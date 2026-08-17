#!/usr/bin/env python3
"""Call an image-capable Responses API model for ROI Pilot pre-review.

This is the independent visual path used when the current Codex session cannot
directly inspect local images.  It sends 10-frame contact sheets, full-resolution
evidence frames, maximum enhancement, provisional TOA/TTP, and vesselness views.
Raw structured responses are saved per candidate and per lesion for resume.

No result is fabricated when OPENAI_API_KEY is unavailable.  This script never
reads private labels and never writes reviewed human statuses.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import ssl
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT = Path("/root/autodl-tmp/aneurysm")
DEFAULT_MODEL = os.environ.get("OPENAI_VISION_MODEL", "gpt-5.5")
AI_STATUSES = [
    "ai_exact_candidate",
    "ai_probable_candidate",
    "ai_ambiguous",
    "ai_unmatched",
]
TRISTATE = ["yes", "no", "uncertain"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--response-root", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_csv(
        temp,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
    )
    os.replace(temp, path)


def image_data_url(path: Path) -> tuple[str, str]:
    data = path.read_bytes()
    suffix = path.suffix.casefold()
    media_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix)
    if media_type is None:
        raise ValueError(f"Unsupported image type for API input: {path}")
    return f"data:{media_type};base64,{base64.b64encode(data).decode('ascii')}", sha256_bytes(data)


def point_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
        "required": ["x", "y"],
    }


def line_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "present": {"type": "boolean"},
            "endpoint1": point_schema(),
            "endpoint2": point_schema(),
            "band_width": {"type": "integer", "minimum": 0},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["present", "endpoint1", "endpoint2", "band_width", "confidence"],
    }


def sac_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "present": {"type": "boolean"},
            "bbox": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "x_min": {"type": "integer"},
                    "y_min": {"type": "integer"},
                    "x_max": {"type": "integer"},
                    "y_max": {"type": "integer"},
                },
                "required": ["x_min", "y_min", "x_max", "y_max"],
            },
            "center": point_schema(),
            "prompt_points": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                        "label": {"type": "integer", "enum": [0, 1]},
                    },
                    "required": ["x", "y", "label"],
                },
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["present", "bbox", "center", "prompt_points", "confidence"],
    }


def phase_schema() -> dict[str, Any]:
    properties: dict[str, Any] = {
        "lesion_visible": {"type": "string", "enum": TRISTATE},
        "sac_opacified": {"type": "string", "enum": TRISTATE},
        "sac_locatable": {"type": "string", "enum": TRISTATE},
        "neck_visible": {"type": "string", "enum": TRISTATE},
        "parent_visible": {"type": "string", "enum": TRISTATE},
        "branch_visible": {"type": "string", "enum": TRISTATE},
        "roi_feasible": {"type": "string", "enum": TRISTATE},
        "selected_internal_series": {"type": "string"},
        "reference_frame_index": {"type": "integer"},
        "image_width": {"type": "integer", "minimum": 0},
        "image_height": {"type": "integer", "minimum": 0},
        "evidence_frame_indices": {"type": "array", "items": {"type": "integer"}},
        "reason": {"type": "string"},
        "sac": sac_schema(),
        "neck": line_schema(),
        "parent_proximal": line_schema(),
        "parent_distal": line_schema(),
        "branch": line_schema(),
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def candidate_schema() -> dict[str, Any]:
    properties: dict[str, Any] = {
        "schema_version": {"type": "string", "enum": ["api_fullseq_v3_ai_candidate_v1"]},
        "lesion_uid": {"type": "string"},
        "candidate_uid": {"type": "string"},
        "candidate_index": {"type": "integer", "minimum": 1},
        "side_concordant": {"type": "string", "enum": TRISTATE},
        "location_concordant": {"type": "string", "enum": TRISTATE},
        "pre": phase_schema(),
        "post": phase_schema(),
        "candidate_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
        "qc_failure_reasons": {"type": "array", "items": {"type": "string"}},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def lesion_schema(candidate_uids: list[str]) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "schema_version": {"type": "string", "enum": ["api_fullseq_v3_ai_lesion_v1"]},
        "lesion_uid": {"type": "string"},
        "selected_pre_candidate": {"type": "string", "enum": ["", *candidate_uids]},
        "selected_post_candidate": {"type": "string", "enum": ["", *candidate_uids]},
        "pre_post_same_lesion": {"type": "string", "enum": TRISTATE},
        "pre_post_view_comparable": {"type": "string", "enum": TRISTATE},
        "ai_status": {"type": "string", "enum": AI_STATUSES},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
        "manual_review_required": {"type": "boolean"},
        "qc_failure_reasons": {"type": "array", "items": {"type": "string"}},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def add_image(content: list[dict[str, Any]], label: str, path: Path, hashes: list[str]) -> None:
    data_url, digest = image_data_url(path)
    hashes.append(f"{path}:{digest}")
    content.append({"type": "input_text", "text": label})
    content.append(
        {"type": "input_image", "image_url": data_url, "detail": "high"}
    )


def reference_frames(cache_dir: Path) -> list[tuple[int, Path, str]]:
    summary = pd.read_csv(cache_dir / "original_frames.csv", dtype=str, keep_default_na=False)
    activity = pd.to_numeric(summary["global_activity_mean_provisional"], errors="coerce")
    peak_position = int(activity.fillna(-1).to_numpy().argmax())
    positions = sorted({0, peak_position, len(summary) - 1})
    return [
        (
            int(summary.iloc[position]["frame_index"]),
            Path(summary.iloc[position]["original_path"]),
            "early" if position == 0 else "late" if position == len(summary) - 1 else "global-activity peak",
        )
        for position in positions
    ]


def candidate_request(
    lesion: dict[str, Any], candidate: dict[str, Any], model: str
) -> tuple[dict[str, Any], str, int]:
    content: list[dict[str, Any]] = []
    source = candidate["candidate_source"]
    prompt = f"""
You are performing blinded research annotation of cerebral angiography for one frozen lesion record.
Do not infer prognosis, treatment outcome, recurrence, or any private label.

Lesion metadata:
- lesion_uid: {lesion['lesion_uid']}
- side: raw={lesion['side_raw']}, normalized={lesion['side_normalized']}
- location: raw={lesion['location_raw']}, normalized={lesion['location_normalized']}
- lesion_index: {lesion['lesion_index_normalized']}

Candidate metadata:
- candidate_uid: {candidate['candidate_uid']}
- candidate_index: {candidate['candidate_index']}
- series_id: {source.get('series_id','')}
- series_path: {source.get('series_path','')}

Definitions:
- sac_opacified: contrast visibly fills the aneurysm sac.
- sac_locatable: the sac region can be spatially localized even if it is not opacified, including Post treatment-device or residual-outline localization.
- roi_feasible: a conservative research ROI draft can be placed without guessing outside visible/localizable anatomy.
- Post feasibility is based on sac_locatable_post, parent_visible_post, and roi_feasible_post; sac_opacified_post is not required.

Inspect all supplied 10-frame contact sheets, original-resolution evidence frames, maximum-enhancement, provisional TOA/TTP, and vesselness views. Return conservative structured output. Coordinates must refer to the explicitly labeled original-resolution reference frame for that phase. When no defensible ROI exists, set present=false and all coordinates/band widths to 0. Never use exact_reviewed/probable_reviewed; this is AI pre-review only.
""".strip()
    content.append({"type": "input_text", "text": prompt})
    hashes: list[str] = []
    image_count = 0
    for phase in ("pre", "post"):
        phase_entries = candidate["phases"].get(phase, [])
        if not phase_entries:
            content.append({"type": "input_text", "text": f"{phase.upper()}: no frozen internal sequence."})
            continue
        for entry in phase_entries:
            cache_dir = Path(entry["cache_dir"])
            label_prefix = (
                f"{phase.upper()} internal_series={entry['internal_series']} "
                f"n_frames={entry['n_frames']} n_pairs={entry['n_contiguous_pairs']}"
            )
            for filename, view_name in (
                ("contact_sheet_10frames.png", "10-frame contact sheet"),
                ("max_enhancement_provisional.png", "maximum enhancement provisional"),
                ("toa_map_provisional.png", "TOA provisional"),
                ("ttp_map_provisional.png", "TTP provisional"),
                ("vesselness_provisional.png", "vesselness provisional"),
            ):
                add_image(content, f"{label_prefix}; view={view_name}", cache_dir / filename, hashes)
                image_count += 1
            refs = reference_frames(cache_dir)
            for frame_index, path, role in refs:
                add_image(
                    content,
                    f"{label_prefix}; original-resolution {role} frame; frame_index={frame_index}; use the global-activity peak frame as the ROI coordinate reference when feasible",
                    path,
                    hashes,
                )
                image_count += 1
    input_hash = sha256_text(prompt + "\n" + "\n".join(hashes))
    body = {
        "model": model,
        "reasoning": {"effort": "medium"},
        "input": [{"role": "user", "content": content}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "roi_pilot_candidate_review",
                "strict": True,
                "schema": candidate_schema(),
            }
        },
        "max_output_tokens": 12000,
        "store": False,
    }
    return body, input_hash, image_count


def lesion_request(
    lesion: dict[str, Any], candidate_results: list[dict[str, Any]], model: str
) -> tuple[dict[str, Any], str, int]:
    candidate_uids = [result["candidate_uid"] for result in candidate_results]
    prompt = f"""
Select the best blinded Pre and Post candidate series for lesion {lesion['lesion_uid']}.
Side={lesion['side_normalized']}; location={lesion['location_normalized']}; lesion_index={lesion['lesion_index_normalized']}.

Use the candidate visual assessments below and the attached contact sheets. Be conservative about whether Pre/Post depict the same lesion. Post candidacy may be accepted when the sac is not opacified if it is locatable, the parent artery is visible, and ROI placement is feasible. Output only an AI pre-review status from the allowed ai_* values. Never output a human reviewed status.

Candidate assessments:
{json.dumps(candidate_results, ensure_ascii=False)}
""".strip()
    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    hashes: list[str] = []
    image_count = 0
    candidate_by_uid = {item["candidate_uid"]: item for item in lesion["candidates"]}
    for result in candidate_results:
        candidate = candidate_by_uid[result["candidate_uid"]]
        for phase in ("pre", "post"):
            for entry in candidate["phases"].get(phase, []):
                path = Path(entry["cache_dir"]) / "contact_sheet_10frames.png"
                add_image(
                    content,
                    f"candidate_uid={result['candidate_uid']} phase={phase} internal_series={entry['internal_series']} contact sheet",
                    path,
                    hashes,
                )
                image_count += 1
    input_hash = sha256_text(prompt + "\n" + "\n".join(hashes))
    body = {
        "model": model,
        "reasoning": {"effort": "medium"},
        "input": [{"role": "user", "content": content}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "roi_pilot_lesion_decision",
                "strict": True,
                "schema": lesion_schema(candidate_uids),
            }
        },
        "max_output_tokens": 6000,
        "store": False,
    }
    return body, input_hash, image_count


def extract_output_text(response: dict[str, Any]) -> str:
    texts: list[str] = []
    for output in response.get("output", []):
        for content in output.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                texts.append(content["text"])
    if not texts and isinstance(response.get("output_text"), str):
        texts.append(response["output_text"])
    if not texts:
        raise ValueError("Responses API returned no output_text")
    return "\n".join(texts)


def call_api(
    body: dict[str, Any], api_key: str, base_url: str, timeout: int, max_retries: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    endpoint = base_url.rstrip("/") + "/responses"
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    context = ssl.create_default_context()
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(
            endpoint,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=context) as handle:
                response = json.loads(handle.read().decode("utf-8"))
            parsed = json.loads(extract_output_text(response))
            return response, parsed
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt >= max_retries:
                raise RuntimeError(f"Responses API HTTP {exc.code}: {error_body}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt >= max_retries:
                raise RuntimeError(f"Responses API network failure: {exc}") from exc
        time.sleep(min(2**attempt, 20))
    raise AssertionError("Unreachable API retry state")


def save_response(
    path: Path,
    response: dict[str, Any],
    parsed: dict[str, Any],
    input_hash: str,
    model: str,
    image_count: int,
) -> None:
    wrapper = {
        "schema_version": "api_fullseq_v3_visual_response_wrapper_v1",
        "model": model,
        "input_hash": input_hash,
        "image_count": image_count,
        "response_id": response.get("id", ""),
        "usage": response.get("usage", {}),
        "parsed": parsed,
    }
    atomic_write_text(json.dumps(wrapper, ensure_ascii=False, indent=2), path)


def valid_resume(path: Path, input_hash: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        wrapper = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if wrapper.get("input_hash") != input_hash or not isinstance(wrapper.get("parsed"), dict):
        return None
    return wrapper["parsed"]


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    root = args.project_root.resolve()
    cache_root = (
        args.cache_root or root / "outputs/api_fullseq_v3_roi_pilot_ai_cache"
    ).resolve()
    response_root = (
        args.response_root or root / "reports/api_fullseq_v3/ai_visual_responses"
    ).resolve()
    manifest_path = cache_root / "pilot_visual_input_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lesions = manifest["lesions"]
    if len(lesions) != 30:
        raise AssertionError(f"Expected 30 lesions in visual input manifest, got {len(lesions)}")

    request_rows: list[dict[str, Any]] = []
    candidate_jobs: list[tuple[dict[str, Any], dict[str, Any], str, int, Path]] = []
    for lesion in lesions:
        for candidate in lesion["candidates"]:
            body, input_hash, image_count = candidate_request(lesion, candidate, args.model)
            output_path = response_root / "candidates" / f"{candidate['candidate_uid']}.json"
            request_rows.append(
                {
                    "request_type": "candidate",
                    "lesion_uid": lesion["lesion_uid"],
                    "candidate_uid": candidate["candidate_uid"],
                    "model": args.model,
                    "image_count": image_count,
                    "input_hash": input_hash,
                    "response_path": str(output_path),
                }
            )
            candidate_jobs.append((lesion, candidate, input_hash, image_count, output_path))
    request_manifest_path = response_root / "visual_request_manifest.csv"
    atomic_write_csv(pd.DataFrame(request_rows), request_manifest_path)
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "prepared_candidate_requests": len(candidate_jobs),
                    "pilot_lesions": len(lesions),
                    "model": args.model,
                    "request_manifest": str(request_manifest_path),
                    "api_called": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        blocker = response_root.parent / "roi_pilot_ai_execution_blocked.md"
        atomic_write_text(
            "\n".join(
                [
                    "# ROI Pilot AI execution blocked",
                    "",
                    "- Visual request packages were prepared, but no visual API call was made.",
                    "- Blocking condition: `OPENAI_API_KEY` is not present in the process environment.",
                    f"- Prepared candidate requests: {len(candidate_jobs)}",
                    f"- Configured model: `{args.model}`",
                    "- No AI candidate decisions, lesion decisions, or ROI annotations were fabricated.",
                    "",
                ]
            ),
            blocker,
        )
        raise SystemExit(
            f"OPENAI_API_KEY is required for real visual review. Blocker report: {blocker}"
        )

    response_root.mkdir(parents=True, exist_ok=True)

    def run_candidate(job: tuple[dict[str, Any], dict[str, Any], str, int, Path]) -> tuple[str, dict[str, Any]]:
        lesion, candidate, input_hash, image_count, output_path = job
        body, rebuilt_hash, rebuilt_image_count = candidate_request(lesion, candidate, args.model)
        if rebuilt_hash != input_hash or rebuilt_image_count != image_count:
            raise AssertionError("Candidate request material changed between prepare and execution")
        if args.resume:
            resumed = valid_resume(output_path, input_hash)
            if resumed is not None:
                return candidate["candidate_uid"], resumed
        response, parsed = call_api(
            body, api_key, args.base_url, args.timeout_seconds, args.max_retries
        )
        if parsed.get("lesion_uid") != lesion["lesion_uid"] or parsed.get(
            "candidate_uid"
        ) != candidate["candidate_uid"]:
            raise AssertionError("Candidate response identity mismatch")
        save_response(output_path, response, parsed, input_hash, args.model, image_count)
        return candidate["candidate_uid"], parsed

    candidate_results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_candidate, job): job for job in candidate_jobs}
        for completed, future in enumerate(as_completed(futures), start=1):
            uid, parsed = future.result()
            candidate_results[uid] = parsed
            print(
                json.dumps(
                    {"candidate_visual_completed": completed, "total": len(candidate_jobs), "candidate_uid": uid}
                ),
                flush=True,
            )

    lesion_results = 0
    for lesion in lesions:
        results = [candidate_results[item["candidate_uid"]] for item in lesion["candidates"]]
        body, input_hash, image_count = lesion_request(lesion, results, args.model)
        output_path = response_root / "lesions" / f"{sha256_text(lesion['lesion_uid'])[:24]}.json"
        parsed = valid_resume(output_path, input_hash) if args.resume else None
        if parsed is None:
            response, parsed = call_api(
                body, api_key, args.base_url, args.timeout_seconds, args.max_retries
            )
            if parsed.get("lesion_uid") != lesion["lesion_uid"]:
                raise AssertionError("Lesion response identity mismatch")
            save_response(output_path, response, parsed, input_hash, args.model, image_count)
        lesion_results += 1
        print(
            json.dumps(
                {"lesion_visual_completed": lesion_results, "total": len(lesions), "lesion_uid": lesion["lesion_uid"]}
            ),
            flush=True,
        )

    print(
        json.dumps(
            {
                "candidate_visual_responses": len(candidate_results),
                "lesion_visual_responses": lesion_results,
                "model": args.model,
                "response_root": str(response_root),
                "private_labels_read": False,
                "human_reviewed_status_written": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

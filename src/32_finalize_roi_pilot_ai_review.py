#!/usr/bin/env python3
"""Validate visual-model responses, create ROI drafts/QC, and render AI review.

The script requires complete real visual response wrappers from code/31.  It
refuses to fabricate missing AI decisions.  It writes AI-only statuses, initial
ROI masks, programmatic QC, review priority, and a light four-action HTML report.
No human reviewed status or final local feature is produced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd


PROJECT = Path("/root/autodl-tmp/aneurysm")
AI_STATUSES = {
    "ai_exact_candidate",
    "ai_probable_candidate",
    "ai_ambiguous",
    "ai_unmatched",
}
TRISTATE = {"yes", "no", "uncertain"}

CANDIDATE_COLUMNS = [
    "lesion_uid",
    "candidate_uid",
    "candidate_index",
    "candidate_series_id",
    "candidate_series_path",
    "candidate_source_type",
    "candidate_valid_in_v2",
    "candidate_selected_in_v2",
    "side_concordant",
    "location_concordant",
    "lesion_visible_pre",
    "lesion_visible_post",
    "sac_opacified_pre",
    "sac_opacified_post",
    "sac_locatable_pre",
    "sac_locatable_post",
    "neck_visible_pre",
    "neck_visible_post",
    "parent_visible_pre",
    "parent_visible_post",
    "branch_visible_pre",
    "branch_visible_post",
    "roi_feasible_pre",
    "roi_feasible_post",
    "selected_internal_series_pre",
    "selected_internal_series_post",
    "reference_frame_index_pre",
    "reference_frame_index_post",
    "evidence_frame_indices_pre",
    "evidence_frame_indices_post",
    "candidate_confidence",
    "reason",
    "model_qc_failure_reasons",
    "candidate_response_path",
]

LESION_COLUMNS = [
    "lesion_uid",
    "patient_id",
    "side_raw",
    "side_normalized",
    "location_raw",
    "location_normalized",
    "lesion_index_normalized",
    "selected_pre_candidate",
    "selected_post_candidate",
    "selected_pre_series_id",
    "selected_post_series_id",
    "selected_pre_series_path",
    "selected_post_series_path",
    "selected_pre_internal_series",
    "selected_post_internal_series",
    "side_concordant",
    "location_concordant",
    "lesion_visible_pre",
    "lesion_visible_post",
    "sac_opacified_pre",
    "sac_opacified_post",
    "sac_locatable_pre",
    "sac_locatable_post",
    "neck_visible_pre",
    "neck_visible_post",
    "parent_visible_pre",
    "parent_visible_post",
    "branch_visible_pre",
    "branch_visible_post",
    "roi_feasible_pre",
    "roi_feasible_post",
    "pre_post_same_lesion",
    "pre_post_view_comparable",
    "ai_status",
    "confidence",
    "evidence_frame_indices_pre",
    "evidence_frame_indices_post",
    "reason",
    "manual_review_required",
    "manual_review_priority_score",
    "manual_review_priority_rank",
    "qc_failure_reasons",
    "post_nonopacified_but_locatable_qc",
    "pre_annotation_path",
    "post_annotation_path",
    "lesion_response_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--response-root", type=Path)
    parser.add_argument("--annotation-root", type=Path)
    parser.add_argument("--candidate-output", type=Path)
    parser.add_argument("--lesion-output", type=Path)
    parser.add_argument("--html-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def atomic_write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.stem}.{uuid.uuid4().hex}{path.suffix}")
    if not cv2.imwrite(str(temp), image):
        raise IOError(f"Failed to write {path}")
    os.replace(temp, path)


def load_wrapper(path: Path) -> dict[str, Any]:
    wrapper = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(wrapper.get("parsed"), dict):
        raise ValueError(f"Visual response wrapper lacks parsed object: {path}")
    return wrapper


def validate_phase(phase: dict[str, Any], label: str) -> None:
    for field in (
        "lesion_visible",
        "sac_opacified",
        "sac_locatable",
        "neck_visible",
        "parent_visible",
        "branch_visible",
        "roi_feasible",
    ):
        if phase.get(field) not in TRISTATE:
            raise ValueError(f"{label}.{field} invalid")
    if not isinstance(phase.get("evidence_frame_indices"), list):
        raise ValueError(f"{label}.evidence_frame_indices invalid")


def validate_candidate(parsed: dict[str, Any], lesion_uid: str, candidate_uid: str) -> None:
    if parsed.get("schema_version") != "api_fullseq_v3_ai_candidate_v1":
        raise ValueError("Candidate schema_version mismatch")
    if parsed.get("lesion_uid") != lesion_uid or parsed.get("candidate_uid") != candidate_uid:
        raise ValueError("Candidate identity mismatch")
    if parsed.get("side_concordant") not in TRISTATE or parsed.get(
        "location_concordant"
    ) not in TRISTATE:
        raise ValueError("Candidate concordance value invalid")
    validate_phase(parsed["pre"], "pre")
    validate_phase(parsed["post"], "post")


def validate_lesion(parsed: dict[str, Any], lesion: dict[str, Any]) -> None:
    if parsed.get("schema_version") != "api_fullseq_v3_ai_lesion_v1":
        raise ValueError("Lesion schema_version mismatch")
    if parsed.get("lesion_uid") != lesion["lesion_uid"]:
        raise ValueError("Lesion identity mismatch")
    allowed = {item["candidate_uid"] for item in lesion["candidates"]} | {""}
    if parsed.get("selected_pre_candidate") not in allowed or parsed.get(
        "selected_post_candidate"
    ) not in allowed:
        raise ValueError("Lesion selected candidate not in lesion package")
    if parsed.get("ai_status") not in AI_STATUSES:
        raise ValueError("Lesion ai_status invalid")
    if parsed.get("pre_post_same_lesion") not in TRISTATE or parsed.get(
        "pre_post_view_comparable"
    ) not in TRISTATE:
        raise ValueError("Lesion Pre/Post assessment invalid")


def candidate_response_path(response_root: Path, candidate_uid: str) -> Path:
    return response_root / "candidates" / f"{candidate_uid}.json"


def lesion_response_path(response_root: Path, lesion_uid: str) -> Path:
    return response_root / "lesions" / f"{sha256_text(lesion_uid)[:24]}.json"


def cache_entry(candidate: dict[str, Any], phase: str, internal_series: str) -> dict[str, Any] | None:
    for entry in candidate["phases"].get(phase, []):
        if str(entry["internal_series"]) == str(internal_series):
            return entry
    return None


def reference_frame(cache_dir: Path, frame_index: int) -> tuple[Path, np.ndarray]:
    frame = pd.read_csv(cache_dir / "original_frames.csv", dtype=str, keep_default_na=False)
    matches = frame.loc[pd.to_numeric(frame["frame_index"]) == int(frame_index)]
    if matches.empty:
        raise ValueError(f"Reference frame {frame_index} not found in {cache_dir}")
    path = Path(matches.iloc[0]["original_path"])
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise IOError(path)
    return path, image


def point_in_bounds(point: dict[str, Any], width: int, height: int) -> bool:
    return 0 <= int(point["x"]) < width and 0 <= int(point["y"]) < height


def line_bounds(line: dict[str, Any], width: int, height: int) -> bool:
    return point_in_bounds(line["endpoint1"], width, height) and point_in_bounds(
        line["endpoint2"], width, height
    )


def ellipse_mask(shape: tuple[int, int], bbox: dict[str, Any]) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = (
        int(bbox["x_min"]),
        int(bbox["y_min"]),
        int(bbox["x_max"]),
        int(bbox["y_max"]),
    )
    x1, x2 = sorted((max(0, x1), min(width - 1, x2)))
    y1, y2 = sorted((max(0, y1), min(height - 1, y2)))
    mask = np.zeros(shape, dtype=np.uint8)
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    axes = (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask


def refine_sac_mask(
    initial: np.ndarray, center: dict[str, Any], cache_dir: Path
) -> np.ndarray:
    activity = cv2.imread(
        str(cache_dir / "max_enhancement_provisional.png"), cv2.IMREAD_GRAYSCALE
    )
    vessel = cv2.imread(str(cache_dir / "vesselness_provisional.png"), cv2.IMREAD_GRAYSCALE)
    if activity is None or vessel is None or activity.shape != initial.shape:
        return initial
    score = cv2.addWeighted(activity, 0.65, vessel, 0.35, 0)
    values = score[initial > 0]
    if values.size < 10:
        return initial
    threshold = max(float(np.percentile(values, 55)), 10.0)
    candidate = ((score >= threshold) & (initial > 0)).astype(np.uint8)
    count, labels, _, _ = cv2.connectedComponentsWithStats(candidate, 8)
    x, y = int(center["x"]), int(center["y"])
    if not (0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]):
        return initial
    label = int(labels[y, x])
    if label <= 0 or label >= count:
        return initial
    refined = np.zeros_like(initial)
    refined[labels == label] = 255
    if int((refined > 0).sum()) < 20:
        return initial
    refined = cv2.morphologyEx(
        refined,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    return refined


def line_mask(shape: tuple[int, int], line: dict[str, Any]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if not line.get("present", False):
        return mask
    point1 = (int(line["endpoint1"]["x"]), int(line["endpoint1"]["y"]))
    point2 = (int(line["endpoint2"]["x"]), int(line["endpoint2"]["y"]))
    cv2.line(mask, point1, point2, 255, max(1, int(line["band_width"])))
    return mask


def masks_adjacent(first: np.ndarray, second: np.ndarray, radius: int) -> bool:
    if not np.any(first) or not np.any(second):
        return False
    size = max(3, radius * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    dilated = cv2.dilate(first, kernel)
    return bool(np.any((dilated > 0) & (second > 0)))


def mask_overlay(
    image: np.ndarray,
    masks: dict[str, np.ndarray],
) -> np.ndarray:
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    colors = {
        "sac": (0, 0, 255),
        "neck": (0, 255, 255),
        "parent_proximal": (0, 255, 0),
        "parent_distal": (255, 255, 0),
        "branch": (255, 0, 255),
    }
    for name, mask in masks.items():
        color = np.asarray(colors[name], dtype=np.float32)
        selected = mask > 0
        overlay[selected] = np.clip(
            overlay[selected].astype(np.float32) * 0.45 + color * 0.55, 0, 255
        ).astype(np.uint8)
    return overlay


def registration_failures(cache_dir: Path, sac_bbox: dict[str, Any]) -> list[str]:
    registration = pd.read_csv(cache_dir / "registration_qc.csv")
    magnitudes = pd.to_numeric(
        registration["translation_magnitude_pixels"], errors="coerce"
    )
    responses = pd.to_numeric(
        registration["phase_correlation_response"], errors="coerce"
    )
    failures: list[str] = []
    if magnitudes.max() > 50 or magnitudes.median() > 20:
        failures.append("roi_cross_frame_propagation_large_translation")
    if int((responses < 0.05).sum()) > max(1, math.ceil(len(responses) * 0.2)):
        failures.append("roi_cross_frame_propagation_low_registration_response")
    width = int(sac_bbox["x_max"]) - int(sac_bbox["x_min"])
    height = int(sac_bbox["y_max"]) - int(sac_bbox["y_min"])
    if width <= 0 or height <= 0:
        failures.append("sac_bbox_degenerate")
    return failures


def create_phase_annotation(
    root: Path,
    annotation_root: Path,
    lesion_uid: str,
    phase: str,
    candidate: dict[str, Any],
    phase_result: dict[str, Any],
) -> tuple[str, list[str], dict[str, Any]]:
    failures: list[str] = []
    selected_internal = str(phase_result["selected_internal_series"])
    entry = cache_entry(candidate, phase, selected_internal)
    if entry is None:
        return "", [f"{phase}_selected_internal_series_not_in_cache"], {}
    cache_dir = Path(entry["cache_dir"])
    reference_path, image = reference_frame(
        cache_dir, int(phase_result["reference_frame_index"])
    )
    height, width = image.shape
    if int(phase_result["image_width"]) != width or int(phase_result["image_height"]) != height:
        failures.append(f"{phase}_model_image_dimensions_mismatch")

    sac = phase_result["sac"]
    bbox = sac["bbox"]
    bbox_in_bounds = (
        0 <= int(bbox["x_min"]) < width
        and 0 <= int(bbox["x_max"]) < width
        and 0 <= int(bbox["y_min"]) < height
        and 0 <= int(bbox["y_max"]) < height
    )
    if sac.get("present", False) and not bbox_in_bounds:
        failures.append(f"{phase}_sac_bbox_out_of_bounds")
    if sac.get("present", False) and not point_in_bounds(sac["center"], width, height):
        failures.append(f"{phase}_sac_center_out_of_bounds")
    for name in ("neck", "parent_proximal", "parent_distal", "branch"):
        line = phase_result[name]
        if line.get("present", False) and not line_bounds(line, width, height):
            failures.append(f"{phase}_{name}_out_of_bounds")

    sac_mask = ellipse_mask(image.shape, bbox) if sac.get("present", False) else np.zeros_like(image)
    if sac.get("present", False):
        sac_mask = refine_sac_mask(sac_mask, sac["center"], cache_dir)
    masks = {
        "sac": sac_mask,
        "neck": line_mask(image.shape, phase_result["neck"]),
        "parent_proximal": line_mask(image.shape, phase_result["parent_proximal"]),
        "parent_distal": line_mask(image.shape, phase_result["parent_distal"]),
        "branch": line_mask(image.shape, phase_result["branch"]),
    }
    area_ratio = float((sac_mask > 0).sum()) / float(width * height)
    if sac.get("present", False) and (area_ratio < 0.00005 or area_ratio > 0.20):
        failures.append(f"{phase}_sac_area_abnormal")
    neck_radius = max(5, int(phase_result["neck"]["band_width"]) * 2)
    if phase_result["neck"].get("present", False) and not masks_adjacent(
        masks["sac"], masks["neck"], neck_radius
    ):
        failures.append(f"{phase}_neck_not_adjacent_to_sac")
    if phase_result["parent_proximal"].get("present", False) and not masks_adjacent(
        masks["neck"], masks["parent_proximal"], 20
    ):
        failures.append(f"{phase}_parent_proximal_not_adjacent_to_neck")
    if phase_result["parent_distal"].get("present", False) and not masks_adjacent(
        masks["neck"], masks["parent_distal"], 20
    ):
        failures.append(f"{phase}_parent_distal_not_adjacent_to_neck")
    failures.extend(registration_failures(cache_dir, bbox))

    output_dir = annotation_root / lesion_uid / phase
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_image(output_dir / "sac_mask.png", masks["sac"])
    atomic_write_image(output_dir / "neck_mask.png", masks["neck"])
    atomic_write_image(output_dir / "parent_proximal_mask.png", masks["parent_proximal"])
    atomic_write_image(output_dir / "parent_distal_mask.png", masks["parent_distal"])
    if phase_result["branch"].get("present", False):
        atomic_write_image(output_dir / "branch_mask.png", masks["branch"])
    atomic_write_image(output_dir / "overlay.png", mask_overlay(image, masks))
    annotation = {
        "schema_version": "api_fullseq_v3_roi_annotation_ai_v1",
        "lesion_uid": lesion_uid,
        "phase": phase,
        "candidate_uid": candidate["candidate_uid"],
        "candidate_series_id": candidate["candidate_source"].get("series_id", ""),
        "candidate_series_path": candidate["candidate_source"].get("series_path", ""),
        "internal_series": selected_internal,
        "reference_frame_index": int(phase_result["reference_frame_index"]),
        "reference_frame_path": str(reference_path),
        "image_width": width,
        "image_height": height,
        "sac": phase_result["sac"],
        "neck": phase_result["neck"],
        "parent_proximal": phase_result["parent_proximal"],
        "parent_distal": phase_result["parent_distal"],
        "branch": phase_result["branch"],
        "mask_generation": "bbox_activity_vesselness_seeded_component_and_line_bands_v1",
        "sac_area_pixels": int((sac_mask > 0).sum()),
        "sac_area_ratio": area_ratio,
        "qc_failure_reasons": failures,
        "ai_draft_only": True,
    }
    annotation_path = output_dir / "annotation.json"
    atomic_write_text(json.dumps(annotation, ensure_ascii=False, indent=2), annotation_path)
    center = {
        "x_normalized": float(phase_result["sac"]["center"]["x"]) / max(width, 1),
        "y_normalized": float(phase_result["sac"]["center"]["y"]) / max(height, 1),
    }
    return str(annotation_path), failures, center


def join_values(values: Iterable[Any]) -> str:
    return "|".join(str(value) for value in values)


def candidate_csv_row(
    lesion: dict[str, Any], candidate: dict[str, Any], parsed: dict[str, Any], path: Path
) -> dict[str, Any]:
    source = candidate["candidate_source"]
    audit = candidate["candidate_audit"]
    return {
        "lesion_uid": lesion["lesion_uid"],
        "candidate_uid": candidate["candidate_uid"],
        "candidate_index": candidate["candidate_index"],
        "candidate_series_id": source.get("series_id", ""),
        "candidate_series_path": source.get("series_path", ""),
        "candidate_source_type": source.get("source_type", ""),
        "candidate_valid_in_v2": audit.get("candidate_valid", False),
        "candidate_selected_in_v2": audit.get("selected_candidate_in_v2", False),
        "side_concordant": parsed["side_concordant"],
        "location_concordant": parsed["location_concordant"],
        "lesion_visible_pre": parsed["pre"]["lesion_visible"],
        "lesion_visible_post": parsed["post"]["lesion_visible"],
        "sac_opacified_pre": parsed["pre"]["sac_opacified"],
        "sac_opacified_post": parsed["post"]["sac_opacified"],
        "sac_locatable_pre": parsed["pre"]["sac_locatable"],
        "sac_locatable_post": parsed["post"]["sac_locatable"],
        "neck_visible_pre": parsed["pre"]["neck_visible"],
        "neck_visible_post": parsed["post"]["neck_visible"],
        "parent_visible_pre": parsed["pre"]["parent_visible"],
        "parent_visible_post": parsed["post"]["parent_visible"],
        "branch_visible_pre": parsed["pre"]["branch_visible"],
        "branch_visible_post": parsed["post"]["branch_visible"],
        "roi_feasible_pre": parsed["pre"]["roi_feasible"],
        "roi_feasible_post": parsed["post"]["roi_feasible"],
        "selected_internal_series_pre": parsed["pre"]["selected_internal_series"],
        "selected_internal_series_post": parsed["post"]["selected_internal_series"],
        "reference_frame_index_pre": parsed["pre"]["reference_frame_index"],
        "reference_frame_index_post": parsed["post"]["reference_frame_index"],
        "evidence_frame_indices_pre": join_values(parsed["pre"]["evidence_frame_indices"]),
        "evidence_frame_indices_post": join_values(parsed["post"]["evidence_frame_indices"]),
        "candidate_confidence": parsed["candidate_confidence"],
        "reason": parsed["reason"],
        "model_qc_failure_reasons": join_values(parsed["qc_failure_reasons"]),
        "candidate_response_path": str(path),
    }


def selected_rollup(
    candidate_result: dict[str, Any] | None, phase: str, field: str, default: str = ""
) -> Any:
    if candidate_result is None:
        return default
    return candidate_result[phase].get(field, default)


def manual_priority(status: str, confidence: float, failures: list[str], conflict: bool) -> float:
    status_weight = {
        "ai_unmatched": 100.0,
        "ai_ambiguous": 80.0,
        "ai_probable_candidate": 30.0,
        "ai_exact_candidate": 10.0,
    }[status]
    return status_weight + 12.0 * len(set(failures)) + (20.0 if conflict else 0.0) + 20.0 * (
        1.0 - confidence
    )


def copy_asset(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    shutil.copy2(source, temp)
    os.replace(temp, destination)
    return destination.name


def render_html(
    html_root: Path,
    lesions: list[dict[str, Any]],
    candidate_results: dict[str, dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    annotation_root: Path,
) -> None:
    assets = html_root / "assets"
    html_root.mkdir(parents=True, exist_ok=True)
    decision_lookup = {row["lesion_uid"]: row for row in decision_rows}
    sections: list[str] = []
    for lesion in lesions:
        uid = lesion["lesion_uid"]
        decision = decision_lookup[uid]
        cards: list[str] = []
        for candidate in lesion["candidates"]:
            result = candidate_results[candidate["candidate_uid"]]
            images: list[str] = []
            for phase in ("pre", "post"):
                for entry in candidate["phases"].get(phase, []):
                    source = Path(entry["cache_dir"]) / "contact_sheet_10frames.png"
                    name = (
                        f"{sha256_text(uid)[:10]}_{candidate['candidate_uid']}_{phase}_"
                        f"{entry['internal_series']}_contact.png"
                    )
                    copy_asset(source, assets / name)
                    images.append(
                        f'<figure><figcaption>{html.escape(phase)} internal {html.escape(str(entry["internal_series"]))}</figcaption><img src="assets/{name}"></figure>'
                    )
            cards.append(
                f"""<article class="candidate"><h3>{html.escape(candidate['candidate_uid'])} · {html.escape(str(candidate['candidate_source'].get('series_id','')))}</h3>
                <p class="path">{html.escape(str(candidate['candidate_source'].get('series_path','')))}</p>
                <p>confidence={result['candidate_confidence']:.3f}; side={html.escape(result['side_concordant'])}; location={html.escape(result['location_concordant'])}</p>
                <p>Pre evidence={html.escape(join_values(result['pre']['evidence_frame_indices']))}; Post evidence={html.escape(join_values(result['post']['evidence_frame_indices']))}</p>
                <div class="images">{''.join(images)}</div></article>"""
            )
        overlays: list[str] = []
        for phase in ("pre", "post"):
            source = annotation_root / uid / phase / "overlay.png"
            if source.is_file():
                name = f"{sha256_text(uid)[:12]}_{phase}_overlay.png"
                copy_asset(source, assets / name)
                overlays.append(
                    f'<figure><figcaption>{phase} ROI overlay</figcaption><img src="assets/{name}"></figure>'
                )
        sections.append(
            f"""<section class="lesion" data-status="{html.escape(decision['ai_status'])}"><h2>{html.escape(uid)}</h2>
            <p>side={html.escape(lesion['side_normalized'])}; location={html.escape(lesion['location_normalized'])}; AI status=<b>{html.escape(decision['ai_status'])}</b>; confidence={float(decision['confidence']):.3f}</p>
            <p>selected Pre={html.escape(str(decision['selected_pre_candidate']))}; selected Post={html.escape(str(decision['selected_post_candidate']))}</p>
            <p>QC={html.escape(str(decision['qc_failure_reasons']) or 'none')}</p><div class="images">{''.join(overlays)}</div>
            <div class="actions" data-uid="{html.escape(uid)}"><button>Accept</button><button>Accept with edits</button><button>Reject</button><button>Ambiguous</button><span class="choice"></span></div>
            {''.join(cards)}</section>"""
        )
    document = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ROI Pilot AI pre-review</title>
<style>body{{font-family:system-ui,sans-serif;background:#eef1f4;margin:20px}}.lesion,.candidate{{background:white;padding:16px;margin:16px 0;border-radius:8px}}.images{{display:grid;grid-template-columns:repeat(2,minmax(300px,1fr));gap:12px}}img{{width:100%;height:auto;border:1px solid #555}}.path{{overflow-wrap:anywhere}}button{{margin:4px;padding:9px 12px}}.choice{{font-weight:bold;margin-left:10px}}</style></head>
<body><h1>30-case ROI Pilot AI pre-review</h1><p>Four-action triage only. Choices are stored in this browser and can be exported.</p><button id="export">Export actions JSON</button>{''.join(sections)}
<script>const key='api_fullseq_v3_ai_pre_review_actions_v1';let state=JSON.parse(localStorage.getItem(key)||'{{}}');document.querySelectorAll('.actions').forEach(box=>{{const uid=box.dataset.uid;const label=box.querySelector('.choice');if(state[uid])label.textContent=state[uid];box.querySelectorAll('button').forEach(button=>button.onclick=()=>{{state[uid]=button.textContent;localStorage.setItem(key,JSON.stringify(state));label.textContent=state[uid];}});}});document.getElementById('export').onclick=()=>{{const blob=new Blob([JSON.stringify(state,null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='roi_pilot_ai_pre_review_actions.json';a.click();}};</script></body></html>"""
    atomic_write_text(document, html_root / "index.html")


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    cache_root = (
        args.cache_root or root / "outputs/api_fullseq_v3_roi_pilot_ai_cache"
    ).resolve()
    response_root = (
        args.response_root or root / "reports/api_fullseq_v3/ai_visual_responses"
    ).resolve()
    annotation_root = (args.annotation_root or root / "roi_annotations_ai").resolve()
    candidate_output = (
        args.candidate_output
        or root / "reports/api_fullseq_v3/roi_pilot_ai_candidate_review.csv"
    ).resolve()
    lesion_output = (
        args.lesion_output
        or root / "reports/api_fullseq_v3/roi_pilot_ai_lesion_decision.csv"
    ).resolve()
    html_root = (
        args.html_root or root / "reports/api_fullseq_v3/ai_pre_review"
    ).resolve()

    manifest = json.loads(
        (cache_root / "pilot_visual_input_manifest.json").read_text(encoding="utf-8")
    )
    lesions = manifest["lesions"]
    if len(lesions) != 30:
        raise AssertionError("Expected 30 lesions in precache manifest")
    missing_candidate: list[str] = []
    missing_lesion: list[str] = []
    for lesion in lesions:
        for candidate in lesion["candidates"]:
            if not candidate_response_path(
                response_root, candidate["candidate_uid"]
            ).is_file():
                missing_candidate.append(candidate["candidate_uid"])
        if not lesion_response_path(response_root, lesion["lesion_uid"]).is_file():
            missing_lesion.append(lesion["lesion_uid"])
    if missing_candidate or missing_lesion:
        raise FileNotFoundError(
            f"Incomplete real visual responses: candidate_missing={len(missing_candidate)}, "
            f"lesion_missing={len(missing_lesion)}. Run code/31 with a valid image-capable API key."
        )

    candidate_results: dict[str, dict[str, Any]] = {}
    candidate_paths: dict[str, Path] = {}
    candidate_rows: list[dict[str, Any]] = []
    lesion_results: dict[str, dict[str, Any]] = {}
    lesion_paths: dict[str, Path] = {}
    for lesion in lesions:
        for candidate in lesion["candidates"]:
            path = candidate_response_path(response_root, candidate["candidate_uid"])
            parsed = load_wrapper(path)["parsed"]
            validate_candidate(parsed, lesion["lesion_uid"], candidate["candidate_uid"])
            candidate_results[candidate["candidate_uid"]] = parsed
            candidate_paths[candidate["candidate_uid"]] = path
            candidate_rows.append(candidate_csv_row(lesion, candidate, parsed, path))
        path = lesion_response_path(response_root, lesion["lesion_uid"])
        parsed = load_wrapper(path)["parsed"]
        validate_lesion(parsed, lesion)
        lesion_results[lesion["lesion_uid"]] = parsed
        lesion_paths[lesion["lesion_uid"]] = path

    if args.validate_only:
        print(
            json.dumps(
                {
                    "candidate_responses_valid": len(candidate_results),
                    "lesion_responses_valid": len(lesion_results),
                    "outputs_written": False,
                },
                indent=2,
            )
        )
        return 0

    candidate_manifest = {
        candidate["candidate_uid"]: candidate
        for lesion in lesions
        for candidate in lesion["candidates"]
    }
    decision_rows: list[dict[str, Any]] = []
    for lesion in lesions:
        uid = lesion["lesion_uid"]
        lesion_result = lesion_results[uid]
        pre_uid = lesion_result["selected_pre_candidate"]
        post_uid = lesion_result["selected_post_candidate"]
        pre_result = candidate_results.get(pre_uid)
        post_result = candidate_results.get(post_uid)
        pre_candidate = candidate_manifest.get(pre_uid)
        post_candidate = candidate_manifest.get(post_uid)
        failures = list(lesion_result["qc_failure_reasons"])
        if pre_result:
            failures.extend(pre_result["qc_failure_reasons"])
        if post_result and post_uid != pre_uid:
            failures.extend(post_result["qc_failure_reasons"])

        pre_annotation = ""
        post_annotation = ""
        centers: dict[str, dict[str, Any]] = {}
        if lesion_result["ai_status"] in {
            "ai_exact_candidate",
            "ai_probable_candidate",
        }:
            if not pre_result or not post_result or not pre_candidate or not post_candidate:
                failures.append("ai_candidate_status_missing_selected_pre_or_post")
            else:
                pre_annotation, pre_failures, centers["pre"] = create_phase_annotation(
                    root,
                    annotation_root,
                    uid,
                    "pre",
                    pre_candidate,
                    pre_result["pre"],
                )
                post_annotation, post_failures, centers["post"] = create_phase_annotation(
                    root,
                    annotation_root,
                    uid,
                    "post",
                    post_candidate,
                    post_result["post"],
                )
                failures.extend(pre_failures)
                failures.extend(post_failures)
                if centers.get("pre") and centers.get("post"):
                    distance = math.hypot(
                        centers["pre"]["x_normalized"] - centers["post"]["x_normalized"],
                        centers["pre"]["y_normalized"] - centers["post"]["y_normalized"],
                    )
                    if distance > 0.25:
                        failures.append("pre_post_roi_centers_anatomically_distant")

        candidate_confidences = sorted(
            [
                float(candidate_results[item["candidate_uid"]]["candidate_confidence"])
                for item in lesion["candidates"]
            ],
            reverse=True,
        )
        conflict = (
            len(candidate_confidences) > 1
            and candidate_confidences[0] - candidate_confidences[1] < 0.10
        )
        if conflict:
            failures.append("candidate_conflict_significant")

        post_nonopacified_locatable = bool(
            post_result
            and post_result["post"]["sac_opacified"] != "yes"
            and post_result["post"]["sac_locatable"] == "yes"
        )
        if post_result and post_result["post"]["sac_locatable"] != "yes" and lesion_result[
            "ai_status"
        ] in {"ai_exact_candidate", "ai_probable_candidate"}:
            failures.append("post_sac_not_locatable")
        if post_result and lesion_result["ai_status"] in {
            "ai_exact_candidate",
            "ai_probable_candidate",
        }:
            if post_result["post"]["parent_visible"] != "yes":
                failures.append("post_parent_not_visible")
            if post_result["post"]["roi_feasible"] != "yes":
                failures.append("post_roi_not_feasible")

        failures = sorted(set(value for value in failures if value))
        confidence = float(lesion_result["confidence"])
        manual_required = bool(lesion_result["manual_review_required"] or failures)
        priority = manual_priority(lesion_result["ai_status"], confidence, failures, conflict)
        pre_source = pre_candidate["candidate_source"] if pre_candidate else {}
        post_source = post_candidate["candidate_source"] if post_candidate else {}
        decision_rows.append(
            {
                "lesion_uid": uid,
                "patient_id": lesion["patient_id"],
                "side_raw": lesion["side_raw"],
                "side_normalized": lesion["side_normalized"],
                "location_raw": lesion["location_raw"],
                "location_normalized": lesion["location_normalized"],
                "lesion_index_normalized": lesion["lesion_index_normalized"],
                "selected_pre_candidate": pre_uid,
                "selected_post_candidate": post_uid,
                "selected_pre_series_id": pre_source.get("series_id", ""),
                "selected_post_series_id": post_source.get("series_id", ""),
                "selected_pre_series_path": pre_source.get("series_path", ""),
                "selected_post_series_path": post_source.get("series_path", ""),
                "selected_pre_internal_series": selected_rollup(
                    pre_result, "pre", "selected_internal_series"
                ),
                "selected_post_internal_series": selected_rollup(
                    post_result, "post", "selected_internal_series"
                ),
                "side_concordant": pre_result["side_concordant"] if pre_result else "uncertain",
                "location_concordant": pre_result["location_concordant"] if pre_result else "uncertain",
                "lesion_visible_pre": selected_rollup(pre_result, "pre", "lesion_visible", "uncertain"),
                "lesion_visible_post": selected_rollup(post_result, "post", "lesion_visible", "uncertain"),
                "sac_opacified_pre": selected_rollup(pre_result, "pre", "sac_opacified", "uncertain"),
                "sac_opacified_post": selected_rollup(post_result, "post", "sac_opacified", "uncertain"),
                "sac_locatable_pre": selected_rollup(pre_result, "pre", "sac_locatable", "uncertain"),
                "sac_locatable_post": selected_rollup(post_result, "post", "sac_locatable", "uncertain"),
                "neck_visible_pre": selected_rollup(pre_result, "pre", "neck_visible", "uncertain"),
                "neck_visible_post": selected_rollup(post_result, "post", "neck_visible", "uncertain"),
                "parent_visible_pre": selected_rollup(pre_result, "pre", "parent_visible", "uncertain"),
                "parent_visible_post": selected_rollup(post_result, "post", "parent_visible", "uncertain"),
                "branch_visible_pre": selected_rollup(pre_result, "pre", "branch_visible", "uncertain"),
                "branch_visible_post": selected_rollup(post_result, "post", "branch_visible", "uncertain"),
                "roi_feasible_pre": selected_rollup(pre_result, "pre", "roi_feasible", "uncertain"),
                "roi_feasible_post": selected_rollup(post_result, "post", "roi_feasible", "uncertain"),
                "pre_post_same_lesion": lesion_result["pre_post_same_lesion"],
                "pre_post_view_comparable": lesion_result["pre_post_view_comparable"],
                "ai_status": lesion_result["ai_status"],
                "confidence": confidence,
                "evidence_frame_indices_pre": join_values(
                    selected_rollup(pre_result, "pre", "evidence_frame_indices", [])
                ),
                "evidence_frame_indices_post": join_values(
                    selected_rollup(post_result, "post", "evidence_frame_indices", [])
                ),
                "reason": lesion_result["reason"],
                "manual_review_required": int(manual_required),
                "manual_review_priority_score": priority,
                "manual_review_priority_rank": 0,
                "qc_failure_reasons": join_values(failures),
                "post_nonopacified_but_locatable_qc": int(post_nonopacified_locatable),
                "pre_annotation_path": pre_annotation,
                "post_annotation_path": post_annotation,
                "lesion_response_path": str(lesion_paths[uid]),
            }
        )

    priority_order = sorted(
        range(len(decision_rows)),
        key=lambda index: (
            -float(decision_rows[index]["manual_review_priority_score"]),
            decision_rows[index]["lesion_uid"],
        ),
    )
    for rank, index in enumerate(priority_order, start=1):
        decision_rows[index]["manual_review_priority_rank"] = rank
    candidate_frame = pd.DataFrame(candidate_rows, columns=CANDIDATE_COLUMNS)
    lesion_frame = pd.DataFrame(decision_rows, columns=LESION_COLUMNS)
    if len(candidate_frame) != 42 or len(lesion_frame) != 30:
        raise AssertionError("AI output scale mismatch")
    if not set(lesion_frame["ai_status"]).issubset(AI_STATUSES):
        raise AssertionError("Unexpected AI status")
    atomic_write_csv(candidate_frame, candidate_output)
    atomic_write_csv(lesion_frame, lesion_output)
    render_html(
        html_root,
        lesions,
        candidate_results,
        decision_rows,
        annotation_root,
    )
    print(
        json.dumps(
            {
                "candidate_review_rows": len(candidate_frame),
                "lesion_decision_rows": len(lesion_frame),
                "ai_status_counts": lesion_frame["ai_status"].value_counts().to_dict(),
                "manual_review_required": int(lesion_frame["manual_review_required"].sum()),
                "annotation_lesions": int(
                    lesion_frame["ai_status"].isin(
                        ["ai_exact_candidate", "ai_probable_candidate"]
                    ).sum()
                ),
                "candidate_output": str(candidate_output),
                "lesion_output": str(lesion_output),
                "annotation_root": str(annotation_root),
                "html": str(html_root / "index.html"),
                "human_reviewed_status_written": False,
                "formal_local_features_generated": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

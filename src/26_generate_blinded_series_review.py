#!/usr/bin/env python3
"""Generate the blinded api_fullseq_v3 local series-review package.

Inputs are limited to the two blinded lesion registries and their frozen
candidate/frame-path payloads.  The restricted private label artifact is never
opened.  No reviewed decision is inferred or populated by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT = Path("/root/autodl-tmp/aneurysm")
QC_SAMPLE_QUOTAS = {"Train": 96, "Valid": 24}
QC_SEED = "api_fullseq_v3_blinded_series_qc_v1"
MAX_FRAMES_PER_INTERNAL_SERIES = 5
THUMBNAIL_SIZE = (180, 180)

FORBIDDEN_OUTPUT_TOKENS = (
    "rroc",
    "adverse",
    "followup_time",
    "follow-up",
    "follow up",
    "随访",
    "不良转归",
    "术后即刻",
    "exact_reviewed",
    "probable_reviewed",
)

REVIEW_COLUMNS = [
    "review_item_id",
    "review_scope",
    "sampling_stratum",
    "sampling_hash",
    "lesion_uid",
    "split",
    "patient_id",
    "source_excel_row_id",
    "side_raw",
    "side_normalized",
    "location_raw",
    "location_normalized",
    "lesion_index_normalized",
    "multiple_aneurysm_normalized",
    "current_registration_status",
    "candidate_index_for_lesion",
    "candidate_count_for_lesion",
    "candidate_source_type",
    "candidate_source_root",
    "candidate_discovery_rank",
    "candidate_series_id",
    "candidate_series_path",
    "candidate_is_fixed_target",
    "candidate_valid",
    "candidate_selected_in_v2",
    "candidate_selection_status_in_v2",
    "candidate_exclusion_reason",
    "pre_api_dir",
    "pre_internal_series",
    "pre_selected_internal_series_in_v2",
    "pre_selected_n_frames",
    "pre_selected_n_contiguous_pairs",
    "post_api_dir",
    "post_internal_series",
    "post_selected_internal_series_in_v2",
    "post_selected_n_frames",
    "post_selected_n_contiguous_pairs",
    "pre_contact_sheet",
    "post_contact_sheet",
    "lesion_review_page",
    "reviewer_candidate_visual_match",
    "reviewer_candidate_rank",
    "reviewer_candidate_notes",
]

DECISION_COLUMNS = [
    "lesion_uid",
    "split",
    "patient_id",
    "source_excel_row_id",
    "side_raw",
    "side_normalized",
    "location_raw",
    "location_normalized",
    "lesion_index_normalized",
    "multiple_aneurysm_normalized",
    "current_registration_status",
    "review_scope",
    "sampling_stratum",
    "sampling_hash",
    "candidate_count",
    "review_item_ids",
    "lesion_review_page",
    "manual_decision_status",
    "chosen_candidate_discovery_rank",
    "chosen_candidate_series_id",
    "chosen_candidate_series_path",
    "chosen_pre_internal_series",
    "chosen_post_internal_series",
    "reviewer_id",
    "reviewed_at_utc",
    "review_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--max-frames-per-internal-series", type=int, default=5)
    return parser.parse_args()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def natural_key(value: str) -> tuple[tuple[int, Any], ...]:
    parts = re.split(r"(\d+)", str(value).casefold())
    return tuple((0, int(part)) if part.isdigit() else (1, part) for part in parts)


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return cleaned.strip("_") or "item"


def stable_hash(value: str) -> str:
    return sha256_text(f"{QC_SEED}|{value}")


def split_pipe(value: Any) -> list[str]:
    text = "" if value is None else str(value)
    return [part for part in text.split("|") if part]


def parse_candidate_payload(value: str) -> dict[str, Any]:
    payload = json.loads(value)
    if not isinstance(payload, dict) or not isinstance(payload.get("candidates"), list):
        raise ValueError("Malformed candidate_series_registry_json")
    return payload


def selected_internal_dimensions(phase: dict[str, Any]) -> str:
    selected = str(phase.get("selected_internal_series_in_v2", ""))
    for item in phase.get("internal_series_audit", []):
        if str(item.get("internal_series_number", "")) == selected:
            dimensions = item.get("dimensions", [])
            return "|".join(str(value) for value in dimensions)
    return ""


def primary_candidate(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = payload["candidates"]
    selected = [
        candidate
        for candidate in candidates
        if candidate["candidate_audit"].get("selected_candidate_in_v2")
    ]
    if selected:
        return selected[0]
    valid = [candidate for candidate in candidates if candidate["candidate_audit"].get("candidate_valid")]
    return valid[0] if valid else candidates[0]


def frame_count_bin(total: int) -> str:
    if total <= 30:
        return "frames_low"
    if total <= 70:
        return "frames_medium"
    return "frames_high"


def dimension_class(candidate: dict[str, Any]) -> str:
    pre = selected_internal_dimensions(candidate["pre"])
    post = selected_internal_dimensions(candidate["post"])
    if not pre and not post:
        return "dimensions_unknown"
    if pre == post:
        return f"dimensions_same_{safe_name(pre)}"
    return f"dimensions_mixed_{safe_name(pre or 'none')}_{safe_name(post or 'none')}"


def sampling_stratum(row: pd.Series) -> str:
    payload = parse_candidate_payload(row["candidate_series_registry_json"])
    candidate = primary_candidate(payload)
    pre_frames = int(candidate["pre"].get("n_frames_in_selected_internal_series") or 0)
    post_frames = int(candidate["post"].get("n_frames_in_selected_internal_series") or 0)
    availability = (
        "prepost"
        if pre_frames >= 2 and post_frames >= 2
        else "post_only"
        if post_frames >= 2
        else "other_availability"
    )
    source = candidate["candidate_source"].get("source_type", "unknown") or "unknown"
    return "|".join(
        [
            str(row["split"]),
            str(row["location_normalized"]),
            str(source),
            availability,
            frame_count_bin(pre_frames + post_frames),
            dimension_class(candidate),
        ]
    )


def stratified_sample(provisional: pd.DataFrame) -> pd.DataFrame:
    selected_indices: list[int] = []
    for split, quota in QC_SAMPLE_QUOTAS.items():
        split_frame = provisional.loc[provisional["split"] == split].copy()
        split_frame["sampling_stratum"] = split_frame.apply(sampling_stratum, axis=1)
        split_frame["sampling_hash"] = split_frame["lesion_uid"].map(stable_hash)
        groups: dict[str, list[int]] = {}
        for stratum, group in split_frame.groupby("sampling_stratum", sort=False):
            ordered = group.sort_values(["sampling_hash", "lesion_uid"], kind="stable")
            groups[stratum] = ordered.index.tolist()
        stratum_order = sorted(groups, key=lambda name: (len(groups[name]), natural_key(name)))
        offsets = defaultdict(int)
        while len([index for index in selected_indices if provisional.loc[index, "split"] == split]) < quota:
            progressed = False
            for stratum in stratum_order:
                offset = offsets[stratum]
                if offset >= len(groups[stratum]):
                    continue
                selected_indices.append(groups[stratum][offset])
                offsets[stratum] += 1
                progressed = True
                split_selected = sum(
                    provisional.loc[index, "split"] == split for index in selected_indices
                )
                if split_selected >= quota:
                    break
            if not progressed:
                raise AssertionError(f"Unable to draw {quota} provisional QC rows for {split}")
    sampled = provisional.loc[selected_indices].copy()
    sampled["sampling_stratum"] = sampled.apply(sampling_stratum, axis=1)
    sampled["sampling_hash"] = sampled["lesion_uid"].map(stable_hash)
    return sampled.sort_values(["split", "sampling_stratum", "sampling_hash"], kind="stable")


def choose_evenly(paths: list[str], limit: int) -> list[str]:
    if len(paths) <= limit:
        return paths
    if limit <= 1:
        return [paths[len(paths) // 2]]
    indices = sorted({round(index * (len(paths) - 1) / (limit - 1)) for index in range(limit)})
    return [paths[index] for index in indices]


def frame_label(path: str) -> str:
    match = re.search(r"IMG-(\d+)-(\d+)\.(?:jpg|jpeg)$", Path(path).name, re.IGNORECASE)
    return f"f={int(match.group(2))}" if match else "frame"


def load_thumbnail(path: str, size: tuple[int, int]) -> tuple[Image.Image, str]:
    try:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("L")
            image = ImageOps.autocontrast(image).convert("RGB")
            image.thumbnail(size, Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", size, "black")
            left = (size[0] - image.width) // 2
            top = (size[1] - image.height) // 2
            canvas.paste(image, (left, top))
            return canvas, ""
    except Exception as exc:
        canvas = Image.new("RGB", size, (60, 20, 20))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), "unreadable", fill="white", font=ImageFont.load_default())
        return canvas, f"{type(exc).__name__}: {exc}"


def render_contact_sheet(
    candidate: dict[str, Any],
    phase_name: str,
    output_path: Path,
    max_frames: int,
) -> tuple[str, list[str]]:
    phase = candidate[phase_name]
    paths_by_internal = phase.get("strict_frame_paths_by_internal_series", {})
    nonempty = [
        (str(internal), list(paths))
        for internal, paths in paths_by_internal.items()
        if paths
    ]
    if not nonempty:
        return "", []

    nonempty.sort(key=lambda item: natural_key(item[0]))
    margin = 12
    label_width = 150
    cell_width = THUMBNAIL_SIZE[0] + 12
    row_height = THUMBNAIL_SIZE[1] + 42
    width = margin * 2 + label_width + max_frames * cell_width
    height = margin * 2 + 34 + len(nonempty) * row_height
    sheet = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    source = candidate["candidate_source"]
    header = (
        f"patient={candidate.get('_patient_id','')} rank={source.get('discovery_rank','')} "
        f"phase={phase_name}"
    )
    draw.text((margin, margin), header, fill="white", font=font)
    errors: list[str] = []
    selected_internal = str(phase.get("selected_internal_series_in_v2", ""))

    for row_number, (internal, paths) in enumerate(nonempty):
        top = margin + 34 + row_number * row_height
        selected_mark = "selected-v2" if internal == selected_internal else "candidate"
        draw.text(
            (margin, top + 6),
            f"internal={internal}\n{selected_mark}\nn={len(paths)}",
            fill=(120, 255, 120) if internal == selected_internal else (220, 220, 220),
            font=font,
        )
        for column, path in enumerate(choose_evenly(paths, max_frames)):
            thumb, error = load_thumbnail(path, THUMBNAIL_SIZE)
            left = margin + label_width + column * cell_width
            sheet.paste(thumb, (left, top))
            border = (40, 200, 80) if internal == selected_internal else (120, 120, 120)
            draw.rectangle(
                [left, top, left + THUMBNAIL_SIZE[0] - 1, top + THUMBNAIL_SIZE[1] - 1],
                outline=border,
                width=2,
            )
            draw.text(
                (left + 4, top + THUMBNAIL_SIZE[1] + 6),
                frame_label(path),
                fill="white",
                font=font,
            )
            if error:
                errors.append(f"{path}: {error}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_name(f".{output_path.name}.tmp")
    sheet.save(temp, format="JPEG", quality=88, optimize=True)
    os.replace(temp, output_path)
    return output_path.as_posix(), errors


def candidate_asset_key(split: str, patient_id: str, candidate: dict[str, Any]) -> str:
    source = candidate["candidate_source"]
    material = "|".join(
        [
            split,
            patient_id,
            str(source.get("discovery_rank", "")),
            str(source.get("series_id", "")),
            str(source.get("series_path", "")),
        ]
    )
    return sha256_text(material)[:16]


def review_item_id(lesion_uid: str, candidate: dict[str, Any]) -> str:
    source = candidate["candidate_source"]
    material = "|".join(
        [
            lesion_uid,
            str(source.get("discovery_rank", "")),
            str(source.get("series_id", "")),
            str(source.get("series_path", "")),
        ]
    )
    return f"review_{sha256_text(material)[:20]}"


def pipe_list(values: Iterable[Any]) -> str:
    return "|".join(str(value) for value in values if str(value))


def html_page(
    lesion: pd.Series,
    scope: str,
    stratum: str,
    candidate_cards: list[dict[str, Any]],
) -> str:
    title = f"Blinded series review: {lesion['lesion_uid']}"
    cards: list[str] = []
    for card in candidate_cards:
        pre_image = (
            f'<img src="../{html.escape(card["pre_asset_relative"])}" alt="Pre candidate contact sheet">'
            if card["pre_asset_relative"]
            else '<div class="missing">No frozen Pre frames</div>'
        )
        post_image = (
            f'<img src="../{html.escape(card["post_asset_relative"])}" alt="Post candidate contact sheet">'
            if card["post_asset_relative"]
            else '<div class="missing">No frozen Post frames</div>'
        )
        cards.append(
            f"""
            <section class="candidate">
              <h2>Candidate {card['candidate_index']} / {card['candidate_count']}</h2>
              <dl>
                <dt>Review item</dt><dd>{html.escape(card['review_item_id'])}</dd>
                <dt>Source</dt><dd>{html.escape(card['source_type'])}</dd>
                <dt>Rank</dt><dd>{html.escape(card['rank'])}</dd>
                <dt>Series</dt><dd>{html.escape(card['series_id'])}</dd>
                <dt>Path</dt><dd class="path">{html.escape(card['series_path'])}</dd>
                <dt>Candidate valid</dt><dd>{card['candidate_valid']}</dd>
                <dt>Selected in v2</dt><dd>{card['selected_in_v2']}</dd>
              </dl>
              <div class="phase"><h3>Pre</h3>{pre_image}</div>
              <div class="phase"><h3>Post</h3>{post_image}</div>
              <div class="entry">
                <label>Visual match <input type="text" placeholder="manual entry in CSV"></label>
                <label>Candidate rank <input type="text" placeholder="manual entry in CSV"></label>
                <label>Notes <textarea placeholder="manual entry in CSV"></textarea></label>
              </div>
            </section>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; background: #f4f5f7; color: #17202a; }}
    a {{ color: #145a8d; }} .meta, .candidate {{ background: white; padding: 18px; margin: 14px 0; border-radius: 8px; box-shadow: 0 1px 5px #ccd; }}
    dl {{ display: grid; grid-template-columns: 180px 1fr; gap: 5px 12px; }} dt {{ font-weight: 700; }} dd {{ margin: 0; }}
    .path {{ overflow-wrap: anywhere; }} .phase {{ margin: 14px 0; }} img {{ max-width: 100%; height: auto; border: 1px solid #555; }}
    .missing {{ padding: 28px; background: #eee; color: #555; }} .entry {{ display: grid; gap: 8px; }} label {{ display: grid; gap: 4px; }} textarea {{ min-height: 70px; }}
  </style>
</head>
<body>
  <p><a href="../index.html">Back to review index</a></p>
  <h1>{html.escape(title)}</h1>
  <section class="meta">
    <dl>
      <dt>Scope</dt><dd>{html.escape(scope)}</dd>
      <dt>Split</dt><dd>{html.escape(str(lesion['split']))}</dd>
      <dt>Patient</dt><dd>{html.escape(str(lesion['patient_id']))}</dd>
      <dt>Side</dt><dd>{html.escape(str(lesion['side_normalized']))}</dd>
      <dt>Location</dt><dd>{html.escape(str(lesion['location_normalized']))}</dd>
      <dt>Lesion index</dt><dd>{html.escape(str(lesion['lesion_index_normalized']))}</dd>
      <dt>Current status</dt><dd>{html.escape(str(lesion['registration_status']))}</dd>
      <dt>Sampling stratum</dt><dd>{html.escape(stratum)}</dd>
    </dl>
  </section>
  {''.join(cards)}
</body>
</html>
"""


def index_page(decisions: pd.DataFrame) -> str:
    rows = []
    for row in decisions.itertuples(index=False):
        rows.append(
            f"<tr data-split=\"{html.escape(row.split)}\" data-scope=\"{html.escape(row.review_scope)}\">"
            f"<td><a href=\"{html.escape(row.lesion_review_page.replace('review_app/', ''))}\">{html.escape(row.lesion_uid)}</a></td>"
            f"<td>{html.escape(row.split)}</td><td>{html.escape(row.review_scope)}</td>"
            f"<td>{html.escape(row.location_normalized)}</td><td>{row.candidate_count}</td>"
            f"<td>{html.escape(row.current_registration_status)}</td></tr>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>api_fullseq_v3 blinded series review</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; }} table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 7px; text-align: left; }} th {{ background: #eef2f5; position: sticky; top: 0; }}
    .filters {{ display: flex; gap: 12px; margin-bottom: 14px; }}
  </style>
</head>
<body>
  <h1>Blinded series review</h1>
  <p>Candidate-level visual review only. Enter reviewer results in the generated CSV tables.</p>
  <div class="filters">
    <label>Split <select id="split"><option value="">All</option><option>Train</option><option>Valid</option></select></label>
    <label>Scope <select id="scope"><option value="">All</option><option>required_review</option><option>provisional_qc</option></select></label>
  </div>
  <table><thead><tr><th>Lesion UID</th><th>Split</th><th>Scope</th><th>Location</th><th>Candidates</th><th>Current status</th></tr></thead>
  <tbody>{''.join(rows)}</tbody></table>
  <script>
    const split = document.getElementById('split'); const scope = document.getElementById('scope');
    function applyFilter() {{ document.querySelectorAll('tbody tr').forEach(row => {{
      row.hidden = (split.value && row.dataset.split !== split.value) || (scope.value && row.dataset.scope !== scope.value);
    }}); }} split.addEventListener('change', applyFilter); scope.addEventListener('change', applyFilter);
  </script>
</body>
</html>
"""


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    df.to_csv(temp, index=False, encoding="utf-8", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    os.replace(temp, path)


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def assert_blinded_text(text: str, name: str) -> None:
    folded = text.casefold()
    found = [token for token in FORBIDDEN_OUTPUT_TOKENS if token.casefold() in folded]
    if found:
        raise AssertionError(f"Forbidden token(s) in {name}: {found}")


def preserve_manual_fields(new: pd.DataFrame, old_path: Path, key: str, fields: list[str]) -> pd.DataFrame:
    if not old_path.is_file():
        return new
    old = pd.read_csv(old_path, dtype=str, keep_default_na=False)
    if key not in old.columns or old[key].duplicated().any():
        return new
    old_lookup = old.set_index(key)
    for field in fields:
        if field not in old_lookup.columns or field not in new.columns:
            continue
        new[field] = [
            old_lookup.at[item_key, field]
            if item_key in old_lookup.index and old_lookup.at[item_key, field] != ""
            else current
            for item_key, current in zip(new[key], new[field])
        ]
    return new


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    if args.max_frames_per_internal_series < 1:
        raise ValueError("--max-frames-per-internal-series must be positive")
    train = pd.read_csv(
        root / "metadata/api_fullseq_v3/lesion_registry_train_blinded.csv",
        dtype=str,
        keep_default_na=False,
    )
    valid = pd.read_csv(
        root / "metadata/api_fullseq_v3/lesion_registry_valid_blinded.csv",
        dtype=str,
        keep_default_na=False,
    )
    registry = pd.concat([train, valid], ignore_index=True)
    if len(registry) != 1446 or not registry["lesion_uid"].is_unique:
        raise AssertionError("Unexpected blinded registry scale or duplicate lesion_uid")

    required = registry.loc[registry["registration_status"] == "review_required"].copy()
    provisional = registry.loc[
        registry["registration_status"] == "provisional_single_lesion"
    ].copy()
    if len(required) != 234 or len(provisional) != 1212:
        raise AssertionError(
            f"Expected 234 review_required and 1212 provisional rows, got {len(required)} and {len(provisional)}"
        )
    required["review_scope"] = "required_review"
    required["sampling_stratum"] = "all_review_required"
    required["sampling_hash"] = required["lesion_uid"].map(stable_hash)
    sampled = stratified_sample(provisional)
    sampled["review_scope"] = "provisional_qc"
    review_lesions = pd.concat([required, sampled], ignore_index=True)
    if len(review_lesions) != 354 or not review_lesions["lesion_uid"].is_unique:
        raise AssertionError("Review lesion set must contain 354 unique lesions")

    report_dir = root / "reports/api_fullseq_v3"
    app_dir = report_dir / "review_app"
    assets_dir = app_dir / "assets"
    lesions_dir = app_dir / "lesions"
    review_csv_path = report_dir / "series_candidate_review.csv"
    decision_csv_path = report_dir / "lesion_series_decision.csv"
    app_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    lesions_dir.mkdir(parents=True, exist_ok=True)

    review_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    rendered_assets: dict[tuple[str, str], str] = {}
    asset_errors: list[str] = []

    for lesion in review_lesions.itertuples(index=False):
        lesion_series = pd.Series(lesion._asdict())
        payload = parse_candidate_payload(lesion.candidate_series_registry_json)
        candidates = payload["candidates"]
        page_name = f"{safe_name(lesion.lesion_uid)}.html"
        page_relative = f"review_app/lesions/{page_name}"
        candidate_cards: list[dict[str, Any]] = []
        item_ids: list[str] = []

        for candidate_index, candidate in enumerate(candidates, start=1):
            candidate["_patient_id"] = lesion.patient_id
            asset_key = candidate_asset_key(lesion.split, lesion.patient_id, candidate)
            source = candidate["candidate_source"]
            audit = candidate["candidate_audit"]
            phase_assets: dict[str, str] = {}
            for phase_name in ("pre", "post"):
                cache_key = (asset_key, phase_name)
                if cache_key not in rendered_assets:
                    filename = (
                        f"{safe_name(lesion.split)}_{safe_name(lesion.patient_id)}_"
                        f"r{safe_name(str(source.get('discovery_rank','')))}_{asset_key}_{phase_name}.jpg"
                    )
                    output_path = assets_dir / filename
                    rendered, errors = render_contact_sheet(
                        candidate,
                        phase_name,
                        output_path,
                        args.max_frames_per_internal_series,
                    )
                    rendered_assets[cache_key] = (
                        f"assets/{filename}" if rendered else ""
                    )
                    asset_errors.extend(errors)
                phase_assets[phase_name] = rendered_assets[cache_key]

            item_id = review_item_id(lesion.lesion_uid, candidate)
            item_ids.append(item_id)
            review_rows.append(
                {
                    "review_item_id": item_id,
                    "review_scope": lesion.review_scope,
                    "sampling_stratum": lesion.sampling_stratum,
                    "sampling_hash": lesion.sampling_hash,
                    "lesion_uid": lesion.lesion_uid,
                    "split": lesion.split,
                    "patient_id": lesion.patient_id,
                    "source_excel_row_id": lesion.source_excel_row_id,
                    "side_raw": lesion.side_raw,
                    "side_normalized": lesion.side_normalized,
                    "location_raw": lesion.location_raw,
                    "location_normalized": lesion.location_normalized,
                    "lesion_index_normalized": lesion.lesion_index_normalized,
                    "multiple_aneurysm_normalized": lesion.multiple_aneurysm_normalized,
                    "current_registration_status": lesion.registration_status,
                    "candidate_index_for_lesion": candidate_index,
                    "candidate_count_for_lesion": len(candidates),
                    "candidate_source_type": source.get("source_type", ""),
                    "candidate_source_root": source.get("source_medical_record_root", ""),
                    "candidate_discovery_rank": source.get("discovery_rank", ""),
                    "candidate_series_id": source.get("series_id", ""),
                    "candidate_series_path": source.get("series_path", ""),
                    "candidate_is_fixed_target": source.get("is_fixed_target", False),
                    "candidate_valid": audit.get("candidate_valid", False),
                    "candidate_selected_in_v2": audit.get("selected_candidate_in_v2", False),
                    "candidate_selection_status_in_v2": audit.get("selection_status_in_v2", ""),
                    "candidate_exclusion_reason": audit.get("candidate_exclusion_reason", ""),
                    "pre_api_dir": candidate["pre"].get("api_dir", ""),
                    "pre_internal_series": pipe_list(candidate["pre"].get("internal_series", [])),
                    "pre_selected_internal_series_in_v2": candidate["pre"].get(
                        "selected_internal_series_in_v2", ""
                    ),
                    "pre_selected_n_frames": candidate["pre"].get(
                        "n_frames_in_selected_internal_series", ""
                    ),
                    "pre_selected_n_contiguous_pairs": candidate["pre"].get(
                        "n_contiguous_pairs_in_selected_internal_series", ""
                    ),
                    "post_api_dir": candidate["post"].get("api_dir", ""),
                    "post_internal_series": pipe_list(candidate["post"].get("internal_series", [])),
                    "post_selected_internal_series_in_v2": candidate["post"].get(
                        "selected_internal_series_in_v2", ""
                    ),
                    "post_selected_n_frames": candidate["post"].get(
                        "n_frames_in_selected_internal_series", ""
                    ),
                    "post_selected_n_contiguous_pairs": candidate["post"].get(
                        "n_contiguous_pairs_in_selected_internal_series", ""
                    ),
                    "pre_contact_sheet": (
                        f"review_app/{phase_assets['pre']}" if phase_assets["pre"] else ""
                    ),
                    "post_contact_sheet": (
                        f"review_app/{phase_assets['post']}" if phase_assets["post"] else ""
                    ),
                    "lesion_review_page": page_relative,
                    "reviewer_candidate_visual_match": "",
                    "reviewer_candidate_rank": "",
                    "reviewer_candidate_notes": "",
                }
            )
            candidate_cards.append(
                {
                    "review_item_id": item_id,
                    "candidate_index": candidate_index,
                    "candidate_count": len(candidates),
                    "source_type": str(source.get("source_type", "")),
                    "rank": str(source.get("discovery_rank", "")),
                    "series_id": str(source.get("series_id", "")),
                    "series_path": str(source.get("series_path", "")),
                    "candidate_valid": audit.get("candidate_valid", False),
                    "selected_in_v2": audit.get("selected_candidate_in_v2", False),
                    "pre_asset_relative": phase_assets["pre"],
                    "post_asset_relative": phase_assets["post"],
                }
            )

        page_text = html_page(
            lesion_series,
            lesion.review_scope,
            lesion.sampling_stratum,
            candidate_cards,
        )
        assert_blinded_text(page_text, page_name)
        atomic_write_text(page_text, lesions_dir / page_name)
        decision_rows.append(
            {
                "lesion_uid": lesion.lesion_uid,
                "split": lesion.split,
                "patient_id": lesion.patient_id,
                "source_excel_row_id": lesion.source_excel_row_id,
                "side_raw": lesion.side_raw,
                "side_normalized": lesion.side_normalized,
                "location_raw": lesion.location_raw,
                "location_normalized": lesion.location_normalized,
                "lesion_index_normalized": lesion.lesion_index_normalized,
                "multiple_aneurysm_normalized": lesion.multiple_aneurysm_normalized,
                "current_registration_status": lesion.registration_status,
                "review_scope": lesion.review_scope,
                "sampling_stratum": lesion.sampling_stratum,
                "sampling_hash": lesion.sampling_hash,
                "candidate_count": len(candidates),
                "review_item_ids": pipe_list(item_ids),
                "lesion_review_page": page_relative,
                "manual_decision_status": "",
                "chosen_candidate_discovery_rank": "",
                "chosen_candidate_series_id": "",
                "chosen_candidate_series_path": "",
                "chosen_pre_internal_series": "",
                "chosen_post_internal_series": "",
                "reviewer_id": "",
                "reviewed_at_utc": "",
                "review_notes": "",
            }
        )

    review_frame = pd.DataFrame(review_rows, columns=REVIEW_COLUMNS)
    decision_frame = pd.DataFrame(decision_rows, columns=DECISION_COLUMNS)
    review_frame = preserve_manual_fields(
        review_frame,
        review_csv_path,
        "review_item_id",
        [
            "reviewer_candidate_visual_match",
            "reviewer_candidate_rank",
            "reviewer_candidate_notes",
        ],
    )
    decision_frame = preserve_manual_fields(
        decision_frame,
        decision_csv_path,
        "lesion_uid",
        [
            "manual_decision_status",
            "chosen_candidate_discovery_rank",
            "chosen_candidate_series_id",
            "chosen_candidate_series_path",
            "chosen_pre_internal_series",
            "chosen_post_internal_series",
            "reviewer_id",
            "reviewed_at_utc",
            "review_notes",
        ],
    )

    if not review_frame["review_item_id"].is_unique:
        raise AssertionError("Duplicate review_item_id")
    if not decision_frame["lesion_uid"].is_unique or len(decision_frame) != 354:
        raise AssertionError("Decision table scale/UID uniqueness failure")
    if set(required["lesion_uid"]) - set(decision_frame["lesion_uid"]):
        raise AssertionError("Not all review_required lesions entered the review system")
    if (decision_frame["review_scope"] == "provisional_qc").sum() != 120:
        raise AssertionError("Provisional QC sample must contain 120 lesions")
    if decision_frame["manual_decision_status"].ne("").any():
        raise AssertionError("This automatic run must not populate manual decision status")

    review_text = review_frame.to_csv(index=False, lineterminator="\n")
    decision_text = decision_frame.to_csv(index=False, lineterminator="\n")
    index_text = index_page(decision_frame)
    assert_blinded_text(review_text, review_csv_path.name)
    assert_blinded_text(decision_text, decision_csv_path.name)
    assert_blinded_text(index_text, "index.html")

    atomic_write_csv(review_frame, review_csv_path)
    atomic_write_csv(decision_frame, decision_csv_path)
    atomic_write_text(index_text, app_dir / "index.html")

    log = {
        "review_required_lesions": 234,
        "provisional_qc_lesions": 120,
        "total_review_lesions": len(decision_frame),
        "candidate_review_rows": len(review_frame),
        "unique_candidate_assets": len({key[0] for key in rendered_assets}),
        "contact_sheet_files": sum(bool(path) for path in rendered_assets.values()),
        "image_read_errors": len(asset_errors),
        "automatic_manual_decisions_written": 0,
        "restricted_private_artifact_accessed": False,
        "qc_seed": QC_SEED,
        "qc_split_quotas": QC_SAMPLE_QUOTAS,
    }
    log_text = json.dumps(log, ensure_ascii=False, indent=2)
    assert_blinded_text(log_text, "generation_log.json")
    atomic_write_text(log_text, app_dir / "generation_log.json")
    if asset_errors:
        error_text = "\n".join(asset_errors)
        assert_blinded_text(error_text, "image_read_errors.log")
        atomic_write_text(error_text, app_dir / "image_read_errors.log")

    print(json.dumps(log, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

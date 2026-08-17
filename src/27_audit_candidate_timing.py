#!/usr/bin/env python3
"""Audit frozen candidate-series timing, frame integrity, and global peak QC.

This stage reads only blinded registries and their 56,641 frozen frame paths.
It does not read private labels, run SEA-RAFT, calculate ROI features, or train
models.  Full-frame intensity-change peaks are provisional censoring QC only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd

try:
    import pydicom  # type: ignore
except ImportError:  # pragma: no cover - current frozen paths are JPEG files.
    pydicom = None


PROJECT = Path("/root/autodl-tmp/aneurysm")
EXPECTED_CANDIDATES = 1462
EXPECTED_STRICT_FRAME_PATHS = 56641
FRAME_RE = re.compile(r"IMG-(\d+)-(\d+)\.(?:jpg|jpeg)$", re.IGNORECASE)

FORBIDDEN_OUTPUT_TOKENS = (
    "rroc",
    "adverse",
    "followup_time",
    "follow-up",
    "follow up",
    "随访",
    "不良转归",
    "术后即刻",
    "washout",
)

COLUMNS = [
    "candidate_uid",
    "split",
    "patient_id",
    "candidate_source_type",
    "candidate_source_root",
    "candidate_discovery_rank",
    "candidate_series_id",
    "candidate_series_path",
    "candidate_valid",
    "candidate_selected_in_v2",
    "candidate_selection_status_in_v2",
    "candidate_exclusion_reason",
    "frozen_strict_frame_path_count",
    "pre_all_strict_frame_path_count",
    "pre_internal_series_count",
    "pre_selected_internal_series",
    "pre_selected_n_frames",
    "pre_selected_n_unique_frame_indices",
    "pre_selected_frame_index_min",
    "pre_selected_frame_index_max",
    "pre_selected_frame_span_frames",
    "pre_selected_missing_frame_count",
    "pre_selected_missing_frame_ranges",
    "pre_selected_contiguous_pair_count",
    "pre_selected_gap_transition_count",
    "pre_selected_duplicate_frame_index_count",
    "pre_selected_unreadable_frame_count",
    "pre_selected_dimensions",
    "pre_fps",
    "pre_frame_time_ms",
    "pre_duration_seconds",
    "pre_timing_source",
    "pre_timing_reliability",
    "global_pre_peak_frame_index_provisional",
    "global_pre_peak_signal_provisional",
    "global_pre_peak_direction_provisional",
    "global_pre_peak_time_seconds_provisional",
    "global_pre_peak_left_censored_qc",
    "global_pre_peak_right_censored_qc",
    "global_pre_peak_near_boundary_qc",
    "global_pre_peak_readable_frame_count_qc",
    "pre_internal_series_timing_json",
    "post_all_strict_frame_path_count",
    "post_internal_series_count",
    "post_selected_internal_series",
    "post_selected_n_frames",
    "post_selected_n_unique_frame_indices",
    "post_selected_frame_index_min",
    "post_selected_frame_index_max",
    "post_selected_frame_span_frames",
    "post_selected_missing_frame_count",
    "post_selected_missing_frame_ranges",
    "post_selected_contiguous_pair_count",
    "post_selected_gap_transition_count",
    "post_selected_duplicate_frame_index_count",
    "post_selected_unreadable_frame_count",
    "post_selected_dimensions",
    "post_fps",
    "post_frame_time_ms",
    "post_duration_seconds",
    "post_timing_source",
    "post_timing_reliability",
    "global_post_peak_frame_index_provisional",
    "global_post_peak_signal_provisional",
    "global_post_peak_direction_provisional",
    "global_post_peak_time_seconds_provisional",
    "global_post_peak_left_censored_qc",
    "global_post_peak_right_censored_qc",
    "global_post_peak_near_boundary_qc",
    "global_post_peak_readable_frame_count_qc",
    "post_internal_series_timing_json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--progress-every", type=int, default=5000)
    return parser.parse_args()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def candidate_uid(split: str, patient_id: str, candidate: dict[str, Any]) -> str:
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
    return f"candidate_{sha256_text(material)[:24]}"


def natural_key(value: str) -> tuple[tuple[int, Any], ...]:
    parts = re.split(r"(\d+)", str(value).casefold())
    return tuple((0, int(part)) if part.isdigit() else (1, part) for part in parts)


def frame_index_from_path(path: str) -> int | None:
    match = FRAME_RE.fullmatch(Path(path).name)
    return int(match.group(2)) if match else None


def dicom_timing(path: str) -> tuple[float, float, str, str]:
    if pydicom is None or Path(path).suffix.casefold() not in {".dcm", ".dicom"}:
        return math.nan, math.nan, "none_reliable", "unavailable"
    try:
        dataset = pydicom.dcmread(path, stop_before_pixels=True, force=False)
    except Exception:
        return math.nan, math.nan, "none_reliable", "unavailable"
    frame_time = getattr(dataset, "FrameTime", None)
    try:
        frame_time_ms = float(frame_time)
    except (TypeError, ValueError):
        frame_time_ms = math.nan
    if math.isfinite(frame_time_ms) and frame_time_ms > 0:
        return 1000.0 / frame_time_ms, frame_time_ms, "dicom_frame_time", "reliable"
    for attribute, source in (
        ("CineRate", "dicom_cine_rate"),
        ("RecommendedDisplayFrameRate", "dicom_recommended_display_frame_rate"),
    ):
        value = getattr(dataset, attribute, None)
        try:
            fps = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fps) and fps > 0:
            return fps, 1000.0 / fps, source, "reliable"
    return math.nan, math.nan, "none_reliable", "unavailable"


def inspect_frame(path: str) -> dict[str, Any]:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return {
            "path": path,
            "readable": False,
            "height": math.nan,
            "width": math.nan,
            "global_mean": math.nan,
            "fps": math.nan,
            "frame_time_ms": math.nan,
            "timing_source": "none_reliable",
            "timing_reliability": "unavailable",
        }
    fps, frame_time_ms, source, reliability = dicom_timing(path)
    return {
        "path": path,
        "readable": True,
        "height": int(image.shape[0]),
        "width": int(image.shape[1]),
        "global_mean": float(cv2.mean(image)[0]),
        "fps": fps,
        "frame_time_ms": frame_time_ms,
        "timing_source": source,
        "timing_reliability": reliability,
    }


def missing_ranges(indices: list[int]) -> tuple[int, list[str], int]:
    if len(indices) < 2:
        return 0, [], 0
    ranges: list[str] = []
    missing_count = 0
    gap_transitions = 0
    for left, right in zip(indices, indices[1:]):
        if right - left <= 1:
            continue
        gap_transitions += 1
        first = left + 1
        last = right - 1
        missing_count += last - first + 1
        ranges.append(str(first) if first == last else f"{first}-{last}")
    return missing_count, ranges, gap_transitions


def consistent_reliable_timing(
    frame_stats: list[dict[str, Any]],
) -> tuple[float, float, str, str]:
    readable = [item for item in frame_stats if item["readable"]]
    reliable = [
        item
        for item in readable
        if item["timing_reliability"] == "reliable" and math.isfinite(item["fps"])
    ]
    if not readable or not reliable:
        return math.nan, math.nan, "none_reliable", "unavailable"
    if len(reliable) != len(readable):
        return math.nan, math.nan, "partial_metadata_rejected", "unreliable"
    fps_values = np.asarray([item["fps"] for item in reliable], dtype=float)
    frame_time_values = np.asarray([item["frame_time_ms"] for item in reliable], dtype=float)
    sources = {item["timing_source"] for item in reliable}
    if len(sources) != 1 or not np.allclose(fps_values, fps_values[0], rtol=1e-6, atol=1e-8):
        return math.nan, math.nan, "inconsistent_metadata_rejected", "unreliable"
    if not np.allclose(frame_time_values, frame_time_values[0], rtol=1e-6, atol=1e-8):
        return math.nan, math.nan, "inconsistent_metadata_rejected", "unreliable"
    return float(fps_values[0]), float(frame_time_values[0]), next(iter(sources)), "reliable"


def sequence_metrics(paths: list[str], stats_by_path: dict[str, dict[str, Any]]) -> dict[str, Any]:
    indexed = [(frame_index_from_path(path), path) for path in paths]
    indexed.sort(
        key=lambda item: (
            item[0] is None,
            item[0] if item[0] is not None else 10**12,
            natural_key(item[1]),
        )
    )
    frame_stats = [stats_by_path[path] for _, path in indexed]
    valid_indices = [index for index, _ in indexed if index is not None]
    unique_indices = sorted(set(valid_indices))
    duplicate_count = len(valid_indices) - len(unique_indices)
    contiguous_pairs = sum(
        1 for left, right in zip(unique_indices, unique_indices[1:]) if right - left == 1
    )
    missing_count, ranges, gap_transitions = missing_ranges(unique_indices)
    frame_min = unique_indices[0] if unique_indices else math.nan
    frame_max = unique_indices[-1] if unique_indices else math.nan
    frame_span = frame_max - frame_min if unique_indices else math.nan
    dimensions = sorted(
        {
            f"{int(item['height'])}x{int(item['width'])}"
            for item in frame_stats
            if item["readable"]
        },
        key=natural_key,
    )
    fps, frame_time_ms, timing_source, timing_reliability = consistent_reliable_timing(
        frame_stats
    )
    duration_seconds = (
        float(frame_span) / fps
        if math.isfinite(frame_span) and math.isfinite(fps) and fps > 0
        else math.nan
    )

    readable_curve = [
        (index, item["global_mean"])
        for (index, _), item in zip(indexed, frame_stats)
        if index is not None and item["readable"] and math.isfinite(item["global_mean"])
    ]
    if readable_curve:
        baseline = readable_curve[0][1]
        changes = [abs(value - baseline) for _, value in readable_curve]
        peak_position = int(np.argmax(np.asarray(changes, dtype=float)))
        peak_index, peak_mean = readable_curve[peak_position]
        peak_signal = float(changes[peak_position])
        direction = (
            "increase" if peak_mean > baseline else "decrease" if peak_mean < baseline else "flat"
        )
        left_censored = peak_position == 0
        right_censored = peak_position == len(readable_curve) - 1
        near_boundary = peak_position <= 1 or peak_position >= len(readable_curve) - 2
        peak_time = (
            (peak_index - readable_curve[0][0]) / fps
            if math.isfinite(fps) and fps > 0
            else math.nan
        )
    else:
        peak_index = math.nan
        peak_signal = math.nan
        direction = "unavailable"
        peak_time = math.nan
        left_censored = False
        right_censored = False
        near_boundary = False

    return {
        "n_frames": len(paths),
        "n_unique_frame_indices": len(unique_indices),
        "frame_index_min": frame_min,
        "frame_index_max": frame_max,
        "frame_span_frames": frame_span,
        "missing_frame_count": missing_count,
        "missing_frame_ranges": "|".join(ranges),
        "contiguous_pair_count": contiguous_pairs,
        "gap_transition_count": gap_transitions,
        "duplicate_frame_index_count": duplicate_count,
        "unreadable_frame_count": sum(not item["readable"] for item in frame_stats),
        "dimensions": "|".join(dimensions),
        "fps": fps,
        "frame_time_ms": frame_time_ms,
        "duration_seconds": duration_seconds,
        "timing_source": timing_source,
        "timing_reliability": timing_reliability,
        "global_peak_frame_index_provisional": peak_index,
        "global_peak_signal_provisional": peak_signal,
        "global_peak_direction_provisional": direction,
        "global_peak_time_seconds_provisional": peak_time,
        "global_peak_left_censored_qc": left_censored,
        "global_peak_right_censored_qc": right_censored,
        "global_peak_near_boundary_qc": near_boundary,
        "global_peak_readable_frame_count_qc": len(readable_curve),
    }


def empty_metrics() -> dict[str, Any]:
    return {
        "n_frames": 0,
        "n_unique_frame_indices": 0,
        "frame_index_min": math.nan,
        "frame_index_max": math.nan,
        "frame_span_frames": math.nan,
        "missing_frame_count": 0,
        "missing_frame_ranges": "",
        "contiguous_pair_count": 0,
        "gap_transition_count": 0,
        "duplicate_frame_index_count": 0,
        "unreadable_frame_count": 0,
        "dimensions": "",
        "fps": math.nan,
        "frame_time_ms": math.nan,
        "duration_seconds": math.nan,
        "timing_source": "none_reliable",
        "timing_reliability": "unavailable",
        "global_peak_frame_index_provisional": math.nan,
        "global_peak_signal_provisional": math.nan,
        "global_peak_direction_provisional": "unavailable",
        "global_peak_time_seconds_provisional": math.nan,
        "global_peak_left_censored_qc": False,
        "global_peak_right_censored_qc": False,
        "global_peak_near_boundary_qc": False,
        "global_peak_readable_frame_count_qc": 0,
    }


def phase_audit(
    candidate: dict[str, Any],
    phase_name: str,
    stats_by_path: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    phase = candidate[phase_name]
    paths_by_internal = phase.get("strict_frame_paths_by_internal_series", {})
    audits: dict[str, dict[str, Any]] = {}
    for internal, paths in sorted(paths_by_internal.items(), key=lambda item: natural_key(str(item[0]))):
        audits[str(internal)] = sequence_metrics(list(paths), stats_by_path)
    selected = str(phase.get("selected_internal_series_in_v2", ""))
    selected_metrics = audits.get(selected, empty_metrics())
    return selected_metrics, audits


def phase_columns(prefix: str, selected: str, all_count: int, audits: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_all_strict_frame_path_count": all_count,
        f"{prefix}_internal_series_count": len(audits),
        f"{prefix}_selected_internal_series": selected,
        f"{prefix}_selected_n_frames": metrics["n_frames"],
        f"{prefix}_selected_n_unique_frame_indices": metrics["n_unique_frame_indices"],
        f"{prefix}_selected_frame_index_min": metrics["frame_index_min"],
        f"{prefix}_selected_frame_index_max": metrics["frame_index_max"],
        f"{prefix}_selected_frame_span_frames": metrics["frame_span_frames"],
        f"{prefix}_selected_missing_frame_count": metrics["missing_frame_count"],
        f"{prefix}_selected_missing_frame_ranges": metrics["missing_frame_ranges"],
        f"{prefix}_selected_contiguous_pair_count": metrics["contiguous_pair_count"],
        f"{prefix}_selected_gap_transition_count": metrics["gap_transition_count"],
        f"{prefix}_selected_duplicate_frame_index_count": metrics[
            "duplicate_frame_index_count"
        ],
        f"{prefix}_selected_unreadable_frame_count": metrics["unreadable_frame_count"],
        f"{prefix}_selected_dimensions": metrics["dimensions"],
        f"{prefix}_fps": metrics["fps"],
        f"{prefix}_frame_time_ms": metrics["frame_time_ms"],
        f"{prefix}_duration_seconds": metrics["duration_seconds"],
        f"{prefix}_timing_source": metrics["timing_source"],
        f"{prefix}_timing_reliability": metrics["timing_reliability"],
        f"global_{prefix}_peak_frame_index_provisional": metrics[
            "global_peak_frame_index_provisional"
        ],
        f"global_{prefix}_peak_signal_provisional": metrics[
            "global_peak_signal_provisional"
        ],
        f"global_{prefix}_peak_direction_provisional": metrics[
            "global_peak_direction_provisional"
        ],
        f"global_{prefix}_peak_time_seconds_provisional": metrics[
            "global_peak_time_seconds_provisional"
        ],
        f"global_{prefix}_peak_left_censored_qc": metrics[
            "global_peak_left_censored_qc"
        ],
        f"global_{prefix}_peak_right_censored_qc": metrics[
            "global_peak_right_censored_qc"
        ],
        f"global_{prefix}_peak_near_boundary_qc": metrics[
            "global_peak_near_boundary_qc"
        ],
        f"global_{prefix}_peak_readable_frame_count_qc": metrics[
            "global_peak_readable_frame_count_qc"
        ],
        f"{prefix}_internal_series_timing_json": json.dumps(
            audits, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=True
        ),
    }


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    df.to_csv(
        temp,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
        na_rep="NaN",
    )
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


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    root = args.project_root.resolve()
    cv2.setNumThreads(1)
    registries = []
    for path in (
        root / "metadata/api_fullseq_v3/lesion_registry_train_blinded.csv",
        root / "metadata/api_fullseq_v3/lesion_registry_valid_blinded.csv",
    ):
        registries.append(pd.read_csv(path, dtype=str, keep_default_na=False))
    registry = pd.concat(registries, ignore_index=True)

    candidates: list[tuple[str, str, dict[str, Any]]] = []
    for row in registry.drop_duplicates(["split", "patient_id"]).itertuples(index=False):
        payload = json.loads(row.candidate_series_registry_json)
        for candidate in payload["candidates"]:
            candidates.append((row.split, row.patient_id, candidate))
    if len(candidates) != EXPECTED_CANDIDATES:
        raise AssertionError(f"Expected {EXPECTED_CANDIDATES} candidates, got {len(candidates)}")
    candidate_uids = [candidate_uid(split, patient_id, candidate) for split, patient_id, candidate in candidates]
    if len(set(candidate_uids)) != EXPECTED_CANDIDATES:
        raise AssertionError("Candidate UID collision")

    all_path_occurrences: list[str] = []
    for _, _, candidate in candidates:
        for phase_name in ("pre", "post"):
            for paths in candidate[phase_name].get(
                "strict_frame_paths_by_internal_series", {}
            ).values():
                all_path_occurrences.extend(paths)
    unique_paths = sorted(set(all_path_occurrences), key=natural_key)
    if len(unique_paths) != EXPECTED_STRICT_FRAME_PATHS:
        raise AssertionError(
            f"Expected {EXPECTED_STRICT_FRAME_PATHS} unique strict frame paths, got {len(unique_paths)}"
        )
    if len(all_path_occurrences) != len(unique_paths):
        raise AssertionError("Frozen strict frame paths are duplicated across candidate/internal groups")

    stats_by_path: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, result in enumerate(executor.map(inspect_frame, unique_paths), start=1):
            stats_by_path[result["path"]] = result
            if args.progress_every > 0 and index % args.progress_every == 0:
                print(
                    json.dumps(
                        {
                            "timing_audit_progress_frames": index,
                            "total_frozen_frames": len(unique_paths),
                        }
                    ),
                    flush=True,
                )

    rows: list[dict[str, Any]] = []
    for (split, patient_id, candidate), uid in zip(candidates, candidate_uids):
        source = candidate["candidate_source"]
        audit = candidate["candidate_audit"]
        pre_metrics, pre_audits = phase_audit(candidate, "pre", stats_by_path)
        post_metrics, post_audits = phase_audit(candidate, "post", stats_by_path)
        pre_count = sum(
            len(paths)
            for paths in candidate["pre"].get(
                "strict_frame_paths_by_internal_series", {}
            ).values()
        )
        post_count = sum(
            len(paths)
            for paths in candidate["post"].get(
                "strict_frame_paths_by_internal_series", {}
            ).values()
        )
        row = {
            "candidate_uid": uid,
            "split": split,
            "patient_id": patient_id,
            "candidate_source_type": source.get("source_type", ""),
            "candidate_source_root": source.get("source_medical_record_root", ""),
            "candidate_discovery_rank": source.get("discovery_rank", ""),
            "candidate_series_id": source.get("series_id", ""),
            "candidate_series_path": source.get("series_path", ""),
            "candidate_valid": audit.get("candidate_valid", False),
            "candidate_selected_in_v2": audit.get("selected_candidate_in_v2", False),
            "candidate_selection_status_in_v2": audit.get("selection_status_in_v2", ""),
            "candidate_exclusion_reason": audit.get("candidate_exclusion_reason", ""),
            "frozen_strict_frame_path_count": pre_count + post_count,
            **phase_columns(
                "pre",
                str(candidate["pre"].get("selected_internal_series_in_v2", "")),
                pre_count,
                pre_audits,
                pre_metrics,
            ),
            **phase_columns(
                "post",
                str(candidate["post"].get("selected_internal_series_in_v2", "")),
                post_count,
                post_audits,
                post_metrics,
            ),
        }
        rows.append(row)

    timing = pd.DataFrame(rows, columns=COLUMNS)
    if len(timing) != EXPECTED_CANDIDATES or not timing["candidate_uid"].is_unique:
        raise AssertionError("Timing registry scale/candidate UID uniqueness failure")
    if int(timing["frozen_strict_frame_path_count"].sum()) != EXPECTED_STRICT_FRAME_PATHS:
        raise AssertionError("Timing registry frozen path total mismatch")

    for prefix in ("pre", "post"):
        unavailable = timing[f"{prefix}_timing_reliability"] != "reliable"
        if timing.loc[unavailable, f"{prefix}_fps"].notna().any():
            raise AssertionError(f"Unreliable {prefix} FPS must be NaN")
        if timing.loc[unavailable, f"{prefix}_duration_seconds"].notna().any():
            raise AssertionError(f"Unreliable {prefix} duration must be NaN")
        if timing.loc[unavailable, f"global_{prefix}_peak_time_seconds_provisional"].notna().any():
            raise AssertionError(f"Unreliable {prefix} peak seconds must be NaN")

    unreadable = sum(not item["readable"] for item in stats_by_path.values())
    timing_source_counts = Counter(
        list(timing["pre_timing_source"]) + list(timing["post_timing_source"])
    )
    timing_reliability_counts = Counter(
        list(timing["pre_timing_reliability"]) + list(timing["post_timing_reliability"])
    )
    total_internal_sequences = int(timing["pre_internal_series_count"].sum()) + int(
        timing["post_internal_series_count"].sum()
    )
    missing_candidates = int(
        (timing["pre_selected_missing_frame_count"] > 0).sum()
        + (timing["post_selected_missing_frame_count"] > 0).sum()
    )
    left_censored = int(
        timing["global_pre_peak_left_censored_qc"].sum()
        + timing["global_post_peak_left_censored_qc"].sum()
    )
    right_censored = int(
        timing["global_pre_peak_right_censored_qc"].sum()
        + timing["global_post_peak_right_censored_qc"].sum()
    )
    near_boundary = int(
        timing["global_pre_peak_near_boundary_qc"].sum()
        + timing["global_post_peak_near_boundary_qc"].sum()
    )

    audit_lines = [
        "# api_fullseq_v3 Candidate Timing and Integrity Audit",
        "",
        "## Scope",
        "",
        "- Audited frozen candidate series and frame paths from blinded Phase 1 registries only.",
        "- No private label artifact was opened.",
        "- No SEA-RAFT execution, ROI calculation, final feature generation, or model training was performed.",
        "- File modification times and filename indices were not treated as physical timing metadata.",
        "- Full-frame intensity-change peak fields are provisional censoring QC only.",
        "",
        "## Frozen scale",
        "",
        f"- Candidate series: **{len(timing)}**",
        f"- Unique strict frame paths: **{len(unique_paths)}**",
        f"- Candidate-phase internal sequences: **{total_internal_sequences}**",
        f"- Unreadable frozen frames: **{unreadable}**",
        "",
        "## Frame integrity",
        "",
        f"- Selected candidate-phases with one or more missing indices: **{missing_candidates}**",
        f"- Total selected Pre contiguous pairs: **{int(timing['pre_selected_contiguous_pair_count'].sum())}**",
        f"- Total selected Post contiguous pairs: **{int(timing['post_selected_contiguous_pair_count'].sum())}**",
        "",
        "## Physical timing metadata",
        "",
        "Reliable FPS and seconds are emitted only when consistent explicit metadata is present on every readable frame in the sequence. Otherwise FPS, FrameTime, duration seconds, and provisional peak seconds are `NaN`.",
        "",
        "| Timing source | Candidate-phase count |",
        "|---|---:|",
    ]
    for source, count in sorted(timing_source_counts.items()):
        audit_lines.append(f"| {source} | {count} |")
    audit_lines.extend(["", "| Timing reliability | Candidate-phase count |", "|---|---:|"])
    for reliability, count in sorted(timing_reliability_counts.items()):
        audit_lines.append(f"| {reliability} | {count} |")
    audit_lines.extend(
        [
            "",
            "## Provisional global peak censoring QC",
            "",
            f"- Left-boundary peaks: **{left_censored}**",
            f"- Right-boundary peaks: **{right_censored}**",
            f"- Peaks at or within one readable frame of a boundary: **{near_boundary}**",
            "- These fields describe full-frame mean-intensity change and are not lesion-level measurements.",
            "",
        ]
    )

    output_path = root / "metadata/api_fullseq_v3/candidate_timing_registry.csv"
    audit_path = root / "reports/api_fullseq_v3/timing_audit.md"
    csv_text = timing.to_csv(index=False, lineterminator="\n", na_rep="NaN")
    audit_text = "\n".join(audit_lines)
    assert_blinded_text(csv_text, output_path.name)
    assert_blinded_text(audit_text, audit_path.name)
    atomic_write_csv(timing, output_path)
    atomic_write_text(audit_text, audit_path)

    print(
        json.dumps(
            {
                "candidate_rows": len(timing),
                "frozen_strict_frame_paths": int(
                    timing["frozen_strict_frame_path_count"].sum()
                ),
                "unreadable_frames": unreadable,
                "timing_reliability_counts": dict(timing_reliability_counts),
                "restricted_private_artifact_accessed": False,
                "sea_raft_run": False,
                "final_features_generated": False,
                "outputs": [str(output_path), str(audit_path)],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

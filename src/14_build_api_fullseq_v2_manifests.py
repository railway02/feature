#!/usr/bin/env python3
"""Build deterministic patient-level api_fullseq_v2 manifests.

This stage is deliberately limited to filesystem discovery and image QC.  It
does not import or run SEA-RAFT, perform optical-flow inference, extract flow
features, or train a model.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import pandas as pd


PROJECT = Path("/root/autodl-tmp/aneurysm")
DATA_ROOT = Path("/root/autodl-tmp/tiantanDSA")
UPDATED_ROOT = PROJECT / "staging/updated_10_cases"
TRAIN_XLSX = PROJECT / "metadata/Train.xlsx"
VALID_XLSX = PROJECT / "metadata/valid.xlsx"

MAX_DISCOVERY_DEPTH = 8
PROGRESS_INTERVAL = 50
DEFAULT_WORKERS = min(16, os.cpu_count() or 1)
STRICT_FRAME_RE = re.compile(r"^IMG-(\d+)-(\d+)\.(jpg|jpeg)$", re.IGNORECASE)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
JPEG_SUFFIXES = {".jpg", ".jpeg"}
PARAMETER_TOKENS = ("CBF", "CBV", "MTT", "TTP")
PHASE_DIR_NAMES = {"pre-api": "pre", "post-api": "post"}

UPDATED_SERIES = {
    "458123": "C6",
    "549117": "C6-1",
    "554148": "C6-1",
    "565733": "C6-1",
    "585192": "C4",
    "593174": "C6-1",
    "640779": "C6-1",
    "667483": "C4",
    "696044": "C6-1",
    "726527": "C6-1",
}
UPDATED_IDS = list(UPDATED_SERIES)
BOUNDARY_IDS = ["453933", "474686", "654075", "654873", "571569", "348817"]
PILOT_IDS = UPDATED_IDS + BOUNDARY_IDS

FINAL_OUTPUTS = {
    "patient_manifest": PROJECT / "manifests/api_fullseq_v2_patient_manifest.csv",
    "train_manifest": PROJECT / "manifests/api_fullseq_v2_train_manifest.csv",
    "valid_manifest": PROJECT / "manifests/api_fullseq_v2_valid_manifest.csv",
    "candidate_audit": PROJECT / "manifests/api_fullseq_v2_candidate_series_audit.csv",
    "frame_audit": PROJECT / "manifests/api_fullseq_v2_frame_audit.csv",
    "pilot_manifest": PROJECT / "manifests/api_fullseq_v2_pilot_manifest.csv",
    "manifest_audit": PROJECT / "reports/api_fullseq_v2/manifest_audit.md",
    "excluded_evidence": PROJECT / "reports/api_fullseq_v2/excluded_evidence.csv",
    "parameter_only": PROJECT / "reports/api_fullseq_v2/parameter_only_directories.csv",
    "nonstandard": PROJECT / "reports/api_fullseq_v2/nonstandard_filename_inventory.csv",
    "unreadable": PROJECT / "reports/api_fullseq_v2/unreadable_files.csv",
    "pilot_report": PROJECT / "reports/api_fullseq_v2/pilot_selection_report.md",
    "config": PROJECT / "configs/api_fullseq_v2_manifest_config.json",
}
FAILURE_PATH = PROJECT / "reports/api_fullseq_v2/failure.md"

PATIENT_COLUMNS = [
    "patient_id", "split", "source_type", "source_medical_record_root",
    "selected_series_id", "selected_series_path", "selected_candidate_rank",
    "selection_reason", "candidate_count", "valid_candidate_count",
    "all_candidate_series", "ignored_series", "pre_api_dir", "post_api_dir",
    "selected_pre_internal_series", "selected_post_internal_series",
    "n_pre_frames", "n_post_frames", "n_pre_contiguous_pairs",
    "n_post_contiguous_pairs", "pre_frame_list_hash", "post_frame_list_hash",
    "can_run_pre", "can_run_post", "can_run_prepost", "patient_status",
    "exclusion_reason", "pre_frame_indices", "post_frame_indices",
    "pre_frame_paths", "post_frame_paths", "pre_dimensions", "post_dimensions",
    "pre_frame_gaps", "post_frame_gaps", "pre_ignored_internal_series",
    "post_ignored_internal_series",
]

CANDIDATE_COLUMNS = [
    "patient_id", "split", "source_type", "source_medical_record_root",
    "discovery_rank", "series_id", "series_path", "is_fixed_target",
    "scan_performed", "validity_evaluated", "pre_api_dir", "post_api_dir",
    "pre_api_dir_exists", "post_api_dir_exists", "pre_extra_api_dirs",
    "post_extra_api_dirs", "pre_parameter_only", "post_parameter_only",
    "pre_total_files", "post_total_files", "pre_image_files", "post_image_files",
    "pre_strict_files", "post_strict_files", "pre_nonstandard_jpeg_files",
    "post_nonstandard_jpeg_files", "pre_unreadable_files", "post_unreadable_files",
    "pre_internal_series", "post_internal_series", "selected_pre_internal_series",
    "selected_post_internal_series", "pre_ignored_internal_series",
    "post_ignored_internal_series", "n_pre_frames", "n_post_frames",
    "n_pre_contiguous_pairs", "n_post_contiguous_pairs", "pre_frame_gaps",
    "post_frame_gaps", "can_run_pre", "can_run_post", "can_run_prepost",
    "candidate_valid", "selected_candidate", "selection_status",
    "candidate_exclusion_reason", "pre_internal_series_audit",
    "post_internal_series_audit",
]

FRAME_COLUMNS = [
    "patient_id", "split", "source_type", "source_medical_record_root",
    "discovery_rank", "series_id", "series_path", "phase", "api_dir",
    "relative_path_in_api", "filename", "absolute_path", "suffix", "size_bytes",
    "strict_filename_match", "internal_series_number", "frame_index",
    "is_parameter_map", "is_nonstandard_jpeg", "read_attempted", "readable", "read_error",
    "height", "width", "channels", "phase_selected_internal_series",
    "primary_for_frame_index", "phase_eligible_frame", "selected",
    "selection_reason",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def natural_key(value: str) -> tuple[tuple[int, Any], ...]:
    parts = re.split(r"(\d+)", value.casefold())
    return tuple((0, int(part)) if part.isdigit() else (1, part) for part in parts)


def patient_key(patient_id: str) -> tuple[int, Any]:
    return (0, int(patient_id)) if patient_id.isdigit() else (1, natural_key(patient_id))


def sorted_paths(paths: Iterable[Path], root: Path | None = None) -> list[Path]:
    if root is None:
        return sorted(paths, key=lambda path: natural_key(path.name))
    return sorted(paths, key=lambda path: natural_key(path.relative_to(root).as_posix()))


def normalize_patient_id(value: Any) -> str:
    if pd.isna(value):
        raise ValueError("Encountered empty 病案号")
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    if not re.fullmatch(r"\d+", text):
        raise ValueError(f"Non-numeric 病案号 after normalization: {value!r}")
    return str(int(text))


def pipe(values: Iterable[Any]) -> str:
    return "|".join(str(value) for value in values if value is not None and str(value) != "")


def bool_value(value: Any) -> bool:
    return bool(value) if value is not None else False


def hash_lines(values: Iterable[str]) -> str:
    material = "\n".join(values).encode("utf-8")
    return hashlib.sha256(material).hexdigest() if material else ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_stat(path: Path) -> os.stat_result | None:
    try:
        return path.stat()
    except OSError:
        return None


def child_dirs(path: Path) -> list[Path]:
    try:
        entries = [Path(entry.path) for entry in os.scandir(path) if entry.is_dir(follow_symlinks=False)]
    except OSError:
        return []
    return sorted_paths(entries)


def walk_files(root: Path) -> list[Path]:
    result: list[Path] = []

    def visit(directory: Path) -> None:
        try:
            entries = list(os.scandir(directory))
        except OSError:
            return
        directories = sorted(
            (Path(entry.path) for entry in entries if entry.is_dir(follow_symlinks=False)),
            key=lambda path: natural_key(path.name),
        )
        files = sorted(
            (Path(entry.path) for entry in entries if entry.is_file(follow_symlinks=False)),
            key=lambda path: natural_key(path.name),
        )
        result.extend(files)
        for child in directories:
            visit(child)

    if root.is_dir():
        visit(root)
    return result


def series_id_for(parent: Path, patient_root: Path) -> str:
    relative = parent.relative_to(patient_root)
    return "main" if str(relative) == "." else relative.as_posix()


def discover_candidates(patient_root: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    def visit(directory: Path, depth: int) -> None:
        children = child_dirs(directory)
        phase_dirs: dict[str, list[Path]] = {"pre": [], "post": []}
        for child in children:
            phase = PHASE_DIR_NAMES.get(child.name.casefold())
            if phase:
                phase_dirs[phase].append(child)
        if phase_dirs["pre"] or phase_dirs["post"]:
            candidates.append({
                "discovery_rank": len(candidates) + 1,
                "series_id": series_id_for(directory, patient_root),
                "series_path": directory.resolve(),
                "pre_api_dir": phase_dirs["pre"][0].resolve() if phase_dirs["pre"] else None,
                "post_api_dir": phase_dirs["post"][0].resolve() if phase_dirs["post"] else None,
                "pre_extra_api_dirs": [path.resolve() for path in phase_dirs["pre"][1:]],
                "post_extra_api_dirs": [path.resolve() for path in phase_dirs["post"][1:]],
            })
        if depth >= MAX_DISCOVERY_DEPTH:
            return
        for child in children:
            if child.name.casefold() in PHASE_DIR_NAMES:
                continue
            visit(child, depth + 1)

    if patient_root.is_dir():
        visit(patient_root, 0)
    return candidates


def inspect_image(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "read_attempted": True,
        "readable": False,
        "height": None,
        "width": None,
        "channels": None,
        "read_error": "",
    }
    try:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            result["read_error"] = "cv2_imread_returned_none"
            return result
        result["readable"] = True
        result["height"] = int(image.shape[0])
        result["width"] = int(image.shape[1])
        result["channels"] = 1 if image.ndim == 2 else int(image.shape[2])
    except Exception as exc:  # OpenCV failures must become audit evidence.
        result["read_error"] = f"{type(exc).__name__}: {exc}"
    return result


def is_parameter_map_path(relative_path: Path) -> bool:
    uppercase_parts = [part.upper() for part in relative_path.parts]
    return any(token in part for part in uppercase_parts for token in PARAMETER_TOKENS)


def format_missing(start: int, end: int) -> str:
    first = start + 1
    last = end - 1
    if first == last:
        return str(first)
    return f"{first}-{last}"


def empty_phase(phase: str, api_dir: Path | None) -> dict[str, Any]:
    return {
        "phase": phase,
        "api_dir": str(api_dir) if api_dir else "",
        "api_dir_exists": bool(api_dir and api_dir.is_dir()),
        "parameter_only": False,
        "total_files": 0,
        "image_files": 0,
        "strict_files": 0,
        "nonstandard_jpeg_files": 0,
        "unreadable_files": 0,
        "internal_series": "",
        "selected_internal_series": "",
        "ignored_internal_series": "",
        "n_frames": 0,
        "n_contiguous_pairs": 0,
        "frame_indices": "",
        "frame_paths": "",
        "frame_list_hash": "",
        "dimensions": "",
        "frame_gaps": "",
        "n_gap_transitions": 0,
        "can_run": False,
        "phase_reason": "missing_phase_directory" if not api_dir else "phase_not_scanned",
        "internal_series_audit": "[]",
        "frame_records": [],
    }

def scan_phase(
    patient_id: str,
    split: str,
    source_type: str,
    source_root: Path,
    candidate: dict[str, Any],
    phase: str,
    api_dir: Path | None,
    executor: ThreadPoolExecutor,
) -> dict[str, Any]:
    if api_dir is None or not api_dir.is_dir():
        return empty_phase(phase, api_dir)

    files = walk_files(api_dir)
    image_paths = [path for path in files if path.suffix.casefold() in IMAGE_SUFFIXES]
    inspections = list(executor.map(inspect_image, image_paths)) if image_paths else []
    inspection_by_path = dict(zip(image_paths, inspections))
    records: list[dict[str, Any]] = []

    for path in files:
        relative = path.relative_to(api_dir)
        suffix = path.suffix.casefold()
        match = STRICT_FRAME_RE.fullmatch(path.name) if suffix in JPEG_SUFFIXES else None
        parameter_map = is_parameter_map_path(relative)
        nonstandard = suffix in JPEG_SUFFIXES and match is None and not parameter_map
        inspection = inspection_by_path.get(path, {
            "read_attempted": False,
            "readable": False,
            "height": None,
            "width": None,
            "channels": None,
            "read_error": "",
        })
        stat_result = safe_stat(path)
        record = {
            "patient_id": patient_id,
            "split": split,
            "source_type": source_type,
            "source_medical_record_root": str(source_root.resolve()),
            "discovery_rank": candidate["discovery_rank"],
            "series_id": candidate["series_id"],
            "series_path": str(candidate["series_path"]),
            "phase": phase,
            "api_dir": str(api_dir),
            "relative_path_in_api": relative.as_posix(),
            "filename": path.name,
            "absolute_path": str(path.resolve()),
            "suffix": suffix,
            "size_bytes": int(stat_result.st_size) if stat_result else None,
            "strict_filename_match": bool(match),
            "internal_series_number": int(match.group(1)) if match else None,
            "frame_index": int(match.group(2)) if match else None,
            "is_parameter_map": parameter_map,
            "is_nonstandard_jpeg": nonstandard,
            "read_attempted": inspection["read_attempted"],
            "readable": inspection["readable"],
            "height": inspection["height"],
            "width": inspection["width"],
            "channels": inspection["channels"],
            "read_error": inspection["read_error"],
            "phase_selected_internal_series": None,
            "primary_for_frame_index": False,
            "phase_eligible_frame": False,
            "selected": False,
            "selection_reason": "",
        }
        if parameter_map:
            record["selection_reason"] = "parameter_map_excluded"
        elif suffix == ".png":
            record["selection_reason"] = "png_excluded"
        elif nonstandard:
            record["selection_reason"] = "nonstandard_jpeg_filename"
        elif match and not inspection["readable"]:
            record["selection_reason"] = "strict_frame_unreadable"
        elif match:
            record["selection_reason"] = "strict_frame_pending_internal_series_qc"
        else:
            record["selection_reason"] = "unsupported_or_nonimage_file"
        records.append(record)

    image_records = [record for record in records if record["suffix"] in IMAGE_SUFFIXES]
    strict_records = [
        record for record in records
        if record["strict_filename_match"] and not record["is_parameter_map"]
    ]
    nonstandard_records = [record for record in records if record["is_nonstandard_jpeg"]]
    unreadable_records = [
        record for record in image_records
        if record["read_attempted"] and not record["readable"]
    ]
    parameter_only = bool(image_records) and all(record["is_parameter_map"] for record in image_records)

    by_internal: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in strict_records:
        by_internal[int(record["internal_series_number"])].append(record)

    internal_audits: list[dict[str, Any]] = []
    runnable_internal_numbers: list[int] = []
    primary_by_internal: dict[int, list[dict[str, Any]]] = {}

    for internal_number in sorted(by_internal):
        internal_records = sorted(
            by_internal[internal_number],
            key=lambda record: (
                int(record["frame_index"]),
                natural_key(str(record["relative_path_in_api"])),
            ),
        )
        readable_records = [record for record in internal_records if record["readable"]]
        dimensions = sorted({(int(record["height"]), int(record["width"])) for record in readable_records})
        mixed_dimensions = len(dimensions) > 1
        primary_records: list[dict[str, Any]] = []
        seen_indices: set[int] = set()
        duplicate_indices: list[int] = []
        for record in readable_records:
            index = int(record["frame_index"])
            if index in seen_indices:
                duplicate_indices.append(index)
                record["selection_reason"] = "duplicate_frame_index_not_primary"
                continue
            seen_indices.add(index)
            record["primary_for_frame_index"] = True
            primary_records.append(record)
        primary_records.sort(key=lambda record: int(record["frame_index"]))
        primary_by_internal[internal_number] = primary_records
        indices = [int(record["frame_index"]) for record in primary_records]
        contiguous_pairs = sum(1 for left, right in zip(indices, indices[1:]) if right - left == 1)
        gaps = [
            f"{left}->{right}[missing:{format_missing(left, right)}]"
            for left, right in zip(indices, indices[1:]) if right - left > 1
        ]
        runnable = len(primary_records) >= 2 and not mixed_dimensions and contiguous_pairs >= 1
        if runnable:
            runnable_internal_numbers.append(internal_number)
        reason_parts: list[str] = []
        if len(primary_records) < 2:
            reason_parts.append("fewer_than_2_readable_unique_frames")
        if mixed_dimensions:
            reason_parts.append("mixed_dimensions")
        if len(primary_records) >= 2 and contiguous_pairs < 1:
            reason_parts.append("no_contiguous_frame_pair")
        if not reason_parts:
            reason_parts.append("runnable")
        internal_audits.append({
            "internal_series_number": internal_number,
            "strict_file_count": len(internal_records),
            "readable_file_count": len(readable_records),
            "unique_frame_count": len(primary_records),
            "duplicate_frame_indices": sorted(set(duplicate_indices)),
            "dimensions": [f"{height}x{width}" for height, width in dimensions],
            "mixed_dimensions": mixed_dimensions,
            "frame_indices": indices,
            "n_contiguous_pairs": contiguous_pairs,
            "frame_gaps": gaps,
            "runnable": runnable,
            "reason": pipe(reason_parts),
        })

    selected_internal = min(runnable_internal_numbers) if runnable_internal_numbers else None
    selected_frames = primary_by_internal.get(selected_internal, []) if selected_internal is not None else []
    selected_indices = [int(record["frame_index"]) for record in selected_frames]
    selected_pairs = sum(
        1 for left, right in zip(selected_indices, selected_indices[1:]) if right - left == 1
    )
    selected_gaps = [
        f"{left}->{right}[missing:{format_missing(left, right)}]"
        for left, right in zip(selected_indices, selected_indices[1:]) if right - left > 1
    ]
    selected_paths = [str(record["absolute_path"]) for record in selected_frames]
    selected_dimensions = sorted({(int(record["height"]), int(record["width"])) for record in selected_frames})

    for record in strict_records:
        record["phase_selected_internal_series"] = selected_internal
        internal_number = int(record["internal_series_number"])
        if not record["readable"]:
            continue
        if selected_internal is None:
            if record["selection_reason"] == "strict_frame_pending_internal_series_qc":
                record["selection_reason"] = "no_runnable_internal_series"
        elif internal_number != selected_internal:
            if record["selection_reason"] == "strict_frame_pending_internal_series_qc":
                record["selection_reason"] = "ignored_internal_series"
        elif record["primary_for_frame_index"]:
            record["phase_eligible_frame"] = True
            record["selection_reason"] = "eligible_frame_in_selected_internal_series"

    ignored_internal = [number for number in sorted(by_internal) if number != selected_internal]
    if selected_internal is not None:
        phase_reason = "runnable"
    elif parameter_only:
        phase_reason = "parameter_only"
    elif strict_records:
        phase_reason = "strict_frames_but_no_runnable_internal_series"
    elif nonstandard_records:
        phase_reason = "only_nonstandard_jpeg_filenames"
    elif image_records:
        phase_reason = "no_eligible_strict_jpeg_frames"
    else:
        phase_reason = "empty_or_no_image_files"

    return {
        "phase": phase,
        "api_dir": str(api_dir),
        "api_dir_exists": True,
        "parameter_only": parameter_only,
        "total_files": len(records),
        "image_files": len(image_records),
        "strict_files": len(strict_records),
        "nonstandard_jpeg_files": len(nonstandard_records),
        "unreadable_files": len(unreadable_records),
        "internal_series": pipe(sorted(by_internal)),
        "selected_internal_series": selected_internal if selected_internal is not None else "",
        "ignored_internal_series": pipe(ignored_internal),
        "n_frames": len(selected_frames),
        "n_contiguous_pairs": selected_pairs,
        "frame_indices": pipe(selected_indices),
        "frame_paths": pipe(selected_paths),
        "frame_list_hash": hash_lines(selected_paths),
        "dimensions": pipe(f"{height}x{width}" for height, width in selected_dimensions),
        "frame_gaps": pipe(selected_gaps),
        "n_gap_transitions": len(selected_gaps),
        "can_run": selected_internal is not None,
        "phase_reason": phase_reason,
        "internal_series_audit": json.dumps(internal_audits, ensure_ascii=False, sort_keys=True),
        "frame_records": records,
    }


def unscanned_phase(phase: str, api_dir: Path | None) -> dict[str, Any]:
    result = empty_phase(phase, api_dir)
    result["api_dir_exists"] = bool(api_dir and api_dir.is_dir())
    result["phase_reason"] = "ignored_fixed_mapping_sibling_not_scanned"
    return result


def scan_candidate(
    patient_id: str,
    split: str,
    source_type: str,
    source_root: Path,
    candidate: dict[str, Any],
    executor: ThreadPoolExecutor,
    scan_performed: bool,
) -> dict[str, Any]:
    if scan_performed:
        pre = scan_phase(
            patient_id, split, source_type, source_root, candidate, "pre",
            candidate.get("pre_api_dir"), executor,
        )
        post = scan_phase(
            patient_id, split, source_type, source_root, candidate, "post",
            candidate.get("post_api_dir"), executor,
        )
    else:
        pre = unscanned_phase("pre", candidate.get("pre_api_dir"))
        post = unscanned_phase("post", candidate.get("post_api_dir"))
    can_run_pre = bool_value(pre["can_run"])
    can_run_post = bool_value(post["can_run"])
    candidate_valid = scan_performed and (can_run_pre or can_run_post)
    exclusion = "" if candidate_valid else pipe(
        [f"pre:{pre['phase_reason']}", f"post:{post['phase_reason']}"]
    )
    result = dict(candidate)
    result.update({
        "scan_performed": scan_performed,
        "validity_evaluated": scan_performed,
        "pre": pre,
        "post": post,
        "can_run_pre": can_run_pre,
        "can_run_post": can_run_post,
        "can_run_prepost": can_run_pre and can_run_post,
        "candidate_valid": candidate_valid,
        "candidate_exclusion_reason": exclusion,
        "selected_candidate": False,
        "selection_status": "",
    })
    return result


def synthesize_fixed_candidate(patient_root: Path, series_id: str, rank: int) -> dict[str, Any]:
    series_path = patient_root / Path(series_id)
    children = child_dirs(series_path) if series_path.is_dir() else []
    pre_dirs = [path for path in children if path.name.casefold() == "pre-api"]
    post_dirs = [path for path in children if path.name.casefold() == "post-api"]
    return {
        "discovery_rank": rank,
        "series_id": series_id,
        "series_path": series_path.resolve(),
        "pre_api_dir": pre_dirs[0].resolve() if pre_dirs else None,
        "post_api_dir": post_dirs[0].resolve() if post_dirs else None,
        "pre_extra_api_dirs": [path.resolve() for path in pre_dirs[1:]],
        "post_extra_api_dirs": [path.resolve() for path in post_dirs[1:]],
    }


def classify_exclusion(source_root: Path, candidates: list[dict[str, Any]]) -> tuple[str, str]:
    if not source_root.is_dir():
        return "missing_medical_record_directory", "source_medical_record_directory_missing"
    evaluated = [candidate for candidate in candidates if candidate["validity_evaluated"]]
    phases = [candidate[phase] for candidate in evaluated for phase in ("pre", "post")]
    strict_count = sum(int(phase["strict_files"]) for phase in phases)
    nonstandard_count = sum(int(phase["nonstandard_jpeg_files"]) for phase in phases)
    parameter_only_count = sum(bool(phase["parameter_only"]) for phase in phases)
    unreadable_count = sum(int(phase["unreadable_files"]) for phase in phases)
    if strict_count:
        reason = "strict_frames_failed_internal_sequence_qc"
        if unreadable_count:
            reason += f"|unreadable_image_files:{unreadable_count}"
        return "unreadable_or_invalid_frames", reason
    if nonstandard_count:
        return "only_nonstandard_filenames", f"nonstandard_jpeg_files:{nonstandard_count}"
    if parameter_only_count:
        return "only_parameter_maps", f"parameter_only_phase_directories:{parameter_only_count}"
    if not candidates:
        return "no_valid_grayscale_api", "no_pre_api_or_post_api_candidate_found"
    return "no_valid_grayscale_api", "no_candidate_contains_runnable_strict_grayscale_sequence"


def candidate_to_row(
    patient_id: str,
    split: str,
    source_type: str,
    source_root: Path,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    pre = candidate["pre"]
    post = candidate["post"]
    return {
        "patient_id": patient_id,
        "split": split,
        "source_type": source_type,
        "source_medical_record_root": str(source_root.resolve()),
        "discovery_rank": candidate["discovery_rank"],
        "series_id": candidate["series_id"],
        "series_path": str(candidate["series_path"]),
        "is_fixed_target": candidate["is_fixed_target"],
        "scan_performed": candidate["scan_performed"],
        "validity_evaluated": candidate["validity_evaluated"],
        "pre_api_dir": pre["api_dir"],
        "post_api_dir": post["api_dir"],
        "pre_api_dir_exists": pre["api_dir_exists"],
        "post_api_dir_exists": post["api_dir_exists"],
        "pre_extra_api_dirs": pipe(candidate.get("pre_extra_api_dirs", [])),
        "post_extra_api_dirs": pipe(candidate.get("post_extra_api_dirs", [])),
        "pre_parameter_only": pre["parameter_only"],
        "post_parameter_only": post["parameter_only"],
        "pre_total_files": pre["total_files"],
        "post_total_files": post["total_files"],
        "pre_image_files": pre["image_files"],
        "post_image_files": post["image_files"],
        "pre_strict_files": pre["strict_files"],
        "post_strict_files": post["strict_files"],
        "pre_nonstandard_jpeg_files": pre["nonstandard_jpeg_files"],
        "post_nonstandard_jpeg_files": post["nonstandard_jpeg_files"],
        "pre_unreadable_files": pre["unreadable_files"],
        "post_unreadable_files": post["unreadable_files"],
        "pre_internal_series": pre["internal_series"],
        "post_internal_series": post["internal_series"],
        "selected_pre_internal_series": pre["selected_internal_series"],
        "selected_post_internal_series": post["selected_internal_series"],
        "pre_ignored_internal_series": pre["ignored_internal_series"],
        "post_ignored_internal_series": post["ignored_internal_series"],
        "n_pre_frames": pre["n_frames"],
        "n_post_frames": post["n_frames"],
        "n_pre_contiguous_pairs": pre["n_contiguous_pairs"],
        "n_post_contiguous_pairs": post["n_contiguous_pairs"],
        "pre_frame_gaps": pre["frame_gaps"],
        "post_frame_gaps": post["frame_gaps"],
        "can_run_pre": candidate["can_run_pre"],
        "can_run_post": candidate["can_run_post"],
        "can_run_prepost": candidate["can_run_prepost"],
        "candidate_valid": candidate["candidate_valid"],
        "selected_candidate": candidate["selected_candidate"],
        "selection_status": candidate["selection_status"],
        "candidate_exclusion_reason": candidate["candidate_exclusion_reason"],
        "pre_internal_series_audit": pre["internal_series_audit"],
        "post_internal_series_audit": post["internal_series_audit"],
    }


def build_patient(
    patient_id: str,
    split: str,
    executor: ThreadPoolExecutor,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    is_updated = patient_id in UPDATED_SERIES
    source_type = "updated_10_cases" if is_updated else "tiantanDSA"
    source_root = (UPDATED_ROOT if is_updated else DATA_ROOT) / patient_id
    discovered = discover_candidates(source_root)
    fixed_series = UPDATED_SERIES.get(patient_id)

    if is_updated and not any(candidate["series_id"] == fixed_series for candidate in discovered):
        discovered.append(synthesize_fixed_candidate(source_root, str(fixed_series), len(discovered) + 1))

    scanned: list[dict[str, Any]] = []
    for candidate in discovered:
        is_fixed_target = bool(is_updated and candidate["series_id"] == fixed_series)
        candidate["is_fixed_target"] = is_fixed_target
        should_scan = not is_updated or is_fixed_target
        scanned.append(
            scan_candidate(
                patient_id, split, source_type, source_root, candidate, executor, should_scan
            )
        )

    selected: dict[str, Any] | None = None
    if is_updated:
        selected = next(
            (candidate for candidate in scanned if candidate["series_id"] == fixed_series),
            None,
        )
    else:
        selected = next((candidate for candidate in scanned if candidate["candidate_valid"]), None)

    for candidate in scanned:
        if candidate is selected:
            candidate["selected_candidate"] = True
            candidate["selection_status"] = (
                "selected_fixed_mapping" if is_updated else "selected_first_valid_by_discovery_rank"
            )
        elif is_updated:
            candidate["selection_status"] = "ignored_fixed_mapping_sibling_not_scanned"
        elif candidate["candidate_valid"]:
            candidate["selection_status"] = "ignored_valid_after_first_valid"
        else:
            candidate["selection_status"] = "ignored_invalid_candidate"

    frame_rows: list[dict[str, Any]] = []
    for candidate in scanned:
        for phase_name in ("pre", "post"):
            for record in candidate[phase_name]["frame_records"]:
                if candidate is selected and record["phase_eligible_frame"]:
                    record["selected"] = True
                    record["selection_reason"] = "selected_strict_runnable_frame"
                elif record["phase_eligible_frame"]:
                    record["selection_reason"] = "eligible_frame_in_ignored_candidate"
                frame_rows.append({column: record.get(column, "") for column in FRAME_COLUMNS})

    candidate_rows = [
        candidate_to_row(patient_id, split, source_type, source_root, candidate)
        for candidate in scanned
    ]
    all_series = [candidate["series_id"] for candidate in scanned]
    ignored_series = [
        candidate["series_id"] for candidate in scanned if candidate is not selected
    ]
    valid_count = sum(bool(candidate["candidate_valid"]) for candidate in scanned)

    selected_valid = bool(selected and selected["candidate_valid"])
    if selected_valid:
        if selected["can_run_prepost"]:
            patient_status = "selected_prepost"
        elif selected["can_run_pre"]:
            patient_status = "selected_pre_only"
        else:
            patient_status = "selected_post_only"
        exclusion_reason = ""
    else:
        patient_status, exclusion_reason = classify_exclusion(source_root, scanned)

    if selected is None:
        pre = empty_phase("pre", None)
        post = empty_phase("post", None)
        selection_reason = "no_valid_candidate"
    else:
        pre = selected["pre"]
        post = selected["post"]
        selection_reason = (
            "fixed_updated_series_mapping_no_fallback"
            if is_updated else "first_valid_candidate_by_discovery_rank"
        )

    patient_row = {
        "patient_id": patient_id,
        "split": split,
        "source_type": source_type,
        "source_medical_record_root": str(source_root.resolve()),
        "selected_series_id": selected["series_id"] if selected else "",
        "selected_series_path": str(selected["series_path"]) if selected else "",
        "selected_candidate_rank": selected["discovery_rank"] if selected else "",
        "selection_reason": selection_reason,
        "candidate_count": len(scanned),
        "valid_candidate_count": valid_count,
        "all_candidate_series": pipe(all_series),
        "ignored_series": pipe(ignored_series),
        "pre_api_dir": pre["api_dir"],
        "post_api_dir": post["api_dir"],
        "selected_pre_internal_series": pre["selected_internal_series"],
        "selected_post_internal_series": post["selected_internal_series"],
        "n_pre_frames": pre["n_frames"],
        "n_post_frames": post["n_frames"],
        "n_pre_contiguous_pairs": pre["n_contiguous_pairs"],
        "n_post_contiguous_pairs": post["n_contiguous_pairs"],
        "pre_frame_list_hash": pre["frame_list_hash"],
        "post_frame_list_hash": post["frame_list_hash"],
        "can_run_pre": bool_value(pre["can_run"]),
        "can_run_post": bool_value(post["can_run"]),
        "can_run_prepost": bool_value(pre["can_run"]) and bool_value(post["can_run"]),
        "patient_status": patient_status,
        "exclusion_reason": exclusion_reason,
        "pre_frame_indices": pre["frame_indices"],
        "post_frame_indices": post["frame_indices"],
        "pre_frame_paths": pre["frame_paths"],
        "post_frame_paths": post["frame_paths"],
        "pre_dimensions": pre["dimensions"],
        "post_dimensions": post["dimensions"],
        "pre_frame_gaps": pre["frame_gaps"],
        "post_frame_gaps": post["frame_gaps"],
        "pre_ignored_internal_series": pre["ignored_internal_series"],
        "post_ignored_internal_series": post["ignored_internal_series"],
    }
    return patient_row, candidate_rows, frame_rows


def load_split(path: Path, split: str) -> tuple[list[str], dict[str, Any]]:
    frame = pd.read_excel(path, dtype={"病案号": str})
    matching_columns = [column for column in frame.columns if str(column).strip() == "病案号"]
    if len(matching_columns) != 1:
        raise ValueError(f"{path}: expected exactly one 病案号 column, found {matching_columns}")
    ids = [normalize_patient_id(value) for value in frame[matching_columns[0]].tolist()]
    unique_ids = sorted(set(ids), key=patient_key)
    return unique_ids, {
        "split": split,
        "excel_path": str(path),
        "excel_rows": len(ids),
        "unique_patient_ids": len(unique_ids),
        "duplicate_excel_rows_after_first": len(ids) - len(unique_ids),
    }


def protected_file_paths() -> list[Path]:
    paths: set[Path] = {TRAIN_XLSX, VALID_XLSX}
    for path in (PROJECT / "manifests").glob("api_fullseq_v1_*"):
        if path.is_file():
            paths.add(path)
    v1_report = PROJECT / "reports/api_fullseq_v1"
    if v1_report.is_dir():
        paths.update(path for path in v1_report.rglob("*") if path.is_file())
    for name in ("flow_manifest.csv", "filesystem_inventory.csv"):
        path = PROJECT / "manifests" / name
        if path.is_file():
            paths.add(path)
    code_dir = PROJECT / "code"
    for entry in os.scandir(code_dir):
        path = Path(entry.path)
        if entry.is_file(follow_symlinks=False) and re.match(r"^(?:0\d|1[0-3])_", path.name):
            paths.add(path)
    return sorted(paths, key=lambda path: natural_key(path.relative_to(PROJECT).as_posix()))


def protected_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(PROJECT)): sha256_file(path)
        for path in protected_file_paths()
    }


def source_tree_signature(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    directory_count = 0
    total_bytes = 0
    if not root.is_dir():
        return {
            "root": str(root),
            "exists": False,
            "sha256_metadata_inventory": "",
            "file_count": 0,
            "directory_count": 0,
            "total_bytes": 0,
        }
    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        dirnames.sort(key=natural_key)
        filenames.sort(key=natural_key)
        current_path = Path(current)
        directory_count += 1
        rel_current = current_path.relative_to(root).as_posix()
        current_stat = current_path.stat()
        digest.update(
            f"D\t{rel_current}\t{current_stat.st_mode}\t{current_stat.st_mtime_ns}\n".encode("utf-8")
        )
        for filename in filenames:
            path = current_path / filename
            try:
                stat_result = path.stat()
            except OSError as exc:
                digest.update(f"E\t{path.relative_to(root).as_posix()}\t{exc}\n".encode("utf-8"))
                continue
            file_count += 1
            total_bytes += int(stat_result.st_size)
            digest.update(
                (
                    f"F\t{path.relative_to(root).as_posix()}\t{stat_result.st_mode}\t"
                    f"{stat_result.st_size}\t{stat_result.st_mtime_ns}\n"
                ).encode("utf-8")
            )
    return {
        "root": str(root),
        "exists": True,
        "sha256_metadata_inventory": digest.hexdigest(),
        "file_count": file_count,
        "directory_count": directory_count,
        "total_bytes": total_bytes,
    }


def preflight_no_conflicts() -> None:
    conflicts = [path for path in [*FINAL_OUTPUTS.values(), FAILURE_PATH] if path.exists()]
    if conflicts:
        formatted = "\n".join(str(path) for path in conflicts)
        raise FileExistsError(f"Refusing to overwrite existing api_fullseq_v2 targets:\n{formatted}")


class RunLogger:
    def __init__(self) -> None:
        log_dir = PROJECT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = log_dir / f"api_fullseq_v2_manifest_{stamp}_{os.getpid()}.log"
        self.handle = self.path.open("x", encoding="utf-8")

    def log(self, message: str) -> None:
        line = f"[{utc_now()}] {message}"
        print(line, flush=True)
        self.handle.write(line + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def build_auxiliary_tables(
    patient_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    frame_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    excluded_columns = [
        "patient_id", "split", "source_type", "source_medical_record_root",
        "patient_status", "exclusion_reason", "candidate_count", "valid_candidate_count",
        "all_candidate_series", "selected_series_id", "selected_series_path",
    ]
    excluded = patient_df[
        ~patient_df["patient_status"].astype(str).str.startswith("selected_")
    ][excluded_columns].copy()

    parameter_rows: list[dict[str, Any]] = []
    for row in candidate_df.to_dict("records"):
        for phase in ("pre", "post"):
            if bool(row[f"{phase}_parameter_only"]):
                parameter_rows.append({
                    "patient_id": row["patient_id"],
                    "split": row["split"],
                    "source_type": row["source_type"],
                    "discovery_rank": row["discovery_rank"],
                    "series_id": row["series_id"],
                    "phase": phase,
                    "api_dir": row[f"{phase}_api_dir"],
                    "image_file_count": row[f"{phase}_image_files"],
                    "selection_status": row["selection_status"],
                    "evidence": "all raster images under phase directory are CBF/CBV/MTT/TTP parameter maps",
                })
    parameter = pd.DataFrame(parameter_rows, columns=[
        "patient_id", "split", "source_type", "discovery_rank", "series_id", "phase",
        "api_dir", "image_file_count", "selection_status", "evidence",
    ])

    nonstandard_mask = frame_df["is_nonstandard_jpeg"].fillna(False).astype(bool)
    nonstandard = frame_df.loc[nonstandard_mask, [
        "patient_id", "split", "source_type", "discovery_rank", "series_id", "phase",
        "api_dir", "relative_path_in_api", "filename", "absolute_path", "suffix",
        "size_bytes", "readable", "height", "width", "selection_reason",
    ]].copy()
    unreadable_mask = (
        frame_df["read_attempted"].fillna(False).astype(bool)
        & ~frame_df["readable"].fillna(False).astype(bool)
    )
    unreadable = frame_df.loc[unreadable_mask, [
        "patient_id", "split", "source_type", "discovery_rank", "series_id", "phase",
        "api_dir", "relative_path_in_api", "filename", "absolute_path", "suffix",
        "size_bytes", "strict_filename_match", "is_parameter_map", "read_error",
    ]].copy()

    patient_by_id = patient_df.set_index("patient_id", drop=False)
    pilot_rows: list[dict[str, Any]] = []
    for order, patient_id in enumerate(PILOT_IDS, start=1):
        if patient_id not in patient_by_id.index:
            continue
        row = patient_by_id.loc[patient_id].to_dict()
        row["pilot_order"] = order
        row["pilot_reason"] = (
            "updated_10_fixed_mapping" if patient_id in UPDATED_SERIES else "boundary_validation_case"
        )
        pilot_rows.append(row)
    pilot = pd.DataFrame(pilot_rows, columns=["pilot_order", "pilot_reason", *PATIENT_COLUMNS])
    return {
        "excluded": excluded,
        "parameter": parameter,
        "nonstandard": nonstandard,
        "unreadable": unreadable,
        "pilot": pilot,
    }


def parse_indices(value: Any) -> list[int]:
    if value is None or pd.isna(value) or str(value) == "":
        return []
    return [int(part) for part in str(value).split("|") if part != ""]


def compute_stats(
    patient_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    frame_df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    workers: int,
    cuda_detected: bool,
) -> dict[str, Any]:
    status_counts = patient_df["patient_status"].value_counts().to_dict()
    runnable = patient_df[patient_df["patient_status"].astype(str).str.startswith("selected_")]
    no_valid = patient_df[~patient_df["patient_status"].astype(str).str.startswith("selected_")]
    gap_phase_count = int(
        candidate_df["pre_frame_gaps"].fillna("").ne("").sum()
        + candidate_df["post_frame_gaps"].fillna("").ne("").sum()
    )
    selected_gap_phase_count = int(
        patient_df["pre_frame_gaps"].fillna("").ne("").sum()
        + patient_df["post_frame_gaps"].fillna("").ne("").sum()
    )
    selected_frames = frame_df[frame_df["selected"].fillna(False).astype(bool)]
    return {
        "train_unique_patients": int((patient_df["split"] == "Train").sum()),
        "valid_unique_patients": int((patient_df["split"] == "Valid").sum()),
        "total_unique_patients": len(patient_df),
        "patient_manifest_rows": len(patient_df),
        "runnable_train_patients": int((runnable["split"] == "Train").sum()),
        "runnable_valid_patients": int((runnable["split"] == "Valid").sum()),
        "selected_prepost": int(status_counts.get("selected_prepost", 0)),
        "selected_pre_only": int(status_counts.get("selected_pre_only", 0)),
        "selected_post_only": int(status_counts.get("selected_post_only", 0)),
        "no_valid_grayscale_api_total": len(no_valid),
        "no_valid_reason_distribution": {
            str(key): int(value) for key, value in no_valid["patient_status"].value_counts().to_dict().items()
        },
        "exclusion_reason_distribution": {
            str(key): int(value) for key, value in no_valid["exclusion_reason"].value_counts().to_dict().items()
        },
        "patients_with_multiple_candidates": int((patient_df["candidate_count"] > 1).sum()),
        "patients_with_multiple_valid_candidates_selected_first": int(
            (patient_df["valid_candidate_count"] > 1).sum()
        ),
        "candidate_series_rows": len(candidate_df),
        "frame_audit_rows": len(frame_df),
        "parameter_only_directories": len(tables["parameter"]),
        "nonstandard_jpeg_files": len(tables["nonstandard"]),
        "unreadable_files": len(tables["unreadable"]),
        "phases_with_frame_gaps_all_candidates": gap_phase_count,
        "selected_phases_with_frame_gaps": selected_gap_phase_count,
        "selected_png_files": int((selected_frames["suffix"].str.casefold() == ".png").sum()),
        "selected_parameter_map_files": int(
            selected_frames["is_parameter_map"].fillna(False).astype(bool).sum()
        ),
        "pilot_manifest_rows": len(tables["pilot"]),
        "updated_cases_in_patient_manifest": int(patient_df["patient_id"].isin(UPDATED_IDS).sum()),
        "updated_train_count": int(
            (patient_df["patient_id"].isin(UPDATED_IDS) & (patient_df["split"] == "Train")).sum()
        ),
        "updated_valid_count": int(
            (patient_df["patient_id"].isin(UPDATED_IDS) & (patient_df["split"] == "Valid")).sum()
        ),
        "cpu_workers": workers,
        "cuda_detected": cuda_detected,
        "gpu_used": False,
    }


def run_assertions(
    patient_df: pd.DataFrame,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    frame_df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    split_metadata: list[dict[str, Any]],
    protected_before: dict[str, str],
    protected_after: dict[str, str],
    source_before: dict[str, dict[str, Any]],
    source_after: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    assertions: list[dict[str, Any]] = []

    def check(name: str, passed: Any, detail: str) -> None:
        assertions.append({"name": name, "passed": bool(passed), "detail": detail})

    train_meta = next(item for item in split_metadata if item["split"] == "Train")
    valid_meta = next(item for item in split_metadata if item["split"] == "Valid")
    train_ids = set(patient_df.loc[patient_df["split"] == "Train", "patient_id"])
    valid_ids = set(patient_df.loc[patient_df["split"] == "Valid", "patient_id"])
    check("train_unique_patient_count", train_meta["unique_patient_ids"] == 1055, str(train_meta))
    check("valid_unique_patient_count", valid_meta["unique_patient_ids"] == 264, str(valid_meta))
    check("train_valid_intersection_zero", not (train_ids & valid_ids), f"intersection={len(train_ids & valid_ids)}")
    check("total_unique_patient_count", len(train_ids | valid_ids) == 1319, f"total={len(train_ids | valid_ids)}")
    check("patient_manifest_rows", len(patient_df) == 1319, f"rows={len(patient_df)}")
    check("patient_manifest_patient_id_unique", patient_df["patient_id"].is_unique, "patient_id must be unique")
    check("patient_id_is_standardized_medical_record_number", patient_df["patient_id"].str.fullmatch(r"\d+").all(), "all patient_id values numeric strings")

    normal_candidates = candidate_df[candidate_df["source_type"] == "tiantanDSA"]
    selected_counts = normal_candidates.groupby("patient_id")["selected_candidate"].sum()
    valid_counts = normal_candidates.groupby("patient_id")["candidate_valid"].sum()
    normal_patient_ids = set(patient_df.loc[patient_df["source_type"] == "tiantanDSA", "patient_id"])
    ordinary_ok = True
    for patient_id in normal_patient_ids:
        selected_count = int(selected_counts.get(patient_id, 0))
        valid_count = int(valid_counts.get(patient_id, 0))
        if selected_count != (1 if valid_count > 0 else 0):
            ordinary_ok = False
            break
    check("ordinary_patient_selected_candidate_cardinality", ordinary_ok, "one selected iff at least one valid candidate")

    first_valid_ok = True
    for patient_id, group in normal_candidates.groupby("patient_id"):
        valid_group = group[group["candidate_valid"].astype(bool)]
        selected_group = group[group["selected_candidate"].astype(bool)]
        if len(valid_group):
            if len(selected_group) != 1 or int(selected_group.iloc[0]["discovery_rank"]) != int(valid_group["discovery_rank"].min()):
                first_valid_ok = False
                break
    check("selected_is_minimum_valid_discovery_rank", first_valid_ok, "ordinary cases use first valid candidate")
    check(
        "multiple_valid_candidates_not_excluded",
        patient_df.loc[
            (patient_df["source_type"] == "tiantanDSA") & (patient_df["valid_candidate_count"] > 1),
            "patient_status",
        ].astype(str).str.startswith("selected_").all(),
        f"multiple_valid={int((patient_df['valid_candidate_count'] > 1).sum())}",
    )

    rank_ok = True
    for _, group in candidate_df.groupby("patient_id"):
        ranks = group["discovery_rank"].astype(int).tolist()
        if ranks != list(range(1, len(ranks) + 1)):
            rank_ok = False
            break
    check("candidate_discovery_ranks_are_sequential", rank_ok, "ranks start at 1 and are deterministic")

    patient_by_id = patient_df.set_index("patient_id", drop=False)
    updated_present = set(UPDATED_IDS).issubset(set(patient_df["patient_id"]))
    check("updated_10_all_in_patient_manifest", updated_present, f"present={sum(x in patient_by_id.index for x in UPDATED_IDS)}")
    updated_rows = patient_df[patient_df["patient_id"].isin(UPDATED_IDS)]
    check(
        "updated_10_split_9_train_1_valid",
        int((updated_rows["split"] == "Train").sum()) == 9
        and int((updated_rows["split"] == "Valid").sum()) == 1
        and patient_by_id.loc["549117", "split"] == "Valid",
        updated_rows[["patient_id", "split"]].to_dict("records"),
    )
    check(
        "updated_10_source_type",
        (updated_rows["source_type"] == "updated_10_cases").all(),
        pipe(updated_rows["source_type"].unique()),
    )
    updated_source_ok = all(
        Path(patient_by_id.loc[patient_id, "source_medical_record_root"])
        == (UPDATED_ROOT / patient_id).resolve()
        for patient_id in UPDATED_IDS
    )
    check("updated_10_source_roots_are_staging", updated_source_ok, "no old tiantanDSA source roots")
    mapping_ok = all(
        str(patient_by_id.loc[patient_id, "selected_series_id"]) == series_id
        for patient_id, series_id in UPDATED_SERIES.items()
    )
    check("updated_10_fixed_series_mapping", mapping_ok, str(UPDATED_SERIES))
    old_source_reads = frame_df[
        frame_df["patient_id"].isin(UPDATED_IDS)
        & frame_df["absolute_path"].astype(str).str.startswith(str(DATA_ROOT.resolve()) + "/")
    ]
    check("updated_10_old_source_read_count_zero", len(old_source_reads) == 0, f"count={len(old_source_reads)}")
    ignored_updated_reads = 0
    for patient_id, fixed_series in UPDATED_SERIES.items():
        fixed_prefix = str((UPDATED_ROOT / patient_id / fixed_series).resolve()) + "/"
        ignored_updated_reads += int(
            (
                (frame_df["patient_id"] == patient_id)
                & ~frame_df["absolute_path"].astype(str).str.startswith(fixed_prefix)
            ).sum()
        )
    check("updated_10_ignored_sibling_read_count_zero", ignored_updated_reads == 0, f"count={ignored_updated_reads}")

    pilot_ids = tables["pilot"]["patient_id"].astype(str).tolist()
    check("pilot_manifest_exact_16_unique_patients", len(pilot_ids) == 16 and len(set(pilot_ids)) == 16, str(pilot_ids))
    check("pilot_contains_all_updated_10", set(UPDATED_IDS).issubset(set(pilot_ids)), str(pilot_ids))

    selected_frames = frame_df[frame_df["selected"].fillna(False).astype(bool)]
    check("selected_png_count_zero", not (selected_frames["suffix"].str.casefold() == ".png").any(), f"count={int((selected_frames['suffix'].str.casefold() == '.png').sum())}")
    check("selected_parameter_map_count_zero", not selected_frames["is_parameter_map"].fillna(False).astype(bool).any(), f"count={int(selected_frames['is_parameter_map'].fillna(False).astype(bool).sum())}")
    check("all_selected_frames_strict_jpeg", selected_frames["strict_filename_match"].fillna(False).astype(bool).all() and selected_frames["suffix"].isin(JPEG_SUFFIXES).all(), f"selected={len(selected_frames)}")
    check("all_selected_frames_readable", selected_frames["readable"].fillna(False).astype(bool).all(), f"selected={len(selected_frames)}")

    runnable_phase_ok = True
    pair_formula_ok = True
    frame_count_ok = True
    for row in patient_df.to_dict("records"):
        for phase in ("pre", "post"):
            indices = parse_indices(row[f"{phase}_frame_indices"])
            actual_pairs = sum(1 for left, right in zip(indices, indices[1:]) if right - left == 1)
            if actual_pairs != int(row[f"n_{phase}_contiguous_pairs"]):
                pair_formula_ok = False
            if bool(row[f"can_run_{phase}"]):
                if len(indices) < 2 or actual_pairs < 1:
                    runnable_phase_ok = False
                selected_count = int(
                    (
                        (selected_frames["patient_id"] == row["patient_id"])
                        & (selected_frames["phase"] == phase)
                    ).sum()
                )
                if selected_count != int(row[f"n_{phase}_frames"]):
                    frame_count_ok = False
    check("all_runnable_phases_have_minimum_frames_and_pair", runnable_phase_ok, "n_frames>=2 and actual pairs>=1")
    check("contiguous_pair_counts_are_actual_adjacent_pairs", pair_formula_ok, "never n_frames-1 across gaps")
    check("selected_frame_counts_match_patient_manifest", frame_count_ok, "frame audit selected counts match")

    dimension_ok = True
    for _, group in selected_frames.groupby(["patient_id", "phase"]):
        if group[["height", "width"]].drop_duplicates().shape[0] != 1:
            dimension_ok = False
            break
    check("selected_internal_series_dimensions_consistent", dimension_ok, "one HxW per selected phase")
    check("frame_audit_unique_paths", not frame_df[["patient_id", "discovery_rank", "phase", "absolute_path"]].duplicated().any(), "no duplicate file audit rows")

    train_expected = patient_df[
        (patient_df["split"] == "Train")
        & patient_df["patient_status"].astype(str).str.startswith("selected_")
    ]["patient_id"].tolist()
    valid_expected = patient_df[
        (patient_df["split"] == "Valid")
        & patient_df["patient_status"].astype(str).str.startswith("selected_")
    ]["patient_id"].tolist()
    check("train_execution_manifest_strict_subset", train_df["patient_id"].tolist() == train_expected, f"rows={len(train_df)}")
    check("valid_execution_manifest_strict_subset", valid_df["patient_id"].tolist() == valid_expected, f"rows={len(valid_df)}")

    forbidden_columns = {"excel_row", "excel_row_number", "lesion_index", "api_sequence_number"}
    check("no_excel_row_or_lesion_join_keys", not (forbidden_columns & set(patient_df.columns)), str(forbidden_columns & set(patient_df.columns)))

    def selected_series(patient_id: str) -> str:
        return str(patient_by_id.loc[patient_id, "selected_series_id"])

    c453933 = candidate_df[candidate_df["patient_id"] == "453933"].sort_values("discovery_rank")
    check(
        "case_453933_candidates_and_first_post_only",
        c453933["series_id"].tolist()[:3] == ["L/Ach", "L/Pcom", "R/Pcom"]
        and selected_series("453933") == "L/Ach"
        and bool(patient_by_id.loc["453933", "can_run_post"])
        and not bool(patient_by_id.loc["453933", "can_run_pre"]),
        c453933[["series_id", "selection_status", "can_run_pre", "can_run_post"]].to_dict("records"),
    )
    c474686 = candidate_df[candidate_df["patient_id"] == "474686"].sort_values("discovery_rank")
    check(
        "case_474686_selects_C6_ignores_Pcom",
        c474686["series_id"].tolist()[:2] == ["C6", "Pcom"]
        and selected_series("474686") == "C6"
        and "Pcom" in str(patient_by_id.loc["474686", "ignored_series"]).split("|"),
        c474686[["series_id", "selection_status"]].to_dict("records"),
    )
    c654075 = candidate_df[candidate_df["patient_id"] == "654075"].set_index("series_id")
    check(
        "case_654075_parameter_main_then_C6",
        "main" in c654075.index and bool(c654075.loc["main", "post_parameter_only"])
        and selected_series("654075") == "C6",
        c654075[["post_parameter_only", "selection_status"]].reset_index().to_dict("records"),
    )
    c654873 = candidate_df[candidate_df["patient_id"] == "654873"].set_index("series_id")
    check(
        "case_654873_parameter_main_then_L",
        "main" in c654873.index
        and bool(c654873.loc["main", "pre_parameter_only"])
        and bool(c654873.loc["main", "post_parameter_only"])
        and selected_series("654873") == "L",
        c654873[["pre_parameter_only", "post_parameter_only", "selection_status"]].reset_index().to_dict("records"),
    )
    r571569 = patient_by_id.loc["571569"]
    check(
        "case_571569_real_gap_pair_count",
        "7->9" in str(r571569["post_frame_gaps"])
        and int(r571569["n_post_frames"]) == 16
        and int(r571569["n_post_contiguous_pairs"]) == 14,
        f"post_frames={r571569['n_post_frames']} post_pairs={r571569['n_post_contiguous_pairs']} gaps={r571569['post_frame_gaps']}",
    )
    r348817 = patient_by_id.loc["348817"]
    check(
        "case_348817_ordinary_complete",
        selected_series("348817") == "main"
        and bool(r348817["can_run_prepost"])
        and r348817["source_type"] == "tiantanDSA",
        r348817[["selected_series_id", "can_run_pre", "can_run_post"]].to_dict(),
    )
    r549117 = patient_by_id.loc["549117"]
    check(
        "case_549117_staging_C6_1_only",
        r549117["source_type"] == "updated_10_cases"
        and selected_series("549117") == "C6-1"
        and str(r549117["selected_series_path"]).startswith(str((UPDATED_ROOT / "549117").resolve()) + "/"),
        r549117[["source_medical_record_root", "selected_series_id", "selected_series_path"]].to_dict(),
    )

    check("protected_v1_metadata_code_hashes_unchanged", protected_before == protected_after, f"files={len(protected_before)}")
    check("raw_and_staging_source_metadata_unchanged", source_before == source_after, json.dumps(source_after, ensure_ascii=False))
    return assertions


def assertion_markdown(assertions: list[dict[str, Any]]) -> list[str]:
    return [
        f"- {'PASS' if item['passed'] else 'FAIL'} — {item['name']}: {item['detail']}"
        for item in assertions
    ]


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(clean(value) for value in row) + " |" for row in rows],
    ]


def updated_case_rows(patient_df: pd.DataFrame) -> list[list[Any]]:
    indexed = patient_df.set_index("patient_id")
    rows: list[list[Any]] = []
    for patient_id in UPDATED_IDS:
        row = indexed.loc[patient_id]
        rows.append([
            patient_id,
            row["split"],
            row["source_medical_record_root"],
            row["selected_series_id"],
            f"{row['n_pre_frames']}/{row['n_pre_contiguous_pairs']}",
            f"{row['n_post_frames']}/{row['n_post_contiguous_pairs']}",
            "yes" if bool(str(row["ignored_series"])) else "no",
            "no",
        ])
    return rows


def boundary_case_rows(
    patient_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
) -> list[list[Any]]:
    patient_index = patient_df.set_index("patient_id")
    rows: list[list[Any]] = []
    for patient_id in BOUNDARY_IDS:
        patient = patient_index.loc[patient_id]
        candidates = candidate_df[candidate_df["patient_id"] == patient_id].sort_values("discovery_rank")
        candidate_desc = "; ".join(
            f"{row.series_id}[pre={bool(row.can_run_pre)},post={bool(row.can_run_post)},"
            f"parameter_pre={bool(row.pre_parameter_only)},parameter_post={bool(row.post_parameter_only)}]"
            for row in candidates.itertuples(index=False)
        )
        detail = (
            f"pre_frames/pairs={patient['n_pre_frames']}/{patient['n_pre_contiguous_pairs']}; "
            f"post_frames/pairs={patient['n_post_frames']}/{patient['n_post_contiguous_pairs']}; "
            f"pre_gaps={patient['pre_frame_gaps'] or '-'}; post_gaps={patient['post_frame_gaps'] or '-'}"
        )
        rows.append([
            patient_id,
            candidate_desc,
            patient["selected_series_id"],
            patient["ignored_series"],
            patient["patient_status"],
            detail,
        ])
    return rows


def build_manifest_audit_markdown(
    stats: dict[str, Any],
    assertions: list[dict[str, Any]],
    patient_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    split_metadata: list[dict[str, Any]],
    source_before: dict[str, dict[str, Any]],
    source_after: dict[str, dict[str, Any]],
    log_path: Path,
) -> str:
    lines = [
        "# api_fullseq_v2 patient-level manifest audit",
        "",
        f"- Generated at: {utc_now()}",
        f"- Run log: {log_path}",
        "- Primary key: standardized Excel 病案号 (patient_id), one unique medical-record number per row.",
        "- Candidate validity: runnable Pre OR runnable Post; Pre-only, Post-only, and Pre+Post are allowed.",
        "- Selection: deterministic natural-order recursive discovery, then minimum valid discovery rank.",
        "- Updated 10: fixed staging mapping only; ignored siblings are not image-scanned; no old-root fallback.",
        "- SEA-RAFT / optical flow / training: not invoked.",
        "",
        "## Input row and unique-patient counts",
        "",
        *markdown_table(
            ["split", "Excel rows", "unique 病案号", "duplicate rows collapsed"],
            [
                [
                    item["split"], item["excel_rows"], item["unique_patient_ids"],
                    item["duplicate_excel_rows_after_first"],
                ]
                for item in split_metadata
            ],
        ),
        "",
        "## Summary statistics",
        "",
        *markdown_table(
            ["metric", "value"],
            [[key, json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, dict) else value]
             for key, value in stats.items()],
        ),
        "",
        "## No-valid-grayscale-API distribution",
        "",
        *markdown_table(
            ["patient_status", "count"],
            [[key, value] for key, value in stats["no_valid_reason_distribution"].items()],
        ),
        "",
        "## Updated 10 fixed-mapping cases",
        "",
        *markdown_table(
            [
                "patient_id", "split", "source root", "fixed selected series",
                "Pre frames/pairs", "Post frames/pairs", "ignored sibling exists",
                "old-root fallback",
            ],
            updated_case_rows(patient_df),
        ),
        "",
        "## Required boundary cases",
        "",
        *markdown_table(
            ["patient_id", "discovered candidates", "selected", "ignored", "status", "frame evidence"],
            boundary_case_rows(patient_df, candidate_df),
        ),
        "",
        "## Source immutability evidence",
        "",
        f"- Before: {json.dumps(source_before, ensure_ascii=False, sort_keys=True)}",
        f"- After: {json.dumps(source_after, ensure_ascii=False, sort_keys=True)}",
        "",
        "## Hard assertions",
        "",
        *assertion_markdown(assertions),
        "",
    ]
    return "\n".join(lines)


def build_pilot_report_markdown(
    pilot_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
) -> str:
    rows: list[list[Any]] = []
    for patient in pilot_df.itertuples(index=False):
        candidates = candidate_df[candidate_df["patient_id"] == patient.patient_id].sort_values("discovery_rank")
        rows.append([
            patient.pilot_order,
            patient.patient_id,
            patient.pilot_reason,
            patient.split,
            patient.source_type,
            patient.selected_series_id,
            patient.patient_status,
            pipe(candidates["series_id"].tolist()),
            patient.ignored_series,
            f"{patient.n_pre_frames}/{patient.n_pre_contiguous_pairs}",
            f"{patient.n_post_frames}/{patient.n_post_contiguous_pairs}",
        ])
    lines = [
        "# api_fullseq_v2 pilot selection report",
        "",
        "This file is a manifest-only pilot list. SEA-RAFT and optical-flow inference were not run.",
        "",
        *markdown_table(
            [
                "order", "patient_id", "reason", "split", "source", "selected series",
                "status", "all candidates", "ignored", "Pre frames/pairs", "Post frames/pairs",
            ],
            rows,
        ),
        "",
    ]
    return "\n".join(lines)


def write_dataframe(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def atomic_publish_file(source: Path, destination: Path) -> None:
    """Publish one completed /tmp file atomically on the destination filesystem."""
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing target: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(destination.parent), os.O_RDWR | os.O_TMPFILE, 0o644)
    try:
        os.fchmod(descriptor, 0o644)
        with source.open("rb") as source_handle:
            while True:
                chunk = source_handle.read(1024 * 1024)
                if not chunk:
                    break
                view = memoryview(chunk)
                while view:
                    written = os.write(descriptor, view)
                    view = view[written:]
        os.fsync(descriptor)
        libc = ctypes.CDLL(None, use_errno=True)
        linkat = libc.linkat
        linkat.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int
        ]
        linkat.restype = ctypes.c_int
        destination_bytes = os.fsencode(destination.resolve())
        result = linkat(descriptor, b"", -100, destination_bytes, 0x1000)
        if result != 0:
            first_error = ctypes.get_errno()
            proc_fd_path = os.fsencode(f"/proc/self/fd/{descriptor}")
            result = linkat(-100, proc_fd_path, -100, destination_bytes, 0x400)
            if result != 0:
                error_number = ctypes.get_errno()
                raise OSError(
                    error_number,
                    (
                        f"{os.strerror(error_number)}; "
                        f"AT_EMPTY_PATH first error={first_error}:{os.strerror(first_error)}"
                    ),
                    str(destination),
                )
        directory_descriptor = os.open(
            str(destination.parent), os.O_RDONLY | os.O_DIRECTORY
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        os.close(descriptor)


def write_failure_report(stage: str, exc: BaseException) -> None:
    if FAILURE_PATH.exists():
        return
    formal_existing = [str(path) for path in FINAL_OUTPUTS.values() if path.exists()]
    text = "\n".join([
        "# api_fullseq_v2 failure",
        "",
        f"- Stage: {stage}",
        f"- Exception: {type(exc).__name__}: {exc}",
        f"- Affected paths: {json.dumps(formal_existing, ensure_ascii=False)}",
        f"- Formal v2 artifacts written: {'yes' if formal_existing else 'no'}",
        "- Source data modified: no writes were issued to metadata, staging, or tiantanDSA inputs.",
        "- CUDA used: no.",
        "- SEA-RAFT used: no.",
        "",
        "## Traceback",
        "",
        traceback.format_exc(),
        "",
    ])
    FAILURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix="api_fullseq_v2_failure_", suffix=".md", dir="/tmp")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(text, encoding="utf-8")
        atomic_publish_file(temporary, FAILURE_PATH)
    finally:
        if temporary.exists():
            temporary.unlink()


def commit_outputs(temp_paths: dict[str, Path]) -> None:
    conflicts = [path for path in FINAL_OUTPUTS.values() if path.exists()]
    if conflicts:
        raise FileExistsError(
            "Output conflict appeared before atomic commit: " + pipe(str(path) for path in conflicts)
        )
    for final_path in FINAL_OUTPUTS.values():
        final_path.parent.mkdir(parents=True, exist_ok=True)
    for key in FINAL_OUTPUTS:
        atomic_publish_file(temp_paths[key], FINAL_OUTPUTS[key])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > DEFAULT_WORKERS:
        raise ValueError(f"--workers must be between 1 and {DEFAULT_WORKERS}")

    stage = "preflight"
    logger: RunLogger | None = None
    temp_dir: Path | None = None
    try:
        preflight_no_conflicts()
        logger = RunLogger()
        logger.log("Starting api_fullseq_v2 patient-level manifest build")
        logger.log(f"CPU workers={args.workers}; OpenCV={cv2.__version__}")
        cuda_devices = cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2, "cuda") else 0
        cuda_detected = cuda_devices > 0
        logger.log(f"CUDA devices detected by OpenCV={cuda_devices}; GPU used=False")

        stage = "protected_input_hashes_before"
        protected_before = protected_hashes()
        logger.log(f"Protected file hashes captured: {len(protected_before)} files")

        stage = "source_tree_signature_before"
        logger.log("Capturing pre-run raw/staging source metadata signatures")
        source_before = {
            "tiantanDSA": source_tree_signature(DATA_ROOT),
            "updated_10_cases": source_tree_signature(UPDATED_ROOT),
        }
        logger.log(
            "Pre-run source inventory: "
            f"tiantanDSA files={source_before['tiantanDSA']['file_count']}; "
            f"updated files={source_before['updated_10_cases']['file_count']}"
        )

        stage = "load_excel_splits"
        train_ids, train_meta = load_split(TRAIN_XLSX, "Train")
        valid_ids, valid_meta = load_split(VALID_XLSX, "Valid")
        split_metadata = [train_meta, valid_meta]
        if len(train_ids) != 1055 or len(valid_ids) != 264 or set(train_ids) & set(valid_ids):
            raise AssertionError(f"Excel split hard counts failed: {split_metadata}")
        patients = [(patient_id, "Train") for patient_id in train_ids]
        patients.extend((patient_id, "Valid") for patient_id in valid_ids)
        logger.log(
            f"Excel authority loaded: Train={len(train_ids)}, Valid={len(valid_ids)}, total={len(patients)}"
        )

        stage = "scan_all_patients"
        patient_rows: list[dict[str, Any]] = []
        candidate_rows: list[dict[str, Any]] = []
        frame_rows: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="api_v2_jpeg") as executor:
            for index, (patient_id, split) in enumerate(patients, start=1):
                patient_row, patient_candidates, patient_frames = build_patient(
                    patient_id, split, executor
                )
                patient_rows.append(patient_row)
                candidate_rows.extend(patient_candidates)
                frame_rows.extend(patient_frames)
                if index % PROGRESS_INTERVAL == 0 or index == len(patients):
                    logger.log(
                        f"Progress {index}/{len(patients)} patients; "
                        f"candidate_rows={len(candidate_rows)}; frame_rows={len(frame_rows)}"
                    )

        stage = "assemble_dataframes"
        patient_df = pd.DataFrame(patient_rows, columns=PATIENT_COLUMNS)
        candidate_df = pd.DataFrame(candidate_rows, columns=CANDIDATE_COLUMNS)
        frame_df = pd.DataFrame(frame_rows, columns=FRAME_COLUMNS)
        runnable_mask = patient_df["patient_status"].astype(str).str.startswith("selected_")
        train_df = patient_df[(patient_df["split"] == "Train") & runnable_mask].copy()
        valid_df = patient_df[(patient_df["split"] == "Valid") & runnable_mask].copy()
        tables = build_auxiliary_tables(patient_df, candidate_df, frame_df)

        stage = "protected_input_hashes_after"
        protected_after = protected_hashes()
        logger.log("Capturing post-run raw/staging source metadata signatures")
        source_after = {
            "tiantanDSA": source_tree_signature(DATA_ROOT),
            "updated_10_cases": source_tree_signature(UPDATED_ROOT),
        }

        stage = "hard_assertions"
        assertions = run_assertions(
            patient_df, train_df, valid_df, candidate_df, frame_df, tables,
            split_metadata, protected_before, protected_after, source_before, source_after,
        )
        failed = [item for item in assertions if not item["passed"]]
        for item in assertions:
            logger.log(f"ASSERT {'PASS' if item['passed'] else 'FAIL'} {item['name']}: {item['detail']}")
        if failed:
            raise AssertionError(
                "Hard assertions failed: " + pipe(item["name"] for item in failed)
            )

        stage = "statistics_and_reports"
        stats = compute_stats(
            patient_df, candidate_df, frame_df, tables, args.workers, cuda_detected
        )
        manifest_audit = build_manifest_audit_markdown(
            stats, assertions, patient_df, candidate_df, split_metadata,
            source_before, source_after, logger.path,
        )
        pilot_report = build_pilot_report_markdown(tables["pilot"], candidate_df)

        stage = "write_temporary_outputs"
        temp_dir = Path(tempfile.mkdtemp(prefix="api_fullseq_v2_", dir="/tmp"))
        temp_paths = {
            key: temp_dir / f"{key}{FINAL_OUTPUTS[key].suffix}"
            for key in FINAL_OUTPUTS
        }
        write_dataframe(patient_df, temp_paths["patient_manifest"])
        write_dataframe(train_df, temp_paths["train_manifest"])
        write_dataframe(valid_df, temp_paths["valid_manifest"])
        write_dataframe(candidate_df, temp_paths["candidate_audit"])
        write_dataframe(frame_df, temp_paths["frame_audit"])
        write_dataframe(tables["pilot"], temp_paths["pilot_manifest"])
        temp_paths["manifest_audit"].write_text(manifest_audit, encoding="utf-8")
        write_dataframe(tables["excluded"], temp_paths["excluded_evidence"])
        write_dataframe(tables["parameter"], temp_paths["parameter_only"])
        write_dataframe(tables["nonstandard"], temp_paths["nonstandard"])
        write_dataframe(tables["unreadable"], temp_paths["unreadable"])
        temp_paths["pilot_report"].write_text(pilot_report, encoding="utf-8")

        output_hashes = {
            key: sha256_file(path)
            for key, path in temp_paths.items()
            if key != "config"
        }
        config = {
            "manifest_version": "api_fullseq_v2",
            "generated_at_utc": utc_now(),
            "project_root": str(PROJECT),
            "data_root": str(DATA_ROOT),
            "updated_root": str(UPDATED_ROOT),
            "train_excel": str(TRAIN_XLSX),
            "valid_excel": str(VALID_XLSX),
            "primary_key": "standardized Excel 病案号 / patient_id",
            "one_row_level": "unique medical-record number, not lesion and not Excel row",
            "recursive_candidate_discovery_max_depth": MAX_DISCOVERY_DEPTH,
            "candidate_discovery_order": "preorder DFS; current directory first; case-insensitive natural-sorted children",
            "candidate_validity": "can_run_pre OR can_run_post",
            "candidate_selection": "minimum discovery_rank among valid candidates",
            "strict_frame_regex": STRICT_FRAME_RE.pattern,
            "parameter_tokens": list(PARAMETER_TOKENS),
            "internal_series_selection": "minimum numeric runnable internal_series_number",
            "contiguous_pair_definition": "sum(frame_index[i+1]-frame_index[i] == 1)",
            "workers": args.workers,
            "progress_interval_patients": PROGRESS_INTERVAL,
            "opencv_version": cv2.__version__,
            "cuda_device_count": cuda_devices,
            "cuda_detected": cuda_detected,
            "gpu_used": False,
            "sea_raft_used": False,
            "optical_flow_run": False,
            "model_training_run": False,
            "updated_fixed_series": UPDATED_SERIES,
            "pilot_patient_ids": PILOT_IDS,
            "split_metadata": split_metadata,
            "statistics": stats,
            "hard_assertions": assertions,
            "protected_hashes_before": protected_before,
            "protected_hashes_after": protected_after,
            "source_signatures_before": source_before,
            "source_signatures_after": source_after,
            "temporary_output_sha256": output_hashes,
            "final_outputs": {key: str(path) for key, path in FINAL_OUTPUTS.items()},
            "run_log": str(logger.path),
        }
        temp_paths["config"].write_text(
            json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        stage = "atomic_commit"
        commit_outputs(temp_paths)
        logger.log("All hard assertions passed and formal outputs committed atomically")
        logger.log(json.dumps(stats, ensure_ascii=False, sort_keys=True))
        return 0
    except BaseException as exc:
        if logger is not None:
            logger.log(f"FAIL stage={stage}: {type(exc).__name__}: {exc}")
        write_failure_report(stage, exc)
        raise
    finally:
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir)
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    sys.exit(main())


#!/usr/bin/env python3
"""Build the frozen API full-sequence manifest using CPU-only filesystem audit."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


PROJECT = Path("/root/autodl-tmp/aneurysm")
DATA_ROOT = Path("/root/autodl-tmp/tiantanDSA")
UPDATED_ROOT = PROJECT / "staging/updated_10_cases"
TRAIN_XLSX = PROJECT / "metadata/Train.xlsx"
VALID_XLSX = PROJECT / "metadata/valid.xlsx"
LEGACY_MANIFEST = PROJECT / "manifests/flow_manifest.csv"

MANIFEST_DIR = PROJECT / "manifests"
REPORT_DIR = PROJECT / "reports/api_fullseq_v1"
CONFIG_DIR = PROJECT / "configs"
FAILURE_PATH = REPORT_DIR / "overnight_failure.md"

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

FRAME_RE = re.compile(r"^IMG-(\d+)-(\d+)\.(jpg|jpeg)$", re.IGNORECASE)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PARAMETER_MAP_KEYS = ("cbf", "cbv", "mtt", "ttp")

OUTPUTS = {
    "series_manifest": MANIFEST_DIR / "api_fullseq_v1_series_manifest.csv",
    "train_manifest": MANIFEST_DIR / "api_fullseq_v1_train_manifest.csv",
    "valid_manifest": MANIFEST_DIR / "api_fullseq_v1_valid_manifest.csv",
    "manifest_audit_md": REPORT_DIR / "manifest_audit.md",
    "manifest_audit_csv": REPORT_DIR / "manifest_audit.csv",
    "excluded": REPORT_DIR / "excluded.csv",
    "manual_review": REPORT_DIR / "manual_series_review.csv",
    "updated_selection": REPORT_DIR / "updated_10_series_selection.csv",
    "frame_audit": REPORT_DIR / "frame_selection_audit.csv",
    "split_audit": REPORT_DIR / "split_audit.csv",
    "config": CONFIG_DIR / "api_fullseq_v1_manifest_config.json",
    "updated_config": CONFIG_DIR / "api_fullseq_v1_updated_series.json",
}


def normalize_patient_id(value: Any) -> str | None:
    if pd.isna(value):
        return None
    value = str(value).strip()
    if value.endswith(".0"):
        value = value[:-2]
    return value or None


def bool_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_split(path: Path, split: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing authoritative split workbook: {path}")
    frame = pd.read_excel(path, dtype=str)
    id_columns = [column for column in frame.columns if "病案号" in str(column)]
    if len(id_columns) != 1:
        raise RuntimeError(f"Expected exactly one patient-id column in {path}; found {id_columns}")
    result = pd.DataFrame(
        {
            "patient_id": frame[id_columns[0]].map(normalize_patient_id),
            "split": split,
        }
    ).dropna(subset=["patient_id"])
    return result.drop_duplicates(subset=["patient_id"], keep="first").reset_index(drop=True)


def sha256_lines(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def dimension_string(shapes: list[tuple[int, int]]) -> str | None:
    if not shapes:
        return None
    return "|".join(f"{height}x{width}" for height, width in sorted(set(shapes)))


def find_parameter_map(api_dir: Path, key: str) -> str | None:
    if not api_dir.is_dir():
        return None
    matches = sorted(
        path
        for path in api_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png" and key in path.name.lower()
    )
    return str(matches[0].resolve()) if matches else None


def find_segmentation(series_dir: Path, phase_title: str) -> Path | None:
    direct_candidates = [
        series_dir / f"{phase_title}-Segmentation.nii.gz",
        series_dir / f"{phase_title}-Segmentation.nii",
    ]
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate
    annotation_dir = series_dir / f"{phase_title}-biaozhu"
    if annotation_dir.is_dir():
        candidates = sorted(
            path
            for path in annotation_dir.iterdir()
            if path.is_file() and (path.name.lower().endswith(".nii") or path.name.lower().endswith(".nii.gz"))
        )
        if candidates:
            return candidates[0]
    return None


def discover_subseries(patient_root: Path) -> tuple[list[Path], list[Path]]:
    complete: list[Path] = []
    any_api: list[Path] = []
    if not patient_root.is_dir():
        return complete, any_api
    for child in sorted((p for p in patient_root.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
        has_pre = (child / "Pre-API").is_dir()
        has_post = (child / "Post-API").is_dir()
        if has_pre or has_post:
            any_api.append(child)
        if has_pre and has_post:
            complete.append(child)
    return complete, any_api


def select_series(
    patient_id: str,
    split: str,
    legacy_row: dict[str, Any] | None,
) -> dict[str, Any]:
    if patient_id in UPDATED_SERIES:
        patient_root = UPDATED_ROOT / patient_id
        selected_id = UPDATED_SERIES[patient_id]
        selected_dir = patient_root / selected_id
        if not patient_root.is_dir():
            raise FileNotFoundError(f"Updated patient directory missing: {patient_root}")
        if not selected_dir.is_dir():
            raise FileNotFoundError(f"Explicit updated series missing: {selected_dir}")
        all_series = sorted(path.name for path in patient_root.iterdir() if path.is_dir())
        ignored = [series for series in all_series if series != selected_id]
        return {
            "patient_id": patient_id,
            "split": split,
            "series_id": selected_id,
            "series_dir": selected_dir,
            "patient_root": patient_root,
            "source_type": "updated_10_cases",
            "reason": "explicit_updated_series_mapping",
            "mapping_status": "selected",
            "ignored_series": ignored,
            "candidate_series": all_series,
            "exclusion_reason": None,
        }

    patient_root = DATA_ROOT / patient_id
    if not patient_root.is_dir():
        return {
            "patient_id": patient_id,
            "split": split,
            "series_id": None,
            "series_dir": None,
            "patient_root": patient_root,
            "source_type": "tiantanDSA",
            "reason": None,
            "mapping_status": "excluded",
            "ignored_series": [],
            "candidate_series": [],
            "exclusion_reason": "missing_patient_directory",
        }

    direct_pre = (patient_root / "Pre-API").is_dir()
    direct_post = (patient_root / "Post-API").is_dir()
    complete_subseries, any_api_subseries = discover_subseries(patient_root)

    if direct_pre or direct_post:
        legacy_recorded = bool(
            legacy_row
            and (
                bool_value(legacy_row.get("pre_api_dir_exists", False))
                or bool_value(legacy_row.get("post_api_dir_exists", False))
            )
        )
        reason = "legacy_flow_manifest_main" if legacy_recorded else "direct_api_at_patient_root"
        return {
            "patient_id": patient_id,
            "split": split,
            "series_id": "main",
            "series_dir": patient_root,
            "patient_root": patient_root,
            "source_type": "tiantanDSA",
            "reason": reason,
            "mapping_status": "selected",
            "ignored_series": [path.name for path in any_api_subseries],
            "candidate_series": ["main"] + [path.name for path in any_api_subseries],
            "exclusion_reason": None,
        }

    if len(complete_subseries) == 1:
        selected = complete_subseries[0]
        return {
            "patient_id": patient_id,
            "split": split,
            "series_id": selected.name,
            "series_dir": selected,
            "patient_root": patient_root,
            "source_type": "tiantanDSA",
            "reason": "unique_subseries_with_pre_and_post_api",
            "mapping_status": "selected",
            "ignored_series": [path.name for path in any_api_subseries if path != selected],
            "candidate_series": [path.name for path in any_api_subseries],
            "exclusion_reason": None,
        }

    if len(complete_subseries) > 1:
        return {
            "patient_id": patient_id,
            "split": split,
            "series_id": None,
            "series_dir": None,
            "patient_root": patient_root,
            "source_type": "tiantanDSA",
            "reason": None,
            "mapping_status": "manual_review",
            "ignored_series": [],
            "candidate_series": [path.name for path in complete_subseries],
            "exclusion_reason": "multiple_complete_api_series_without_legacy_mapping",
        }

    partial = [path.name for path in any_api_subseries]
    return {
        "patient_id": patient_id,
        "split": split,
        "series_id": None,
        "series_dir": None,
        "patient_root": patient_root,
        "source_type": "tiantanDSA",
        "reason": None,
        "mapping_status": "excluded",
        "ignored_series": [],
        "candidate_series": partial,
        "exclusion_reason": "no_direct_or_unique_complete_api_series",
    }


def scan_phase(
    patient_id: str,
    split: str,
    series_id: str,
    series_dir: Path,
    phase: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    phase_title = "Pre" if phase == "pre" else "Post"
    api_dir = series_dir / f"{phase_title}-API"
    audit_rows: list[dict[str, Any]] = []
    selected: list[tuple[int, int, Path, int, int, bool]] = []

    files = sorted((path for path in api_dir.iterdir() if path.is_file()), key=lambda p: p.name.lower()) if api_dir.is_dir() else []
    for path in files:
        suffix = path.suffix.lower()
        match = FRAME_RE.fullmatch(path.name)
        contains_parameter_key = any(key in path.name.lower() for key in PARAMETER_MAP_KEYS)
        is_selected = bool(match and suffix in {".jpg", ".jpeg"} and not contains_parameter_key)
        if is_selected:
            selection_reason = "strict_img_jpg_jpeg_match"
        elif suffix == ".png":
            selection_reason = "excluded_png"
        elif contains_parameter_key:
            selection_reason = "excluded_parameter_map"
        elif suffix not in {".jpg", ".jpeg"}:
            selection_reason = "excluded_suffix"
        else:
            selection_reason = "excluded_filename_pattern"

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if suffix in IMAGE_SUFFIXES else None
        readable = image is not None
        height = int(image.shape[0]) if readable else None
        width = int(image.shape[1]) if readable else None
        series_number = int(match.group(1)) if match else None
        parsed_frame_index = int(match.group(2)) if match else None
        if is_selected:
            selected.append((series_number, parsed_frame_index, path.resolve(), height, width, readable))
        audit_rows.append(
            {
                "patient_id": patient_id,
                "split": split,
                "series_id": series_id,
                "phase": phase,
                "sequence_index": None,
                "frame_index": parsed_frame_index,
                "series_number": series_number,
                "filename": path.name,
                "absolute_path": str(path.resolve()),
                "suffix": suffix,
                "selected": is_selected,
                "selection_reason": selection_reason,
                "height": height,
                "width": width,
                "readable": readable,
            }
        )

    selected.sort(key=lambda item: (item[0], item[1], item[2].name.lower()))
    order_by_path = {str(item[2]): index for index, item in enumerate(selected)}
    for row in audit_rows:
        if row["selected"]:
            row["sequence_index"] = order_by_path[row["absolute_path"]]

    selected_paths = [item[2] for item in selected]
    shapes = [(item[3], item[4]) for item in selected if item[5]]
    unreadable_count = sum(not item[5] for item in selected)
    internal_series = [item[0] for item in selected]
    index_keys = [(item[0], item[1]) for item in selected]
    duplicate_frame_index = len(index_keys) != len(set(index_keys))
    multiple_internal_series = len(set(internal_series)) > 1

    n_contiguous_pairs = 0
    gap_deltas: list[int] = []
    missing_frame_count = 0
    non_monotonic = False
    for left, right in zip(selected, selected[1:]):
        if left[0] != right[0]:
            continue
        delta = right[1] - left[1]
        if delta <= 0:
            non_monotonic = True
        elif delta == 1:
            n_contiguous_pairs += 1
        else:
            gap_deltas.append(delta)
            missing_frame_count += delta - 1

    n_frames = len(selected)
    denominator = n_frames + missing_frame_count
    missing_ratio = (missing_frame_count / denominator) if denominator else 0.0
    can_run = bool(
        n_frames >= 2
        and n_contiguous_pairs >= 1
        and unreadable_count == 0
        and not multiple_internal_series
        and not duplicate_frame_index
        and not non_monotonic
    )

    segmentation = find_segmentation(series_dir, phase_title)
    stats = {
        "api_dir": str(api_dir.resolve()),
        "api_dir_exists": api_dir.is_dir(),
        "n_frames": n_frames,
        "n_pairs": max(n_frames - 1, 0),
        "n_contiguous_pairs": n_contiguous_pairs,
        "first_frame": str(selected_paths[0]) if selected_paths else None,
        "last_frame": str(selected_paths[-1]) if selected_paths else None,
        "frame_list_hash": sha256_lines(selected_paths),
        "dimensions": dimension_string(shapes),
        "mixed_dimensions": len(set(shapes)) > 1,
        "n_frame_gaps": len(gap_deltas),
        "max_frame_gap": max(gap_deltas) if gap_deltas else 0,
        "missing_frame_count": missing_frame_count,
        "missing_frame_ratio": missing_ratio,
        "multiple_internal_series": multiple_internal_series,
        "duplicate_frame_index": duplicate_frame_index,
        "non_monotonic_frame_index": non_monotonic,
        "selected_unreadable_count": unreadable_count,
        "can_run": can_run,
        "height": shapes[0][0] if shapes and len(set(shapes)) == 1 else None,
        "width": shapes[0][1] if shapes and len(set(shapes)) == 1 else None,
        "fps": None,
        "frame_interval": None,
        "pixel_spacing": None,
        "projection_primary_angle": None,
        "projection_secondary_angle": None,
        "segmentation_path": str(segmentation.resolve()) if segmentation else None,
        "segmentation_exists": bool(segmentation and segmentation.is_file()),
        "segmentation_size_bytes": segmentation.stat().st_size if segmentation else None,
        "segmentation_shape": None,
        "segmentation_affine_summary": None,
        "segmentation_reader": "unavailable_nibabel_not_installed" if segmentation else None,
        "cbf_path": find_parameter_map(api_dir, "cbf"),
        "cbv_path": find_parameter_map(api_dir, "cbv"),
        "mtt_path": find_parameter_map(api_dir, "mtt"),
        "ttp_path": find_parameter_map(api_dir, "ttp"),
    }
    return stats, audit_rows


def failure_report(reason: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if FAILURE_PATH.exists():
        print(f"Failure report already exists; refusing to overwrite: {FAILURE_PATH}", file=sys.stderr)
        return
    FAILURE_PATH.write_text(
        "# API full-sequence overnight failure\n\n"
        f"- stage: A manifest build\n"
        f"- reason: {reason}\n"
        "- GPU used: false\n"
        "- SEA-RAFT model loaded: false\n"
        "- protected inputs modified: false\n",
        encoding="utf-8",
    )


def main() -> int:
    conflicts = [str(path) for path in OUTPUTS.values() if path.exists()]
    if conflicts:
        raise FileExistsError("Output conflict: " + "; ".join(conflicts))
    if FAILURE_PATH.exists():
        raise FileExistsError(f"Existing failure report conflict: {FAILURE_PATH}")
    for required in (DATA_ROOT, UPDATED_ROOT, TRAIN_XLSX, VALID_XLSX, LEGACY_MANIFEST):
        if not required.exists():
            raise FileNotFoundError(f"Required input missing: {required}")

    train = load_split(TRAIN_XLSX, "train")
    valid = load_split(VALID_XLSX, "valid")
    train_ids = set(train["patient_id"])
    valid_ids = set(valid["patient_id"])
    overlap = train_ids & valid_ids
    split_frame = pd.concat([train, valid], ignore_index=True)
    if split_frame["patient_id"].duplicated().any():
        raise AssertionError("A patient_id belongs to more than one split")

    legacy = pd.read_csv(LEGACY_MANIFEST, dtype={"patient_id": str})
    legacy["patient_id"] = legacy["patient_id"].map(normalize_patient_id)
    if legacy["patient_id"].duplicated().any():
        raise AssertionError("Legacy manifest has duplicate patient_id rows")
    legacy_by_id = legacy.set_index("patient_id").to_dict(orient="index")

    manifest_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    manual_rows: list[dict[str, Any]] = []
    updated_rows: list[dict[str, Any]] = []

    for split_row in split_frame.sort_values(["split", "patient_id"]).itertuples(index=False):
        selected = select_series(split_row.patient_id, split_row.split, legacy_by_id.get(split_row.patient_id))
        base = {
            "patient_id": split_row.patient_id,
            "split": split_row.split,
            "series_id": selected["series_id"],
            "series_relpath": (
                safe_relpath(selected["series_dir"], selected["patient_root"])
                if selected["series_dir"] is not None
                else None
            ),
            "source_type": selected["source_type"],
            "source_patient_root": str(selected["patient_root"].resolve()),
            "selected_series_reason": selected["reason"],
            "is_updated_case": split_row.patient_id in UPDATED_SERIES,
            "ignored_series": "|".join(selected["ignored_series"]),
            "mapping_status": selected["mapping_status"],
            "exclusion_reason": selected["exclusion_reason"],
        }

        if selected["mapping_status"] == "manual_review":
            manual_rows.append(
                {
                    "patient_id": split_row.patient_id,
                    "split": split_row.split,
                    "source_patient_root": str(selected["patient_root"].resolve()),
                    "candidate_series": "|".join(selected["candidate_series"]),
                    "candidate_count": len(selected["candidate_series"]),
                    "mapping_status": "manual_review",
                    "reason": selected["exclusion_reason"],
                }
            )

        if selected["mapping_status"] == "selected":
            phase_stats: dict[str, dict[str, Any]] = {}
            for phase in ("pre", "post"):
                stats, rows = scan_phase(
                    split_row.patient_id,
                    split_row.split,
                    selected["series_id"],
                    selected["series_dir"],
                    phase,
                )
                phase_stats[phase] = stats
                frame_rows.extend(rows)
                phase_rows.append(
                    {
                        "patient_id": split_row.patient_id,
                        "split": split_row.split,
                        "series_id": selected["series_id"],
                        "phase": phase,
                        "source_type": selected["source_type"],
                        "api_dir": stats["api_dir"],
                        **{key: value for key, value in stats.items() if key != "api_dir"},
                    }
                )

            row = dict(base)
            for phase in ("pre", "post"):
                stats = phase_stats[phase]
                prefix = f"{phase}_"
                row.update(
                    {
                        f"{prefix}api_dir": stats["api_dir"],
                        f"n_{phase}_frames": stats["n_frames"],
                        f"n_{phase}_pairs": stats["n_pairs"],
                        f"n_{phase}_contiguous_pairs": stats["n_contiguous_pairs"],
                        f"{prefix}first_frame": stats["first_frame"],
                        f"{prefix}last_frame": stats["last_frame"],
                        f"{prefix}frame_list_hash": stats["frame_list_hash"],
                        f"{prefix}dimensions": stats["dimensions"],
                        f"{prefix}mixed_dimensions": stats["mixed_dimensions"],
                        f"{prefix}n_frame_gaps": stats["n_frame_gaps"],
                        f"{prefix}max_frame_gap": stats["max_frame_gap"],
                        f"{prefix}missing_frame_count": stats["missing_frame_count"],
                        f"{prefix}missing_frame_ratio": stats["missing_frame_ratio"],
                        f"{prefix}multiple_internal_series": stats["multiple_internal_series"],
                        f"{prefix}duplicate_frame_index": stats["duplicate_frame_index"],
                        f"{prefix}non_monotonic_frame_index": stats["non_monotonic_frame_index"],
                        f"can_run_{phase}": stats["can_run"],
                        f"{prefix}fps": stats["fps"],
                        f"{prefix}frame_interval": stats["frame_interval"],
                        f"{prefix}pixel_spacing": stats["pixel_spacing"],
                        f"{prefix}projection_primary_angle": stats["projection_primary_angle"],
                        f"{prefix}projection_secondary_angle": stats["projection_secondary_angle"],
                        f"{prefix}segmentation_path": stats["segmentation_path"],
                        f"{prefix}segmentation_exists": stats["segmentation_exists"],
                        f"{prefix}cbf_path": stats["cbf_path"],
                        f"{prefix}cbv_path": stats["cbv_path"],
                        f"{prefix}mtt_path": stats["mtt_path"],
                        f"{prefix}ttp_path": stats["ttp_path"],
                    }
                )
            if not row["can_run_pre"] and not row["can_run_post"]:
                row["exclusion_reason"] = "no_runnable_phase_after_strict_frame_qc"
            manifest_rows.append(row)
        else:
            empty = dict(base)
            for phase in ("pre", "post"):
                prefix = f"{phase}_"
                empty.update(
                    {
                        f"{prefix}api_dir": None,
                        f"n_{phase}_frames": 0,
                        f"n_{phase}_pairs": 0,
                        f"n_{phase}_contiguous_pairs": 0,
                        f"{prefix}first_frame": None,
                        f"{prefix}last_frame": None,
                        f"{prefix}frame_list_hash": hashlib.sha256(b"").hexdigest(),
                        f"{prefix}dimensions": None,
                        f"{prefix}mixed_dimensions": False,
                        f"{prefix}n_frame_gaps": 0,
                        f"{prefix}max_frame_gap": 0,
                        f"{prefix}missing_frame_count": 0,
                        f"{prefix}missing_frame_ratio": 0.0,
                        f"{prefix}multiple_internal_series": False,
                        f"{prefix}duplicate_frame_index": False,
                        f"{prefix}non_monotonic_frame_index": False,
                        f"can_run_{phase}": False,
                        f"{prefix}fps": None,
                        f"{prefix}frame_interval": None,
                        f"{prefix}pixel_spacing": None,
                        f"{prefix}projection_primary_angle": None,
                        f"{prefix}projection_secondary_angle": None,
                        f"{prefix}segmentation_path": None,
                        f"{prefix}segmentation_exists": False,
                        f"{prefix}cbf_path": None,
                        f"{prefix}cbv_path": None,
                        f"{prefix}mtt_path": None,
                        f"{prefix}ttp_path": None,
                    }
                )
            manifest_rows.append(empty)

        if split_row.patient_id in UPDATED_SERIES:
            selected_root = UPDATED_ROOT / split_row.patient_id
            all_series = sorted(path.name for path in selected_root.iterdir() if path.is_dir())
            updated_rows.append(
                {
                    "patient_id": split_row.patient_id,
                    "split": split_row.split,
                    "source_patient_root": str(selected_root.resolve()),
                    "selected_series": UPDATED_SERIES[split_row.patient_id],
                    "ignored_series": "|".join(series for series in all_series if series != UPDATED_SERIES[split_row.patient_id]),
                    "all_scanned_series": "|".join(all_series),
                    "mapping_status": selected["mapping_status"],
                }
            )

    manifest = pd.DataFrame(manifest_rows)
    phase_audit = pd.DataFrame(phase_rows)
    frame_audit = pd.DataFrame(frame_rows)
    manual = pd.DataFrame(manual_rows, columns=[
        "patient_id", "split", "source_patient_root", "candidate_series", "candidate_count", "mapping_status", "reason"
    ])
    updated = pd.DataFrame(updated_rows)
    runnable = manifest[
        (manifest["mapping_status"] == "selected") & (manifest["can_run_pre"] | manifest["can_run_post"])
    ].copy()
    train_manifest = runnable[runnable["split"] == "train"].copy()
    valid_manifest = runnable[runnable["split"] == "valid"].copy()
    excluded = manifest[~((manifest["mapping_status"] == "selected") & (manifest["can_run_pre"] | manifest["can_run_post"]))].copy()

    selected_frames = frame_audit[frame_audit["selected"] == True].copy()  # noqa: E712
    assertion_rows: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        assertion_rows.append({"assertion": name, "passed": bool(condition), "detail": detail})
        if not condition:
            raise AssertionError(f"{name}: {detail}")

    check("train_valid_intersection_zero", len(overlap) == 0, f"intersection={len(overlap)}")
    check("patient_one_split", manifest["patient_id"].nunique() == len(manifest), f"rows={len(manifest)}")
    check(
        "patient_series_unique",
        not manifest[["patient_id", "series_id"]].astype(str).duplicated().any(),
        "patient_id+series_id must be unique",
    )
    check("updated_10_all_audited", set(UPDATED_SERIES) == set(updated["patient_id"]), f"audited={len(updated)}")
    check(
        "updated_10_source_is_staging",
        manifest[manifest["is_updated_case"]]["source_patient_root"].str.startswith(str(UPDATED_ROOT)).all(),
        "all updated roots must be staging/updated_10_cases",
    )
    check(
        "updated_10_old_source_count_zero",
        int(manifest[manifest["is_updated_case"]]["source_patient_root"].str.startswith(str(DATA_ROOT)).sum()) == 0,
        "old-source updated count must be zero",
    )
    selected_updated = manifest[manifest["is_updated_case"]].set_index("patient_id")["series_id"].to_dict()
    check("updated_10_explicit_series", selected_updated == UPDATED_SERIES, str(selected_updated))
    ignored_read_count = 0
    for row in updated.itertuples(index=False):
        for ignored_series in str(row.ignored_series).split("|") if row.ignored_series else []:
            needle = str((UPDATED_ROOT / row.patient_id / ignored_series).resolve()) + "/"
            ignored_read_count += int(selected_frames["absolute_path"].str.startswith(needle).sum())
    check("ignored_series_selected_count_zero", ignored_read_count == 0, f"count={ignored_read_count}")
    check("selected_png_count_zero", int(selected_frames["suffix"].str.lower().eq(".png").sum()) == 0, "selected PNG must be zero")
    parameter_selected = selected_frames["filename"].str.lower().str.contains("cbf|cbv|mtt|ttp", regex=True).sum()
    check("selected_parameter_map_count_zero", int(parameter_selected) == 0, f"count={parameter_selected}")
    strict_name_ok = selected_frames["filename"].map(lambda name: FRAME_RE.fullmatch(str(name)) is not None).all()
    check("selected_filename_pattern", bool(strict_name_ok), "all selected files must match strict IMG numeric JPG/JPEG regex")
    pairs_ok = ((manifest["n_pre_pairs"] == (manifest["n_pre_frames"] - 1).clip(lower=0)) & (manifest["n_post_pairs"] == (manifest["n_post_frames"] - 1).clip(lower=0))).all()
    check("pair_count_formula", bool(pairs_ok), "n_pairs=max(n_frames-1,0)")
    runnable_frame_ok = (
        (~manifest["can_run_pre"] | (manifest["n_pre_frames"] >= 2))
        & (~manifest["can_run_post"] | (manifest["n_post_frames"] >= 2))
    ).all()
    check("runnable_has_two_frames", bool(runnable_frame_ok), "each runnable phase has at least two frames")
    check("selected_paths_exist", selected_frames["absolute_path"].map(lambda value: Path(value).is_file()).all(), "all selected paths exist")
    check("selected_frames_readable", selected_frames["readable"].astype(bool).all(), "all selected frames are OpenCV-readable")
    check("manual_review_reported", len(manual) == int((manifest["mapping_status"] == "manual_review").sum()), f"manual={len(manual)}")
    mixed_manifest = int(manifest["pre_mixed_dimensions"].sum() + manifest["post_mixed_dimensions"].sum())
    mixed_phase = int(phase_audit["mixed_dimensions"].sum()) if len(phase_audit) else 0
    check("mixed_dimensions_recorded", mixed_manifest == mixed_phase, f"manifest={mixed_manifest}, phase_audit={mixed_phase}")
    check("no_automatic_deletion_or_repair", True, "script is read-only for all source data")

    split_audit = pd.DataFrame(assertion_rows + [
        {"assertion": "train_unique_patients", "passed": True, "detail": str(len(train_ids))},
        {"assertion": "valid_unique_patients", "passed": True, "detail": str(len(valid_ids))},
        {"assertion": "total_unique_patients", "passed": True, "detail": str(len(train_ids | valid_ids))},
    ])

    counts = {
        "train_unique_patients": len(train_ids),
        "valid_unique_patients": len(valid_ids),
        "total_unique_patients": len(train_ids | valid_ids),
        "runnable_patients": len(runnable),
        "excluded_patients": len(excluded),
        "manual_review_patients": len(manual),
        "selected_frames": len(selected_frames),
        "selected_png": int(selected_frames["suffix"].str.lower().eq(".png").sum()),
        "selected_parameter_maps": int(parameter_selected),
        "mixed_dimension_phases": mixed_phase,
        "frame_gap_phases": int((phase_audit["n_frame_gaps"] > 0).sum()) if len(phase_audit) else 0,
        "multiple_internal_series_phases": int(phase_audit["multiple_internal_series"].sum()) if len(phase_audit) else 0,
    }
    reason_counts = Counter(excluded["exclusion_reason"].fillna("unspecified"))

    audit_md = [
        "# API full-sequence v1 manifest audit",
        "",
        f"- authoritative Train workbook: `{TRAIN_XLSX}`",
        f"- authoritative Valid workbook: `{VALID_XLSX}`",
        f"- Train unique patients: {counts['train_unique_patients']}",
        f"- Valid unique patients: {counts['valid_unique_patients']}",
        f"- Total unique patients: {counts['total_unique_patients']}",
        f"- Runnable patients: {counts['runnable_patients']}",
        f"- Excluded/manual-review patients: {counts['excluded_patients']}",
        f"- Manual series review patients: {counts['manual_review_patients']}",
        f"- Selected frames: {counts['selected_frames']}",
        f"- Selected PNG files: {counts['selected_png']}",
        f"- Selected CBF/CBV/MTT/TTP files: {counts['selected_parameter_maps']}",
        f"- Mixed-dimension phases: {counts['mixed_dimension_phases']}",
        f"- Phases with frame gaps: {counts['frame_gap_phases']}",
        f"- Phases with multiple internal IMG series numbers: {counts['multiple_internal_series_phases']}",
        "- NIfTI shape/affine audit: dependency unavailable (`nibabel` not installed); paths and file sizes only.",
        "- Source data were read only; no patient directory was repaired, removed, renamed, or modified.",
        "",
        "## Mapping policy",
        "",
        "Updated cases use the explicit fixed series mapping and only `staging/updated_10_cases`. Ordinary cases reuse the legacy root-level API mapping when present; otherwise exactly one child series with both Pre-API and Post-API is selected. Multiple complete child series are reported for manual review and excluded from runnable manifests.",
        "",
        "## Exclusion counts",
        "",
    ]
    audit_md.extend(f"- {reason}: {count}" for reason, count in sorted(reason_counts.items()))
    audit_md.extend(["", "## Hard assertions", ""])
    audit_md.extend(f"- {row['assertion']}: {'PASS' if row['passed'] else 'FAIL'} ({row['detail']})" for row in assertion_rows)
    audit_md.extend(
        [
            "",
            "## Updated-case caveat",
            "",
            "Patient 549117 is retained only for the patient-level adverse-outcome path. The fixed C6-1 series has not been claimed to be lesion-level matched to the C4/C6 labels in Excel; future RROC lesion tasks require a separate manual mapping.",
        ]
    )

    config = {
        "version": "api_fullseq_v1",
        "cpu_only": True,
        "train_workbook": str(TRAIN_XLSX),
        "valid_workbook": str(VALID_XLSX),
        "ordinary_root": str(DATA_ROOT),
        "updated_root": str(UPDATED_ROOT),
        "legacy_mapping_source": str(LEGACY_MANIFEST),
        "frame_regex": FRAME_RE.pattern,
        "allowed_suffixes": [".jpg", ".jpeg"],
        "sort_order": ["series_number", "frame_index", "filename_casefold"],
        "parameter_maps_recorded_only": ["CBF", "CBV", "MTT", "TTP"],
        "nifti_dependency_available": False,
        "counts": counts,
    }

    for directory in (MANIFEST_DIR, REPORT_DIR, CONFIG_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUTPUTS["series_manifest"], index=False, encoding="utf-8-sig")
    train_manifest.to_csv(OUTPUTS["train_manifest"], index=False, encoding="utf-8-sig")
    valid_manifest.to_csv(OUTPUTS["valid_manifest"], index=False, encoding="utf-8-sig")
    phase_audit.to_csv(OUTPUTS["manifest_audit_csv"], index=False, encoding="utf-8-sig")
    excluded.to_csv(OUTPUTS["excluded"], index=False, encoding="utf-8-sig")
    manual.to_csv(OUTPUTS["manual_review"], index=False, encoding="utf-8-sig")
    updated.to_csv(OUTPUTS["updated_selection"], index=False, encoding="utf-8-sig")
    frame_audit.to_csv(OUTPUTS["frame_audit"], index=False, encoding="utf-8-sig")
    split_audit.to_csv(OUTPUTS["split_audit"], index=False, encoding="utf-8-sig")
    OUTPUTS["manifest_audit_md"].write_text("\n".join(audit_md) + "\n", encoding="utf-8")
    OUTPUTS["config"].write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUTPUTS["updated_config"].write_text(json.dumps(UPDATED_SERIES, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(counts, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # failure evidence is mandatory for unattended execution
        reason = f"{type(exc).__name__}: {exc}"
        failure_report(reason + "\n\n" + traceback.format_exc())
        print(reason, file=sys.stderr)
        raise SystemExit(1)

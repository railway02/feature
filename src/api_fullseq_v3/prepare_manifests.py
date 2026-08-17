#!/usr/bin/env python3
"""Audit and freeze the all-series manifests for api_fullseq_v3.

This script is deliberately label-blind.  It never scans the raw image tree to
select a series; it only validates the already-selected rows and frozen frame
lists in the supplied manifests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "patient_id", "split", "source_type", "source_medical_record_root",
    "series_uid", "series_id", "series_path", "selected_series_id",
    "selected_for_extraction", "candidate_valid", "selected_candidate",
    "selection_status", "selected_pre_internal_series",
    "selected_post_internal_series", "can_run_pre", "can_run_post",
    "can_run_prepost", "n_pre_frames", "n_post_frames",
    "n_pre_contiguous_pairs", "n_post_contiguous_pairs",
    "pre_frame_indices", "post_frame_indices", "pre_frame_paths",
    "post_frame_paths", "pre_frame_list_hash", "post_frame_list_hash",
    "pre_dimensions", "post_dimensions",
}
JPEG_SUFFIXES = {".jpg", ".jpeg"}
PARAMETER_TOKENS = ("CBF", "CBV", "MTT", "TTP")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def parse_pipe_strings(value: Any) -> list[str]:
    if value is None or pd.isna(value) or str(value) == "":
        return []
    return [item for item in str(value).split("|") if item]


def parse_pipe_ints(value: Any) -> list[int]:
    return [int(item) for item in parse_pipe_strings(value)]


def hash_lines(values: Iterable[str]) -> str:
    material = "\n".join(str(v) for v in values).encode("utf-8")
    return hashlib.sha256(material).hexdigest() if material else ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def frame_blocks(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    blocks = [[indices[0]]]
    for value in indices[1:]:
        if value == blocks[-1][-1] + 1:
            blocks[-1].append(value)
        else:
            blocks.append([value])
    return blocks


def expected_pair_count(indices: list[int]) -> int:
    return sum(max(len(block) - 1, 0) for block in frame_blocks(indices))


def validate_phase(row: pd.Series, phase: str, verify_files: bool) -> dict[str, Any]:
    can_run = as_bool(row[f"can_run_{phase}"])
    paths = [Path(item) for item in parse_pipe_strings(row[f"{phase}_frame_paths"])]
    indices = parse_pipe_ints(row[f"{phase}_frame_indices"])
    frame_hash = "" if pd.isna(row[f"{phase}_frame_list_hash"]) else str(row[f"{phase}_frame_list_hash"])
    expected_frames = int(pd.to_numeric(row[f"n_{phase}_frames"], errors="coerce") or 0)
    expected_pairs = int(pd.to_numeric(row[f"n_{phase}_contiguous_pairs"], errors="coerce") or 0)

    if not can_run:
        if paths or indices or expected_frames or expected_pairs:
            raise AssertionError(
                f"{row.series_uid} {phase}: non-runnable phase contains frozen frames/pairs"
            )
        return {
            "can_run": False,
            "frames": 0,
            "pairs": 0,
            "blocks": 0,
            "gaps": 0,
        }

    if len(paths) != len(indices):
        raise AssertionError(f"{row.series_uid} {phase}: path/index length mismatch")
    if len(paths) != expected_frames:
        raise AssertionError(
            f"{row.series_uid} {phase}: expected frames={expected_frames}, actual={len(paths)}"
        )
    if len(paths) < 2:
        raise AssertionError(f"{row.series_uid} {phase}: fewer than two frames")
    if len(indices) != len(set(indices)) or indices != sorted(indices):
        raise AssertionError(f"{row.series_uid} {phase}: frame indices are not unique/sorted")
    reconstructed_pairs = expected_pair_count(indices)
    if reconstructed_pairs != expected_pairs:
        raise AssertionError(
            f"{row.series_uid} {phase}: expected pairs={expected_pairs}, "
            f"reconstructed={reconstructed_pairs}"
        )
    actual_hash = hash_lines(str(path) for path in paths)
    if actual_hash != frame_hash:
        raise AssertionError(f"{row.series_uid} {phase}: frame-list hash mismatch")

    for path in paths:
        if path.suffix.casefold() not in JPEG_SUFFIXES:
            raise AssertionError(f"{row.series_uid} {phase}: selected non-JPEG {path}")
        upper = path.name.upper()
        if any(token in upper for token in PARAMETER_TOKENS):
            raise AssertionError(f"{row.series_uid} {phase}: selected parameter map {path}")
        if verify_files and not path.is_file():
            raise FileNotFoundError(path)

    blocks = frame_blocks(indices)
    return {
        "can_run": True,
        "frames": len(indices),
        "pairs": reconstructed_pairs,
        "blocks": len(blocks),
        "gaps": max(len(blocks) - 1, 0),
        "first_frame": indices[0],
        "last_frame": indices[-1],
    }


def audit_manifest(path: Path, expected_split: str, verify_files: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(path, dtype={"patient_id": str, "series_uid": str})
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if frame.empty:
        raise AssertionError(f"{path}: empty manifest")
    if frame["series_uid"].duplicated().any():
        duplicate = frame.loc[frame["series_uid"].duplicated(), "series_uid"].head().tolist()
        raise AssertionError(f"{path}: duplicate series_uid {duplicate}")
    if not (frame["split"].astype(str).str.casefold() == expected_split.casefold()).all():
        raise AssertionError(f"{path}: split contamination")
    for column in ("selected_for_extraction", "candidate_valid", "selected_candidate"):
        if not frame[column].map(as_bool).all():
            raise AssertionError(f"{path}: {column} contains false rows")

    phase_rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        patient_id = str(row["patient_id"])
        series_uid = str(row["series_uid"])
        if not patient_id or patient_id.casefold() == "nan":
            raise AssertionError(f"{series_uid}: invalid patient_id")
        for phase in ("pre", "post"):
            result = validate_phase(row, phase, verify_files)
            phase_rows.append({
                "patient_id": patient_id,
                "series_uid": series_uid,
                "split": expected_split,
                "source_type": str(row["source_type"]),
                "phase": phase,
                **result,
            })

    phase_table = pd.DataFrame(phase_rows)
    runnable = phase_table[phase_table["can_run"]].copy()
    patient_sizes = frame.groupby("patient_id").size()
    summary = {
        "manifest": str(path),
        "manifest_sha256": sha256_file(path),
        "split": expected_split,
        "series_rows": int(len(frame)),
        "unique_patients": int(frame["patient_id"].nunique()),
        "unique_series_uid": int(frame["series_uid"].nunique()),
        "runnable_pre": int(frame["can_run_pre"].map(as_bool).sum()),
        "runnable_post": int(frame["can_run_post"].map(as_bool).sum()),
        "runnable_prepost": int(frame["can_run_prepost"].map(as_bool).sum()),
        "post_only": int((~frame["can_run_pre"].map(as_bool) & frame["can_run_post"].map(as_bool)).sum()),
        "runnable_phases": int(len(runnable)),
        "contiguous_pairs": int(runnable["pairs"].sum()),
        "frame_blocks": int(runnable["blocks"].sum()),
        "gap_phases": int((runnable["gaps"] > 0).sum()),
        "source_type_counts": {str(k): int(v) for k, v in frame["source_type"].value_counts().items()},
        "patients_with_multiple_series": int((patient_sizes > 1).sum()),
        "max_series_per_patient": int(patient_sizes.max()),
        "v2_fully_reusable": (
            int(frame["v2_pairdata_fully_reusable"].map(as_bool).sum())
            if "v2_pairdata_fully_reusable" in frame.columns else None
        ),
        "needs_incremental_searaft": (
            int(frame["needs_incremental_searaft"].map(as_bool).sum())
            if "needs_incremental_searaft" in frame.columns else None
        ),
        "verified_source_files": bool(verify_files),
    }
    return frame, summary


def deterministic_pilot(train: pd.DataFrame, count: int) -> pd.DataFrame:
    table = train.copy()
    table["_prepost"] = table["can_run_prepost"].map(as_bool)
    table["_pairs"] = (
        pd.to_numeric(table["n_pre_contiguous_pairs"], errors="coerce").fillna(0)
        + pd.to_numeric(table["n_post_contiguous_pairs"], errors="coerce").fillna(0)
    )
    rank = table["_pairs"].rank(method="first")
    bins = min(5, max(1, len(table)))
    try:
        table["_length_bin"] = pd.qcut(rank, q=bins, labels=False, duplicates="drop")
    except ValueError:
        table["_length_bin"] = 0
    table["_has_gap"] = (
        table["pre_frame_gaps"].notna() | table["post_frame_gaps"].notna()
        if {"pre_frame_gaps", "post_frame_gaps"}.issubset(table.columns)
        else False
    )
    table["_stratum"] = (
        table["source_type"].astype(str)
        + "|" + table["_prepost"].astype(str)
        + "|" + table["_length_bin"].astype(str)
        + "|" + table["_has_gap"].astype(str)
    )
    table["_hash"] = table["series_uid"].astype(str).map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )

    selected_indices: list[int] = []
    groups = [group.sort_values("_hash") for _, group in table.groupby("_stratum", sort=True)]
    position = 0
    while len(selected_indices) < min(count, len(table)):
        added = False
        for group in groups:
            if position < len(group):
                index = int(group.index[position])
                if index not in selected_indices:
                    selected_indices.append(index)
                    added = True
                    if len(selected_indices) >= count:
                        break
        if not added:
            break
        position += 1

    # Force at least one known gap sequence when present.
    gap_indices = table.index[table["_has_gap"]].tolist()
    if gap_indices and not any(index in gap_indices for index in selected_indices):
        selected_indices[-1] = int(gap_indices[0])

    pilot = table.loc[selected_indices].copy()
    pilot = pilot.sort_values(["patient_id", "series_uid"]).drop(
        columns=[column for column in pilot.columns if column.startswith("_")]
    )
    return pilot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--valid-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pilot-count", type=int, default=40)
    parser.add_argument("--verify-files", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    train_path = Path(args.train_manifest).resolve()
    valid_path = Path(args.valid_manifest).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    targets = {
        "train": output / "api_fullseq_v3_train_all_series_frozen.csv",
        "valid": output / "api_fullseq_v3_valid_all_series_frozen.csv",
        "pilot": output / "api_fullseq_v3_pilot_train_all_series.csv",
        "audit": output / "api_fullseq_v3_manifest_audit.json",
        "hashes": output / "api_fullseq_v3_manifest_hashes.json",
        "success": output / ".MANIFESTS_FROZEN_SUCCESS",
    }
    existing = [path for path in targets.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Refusing to overwrite: " + " | ".join(map(str, existing)))

    train, train_summary = audit_manifest(train_path, "Train", args.verify_files)
    valid, valid_summary = audit_manifest(valid_path, "Valid", args.verify_files)
    overlap = sorted(set(train["patient_id"]) & set(valid["patient_id"]))
    if overlap:
        raise AssertionError(f"Train/Valid patient overlap: {overlap[:10]}")
    series_overlap = sorted(set(train["series_uid"]) & set(valid["series_uid"]))
    if series_overlap:
        raise AssertionError(f"Train/Valid series overlap: {series_overlap[:10]}")

    expected = {
        "Train": {"series_rows": 1147, "unique_patients": 1055, "runnable_phases": 2087, "contiguous_pairs": 43364},
        "Valid": {"series_rows": 287, "unique_patients": 264, "runnable_phases": 535, "contiguous_pairs": 11040},
    }
    for summary in (train_summary, valid_summary):
        for key, value in expected[summary["split"]].items():
            if int(summary[key]) != int(value):
                raise AssertionError(
                    f"{summary['split']} {key}: expected={value}, actual={summary[key]}"
                )

    pilot = deterministic_pilot(train, args.pilot_count)
    if len(pilot) != args.pilot_count:
        raise AssertionError(f"Pilot expected={args.pilot_count}, actual={len(pilot)}")

    shutil.copy2(train_path, targets["train"])
    shutil.copy2(valid_path, targets["valid"])
    pilot.to_csv(targets["pilot"], index=False, encoding="utf-8", lineterminator="\n")

    audit = {
        "version": "api_fullseq_v3_manifest_freeze_v1",
        "created_utc": utc_now(),
        "train": train_summary,
        "valid": valid_summary,
        "train_valid_patient_overlap": 0,
        "train_valid_series_overlap": 0,
        "pilot_series": int(len(pilot)),
        "pilot_patients": int(pilot["patient_id"].nunique()),
        "pilot_prepost": int(pilot["can_run_prepost"].map(as_bool).sum()),
        "pilot_post_only": int((~pilot["can_run_pre"].map(as_bool) & pilot["can_run_post"].map(as_bool)).sum()),
        "pilot_pairs": int(
            pd.to_numeric(pilot["n_pre_contiguous_pairs"], errors="coerce").fillna(0).sum()
            + pd.to_numeric(pilot["n_post_contiguous_pairs"], errors="coerce").fillna(0).sum()
        ),
        "labels_read": False,
        "raw_directory_rescanned": False,
    }
    write_json_atomic(targets["audit"], audit)
    hashes = {
        "train_frozen": {"path": str(targets["train"]), "sha256": sha256_file(targets["train"])},
        "valid_frozen": {"path": str(targets["valid"]), "sha256": sha256_file(targets["valid"])},
        "pilot_frozen": {"path": str(targets["pilot"]), "sha256": sha256_file(targets["pilot"])},
    }
    write_json_atomic(targets["hashes"], hashes)
    write_json_atomic(targets["success"], {"created_utc": utc_now(), **hashes})
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

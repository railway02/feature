#!/usr/bin/env python3
"""Build extraction-ready manifests containing ALL runnable 2D API series.

This is a thin, non-destructive wrapper around the frozen
14_build_api_fullseq_v2_manifests.py implementation.

It reuses code/14 for:
- patient directory discovery;
- candidate series discovery;
- grayscale JPEG and parameter-map filtering;
- image readability and dimension QC;
- internal-series selection;
- true contiguous-frame-pair counting.

It changes only the final selection policy:
- old v2: select the first valid candidate per patient;
- api_record_v1 preparation: select EVERY valid candidate per patient.

Outputs are series-level, not record-level. Each valid image series is listed
once and contains the exact selected Pre/Post filenames, paths, frame indices,
and hashes. No SEA-RAFT inference or feature extraction is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PROJECT = Path("/root/autodl-tmp/aneurysm")
DEFAULT_CODE14 = DEFAULT_PROJECT / "code/14_build_api_fullseq_v2_manifests.py"
DEFAULT_OUTDIR = DEFAULT_PROJECT / "manifests/api_record_v1"
EXPECTED_TRAIN_RECORDS = 1157
EXPECTED_VALID_RECORDS = 289


def load_code14(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"code/14 not found: {path}")
    spec = importlib.util.spec_from_file_location("api_fullseq_v2_code14", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalize_patient_id(value: Any) -> str:
    if pd.isna(value):
        raise ValueError("Encountered empty 病案号")
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    if not re.fullmatch(r"\d+", text):
        raise ValueError(f"Invalid 病案号: {value!r}")
    return str(int(text))


def safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("_")
    return cleaned or "main"


def series_uid(split: str, patient_id: str, series_id: str) -> str:
    digest = hashlib.sha1(series_id.encode("utf-8")).hexdigest()[:10]
    return f"{split}__{patient_id}__{safe_component(series_id)}__{digest}"


def basenames_from_pipe(value: Any) -> str:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return ""
    return "|".join(Path(part).name for part in str(value).split("|") if part)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temp, index=False, encoding="utf-8-sig", lineterminator="\n")
    os.replace(temp, path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def build_record_table(xlsx: Path, split: str) -> pd.DataFrame:
    """Preserve every Excel row; this table is not used to choose image frames."""
    frame = pd.read_excel(xlsx, dtype={"病案号": str})
    if "病案号" not in frame.columns:
        raise ValueError(f"{xlsx}: missing 病案号 column")
    frame = frame.copy()
    frame.insert(0, "excel_row_number", range(2, len(frame) + 2))
    frame.insert(1, "source_record_index", range(1, len(frame) + 1))
    frame.insert(2, "split", split)
    frame.insert(3, "patient_id", frame["病案号"].map(normalize_patient_id))
    frame.insert(
        4,
        "record_uid",
        [
            f"{split}:{pid}:excel_row_{row:06d}"
            for pid, row in zip(frame["patient_id"], frame["excel_row_number"])
        ],
    )
    frame.insert(
        5,
        "record_index_within_patient",
        frame.groupby("patient_id").cumcount() + 1,
    )
    return frame


def load_old_v2_index(train_path: Path, valid_path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for split, path in (("Train", train_path), ("Valid", valid_path)):
        if not path.is_file():
            continue
        frame = pd.read_csv(path, dtype={"patient_id": str})
        for row in frame.to_dict("records"):
            patient_id = normalize_patient_id(row["patient_id"])
            sid = str(row.get("selected_series_id", ""))
            index[(split, patient_id, sid)] = row
    return index


def hash_equal_for_phase(new_phase: dict[str, Any], old: dict[str, Any] | None, phase: str) -> bool:
    if not bool(new_phase["can_run"]):
        return True  # no pairdata needed for this phase
    if old is None:
        return False
    new_hash = str(new_phase.get("frame_list_hash", ""))
    old_hash = str(old.get(f"{phase}_frame_list_hash", ""))
    new_internal = str(new_phase.get("selected_internal_series", ""))
    old_internal = str(old.get(f"selected_{phase}_internal_series", ""))
    new_pairs = int(new_phase.get("n_contiguous_pairs", 0))
    old_pairs = int(float(old.get(f"n_{phase}_contiguous_pairs", 0) or 0))
    return bool(new_hash) and new_hash == old_hash and new_internal == old_internal and new_pairs == old_pairs


def build_all_series_for_patient(
    code14,
    patient_id: str,
    split: str,
    executor: ThreadPoolExecutor,
    old_v2_index: dict[tuple[str, str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Scan all candidates and select every valid candidate for extraction."""
    is_updated = patient_id in code14.UPDATED_SERIES
    source_type = "updated_10_cases" if is_updated else "tiantanDSA"
    source_root = (code14.UPDATED_ROOT if is_updated else code14.DATA_ROOT) / patient_id
    discovered = code14.discover_candidates(source_root)
    fixed_series = code14.UPDATED_SERIES.get(patient_id)

    # Preserve code/14's fallback synthesis, but DO NOT restrict scanning to fixed target.
    if is_updated and fixed_series and not any(c["series_id"] == fixed_series for c in discovered):
        discovered.append(
            code14.synthesize_fixed_candidate(source_root, str(fixed_series), len(discovered) + 1)
        )

    scanned: list[dict[str, Any]] = []
    for candidate in discovered:
        candidate["is_fixed_target"] = bool(is_updated and candidate["series_id"] == fixed_series)
        # Critical change from v2: scan every sibling candidate, including updated_10 cases.
        scanned_candidate = code14.scan_candidate(
            patient_id=patient_id,
            split=split,
            source_type=source_type,
            source_root=source_root,
            candidate=candidate,
            executor=executor,
            scan_performed=True,
        )
        if scanned_candidate["candidate_valid"]:
            scanned_candidate["selected_candidate"] = True
            scanned_candidate["selection_status"] = "selected_all_valid_candidates"
        else:
            scanned_candidate["selected_candidate"] = False
            scanned_candidate["selection_status"] = "excluded_invalid_candidate"
        scanned.append(scanned_candidate)

    series_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []

    for candidate in scanned:
        pre = candidate["pre"]
        post = candidate["post"]
        sid = str(candidate["series_id"])
        uid = series_uid(split, patient_id, sid)
        old = old_v2_index.get((split, patient_id, sid))
        pre_reuse = hash_equal_for_phase(pre, old, "pre")
        post_reuse = hash_equal_for_phase(post, old, "post")
        reusable = bool(candidate["candidate_valid"] and pre_reuse and post_reuse)

        row = code14.candidate_to_row(patient_id, split, source_type, source_root, candidate)
        row.update({
            "series_uid": uid,
            "selected_for_extraction": bool(candidate["candidate_valid"]),
            "pre_frame_indices": pre["frame_indices"],
            "post_frame_indices": post["frame_indices"],
            "pre_selected_filenames": basenames_from_pipe(pre["frame_paths"]),
            "post_selected_filenames": basenames_from_pipe(post["frame_paths"]),
            "pre_frame_paths": pre["frame_paths"],
            "post_frame_paths": post["frame_paths"],
            "pre_frame_list_hash": pre["frame_list_hash"],
            "post_frame_list_hash": post["frame_list_hash"],
            "pre_dimensions": pre["dimensions"],
            "post_dimensions": post["dimensions"],
            "v2_series_match_found": old is not None,
            "v2_pre_pairdata_reusable": pre_reuse,
            "v2_post_pairdata_reusable": post_reuse,
            "v2_pairdata_fully_reusable": reusable,
            "needs_incremental_searaft": bool(candidate["candidate_valid"] and not reusable),
            "fixed_mapping_series": fixed_series or "",
        })
        series_rows.append(row)

        for phase_name in ("pre", "post"):
            for record in candidate[phase_name]["frame_records"]:
                selected = bool(candidate["candidate_valid"] and record["phase_eligible_frame"])
                record["selected"] = selected
                if selected:
                    record["selection_reason"] = "selected_frame_in_all_valid_candidate"
                elif record["phase_eligible_frame"]:
                    record["selection_reason"] = "eligible_frame_in_invalid_candidate"
                output = {column: record.get(column, "") for column in code14.FRAME_COLUMNS}
                output["series_uid"] = uid
                output["selected_for_extraction"] = selected
                frame_rows.append(output)

    return series_rows, frame_rows


def validate(
    train_records: pd.DataFrame,
    valid_records: pd.DataFrame,
    series_df: pd.DataFrame,
    frame_df: pd.DataFrame,
) -> dict[str, Any]:
    errors: list[str] = []
    if len(train_records) != EXPECTED_TRAIN_RECORDS:
        errors.append(f"Train record count {len(train_records)} != {EXPECTED_TRAIN_RECORDS}")
    if len(valid_records) != EXPECTED_VALID_RECORDS:
        errors.append(f"Valid record count {len(valid_records)} != {EXPECTED_VALID_RECORDS}")
    if not train_records["record_uid"].is_unique:
        errors.append("Train record_uid is not unique")
    if not valid_records["record_uid"].is_unique:
        errors.append("Valid record_uid is not unique")
    intersection = set(train_records["patient_id"]) & set(valid_records["patient_id"])
    if intersection:
        errors.append(f"Train/Valid patient intersection: {len(intersection)}")
    if not series_df["series_uid"].is_unique:
        errors.append("series_uid is not unique")

    selected = series_df[series_df["selected_for_extraction"].astype(bool)]
    if not ((selected["can_run_pre"].astype(bool)) | (selected["can_run_post"].astype(bool))).all():
        errors.append("A selected series has neither runnable Pre nor runnable Post")
    if (selected["selection_status"] != "selected_all_valid_candidates").any():
        errors.append("Selected series contains an unexpected selection_status")
    if (series_df["selection_status"] == "ignored_valid_after_first_valid").any():
        errors.append("Old first-valid suppression still exists")

    selected_frames = frame_df[frame_df["selected_for_extraction"].astype(bool)]
    missing_paths = [p for p in selected_frames["absolute_path"].astype(str) if not Path(p).is_file()]
    if missing_paths:
        errors.append(f"Selected frame paths missing: {len(missing_paths)}")
    bad_names = [
        name for name in selected_frames["filename"].astype(str)
        if code14_global.STRICT_FRAME_RE.fullmatch(name) is None
    ]
    if bad_names:
        errors.append(f"Selected frames with non-strict names: {len(bad_names)}")

    summary = {
        "train_record_rows": len(train_records),
        "valid_record_rows": len(valid_records),
        "unique_train_patients": train_records["patient_id"].nunique(),
        "unique_valid_patients": valid_records["patient_id"].nunique(),
        "all_candidate_series_rows": len(series_df),
        "selected_valid_series_rows": len(selected),
        "selected_train_series_rows": int(((selected["split"] == "Train")).sum()),
        "selected_valid_split_series_rows": int(((selected["split"] == "Valid")).sum()),
        "fully_reusable_v2_series": int(selected["v2_pairdata_fully_reusable"].astype(bool).sum()),
        "incremental_searaft_series": int(selected["needs_incremental_searaft"].astype(bool).sum()),
        "selected_frame_rows": len(selected_frames),
        "errors": errors,
    }
    if errors:
        raise AssertionError(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--code14", type=Path, default=DEFAULT_CODE14)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    global code14_global
    code14_global = load_code14(args.code14)
    code14 = code14_global

    train_xlsx = args.project / "metadata/Train.xlsx"
    valid_xlsx = args.project / "metadata/valid.xlsx"
    old_train = args.project / "manifests/api_fullseq_v2_train_manifest.csv"
    old_valid = args.project / "manifests/api_fullseq_v2_valid_manifest.csv"

    outputs = {
        "train_records": args.outdir / "train_record_table.csv",
        "valid_records": args.outdir / "valid_record_table.csv",
        "train_series": args.outdir / "train_all_series_manifest.csv",
        "valid_series": args.outdir / "valid_all_series_manifest.csv",
        "all_candidates": args.outdir / "all_candidate_series_audit.csv",
        "frames": args.outdir / "all_series_frame_audit.csv",
        "incremental": args.outdir / "incremental_searaft_series_manifest.csv",
        "reusable": args.outdir / "reusable_v2_series_manifest.csv",
        "summary": args.outdir / "manifest_summary.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing outputs. Use --overwrite after review:\n"
            + "\n".join(existing)
        )

    train_records = build_record_table(train_xlsx, "Train")
    valid_records = build_record_table(valid_xlsx, "Valid")
    old_index = load_old_v2_index(old_train, old_valid)

    unique_train = sorted(train_records["patient_id"].unique(), key=code14.patient_key)
    unique_valid = sorted(valid_records["patient_id"].unique(), key=code14.patient_key)
    patients = [(pid, "Train") for pid in unique_train] + [(pid, "Valid") for pid in unique_valid]

    series_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="api_record_v1") as executor:
        for i, (pid, split) in enumerate(patients, start=1):
            patient_series, patient_frames = build_all_series_for_patient(
                code14, pid, split, executor, old_index
            )
            series_rows.extend(patient_series)
            frame_rows.extend(patient_frames)
            if i % 50 == 0 or i == len(patients):
                print(
                    f"[{i}/{len(patients)}] candidate_series={len(series_rows)} "
                    f"frame_audit={len(frame_rows)}",
                    flush=True,
                )

    series_df = pd.DataFrame(series_rows)
    frame_df = pd.DataFrame(frame_rows)
    selected_df = series_df[series_df["selected_for_extraction"].astype(bool)].copy()
    train_series = selected_df[selected_df["split"] == "Train"].copy()
    valid_series = selected_df[selected_df["split"] == "Valid"].copy()
    incremental = selected_df[selected_df["needs_incremental_searaft"].astype(bool)].copy()
    reusable = selected_df[selected_df["v2_pairdata_fully_reusable"].astype(bool)].copy()

    summary = validate(train_records, valid_records, series_df, frame_df)

    atomic_csv(train_records, outputs["train_records"])
    atomic_csv(valid_records, outputs["valid_records"])
    atomic_csv(train_series, outputs["train_series"])
    atomic_csv(valid_series, outputs["valid_series"])
    atomic_csv(series_df, outputs["all_candidates"])
    atomic_csv(frame_df, outputs["frames"])
    atomic_csv(incremental, outputs["incremental"])
    atomic_csv(reusable, outputs["reusable"])
    atomic_json(summary, outputs["summary"])

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Outputs: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

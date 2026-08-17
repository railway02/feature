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
- api_record_v1 preparation: select EVERY valid candidate per patient, including valid siblings under updated_10_cases.

Outputs are series-level, not record-level. Each valid image series is listed
once and contains the exact selected Pre/Post filenames, paths, frame indices,
and hashes. No SEA-RAFT inference or feature extraction is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PROJECT = Path("/root/autodl-tmp/aneurysm")
DEFAULT_CODE14 = DEFAULT_PROJECT / "code/14_build_api_fullseq_v2_manifests.py"
DEFAULT_OUTDIR = DEFAULT_PROJECT / "manifests/api_record_v1_all_series"
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



def normalize_side(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().upper()
    aliases = {
        "左": "L", "左侧": "L", "LEFT": "L", "L": "L",
        "右": "R", "右侧": "R", "RIGHT": "R", "R": "R",
    }
    return aliases.get(text, text)


def normalize_location(value: Any) -> str:
    """Normalize common aneurysm-location labels without using outcome fields."""
    if value is None or pd.isna(value):
        return ""
    text = re.sub(r"[\s_\-]+", "", str(value).strip().upper())
    aliases = {
        "PCOM": "PCOM", "PCOA": "PCOM", "PCA": "PCA",
        "ACOM": "ACOM", "ACOA": "ACOM",
        "MCA": "MCA", "ACA": "ACA",
        "ACH": "ACHA", "ACHA": "ACHA", "ACHA.": "ACHA",
        "BA": "BA", "VA": "VA", "PICA": "PICA", "SCA": "SCA",
        "ICA": "ICA", "OPH": "OPH",
    }
    if text in aliases:
        return aliases[text]
    match = re.search(r"\bC([1-7])\b", text)
    if match:
        return f"C{match.group(1)}"
    for key, value_norm in aliases.items():
        if key in text:
            return value_norm
    return text


def series_location_and_suffix(series_id: str) -> tuple[str, int | None]:
    text = str(series_id).strip()
    normalized = normalize_location(text)
    c_match = re.search(r"(?i)(C[1-7])(?:[-_]?(\d+))?$", text)
    if c_match:
        return c_match.group(1).upper(), int(c_match.group(2)) if c_match.group(2) else None
    suffix_match = re.search(r"(?:[-_])(\d+)$", text)
    suffix = int(suffix_match.group(1)) if suffix_match else None
    return normalized, suffix


def build_record_series_suggestions(
    records: pd.DataFrame,
    selected_series: pd.DataFrame,
) -> pd.DataFrame:
    """Create deterministic suggestions; extraction still uses ALL valid series.

    The suggestions are not treated as ground truth. Exact unique-location matches
    are high confidence. Same-location count/order matches (e.g. two C6 rows and
    C6-1/C6-2) are medium confidence and remain auditable.
    """
    records = records.copy()
    selected = selected_series.copy()
    if selected.empty:
        output = records[[
            "record_uid", "split", "patient_id", "excel_row_number",
            "record_index_within_patient", "normalized_side", "normalized_location",
        ]].copy()
        output["suggested_series_uid"] = ""
        output["suggested_series_id"] = ""
        output["mapping_status"] = "no_valid_series"
        output["mapping_confidence"] = "unavailable"
        output["patient_valid_series_count"] = 0
        output["location_valid_series_count"] = 0
        output["candidate_series_ids"] = ""
        return output

    parsed = selected["selected_series_id"].astype(str).map(series_location_and_suffix)
    selected["normalized_series_location"] = [item[0] for item in parsed]
    selected["series_suffix_index"] = [item[1] for item in parsed]

    rows: list[dict[str, Any]] = []
    for patient_id, patient_records in records.groupby("patient_id", sort=False):
        patient_series = selected[selected["patient_id"].astype(str) == str(patient_id)].copy()
        patient_series = patient_series.sort_values(
            ["normalized_series_location", "series_suffix_index", "selected_series_id"],
            na_position="last",
        )
        assignment: dict[str, tuple[pd.Series, str, str]] = {}
        used_series_uids: set[str] = set()

        # First pass: location groups with equal counts.
        for location, record_group in patient_records.groupby("normalized_location", sort=False):
            if not location:
                continue
            series_group = patient_series[
                patient_series["normalized_series_location"] == location
            ].copy()
            if len(record_group) == len(series_group) and len(record_group) > 0:
                ordered_records = record_group.sort_values("excel_row_number")
                ordered_series = series_group.sort_values(
                    ["series_suffix_index", "selected_series_id"], na_position="last"
                )
                confidence = "high" if len(record_group) == 1 else "medium"
                status = (
                    "auto_unique_location"
                    if len(record_group) == 1
                    else "auto_same_location_ordered"
                )
                for (_, record), (_, series) in zip(
                    ordered_records.iterrows(), ordered_series.iterrows()
                ):
                    assignment[str(record["record_uid"])] = (
                        series, status, confidence
                    )
                    used_series_uids.add(str(series["series_uid"]))

        # Second pass: only one unmatched record and one unmatched series.
        unmatched_records = patient_records[
            ~patient_records["record_uid"].astype(str).isin(assignment)
        ]
        unmatched_series = patient_series[
            ~patient_series["series_uid"].astype(str).isin(used_series_uids)
        ]
        if len(unmatched_records) == 1 and len(unmatched_series) == 1:
            record = unmatched_records.iloc[0]
            series = unmatched_series.iloc[0]
            assignment[str(record["record_uid"])] = (
                series, "auto_single_remaining", "medium"
            )
            used_series_uids.add(str(series["series_uid"]))

        candidate_ids = "|".join(patient_series["selected_series_id"].astype(str).tolist())
        for _, record in patient_records.sort_values("excel_row_number").iterrows():
            record_uid = str(record["record_uid"])
            location_count = int(
                (
                    patient_series["normalized_series_location"]
                    == str(record["normalized_location"])
                ).sum()
            )
            base = {
                "record_uid": record_uid,
                "split": record["split"],
                "patient_id": str(patient_id),
                "excel_row_number": int(record["excel_row_number"]),
                "record_index_within_patient": int(record["record_index_within_patient"]),
                "normalized_side": record["normalized_side"],
                "normalized_location": record["normalized_location"],
                "patient_valid_series_count": int(len(patient_series)),
                "location_valid_series_count": location_count,
                "candidate_series_ids": candidate_ids,
            }
            if record_uid in assignment:
                series, status, confidence = assignment[record_uid]
                base.update({
                    "suggested_series_uid": str(series["series_uid"]),
                    "suggested_series_id": str(series["selected_series_id"]),
                    "mapping_status": status,
                    "mapping_confidence": confidence,
                    "suggested_pre_frame_paths": str(series.get("pre_frame_paths", "")),
                    "suggested_post_frame_paths": str(series.get("post_frame_paths", "")),
                    "suggested_pre_selected_filenames": str(series.get("pre_selected_filenames", "")),
                    "suggested_post_selected_filenames": str(series.get("post_selected_filenames", "")),
                    "suggested_pairdata_reusable": bool(series.get("v2_pairdata_fully_reusable", False)),
                    "suggested_needs_incremental_searaft": bool(series.get("needs_incremental_searaft", False)),
                })
            else:
                status = "no_valid_series" if patient_series.empty else "ambiguous"
                base.update({
                    "suggested_series_uid": "",
                    "suggested_series_id": "",
                    "mapping_status": status,
                    "mapping_confidence": "unavailable" if status == "no_valid_series" else "low",
                    "suggested_pre_frame_paths": "",
                    "suggested_post_frame_paths": "",
                    "suggested_pre_selected_filenames": "",
                    "suggested_post_selected_filenames": "",
                    "suggested_pairdata_reusable": False,
                    "suggested_needs_incremental_searaft": False,
                })
            rows.append(base)
    return pd.DataFrame(rows)


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
    """Preserve every Excel row; outcome columns are never used for frame selection."""
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
    frame["normalized_side"] = (
        frame["侧别"].map(normalize_side) if "侧别" in frame.columns else ""
    )
    frame["normalized_location"] = (
        frame["部位"].map(normalize_location) if "部位" in frame.columns else ""
    )
    frame["record_index_within_location"] = (
        frame.groupby(["patient_id", "normalized_location"]).cumcount() + 1
    )
    frame["record_count_within_patient"] = (
        frame.groupby("patient_id")["patient_id"].transform("size")
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


def normalize_manifest_scalar(value: Any) -> str:
    """Normalize CSV scalar IDs so 89, 89.0 and "89.0" compare equally."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "<na>"}:
        return ""
    try:
        number = float(text)
    except (TypeError, ValueError):
        return text
    if math.isfinite(number) and number.is_integer():
        return str(int(number))
    return text


def normalize_nonnegative_int(value: Any) -> int:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return 0
    number = float(str(value).strip())
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError(f"Expected non-negative integer-like value, got {value!r}")
    return int(number)


def hash_equal_for_phase(
    new_phase: dict[str, Any],
    old: dict[str, Any] | None,
    phase: str,
) -> bool:
    if not bool(new_phase["can_run"]):
        return True  # no pairdata is required for this phase
    if old is None:
        return False

    new_hash = str(new_phase.get("frame_list_hash", "")).strip()
    old_hash = str(old.get(f"{phase}_frame_list_hash", "")).strip()
    new_internal = normalize_manifest_scalar(
        new_phase.get("selected_internal_series", "")
    )
    old_internal = normalize_manifest_scalar(
        old.get(f"selected_{phase}_internal_series", "")
    )
    new_pairs = normalize_nonnegative_int(
        new_phase.get("n_contiguous_pairs", 0)
    )
    old_pairs = normalize_nonnegative_int(
        old.get(f"n_{phase}_contiguous_pairs", 0)
    )

    return (
        bool(new_hash)
        and new_hash == old_hash
        and new_internal == old_internal
        and new_pairs == old_pairs
    )


def build_all_series_for_patient(
    code14,
    project: Path,
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

    # The ten supplemented cases must stay inside staging/updated_10_cases,
    # but api_record_v1 is an all-series inventory. The frozen fixed series
    # identifies the old-v2 selected series; it must not suppress valid siblings.
    if (
        is_updated
        and fixed_series
        and not any(
            str(candidate["series_id"]) == str(fixed_series)
            for candidate in discovered
        )
    ):
        discovered.append(
            code14.synthesize_fixed_candidate(
                source_root,
                str(fixed_series),
                len(discovered) + 1,
            )
        )

    scanned: list[dict[str, Any]] = []
    for candidate in discovered:
        candidate["is_fixed_target"] = bool(is_updated and candidate["series_id"] == fixed_series)
        # Scan every candidate discovered under the selected source root.
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
            "series_id": sid,
            "selected_series_id": sid,
            "series_location_base": series_location_and_suffix(sid)[0],
            "series_suffix_index": series_location_and_suffix(sid)[1],
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
            "v2_pairdata_source_root": (
                str(
                    project
                    / "outputs/api_fullseq_v2_pairdata/full"
                    / split.lower()
                    / patient_id
                )
                if reusable else ""
            ),
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




def build_updated_10_audit(code14, series_df: pd.DataFrame) -> pd.DataFrame:
    """Audit all valid sibling series under the authoritative staging roots."""
    rows: list[dict[str, Any]] = []
    for raw_patient_id in code14.UPDATED_IDS:
        patient_id = str(raw_patient_id)
        fixed_series = str(code14.UPDATED_SERIES[raw_patient_id])
        group = series_df[
            series_df["patient_id"].astype(str) == patient_id
        ].copy()
        valid_group = group[group["candidate_valid"].astype(bool)].copy()
        selected_group = group[
            group["selected_for_extraction"].astype(bool)
        ].copy()
        fixed_group = group[
            group["selected_series_id"].astype(str) == fixed_series
        ].copy()
        expected_root = (code14.UPDATED_ROOT / patient_id).resolve()

        all_paths_under_updated_root = True
        for path_text in group.get("series_path", pd.Series(dtype=str)).astype(str):
            try:
                Path(path_text).resolve().relative_to(expected_root)
            except Exception:
                all_paths_under_updated_root = False
                break

        sibling_valid = valid_group[
            valid_group["selected_series_id"].astype(str) != fixed_series
        ].copy()

        rows.append({
            "patient_id": patient_id,
            "expected_split": "Valid" if patient_id == "549117" else "Train",
            "source_root": str(expected_root),
            "source_root_exists": expected_root.is_dir(),
            "fixed_mapping_series": fixed_series,
            "candidate_series_count": int(len(group)),
            "candidate_series_ids": "|".join(
                group["selected_series_id"].astype(str).tolist()
            ),
            "all_candidates_scanned": bool(
                len(group) > 0 and group["scan_performed"].astype(bool).all()
            ),
            "valid_series_count": int(len(valid_group)),
            "selected_valid_series_count": int(len(selected_group)),
            "all_valid_series_selected": bool(
                len(valid_group) == len(selected_group)
                and set(valid_group["series_uid"].astype(str))
                == set(selected_group["series_uid"].astype(str))
            ),
            "fixed_target_discovered": bool(len(fixed_group) == 1),
            "fixed_target_path_exists": bool(
                len(fixed_group) == 1
                and Path(str(fixed_group.iloc[0]["series_path"])).is_dir()
            ),
            "fixed_target_valid": bool(
                len(fixed_group) == 1
                and bool(fixed_group.iloc[0]["candidate_valid"])
            ),
            "fixed_target_manifest_reusable": bool(
                len(fixed_group) == 1
                and bool(fixed_group.iloc[0]["v2_pairdata_fully_reusable"])
            ),
            "sibling_valid_series_count": int(len(sibling_valid)),
            "sibling_valid_series_ids": "|".join(
                sibling_valid["selected_series_id"].astype(str).tolist()
            ),
            "sibling_incremental_series_count": int(
                sibling_valid["needs_incremental_searaft"].astype(bool).sum()
            ) if len(sibling_valid) else 0,
            "source_type_all_updated_10": bool(
                len(group) > 0
                and (group["source_type"].astype(str) == "updated_10_cases").all()
            ),
            "all_paths_under_updated_root": all_paths_under_updated_root,
            "old_tiantan_root_used": bool(
                group.get("series_path", pd.Series(dtype=str))
                .astype(str)
                .str.startswith(str(code14.DATA_ROOT.resolve()) + os.sep)
                .any()
            ),
        })
    return pd.DataFrame(rows)

def build_record_series_coverage_audit(
    records: pd.DataFrame,
    selected_series: pd.DataFrame,
    suggestions: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize record/series coverage without forcing ambiguous links."""
    rows: list[dict[str, Any]] = []
    for patient_id, record_group in records.groupby("patient_id", sort=False):
        series_group = selected_series[
            selected_series["patient_id"].astype(str) == str(patient_id)
        ]
        suggestion_group = suggestions[
            suggestions["patient_id"].astype(str) == str(patient_id)
        ]
        record_count = int(len(record_group))
        series_count = int(len(series_group))
        mapped_count = int(
            suggestion_group["suggested_series_uid"].astype(str).ne("").sum()
        )
        ambiguous_count = int(
            suggestion_group["mapping_status"].isin(["ambiguous", "no_valid_series"]).sum()
        )
        if series_count == record_count:
            count_relation = "equal"
        elif series_count > record_count:
            count_relation = "more_series_than_records"
        else:
            count_relation = "fewer_series_than_records"
        rows.append({
            "split": str(record_group.iloc[0]["split"]),
            "patient_id": str(patient_id),
            "record_count": record_count,
            "selected_valid_series_count": series_count,
            "auto_suggested_record_count": mapped_count,
            "ambiguous_or_unavailable_record_count": ambiguous_count,
            "record_series_count_relation": count_relation,
            "record_uids": "|".join(record_group["record_uid"].astype(str).tolist()),
            "series_ids": "|".join(series_group["selected_series_id"].astype(str).tolist()),
            "mapping_statuses": "|".join(
                suggestion_group["mapping_status"].astype(str).tolist()
            ),
        })
    return pd.DataFrame(rows)


def validate(
    code14,
    train_records: pd.DataFrame,
    valid_records: pd.DataFrame,
    series_df: pd.DataFrame,
    frame_df: pd.DataFrame,
    updated_10_audit: pd.DataFrame,
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

    train_patients = set(train_records["patient_id"].astype(str))
    valid_patients = set(valid_records["patient_id"].astype(str))
    intersection = train_patients & valid_patients
    if intersection:
        errors.append(f"Train/Valid patient intersection: {len(intersection)}")

    if series_df.empty:
        errors.append("No candidate series were discovered")
    elif not series_df["series_uid"].is_unique:
        errors.append("series_uid is not unique")

    selected = series_df[
        series_df["selected_for_extraction"].astype(bool)
    ].copy()
    if len(selected):
        if not (
            selected["can_run_pre"].astype(bool)
            | selected["can_run_post"].astype(bool)
        ).all():
            errors.append("A selected series has neither runnable Pre nor runnable Post")
        if (
            selected["selection_status"].astype(str)
            != "selected_all_valid_candidates"
        ).any():
            errors.append("Selected series contains an unexpected selection_status")

    if (
        series_df["selection_status"].astype(str)
        == "ignored_valid_after_first_valid"
    ).any():
        errors.append("Old first-valid suppression still exists")

    selected_frames = frame_df[
        frame_df["selected_for_extraction"].astype(bool)
    ].copy()
    missing_paths = [
        path for path in selected_frames["absolute_path"].astype(str)
        if not Path(path).is_file()
    ]
    if missing_paths:
        errors.append(f"Selected frame paths missing: {len(missing_paths)}")

    bad_names = [
        name for name in selected_frames["filename"].astype(str)
        if code14_global.STRICT_FRAME_RE.fullmatch(name) is None
    ]
    if bad_names:
        errors.append(f"Selected frames with non-strict names: {len(bad_names)}")

    expected_updated_ids = set(map(str, code14.UPDATED_IDS))
    actual_updated_ids = set(updated_10_audit["patient_id"].astype(str))
    if actual_updated_ids != expected_updated_ids:
        errors.append(
            "Updated-10 audit IDs mismatch: "
            f"missing={sorted(expected_updated_ids - actual_updated_ids)}, "
            f"extra={sorted(actual_updated_ids - expected_updated_ids)}"
        )

    for column in [
        "source_root_exists",
        "all_candidates_scanned",
        "all_valid_series_selected",
        "fixed_target_discovered",
        "fixed_target_path_exists",
        "fixed_target_valid",
        "fixed_target_manifest_reusable",
        "source_type_all_updated_10",
        "all_paths_under_updated_root",
    ]:
        if not updated_10_audit[column].astype(bool).all():
            bad = updated_10_audit.loc[
                ~updated_10_audit[column].astype(bool), "patient_id"
            ].astype(str).tolist()
            errors.append(f"Updated-10 audit failed {column}: {bad}")

    split_lookup = {
        **{patient_id: "Train" for patient_id in train_patients},
        **{patient_id: "Valid" for patient_id in valid_patients},
    }
    split_mismatches = []
    for row in updated_10_audit.to_dict("records"):
        actual = split_lookup.get(str(row["patient_id"]), "missing")
        if actual != str(row["expected_split"]):
            split_mismatches.append(
                (str(row["patient_id"]), str(row["expected_split"]), actual)
            )
    if split_mismatches:
        errors.append(f"Updated-10 split mismatches: {split_mismatches}")

    if updated_10_audit["old_tiantan_root_used"].astype(bool).any():
        bad = updated_10_audit.loc[
            updated_10_audit["old_tiantan_root_used"].astype(bool),
            "patient_id",
        ].astype(str).tolist()
        errors.append(f"Updated-10 cases fell back to old tiantanDSA root: {bad}")

    summary = {
        "train_record_rows": int(len(train_records)),
        "valid_record_rows": int(len(valid_records)),
        "unique_train_patients": int(train_records["patient_id"].nunique()),
        "unique_valid_patients": int(valid_records["patient_id"].nunique()),
        "all_candidate_series_rows": int(len(series_df)),
        "selected_valid_series_rows": int(len(selected)),
        "selected_train_series_rows": int((selected["split"] == "Train").sum()),
        "selected_valid_split_series_rows": int((selected["split"] == "Valid").sum()),
        "fully_reusable_v2_series": int(
            selected["v2_pairdata_fully_reusable"].astype(bool).sum()
        ),
        "incremental_searaft_series": int(
            selected["needs_incremental_searaft"].astype(bool).sum()
        ),
        "selected_frame_rows": int(len(selected_frames)),
        "updated_10_rows": int(len(updated_10_audit)),
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
    outdir = args.outdir.resolve()
    success_path = outdir / ".SUCCESS"
    failed_path = outdir / ".FAILED"

    try:
        success_path.unlink(missing_ok=True)
        failed_path.unlink(missing_ok=True)

        code14_global = load_code14(args.code14)
        code14 = code14_global

        train_xlsx = args.project / "metadata/Train.xlsx"
        valid_xlsx = args.project / "metadata/valid.xlsx"
        old_train = args.project / "manifests/api_fullseq_v2_train_manifest.csv"
        old_valid = args.project / "manifests/api_fullseq_v2_valid_manifest.csv"

        outputs = {
            "train_records": outdir / "train_record_table.csv",
            "valid_records": outdir / "valid_record_table.csv",
            "train_series": outdir / "train_all_series_manifest.csv",
            "valid_series": outdir / "valid_all_series_manifest.csv",
            "all_candidates": outdir / "all_candidate_series_audit.csv",
            "frames": outdir / "all_series_frame_audit.csv",
            "incremental": outdir / "incremental_searaft_series_manifest.csv",
            "reusable": outdir / "reusable_v2_series_manifest.csv",
            "train_record_series": outdir / "train_record_series_suggestions.csv",
            "valid_record_series": outdir / "valid_record_series_suggestions.csv",
            "mapping_review": outdir / "record_series_mapping_review.csv",
            "coverage": outdir / "record_series_coverage_audit.csv",
            "updated_10_audit": outdir / "updated_10_series_audit.csv",
            "summary": outdir / "manifest_summary.json",
            "success": success_path,
            "failed": failed_path,
        }

        existing = [
            str(path) for key, path in outputs.items()
            if key not in {"failed"} and path.exists()
        ]
        if existing and not args.overwrite:
            raise FileExistsError(
                "Refusing to overwrite existing outputs. "
                "Use --overwrite only after backing up/reviewing:\n"
                + "\n".join(existing)
            )

        train_records = build_record_table(train_xlsx, "Train")
        valid_records = build_record_table(valid_xlsx, "Valid")
        old_index = load_old_v2_index(old_train, old_valid)

        unique_train = sorted(
            train_records["patient_id"].unique(), key=code14.patient_key
        )
        unique_valid = sorted(
            valid_records["patient_id"].unique(), key=code14.patient_key
        )
        patients = (
            [(pid, "Train") for pid in unique_train]
            + [(pid, "Valid") for pid in unique_valid]
        )

        series_rows: list[dict[str, Any]] = []
        frame_rows: list[dict[str, Any]] = []
        with ThreadPoolExecutor(
            max_workers=args.workers,
            thread_name_prefix="api_record_v1",
        ) as executor:
            for i, (pid, split) in enumerate(patients, start=1):
                patient_series, patient_frames = build_all_series_for_patient(
                    code14, args.project, pid, split, executor, old_index
                )
                series_rows.extend(patient_series)
                frame_rows.extend(patient_frames)
                if i % 50 == 0 or i == len(patients):
                    print(
                        f"[{i}/{len(patients)}] "
                        f"candidate_series={len(series_rows)} "
                        f"frame_audit={len(frame_rows)}",
                        flush=True,
                    )

        series_df = pd.DataFrame(series_rows)
        frame_df = pd.DataFrame(frame_rows)
        selected_df = series_df[
            series_df["selected_for_extraction"].astype(bool)
        ].copy()
        train_series = selected_df[selected_df["split"] == "Train"].copy()
        valid_series = selected_df[selected_df["split"] == "Valid"].copy()
        incremental = selected_df[
            selected_df["needs_incremental_searaft"].astype(bool)
        ].copy()
        reusable = selected_df[
            selected_df["v2_pairdata_fully_reusable"].astype(bool)
        ].copy()

        train_record_series = build_record_series_suggestions(
            train_records, train_series
        )
        valid_record_series = build_record_series_suggestions(
            valid_records, valid_series
        )
        all_suggestions = pd.concat(
            [train_record_series, valid_record_series],
            ignore_index=True,
        )
        mapping_review = all_suggestions[
            all_suggestions["mapping_confidence"].isin(["low", "unavailable"])
        ].copy()
        coverage = pd.concat(
            [
                build_record_series_coverage_audit(
                    train_records, train_series, train_record_series
                ),
                build_record_series_coverage_audit(
                    valid_records, valid_series, valid_record_series
                ),
            ],
            ignore_index=True,
        )
        updated_10_audit = build_updated_10_audit(code14, series_df)

        summary = validate(
            code14,
            train_records,
            valid_records,
            series_df,
            frame_df,
            updated_10_audit,
        )
        summary.update({
            "auto_suggested_train_records": int(
                train_record_series["suggested_series_uid"].astype(str).ne("").sum()
            ),
            "auto_suggested_valid_records": int(
                valid_record_series["suggested_series_uid"].astype(str).ne("").sum()
            ),
            "mapping_review_rows": int(len(mapping_review)),
            "patients_equal_record_series_count": int(
                (coverage["record_series_count_relation"] == "equal").sum()
            ),
            "patients_more_series_than_records": int(
                (coverage["record_series_count_relation"] == "more_series_than_records").sum()
            ),
            "patients_fewer_series_than_records": int(
                (coverage["record_series_count_relation"] == "fewer_series_than_records").sum()
            ),
            "updated_10_all_valid_siblings_scanned": True,
            "updated_10_old_root_fallback_forbidden": True,
            "reuse_internal_id_numeric_normalization": True,
            "record_series_suggestions_are_ground_truth": False,
        })

        atomic_csv(train_records, outputs["train_records"])
        atomic_csv(valid_records, outputs["valid_records"])
        atomic_csv(train_series, outputs["train_series"])
        atomic_csv(valid_series, outputs["valid_series"])
        atomic_csv(series_df, outputs["all_candidates"])
        atomic_csv(frame_df, outputs["frames"])
        atomic_csv(incremental, outputs["incremental"])
        atomic_csv(reusable, outputs["reusable"])
        atomic_csv(train_record_series, outputs["train_record_series"])
        atomic_csv(valid_record_series, outputs["valid_record_series"])
        atomic_csv(mapping_review, outputs["mapping_review"])
        atomic_csv(coverage, outputs["coverage"])
        atomic_csv(updated_10_audit, outputs["updated_10_audit"])
        atomic_json(summary, outputs["summary"])
        atomic_json(
            {
                "status": "success",
                "manifest_summary": str(outputs["summary"]),
                "selected_series_rows": int(len(selected_df)),
                "train_record_rows": int(len(train_records)),
                "valid_record_rows": int(len(valid_records)),
                "record_series_suggestions_are_ground_truth": False,
            },
            outputs["success"],
        )

        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Outputs: {outdir}")
        return 0

    except BaseException as exc:
        outdir.mkdir(parents=True, exist_ok=True)
        success_path.unlink(missing_ok=True)
        atomic_json(
            {
                "status": "failed",
                "exception": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
            failed_path,
        )
        print(traceback.format_exc(), file=sys.stderr)
        return 42


if __name__ == "__main__":
    raise SystemExit(main())

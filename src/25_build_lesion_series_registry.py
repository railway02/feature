#!/usr/bin/env python3
"""Build the blinded api_fullseq_v3 lesion-to-series candidate registry.

Phase 1 only: register frozen Excel lesion records against existing v2
patient/candidate/frame audits.  This script does not inspect image pixels,
run SEA-RAFT, extract features, use outcomes for series selection, or train a
model.  All v2 files are read-only inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PROJECT = Path("/root/autodl-tmp/aneurysm")

EXPECTED = {
    "Train": {"records": 1157, "patients": 1055},
    "Valid": {"records": 289, "patients": 264},
}

ALLOWED_INITIAL_STATUSES = {
    "provisional_single_lesion",
    "candidate_available",
    "review_required",
    "ambiguous",
    "unmatched",
}

FORBIDDEN_REVIEWED_STATUSES = {"exact_reviewed", "probable_reviewed"}

EXCEL_COLUMNS = {
    "patient_id": "病案号",
    "multiple": "是否多发",
    "side": "侧别",
    "location": "部位",
}

LESION_INDEX_COLUMN_CANDIDATES = (
    "lesion_index",
    "病灶序号",
    "病灶编号",
    "动脉瘤序号",
    "动脉瘤编号",
)

OUTCOME_COLUMNS = {
    "immediate_rroc": "术后即刻RROC",
    "followup_rroc": "随访RROC123",
    "adverse_outcome": "不良转归：1是；0否",
    "dsa_date": "DSA时间",
    "last_followup_date": "最后一次随访时间",
    "followup_interval_months": "随访间隔（月）",
    "followup_time": "随访时间",
    "followup_time_duplicate": "随访时间.1",
}

PATIENT_REQUIRED_COLUMNS = {
    "patient_id",
    "split",
    "source_type",
    "source_medical_record_root",
    "selected_series_id",
    "selected_series_path",
    "selected_candidate_rank",
    "selected_pre_internal_series",
    "selected_post_internal_series",
    "pre_frame_paths",
    "post_frame_paths",
    "patient_status",
    "exclusion_reason",
}

CANDIDATE_REQUIRED_COLUMNS = {
    "patient_id",
    "split",
    "source_type",
    "source_medical_record_root",
    "discovery_rank",
    "series_id",
    "series_path",
    "is_fixed_target",
    "scan_performed",
    "validity_evaluated",
    "pre_api_dir",
    "post_api_dir",
    "pre_api_dir_exists",
    "post_api_dir_exists",
    "pre_extra_api_dirs",
    "post_extra_api_dirs",
    "pre_parameter_only",
    "post_parameter_only",
    "pre_internal_series",
    "post_internal_series",
    "selected_pre_internal_series",
    "selected_post_internal_series",
    "pre_ignored_internal_series",
    "post_ignored_internal_series",
    "n_pre_frames",
    "n_post_frames",
    "n_pre_contiguous_pairs",
    "n_post_contiguous_pairs",
    "pre_frame_gaps",
    "post_frame_gaps",
    "can_run_pre",
    "can_run_post",
    "can_run_prepost",
    "candidate_valid",
    "selected_candidate",
    "selection_status",
    "candidate_exclusion_reason",
    "pre_internal_series_audit",
    "post_internal_series_audit",
}

FRAME_REQUIRED_COLUMNS = {
    "patient_id",
    "split",
    "discovery_rank",
    "series_id",
    "series_path",
    "phase",
    "absolute_path",
    "strict_filename_match",
    "internal_series_number",
    "is_parameter_map",
    "is_nonstandard_jpeg",
    "phase_eligible_frame",
    "selected",
}

BLINDED_COLUMNS = [
    "lesion_uid",
    "split",
    "patient_id",
    "source_excel_file",
    "source_excel_sheet",
    "source_excel_row_id",
    "side_raw",
    "side_normalized",
    "location_raw",
    "location_normalized",
    "lesion_index_raw",
    "lesion_index_normalized",
    "lesion_index_source",
    "multiple_aneurysm_raw",
    "multiple_aneurysm_normalized",
    "patient_lesion_record_count",
    "patient_registry_multiplicity",
    "metadata_ambiguity_flag",
    "metadata_ambiguity_reason",
    "patient_manifest_matched",
    "patient_manifest_source_type",
    "patient_manifest_source_medical_record_root",
    "patient_manifest_status",
    "patient_manifest_exclusion_reason",
    "candidate_series_count",
    "valid_candidate_series_count",
    "v2_selected_series_id_candidate",
    "v2_selected_series_path_candidate",
    "v2_selected_candidate_rank",
    "v2_selected_pre_internal_series_candidate",
    "v2_selected_post_internal_series_candidate",
    "v2_selected_pre_frame_paths_candidate",
    "v2_selected_post_frame_paths_candidate",
    "candidate_series_registry_json",
    "candidate_available",
    "registration_status",
    "requires_manual_review",
    "candidate_assignment_scope",
]

PRIVATE_COLUMNS = [
    "lesion_uid",
    "split",
    "patient_id",
    "source_excel_file",
    "source_excel_sheet",
    "source_excel_row_id",
    "immediate_rroc_raw",
    "immediate_rroc_normalized",
    "followup_rroc_raw",
    "followup_rroc_normalized",
    "adverse_outcome_raw",
    "adverse_outcome_normalized",
    "dsa_date_raw",
    "dsa_date_normalized",
    "last_followup_date_raw",
    "last_followup_date_normalized",
    "followup_interval_months_raw",
    "followup_interval_months_normalized",
    "followup_time_raw",
    "followup_time_normalized",
    "followup_time_duplicate_raw",
    "followup_time_duplicate_normalized",
]

MANUAL_REVIEW_COLUMNS = [
    "lesion_uid",
    "split",
    "patient_id",
    "source_excel_file",
    "source_excel_row_id",
    "side_raw",
    "side_normalized",
    "location_raw",
    "location_normalized",
    "lesion_index_raw",
    "lesion_index_normalized",
    "multiple_aneurysm_raw",
    "multiple_aneurysm_normalized",
    "patient_lesion_record_count",
    "metadata_ambiguity_flag",
    "metadata_ambiguity_reason",
    "current_registration_status",
    "manual_review_reason",
    "candidate_series_count",
    "valid_candidate_series_count",
    "candidate_series_registry_json",
    "reviewer_id",
    "reviewed_at_utc",
    "review_decision",
    "reviewed_series_id",
    "reviewed_pre_internal_series",
    "reviewed_post_internal_series",
    "review_notes",
]

BLINDED_FORBIDDEN_COLUMN_TOKENS = (
    "rroc",
    "adverse",
    "outcome",
    "followup",
    "follow_up",
    "follow-up",
    "随访",
    "不良转归",
    "术后即刻",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--train-xlsx", type=Path)
    parser.add_argument("--valid-xlsx", type=Path)
    parser.add_argument("--patient-manifest", type=Path)
    parser.add_argument("--candidate-audit", type=Path)
    parser.add_argument(
        "--frame-audit",
        type=Path,
        help=(
            "Read-only existing v2 frame-path audit. Defaults to "
            "manifests/api_fullseq_v2_frame_audit.csv."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_columns(df: pd.DataFrame, required: Iterable[str], source: Path) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def raw_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    text = str(value).strip()
    if re.fullmatch(r"[-+]?\d+\.0+", text):
        return text.split(".", 1)[0]
    return text


def normalize_patient_id(value: Any) -> str:
    text = raw_text(value)
    if not text:
        raise ValueError("Encountered empty patient_id/病案号")
    if not re.fullmatch(r"\d+", text):
        raise ValueError(f"Non-numeric patient_id after normalization: {value!r}")
    return str(int(text))


def canonical_split(value: Any) -> str:
    text = raw_text(value).casefold()
    mapping = {"train": "Train", "valid": "Valid", "validation": "Valid"}
    if text not in mapping:
        raise ValueError(f"Unexpected split value: {value!r}")
    return mapping[text]


def bool_value(value: Any) -> bool:
    text = raw_text(value).casefold()
    if text in {"true", "1", "yes", "y", "是"}:
        return True
    if text in {"false", "0", "no", "n", "否", ""}:
        return False
    raise ValueError(f"Unexpected boolean value: {value!r}")


def normalize_binary(value: Any) -> str:
    text = raw_text(value).casefold()
    if text in {"1", "true", "yes", "y", "是"}:
        return "1"
    if text in {"0", "false", "no", "n", "否"}:
        return "0"
    return ""


def normalize_side(value: Any) -> str:
    text = raw_text(value)
    if not text:
        return "unspecified"
    folded = re.sub(r"[\s_\-]+", "", text).casefold()
    if folded in {"l", "left", "左", "左侧"}:
        return "left"
    if folded in {"r", "right", "右", "右侧"}:
        return "right"
    if folded in {"b", "bilateral", "双", "双侧"}:
        return "bilateral"
    return text.strip().casefold()


def normalize_location(value: Any) -> str:
    text = raw_text(value)
    if not text:
        return "unspecified"
    compact = re.sub(r"[\s_\-]+", "", text).casefold()
    mapping = {
        "acom": "ACOM",
        "aom": "ACOM",
        "pcom": "PCOM",
        "ach": "ACHA",
        "acha": "ACHA",
        "mca": "MCA",
        "aca": "ACA",
        "pca": "PCA",
        "pica": "PICA",
        "sca": "SCA",
        "va": "VA",
        "ba": "BA",
        "c4": "C4",
        "c5": "C5",
        "c6": "C6",
        "颈内末端分叉": "ICA_TERMINUS",
        "颈内末端分成": "ICA_TERMINUS",
        "颈内末端": "ICA_TERMINUS",
    }
    return mapping.get(compact, text.strip().upper())


def normalize_integral_text(value: Any) -> str:
    text = raw_text(value)
    if not text:
        return ""
    try:
        number = Decimal(text)
    except InvalidOperation:
        return text
    if number == number.to_integral_value():
        return str(int(number))
    return format(number.normalize(), "f")


def normalize_number(value: Any) -> str:
    text = raw_text(value)
    if not text:
        return ""
    try:
        number = Decimal(text)
    except InvalidOperation:
        return text
    if number == number.to_integral_value():
        return str(int(number))
    return format(number.normalize(), "f")


def normalize_date(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = raw_text(value)
    if not text:
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return text
    return parsed.date().isoformat()


def parse_json_list(value: Any, field_name: str) -> list[Any]:
    text = raw_text(value)
    if not text:
        return []
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError(f"{field_name} must contain a JSON list")
    return parsed


def natural_key(value: str) -> tuple[tuple[int, Any], ...]:
    parts = re.split(r"(\d+)", value.casefold())
    return tuple((0, int(part)) if part.isdigit() else (1, part) for part in parts)


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def pipe_values(value: Any) -> list[str]:
    return [part for part in raw_text(value).split("|") if part]


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    df.to_csv(
        temp,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
    )
    os.replace(temp, path)


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def load_excel_records(path: Path, split: str) -> tuple[pd.DataFrame, str, str | None]:
    excel = pd.ExcelFile(path)
    if not excel.sheet_names:
        raise ValueError(f"No worksheet found in {path}")
    sheet = excel.sheet_names[0]
    df = pd.read_excel(excel, sheet_name=sheet, dtype=object)
    require_columns(df, set(EXCEL_COLUMNS.values()) | set(OUTCOME_COLUMNS.values()), path)

    missing_patient_rows = df[EXCEL_COLUMNS["patient_id"]].isna()
    if missing_patient_rows.any():
        rows = (df.index[missing_patient_rows] + 2).tolist()
        raise ValueError(f"{path} has empty patient IDs at Excel rows {rows}")

    df = df.copy()
    df["_split"] = split
    df["_source_excel_file"] = path.name
    df["_source_excel_sheet"] = sheet
    df["_source_excel_row_id"] = df.index + 2
    df["_patient_id"] = df[EXCEL_COLUMNS["patient_id"]].map(normalize_patient_id)

    lesion_index_column = next(
        (column for column in LESION_INDEX_COLUMN_CANDIDATES if column in df.columns),
        None,
    )
    if lesion_index_column is None:
        df["_lesion_index_raw"] = ""
    else:
        df["_lesion_index_raw"] = df[lesion_index_column].map(raw_text)

    df["_lesion_index_normalized"] = (
        df.groupby("_patient_id", sort=False).cumcount() + 1
    ).astype(int)
    df["_lesion_index_source"] = (
        "normalized_from_source_column"
        if lesion_index_column is not None
        else "derived_from_frozen_excel_row_order"
    )
    df["_lesion_uid"] = df.apply(
        lambda row: (
            f"{row['_split']}:{row['_patient_id']}:"
            f"excel_row_{int(row['_source_excel_row_id']):06d}"
        ),
        axis=1,
    )
    return df, sheet, lesion_index_column


def load_v2_csv(path: Path, required: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    require_columns(df, required, path)
    df = df.copy()
    df["split"] = df["split"].map(canonical_split)
    df["patient_id"] = df["patient_id"].map(normalize_patient_id)
    return df


def candidate_key(row: Any) -> tuple[str, str, str, str, str]:
    return (
        canonical_split(getattr(row, "split")),
        normalize_patient_id(getattr(row, "patient_id")),
        normalize_integral_text(getattr(row, "discovery_rank")),
        raw_text(getattr(row, "series_id")),
        raw_text(getattr(row, "series_path")),
    )


def build_frame_index(frame_audit: pd.DataFrame) -> dict[tuple[str, str, str, str, str], dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "pre": {
                "strict_paths": defaultdict(list),
                "eligible_paths": defaultdict(list),
                "selected_paths": [],
                "nonstandard_image_paths": [],
            },
            "post": {
                "strict_paths": defaultdict(list),
                "eligible_paths": defaultdict(list),
                "selected_paths": [],
                "nonstandard_image_paths": [],
            },
        }
    )

    for row in frame_audit.itertuples(index=False):
        key = candidate_key(row)
        phase = raw_text(row.phase).casefold()
        if phase not in {"pre", "post"}:
            raise ValueError(f"Unexpected phase in frame audit: {row.phase!r}")
        path = raw_text(row.absolute_path)
        if not path:
            continue
        group = groups[key][phase]
        if bool_value(row.is_nonstandard_jpeg):
            group["nonstandard_image_paths"].append(path)
        if bool_value(row.strict_filename_match) and not bool_value(row.is_parameter_map):
            internal = normalize_integral_text(row.internal_series_number) or "unassigned"
            group["strict_paths"][internal].append(path)
            if bool_value(row.phase_eligible_frame):
                group["eligible_paths"][internal].append(path)
            if bool_value(row.selected):
                group["selected_paths"].append(path)

    final: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for key, phases in groups.items():
        final[key] = {}
        for phase, values in phases.items():
            strict_paths = {
                internal: unique_preserving_order(paths)
                for internal, paths in sorted(
                    values["strict_paths"].items(), key=lambda item: natural_key(item[0])
                )
            }
            eligible_paths = {
                internal: unique_preserving_order(paths)
                for internal, paths in sorted(
                    values["eligible_paths"].items(), key=lambda item: natural_key(item[0])
                )
            }
            final[key][phase] = {
                "strict_frame_paths_by_internal_series": strict_paths,
                "eligible_frame_paths_by_internal_series": eligible_paths,
                "selected_frame_paths": unique_preserving_order(values["selected_paths"]),
                "nonstandard_image_paths": unique_preserving_order(
                    values["nonstandard_image_paths"]
                ),
                "strict_frame_path_count": sum(len(paths) for paths in strict_paths.values()),
                "eligible_frame_path_count": sum(len(paths) for paths in eligible_paths.values()),
            }
    return final


def build_candidate_bundle(
    split: str,
    patient_id: str,
    candidate_rows: pd.DataFrame,
    frame_index: dict[tuple[str, str, str, str, str], dict[str, Any]],
) -> tuple[str, int, int]:
    rows = candidate_rows.copy()
    rows["_rank_sort"] = rows["discovery_rank"].map(
        lambda value: int(normalize_integral_text(value))
        if normalize_integral_text(value).isdigit()
        else 10**9
    )
    rows = rows.sort_values(["_rank_sort", "series_id", "series_path"], kind="stable")
    candidates: list[dict[str, Any]] = []

    for row in rows.itertuples(index=False):
        key = candidate_key(row)
        frames = frame_index.get(
            key,
            {
                "pre": {
                    "strict_frame_paths_by_internal_series": {},
                    "eligible_frame_paths_by_internal_series": {},
                    "selected_frame_paths": [],
                    "nonstandard_image_paths": [],
                    "strict_frame_path_count": 0,
                    "eligible_frame_path_count": 0,
                },
                "post": {
                    "strict_frame_paths_by_internal_series": {},
                    "eligible_frame_paths_by_internal_series": {},
                    "selected_frame_paths": [],
                    "nonstandard_image_paths": [],
                    "strict_frame_path_count": 0,
                    "eligible_frame_path_count": 0,
                },
            },
        )
        candidate = {
            "candidate_source": {
                "source_type": raw_text(row.source_type),
                "source_medical_record_root": raw_text(row.source_medical_record_root),
                "discovery_rank": normalize_integral_text(row.discovery_rank),
                "series_id": raw_text(row.series_id),
                "series_path": raw_text(row.series_path),
                "is_fixed_target": bool_value(row.is_fixed_target),
            },
            "candidate_audit": {
                "scan_performed": bool_value(row.scan_performed),
                "validity_evaluated": bool_value(row.validity_evaluated),
                "candidate_valid": bool_value(row.candidate_valid),
                "selected_candidate_in_v2": bool_value(row.selected_candidate),
                "selection_status_in_v2": raw_text(row.selection_status),
                "candidate_exclusion_reason": raw_text(row.candidate_exclusion_reason),
                "can_run_pre": bool_value(row.can_run_pre),
                "can_run_post": bool_value(row.can_run_post),
                "can_run_prepost": bool_value(row.can_run_prepost),
            },
            "pre": {
                "api_dir": raw_text(row.pre_api_dir),
                "api_dir_exists": bool_value(row.pre_api_dir_exists),
                "extra_api_dirs": pipe_values(row.pre_extra_api_dirs),
                "parameter_only": bool_value(row.pre_parameter_only),
                "internal_series": pipe_values(row.pre_internal_series),
                "selected_internal_series_in_v2": normalize_integral_text(
                    row.selected_pre_internal_series
                ),
                "ignored_internal_series": pipe_values(row.pre_ignored_internal_series),
                "n_frames_in_selected_internal_series": normalize_integral_text(row.n_pre_frames),
                "n_contiguous_pairs_in_selected_internal_series": normalize_integral_text(
                    row.n_pre_contiguous_pairs
                ),
                "frame_gaps_in_selected_internal_series": pipe_values(row.pre_frame_gaps),
                "internal_series_audit": parse_json_list(
                    row.pre_internal_series_audit, "pre_internal_series_audit"
                ),
                "frame_path_source": "api_fullseq_v2_frame_audit.csv",
                **frames["pre"],
            },
            "post": {
                "api_dir": raw_text(row.post_api_dir),
                "api_dir_exists": bool_value(row.post_api_dir_exists),
                "extra_api_dirs": pipe_values(row.post_extra_api_dirs),
                "parameter_only": bool_value(row.post_parameter_only),
                "internal_series": pipe_values(row.post_internal_series),
                "selected_internal_series_in_v2": normalize_integral_text(
                    row.selected_post_internal_series
                ),
                "ignored_internal_series": pipe_values(row.post_ignored_internal_series),
                "n_frames_in_selected_internal_series": normalize_integral_text(row.n_post_frames),
                "n_contiguous_pairs_in_selected_internal_series": normalize_integral_text(
                    row.n_post_contiguous_pairs
                ),
                "frame_gaps_in_selected_internal_series": pipe_values(row.post_frame_gaps),
                "internal_series_audit": parse_json_list(
                    row.post_internal_series_audit, "post_internal_series_audit"
                ),
                "frame_path_source": "api_fullseq_v2_frame_audit.csv",
                **frames["post"],
            },
            "registration_scope": "patient_level_candidate_only_pending_lesion_review",
        }
        candidates.append(candidate)

    valid_count = sum(
        1 for candidate in candidates if candidate["candidate_audit"]["candidate_valid"]
    )
    payload = {
        "split": split,
        "patient_id": patient_id,
        "candidate_count": len(candidates),
        "valid_candidate_count": valid_count,
        "candidates": candidates,
    }
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        len(candidates),
        valid_count,
    )


def build_metadata_flags(records: pd.DataFrame) -> pd.DataFrame:
    records = records.copy()
    records["_side_raw"] = records[EXCEL_COLUMNS["side"]].map(raw_text)
    records["_side_normalized"] = records[EXCEL_COLUMNS["side"]].map(normalize_side)
    records["_location_raw"] = records[EXCEL_COLUMNS["location"]].map(raw_text)
    records["_location_normalized"] = records[EXCEL_COLUMNS["location"]].map(
        normalize_location
    )
    records["_multiple_raw"] = records[EXCEL_COLUMNS["multiple"]].map(raw_text)
    records["_multiple_normalized"] = records[EXCEL_COLUMNS["multiple"]].map(
        normalize_binary
    )
    records["_patient_lesion_record_count"] = records.groupby("_patient_id")[
        "_patient_id"
    ].transform("size")

    patient_multiple_values = records.groupby("_patient_id")["_multiple_normalized"].agg(
        lambda series: sorted({value for value in series if value})
    )
    inconsistent_multiple_patients = {
        patient_id for patient_id, values in patient_multiple_values.items() if len(values) > 1
    }

    records["_patient_is_multiple"] = (
        records["_multiple_normalized"].eq("1")
        | records["_patient_lesion_record_count"].gt(1)
    )

    duplicated_attributes = records.duplicated(
        subset=["_patient_id", "_side_normalized", "_location_normalized"],
        keep=False,
    ) & records["_patient_is_multiple"]

    flags: list[bool] = []
    reasons: list[str] = []
    for index, row in records.iterrows():
        row_reasons: list[str] = []
        if row["_patient_id"] in inconsistent_multiple_patients:
            row_reasons.append("inconsistent_multiple_aneurysm_values_within_patient")
        if duplicated_attributes.loc[index]:
            row_reasons.append("repeated_side_location_attributes_within_multi_lesion_patient")
        if row["_side_normalized"] == "unspecified":
            row_reasons.append("side_unspecified")
        if row["_location_normalized"] == "unspecified":
            row_reasons.append("location_unspecified")
        if not row["_multiple_normalized"]:
            row_reasons.append("multiple_aneurysm_value_unrecognized")
        flags.append(bool(row_reasons))
        reasons.append("|".join(row_reasons))
    records["_metadata_ambiguity_flag"] = flags
    records["_metadata_ambiguity_reason"] = reasons
    return records


def build_private_row(row: pd.Series) -> dict[str, Any]:
    return {
        "lesion_uid": row["_lesion_uid"],
        "split": row["_split"],
        "patient_id": row["_patient_id"],
        "source_excel_file": row["_source_excel_file"],
        "source_excel_sheet": row["_source_excel_sheet"],
        "source_excel_row_id": int(row["_source_excel_row_id"]),
        "immediate_rroc_raw": raw_text(row[OUTCOME_COLUMNS["immediate_rroc"]]),
        "immediate_rroc_normalized": normalize_integral_text(
            row[OUTCOME_COLUMNS["immediate_rroc"]]
        ),
        "followup_rroc_raw": raw_text(row[OUTCOME_COLUMNS["followup_rroc"]]),
        "followup_rroc_normalized": normalize_integral_text(
            row[OUTCOME_COLUMNS["followup_rroc"]]
        ),
        "adverse_outcome_raw": raw_text(row[OUTCOME_COLUMNS["adverse_outcome"]]),
        "adverse_outcome_normalized": normalize_binary(
            row[OUTCOME_COLUMNS["adverse_outcome"]]
        ),
        "dsa_date_raw": raw_text(row[OUTCOME_COLUMNS["dsa_date"]]),
        "dsa_date_normalized": normalize_date(row[OUTCOME_COLUMNS["dsa_date"]]),
        "last_followup_date_raw": raw_text(row[OUTCOME_COLUMNS["last_followup_date"]]),
        "last_followup_date_normalized": normalize_date(
            row[OUTCOME_COLUMNS["last_followup_date"]]
        ),
        "followup_interval_months_raw": raw_text(
            row[OUTCOME_COLUMNS["followup_interval_months"]]
        ),
        "followup_interval_months_normalized": normalize_number(
            row[OUTCOME_COLUMNS["followup_interval_months"]]
        ),
        "followup_time_raw": raw_text(row[OUTCOME_COLUMNS["followup_time"]]),
        "followup_time_normalized": normalize_number(row[OUTCOME_COLUMNS["followup_time"]]),
        "followup_time_duplicate_raw": raw_text(
            row[OUTCOME_COLUMNS["followup_time_duplicate"]]
        ),
        "followup_time_duplicate_normalized": normalize_number(
            row[OUTCOME_COLUMNS["followup_time_duplicate"]]
        ),
    }


def list_text(values: Iterable[str]) -> str:
    items = sorted(set(values), key=natural_key)
    return ", ".join(items) if items else "None"


def validate_blinded_columns(columns: Iterable[str], output_name: str) -> None:
    violations = [
        column
        for column in columns
        if any(token in column.casefold() for token in BLINDED_FORBIDDEN_COLUMN_TOKENS)
    ]
    if violations:
        raise AssertionError(f"Forbidden outcome/time columns in {output_name}: {violations}")


def build_registry_audit(
    registries: dict[str, pd.DataFrame],
    manual_review: pd.DataFrame,
    private_labels: pd.DataFrame,
    patient_overlap: set[str],
    input_hashes: dict[str, str],
    lesion_index_columns: dict[str, str | None],
) -> str:
    lines = [
        "# api_fullseq_v3 Lesion Registry Audit",
        "",
        "## Scope and safety boundary",
        "",
        "- Phase completed: lesion record to candidate imaging-series registration only.",
        "- Candidate registration used existing v2 patient/candidate/frame audits; image pixels were not opened.",
        "- No ROI work, temporal audit, feature extraction, SEA-RAFT execution, or model training was performed.",
        "- Candidate/status logic did not read private label fields.",
        "- Patient-level selected series are stored only as candidates and are never lesion-confirmed assignments.",
        "- No reviewed exact/probable registration status was generated.",
        "",
        "## Frozen lesion identity",
        "",
        "`lesion_uid = split + patient_id + source_excel_row_id`, formatted as "
        "`Split:patient_id:excel_row_######`.",
        "",
        "`source_excel_row_id` is the physical worksheet row number, including the header at row 1. "
        "The first lesion record is therefore row 2. Side, location, lesion index, and multiple-aneurysm "
        "fields are attributes only and do not participate in identity.",
        "",
        "The current Excel files contain no explicit lesion-index column. `lesion_index_raw` is therefore "
        "blank, while `lesion_index_normalized` is the deterministic 1-based order of frozen Excel rows "
        "within each split/patient.",
        "",
        "## Hard counts",
        "",
        "| Split | Lesion records | Unique patients | Unique lesion_uid | Single-lesion patients | Multiple-lesion patients | Lesion rows in single patients | Lesion rows in multiple patients | Candidate available rows | No-candidate rows | Manual-review rows | Manifest-unmatched rows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("Train", "Valid"):
        df = registries[split]
        patient_class = df.groupby("patient_id")["patient_registry_multiplicity"].first()
        single_patients = int((patient_class == "single_lesion").sum())
        multi_patients = int((patient_class == "multiple_lesion").sum())
        single_rows = int((df["patient_registry_multiplicity"] == "single_lesion").sum())
        multi_rows = int((df["patient_registry_multiplicity"] == "multiple_lesion").sum())
        lines.append(
            f"| {split} | {len(df)} | {df['patient_id'].nunique()} | {df['lesion_uid'].nunique()} | "
            f"{single_patients} | {multi_patients} | {single_rows} | {multi_rows} | "
            f"{int(df['candidate_available'].sum())} | {int((~df['candidate_available']).sum())} | "
            f"{int(df['requires_manual_review'].sum())} | "
            f"{int((~df['patient_manifest_matched']).sum())} |"
        )

    lines.extend(
        [
            "",
            f"- Train/Valid patient_id intersection: **{len(patient_overlap)}**",
            f"- Combined private label rows: **{len(private_labels)}**",
            f"- Combined manual-review rows: **{len(manual_review)}**",
            "",
            "## Initial registration statuses",
            "",
            "| Split | Status | Lesion rows |",
            "|---|---|---:|",
        ]
    )
    for split in ("Train", "Valid"):
        counts = registries[split]["registration_status"].value_counts()
        for status in sorted(ALLOWED_INITIAL_STATUSES):
            lines.append(f"| {split} | {status} | {int(counts.get(status, 0))} |")

    lines.extend(
        [
            "",
            "## Patient-manifest connection failures",
            "",
        ]
    )
    for split in ("Train", "Valid"):
        unmatched = registries[split].loc[
            ~registries[split]["patient_manifest_matched"], "patient_id"
        ]
        lines.append(f"- {split}: {len(unmatched)} lesion rows; patient IDs: {list_text(unmatched)}")

    lines.extend(
        [
            "",
            "## Blinding checks",
            "",
            "- Blinded registry forbidden label/time columns: 0",
            "- Manual-review forbidden label/time columns: 0",
            "- Private labels are keyed only by frozen lesion identity and stored separately.",
            "- Reviewed exact/probable statuses in generated registries: 0",
            "",
            "## Input hashes (read-only)",
            "",
            "| Input | SHA-256 |",
            "|---|---|",
        ]
    )
    for path, digest in input_hashes.items():
        lines.append(f"| `{path}` | `{digest}` |")
    lines.extend(
        [
            "",
            "## Lesion-index source columns",
            "",
            f"- Train explicit lesion-index source column: {lesion_index_columns['Train'] or 'None'}",
            f"- Valid explicit lesion-index source column: {lesion_index_columns['Valid'] or 'None'}",
            "",
        ]
    )
    return "\n".join(lines)


def build_alignment_audit(
    excel_ids: dict[str, set[str]],
    patient_manifest: pd.DataFrame,
    candidate_audit: pd.DataFrame,
    frame_index: dict[tuple[str, str, str, str, str], dict[str, Any]],
    candidate_keys: set[tuple[str, str, str, str, str]],
) -> str:
    lines = [
        "# api_fullseq_v3 Split ID Alignment Audit",
        "",
        "## Split-level alignment",
        "",
        "| Split | Excel patients | Patient manifest patients | Candidate audit patients | Excel missing from patient manifest | Patient manifest extras | Excel missing from candidate audit | Candidate audit extras |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    details: list[str] = []
    for split in ("Train", "Valid"):
        manifest_ids = set(patient_manifest.loc[patient_manifest["split"] == split, "patient_id"])
        candidate_ids = set(candidate_audit.loc[candidate_audit["split"] == split, "patient_id"])
        excel_missing_manifest = excel_ids[split] - manifest_ids
        manifest_extras = manifest_ids - excel_ids[split]
        excel_missing_candidate = excel_ids[split] - candidate_ids
        candidate_extras = candidate_ids - excel_ids[split]
        lines.append(
            f"| {split} | {len(excel_ids[split])} | {len(manifest_ids)} | {len(candidate_ids)} | "
            f"{len(excel_missing_manifest)} | {len(manifest_extras)} | "
            f"{len(excel_missing_candidate)} | {len(candidate_extras)} |"
        )
        details.extend(
            [
                f"### {split}",
                "",
                f"- Excel missing from patient manifest: {list_text(excel_missing_manifest)}",
                f"- Patient manifest extras: {list_text(manifest_extras)}",
                f"- Excel missing from candidate audit: {list_text(excel_missing_candidate)}",
                f"- Candidate audit extras: {list_text(candidate_extras)}",
                "",
            ]
        )

    overlap = excel_ids["Train"] & excel_ids["Valid"]
    candidate_without_frame_rows = candidate_keys - set(frame_index)
    lines.extend(
        [
            "",
            "## Isolation and candidate-frame linkage",
            "",
            f"- Train/Valid Excel patient_id intersection: **{len(overlap)}** ({list_text(overlap)})",
            f"- Candidate audit rows/keys without frame-audit records: **{len(candidate_without_frame_rows)}**",
            "- A candidate may legitimately have no frame-audit rows when its phase directories were absent or unscanned; the candidate row is still preserved.",
            "",
        ]
    )
    if candidate_without_frame_rows:
        rendered = [
            f"{split}:{patient_id}:rank={rank}:series={series_id}:path={series_path}"
            for split, patient_id, rank, series_id, series_path in sorted(candidate_without_frame_rows)
        ]
        lines.append("Candidate keys without frame rows:")
        lines.append("")
        lines.extend(f"- `{item}`" for item in rendered)
        lines.append("")
    lines.extend(details)
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    train_xlsx = (args.train_xlsx or root / "metadata/Train.xlsx").resolve()
    valid_xlsx = (args.valid_xlsx or root / "metadata/valid.xlsx").resolve()
    patient_manifest_path = (
        args.patient_manifest or root / "manifests/api_fullseq_v2_patient_manifest.csv"
    ).resolve()
    candidate_audit_path = (
        args.candidate_audit or root / "manifests/api_fullseq_v2_candidate_series_audit.csv"
    ).resolve()
    frame_audit_path = (
        args.frame_audit or root / "manifests/api_fullseq_v2_frame_audit.csv"
    ).resolve()

    input_paths = [
        train_xlsx,
        valid_xlsx,
        patient_manifest_path,
        candidate_audit_path,
        frame_audit_path,
    ]
    missing_inputs = [str(path) for path in input_paths if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"Missing required inputs: {missing_inputs}")
    input_hashes = {str(path.relative_to(root)): sha256_file(path) for path in input_paths}

    train_records, _, train_index_column = load_excel_records(train_xlsx, "Train")
    valid_records, _, valid_index_column = load_excel_records(valid_xlsx, "Valid")
    excel_records = {
        "Train": build_metadata_flags(train_records),
        "Valid": build_metadata_flags(valid_records),
    }

    patient_manifest = load_v2_csv(patient_manifest_path, PATIENT_REQUIRED_COLUMNS)
    candidate_audit = load_v2_csv(candidate_audit_path, CANDIDATE_REQUIRED_COLUMNS)
    frame_audit = load_v2_csv(frame_audit_path, FRAME_REQUIRED_COLUMNS)

    if patient_manifest.duplicated(["split", "patient_id"]).any():
        duplicate_rows = patient_manifest.loc[
            patient_manifest.duplicated(["split", "patient_id"], keep=False),
            ["split", "patient_id"],
        ].drop_duplicates()
        raise AssertionError(
            "Patient manifest is not unique by split/patient_id: "
            f"{duplicate_rows.to_dict(orient='records')}"
        )
    if candidate_audit.duplicated(
        ["split", "patient_id", "discovery_rank", "series_id", "series_path"]
    ).any():
        raise AssertionError("Candidate audit keys are not unique")

    frame_index = build_frame_index(frame_audit)
    patient_lookup = {
        (row.split, row.patient_id): row
        for row in patient_manifest.itertuples(index=False)
    }
    candidate_groups = {
        (split, patient_id): group.copy()
        for (split, patient_id), group in candidate_audit.groupby(
            ["split", "patient_id"], sort=False
        )
    }
    candidate_bundles = {
        key: build_candidate_bundle(key[0], key[1], group, frame_index)
        for key, group in candidate_groups.items()
    }

    registries: dict[str, pd.DataFrame] = {}
    private_rows: list[dict[str, Any]] = []
    manual_rows: list[dict[str, Any]] = []

    for split in ("Train", "Valid"):
        source = excel_records[split]
        registry_rows: list[dict[str, Any]] = []
        for _, row in source.iterrows():
            patient_id = row["_patient_id"]
            key = (split, patient_id)
            manifest_row = patient_lookup.get(key)
            bundle_json, candidate_count, valid_candidate_count = candidate_bundles.get(
                key,
                (
                    json.dumps(
                        {
                            "split": split,
                            "patient_id": patient_id,
                            "candidate_count": 0,
                            "valid_candidate_count": 0,
                            "candidates": [],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    0,
                    0,
                ),
            )
            manifest_matched = manifest_row is not None
            selected_series_id = raw_text(manifest_row.selected_series_id) if manifest_row else ""
            has_candidate = candidate_count > 0 or bool(selected_series_id)
            is_multiple = bool(row["_patient_is_multiple"])

            if not manifest_matched and not has_candidate:
                status = "unmatched"
            elif is_multiple:
                status = "review_required"
            elif manifest_matched and selected_series_id:
                status = "provisional_single_lesion"
            elif has_candidate:
                status = "candidate_available"
            elif bool(row["_metadata_ambiguity_flag"]):
                status = "ambiguous"
            else:
                status = "unmatched"

            requires_review = status != "provisional_single_lesion"
            registry_row = {
                "lesion_uid": row["_lesion_uid"],
                "split": split,
                "patient_id": patient_id,
                "source_excel_file": row["_source_excel_file"],
                "source_excel_sheet": row["_source_excel_sheet"],
                "source_excel_row_id": int(row["_source_excel_row_id"]),
                "side_raw": row["_side_raw"],
                "side_normalized": row["_side_normalized"],
                "location_raw": row["_location_raw"],
                "location_normalized": row["_location_normalized"],
                "lesion_index_raw": row["_lesion_index_raw"],
                "lesion_index_normalized": int(row["_lesion_index_normalized"]),
                "lesion_index_source": row["_lesion_index_source"],
                "multiple_aneurysm_raw": row["_multiple_raw"],
                "multiple_aneurysm_normalized": row["_multiple_normalized"],
                "patient_lesion_record_count": int(row["_patient_lesion_record_count"]),
                "patient_registry_multiplicity": (
                    "multiple_lesion" if is_multiple else "single_lesion"
                ),
                "metadata_ambiguity_flag": bool(row["_metadata_ambiguity_flag"]),
                "metadata_ambiguity_reason": row["_metadata_ambiguity_reason"],
                "patient_manifest_matched": manifest_matched,
                "patient_manifest_source_type": (
                    raw_text(manifest_row.source_type) if manifest_row else ""
                ),
                "patient_manifest_source_medical_record_root": (
                    raw_text(manifest_row.source_medical_record_root) if manifest_row else ""
                ),
                "patient_manifest_status": (
                    raw_text(manifest_row.patient_status) if manifest_row else ""
                ),
                "patient_manifest_exclusion_reason": (
                    raw_text(manifest_row.exclusion_reason) if manifest_row else ""
                ),
                "candidate_series_count": candidate_count,
                "valid_candidate_series_count": valid_candidate_count,
                "v2_selected_series_id_candidate": selected_series_id,
                "v2_selected_series_path_candidate": (
                    raw_text(manifest_row.selected_series_path) if manifest_row else ""
                ),
                "v2_selected_candidate_rank": (
                    normalize_integral_text(manifest_row.selected_candidate_rank)
                    if manifest_row
                    else ""
                ),
                "v2_selected_pre_internal_series_candidate": (
                    normalize_integral_text(manifest_row.selected_pre_internal_series)
                    if manifest_row
                    else ""
                ),
                "v2_selected_post_internal_series_candidate": (
                    normalize_integral_text(manifest_row.selected_post_internal_series)
                    if manifest_row
                    else ""
                ),
                "v2_selected_pre_frame_paths_candidate": (
                    raw_text(manifest_row.pre_frame_paths) if manifest_row else ""
                ),
                "v2_selected_post_frame_paths_candidate": (
                    raw_text(manifest_row.post_frame_paths) if manifest_row else ""
                ),
                "candidate_series_registry_json": bundle_json,
                "candidate_available": has_candidate,
                "registration_status": status,
                "requires_manual_review": requires_review,
                "candidate_assignment_scope": "patient_level_candidate_only_pending_lesion_review",
            }
            registry_rows.append(registry_row)
            private_rows.append(build_private_row(row))

            if requires_review:
                review_reasons: list[str] = []
                if is_multiple:
                    review_reasons.append("multi_lesion_patient_requires_image_review")
                if status == "candidate_available":
                    review_reasons.append("candidate_available_without_provisional_selected_series")
                if status == "ambiguous":
                    review_reasons.append("metadata_or_linkage_ambiguous")
                if status == "unmatched":
                    review_reasons.append("patient_or_candidate_linkage_unmatched")
                if row["_metadata_ambiguity_reason"]:
                    review_reasons.append(row["_metadata_ambiguity_reason"])
                manual_rows.append(
                    {
                        "lesion_uid": registry_row["lesion_uid"],
                        "split": split,
                        "patient_id": patient_id,
                        "source_excel_file": registry_row["source_excel_file"],
                        "source_excel_row_id": registry_row["source_excel_row_id"],
                        "side_raw": registry_row["side_raw"],
                        "side_normalized": registry_row["side_normalized"],
                        "location_raw": registry_row["location_raw"],
                        "location_normalized": registry_row["location_normalized"],
                        "lesion_index_raw": registry_row["lesion_index_raw"],
                        "lesion_index_normalized": registry_row["lesion_index_normalized"],
                        "multiple_aneurysm_raw": registry_row["multiple_aneurysm_raw"],
                        "multiple_aneurysm_normalized": registry_row[
                            "multiple_aneurysm_normalized"
                        ],
                        "patient_lesion_record_count": registry_row[
                            "patient_lesion_record_count"
                        ],
                        "metadata_ambiguity_flag": registry_row["metadata_ambiguity_flag"],
                        "metadata_ambiguity_reason": registry_row[
                            "metadata_ambiguity_reason"
                        ],
                        "current_registration_status": status,
                        "manual_review_reason": "|".join(unique_preserving_order(review_reasons)),
                        "candidate_series_count": candidate_count,
                        "valid_candidate_series_count": valid_candidate_count,
                        "candidate_series_registry_json": bundle_json,
                        "reviewer_id": "",
                        "reviewed_at_utc": "",
                        "review_decision": "",
                        "reviewed_series_id": "",
                        "reviewed_pre_internal_series": "",
                        "reviewed_post_internal_series": "",
                        "review_notes": "",
                    }
                )

        registries[split] = pd.DataFrame(registry_rows, columns=BLINDED_COLUMNS)

    private_labels = pd.DataFrame(private_rows, columns=PRIVATE_COLUMNS)
    manual_review = pd.DataFrame(manual_rows, columns=MANUAL_REVIEW_COLUMNS)

    excel_ids = {
        split: set(records["_patient_id"])
        for split, records in excel_records.items()
    }
    patient_overlap = excel_ids["Train"] & excel_ids["Valid"]

    for split in ("Train", "Valid"):
        df = registries[split]
        expected = EXPECTED[split]
        assert len(df) == expected["records"], (split, len(df), expected["records"])
        assert df["patient_id"].nunique() == expected["patients"], (
            split,
            df["patient_id"].nunique(),
            expected["patients"],
        )
        assert df["lesion_uid"].is_unique, f"Duplicate lesion_uid in {split}"
        assert set(df["registration_status"]).issubset(ALLOWED_INITIAL_STATUSES)
        assert not set(df["registration_status"]) & FORBIDDEN_REVIEWED_STATUSES
        assert (
            df.loc[df["patient_registry_multiplicity"] == "multiple_lesion", "registration_status"]
            == "review_required"
        ).all()
        assert (
            df.loc[df["patient_registry_multiplicity"] == "single_lesion", "registration_status"]
            == "provisional_single_lesion"
        ).all()
        assert df["patient_manifest_matched"].all()
        assert df["candidate_available"].all()
        assert not df["candidate_assignment_scope"].str.contains("confirmed", case=False).any()

    assert len(patient_overlap) == 0, f"Train/Valid patient overlap: {sorted(patient_overlap)}"
    assert private_labels["lesion_uid"].is_unique
    assert len(private_labels) == EXPECTED["Train"]["records"] + EXPECTED["Valid"]["records"]
    assert set(manual_review["current_registration_status"]).issubset(
        ALLOWED_INITIAL_STATUSES
    )
    assert not set(manual_review["current_registration_status"]) & FORBIDDEN_REVIEWED_STATUSES
    assert manual_review["review_decision"].eq("").all()
    assert manual_review["reviewed_series_id"].eq("").all()
    assert manual_review["reviewed_pre_internal_series"].eq("").all()
    assert manual_review["reviewed_post_internal_series"].eq("").all()

    validate_blinded_columns(registries["Train"].columns, "train blinded registry")
    validate_blinded_columns(registries["Valid"].columns, "valid blinded registry")
    validate_blinded_columns(manual_review.columns, "manual review")

    manifest_ids = {
        split: set(patient_manifest.loc[patient_manifest["split"] == split, "patient_id"])
        for split in ("Train", "Valid")
    }
    candidate_ids = {
        split: set(candidate_audit.loc[candidate_audit["split"] == split, "patient_id"])
        for split in ("Train", "Valid")
    }
    for split in ("Train", "Valid"):
        assert excel_ids[split] == manifest_ids[split], f"Excel/manifest mismatch in {split}"
        assert excel_ids[split] == candidate_ids[split], f"Excel/candidate mismatch in {split}"

    output_metadata = root / "metadata/api_fullseq_v3"
    output_reports = root / "reports/api_fullseq_v3"
    output_paths = {
        "train_registry": output_metadata / "lesion_registry_train_blinded.csv",
        "valid_registry": output_metadata / "lesion_registry_valid_blinded.csv",
        "private_labels": output_metadata / "lesion_outcome_labels_private.csv",
        "registry_audit": output_reports / "lesion_registry_audit.md",
        "manual_review": output_reports / "lesion_series_manual_review.csv",
        "alignment_audit": output_reports / "split_id_alignment_audit.md",
    }

    candidate_keys = {
        candidate_key(row) for row in candidate_audit.itertuples(index=False)
    }
    registry_audit = build_registry_audit(
        registries,
        manual_review,
        private_labels,
        patient_overlap,
        input_hashes,
        {"Train": train_index_column, "Valid": valid_index_column},
    )
    alignment_audit = build_alignment_audit(
        excel_ids,
        patient_manifest,
        candidate_audit,
        frame_index,
        candidate_keys,
    )

    atomic_write_csv(registries["Train"], output_paths["train_registry"])
    atomic_write_csv(registries["Valid"], output_paths["valid_registry"])
    atomic_write_csv(private_labels, output_paths["private_labels"])
    atomic_write_csv(manual_review, output_paths["manual_review"])
    atomic_write_text(registry_audit, output_paths["registry_audit"])
    atomic_write_text(alignment_audit, output_paths["alignment_audit"])

    summary = {
        "phase": "api_fullseq_v3_phase_1_lesion_series_registry",
        "train_records": len(registries["Train"]),
        "train_patients": registries["Train"]["patient_id"].nunique(),
        "valid_records": len(registries["Valid"]),
        "valid_patients": registries["Valid"]["patient_id"].nunique(),
        "train_valid_patient_overlap": len(patient_overlap),
        "manual_review_rows": len(manual_review),
        "patient_manifest_unmatched_rows": sum(
            int((~df["patient_manifest_matched"]).sum()) for df in registries.values()
        ),
        "outputs": {name: str(path) for name, path in output_paths.items()},
        "stopped_before_downstream_work": True,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

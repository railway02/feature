#!/usr/bin/env python3
"""Build the formal strict-Pre+Post SERIES-level adverse-outcome task (V3).

Scientific contract
-------------------
- Prediction unit: one unique `series_uid`.
- Mapping/audit unit: `record_uid`.
- Leakage-control grouping unit: `patient_id`.
- Mapping: only a finalized record_uid -> series_uid mapping is accepted.
- Image eligibility: BOTH Local-CAVE Pre and Post embeddings must be complete.
- Same series + same label: collapse to one training sample.
- Same series + conflicting labels: exclude only that series.
- Different series from one patient may have different labels and remain valid.
- Scalar schema: selected from Train columns and Train values only. Valid never
  determines which scalar features exist or are retained.
- Official Valid never contributes to fold construction or preprocessing.

The output contains a complete inclusion/exclusion audit, fixed patient-grouped
outer folds, and NPZ files consumed by the formal model trainer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


SEED = 20260804
N_FOLDS = 5
LABEL_COLUMN_PREFIX = "不良转归"
PATIENT_COLUMN = "病案号"
EXPECTED_SOURCE_ROWS = {"Train": 1157, "Valid": 289}

IDENTITY_SCALAR_COLUMNS = {
    "patient_id",
    "series_uid",
    "series_id",
    "source_type",
    "split",
    "missing_pre",
    "missing_post",
    "has_pre",
    "has_post",
    "local_pre_available",
    "local_post_available",
    "local_both_available",
    "local_any_available",
}


def normalize_patient_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = re.sub(r"\.0$", "", str(value).strip())
    if not text:
        return ""
    if not text.isdigit():
        raise ValueError(f"Invalid patient_id: {value!r}")
    return str(int(text))


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return clean_text(value).casefold() in {"1", "true", "yes", "y"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_lines(values: list[str]) -> str:
    material = "\n".join(values).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(
        temporary,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )
    os.replace(temporary, path)


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite non-empty output directory: {path}"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_excel_records(path: Path, split: str) -> pd.DataFrame:
    frame = pd.read_excel(path, dtype=object).copy()
    if PATIENT_COLUMN not in frame.columns:
        raise KeyError(f"{path}: missing {PATIENT_COLUMN}")
    label_column = next(
        (
            column
            for column in frame.columns
            if str(column).startswith(LABEL_COLUMN_PREFIX)
        ),
        None,
    )
    if label_column is None:
        raise KeyError(f"{path}: cannot find adverse label column")

    frame.insert(0, "excel_row_number", range(2, len(frame) + 2))
    frame.insert(1, "split", split)
    frame.insert(2, "patient_id", frame[PATIENT_COLUMN].map(normalize_patient_id))
    frame.insert(
        3,
        "record_uid",
        [
            f"{split}:{patient_id}:excel_row_{row_number:06d}"
            for patient_id, row_number in zip(
                frame["patient_id"],
                frame["excel_row_number"],
            )
        ],
    )
    frame["target"] = pd.to_numeric(frame[label_column], errors="coerce")
    if frame["record_uid"].duplicated().any():
        raise AssertionError(f"{path}: duplicate record_uid")
    if (frame["patient_id"] == "").any():
        raise AssertionError(f"{path}: empty patient_id")

    frame["records_for_patient_source"] = (
        frame.groupby("patient_id")["record_uid"].transform("count").astype(int)
    )
    return frame


def discover_mapping_dir(project: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        root = explicit.resolve()
        required = (
            root / "train_record_series_map.csv",
            root / "valid_record_series_map.csv",
        )
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Explicit mapping directory is incomplete:\n"
                + "\n".join(str(path) for path in missing)
            )
        return root

    candidates: list[Path] = []
    for train_path in (project / "manifests").rglob(
        "train_record_series_map.csv"
    ):
        candidate = train_path.parent.resolve()
        if (candidate / "valid_record_series_map.csv").is_file():
            candidates.append(candidate)
    candidates = sorted(set(candidates), key=str)
    if len(candidates) != 1:
        raise RuntimeError(
            "Could not uniquely resolve finalized mapping directory. "
            "Set --mapping-dir explicitly.\nCandidates:\n"
            + "\n".join(f"  - {path}" for path in candidates)
        )
    return candidates[0]


def mapping_series_column(frame: pd.DataFrame) -> str:
    for column in ("series_uid", "suggested_series_uid"):
        if column in frame.columns:
            return column
    raise KeyError("Mapping table has no series_uid/suggested_series_uid")


def mapping_acceptance(frame: pd.DataFrame, series_column: str) -> pd.Series:
    nonempty = frame[series_column].fillna("").astype(str).str.strip().ne("")
    if "mapping_accepted" in frame.columns:
        return nonempty & frame["mapping_accepted"].map(as_bool)

    status = (
        frame.get("mapping_status", pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
        .str.casefold()
    )
    confidence = (
        frame.get("mapping_confidence", pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
        .str.casefold()
    )
    accepted_status = status.isin(
        {
            "matched",
            "shared_series",
            "resolved",
            "accepted",
            "auto_resolved",
            "manual_resolved",
            "rule_resolved",
        }
    )
    rejected_confidence = confidence.isin(
        {"low", "unavailable", "rejected"}
    )
    reviewed = (
        frame["reviewed"].map(as_bool)
        if "reviewed" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    return nonempty & ((accepted_status & ~rejected_confidence) | reviewed)


def load_mapping(path: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype={"patient_id": str, "record_uid": str},
        keep_default_na=False,
    )
    required = {"record_uid", "patient_id", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"{path}: missing columns {sorted(missing)}")
    if frame["record_uid"].duplicated().any():
        raise AssertionError(f"{path}: duplicate record_uid")
    if not frame["split"].astype(str).eq(split).all():
        raise AssertionError(f"{path}: split mismatch")

    series_column = mapping_series_column(frame)
    result = frame.copy()
    result["patient_id"] = result["patient_id"].map(normalize_patient_id)
    result["mapped_series_uid"] = result[series_column].map(clean_text)
    result["mapping_accepted_final"] = mapping_acceptance(
        result,
        series_column,
    )
    keep = [
        "record_uid",
        "patient_id",
        "split",
        "mapped_series_uid",
        "mapping_accepted_final",
    ]
    for column in (
        "series_id",
        "suggested_series_id",
        "mapping_status",
        "mapping_confidence",
        "mapping_source",
        "mapping_reason",
        "reviewed",
        "manual_review_required",
        "missing_image",
        "shared_series",
    ):
        if column in result.columns:
            keep.append(column)
    return result[keep].copy()


def resolve_embedding_key(payload: np.lib.npyio.NpzFile) -> str:
    for key in ("embeddings", "embedding"):
        if key in payload.files:
            return key
    raise KeyError("series_embeddings_5120.npz has no embeddings array")


def infer_patient_from_series_uid(series_uid: str) -> str:
    parts = str(series_uid).split("__")
    return normalize_patient_id(parts[1]) if len(parts) >= 2 else ""


def load_scalar_frame(path: Path) -> pd.DataFrame:
    csv_path = path / "series_scalar_features.csv"
    parquet_path = path / "series_scalar_features.parquet"
    if csv_path.is_file():
        frame = pd.read_csv(
            csv_path,
            dtype={"patient_id": str, "series_uid": str},
            keep_default_na=False,
        )
        frame.attrs["source_path"] = str(csv_path)
        return frame
    if parquet_path.is_file():
        frame = pd.read_parquet(parquet_path)
        frame.attrs["source_path"] = str(parquet_path)
        return frame
    raise FileNotFoundError(f"No series scalar table under {path}")


def load_series_table(table_dir: Path, split: str) -> dict[str, Any]:
    npz_path = table_dir / "series_embeddings_5120.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)

    with np.load(npz_path, allow_pickle=False) as raw:
        if "series_uid" not in raw.files:
            raise KeyError(f"{npz_path}: missing series_uid")
        series_uid = raw["series_uid"].astype(str)
        embeddings = np.asarray(
            raw[resolve_embedding_key(raw)],
            dtype=np.float32,
        )
        patient_id = (
            raw["patient_id"].astype(str)
            if "patient_id" in raw.files
            else np.asarray(
                [infer_patient_from_series_uid(uid) for uid in series_uid],
                dtype=str,
            )
        )

    if embeddings.ndim != 3 or embeddings.shape[1:] != (2, 5120):
        raise AssertionError(
            f"{npz_path}: expected [N,2,5120], found {embeddings.shape}"
        )
    if len(series_uid) != len(embeddings):
        raise AssertionError(f"{npz_path}: UID/embedding length mismatch")
    if len(set(series_uid.tolist())) != len(series_uid):
        raise AssertionError(f"{npz_path}: duplicate series_uid")
    patient_id = np.asarray(
        [normalize_patient_id(value) for value in patient_id],
        dtype=str,
    )

    scalar = load_scalar_frame(table_dir)
    scalar_path = Path(str(scalar.attrs["source_path"]))
    if "series_uid" not in scalar.columns:
        raise KeyError(f"{scalar_path}: missing series_uid")
    scalar["series_uid"] = scalar["series_uid"].astype(str)
    if scalar["series_uid"].duplicated().any():
        raise AssertionError(f"{scalar_path}: duplicate series_uid")
    scalar_by_uid = scalar.set_index("series_uid", drop=False)
    missing_scalar = sorted(set(series_uid) - set(scalar_by_uid.index))
    if missing_scalar:
        raise AssertionError(
            f"{scalar_path}: missing {len(missing_scalar)} series rows"
        )
    scalar = scalar_by_uid.loc[series_uid.tolist()].reset_index(drop=True)

    pre_finite = np.isfinite(embeddings[:, 0, :]).all(axis=1)
    post_finite = np.isfinite(embeddings[:, 1, :]).all(axis=1)
    availability_path = (
        table_dir.parent
        / f"local_series_availability_{split.casefold()}.csv"
    )
    availability_both = np.ones(len(series_uid), dtype=bool)
    if availability_path.is_file():
        availability = pd.read_csv(
            availability_path,
            dtype={"patient_id": str, "series_uid": str},
            keep_default_na=False,
        )
        if "series_uid" not in availability.columns:
            raise KeyError(f"{availability_path}: missing series_uid")
        if availability["series_uid"].duplicated().any():
            raise AssertionError(f"{availability_path}: duplicate series_uid")
        availability_by_uid = availability.set_index("series_uid")
        if "local_both_available" not in availability_by_uid.columns:
            raise KeyError(
                f"{availability_path}: missing local_both_available"
            )
        availability_both = np.asarray(
            [
                as_bool(
                    availability_by_uid.at[uid, "local_both_available"]
                    if uid in availability_by_uid.index
                    else False
                )
                for uid in series_uid
            ],
            dtype=bool,
        )

    return {
        "split": split,
        "npz_path": npz_path,
        "npz_sha256": sha256_file(npz_path),
        "scalar_path": scalar_path,
        "scalar_sha256": sha256_file(scalar_path),
        "availability_path": (
            str(availability_path) if availability_path.is_file() else ""
        ),
        "series_uid": series_uid,
        "patient_id": patient_id,
        "embeddings": embeddings,
        "scalar_frame": scalar,
        "pre_finite": pre_finite,
        "post_finite": post_finite,
        "both_available": pre_finite & post_finite & availability_both,
    }


def train_only_scalar_schema(
    train_store: dict[str, Any],
    valid_store: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    train_frame = train_store["scalar_frame"]
    valid_frame = valid_store["scalar_frame"]
    selected: list[str] = []
    train_arrays: list[np.ndarray] = []

    for column in train_frame.columns:
        if str(column) in IDENTITY_SCALAR_COLUMNS:
            continue
        values = pd.to_numeric(
            train_frame[column],
            errors="coerce",
        ).to_numpy(np.float64)
        if np.isinf(values).any():
            raise AssertionError(
                f"{train_store['scalar_path']}: infinity in {column}"
            )
        if np.isfinite(values).any():
            selected.append(str(column))
            train_arrays.append(values.astype(np.float32))

    if not selected:
        raise RuntimeError("Train-only scalar schema is empty")

    valid_arrays: list[np.ndarray] = []
    missing_valid_columns: list[str] = []
    for column in selected:
        if column not in valid_frame.columns:
            values = np.full(len(valid_frame), np.nan, dtype=np.float32)
            missing_valid_columns.append(column)
        else:
            numeric = pd.to_numeric(
                valid_frame[column],
                errors="coerce",
            ).to_numpy(np.float64)
            if np.isinf(numeric).any():
                raise AssertionError(
                    f"{valid_store['scalar_path']}: infinity in {column}"
                )
            values = numeric.astype(np.float32)
        valid_arrays.append(values)

    valid_extra = [
        str(column)
        for column in valid_frame.columns
        if str(column) not in set(selected)
        and str(column) not in IDENTITY_SCALAR_COLUMNS
    ]
    train_matrix = np.column_stack(train_arrays).astype(np.float32)
    valid_matrix = np.column_stack(valid_arrays).astype(np.float32)
    audit = {
        "policy": "feature names selected from Train table and Train values only",
        "selected_feature_count": len(selected),
        "selected_feature_hash": hash_lines(selected),
        "valid_missing_selected_columns": missing_valid_columns,
        "valid_extra_columns_not_used": valid_extra,
        "valid_used_to_select_schema": False,
    }
    return train_matrix, valid_matrix, selected, audit


def assign_record_exclusion_reason(frame: pd.DataFrame) -> pd.Series:
    """Assign reasons that can be decided before grouping by series_uid."""
    reason = pd.Series("", index=frame.index, dtype=object)

    invalid_label = ~frame["target"].isin([0, 1])
    reason.loc[invalid_label] = "invalid_or_missing_adverse_label"

    not_mapped = ~frame["mapping_accepted_final"].fillna(False).map(as_bool)
    reason.loc[(reason == "") & not_mapped] = (
        "record_to_series_mapping_not_accepted"
    )

    no_series = frame["series_row"].isna()
    reason.loc[(reason == "") & no_series] = (
        "mapped_series_not_in_local_table"
    )

    patient_mismatch = (
        frame["series_patient_id"].fillna("").astype(str)
        != frame["patient_id"].fillna("").astype(str)
    )
    reason.loc[
        (reason == "") & ~no_series & patient_mismatch
    ] = "mapped_series_patient_mismatch"

    pre_missing = ~frame["pre_finite"].map(as_bool)
    post_missing = ~frame["post_finite"].map(as_bool)
    reason.loc[
        (reason == "") & ~no_series & (pre_missing ^ post_missing)
    ] = "strict_prepost_exclusion_only_one_phase_available"
    reason.loc[
        (reason == "") & ~no_series & pre_missing & post_missing
    ] = "strict_prepost_exclusion_no_phase_available"

    not_both = ~frame["both_available"].map(as_bool)
    reason.loc[
        (reason == "") & ~no_series & not_both
    ] = "strict_prepost_exclusion_availability_not_both"
    return reason


def build_split(
    records: pd.DataFrame,
    mapping: pd.DataFrame,
    store: dict[str, Any],
    scalar_matrix: np.ndarray,
    scalar_columns: list[str],
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create one model row per unique Local-CAVE series."""
    split = str(store["split"])
    series_index = pd.DataFrame(
        {
            "mapped_series_uid": store["series_uid"],
            "series_row": np.arange(len(store["series_uid"]), dtype=int),
            "series_patient_id": store["patient_id"],
            "pre_finite": store["pre_finite"],
            "post_finite": store["post_finite"],
            "both_available": store["both_available"],
        }
    )

    merged = records.merge(
        mapping,
        on=["record_uid", "patient_id", "split"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_mapping"),
    )
    merged["mapping_accepted_final"] = (
        merged["mapping_accepted_final"].fillna(False).map(as_bool)
    )
    merged["mapped_series_uid"] = (
        merged["mapped_series_uid"].fillna("").astype(str)
    )
    merged = merged.merge(
        series_index,
        on="mapped_series_uid",
        how="left",
        validate="many_to_one",
    )
    merged["record_exclusion_reason"] = assign_record_exclusion_reason(merged)
    merged["record_candidate_for_series_task"] = (
        merged["record_exclusion_reason"].eq("")
    )

    candidate = merged[merged["record_candidate_for_series_task"]].copy()
    candidate["target"] = candidate["target"].astype(int)
    candidate["series_row"] = candidate["series_row"].astype(int)

    series_rows: list[dict[str, Any]] = []
    for series_uid, group in candidate.groupby(
        "mapped_series_uid",
        sort=False,
        dropna=False,
    ):
        patient_ids = sorted(set(group["patient_id"].astype(str)))
        series_rows_in_table = sorted(
            set(group["series_row"].astype(int).tolist())
        )
        labels = sorted(set(group["target"].astype(int).tolist()))
        record_uids = group["record_uid"].astype(str).tolist()
        excel_rows = group["excel_row_number"].astype(int).tolist()

        if len(patient_ids) != 1:
            status = "excluded"
            exclusion_reason = "series_maps_to_multiple_patients"
        elif len(series_rows_in_table) != 1:
            status = "excluded"
            exclusion_reason = "series_maps_to_multiple_feature_rows"
        elif len(labels) != 1:
            status = "excluded"
            exclusion_reason = "series_adverse_label_conflict"
        else:
            status = "included"
            exclusion_reason = ""

        series_rows.append(
            {
                "split": split,
                "series_uid": str(series_uid),
                "patient_id": patient_ids[0] if len(patient_ids) == 1 else "|".join(patient_ids),
                "series_row": (
                    series_rows_in_table[0]
                    if len(series_rows_in_table) == 1
                    else -1
                ),
                "target": labels[0] if len(labels) == 1 else np.nan,
                "source_record_count": int(len(group)),
                "source_record_uids": "|".join(record_uids),
                "source_excel_rows": "|".join(str(value) for value in excel_rows),
                "duplicate_same_label_records_collapsed": int(
                    len(group) - 1 if len(labels) == 1 else 0
                ),
                "series_status": status,
                "series_exclusion_reason": exclusion_reason,
            }
        )

    series_audit = pd.DataFrame(series_rows)
    if series_audit.empty:
        raise RuntimeError(f"{split}: no record candidates reached series grouping")

    # Propagate series-level decisions back to every candidate record.
    decision = series_audit[
        ["series_uid", "series_status", "series_exclusion_reason"]
    ].rename(columns={"series_uid": "mapped_series_uid"})
    merged = merged.merge(
        decision,
        on="mapped_series_uid",
        how="left",
        validate="many_to_one",
    )
    merged["series_status"] = merged["series_status"].fillna("")
    merged["series_exclusion_reason"] = (
        merged["series_exclusion_reason"].fillna("")
    )
    merged["included_in_series_task"] = (
        merged["record_candidate_for_series_task"]
        & merged["series_status"].eq("included")
    )
    merged["final_exclusion_reason"] = merged["record_exclusion_reason"]
    needs_series_reason = (
        merged["final_exclusion_reason"].eq("")
        & ~merged["included_in_series_task"]
    )
    merged.loc[needs_series_reason, "final_exclusion_reason"] = (
        merged.loc[needs_series_reason, "series_exclusion_reason"]
    )

    accepted = series_audit[
        series_audit["series_status"].eq("included")
    ].copy()
    accepted["target"] = accepted["target"].astype(int)
    accepted["series_row"] = accepted["series_row"].astype(int)
    if accepted.empty:
        raise RuntimeError(f"{split}: no series survived strict cohort rules")
    if accepted["series_uid"].duplicated().any():
        raise AssertionError(f"{split}: duplicate series_uid after collapse")
    if accepted["series_row"].duplicated().any():
        raise AssertionError(f"{split}: duplicate feature row after collapse")

    rows = accepted["series_row"].to_numpy(dtype=int)
    deep = store["embeddings"][rows].reshape(len(rows), 10240).astype(np.float32)
    scalar = scalar_matrix[rows].astype(np.float32)
    if not np.isfinite(deep).all():
        raise AssertionError(f"{split}: nonfinite strict Pre+Post deep matrix")
    if scalar.shape[1] != len(scalar_columns):
        raise AssertionError(f"{split}: scalar schema mismatch")

    accepted["series_for_patient_included"] = (
        accepted.groupby("patient_id")["series_uid"]
        .transform("count")
        .astype(int)
    )
    accepted["patient_has_mixed_series_labels"] = (
        accepted.groupby("patient_id")["target"]
        .transform("nunique")
        .gt(1)
        .astype(int)
    )

    metadata_columns = [
        "series_uid",
        "split",
        "patient_id",
        "target",
        "source_record_count",
        "source_record_uids",
        "source_excel_rows",
        "duplicate_same_label_records_collapsed",
        "series_for_patient_included",
        "patient_has_mixed_series_labels",
    ]
    metadata = accepted[metadata_columns].copy()

    record_audit_path = (
        output_dir / f"{split.casefold()}_record_to_series_inclusion_audit.csv"
    )
    series_audit_path = (
        output_dir / f"{split.casefold()}_series_inclusion_audit.csv"
    )
    metadata_path = output_dir / f"{split.casefold()}_series_samples.csv"
    npz_path = output_dir / f"{split.casefold()}_features.npz"

    atomic_csv(merged, record_audit_path)
    atomic_csv(series_audit, series_audit_path)
    atomic_csv(metadata, metadata_path)
    atomic_npz(
        npz_path,
        deep=deep,
        scalar=scalar,
        target=accepted["target"].to_numpy(np.int64),
        series_uid=np.asarray(
            accepted["series_uid"].astype(str).tolist(),
            dtype=str,
        ),
        patient_id=np.asarray(
            accepted["patient_id"].astype(str).tolist(),
            dtype=str,
        ),
        source_record_count=accepted[
            "source_record_count"
        ].to_numpy(np.int64),
        scalar_feature_names=np.asarray(scalar_columns, dtype=str),
    )

    record_exclusion_counts = (
        merged.loc[
            ~merged["record_candidate_for_series_task"],
            "record_exclusion_reason",
        ]
        .value_counts(dropna=False)
        .to_dict()
    )
    series_exclusion_counts = (
        series_audit.loc[
            series_audit["series_status"].eq("excluded"),
            "series_exclusion_reason",
        ]
        .value_counts(dropna=False)
        .to_dict()
    )
    summary = {
        "split": split,
        "source_excel_records": int(len(records)),
        "source_patients": int(records["patient_id"].nunique()),
        "source_positive_records": int((records["target"] == 1).sum()),
        "record_candidates_after_mapping_and_phase_qc": int(len(candidate)),
        "candidate_unique_series": int(
            candidate["mapped_series_uid"].nunique()
        ),
        "included_series": int(len(accepted)),
        "included_patients": int(accepted["patient_id"].nunique()),
        "positive_series": int((accepted["target"] == 1).sum()),
        "negative_series": int((accepted["target"] == 0).sum()),
        "series_with_mixed_patient_level_labels_preserved": int(
            accepted.loc[
                accepted["patient_has_mixed_series_labels"].eq(1),
                "series_uid",
            ].nunique()
        ),
        "patients_with_mixed_series_labels_preserved": int(
            accepted.loc[
                accepted["patient_has_mixed_series_labels"].eq(1),
                "patient_id",
            ].nunique()
        ),
        "duplicate_same_label_records_collapsed": int(
            accepted["duplicate_same_label_records_collapsed"].sum()
        ),
        "excluded_conflicting_series": int(
            series_audit["series_exclusion_reason"]
            .eq("series_adverse_label_conflict")
            .sum()
        ),
        "deep_shape": list(deep.shape),
        "scalar_shape": list(scalar.shape),
        "prediction_unit": "series_uid",
        "grouping_unit": "patient_id",
        "strict_prepost_only": True,
        "record_exclusion_reasons": {
            str(key): int(value)
            for key, value in record_exclusion_counts.items()
        },
        "series_exclusion_reasons": {
            str(key): int(value)
            for key, value in series_exclusion_counts.items()
        },
        "record_audit_path": str(record_audit_path),
        "series_audit_path": str(series_audit_path),
        "samples_path": str(metadata_path),
        "npz_path": str(npz_path),
    }
    return metadata, summary


def balanced_grouped_folds(
    metadata: pd.DataFrame,
    n_splits: int,
    seed: int,
    retries: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    y = metadata["target"].to_numpy(np.int64)
    groups = metadata["patient_id"].astype(str).to_numpy()
    if len(np.unique(y)) != 2:
        raise AssertionError("Train cohort must contain both classes")
    # One patient may legitimately own distinct series with different labels.
    # patient_id is a leakage-control group, not the prediction/label unit.
    global_rate = float(np.mean(y))
    target_rows = len(y) / n_splits
    best: tuple[float, np.ndarray, list[dict[str, Any]], int] | None = None

    for offset in range(retries):
        current_seed = seed + offset
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=current_seed,
        )
        fold_assignment = np.zeros(len(metadata), dtype=np.int64)
        rows: list[dict[str, Any]] = []
        valid_candidate = True
        for fold_number, (development, holdout) in enumerate(
            splitter.split(np.zeros(len(y)), y, groups),
            start=1,
        ):
            if (
                len(np.unique(y[development])) != 2
                or len(np.unique(y[holdout])) != 2
            ):
                valid_candidate = False
                break
            fold_assignment[holdout] = fold_number
            rows.append(
                {
                    "fold": fold_number,
                    "records": int(len(holdout)),
                    "patients": int(len(np.unique(groups[holdout]))),
                    "positive": int(y[holdout].sum()),
                    "negative": int((y[holdout] == 0).sum()),
                    "positive_fraction": float(np.mean(y[holdout])),
                }
            )
        if not valid_candidate or (fold_assignment == 0).any():
            continue
        audit = pd.DataFrame(rows)
        score = float(
            np.max(np.abs(audit["records"] - target_rows)) / max(target_rows, 1)
            + 5.0
            * np.max(
                np.abs(audit["positive_fraction"] - global_rate)
            )
        )
        candidate = (score, fold_assignment, rows, current_seed)
        if best is None or candidate[0] < best[0]:
            best = candidate

    if best is None:
        raise RuntimeError("Unable to construct valid patient-grouped 5 folds")
    _, fold_assignment, rows, selected_seed = best
    output = metadata.copy()
    output["fold"] = fold_assignment
    if int(output.groupby("patient_id")["fold"].nunique().max()) != 1:
        raise AssertionError("Patient leakage across folds")
    audit = pd.DataFrame(rows)
    audit["selected_seed"] = selected_seed
    return output, audit, selected_seed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("/root/autodl-tmp/aneurysm"),
    )
    parser.add_argument("--train-xlsx", type=Path)
    parser.add_argument("--valid-xlsx", type=Path)
    parser.add_argument("--mapping-dir", type=Path)
    parser.add_argument("--table-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-source-count-change",
        action="store_true",
    )
    args = parser.parse_args()

    project = args.project.resolve()
    train_xlsx = (
        args.train_xlsx.resolve()
        if args.train_xlsx
        else project / "metadata/Train.xlsx"
    )
    valid_xlsx = (
        args.valid_xlsx.resolve()
        if args.valid_xlsx
        else project / "metadata/valid.xlsx"
    )
    table_root = (
        args.table_root.resolve()
        if args.table_root
        else project
        / "outputs/api_gtmask_roi_cave_v5_fullmask_fullseq/tables/local_eligible"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else project
        / "outputs/api_gtmask_roi_cave_v5_fullmask_fullseq/"
        "adverse_prepost_series_task_v3"
    )
    prepare_output(output_dir, args.overwrite)

    for path in (train_xlsx, valid_xlsx):
        if not path.is_file():
            raise FileNotFoundError(path)
    mapping_dir = discover_mapping_dir(project, args.mapping_dir)

    train_records = read_excel_records(train_xlsx, "Train")
    valid_records = read_excel_records(valid_xlsx, "Valid")
    if not args.allow_source_count_change:
        for split, frame in (
            ("Train", train_records),
            ("Valid", valid_records),
        ):
            expected = EXPECTED_SOURCE_ROWS[split]
            if len(frame) != expected:
                raise AssertionError(
                    f"{split} source rows changed: {len(frame)} != {expected}"
                )
    overlap = set(train_records["patient_id"]) & set(
        valid_records["patient_id"]
    )
    if overlap:
        raise AssertionError(
            f"Train/Valid patient overlap: {sorted(overlap)[:10]}"
        )

    train_mapping_path = mapping_dir / "train_record_series_map.csv"
    valid_mapping_path = mapping_dir / "valid_record_series_map.csv"
    train_mapping = load_mapping(train_mapping_path, "Train")
    valid_mapping = load_mapping(valid_mapping_path, "Valid")
    if set(train_records["record_uid"]) != set(train_mapping["record_uid"]):
        raise AssertionError(
            "Train Excel and finalized mapping record_uid sets differ"
        )
    if set(valid_records["record_uid"]) != set(valid_mapping["record_uid"]):
        raise AssertionError(
            "Valid Excel and finalized mapping record_uid sets differ"
        )

    train_store = load_series_table(table_root / "train", "Train")
    valid_store = load_series_table(table_root / "valid", "Valid")
    (
        train_scalar,
        valid_scalar,
        scalar_columns,
        scalar_schema_audit,
    ) = train_only_scalar_schema(train_store, valid_store)
    atomic_json(
        {
            **scalar_schema_audit,
            "feature_names": scalar_columns,
        },
        output_dir / "scalar_schema_train_only.json",
    )

    train_meta, train_summary = build_split(
        train_records,
        train_mapping,
        train_store,
        train_scalar,
        scalar_columns,
        output_dir,
    )
    valid_meta, valid_summary = build_split(
        valid_records,
        valid_mapping,
        valid_store,
        valid_scalar,
        scalar_columns,
        output_dir,
    )

    fold_table, fold_audit, fold_seed = balanced_grouped_folds(
        train_meta,
        N_FOLDS,
        SEED,
    )
    atomic_csv(fold_table, output_dir / "train_grouped_folds.csv")
    atomic_csv(fold_audit, output_dir / "train_fold_balance_audit.csv")

    train_npz_path = output_dir / "train_features.npz"
    with np.load(train_npz_path, allow_pickle=False) as raw:
        arrays = {name: np.asarray(raw[name]) for name in raw.files}
    if not np.array_equal(
        arrays["series_uid"].astype(str),
        fold_table["series_uid"].astype(str).to_numpy(),
    ):
        raise AssertionError("Fold table order differs from Train NPZ")
    arrays["fold"] = fold_table["fold"].to_numpy(np.int64)
    atomic_npz(train_npz_path, **arrays)

    summary = {
        "status": "success",
        "version": "formal_adverse_prepost_local_cave_series_v3",
        "task": "adverse_outcome_series",
        "prediction_unit": "series_uid",
        "grouping_unit": "patient_id",
        "strict_phase_policy": "both Pre and Post required",
        "conflict_policy": (
            "collapse same-series same-label records; exclude only same-series label conflicts"
        ),
        "scalar_schema_policy": "Train-only; Valid cannot select features",
        "label_column_prefix": LABEL_COLUMN_PREFIX,
        "train": train_summary,
        "valid": valid_summary,
        "outer_folds": N_FOLDS,
        "outer_fold_seed_selected": fold_seed,
        "base_seed": SEED,
        "inputs": {
            "train_xlsx": str(train_xlsx),
            "train_xlsx_sha256": sha256_file(train_xlsx),
            "valid_xlsx": str(valid_xlsx),
            "valid_xlsx_sha256": sha256_file(valid_xlsx),
            "mapping_dir": str(mapping_dir),
            "train_mapping": str(train_mapping_path),
            "train_mapping_sha256": sha256_file(train_mapping_path),
            "valid_mapping": str(valid_mapping_path),
            "valid_mapping_sha256": sha256_file(valid_mapping_path),
            "table_root": str(table_root),
            "train_embedding_npz_sha256": train_store["npz_sha256"],
            "valid_embedding_npz_sha256": valid_store["npz_sha256"],
            "train_scalar_sha256": train_store["scalar_sha256"],
            "valid_scalar_sha256": valid_store["scalar_sha256"],
        },
        "features": {
            "deep": "Pre 5120 concatenated with Post 5120",
            "deep_dimension": 10240,
            "scalar_dimension": len(scalar_columns),
            "scalar_feature_hash": hash_lines(scalar_columns),
        },
        "valid_used_for_schema_preprocessing_or_selection": False,
        "different_series_from_same_patient_may_have_different_labels": True,
    }
    atomic_json(summary, output_dir / "task_summary.json")
    atomic_json(summary, output_dir / ".TASK_SUCCESS.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build the strict Pre+Post record-level adverse-outcome task.

This script is intentionally label-blind while matching records to image
features: it uses an existing finalized record_uid -> series_uid mapping, then
reads the adverse label only after the mapping is fixed.

Inclusion policy
----------------
A record is included only when:
1. its adverse label is binary (0/1);
2. its finalized record-to-series mapping is accepted;
3. the mapped Local-CAVE series exists;
4. BOTH Pre and Post embeddings are present and finite;
5. local_series_availability marks both phases available, when that audit file
   is present.

Records with only Pre or only Post are explicitly excluded and written to the
audit table. Multiple records from one patient are preserved; downstream folds
are grouped by patient_id.
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

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("scikit-learn with StratifiedGroupKFold is required") from exc


SEED = 20260803
N_FOLDS = 5
LABEL_COLUMN_PREFIX = "不良转归"
PATIENT_COLUMN = "病案号"


def normalize_patient_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    text = re.sub(r"\.0$", "", text)
    if not text:
        return ""
    if text.isdigit():
        return str(int(text))
    raise ValueError(f"Invalid patient_id: {value!r}")


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
        (column for column in frame.columns if str(column).startswith(LABEL_COLUMN_PREFIX)),
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
                frame["patient_id"], frame["excel_row_number"]
            )
        ],
    )
    frame["target"] = pd.to_numeric(frame[label_column], errors="coerce")
    if frame["record_uid"].duplicated().any():
        raise AssertionError(f"{path}: duplicate record_uid")
    if (frame["patient_id"] == "").any():
        raise AssertionError(f"{path}: empty patient_id")
    return frame


def discover_mapping_dir(project: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        root = explicit.resolve()
        required = [
            root / "train_record_series_map.csv",
            root / "valid_record_series_map.csv",
        ]
        if not all(path.is_file() for path in required):
            raise FileNotFoundError(
                "Explicit mapping directory is missing train/valid maps: "
                + str(root)
            )
        return root

    candidates: list[Path] = []
    manifests = project / "manifests"
    for train_path in manifests.rglob("train_record_series_map.csv"):
        root = train_path.parent
        if (root / "valid_record_series_map.csv").is_file():
            candidates.append(root.resolve())
    candidates = sorted(set(candidates), key=lambda path: str(path))
    if len(candidates) != 1:
        message = [
            "Could not uniquely auto-resolve finalized mapping directory.",
            "Pass --mapping-dir explicitly.",
            "Candidates:",
            *[f"  - {path}" for path in candidates],
        ]
        raise RuntimeError("\n".join(message))
    return candidates[0]


def mapping_series_column(frame: pd.DataFrame) -> str:
    for column in ("series_uid", "suggested_series_uid"):
        if column in frame.columns:
            return column
    raise KeyError("Mapping table has neither series_uid nor suggested_series_uid")


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
            "resolved",
            "accepted",
            "auto_resolved",
            "manual_resolved",
            "rule_resolved",
        }
    )
    rejected_confidence = confidence.isin({"low", "unavailable", "rejected"})
    reviewed = (
        frame.get("reviewed", pd.Series(False, index=frame.index)).map(as_bool)
        if "reviewed" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    # Finalized maps normally use mapping_status=matched. reviewed+nonempty is
    # accepted as a fallback for manually finalized rows.
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
        raise KeyError(f"{path}: missing mapping columns {sorted(missing)}")
    if frame["record_uid"].duplicated().any():
        raise AssertionError(f"{path}: duplicate record_uid")
    if not frame["split"].astype(str).eq(split).all():
        raise AssertionError(f"{path}: split mismatch")

    series_column = mapping_series_column(frame)
    result = frame.copy()
    result["patient_id"] = result["patient_id"].map(normalize_patient_id)
    result["mapped_series_uid"] = result[series_column].map(clean_text)
    result["mapping_accepted_final"] = mapping_acceptance(result, series_column)
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
    ):
        if column in result.columns:
            keep.append(column)
    return result[keep].copy()


def resolve_embedding_key(payload: np.lib.npyio.NpzFile) -> str:
    for key in ("embeddings", "embedding"):
        if key in payload.files:
            return key
    raise KeyError("series_embeddings_5120.npz missing embeddings/embedding")


def infer_patient_from_series_uid(series_uid: str) -> str:
    parts = str(series_uid).split("__")
    if len(parts) >= 2:
        return normalize_patient_id(parts[1])
    return ""


def load_series_table(table_dir: Path, split: str) -> dict[str, Any]:
    npz_path = table_dir / "series_embeddings_5120.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)

    with np.load(npz_path, allow_pickle=False) as raw:
        if "series_uid" not in raw.files:
            raise KeyError(f"{npz_path}: missing series_uid")
        series_uid = raw["series_uid"].astype(str)
        key = resolve_embedding_key(raw)
        embeddings = np.asarray(raw[key], dtype=np.float32)
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
    patient_id = np.asarray([normalize_patient_id(value) for value in patient_id])

    scalar_path = table_dir / "series_scalar_features.parquet"
    if scalar_path.is_file():
        scalar = pd.read_parquet(scalar_path)
    else:
        scalar_path = table_dir / "series_scalar_features.csv"
        if not scalar_path.is_file():
            raise FileNotFoundError(
                f"Missing series scalar table under {table_dir}"
            )
        scalar = pd.read_csv(
            scalar_path,
            dtype={"patient_id": str, "series_uid": str},
            keep_default_na=False,
        )
    if "series_uid" not in scalar.columns:
        raise KeyError(f"{scalar_path}: missing series_uid")
    scalar["series_uid"] = scalar["series_uid"].astype(str)
    if scalar["series_uid"].duplicated().any():
        raise AssertionError(f"{scalar_path}: duplicate series_uid")

    scalar_by_uid = scalar.set_index("series_uid", drop=False)
    missing_scalar = sorted(set(series_uid.tolist()) - set(scalar_by_uid.index))
    if missing_scalar:
        raise AssertionError(
            f"{scalar_path}: missing scalar rows for {len(missing_scalar)} series"
        )
    scalar = scalar_by_uid.loc[series_uid.tolist()].reset_index(drop=True)

    identity_columns = {
        "patient_id",
        "series_uid",
        "series_id",
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
    scalar_columns: list[str] = []
    scalar_arrays: list[np.ndarray] = []
    for column in scalar.columns:
        if column in identity_columns:
            continue
        values = pd.to_numeric(scalar[column], errors="coerce").to_numpy(np.float64)
        if np.isinf(values).any():
            raise AssertionError(f"{scalar_path}: infinity in {column}")
        # Keep columns with at least one observed numeric value. Fold-local
        # preprocessing decides which are usable in each development split.
        if np.isfinite(values).any():
            scalar_columns.append(str(column))
            scalar_arrays.append(values.astype(np.float32))
    if not scalar_arrays:
        raise RuntimeError(f"{scalar_path}: no numeric scalar features")
    scalar_matrix = np.column_stack(scalar_arrays).astype(np.float32)

    pre_finite = np.isfinite(embeddings[:, 0, :]).all(axis=1)
    post_finite = np.isfinite(embeddings[:, 1, :]).all(axis=1)
    both_finite = pre_finite & post_finite

    availability_path = table_dir.parent / f"local_series_availability_{split.casefold()}.csv"
    availability_both = np.ones(len(series_uid), dtype=bool)
    availability_source = ""
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
        available_by_uid = availability.set_index("series_uid")
        if "local_both_available" not in available_by_uid.columns:
            raise KeyError(f"{availability_path}: missing local_both_available")
        availability_both = np.asarray(
            [
                as_bool(
                    available_by_uid.at[uid, "local_both_available"]
                    if uid in available_by_uid.index
                    else False
                )
                for uid in series_uid
            ],
            dtype=bool,
        )
        availability_source = str(availability_path)

    both_available = both_finite & availability_both
    return {
        "split": split,
        "npz_path": npz_path,
        "npz_sha256": sha256_file(npz_path),
        "scalar_path": scalar_path,
        "scalar_sha256": sha256_file(scalar_path),
        "availability_path": availability_source,
        "series_uid": series_uid,
        "patient_id": patient_id,
        "embeddings": embeddings,
        "scalar": scalar_matrix,
        "scalar_columns": scalar_columns,
        "pre_finite": pre_finite,
        "post_finite": post_finite,
        "both_available": both_available,
    }


def shared_scalar_schema(
    train_store: dict[str, Any],
    valid_store: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    train_names = list(train_store["scalar_columns"])
    valid_names = list(valid_store["scalar_columns"])
    valid_index = {name: index for index, name in enumerate(valid_names)}
    shared = [name for name in train_names if name in valid_index]
    if not shared:
        raise RuntimeError("Train/Valid have no shared scalar features")
    train_index = np.asarray([train_names.index(name) for name in shared], dtype=int)
    valid_columns = np.asarray([valid_index[name] for name in shared], dtype=int)
    return (
        train_store["scalar"][:, train_index],
        valid_store["scalar"][:, valid_columns],
        shared,
    )


def assign_exclusion_reason(frame: pd.DataFrame) -> pd.Series:
    reasons = pd.Series("", index=frame.index, dtype=object)
    invalid_label = ~frame["target"].isin([0, 1])
    reasons.loc[invalid_label] = "invalid_or_missing_adverse_label"

    not_mapped = ~frame["mapping_accepted_final"].map(as_bool)
    reasons.loc[(reasons == "") & not_mapped] = "record_to_series_mapping_not_accepted"

    no_series = frame["series_row"].isna()
    reasons.loc[(reasons == "") & no_series] = "mapped_series_not_in_local_table"

    patient_mismatch = (
        frame["series_patient_id"].fillna("").astype(str)
        != frame["patient_id"].fillna("").astype(str)
    )
    reasons.loc[
        (reasons == "") & ~no_series & patient_mismatch
    ] = "mapped_series_patient_mismatch"

    pre_missing = ~frame["pre_finite"].fillna(False).map(as_bool)
    post_missing = ~frame["post_finite"].fillna(False).map(as_bool)
    only_one = pre_missing ^ post_missing
    neither = pre_missing & post_missing
    reasons.loc[(reasons == "") & ~no_series & only_one] = (
        "strict_prepost_exclusion_only_one_phase_available"
    )
    reasons.loc[(reasons == "") & ~no_series & neither] = (
        "strict_prepost_exclusion_no_phase_available"
    )

    not_both = ~frame["both_available"].fillna(False).map(as_bool)
    reasons.loc[(reasons == "") & ~no_series & not_both] = (
        "strict_prepost_exclusion_availability_not_both"
    )
    return reasons


def build_split(
    records: pd.DataFrame,
    mapping: pd.DataFrame,
    store: dict[str, Any],
    scalar_matrix: np.ndarray,
    scalar_columns: list[str],
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
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
    merged["exclusion_reason"] = assign_exclusion_reason(merged)
    merged["included"] = merged["exclusion_reason"].eq("")

    accepted = merged[merged["included"]].copy()
    accepted["target"] = accepted["target"].astype(int)
    accepted["series_row"] = accepted["series_row"].astype(int)

    rows = accepted["series_row"].to_numpy(dtype=int)
    deep = store["embeddings"][rows].reshape(len(rows), -1).astype(np.float32)
    scalar = scalar_matrix[rows].astype(np.float32)
    if deep.shape[1] != 10240:
        raise AssertionError(f"{split}: deep feature shape changed: {deep.shape}")
    if not np.isfinite(deep).all():
        raise AssertionError(f"{split}: strict both-phase deep contains nonfinite values")
    if scalar.shape[1] != len(scalar_columns):
        raise AssertionError(f"{split}: scalar schema mismatch")

    patient_record_count = (
        accepted.groupby("patient_id")["record_uid"].transform("count").astype(int)
    )
    series_record_count = (
        accepted.groupby("mapped_series_uid")["record_uid"].transform("count").astype(int)
    )
    patient_label_count = (
        accepted.groupby("patient_id")["target"].transform("nunique").astype(int)
    )
    accepted["records_for_patient"] = patient_record_count
    accepted["records_using_same_series"] = series_record_count
    accepted["patient_has_conflicting_record_labels"] = (patient_label_count > 1).astype(int)

    metadata_columns = [
        "record_uid",
        "excel_row_number",
        "split",
        "patient_id",
        "mapped_series_uid",
        "target",
        "records_for_patient",
        "records_using_same_series",
        "patient_has_conflicting_record_labels",
    ]
    for column in (
        "mapping_status",
        "mapping_confidence",
        "mapping_source",
        "mapping_reason",
        "reviewed",
    ):
        if column in accepted.columns:
            metadata_columns.append(column)
    metadata = accepted[metadata_columns].rename(
        columns={"mapped_series_uid": "series_uid"}
    )

    atomic_csv(merged, output_dir / f"{split.casefold()}_record_inclusion_audit.csv")
    atomic_csv(metadata, output_dir / f"{split.casefold()}_records.csv")
    atomic_npz(
        output_dir / f"{split.casefold()}_features.npz",
        deep=deep,
        scalar=scalar,
        target=accepted["target"].to_numpy(np.int64),
        record_uid=np.asarray(accepted["record_uid"].astype(str).tolist(), dtype=str),
        patient_id=np.asarray(accepted["patient_id"].astype(str).tolist(), dtype=str),
        series_uid=np.asarray(accepted["mapped_series_uid"].astype(str).tolist(), dtype=str),
        scalar_feature_names=np.asarray(scalar_columns, dtype=str),
    )

    reason_counts = (
        merged.loc[~merged["included"], "exclusion_reason"]
        .value_counts(dropna=False)
        .to_dict()
    )
    summary = {
        "split": split,
        "source_excel_records": int(len(records)),
        "included_records": int(len(accepted)),
        "excluded_records": int((~merged["included"]).sum()),
        "included_patients": int(accepted["patient_id"].nunique()),
        "included_series": int(accepted["mapped_series_uid"].nunique()),
        "positive_records": int((accepted["target"] == 1).sum()),
        "negative_records": int((accepted["target"] == 0).sum()),
        "conflicting_label_patients_preserved": int(
            accepted.loc[
                accepted["patient_has_conflicting_record_labels"] == 1,
                "patient_id",
            ].nunique()
        ),
        "records_from_conflicting_label_patients": int(
            (accepted["patient_has_conflicting_record_labels"] == 1).sum()
        ),
        "deep_shape": list(deep.shape),
        "scalar_shape": list(scalar.shape),
        "strict_prepost_only": True,
        "exclusion_reasons": {
            str(key): int(value) for key, value in reason_counts.items()
        },
    }
    return metadata, summary


def assign_grouped_folds(metadata: pd.DataFrame) -> pd.DataFrame:
    y = metadata["target"].to_numpy(np.int64)
    groups = metadata["patient_id"].astype(str).to_numpy()
    if len(np.unique(y)) != 2:
        raise AssertionError("Train strict Pre+Post cohort must contain both classes")
    splitter = StratifiedGroupKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED,
    )
    fold = np.zeros(len(metadata), dtype=np.int64)
    for fold_number, (_, holdout) in enumerate(
        splitter.split(np.zeros(len(y)), y, groups),
        start=1,
    ):
        fold[holdout] = fold_number
    if (fold == 0).any():
        raise AssertionError("Incomplete fold assignment")
    audit = metadata.copy()
    audit["fold"] = fold
    leakage = audit.groupby("patient_id")["fold"].nunique().max()
    if int(leakage) != 1:
        raise AssertionError("A patient appears in multiple folds")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("/root/autodl-tmp/aneurysm"))
    parser.add_argument("--train-xlsx", type=Path)
    parser.add_argument("--valid-xlsx", type=Path)
    parser.add_argument("--mapping-dir", type=Path)
    parser.add_argument("--table-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
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
        else project / "outputs/api_gtmask_roi_cave_v5_fullmask_fullseq/adverse_prepost_record_task"
    )
    prepare_output(output_dir, args.overwrite)

    for path in (train_xlsx, valid_xlsx):
        if not path.is_file():
            raise FileNotFoundError(path)
    mapping_dir = discover_mapping_dir(project, args.mapping_dir)

    train_records = read_excel_records(train_xlsx, "Train")
    valid_records = read_excel_records(valid_xlsx, "Valid")
    if len(train_records) != 1157 or len(valid_records) != 289:
        raise AssertionError(
            f"Excel row counts changed: Train={len(train_records)}, "
            f"Valid={len(valid_records)}"
        )
    overlap = set(train_records["patient_id"]) & set(valid_records["patient_id"])
    if overlap:
        raise AssertionError(f"Train/Valid patient overlap: {sorted(overlap)[:10]}")

    train_mapping_path = mapping_dir / "train_record_series_map.csv"
    valid_mapping_path = mapping_dir / "valid_record_series_map.csv"
    train_mapping = load_mapping(train_mapping_path, "Train")
    valid_mapping = load_mapping(valid_mapping_path, "Valid")

    if set(train_records["record_uid"]) != set(train_mapping["record_uid"]):
        raise AssertionError("Train Excel and finalized mapping record_uid sets differ")
    if set(valid_records["record_uid"]) != set(valid_mapping["record_uid"]):
        raise AssertionError("Valid Excel and finalized mapping record_uid sets differ")

    train_store = load_series_table(table_root / "train", "Train")
    valid_store = load_series_table(table_root / "valid", "Valid")
    train_scalar, valid_scalar, scalar_columns = shared_scalar_schema(
        train_store, valid_store
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
    fold_table = assign_grouped_folds(train_meta)
    atomic_csv(fold_table, output_dir / "train_grouped_folds.csv")

    # Add the fixed fold assignment to the task NPZ without changing row order.
    train_npz_path = output_dir / "train_features.npz"
    with np.load(train_npz_path, allow_pickle=False) as raw:
        arrays = {name: raw[name] for name in raw.files}
    if not np.array_equal(
        arrays["record_uid"].astype(str),
        fold_table["record_uid"].astype(str).to_numpy(),
    ):
        raise AssertionError("Fold metadata order differs from Train NPZ")
    arrays["fold"] = fold_table["fold"].to_numpy(np.int64)
    atomic_npz(train_npz_path, **arrays)

    summary = {
        "status": "success",
        "version": "formal_adverse_prepost_local_cave_record_v1",
        "task": "adverse_outcome_record",
        "prediction_unit": "record_uid",
        "grouping_unit": "patient_id",
        "strict_phase_policy": "require_both_pre_and_post; ignore pre-only/post-only",
        "label_column_prefix": LABEL_COLUMN_PREFIX,
        "train": train_summary,
        "valid": valid_summary,
        "folds": N_FOLDS,
        "seed": SEED,
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
            "scalar_feature_names": scalar_columns,
        },
        "valid_used_for_fitting_or_selection": False,
    }
    atomic_json(summary, output_dir / "task_summary.json")
    atomic_json(summary, output_dir / ".TASK_SUCCESS.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build CAVE feature-bank prediction tasks without duplicating 10k features into CSV.

The feature bank stays label-blind. This script is the first stage that reads
outcome labels. It preserves the v3 patient/record task definitions and writes
aligned NPZ arrays plus small metadata CSV files for downstream training.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


META_COLUMNS = {
    "patient_id", "series_uid", "split", "source_type", "series_id",
    "record_uid", "mapping_status", "mapping_confidence", "target",
}


def normalize_patient_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\.0$", "", str(value).strip())


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(tmp, path)


def read_excel_records(path: Path, split: str) -> pd.DataFrame:
    frame = pd.read_excel(path, dtype=object).copy()
    if "病案号" not in frame.columns:
        raise KeyError(f"{path}: missing 病案号")
    frame.insert(0, "excel_row_number", range(2, len(frame) + 2))
    frame.insert(1, "split", split)
    frame.insert(2, "patient_id", frame["病案号"].map(normalize_patient_id))
    frame.insert(3, "record_uid", [
        f"{split}:{patient_id}:excel_row_{row_number:06d}"
        for patient_id, row_number in zip(frame["patient_id"], frame["excel_row_number"])
    ])
    if frame["record_uid"].duplicated().any():
        raise AssertionError(f"{path}: duplicate record_uid")
    return frame


def find_mapping_files(project: Path) -> tuple[Path, Path, Path]:
    candidates = [
        project / "manifests/api_record_v1_all_series_14e",
        project / "manifests/api_record_v1_all_series",
        *sorted([p for p in (project / "manifests").glob("api_record_v1_all_series*") if p.is_dir()], reverse=True),
    ]
    seen: set[Path] = set()
    for root in candidates:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        train = root / "train_record_series_suggestions.csv"
        valid = root / "valid_record_series_suggestions.csv"
        if train.is_file() and valid.is_file():
            return root, train, valid
    raise FileNotFoundError("Could not locate train/valid_record_series_suggestions.csv under manifests/")


def numeric_label(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_patient_label(records: pd.DataFrame, label_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for patient_id, group in records.groupby("patient_id", sort=True):
        values = numeric_label(group[label_column]).dropna().astype(int).unique().tolist()
        if len(values) == 1:
            rows.append({"patient_id": str(patient_id), "target": int(values[0])})
        elif len(values) > 1:
            conflicts.append({
                "patient_id": str(patient_id),
                "label_column": label_column,
                "observed_values": "|".join(map(str, sorted(values))),
                "record_count": int(len(group)),
            })
    return pd.DataFrame(rows), pd.DataFrame(conflicts)


class FeatureStore:
    def __init__(self, table_dir: Path, level: str):
        if level not in {"series", "patient"}:
            raise ValueError(level)
        self.level = level
        if level == "series":
            parquet_path = table_dir / "series_scalar_features.parquet"
            csv_path = table_dir / "series_scalar_features.csv"
            self.npz_path = table_dir / "series_embeddings_5120.npz"
            self.key = "series_uid"
            self.missing_columns = ("missing_pre", "missing_post")
        else:
            parquet_path = table_dir / "patient_median_scalar_features.parquet"
            csv_path = table_dir / "patient_median_scalar_features.csv"
            self.npz_path = table_dir / "patient_median_embeddings_5120.npz"
            self.key = "patient_id"
            self.missing_columns = ("missing_pre_all", "missing_post_all")
        self.scalar_path = parquet_path if parquet_path.is_file() else csv_path
        if not self.scalar_path.is_file() or not self.npz_path.is_file():
            raise FileNotFoundError(f"Missing CAVE tables under {table_dir}")
        self.scalar = pd.read_parquet(self.scalar_path) if self.scalar_path.suffix == ".parquet" else pd.read_csv(self.scalar_path)
        self.scalar[self.key] = self.scalar[self.key].astype(str)
        if self.scalar[self.key].duplicated().any():
            raise AssertionError(f"Duplicate {self.key}: {self.scalar_path}")
        raw = np.load(self.npz_path)
        ids = raw[self.key].astype(str)
        embeddings = raw["embeddings"].astype(np.float32)
        if embeddings.ndim != 3 or embeddings.shape[1:] != (2, 5120):
            raise AssertionError(f"Unexpected embeddings shape {embeddings.shape}: {self.npz_path}")
        if len(ids) != len(embeddings) or len(set(ids.tolist())) != len(ids):
            raise AssertionError(f"Invalid IDs in {self.npz_path}")
        self.embedding_by_id = {uid: embeddings[index] for index, uid in enumerate(ids)}
        scalar_ids = set(self.scalar[self.key])
        if scalar_ids != set(ids):
            raise AssertionError(f"Scalar/embedding ID mismatch at {table_dir}")
        excluded = META_COLUMNS | set(self.missing_columns) | {"missing_pre", "missing_post", "missing_pre_all", "missing_post_all"}
        self.scalar_columns = [
            column for column in self.scalar.columns
            if column not in excluded and pd.api.types.is_numeric_dtype(self.scalar[column])
        ]
        if not self.scalar_columns:
            raise AssertionError(f"No scalar features in {self.scalar_path}")
        self.row_by_id = self.scalar.set_index(self.key, drop=False)

    def extract(self, ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        missing = sorted(set(ids) - set(self.row_by_id.index.astype(str)))
        if missing:
            raise KeyError(f"Missing {self.level} features: {missing[:10]}")
        deep = np.stack([self.embedding_by_id[str(uid)].reshape(-1) for uid in ids]).astype(np.float32)
        rows = self.row_by_id.loc[[str(uid) for uid in ids]]
        scalar = rows[self.scalar_columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        flags: list[np.ndarray] = []
        for column in self.missing_columns:
            if column in rows.columns:
                flags.append(pd.to_numeric(rows[column], errors="coerce").fillna(1).to_numpy(np.float32))
            else:
                # Patient NPZ always carries missing_pre/missing_post even if the scalar table changes.
                flags.append(np.isnan(deep[:, :5120]).all(axis=1).astype(np.float32) if len(flags) == 0 else np.isnan(deep[:, 5120:]).all(axis=1).astype(np.float32))
        missing_flags = np.stack(flags, axis=1).astype(np.float32)
        if deep.shape[1] != 10240 or missing_flags.shape[1] != 2:
            raise AssertionError("CAVE feature shape changed")
        return deep, scalar, missing_flags


def task_summary(meta: pd.DataFrame) -> dict[str, Any]:
    target = pd.to_numeric(meta["target"], errors="raise").astype(int)
    return {
        "rows": int(len(meta)),
        "patients": int(meta["patient_id"].astype(str).nunique()),
        "positive": int((target == 1).sum()),
        "negative": int((target == 0).sum()),
    }


def write_task(
    output_root: Path,
    task_name: str,
    train_meta: pd.DataFrame,
    valid_meta: pd.DataFrame,
    train_store: FeatureStore,
    valid_store: FeatureStore,
    id_column: str,
    task_level: str,
    label_definition: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task_dir = output_root / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    for split_name, meta in (("Train", train_meta), ("Valid", valid_meta)):
        if meta.empty or meta["target"].isna().any():
            raise AssertionError(f"{task_name} {split_name}: empty or missing target")
        if not set(meta["target"].astype(int).unique()).issubset({0, 1}):
            raise AssertionError(f"{task_name} {split_name}: non-binary target")
    overlap = set(train_meta["patient_id"].astype(str)) & set(valid_meta["patient_id"].astype(str))
    if overlap:
        raise AssertionError(f"{task_name}: Train/Valid patient overlap={len(overlap)}")
    train_ids = train_meta[id_column].astype(str).tolist()
    valid_ids = valid_meta[id_column].astype(str).tolist()
    train_deep, train_scalar, train_missing = train_store.extract(train_ids)
    valid_deep, valid_scalar, valid_missing = valid_store.extract(valid_ids)
    if train_store.scalar_columns != valid_store.scalar_columns:
        raise AssertionError(f"{task_name}: Train/Valid scalar schema differs")
    np.savez_compressed(
        task_dir / "train_features.npz",
        deep=train_deep,
        scalar=train_scalar,
        missing=train_missing,
        target=train_meta["target"].astype(np.int64).to_numpy(),
    )
    np.savez_compressed(
        task_dir / "valid_features.npz",
        deep=valid_deep,
        scalar=valid_scalar,
        missing=valid_missing,
        target=valid_meta["target"].astype(np.int64).to_numpy(),
    )
    atomic_csv(train_meta, task_dir / "train_meta.csv")
    atomic_csv(valid_meta, task_dir / "valid_meta.csv")
    config = {
        "version": "api_fullseq_cave_v3_prediction_task_1",
        "task_name": task_name,
        "task_level": task_level,
        "label_definition": label_definition,
        "deep_dimension": 10240,
        "scalar_dimension": len(train_store.scalar_columns),
        "missing_dimension": 2,
        "scalar_columns": train_store.scalar_columns,
        "train": task_summary(train_meta),
        "valid": task_summary(valid_meta),
        "train_valid_patient_overlap": 0,
        **(extra or {}),
    }
    atomic_json(config, task_dir / "task_config.json")
    return config


def record_task_meta(records: pd.DataFrame, mappings: pd.DataFrame, source_label: str, store: FeatureStore) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"record_uid", "patient_id", "suggested_series_uid", "mapping_status", "mapping_confidence", "split"}
    missing = required - set(mappings.columns)
    if missing:
        raise KeyError(f"mapping missing {sorted(missing)}")
    mappings = mappings.copy()
    mappings["patient_id"] = mappings["patient_id"].map(normalize_patient_id)
    if mappings["record_uid"].duplicated().any():
        raise AssertionError("mapping record_uid duplicated")
    table = records.merge(mappings, on=["record_uid", "patient_id", "split"], how="left", validate="one_to_one", suffixes=("", "_mapping"))
    table["mapping_accepted"] = (
        table["suggested_series_uid"].fillna("").astype(str).ne("")
        & table["mapping_confidence"].fillna("").astype(str).isin(["high", "medium"])
    )
    audit = table.loc[~table["mapping_accepted"], [
        "record_uid", "patient_id", "split", "suggested_series_uid", "mapping_status", "mapping_confidence"
    ]].copy()
    accepted = table[table["mapping_accepted"]].copy()
    accepted["series_uid"] = accepted["suggested_series_uid"].astype(str)
    available_series = set(store.scalar["series_uid"].astype(str))
    feature_missing = accepted[~accepted["series_uid"].isin(available_series)].copy()
    if not feature_missing.empty:
        extra_audit = feature_missing[["record_uid", "patient_id", "split", "suggested_series_uid", "mapping_status", "mapping_confidence"]].copy()
        extra_audit["mapping_status"] = "mapped_series_missing_from_cave_featurebank"
        audit = pd.concat([audit, extra_audit], ignore_index=True)
    accepted = accepted[accepted["series_uid"].isin(available_series)].copy()
    label = numeric_label(accepted[source_label])
    valid_label = label.isin([1, 2, 3])
    accepted = accepted[valid_label].copy()
    label = label[valid_label].astype(int)
    accepted["target"] = (label != 1).astype(int).to_numpy()
    keep = ["record_uid", "patient_id", "split", "series_uid", "mapping_status", "mapping_confidence", "target"]
    return accepted[keep].reset_index(drop=True), audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="/root/autodl-tmp/aneurysm")
    parser.add_argument("--train-table-dir", required=True)
    parser.add_argument("--valid-table-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project = Path(args.project).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite non-empty {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    train_excel = read_excel_records(project / "metadata/Train.xlsx", "Train")
    valid_excel = read_excel_records(project / "metadata/valid.xlsx", "Valid")
    if len(train_excel) != 1157 or len(valid_excel) != 289:
        raise AssertionError(f"Excel rows changed Train={len(train_excel)} Valid={len(valid_excel)}")
    if set(train_excel["patient_id"]) & set(valid_excel["patient_id"]):
        raise AssertionError("Excel Train/Valid patient overlap")

    train_series = FeatureStore(Path(args.train_table_dir), "series")
    valid_series = FeatureStore(Path(args.valid_table_dir), "series")
    train_patient = FeatureStore(Path(args.train_table_dir), "patient")
    valid_patient = FeatureStore(Path(args.valid_table_dir), "patient")

    adverse_column = next((column for column in train_excel.columns if str(column).startswith("不良转归")), None)
    if adverse_column is None:
        raise KeyError("Could not find adverse label column")
    train_adverse_labels, train_conflicts = build_patient_label(train_excel, adverse_column)
    valid_adverse_labels, valid_conflicts = build_patient_label(valid_excel, adverse_column)
    train_adverse = train_patient.scalar[["patient_id", "split"]].merge(train_adverse_labels, on="patient_id", how="inner", validate="one_to_one")
    valid_adverse = valid_patient.scalar[["patient_id", "split"]].merge(valid_adverse_labels, on="patient_id", how="inner", validate="one_to_one")

    configs: dict[str, Any] = {}
    configs["adverse_patient"] = write_task(
        output, "adverse_patient", train_adverse, valid_adverse,
        train_patient, valid_patient, "patient_id", "patient",
        "Patient-level adverse outcome: Excel adverse label 1 vs 0; conflicting patient labels excluded.",
        {"train_conflicting_patients": int(len(train_conflicts)), "valid_conflicting_patients": int(len(valid_conflicts))},
    )
    atomic_csv(pd.concat([
        train_conflicts.assign(split="Train"), valid_conflicts.assign(split="Valid")
    ], ignore_index=True), output / "adverse_label_conflicts.csv")

    mapping_root, train_mapping_path, valid_mapping_path = find_mapping_files(project)
    train_mapping = pd.read_csv(train_mapping_path, dtype={"patient_id": str, "suggested_series_uid": str})
    valid_mapping = pd.read_csv(valid_mapping_path, dtype={"patient_id": str, "suggested_series_uid": str})
    unresolved: list[pd.DataFrame] = []
    for task_name, label_column, definition in [
        ("immediate_rroc_record", "术后即刻RROC", "Record-level immediate incomplete occlusion: RROC 2/3 positive, RROC 1 negative."),
        ("followup_rroc_record", "随访RROC123", "Record-level follow-up incomplete occlusion: RROC 2/3 positive, RROC 1 negative."),
    ]:
        train_meta, train_unresolved = record_task_meta(train_excel, train_mapping, label_column, train_series)
        valid_meta, valid_unresolved = record_task_meta(valid_excel, valid_mapping, label_column, valid_series)
        configs[task_name] = write_task(
            output, task_name, train_meta, valid_meta,
            train_series, valid_series, "series_uid", "record", definition,
            {
                "mapping_root": str(mapping_root),
                "accepted_mapping_confidence": ["high", "medium"],
                "train_unresolved_records": int(len(train_unresolved)),
                "valid_unresolved_records": int(len(valid_unresolved)),
            },
        )
        unresolved.extend([train_unresolved.assign(task_name=task_name), valid_unresolved.assign(task_name=task_name)])
    atomic_csv(pd.concat(unresolved, ignore_index=True), output / "record_mapping_unresolved.csv")

    summary = {
        "version": "api_fullseq_cave_v3_prediction_tasks_1",
        "mapping_root": str(mapping_root),
        "tasks": configs,
        "feature_extraction_completed_before_labels_read": True,
        "train_valid_patient_overlap": 0,
    }
    atomic_json(summary, output / "task_summary.json")
    atomic_json(summary, output / ".TASKS_SUCCESS")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

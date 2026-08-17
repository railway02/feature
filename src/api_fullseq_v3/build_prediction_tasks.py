#!/usr/bin/env python3
"""Build label tables for api_fullseq_v3 prediction experiments.

Tasks
-----
1. adverse_patient
   Patient-level adverse-outcome classification using label-blind median
   aggregation across all selected series for a patient.
2. immediate_rroc_record
   Record/lesion-level incomplete immediate occlusion: RROC 2/3 vs RROC 1.
3. followup_rroc_record
   Record/lesion-level incomplete follow-up occlusion: RROC 2/3 vs RROC 1.

The record tasks only use deterministic record-to-series mappings already
produced by the all-series manifest workflow. Low/unavailable mappings remain in
an audit table and are not silently assigned image features.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def normalize_patient_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return re.sub(r"\.0$", "", text)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(temporary, path)


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_excel_records(path: Path, split: str) -> pd.DataFrame:
    frame = pd.read_excel(path, dtype=object).copy()
    if "病案号" not in frame.columns:
        raise KeyError(f"{path}: missing 病案号")
    frame.insert(0, "excel_row_number", range(2, len(frame) + 2))
    frame.insert(1, "split", split)
    frame.insert(2, "patient_id", frame["病案号"].map(normalize_patient_id))
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
    if frame["record_uid"].duplicated().any():
        raise AssertionError(f"{path}: duplicate record_uid")
    return frame


def find_mapping_files(project: Path) -> tuple[Path, Path, Path]:
    candidates = [
        project / "manifests/api_record_v1_all_series_14e",
        project / "manifests/api_record_v1_all_series",
    ]
    # Also allow any later all-series directory containing both files.
    candidates.extend(
        sorted(
            [p for p in (project / "manifests").glob("api_record_v1_all_series*") if p.is_dir()],
            reverse=True,
        )
    )
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
    raise FileNotFoundError(
        "Could not locate train/valid_record_series_suggestions.csv under manifests/"
    )


def core_feature_columns(schema_path: Path) -> list[str]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    raw = schema["phase_core_features"]
    names = [item["feature_name"] if isinstance(item, dict) else str(item) for item in raw]
    columns = [f"{phase}_{name}" for phase in ("pre", "post") for name in names]
    if len(columns) != 212:
        raise AssertionError(f"Expected 212 default features, found {len(columns)}")
    return columns


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


def task_summary(frame: pd.DataFrame) -> dict[str, Any]:
    target = pd.to_numeric(frame["target"], errors="coerce")
    return {
        "rows": int(len(frame)),
        "patients": int(frame["patient_id"].astype(str).nunique()),
        "positive": int((target == 1).sum()),
        "negative": int((target == 0).sum()),
        "missing_target": int(target.isna().sum()),
    }


def write_task(
    output_root: Path,
    task_name: str,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_columns: list[str],
    task_level: str,
    label_definition: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task_dir = output_root / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    required = ["patient_id", "target", *feature_columns]
    for split_name, frame in (("Train", train), ("Valid", valid)):
        missing = set(required) - set(frame.columns)
        if missing:
            raise KeyError(f"{task_name} {split_name}: missing {sorted(missing)}")
        if frame["target"].isna().any():
            raise AssertionError(f"{task_name} {split_name}: target contains NaN")
        if not set(frame["target"].astype(int).unique()).issubset({0, 1}):
            raise AssertionError(f"{task_name} {split_name}: target is not binary")
    train_patients = set(train["patient_id"].astype(str))
    valid_patients = set(valid["patient_id"].astype(str))
    overlap = train_patients & valid_patients
    if overlap:
        raise AssertionError(f"{task_name}: Train/Valid patient overlap={len(overlap)}")
    atomic_csv(train, task_dir / "train.csv")
    atomic_csv(valid, task_dir / "valid.csv")
    config = {
        "task_name": task_name,
        "task_level": task_level,
        "label_definition": label_definition,
        "feature_columns": feature_columns,
        "train": task_summary(train),
        "valid": task_summary(valid),
        "train_valid_patient_overlap": 0,
        **(extra or {}),
    }
    atomic_json(config, task_dir / "task_config.json")
    return config


def build_record_task(
    records: pd.DataFrame,
    mappings: pd.DataFrame,
    series_features: pd.DataFrame,
    feature_columns: list[str],
    source_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_mapping = {
        "record_uid", "patient_id", "suggested_series_uid",
        "mapping_status", "mapping_confidence",
    }
    missing_mapping = required_mapping - set(mappings.columns)
    if missing_mapping:
        raise KeyError(f"mapping missing {sorted(missing_mapping)}")
    mappings = mappings.copy()
    mappings["patient_id"] = mappings["patient_id"].map(normalize_patient_id)
    if mappings["record_uid"].duplicated().any():
        raise AssertionError("mapping record_uid duplicated")

    table = records.merge(
        mappings,
        on=["record_uid", "patient_id", "split"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_mapping"),
    )
    table["mapping_accepted"] = (
        table["suggested_series_uid"].fillna("").astype(str).ne("")
        & table["mapping_confidence"].fillna("").astype(str).isin(["high", "medium"])
    )
    audit = table.loc[
        ~table["mapping_accepted"],
        [
            "record_uid", "patient_id", "split", "suggested_series_uid",
            "mapping_status", "mapping_confidence",
        ],
    ].copy()

    accepted = table[table["mapping_accepted"]].copy()
    accepted = accepted.merge(
        series_features,
        left_on=["patient_id", "suggested_series_uid"],
        right_on=["patient_id", "series_uid"],
        how="inner",
        validate="many_to_one",
        suffixes=("", "_feature"),
    )
    label = numeric_label(accepted[source_label])
    valid_label = label.isin([1, 2, 3])
    accepted = accepted[valid_label].copy()
    label = label[valid_label].astype(int)
    accepted["target"] = (label != 1).astype(int).to_numpy()

    keep_metadata = [
        "record_uid", "patient_id", "split", "series_uid", "series_id",
        "mapping_status", "mapping_confidence", "target",
    ]
    result = accepted[[*keep_metadata, *feature_columns]].copy()
    return result, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="/root/autodl-tmp/aneurysm")
    parser.add_argument("--train-feature-dir", required=True)
    parser.add_argument("--valid-feature-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project = Path(args.project).resolve()
    train_feature_dir = Path(args.train_feature_dir).resolve()
    valid_feature_dir = Path(args.valid_feature_dir).resolve()
    output = Path(args.output_dir).resolve()
    success = output / ".TASKS_SUCCESS"
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)

    train_excel = read_excel_records(project / "metadata/Train.xlsx", "Train")
    valid_excel = read_excel_records(project / "metadata/valid.xlsx", "Valid")
    if len(train_excel) != 1157 or len(valid_excel) != 289:
        raise AssertionError(
            f"Excel rows changed Train={len(train_excel)} Valid={len(valid_excel)}"
        )
    if set(train_excel["patient_id"]) & set(valid_excel["patient_id"]):
        raise AssertionError("Excel Train/Valid patient overlap")

    schema_path = train_feature_dir / "feature_schema.json"
    feature_columns = core_feature_columns(schema_path)
    train_series = pd.read_csv(
        train_feature_dir / "series_features.csv",
        dtype={"patient_id": str, "series_uid": str},
    )
    valid_series = pd.read_csv(
        valid_feature_dir / "series_features.csv",
        dtype={"patient_id": str, "series_uid": str},
    )
    train_patient = pd.read_csv(
        train_feature_dir / "patient_median_features.csv", dtype={"patient_id": str}
    )
    valid_patient = pd.read_csv(
        valid_feature_dir / "patient_median_features.csv", dtype={"patient_id": str}
    )
    for name, frame in {
        "train_series": train_series,
        "valid_series": valid_series,
        "train_patient": train_patient,
        "valid_patient": valid_patient,
    }.items():
        missing = set(feature_columns) - set(frame.columns)
        if missing:
            raise KeyError(f"{name}: missing {len(missing)} model features")

    adverse_column = next(
        (column for column in train_excel.columns if str(column).startswith("不良转归")),
        None,
    )
    if adverse_column is None:
        raise KeyError("Could not find adverse label column")
    train_adverse_labels, train_conflicts = build_patient_label(train_excel, adverse_column)
    valid_adverse_labels, valid_conflicts = build_patient_label(valid_excel, adverse_column)
    train_adverse = train_patient.merge(
        train_adverse_labels, on="patient_id", how="inner", validate="one_to_one"
    )
    valid_adverse = valid_patient.merge(
        valid_adverse_labels, on="patient_id", how="inner", validate="one_to_one"
    )
    train_adverse = train_adverse[["patient_id", "split", "target", *feature_columns]]
    valid_adverse = valid_adverse[["patient_id", "split", "target", *feature_columns]]
    configs: dict[str, Any] = {}
    configs["adverse_patient"] = write_task(
        output,
        "adverse_patient",
        train_adverse,
        valid_adverse,
        feature_columns,
        "patient",
        "Patient-level adverse outcome: Excel adverse label 1 vs 0; conflicting patient labels excluded.",
        {
            "train_conflicting_patients": int(len(train_conflicts)),
            "valid_conflicting_patients": int(len(valid_conflicts)),
        },
    )
    atomic_csv(
        pd.concat([train_conflicts.assign(split="Train"), valid_conflicts.assign(split="Valid")], ignore_index=True),
        output / "adverse_label_conflicts.csv",
    )

    mapping_root, train_mapping_path, valid_mapping_path = find_mapping_files(project)
    train_mapping = pd.read_csv(train_mapping_path, dtype={"patient_id": str, "suggested_series_uid": str})
    valid_mapping = pd.read_csv(valid_mapping_path, dtype={"patient_id": str, "suggested_series_uid": str})

    task_specs = [
        (
            "immediate_rroc_record",
            "术后即刻RROC",
            "Record-level immediate incomplete occlusion: RROC 2/3 positive, RROC 1 negative.",
        ),
        (
            "followup_rroc_record",
            "随访RROC123",
            "Record-level follow-up incomplete occlusion: RROC 2/3 positive, RROC 1 negative.",
        ),
    ]
    unresolved_tables: list[pd.DataFrame] = []
    for task_name, label_column, definition in task_specs:
        train_task, train_unresolved = build_record_task(
            train_excel, train_mapping, train_series, feature_columns, label_column
        )
        valid_task, valid_unresolved = build_record_task(
            valid_excel, valid_mapping, valid_series, feature_columns, label_column
        )
        configs[task_name] = write_task(
            output,
            task_name,
            train_task,
            valid_task,
            feature_columns,
            "record",
            definition,
            {
                "mapping_root": str(mapping_root),
                "accepted_mapping_confidence": ["high", "medium"],
                "train_unresolved_records": int(len(train_unresolved)),
                "valid_unresolved_records": int(len(valid_unresolved)),
            },
        )
        unresolved_tables.extend([
            train_unresolved.assign(task_name=task_name),
            valid_unresolved.assign(task_name=task_name),
        ])
    atomic_csv(
        pd.concat(unresolved_tables, ignore_index=True),
        output / "record_mapping_unresolved.csv",
    )

    summary = {
        "version": "api_fullseq_v3_prediction_tasks_v1",
        "feature_count": len(feature_columns),
        "mapping_root": str(mapping_root),
        "tasks": configs,
        "labels_used_only_after_feature_extraction": True,
        "train_valid_patient_overlap": 0,
    }
    atomic_json(summary, output / "task_summary.json")
    atomic_json(summary, success)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

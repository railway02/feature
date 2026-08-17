#!/usr/bin/env python3
"""Build patient-level adverse-outcome Pre+Post task tables from frozen full-sequence features.

This script follows the teacher's requested experiment:
    [147 Pre model-candidate features, 147 Post model-candidate features] -> adverse 0/1

It does not modify or rerun SEA-RAFT. It only aligns frozen patient features with
patient-level ground-truth labels, excludes Post-only and conflicting-label patients,
and writes one row per patient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LABEL_COLUMN_DEFAULT = "不良转归：1是；0否"
PID_COLUMN_DEFAULT = "病案号"
EXPECTED_PHASE_CANDIDATES = 147
EXPECTED_INPUT_FEATURES = 294
EXPECTED = {
    "train": {"rows": 855, "positive": 137, "negative": 718, "conflicts": 13},
    "valid": {"rows": 226, "positive": 38, "negative": 188, "conflicts": 3},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_patient_id(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            pass
    return text


def discover_label_column(frame: pd.DataFrame, requested: str) -> str:
    if requested in frame.columns:
        return requested
    matches = [column for column in frame.columns if "不良转归" in str(column)]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        f"Could not resolve adverse label column. requested={requested!r}, matches={matches}"
    )


def load_patient_labels(
    path: Path,
    split_name: str,
    patient_column: str,
    label_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(path)
    label_column = discover_label_column(raw, label_column)
    if patient_column not in raw.columns:
        raise ValueError(f"{path}: missing patient column {patient_column!r}")

    audit = raw.copy()
    audit["patient_id"] = audit[patient_column].map(normalize_patient_id)
    audit["adverse"] = pd.to_numeric(audit[label_column], errors="coerce")
    audit = audit[(audit["patient_id"] != "") & audit["adverse"].isin([0, 1])].copy()
    audit["adverse"] = audit["adverse"].astype(int)

    grouped = audit.groupby("patient_id", sort=False)["adverse"].agg(
        lambda series: sorted(set(int(value) for value in series.tolist()))
    )
    conflict_ids = grouped[grouped.map(len) > 1].index.astype(str)
    resolved_ids = grouped[grouped.map(len) == 1].index.astype(str)

    conflicts = (
        audit[audit["patient_id"].isin(conflict_ids)]
        .groupby("patient_id", as_index=False)
        .agg(
            labels=("adverse", lambda values: "|".join(map(str, sorted(set(values))))),
            record_count=("adverse", "size"),
        )
    )
    conflicts.insert(1, "split", split_name)

    resolved = pd.DataFrame(
        {
            "patient_id": resolved_ids,
            "adverse": [int(grouped.loc[patient_id][0]) for patient_id in resolved_ids],
        }
    )
    resolved.insert(1, "split", split_name)
    return resolved, conflicts, audit


def load_schema_candidates(schema_path: Path) -> list[str]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    definitions = schema.get("phase_features")
    if not isinstance(definitions, list):
        raise ValueError(f"{schema_path}: missing phase_features list")
    candidates = [
        str(item["feature_name"])
        for item in definitions
        if bool(item.get("model_candidate", False))
    ]
    if len(candidates) != EXPECTED_PHASE_CANDIDATES:
        raise AssertionError(
            f"Expected {EXPECTED_PHASE_CANDIDATES} phase model candidates, found {len(candidates)}"
        )
    if len(set(candidates)) != len(candidates):
        raise AssertionError("Duplicate phase model-candidate feature names in schema")
    return candidates


def load_features(path: Path, expected_split: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"patient_id": str})
    required = {
        "patient_id",
        "split",
        "missing_pre",
        "missing_post",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")
    frame["patient_id"] = frame["patient_id"].map(normalize_patient_id)
    if frame["patient_id"].duplicated().any():
        duplicate_ids = frame.loc[frame["patient_id"].duplicated(), "patient_id"].tolist()
        raise AssertionError(f"{path}: duplicate patient IDs, examples={duplicate_ids[:10]}")
    if not frame["split"].astype(str).eq(expected_split).all():
        raise AssertionError(f"{path}: expected split={expected_split}")
    return frame


def build_task(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    conflicts: pd.DataFrame,
    split_name: str,
    phase_candidates: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_columns = [f"pre_{name}" for name in phase_candidates] + [
        f"post_{name}" for name in phase_candidates
    ]
    missing_columns = [column for column in feature_columns if column not in features.columns]
    if missing_columns:
        raise AssertionError(
            f"{split_name}: missing {len(missing_columns)} schema-selected feature columns; "
            f"examples={missing_columns[:10]}"
        )
    if len(feature_columns) != EXPECTED_INPUT_FEATURES:
        raise AssertionError(f"Expected {EXPECTED_INPUT_FEATURES} input columns")

    conflict_ids = set(conflicts["patient_id"].astype(str))
    eligible = features[
        pd.to_numeric(features["missing_pre"], errors="coerce").eq(0)
        & pd.to_numeric(features["missing_post"], errors="coerce").eq(0)
        & ~features["patient_id"].isin(conflict_ids)
    ].copy()

    joined = eligible.merge(
        labels[["patient_id", "adverse"]],
        on="patient_id",
        how="inner",
        validate="one_to_one",
    )
    joined.insert(1, "task_split", split_name)

    for column in feature_columns:
        joined[column] = pd.to_numeric(joined[column], errors="coerce")
        values = joined[column].to_numpy(dtype=np.float64)
        if np.isinf(values).any():
            raise AssertionError(f"{split_name}: infinity in {column}")

    output = joined[["patient_id", "task_split", "adverse", *feature_columns]].copy()
    output = output.sort_values("patient_id").reset_index(drop=True)
    if output["patient_id"].duplicated().any():
        raise AssertionError(f"{split_name}: duplicate patient rows after task construction")
    if not set(output["adverse"].unique()).issubset({0, 1}):
        raise AssertionError(f"{split_name}: adverse labels are not binary")

    summary = {
        "split": split_name,
        "rows": int(len(output)),
        "positive": int(output["adverse"].sum()),
        "negative": int(len(output) - output["adverse"].sum()),
        "feature_count": len(feature_columns),
        "pre_feature_count": len(phase_candidates),
        "post_feature_count": len(phase_candidates),
        "conflict_patient_count": int(len(conflicts)),
        "excluded_missing_pre_or_post": int(
            (
                pd.to_numeric(features["missing_pre"], errors="coerce").eq(1)
                | pd.to_numeric(features["missing_post"], errors="coerce").eq(1)
            ).sum()
        ),
        "label_match_count": int(len(output)),
        "feature_nan_cells": int(output[feature_columns].isna().sum().sum()),
        "features_with_any_nan": int(output[feature_columns].isna().any().sum()),
    }
    return output, summary


def assert_expected(summary: dict[str, Any], key: str) -> None:
    expected = EXPECTED[key]
    for field in ("rows", "positive", "negative", "conflicts"):
        actual_field = "conflict_patient_count" if field == "conflicts" else field
        if int(summary[actual_field]) != int(expected[field]):
            raise AssertionError(
                f"{key} {field}: expected={expected[field]} actual={summary[actual_field]}"
            )


def write_report(path: Path, payload: dict[str, Any]) -> None:
    train = payload["train"]
    valid = payload["valid"]
    lines = [
        "# adverse_prepost_fullseq_v2 task build report",
        "",
        f"- Generated: {payload['generated_utc']}",
        f"- Phase model-candidate features: {payload['phase_candidate_count']}",
        f"- Input features: {payload['input_feature_count']} (Pre + Post)",
        f"- Train: {train['rows']} patients, {train['positive']} positive, {train['negative']} negative",
        f"- Valid: {valid['rows']} patients, {valid['positive']} positive, {valid['negative']} negative",
        f"- Train conflict patients excluded: {train['conflict_patient_count']}",
        f"- Valid conflict patients excluded: {valid['conflict_patient_count']}",
        "- Post-only patients are excluded from this Pre+Post task; missing Pre is never zero-filled.",
        "- NaN values within genuine Pre/Post phases are preserved for fold-local preprocessing.",
        "- No QC, identifier, Delta, runtime, frame-count, or label-derived fields enter the input.",
        "",
        "## Input hashes",
        "",
    ]
    for name, value in payload["input_sha256"].items():
        lines.append(f"- {name}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--valid-features", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--valid-labels", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patient-column", default=PID_COLUMN_DEFAULT)
    parser.add_argument("--label-column", default=LABEL_COLUMN_DEFAULT)
    parser.add_argument(
        "--skip-expected-assertions",
        action="store_true",
        help="Disable current-cohort assertions (855/226 and conflict counts).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    phase_candidates = load_schema_candidates(args.schema)
    train_features = load_features(args.train_features, "Train")
    valid_features = load_features(args.valid_features, "Valid")
    overlap = set(train_features["patient_id"]) & set(valid_features["patient_id"])
    if overlap:
        raise AssertionError(f"Train/Valid feature patient overlap: {sorted(overlap)[:10]}")

    train_labels, train_conflicts, train_label_audit = load_patient_labels(
        args.train_labels, "Train", args.patient_column, args.label_column
    )
    valid_labels, valid_conflicts, valid_label_audit = load_patient_labels(
        args.valid_labels, "Valid", args.patient_column, args.label_column
    )

    train_task, train_summary = build_task(
        train_features, train_labels, train_conflicts, "Train", phase_candidates
    )
    valid_task, valid_summary = build_task(
        valid_features, valid_labels, valid_conflicts, "Valid", phase_candidates
    )
    task_overlap = set(train_task["patient_id"]) & set(valid_task["patient_id"])
    if task_overlap:
        raise AssertionError(f"Task Train/Valid overlap: {sorted(task_overlap)[:10]}")

    if not args.skip_expected_assertions:
        assert_expected(train_summary, "train")
        assert_expected(valid_summary, "valid")

    train_path = args.output_dir / "adverse_prepost_fullseq_v2_train.csv"
    valid_path = args.output_dir / "adverse_prepost_fullseq_v2_valid.csv"
    conflict_path = args.output_dir / "adverse_label_conflicts.csv"
    train_audit_path = args.output_dir / "train_label_records_audit.csv"
    valid_audit_path = args.output_dir / "valid_label_records_audit.csv"

    train_task.to_csv(train_path, index=False)
    valid_task.to_csv(valid_path, index=False)
    pd.concat([train_conflicts, valid_conflicts], ignore_index=True).to_csv(
        conflict_path, index=False
    )
    train_label_audit.to_csv(train_audit_path, index=False)
    valid_label_audit.to_csv(valid_audit_path, index=False)

    payload = {
        "generated_utc": utc_now(),
        "phase_candidate_count": len(phase_candidates),
        "input_feature_count": EXPECTED_INPUT_FEATURES,
        "feature_order": [
            *[f"pre_{name}" for name in phase_candidates],
            *[f"post_{name}" for name in phase_candidates],
        ],
        "train": train_summary,
        "valid": valid_summary,
        "input_sha256": {
            "train_features": sha256_file(args.train_features),
            "valid_features": sha256_file(args.valid_features),
            "train_labels": sha256_file(args.train_labels),
            "valid_labels": sha256_file(args.valid_labels),
            "schema": sha256_file(args.schema),
        },
        "outputs": {
            "train_task": str(train_path),
            "valid_task": str(valid_path),
            "conflicts": str(conflict_path),
        },
    }
    (args.output_dir / "task_build_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(args.output_dir / "task_build_report.md", payload)
    (args.output_dir / ".SUCCESS").write_text(utc_now() + "\n", encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

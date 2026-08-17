#!/usr/bin/env python3
"""Independently audit the patient-level 2D task datasets.

The audit is deliberately read-only with respect to metadata, feature sources,
task datasets, and existing reports. It reconstructs each task from the raw
Excel labels and patient-level feature tables, then writes audit artifacts only
under reports/independent_task_audit/.
"""

from __future__ import annotations

import csv
import math
import re
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT / "reports" / "independent_task_audit"

ID_COLUMN = "病案号"
LABEL_COLUMNS = {
    "adverse": "不良转归：1是；0否",
    "immediate": "术后即刻RROC",
    "followup": "随访RROC123",
}
LABEL_RANGES = {
    "adverse": {0, 1},
    "immediate": {1, 2, 3},
    "followup": {1, 2, 3},
}

METADATA_PATHS = {
    "train": PROJECT / "metadata" / "Train.xlsx",
    "valid": PROJECT / "metadata" / "valid.xlsx",
}
FEATURE_PATHS = {
    "train": PROJECT / "outputs" / "features" / "train_patient_flow_features.csv",
    "valid": PROJECT / "outputs" / "features" / "valid_patient_flow_features.csv",
    "combined": PROJECT / "outputs" / "features" / "patient_flow_features.csv",
}
EXISTING_CONFLICT_PATH = PROJECT / "reports" / "patient_label_conflicts.csv"

TASKS = {
    "adverse_pre": {
        "label": "adverse",
        "mode": "pre",
        "expected_rows": {"train": 794, "valid": 209},
        "expected_features": 48,
    },
    "immediate_pre": {
        "label": "immediate",
        "mode": "pre",
        "expected_rows": {"train": 794, "valid": 209},
        "expected_features": 48,
    },
    "immediate_post": {
        "label": "immediate",
        "mode": "post",
        "expected_rows": {"train": 965, "valid": 240},
        "expected_features": 48,
    },
    "followup_prepost": {
        "label": "followup",
        "mode": "prepost",
        "expected_rows": {"train": 794, "valid": 209},
        "expected_features": 144,
    },
}

SUMMARY_COLUMNS = ["scope", "check", "status", "issue_count", "details"]
PATIENT_SET_COLUMNS = [
    "task", "split", "patient_id", "mismatch_type", "expected", "actual", "details"
]
LABEL_MISMATCH_COLUMNS = [
    "task", "split", "patient_id", "issue_type", "expected_label", "actual_label", "details"
]
FEATURE_ISSUE_COLUMNS = [
    "task", "split", "issue_type", "column", "related_column", "position", "details"
]
CONFLICT_COLUMNS = [
    "patient_id",
    "split",
    "label",
    "source_label_column",
    "values",
    "record_count",
    "in_existing_conflict_report",
    "existing_values",
    "existing_record_count",
    "existing_report_match",
    "relevant_task_presence",
]
NUMERIC_ISSUE_COLUMNS = [
    "task", "split", "field_type", "column", "issue_type", "count", "patient_ids"
]
CLASS_DISTRIBUTION_COLUMNS = ["task", "split", "label", "count", "proportion"]
VALUE_MISMATCH_COLUMNS = [
    "scope", "task", "split", "patient_id", "column", "expected", "actual", "issue_type"
]


def normalize_patient_id(value: Any) -> str | None:
    """Strip whitespace and a trailing Excel-style '.0' from a patient ID."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return None
    return re.sub(r"\.0$", "", text)


def canonical_value(value: Any) -> int | float | str | None:
    """Return a stable scalar representation for labels and report comparisons."""
    if pd.isna(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value).strip()
    if math.isfinite(numeric) and numeric.is_integer():
        return int(numeric)
    return numeric


def display_value(value: Any) -> str:
    value = canonical_value(value)
    if value is None:
        return ""
    return str(value)


def join_values(values: Iterable[Any]) -> str:
    return "|".join(sorted((display_value(value) for value in values), key=str))


def write_table(rows: list[dict[str, Any]], columns: list[str], path: Path) -> None:
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def read_csv_checked(path: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        try:
            header = next(csv.reader(handle))
        except StopIteration:
            header = []
    duplicate_headers = sorted(name for name, count in Counter(header).items() if count > 1)
    frame = pd.read_csv(path, dtype={"patient_id": "string"})
    return frame, header, duplicate_headers


def select_expected_features(columns: Iterable[str], mode: str) -> list[str]:
    selected: list[str] = []
    excluded = {"patient_id", "split", "missing_pre", "missing_post"}
    for column in columns:
        if column in excluded or "runtime_s" in column:
            continue
        if column in {"pre_n_pairs", "post_n_pairs"}:
            continue
        if mode == "pre" and column.startswith("pre_"):
            selected.append(column)
        elif mode == "post" and column.startswith("post_"):
            selected.append(column)
        elif mode == "prepost" and column.startswith(("pre_", "post_", "delta_")):
            selected.append(column)
    return selected


def series_equal(left: pd.Series, right: pd.Series) -> np.ndarray:
    """Compare two aligned series, allowing only tiny CSV float round-off."""
    left_numeric = pd.to_numeric(left, errors="coerce")
    right_numeric = pd.to_numeric(right, errors="coerce")
    numeric_mask = left_numeric.notna() & right_numeric.notna()
    both_missing = left.isna() & right.isna()
    result = np.zeros(len(left), dtype=bool)
    if numeric_mask.any():
        result[numeric_mask.to_numpy()] = np.isclose(
            left_numeric[numeric_mask].to_numpy(dtype=float),
            right_numeric[numeric_mask].to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
    result[both_missing.to_numpy()] = True
    other_mask = ~(numeric_mask | both_missing)
    if other_mask.any():
        result[other_mask.to_numpy()] = (
            left[other_mask].astype("string").fillna("<NA>").to_numpy()
            == right[other_mask].astype("string").fillna("<NA>").to_numpy()
        )
    return result


def duplicate_feature_pairs(frame: pd.DataFrame, columns: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for index, left_column in enumerate(columns):
        left = frame[left_column]
        for right_column in columns[index + 1 :]:
            if left.equals(frame[right_column]):
                pairs.append((left_column, right_column))
    return pairs


class Audit:
    def __init__(self) -> None:
        self.summary: list[dict[str, Any]] = []
        self.patient_set_issues: list[dict[str, Any]] = []
        self.label_issues: list[dict[str, Any]] = []
        self.feature_issues: list[dict[str, Any]] = []
        self.conflicts: list[dict[str, Any]] = []
        self.numeric_issues: list[dict[str, Any]] = []
        self.class_distribution: list[dict[str, Any]] = []
        self.value_mismatches: list[dict[str, Any]] = []
        self.task_metrics: dict[tuple[str, str], dict[str, Any]] = {}
        self.checked_files: list[Path] = []
        self.task_frames: dict[tuple[str, str], pd.DataFrame] = {}
        self.metadata: dict[str, pd.DataFrame] = {}
        self.features: dict[str, pd.DataFrame] = {}
        self.label_values: dict[str, dict[str, dict[str, tuple[Any, ...]]]] = defaultdict(
            lambda: defaultdict(dict)
        )

    def check(self, scope: str, name: str, failures: int, details: str) -> None:
        self.summary.append(
            {
                "scope": scope,
                "check": name,
                "status": "FAIL" if failures else "PASS",
                "issue_count": int(failures),
                "details": details,
            }
        )

    def add_patient_issue(
        self,
        task: str,
        split: str,
        patient_id: str,
        mismatch_type: str,
        expected: Any = "",
        actual: Any = "",
        details: str = "",
    ) -> None:
        self.patient_set_issues.append(
            {
                "task": task,
                "split": split,
                "patient_id": patient_id,
                "mismatch_type": mismatch_type,
                "expected": expected,
                "actual": actual,
                "details": details,
            }
        )

    def add_feature_issue(
        self,
        task: str,
        split: str,
        issue_type: str,
        column: str = "",
        related_column: str = "",
        position: Any = "",
        details: str = "",
    ) -> None:
        self.feature_issues.append(
            {
                "task": task,
                "split": split,
                "issue_type": issue_type,
                "column": column,
                "related_column": related_column,
                "position": position,
                "details": details,
            }
        )

    def load_metadata(self) -> None:
        for split, path in METADATA_PATHS.items():
            self.checked_files.append(path)
            frame = pd.read_excel(path)
            required = [ID_COLUMN, *LABEL_COLUMNS.values()]
            missing_columns = [column for column in required if column not in frame.columns]
            self.check(
                f"metadata:{split}",
                "required_columns",
                len(missing_columns),
                "missing=" + ",".join(missing_columns),
            )
            if missing_columns:
                raise KeyError(f"{path} missing required columns: {missing_columns}")

            frame = frame.copy()
            frame["patient_id"] = frame[ID_COLUMN].map(normalize_patient_id)
            invalid_ids = frame["patient_id"].isna()
            self.check(
                f"metadata:{split}",
                "valid_normalized_patient_ids",
                int(invalid_ids.sum()),
                f"rows={len(frame)}, invalid_ids={int(invalid_ids.sum())}",
            )
            frame = frame.loc[~invalid_ids].copy()
            self.metadata[split] = frame

            for label, source_column in LABEL_COLUMNS.items():
                non_numeric = pd.to_numeric(frame[source_column], errors="coerce").isna() & frame[
                    source_column
                ].notna()
                numeric = pd.to_numeric(frame[source_column], errors="coerce")
                invalid_range = numeric.notna() & ~numeric.isin(LABEL_RANGES[label])
                source_label_failures = int(non_numeric.sum() + invalid_range.sum())
                self.check(
                    f"metadata:{split}:{label}",
                    "source_label_range_and_type",
                    source_label_failures,
                    f"non_numeric={int(non_numeric.sum())}, out_of_range={int(invalid_range.sum())}",
                )

                for patient_id, group in frame.groupby("patient_id", sort=True):
                    values: list[Any] = []
                    for raw_value in group[source_column].tolist():
                        value = canonical_value(raw_value)
                        if value is not None and value not in values:
                            values.append(value)
                    self.label_values[split][label][patient_id] = tuple(values)
                    if len(values) > 1:
                        self.conflicts.append(
                            {
                                "patient_id": patient_id,
                                "split": split,
                                "label": label,
                                "source_label_column": source_column,
                                "values": join_values(values),
                                "record_count": len(group),
                                "in_existing_conflict_report": False,
                                "existing_values": "",
                                "existing_record_count": "",
                                "existing_report_match": False,
                                "relevant_task_presence": "",
                            }
                        )

                label_groups = self.label_values[split][label]
                conflict_count = sum(len(values) > 1 for values in label_groups.values())
                missing_count = sum(len(values) == 0 for values in label_groups.values())
                self.check(
                    f"metadata:{split}:{label}",
                    "patient_level_label_uniqueness",
                    0,
                    f"conflicts_reported={conflict_count}, all_missing={missing_count}",
                )

        train_ids = set(self.metadata["train"]["patient_id"])
        valid_ids = set(self.metadata["valid"]["patient_id"])
        overlap = sorted(train_ids & valid_ids)
        for patient_id in overlap:
            self.add_patient_issue(
                "all", "train|valid", patient_id, "metadata_train_valid_overlap", 0, 1
            )
        self.check(
            "metadata",
            "train_valid_patient_overlap",
            len(overlap),
            f"overlap={len(overlap)}",
        )

    def load_features(self) -> None:
        for split, path in FEATURE_PATHS.items():
            self.checked_files.append(path)
            frame, header, duplicate_headers = read_csv_checked(path)
            for column in duplicate_headers:
                self.add_feature_issue(
                    "source_features", split, "duplicate_header_name", column=column
                )
            self.check(
                f"source_features:{split}",
                "unique_header_names",
                len(duplicate_headers),
                f"columns={len(header)}, duplicate_names={len(duplicate_headers)}",
            )

            required = ["patient_id", "split", "missing_pre", "missing_post"]
            missing_columns = [column for column in required if column not in frame.columns]
            self.check(
                f"source_features:{split}",
                "required_columns",
                len(missing_columns),
                "missing=" + ",".join(missing_columns),
            )
            if missing_columns:
                raise KeyError(f"{path} missing required columns: {missing_columns}")

            frame = frame.copy()
            frame["patient_id"] = frame["patient_id"].map(normalize_patient_id)
            invalid_ids = int(frame["patient_id"].isna().sum())
            duplicate_ids = sorted(
                frame.loc[frame["patient_id"].duplicated(keep=False), "patient_id"]
                .dropna()
                .unique()
                .tolist()
            )
            for patient_id in duplicate_ids:
                self.add_patient_issue(
                    "source_features", split, patient_id, "duplicate_source_feature_patient"
                )
            self.check(
                f"source_features:{split}",
                "unique_valid_patient_ids",
                invalid_ids + len(duplicate_ids),
                f"invalid_ids={invalid_ids}, duplicate_patients={len(duplicate_ids)}",
            )

            allowed_splits = {"train", "valid"} if split == "combined" else {split}
            normalized_split = frame["split"].astype("string").str.strip().str.lower()
            wrong_split = ~normalized_split.isin(allowed_splits)
            self.check(
                f"source_features:{split}",
                "split_values",
                int(wrong_split.sum()),
                f"values={dict(normalized_split.value_counts(dropna=False))}",
            )
            frame["split"] = normalized_split

            for flag in ["missing_pre", "missing_post"]:
                numeric_flag = pd.to_numeric(frame[flag], errors="coerce")
                invalid_flag = numeric_flag.isna() | ~numeric_flag.isin([0, 1])
                self.check(
                    f"source_features:{split}",
                    f"{flag}_binary",
                    int(invalid_flag.sum()),
                    f"invalid={int(invalid_flag.sum())}",
                )
                frame[flag] = numeric_flag

            self.features[split] = frame

        train_columns = list(self.features["train"].columns)
        valid_columns = list(self.features["valid"].columns)
        same_split_columns = train_columns == valid_columns
        if not same_split_columns:
            self.add_feature_issue(
                "source_features",
                "train|valid",
                "source_feature_column_order_mismatch",
                details=f"train_columns={len(train_columns)}, valid_columns={len(valid_columns)}",
            )
        self.check(
            "source_features",
            "train_valid_column_names_order_count",
            0 if same_split_columns else 1,
            f"train={len(train_columns)}, valid={len(valid_columns)}, exact_match={same_split_columns}",
        )

        metadata_ids = {
            split: set(self.metadata[split]["patient_id"]) for split in ["train", "valid"]
        }
        for split in ["train", "valid"]:
            feature_ids = set(self.features[split]["patient_id"].dropna())
            extras = sorted(feature_ids - metadata_ids[split])
            for patient_id in extras:
                self.add_patient_issue(
                    "source_features",
                    split,
                    patient_id,
                    "source_feature_patient_not_in_metadata_split",
                    expected=split,
                    actual="absent",
                )
            self.check(
                f"source_features:{split}",
                "patient_membership_in_metadata",
                len(extras),
                f"feature_patients={len(feature_ids)}, not_in_metadata={len(extras)}",
            )

        expected_combined = pd.concat(
            [self.features["train"], self.features["valid"]], ignore_index=True
        )
        combined = self.features["combined"]
        expected_ids = set(expected_combined["patient_id"].dropna())
        combined_ids = set(combined["patient_id"].dropna())
        missing_ids = sorted(expected_ids - combined_ids)
        extra_ids = sorted(combined_ids - expected_ids)
        for patient_id in missing_ids:
            expected_split = expected_combined.loc[
                expected_combined["patient_id"] == patient_id, "split"
            ].iloc[0]
            self.add_patient_issue(
                "source_features",
                str(expected_split),
                patient_id,
                "missing_from_combined_feature_source",
                expected=1,
                actual=0,
            )
        for patient_id in extra_ids:
            actual_split = combined.loc[combined["patient_id"] == patient_id, "split"].iloc[0]
            self.add_patient_issue(
                "source_features",
                str(actual_split),
                patient_id,
                "extra_in_combined_feature_source",
                expected=0,
                actual=1,
            )
        self.check(
            "source_features:combined",
            "patient_set_equals_split_sources",
            len(missing_ids) + len(extra_ids),
            f"missing={len(missing_ids)}, extra={len(extra_ids)}",
        )

        same_combined_columns = list(combined.columns) == train_columns
        if not same_combined_columns:
            self.add_feature_issue(
                "source_features",
                "combined",
                "combined_source_column_order_mismatch",
                details=f"combined={len(combined.columns)}, split_source={len(train_columns)}",
            )
        self.check(
            "source_features:combined",
            "column_names_order_count_equals_split_sources",
            0 if same_combined_columns else 1,
            f"exact_match={same_combined_columns}",
        )

        value_mismatch_count = 0
        if same_combined_columns:
            expected_indexed = expected_combined.drop_duplicates("patient_id").set_index("patient_id")
            combined_indexed = combined.drop_duplicates("patient_id").set_index("patient_id")
            common_ids = sorted(expected_ids & combined_ids)
            for column in [name for name in train_columns if name != "patient_id"]:
                expected_series = expected_indexed.loc[common_ids, column]
                actual_series = combined_indexed.loc[common_ids, column]
                equal = series_equal(expected_series, actual_series)
                for position in np.flatnonzero(~equal):
                    patient_id = common_ids[int(position)]
                    self.value_mismatches.append(
                        {
                            "scope": "source_features",
                            "task": "combined",
                            "split": expected_indexed.loc[patient_id, "split"],
                            "patient_id": patient_id,
                            "column": column,
                            "expected": display_value(expected_series.iloc[position]),
                            "actual": display_value(actual_series.iloc[position]),
                            "issue_type": "combined_vs_split_source_value_mismatch",
                        }
                    )
                    value_mismatch_count += 1
        self.check(
            "source_features:combined",
            "values_equal_split_sources",
            value_mismatch_count,
            f"value_mismatches={value_mismatch_count}",
        )

    def audit_existing_conflicts(self) -> None:
        self.checked_files.append(EXISTING_CONFLICT_PATH)
        existing, _, duplicate_headers = read_csv_checked(EXISTING_CONFLICT_PATH)
        required = ["patient_id", "split", "label", "values", "record_count"]
        missing_columns = [column for column in required if column not in existing.columns]
        failures = len(duplicate_headers) + len(missing_columns)
        self.check(
            "existing_conflict_report",
            "schema",
            failures,
            f"missing={missing_columns}, duplicate_headers={duplicate_headers}",
        )
        if missing_columns:
            return

        existing = existing.copy()
        existing["patient_id"] = existing["patient_id"].map(normalize_patient_id)
        source_to_short = {source: short for short, source in LABEL_COLUMNS.items()}
        existing["label_short"] = existing["label"].map(source_to_short)

        def conflict_key(row: pd.Series) -> tuple[str, str, str]:
            return str(row["split"]).strip().lower(), str(row["label_short"]), str(row["patient_id"])

        existing_by_key: dict[tuple[str, str, str], list[pd.Series]] = defaultdict(list)
        for _, row in existing.iterrows():
            existing_by_key[conflict_key(row)].append(row)

        derived_by_key = {
            (row["split"], row["label"], row["patient_id"]): row for row in self.conflicts
        }
        missing_keys = sorted(set(derived_by_key) - set(existing_by_key))
        extra_keys = sorted(set(existing_by_key) - set(derived_by_key))
        mismatch_count = len(missing_keys) + len(extra_keys)

        for key, derived in derived_by_key.items():
            rows = existing_by_key.get(key, [])
            if not rows:
                continue
            existing_row = rows[0]
            existing_values = {
                canonical_value(value) for value in str(existing_row["values"]).split("|")
            }
            derived_values = {
                canonical_value(value) for value in str(derived["values"]).split("|")
            }
            record_count_match = canonical_value(existing_row["record_count"]) == canonical_value(
                derived["record_count"]
            )
            match = len(rows) == 1 and existing_values == derived_values and record_count_match
            derived["in_existing_conflict_report"] = True
            derived["existing_values"] = str(existing_row["values"])
            derived["existing_record_count"] = existing_row["record_count"]
            derived["existing_report_match"] = match
            if not match:
                mismatch_count += 1

        self.check(
            "existing_conflict_report",
            "content_equals_independent_derivation",
            mismatch_count,
            f"derived={len(derived_by_key)}, existing={len(existing)}, missing={len(missing_keys)}, "
            f"extra={len(extra_keys)}, mismatched={mismatch_count - len(missing_keys) - len(extra_keys)}",
        )

    def expected_patient_ids(self, task: str, split: str) -> tuple[set[str], dict[str, Any]]:
        config = TASKS[task]
        label = str(config["label"])
        mode = str(config["mode"])
        feature_frame = self.features[split]
        if mode == "pre":
            available = feature_frame["missing_pre"].eq(0)
        elif mode == "post":
            available = feature_frame["missing_post"].eq(0)
        else:
            available = feature_frame["missing_pre"].eq(0) & feature_frame["missing_post"].eq(0)

        labels = self.label_values[split][label]
        unique_labels = {
            patient_id: values[0] for patient_id, values in labels.items() if len(values) == 1
        }
        available_ids = set(feature_frame.loc[available, "patient_id"].dropna())
        expected_ids = available_ids & set(unique_labels)
        return expected_ids, unique_labels

    def audit_tasks(self) -> None:
        actual_features_by_task: dict[tuple[str, str], list[str]] = {}
        expected_features_by_task: dict[tuple[str, str], list[str]] = {}

        for task, config in TASKS.items():
            label = str(config["label"])
            mode = str(config["mode"])
            for split in ["train", "valid"]:
                path = PROJECT / "outputs" / "task_datasets" / f"{task}_{split}.csv"
                self.checked_files.append(path)
                frame, header, duplicate_headers = read_csv_checked(path)
                self.task_frames[(task, split)] = frame

                for column in duplicate_headers:
                    self.add_feature_issue(task, split, "duplicate_header_name", column=column)
                self.check(
                    f"task:{task}:{split}",
                    "unique_header_names",
                    len(duplicate_headers),
                    f"columns={len(header)}, duplicate_names={len(duplicate_headers)}",
                )

                required = ["patient_id", "split", label]
                missing_required = [column for column in required if column not in frame.columns]
                self.check(
                    f"task:{task}:{split}",
                    "required_structural_columns",
                    len(missing_required),
                    "missing=" + ",".join(missing_required),
                )
                if missing_required:
                    for column in missing_required:
                        self.add_feature_issue(task, split, "missing_structural_column", column=column)
                    continue

                frame = frame.copy()
                frame["patient_id"] = frame["patient_id"].map(normalize_patient_id)
                frame["split"] = frame["split"].astype("string").str.strip().str.lower()
                self.task_frames[(task, split)] = frame

                invalid_id_count = int(frame["patient_id"].isna().sum())
                duplicate_ids = sorted(
                    frame.loc[frame["patient_id"].duplicated(keep=False), "patient_id"]
                    .dropna()
                    .unique()
                    .tolist()
                )
                for patient_id in duplicate_ids:
                    self.add_patient_issue(
                        task, split, patient_id, "duplicate_patient_id", expected=1, actual=2
                    )
                self.check(
                    f"task:{task}:{split}",
                    "unique_valid_patient_ids",
                    invalid_id_count + len(duplicate_ids),
                    f"invalid_ids={invalid_id_count}, duplicate_patients={len(duplicate_ids)}",
                )

                wrong_split_rows = frame["split"].ne(split)
                for _, row in frame.loc[wrong_split_rows, ["patient_id", "split"]].iterrows():
                    self.add_patient_issue(
                        task,
                        split,
                        str(row["patient_id"]),
                        "wrong_split_value",
                        expected=split,
                        actual=row["split"],
                    )
                self.check(
                    f"task:{task}:{split}",
                    "split_column_values",
                    int(wrong_split_rows.sum()),
                    f"wrong_rows={int(wrong_split_rows.sum())}",
                )

                metadata_ids = set(self.metadata[split]["patient_id"])
                actual_ids = set(frame["patient_id"].dropna())
                wrong_membership = sorted(actual_ids - metadata_ids)
                for patient_id in wrong_membership:
                    other_split = "valid" if split == "train" else "train"
                    actual_membership = (
                        other_split
                        if patient_id in set(self.metadata[other_split]["patient_id"])
                        else "neither"
                    )
                    self.add_patient_issue(
                        task,
                        split,
                        patient_id,
                        "patient_not_in_expected_metadata_split",
                        expected=split,
                        actual=actual_membership,
                    )
                self.check(
                    f"task:{task}:{split}",
                    "patient_membership_in_metadata_split",
                    len(wrong_membership),
                    f"wrong_membership={len(wrong_membership)}",
                )

                expected_ids, unique_labels = self.expected_patient_ids(task, split)
                missing_patients = sorted(expected_ids - actual_ids)
                extra_patients = sorted(actual_ids - expected_ids)
                for patient_id in missing_patients:
                    self.add_patient_issue(
                        task,
                        split,
                        patient_id,
                        "missing_expected_patient",
                        expected=1,
                        actual=0,
                    )
                for patient_id in extra_patients:
                    label_values = self.label_values[split][label].get(patient_id, ())
                    details = f"source_label_values={join_values(label_values)}"
                    self.add_patient_issue(
                        task,
                        split,
                        patient_id,
                        "unexpected_patient",
                        expected=0,
                        actual=1,
                        details=details,
                    )
                self.check(
                    f"task:{task}:{split}",
                    "patient_set_equals_independent_derivation",
                    len(missing_patients) + len(extra_patients),
                    f"expected={len(expected_ids)}, actual={len(actual_ids)}, "
                    f"missing={len(missing_patients)}, extra={len(extra_patients)}",
                )

                label_numeric = pd.to_numeric(frame[label], errors="coerce")
                non_numeric_label = label_numeric.isna() & frame[label].notna()
                missing_label = frame[label].isna()
                infinite_label = np.isinf(label_numeric.fillna(0).to_numpy(dtype=float))
                invalid_range = label_numeric.notna() & ~label_numeric.isin(LABEL_RANGES[label])
                label_quality_failures = int(
                    non_numeric_label.sum()
                    + missing_label.sum()
                    + infinite_label.sum()
                    + invalid_range.sum()
                )
                if non_numeric_label.any():
                    ids = frame.loc[non_numeric_label, "patient_id"].astype(str).tolist()
                    self.numeric_issues.append(
                        {
                            "task": task,
                            "split": split,
                            "field_type": "label",
                            "column": label,
                            "issue_type": "non_numeric_string",
                            "count": len(ids),
                            "patient_ids": "|".join(ids),
                        }
                    )
                if missing_label.any():
                    ids = frame.loc[missing_label, "patient_id"].astype(str).tolist()
                    self.numeric_issues.append(
                        {
                            "task": task,
                            "split": split,
                            "field_type": "label",
                            "column": label,
                            "issue_type": "NaN",
                            "count": len(ids),
                            "patient_ids": "|".join(ids),
                        }
                    )
                if infinite_label.any():
                    ids = frame.loc[infinite_label, "patient_id"].astype(str).tolist()
                    self.numeric_issues.append(
                        {
                            "task": task,
                            "split": split,
                            "field_type": "label",
                            "column": label,
                            "issue_type": "Inf",
                            "count": len(ids),
                            "patient_ids": "|".join(ids),
                        }
                    )
                if invalid_range.any():
                    ids = frame.loc[invalid_range, "patient_id"].astype(str).tolist()
                    self.numeric_issues.append(
                        {
                            "task": task,
                            "split": split,
                            "field_type": "label",
                            "column": label,
                            "issue_type": "out_of_range",
                            "count": len(ids),
                            "patient_ids": "|".join(ids),
                        }
                    )
                self.check(
                    f"task:{task}:{split}",
                    "label_numeric_quality_and_range",
                    label_quality_failures,
                    f"missing={int(missing_label.sum())}, non_numeric={int(non_numeric_label.sum())}, "
                    f"inf={int(infinite_label.sum())}, out_of_range={int(invalid_range.sum())}",
                )

                label_mismatch_count = 0
                for _, row in frame.iterrows():
                    patient_id = row["patient_id"]
                    if patient_id is None:
                        continue
                    source_values = self.label_values[split][label].get(patient_id, ())
                    if len(source_values) != 1:
                        issue_type = "source_label_conflict" if len(source_values) > 1 else "source_label_missing"
                        self.label_issues.append(
                            {
                                "task": task,
                                "split": split,
                                "patient_id": patient_id,
                                "issue_type": issue_type,
                                "expected_label": join_values(source_values),
                                "actual_label": display_value(row[label]),
                                "details": "Task patient does not have one unique raw label.",
                            }
                        )
                        label_mismatch_count += 1
                        continue
                    actual_label = canonical_value(row[label])
                    if actual_label != source_values[0]:
                        self.label_issues.append(
                            {
                                "task": task,
                                "split": split,
                                "patient_id": patient_id,
                                "issue_type": "label_value_mismatch",
                                "expected_label": display_value(source_values[0]),
                                "actual_label": display_value(actual_label),
                                "details": LABEL_COLUMNS[label],
                            }
                        )
                        label_mismatch_count += 1
                self.check(
                    f"task:{task}:{split}",
                    "labels_equal_unique_raw_excel_labels",
                    label_mismatch_count,
                    f"mismatches={label_mismatch_count}",
                )

                structural = {"patient_id", "split", label}
                actual_features = [column for column in frame.columns if column not in structural]
                expected_features = select_expected_features(self.features[split].columns, mode)
                actual_features_by_task[(task, split)] = actual_features
                expected_features_by_task[(task, split)] = expected_features

                other_labels = (set(LABEL_COLUMNS) - {label}) & set(actual_features)
                allowed_prefixes = {
                    "pre": ("pre_",),
                    "post": ("post_",),
                    "prepost": ("pre_", "post_", "delta_"),
                }[mode]
                bad_prefix = [
                    column for column in actual_features if not column.startswith(allowed_prefixes)
                ]
                runtime_columns = [column for column in actual_features if "runtime_s" in column]
                pair_columns = [
                    column
                    for column in actual_features
                    if column in {"pre_n_pairs", "post_n_pairs"}
                ]
                reserved_columns = [
                    column
                    for column in actual_features
                    if column in {"patient_id", "split", *LABEL_COLUMNS.keys()}
                ]
                for column in sorted(set(bad_prefix)):
                    self.add_feature_issue(task, split, "disallowed_feature_prefix", column=column)
                for column in runtime_columns:
                    self.add_feature_issue(task, split, "runtime_feature_leakage", column=column)
                for column in pair_columns:
                    self.add_feature_issue(task, split, "pair_count_excluded_from_main_analysis", column=column)
                for column in sorted(set(reserved_columns) | set(other_labels)):
                    self.add_feature_issue(task, split, "reserved_or_label_column_as_feature", column=column)

                rule_failures = len(
                    set(bad_prefix) | set(runtime_columns) | set(pair_columns) | set(reserved_columns)
                )
                self.check(
                    f"task:{task}:{split}",
                    "feature_prefix_and_leakage_rules",
                    rule_failures,
                    f"bad_prefix={len(set(bad_prefix))}, runtime={len(runtime_columns)}, "
                    f"pair_count={len(pair_columns)}, reserved={len(set(reserved_columns))}",
                )

                missing_features = [column for column in expected_features if column not in actual_features]
                extra_features = [column for column in actual_features if column not in expected_features]
                order_mismatch = not missing_features and not extra_features and actual_features != expected_features
                for column in missing_features:
                    self.add_feature_issue(task, split, "missing_expected_feature", column=column)
                for column in extra_features:
                    self.add_feature_issue(task, split, "unexpected_feature", column=column)
                if order_mismatch:
                    first_position = next(
                        index
                        for index, (actual, expected) in enumerate(
                            zip(actual_features, expected_features)
                        )
                        if actual != expected
                    )
                    self.add_feature_issue(
                        task,
                        split,
                        "feature_order_mismatch",
                        column=actual_features[first_position],
                        related_column=expected_features[first_position],
                        position=first_position,
                    )
                self.check(
                    f"task:{task}:{split}",
                    "features_equal_independent_source_selection",
                    len(missing_features) + len(extra_features) + int(order_mismatch),
                    f"expected={len(expected_features)}, actual={len(actual_features)}, "
                    f"missing={len(missing_features)}, extra={len(extra_features)}, "
                    f"order_match={not order_mismatch}",
                )

                numeric_feature_failures = 0
                for column in actual_features:
                    numeric = pd.to_numeric(frame[column], errors="coerce")
                    non_numeric = frame[column].notna() & numeric.isna()
                    missing = frame[column].isna()
                    positive_inf = numeric.eq(np.inf)
                    negative_inf = numeric.eq(-np.inf)
                    issue_masks = {
                        "non_numeric_string": non_numeric,
                        "NaN": missing,
                        "positive_Inf": positive_inf,
                        "negative_Inf": negative_inf,
                    }
                    for issue_type, mask in issue_masks.items():
                        count = int(mask.sum())
                        if not count:
                            continue
                        self.numeric_issues.append(
                            {
                                "task": task,
                                "split": split,
                                "field_type": "feature",
                                "column": column,
                                "issue_type": issue_type,
                                "count": count,
                                "patient_ids": "|".join(
                                    frame.loc[mask, "patient_id"].astype(str).tolist()
                                ),
                            }
                        )
                        numeric_feature_failures += count
                self.check(
                    f"task:{task}:{split}",
                    "feature_nan_inf_and_numeric_strings",
                    numeric_feature_failures,
                    f"problem_cells={numeric_feature_failures}",
                )

                constant_columns = [
                    column for column in actual_features if frame[column].nunique(dropna=False) <= 1
                ]
                for column in constant_columns:
                    self.add_feature_issue(
                        task,
                        split,
                        "constant_feature",
                        column=column,
                        details=f"value={display_value(frame[column].iloc[0]) if len(frame) else ''}",
                    )
                duplicate_pairs = duplicate_feature_pairs(frame, actual_features)
                for left_column, right_column in duplicate_pairs:
                    self.add_feature_issue(
                        task,
                        split,
                        "duplicate_feature_column",
                        column=right_column,
                        related_column=left_column,
                    )
                self.check(
                    f"task:{task}:{split}",
                    "constant_and_duplicate_features",
                    len(constant_columns) + len(duplicate_pairs),
                    f"constant={len(constant_columns)}, duplicate_pairs={len(duplicate_pairs)}",
                )

                value_mismatch_count = 0
                if not duplicate_ids:
                    actual_indexed = frame.set_index("patient_id")
                    source_indexed = self.features[split].set_index("patient_id")
                    common_ids = sorted(expected_ids & actual_ids)
                    for column in [
                        name
                        for name in expected_features
                        if name in actual_features and name in source_indexed.columns
                    ]:
                        expected_series = source_indexed.loc[common_ids, column]
                        actual_series = actual_indexed.loc[common_ids, column]
                        equal = series_equal(expected_series, actual_series)
                        for position in np.flatnonzero(~equal):
                            patient_id = common_ids[int(position)]
                            self.value_mismatches.append(
                                {
                                    "scope": "task_feature",
                                    "task": task,
                                    "split": split,
                                    "patient_id": patient_id,
                                    "column": column,
                                    "expected": display_value(expected_series.iloc[position]),
                                    "actual": display_value(actual_series.iloc[position]),
                                    "issue_type": "task_vs_patient_source_value_mismatch",
                                }
                            )
                            value_mismatch_count += 1
                self.check(
                    f"task:{task}:{split}",
                    "feature_values_equal_patient_feature_source",
                    value_mismatch_count,
                    f"value_mismatches={value_mismatch_count}",
                )

                counts = label_numeric.value_counts(dropna=False).sort_index()
                for value, count in counts.items():
                    self.class_distribution.append(
                        {
                            "task": task,
                            "split": split,
                            "label": display_value(value) or "NaN",
                            "count": int(count),
                            "proportion": float(count / len(frame)) if len(frame) else np.nan,
                        }
                    )

                expected_target_rows = int(config["expected_rows"][split])
                expected_target_features = int(config["expected_features"])
                scale_failures = int(len(expected_ids) != expected_target_rows) + int(
                    len(frame) != expected_target_rows
                ) + int(len(actual_features) != expected_target_features)
                self.check(
                    f"task:{task}:{split}",
                    "expected_scale",
                    scale_failures,
                    f"derived_rows={len(expected_ids)}, actual_rows={len(frame)}, "
                    f"target_rows={expected_target_rows}, actual_features={len(actual_features)}, "
                    f"target_features={expected_target_features}",
                )

                self.task_metrics[(task, split)] = {
                    "derived_rows": len(expected_ids),
                    "actual_rows": len(frame),
                    "actual_features": len(actual_features),
                    "target_rows": expected_target_rows,
                    "target_features": expected_target_features,
                    "classes": ", ".join(
                        f"{display_value(value) or 'NaN'}:{int(count)}" for value, count in counts.items()
                    ),
                }

            train_features = actual_features_by_task.get((task, "train"), [])
            valid_features = actual_features_by_task.get((task, "valid"), [])
            exact_match = train_features == valid_features
            if not exact_match:
                self.add_feature_issue(
                    task,
                    "train|valid",
                    "train_valid_feature_columns_mismatch",
                    details=f"train={len(train_features)}, valid={len(valid_features)}",
                )
            self.check(
                f"task:{task}",
                "train_valid_feature_names_order_count",
                0 if exact_match else 1,
                f"train={len(train_features)}, valid={len(valid_features)}, exact_match={exact_match}",
            )

        conflict_presence: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        for task, config in TASKS.items():
            label = str(config["label"])
            for split in ["train", "valid"]:
                frame = self.task_frames.get((task, split))
                if frame is None or "patient_id" not in frame.columns:
                    continue
                ids = set(frame["patient_id"].dropna())
                for patient_id in ids:
                    conflict_presence[(split, label, patient_id)].append(task)
        for row in self.conflicts:
            row["relevant_task_presence"] = "|".join(
                sorted(conflict_presence.get((row["split"], row["label"], row["patient_id"]), []))
            )

    def write_reports(self) -> bool:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        write_table(self.summary, SUMMARY_COLUMNS, REPORT_DIR / "audit_summary.csv")
        write_table(
            self.patient_set_issues,
            PATIENT_SET_COLUMNS,
            REPORT_DIR / "patient_set_mismatches.csv",
        )
        write_table(
            self.label_issues,
            LABEL_MISMATCH_COLUMNS,
            REPORT_DIR / "label_mismatches.csv",
        )
        write_table(
            self.feature_issues,
            FEATURE_ISSUE_COLUMNS,
            REPORT_DIR / "feature_column_issues.csv",
        )
        write_table(self.conflicts, CONFLICT_COLUMNS, REPORT_DIR / "conflict_patients.csv")
        write_table(
            self.numeric_issues,
            NUMERIC_ISSUE_COLUMNS,
            REPORT_DIR / "numeric_quality_issues.csv",
        )
        write_table(
            self.class_distribution,
            CLASS_DISTRIBUTION_COLUMNS,
            REPORT_DIR / "class_distribution.csv",
        )
        write_table(
            self.value_mismatches,
            VALUE_MISMATCH_COLUMNS,
            REPORT_DIR / "feature_value_mismatches.csv",
        )

        failed = [row for row in self.summary if row["status"] == "FAIL"]
        report = self.render_markdown(failed)
        (REPORT_DIR / "audit_report.md").write_text(report, encoding="utf-8")
        return not failed

    def render_markdown(self, failed: list[dict[str, Any]]) -> str:
        overall = "PASS" if not failed else "FAIL"
        modeling = "可以开始建模" if not failed else "不可以开始建模"
        lines = [
            "# 独立任务数据审计报告",
            "",
            f"- 审计结论：**{overall}**",
            f"- 建模许可：**{modeling}**",
            f"- 失败检查数：{len(failed)}",
            f"- 原始标签冲突记录数（患者-标签）：{len(self.conflicts)}",
            "",
            "## 已检查文件",
            "",
        ]
        for path in self.checked_files:
            lines.append(f"- `{path.relative_to(PROJECT)}`")

        lines.extend(
            [
                "",
                "## ID、划分与标签",
                "",
                self.summary_sentence("metadata", "train_valid_patient_overlap", "Train/Valid 病案号交集"),
                f"- 患者集合问题：{len(self.patient_set_issues)} 条。",
                f"- 任务标签不匹配：{len(self.label_issues)} 条。",
                f"- 原始标签冲突：{len(self.conflicts)} 条患者-标签记录；冲突患者按对应任务标签唯一性规则处理。",
                "",
                "## 任务规模与类别分布",
                "",
                "| 任务 | 划分 | 重建患者 | 实际患者 | 特征数 | 目标规模 | 类别分布 |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for task in TASKS:
            for split in ["train", "valid"]:
                metrics = self.task_metrics.get((task, split), {})
                lines.append(
                    f"| {task} | {split} | {metrics.get('derived_rows', '')} | "
                    f"{metrics.get('actual_rows', '')} | {metrics.get('actual_features', '')} | "
                    f"{metrics.get('target_rows', '')}/{metrics.get('target_features', '')} | "
                    f"{metrics.get('classes', '')} |"
                )

        lines.extend(
            [
                "",
                "## 特征与数值质量",
                "",
                f"- 特征列问题：{len(self.feature_issues)} 条。",
                f"- NaN、Inf 或非数值问题：{sum(int(row['count']) for row in self.numeric_issues)} 个单元格/标签值。",
                f"- 特征值相对患者级源表不一致：{len(self.value_mismatches)} 条。",
                "- 已检查特征前缀、`runtime_s`、`pre_n_pairs`/`post_n_pairs`、结构列泄漏、Train/Valid 列顺序、常数列和完全重复列。",
                "",
                "## 全部问题",
                "",
            ]
        )
        if not failed:
            lines.append("未发现阻止建模的数据问题。")
        else:
            lines.extend(
                [
                    "| 范围 | 检查 | 问题数 | 详情 |",
                    "|---|---|---:|---|",
                ]
            )
            for row in failed:
                details = str(row["details"]).replace("|", "/")
                lines.append(
                    f"| {row['scope']} | {row['check']} | {row['issue_count']} | {details} |"
                )

        lines.extend(
            [
                "",
                "## 机器可读明细",
                "",
                "- `audit_summary.csv`：所有检查及 PASS/FAIL 状态",
                "- `patient_set_mismatches.csv`：缺少、多余、串集和重复患者",
                "- `label_mismatches.csv`：逐患者标签冲突或值不一致",
                "- `feature_column_issues.csv`：泄漏、列差异、常数及重复特征",
                "- `conflict_patients.csv`：从原始 Excel 独立识别的冲突患者",
                "- `numeric_quality_issues.csv`：NaN、Inf 和非数值字符串",
                "- `class_distribution.csv`：每个任务和划分的类别分布",
                "- `feature_value_mismatches.csv`：任务特征对患者级源特征的值差异",
                "",
            ]
        )
        return "\n".join(lines)

    def summary_sentence(self, scope: str, check: str, label: str) -> str:
        for row in self.summary:
            if row["scope"] == scope and row["check"] == check:
                return f"- {label}：{row['status']}（{row['details']}）。"
        return f"- {label}：未完成。"


def write_failure_reports(exc: BaseException) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    detail = f"{type(exc).__name__}: {exc}"
    write_table(
        [
            {
                "scope": "audit_runtime",
                "check": "audit_completed",
                "status": "FAIL",
                "issue_count": 1,
                "details": detail,
            }
        ],
        SUMMARY_COLUMNS,
        REPORT_DIR / "audit_summary.csv",
    )
    for columns, filename in [
        (PATIENT_SET_COLUMNS, "patient_set_mismatches.csv"),
        (LABEL_MISMATCH_COLUMNS, "label_mismatches.csv"),
        (FEATURE_ISSUE_COLUMNS, "feature_column_issues.csv"),
        (CONFLICT_COLUMNS, "conflict_patients.csv"),
        (NUMERIC_ISSUE_COLUMNS, "numeric_quality_issues.csv"),
        (CLASS_DISTRIBUTION_COLUMNS, "class_distribution.csv"),
        (VALUE_MISMATCH_COLUMNS, "feature_value_mismatches.csv"),
    ]:
        write_table([], columns, REPORT_DIR / filename)
    report = "\n".join(
        [
            "# 独立任务数据审计报告",
            "",
            "- 审计结论：**FAIL**",
            "- 建模许可：**不可以开始建模**",
            f"- 审计运行错误：`{detail}`",
            "",
        ]
    )
    (REPORT_DIR / "audit_report.md").write_text(report, encoding="utf-8")


def main() -> int:
    audit = Audit()
    try:
        audit.load_metadata()
        audit.load_features()
        audit.audit_existing_conflicts()
        audit.audit_tasks()
        passed = audit.write_reports()
    except Exception as exc:  # Ensure required failure artifacts exist for CI use.
        write_failure_reports(exc)
        traceback.print_exc()
        print(f"AUDIT FAIL: {exc}", file=sys.stderr)
        return 2

    failed_count = sum(row["status"] == "FAIL" for row in audit.summary)
    print(f"Audit reports: {REPORT_DIR}")
    print(f"Checks: {len(audit.summary)}, failed: {failed_count}")
    print("AUDIT PASS" if passed else "AUDIT FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

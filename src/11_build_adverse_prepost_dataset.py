#!/usr/bin/env python3
"""Build the adverse Pre+Post datasets without training a model.

The adverse Pre task tables are authoritative for the patient cohort, split,
label, and 48 Pre features. Split-specific patient-flow tables only contribute
the 48 core Post features after a strict one-to-one patient_id match.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/root/autodl-tmp/aneurysm")
TASK_DIR = PROJECT_ROOT / "outputs" / "task_datasets"
FLOW_DIR = PROJECT_ROOT / "outputs" / "features"
REPORT_DIR = PROJECT_ROOT / "reports" / "adverse_prepost_mlp_v1"

TRAIN_TASK_PATH = TASK_DIR / "adverse_pre_train.csv"
VALID_TASK_PATH = TASK_DIR / "adverse_pre_valid.csv"
TRAIN_OUTPUT_PATH = TASK_DIR / "adverse_prepost_train.csv"
VALID_OUTPUT_PATH = TASK_DIR / "adverse_prepost_valid.csv"
REPORT_PATH = REPORT_DIR / "data_build_report.md"

EXPECTED_ROWS = {"train": 794, "valid": 209}
EXPECTED_POSITIVES = {"train": 132, "valid": 36}
EXPECTED_FEATURE_COUNT = 48
EXPECTED_OUTPUT_COLUMNS = 99


class BuildError(RuntimeError):
    """Raised when any required data-build assertion fails."""


@dataclass(frozen=True)
class FileInfo:
    path: Path
    sha256: str
    rows: int
    columns: int
    unique_patient_ids: int
    modified_utc: str


@dataclass
class SplitResult:
    split: str
    task_path: Path
    flow_path: Path
    task_info: FileInfo
    flow_info: FileInfo
    output_path: Path
    output: pd.DataFrame
    pre_columns: list[str]
    post_columns: list[str]
    matched_patients: int
    positive_count: int
    pre_allclose: bool
    assertions: list[str]


def fail(message: str) -> None:
    raise BuildError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_patient_id(value: object) -> str:
    if pd.isna(value):
        fail("patient_id contains a missing value")
    normalized = str(value).strip()
    if not normalized:
        fail("patient_id contains an empty value after trimming whitespace")
    if re.fullmatch(r"[+-]?\d+\.0+", normalized):
        normalized = normalized[: normalized.index(".")]
        if normalized.startswith("+"):
            normalized = normalized[1:]
    return normalized


def normalize_patient_ids(series: pd.Series, context: str) -> pd.Series:
    try:
        normalized = series.map(normalize_patient_id)
    except BuildError as exc:
        fail(f"{context}: {exc}")
    duplicates = normalized[normalized.duplicated(keep=False)]
    if not duplicates.empty:
        examples = sorted(duplicates.unique().tolist())[:10]
        fail(
            f"{context}: patient_id is not unique after normalization; "
            f"duplicate examples={examples}"
        )
    return normalized


def read_csv_strict(path: Path, role: str) -> pd.DataFrame:
    if not path.is_file():
        fail(f"{role}: input file does not exist: {path}")
    try:
        frame = pd.read_csv(path, dtype={"patient_id": "string"})
    except Exception as exc:  # pandas supplies the useful parser detail.
        fail(f"{role}: failed to read CSV {path}: {exc}")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        fail(f"{role}: duplicate CSV column names: {duplicates}")
    if "patient_id" not in frame.columns:
        fail(f"{role}: required column patient_id is absent")
    return frame


def core_pre_columns(columns: Iterable[str]) -> list[str]:
    return [
        column
        for column in columns
        if column.startswith("pre_")
        and column != "pre_n_pairs"
        and "runtime_s" not in column
    ]


def core_post_columns(columns: Iterable[str]) -> list[str]:
    return [
        column
        for column in columns
        if column.startswith("post_")
        and column != "post_n_pairs"
        and "runtime_s" not in column
    ]


def delta_columns(columns: Iterable[str]) -> list[str]:
    return [column for column in columns if column.startswith("delta_")]


def file_info(path: Path, frame: pd.DataFrame, normalized_ids: pd.Series) -> FileInfo:
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return FileInfo(
        path=path.resolve(),
        sha256=sha256_file(path),
        rows=int(frame.shape[0]),
        columns=int(frame.shape[1]),
        unique_patient_ids=int(normalized_ids.nunique(dropna=False)),
        modified_utc=modified.isoformat(),
    )


def candidate_description(path: Path, expected_keys: set[str]) -> tuple[str, bool]:
    try:
        frame = read_csv_strict(path, f"flow candidate {path.name}")
        keys = normalize_patient_ids(frame["patient_id"], f"flow candidate {path.name}")
        pre_count = len(core_pre_columns(frame.columns))
        post_count = len(core_post_columns(frame.columns))
        delta_count = len(delta_columns(frame.columns))
        info = file_info(path, frame, keys)
        covers_task = expected_keys.issubset(set(keys))
        structurally_valid = (
            info.rows == info.unique_patient_ids
            and pre_count == EXPECTED_FEATURE_COUNT
            and post_count == EXPECTED_FEATURE_COUNT
            and delta_count == EXPECTED_FEATURE_COUNT
            and covers_task
        )
        description = (
            f"{info.path} | rows={info.rows} | unique_patient_id="
            f"{info.unique_patient_ids} | core_pre={pre_count} | "
            f"core_post={post_count} | delta={delta_count} | "
            f"covers_task={covers_task} | sha256={info.sha256} | "
            f"modified_utc={info.modified_utc}"
        )
        return description, structurally_valid
    except BuildError as exc:
        stat = path.stat()
        digest = sha256_file(path)
        modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
        return (
            f"{path.resolve()} | inspection_error={exc} | sha256={digest} | "
            f"modified_utc={modified}",
            False,
        )


def select_flow_path(pattern: str, split: str, expected_keys: set[str]) -> Path:
    candidates = sorted(path.resolve() for path in FLOW_DIR.glob(pattern) if path.is_file())
    if not candidates:
        fail(f"{split}: no flow feature candidate matched {FLOW_DIR / pattern}")
    if len(candidates) == 1:
        return candidates[0]

    inspected = [candidate_description(path, expected_keys) for path in candidates]
    valid_candidates = [
        path for path, (_, valid) in zip(candidates, inspected, strict=True) if valid
    ]
    if len(valid_candidates) == 1:
        return valid_candidates[0]

    details = "\n".join(f"  - {description}" for description, _ in inspected)
    fail(
        f"{split}: unable to uniquely select a flow feature table from "
        f"{len(candidates)} candidates; refusing to guess. Candidates:\n{details}"
    )


def numeric_matrix(frame: pd.DataFrame, columns: list[str], context: str) -> np.ndarray:
    try:
        numeric = frame.loc[:, columns].apply(pd.to_numeric, errors="raise")
    except Exception as exc:
        fail(f"{context}: feature conversion to numeric failed: {exc}")
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        bad_positions = np.argwhere(~np.isfinite(values))
        examples = [
            {
                "row": int(row),
                "column": columns[int(column)],
                "value": repr(values[int(row), int(column)]),
            }
            for row, column in bad_positions[:10]
        ]
        fail(f"{context}: features contain NaN or positive/negative infinity: {examples}")
    return values


def validate_task_table(
    path: Path, split: str
) -> tuple[pd.DataFrame, pd.Series, list[str], FileInfo, list[str]]:
    assertions: list[str] = []
    frame = read_csv_strict(path, f"{split} task table")
    required = {"patient_id", "split", "adverse"}
    missing = sorted(required - set(frame.columns))
    if missing:
        fail(f"{split} task table: missing required columns {missing}")

    keys = normalize_patient_ids(frame["patient_id"], f"{split} task table")
    assertions.append("PASS: normalized patient_id values are nonempty and unique")

    pre_columns = core_pre_columns(frame.columns)
    if len(pre_columns) != EXPECTED_FEATURE_COUNT:
        fail(
            f"{split} task table: expected 48 core Pre features, found "
            f"{len(pre_columns)}: {pre_columns}"
        )
    assertions.append("PASS: task table contains exactly 48 core Pre features")
    numeric_matrix(frame, pre_columns, f"{split} task Pre features")
    assertions.append("PASS: all task Pre feature values are finite numeric values")

    if len(frame) != EXPECTED_ROWS[split]:
        fail(
            f"{split} task table: expected {EXPECTED_ROWS[split]} rows, "
            f"found {len(frame)}"
        )
    assertions.append(f"PASS: task table has exactly {EXPECTED_ROWS[split]} rows")

    split_values = set(frame["split"].astype(str))
    if split_values != {split}:
        fail(f"{split} task table: split values are {sorted(split_values)}, expected only {split}")
    assertions.append(f"PASS: every task split value is {split}")

    try:
        labels = pd.to_numeric(frame["adverse"], errors="raise")
    except Exception as exc:
        fail(f"{split} task table: adverse labels are not numeric: {exc}")
    label_values = set(labels.unique().tolist())
    if not label_values.issubset({0, 1}):
        fail(f"{split} task table: adverse labels are not binary 0/1: {label_values}")
    positives = int((labels == 1).sum())
    if positives != EXPECTED_POSITIVES[split]:
        fail(
            f"{split} task table: expected {EXPECTED_POSITIVES[split]} positive labels, "
            f"found {positives}"
        )
    assertions.append(
        f"PASS: adverse labels are unchanged binary values with {positives} positives"
    )

    return frame, keys, pre_columns, file_info(path, frame, keys), assertions


def build_split(task_path: Path, flow_pattern: str, split: str, output_path: Path) -> SplitResult:
    task, task_keys, pre_columns, task_info, assertions = validate_task_table(task_path, split)
    flow_path = select_flow_path(flow_pattern, split, set(task_keys))
    flow = read_csv_strict(flow_path, f"{split} flow table")
    flow_keys = normalize_patient_ids(flow["patient_id"], f"{split} flow table")
    flow_input_info = file_info(flow_path, flow, flow_keys)
    assertions.append("PASS: flow-table normalized patient_id values are nonempty and unique")

    if "split" in flow.columns:
        flow_split_values = set(flow["split"].astype(str))
        if flow_split_values != {split}:
            fail(
                f"{split} flow table: split values are {sorted(flow_split_values)}, "
                f"expected only {split}"
            )
        assertions.append(f"PASS: every flow-table split value is {split}")

    flow_pre_columns = core_pre_columns(flow.columns)
    post_columns = core_post_columns(flow.columns)
    if len(flow_pre_columns) != EXPECTED_FEATURE_COUNT:
        fail(
            f"{split} flow table: expected 48 core Pre features, found "
            f"{len(flow_pre_columns)}: {flow_pre_columns}"
        )
    if len(post_columns) != EXPECTED_FEATURE_COUNT:
        fail(
            f"{split} flow table: expected 48 core Post features, found "
            f"{len(post_columns)}: {post_columns}"
        )
    assertions.append("PASS: flow table contains exactly 48 core Pre features")
    assertions.append("PASS: flow table contains exactly 48 core Post features")

    if set(flow_pre_columns) != set(pre_columns):
        fail(
            f"{split}: task/flow core Pre names differ; "
            f"task_only={sorted(set(pre_columns) - set(flow_pre_columns))}, "
            f"flow_only={sorted(set(flow_pre_columns) - set(pre_columns))}"
        )
    assertions.append("PASS: task and flow tables have identical core Pre feature names")

    flow = flow.copy()
    flow["_patient_id_key"] = flow_keys
    task = task.copy()
    task["_patient_id_key"] = task_keys
    task["_task_row_order"] = np.arange(len(task), dtype=np.int64)

    flow_pre_renames = {column: f"_flow_{column}" for column in pre_columns}
    flow_for_join = flow.loc[:, ["_patient_id_key", *pre_columns, *post_columns]].rename(
        columns=flow_pre_renames
    )
    try:
        merged = task.merge(
            flow_for_join,
            on="_patient_id_key",
            how="left",
            sort=False,
            validate="one_to_one",
            indicator=True,
        )
    except Exception as exc:
        fail(f"{split}: one-to-one patient join failed: {exc}")

    unmatched = merged.loc[merged["_merge"] != "both", "_patient_id_key"].tolist()
    if unmatched:
        fail(
            f"{split}: {len(unmatched)} task patient_id values were not found in the "
            f"split-specific flow table; examples={unmatched[:10]}"
        )
    matched_patients = int((merged["_merge"] == "both").sum())
    if matched_patients != len(task):
        fail(
            f"{split}: matched {matched_patients} patients but task table has {len(task)} rows"
        )
    assertions.append(
        f"PASS: one-to-one join matched all {matched_patients} task patients without deletion"
    )

    if not np.array_equal(merged["_task_row_order"].to_numpy(), np.arange(len(task))):
        fail(f"{split}: the one-to-one join changed task-table row order")
    assertions.append("PASS: task-table row order is preserved")

    task_pre_values = numeric_matrix(merged, pre_columns, f"{split} task Pre features after join")
    flow_pre_joined_columns = [flow_pre_renames[column] for column in pre_columns]
    flow_pre_values = numeric_matrix(
        merged, flow_pre_joined_columns, f"{split} matched flow Pre features"
    )
    post_values = numeric_matrix(merged, post_columns, f"{split} matched flow Post features")
    assertions.append("PASS: all matched flow Pre and Post feature values are finite")

    pre_allclose = bool(
        np.allclose(
            task_pre_values,
            flow_pre_values,
            rtol=1e-7,
            atol=1e-9,
            equal_nan=False,
        )
    )
    if not pre_allclose:
        close_mask = np.isclose(
            task_pre_values,
            flow_pre_values,
            rtol=1e-7,
            atol=1e-9,
            equal_nan=False,
        )
        bad_positions = np.argwhere(~close_mask)
        examples = [
            {
                "patient_id": merged.iloc[int(row)]["_patient_id_key"],
                "feature": pre_columns[int(column)],
                "task": float(task_pre_values[int(row), int(column)]),
                "flow": float(flow_pre_values[int(row), int(column)]),
            }
            for row, column in bad_positions[:10]
        ]
        fail(
            f"{split}: core Pre features are inconsistent under "
            f"np.allclose(rtol=1e-7, atol=1e-9, equal_nan=False); examples={examples}"
        )
    assertions.append(
        "PASS: all 48 task Pre features equal matched flow Pre features under "
        "np.allclose(rtol=1e-7, atol=1e-9, equal_nan=False)"
    )

    output = pd.DataFrame(
        {
            "patient_id": merged["_patient_id_key"],
            "split": merged["split"],
            "adverse": merged["adverse"],
        }
    )
    for index, column in enumerate(pre_columns):
        output[column] = task_pre_values[:, index]
    for index, column in enumerate(post_columns):
        output[column] = post_values[:, index]

    expected_order = ["patient_id", "split", "adverse", *pre_columns, *post_columns]
    if output.columns.tolist() != expected_order:
        fail(f"{split}: output column order is not the required strict order")
    if output.shape != (EXPECTED_ROWS[split], EXPECTED_OUTPUT_COLUMNS):
        fail(
            f"{split}: expected output shape "
            f"{EXPECTED_ROWS[split]}x{EXPECTED_OUTPUT_COLUMNS}, found "
            f"{output.shape[0]}x{output.shape[1]}"
        )
    assertions.append(
        f"PASS: output shape is {output.shape[0]} rows x {output.shape[1]} columns"
    )

    if output["patient_id"].duplicated().any():
        fail(f"{split}: output patient_id is not unique")
    assertions.append("PASS: output patient_id is unique")

    if set(output["split"].astype(str)) != {split}:
        fail(f"{split}: output split column is not uniformly {split}")
    assertions.append(f"PASS: output split is uniformly {split}")

    output_labels = pd.to_numeric(output["adverse"], errors="raise")
    positive_count = int((output_labels == 1).sum())
    if positive_count != EXPECTED_POSITIVES[split]:
        fail(
            f"{split}: expected {EXPECTED_POSITIVES[split]} output positives, "
            f"found {positive_count}"
        )
    if not output["adverse"].reset_index(drop=True).equals(
        task["adverse"].reset_index(drop=True)
    ):
        fail(f"{split}: adverse labels changed during construction")
    assertions.append(
        f"PASS: labels are unchanged and output has {positive_count} positive patients"
    )

    forbidden_columns = [
        column
        for column in output.columns
        if column.startswith("delta_")
        or column in {"pre_n_pairs", "post_n_pairs", "missing_pre", "missing_post"}
        or "runtime_s" in column
        or column.startswith("Unnamed:")
    ]
    if forbidden_columns:
        fail(f"{split}: output contains prohibited columns: {forbidden_columns}")
    assertions.append(
        "PASS: output contains no delta, pair-count, runtime, missing-flag, or CSV-index columns"
    )

    numeric_matrix(output, [*pre_columns, *post_columns], f"{split} final output features")
    assertions.append("PASS: all 96 final output feature values are finite")

    return SplitResult(
        split=split,
        task_path=task_path.resolve(),
        flow_path=flow_path.resolve(),
        task_info=task_info,
        flow_info=flow_input_info,
        output_path=output_path.resolve(),
        output=output,
        pre_columns=pre_columns,
        post_columns=post_columns,
        matched_patients=matched_patients,
        positive_count=positive_count,
        pre_allclose=pre_allclose,
        assertions=assertions,
    )


def markdown_numbered(columns: list[str]) -> str:
    return "\n".join(f"{index}. `{column}`" for index, column in enumerate(columns, start=1))


def input_table(results: list[SplitResult]) -> str:
    rows = [
        "| Role | Absolute path | SHA256 | Rows | Columns | Unique patient_id | Modified UTC |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for result in results:
        for role, info in ((f"{result.split} task", result.task_info), (f"{result.split} flow", result.flow_info)):
            rows.append(
                f"| {role} | `{info.path}` | `{info.sha256}` | {info.rows} | "
                f"{info.columns} | {info.unique_patient_ids} | {info.modified_utc} |"
            )
    return "\n".join(rows)


def output_table(results: list[SplitResult]) -> str:
    rows = [
        "| Split | Absolute output path | Matched patients | Rows | Columns | Positives | Pre allclose |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for result in results:
        rows.append(
            f"| {result.split} | `{result.output_path}` | {result.matched_patients} | "
            f"{result.output.shape[0]} | {result.output.shape[1]} | "
            f"{result.positive_count} | {result.pre_allclose} |"
        )
    return "\n".join(rows)


def build_report(results: list[SplitResult], cross_split_assertions: list[str]) -> str:
    train, valid = results
    if train.pre_columns != valid.pre_columns:
        fail("Train and Valid core Pre column order differs")
    if train.post_columns != valid.post_columns:
        fail("Train and Valid core Post column order differs")

    assertion_sections = []
    for result in results:
        assertion_sections.append(
            f"### {result.split.capitalize()}\n\n"
            + "\n".join(f"- {assertion}" for assertion in result.assertions)
        )
    assertion_sections.append(
        "### Cross-split and write safety\n\n"
        + "\n".join(f"- {assertion}" for assertion in cross_split_assertions)
    )

    return f"""# Adverse Pre+Post data build report

- Build status: SUCCESS
- Generated UTC: {datetime.now(timezone.utc).isoformat()}
- Builder: `{Path(__file__).resolve()}`
- Model training performed: No

## Actual input files

{input_table(results)}

## Matching and outputs

{output_table(results)}

Train and Valid task patients were matched only to their corresponding split-specific flow table. No patients or labels were removed or changed.

## Core Pre features (48)

{markdown_numbered(train.pre_columns)}

## Core Post features (48)

{markdown_numbered(train.post_columns)}

## Pre consistency check

- Comparison: `np.allclose(rtol=1e-7, atol=1e-9, equal_nan=False)`
- Train result: `{train.pre_allclose}` for all {train.matched_patients} matched patients and all 48 Pre features.
- Valid result: `{valid.pre_allclose}` for all {valid.matched_patients} matched patients and all 48 Pre features.

## Assertion results

{os.linesep.join(assertion_sections)}
"""


def write_csv_exclusive(frame: pd.DataFrame, path: Path) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
    except FileExistsError:
        fail(f"refusing to overwrite existing output file: {path}")


def write_text_exclusive(text: str, path: Path) -> None:
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
    except FileExistsError:
        fail(f"refusing to overwrite existing report file: {path}")


def main() -> int:
    protected_outputs = [TRAIN_OUTPUT_PATH, VALID_OUTPUT_PATH, REPORT_PATH]
    existing = [path.resolve() for path in protected_outputs if path.exists()]
    if existing:
        fail(f"refusing to overwrite existing build artifacts: {existing}")

    train = build_split(
        TRAIN_TASK_PATH,
        "train_patient_flow_features*.csv",
        "train",
        TRAIN_OUTPUT_PATH,
    )
    valid = build_split(
        VALID_TASK_PATH,
        "valid_patient_flow_features*.csv",
        "valid",
        VALID_OUTPUT_PATH,
    )

    cross_split_assertions = [
        "PASS: all three intended build artifacts were absent before construction"
    ]
    train_ids = set(train.output["patient_id"])
    valid_ids = set(valid.output["patient_id"])
    overlap = sorted(train_ids & valid_ids)
    if overlap:
        fail(
            f"Train and Valid outputs share {len(overlap)} patient_id values; "
            f"examples={overlap[:10]}"
        )
    cross_split_assertions.append("PASS: Train/Valid normalized patient_id intersection is 0")

    if train.pre_columns != valid.pre_columns:
        fail("Train and Valid output Pre feature names/order differ")
    if train.post_columns != valid.post_columns:
        fail("Train and Valid output Post feature names/order differ")
    cross_split_assertions.append("PASS: Train/Valid Pre feature names and order are identical")
    cross_split_assertions.append("PASS: Train/Valid Post feature names and order are identical")

    report = build_report([train, valid], cross_split_assertions)

    # Recheck immediately before exclusive creation to avoid silent overwrite.
    existing = [path.resolve() for path in protected_outputs if path.exists()]
    if existing:
        fail(f"build artifact appeared during validation; refusing to overwrite: {existing}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv_exclusive(train.output, TRAIN_OUTPUT_PATH)
    write_csv_exclusive(valid.output, VALID_OUTPUT_PATH)
    write_text_exclusive(report, REPORT_PATH)

    print(f"TRAIN_FLOW={train.flow_path}")
    print(f"VALID_FLOW={valid.flow_path}")
    print(f"TRAIN_MATCHED={train.matched_patients}/{EXPECTED_ROWS['train']}")
    print(f"VALID_MATCHED={valid.matched_patients}/{EXPECTED_ROWS['valid']}")
    print(f"TRAIN_PRE_ALLCLOSE={train.pre_allclose}")
    print(f"VALID_PRE_ALLCLOSE={valid.pre_allclose}")
    print(f"TRAIN_OUTPUT_SHAPE={train.output.shape[0]}x{train.output.shape[1]}")
    print(f"VALID_OUTPUT_SHAPE={valid.output.shape[0]}x{valid.output.shape[1]}")
    print(f"TRAIN_POSITIVES={train.positive_count}")
    print(f"VALID_POSITIVES={valid.positive_count}")
    print("POST_FEATURES_PER_SPLIT=48")
    print("PROHIBITED_OUTPUT_COLUMNS=NONE")
    print(f"REPORT={REPORT_PATH.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        raise SystemExit(1)

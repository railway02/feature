#!/usr/bin/env python3
"""Strict validation for matched-ablation tasks before any model is run."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from matched_common import atomic_json, read_npz, sha256_file


ROOT = Path("/root/autodl-tmp/aneurysm")
BASE_TASK = (
    ROOT
    / "outputs/api_gtmask_roi_cave_v5_fullmask_fullseq"
    / "adverse_prepost_series_task_v3"
)
TASK_ROOT = (
    ROOT
    / "outputs/api_gtmask_roi_cave_v5_fullmask_fullseq"
    / "adverse_prepost_matched_ablation_task_v1"
)
REPORT_DIR = ROOT / "reports/api_gtmask_roi_cave_v5_fullmask_fullseq"

IDENTITY_KEYS = ("series_uid", "patient_id", "target")
EXPECTED = {
    "M0": {"mask_features": 36},
    "W0": {"whole_deep": 10240},
    "WL": {"whole_deep": 10240, "local_deep": 10240},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-task", type=Path, default=BASE_TASK)
    parser.add_argument("--task-root", type=Path, default=TASK_ROOT)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    return parser.parse_args()


def _as_text(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).astype(str)


def _assert_identity(
    base: Dict[str, np.ndarray], candidate: Dict[str, np.ndarray], context: str
) -> None:
    for key in IDENTITY_KEYS:
        if key not in candidate:
            raise AssertionError(f"{context}: missing identity key {key}")
        if not np.array_equal(_as_text(base[key]), _as_text(candidate[key])):
            raise AssertionError(f"{context}: {key} differs from the fixed Local task")
    if "fold" in base:
        if "fold" not in candidate:
            raise AssertionError(f"{context}: missing identity key fold")
        if not np.array_equal(
            np.asarray(base["fold"], dtype=np.int64),
            np.asarray(candidate["fold"], dtype=np.int64),
        ):
            raise AssertionError(f"{context}: fold differs from the fixed Local task")


def _assert_group_folds(data: Dict[str, np.ndarray], context: str) -> None:
    frame = pd.DataFrame(
        {
            "patient_id": _as_text(data["patient_id"]),
            "fold": np.asarray(data["fold"], dtype=np.int64),
        }
    )
    fold_counts = frame.groupby("patient_id")["fold"].nunique()
    if int(fold_counts.max()) != 1:
        raise AssertionError(f"{context}: a patient spans multiple outer folds")


def validate(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.task_root.is_dir():
        raise FileNotFoundError(args.task_root)

    base = {
        split: read_npz(args.base_task / f"{split}_features.npz")
        for split in ("train", "valid")
    }
    if len(base["train"]["series_uid"]) != 908 or len(base["valid"]["series_uid"]) != 239:
        raise AssertionError("fixed Local task is not the expected 908/239 cohort")
    if set(_as_text(base["train"]["series_uid"])) & set(
        _as_text(base["valid"]["series_uid"])
    ):
        raise AssertionError("fixed Train and Valid series overlap")

    results: Dict[str, Any] = {}
    printed_rows: List[Dict[str, Any]] = []
    for experiment, feature_spec in EXPECTED.items():
        experiment_result: Dict[str, Any] = {}
        for split in ("train", "valid"):
            path = args.task_root / experiment / f"{split}_features.npz"
            data = read_npz(path)
            _assert_identity(base[split], data, f"{experiment}/{split}")
            if split == "train":
                _assert_group_folds(data, f"{experiment}/{split}")

            n = len(data["series_uid"])
            nonfinite = 0
            shapes: Dict[str, List[int]] = {}
            for feature_name, width in feature_spec.items():
                values = np.asarray(data[feature_name])
                if values.shape != (n, width):
                    raise AssertionError(
                        f"{experiment}/{split}: {feature_name} shape {values.shape}, "
                        f"expected {(n, width)}"
                    )
                nonfinite += int((~np.isfinite(values)).sum())
                shapes[feature_name] = list(values.shape)
            if nonfinite:
                raise AssertionError(f"{experiment}/{split}: {nonfinite} non-finite values")

            experiment_result[split] = {
                "samples": n,
                "feature_shapes": shapes,
                "uid_rowwise_equal": True,
                "target_rowwise_equal": True,
                "patient_rowwise_equal": True,
                "fold_rowwise_equal": True if split == "train" else None,
                "nonfinite_values": nonfinite,
                "sha256": sha256_file(path),
            }
            printed_rows.append(
                {
                    "experiment": experiment,
                    "split": split,
                    "samples": n,
                    "feature_shape": "; ".join(
                        f"{name}={tuple(shape)}" for name, shape in shapes.items()
                    ),
                    "uid_equal": True,
                    "target_equal": True,
                    "patient_equal": True,
                    "fold_equal": True if split == "train" else "N/A",
                    "nonfinite": nonfinite,
                }
            )
        results[experiment] = experiment_result

    coverage_path = args.task_root / "coverage.csv"
    coverage = pd.read_csv(coverage_path, dtype=str, keep_default_na=False)
    expected_phases = 2 * (908 + 239)
    if len(coverage) != expected_phases:
        raise AssertionError(
            f"coverage has {len(coverage)} rows, expected {expected_phases} phase rows"
        )
    required_flags = (
        "morphology_available",
        "whole_metadata_available",
        "whole_embedding_available",
        "local_metadata_available",
        "frame_list_hash_equal",
        "temporal_indices_equal",
    )
    temporal_result: Dict[str, Any] = {"phase_rows": len(coverage)}
    for column in required_flags:
        if column not in coverage.columns:
            raise AssertionError(f"coverage missing {column}")
        false_count = int(
            (~coverage[column].isin(("1", "1.0", "true", "True"))).sum()
        )
        if false_count:
            raise AssertionError(f"coverage {column} has {false_count} failures")
        temporal_result[column] = {"pass": len(coverage), "fail": 0}

    missing_path = args.task_root / "missing_or_mismatch.csv"
    missing = pd.read_csv(missing_path, dtype=str, keep_default_na=False)
    if len(missing):
        raise AssertionError(f"missing_or_mismatch.csv contains {len(missing)} rows")

    summary: Dict[str, Any] = {
        "status": "PASS",
        "fixed_cohort": {"train": 908, "valid": 239},
        "identity_keys": [*IDENTITY_KEYS, "fold (Train only)"],
        "experiments": results,
        "whole_local_temporal_validation": temporal_result,
        "missing_or_mismatch_rows": 0,
        "formal_training_started": False,
    }
    atomic_json(summary, args.task_root / "task_validation.json")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(summary, args.report_dir / "MATCHED_ABLATION_TASK_VALIDATION_V1.json")

    print(pd.DataFrame(printed_rows).to_string(index=False))
    print(
        "Whole/Local temporal verification: "
        f"{len(coverage)}/{len(coverage)} phase rows matched "
        "frame_list_hash and temporal-view indices"
    )
    print("Task validation: PASS")
    return summary


def main() -> None:
    validate(parse_args())


if __name__ == "__main__":
    main()

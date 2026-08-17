#!/usr/bin/env python3
"""Paired uncertainty audit for the pure-image key-feature MLP probe."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


SEED = 20260721
LINEAR_SUFFIX = {
    "cave_embedding_key": "cave_embedding",
    "cave_scalar_key": "cave_scalar",
    "sea_full_key": "sea_full",
    "cave_embedding_sea_key": "cave_embedding_sea_full",
    "cave_all_sea_key": "cave_all_sea_full",
}


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


def align(reference: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    reference_ids = reference["patient_id"].astype(str).tolist()
    other = other.copy()
    other["patient_id"] = other["patient_id"].astype(str)
    if set(reference_ids) != set(other["patient_id"]):
        raise AssertionError("Prediction patient IDs differ")
    aligned = other.set_index("patient_id").loc[reference_ids].reset_index()
    if not np.array_equal(reference["target"].to_numpy(), aligned["target"].to_numpy()):
        raise AssertionError("Prediction targets differ")
    return aligned


def stratified_indices(y: np.ndarray, repeats: int) -> list[np.ndarray]:
    rng = np.random.default_rng(SEED)
    negative = np.flatnonzero(y == 0)
    positive = np.flatnonzero(y == 1)
    return [
        np.concatenate([
            rng.choice(negative, len(negative), replace=True),
            rng.choice(positive, len(positive), replace=True),
        ])
        for _ in range(repeats)
    ]


def metrics(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    return float(roc_auc_score(y, probability)), float(average_precision_score(y, probability))


def percentile(values: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def comparison_rows(
    mlp: pd.DataFrame,
    sparse: pd.DataFrame,
    linear: pd.DataFrame,
    split: str,
    repeats: int,
) -> list[dict[str, Any]]:
    sparse = align(mlp, sparse)
    linear = align(mlp, linear)
    y = mlp["target"].to_numpy(dtype=np.int64)
    samples = stratified_indices(y, repeats)
    rows: list[dict[str, Any]] = []
    for variant, linear_suffix in LINEAR_SUFFIX.items():
        mlp_name = f"KeyMLP_{variant}"
        mlp_probability = mlp[f"{mlp_name.lower()}_probability"].to_numpy(dtype=np.float64)
        comparators = {
            f"StableSparse_{variant}": sparse[
                f"stablesparse_{variant}_probability"
            ].to_numpy(dtype=np.float64),
            f"LinearProbe_{linear_suffix}": linear[
                f"linearprobe_{linear_suffix}_probability"
            ].to_numpy(dtype=np.float64),
        }
        mlp_auc, mlp_ap = metrics(y, mlp_probability)
        mlp_auc_boot = np.empty(repeats, dtype=np.float64)
        mlp_ap_boot = np.empty(repeats, dtype=np.float64)
        for repeat, positions in enumerate(samples):
            mlp_auc_boot[repeat], mlp_ap_boot[repeat] = metrics(
                y[positions], mlp_probability[positions]
            )
        mlp_auc_low, mlp_auc_high = percentile(mlp_auc_boot)
        mlp_ap_low, mlp_ap_high = percentile(mlp_ap_boot)
        for comparator_name, comparator_probability in comparators.items():
            comparator_auc, comparator_ap = metrics(y, comparator_probability)
            comparator_auc_boot = np.empty(repeats, dtype=np.float64)
            comparator_ap_boot = np.empty(repeats, dtype=np.float64)
            for repeat, positions in enumerate(samples):
                comparator_auc_boot[repeat], comparator_ap_boot[repeat] = metrics(
                    y[positions], comparator_probability[positions]
                )
            auc_delta = mlp_auc_boot - comparator_auc_boot
            ap_delta = mlp_ap_boot - comparator_ap_boot
            auc_delta_low, auc_delta_high = percentile(auc_delta)
            ap_delta_low, ap_delta_high = percentile(ap_delta)
            rows.append({
                "split": split,
                "mlp_model": mlp_name,
                "comparator": comparator_name,
                "rows": int(len(y)),
                "positive": int(y.sum()),
                "mlp_auroc": mlp_auc,
                "mlp_auroc_ci_low": mlp_auc_low,
                "mlp_auroc_ci_high": mlp_auc_high,
                "mlp_auprc": mlp_ap,
                "mlp_auprc_ci_low": mlp_ap_low,
                "mlp_auprc_ci_high": mlp_ap_high,
                "comparator_auroc": comparator_auc,
                "comparator_auprc": comparator_ap,
                "delta_mlp_minus_comparator_auroc": mlp_auc - comparator_auc,
                "delta_auroc_ci_low": auc_delta_low,
                "delta_auroc_ci_high": auc_delta_high,
                "delta_mlp_minus_comparator_auprc": mlp_ap - comparator_ap,
                "delta_auprc_ci_low": ap_delta_low,
                "delta_auprc_ci_high": ap_delta_high,
                "delta_auroc_ci_excludes_zero": bool(auc_delta_low > 0 or auc_delta_high < 0),
                "delta_auprc_ci_excludes_zero": bool(ap_delta_low > 0 or ap_delta_high < 0),
                "bootstrap_repeats": repeats,
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlp-dir", required=True)
    parser.add_argument("--sparse-dir", required=True)
    parser.add_argument("--linear-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    args = parser.parse_args()
    mlp_dir = Path(args.mlp_dir).resolve()
    sparse_dir = Path(args.sparse_dir).resolve()
    linear_dir = Path(args.linear_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for directory in (mlp_dir, sparse_dir, linear_dir):
        if not (directory / ".MODELS_SUCCESS").is_file():
            raise FileNotFoundError(f"Success marker missing: {directory}")

    training = pd.read_csv(mlp_dir / "training_audit.csv")
    if len(training) != 180 or not np.isfinite(training["final_training_loss"]).all():
        raise AssertionError("MLP training audit failed")
    rows: list[dict[str, Any]] = []
    for split, filename in (("Train_OOF", "train_oof_predictions.csv"), ("Valid", "valid_predictions.csv")):
        mlp = pd.read_csv(mlp_dir / filename, dtype={"patient_id": str})
        sparse = pd.read_csv(sparse_dir / filename, dtype={"patient_id": str})
        linear = pd.read_csv(linear_dir / filename, dtype={"patient_id": str})
        rows.extend(comparison_rows(mlp, sparse, linear, split, args.bootstrap_repeats))
    comparison = pd.DataFrame(rows)
    atomic_csv(comparison, output_dir / "bootstrap_comparison.csv")

    importance = pd.read_csv(mlp_dir / "group_permutation_importance.csv")
    consistent = importance[
        (importance["outer_folds"] == 5)
        & (importance["mean_auprc_decrease"] > 0)
        & (importance["mean_auroc_decrease"] > 0)
        & (importance["positive_auprc_decrease_fraction"] >= 2 / 3)
        & (importance["positive_auroc_decrease_fraction"] >= 2 / 3)
    ].copy()
    consistent = consistent.sort_values(
        ["variant", "mean_auprc_decrease", "mean_auroc_decrease"],
        ascending=[True, False, False],
    )
    atomic_csv(consistent, output_dir / "consistent_group_importance.csv")

    significant = comparison[
        comparison["delta_auroc_ci_excludes_zero"]
        | comparison["delta_auprc_ci_excludes_zero"]
    ]
    summary = {
        "version": "api_fullseq_image_probe_v3_key_mlp_audit_1",
        "training_rows": int(len(training)),
        "nonfinite_training_losses": int((~np.isfinite(training["final_training_loss"])).sum()),
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap": "patient-level stratified paired percentile bootstrap",
        "comparison_rows": int(len(comparison)),
        "comparisons_with_any_metric_delta_ci_excluding_zero": int(len(significant)),
        "consistent_group_counts": consistent.groupby("variant").size().astype(int).to_dict(),
        "valid_used_for_model_development": False,
        "valid_labels_used_for": "post-hoc final metrics and uncertainty only",
        "seed": SEED,
    }
    atomic_json(summary, output_dir / "audit_summary.json")
    atomic_json(summary, output_dir / ".AUDIT_SUCCESS")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

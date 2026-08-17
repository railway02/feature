#!/usr/bin/env python3
"""Audit uncertainty and reproducibility of the stable sparse image probes."""
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
MODEL_MAP = {
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


def stratified_bootstrap_indices(y: np.ndarray, repeats: int) -> list[np.ndarray]:
    rng = np.random.default_rng(SEED)
    negative = np.flatnonzero(y == 0)
    positive = np.flatnonzero(y == 1)
    return [
        np.concatenate([
            rng.choice(negative, size=len(negative), replace=True),
            rng.choice(positive, size=len(positive), replace=True),
        ])
        for _ in range(repeats)
    ]


def interval(values: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def metric_values(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    return float(roc_auc_score(y, probability)), float(average_precision_score(y, probability))


def align_predictions(sparse: pd.DataFrame, linear: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sparse = sparse.copy()
    linear = linear.copy()
    sparse["patient_id"] = sparse["patient_id"].astype(str)
    linear["patient_id"] = linear["patient_id"].astype(str)
    if set(sparse["patient_id"]) != set(linear["patient_id"]):
        raise AssertionError("Sparse/linear patient IDs differ")
    linear = linear.set_index("patient_id").loc[sparse["patient_id"]].reset_index()
    if "target" in sparse and "target" in linear:
        if not np.array_equal(sparse["target"].to_numpy(), linear["target"].to_numpy()):
            raise AssertionError("Sparse/linear targets differ")
    return sparse, linear


def bootstrap_comparison(
    sparse: pd.DataFrame,
    linear: pd.DataFrame,
    split: str,
    repeats: int,
) -> pd.DataFrame:
    sparse, linear = align_predictions(sparse, linear)
    y = sparse["target"].to_numpy(dtype=np.int64)
    bootstrap = stratified_bootstrap_indices(y, repeats)
    rows: list[dict[str, Any]] = []
    for sparse_suffix, linear_suffix in MODEL_MAP.items():
        sparse_name = f"StableSparse_{sparse_suffix}"
        linear_name = f"LinearProbe_{linear_suffix}"
        sparse_probability = sparse[f"{sparse_name.lower()}_probability"].to_numpy(dtype=np.float64)
        linear_probability = linear[f"{linear_name.lower()}_probability"].to_numpy(dtype=np.float64)
        sparse_auc, sparse_ap = metric_values(y, sparse_probability)
        linear_auc, linear_ap = metric_values(y, linear_probability)
        sparse_auc_boot = np.empty(repeats, dtype=np.float64)
        sparse_ap_boot = np.empty(repeats, dtype=np.float64)
        linear_auc_boot = np.empty(repeats, dtype=np.float64)
        linear_ap_boot = np.empty(repeats, dtype=np.float64)
        for repeat, indices in enumerate(bootstrap):
            sampled_y = y[indices]
            sparse_auc_boot[repeat], sparse_ap_boot[repeat] = metric_values(
                sampled_y, sparse_probability[indices]
            )
            linear_auc_boot[repeat], linear_ap_boot[repeat] = metric_values(
                sampled_y, linear_probability[indices]
            )
        sparse_auc_low, sparse_auc_high = interval(sparse_auc_boot)
        sparse_ap_low, sparse_ap_high = interval(sparse_ap_boot)
        delta_auc_low, delta_auc_high = interval(sparse_auc_boot - linear_auc_boot)
        delta_ap_low, delta_ap_high = interval(sparse_ap_boot - linear_ap_boot)
        rows.append({
            "split": split,
            "sparse_model": sparse_name,
            "linear_baseline": linear_name,
            "rows": int(len(y)),
            "positive": int(y.sum()),
            "sparse_auroc": sparse_auc,
            "sparse_auroc_ci_low": sparse_auc_low,
            "sparse_auroc_ci_high": sparse_auc_high,
            "sparse_auprc": sparse_ap,
            "sparse_auprc_ci_low": sparse_ap_low,
            "sparse_auprc_ci_high": sparse_ap_high,
            "linear_auroc": linear_auc,
            "linear_auprc": linear_ap,
            "delta_sparse_minus_linear_auroc": sparse_auc - linear_auc,
            "delta_auroc_ci_low": delta_auc_low,
            "delta_auroc_ci_high": delta_auc_high,
            "delta_sparse_minus_linear_auprc": sparse_ap - linear_ap,
            "delta_auprc_ci_low": delta_ap_low,
            "delta_auprc_ci_high": delta_ap_high,
            "paired_delta_auroc_ci_excludes_zero": bool(delta_auc_low > 0 or delta_auc_high < 0),
            "paired_delta_auprc_ci_excludes_zero": bool(delta_ap_low > 0 or delta_ap_high < 0),
            "bootstrap_repeats": repeats,
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparse-dir", required=True)
    parser.add_argument("--linear-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    args = parser.parse_args()
    sparse_dir = Path(args.sparse_dir).resolve()
    linear_dir = Path(args.linear_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (sparse_dir / ".MODELS_SUCCESS").is_file():
        raise FileNotFoundError("Sparse success marker missing")
    if not (linear_dir / ".MODELS_SUCCESS").is_file():
        raise FileNotFoundError("Linear baseline success marker missing")
    fit_audit = pd.read_csv(sparse_dir / "logistic_fit_audit.csv")
    if len(fit_audit) != 1380 or fit_audit["convergence_warning"].astype(bool).any():
        raise AssertionError("Sparse fit audit failed")

    train_sparse = pd.read_csv(sparse_dir / "train_oof_predictions.csv", dtype={"patient_id": str})
    valid_sparse = pd.read_csv(sparse_dir / "valid_predictions.csv", dtype={"patient_id": str})
    train_linear = pd.read_csv(linear_dir / "train_oof_predictions.csv", dtype={"patient_id": str})
    valid_linear = pd.read_csv(linear_dir / "valid_predictions.csv", dtype={"patient_id": str})
    comparisons = pd.concat([
        bootstrap_comparison(train_sparse, train_linear, "Train_OOF", args.bootstrap_repeats),
        bootstrap_comparison(valid_sparse, valid_linear, "Valid", args.bootstrap_repeats),
    ], ignore_index=True)
    atomic_csv(comparisons, output_dir / "bootstrap_comparison.csv")

    stability = pd.read_csv(sparse_dir / "stable_feature_importance.csv")
    confirmed = stability[
        stability["stable_key_feature"].astype(bool)
        & stability["full_train_nonzero"].astype(bool)
        & (stability["source"] != "missing")
    ].copy()
    confirmed["confirmed_stable_key_feature"] = True
    confirmed = confirmed.sort_values(
        ["variant", "outer_nonzero_count", "outer_selection_count", "mean_abs_coefficient_across_outer_folds"],
        ascending=[True, False, False, False],
    )
    atomic_csv(confirmed, output_dir / "confirmed_stable_features.csv")

    consensus = (
        confirmed.groupby(["feature_name", "source", "group"], as_index=False)
        .agg(
            variant_count=("variant", "nunique"),
            variants=("variant", lambda values: "|".join(sorted(set(values)))),
            minimum_outer_selection_count=("outer_selection_count", "min"),
            minimum_outer_nonzero_count=("outer_nonzero_count", "min"),
            minimum_direction_consistency=("direction_consistency", "min"),
        )
        .sort_values(
            ["variant_count", "minimum_outer_nonzero_count", "minimum_outer_selection_count"],
            ascending=False,
        )
    )
    atomic_csv(consensus, output_dir / "cross_variant_consensus_features.csv")

    summary = {
        "version": "api_fullseq_image_probe_v3_stable_sparse_audit_1",
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap": "patient-level stratified paired percentile bootstrap",
        "logistic_fit_count": int(len(fit_audit)),
        "convergence_warning_count": int(fit_audit["convergence_warning"].astype(bool).sum()),
        "confirmed_stable_feature_counts": (
            confirmed.groupby("variant").size().astype(int).to_dict()
        ),
        "cross_variant_consensus_feature_count_at_least_2": int((consensus["variant_count"] >= 2).sum()),
        "cross_variant_consensus_feature_count_at_least_3": int((consensus["variant_count"] >= 3).sum()),
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

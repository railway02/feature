#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from common import atomic_csv, atomic_json, configure_runtime, load_config, write_success


def import_fixed(path: Path):
    spec = importlib.util.spec_from_file_location("fast_v1_fixed_shared", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class MorphologyPreprocessor:
    branch: object | None = None
    final_scaler: StandardScaler | None = None

    def fit(self, values: np.ndarray, seed: int, fixed) -> "MorphologyPreprocessor":
        self.branch = fixed.NumericBranch(32, "robust", 0.25, True).fit(values, seed)
        reduced = self.branch.transform(values)
        self.final_scaler = StandardScaler().fit(reduced)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.branch is None or self.final_scaler is None:
            raise RuntimeError("Morphology preprocessor not fitted")
        output = self.final_scaler.transform(self.branch.transform(values)).astype(np.float64)
        if not np.isfinite(output).all():
            raise AssertionError("Nonfinite morphology features")
        return output


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def safe_ap(y, p):
    return float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()
    config = load_config(args.config)
    configure_runtime(config)
    fixed = import_fixed(Path(config["fixed_trainer"]))
    outputs = Path(config["paths"]["outputs"])
    reports = Path(config["paths"]["reports"])
    upstream = Path(config["upstream_v1_outputs"]) / "mask_morphology" / "pred_patient_median.csv"
    morphology = pd.read_csv(upstream, dtype={"patient_id": str})
    morphology = morphology[morphology["split"] == "Train"].copy()
    meta = pd.read_csv(reports / "train_oof_predictions.csv", dtype={"patient_id": str})
    folds = pd.read_csv(reports / "patient_fold_assignments.csv", dtype={"patient_id": str})
    meta = meta[["patient_id", "target"]].merge(folds, on="patient_id", validate="one_to_one")
    store = morphology.set_index("patient_id")
    if not set(meta["patient_id"]).issubset(store.index):
        raise AssertionError("Morphology missing patients from main OOF cohort")
    columns = [
        column for column in morphology.columns
        if column not in {"patient_id", "split", "series_count"} and pd.api.types.is_numeric_dtype(morphology[column])
    ]
    values = store.loc[meta["patient_id"], columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    y = meta["target"].to_numpy(np.int64)
    groups = meta["patient_id"].astype(str).to_numpy()
    fold_values = meta["fold"].to_numpy(int)
    oof = np.full(len(y), np.nan)
    audit_rows = []
    audit_path = reports / "morphology_logistic_convergence_audit.csv"
    model_root = outputs / "minimal_models" / "pred_morphology"
    fold_rows = []

    for fold in sorted(np.unique(fold_values)):
        holdout = np.flatnonzero(fold_values == fold)
        development = np.flatnonzero(fold_values != fold)
        predictions = {float(c): np.full(len(development), np.nan) for c in config["prediction"]["c_grid"]}
        inner_splits = fixed.grouped_splits(
            y[development], groups[development], int(config["prediction"]["inner_folds"]),
            int(config["prediction"]["seed"]) + int(fold) * 1000,
        )
        for inner_fold, (fit_index, inner_holdout) in enumerate(inner_splits, 1):
            pre = MorphologyPreprocessor().fit(values[development][fit_index], int(config["prediction"]["seed"]) + int(fold) * 100 + inner_fold, fixed)
            x_fit = pre.transform(values[development][fit_index])
            x_hold = pre.transform(values[development][inner_holdout])
            for c_value in predictions:
                model = fixed.fit_logistic_checked(
                    x_fit, y[development][fit_index], c_value,
                    {
                        "task": "adverse_patient", "representation": "pred_morphology",
                        "outer_fold": int(fold), "stage": "inner_cv",
                        "variant": "morphology", "inner_fold": inner_fold,
                    },
                    audit_rows, audit_path, x_hold, y[development][inner_holdout],
                )
                predictions[c_value][inner_holdout] = model.predict_proba(x_hold)[:, 1]
        scores = {str(c): safe_ap(y[development], probability) for c, probability in predictions.items()}
        selected_c = max(predictions, key=lambda value: (scores[str(value)], -float(value)))
        pre = MorphologyPreprocessor().fit(values[development], int(config["prediction"]["seed"]) + int(fold) * 100, fixed)
        x_dev = pre.transform(values[development])
        x_hold = pre.transform(values[holdout])
        model = fixed.fit_logistic_checked(
            x_dev, y[development], float(selected_c),
            {
                "task": "adverse_patient", "representation": "pred_morphology",
                "outer_fold": int(fold), "stage": "outer_development_refit",
                "variant": "morphology", "inner_fold": 0,
            },
            audit_rows, audit_path, x_hold, y[holdout],
        )
        oof[holdout] = model.predict_proba(x_hold)[:, 1]
        model_root.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "preprocessor": pre, "model": model, "selected_c": float(selected_c),
            "columns": columns, "development_patient_ids": groups[development].tolist(),
            "holdout_patient_ids": groups[holdout].tolist(),
        }, model_root / f"fold_{int(fold)}.joblib")
        fold_rows.append({
            "fold": int(fold), "selected_c": float(selected_c),
            "holdout_auroc": safe_auc(y[holdout], oof[holdout]),
            "holdout_auprc": safe_ap(y[holdout], oof[holdout]),
        })

    if not np.isfinite(oof).all():
        raise AssertionError("Incomplete morphology OOF")
    output = meta.copy()
    output["pred_morphology_probability"] = oof
    atomic_csv(output, reports / "morphology_train_oof_predictions.csv")
    atomic_csv(pd.DataFrame(fold_rows), reports / "morphology_fold_metrics.csv")
    summary = {
        "rows": len(y), "positive": int(y.sum()), "columns": len(columns),
        "auroc": safe_auc(y, oof), "auprc": safe_ap(y, oof),
        "brier": float(brier_score_loss(y, oof)),
        "threshold": float(fixed.youden_threshold(y, oof)),
        "same_outer_folds_as_whole_and_roi": True,
    }
    atomic_json(summary, reports / "morphology_oof_metrics.json")
    write_success(reports / ".MORPHOLOGY_OOF_SUCCESS", "07_train_morphology_oof", config, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

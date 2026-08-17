#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from common import atomic_csv, atomic_json, configure_runtime, load_config, normalize_patient_id, write_success


def import_fixed(path: Path):
    spec = importlib.util.spec_from_file_location("fast_v1_fixed_shared", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class DeepOnlyPreprocessor:
    branch: object | None = None
    final_scaler: StandardScaler | None = None

    def fit(self, deep: np.ndarray, missing: np.ndarray, seed: int, fixed) -> "DeepOnlyPreprocessor":
        self.branch = fixed.DeepBranch().fit(deep, seed)
        base = np.concatenate([self.branch.transform(deep), missing.astype(np.float64)], axis=1)
        self.final_scaler = StandardScaler().fit(base)
        return self

    def transform(self, deep: np.ndarray, missing: np.ndarray) -> np.ndarray:
        if self.branch is None or self.final_scaler is None:
            raise RuntimeError("Preprocessor not fitted")
        base = np.concatenate([self.branch.transform(deep), missing.astype(np.float64)], axis=1)
        output = self.final_scaler.transform(base).astype(np.float64)
        if not np.isfinite(output).all():
            raise AssertionError("Nonfinite deep features")
        return output


def load_embeddings(table_dir: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    raw = np.load(table_dir / "patient_median_embeddings_5120.npz")
    ids = raw["patient_id"].astype(str).tolist()
    embeddings = raw["embeddings"].astype(np.float32)
    if embeddings.ndim != 3 or embeddings.shape[1:] != (2, 5120):
        raise AssertionError(f"Unexpected embeddings: {embeddings.shape}")
    missing = np.stack([
        np.isnan(embeddings[:, 0]).all(axis=1),
        np.isnan(embeddings[:, 1]).all(axis=1),
    ], axis=1).astype(np.float64)
    return ids, embeddings.reshape(len(ids), -1), missing


def labels(path: Path, split: str) -> pd.DataFrame:
    frame = pd.read_excel(path, dtype=object)
    frame["patient_id"] = frame["病案号"].map(normalize_patient_id)
    adverse = next(column for column in frame.columns if str(column).startswith("不良转归"))
    rows = []
    for patient_id, group in frame.groupby("patient_id"):
        values = pd.to_numeric(group[adverse], errors="coerce").dropna().astype(int).unique().tolist()
        if len(values) == 1 and values[0] in {0, 1}:
            rows.append({"patient_id": patient_id, "split": split, "target": int(values[0])})
    return pd.DataFrame(rows)


def align_store(ids: list[str], deep: np.ndarray, missing: np.ndarray, ordered_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    lookup = {uid: index for index, uid in enumerate(ids)}
    indices = [lookup[uid] for uid in ordered_ids]
    return deep[indices], missing[indices]


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def select_c(deep: np.ndarray, missing: np.ndarray, y: np.ndarray, groups: np.ndarray, outer_fold: int,
             fixed, config: dict, representation: str, audit_rows: list[dict], audit_path: Path) -> tuple[float, dict[str, float]]:
    predictions = {float(c): np.full(len(y), np.nan, dtype=np.float64) for c in config["prediction"]["c_grid"]}
    splits = fixed.grouped_splits(y, groups, int(config["prediction"]["inner_folds"]), int(config["prediction"]["seed"]) + outer_fold * 1000)
    for inner_fold, (fit_index, holdout_index) in enumerate(splits, 1):
        pre = DeepOnlyPreprocessor().fit(deep[fit_index], missing[fit_index], int(config["prediction"]["seed"]) + outer_fold * 100 + inner_fold, fixed)
        x_fit = pre.transform(deep[fit_index], missing[fit_index])
        x_holdout = pre.transform(deep[holdout_index], missing[holdout_index])
        for c_value in predictions:
            model = fixed.fit_logistic_checked(
                x_fit, y[fit_index], c_value,
                {
                    "task": "adverse_patient", "representation": representation,
                    "outer_fold": outer_fold, "stage": "inner_cv",
                    "variant": "deep", "inner_fold": inner_fold,
                },
                audit_rows, audit_path, x_holdout, y[holdout_index],
            )
            predictions[c_value][holdout_index] = model.predict_proba(x_holdout)[:, 1]
    scores = {str(c): safe_ap(y, probability) for c, probability in predictions.items()}
    selected = max(predictions, key=lambda value: (scores[str(value)], -float(value)))
    return float(selected), scores


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()
    config = load_config(args.config)
    configure_runtime(config)
    fixed = import_fixed(Path(config["fixed_trainer"]))
    outputs = Path(config["paths"]["outputs"])
    reports = Path(config["paths"]["reports"])
    roi_dir = outputs / "cave_pred_roi_tables" / "train"
    whole_dir = Path(config["whole_train_tables"])
    roi_ids, roi_deep, roi_missing = load_embeddings(roi_dir)
    whole_ids, whole_deep, whole_missing = load_embeddings(whole_dir)
    label_frame = labels(Path(config["train_excel"]), "Train")

    common_ids = sorted(set(roi_ids) & set(whole_ids) & set(label_frame["patient_id"]))
    meta = label_frame[label_frame["patient_id"].isin(common_ids)].sort_values("patient_id").reset_index(drop=True)
    ordered_ids = meta["patient_id"].astype(str).tolist()
    y = meta["target"].to_numpy(np.int64)
    groups = np.asarray(ordered_ids)
    if len(np.unique(y)) != 2:
        raise AssertionError("Train cohort does not contain both classes")
    roi_x, roi_m = align_store(roi_ids, roi_deep, roi_missing, ordered_ids)
    whole_x, whole_m = align_store(whole_ids, whole_deep, whole_missing, ordered_ids)

    folds = fixed.grouped_splits(y, groups, int(config["prediction"]["folds"]), int(config["prediction"]["seed"]))
    fold_assignment = np.zeros(len(y), dtype=np.int64)
    whole_oof = np.full(len(y), np.nan)
    roi_oof = np.full(len(y), np.nan)
    audit_rows = []
    audit_path = reports / "minimal_logistic_convergence_audit.csv"
    model_root = outputs / "minimal_models"
    fold_rows = []

    for fold, (development, holdout) in enumerate(folds, 1):
        fold_assignment[holdout] = fold
        for name, deep, missing, destination in (
            ("whole", whole_x, whole_m, whole_oof),
            ("pred_roi", roi_x, roi_m, roi_oof),
        ):
            selected_c, inner_scores = select_c(
                deep[development], missing[development], y[development], groups[development],
                fold, fixed, config, name, audit_rows, audit_path,
            )
            pre = DeepOnlyPreprocessor().fit(
                deep[development], missing[development],
                int(config["prediction"]["seed"]) + fold * 100, fixed,
            )
            x_dev = pre.transform(deep[development], missing[development])
            x_hold = pre.transform(deep[holdout], missing[holdout])
            model = fixed.fit_logistic_checked(
                x_dev, y[development], selected_c,
                {
                    "task": "adverse_patient", "representation": name,
                    "outer_fold": fold, "stage": "outer_development_refit",
                    "variant": "deep", "inner_fold": 0,
                },
                audit_rows, audit_path, x_hold, y[holdout],
            )
            destination[holdout] = model.predict_proba(x_hold)[:, 1]
            directory = model_root / name
            directory.mkdir(parents=True, exist_ok=True)
            joblib.dump({
                "preprocessor": pre,
                "model": model,
                "selected_c": selected_c,
                "inner_ap_by_c": inner_scores,
                "development_patient_ids": groups[development].tolist(),
                "holdout_patient_ids": groups[holdout].tolist(),
            }, directory / f"fold_{fold}.joblib")
            fold_rows.append({
                "fold": fold, "representation": name,
                "development_rows": len(development), "holdout_rows": len(holdout),
                "selected_c": selected_c,
                "holdout_auroc": safe_auc(y[holdout], destination[holdout]),
                "holdout_auprc": safe_ap(y[holdout], destination[holdout]),
            })

    if not np.isfinite(whole_oof).all() or not np.isfinite(roi_oof).all():
        raise AssertionError("Incomplete OOF probabilities")
    predictions = meta.copy()
    predictions["fold"] = fold_assignment
    predictions["whole_probability"] = whole_oof
    predictions["pred_roi_probability"] = roi_oof
    atomic_csv(predictions, reports / "train_oof_predictions.csv")
    atomic_csv(pd.DataFrame(fold_rows), reports / "train_fold_metrics.csv")
    atomic_csv(pd.DataFrame({"patient_id": ordered_ids, "fold": fold_assignment}), reports / "patient_fold_assignments.csv")

    whole_metrics = {
        "auroc": safe_auc(y, whole_oof),
        "auprc": safe_ap(y, whole_oof),
        "brier": float(brier_score_loss(y, whole_oof)),
        "threshold": float(fixed.youden_threshold(y, whole_oof)),
    }
    roi_metrics = {
        "auroc": safe_auc(y, roi_oof),
        "auprc": safe_ap(y, roi_oof),
        "brier": float(brier_score_loss(y, roi_oof)),
        "threshold": float(fixed.youden_threshold(y, roi_oof)),
    }
    fold_frame = pd.DataFrame(fold_rows)
    pivot_ap = fold_frame.pivot(index="fold", columns="representation", values="holdout_auprc")
    consistent_folds = int((pivot_ap["pred_roi"] > pivot_ap["whole"]).sum())
    leave_one_out_positive = 0
    for fold in sorted(np.unique(fold_assignment)):
        keep = fold_assignment != fold
        if safe_ap(y[keep], roi_oof[keep]) - safe_ap(y[keep], whole_oof[keep]) > 0:
            leave_one_out_positive += 1

    delta = {
        "auroc": roi_metrics["auroc"] - whole_metrics["auroc"],
        "auprc": roi_metrics["auprc"] - whole_metrics["auprc"],
        "brier_roi_minus_whole": roi_metrics["brier"] - whole_metrics["brier"],
    }
    gate_cfg = config["oof_gain_audit"]
    condition_a = delta["auprc"] >= float(gate_cfg["minimum_auprc_gain"]) and delta["auroc"] >= -float(gate_cfg["maximum_auroc_drop_when_auprc_gate"])
    condition_b = delta["auroc"] >= float(gate_cfg["minimum_auroc_gain"]) and delta["auprc"] >= 0
    gain_criteria_met = bool(
        (condition_a or condition_b) and
        delta["auprc"] > 0 and
        consistent_folds >= int(gate_cfg["minimum_consistent_folds"]) and
        leave_one_out_positive >= 4 and
        delta["brier_roi_minus_whole"] <= float(gate_cfg["maximum_brier_worsening"])
    )
    summary = {
        "train_rows": len(meta),
        "train_positive": int(y.sum()),
        "whole": whole_metrics,
        "pred_roi": roi_metrics,
        "delta_roi_minus_whole": delta,
        "consistent_folds_by_auprc": consistent_folds,
        "leave_one_fold_out_positive": leave_one_out_positive,
        "condition_a": condition_a,
        "condition_b": condition_b,
        "oof_gain_criteria_met": gain_criteria_met,
        "oof_gain_controls_valid_execution": False,
        "pred_roi_pipeline_is_mandatory": True,
        "valid_used": False,
        "models": ["Dummy", "Logistic_deep"],
    }
    atomic_json(summary, reports / "train_oof_comparison.json")
    for stale_name in (
        ".OOF_GATE_PASS", ".STOPPED_NO_OOF_GAIN",
        ".OOF_GAIN_PASS", ".OOF_GAIN_NOT_DEMONSTRATED",
    ):
        (reports / stale_name).unlink(missing_ok=True)
    write_success(reports / ".OOF_AUDIT_COMPLETE", "06_train_oof_comparison", config, summary)
    gain_marker = reports / (
        ".OOF_GAIN_PASS" if gain_criteria_met else ".OOF_GAIN_NOT_DEMONSTRATED"
    )
    write_success(gain_marker, "06_train_oof_gain_classification", config, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

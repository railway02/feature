#!/usr/bin/env python3
"""Strict PredROI+CAVE-deep Logistic using the formal v31 Logistic core."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from common import atomic_csv, atomic_json, brier, load_config, load_temporal, load_train_folds, resolve_path, safe_ap, safe_auc


def load_formal_core(path: Path):
    spec = importlib.util.spec_from_file_location("formal_logistic_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_pred_combined(feature_root: Path, fold: int, split: str, expected_uid):
    path = feature_root / "seg_features" / f"fold_{fold}" / f"{split.casefold()}.npz"
    with np.load(path, allow_pickle=False) as z:
        x = np.asarray(z["pred_combined"], dtype=np.float32)
        uid = z["series_uid"].astype(str)
    if not np.array_equal(uid, expected_uid.astype(str)):
        raise AssertionError(f"UID order mismatch: {path}")
    if x.shape[1] != 1024 or not np.isfinite(x).all():
        raise AssertionError(f"Invalid pred_combined: {path}")
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Existing primary A config")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--report-root", required=True)
    ap.add_argument("--formal-core", required=True)
    ap.add_argument("--cpu-threads", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if cfg["spatial"].get("representation") != "global_pred_roi":
        raise AssertionError("Expected primary PredROI config")
    if cfg.get("temporal", {}).get("representation") != "deep_only":
        raise AssertionError("Expected deep_only temporal config")
    out = Path(args.output_root).resolve()
    rep = Path(args.report_root).resolve()
    feature_root = resolve_path(cfg["spatial"]["feature_root"], cfg["project_root"])
    core = load_formal_core(Path(args.formal_core).resolve())
    core.set_seed(core.SEED)
    runtime = core.configure_runtime(int(args.cpu_threads), torch.device("cpu"))

    train = load_temporal(cfg, "Train")
    valid = load_temporal(cfg, "Valid")
    folds = load_train_folds(cfg, train)
    y = train["target"].astype(int)
    yv = valid["target"].astype(int)
    groups = train["patient_id"].astype(str)
    oof = np.full(len(y), np.nan, dtype=float)
    valid_fold_prob = []
    fold_rows = []

    for fold in range(1, 6):
        dev = np.flatnonzero(folds != fold)
        hold = np.flatnonzero(folds == fold)
        if set(groups[dev]) & set(groups[hold]):
            raise AssertionError("Patient leakage between outer development and holdout")
        spatial_train = load_pred_combined(feature_root, fold, "Train", train["series_uid"])
        spatial_valid = load_pred_combined(feature_root, fold, "Valid", valid["series_uid"])
        x_train = np.concatenate([spatial_train, train["deep"]], axis=1).astype(np.float32)
        x_valid = np.concatenate([spatial_valid, valid["deep"]], axis=1).astype(np.float32)
        if x_train.shape != (781, 11264) or x_valid.shape != (207, 11264):
            raise AssertionError(f"Unexpected composite shape fold={fold}: {x_train.shape}/{x_valid.shape}")
        dummy_train = np.empty((len(x_train), 0), dtype=np.float32)
        dummy_valid = np.empty((len(x_valid), 0), dtype=np.float32)

        best, inner_rows = core.select_logistic(
            x_train[dev], dummy_train[dev], y[dev], groups[dev], False, core.SEED + fold * 100
        )
        preprocessor = core.FusionPreprocessor(
            deep_pca_dim=int(best["deep_pca_dim"]), use_scalar=False, seed=core.SEED + fold * 1000
        ).fit(x_train[dev], dummy_train[dev])
        x_dev = preprocessor.transform(x_train[dev], dummy_train[dev])
        x_hold = preprocessor.transform(x_train[hold], dummy_train[hold])
        x_val = preprocessor.transform(x_valid, dummy_valid)
        model = core.LogisticRegression(
            C=float(best["C"]), class_weight="balanced", solver="liblinear", max_iter=10000, random_state=core.SEED + fold
        )
        model.fit(x_dev, y[dev])
        hold_prob = model.predict_proba(x_hold)[:, 1]
        valid_prob = model.predict_proba(x_val)[:, 1]
        oof[hold] = hold_prob
        valid_fold_prob.append(valid_prob)

        fold_dir = out / "folds" / "PredROI_CAVE_Logistic" / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"preprocessor": preprocessor, "model": model, "best_inner_config": best}, fold_dir / "model.joblib")
        atomic_csv(pd.DataFrame(inner_rows), fold_dir / "inner_search.csv")
        np.savez_compressed(
            fold_dir / "predictions.npz",
            holdout_series_uid=train["series_uid"][hold].astype(str),
            holdout_probability=hold_prob.astype(np.float64),
            valid_series_uid=valid["series_uid"].astype(str),
            valid_probability=valid_prob.astype(np.float64),
        )
        payload = {
            "status": "success", "model": "PredROI_CAVE_Logistic", "fold": fold,
            "development_rows": int(len(dev)), "holdout_rows": int(len(hold)),
            "strict_spatial_feature_root": str(feature_root), "spatial_key": "pred_combined",
            "temporal_key": "deep", "input_dim": 11264,
            "best_inner_config": best, "preprocessor": preprocessor.audit(),
        }
        atomic_json(payload, fold_dir / ".SUCCESS.json")
        fold_rows.append({
            "fold": fold, "best_epoch": None, "best_inner_AUPRC": float(best["inner_pooled_AUPRC"]),
            "best_inner_AUROC": float(best["inner_pooled_AUROC"]), "selected_pca_dim": int(best["deep_pca_dim"]), "selected_C": float(best["C"]),
            "OOF_AUPRC": safe_ap(y[hold], hold_prob), "OOF_AUROC": safe_auc(y[hold], hold_prob), "OOF_Brier": brier(y[hold], hold_prob),
        })

    if not np.isfinite(oof).all():
        raise RuntimeError("Incomplete OOF")
    valid_probability = np.mean(np.stack(valid_fold_prob), axis=0)
    metrics = {
        "model": "PredROI_CAVE_Logistic", "strategy": "strict_crossfit", "spatial_representation": "global_pred_roi", "temporal_representation": "deep_only",
        "input": "concat(pred_combined[1024], cave_deep[10240])", "input_dim": 11264,
        "formal_logistic_core": str(Path(args.formal_core).resolve()), "pca_dims": list(core.LOGISTIC_DEEP_DIMS), "C_grid": list(core.C_GRID),
        "selection": "3-fold grouped Train-only inner pooled AUPRC; tie AUROC, lower PCA dimension, lower C", "uses_scalar": False, "valid_used_for_selection": False,
        "train_oof": {"AUROC": safe_auc(y, oof), "AUPRC": safe_ap(y, oof), "Brier": brier(y, oof)},
        "valid": {"AUROC": safe_auc(yv, valid_probability), "AUPRC": safe_ap(yv, valid_probability), "Brier": brier(yv, valid_probability)},
        "runtime": runtime,
    }
    atomic_json(metrics, out / "metrics.json")
    atomic_csv(pd.DataFrame(fold_rows), out / "fold_metrics.csv")
    atomic_csv(pd.DataFrame({"series_uid": train["series_uid"].astype(str), "patient_id": train["patient_id"].astype(str), "target": y, "fold": folds, "probability": oof, "representation_oof_status": "strict_crossfit"}), out / "train_oof_predictions.csv")
    atomic_csv(pd.DataFrame({"series_uid": valid["series_uid"].astype(str), "patient_id": valid["patient_id"].astype(str), "target": yv, "probability": valid_probability}), out / "valid_predictions.csv")
    atomic_json({"status": "success", "model": "PredROI_CAVE_Logistic", "runtime": runtime, "input_dim": 11264, "folds": [1,2,3,4,5]}, out / ".SUCCESS.json")
    rep.mkdir(parents=True, exist_ok=True)
    atomic_json(metrics, rep / "metrics.json")
    atomic_csv(pd.DataFrame(fold_rows), rep / "fold_metrics.csv")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

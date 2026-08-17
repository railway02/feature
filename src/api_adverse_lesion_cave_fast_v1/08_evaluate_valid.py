#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, roc_auc_score

from common import atomic_csv, atomic_json, configure_runtime, load_config, normalize_patient_id, write_success


def import_fixed(path: Path):
    spec = importlib.util.spec_from_file_location("fast_v1_fixed_shared", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_embeddings(table_dir: Path):
    raw = np.load(table_dir / "patient_median_embeddings_5120.npz")
    ids = raw["patient_id"].astype(str).tolist()
    embeddings = raw["embeddings"].astype(np.float32)
    missing = np.stack([np.isnan(embeddings[:, 0]).all(1), np.isnan(embeddings[:, 1]).all(1)], axis=1).astype(np.float64)
    return ids, embeddings.reshape(len(ids), -1), missing


def labels(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, dtype=object)
    frame["patient_id"] = frame["病案号"].map(normalize_patient_id)
    adverse = next(column for column in frame.columns if str(column).startswith("不良转归"))
    rows = []
    for patient_id, group in frame.groupby("patient_id"):
        values = pd.to_numeric(group[adverse], errors="coerce").dropna().astype(int).unique().tolist()
        if len(values) == 1 and values[0] in {0, 1}:
            rows.append({"patient_id": patient_id, "split": "Valid", "target": int(values[0])})
    return pd.DataFrame(rows)


def align(ids, values, missing, ordered):
    lookup = {uid: index for index, uid in enumerate(ids)}
    indices = [lookup[uid] for uid in ordered]
    return values[indices], missing[indices]


def metrics(name, y, probability, threshold):
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "model": name, "rows": len(y), "positive": int(y.sum()),
        "auroc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "threshold_from_train_oof": float(threshold),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()
    config = load_config(args.config)
    configure_runtime(config)
    fixed = import_fixed(Path(config["fixed_trainer"]))
    outputs = Path(config["paths"]["outputs"])
    reports = Path(config["paths"]["reports"])
    if not (reports / ".OOF_AUDIT_COMPLETE").is_file():
        raise RuntimeError("Train OOF comparison audit is incomplete; Valid evaluation forbidden")

    train_gate = json.loads((reports / "train_oof_comparison.json").read_text(encoding="utf-8"))
    roi_ids, roi_deep, roi_missing = load_embeddings(outputs / "cave_pred_roi_tables" / "valid")
    whole_ids, whole_deep, whole_missing = load_embeddings(Path(config["whole_valid_tables"]))
    morphology = pd.read_csv(Path(config["upstream_v1_outputs"]) / "mask_morphology" / "pred_patient_median.csv", dtype={"patient_id": str})
    morphology = morphology[morphology["split"] == "Valid"].set_index("patient_id")
    label_frame = labels(Path(config["valid_excel"]))
    common = sorted(set(roi_ids) & set(whole_ids) & set(morphology.index) & set(label_frame["patient_id"]))
    meta = label_frame[label_frame["patient_id"].isin(common)].sort_values("patient_id").reset_index(drop=True)
    ordered = meta["patient_id"].astype(str).tolist()
    y = meta["target"].to_numpy(int)
    roi_x, roi_m = align(roi_ids, roi_deep, roi_missing, ordered)
    whole_x, whole_m = align(whole_ids, whole_deep, whole_missing, ordered)

    probabilities = {}
    for name, deep, missing in (("whole", whole_x, whole_m), ("pred_roi", roi_x, roi_m)):
        fold_predictions = []
        for fold in range(1, int(config["prediction"]["folds"]) + 1):
            payload = joblib.load(outputs / "minimal_models" / name / f"fold_{fold}.joblib")
            x = payload["preprocessor"].transform(deep, missing)
            fold_predictions.append(payload["model"].predict_proba(x)[:, 1])
        probabilities[name] = np.mean(np.stack(fold_predictions), axis=0)

    morphology_fold_predictions = []
    for fold in range(1, int(config["prediction"]["folds"]) + 1):
        payload = joblib.load(outputs / "minimal_models" / "pred_morphology" / f"fold_{fold}.joblib")
        values = morphology.loc[ordered, payload["columns"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        x = payload["preprocessor"].transform(values)
        morphology_fold_predictions.append(payload["model"].predict_proba(x)[:, 1])
    probabilities["pred_morphology"] = np.mean(np.stack(morphology_fold_predictions), axis=0)

    train_morphology = json.loads((reports / "morphology_oof_metrics.json").read_text(encoding="utf-8"))
    thresholds = {
        "whole": float(train_gate["whole"]["threshold"]),
        "pred_roi": float(train_gate["pred_roi"]["threshold"]),
        "pred_morphology": float(train_morphology["threshold"]),
    }
    output = meta.copy()
    for name, probability in probabilities.items():
        output[f"{name}_probability"] = probability
    atomic_csv(output, reports / "valid_predictions.csv")
    metric_rows = [
        metrics(name, y, probabilities[name], thresholds[name])
        for name in ("whole", "pred_morphology", "pred_roi")
    ]
    metric_frame = pd.DataFrame(metric_rows)
    atomic_csv(metric_frame, reports / "valid_metrics.csv")
    upstream_provenance = json.loads((reports / "upstream_provenance.json").read_text(encoding="utf-8"))
    segmentation_gate_passed = bool(upstream_provenance["audit"]["segmentation_gate_passed"])
    summary = {
        "valid_rows": len(meta), "valid_positive": int(y.sum()),
        "valid_used_for_selection": False,
        "valid_executed_regardless_of_oof_gain": True,
        "oof_gain_criteria_met": bool(train_gate["oof_gain_criteria_met"]),
        "segmentation_gate_passed": segmentation_gate_passed,
        "segmentation_gate_controls_execution": False,
        "deployable_claim_allowed_by_segmentation_gate": segmentation_gate_passed,
        "models": metric_rows,
        "whole_vs_pred_roi": {
            "auroc_delta": metric_rows[2]["auroc"] - metric_rows[0]["auroc"],
            "auprc_delta": metric_rows[2]["auprc"] - metric_rows[0]["auprc"],
            "brier_delta": metric_rows[2]["brier"] - metric_rows[0]["brier"],
        },
    }
    atomic_json(summary, reports / "final_fast_summary.json")
    for stale_name in (
        ".FAST_MAIN_SUCCESS",
        ".FAST_MAIN_TECHNICAL_COMPLETE",
        ".FAST_MAIN_COMPLETE_WITH_SEGMENTATION_WARNING",
    ):
        (reports / stale_name).unlink(missing_ok=True)
    write_success(reports / ".FAST_MAIN_TECHNICAL_COMPLETE", "08_evaluate_valid", config, summary)
    final_marker = reports / (
        ".FAST_MAIN_SUCCESS"
        if segmentation_gate_passed
        else ".FAST_MAIN_COMPLETE_WITH_SEGMENTATION_WARNING"
    )
    write_success(final_marker, "08_evaluate_valid_scientific_status", config, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

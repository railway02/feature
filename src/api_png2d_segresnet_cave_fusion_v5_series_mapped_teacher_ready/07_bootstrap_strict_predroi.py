#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from common import atomic_csv, atomic_json


ROOT = Path("/root/autodl-tmp/aneurysm")
KEYS = ["series_uid", "patient_id", "target"]
BOOTSTRAP_SEED = 20260831
N_BOOTSTRAP = 5000


def load_predictions(path: Path, probability_column: str, split: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    cols = KEYS + (["fold"] if split == "Train_OOF" else [])
    if probability_column not in frame.columns:
        raise KeyError(f"{path}: missing {probability_column}")
    result = frame[cols].copy()
    result["patient_id"] = result["patient_id"].astype(str)
    result["series_uid"] = result["series_uid"].astype(str)
    result["target"] = result["target"].astype(int)
    result["probability"] = pd.to_numeric(frame[probability_column], errors="raise")
    if result["series_uid"].duplicated().any() or not np.isfinite(result["probability"]).all():
        raise AssertionError(f"{path}: duplicate UID or nonfinite probability")
    return result


def align(left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str):
    l = left.sort_values("series_uid").reset_index(drop=True)
    r = right.sort_values("series_uid").reset_index(drop=True)
    if len(l) != len(r) or not l[KEYS].equals(r[KEYS]):
        raise AssertionError(f"Alignment failed: {left_name} vs {right_name}")
    return l, r


def cluster_indices(patient_id: np.ndarray):
    patient_id = np.asarray(patient_id).astype(str)
    patients = np.unique(patient_id)
    mapping = [np.flatnonzero(patient_id == patient) for patient in patients]
    return patients, mapping


def metric_values(y, probability):
    return {
        "AUROC": float(roc_auc_score(y, probability)),
        "AUPRC": float(average_precision_score(y, probability)),
    }


def bootstrap_pair(left, right, seed):
    y = left["target"].to_numpy(int)
    p_left = left["probability"].to_numpy(float)
    p_right = right["probability"].to_numpy(float)
    _, clusters = cluster_indices(left["patient_id"].to_numpy(str))
    rng = np.random.default_rng(seed)
    values = {"AUROC": [], "AUPRC": []}
    n_patients = len(clusters)
    for _ in range(N_BOOTSTRAP):
        drawn = rng.integers(0, n_patients, size=n_patients)
        index = np.concatenate([clusters[i] for i in drawn])
        sample_y = y[index]
        for metric, fn in (("AUROC", roc_auc_score), ("AUPRC", average_precision_score)):
            values[metric].append((float(fn(sample_y, p_left[index])), float(fn(sample_y, p_right[index]))))
    return {metric: np.asarray(pair, dtype=float) for metric, pair in values.items()}


def bootstrap_self(frame, seed):
    y = frame["target"].to_numpy(int)
    p = frame["probability"].to_numpy(float)
    _, clusters = cluster_indices(frame["patient_id"].to_numpy(str))
    rng = np.random.default_rng(seed)
    values = {"AUROC": [], "AUPRC": []}
    n_patients = len(clusters)
    for _ in range(N_BOOTSTRAP):
        drawn = rng.integers(0, n_patients, size=n_patients)
        index = np.concatenate([clusters[i] for i in drawn])
        sample_y = y[index]
        values["AUROC"].append(float(roc_auc_score(sample_y, p[index])))
        values["AUPRC"].append(float(average_precision_score(sample_y, p[index])))
    return {metric: np.asarray(value, dtype=float) for metric, value in values.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-root", required=True)
    args = ap.parse_args()
    report = Path(args.report_root).resolve()

    paths = {
        "A": {
            "Train_OOF": (ROOT / "outputs/api_png2d_segresnet_cave_fusion_v5_strict_cell_a_predroi_deep/fusion/gated_interaction/train_oof_predictions.csv", "probability"),
            "Valid": (ROOT / "outputs/api_png2d_segresnet_cave_fusion_v5_strict_cell_a_predroi_deep/fusion/gated_interaction/valid_predictions.csv", "probability"),
        },
        "B0_Historical_CAVE_Deep_Logistic": {
            "Train_OOF": (ROOT / "outputs/api_png2d_gtmask_roi_cave_v2_complete_mapping_fullseq/adverse_prepost_series_formal_models_v31/train_oof_predictions.csv", "Logistic_Deep_probability"),
            "Valid": (ROOT / "outputs/api_png2d_gtmask_roi_cave_v2_complete_mapping_fullseq/adverse_prepost_series_formal_models_v31/valid_predictions.csv", "Logistic_Deep_probability"),
        },
        "C_Global_Deep_Gated": {
            "Train_OOF": (ROOT / "outputs/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready_strict/fusion/gated_interaction/train_oof_predictions.csv", "probability"),
            "Valid": (ROOT / "outputs/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready_strict/fusion/gated_interaction/valid_predictions.csv", "probability"),
        },
    }
    loaded = {
        name: {split: load_predictions(path, column, split) for split, (path, column) in by_split.items()}
        for name, by_split in paths.items()
    }

    paired_rows = []
    self_rows = []
    alignment = []
    for split, seed_offset in (("Train_OOF", 0), ("Valid", 100)):
        a = loaded["A"][split]
        for other_name, other_seed_offset in (("B0_Historical_CAVE_Deep_Logistic", 10), ("C_Global_Deep_Gated", 20)):
            left, right = align(a, loaded[other_name][split], "A", other_name)
            alignment.append({"split": split, "model_A": "A", "model_B": other_name, "rows": int(len(left)), "patients": int(left["patient_id"].nunique()), "strict_alignment": True})
            boot = bootstrap_pair(left, right, BOOTSTRAP_SEED + seed_offset + other_seed_offset)
            point_a = metric_values(left["target"], left["probability"])
            point_b = metric_values(right["target"], right["probability"])
            for metric, values in boot.items():
                delta = values[:, 0] - values[:, 1]
                paired_rows.append({
                    "model_A": "A_PredROI_CAVE_Deep_Gated",
                    "model_B": other_name,
                    "split": split,
                    "metric": metric,
                    "point_A": point_a[metric],
                    "point_B": point_b[metric],
                    "delta_A_minus_B": point_a[metric] - point_b[metric],
                    "CI_low": float(np.quantile(delta, 0.025)),
                    "CI_high": float(np.quantile(delta, 0.975)),
                    "n_bootstrap": N_BOOTSTRAP,
                    "bootstrap_seed": BOOTSTRAP_SEED + seed_offset + other_seed_offset,
                    "cluster_unit": "patient_id_with_replacement_multiplicity_preserved",
                })

        self_boot = bootstrap_self(a, BOOTSTRAP_SEED + seed_offset + 30)
        point = metric_values(a["target"], a["probability"])
        for metric, values in self_boot.items():
            self_rows.append({
                "model": "A_PredROI_CAVE_Deep_Gated",
                "split": split,
                "metric": metric,
                "point": point[metric],
                "CI_low": float(np.quantile(values, 0.025)),
                "CI_high": float(np.quantile(values, 0.975)),
                "n_bootstrap": N_BOOTSTRAP,
                "bootstrap_seed": BOOTSTRAP_SEED + seed_offset + 30,
                "cluster_unit": "patient_id_with_replacement_multiplicity_preserved",
            })

    paired = pd.DataFrame(paired_rows)
    self_ci = pd.DataFrame(self_rows)
    alignment_df = pd.DataFrame(alignment)
    atomic_csv(paired, report / "07_paired_patient_bootstrap.csv")
    atomic_csv(self_ci, report / "07_primary_patient_bootstrap_ci.csv")
    atomic_csv(alignment_df, report / "07_bootstrap_alignment_audit.csv")
    atomic_json({
        "status": "success",
        "n_bootstrap": N_BOOTSTRAP,
        "bootstrap_seed_base": BOOTSTRAP_SEED,
        "cluster_bootstrap": "patient_id sampled with replacement; each sampled patient contributes all of its series for every draw, including repeated draws",
        "valid_used_for_selection": False,
        "paired_comparisons": paired.to_dict("records"),
        "primary_self_ci": self_ci.to_dict("records"),
        "alignment": alignment_df.to_dict("records"),
    }, report / "07_paired_patient_bootstrap.json")


if __name__ == "__main__":
    main()

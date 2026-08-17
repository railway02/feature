#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from common import atomic_csv, atomic_json, load_config, load_temporal, load_train_folds, resolve_path
from fusion_models import OutcomeModel


def load_spatial(feature_root, fold, task_uid):
    path = feature_root / "seg_features" / f"fold_{fold}" / "train.npz"
    with np.load(path, allow_pickle=False) as z:
        values = np.asarray(z["pred_combined"], dtype=np.float32)
        uid = z["series_uid"].astype(str)
    if not np.array_equal(uid, task_uid.astype(str)):
        raise AssertionError(f"UID order mismatch: {path}")
    return values


def gate_statistics(name, values, target):
    clipped = np.clip(values, 1e-7, 1.0 - 1e-7)
    entropy = -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))
    case_mean = values.mean(axis=1)
    hist, edges = np.histogram(values.ravel(), bins=50, range=(0.0, 1.0))
    dim = pd.DataFrame({
        "gate": name,
        "dimension": np.arange(values.shape[1]),
        "mean": values.mean(axis=0),
        "std": values.std(axis=0, ddof=0),
    })
    histogram = pd.DataFrame({
        "gate": name,
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "count": hist,
        "fraction": hist / float(values.size),
    })
    groups = []
    for label, mask in (("all", np.ones(len(target), dtype=bool)), ("negative", target == 0), ("positive", target == 1)):
        subset = case_mean[mask]
        groups.append({
            "gate": name,
            "group": label,
            "n_cases": int(mask.sum()),
            "case_mean_mean": float(subset.mean()),
            "case_mean_std": float(subset.std(ddof=0)),
            "case_mean_median": float(np.median(subset)),
        })
    summary = {
        "overall_mean": float(values.mean()),
        "overall_std": float(values.std(ddof=0)),
        "fraction_lt_0_1": float((values < 0.1).mean()),
        "fraction_gt_0_9": float((values > 0.9).mean()),
        "fraction_0_4_to_0_6": float(((values >= 0.4) & (values <= 0.6)).mean()),
        "entropy_mean": float(entropy.mean()),
        "entropy_median": float(np.median(entropy)),
    }
    return summary, dim, histogram, pd.DataFrame(groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--report-root", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if cfg["spatial"].get("representation") != "global_pred_roi":
        raise AssertionError("Gate audit is fixed to A global_pred_roi config")
    if cfg.get("temporal", {}).get("representation") != "deep_only":
        raise AssertionError("Gate audit is fixed to A deep_only config")
    if cfg["fusion"]["modes"] != ["gated_interaction"]:
        raise AssertionError("Gate audit is fixed to gated_interaction only")

    output_root = resolve_path(cfg["output_root"], cfg["project_root"])
    feature_root = resolve_path(cfg["spatial"]["feature_root"], cfg["project_root"])
    report = Path(args.report_root).resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    train = load_temporal(cfg, "Train")
    folds = load_train_folds(cfg, train)
    n = len(train["target"])
    a2d = np.full((n, 256), np.nan, dtype=np.float32)
    at = np.full((n, 256), np.nan, dtype=np.float32)
    seen = np.zeros(n, dtype=bool)

    for fold in range(1, 6):
        holdout = np.flatnonzero(folds == fold)
        spatial = load_spatial(feature_root, fold, train["series_uid"])
        temporal = np.asarray(train["deep"], dtype=np.float32)
        checkpoint = output_root / "fusion" / "gated_interaction" / f"fold_{fold}" / "model.pt"
        raw = torch.load(checkpoint, map_location="cpu")
        model = OutcomeModel(
            mode="gated_interaction",
            spatial_dim=spatial.shape[1],
            temporal_dim=temporal.shape[1],
            hidden_dim=int(cfg["fusion"]["hidden_dim"]),
            fusion_mid_dim=int(cfg["fusion"]["fusion_mid_dim"]),
            dropout=float(cfg["fusion"]["dropout"]),
        ).to(device)
        model.load_state_dict(raw["state_dict"], strict=True)
        model.eval()
        ds = TensorDataset(torch.from_numpy(spatial[holdout]), torch.from_numpy(temporal[holdout]))
        dl = DataLoader(ds, batch_size=int(cfg["fusion"]["batch_size"]), shuffle=False, drop_last=False)
        offset = 0
        with torch.no_grad():
            for s, t in dl:
                out = model(s.to(device), t.to(device))
                count = len(s)
                a2d[holdout[offset:offset + count]] = out["gate_2d"].cpu().numpy().astype(np.float32)
                at[holdout[offset:offset + count]] = out["gate_t"].cpu().numpy().astype(np.float32)
                offset += count
        if offset != len(holdout) or seen[holdout].any():
            raise AssertionError(f"Fold {fold}: holdout assignment failure")
        seen[holdout] = True

    if not seen.all() or a2d.shape != (781, 256) or at.shape != (781, 256):
        raise AssertionError("Expected exactly one 256-D gate vector per Train series")
    if not (np.isfinite(a2d).all() and np.isfinite(at).all()):
        raise AssertionError("Nonfinite gate value")
    if len(np.unique(train["series_uid"].astype(str))) != 781:
        raise AssertionError("series_uid must be unique")

    np.savez_compressed(
        report / "08_primary_oof_full_gates.npz",
        a2D=a2d,
        aT=at,
        series_uid=train["series_uid"].astype(str),
        patient_id=train["patient_id"].astype(str),
        target=train["target"].astype(np.int64),
        fold=folds.astype(np.int64),
    )
    s2d, d2d, h2d, g2d = gate_statistics("a2D", a2d, train["target"])
    st, dt, ht, gt = gate_statistics("aT", at, train["target"])
    atomic_csv(pd.concat([d2d, dt], ignore_index=True), report / "08_primary_gate_per_dimension.csv")
    atomic_csv(pd.concat([h2d, ht], ignore_index=True), report / "08_primary_gate_histogram.csv")
    atomic_csv(pd.concat([g2d, gt], ignore_index=True), report / "08_primary_gate_case_mean_by_target.csv")
    atomic_json({
        "status": "success",
        "evaluation_only": True,
        "model_config": str(Path(args.config).resolve()),
        "checkpoint_root": str(output_root),
        "feature_root": str(feature_root),
        "model_eval": True,
        "torch_no_grad": True,
        "oof_shape_a2D": list(a2d.shape),
        "oof_shape_aT": list(at.shape),
        "unique_series_uid": int(len(np.unique(train["series_uid"].astype(str)))),
        "a2D": s2d,
        "aT": st,
        "entropy_clip": [1e-7, 1.0 - 1e-7],
    }, report / "08_primary_gate_audit.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from common import atomic_csv, atomic_json, load_config, resolve_path
from data import SegPhaseDataset
from segresnet_model import build_segresnet


def load_model(checkpoint: Path, cfg, device):
    model = build_segresnet(cfg)
    raw = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(raw["state_dict"], strict=True)
    model.to(device).eval()
    return model


@torch.no_grad()
def audit_fold(model, cases, cfg, device, fold):
    ds = SegPhaseDataset(cases, cfg, augment=False)
    dl = DataLoader(
        ds,
        batch_size=int(cfg["feature_extraction"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["feature_extraction"]["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(cfg["feature_extraction"]["num_workers"]) > 0,
    )
    amp = bool(cfg["feature_extraction"]["amp"] and device.type == "cuda")
    rows = []
    for x, y, idx in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast(enabled=amp):
            pred = (torch.sigmoid(model(x)) >= 0.5).float()
        inter = (pred * y).sum(dim=(1, 2, 3))
        pred_area = pred.sum(dim=(1, 2, 3))
        gt_area = y.sum(dim=(1, 2, 3))
        dice = (2.0 * inter + 1e-6) / (pred_area + gt_area + 1e-6)
        iou = (inter + 1e-6) / (pred_area + gt_area - inter + 1e-6)
        for j, source_idx in enumerate(idx.tolist()):
            meta = ds.rows.iloc[int(source_idx)]
            rows.append({
                "fold": int(fold),
                "series_uid": str(meta["series_uid"]),
                "patient_id": str(meta["patient_id"]),
                "phase": str(meta["phase"]),
                "dice": float(dice[j].cpu()),
                "iou": float(iou[j].cpu()),
                "pred_empty": bool(pred_area[j].item() == 0),
                "pred_pixels": int(pred_area[j].item()),
                "gt_pixels": int(gt_area[j].item()),
                "pred_gt_area_ratio": float((pred_area[j] / gt_area[j].clamp_min(1.0)).cpu()),
            })
    return pd.DataFrame(rows)


def summarize(rows):
    groups = []
    for fold in sorted(rows["fold"].unique()):
        fold_rows = rows[rows["fold"] == fold]
        for phase, subset in (("Pre", fold_rows[fold_rows["phase"] == "Pre"]), ("Post", fold_rows[fold_rows["phase"] == "Post"]), ("Overall", fold_rows)):
            groups.append({
                "fold": int(fold),
                "phase": phase,
                "n": int(len(subset)),
                "dice_mean": float(subset["dice"].mean()),
                "iou_mean": float(subset["iou"].mean()),
                "empty_pred_rate": float(subset["pred_empty"].mean()),
                "pred_gt_area_ratio_mean": float(subset["pred_gt_area_ratio"].mean()),
                "pred_gt_area_ratio_median": float(subset["pred_gt_area_ratio"].median()),
            })
    return pd.DataFrame(groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source-output-root", required=True)
    ap.add_argument("--report-root", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = load_config(args.config)
    source = Path(args.source_output_root).resolve()
    report = Path(args.report_root).resolve()
    manifest = pd.read_csv(source / "case_manifest.csv", dtype={"patient_id": str, "series_uid": str})
    train = manifest[manifest["split"] == "Train"].copy()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    phase_rows = []
    for fold in range(1, 6):
        holdout = train[train["fold"].astype(int) == fold].copy()
        checkpoint = source / "segmentation" / f"fold_{fold}" / "model.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        phase_rows.append(audit_fold(load_model(checkpoint, cfg, device), holdout, cfg, device, fold))

    detail = pd.concat(phase_rows, ignore_index=True)
    summary = summarize(detail)
    atomic_csv(detail, report / "05_strict_outer_segmentation_phase_predictions.csv")
    atomic_csv(summary, report / "05_strict_outer_segmentation_metrics.csv")
    atomic_json({
        "status": "success",
        "evaluation_only": True,
        "selection_or_training_used": False,
        "source_output_root": str(source),
        "folds": [1, 2, 3, 4, 5],
        "rows": int(len(detail)),
        "metrics": summary.to_dict("records"),
    }, report / "05_strict_outer_segmentation_audit.json")


if __name__ == "__main__":
    main()

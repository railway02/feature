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

from common import atomic_json, atomic_npz, load_config, load_temporal, resolve_path
from data import FeaturePhaseDataset
from segresnet_model import (
    build_segresnet,
    encode_and_decode,
    global_pool,
    mask_pool,
    maybe_load_external_checkpoint,
)


def load_model(model_name, cfg, out, device):
    strategy = str(cfg["spatial"]["strategy"]).casefold()
    model = build_segresnet(cfg)

    if strategy == "external_checkpoint":
        info = maybe_load_external_checkpoint(model, cfg)
        if not info.get("used"):
            raise RuntimeError("external_checkpoint strategy has no checkpoint")
    else:
        ckpt = out / "segmentation" / model_name / "model.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        raw = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(raw["state_dict"], strict=True)
        info = {"used": True, "path": str(ckpt)}

    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, info


@torch.no_grad()
def extract_phase_features(model, case_frame, cfg, device):
    ds = FeaturePhaseDataset(case_frame, cfg)
    dl = DataLoader(
        ds,
        batch_size=int(cfg["feature_extraction"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["feature_extraction"]["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(cfg["feature_extraction"]["num_workers"]) > 0,
    )

    amp = bool(cfg["feature_extraction"]["amp"] and device.type == "cuda")
    eps = float(cfg["feature_extraction"]["epsilon"])
    temp = float(cfg["feature_extraction"]["pred_mask_temperature"])

    global_feat = None
    gt_roi_feat = None
    pred_roi_feat = None
    gt_feature_mass = None
    pred_feature_mass = None

    for x, gt_mask, idx in dl:
        x = x.to(device, non_blocking=True)
        gt_mask = gt_mask.to(device, non_blocking=True)

        with autocast(enabled=amp):
            fmap, logits = encode_and_decode(model, x)

            zg = global_pool(fmap)

            pred_prob = torch.sigmoid(logits / temp)
            z_pred, pred_mass = mask_pool(
                fmap,
                pred_prob,
                resize_mode="bilinear",
                eps=eps,
            )

            z_gt, gt_mass = mask_pool(
                fmap,
                gt_mask,
                resize_mode=str(cfg["feature_extraction"].get("gt_roi_resize_mode", "area")),
                eps=eps,
                require_nonzero=True,
            )

        zg = zg.float().cpu().numpy()
        z_pred = z_pred.float().cpu().numpy()
        z_gt = z_gt.float().cpu().numpy()
        idx = np.asarray(idx)

        if global_feat is None:
            global_feat = np.empty((len(ds), zg.shape[1]), dtype=np.float32)
            pred_roi_feat = np.empty((len(ds), z_pred.shape[1]), dtype=np.float32)
            gt_roi_feat = np.empty((len(ds), z_gt.shape[1]), dtype=np.float32)
            gt_feature_mass = np.empty(len(ds), dtype=np.float32)
            pred_feature_mass = np.empty(len(ds), dtype=np.float32)

        global_feat[idx] = zg
        pred_roi_feat[idx] = z_pred
        gt_roi_feat[idx] = z_gt
        gt_feature_mass[idx] = gt_mass.float().cpu().numpy().reshape(-1)
        pred_feature_mass[idx] = pred_mass.float().cpu().numpy().reshape(-1)

    audit = ds.rows.copy()
    audit["gt_feature_roi_mass"] = gt_feature_mass
    audit["pred_feature_roi_mass"] = pred_feature_mass
    audit["gt_roi_vanished"] = audit["gt_feature_roi_mass"] <= eps
    return ds.rows, global_feat, gt_roi_feat, pred_roi_feat, audit


def pack_series(case_frame, phase_rows, zg, zgt, zpred):
    lookup = {
        (str(r.series_uid), str(r.phase)): i
        for i, r in enumerate(phase_rows.itertuples(index=False))
    }

    global_series = []
    gt_roi_series = []
    pred_roi_series = []
    gt_combined = []
    pred_combined = []

    for r in case_frame.itertuples(index=False):
        pre = lookup[(str(r.series_uid), "Pre")]
        post = lookup[(str(r.series_uid), "Post")]

        g = np.concatenate([zg[pre], zg[post]])
        gt = np.concatenate([zgt[pre], zgt[post]])
        pred = np.concatenate([zpred[pre], zpred[post]])

        # Teacher-aligned z2D_raw:
        # [Pre-global, Pre-ROI, Post-global, Post-ROI]
        gt_c = np.concatenate([zg[pre], zgt[pre], zg[post], zgt[post]])
        pred_c = np.concatenate([zg[pre], zpred[pre], zg[post], zpred[post]])

        global_series.append(g)
        gt_roi_series.append(gt)
        pred_roi_series.append(pred)
        gt_combined.append(gt_c)
        pred_combined.append(pred_c)

    return {
        "global": np.stack(global_series).astype(np.float32),
        "gt_roi": np.stack(gt_roi_series).astype(np.float32),
        "pred_roi": np.stack(pred_roi_series).astype(np.float32),
        "gt_combined": np.stack(gt_combined).astype(np.float32),
        "pred_combined": np.stack(pred_combined).astype(np.float32),
    }


def run_model(model_name, cfg, out, device):
    model, model_info = load_model(model_name, cfg, out, device)

    feature_root = out / "seg_features" / model_name
    feature_root.mkdir(parents=True, exist_ok=True)

    summary = {"model_name": model_name, "model_info": model_info, "splits": {}}

    for split in ("Train", "Valid"):
        temporal = load_temporal(cfg, split)
        manifest = pd.read_csv(
            out / "case_manifest.csv",
            dtype={"patient_id": str, "series_uid": str},
        )
        case_frame = (
            manifest[manifest["split"] == split]
            .sort_values("task_row")
            .reset_index(drop=True)
        )

        if not np.array_equal(
            case_frame["series_uid"].astype(str).to_numpy(),
            temporal["series_uid"].astype(str),
        ):
            raise AssertionError(f"{split}: manifest/temporal order mismatch")

        phase_rows, zg, zgt, zpred, roi_audit = extract_phase_features(
            model,
            case_frame,
            cfg,
            device,
        )
        features = pack_series(case_frame, phase_rows, zg, zgt, zpred)
        roi_audit["split"] = split
        roi_audit["model_name"] = model_name
        roi_audit.to_csv(feature_root / f"{split.casefold()}_roi_pool_audit.csv", index=False)
        if bool(roi_audit["gt_roi_vanished"].any()):
            raise RuntimeError(f"{split}: GT ROI vanished after area pooling")

        atomic_npz(
            feature_root / f"{split.casefold()}.npz",
            **features,
            series_uid=case_frame["series_uid"].astype(str).to_numpy(dtype=str),
            patient_id=case_frame["patient_id"].astype(str).to_numpy(dtype=str),
            target=case_frame["target"].to_numpy(np.int64),
        )

        summary["splits"][split] = {
            key: list(value.shape)
            for key, value in features.items()
        }

    summary["status"] = "success"
    atomic_json(summary, feature_root / ".SUCCESS.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", default="all", help="Only used by strict_crossfit")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = resolve_path(cfg["output_root"], cfg["project_root"])

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)

    strategy = str(cfg["spatial"]["strategy"]).casefold()

    if strategy == "pilot_single":
        run_model("pilot", cfg, out, device)
        return

    if strategy == "external_checkpoint":
        run_model("external", cfg, out, device)
        return

    if strategy != "strict_crossfit":
        raise ValueError(strategy)

    folds = range(1, 6) if args.fold == "all" else [int(args.fold)]
    for fold in folds:
        run_model(f"fold_{fold}", cfg, out, device)


if __name__ == "__main__":
    main()

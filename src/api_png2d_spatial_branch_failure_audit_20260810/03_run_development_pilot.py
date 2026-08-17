#!/usr/bin/env python3
"""Leakage-safe development-only spatial segmentation pilot.

The pilot uses one predefined outer-development population and its original
patient-level inner split.  It never evaluates the corresponding outer
holdout and never reads the independent Valid cohort for selection.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from pilot_data import PilotSegmentationDataset
from pilot_models import build_pilot_model


PROJECT = Path("/root/autodl-tmp/aneurysm")
V5_CODE = PROJECT / "code/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready"
CONFIG = PROJECT / "configs/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready_strict.json"
SOURCE = PROJECT / "outputs/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready_strict"
OUTPUT = PROJECT / "outputs/api_png2d_spatial_branch_failure_audit_20260810_pilot"
REPORT = PROJECT / "reports/api_png2d_spatial_branch_failure_audit_20260810/pilot"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=[
        "frozen_baseline",
        "segresnet_geometry",
        "segresnet_geometry_pos3",
        "deeplabv3_resnet50_pretrained",
    ])
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--source-output-root", default=str(SOURCE))
    ap.add_argument("--v5-code-root", default=str(V5_CODE))
    ap.add_argument("--output-root", default=str(OUTPUT))
    ap.add_argument("--report-root", default=str(REPORT))
    ap.add_argument("--pilot-outer-fold", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--min-epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--smoke-steps", type=int, default=0)
    ap.add_argument("--run-name", default="")
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


def make_loader(frame, cfg, train, seed, prepare_pair, geometry_enabled, batch_size, num_workers, seed_worker):
    dataset = PilotSegmentationDataset(
        frame,
        cfg,
        augment=train,
        prepare_pair=prepare_pair,
        geometry_enabled=geometry_enabled,
    )
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(train),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(num_workers) > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def train_epoch(model, loader, optimizer, scaler, device, pos_weight, cfg, segmentation_loss, smoke_steps=0):
    model.train()
    amp = bool(cfg["segresnet"]["amp"] and device.type == "cuda")
    losses, dices = [], []
    started = time.time()
    for step, (x, y, _) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            logits = model(x)
            loss = segmentation_loss(logits, y, pos_weight, cfg)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            pred = torch.sigmoid(logits) >= 0.5
            inter = (pred * (y > 0)).sum(dim=(1, 2, 3))
            denom = pred.sum(dim=(1, 2, 3)) + (y > 0).sum(dim=(1, 2, 3))
            batch_dice = (2 * inter + 1e-6) / (denom + 1e-6)
        losses.extend([float(loss.detach().cpu())] * len(x))
        dices.extend(batch_dice.float().cpu().tolist())
        if smoke_steps > 0 and step >= smoke_steps:
            break
    return {
        "train_loss": float(np.mean(losses)),
        "train_dice": float(np.mean(dices)),
        "train_seconds": float(time.time() - started),
        "train_steps": int(step),
    }


@torch.no_grad()
def evaluate(model, loader, device, cfg, max_batches=0):
    model.eval()
    amp = bool(cfg["segresnet"]["amp"] and device.type == "cuda")
    records = []
    for batch_index, (x, y, indices) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast(enabled=amp):
            probability = torch.sigmoid(model(x))
        pred = probability >= 0.5
        target = y > 0
        inter = (pred & target).sum(dim=(1, 2, 3))
        pred_area = pred.sum(dim=(1, 2, 3))
        gt_area = target.sum(dim=(1, 2, 3))
        dice = (2 * inter + 1e-6) / (pred_area + gt_area + 1e-6)
        iou = (inter + 1e-6) / (pred_area + gt_area - inter + 1e-6)
        for j, source_index in enumerate(indices.tolist()):
            meta = loader.dataset.rows.iloc[int(source_index)]
            records.append({
                "patient_id": str(meta.patient_id),
                "series_uid": str(meta.series_uid),
                "phase": str(meta.phase),
                "dice": float(dice[j].cpu()),
                "iou": float(iou[j].cpu()),
                "inter": int(inter[j].cpu()),
                "pred_pixels": int(pred_area[j].cpu()),
                "gt_pixels": int(gt_area[j].cpu()),
                "pred_gt_area_ratio": float((pred_area[j] / gt_area[j].clamp_min(1)).cpu()),
            })
        if max_batches > 0 and batch_index >= max_batches:
            break
    rows = pd.DataFrame(records)
    total_inter = rows.inter.sum()
    total_pred = rows.pred_pixels.sum()
    total_gt = rows.gt_pixels.sum()
    metrics = {
        "n": int(len(rows)),
        "macro_dice": float(rows.dice.mean()),
        "micro_dice": float((2 * total_inter + 1e-6) / (total_pred + total_gt + 1e-6)),
        "macro_iou": float(rows.iou.mean()),
        "pre_dice": float(rows.loc[rows.phase == "Pre", "dice"].mean()),
        "post_dice": float(rows.loc[rows.phase == "Post", "dice"].mean()),
        "failure_lt_02": float((rows.dice < 0.2).mean()),
        "failure_lt_05": float((rows.dice < 0.5).mean()),
        "pred_gt_area_ratio_mean": float(rows.pred_gt_area_ratio.mean()),
        "pred_gt_area_ratio_total": float(total_pred / max(1, total_gt)),
    }
    return metrics, rows


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def main():
    args = parse_args()
    sys.path.insert(0, str(Path(args.v5_code_root).resolve()))
    from common import load_config, patient_group_split, seed_worker, set_seed  # pylint: disable=import-error,import-outside-toplevel
    from data import estimate_bce_pos_weight, prepare_pair  # pylint: disable=import-error,import-outside-toplevel
    from segresnet_model import build_segresnet, segmentation_loss  # pylint: disable=import-error,import-outside-toplevel

    cfg = copy.deepcopy(load_config(args.config))
    cfg["segresnet"]["augmentation"].update({
        "geometry_probability": 0.80,
        "rotation_degrees": 10.0,
        "translate_fraction": 0.06,
        "scale_delta": 0.10,
    })
    if args.variant in {"segresnet_geometry_pos3", "deeplabv3_resnet50_pretrained"}:
        cfg["segresnet"]["max_bce_pos_weight"] = 3.0

    source = Path(args.source_output_root).resolve()
    output_root = Path(args.output_root).resolve()
    report_root = Path(args.report_root).resolve()
    run_name = args.run_name.strip() or args.variant
    if args.smoke_steps > 0 and not args.run_name.strip():
        run_name = f"smoke_{args.variant}"
    run_dir = output_root / f"outer_development_fold_{args.pilot_outer_fold}" / run_name
    run_report = report_root / f"outer_development_fold_{args.pilot_outer_fold}" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    run_report.mkdir(parents=True, exist_ok=True)

    success = run_dir / "SUCCESS.json"
    if success.is_file() and args.smoke_steps == 0:
        print(success.read_text(encoding="utf-8"))
        return

    manifest = pd.read_csv(
        source / "case_manifest.csv", dtype={"patient_id": str, "series_uid": str}
    )
    train = manifest[manifest.split == "Train"].copy()
    development = train[train.fold.astype(int) != args.pilot_outer_fold].copy()
    outer_holdout = train[train.fold.astype(int) == args.pilot_outer_fold].copy()
    if set(development.patient_id) & set(outer_holdout.patient_id):
        raise AssertionError("Outer patient leakage")

    seed = int(cfg["segresnet"]["seed"]) + args.pilot_outer_fold * 100
    patients = development[["patient_id"]].drop_duplicates().reset_index(drop=True)
    train_idx, valid_idx = patient_group_split(
        patients.patient_id.to_numpy(),
        float(cfg["segresnet"]["inner_val_fraction"]),
        seed,
    )
    inner_train_patients = set(patients.iloc[train_idx].patient_id)
    inner_valid_patients = set(patients.iloc[valid_idx].patient_id)
    inner_train = development[development.patient_id.isin(inner_train_patients)].copy()
    inner_valid = development[development.patient_id.isin(inner_valid_patients)].copy()
    if inner_train_patients & inner_valid_patients:
        raise AssertionError("Inner patient leakage")
    if set(inner_train.patient_id) & set(outer_holdout.patient_id):
        raise AssertionError("Outer holdout entered inner training")
    if set(inner_valid.patient_id) & set(outer_holdout.patient_id):
        raise AssertionError("Outer holdout entered inner validation")

    split_rows = pd.concat([
        inner_train.assign(pilot_partition="inner_train"),
        inner_valid.assign(pilot_partition="inner_valid"),
        outer_holdout.assign(pilot_partition="forbidden_outer_holdout_not_used"),
    ], ignore_index=True)
    split_rows.to_csv(run_report / "pilot_patient_split.csv", index=False)

    batch_size = args.batch_size or (
        2 if args.variant == "deeplabv3_resnet50_pretrained" else int(cfg["segresnet"]["batch_size"])
    )
    learning_rate = args.learning_rate or (
        5e-5 if args.variant == "deeplabv3_resnet50_pretrained" else float(cfg["segresnet"]["learning_rate"])
    )
    geometry_enabled = args.variant != "frozen_baseline"

    valid_loader = make_loader(
        inner_valid, cfg, False, seed + 1, prepare_pair, False,
        batch_size, args.num_workers, seed_worker,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    set_seed(seed)

    if args.variant == "frozen_baseline":
        model = build_segresnet(cfg).to(device)
        checkpoint = source / "segmentation" / f"fold_{args.pilot_outer_fold}" / "search_best.pt"
        raw = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(raw["state_dict"], strict=True)
        metrics, rows = evaluate(model, valid_loader, device, cfg)
        rows.to_csv(run_report / "inner_valid_predictions.csv", index=False)
        result = {
            "status": "success",
            "variant": args.variant,
            "selection_population": f"Train folds != {args.pilot_outer_fold}; original patient inner split",
            "outer_holdout_evaluated": False,
            "independent_valid_used": False,
            "checkpoint": str(checkpoint),
            "checkpoint_epoch": int(raw["epoch"]),
            "metrics": metrics,
        }
        save_json(success, result)
        save_json(run_report / "metrics.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    train_loader = make_loader(
        inner_train, cfg, True, seed, prepare_pair, geometry_enabled,
        batch_size, args.num_workers, seed_worker,
    )
    model, model_info = build_pilot_model(args.variant, cfg, build_segresnet)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        test_input = torch.zeros(1, 1, int(cfg["spatial"]["input_size"]), int(cfg["spatial"]["input_size"]), device=device)
        if hasattr(model, "encode_and_decode"):
            fmap, test_logits = model.encode_and_decode(test_input)
            model_info["observed_feature_map_shape"] = list(fmap.shape[1:])
        else:
            test_logits = model(test_input)
        model_info["observed_logits_shape"] = list(test_logits.shape[1:])
    del test_input, test_logits
    if device.type == "cuda":
        torch.cuda.empty_cache()

    pos_weight = estimate_bce_pos_weight(inner_train, cfg)
    optimizer = AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(cfg["segresnet"]["weight_decay"])
    )
    scaler = GradScaler(enabled=bool(cfg["segresnet"]["amp"] and device.type == "cuda"))
    start_epoch, best_dice, best_epoch, bad_epochs = 1, -1.0, 0, 0
    history = []
    last_checkpoint = run_dir / "last.pt"
    if args.resume and last_checkpoint.is_file():
        state = torch.load(last_checkpoint, map_location="cpu")
        model.load_state_dict(state["state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        best_dice = float(state["best_dice"])
        best_epoch = int(state["best_epoch"])
        bad_epochs = int(state["bad_epochs"])
        history = list(state["history"])

    max_epochs = 1 if args.smoke_steps > 0 else int(args.max_epochs)
    config_snapshot = {
        "variant": args.variant,
        "run_name": run_name,
        "pilot_outer_fold": args.pilot_outer_fold,
        "seed": seed,
        "development_series": int(len(development)),
        "inner_train_series": int(len(inner_train)),
        "inner_valid_series": int(len(inner_valid)),
        "forbidden_outer_holdout_series": int(len(outer_holdout)),
        "outer_holdout_evaluated": False,
        "independent_valid_used": False,
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "max_epochs": int(max_epochs),
        "min_epochs": int(args.min_epochs),
        "patience": int(args.patience),
        "bce_pos_weight": float(pos_weight),
        "augmentation": cfg["segresnet"]["augmentation"],
        "model": model_info,
    }
    save_json(run_report / "config_snapshot.json", config_snapshot)

    for epoch in range(start_epoch, max_epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, device, pos_weight, cfg,
            segmentation_loss, smoke_steps=args.smoke_steps,
        )
        valid_metrics, _ = evaluate(
            model, valid_loader, device, cfg,
            max_batches=1 if args.smoke_steps > 0 else 0,
        )
        epoch_record = {"epoch": epoch, **train_metrics, **{f"valid_{k}": v for k, v in valid_metrics.items()}}
        history.append(epoch_record)
        pd.DataFrame(history).to_csv(run_report / "history.csv", index=False)
        print(json.dumps(epoch_record, ensure_ascii=False), flush=True)

        if valid_metrics["macro_dice"] > best_dice + 1e-5:
            best_dice = float(valid_metrics["macro_dice"])
            best_epoch = int(epoch)
            bad_epochs = 0
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "valid_metrics": valid_metrics,
                "model_info": model_info,
                "config_snapshot": config_snapshot,
            }, run_dir / "best.pt")
        else:
            bad_epochs += 1

        torch.save({
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_dice": best_dice,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
            "history": history,
        }, last_checkpoint)
        if args.smoke_steps == 0 and epoch >= int(args.min_epochs) and bad_epochs >= int(args.patience):
            break

    best = torch.load(run_dir / "best.pt", map_location="cpu")
    model.load_state_dict(best["state_dict"], strict=True)
    final_metrics, final_rows = evaluate(
        model, valid_loader, device, cfg,
        max_batches=1 if args.smoke_steps > 0 else 0,
    )
    final_rows.to_csv(run_report / "best_inner_valid_predictions.csv", index=False)
    result = {
        "status": "smoke_success" if args.smoke_steps > 0 else "success",
        "variant": args.variant,
        "run_name": run_name,
        "selection_population": f"Train folds != {args.pilot_outer_fold}; original patient inner split",
        "outer_holdout_evaluated": False,
        "independent_valid_used": False,
        "best_epoch": int(best_epoch),
        "best_inner_valid_metrics": final_metrics,
        "config_snapshot": config_snapshot,
    }
    save_json(run_dir / ("SMOKE_SUCCESS.json" if args.smoke_steps > 0 else "SUCCESS.json"), result)
    save_json(run_report / "metrics.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

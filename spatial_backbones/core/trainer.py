from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common import (
    assert_signature,
    atomic_csv,
    atomic_json,
    atomic_torch_save,
    run_signature,
    seed_worker,
    set_seed,
    sha256_file,
)
from data import SegmentationDataset, estimate_pos_weight, split_hash
from losses import segmentation_loss
from model_interface import build_model, model_parameter_count


def make_loader(frame, cfg, train: bool, seed: int, batch_size: int):
    dataset = SegmentationDataset(frame, cfg, augment=train)
    generator = torch.Generator().manual_seed(int(seed))
    workers = int(cfg["development"]["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(train),
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def train_epoch(model, loader, optimizer, scaler, device, pos_weight, cfg, accumulation: int):
    model.train()
    amp = bool(cfg["development"]["amp"] and device.type == "cuda")
    optimizer.zero_grad(set_to_none=True)
    losses = []
    intersections = 0.0
    denominators = 0.0
    geometry_applied = 0
    geometry_fallback = 0
    started = time.time()
    for step, (x, y, _, applied, fallback) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast(enabled=amp):
            logits = model(x)
            full_loss = segmentation_loss(logits, y, pos_weight, cfg)
            loss = full_loss / int(accumulation)
        scaler.scale(loss).backward()
        if step % int(accumulation) == 0 or step == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            prediction = torch.sigmoid(logits) >= float(cfg["development"]["threshold"])
            target = y > 0
            intersections += float((prediction & target).sum().cpu())
            denominators += float(prediction.sum().cpu() + target.sum().cpu())
        losses.extend([float(full_loss.detach().cpu())] * len(x))
        geometry_applied += int(applied.sum())
        geometry_fallback += int(fallback.sum())
    return {
        "train_loss": float(np.mean(losses)),
        "train_micro_dice": float((2.0 * intersections + 1e-6) / (denominators + 1e-6)),
        "train_seconds": float(time.time() - started),
        "train_steps": int(step),
        "geometry_applied": int(geometry_applied),
        "geometry_fallback": int(geometry_fallback),
    }


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    amp = bool(cfg["development"]["amp"] and device.type == "cuda")
    threshold = float(cfg["development"]["threshold"])
    records = []
    for x, y, indices, _, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast(enabled=amp):
            probability = torch.sigmoid(model(x))
        prediction = probability >= threshold
        target = y > 0
        intersection = (prediction & target).sum(dim=(1, 2, 3))
        pred_area = prediction.sum(dim=(1, 2, 3))
        gt_area = target.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + 1e-6) / (pred_area + gt_area + 1e-6)
        iou = (intersection + 1e-6) / (pred_area + gt_area - intersection + 1e-6)
        for item, source_index in enumerate(indices.tolist()):
            metadata = loader.dataset.rows.iloc[int(source_index)]
            gt_pixels = target[item]
            bg_pixels = ~gt_pixels
            records.append({
                "patient_id": str(metadata.patient_id),
                "series_uid": str(metadata.series_uid),
                "phase": str(metadata.phase),
                "image_path": str(metadata.image_path),
                "mask_path": str(metadata.mask_path),
                "dice": float(dice[item].cpu()),
                "iou": float(iou[item].cpu()),
                "intersection_pixels": int(intersection[item].cpu()),
                "pred_pixels": int(pred_area[item].cpu()),
                "gt_pixels": int(gt_area[item].cpu()),
                "pred_gt_area_ratio": float((pred_area[item] / gt_area[item].clamp_min(1)).cpu()),
                "zero_overlap": bool(intersection[item].item() == 0),
                "probability_mean_gt": float(probability[item][gt_pixels].mean().cpu()),
                "probability_mean_background": float(probability[item][bg_pixels].mean().cpu()),
                "probability_entropy_mean": float((-(probability[item].clamp(1e-6, 1-1e-6) * probability[item].clamp(1e-6, 1-1e-6).log() + (1-probability[item]).clamp(1e-6, 1-1e-6) * (1-probability[item]).clamp(1e-6, 1-1e-6).log())).mean().cpu()),
            })
    rows = pd.DataFrame(records)
    total_intersection = int(rows["intersection_pixels"].sum())
    total_prediction = int(rows["pred_pixels"].sum())
    total_gt = int(rows["gt_pixels"].sum())
    metrics = {
        "n_phase_images": int(len(rows)),
        "macro_dice": float(rows["dice"].mean()),
        "micro_dice": float((2.0 * total_intersection + 1e-6) / (total_prediction + total_gt + 1e-6)),
        "macro_iou": float(rows["iou"].mean()),
        "pre_dice": float(rows.loc[rows["phase"].eq("Pre"), "dice"].mean()),
        "post_dice": float(rows.loc[rows["phase"].eq("Post"), "dice"].mean()),
        "failure_lt_02": float(rows["dice"].lt(0.2).mean()),
        "failure_lt_05": float(rows["dice"].lt(0.5).mean()),
        "zero_overlap_rate": float(rows["zero_overlap"].mean()),
        "zero_overlap_count": int(rows["zero_overlap"].sum()),
        "pred_gt_area_ratio_mean": float(rows["pred_gt_area_ratio"].mean()),
        "pred_gt_area_ratio_median": float(rows["pred_gt_area_ratio"].median()),
        "pred_gt_area_ratio_total": float(total_prediction / max(1, total_gt)),
    }
    return metrics, rows


def is_better(metrics: dict[str, float], best: dict[str, float] | None, tolerance: float = 1e-5) -> bool:
    if best is None:
        return True
    if metrics["macro_dice"] > best["macro_dice"] + tolerance:
        return True
    if abs(metrics["macro_dice"] - best["macro_dice"]) <= tolerance:
        if metrics["post_dice"] > best["post_dice"] + tolerance:
            return True
        if abs(metrics["post_dice"] - best["post_dice"]) <= tolerance:
            return metrics["failure_lt_02"] < best["failure_lt_02"] - tolerance
    return False


def runtime_batch(cfg, family: str):
    if family == "segresnet":
        settings = cfg["models"][family]
        return int(settings["physical_batch_size"]), int(settings["gradient_accumulation"])
    runtime_path = Path(cfg["report_root"]) / "00_preflight/runtime_selection.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    selected = runtime[family]
    return int(selected["physical_batch_size"]), int(selected["gradient_accumulation"])


def train_development_model(cfg, family: str, outer_fold: int, inner_train, inner_valid, split_frame, device):
    model_cfg = cfg["models"][family]
    output_dir = Path(cfg["output_root"]) / "development" / f"fold_{outer_fold}" / family
    report_dir = Path(cfg["report_root"]) / "development" / f"fold_{outer_fold}" / family
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    split_digest = split_hash(split_frame)
    pretrained_digest = cfg["pretrained"]["sha256"] if bool(model_cfg["pretrained"]) else "random_initialization"
    signature = run_signature(cfg, split_digest, family, pretrained_digest)
    success_path = output_dir / "SUCCESS.json"
    if success_path.is_file():
        existing = json.loads(success_path.read_text(encoding="utf-8"))
        assert_signature(existing["run_signature"], signature)
        print(json.dumps(existing, ensure_ascii=False, indent=2), flush=True)
        return existing

    seed = int(cfg["development"]["base_seed"]) + int(outer_fold) * 100
    batch_size, accumulation = runtime_batch(cfg, family)
    pos_weight = estimate_pos_weight(inner_train, cfg)
    set_seed(seed)
    model = build_model(family, cfg, load_pretrained=True).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(model_cfg["learning_rate"]),
        weight_decay=float(model_cfg["weight_decay"]),
    )
    scaler = GradScaler(enabled=bool(cfg["development"]["amp"] and device.type == "cuda"))
    train_loader = make_loader(inner_train, cfg, True, seed, batch_size)
    valid_loader = make_loader(inner_valid, cfg, False, seed + 1, batch_size)

    snapshot = {
        "run_signature": signature,
        "outer_fold": int(outer_fold),
        "development_only": True,
        "outer_holdout_evaluated": False,
        "independent_valid_used": False,
        "seed": int(seed),
        "inner_train_series": int(len(inner_train)),
        "inner_valid_series": int(len(inner_valid)),
        "inner_train_patients": int(inner_train["patient_id"].nunique()),
        "inner_valid_patients": int(inner_valid["patient_id"].nunique()),
        "physical_batch_size": int(batch_size),
        "gradient_accumulation": int(accumulation),
        "effective_batch_size": int(batch_size * accumulation),
        "pos_weight": float(pos_weight),
        "threshold": float(cfg["development"]["threshold"]),
        "model_settings": model_cfg,
        "augmentation": cfg["augmentation"],
        "loss": cfg["loss"],
        "model_parameters": model_parameter_count(model),
    }
    atomic_json(snapshot, report_dir / "config_snapshot.json")
    atomic_csv(split_frame, report_dir / "split_manifest.csv")

    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_metrics = None
    best_epoch = 0
    bad_epochs = 0
    last_path = output_dir / "last.pt"
    if bool(cfg["development"].get("resume", True)) and last_path.is_file():
        state = torch.load(last_path, map_location="cpu")
        assert_signature(state["run_signature"], signature)
        model.load_state_dict(state["state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        history = list(state["history"])
        start_epoch = int(state["epoch"]) + 1
        best_metrics = state["best_metrics"]
        best_epoch = int(state["best_epoch"])
        bad_epochs = int(state["bad_epochs"])

    for epoch in range(start_epoch, int(cfg["development"]["max_epochs"]) + 1):
        training = train_epoch(model, train_loader, optimizer, scaler, device, pos_weight, cfg, accumulation)
        validation, _ = evaluate(model, valid_loader, device, cfg)
        record = {"epoch": int(epoch), **training, **{f"valid_{key}": value for key, value in validation.items()}}
        history.append(record)
        atomic_csv(pd.DataFrame(history), report_dir / "history.csv")
        print(json.dumps({"family": family, "outer_fold": outer_fold, **record}, ensure_ascii=False), flush=True)

        if is_better(validation, best_metrics):
            best_metrics = validation
            best_epoch = int(epoch)
            bad_epochs = 0
            atomic_torch_save({
                "state_dict": model.state_dict(),
                "epoch": int(epoch),
                "metrics": validation,
                "run_signature": signature,
                "config_snapshot": snapshot,
            }, output_dir / "search_best.pt")
        else:
            bad_epochs += 1

        atomic_torch_save({
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": int(epoch),
            "best_metrics": best_metrics,
            "best_epoch": int(best_epoch),
            "bad_epochs": int(bad_epochs),
            "history": history,
            "run_signature": signature,
        }, last_path)
        if epoch >= int(cfg["development"]["min_epochs"]) and bad_epochs >= int(cfg["development"]["patience"]):
            break

    best = torch.load(output_dir / "search_best.pt", map_location="cpu")
    assert_signature(best["run_signature"], signature)
    model.load_state_dict(best["state_dict"], strict=True)
    final_metrics, predictions = evaluate(model, valid_loader, device, cfg)
    predictions["gt_size_quartile"] = pd.qcut(
        predictions["gt_pixels"], 4, labels=["Q1-smallest", "Q2", "Q3", "Q4-largest"]
    ).astype(str)
    atomic_csv(predictions, report_dir / "best_inner_valid_predictions.csv")
    result = {
        "status": "success",
        "family": family,
        "outer_fold": int(outer_fold),
        "development_only": True,
        "outer_holdout_evaluated": False,
        "independent_valid_used": False,
        "best_epoch": int(best_epoch),
        "best_inner_valid_metrics": final_metrics,
        "run_signature": signature,
        "checkpoint": str(output_dir / "search_best.pt"),
        "checkpoint_sha256": sha256_file(output_dir / "search_best.pt"),
        "report_dir": str(report_dir),
    }
    atomic_json(result, report_dir / "metrics.json")
    atomic_json(result, success_path)
    return result

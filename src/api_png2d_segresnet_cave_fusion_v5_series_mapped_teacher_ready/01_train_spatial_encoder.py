#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common import (
    atomic_json,
    load_config,
    patient_group_split,
    resolve_path,
    seed_worker,
    set_seed,
)
from data import SegPhaseDataset, estimate_bce_pos_weight
from segresnet_model import (
    build_segresnet,
    dice_metric,
    maybe_load_external_checkpoint,
    segmentation_loss,
)


def make_loader(frame, cfg, train, seed):
    ds = SegPhaseDataset(frame, cfg, augment=train)
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=int(cfg["segresnet"]["batch_size"]),
        shuffle=train,
        num_workers=int(cfg["segresnet"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(cfg["segresnet"]["num_workers"]) > 0,
        worker_init_fn=seed_worker,
        generator=g,
    )


def fresh_model(cfg, device):
    model = build_segresnet(cfg)
    init_info = maybe_load_external_checkpoint(model, cfg)
    return model.to(device), init_info


def train_epoch(model, loader, optimizer, scaler, device, pos_weight, cfg):
    model.train()
    losses, dices = [], []
    amp = bool(cfg["segresnet"]["amp"] and device.type == "cuda")

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            logits = model(x)
            loss = segmentation_loss(logits, y, pos_weight, cfg)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(float(loss.detach().cpu()))
        dices.append(float(dice_metric(logits.detach(), y).cpu()))

    return float(np.mean(losses)), float(np.mean(dices))


@torch.no_grad()
def evaluate(model, loader, device, pos_weight, cfg):
    model.eval()
    losses, dices = [], []
    amp = bool(cfg["segresnet"]["amp"] and device.type == "cuda")

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast(enabled=amp):
            logits = model(x)
            loss = segmentation_loss(logits, y, pos_weight, cfg)

        losses.append(float(loss.cpu()))
        dices.append(float(dice_metric(logits, y).cpu()))

    return float(np.mean(losses)), float(np.mean(dices))


def search_epoch(train_cases, val_cases, cfg, device, seed, out_dir):
    set_seed(seed)
    model, init_info = fresh_model(cfg, device)

    pos_weight = estimate_bce_pos_weight(train_cases, cfg)
    c = cfg["segresnet"]

    optimizer = AdamW(
        model.parameters(),
        lr=float(c["learning_rate"]),
        weight_decay=float(c["weight_decay"]),
    )
    scaler = GradScaler(enabled=bool(c["amp"] and device.type == "cuda"))

    tr = make_loader(train_cases, cfg, True, seed)
    va = make_loader(val_cases, cfg, False, seed + 1)

    best_dice = -1.0
    best_epoch = 0
    bad = 0
    history = []

    for epoch in range(1, int(c["max_epochs"]) + 1):
        tr_loss, tr_dice = train_epoch(
            model, tr, optimizer, scaler, device, pos_weight, cfg
        )
        va_loss, va_dice = evaluate(
            model, va, device, pos_weight, cfg
        )

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_dice": tr_dice,
            "valid_loss": va_loss,
            "valid_dice": va_dice,
            "bce_pos_weight": pos_weight,
        })

        print(
            f"[seg-search] {out_dir.name} epoch={epoch:03d} "
            f"train_dice={tr_dice:.4f} valid_dice={va_dice:.4f}",
            flush=True,
        )

        if va_dice > best_dice + 1e-5:
            best_dice = va_dice
            best_epoch = epoch
            bad = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "valid_dice": va_dice,
                    "init_info": init_info,
                },
                out_dir / "search_best.pt",
            )
        else:
            bad += 1

        if epoch >= int(c["min_epochs"]) and bad >= int(c["patience"]):
            break

    pd.DataFrame(history).to_csv(out_dir / "search_history.csv", index=False)
    if best_epoch <= 0:
        raise RuntimeError("No best epoch selected")
    return best_epoch, best_dice, init_info


def refit(all_cases, epochs, cfg, device, seed, out_dir):
    set_seed(seed)
    model, init_info = fresh_model(cfg, device)

    pos_weight = estimate_bce_pos_weight(all_cases, cfg)
    c = cfg["segresnet"]

    optimizer = AdamW(
        model.parameters(),
        lr=float(c["learning_rate"]),
        weight_decay=float(c["weight_decay"]),
    )
    scaler = GradScaler(enabled=bool(c["amp"] and device.type == "cuda"))
    dl = make_loader(all_cases, cfg, True, seed)

    history = []
    for epoch in range(1, epochs + 1):
        loss, dice = train_epoch(
            model, dl, optimizer, scaler, device, pos_weight, cfg
        )
        history.append({
            "epoch": epoch,
            "train_loss": loss,
            "train_dice": dice,
            "bce_pos_weight": pos_weight,
        })
        print(
            f"[seg-refit] {out_dir.name} {epoch:03d}/{epochs} dice={dice:.4f}",
            flush=True,
        )

    pd.DataFrame(history).to_csv(out_dir / "refit_history.csv", index=False)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "selected_epochs": int(epochs),
            "segresnet_config": cfg["segresnet"],
            "init_info": init_info,
        },
        out_dir / "model.pt",
    )


def train_one(name, development, cfg, device, seed, out):
    out_dir = out / "segmentation" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    success = out_dir / ".SUCCESS.json"
    if success.is_file():
        print(f"{name} already complete")
        return

    patient = development[["patient_id"]].drop_duplicates("patient_id").reset_index(drop=True)
    tr_i, va_i = patient_group_split(
        patient["patient_id"].to_numpy(),
        float(cfg["segresnet"]["inner_val_fraction"]),
        seed,
    )

    tr_pat = set(patient.iloc[tr_i]["patient_id"])
    va_pat = set(patient.iloc[va_i]["patient_id"])

    inner_train = development[development["patient_id"].isin(tr_pat)].copy()
    inner_valid = development[development["patient_id"].isin(va_pat)].copy()

    best_epoch, best_dice, init_info = search_epoch(
        inner_train,
        inner_valid,
        cfg,
        device,
        seed,
        out_dir,
    )

    # Fresh refit on all allowed development patients.
    refit(
        development,
        best_epoch,
        cfg,
        device,
        seed + 10000,
        out_dir,
    )

    atomic_json(
        {
            "status": "success",
            "name": name,
            "development_cases": int(len(development)),
            "development_patients": int(development["patient_id"].nunique()),
            "best_epoch": int(best_epoch),
            "best_inner_valid_dice": float(best_dice),
            "external_initialization": init_info,
        },
        success,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", default="all", help="Only used by strict_crossfit")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = resolve_path(cfg["output_root"], cfg["project_root"])
    manifest = pd.read_csv(
        out / "case_manifest.csv",
        dtype={"patient_id": str, "series_uid": str},
    )
    segmentation_manifest = pd.read_csv(
        out / "segmentation_manifest.csv",
        dtype={"patient_id": str},
    )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)

    strategy = str(cfg["spatial"]["strategy"]).casefold()
    train = manifest[manifest["split"] == "Train"].copy()

    if strategy == "external_checkpoint":
        print("external_checkpoint strategy: no SegResNet training is performed.")
        return

    if strategy == "pilot_single":
        train_one(
            "pilot",
            segmentation_manifest,
            cfg,
            device,
            int(cfg["segresnet"]["seed"]),
            out,
        )
        return

    if strategy != "strict_crossfit":
        raise ValueError(strategy)

    folds = range(1, 6) if args.fold == "all" else [int(args.fold)]
    for fold in folds:
        development = train[train["fold"].astype(int) != fold].copy()
        holdout = train[train["fold"].astype(int) == fold].copy()

        if set(development["patient_id"]) & set(holdout["patient_id"]):
            raise AssertionError("Outer patient leakage")

        train_one(
            f"fold_{fold}",
            development,
            cfg,
            device,
            int(cfg["segresnet"]["seed"]) + fold * 100,
            out,
        )


if __name__ == "__main__":
    main()

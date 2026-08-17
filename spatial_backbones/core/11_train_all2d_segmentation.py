#!/usr/bin/env python3
"""Train each V6 segmentation backbone on the full annotated 2-D population.

Epoch selection uses only a patient-level split of the 2,233-pair inventory.
The final checkpoint is then freshly initialized and refit on all 2,233 pairs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW

from common import assert_signature, atomic_csv, atomic_json, atomic_torch_save, load_config, run_signature, set_seed, sha256_file
from data import all2d_inner_split, estimate_pos_weight, load_all2d_segmentation_manifest, split_hash
from model_interface import build_model, model_parameter_count
from trainer import evaluate, is_better, make_loader, runtime_batch, train_epoch


FAMILIES = ["segresnet", "deeplabv3plus_resnet50_imagenet"]


def _paths(cfg, family):
    output = Path(cfg["output_root"]) / "all2d_segmentation" / family
    report = Path(cfg["report_root"]) / "all2d_segmentation" / family
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    return output, report


def _signature(cfg, family, split):
    pretrained = cfg["pretrained"]["sha256"] if cfg["models"][family]["pretrained"] else "random_initialization"
    return run_signature(cfg, split_hash(split), family, pretrained)


def _search(cfg, family, inner_train, inner_valid, split, device, output, report):
    signature = _signature(cfg, family, split)
    seed = int(cfg["all2d_segmentation"]["seed"])
    batch_size, accumulation = runtime_batch(cfg, family)
    pos_weight = estimate_pos_weight(inner_train, cfg)
    set_seed(seed)
    model = build_model(family, cfg, load_pretrained=True).to(device)
    settings = cfg["models"][family]
    optimizer = AdamW(model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))
    scaler = GradScaler(enabled=bool(cfg["development"]["amp"] and device.type == "cuda"))
    train_loader = make_loader(inner_train, cfg, True, seed, batch_size)
    valid_loader = make_loader(inner_valid, cfg, False, seed + 1, batch_size)
    snapshot = {
        "protocol": "post_strict_final_deployment_encoder_all2233_pairs",
        "strict_segmentation_oof": False,
        "strict_oof_or_valid_reporting_use": False,
        "run_signature": signature,
        "selection_population": "all 2D inventory patient-level inner train/valid",
        "inner_train_rows": int(len(inner_train)), "inner_valid_rows": int(len(inner_valid)),
        "inner_train_patients": int(inner_train.patient_id.nunique()), "inner_valid_patients": int(inner_valid.patient_id.nunique()),
        "physical_batch_size": batch_size, "gradient_accumulation": accumulation,
        "pos_weight": pos_weight, "model_settings": settings, "augmentation": cfg["augmentation"], "loss": cfg["loss"],
        "model_parameters": model_parameter_count(model),
    }
    atomic_json(snapshot, report / "config_snapshot.json")
    atomic_csv(split, report / "all2d_inner_split.csv")
    history, start_epoch, best_metrics, best_epoch, bad_epochs = [], 1, None, 0, 0
    last = output / "search_last.pt"
    if cfg["development"]["resume"] and last.is_file():
        state = torch.load(last, map_location="cpu")
        assert_signature(state["run_signature"], signature)
        model.load_state_dict(state["state_dict"], strict=True); optimizer.load_state_dict(state["optimizer"]); scaler.load_state_dict(state["scaler"])
        history, start_epoch, best_metrics, best_epoch, bad_epochs = list(state["history"]), int(state["epoch"])+1, state["best_metrics"], int(state["best_epoch"]), int(state["bad_epochs"])
    for epoch in range(start_epoch, int(cfg["development"]["max_epochs"])+1):
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, device, pos_weight, cfg, accumulation)
        valid_metrics, _ = evaluate(model, valid_loader, device, cfg)
        record = {"epoch": epoch, **train_metrics, **{f"valid_{key}": value for key, value in valid_metrics.items()}}
        history.append(record); atomic_csv(pd.DataFrame(history), report / "search_history.csv")
        print(json.dumps({"family": family, "stage": "selection", **record}, ensure_ascii=False), flush=True)
        if is_better(valid_metrics, best_metrics):
            best_metrics, best_epoch, bad_epochs = valid_metrics, epoch, 0
            atomic_torch_save({"state_dict": model.state_dict(), "epoch": epoch, "metrics": valid_metrics, "run_signature": signature, "config_snapshot": snapshot}, output / "search_best.pt")
        else: bad_epochs += 1
        atomic_torch_save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch, "best_metrics": best_metrics, "best_epoch": best_epoch, "bad_epochs": bad_epochs, "history": history, "run_signature": signature}, last)
        if epoch >= int(cfg["development"]["min_epochs"]) and bad_epochs >= int(cfg["development"]["patience"]): break
    best = torch.load(output / "search_best.pt", map_location="cpu")
    model.load_state_dict(best["state_dict"], strict=True)
    metrics, rows = evaluate(model, valid_loader, device, cfg)
    rows["gt_size_quartile"] = pd.qcut(rows["gt_pixels"], 4, labels=["Q1-smallest","Q2","Q3","Q4-largest"]).astype(str)
    atomic_csv(rows, report / "best_inner_valid_predictions.csv")
    atomic_json({"best_epoch": int(best_epoch), "metrics": metrics, "run_signature": signature}, report / "search_metrics.json")
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return int(best_epoch), metrics, snapshot, signature


def _refit(cfg, family, all_cases, selected_epochs, snapshot, signature, device, output, report):
    if not cfg["all2d_segmentation"]["refit_after_selection"]: return None
    model_path = output / "model.pt"
    if model_path.is_file():
        existing = torch.load(model_path, map_location="cpu")
        assert_signature(existing["run_signature"], signature)
        return {"checkpoint": str(model_path), "sha256": sha256_file(model_path), "resumed": True}
    seed = int(cfg["all2d_segmentation"]["seed"]) + 10000
    batch_size, accumulation = runtime_batch(cfg, family)
    pos_weight = estimate_pos_weight(all_cases, cfg)
    set_seed(seed)
    model = build_model(family, cfg, load_pretrained=True).to(device)
    settings = cfg["models"][family]
    optimizer = AdamW(model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))
    scaler = GradScaler(enabled=bool(cfg["development"]["amp"] and device.type == "cuda"))
    loader = make_loader(all_cases, cfg, True, seed, batch_size)
    history = []
    for epoch in range(1, selected_epochs+1):
        train_metrics = train_epoch(model, loader, optimizer, scaler, device, pos_weight, cfg, accumulation)
        record = {"epoch": epoch, **train_metrics}; history.append(record)
        atomic_csv(pd.DataFrame(history), report / "refit_history.csv")
        print(json.dumps({"family": family, "stage": "full_all2d_refit", **record}, ensure_ascii=False), flush=True)
    atomic_torch_save({"state_dict": model.state_dict(), "selected_epochs": selected_epochs, "run_signature": signature, "config_snapshot": snapshot, "refit_population_rows": int(len(all_cases)), "refit_population_patients": int(all_cases.patient_id.nunique())}, model_path)
    result = {"checkpoint": str(model_path), "sha256": sha256_file(model_path), "selected_epochs": selected_epochs, "refit_rows": int(len(all_cases)), "refit_patients": int(all_cases.patient_id.nunique()), "resumed": False}
    atomic_json(result, report / "refit_metrics.json")
    return result


def train_one(cfg, family, device):
    manifest = load_all2d_segmentation_manifest(cfg)
    inner_train, inner_valid, split = all2d_inner_split(manifest, cfg)
    output, report = _paths(cfg, family)
    success = output / "SUCCESS.json"
    signature = _signature(cfg, family, split)
    if success.is_file():
        result = json.loads(success.read_text(encoding="utf-8")); assert_signature(result["run_signature"], signature); return result
    epochs, metrics, snapshot, signature = _search(cfg, family, inner_train, inner_valid, split, device, output, report)
    refit = _refit(cfg, family, manifest, epochs, snapshot, signature, device, output, report)
    result = {"status": "success", "family": family, "protocol": "post_strict_final_deployment_encoder_all2233_pairs", "strict_segmentation_oof": False, "strict_oof_or_valid_reporting_use": False, "selection_inner_valid_metrics": metrics, "selected_epoch": epochs, "refit": refit, "run_signature": signature}
    atomic_json(result, report / "metrics.json"); atomic_json(result, success); return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--family", choices=FAMILIES+['all'], default='all'); parser.add_argument("--device", default='cuda:0')
    args = parser.parse_args(); cfg = load_config(args.config); os.environ['TORCH_HOME'] = cfg['torch_home']
    final_report = Path(cfg['report_root']) / 'FINAL_EXPANDED_STRICT_REPORT.json'
    if not final_report.is_file() or json.loads(final_report.read_text(encoding='utf-8')).get('status') != 'success':
        raise SystemExit('deployment encoder is prohibited until the full strict segmentation/OOF/Valid report succeeds')
    if not (Path(cfg['report_root']) / 'all2d_segmentation_preflight/SUCCESS.json').is_file(): raise RuntimeError('Run 10_preflight_all2d_segmentation.py first')
    device = torch.device(args.device); families = FAMILIES if args.family == 'all' else [args.family]
    results = [train_one(cfg, family, device) for family in families]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()

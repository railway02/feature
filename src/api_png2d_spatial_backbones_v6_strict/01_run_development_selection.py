#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from common import atomic_csv, load_config
from data import development_split, load_train_manifest, split_manifest
from trainer import train_development_model


FAMILIES = ["segresnet", "deeplabv3plus_resnet50_imagenet"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--family", choices=FAMILIES + ["all"], default="all")
    parser.add_argument("--outer-fold", choices=["1", "4", "all"], default="all")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    cfg = load_config(args.config)
    preflight = Path(cfg["report_root"]) / "00_preflight/SUCCESS.json"
    if not preflight.is_file():
        raise RuntimeError(f"Preflight SUCCESS is required: {preflight}")
    train = load_train_manifest(cfg)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    families = FAMILIES if args.family == "all" else [args.family]
    folds = [1, 4] if args.outer_fold == "all" else [int(args.outer_fold)]
    results = []
    for outer_fold in folds:
        inner_train, inner_valid, _ = development_split(train, outer_fold, cfg)
        split_frame = split_manifest(train, outer_fold, cfg)
        split_path = Path(cfg["report_root"]) / "development/splits" / f"fold_{outer_fold}.csv"
        atomic_csv(split_frame, split_path)
        for family in families:
            results.append(train_development_model(
                cfg, family, outer_fold, inner_train, inner_valid, split_frame, device
            ))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

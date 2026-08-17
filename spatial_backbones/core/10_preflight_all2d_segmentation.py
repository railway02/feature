#!/usr/bin/env python3
"""Preflight for the user-directed all-2D segmentation protocol.

This protocol deliberately differs from strict segmentation cross-fit: all
2,233 annotated 2-D image/mask pairs are eligible for final segmentation
refit. No adverse-outcome label is read here.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import pandas as pd
import torch

from common import atomic_csv, atomic_json, atomic_text, load_config, sha256_file, tree_hash
from data import all2d_inner_split, load_all2d_segmentation_manifest, prepare_pair
from model_interface import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    cfg = load_config(args.config)
    final_report = Path(cfg["report_root"]) / "FINAL_EXPANDED_STRICT_REPORT.json"
    if not final_report.is_file() or json.loads(final_report.read_text(encoding="utf-8")).get("status") != "success":
        raise SystemExit("deployment encoder is prohibited until the full strict segmentation/OOF/Valid report succeeds")
    os.environ["TORCH_HOME"] = cfg["torch_home"]
    output = Path(cfg["output_root"]) / "all2d_segmentation"
    report = Path(cfg["report_root"]) / "all2d_segmentation_preflight"
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    inventory_path = Path(cfg["sources"]["segmentation_inventory"])
    inventory_hash = sha256_file(inventory_path)
    if inventory_hash != cfg["sources"]["segmentation_inventory_sha256"]:
        raise RuntimeError("Frozen all-2D inventory checksum mismatch")
    manifest = load_all2d_segmentation_manifest(cfg)
    inner_train, inner_valid, split = all2d_inner_split(manifest, cfg)
    if set(inner_train.patient_id) & set(inner_valid.patient_id):
        raise RuntimeError("Patient leakage in all-2D inner split")
    atomic_csv(manifest, output / "all2d_segmentation_manifest.csv")
    atomic_csv(split, report / "all2d_patient_inner_split.csv")

    rows = []
    for item in manifest.itertuples(index=False):
        image = cv2.imread(item.image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(item.mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"Unreadable all-2D pair {item.segmentation_key}")
        if image.shape != mask.shape or not (mask > 0).any():
            raise RuntimeError(f"Invalid all-2D pair {item.segmentation_key}")
        _, prepared_mask, letterbox = prepare_pair(item.image_path, item.mask_path, cfg)
        rows.append({
            "segmentation_key": item.segmentation_key,
            "patient_id": item.patient_id,
            "phase": item.phase,
            "image_height": int(image.shape[0]),
            "image_width": int(image.shape[1]),
            "raw_foreground": int((mask > 0).sum()),
            "prepared_foreground": int(prepared_mask.sum()),
            "padding": json.dumps(letterbox["padding"]),
        })
    audit = pd.DataFrame(rows)
    atomic_csv(audit, report / "all2d_image_mask_audit.csv")

    unit = subprocess.run([sys.executable, "-m", "unittest", "-v", "test_v6.py"], cwd=cfg["code_root"], check=True, text=True, capture_output=True)
    atomic_text(unit.stdout + unit.stderr, report / "unit_tests.txt")
    subprocess.run([sys.executable, "-m", "compileall", "-q", cfg["code_root"]], check=True)

    # Confirm both code paths still accept actual all-2D data at 768x768.
    sample = manifest.iloc[0]
    image, _, _ = prepare_pair(sample.image_path, sample.mask_path, cfg)
    x = torch.from_numpy(image[None, None]).float().to(device)
    shapes = {}
    for family in ("segresnet", "deeplabv3plus_resnet50_imagenet"):
        model = build_model(family, cfg, load_pretrained=True).to(device).eval()
        with torch.no_grad():
            fmap, logits = model.encode_and_decode(x)
        shapes[family] = {"fmap": list(fmap.shape), "logits": list(logits.shape)}
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    if shapes["segresnet"]["fmap"][1:] != [256, 96, 96]:
        raise RuntimeError(f"Unexpected SegResNet shape {shapes['segresnet']}")
    if shapes["deeplabv3plus_resnet50_imagenet"]["fmap"][1] != 256:
        raise RuntimeError(f"Unexpected DeepLab shape {shapes['deeplabv3plus_resnet50_imagenet']}")

    protected = {path: tree_hash(path) if Path(path).is_dir() else sha256_file(path) for path in cfg["protected_roots"]}
    result = {
        "status": "PASS",
        "protocol": "post_strict_final_deployment_encoder_all2233_pairs",
        "strict_segmentation_oof": False,
        "strict_oof_or_valid_reporting_use": False,
        "outcome_labels_read": False,
        "all2d_inventory": {"path": str(inventory_path), "sha256": inventory_hash, "rows": int(len(manifest)), "patients": int(manifest.patient_id.nunique())},
        "inner_split": {"train_rows": int(len(inner_train)), "valid_rows": int(len(inner_valid)), "train_patients": int(inner_train.patient_id.nunique()), "valid_patients": int(inner_valid.patient_id.nunique())},
        "model_shapes": shapes,
        "protected_v5_assets_current_hash": protected,
        "config_sha256": cfg["_config_sha256"],
    }
    atomic_json(result, report / "PRELIGHT_ALL2D.json")
    atomic_json(result, report / "SUCCESS.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

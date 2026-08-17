#!/usr/bin/env python3
"""Freeze and audit the expanded strict segmentation protocol before training."""
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
from data import build_expanded_strict_population, expanded_strict_fold_split, prepare_pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    spec = cfg["expanded_strict_segmentation"]
    out = Path(cfg["output_root"]) / "expanded_strict"
    report = Path(cfg["report_root"]) / "expanded_strict_preflight"
    out.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    locks = {}
    for key in ["segmentation_train_metadata", "segmentation_valid_metadata", "phase_eligible_manifest"]:
        actual = sha256_file(spec[key]); expected = spec[f"{key}_sha256"]
        locks[key] = {"path": spec[key], "expected_sha256": expected, "actual_sha256": actual, "match": actual == expected}
        if actual != expected: raise RuntimeError(f"Checksum mismatch: {key}")
    train_pool, valid_pool, adverse, master_audit = build_expanded_strict_population(cfg)
    atomic_csv(train_pool, out / "segmentation_train_master_manifest.csv")
    atomic_csv(valid_pool, out / "segmentation_valid_forbidden_manifest.csv")
    fold_audits = []
    extra_rows = []
    for fold in range(1, 6):
        legal, inner_train, inner_valid, forbidden_valid, audit = expanded_strict_fold_split(cfg, fold)
        holdout_patients = set(adverse.loc[adverse.fold.eq(fold), "patient_id"].astype(str))
        if set(legal.patient_id) & holdout_patients: raise RuntimeError(f"fold {fold}: outer holdout overlap")
        if set(legal.patient_id) & set(valid_pool.patient_id): raise RuntimeError(f"fold {fold}: valid.xlsx patient overlap")
        if set(legal.png_key) & set(valid_pool.png_key): raise RuntimeError(f"fold {fold}: valid.xlsx png overlap")
        atomic_csv(legal, out / f"fold_{fold}" / "segmentation_legal_split_manifest.csv")
        audit["outer_holdout_patient_overlap"] = 0
        audit["valid_patient_overlap"] = 0
        audit["valid_png_overlap"] = 0
        fold_audits.append(audit)
        extras = legal.loc[legal.png_key.isin(spec["adverse_extra_png_keys"]), ["png_key", "patient_id", "fold", "development_partition"]].copy()
        extras["outer_fold_context"] = fold
        extra_rows.append(extras)
    atomic_csv(pd.DataFrame(fold_audits), report / "five_fold_legality_audit.csv")
    atomic_csv(pd.concat(extra_rows, ignore_index=True), report / "adverse_extra_series_inheritance_audit.csv")
    # Read every image/mask pair exactly once; this validates the full 1780+453 pools.
    file_rows = []
    for population, frame in (("Train", train_pool), ("Valid_forbidden", valid_pool)):
        for item in frame.itertuples(index=False):
            image = cv2.imread(item.image_path, cv2.IMREAD_GRAYSCALE); mask = cv2.imread(item.mask_path, cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None or image.shape != mask.shape or not (mask > 0).any():
                raise RuntimeError(f"Invalid image/mask pair: {item.png_key}")
            _, prepared, letterbox = prepare_pair(item.image_path, item.mask_path, cfg)
            file_rows.append({"population": population, "png_key": item.png_key, "patient_id": item.patient_id, "native_shape": f"{image.shape[0]}x{image.shape[1]}", "raw_fg": int((mask>0).sum()), "prepared_fg": int(prepared.sum()), "padding": json.dumps(letterbox["padding"])})
    atomic_csv(pd.DataFrame(file_rows), report / "all_2233_image_mask_audit.csv")
    unit = subprocess.run([sys.executable, "-m", "unittest", "-v", "test_v6.py"], cwd=cfg["code_root"], check=True, text=True, capture_output=True)
    atomic_text(unit.stdout + unit.stderr, report / "unit_tests.txt")
    subprocess.run([sys.executable, "-m", "compileall", "-q", cfg["code_root"]], check=True)
    protocol = {
        "status": "frozen_before_full_strict",
        "protocol": spec["protocol"],
        "backbones": ["segresnet", "deeplabv3plus_resnet50_imagenet"],
        "confirmatory_backbones": True,
        "development_benchmarks_skipped": True,
        "promotion_decision_not_used": True,
        "config_path": cfg["_config_path"], "config_sha256": cfg["_config_sha256"],
        "threshold": cfg["development"]["threshold"], "loss": cfg["loss"], "augmentation": cfg["augmentation"], "models": cfg["models"], "pretrained": cfg["pretrained"],
        "inner_selection": {"patient_level": True, "adverse_development_only": True, "segmentation_only_patients": "inner_train_only", "max_epochs": cfg["development"]["max_epochs"], "min_epochs": cfg["development"]["min_epochs"], "patience": cfg["development"]["patience"]},
        "data_boundary": {"train_rows": 1780, "valid_rows_forbidden": 453, "valid_training_or_selection_allowed": False, "outer_holdout_training_or_selection_allowed": False},
        "legal_fold_audit": fold_audits,
        "input_locks": locks,
        "protected_v5_hashes": {path: tree_hash(path) if Path(path).is_dir() else sha256_file(path) for path in cfg["protected_roots"]},
    }
    atomic_json(protocol, Path(cfg["report_root"]) / "FROZEN_TWO_BACKBONE_PROTOCOL.json")
    atomic_json({"status": "superseded_not_for_strict", "reason": "Expanded strict protocol uses fold-specific Train.xlsx patient pools; all2d single encoder is deployment-only after strict evaluation."}, report / "ALL2D_SINGLE_ENCODER_SUPERSEDED.json")
    result = {"status": "PASS", "master": master_audit, "folds": fold_audits, "protocol": str(Path(cfg["report_root"]) / "FROZEN_TWO_BACKBONE_PROTOCOL.json")}
    atomic_json(result, report / "PRELIGHT_EXPANDED_STRICT.json")
    atomic_json(result, report / "SUCCESS.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()

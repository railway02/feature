#!/usr/bin/env python3
"""Build a machine-readable delivery manifest for the completed interface."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from loader import sha256, write_json


WORKSPACE = Path("/root/autodl-tmp")
ROOT = WORKSPACE / "aneurysm"
DEFAULT_OUT = ROOT / "outputs/api_png2d_segresnet_cave_main_fusion_v6_strict"
DEFAULT_CODE = ROOT / "code/api_png2d_segresnet_cave_main_fusion_v6_strict"
DEFAULT_HANDOFF = WORKSPACE / "DSA_2D_CAVE_MAIN_FUSION_INTERFACE_HANDOFF_FINAL.md"


def npz_schema(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        forbidden = {"target", "label", "outcome", "y", "adverse_outcome"} & set(data.files)
        if forbidden:
            raise AssertionError(f"{path}: public output contains label fields: {sorted(forbidden)}")
        return {
            name: {"shape": list(data[name].shape), "dtype": str(data[name].dtype)}
            for name in data.files
        }


def file_record(path: Path, role: str, usage: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "role": role,
        "usage": usage,
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--code-root", type=Path, default=DEFAULT_CODE)
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    args = parser.parse_args()

    train_oof = args.output_root / "train_oof_main_outputs.npz"
    train_by_fold = args.output_root / "train_main_outputs_by_fold.npz"
    valid_by_fold = args.output_root / "valid_main_outputs_by_fold.npz"
    interface_2d = ROOT / "outputs/api_png2d_spatial_backbones_v6_strict/expanded_strict/2d_spatial_interface/dsa_2d_spatial_v1_segresnet_strict_soft_predroi"
    code_files = [
        "model.py", "loader.py", "package_raw_modalities.py", "train_main_fusion.py",
        "export_train_by_fold.py", "downstream_loader.py", "audit_outputs.py",
        "build_delivery_manifest.py", "MAIN_FUSION_INTERFACE_SPEC.md",
    ]
    checkpoints = [args.output_root / "main_fusion" / f"fold_{fold}" / "model.pt" for fold in range(1, 6)]

    manifest = {
        "status": "READY_FOR_HANDOFF",
        "interface_version": "dsa_2d_cave_main_fusion_v6_strict",
        "completed_scope": "teacher scheme IV.2: 2D--temporal main-modality fusion",
        "not_completed_scope": ["3D residual correction", "TabPFN residual correction", "auxiliary reliability gates", "final four-modality probability"],
        "cohort": {"train_series": 781, "valid_series": 207, "valid_patients": 206},
        "formal_inputs": {
            "z_2d_raw": {"shape": ["B", 1024], "semantics": ["G_pre", "soft_PredROI_pre", "G_post", "soft_PredROI_post"]},
            "z_time_raw": {"shape": ["B", 10240], "semantics": ["CAVE_pre_5120", "CAVE_post_5120"]},
        },
        "formal_outputs": {"z_main": ["B", 256], "main_logit": ["B", 1], "main_prob": ["B", 1]},
        "strict_rules": {
            "train_oof_for_main_path_evaluation": True,
            "downstream_outer_fold_k_uses_train_and_valid_column_k_minus_1": True,
            "latent_averaging_allowed": False,
            "valid_used_for_selection": False,
            "public_representation_files_contain_outcome_label": False,
            "downstream_inner_search_requires_separate_nested_design": True,
        },
        "primary_data_files": [
            {**file_record(train_by_fold, "fold-specific Train main representation", "primary neural downstream input; fold k reads [:,k-1,:]"), "schema": npz_schema(train_by_fold)},
            {**file_record(valid_by_fold, "fold-specific Valid main representation", "fold k reads [:,k-1,:]; do not average latent vectors"), "schema": npz_schema(valid_by_fold)},
            {**file_record(train_oof, "strict Train OOF main output", "main-path evaluation/audit/calibration only; not a globally aligned neural latent bank"), "schema": npz_schema(train_oof)},
        ],
        "upstream_public_interfaces": [
            {**file_record(interface_2d / "train_oof_z_2d_raw.npz", "strict Train OOF 2D input", "formal z_2d_raw evaluation/interface view"), "schema": npz_schema(interface_2d / "train_oof_z_2d_raw.npz")},
            {**file_record(interface_2d / "valid_z_2d_raw_by_fold.npz", "fold-specific Valid 2D input", "fusion fold k reads [:,k-1,:]"), "schema": npz_schema(interface_2d / "valid_z_2d_raw_by_fold.npz")},
            {**file_record(args.output_root / "raw_modalities/cave_train_z_time_raw.npz", "Train CAVE raw input", "label-free z_time_raw [781,10240]"), "schema": npz_schema(args.output_root / "raw_modalities/cave_train_z_time_raw.npz")},
            {**file_record(args.output_root / "raw_modalities/cave_valid_z_time_raw.npz", "Valid CAVE raw input", "label-free z_time_raw [207,10240]"), "schema": npz_schema(args.output_root / "raw_modalities/cave_valid_z_time_raw.npz")},
        ],
        "frozen_checkpoints": [file_record(path, f"main-fusion fold {fold} checkpoint", "frozen inference; do not alter architecture") for fold, path in enumerate(checkpoints, start=1)],
        "audits": [
            file_record(args.output_root / "alignment/alignment_audit.json", "upstream alignment audit", "UID/patient/target/fold and split checks"),
            file_record(args.output_root / "main_fusion/SMOKE_TEST.json", "model smoke test", "shape/dtype/finite/range/reload checks"),
            file_record(args.output_root / "main_fusion/TRAIN_BY_FOLD_EXPORT_AUDIT.json", "frozen Train by-fold export audit", "checkpoint hashes and OOF reconstruction"),
            file_record(args.output_root / "main_fusion/DOWNSTREAM_LOADER_SMOKE_TEST.json", "downstream loader smoke test", "fold views, identifiers and holdout/OOF match"),
            file_record(args.output_root / "main_fusion/metrics.json", "frozen model metrics", "strict OOF and independent Valid results"),
        ],
        "code": [file_record(args.code_root / name, "implementation", "reproduction and fail-closed loading") for name in code_files],
        "human_handoff": file_record(args.handoff, "authoritative Chinese handoff", "read before downstream integration"),
        "integrity_index": str(args.output_root / "SHA256SUMS.txt"),
    }
    write_json(args.output_root / "DELIVERY_MANIFEST.json", manifest)
    print(json.dumps({"status": manifest["status"], "output": str(args.output_root / "DELIVERY_MANIFEST.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

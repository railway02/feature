#!/usr/bin/env python3
"""Frozen export of all five main-fusion coordinate views for Train.

This script never updates model parameters.  Each frozen fusion fold is
applied to all 781 aligned Train rows using the matching SegResNet fold view.
The result is intended for fold-specific downstream 3D/TabPFN modelling; the
strict OOF file remains the authoritative file for main-path evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from loader import load_and_audit, sha256, write_json
from model import MainFusionModel


ROOT = Path("/root/autodl-tmp/aneurysm")
DEFAULT_SP = ROOT / "outputs/api_png2d_spatial_backbones_v6_strict/expanded_strict/featurebanks/segresnet"
DEFAULT_CAVE = ROOT / "outputs/api_png2d_gtmask_roi_cave_v2_complete_mapping_fullseq/adverse_prepost_series_task_v3"
DEFAULT_OUT = ROOT / "outputs/api_png2d_segresnet_cave_main_fusion_v6_strict"
OUTPUT_FIELDS = ("z_main", "main_logit", "main_prob", "spatial_gate", "temporal_gate")
OOF_COMPARE_RTOL = 1e-5
OOF_COMPARE_ATOL = 1e-5


@torch.inference_mode()
def infer_all(
    model: MainFusionModel,
    spatial: np.ndarray,
    temporal: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    dataset = TensorDataset(torch.from_numpy(spatial), torch.from_numpy(temporal))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    collected = {name: [] for name in OUTPUT_FIELDS}
    model.eval()
    for spatial_batch, temporal_batch in loader:
        outputs = model(spatial_batch.to(device), temporal_batch.to(device))
        for name in OUTPUT_FIELDS:
            collected[name].append(outputs[name].detach().cpu().numpy().astype(np.float32))
    return {name: np.concatenate(parts, axis=0) for name, parts in collected.items()}


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial-root", type=Path, default=DEFAULT_SP)
    parser.add_argument("--cave-root", type=Path, default=DEFAULT_CAVE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    spatial_train = args.spatial_root / "train_spatial_features.npz"
    spatial_valid = args.spatial_root / "valid_spatial_features.npz"
    cave_train = args.cave_root / "train_features.npz"
    cave_valid = args.cave_root / "valid_features.npz"
    train, _, alignment, _, _ = load_and_audit(
        spatial_train, spatial_valid, cave_train, cave_valid
    )
    if alignment["status"] != "PASS":
        raise AssertionError("upstream alignment audit did not pass")

    by_fold: dict[str, list[np.ndarray]] = {name: [] for name in OUTPUT_FIELDS}
    checkpoint_hashes: dict[str, str] = {}
    selected_epochs: list[int] = []
    for fold in range(1, 6):
        checkpoint_path = args.output_root / "main_fusion" / f"fold_{fold}" / "model.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if checkpoint.get("protocol") != "fresh_refit_after_inner_patient_level_AUPRC_selection":
            raise AssertionError(f"fold {fold}: unexpected checkpoint protocol")
        config = checkpoint["model_config"]
        if config.get("spatial_dim") != 1024 or config.get("temporal_dim") != 10240 or config.get("hidden_dim") != 256:
            raise AssertionError(f"fold {fold}: unexpected model dimensions: {config}")
        model = MainFusionModel(**config).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        outputs = infer_all(
            model=model,
            spatial=train["spatial_by_fold"][:, fold - 1, :].astype(np.float32, copy=False),
            temporal=train["temporal"].astype(np.float32, copy=False),
            batch_size=args.batch_size,
            device=device,
        )
        for name in OUTPUT_FIELDS:
            by_fold[name].append(outputs[name])
        checkpoint_hashes[str(fold)] = sha256(checkpoint_path)
        selected_epochs.append(int(checkpoint["selected_epoch"]))

    stacked = {f"{name}_by_fold": np.stack(parts, axis=1) for name, parts in by_fold.items()}
    expected_shapes = {
        "z_main_by_fold": (781, 5, 256),
        "main_logit_by_fold": (781, 5, 1),
        "main_prob_by_fold": (781, 5, 1),
        "spatial_gate_by_fold": (781, 5, 256),
        "temporal_gate_by_fold": (781, 5, 256),
    }
    for name, shape in expected_shapes.items():
        value = stacked[name]
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise AssertionError(f"{name}: invalid shape/dtype/finite state")
    for name in ("main_prob_by_fold", "spatial_gate_by_fold", "temporal_gate_by_fold"):
        if not np.all((stacked[name] >= 0.0) & (stacked[name] <= 1.0)):
            raise AssertionError(f"{name}: values outside [0,1]")

    existing_oof_path = args.output_root / "train_oof_main_outputs.npz"
    existing_oof = read_npz(existing_oof_path)
    if not np.array_equal(existing_oof["series_uid"].astype(str), train["series_uid"].astype(str)):
        raise AssertionError("series_uid differs from existing OOF file")
    if not np.array_equal(existing_oof["patient_id"].astype(str), train["patient_id"].astype(str)):
        raise AssertionError("patient_id differs from existing OOF file")
    if not np.array_equal(existing_oof["outer_fold"], train["fold"]):
        raise AssertionError("outer_fold differs from existing OOF file")

    row_index = np.arange(len(train["fold"]))
    fold_index = train["fold"] - 1
    oof_max_abs_difference: dict[str, float] = {}
    for name in OUTPUT_FIELDS:
        selected = stacked[f"{name}_by_fold"][row_index, fold_index]
        reference = existing_oof[name]
        maximum = float(np.max(np.abs(selected - reference)))
        oof_max_abs_difference[name] = maximum
        if not np.allclose(selected, reference, rtol=OOF_COMPARE_RTOL, atol=OOF_COMPARE_ATOL):
            raise AssertionError(f"{name}: by-fold OOF selection differs from frozen OOF; max_abs={maximum}")

    output_path = args.output_root / "train_main_outputs_by_fold.npz"
    np.savez_compressed(
        output_path,
        series_uid=train["series_uid"],
        patient_id=train["patient_id"],
        split=np.repeat("Train", len(train["fold"])),
        outer_fold=train["fold"].astype(np.int64),
        source_fusion_folds=np.arange(1, 6, dtype=np.int64),
        selected_epochs=np.asarray(selected_epochs, dtype=np.int64),
        model_family=np.asarray("segresnet_cave_main_fusion"),
        feature_version=np.asarray("dsa_2d_cave_main_fusion_v6_strict_by_fold"),
        **stacked,
    )

    # Re-open the serialized artifact before declaring success.
    serialized = read_npz(output_path)
    for name, shape in expected_shapes.items():
        if serialized[name].shape != shape or serialized[name].dtype != np.float32 or not np.isfinite(serialized[name]).all():
            raise AssertionError(f"serialized {name}: invalid shape/dtype/finite state")
    if not np.array_equal(serialized["source_fusion_folds"], np.arange(1, 6)):
        raise AssertionError("serialized source_fusion_folds mismatch")

    audit = {
        "status": "PASS",
        "operation": "frozen inference only; no model parameters updated",
        "rows": 781,
        "source_fusion_folds": [1, 2, 3, 4, 5],
        "selected_epochs": selected_epochs,
        "shapes": {name: list(shape) for name, shape in expected_shapes.items()},
        "dtype": "float32",
        "all_finite": True,
        "probability_and_gate_range": True,
        "identifier_order_matches_existing_oof": True,
        "outer_fold_matches_existing_oof": True,
        "oof_selection_matches_existing_strict_oof": True,
        "oof_comparison_tolerance": {"rtol": OOF_COMPARE_RTOL, "atol": OOF_COMPARE_ATOL},
        "oof_selection_max_abs_difference": oof_max_abs_difference,
        "latent_averaging_applied": False,
        "formal_downstream_usage": "downstream outer fold k reads [:, k-1, :] for both development and holdout",
        "not_for_strict_main_path_evaluation": True,
        "checkpoint_sha256": checkpoint_hashes,
        "input_sha256": {
            "spatial_train": sha256(spatial_train),
            "spatial_valid": sha256(spatial_valid),
            "cave_train": sha256(cave_train),
            "cave_valid": sha256(cave_valid),
            "existing_train_oof": sha256(existing_oof_path),
        },
        "output": str(output_path),
        "output_sha256": sha256(output_path),
    }
    audit_path = args.output_root / "main_fusion" / "TRAIN_BY_FOLD_EXPORT_AUDIT.json"
    write_json(audit_path, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

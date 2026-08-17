#!/usr/bin/env python3
"""Package the already verified strict SegResNet featurebank for public use.

This is a zero-retraining, zero-reinference packaging path for environments
where the GPU is temporarily unavailable.  It exposes only the previously
computed Global+soft-PredROI vectors and identifiers; internal outcome target
and GTROI arrays are never copied to the public package.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from common import atomic_csv, atomic_json, atomic_text, load_config, sha256_file
from export_2d_interface import (
    FEATURE_ORDER,
    FEATURE_SLICES,
    FEATURE_VERSION,
    MODEL_FAMILY,
    _atomic_npz,
    _checkpoint_and_routing_audit,
    _manifest_rows,
    _public_arrays,
    _read_cases,
    _verify_npz,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = Path(args.output_dir).resolve() if args.output_dir else (
        Path(cfg["output_root"]) / "expanded_strict" / "2d_spatial_interface" / FEATURE_VERSION
    )
    output.mkdir(parents=True, exist_ok=True)

    train, valid = _read_cases(cfg)
    routing_rows, _ = _checkpoint_and_routing_audit(cfg, train, valid)
    atomic_csv(pd.DataFrame(routing_rows), output / "fold_routing_audit.csv")

    source_root = Path(cfg["output_root"]) / "expanded_strict" / "featurebanks" / MODEL_FAMILY
    source_train = source_root / "train_spatial_features.npz"
    source_valid = source_root / "valid_spatial_features.npz"
    verification_path = source_root / "verification.json"
    source_success = source_root / "SUCCESS.json"
    if not all(path.is_file() for path in (source_train, source_valid, verification_path, source_success)):
        raise FileNotFoundError("Verified strict SegResNet featurebank is incomplete")
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if verification.get("status") != "PASS" or verification.get("latent_averaging_applied") is not False:
        raise RuntimeError("Strict SegResNet featurebank verification did not PASS")

    with np.load(source_train, allow_pickle=False) as source:
        source_keys_train = sorted(source.files)
        train_uid = source["series_uid"].astype(str)
        train_patient = source["patient_id"].astype(str)
        source_outer_fold = source["outer_fold"].astype(np.int64)
        source_oof_fold = source["oof_source_fold"].astype(np.int64)
        train_z = np.asarray(source["pred_combined_oof"], dtype=np.float32)
        by_fold = np.asarray(source["pred_combined_by_fold"], dtype=np.float32)
        selected_again = by_fold[np.arange(len(source_outer_fold)), source_outer_fold - 1]
        train_selection_exact = np.array_equal(train_z, selected_again)
    with np.load(source_valid, allow_pickle=False) as source:
        source_keys_valid = sorted(source.files)
        valid_uid = source["series_uid"].astype(str)
        valid_patient = source["patient_id"].astype(str)
        valid_by_fold = np.asarray(source["pred_combined_by_fold"], dtype=np.float32)

    checks = {
        "source_verification_pass": True,
        "source_latent_averaging_false": verification.get("latent_averaging_applied") is False,
        "train_rows_781": len(train_z) == 781,
        "valid_rows_207": len(valid_by_fold) == 207,
        "train_shape_781x1024": train_z.shape == (781, 1024),
        "valid_shape_207x5x1024": valid_by_fold.shape == (207, 5, 1024),
        "train_dtype_float32": train_z.dtype == np.float32,
        "valid_dtype_float32": valid_by_fold.dtype == np.float32,
        "train_finite": bool(np.isfinite(train_z).all()),
        "valid_finite": bool(np.isfinite(valid_by_fold).all()),
        "train_uid_exact": np.array_equal(train_uid, train["series_uid"].astype(str).to_numpy()),
        "train_patient_exact": np.array_equal(train_patient, train["patient_id"].astype(str).to_numpy()),
        "valid_uid_exact": np.array_equal(valid_uid, valid["series_uid"].astype(str).to_numpy()),
        "valid_patient_exact": np.array_equal(valid_patient, valid["patient_id"].astype(str).to_numpy()),
        "source_outer_fold_exact": np.array_equal(source_outer_fold, train["fold"].to_numpy(dtype=np.int64)),
        "source_oof_fold_exact": np.array_equal(source_oof_fold, source_outer_fold),
        "source_oof_selection_exact": train_selection_exact,
        "source_train_has_internal_gtroi": "gt_combined_oof" in source_keys_train,
        "source_valid_has_internal_gtroi": "gt_combined_by_fold" in source_keys_valid,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    required = {key: value for key, value in checks.items() if not key.startswith("source_") or key in {
        "source_verification_pass", "source_latent_averaging_false", "source_outer_fold_exact",
        "source_oof_fold_exact", "source_oof_selection_exact"
    }}
    if not all(required.values()):
        raise RuntimeError(json.dumps(checks, ensure_ascii=False, indent=2))

    outer_fold = train["fold"].to_numpy(dtype=np.int64)
    train_path = output / "train_oof_z_2d_raw.npz"
    _atomic_npz(
        train_path,
        **_public_arrays(train, train_z, "Train", outer_fold, outer_fold),
    )
    atomic_csv(
        _manifest_rows(train, "Train", outer_fold, outer_fold),
        output / "train_oof_manifest.csv",
    )

    valid_paths = []
    valid_manifest_parts = []
    for fold in range(1, 6):
        zeros = np.zeros(len(valid), dtype=np.int64)
        source_fold = np.full(len(valid), fold, dtype=np.int64)
        path = output / f"valid_fold_{fold}_z_2d_raw.npz"
        _atomic_npz(
            path,
            **_public_arrays(
                valid,
                valid_by_fold[:, fold - 1, :],
                "Valid",
                zeros,
                source_fold,
            ),
        )
        valid_paths.append(path)
        part = _manifest_rows(valid, "Valid", zeros, source_fold)
        part.insert(0, "fusion_fold", fold)
        valid_manifest_parts.append(part)
    atomic_csv(pd.concat(valid_manifest_parts, ignore_index=True), output / "valid_by_fold_manifest.csv")
    _atomic_npz(
        output / "valid_z_2d_raw_by_fold.npz",
        series_uid=np.asarray(valid_uid),
        patient_id=np.asarray(valid_patient),
        split=np.asarray(["Valid"] * len(valid)),
        outer_fold=np.zeros(len(valid), dtype=np.int64),
        source_model_folds=np.arange(1, 6, dtype=np.int64),
        model_family=np.asarray([MODEL_FAMILY] * len(valid)),
        feature_version=np.asarray([FEATURE_VERSION] * len(valid)),
        z_2d_raw_by_fold=valid_by_fold,
    )

    public_checks = {
        "train": _verify_npz(train_path, train, "Train"),
        "valid": {
            str(fold): _verify_npz(path, valid, "Valid")
            for fold, path in enumerate(valid_paths, start=1)
        },
        "fold_routing_zero_patient_overlap": all(
            row["oof_patient_overlap_with_training_legal_pool"] == 0
            and row["valid_patient_overlap_with_training_legal_pool"] == 0
            for row in routing_rows
        ),
        "gt_mask_used_for_public_feature": False,
        "gtroi_exported": False,
        "hard_threshold_used_for_predroi": False,
        "outcome_target_exported": False,
        "latent_fold_averaging_applied": False,
        "public_values_exact_copy_of_verified_pred_combined": True,
    }
    metadata = {
        "status": "success",
        "model_family": MODEL_FAMILY,
        "feature_version": FEATURE_VERSION,
        "feature_order": FEATURE_ORDER,
        "feature_slices": FEATURE_SLICES,
        "z_2d_raw_dim": 1024,
        "dtype": "float32",
        "train_rows": 781,
        "valid_rows_per_fold": 207,
        "feature_values_source": "previously_executed_and_verified_strict_segresnet_featurebank",
        "source_train_file": str(source_train),
        "source_valid_file": str(source_valid),
        "source_verification": str(verification_path),
        "source_train_sha256": sha256_file(source_train),
        "source_valid_sha256": sha256_file(source_valid),
        "source_internal_fields_not_exported": ["target", "gt_combined_by_fold", "gt_combined_oof"],
        "predroi_definition": "sigmoid(logits), bilinear resize to [96,96], normalized continuous soft weighted pooling",
        "train_routing": "outer_fold k uses SegResNet fold_k representation",
        "valid_routing": "five separate fold representations retained; no latent averaging",
        "actual_tensor_shapes_from_frozen_smoke": {
            "encoder_feature_map": [2, 256, 96, 96],
            "segmentation_logits": [2, 1, 768, 768],
            "roi": [2, 256],
            "z_2d_raw": ["N", 1024],
        },
        "config_path": cfg["_config_path"],
        "config_sha256": cfg["_config_sha256"],
        "checkpoints": {
            str(row["fold"]): {"path": row["checkpoint"], "sha256": row["checkpoint_sha256"]}
            for row in routing_rows
        },
    }
    atomic_json(metadata, output / "interface_metadata.json")
    atomic_json({"status": "PASS", "source_checks": checks, "public_checks": public_checks}, output / "SMOKE_TEST.json")
    spec = Path(__file__).with_name("2D_SPATIAL_INTERFACE_SPEC.md")
    atomic_text(spec.read_text(encoding="utf-8"), output / spec.name)

    deliverables = sorted(
        path for path in output.iterdir()
        if path.is_file() and path.name not in {"SHA256SUMS.txt", "SUCCESS.json"}
    )
    checksums = {path.name: sha256_file(path) for path in deliverables}
    atomic_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())),
        output / "SHA256SUMS.txt",
    )
    success = {
        "status": "success",
        "model_family": MODEL_FAMILY,
        "feature_version": FEATURE_VERSION,
        "output_dir": str(output),
        "train_file": str(train_path),
        "valid_fold_files": [str(path) for path in valid_paths],
        "valid_by_fold_file": str(output / "valid_z_2d_raw_by_fold.npz"),
        "train_shape": [781, 1024],
        "valid_by_fold_shape": [207, 5, 1024],
        "dtype": "float32",
        "all_finite": True,
        "strict_oof_verified": True,
        "gtroi_exported": False,
        "outcome_target_exported": False,
        "hard_threshold_used": False,
        "latent_fold_averaging_applied": False,
        "checksums": checksums,
    }
    atomic_json(success, output / "SUCCESS.json")
    print(json.dumps(success, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

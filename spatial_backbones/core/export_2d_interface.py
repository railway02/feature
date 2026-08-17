#!/usr/bin/env python3
"""Export the frozen strict SegResNet as a clean patient-level 2-D interface.

The public feature path deliberately does not load outcome targets, GT masks,
GTROI features, or hard segmentation masks.  Train rows are extracted only by
their matching outer-fold checkpoint; Valid rows are retained separately for
all five frozen checkpoints.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset

from common import atomic_csv, atomic_json, atomic_text, load_config, sha256_file
from data import percentile_normalize
from model_interface import build_model, global_pool, roi_pool


MODEL_FAMILY = "segresnet"
FEATURE_VERSION = "dsa_2d_spatial_v1_segresnet_strict_soft_predroi"
FEATURE_ORDER = ["G_pre", "PredROI_pre", "G_post", "PredROI_post"]
FEATURE_SLICES = {
    "G_pre": [0, 256],
    "PredROI_pre": [256, 512],
    "G_post": [512, 768],
    "PredROI_post": [768, 1024],
}
PUBLIC_KEYS = {
    "series_uid",
    "patient_id",
    "split",
    "outer_fold",
    "source_model_fold",
    "model_family",
    "feature_version",
    "z_2d_raw",
}


def _unicode(values) -> np.ndarray:
    strings = [str(value) for value in values]
    width = max(1, max((len(value) for value in strings), default=1))
    return np.asarray(strings, dtype=f"<U{width}")


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _prepare_image(path: str, cfg: dict[str, Any]) -> np.ndarray:
    """Image-only equivalent of the frozen segmentation preprocessing."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    image = percentile_normalize(
        image,
        float(cfg["data"]["percentile_low"]),
        float(cfg["data"]["percentile_high"]),
    )
    target = int(cfg["data"]["input_size"])
    height, width = image.shape
    scale = min(target / height, target / width)
    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    top = (target - new_h) // 2
    bottom = target - new_h - top
    left = (target - new_w) // 2
    right = target - new_w - left
    output = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=float(np.median(resized)),
    )
    if output.shape != (target, target):
        raise RuntimeError(f"Unexpected prepared image shape {output.shape}: {path}")
    return output.astype(np.float32)


class PairedMeanPngDataset(Dataset):
    def __init__(self, cases: pd.DataFrame, cfg: dict[str, Any]):
        self.cases = cases.reset_index(drop=True)
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int):
        row = self.cases.iloc[int(index)]
        pre = _prepare_image(str(row.pre_image), self.cfg)
        post = _prepare_image(str(row.post_image), self.cfg)
        return (
            torch.from_numpy(pre[None]).float(),
            torch.from_numpy(post[None]).float(),
            int(index),
        )


def _read_cases(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Deliberately exclude target, pre_mask and post_mask from the read itself.
    columns = [
        "split",
        "task_row",
        "series_uid",
        "patient_id",
        "fold",
        "pre_image",
        "post_image",
    ]
    frame = pd.read_csv(
        cfg["sources"]["v5_case_manifest"],
        usecols=columns,
        dtype={"series_uid": str, "patient_id": str},
    )
    if frame["series_uid"].duplicated().any():
        raise RuntimeError("case_manifest contains duplicate series_uid")
    train = frame.loc[frame["split"].eq("Train")].copy().reset_index(drop=True)
    valid = frame.loc[frame["split"].eq("Valid")].copy().reset_index(drop=True)
    train["fold"] = pd.to_numeric(train["fold"], errors="raise").astype(int)
    if len(train) != 781 or train["series_uid"].nunique() != 781:
        raise RuntimeError(f"Unexpected Train cases: {len(train)}")
    if len(valid) != 207 or valid["series_uid"].nunique() != 207:
        raise RuntimeError(f"Unexpected Valid cases: {len(valid)}")
    expected_counts = {int(key): int(value) for key, value in cfg["data"]["expected_fold_counts"].items()}
    actual_counts = train["fold"].value_counts().sort_index().to_dict()
    if actual_counts != expected_counts:
        raise RuntimeError(f"Unexpected Train fold counts: {actual_counts} != {expected_counts}")
    train["feature_row"] = np.arange(len(train), dtype=int)
    valid["feature_row"] = np.arange(len(valid), dtype=int)
    return train, valid


def _checkpoint_and_routing_audit(
    cfg: dict[str, Any], train: pd.DataFrame, valid: pd.DataFrame
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    rows = []
    checkpoints: dict[int, dict[str, Any]] = {}
    report_root = Path(cfg["report_root"])
    output_root = Path(cfg["output_root"])
    valid_patients = set(valid["patient_id"].astype(str))
    for fold in range(1, 6):
        checkpoint = output_root / "expanded_strict" / "segmentation" / MODEL_FAMILY / f"fold_{fold}" / "model.pt"
        success = checkpoint.with_name("SUCCESS.json")
        legal_manifest = report_root / "expanded_strict_segmentation" / MODEL_FAMILY / f"fold_{fold}" / "segmentation_legal_split_manifest.csv"
        if not checkpoint.is_file() or not success.is_file() or not legal_manifest.is_file():
            raise FileNotFoundError(f"Missing frozen fold artifact for fold {fold}")
        raw = torch.load(checkpoint, map_location="cpu")
        snapshot = raw.get("config_snapshot", {})
        signature = raw.get("run_signature", {})
        if int(snapshot.get("outer_fold", -1)) != fold:
            raise RuntimeError(f"fold {fold}: checkpoint outer_fold mismatch")
        if signature.get("model_family") != MODEL_FAMILY:
            raise RuntimeError(f"fold {fold}: checkpoint model family mismatch")
        success_json = json.loads(success.read_text(encoding="utf-8"))
        digest = sha256_file(checkpoint)
        if success_json.get("model_sha256") != digest:
            raise RuntimeError(f"fold {fold}: checkpoint SHA256 mismatch")
        legal = pd.read_csv(legal_manifest, dtype={"patient_id": str})
        legal_patients = set(legal["patient_id"].astype(str))
        oof = train.loc[train["fold"].eq(fold)]
        oof_patients = set(oof["patient_id"].astype(str))
        oof_overlap = legal_patients & oof_patients
        valid_overlap = legal_patients & valid_patients
        if oof_overlap:
            raise RuntimeError(f"fold {fold}: OOF patient leakage: {sorted(oof_overlap)[:5]}")
        if valid_overlap:
            raise RuntimeError(f"fold {fold}: Valid patient leakage: {sorted(valid_overlap)[:5]}")
        rows.append({
            "fold": fold,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest,
            "checkpoint_outer_fold": int(snapshot["outer_fold"]),
            "legal_segmentation_rows": int(len(legal)),
            "legal_segmentation_patients": int(legal["patient_id"].nunique()),
            "train_oof_cases": int(len(oof)),
            "train_oof_patients": int(oof["patient_id"].nunique()),
            "oof_patient_overlap_with_training_legal_pool": 0,
            "valid_patient_overlap_with_training_legal_pool": 0,
        })
        checkpoints[fold] = {"path": checkpoint, "state": raw["state_dict"], "sha256": digest}
    return rows, checkpoints


@torch.no_grad()
def _extract_cases(
    model: torch.nn.Module,
    cases: pd.DataFrame,
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, dict[str, list[int]]]:
    # Two cases per batch = four phase images, matching the frozen SegResNet
    # physical phase-image batch size of four.
    dataset = PairedMeanPngDataset(cases, cfg)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=int(cfg["development"]["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(cfg["development"]["num_workers"]) > 0,
    )
    features = np.empty((len(dataset), 1024), dtype=np.float32)
    observed_shapes: dict[str, list[int]] = {}
    for pre, post, indices in loader:
        pre = pre.to(device, non_blocking=True)
        post = post.to(device, non_blocking=True)
        images = torch.cat([pre, post], dim=0)
        with autocast(enabled=device.type == "cuda"):
            feature_map, logits = model.encode_and_decode(images)
            probability = torch.sigmoid(logits)
            global_feature = global_pool(feature_map)
            pred_roi, probability_mass = roi_pool(
                feature_map,
                probability,
                mode="bilinear",
            )
        if feature_map.shape[1:] != (256, 96, 96):
            raise RuntimeError(f"Unexpected SegResNet feature map {tuple(feature_map.shape)}")
        if logits.shape[1:] != (1, 768, 768):
            raise RuntimeError(f"Unexpected SegResNet logits {tuple(logits.shape)}")
        if global_feature.shape[1] != 256 or pred_roi.shape[1] != 256:
            raise RuntimeError("Unexpected Global/PredROI dimension")
        if not torch.isfinite(global_feature).all() or not torch.isfinite(pred_roi).all():
            raise RuntimeError("Non-finite spatial feature")
        if not torch.isfinite(probability_mass).all() or torch.any(probability_mass <= 0):
            raise RuntimeError("Invalid soft probability mass")
        n = len(indices)
        g_pre, g_post = global_feature[:n], global_feature[n:]
        r_pre, r_post = pred_roi[:n], pred_roi[n:]
        z = torch.cat([g_pre, r_pre, g_post, r_post], dim=1)
        if z.shape != (n, 1024):
            raise RuntimeError(f"Unexpected z_2d_raw shape {tuple(z.shape)}")
        features[np.asarray(indices)] = z.float().cpu().numpy()
        if not observed_shapes:
            observed_shapes = {
                "phase_input": list(images.shape),
                "encoder_feature_map": list(feature_map.shape),
                "segmentation_logits": list(logits.shape),
                "continuous_probability_map": list(probability.shape),
                "global": list(global_feature.shape),
                "pred_roi": list(pred_roi.shape),
                "case_z_2d_raw": list(z.shape),
            }
    if features.dtype != np.float32 or features.shape != (len(cases), 1024):
        raise RuntimeError("Invalid exported feature array")
    if not np.isfinite(features).all():
        raise RuntimeError("Exported feature array contains NaN/Inf")
    return features, observed_shapes


def _public_arrays(
    cases: pd.DataFrame,
    z: np.ndarray,
    split: str,
    outer_fold: np.ndarray,
    source_model_fold: np.ndarray,
) -> dict[str, np.ndarray]:
    n = len(cases)
    arrays = {
        "series_uid": _unicode(cases["series_uid"]),
        "patient_id": _unicode(cases["patient_id"]),
        "split": _unicode([split] * n),
        "outer_fold": np.asarray(outer_fold, dtype=np.int64),
        "source_model_fold": np.asarray(source_model_fold, dtype=np.int64),
        "model_family": _unicode([MODEL_FAMILY] * n),
        "feature_version": _unicode([FEATURE_VERSION] * n),
        "z_2d_raw": np.asarray(z, dtype=np.float32),
    }
    if set(arrays) != PUBLIC_KEYS:
        raise RuntimeError(f"Unexpected public keys: {sorted(arrays)}")
    return arrays


def _manifest_rows(
    cases: pd.DataFrame,
    split: str,
    outer_fold: np.ndarray,
    source_model_fold: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame({
        "feature_row": np.arange(len(cases), dtype=int),
        "series_uid": cases["series_uid"].astype(str).to_numpy(),
        "patient_id": cases["patient_id"].astype(str).to_numpy(),
        "split": split,
        "outer_fold": np.asarray(outer_fold, dtype=np.int64),
        "source_model_fold": np.asarray(source_model_fold, dtype=np.int64),
        "model_family": MODEL_FAMILY,
        "feature_version": FEATURE_VERSION,
        "z_2d_raw_dim": 1024,
    })


def _verify_npz(path: Path, expected_cases: pd.DataFrame, split: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        arrays = {key: np.asarray(z[key]) for key in z.files}
    checks = {
        "public_keys_exact": set(arrays) == PUBLIC_KEYS,
        "rows": len(arrays["series_uid"]) == len(expected_cases),
        "series_uid_exact": np.array_equal(arrays["series_uid"].astype(str), expected_cases["series_uid"].astype(str).to_numpy()),
        "patient_id_exact": np.array_equal(arrays["patient_id"].astype(str), expected_cases["patient_id"].astype(str).to_numpy()),
        "split_exact": np.all(arrays["split"].astype(str) == split),
        "z_shape": arrays["z_2d_raw"].shape == (len(expected_cases), 1024),
        "z_dtype_float32": arrays["z_2d_raw"].dtype == np.float32,
        "z_finite": bool(np.isfinite(arrays["z_2d_raw"]).all()),
        "no_target": "target" not in arrays,
        "no_gtroi": not any("gt" in key.casefold() for key in arrays),
        "no_mask": not any("mask" in key.casefold() for key in arrays),
    }
    if split == "Train":
        expected_fold = expected_cases["fold"].to_numpy(dtype=np.int64)
        checks["outer_fold_exact"] = np.array_equal(arrays["outer_fold"], expected_fold)
        checks["source_model_fold_exact"] = np.array_equal(arrays["source_model_fold"], expected_fold)
        checks["strict_oof_fold_mapping"] = np.array_equal(arrays["outer_fold"], arrays["source_model_fold"])
    else:
        checks["outer_fold_is_zero"] = bool(np.all(arrays["outer_fold"] == 0))
        checks["source_model_fold_single_value"] = (
            len(np.unique(arrays["source_model_fold"])) == 1
        )
    checks = {key: bool(value) for key, value in checks.items()}
    if not all(checks.values()):
        raise RuntimeError(f"{path}: {json.dumps(checks)}")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    cfg = load_config(args.config)
    os.environ["TORCH_HOME"] = cfg["torch_home"]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = Path(args.output_dir).resolve() if args.output_dir else (
        Path(cfg["output_root"]) / "expanded_strict" / "2d_spatial_interface" / FEATURE_VERSION
    )
    output.mkdir(parents=True, exist_ok=True)

    train, valid = _read_cases(cfg)
    routing_rows, checkpoints = _checkpoint_and_routing_audit(cfg, train, valid)
    atomic_csv(pd.DataFrame(routing_rows), output / "fold_routing_audit.csv")

    train_z = np.empty((len(train), 1024), dtype=np.float32)
    valid_by_fold = np.empty((len(valid), 5, 1024), dtype=np.float32)
    shape_evidence: dict[str, list[int]] = {}

    for fold in range(1, 6):
        model = build_model(MODEL_FAMILY, cfg, load_pretrained=False).to(device).eval()
        model.load_state_dict(checkpoints[fold]["state"], strict=True)
        train_subset = train.loc[train["fold"].eq(fold)].copy().reset_index(drop=True)
        train_features, observed = _extract_cases(model, train_subset, cfg, device)
        train_z[train_subset["feature_row"].to_numpy(dtype=int)] = train_features
        valid_features, _ = _extract_cases(model, valid, cfg, device)
        valid_by_fold[:, fold - 1, :] = valid_features
        if not shape_evidence:
            shape_evidence = observed
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    train_outer_fold = train["fold"].to_numpy(dtype=np.int64)
    train_arrays = _public_arrays(
        train,
        train_z,
        "Train",
        train_outer_fold,
        train_outer_fold,
    )
    train_path = output / "train_oof_z_2d_raw.npz"
    _atomic_npz(train_path, **train_arrays)
    atomic_csv(
        _manifest_rows(train, "Train", train_outer_fold, train_outer_fold),
        output / "train_oof_manifest.csv",
    )

    valid_manifest_parts = []
    valid_paths = []
    for fold in range(1, 6):
        zeros = np.zeros(len(valid), dtype=np.int64)
        source = np.full(len(valid), fold, dtype=np.int64)
        arrays = _public_arrays(valid, valid_by_fold[:, fold - 1, :], "Valid", zeros, source)
        path = output / f"valid_fold_{fold}_z_2d_raw.npz"
        _atomic_npz(path, **arrays)
        valid_paths.append(path)
        part = _manifest_rows(valid, "Valid", zeros, source)
        part.insert(0, "fusion_fold", fold)
        valid_manifest_parts.append(part)
    atomic_csv(pd.concat(valid_manifest_parts, ignore_index=True), output / "valid_by_fold_manifest.csv")
    _atomic_npz(
        output / "valid_z_2d_raw_by_fold.npz",
        series_uid=_unicode(valid["series_uid"]),
        patient_id=_unicode(valid["patient_id"]),
        split=_unicode(["Valid"] * len(valid)),
        outer_fold=np.zeros(len(valid), dtype=np.int64),
        source_model_folds=np.arange(1, 6, dtype=np.int64),
        model_family=_unicode([MODEL_FAMILY] * len(valid)),
        feature_version=_unicode([FEATURE_VERSION] * len(valid)),
        z_2d_raw_by_fold=valid_by_fold.astype(np.float32),
    )

    smoke_checks = {
        "train": _verify_npz(train_path, train, "Train"),
        "valid_by_fold": {
            str(fold): _verify_npz(path, valid, "Valid")
            for fold, path in enumerate(valid_paths, start=1)
        },
        "train_fold_values_1_to_5": bool(np.isin(train_outer_fold, [1, 2, 3, 4, 5]).all()),
        "routing_audit_all_zero_overlap": all(
            row["oof_patient_overlap_with_training_legal_pool"] == 0
            and row["valid_patient_overlap_with_training_legal_pool"] == 0
            for row in routing_rows
        ),
        "gt_mask_used_for_feature_generation": False,
        "gtroi_used_in_public_interface": False,
        "hard_threshold_used_for_predroi": False,
        "outcome_target_loaded_or_used": False,
        "latent_fold_averaging_applied": False,
    }
    if not all([
        smoke_checks["train_fold_values_1_to_5"],
        smoke_checks["routing_audit_all_zero_overlap"],
    ]):
        raise RuntimeError("Strict interface smoke checks failed")

    metadata = {
        "status": "success",
        "model_family": MODEL_FAMILY,
        "feature_version": FEATURE_VERSION,
        "feature_order": FEATURE_ORDER,
        "feature_slices": FEATURE_SLICES,
        "z_2d_raw_dim": 1024,
        "dtype": "float32",
        "train_rows": int(len(train)),
        "valid_rows_per_fold": int(len(valid)),
        "valid_model_folds": [1, 2, 3, 4, 5],
        "input_columns_used": ["split", "task_row", "series_uid", "patient_id", "fold", "pre_image", "post_image"],
        "excluded_inputs": ["target", "pre_mask", "post_mask", "GTROI", "hard_threshold_mask"],
        "predroi_definition": "bilinear_resize(sigmoid(logits)) to feature-map space, then normalized soft weighted pooling",
        "train_routing": "outer_fold k uses only frozen SegResNet fold_k checkpoint",
        "valid_routing": "retain five separate representations; fusion fold k reads valid_fold_k_z_2d_raw.npz",
        "latent_fold_averaging_applied": False,
        "observed_tensor_shapes": shape_evidence,
        "config_path": cfg["_config_path"],
        "config_sha256": cfg["_config_sha256"],
        "checkpoints": {str(row["fold"]): {"path": row["checkpoint"], "sha256": row["checkpoint_sha256"]} for row in routing_rows},
    }
    atomic_json(metadata, output / "interface_metadata.json")
    spec_path = Path(__file__).with_name("2D_SPATIAL_INTERFACE_SPEC.md")
    atomic_text(
        spec_path.read_text(encoding="utf-8"), output / "2D_SPATIAL_INTERFACE_SPEC.md"
    )
    atomic_json({"status": "PASS", "checks": smoke_checks, "observed_tensor_shapes": shape_evidence}, output / "SMOKE_TEST.json")

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
        "train_shape": list(train_z.shape),
        "valid_by_fold_shape": list(valid_by_fold.shape),
        "dtype": str(train_z.dtype),
        "all_finite": bool(np.isfinite(train_z).all() and np.isfinite(valid_by_fold).all()),
        "strict_oof_verified": True,
        "gt_mask_used": False,
        "gtroi_used": False,
        "outcome_target_used": False,
        "hard_threshold_used": False,
        "latent_fold_averaging_applied": False,
        "checksums": checksums,
    }
    atomic_json(success, output / "SUCCESS.json")
    print(json.dumps(success, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

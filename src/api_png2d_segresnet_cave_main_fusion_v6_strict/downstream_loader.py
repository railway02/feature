#!/usr/bin/env python3
"""Fail-closed loader for fold-specific 2D--CAVE main representations.

The loader exposes one fusion coordinate system at a time.  It deliberately
does not load outcome targets: downstream labels must come from an independently
aligned outcome manifest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from loader import write_json


ROOT = Path("/root/autodl-tmp/aneurysm")
DEFAULT_OUT = ROOT / "outputs/api_png2d_segresnet_cave_main_fusion_v6_strict"
FIELDS = ("z_main", "main_logit", "main_prob", "spatial_gate", "temporal_gate")
FORBIDDEN_LABEL_FIELDS = {"target", "label", "outcome", "y", "adverse_outcome"}


def _read(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        fields = set(data.files)
        forbidden = fields & FORBIDDEN_LABEL_FIELDS
        if forbidden:
            raise AssertionError(f"{path}: label fields are forbidden: {sorted(forbidden)}")
        return {name: np.asarray(data[name]) for name in data.files}


def _as_str(value: np.ndarray) -> np.ndarray:
    return np.asarray(value).astype(str)


def load_fold_specific_main_interface(
    fold: int,
    output_root: str | Path = DEFAULT_OUT,
) -> dict[str, dict[str, np.ndarray] | int]:
    """Load Train and Valid representations for one downstream outer fold.

    For fold ``k``, both Train and Valid use column ``k-1`` from their
    respective by-fold banks.  This keeps every row within a downstream model
    in the same frozen main-fusion latent coordinate system.

    The returned development/holdout masks describe the outcome outer split.
    They do not create a leakage-free inner validation split for new downstream
    hyperparameter search; such search needs its own nested protocol.
    """
    if fold not in {1, 2, 3, 4, 5}:
        raise ValueError("fold must be one of 1,2,3,4,5")
    root = Path(output_root)
    train = _read(root / "train_main_outputs_by_fold.npz")
    valid = _read(root / "valid_main_outputs_by_fold.npz")
    oof = _read(root / "train_oof_main_outputs.npz")

    required_train = {
        "series_uid", "patient_id", "outer_fold", "source_fusion_folds",
        "z_main_by_fold", "main_logit_by_fold", "main_prob_by_fold",
        "spatial_gate_by_fold", "temporal_gate_by_fold",
    }
    required_valid = {
        "series_uid", "patient_id", "source_fusion_folds",
        "z_main_by_fold", "main_logit_by_fold", "main_prob_by_fold",
        "spatial_gate_by_fold", "temporal_gate_by_fold",
    }
    required_oof = {
        "series_uid", "patient_id", "outer_fold", "source_fusion_fold",
        *FIELDS,
    }
    for name, raw, required in (
        ("Train by-fold", train, required_train),
        ("Valid by-fold", valid, required_valid),
        ("Train OOF", oof, required_oof),
    ):
        missing = required - set(raw)
        if missing:
            raise KeyError(f"{name}: missing {sorted(missing)}")

    if train["z_main_by_fold"].shape != (781, 5, 256):
        raise AssertionError("Train z_main_by_fold shape mismatch")
    if valid["z_main_by_fold"].shape != (207, 5, 256):
        raise AssertionError("Valid z_main_by_fold shape mismatch")
    if not np.array_equal(train["source_fusion_folds"], np.arange(1, 6)):
        raise AssertionError("Train source_fusion_folds mismatch")
    if not np.array_equal(valid["source_fusion_folds"], np.arange(1, 6)):
        raise AssertionError("Valid source_fusion_folds mismatch")

    train_uid, valid_uid = _as_str(train["series_uid"]), _as_str(valid["series_uid"])
    train_pid, valid_pid = _as_str(train["patient_id"]), _as_str(valid["patient_id"])
    if len(np.unique(train_uid)) != 781 or len(np.unique(valid_uid)) != 207:
        raise AssertionError("duplicate series_uid")
    if set(train_uid) & set(valid_uid) or set(train_pid) & set(valid_pid):
        raise AssertionError("Train/Valid identifier leakage")
    if not np.array_equal(train_uid, _as_str(oof["series_uid"])):
        raise AssertionError("Train UID order differs from strict OOF")
    if not np.array_equal(train_pid, _as_str(oof["patient_id"])):
        raise AssertionError("Train patient order differs from strict OOF")
    outer_fold = np.asarray(train["outer_fold"], dtype=np.int64)
    if not np.array_equal(outer_fold, oof["outer_fold"]):
        raise AssertionError("Train outer fold differs from strict OOF")
    if not np.array_equal(oof["source_fusion_fold"], outer_fold):
        raise AssertionError("strict OOF source fold mismatch")

    column = fold - 1
    train_view = {
        "series_uid": train_uid,
        "patient_id": train_pid,
        "outer_fold": outer_fold,
        "development_mask": outer_fold != fold,
        "holdout_mask": outer_fold == fold,
        "source_fusion_fold": np.asarray(fold, dtype=np.int64),
    }
    valid_view = {
        "series_uid": valid_uid,
        "patient_id": valid_pid,
        "source_fusion_fold": np.asarray(fold, dtype=np.int64),
    }
    for base in FIELDS:
        train_value = np.asarray(train[f"{base}_by_fold"][:, column], dtype=np.float32)
        valid_value = np.asarray(valid[f"{base}_by_fold"][:, column], dtype=np.float32)
        if not np.isfinite(train_value).all() or not np.isfinite(valid_value).all():
            raise AssertionError(f"{base}: non-finite values")
        train_view[base] = train_value
        valid_view[base] = valid_value

    for split_name, view in (("Train", train_view), ("Valid", valid_view)):
        for base in ("main_prob", "spatial_gate", "temporal_gate"):
            value = view[base]
            if not np.all((value >= 0.0) & (value <= 1.0)):
                raise AssertionError(f"{split_name} {base}: outside [0,1]")

    holdout = train_view["holdout_mask"]
    for base in FIELDS:
        if not np.allclose(train_view[base][holdout], oof[base][holdout], rtol=1e-5, atol=1e-5):
            raise AssertionError(f"fold {fold}: holdout {base} differs from strict OOF")

    return {"fold": fold, "train": train_view, "valid": valid_view}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    folds = []
    for fold in range(1, 6):
        data = load_fold_specific_main_interface(fold, args.output_root)
        train, valid = data["train"], data["valid"]
        folds.append({
            "fold": fold,
            "development_rows": int(train["development_mask"].sum()),
            "holdout_rows": int(train["holdout_mask"].sum()),
            "valid_rows": int(len(valid["series_uid"])),
            "train_z_main_shape": list(train["z_main"].shape),
            "valid_z_main_shape": list(valid["z_main"].shape),
            "holdout_matches_strict_oof": True,
        })
    report = {
        "status": "PASS",
        "label_fields_loaded": False,
        "same_fold_coordinate_for_train_and_valid": True,
        "train_valid_identifier_overlap": 0,
        "latent_averaging_applied": False,
        "inner_selection_warning": "fold-specific banks guarantee outer-fold routing, not a new downstream leakage-free inner search",
        "folds": folds,
    }
    write_json(args.output_root / "main_fusion" / "DOWNSTREAM_LOADER_SMOKE_TEST.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

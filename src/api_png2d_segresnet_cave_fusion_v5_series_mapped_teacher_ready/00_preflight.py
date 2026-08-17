#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import (
    atomic_csv,
    atomic_json,
    load_config,
    load_temporal,
    load_train_folds,
    resolve_path,
)
from data import (
    build_case_manifest,
    build_segmentation_manifest,
    inspect_files,
    inspect_segmentation_inventory,
)
from segresnet_model import build_segresnet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = resolve_path(cfg["output_root"], cfg["project_root"])
    rep = resolve_path(cfg["report_root"], cfg["project_root"])
    out.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)

    tr = load_temporal(cfg, "Train")
    va = load_temporal(cfg, "Valid")
    folds = load_train_folds(cfg, tr)

    issues = []
    def check(ok, msg):
        if not ok:
            issues.append(msg)

    def sha256_file(path):
        h = hashlib.sha256()
        with Path(path).open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    locked_inputs = {
        "phase_mapping_manifest": "phase_mapping_manifest_sha256",
        "segmentation_inventory": "segmentation_inventory_sha256",
        "temporal_train_npz": "temporal_train_npz_sha256",
        "temporal_valid_npz": "temporal_valid_npz_sha256",
    }
    input_hashes = {}
    for path_key, hash_key in locked_inputs.items():
        path = resolve_path(cfg["data"][path_key], cfg["project_root"])
        actual = sha256_file(path)
        expected_hash = str(cfg["data"].get(hash_key, "")).strip()
        input_hashes[path_key] = {
            "path": str(path),
            "expected_sha256": expected_hash,
            "actual_sha256": actual,
            "match": bool(expected_hash and actual == expected_hash),
        }
        check(bool(expected_hash), f"Missing input lock {hash_key}")
        check(actual == expected_hash, f"Input SHA256 mismatch: {path_key}")

    check(tr["deep"].ndim == 2 and va["deep"].ndim == 2, "CAVE deep must be 2D")
    check(np.isfinite(tr["deep"]).all(), "Train CAVE deep has NaN/Inf")
    check(np.isfinite(va["deep"]).all(), "Valid CAVE deep has NaN/Inf")
    check(tr["deep"].shape[1] == va["deep"].shape[1], "Train/Valid temporal dim mismatch")
    check(set(np.unique(folds)) == {1,2,3,4,5}, "Fold IDs must be 1..5")
    check(len(set(tr["patient_id"]) & set(va["patient_id"])) == 0, "Train/Valid patient overlap")

    exp = cfg.get("expected", {})
    if bool(exp.get("strict", False)):
        check(len(tr["target"]) == int(exp["train_rows"]), f"Train rows={len(tr['target'])}")
        check(len(va["target"]) == int(exp["valid_rows"]), f"Valid rows={len(va['target'])}")
        check(int(tr["target"].sum()) == int(exp["train_positive"]), f"Train positive={int(tr['target'].sum())}")
        check(int(va["target"].sum()) == int(exp["valid_positive"]), f"Valid positive={int(va['target'].sum())}")
        check(tr["deep"].shape[1] == int(exp["cave_deep_dim"]), f"CAVE dim={tr['deep'].shape[1]}")

    fold_df = pd.DataFrame({"patient_id": tr["patient_id"], "fold": folds})
    check(
        int(fold_df.groupby("patient_id")["fold"].nunique().max()) == 1,
        "Patient appears in multiple outcome folds",
    )

    try:
        manifest = build_case_manifest(cfg)
        segmentation_manifest = build_segmentation_manifest(cfg)
        atomic_csv(manifest, out / "case_manifest.csv")
        atomic_csv(segmentation_manifest, out / "segmentation_manifest.csv")
        if bool(exp.get("strict", False)):
            check(
                len(segmentation_manifest) == int(exp["segmentation_phase_rows"]),
                f"Segmentation phase rows={len(segmentation_manifest)}",
            )

        detail, failures = inspect_files(manifest, cfg)
        atomic_csv(detail, rep / "00_task_image_mask_audit.csv")
        atomic_csv(failures, rep / "00_task_image_mask_failures.csv")
        check(len(failures) == 0, f"Task image/mask failures={len(failures)}")

        seg_detail, seg_failures = inspect_segmentation_inventory(
            segmentation_manifest
        )
        atomic_csv(seg_detail, rep / "00_segmentation_inventory_audit.csv")
        atomic_csv(seg_failures, rep / "00_segmentation_inventory_failures.csv")
        check(
            len(seg_failures) == 0,
            f"Segmentation inventory failures={len(seg_failures)}",
        )

        task_png_keys = set(manifest["pre_png_key"]) | set(manifest["post_png_key"])
        segmentation_keys = set(segmentation_manifest["segmentation_key"])
        missing_task_png = sorted(task_png_keys - segmentation_keys)
        check(
            not missing_task_png,
            f"Task PNG keys absent from segmentation inventory={len(missing_task_png)}",
        )
        check(
            manifest["series_uid"].nunique() == len(manifest),
            "Task case_manifest series_uid is not unique",
        )
        check(
            manifest["pre_phase_uid"].nunique() == len(manifest)
            and manifest["post_phase_uid"].nunique() == len(manifest),
            "Task phase_uid is not one-to-one",
        )
    except Exception as e:
        manifest = pd.DataFrame()
        segmentation_manifest = pd.DataFrame()
        detail = pd.DataFrame()
        failures = pd.DataFrame()
        seg_detail = pd.DataFrame()
        seg_failures = pd.DataFrame()
        missing_task_png = []
        issues.append(f"Manifest/image audit failed: {type(e).__name__}:{e}")

    try:
        import monai
        _ = build_segresnet(cfg)
        monai_version = monai.__version__
    except Exception as e:
        monai_version = f"FAILED:{type(e).__name__}:{e}"
        issues.append(f"MONAI SegResNet init failed: {e}")

    strategy = str(cfg["spatial"]["strategy"]).strip().casefold()
    check(
        strategy in {"pilot_single", "strict_crossfit", "external_checkpoint"},
        f"Unknown spatial.strategy={strategy}",
    )
    if strategy == "external_checkpoint":
        path = str(cfg["spatial"].get("external_checkpoint", "") or "").strip()
        check(bool(path), "external_checkpoint strategy requires a checkpoint path")

    payload = {
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "teacher_alignment": {
            "direct_png_mean_images": True,
            "direct_png_masks": True,
            "recompute_mean_from_sequence": False,
            "reuse_existing_cave_deep": True,
            "fusion_pca": False,
            "projection": "Linear(input_dim,256)->LayerNorm->GELU->Dropout",
            "main_fusion": "bidirectional_gate + [z2d,zt,product,absdiff] + 1024->256",
        },
        "strategy": strategy,
        "spatial_representation": cfg["spatial"].get("representation", "global_only"),
        "roi_source": cfg["spatial"].get("roi_source", ""),
        "task": {
            "train_rows": len(tr["target"]),
            "valid_rows": len(va["target"]),
            "train_positive": int(tr["target"].sum()),
            "valid_positive": int(va["target"].sum()),
            "cave_train_shape": list(tr["deep"].shape),
            "cave_valid_shape": list(va["deep"].shape),
            "fold_counts": pd.Series(folds).value_counts().sort_index().to_dict(),
        },
        "input_locks": input_hashes,
        "task_mapping": {
            "case_rows": int(len(manifest)),
            "series_uid_unique": int(manifest["series_uid"].nunique()) if len(manifest) else 0,
            "phase_rows_ok": int(len(detail)),
            "phase_rows_expected": int(len(manifest) * len(cfg["data"]["phases"])),
            "missing_from_segmentation_inventory": int(len(missing_task_png)),
            "failures": int(len(failures)),
            "mask_area_ratio_min": float(detail["mask_area_ratio"].min()) if len(detail) else None,
            "mask_area_ratio_median": float(detail["mask_area_ratio"].median()) if len(detail) else None,
            "mask_area_ratio_max": float(detail["mask_area_ratio"].max()) if len(detail) else None,
        },
        "segmentation_population": {
            "policy": cfg["spatial"].get("pilot_segmentation_population", "all_2d_inventory"),
            "phase_rows": int(len(segmentation_manifest)),
            "patients": int(segmentation_manifest["patient_id"].nunique()) if len(segmentation_manifest) else 0,
            "failures": int(len(seg_failures)),
        },
        "environment": {
            "torch": torch.__version__,
            "monai": monai_version,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        }
    }

    atomic_json(payload, rep / "00_preflight.json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if issues:
        raise SystemExit("PREFLIGHT_FAILED")
    print("PREFLIGHT_OK")


if __name__ == "__main__":
    main()

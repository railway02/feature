#!/usr/bin/env python3
"""Create label-free modality packages and repeat the 781/207 alignment audit."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from loader import load_and_audit, sha256, write_json

ROOT = Path("/root/autodl-tmp/aneurysm")
DEFAULT_SP = ROOT / "outputs/api_png2d_spatial_backbones_v6_strict/expanded_strict/featurebanks/segresnet"
DEFAULT_CAVE = ROOT / "outputs/api_png2d_gtmask_roi_cave_v2_complete_mapping_fullseq/adverse_prepost_series_task_v3"
DEFAULT_OUT = ROOT / "outputs/api_png2d_segresnet_cave_main_fusion_v6_strict"

def save(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spatial-root", type=Path, default=DEFAULT_SP)
    ap.add_argument("--cave-root", type=Path, default=DEFAULT_CAVE)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    st, sv = args.spatial_root / "train_spatial_features.npz", args.spatial_root / "valid_spatial_features.npz"
    ct, cv = args.cave_root / "train_features.npz", args.cave_root / "valid_features.npz"
    train, valid, audit, train_df, valid_df = load_and_audit(st, sv, ct, cv)
    raw = args.output_root / "raw_modalities"; align = args.output_root / "alignment"
    raw.mkdir(parents=True, exist_ok=True)
    align.mkdir(parents=True, exist_ok=True)
    # 2D source banks are intentionally referenced, not copied (large and immutable).
    save(raw / "cave_train_z_time_raw.npz", series_uid=train["series_uid"], patient_id=train["patient_id"], split=np.repeat("Train", len(train["target"])), outer_fold=train["fold"], feature_version=np.repeat("cave_deep_pre5120_post5120_v3", len(train["target"])), z_time_raw=train["temporal"])
    save(raw / "cave_valid_z_time_raw.npz", series_uid=valid["series_uid"], patient_id=valid["patient_id"], split=np.repeat("Valid", len(valid["target"])), feature_version=np.repeat("cave_deep_pre5120_post5120_v3", len(valid["target"])), z_time_raw=valid["temporal"])
    train_df.to_csv(align / "train_alignment.csv", index=False, encoding="utf-8")
    valid_df.to_csv(align / "valid_alignment.csv", index=False, encoding="utf-8")
    audit["input_sha256"] = {str(p): sha256(p) for p in (st, sv, ct, cv)}
    audit["raw_modality_files"] = {"segresnet_train_spatial_features": str(st), "segresnet_valid_spatial_features": str(sv), "cave_train_z_time_raw": str(raw / "cave_train_z_time_raw.npz"), "cave_valid_z_time_raw": str(raw / "cave_valid_z_time_raw.npz")}
    write_json(align / "alignment_audit.json", audit)
    write_json(raw / "RAW_MODALITY_PACKAGE.json", {"status":"PASS", "target_fields_exported": False, "spatial_banks_referenced_not_copied": True, **audit})
    print((align / "alignment_audit.json").read_text(encoding="utf-8"))

if __name__ == "__main__": main()

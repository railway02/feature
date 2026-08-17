#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import common as fast_common


def import_frozen_cave(code_root: Path):
    sys.path.insert(0, str(code_root.resolve()))
    for name in ["common", "io_ops", "manifest", "pooling", "release", "scalar_features", "schema", "v3_bridge", "cave_model", "extract_cave_featurebank"]:
        sys.modules.pop(name, None)
    cave = importlib.import_module("extract_cave_featurebank")
    io_ops = importlib.import_module("io_ops")
    return cave, io_ops


def import_roi_helpers(upstream_code: Path):
    spec = importlib.util.spec_from_file_location("fast_v1_upstream_roi", upstream_code / "roi.py")
    if spec is None or spec.loader is None:
        raise ImportError(upstream_code / "roi.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def activity_masks_with_fallback(original, enhancement, fov_mask, cave_config, fallback_config):
    try:
        return original(enhancement, fov_mask, cave_config)
    except AssertionError as exc:
        if not fallback_config.get("enabled") or not str(exc).startswith("Activity ROI too small:"):
            raise
        activity = np.percentile(
            enhancement, cave_config["activity"]["activity_temporal_percentile"], axis=0
        ).astype(np.float32)
        fov_positions = np.flatnonzero(fov_mask.ravel())
        minimum_active = int(cave_config["activity"]["minimum_active_pixels"])
        minimum_background = int(cave_config["activity"]["minimum_background_pixels"])
        if len(fov_positions) < minimum_active + minimum_background:
            raise AssertionError(f"FOV too small for activity fallback: {len(fov_positions)}") from exc
        values = activity.ravel()[fov_positions]
        order = np.argsort(values, kind="stable")
        active_positions = fov_positions[order[-minimum_active:]]
        high_count = max(1, minimum_active // 2)
        high_positions = fov_positions[order[-high_count:]]
        background_positions = fov_positions[order[:minimum_background]]
        active = np.zeros_like(fov_mask, dtype=bool)
        high = np.zeros_like(fov_mask, dtype=bool)
        background = np.zeros_like(fov_mask, dtype=bool)
        active.ravel()[active_positions] = True
        high.ravel()[high_positions] = True
        background.ravel()[background_positions] = True
        vessel = active.copy()
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        active_threshold = float(np.min(activity.ravel()[active_positions]))
        high_threshold = float(np.min(activity.ravel()[high_positions]))
        background_threshold = float(np.max(activity.ravel()[background_positions]))
        masks = {
            "active": active,
            "high_activity": high,
            "vessel": vessel,
            "background": background,
            "fov": fov_mask,
        }
        qc = {
            "activity_median": median,
            "activity_mad": mad,
            "active_threshold": active_threshold,
            "high_activity_threshold": high_threshold,
            "background_threshold": background_threshold,
            "active_pixels": int(active.sum()),
            "high_activity_pixels": int(high.sum()),
            "vessel_pixels": int(vessel.sum()),
            "background_pixels": int(background.sum()),
            "active_ratio_fov": float(active.sum() / max(fov_mask.sum(), 1)),
            "high_activity_ratio_fov": float(high.sum() / max(fov_mask.sum(), 1)),
            "vessel_ratio_fov": float(vessel.sum() / max(fov_mask.sum(), 1)),
            "background_ratio_fov": float(background.sum() / max(fov_mask.sum(), 1)),
            "vesselness_threshold": float("nan"),
            "vessel_fallback_to_active": True,
            "background_fallback": True,
            "activity_fallback": "topk_activity_within_fov",
            "activity_fallback_original_error": str(exc),
        }
        return masks, qc, activity


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--split", choices=["Train", "Valid"], required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--series-uids-file", type=Path)
    parser.add_argument("--max-series", type=int)
    parser.add_argument("--io-workers", type=int, default=4)
    parser.add_argument("--disable-empty-cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = fast_common.load_config(args.config)
    fast_common.configure_runtime(config)
    manifests = Path(config["paths"]["manifests"])
    roi_path = manifests / "roi_manifest_pred.csv"
    manifest_path = manifests / f"cave_manifest_pred_{args.split.casefold()}.csv"
    roi_frame = pd.read_csv(roi_path, dtype=str, keep_default_na=False)
    roi_frame = roi_frame[(roi_frame["split"] == args.split) & (~roi_frame["duplicate_excluded"].astype(str).str.casefold().eq("true"))].copy()
    by_hash = {str(row.frame_list_hash): row._asdict() for row in roi_frame.itertuples(index=False)}
    if len(by_hash) != len(roi_frame):
        raise AssertionError("ROI frame-list hash collision")

    selected = []
    if args.series_uids_file:
        selected = [line.strip() for line in args.series_uids_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    base = json.loads(Path(config["base_cave_config"]).read_text(encoding="utf-8"))
    roi_sha = fast_common.sha256_file(roi_path)
    manifest_sha = fast_common.sha256_file(manifest_path)
    base.update({
        "roi_pipeline_version": config["version"],
        "roi_branch": "pred",
        "roi_split": args.split,
        "roi_manifest_sha256": roi_sha,
        "roi_cave_manifest_sha256": manifest_sha,
        "roi_adapter": "post_load_gray_frames_pre_v3_preprocess_fast_v1",
        "roi_activity_fallback": config["roi_cave"]["activity_fallback"],
    })
    frozen_path = Path(config["paths"]["outputs"]) / "cave_frozen_configs" / f"pred_{args.split.casefold()}.json"
    fast_common.atomic_json(base, frozen_path)
    frozen_hash = fast_common.sha256_text(json.dumps(base, ensure_ascii=False, sort_keys=True, separators=(",", ":")))

    cave, io_ops = import_frozen_cave(Path(config["cave_code_root"]))
    roi_helpers = import_roi_helpers(Path(config["upstream_v1_code"]))
    original_load = io_ops.load_gray_frames
    original_process = cave.process_phase

    if args.disable_empty_cache:
        cave.torch.cuda.empty_cache = lambda: None

    def roi_load(paths, num_workers=4):
        frames = original_load(paths, num_workers=args.io_workers)
        key = io_ops.hash_lines([str(value) for value in paths])
        if key not in by_hash:
            raise KeyError(f"No ROI mapping for frame-list hash {key}")
        box = roi_helpers.bbox_from_text(by_hash[key]["expanded_bbox"])
        return roi_helpers.crop_frames(frames, box)

    def roi_process(args_inner, extractor, v3, plan, provenance, schema_path):
        key = plan.frame_list_hash
        if key not in by_hash:
            raise KeyError(f"No ROI row for {plan.series_uid} {plan.phase}")
        row = by_hash[key]
        original_activity = v3.module.build_activity_masks
        fallback_config = config["roi_cave"]["activity_fallback"]
        def patched_activity(enhancement, fov_mask, cave_config):
            return activity_masks_with_fallback(original_activity, enhancement, fov_mask, cave_config, fallback_config)
        v3.module.build_activity_masks = patched_activity
        try:
            result = original_process(args_inner, extractor, v3, plan, provenance, schema_path)
        finally:
            v3.module.build_activity_masks = original_activity
        directory = cave._phase_output_dir(args_inner.output_root, plan)
        metadata_path = directory / "metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["roi"] = {
                "roi_pipeline_version": config["version"],
                "roi_branch": "pred",
                "mask_source": "oof_pred" if args.split == "Train" else "valid_pred",
                "segmentation_model_hash": row.get("segmentation_model_hash", ""),
                "segmentation_fold": int(float(row.get("segmentation_fold", 0) or 0)),
                "original_bbox": row.get("original_bbox", ""),
                "expanded_bbox": row["expanded_bbox"],
                "crop_padding_factor": float(row["crop_padding_factor"]),
                "crop_padding": row.get("crop_padding", "0|0|0|0"),
                "roi_area_ratio": float(row["roi_area_ratio"]),
                "mask_area_ratio": float(row["mask_area_ratio"]),
                "alignment_transform": row["orientation_transform"],
                "fallback_type": row["fallback_type"],
                "roi_manifest_sha256": roi_sha,
            }
            fast_common.atomic_json(metadata, metadata_path)
            qc_path = directory / "qc.json"
            qc = json.loads(qc_path.read_text(encoding="utf-8"))
            metadata["roi"]["cave_activity_fallback"] = qc.get("activity_fallback", "none")
            metadata["roi"]["cave_activity_fallback_original_error"] = qc.get("activity_fallback_original_error", "")
            fast_common.atomic_json(metadata, metadata_path)
            qc.update({
                "roi_area_ratio": float(row["roi_area_ratio"]),
                "roi_mask_area_ratio": float(row["mask_area_ratio"]),
                "roi_fallback": int(row["fallback_type"] != "none"),
            })
            fast_common.atomic_json(qc, qc_path)
            success_path = directory / ".SUCCESS.json"
            success = json.loads(success_path.read_text(encoding="utf-8"))
            success.update({
                "roi_manifest_sha256": roi_sha,
                "roi_branch": "pred",
                "expanded_bbox": row["expanded_bbox"],
            })
            fast_common.atomic_json(success, success_path)
        return result

    cave.load_gray_frames = roi_load
    cave.process_phase = roi_process
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)
    argv = [
        "extract_cave_featurebank.py", "--mode", "custom",
        "--manifest", str(manifest_path),
        "--cave-repo", config["cave_repo"],
        "--checkpoint", config["checkpoint"],
        "--v3-extractor", config["v3_extractor"],
        "--v3-base-config", config["v3_base_config"],
        "--v3-override-config", config["v3_override_config"],
        "--output-root", str(args.output_root),
        "--report-root", str(args.report_root),
        "--frozen-config", str(frozen_path),
        "--io-workers", str(args.io_workers),
    ]
    if args.max_series:
        argv.extend(["--max-series", str(args.max_series)])
    for uid in selected:
        argv.extend(["--series-uid", uid])
    if args.overwrite:
        argv.append("--overwrite")
    old_argv = sys.argv
    sys.argv = argv
    try:
        code = cave.main()
    finally:
        sys.argv = old_argv
    summary = {
        "split": args.split,
        "output_root": str(args.output_root),
        "selected_series": len(selected),
        "max_series": args.max_series,
        "io_workers": args.io_workers,
        "disable_empty_cache": args.disable_empty_cache,
        "roi_manifest_sha256": roi_sha,
        "manifest_sha256": manifest_sha,
        "frozen_config_hash": frozen_hash,
        "exit_code": int(code),
    }
    fast_common.atomic_json(summary, args.report_root / "worker_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())

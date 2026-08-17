#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


RETAINED = (
    "embedding_5120.npy",
    "embedding_views_5120.npz",
    "f4_last_ensemble.fp16.npy",
    "f5_last_ensemble.fp16.npy",
    "phase_trajectories_16.fp16.npz",
    "curves.npz",
    "scalar_features.json",
    "metadata.json",
    "qc.json",
    ".SUCCESS.json",
)
REMOVABLE = (
    "probabilities_original.fp16.npz",
    "input_mosaic.jpg",
    "artery_vein_overlay.png",
    "artery_probability.png",
    "vein_probability.png",
    "vessel_probability.png",
    "vessel_union_probability.png",
)


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def compact_phase(directory: Path, common) -> dict[str, object]:
    missing = [name for name in RETAINED if not (directory / name).is_file()]
    if missing:
        raise RuntimeError(f"Cannot compact incomplete phase {directory}: {missing}")
    removed = []
    removed_bytes = 0
    blocks = directory / "blocks"
    if blocks.exists():
        removed_bytes += sum(
            path.stat().st_size for path in blocks.rglob("*") if path.is_file()
        )
        shutil.rmtree(blocks)
        removed.append("blocks/")
    for name in REMOVABLE:
        path = directory / name
        if path.is_file():
            removed_bytes += path.stat().st_size
            path.unlink()
            removed.append(name)
    payload = {
        "status": "compacted",
        "branch": "gt_oracle_1p5",
        "removed": removed,
        "removed_bytes": removed_bytes,
    }
    common.atomic_json(payload, directory / ".COMPACTED.json")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["Train", "Valid"], required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--series-uids-file", type=Path, required=True)
    parser.add_argument("--io-workers", type=int, default=8)
    parser.add_argument("--disable-empty-cache", action="store_true")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root = Path(config["project_root"])
    fast_code = root / "code/api_adverse_lesion_cave_fast_v1"
    sys.path.insert(0, str(fast_code))
    import common as fast_common
    import fast_roi as roi_helpers

    pred_worker = import_file(
        fast_code / "02_extract_pred_worker.py", "gt_oracle_pred_worker_base"
    )
    roi_path = (
        root / "manifests/api_adverse_lesion_cave_v1/roi_manifest_gt.csv"
    )
    manifest_path = (
        root
        / "manifests/api_adverse_lesion_cave_v1"
        / f"cave_manifest_gt_{args.split.casefold()}.csv"
    )
    roi = pd.read_csv(roi_path, dtype=str, keep_default_na=False)
    roi = roi[
        (roi["split"] == args.split)
        & (~roi["duplicate_excluded"].str.casefold().eq("true"))
    ].copy()
    by_hash = {
        str(row.frame_list_hash): row._asdict()
        for row in roi.itertuples(index=False)
    }
    if len(by_hash) != len(roi):
        raise AssertionError("GT ROI frame-list hash collision")

    selected = [
        line.strip()
        for line in args.series_uids_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    base = json.loads(Path(config["base_cave_config"]).read_text(encoding="utf-8"))
    roi_sha = fast_common.sha256_file(roi_path)
    manifest_sha = fast_common.sha256_file(manifest_path)
    base.update(
        {
            "roi_pipeline_version": "api_adverse_lesion_cave_gt_oracle_v1",
            "roi_branch": "gt_oracle_1p5",
            "roi_split": args.split,
            "roi_manifest_sha256": roi_sha,
            "roi_cave_manifest_sha256": manifest_sha,
            "roi_adapter": "post_load_gray_frames_pre_v3_preprocess_gt_oracle",
            "roi_activity_fallback": config["roi_cave"]["activity_fallback"],
        }
    )
    frozen_path = (
        args.output_root.parent
        / "cave_frozen_configs"
        / f"gt_oracle_1p5_{args.split.casefold()}.json"
    )
    fast_common.atomic_json(base, frozen_path)

    cave, io_ops = pred_worker.import_frozen_cave(
        Path(config["cave_code_root"])
    )
    original_load = io_ops.load_gray_frames
    original_process = cave.process_phase
    if args.disable_empty_cache:
        cave.torch.cuda.empty_cache = lambda: None

    def roi_load(paths, num_workers=4):
        frames = original_load(paths, num_workers=args.io_workers)
        key = io_ops.hash_lines([str(value) for value in paths])
        if key not in by_hash:
            raise KeyError(f"No GT ROI mapping for frame-list hash {key}")
        box = roi_helpers.bbox_from_text(by_hash[key]["expanded_bbox"])
        return roi_helpers.crop_frames(frames, box)

    def roi_process(args_inner, extractor, v3, plan, provenance, schema_path):
        row = by_hash[plan.frame_list_hash]
        original_activity = v3.module.build_activity_masks
        fallback_config = config["roi_cave"]["activity_fallback"]

        def patched_activity(enhancement, fov_mask, cave_config):
            return pred_worker.activity_masks_with_fallback(
                original_activity,
                enhancement,
                fov_mask,
                cave_config,
                fallback_config,
            )

        v3.module.build_activity_masks = patched_activity
        try:
            result = original_process(
                args_inner, extractor, v3, plan, provenance, schema_path
            )
        finally:
            v3.module.build_activity_masks = original_activity
        directory = cave._phase_output_dir(args_inner.output_root, plan)
        metadata_path = directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        qc_path = directory / "qc.json"
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        metadata["roi"] = {
            "roi_pipeline_version": "api_adverse_lesion_cave_gt_oracle_v1",
            "roi_branch": "gt_oracle_1p5",
            "mask_source": "gt",
            "original_bbox": row.get("original_bbox", ""),
            "expanded_bbox": row["expanded_bbox"],
            "crop_padding_factor": float(row["crop_padding_factor"]),
            "crop_padding": row.get("crop_padding", "0|0|0|0"),
            "roi_area_ratio": float(row["roi_area_ratio"]),
            "mask_area_ratio": float(row["mask_area_ratio"]),
            "alignment_transform": row["orientation_transform"],
            "fallback_type": "none",
            "roi_manifest_sha256": roi_sha,
            "cave_activity_fallback": qc.get("activity_fallback", "none"),
            "cave_activity_fallback_original_error": qc.get(
                "activity_fallback_original_error", ""
            ),
        }
        fast_common.atomic_json(metadata, metadata_path)
        qc.update(
            {
                "roi_area_ratio": float(row["roi_area_ratio"]),
                "roi_mask_area_ratio": float(row["mask_area_ratio"]),
                "roi_fallback": 0,
            }
        )
        fast_common.atomic_json(qc, qc_path)
        success_path = directory / ".SUCCESS.json"
        success = json.loads(success_path.read_text(encoding="utf-8"))
        success.update(
            {
                "roi_manifest_sha256": roi_sha,
                "roi_branch": "gt_oracle_1p5",
                "expanded_bbox": row["expanded_bbox"],
            }
        )
        fast_common.atomic_json(success, success_path)
        compact_phase(directory, fast_common)
        return result

    cave.load_gray_frames = roi_load
    cave.process_phase = roi_process
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)
    argv = [
        "extract_cave_featurebank.py",
        "--mode",
        "custom",
        "--manifest",
        str(manifest_path),
        "--cave-repo",
        config["cave_repo"],
        "--checkpoint",
        config["checkpoint"],
        "--v3-extractor",
        config["v3_extractor"],
        "--v3-base-config",
        config["v3_base_config"],
        "--v3-override-config",
        config["v3_override_config"],
        "--output-root",
        str(args.output_root),
        "--report-root",
        str(args.report_root),
        "--frozen-config",
        str(frozen_path),
        "--io-workers",
        str(args.io_workers),
    ]
    for uid in selected:
        argv.extend(["--series-uid", uid])
    old_argv = sys.argv
    sys.argv = argv
    try:
        code = cave.main()
    finally:
        sys.argv = old_argv
    summary = {
        "split": args.split,
        "series": len(selected),
        "output_root": str(args.output_root),
        "roi_manifest_sha256": roi_sha,
        "manifest_sha256": manifest_sha,
        "exit_code": int(code),
    }
    fast_common.atomic_json(summary, args.report_root / "worker_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())

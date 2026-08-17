#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


RETAINED = (
    "embedding_5120.npy", "embedding_views_5120.npz",
    "f4_last_ensemble.fp16.npy", "f5_last_ensemble.fp16.npy",
    "phase_trajectories_16.fp16.npz", "curves.npz",
    "scalar_features.json", "metadata.json", "qc.json", ".SUCCESS.json",
)
REMOVABLE = (
    "probabilities_original.fp16.npz", "input_mosaic.jpg",
    "artery_vein_overlay.png", "artery_probability.png",
    "vein_probability.png", "vessel_probability.png", "vessel_union_probability.png",
)


def import_frozen_cave(code_root: Path):
    sys.path.insert(0, str(code_root.resolve()))
    for name in [
        "common", "io_ops", "manifest", "pooling", "release", "scalar_features",
        "schema", "v3_bridge", "cave_model", "extract_cave_featurebank",
    ]:
        sys.modules.pop(name, None)
    cave = importlib.import_module("extract_cave_featurebank")
    io_ops = importlib.import_module("io_ops")
    return cave, io_ops


def bbox_from_text(value: str) -> tuple[int, int, int, int]:
    parts = tuple(int(item) for item in str(value).split("|"))
    if len(parts) != 4:
        raise ValueError(value)
    return parts


def crop_frames(frames: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    side = x1 - x0
    if side <= 0 or y1 - y0 != side:
        raise AssertionError(f"Invalid square bbox {box}")
    height, width = frames.shape[1:]
    left, top = max(0, -x0), max(0, -y0)
    right, bottom = max(0, x1 - width), max(0, y1 - height)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    if sx0 >= sx1 or sy0 >= sy1:
        raise AssertionError(f"ROI outside image {box} {frames.shape}")
    cropped = frames[:, sy0:sy1, sx0:sx1]
    if any((left, top, right, bottom)):
        padded = np.empty((len(frames), side, side), dtype=frames.dtype)
        for index, frame in enumerate(frames):
            border = np.concatenate((frame[0], frame[-1], frame[:, 0], frame[:, -1]))
            padded[index].fill(np.asarray(np.median(border), dtype=frames.dtype))
        padded[:, top:top + cropped.shape[1], left:left + cropped.shape[2]] = cropped
        cropped = padded
    if cropped.shape[1:] != (side, side):
        raise AssertionError(f"Unexpected crop {cropped.shape}")
    return cropped


def crop_zero(array: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    original_ndim = array.ndim
    frames = array[None] if original_ndim == 2 else array
    x0, y0, x1, y1 = box
    side = x1 - x0
    height, width = frames.shape[1:]
    output = np.zeros((len(frames), side, side), dtype=frames.dtype)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    left, top = max(0, -x0), max(0, -y0)
    output[:, top:top + (sy1 - sy0), left:left + (sx1 - sx0)] = frames[:, sy0:sy1, sx0:sx1]
    return output[0] if original_ndim == 2 else output


def crop_preprocessing(whole: dict, box: tuple[int, int, int, int]) -> dict:
    return {
        "normalized": crop_frames(whole["normalized"], box),
        "normalization_low": whole["normalization_low"],
        "normalization_high": whole["normalization_high"],
        "fov": crop_zero(whole["fov"], box).astype(bool),
        "baseline": crop_frames(whole["baseline"][None], box)[0],
        "enhancement": crop_zero(whole["enhancement"], box),
        "masks": {name: crop_zero(value, box).astype(bool) for name, value in whole["masks"].items()},
        "activity": crop_zero(whole["activity"], box),
        "qc": {**whole["qc"], "preprocess_scope": "whole_then_crop"},
    }


def compact_phase(directory: Path) -> None:
    missing = [name for name in RETAINED if not (directory / name).is_file()]
    if missing:
        raise RuntimeError(f"Cannot compact incomplete phase {directory}: {missing}")
    blocks = directory / "blocks"
    if blocks.exists():
        shutil.rmtree(blocks)
    for name in REMOVABLE:
        path = directory / name
        if path.is_file():
            path.unlink()
    (directory / ".COMPACTED.json").write_text(
        json.dumps({"status": "compacted", "branch": "record_gt_local"}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--scale", choices=["30", "40"], required=True)
    parser.add_argument("--split", choices=["Train", "Valid"], required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--series-uids-file", type=Path, required=True)
    parser.add_argument("--io-workers", type=int, default=8)
    parser.add_argument("--disable-empty-cache", action="store_true")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root = Path(config["project_root"])
    sys.path.insert(0, str(root / "code/api_adverse_lesion_record_v2"))
    import common as record_common

    phase_path = Path(config["paths"]["manifests"]) / f"gt_context{args.scale}_phase_manifest.csv"
    manifest_path = Path(config["paths"]["manifests"]) / f"cave_manifest_gt_context{args.scale}_{args.split.casefold()}.csv"
    phase = pd.read_csv(phase_path, dtype=str, keep_default_na=False)
    phase = phase[phase.split == args.split].copy()
    by_hash = {str(row.frame_list_hash): row._asdict() for row in phase.itertuples(index=False)}
    if len(by_hash) != len(phase):
        raise AssertionError("Frame-list hash collision in record GT context manifest")
    selected = [line.strip() for line in args.series_uids_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    base = json.loads(Path(config["base_cave_config"]).read_text(encoding="utf-8"))
    base.update({
        "record_pipeline_version": config["version"],
        "roi_branch": f"gt_context{args.scale}",
        "roi_split": args.split,
        "roi_manifest_sha256": record_common.sha256_file(phase_path),
        "roi_cave_manifest_sha256": record_common.sha256_file(manifest_path),
        "record_temporal_view_policy": "locked_to_whole_metadata",
        "record_preprocess_policy": "whole_then_crop",
    })
    frozen_path = Path(config["paths"]["outputs"]) / "cave_frozen_configs" / f"gt_context{args.scale}_{args.split.casefold()}.json"
    record_common.atomic_json(base, frozen_path)
    cave, io_ops = import_frozen_cave(Path(config["cave_code_root"]))
    original_load = io_ops.load_gray_frames
    original_process = cave.process_phase
    holder: dict[str, dict] = {}
    if args.disable_empty_cache:
        cave.torch.cuda.empty_cache = lambda: None

    def roi_load(paths, num_workers=4):
        frames = original_load(paths, num_workers=args.io_workers)
        key = io_ops.hash_lines([str(value) for value in paths])
        row = by_hash[key]
        box = bbox_from_text(row["expanded_bbox"])
        holder[key] = {"frames": frames, "box": box}
        return crop_frames(frames, box)

    def roi_process(args_inner, extractor, v3, plan, provenance, schema_path):
        row = by_hash[plan.frame_list_hash]
        reference = json.loads(Path(row["whole_metadata_path"]).read_text(encoding="utf-8"))
        reference_blocks = reference["blocks"]
        original_preprocess = v3.preprocess
        original_views = cave.temporal_views
        view_call = [0]

        def patched_preprocess(_cropped_frames):
            state = holder[plan.frame_list_hash]
            return crop_preprocessing(original_preprocess(state["frames"]), state["box"])

        def patched_views(_frames, max_len=20):
            index = view_call[0]
            if index >= len(reference_blocks):
                raise AssertionError("More temporal blocks than Whole reference")
            block = reference_blocks[index]
            view_call[0] += 1
            block_indices = [int(value) for value in block["indices"]]
            position = {value: idx for idx, value in enumerate(block_indices)}
            return {
                name: np.asarray([position[int(value)] for value in indices], dtype=np.int64)
                for name, indices in block["view_indices"].items()
            }

        v3.preprocess = patched_preprocess
        cave.temporal_views = patched_views
        try:
            result = original_process(args_inner, extractor, v3, plan, provenance, schema_path)
        finally:
            v3.preprocess = original_preprocess
            cave.temporal_views = original_views
            holder.pop(plan.frame_list_hash, None)
        if view_call[0] != len(reference_blocks):
            raise AssertionError("Temporal block count differs from Whole reference")
        directory = cave._phase_output_dir(args_inner.output_root, plan)
        metadata_path = directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for actual, expected in zip(metadata["blocks"], reference_blocks):
            if actual["view_indices"] != expected["view_indices"]:
                raise AssertionError("Local temporal views differ from Whole reference")
        metadata["record_roi"] = {
            "record_uid": row["record_uid"], "mapping_tier": row["mapping_tier"],
            "roi_branch": f"gt_context{args.scale}", "mask_source": "gt",
            "expanded_bbox": row["expanded_bbox"], "crop_padding": row["crop_padding"],
            "roi_area_ratio": float(row["roi_area_ratio"]),
            "large_lesion_override": int(float(row["large_lesion_override"])),
            "temporal_views_locked": True, "whole_preprocess_then_crop": True,
            "whole_metadata_path": row["whole_metadata_path"],
        }
        record_common.atomic_json(metadata, metadata_path)
        success_path = directory / ".SUCCESS.json"
        success = json.loads(success_path.read_text(encoding="utf-8"))
        success.update({"record_uid": row["record_uid"], "roi_branch": f"gt_context{args.scale}", "temporal_views_locked": True})
        record_common.atomic_json(success, success_path)
        compact_phase(directory)
        return result

    cave.load_gray_frames = roi_load
    cave.process_phase = roi_process
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)
    argv = [
        "extract_cave_featurebank.py", "--mode", "custom",
        "--manifest", str(manifest_path), "--cave-repo", config["cave_repo"],
        "--checkpoint", config["checkpoint"], "--v3-extractor", config["v3_extractor"],
        "--v3-base-config", config["v3_base_config"], "--v3-override-config", config["v3_override_config"],
        "--output-root", str(args.output_root), "--report-root", str(args.report_root),
        "--frozen-config", str(frozen_path), "--io-workers", str(args.io_workers),
    ]
    for uid in selected:
        argv.extend(["--series-uid", uid])
    old = sys.argv
    sys.argv = argv
    try:
        code = cave.main()
    finally:
        sys.argv = old
    summary = {"scale": args.scale, "split": args.split, "selected_series": len(selected), "exit_code": int(code)}
    record_common.atomic_json(summary, args.report_root / "worker_summary.json")
    print(json.dumps(summary, indent=2))
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import common as local_common
from roi import bbox_from_text, crop_frames


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


def is_activity_error(exc: BaseException) -> bool:
    return "Activity ROI too small" in str(exc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["Train", "Valid"], required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--series-uids-file", type=Path)
    parser.add_argument("--max-series", type=int)
    parser.add_argument("--io-workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = local_common.load_config(args.config)
    local_common.configure_runtime(cfg)
    manifests = Path(cfg["paths"]["manifests"])
    stage1_success = manifests / ".STAGE1_ELIGIBLE_SUCCESS.json"
    if not stage1_success.is_file():
        raise RuntimeError(f"缺少第一阶段闭合标记：{stage1_success}")

    roi_path = manifests / "roi_phase_manifest_eligible.csv"
    manifest_path = manifests / f"cave_manifest_local_{args.split.casefold()}_eligible.csv"
    views_path = manifests / "whole_temporal_views_eligible.csv"
    roi = pd.read_csv(roi_path, dtype=str, keep_default_na=False)
    roi = roi[roi["split"] == args.split].copy()
    by_hash = {str(row.frame_list_hash): row._asdict() for row in roi.itertuples(index=False)}
    if len(by_hash) != len(roi):
        raise AssertionError("ROI frame_list_hash 不唯一")
    views = pd.read_csv(views_path, dtype=str, keep_default_na=False)
    views = views[views["split"] == args.split]
    views_by_hash = {str(row.frame_list_hash): json.loads(row.blocks_json) for row in views.itertuples(index=False)}
    missing_views = set(by_hash) - set(views_by_hash)
    if missing_views:
        raise AssertionError(f"{len(missing_views)} 个 ROI phase 缺少 Whole 时间索引")

    selected: list[str] = []
    if args.series_uids_file:
        selected = [line.strip() for line in args.series_uids_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    base = json.loads(Path(cfg["base_cave_config"]).read_text(encoding="utf-8"))
    roi_cfg = cfg.get("roi", {})
    target_rule = str(roi_cfg.get("target_rule", ""))
    selected_labels = [int(value) for value in roi_cfg.get("foreground_labels", [])]
    if target_rule != "png2d_gt_labels_1_2_nonzero" or selected_labels != [1, 2]:
        raise ValueError(
            f"Invalid labels12 ROI config: rule={target_rule}, labels={selected_labels}"
        )
    roi_branch = str(roi_cfg.get("branch", "png2d_gt_fullseq"))
    temporal_policy = cfg.get("temporal", {}).get("policy", "freeze_whole_indices")
    base.update({
        "roi_pipeline_version": cfg["version"],
        "roi_branch": roi_branch,
        "foreground_rule": "labels_in_1_2_equivalent_nonzero",
        "selected_labels": selected_labels,
        "roi_split": args.split,
        "roi_manifest_sha256": local_common.sha256_file(roi_path),
        "local_cave_manifest_sha256": local_common.sha256_file(manifest_path),
        "stage1_success_sha256": local_common.sha256_file(stage1_success),
        "roi_adapter": "one_gt_mask_roi_per_series_phase_applied_to_all_frames_in_memory",
        "segmentation_model_used": False,
        "local_frames_saved": False,
        "temporal_policy": temporal_policy,
    })
    frozen = Path(cfg["paths"]["outputs"]) / "cave_frozen_configs" / f"local_full_{args.split.casefold()}.json"
    local_common.atomic_json(base, frozen)

    cave, io_ops = import_frozen_cave(Path(cfg["cave_code_root"]))
    original_load = io_ops.load_gray_frames
    original_process = cave.process_phase
    original_temporal_views = cave.temporal_views
    active_mode: dict[str, str] = {}

    def roi_load(paths, num_workers=4):
        frames = original_load(paths, num_workers=args.io_workers)
        key = io_ops.hash_lines([str(value) for value in paths])
        row = by_hash.get(key)
        if row is None:
            raise KeyError(f"当前 frame_list_hash 没有 ROI：{key}")
        fields = {
            "primary": "expanded_bbox",
            "fallback": "fallback_bbox",
            "extended": "extended_fallback_bbox",
        }
        field = fields.get(active_mode.get(key, "primary"), "expanded_bbox")
        return crop_frames(frames, bbox_from_text(row[field]))

    def cache_state(args_inner, plan, row):
        """Return (state, directory, reason).

        state is one of:
          - "valid": complete Local-CAVE result that can be safely resumed/skipped;
          - "partial": an existing but incomplete or stale result;
          - "missing": no previous result.
        """
        directory = cave._phase_output_dir(args_inner.output_root, plan)
        if not directory.exists():
            return "missing", directory, ""

        success_path = directory / ".SUCCESS.json"
        metadata_path = directory / "metadata.json"
        embedding_path = directory / "embedding_5120.npy"
        missing = [
            path.name
            for path in (success_path, metadata_path, embedding_path)
            if not path.is_file()
        ]
        if missing:
            return "partial", directory, "missing:" + ",".join(missing)

        try:
            success = json.loads(success_path.read_text(encoding="utf-8"))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return "partial", directory, f"unreadable_metadata:{exc!r}"

        expected = {
            "phase_uid": row["phase_uid"],
            "annotation_source": "png_2d_gt",
            "mask_sha256": row["mask_sha256"],
            "effective_mask_array_sha256": row["effective_mask_array_sha256"],
            "temporal_policy": temporal_policy,
            "roi_pipeline_version": cfg["version"],
            "target_rule": target_rule,
            "foreground_rule": "labels_in_1_2_equivalent_nonzero",
            "selected_labels": selected_labels,
        }
        for key, value in expected.items():
            if str(success.get(key, "")) != str(value):
                return (
                    "partial",
                    directory,
                    f"success_{key}_mismatch:{success.get(key)!r}!={value!r}",
                )

        valid_bboxes = {
            str(row["expanded_bbox"]),
            str(row["fallback_bbox"]),
            str(row["extended_fallback_bbox"]),
        }
        if str(success.get("used_bbox", "")) not in valid_bboxes:
            return "partial", directory, "success_used_bbox_mismatch"

        roi_meta = metadata.get("roi", {})
        for key, value in expected.items():
            if str(roi_meta.get(key, "")) != str(value):
                return (
                    "partial",
                    directory,
                    f"metadata_roi_{key}_mismatch:{roi_meta.get(key)!r}!={value!r}",
                )
        if str(roi_meta.get("used_bbox", "")) not in valid_bboxes:
            return "partial", directory, "metadata_roi_used_bbox_mismatch"

        return "valid", directory, ""

    def quarantine_partial(directory: Path, phase_uid: str, reason: str) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        safe_uid = phase_uid.replace("/", "_").replace("::", "__")
        quarantine_root = args.report_root / "partial_cache_quarantine"
        quarantine_root.mkdir(parents=True, exist_ok=True)
        target = quarantine_root / f"{safe_uid}__{stamp}"
        shutil.move(str(directory), str(target))
        local_common.atomic_json(
            {
                "phase_uid": phase_uid,
                "reason": reason,
                "source_directory": str(directory),
                "quarantined_directory": str(target),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
            target / "QUARANTINE_REASON.json",
        )
        return target

    def run_once(args_inner, extractor, v3, plan, provenance, schema_path, mode):
        key = plan.frame_list_hash
        active_mode[key] = mode
        blocks = views_by_hash[key]
        counter = {"value": 0}

        def frozen_views(frames01, max_len=20):
            block_index = counter["value"]
            if block_index >= len(blocks):
                raise AssertionError("Local CAVE 请求的 block 多于 Whole metadata")
            block = blocks[block_index]
            counter["value"] += 1
            result = {}
            for name in ("uniform_full20", "contrast_core20"):
                positions = block.get("view_positions_local", {}).get(name)
                if positions is None:
                    absolute = block.get("view_indices", {}).get(name)
                    indices = block.get("indices")
                    if absolute is None or indices is None:
                        raise KeyError(f"Whole block 缺少 {name}")
                    lookup = {int(value): position for position, value in enumerate(indices)}
                    positions = [lookup[int(value)] for value in absolute]
                positions = np.asarray(positions, dtype=np.int64)
                if len(positions) and (positions.min() < 0 or positions.max() >= len(frames01)):
                    raise AssertionError(f"冻结时间索引越界：{name}")
                result[name] = positions
            return result

        cave.temporal_views = frozen_views if temporal_policy == "freeze_whole_indices" else original_temporal_views
        try:
            result = original_process(args_inner, extractor, v3, plan, provenance, schema_path)
        finally:
            cave.temporal_views = original_temporal_views
        if temporal_policy == "freeze_whole_indices" and counter["value"] != len(blocks):
            raise AssertionError(f"Local/Whole block 数不同：{counter['value']} vs {len(blocks)}")
        return result

    def roi_process(args_inner, extractor, v3, plan, provenance, schema_path):
        key = plan.frame_list_hash
        row = by_hash[key]

        # Resume must be handled before installing/counting temporal-view hooks.
        # A valid cache makes frozen CAVE return early, so the hook counter would
        # correctly remain zero; that is a cache skip, not a block mismatch.
        state, directory, reason = cache_state(args_inner, plan, row)
        overwrite = bool(getattr(args_inner, "overwrite", False))
        if state == "valid" and not overwrite:
            return {
                "status": "skipped",
                "patient_id": plan.patient_id,
                "series_uid": plan.series_uid,
                "phase": plan.phase,
                "skip_source": "worker_validated_resume_cache",
            }
        if state == "partial" and not overwrite:
            quarantine_partial(directory, row["phase_uid"], reason)

        enabled = bool(cfg.get("roi", {}).get("activity_fallback_enabled", True))
        mode_bbox = [
            ("primary", row["expanded_bbox"]),
            ("fallback", row["fallback_bbox"]),
            ("extended", row["extended_fallback_bbox"]),
        ]
        if not enabled:
            mode_bbox = mode_bbox[:1]
        modes: list[tuple[str, str]] = []
        for candidate in mode_bbox:
            if not modes or candidate[1] != modes[-1][1]:
                modes.append(candidate)

        result = None
        mode = "primary"
        for index, (mode, _) in enumerate(modes):
            try:
                result = run_once(
                    args_inner, extractor, v3, plan, provenance, schema_path, mode
                )
                break
            except (AssertionError, RuntimeError, ValueError) as exc:
                if not is_activity_error(exc) or index == len(modes) - 1:
                    raise
                directory = cave._phase_output_dir(args_inner.output_root, plan)
                if directory.exists() and not (directory / ".SUCCESS.json").is_file():
                    shutil.rmtree(directory)
        if result is None:
            raise RuntimeError("ROI fallback modes produced no result")

        directory = cave._phase_output_dir(args_inner.output_root, plan)
        used_bbox = dict(modes)[mode]
        metadata_path = directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["roi"] = {
            "pipeline_version": cfg["version"],
            "roi_pipeline_version": cfg["version"],
            "branch": roi_branch,
            "phase_uid": row["phase_uid"],
            "series_uid": row["series_uid"],
            "phase": row["phase"],
            "annotation_source": "png_2d_gt",
            "reference_image_path": row["reference_image_path"],
            "reference_sha256": row["reference_sha256"],
            "mask_path": row["mask_path"],
            "mask_sha256": row["mask_sha256"],
            "annotation_shape": row["annotation_shape"],
            "effective_mask_shape": row["effective_mask_shape"],
            "mask_resized_to_frame": local_common.as_bool(row["mask_resized_to_frame"]),
            "mask_resize_interpolation": row["mask_resize_interpolation"],
            "resize_scale_x": float(row["resize_scale_x"]),
            "resize_scale_y": float(row["resize_scale_y"]),
            "resize_uniformity_error": float(row["resize_uniformity_error"]),
            "effective_mask_array_sha256": row["effective_mask_array_sha256"],
            "storage_layout": row["storage_layout"],
            "mapping_method": row["mapping_method"],
            "orientation_transform": row["orientation_transform"],
            "orientation_status": row["orientation_status"],
            "target_rule": target_rule,
            "foreground_rule": "labels_in_1_2_equivalent_nonzero",
            "selected_labels": selected_labels,
            "labels_present": json.loads(row["labels_present"]),
            "labels_ignored": json.loads(row["labels_ignored"]),
            "selected_foreground_pixels": int(row["selected_foreground_pixels"]),
            "ignored_foreground_pixels": int(row["ignored_foreground_pixels"]),
            "original_bbox": row["original_bbox"],
            "primary_bbox": row["expanded_bbox"],
            "fallback_bbox": row["fallback_bbox"],
            "extended_fallback_bbox": row["extended_fallback_bbox"],
            "used_bbox": used_bbox,
            "fallback_used": mode != "primary",
            "fallback_level": mode,
            "temporal_policy": temporal_policy,
            "segmentation_model_used": False,
            "local_frames_saved": False,
        }
        local_common.atomic_json(metadata, metadata_path)
        success_path = directory / ".SUCCESS.json"
        success = json.loads(success_path.read_text(encoding="utf-8"))
        success.update({
            "phase_uid": row["phase_uid"],
            "annotation_source": "png_2d_gt",
            "roi_branch": roi_branch,
            "roi_pipeline_version": cfg["version"],
            "effective_mask_array_sha256": row["effective_mask_array_sha256"],
            "mask_resized_to_frame": local_common.as_bool(row["mask_resized_to_frame"]),
            "mask_resize_interpolation": row["mask_resize_interpolation"],
            "target_rule": target_rule,
            "foreground_rule": "labels_in_1_2_equivalent_nonzero",
            "selected_labels": selected_labels,
            "labels_present": json.loads(row["labels_present"]),
            "labels_ignored": json.loads(row["labels_ignored"]),
            "selected_foreground_pixels": int(row["selected_foreground_pixels"]),
            "ignored_foreground_pixels": int(row["ignored_foreground_pixels"]),
            "mask_sha256": row["mask_sha256"],
            "fallback_used": mode != "primary",
            "fallback_level": mode,
            "used_bbox": used_bbox,
            "temporal_policy": temporal_policy,
            "segmentation_model_used": False,
        })
        local_common.atomic_json(success, success_path)
        return result

    cave.load_gray_frames = roi_load
    cave.process_phase = roi_process
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)
    argv = [
        "extract_cave_featurebank.py",
        "--mode", "custom",
        "--manifest", str(manifest_path),
        "--cave-repo", cfg["cave_repo"],
        "--checkpoint", cfg["checkpoint"],
        "--v3-extractor", cfg["v3_extractor"],
        "--v3-base-config", cfg["v3_base_config"],
        "--v3-override-config", cfg["v3_override_config"],
        "--output-root", str(args.output_root),
        "--report-root", str(args.report_root),
        "--frozen-config", str(frozen),
        "--io-workers", str(args.io_workers),
    ]
    if args.max_series:
        argv += ["--max-series", str(args.max_series)]
    for uid in selected:
        argv += ["--series-uid", uid]
    if args.overwrite:
        argv.append("--overwrite")
    old_argv = sys.argv
    sys.argv = argv
    try:
        return int(cave.main())
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())

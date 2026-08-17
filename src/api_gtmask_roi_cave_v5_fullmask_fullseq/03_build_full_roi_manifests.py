#!/usr/bin/env python3
"""Build eligible-phase ROI manifests from full source-phase coverage.

Policy ``eligible_only``: every source phase is accounted for. A phase is
``local_eligible`` only when its own GT mask is mapped, readable, shape
compatible, non-empty, yields a legal ROI, and has Whole-CAVE temporal
metadata. Everything else is excluded with an explicit reason — no silent
drops, no mask reuse across phases/series, no automatic resize.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

from common import as_bool, atomic_csv, atomic_json, load_config, parse_pipe, sha256_file
from nifti_io import apply_orientation, load_label_mask, resize_labels
from roi import bbox_from_mask, bbox_to_text, context_square_bbox, crop_padding, mask_statistics

ORIENTATION_STATUS_BY_METHOD = {
    "upstream": "authoritative_upstream",
    "manual": "authoritative_manual",
    "path_exact_with_reference": "reference_verified",
    "reference_match": "reference_verified",
    "path_exact": "default_identity_unverified",
    "unique_series_for_phase": "default_identity_unverified",
    "unique_phase_for_series": "default_identity_unverified",
}


def read_shape(path: str) -> tuple[int, int]:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return int(image.shape[0]), int(image.shape[1])


def aspect_error(a: tuple[int, int], b: tuple[int, int]) -> float:
    ah, aw = a
    bh, bw = b
    ra = aw / max(ah, 1)
    rb = bw / max(bh, 1)
    return abs(ra - rb) / max(abs(rb), 1e-8)


def locate_whole_metadata(root: Path, split: str, patient_id: str, series_uid: str, phase: str) -> Path:
    candidates = [
        root / split.casefold() / patient_id / series_uid / phase / "metadata.json",
        root / split / patient_id / series_uid / phase / "metadata.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = list(root.glob(f"**/{patient_id}/{series_uid}/{phase}/metadata.json"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Whole-CAVE metadata 不存在或不唯一：{candidates}; glob={len(matches)}")


def base_coverage(mapped: dict[str, Any]) -> dict[str, Any]:
    mapping_status = str(mapped.get("phase_mapping_status", ""))
    mapping_method = str(mapped.get("mapping_method", ""))
    return {
        "phase_uid": mapped["phase_uid"],
        "split": mapped["split"],
        "source_series_order": mapped["source_series_order"],
        "source_phase_order": mapped["source_phase_order"],
        "patient_id": mapped["patient_id"],
        "series_uid": mapped["series_uid"],
        "phase": mapped["phase"],
        "frame_list_hash": mapped["frame_list_hash"],
        "mask_available": int(mapping_status == "accepted"),
        "mapping_status": mapping_status,
        "mapping_method": mapping_method,
        "mask_path": str(mapped.get("mask_path", "")),
        "mask_sha256": str(mapped.get("mask_sha256", "")),
        "orientation_transform": str(mapped.get("orientation_transform", "")),
        "orientation_status": ORIENTATION_STATUS_BY_METHOD.get(mapping_method, "none"),
        "mask_readable": 0,
        "shape_compatible": 0,
        "foreground_nonempty": 0,
        "roi_available": 0,
        "whole_metadata_available": 0,
        "local_eligible": 0,
        "local_exclusion_reason": "",
        "mask_shape": "",
        "frame_shape": "",
        "mask_resized_to_frame": 0,
    }


def evaluate_phase(mapped: dict[str, Any], roi_cfg: dict, whole_root: Path) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Evaluate one source phase; return (coverage, roi_row, morph_row, view_row)."""
    cov = base_coverage(mapped)
    if str(mapped.get("phase_mapping_status", "")) != "accepted":
        status = str(mapped.get("phase_mapping_status", "")) or "unknown"
        reason = {
            "missing": "no_mask_mapping",
            "needs_review": "mapping_needs_review",
            "conflict": "mapping_conflict",
        }.get(status, f"mapping_{status}")
        cov["local_exclusion_reason"] = reason
        return cov, None, None, None

    paths = parse_pipe(mapped["frame_paths"])
    if not paths:
        cov["local_exclusion_reason"] = "empty_frame_paths"
        return cov, None, None, None
    frame_shape = read_shape(paths[0])
    cov["frame_shape"] = f"{frame_shape[0]}x{frame_shape[1]}"

    try:
        labels, nifti_meta = load_label_mask(mapped["mask_path"])
    except Exception as exc:
        cov["local_exclusion_reason"] = f"mask_unreadable:{type(exc).__name__}:{exc}"
        return cov, None, None, None
    cov["mask_readable"] = 1

    orientation = str(mapped.get("orientation_transform", "identity") or "identity")
    labels = apply_orientation(labels, orientation)
    cov["mask_shape"] = f"{labels.shape[0]}x{labels.shape[1]}"
    resized = False
    if labels.shape != frame_shape:
        allow_resize = bool(roi_cfg.get("allow_mask_resize", False))
        error = aspect_error(labels.shape, frame_shape)
        if not allow_resize or error > float(roi_cfg.get("max_resize_aspect_ratio_error", 0.01)):
            cov["local_exclusion_reason"] = (
                f"shape_mismatch:mask={labels.shape[0]}x{labels.shape[1]},"
                f"frame={frame_shape[0]}x{frame_shape[1]},aspect_error={error:.6f}"
            )
            return cov, None, None, None
        labels = resize_labels(labels, frame_shape)
        resized = True
    cov["shape_compatible"] = 1
    cov["mask_resized_to_frame"] = int(resized)

    foreground = labels != 0
    if not bool(foreground.any()):
        cov["local_exclusion_reason"] = "empty_foreground"
        return cov, None, None, None
    cov["foreground_nonempty"] = 1

    try:
        object_bbox = bbox_from_mask(foreground)
        primary, primary_audit = context_square_bbox(
            object_bbox,
            frame_shape,
            bbox_factor=float(roi_cfg.get("bbox_factor", 1.5)),
            min_side_pixels=int(roi_cfg.get("min_side_pixels", 96)),
            min_margin_pixels=int(roi_cfg.get("min_margin_pixels", 8)),
            round_multiple=int(roi_cfg.get("round_multiple", 32)),
        )
        fallback, fallback_audit = context_square_bbox(
            object_bbox,
            frame_shape,
            bbox_factor=float(roi_cfg.get("fallback_bbox_factor", 2.0)),
            min_side_pixels=int(roi_cfg.get("fallback_min_side_pixels", 160)),
            min_margin_pixels=int(roi_cfg.get("min_margin_pixels", 8)),
            round_multiple=int(roi_cfg.get("round_multiple", 32)),
        )
    except Exception as exc:
        cov["local_exclusion_reason"] = f"roi_build_error:{type(exc).__name__}:{exc}"
        return cov, None, None, None
    cov["roi_available"] = 1
    stats = mask_statistics(labels, foreground)

    try:
        metadata_path = locate_whole_metadata(
            whole_root,
            str(mapped["split"]),
            str(mapped["patient_id"]),
            str(mapped["series_uid"]),
            str(mapped["phase"]),
        )
        whole_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        blocks = whole_metadata.get("blocks")
        if not isinstance(blocks, list) or not blocks:
            raise ValueError("missing blocks")
    except Exception as exc:
        cov["local_exclusion_reason"] = f"whole_metadata_missing:{type(exc).__name__}"
        return cov, None, None, None
    cov["whole_metadata_available"] = 1
    cov["local_eligible"] = 1

    roi_row = {
        "phase_uid": mapped["phase_uid"],
        "split": mapped["split"],
        "source_series_order": mapped["source_series_order"],
        "source_phase_order": mapped["source_phase_order"],
        "patient_id": mapped["patient_id"],
        "series_uid": mapped["series_uid"],
        "series_id": mapped.get("series_id", ""),
        "phase": mapped["phase"],
        "frame_paths": mapped["frame_paths"],
        "frame_list_hash": mapped["frame_list_hash"],
        "n_frames": len(paths),
        "frame_height": frame_shape[0],
        "frame_width": frame_shape[1],
        "mask_path": mapped["mask_path"],
        "mask_sha256": mapped["mask_sha256"],
        "reference_image_path": mapped.get("reference_image_path", ""),
        "reference_sha256": mapped.get("reference_sha256", ""),
        "storage_layout": mapped.get("storage_layout", ""),
        "mapping_method": mapped.get("mapping_method", ""),
        "mapping_priority": mapped.get("mapping_priority", ""),
        "mapping_score": mapped.get("match_score", ""),
        "mapping_margin": mapped.get("score_margin", ""),
        "orientation_transform": orientation,
        "orientation_status": cov["orientation_status"],
        "reference_plane": mapped.get("reference_plane", ""),
        "matched_frame_path": mapped.get("matched_frame_path", ""),
        "matched_frame_index": mapped.get("matched_frame_index", ""),
        "mask_resized_to_frame": resized,
        **nifti_meta,
        "target_rule": "all_nonzero_labels_for_roi_only",
        "original_bbox": bbox_to_text(object_bbox),
        "expanded_bbox": bbox_to_text(primary),
        "fallback_bbox": bbox_to_text(fallback),
        "padding_left": crop_padding(primary, frame_shape)[0],
        "padding_top": crop_padding(primary, frame_shape)[1],
        "padding_right": crop_padding(primary, frame_shape)[2],
        "padding_bottom": crop_padding(primary, frame_shape)[3],
        "roi_side": primary_audit["roi_side"],
        "roi_area_ratio": primary_audit["roi_area_ratio"],
        "fallback_roi_side": fallback_audit["roi_side"],
        "fallback_roi_area_ratio": fallback_audit["roi_area_ratio"],
        **stats,
        "whole_metadata_path": str(metadata_path),
        "whole_metadata_sha256": sha256_file(metadata_path),
    }
    morph_row = {
        key: roi_row[key]
        for key in (
            "phase_uid", "split", "patient_id", "series_uid", "phase", "mask_path", "mask_sha256",
            "mask_area_ratio", "bbox_width_ratio", "bbox_height_ratio", "bbox_aspect_ratio",
            "bbox_fill_ratio", "centroid_x_ratio", "centroid_y_ratio", "circularity", "solidity",
            "component_count", "largest_component_ratio", "positive_labels", "label_pixel_counts",
            "roi_area_ratio",
        )
    }
    view_row = {
        "phase_uid": mapped["phase_uid"],
        "split": mapped["split"],
        "patient_id": mapped["patient_id"],
        "series_uid": mapped["series_uid"],
        "phase": mapped["phase"],
        "frame_list_hash": mapped["frame_list_hash"],
        "blocks_json": json.dumps(blocks, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        "whole_metadata_path": str(metadata_path),
        "whole_metadata_sha256": sha256_file(metadata_path),
    }
    return cov, roi_row, morph_row, view_row


def build_eligible_cave_manifest(source: pd.DataFrame, eligible_keys: set[tuple[str, str]], version: str) -> pd.DataFrame:
    result = source.copy()
    keep = []
    for index, row in result.iterrows():
        uid = str(row["series_uid"])
        pre = as_bool(row.get("can_run_pre")) and (uid, "pre") in eligible_keys
        post = as_bool(row.get("can_run_post")) and (uid, "post") in eligible_keys
        for phase, runnable in (("pre", pre), ("post", post)):
            if runnable or not as_bool(row.get(f"can_run_{phase}")):
                continue
            # CAVE 校验：不可运行 phase 不得携带任何帧信息
            result.at[index, f"{phase}_frame_paths"] = ""
            result.at[index, f"{phase}_frame_list_hash"] = ""
            result.at[index, f"{phase}_frame_indices"] = ""
            result.at[index, f"{phase}_selected_filenames"] = ""
            if f"n_{phase}_frames" in result.columns:
                result.at[index, f"n_{phase}_frames"] = "0"
            if f"n_{phase}_contiguous_pairs" in result.columns:
                result.at[index, f"n_{phase}_contiguous_pairs"] = "0"
        result.at[index, "can_run_pre"] = "True" if pre else "False"
        result.at[index, "can_run_post"] = "True" if post else "False"
        result.at[index, "can_run_prepost"] = "True" if pre and post else "False"
        for col in ("candidate_valid", "selected_candidate", "selected_for_extraction"):
            if col in result.columns:
                result.at[index, col] = "True" if (pre or post) else "False"
        result.at[index, "local_pre_eligible"] = "True" if pre else "False"
        result.at[index, "local_post_eligible"] = "True" if post else "False"
        keep.append(pre or post)
    result["local_roi_pipeline_version"] = version
    selected = result.loc[keep].reset_index(drop=True)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifests = Path(cfg["paths"]["manifests"])
    reports = Path(cfg["paths"]["reports"])
    reports.mkdir(parents=True, exist_ok=True)

    phase_map = pd.read_csv(manifests / "source_phase_with_mask_map.csv", dtype=str, keep_default_na=False)
    phase_map["_so"] = phase_map["source_series_order"].astype(int)
    phase_map["_po"] = phase_map["source_phase_order"].astype(int)
    phase_map = (
        phase_map.sort_values(["split", "_so", "_po"], kind="stable")
        .drop(columns=["_so", "_po"])
        .reset_index(drop=True)
    )

    roi_cfg = cfg.get("roi", {})
    whole_root = Path(cfg["whole_featurebank"])

    coverage_rows: list[dict[str, Any]] = []
    roi_rows: list[dict[str, Any]] = []
    morphology_rows: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []

    for mapped in phase_map.to_dict("records"):
        cov, roi_row, morph_row, view_row = evaluate_phase(mapped, roi_cfg, whole_root)
        coverage_rows.append(cov)
        if roi_row is not None:
            roi_rows.append(roi_row)
            morphology_rows.append(morph_row)
            temporal_rows.append(view_row)

    coverage = pd.DataFrame(coverage_rows)
    roi = pd.DataFrame(roi_rows)
    if coverage.empty or roi.empty:
        raise RuntimeError("没有任何 eligible phase；检查映射与源数据")
    eligible_count = int(coverage["local_eligible"].sum())
    excluded_count = int((coverage["local_eligible"] == 0).sum())
    if eligible_count + excluded_count != len(coverage):
        raise AssertionError("coverage 计数不闭合")
    if eligible_count != len(roi):
        raise AssertionError(f"eligible 与 ROI 行数不一致：{eligible_count} vs {len(roi)}")
    no_reason = int(((coverage["local_eligible"] == 0) & (coverage["local_exclusion_reason"] == "")).sum())
    if no_reason:
        raise AssertionError(f"{no_reason} 个 excluded phase 没有原因")
    if roi["phase_uid"].duplicated().any() or roi["frame_list_hash"].duplicated().any():
        raise AssertionError("ROI manifest phase_uid/frame_list_hash 不唯一")

    atomic_csv(coverage, manifests / "local_phase_coverage_all.csv")
    atomic_csv(roi, manifests / "roi_phase_manifest_eligible.csv")
    atomic_csv(pd.DataFrame(morphology_rows), manifests / "mask_morphology_phase_eligible.csv")
    temporal = pd.DataFrame(temporal_rows).sort_values(["split", "series_uid", "phase"]).reset_index(drop=True)
    atomic_csv(temporal, manifests / "whole_temporal_views_eligible.csv")

    eligible_keys = {(str(row.series_uid), str(row.phase)) for row in roi.itertuples(index=False)}
    split_outputs: dict[str, Any] = {}
    for split in ("Train", "Valid"):
        source = pd.read_csv(cfg["source_series_manifests"][split], dtype=str, keep_default_na=False)
        local_manifest = build_eligible_cave_manifest(source, eligible_keys, cfg["version"])
        output_path = manifests / f"cave_manifest_local_{split.casefold()}_eligible.csv"
        atomic_csv(local_manifest, output_path)
        split_cov = coverage[coverage["split"] == split]
        split_roi = roi[roi["split"] == split]
        split_outputs[split] = {
            "source_series": int(len(source)),
            "extract_series": int(len(local_manifest)),
            "source_phases": int(len(split_cov)),
            "eligible_phases": int(len(split_roi)),
            "eligible_pre": int((split_roi["phase"] == "pre").sum()),
            "eligible_post": int((split_roi["phase"] == "post").sum()),
            "excluded_phases": int((split_cov["local_eligible"] == 0).sum()),
            "manifest": str(output_path),
            "manifest_sha256": sha256_file(output_path),
        }

    reason_counts = (
        coverage.loc[coverage["local_eligible"] == 0, "local_exclusion_reason"]
        .str.split(":", n=1).str[0].value_counts().to_dict()
    )
    exclusions = coverage[coverage["local_eligible"] == 0].copy()
    atomic_csv(exclusions, reports / "03_local_feature_exclusion.csv")

    roi_side = pd.to_numeric(roi["roi_side"])
    roi_ratio = pd.to_numeric(roi["roi_area_ratio"])
    lock = {
        "coverage_policy": "eligible_only",
        "version": cfg["version"],
        "source_phase_count": int(len(coverage)),
        "eligible_phase_count": eligible_count,
        "excluded_phase_count": excluded_count,
        "exclusion_reason_counts": reason_counts,
        "splits": split_outputs,
        "roi_manifest": str(manifests / "roi_phase_manifest_eligible.csv"),
        "roi_manifest_sha256": sha256_file(manifests / "roi_phase_manifest_eligible.csv"),
        "temporal_views_sha256": sha256_file(manifests / "whole_temporal_views_eligible.csv"),
        "roi_side_stats": {
            "min": float(roi_side.min()), "median": float(roi_side.median()), "max": float(roi_side.max()),
        },
        "roi_area_ratio_stats": {
            "min": float(roi_ratio.min()), "median": float(roi_ratio.median()), "max": float(roi_ratio.max()),
        },
        "roi_rule": {
            "foreground": "segmentation != 0",
            "bbox_factor": float(roi_cfg.get("bbox_factor", 1.5)),
            "min_side_pixels": int(roi_cfg.get("min_side_pixels", 96)),
            "min_margin_pixels": int(roi_cfg.get("min_margin_pixels", 8)),
            "round_multiple": int(roi_cfg.get("round_multiple", 32)),
            "allow_mask_resize": bool(roi_cfg.get("allow_mask_resize", False)),
            "local_frames_saved": False,
            "crop_mode": "on_the_fly_in_memory",
        },
    }
    atomic_json(lock, manifests / "local_eligible_manifest_lock.json")
    atomic_json(lock, reports / "03_eligible_roi_summary.json")
    print(json.dumps(lock, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

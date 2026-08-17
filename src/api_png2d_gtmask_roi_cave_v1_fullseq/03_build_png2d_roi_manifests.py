#!/usr/bin/env python3
"""Build Local-CAVE ROI manifests from authoritative paired 2-D PNG GT masks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from common import as_bool, atomic_csv, atomic_json, load_config, parse_pipe, sha256_file
from roi import bbox_from_mask, bbox_to_text, context_square_bbox, crop_padding, mask_statistics


def read_gray(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.ndim != 2:
        raise ValueError(f"unreadable 2-D grayscale image: {path}")
    return image


def shape_text(shape: tuple[int, int]) -> str:
    return f"{int(shape[0])}x{int(shape[1])}"


def aspect_error(left: tuple[int, int], right: tuple[int, int]) -> float:
    left_ratio = float(left[1]) / max(float(left[0]), 1.0)
    right_ratio = float(right[1]) / max(float(right[0]), 1.0)
    return abs(left_ratio - right_ratio) / max(abs(right_ratio), 1e-12)


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def locate_whole_metadata(
    root: Path, split: str, patient_id: str, series_uid: str, phase: str
) -> Path:
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
    raise FileNotFoundError(
        f"Whole-CAVE metadata missing/not unique: {series_uid}/{phase}, matches={len(matches)}"
    )


def base_coverage(mapped: dict[str, Any]) -> dict[str, Any]:
    status = str(mapped.get("phase_mapping_status", ""))
    return {
        "phase_uid": mapped["phase_uid"],
        "split": mapped["split"],
        "source_series_order": mapped["source_series_order"],
        "source_phase_order": mapped["source_phase_order"],
        "patient_id": mapped["patient_id"],
        "series_uid": mapped["series_uid"],
        "phase": mapped["phase"],
        "frame_list_hash": mapped["frame_list_hash"],
        "annotation_source": "png_2d_gt",
        "segmentation_model_used": 0,
        "mask_available": int(status == "accepted"),
        "mapping_status": status,
        "mapping_method": str(mapped.get("mapping_method", "")),
        "mapping_reason": str(mapped.get("mapping_reason", "")),
        "png_key": str(mapped.get("png_key", "")),
        "reference_image_path": str(mapped.get("reference_image_path", "")),
        "reference_sha256": str(mapped.get("reference_sha256", "")),
        "mask_path": str(mapped.get("mask_path", "")),
        "mask_sha256": str(mapped.get("mask_sha256", "")),
        "identity_pearson_correlation": str(mapped.get("identity_pearson_correlation", "")),
        "orientation_transform": str(mapped.get("orientation_transform", "")),
        "orientation_status": str(mapped.get("orientation_status", "")),
        "mask_readable": 0,
        "reference_readable": 0,
        "shape_compatible": 0,
        "foreground_nonempty": 0,
        "roi_available": 0,
        "whole_metadata_available": 0,
        "local_eligible": 0,
        "local_exclusion_reason": "",
        "annotation_shape": "",
        "frame_shape": "",
        "effective_mask_shape": "",
        "mask_resized_to_frame": 0,
        "mask_resize_interpolation": "none",
        "resize_scale_x": "",
        "resize_scale_y": "",
        "resize_uniformity_error": "",
        "resize_aspect_ratio_error": "",
        "effective_mask_array_sha256": "",
        "foreground_rule": "labels_in_1_2_equivalent_nonzero",
        "selected_labels": "[1,2]",
        "labels_present_original": "[]",
        "labels_present_effective": "[]",
        "labels_ignored": "[]",
        "selected_foreground_pixels_original": 0,
        "selected_foreground_pixels_effective": 0,
    }


def effective_mask(
    labels: np.ndarray,
    frame_shape: tuple[int, int],
    roi_cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    source_shape = tuple(map(int, labels.shape))
    scale_x = frame_shape[1] / max(source_shape[1], 1)
    scale_y = frame_shape[0] / max(source_shape[0], 1)
    uniformity = abs(scale_x - scale_y) / max(abs(scale_x), abs(scale_y), 1e-12)
    aspect = aspect_error(source_shape, frame_shape)
    resized = source_shape != frame_shape
    audit = {
        "mask_resized_to_frame": int(resized),
        "mask_resize_interpolation": "nearest" if resized else "none",
        "resize_scale_x": float(scale_x),
        "resize_scale_y": float(scale_y),
        "resize_uniformity_error": float(uniformity),
        "resize_aspect_ratio_error": float(aspect),
    }
    if not resized:
        return labels.copy(), audit
    if not bool(roi_cfg.get("allow_mask_resize", False)):
        raise ValueError("mask_resize_disabled")
    if str(roi_cfg.get("resize_interpolation", "")).casefold() != "nearest":
        raise ValueError("mask_resize_interpolation_must_be_nearest")
    maximum = float(roi_cfg.get("max_resize_aspect_ratio_error", 0.001))
    if aspect > maximum or (
        bool(roi_cfg.get("require_uniform_xy_scale", True)) and uniformity > maximum
    ):
        raise ValueError(
            f"nonuniform_scale_not_allowed:mask={shape_text(source_shape)},"
            f"frame={shape_text(frame_shape)},aspect_error={aspect:.6f},"
            f"uniformity_error={uniformity:.6f}"
        )
    resized_labels = cv2.resize(
        labels,
        (frame_shape[1], frame_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized_labels.astype(labels.dtype, copy=False), audit


def evaluate_phase(
    mapped: dict[str, Any],
    roi_cfg: dict[str, Any],
    whole_root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    cov = base_coverage(mapped)
    status = str(mapped.get("phase_mapping_status", ""))
    if status != "accepted":
        cov["local_exclusion_reason"] = (
            "mapping_needs_review"
            if status == "needs_review"
            else "no_png2d_mask_mapping"
        )
        return cov, None, None, None

    paths = parse_pipe(mapped.get("frame_paths", ""))
    if not paths:
        cov["local_exclusion_reason"] = "empty_frame_paths"
        return cov, None, None, None
    try:
        first_frame = read_gray(paths[0])
        frame_shape = tuple(map(int, first_frame.shape))
    except Exception as exc:
        cov["local_exclusion_reason"] = f"frame_unreadable:{type(exc).__name__}:{exc}"
        return cov, None, None, None
    cov["frame_shape"] = shape_text(frame_shape)

    try:
        reference = read_gray(mapped["reference_image_path"])
        labels = read_gray(mapped["mask_path"])
    except Exception as exc:
        cov["local_exclusion_reason"] = f"png_unreadable:{type(exc).__name__}:{exc}"
        return cov, None, None, None
    cov["mask_readable"] = 1
    cov["reference_readable"] = 1
    cov["annotation_shape"] = shape_text(labels.shape)
    if reference.shape != labels.shape:
        cov["local_exclusion_reason"] = (
            f"reference_mask_shape_mismatch:{reference.shape}!={labels.shape}"
        )
        return cov, None, None, None

    allowed = {0, 1, 2}
    original_values = sorted(int(value) for value in np.unique(labels))
    if not set(original_values).issubset(allowed):
        cov["local_exclusion_reason"] = f"unexpected_png_mask_labels:{original_values}"
        return cov, None, None, None
    if sha256_file(mapped["mask_path"]) != str(mapped["mask_sha256"]):
        cov["local_exclusion_reason"] = "mask_sha256_changed"
        return cov, None, None, None
    if sha256_file(mapped["reference_image_path"]) != str(mapped["reference_sha256"]):
        cov["local_exclusion_reason"] = "reference_sha256_changed"
        return cov, None, None, None

    try:
        labels_frame, resize_audit = effective_mask(labels, frame_shape, roi_cfg)
    except Exception as exc:
        cov["local_exclusion_reason"] = str(exc)
        return cov, None, None, None
    cov.update(resize_audit)
    cov["shape_compatible"] = 1
    cov["effective_mask_shape"] = shape_text(labels_frame.shape)
    cov["effective_mask_array_sha256"] = array_sha256(labels_frame)

    effective_values = sorted(int(value) for value in np.unique(labels_frame))
    if not set(effective_values).issubset(allowed):
        cov["local_exclusion_reason"] = f"resized_mask_label_corruption:{effective_values}"
        cov["shape_compatible"] = 0
        return cov, None, None, None

    selected_labels = [int(value) for value in roi_cfg.get("foreground_labels", [])]
    if selected_labels != [1, 2]:
        raise ValueError(f"PNG2D pipeline requires foreground_labels=[1,2], got {selected_labels}")
    foreground_original = np.isin(labels, selected_labels)
    foreground = np.isin(labels_frame, selected_labels)
    labels_present_original = [value for value in original_values if value != 0]
    labels_present_effective = [value for value in effective_values if value != 0]
    labels_ignored = [
        value for value in labels_present_effective if value not in set(selected_labels)
    ]
    cov.update({
        "labels_present_original": json.dumps(labels_present_original, separators=(",", ":")),
        "labels_present_effective": json.dumps(labels_present_effective, separators=(",", ":")),
        "labels_ignored": json.dumps(labels_ignored, separators=(",", ":")),
        "selected_foreground_pixels_original": int(foreground_original.sum()),
        "selected_foreground_pixels_effective": int(foreground.sum()),
    })
    if not foreground.any():
        cov["local_exclusion_reason"] = "empty_png2d_gt_foreground_after_mapping"
        return cov, None, None, None
    if labels_ignored:
        cov["local_exclusion_reason"] = f"unexpected_ignored_labels:{labels_ignored}"
        return cov, None, None, None
    cov["foreground_nonempty"] = 1

    try:
        tight = bbox_from_mask(foreground)
        primary, primary_audit = context_square_bbox(
            tight,
            frame_shape,
            bbox_factor=float(roi_cfg["bbox_factor"]),
            min_side_pixels=int(roi_cfg["min_side_pixels"]),
            min_margin_pixels=int(roi_cfg["min_margin_pixels"]),
            round_multiple=int(roi_cfg["round_multiple"]),
        )
        fallback, fallback_audit = context_square_bbox(
            tight,
            frame_shape,
            bbox_factor=float(roi_cfg["fallback_bbox_factor"]),
            min_side_pixels=int(roi_cfg["fallback_min_side_pixels"]),
            min_margin_pixels=int(roi_cfg["min_margin_pixels"]),
            round_multiple=int(roi_cfg["round_multiple"]),
        )
        extended_fallback, extended_fallback_audit = context_square_bbox(
            tight,
            frame_shape,
            bbox_factor=float(roi_cfg["extended_fallback_bbox_factor"]),
            min_side_pixels=int(roi_cfg["extended_fallback_min_side_pixels"]),
            min_margin_pixels=int(roi_cfg["min_margin_pixels"]),
            round_multiple=int(roi_cfg["round_multiple"]),
        )
    except Exception as exc:
        cov["local_exclusion_reason"] = f"roi_build_error:{type(exc).__name__}:{exc}"
        return cov, None, None, None
    cov["roi_available"] = 1

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
            raise ValueError("missing Whole temporal blocks")
    except Exception as exc:
        cov["local_exclusion_reason"] = f"whole_metadata_missing:{type(exc).__name__}:{exc}"
        return cov, None, None, None
    cov["whole_metadata_available"] = 1
    cov["local_eligible"] = 1

    stats = mask_statistics(labels_frame, foreground)
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
        "annotation_source": "png_2d_gt",
        "segmentation_model_used": 0,
        "png_key": mapped["png_key"],
        "reference_image_path": mapped["reference_image_path"],
        "reference_sha256": mapped["reference_sha256"],
        "mask_path": mapped["mask_path"],
        "mask_sha256": mapped["mask_sha256"],
        "storage_layout": "paired_png_2d",
        "mapping_method": mapped["mapping_method"],
        "mapping_score": mapped["identity_pearson_correlation"],
        "orientation_transform": "identity",
        "orientation_status": mapped["orientation_status"],
        "annotation_shape": shape_text(labels.shape),
        "effective_mask_shape": shape_text(labels_frame.shape),
        **resize_audit,
        "effective_mask_array_sha256": array_sha256(labels_frame),
        "target_rule": str(roi_cfg["target_rule"]),
        "foreground_rule": "labels_in_1_2_equivalent_nonzero",
        "selected_labels": "[1,2]",
        "labels_present_original": json.dumps(labels_present_original, separators=(",", ":")),
        "labels_present_effective": json.dumps(labels_present_effective, separators=(",", ":")),
        "labels_present": json.dumps(labels_present_effective, separators=(",", ":")),
        "labels_ignored": "[]",
        "selected_foreground_pixels_original": int(foreground_original.sum()),
        "selected_foreground_pixels_effective": int(foreground.sum()),
        "selected_foreground_pixels": int(foreground.sum()),
        "ignored_foreground_pixels": 0,
        "original_bbox": bbox_to_text(tight),
        "expanded_bbox": bbox_to_text(primary),
        "fallback_bbox": bbox_to_text(fallback),
        "extended_fallback_bbox": bbox_to_text(extended_fallback),
        "padding_left": crop_padding(primary, frame_shape)[0],
        "padding_top": crop_padding(primary, frame_shape)[1],
        "padding_right": crop_padding(primary, frame_shape)[2],
        "padding_bottom": crop_padding(primary, frame_shape)[3],
        "roi_side": primary_audit["roi_side"],
        "roi_area_ratio": primary_audit["roi_area_ratio"],
        "fallback_roi_side": fallback_audit["roi_side"],
        "fallback_roi_area_ratio": fallback_audit["roi_area_ratio"],
        "extended_fallback_roi_side": extended_fallback_audit["roi_side"],
        "extended_fallback_roi_area_ratio": extended_fallback_audit["roi_area_ratio"],
        **stats,
        "whole_metadata_path": str(metadata_path),
        "whole_metadata_sha256": sha256_file(metadata_path),
    }
    morphology = {
        key: roi_row[key]
        for key in (
            "phase_uid", "split", "patient_id", "series_uid", "phase",
            "annotation_source", "mask_path", "mask_sha256",
            "annotation_shape", "effective_mask_shape", "mask_resized_to_frame",
            "resize_scale_x", "resize_scale_y", "resize_uniformity_error",
            "effective_mask_array_sha256", "mask_area_ratio",
            "bbox_width_ratio", "bbox_height_ratio", "bbox_aspect_ratio",
            "bbox_fill_ratio", "centroid_x_ratio", "centroid_y_ratio",
            "circularity", "solidity", "component_count",
            "largest_component_ratio", "positive_labels", "label_pixel_counts",
            "foreground_rule", "selected_labels", "labels_present",
            "selected_foreground_pixels", "roi_area_ratio",
        )
    }
    temporal = {
        "phase_uid": mapped["phase_uid"],
        "split": mapped["split"],
        "patient_id": mapped["patient_id"],
        "series_uid": mapped["series_uid"],
        "phase": mapped["phase"],
        "frame_list_hash": mapped["frame_list_hash"],
        "blocks_json": json.dumps(
            blocks, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
        "whole_metadata_path": str(metadata_path),
        "whole_metadata_sha256": sha256_file(metadata_path),
    }
    return cov, roi_row, morphology, temporal


def build_local_manifest(
    source: pd.DataFrame,
    eligible_keys: set[tuple[str, str]],
    version: str,
) -> pd.DataFrame:
    result = source.copy()
    keep: list[bool] = []
    for index, row in result.iterrows():
        uid = str(row["series_uid"])
        pre = as_bool(row.get("can_run_pre")) and (uid, "pre") in eligible_keys
        post = as_bool(row.get("can_run_post")) and (uid, "post") in eligible_keys
        for phase, runnable in (("pre", pre), ("post", post)):
            if runnable or not as_bool(row.get(f"can_run_{phase}")):
                continue
            for suffix in (
                "frame_paths", "frame_list_hash", "frame_indices", "selected_filenames"
            ):
                column = f"{phase}_{suffix}"
                if column in result.columns:
                    result.at[index, column] = ""
            for column in (f"n_{phase}_frames", f"n_{phase}_contiguous_pairs"):
                if column in result.columns:
                    result.at[index, column] = "0"
        result.at[index, "can_run_pre"] = "True" if pre else "False"
        result.at[index, "can_run_post"] = "True" if post else "False"
        result.at[index, "can_run_prepost"] = "True" if pre and post else "False"
        for column in ("candidate_valid", "selected_candidate", "selected_for_extraction"):
            if column in result.columns:
                result.at[index, column] = "True" if (pre or post) else "False"
        result.at[index, "local_pre_eligible"] = "True" if pre else "False"
        result.at[index, "local_post_eligible"] = "True" if post else "False"
        keep.append(pre or post)
    result["local_roi_pipeline_version"] = version
    return result.loc[keep].reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifests = Path(cfg["paths"]["manifests"])
    reports = Path(cfg["paths"]["reports"])
    reports.mkdir(parents=True, exist_ok=True)
    phase_map = pd.read_csv(
        manifests / "source_phase_with_mask_map.csv",
        dtype=str,
        keep_default_na=False,
    )
    phase_map["_series_order"] = pd.to_numeric(phase_map["source_series_order"])
    phase_map["_phase_order"] = pd.to_numeric(phase_map["source_phase_order"])
    phase_map = phase_map.sort_values(
        ["split", "_series_order", "_phase_order"], kind="stable"
    ).drop(columns=["_series_order", "_phase_order"]).reset_index(drop=True)

    roi_cfg = cfg["roi"]
    whole_root = Path(cfg["whole_featurebank"])
    coverage_rows: list[dict[str, Any]] = []
    roi_rows: list[dict[str, Any]] = []
    morphology_rows: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []
    for mapped in phase_map.to_dict("records"):
        coverage, roi_row, morphology, temporal = evaluate_phase(
            mapped, roi_cfg, whole_root
        )
        coverage_rows.append(coverage)
        if roi_row is not None:
            roi_rows.append(roi_row)
            morphology_rows.append(morphology)
            temporal_rows.append(temporal)

    coverage = pd.DataFrame(coverage_rows)
    roi = pd.DataFrame(roi_rows)
    morphology = pd.DataFrame(morphology_rows)
    temporal = pd.DataFrame(temporal_rows)
    eligible = coverage[coverage["local_eligible"] == 1]
    excluded = coverage[coverage["local_eligible"] == 0]
    if len(coverage) != len(phase_map):
        raise AssertionError("coverage does not close over source phase map")
    if (excluded["local_exclusion_reason"] == "").any():
        raise AssertionError("excluded phase without reason")
    if len(eligible) != len(roi):
        raise AssertionError("eligible/ROI row mismatch")
    if roi["phase_uid"].duplicated().any() or roi["frame_list_hash"].duplicated().any():
        raise AssertionError("ROI phase_uid/frame_list_hash is not unique")

    expected = cfg["expected_png2d_stage1"]
    if len(coverage) != int(expected["source_phases"]):
        raise AssertionError(f"source phase count={len(coverage)}")
    if len(roi) != int(expected["eligible_phases"]):
        raise AssertionError(f"eligible phase count={len(roi)}")

    atomic_csv(coverage, manifests / "local_phase_coverage_all.csv")
    atomic_csv(coverage, manifests / "png2d_phase_coverage_all.csv")
    atomic_csv(roi, manifests / "roi_phase_manifest_eligible.csv")
    atomic_csv(morphology, manifests / "mask_morphology_phase_eligible.csv")
    temporal = temporal.sort_values(
        ["split", "series_uid", "phase"], kind="stable"
    ).reset_index(drop=True)
    atomic_csv(temporal, manifests / "whole_temporal_views_eligible.csv")
    atomic_csv(excluded, reports / "03_local_feature_exclusion.csv")

    eligible_keys = {
        (str(row.series_uid), str(row.phase))
        for row in roi.itertuples(index=False)
    }
    split_outputs: dict[str, Any] = {}
    strict_counts: dict[str, int] = {}
    for split in ("Train", "Valid"):
        source = pd.read_csv(
            cfg["source_series_manifests"][split],
            dtype=str,
            keep_default_na=False,
        )
        local = build_local_manifest(source, eligible_keys, cfg["version"])
        output_path = (
            manifests / f"cave_manifest_local_{split.casefold()}_eligible.csv"
        )
        atomic_csv(local, output_path)
        split_roi = roi[roi["split"] == split]
        strict = int(
            (
                local["local_pre_eligible"].map(as_bool)
                & local["local_post_eligible"].map(as_bool)
            ).sum()
        )
        strict_counts[split] = strict
        split_outputs[split] = {
            "source_series": len(source),
            "extract_series": len(local),
            "eligible_phases": len(split_roi),
            "eligible_pre": int((split_roi["phase"] == "pre").sum()),
            "eligible_post": int((split_roi["phase"] == "post").sum()),
            "strict_prepost_series": strict,
            "manifest": str(output_path),
            "manifest_sha256": sha256_file(output_path),
        }
    if split_outputs["Train"]["eligible_phases"] != int(expected["train_eligible_phases"]):
        raise AssertionError("Train eligible phase count changed")
    if split_outputs["Valid"]["eligible_phases"] != int(expected["valid_eligible_phases"]):
        raise AssertionError("Valid eligible phase count changed")
    if strict_counts["Train"] != int(expected["strict_prepost_series_train"]):
        raise AssertionError("Train strict Pre/Post count changed")
    if strict_counts["Valid"] != int(expected["strict_prepost_series_valid"]):
        raise AssertionError("Valid strict Pre/Post count changed")

    reason_counts = (
        excluded["local_exclusion_reason"].astype(str)
        .str.split(":", n=1).str[0].value_counts().to_dict()
    )
    roi_side = pd.to_numeric(roi["roi_side"])
    roi_ratio = pd.to_numeric(roi["roi_area_ratio"])
    resized = roi["mask_resized_to_frame"].map(as_bool)
    summary = {
        "status": "success",
        "version": cfg["version"],
        "annotation_source": "paired_png_2d_mean_and_gt_mask",
        "segmentation_model_used": False,
        "coverage_policy": "eligible_only",
        "source_phase_count": len(coverage),
        "mapped_phase_count": int(coverage["mask_available"].sum()),
        "eligible_phase_count": len(roi),
        "excluded_phase_count": len(excluded),
        "exclusion_reason_counts": reason_counts,
        "mask_resized_to_frame_count": int(resized.sum()),
        "mask_exact_shape_count": int((~resized).sum()),
        "resize_interpolation": "nearest",
        "nonuniform_scale_policy": "exclude",
        "foreground_rule": "labels_in_1_2_equivalent_nonzero",
        "selected_labels": [1, 2],
        "splits": split_outputs,
        "roi_side_stats": {
            "min": float(roi_side.min()),
            "median": float(roi_side.median()),
            "max": float(roi_side.max()),
        },
        "roi_area_ratio_stats": {
            "min": float(roi_ratio.min()),
            "median": float(roi_ratio.median()),
            "max": float(roi_ratio.max()),
        },
        "roi_manifest": str(manifests / "roi_phase_manifest_eligible.csv"),
        "roi_manifest_sha256": sha256_file(
            manifests / "roi_phase_manifest_eligible.csv"
        ),
        "temporal_views_sha256": sha256_file(
            manifests / "whole_temporal_views_eligible.csv"
        ),
        "local_frames_saved": False,
        "crop_mode": "on_the_fly_in_memory",
    }
    atomic_json(summary, manifests / "local_eligible_manifest_lock.json")
    atomic_json(summary, reports / "03_png2d_roi_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


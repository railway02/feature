#!/usr/bin/env python3
"""Validate complete PNG2D stage-1 coverage and write the extraction gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import as_bool, atomic_json, load_config, parse_pipe, sha256_file, write_success
from roi import bbox_from_text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifests = Path(cfg["paths"]["manifests"])
    reports = Path(cfg["paths"]["reports"])
    failures: list[str] = []

    required_files = [
        manifests / "source_manifest_lock.json",
        manifests / "png2d_input_lock.json",
        manifests / "source_phase_index_all.csv",
        manifests / "source_phase_with_mask_map.csv",
        manifests / "local_phase_coverage_all.csv",
        manifests / "png2d_phase_coverage_all.csv",
        manifests / "roi_phase_manifest_eligible.csv",
        manifests / "mask_morphology_phase_eligible.csv",
        manifests / "whole_temporal_views_eligible.csv",
        manifests / "cave_manifest_local_train_eligible.csv",
        manifests / "cave_manifest_local_valid_eligible.csv",
        reports / "04_roi_qa_summary.json",
    ]
    for path in required_files:
        if not path.is_file():
            failures.append(f"missing stage1 file: {path}")

    if failures:
        summary = {"status": "failed", "failures": failures}
        atomic_json(summary, reports / "05_png2d_stage1_validation.json")
        raise AssertionError("; ".join(failures))

    input_lock = json.loads(
        (manifests / "png2d_input_lock.json").read_text(encoding="utf-8")
    )
    if input_lock.get("status") != "success":
        failures.append("PNG2D input audit lock is not success")
    qa = json.loads((reports / "04_roi_qa_summary.json").read_text(encoding="utf-8"))
    if qa.get("status") != "success" or int(qa.get("qa_failures", -1)) != 0:
        failures.append("PNG2D ROI QA did not close")

    source = pd.read_csv(
        manifests / "source_phase_index_all.csv", dtype=str, keep_default_na=False
    )
    phase_map = pd.read_csv(
        manifests / "source_phase_with_mask_map.csv", dtype=str, keep_default_na=False
    )
    coverage = pd.read_csv(
        manifests / "local_phase_coverage_all.csv", dtype=str, keep_default_na=False
    )
    canonical = pd.read_csv(
        manifests / "png2d_phase_coverage_all.csv", dtype=str, keep_default_na=False
    )
    roi = pd.read_csv(
        manifests / "roi_phase_manifest_eligible.csv", dtype=str, keep_default_na=False
    )
    temporal = pd.read_csv(
        manifests / "whole_temporal_views_eligible.csv", dtype=str, keep_default_na=False
    )
    expected = cfg["expected_png2d_stage1"]

    expected_uids = source["phase_uid"].astype(str).tolist()
    for name, frame in (("phase_map", phase_map), ("coverage", coverage), ("canonical", canonical)):
        if frame["phase_uid"].astype(str).tolist() != expected_uids:
            failures.append(f"{name} changed source phase order/coverage")
    if sha256_file(manifests / "local_phase_coverage_all.csv") != sha256_file(
        manifests / "png2d_phase_coverage_all.csv"
    ):
        failures.append("canonical PNG2D coverage differs from local coverage")

    eligible = coverage[coverage["local_eligible"].map(as_bool)]
    excluded = coverage[~coverage["local_eligible"].map(as_bool)]
    if len(source) != int(expected["source_phases"]):
        failures.append(f"source phases={len(source)}")
    if len(eligible) != int(expected["eligible_phases"]):
        failures.append(f"eligible phases={len(eligible)}")
    if len(excluded) + len(eligible) != len(source):
        failures.append("eligible/excluded coverage does not close")
    if (excluded["local_exclusion_reason"].astype(str) == "").any():
        failures.append("excluded phase without reason")
    if set(eligible["phase_uid"]) != set(roi["phase_uid"]):
        failures.append("eligible/ROI phase sets differ")
    if set(eligible["phase_uid"]) != set(temporal["phase_uid"]):
        failures.append("eligible/temporal phase sets differ")
    if roi["phase_uid"].duplicated().any() or roi["frame_list_hash"].duplicated().any():
        failures.append("ROI phase_uid/frame_list_hash not unique")

    required_roi_columns = {
        "annotation_source", "segmentation_model_used", "reference_image_path",
        "reference_sha256", "mask_path", "mask_sha256", "annotation_shape",
        "effective_mask_shape", "mask_resized_to_frame",
        "mask_resize_interpolation", "resize_scale_x", "resize_scale_y",
        "resize_uniformity_error", "resize_aspect_ratio_error",
        "effective_mask_array_sha256", "foreground_rule", "selected_labels",
        "labels_present", "labels_ignored", "selected_foreground_pixels",
        "original_bbox", "expanded_bbox", "fallback_bbox",
        "extended_fallback_bbox",
    }
    missing_columns = sorted(required_roi_columns - set(roi.columns))
    if missing_columns:
        failures.append(f"ROI missing provenance columns: {missing_columns}")
    if not roi["annotation_source"].eq("png_2d_gt").all():
        failures.append("ROI annotation_source is not png_2d_gt")
    if roi["segmentation_model_used"].map(as_bool).any():
        failures.append("ROI unexpectedly declares a segmentation model")
    if not roi["orientation_transform"].eq("identity").all():
        failures.append("eligible ROI has non-identity orientation")
    if not roi["orientation_status"].eq("identity_mean_verified").all():
        failures.append("eligible ROI orientation is not mean-verified")
    if not roi["foreground_rule"].eq("labels_in_1_2_equivalent_nonzero").all():
        failures.append("ROI foreground rule mismatch")
    if not roi["selected_labels"].eq("[1,2]").all():
        failures.append("ROI selected labels mismatch")
    if not roi["labels_ignored"].eq("[]").all():
        failures.append("ROI unexpectedly ignores PNG labels")
    if (pd.to_numeric(roi["selected_foreground_pixels"]) <= 0).any():
        failures.append("eligible ROI has empty foreground")
    resized = roi["mask_resized_to_frame"].map(as_bool)
    if int(resized.sum()) != int(expected["eligible_nearest_resized"]):
        failures.append(f"nearest-resized count={int(resized.sum())}")
    if int((~resized).sum()) != int(expected["eligible_exact_shape"]):
        failures.append(f"exact-shape count={int((~resized).sum())}")
    if not roi.loc[resized, "mask_resize_interpolation"].eq("nearest").all():
        failures.append("resized mask did not use nearest interpolation")
    if not roi.loc[~resized, "mask_resize_interpolation"].eq("none").all():
        failures.append("exact mask unexpectedly declares interpolation")
    maximum = float(cfg["roi"]["max_resize_aspect_ratio_error"])
    if (pd.to_numeric(roi["resize_aspect_ratio_error"]) > maximum + 1e-12).any():
        failures.append("eligible ROI exceeds aspect-ratio resize limit")
    if (pd.to_numeric(roi["resize_uniformity_error"]) > maximum + 1e-12).any():
        failures.append("eligible ROI exceeds uniform-scale limit")

    for row in roi.to_dict("records"):
        tight = bbox_from_text(row["original_bbox"])
        primary = bbox_from_text(row["expanded_bbox"])
        fallback = bbox_from_text(row["fallback_bbox"])
        extended = bbox_from_text(row["extended_fallback_bbox"])
        for name, box in (
            ("primary", primary),
            ("fallback", fallback),
            ("extended", extended),
        ):
            if box[0] > tight[0] or box[1] > tight[1] or box[2] < tight[2] or box[3] < tight[3]:
                failures.append(f"{row['phase_uid']}: {name} bbox does not contain GT")
        paths = parse_pipe(row["frame_paths"])
        if len(paths) != int(float(row["n_frames"])):
            failures.append(f"{row['phase_uid']}: frame count mismatch")
        for key in ("reference_image_path", "mask_path", "whole_metadata_path"):
            if not Path(row[key]).is_file():
                failures.append(f"{row['phase_uid']}: missing {key}")
        if Path(row["mask_path"]).is_file() and sha256_file(row["mask_path"]) != row["mask_sha256"]:
            failures.append(f"{row['phase_uid']}: mask SHA changed")
        if Path(row["reference_image_path"]).is_file() and sha256_file(
            row["reference_image_path"]
        ) != row["reference_sha256"]:
            failures.append(f"{row['phase_uid']}: reference SHA changed")

    reason_prefix = (
        excluded["local_exclusion_reason"].str.split(":", n=1).str[0].value_counts().to_dict()
    )
    nonuniform = sum(
        str(value).startswith("nonuniform_scale_not_allowed")
        for value in excluded["local_exclusion_reason"]
    )
    if nonuniform != int(expected["excluded_nonuniform_scale"]):
        failures.append(f"nonuniform-scale exclusions={nonuniform}")
    bad_uid = "Train__631324__main__b28b7af693::pre"
    bad = coverage[coverage["phase_uid"] == bad_uid]
    if len(bad) != 1 or not str(bad.iloc[0]["local_exclusion_reason"]).startswith(
        "nonuniform_scale_not_allowed"
    ):
        failures.append("631324_Pre is not explicitly excluded for nonuniform scale")

    split_summary = {}
    eligible_keys = set(zip(eligible["series_uid"], eligible["phase"]))
    for split in ("Train", "Valid"):
        source_series = pd.read_csv(
            cfg["source_series_manifests"][split],
            dtype=str,
            keep_default_na=False,
        )
        local = pd.read_csv(
            manifests / f"cave_manifest_local_{split.casefold()}_eligible.csv",
            dtype=str,
            keep_default_na=False,
        )
        retained = source_series[
            source_series["series_uid"].isin(set(local["series_uid"]))
        ]
        if retained["series_uid"].tolist() != local["series_uid"].tolist():
            failures.append(f"{split}: local manifest changed source series order")
        strict_prepost = int(
            (
                local["local_pre_eligible"].map(as_bool)
                & local["local_post_eligible"].map(as_bool)
            ).sum()
        )
        expected_strict = int(expected[f"strict_prepost_series_{split.casefold()}"])
        if strict_prepost != expected_strict:
            failures.append(f"{split}: strict Pre/Post={strict_prepost}")
        for row in local.to_dict("records"):
            uid = row["series_uid"]
            for phase in ("pre", "post"):
                got = as_bool(row[f"local_{phase}_eligible"])
                want = (uid, phase) in eligible_keys
                if got != want:
                    failures.append(f"{uid}/{phase}: local manifest eligibility mismatch")
        split_summary[split] = {
            "source_series": len(source_series),
            "extract_series": len(local),
            "strict_prepost_series": strict_prepost,
            "eligible_phases": int((eligible["split"] == split).sum()),
            "local_manifest_sha256": sha256_file(
                manifests / f"cave_manifest_local_{split.casefold()}_eligible.csv"
            ),
        }

    summary = {
        "status": "failed" if failures else "success",
        "failures": failures,
        "annotation_source": "paired_png_2d_mean_and_gt_mask",
        "segmentation_model_used": False,
        "source_phases": len(source),
        "eligible_phases": len(eligible),
        "excluded_phases": len(excluded),
        "exclusion_reason_counts": reason_prefix,
        "exact_shape_phases": int((~resized).sum()),
        "nearest_resized_phases": int(resized.sum()),
        "resize_interpolation": "nearest",
        "nonuniform_scale_exclusions": nonuniform,
        "foreground_rule": "labels_in_1_2_equivalent_nonzero",
        "selected_labels": [1, 2],
        "all_nonzero_equals_selected_foreground": True,
        "orientation_status_counts": roi["orientation_status"].value_counts().to_dict(),
        "splits": split_summary,
        "qa_generated": int(qa.get("qa_generated", 0)),
        "qa_failures": int(qa.get("qa_failures", -1)),
        "temporal_policy": cfg["temporal"]["policy"],
        "local_frames_saved": False,
    }
    atomic_json(summary, reports / "05_png2d_stage1_validation.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise AssertionError("; ".join(failures[:30]))
    write_success(
        manifests / ".STAGE1_ELIGIBLE_SUCCESS.json",
        "png2d_stage1_eligible",
        cfg,
        summary,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


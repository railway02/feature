#!/usr/bin/env python3
"""Stage-1 closure validation under the eligible_only coverage policy.

The lock means: every eligible phase has a correct ROI, and every excluded
phase has an explicit reason — not that all 2622 phases have masks.
"""
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

    phase_index = pd.read_csv(manifests / "source_phase_index_all.csv", dtype=str, keep_default_na=False)
    coverage = pd.read_csv(manifests / "local_phase_coverage_all.csv", dtype=str, keep_default_na=False)
    roi = pd.read_csv(manifests / "roi_phase_manifest_eligible.csv", dtype=str, keep_default_na=False)
    temporal = pd.read_csv(manifests / "whole_temporal_views_eligible.csv", dtype=str, keep_default_na=False)


    labels12_path = manifests / "labels12_phase_coverage_all.csv"
    if not labels12_path.is_file():
        failures.append("缺少 labels12_phase_coverage_all.csv")
    elif sha256_file(labels12_path) != sha256_file(manifests / "local_phase_coverage_all.csv"):
        failures.append("labels12 coverage 与 canonical coverage 不一致")
    roi_cfg = cfg.get("roi", {})
    if roi_cfg.get("target_rule") != "labels_1_2_for_roi_only":
        failures.append("config target_rule 不是 labels_1_2_for_roi_only")
    if [int(value) for value in roi_cfg.get("foreground_labels", [])] != [1, 2]:
        failures.append("config foreground_labels 不是 [1, 2]")
    required_label_columns = {
        "foreground_rule", "selected_labels", "labels_present", "labels_ignored",
        "selected_foreground_pixels", "ignored_foreground_pixels",
    }
    if not required_label_columns.issubset(coverage.columns):
        failures.append("coverage 缺少 labels12 provenance 列")
    if not required_label_columns.issubset(roi.columns):
        failures.append("ROI manifest 缺少 labels12 provenance 列")
    if "foreground_rule" in roi and not roi["foreground_rule"].eq("labels_in_1_2").all():
        failures.append("ROI foreground_rule 不唯一或不是 labels_in_1_2")
    if "selected_labels" in roi and not roi["selected_labels"].eq("[1,2]").all():
        failures.append("ROI selected_labels 不唯一或不是 [1,2]")
    if "selected_foreground_pixels" in roi and (pd.to_numeric(roi["selected_foreground_pixels"]) <= 0).any():
        failures.append("eligible ROI 存在空 Labels 1+2 前景")
    expected_uids = phase_index["phase_uid"].astype(str).tolist()
    if len(expected_uids) != len(set(expected_uids)):
        failures.append("source phase_uid 重复")
    if coverage["phase_uid"].astype(str).tolist() != expected_uids:
        failures.append("coverage 行与 source_phase_index 不一致（数量或顺序）")

    eligible = coverage[coverage["local_eligible"].map(as_bool)]
    excluded = coverage[~coverage["local_eligible"].map(as_bool)]
    if (excluded["local_exclusion_reason"].astype(str) == "").any():
        failures.append("存在没有原因的 excluded phase")
    if set(eligible["phase_uid"].astype(str)) != set(roi["phase_uid"].astype(str)):
        failures.append("eligible phase 与 ROI manifest 集合不一致")
    if set(eligible["phase_uid"].astype(str)) != set(temporal["phase_uid"].astype(str)):
        failures.append("eligible phase 与 temporal views 集合不一致")
    if len(eligible) + len(excluded) != len(coverage):
        failures.append("eligible+excluded != source phase 总数")
    if roi["frame_list_hash"].duplicated().any():
        failures.append("ROI frame_list_hash 重复")
    if roi["mask_path"].astype(str).eq("").any():
        failures.append("eligible ROI 存在空 mask_path")

    for row in roi.to_dict("records"):
        tight = bbox_from_text(row["original_bbox"])
        expanded = bbox_from_text(row["expanded_bbox"])
        if expanded[0] > tight[0] or expanded[1] > tight[1] or expanded[2] < tight[2] or expanded[3] < tight[3]:
            failures.append(f"ROI 裁掉 Mask：{row['phase_uid']}")
        paths = parse_pipe(row["frame_paths"])
        if len(paths) != int(float(row["n_frames"])):
            failures.append(f"帧数不闭合：{row['phase_uid']}")
        if not Path(row["mask_path"]).is_file():
            failures.append(f"Mask 不存在：{row['mask_path']}")
        if not Path(row["whole_metadata_path"]).is_file():
            failures.append(f"Whole metadata 不存在：{row['whole_metadata_path']}")

    split_summary = {}
    for split in ("Train", "Valid"):
        source = pd.read_csv(cfg["source_series_manifests"][split], dtype=str, keep_default_na=False)
        local_path = manifests / f"cave_manifest_local_{split.casefold()}_eligible.csv"
        local = pd.read_csv(local_path, dtype=str, keep_default_na=False)
        kept_source = source[source["series_uid"].isin(set(local["series_uid"].astype(str)))]
        if kept_source["series_uid"].astype(str).tolist() != local["series_uid"].astype(str).tolist():
            failures.append(f"{split} eligible manifest 改变了 series_uid 顺序")
        merged = source.merge(
            local[["series_uid", "can_run_pre", "can_run_post", "local_pre_eligible", "local_post_eligible"]],
            on="series_uid", how="inner", suffixes=("_src", "_local"),
        )
        eligible_keys = set(zip(eligible["series_uid"], eligible["phase"]))
        for row in merged.to_dict("records"):
            uid = str(row["series_uid"])
            for phase in ("pre", "post"):
                want = as_bool(row[f"can_run_{phase}_src"]) and (uid, phase) in eligible_keys
                got = as_bool(row[f"can_run_{phase}_local"])
                if want != got:
                    failures.append(f"{split} {uid} can_run_{phase} 与 eligibility 不一致")
        split_cov = coverage[coverage["split"] == split]
        split_summary[split] = {
            "source_series": int(len(source)),
            "extract_series": int(len(local)),
            "source_phases": int(len(split_cov)),
            "eligible_phases": int(split_cov["local_eligible"].map(as_bool).sum()),
            "excluded_phases": int((~split_cov["local_eligible"].map(as_bool)).sum()),
            "local_manifest": str(local_path),
            "local_manifest_sha256": sha256_file(local_path),
        }

    reason_counts = (
        excluded["local_exclusion_reason"].astype(str).str.split(":", n=1).str[0].value_counts().to_dict()
    )
    orientation_counts = eligible["orientation_status"].value_counts().to_dict() if "orientation_status" in eligible else {}
    summary = {
        "status": "failed" if failures else "success",
        "failures": failures,
        "coverage_policy": "eligible_only",
        "foreground_rule": "labels_in_1_2",
        "selected_labels": [1, 2],
        "empty_selected_labels_policy": "exclude_without_nonzero_fallback",
        "labels12_coverage_sha256": sha256_file(labels12_path) if labels12_path.is_file() else "",
        "source_phase_count": int(len(coverage)),
        "eligible_phase_count": int(len(eligible)),
        "excluded_phase_count": int(len(excluded)),
        "exclusion_reason_counts": reason_counts,
        "eligible_orientation_status_counts": orientation_counts,
        "splits": split_summary,
        "segmentation_model_used": False,
        "allow_mask_resize": bool(cfg.get("roi", {}).get("allow_mask_resize", False)),
        "local_frames_saved": False,
    }
    atomic_json(summary, reports / "05_stage1_eligible_validation.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise AssertionError("; ".join(failures[:25]))
    write_success(manifests / ".STAGE1_ELIGIBLE_SUCCESS.json", "stage1_eligible", cfg, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

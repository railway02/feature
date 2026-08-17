#!/usr/bin/env python3
"""ROI QA overlays for eligible phases.

Frame picks come from the frozen Whole-CAVE contrast_core20 indices (contrast
peak region), never just the first frame — early frames may carry no
contrast. Sampling: every non-upstream/manual mapping (unverified
orientation), all anomalies, ROI side extremes, one row per resolution, plus
a seeded random upstream sample.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common import atomic_csv, atomic_json, load_config, parse_pipe
from nifti_io import apply_orientation, load_label_mask
from roi import bbox_from_mask, bbox_from_text, crop_frame


def normalize_u8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    return np.clip((image - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)


def overlay(image: np.ndarray, labels: np.ndarray, old_tight: tuple[int, int, int, int], tight: tuple[int, int, int, int], roi: tuple[int, int, int, int], fallback: tuple[int, int, int, int]) -> np.ndarray:
    canvas = cv2.cvtColor(normalize_u8(image), cv2.COLOR_GRAY2BGR)
    colors = {
        1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 0),
        4: (0, 165, 255), 5: (255, 255, 0), 6: (255, 0, 255),
    }
    for value, color in colors.items():
        contours, _ = cv2.findContours((labels == value).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, 2)
    cv2.rectangle(canvas, (old_tight[0], old_tight[1]), (old_tight[2] - 1, old_tight[3] - 1), (0, 165, 255), 2)
    cv2.rectangle(canvas, (tight[0], tight[1]), (tight[2] - 1, tight[3] - 1), (0, 0, 255), 2)
    cv2.rectangle(canvas, (roi[0], roi[1]), (roi[2] - 1, roi[3] - 1), (0, 255, 0), 2)
    cv2.rectangle(canvas, (fallback[0], fallback[1]), (fallback[2] - 1, fallback[3] - 1), (255, 255, 0), 1)
    cv2.putText(canvas, "L1 red; L2 green; L3-6 ignored", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(canvas, "old tight orange; new tight red", (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(canvas, "primary green; fallback cyan", (12, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return canvas


def representative_indices(blocks_json: str, n_frames: int) -> list[int]:
    """Return early, contrast-peak, and late indices from frozen Whole views."""
    fallback = sorted({0, n_frames // 2, n_frames - 1})
    try:
        blocks = json.loads(blocks_json)
        uniform: list[int] = []
        core: list[int] = []
        for block in blocks:
            views = block.get("view_indices", {})
            uniform.extend(int(value) for value in views.get("uniform_full20", []))
            core.extend(int(value) for value in views.get("contrast_core20", []))
        if not uniform or not core:
            return fallback
        early = min(max(min(uniform), 0), n_frames - 1)
        peak = min(max(core[len(core) // 2], 0), n_frames - 1)
        late = min(max(max(uniform), 0), n_frames - 1)
        return [early, peak, late]
    except Exception:
        return fallback


def select_samples(group: pd.DataFrame, upstream_sample: int, seed: int) -> pd.DataFrame:
    def tagged(frame: pd.DataFrame, reason: str) -> pd.DataFrame:
        out = frame.copy()
        out["qa_reason"] = reason
        return out

    old_ratio = pd.to_numeric(group["old_roi_area_ratio"], errors="coerce")
    bbox_score = pd.to_numeric(group["bbox_change_score"], errors="coerce")
    ignored_pixels = pd.to_numeric(group["ignored_foreground_pixels"], errors="coerce")
    fallback_gain = pd.to_numeric(group["fallback_roi_area_ratio"], errors="coerce") - pd.to_numeric(group["roi_area_ratio"], errors="coerce")
    review = group[group["orientation_status"] == "default_identity_unverified"]
    largest_change = group.assign(_score=bbox_score).nlargest(min(20, len(group)), "_score").drop(columns="_score")
    old_whole = group[old_ratio >= 1.0]
    ignored = group.assign(_ignored=ignored_pixels).nlargest(min(20, len(group)), "_ignored").drop(columns="_ignored")
    fallback = group.assign(_fallback=fallback_gain).nlargest(min(12, len(group)), "_fallback").drop(columns="_fallback")
    actual_fallback = group[group["actual_fallback_used"]]
    resolutions = group.drop_duplicates(subset=["frame_height", "frame_width"])

    target_parts: list[pd.DataFrame] = []
    for target_value in (0, 1):
        candidates = group[pd.to_numeric(group["target"], errors="coerce") == target_value]
        if len(candidates):
            target_parts.append(
                tagged(candidates.sample(n=min(6, len(candidates)), random_state=seed + target_value), f"random_target_{target_value}")
            )

    ratio = pd.to_numeric(group["roi_area_ratio"], errors="coerce")
    anomaly = group[(ratio > 0.50) | (ratio < 0.005) | (group["mask_resized_to_frame"].astype(str).str.casefold() == "true")]
    base = pd.concat(
        [
            tagged(largest_change, "largest_old_new_bbox_change"),
            tagged(old_whole, "old_roi_area_ratio_ge_1"),
            tagged(ignored, "typical_labels_3_6_ignored"),
            tagged(review, "default_identity_unverified"),
            tagged(actual_fallback, "actual_gpu_fallback_used"),
            tagged(fallback, "fallback_roi_candidate"),
            tagged(resolutions, "resolution_coverage"),
            tagged(anomaly, "new_roi_area_anomaly"),
            *target_parts,
        ],
        ignore_index=True,
    ).drop_duplicates("phase_uid")
    rest = group[~group["phase_uid"].isin(set(base["phase_uid"].astype(str)))]
    upstream_rest = rest[rest["mapping_method"].isin(["upstream", "manual"])]
    sample = upstream_rest.sample(n=min(upstream_sample, len(upstream_rest)), random_state=seed) if len(upstream_rest) else upstream_rest
    return pd.concat([base, tagged(sample, "random_general")]).drop_duplicates("phase_uid")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--upstream-sample", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifests = Path(cfg["paths"]["manifests"])
    reports = Path(cfg["paths"]["reports"])
    qa_root = reports / "roi_qa"
    roi = pd.read_csv(manifests / "roi_phase_manifest_eligible.csv", dtype=str, keep_default_na=False)
    old_path = Path("/root/autodl-tmp/aneurysm/manifests/api_gtmask_roi_cave_v5_fullmask_fullseq/roi_phase_manifest_eligible.csv")
    old = pd.read_csv(old_path, dtype=str, keep_default_na=False)[["phase_uid", "expanded_bbox", "roi_area_ratio"]]
    old = old.rename(columns={"expanded_bbox": "old_expanded_bbox", "roi_area_ratio": "old_roi_area_ratio"})
    roi = roi.merge(old, on="phase_uid", how="left", validate="one_to_one")
    if roi[["old_expanded_bbox", "old_roi_area_ratio"]].eq("").any().any():
        raise AssertionError("New labels12 ROI is missing its v5 QA comparator")
    views = pd.read_csv(manifests / "whole_temporal_views_eligible.csv", dtype=str, keep_default_na=False)

    def bbox_change_score(row: pd.Series) -> float:
        old_box = bbox_from_text(row["old_expanded_bbox"])
        new_box = bbox_from_text(row["expanded_bbox"])
        scale = max(float(row["frame_width"]) + float(row["frame_height"]), 1.0)
        return float(sum(abs(left - right) for left, right in zip(old_box, new_box)) / scale)

    roi["bbox_change_score"] = roi.apply(bbox_change_score, axis=1)
    actual_fallback_uids: set[str] = set()
    smoke_root = Path(cfg["paths"]["outputs"]) / "smoke_local_eligible_featurebank"
    if smoke_root.is_dir():
        for success_path in smoke_root.rglob(".SUCCESS.json"):
            payload = json.loads(success_path.read_text(encoding="utf-8"))
            if bool(payload.get("fallback_used", False)):
                actual_fallback_uids.add(str(payload.get("phase_uid", "")))
    roi["actual_fallback_used"] = roi["phase_uid"].isin(actual_fallback_uids)
    roi["target"] = np.nan
    task_root = Path("/root/autodl-tmp/aneurysm/outputs/api_gtmask_roi_cave_v5_fullmask_fullseq/adverse_prepost_series_task_v3")
    for split in ("Train", "Valid"):
        samples_path = task_root / f"{split.casefold()}_series_samples.csv"
        if not samples_path.is_file():
            continue
        fixed_samples = pd.read_csv(samples_path, dtype=str, keep_default_na=False)
        target_by_uid = dict(zip(fixed_samples["series_uid"].astype(str), fixed_samples["target"].astype(int)))
        split_mask = roi["split"].eq(split)
        roi.loc[split_mask, "target"] = roi.loc[split_mask, "series_uid"].map(target_by_uid)
    atomic_csv(roi[["phase_uid", "split", "series_uid", "old_expanded_bbox", "expanded_bbox", "old_roi_area_ratio", "roi_area_ratio", "bbox_change_score", "labels_present", "labels_ignored", "selected_foreground_pixels", "ignored_foreground_pixels"]], reports / "04_labels12_bbox_comparison.csv")
    blocks_by_hash = {str(r.frame_list_hash): str(r.blocks_json) for r in views.itertuples(index=False)}

    samples: list[pd.DataFrame] = []
    for split in ("Train", "Valid"):
        group = roi[roi["split"] == split].copy()
        samples.append(select_samples(group, args.upstream_sample, args.seed))
    sample = pd.concat(samples, ignore_index=True)

    index_rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    for row in sample.to_dict("records"):
        try:
            paths = parse_pipe(row["frame_paths"])
            if not paths:
                raise ValueError("empty frame paths")
            selected = representative_indices(blocks_by_hash.get(str(row["frame_list_hash"]), ""), len(paths))
            frames, valid_indices = [], []
            for index in selected:
                image = cv2.imread(paths[index], cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    frames.append(image)
                    valid_indices.append(index)
            if not frames:
                raise FileNotFoundError("QA frames unreadable")
            labels, _ = load_label_mask(row["mask_path"])
            labels = apply_orientation(labels, row["orientation_transform"])
            if labels.shape != frames[0].shape:
                raise AssertionError(f"QA mask/frame shape mismatch without resize: {labels.shape} vs {frames[0].shape}")
            old_tight = bbox_from_mask(labels != 0)
            tight = bbox_from_text(row["original_bbox"])
            box = bbox_from_text(row["expanded_bbox"])
            fallback = bbox_from_text(row["fallback_bbox"])
            reference_path = str(row.get("matched_frame_path", ""))
            reference = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE) if reference_path else None
            if reference is None:
                reference = frames[len(frames) // 2]
            panels = []
            reference_full = overlay(reference, labels, old_tight, tight, box, fallback)
            reference_local = cv2.cvtColor(normalize_u8(crop_frame(reference, box)), cv2.COLOR_GRAY2BGR)
            reference_local = cv2.resize(reference_local, (reference_full.shape[1], reference_full.shape[0]), interpolation=cv2.INTER_AREA)
            cv2.putText(reference_full, "reference + label map", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(reference_local, "reference local", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            panels.append(np.concatenate([reference_full, reference_local], axis=1))
            frame_names = ("early", "peak", "late")
            for panel_index, (frame, frame_index) in enumerate(zip(frames, valid_indices)):
                full = overlay(frame, labels, old_tight, tight, box, fallback)
                local = cv2.cvtColor(normalize_u8(crop_frame(frame, box)), cv2.COLOR_GRAY2BGR)
                local = cv2.resize(local, (full.shape[1], full.shape[0]), interpolation=cv2.INTER_AREA)
                frame_name = frame_names[min(panel_index, len(frame_names) - 1)]
                cv2.putText(full, f"{frame_name} frame={frame_index} full", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(local, f"{frame_name} frame={frame_index} local", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                panels.append(np.concatenate([full, local], axis=1))
            contact = np.concatenate(panels, axis=0)
            safe_uid = str(row["series_uid"]).replace("/", "_").replace("\\", "_")
            out = qa_root / row["split"].casefold() / str(row["patient_id"]) / f"{safe_uid}__{row['phase']}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(out), contact, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise IOError(f"cannot write {out}")
            index_rows.append({
                "phase_uid": row["phase_uid"], "split": row["split"], "patient_id": row["patient_id"],
                "series_uid": row["series_uid"], "phase": row["phase"], "mapping_method": row["mapping_method"],
                "orientation_transform": row["orientation_transform"], "orientation_status": row.get("orientation_status", ""),
                "qa_reason": row.get("qa_reason", ""), "target": row.get("target", ""), "actual_fallback_used": row.get("actual_fallback_used", False),
                "qa_frame_indices": "|".join(map(str, valid_indices)),
                "old_expanded_bbox": row["old_expanded_bbox"], "new_tight_bbox": row["original_bbox"],
                "new_expanded_bbox": row["expanded_bbox"], "fallback_bbox": row["fallback_bbox"],
                "labels_present": row["labels_present"], "labels_ignored": row["labels_ignored"],
                "mask_path": row["mask_path"], "qa_path": str(out),
            })
        except Exception as exc:
            failures.append({**row, "qa_error": f"{type(exc).__name__}:{exc}"})

    atomic_csv(pd.DataFrame(index_rows), reports / "04_roi_qa_index.csv")
    atomic_csv(pd.DataFrame(failures), reports / "04_roi_qa_failures.csv")
    method_counts = pd.Series([r["mapping_method"] for r in index_rows]).value_counts().to_dict()
    summary = {
        "qa_images": len(index_rows),
        "qa_failures": len(failures),
        "qa_root": str(qa_root),
        "qa_by_mapping_method": method_counts,
        "frame_policy": "frozen_whole_uniform_early_late_plus_contrast_core_peak",
        "foreground_rule": "labels_in_1_2",
        "old_v5_manifest_used_for_qa_only": str(old_path),
        "mask_resize_used_for_qa": False,
    }
    atomic_json(summary, reports / "04_roi_qa_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

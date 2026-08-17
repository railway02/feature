#!/usr/bin/env python3
"""Create PNG2D-to-full-sequence ROI QA mosaics on the source-frame grid."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from common import atomic_csv, atomic_json, load_config, parse_pipe
from roi import bbox_from_text, crop_frame


def load_roi_module(code_dir: Path):
    path = code_dir / "03_build_png2d_roi_manifests.py"
    spec = importlib.util.spec_from_file_location("png2d_roi_builder", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_u8(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image, dtype=np.float32)
    finite = value[np.isfinite(value)]
    if not finite.size:
        return np.zeros(value.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        return np.zeros(value.shape, dtype=np.uint8)
    return np.clip((value - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)


def source_mean(paths: list[str]) -> np.ndarray:
    total: np.ndarray | None = None
    shape: tuple[int, int] | None = None
    for path in paths:
        frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise FileNotFoundError(path)
        if shape is None:
            shape = frame.shape
            total = np.zeros(shape, dtype=np.float64)
        elif frame.shape != shape:
            raise ValueError(f"mixed source frame shapes: {frame.shape} vs {shape}")
        assert total is not None
        total += frame
    if total is None:
        raise ValueError("empty frame list")
    return (total / len(paths)).astype(np.float32)


def draw_overlay(
    image: np.ndarray,
    labels: np.ndarray,
    tight: tuple[int, int, int, int],
    primary: tuple[int, int, int, int],
    fallback: tuple[int, int, int, int],
    extended: tuple[int, int, int, int],
    title: str,
) -> np.ndarray:
    canvas = cv2.cvtColor(normalize_u8(image), cv2.COLOR_GRAY2BGR)
    colors = {1: (0, 0, 255), 2: (0, 255, 0)}
    for value, color in colors.items():
        contours, _ = cv2.findContours(
            (labels == value).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(canvas, contours, -1, color, 2)
    cv2.rectangle(
        canvas, (tight[0], tight[1]), (tight[2] - 1, tight[3] - 1), (0, 0, 255), 2
    )
    cv2.rectangle(
        canvas,
        (primary[0], primary[1]),
        (primary[2] - 1, primary[3] - 1),
        (0, 255, 0),
        2,
    )
    cv2.rectangle(
        canvas,
        (fallback[0], fallback[1]),
        (fallback[2] - 1, fallback[3] - 1),
        (255, 255, 0),
        1,
    )
    cv2.rectangle(
        canvas,
        (extended[0], extended[1]),
        (extended[2] - 1, extended[3] - 1),
        (255, 0, 255),
        1,
    )
    cv2.putText(
        canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    cv2.putText(
        canvas,
        "L1 red; L2 green; ROI green; F2 cyan; F3 magenta",
        (12, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
    )
    return canvas


def frame_indices(blocks_json: str, n_frames: int) -> list[int]:
    fallback = [0, n_frames // 2, n_frames - 1]
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
        return [
            max(0, min(n_frames - 1, min(uniform))),
            max(0, min(n_frames - 1, core[len(core) // 2])),
            max(0, min(n_frames - 1, max(uniform))),
        ]
    except Exception:
        return fallback


def tag(frame: pd.DataFrame, reason: str) -> pd.DataFrame:
    result = frame.copy()
    result["qa_reason"] = reason
    return result


def select_samples(roi: pd.DataFrame, reports: Path, seed: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    weak_path = reports / "01_weak_identity_orientation_audit.csv"
    if weak_path.is_file():
        weak = pd.read_csv(weak_path, dtype=str, keep_default_na=False)
        parts.append(tag(roi[roi["phase_uid"].isin(set(weak["phase_uid"]))], "weak_identity_all_8_checked"))
    alias_or_manual = roi[
        roi["mapping_method"].isin([
            "series_path_alias_identity_mean_verified",
            "manual_visual_identity_confirmed",
        ])
    ]
    if len(alias_or_manual):
        parts.append(tag(alias_or_manual, "complete_mapping_alias_or_manual_all"))
    resized = roi[roi["mask_resized_to_frame"].astype(str).str.casefold().isin({"1", "true"})]
    if len(resized):
        parts.append(tag(resized.sample(min(40, len(resized)), random_state=seed), "resized_mask_random"))
    parts.append(tag(roi.drop_duplicates(["frame_height", "frame_width"]), "source_resolution_coverage"))
    ratio = pd.to_numeric(roi["roi_area_ratio"], errors="coerce")
    parts.append(tag(roi.assign(_ratio=ratio).nlargest(min(12, len(roi)), "_ratio").drop(columns="_ratio"), "largest_roi"))
    parts.append(tag(roi.assign(_ratio=ratio).nsmallest(min(12, len(roi)), "_ratio").drop(columns="_ratio"), "smallest_roi"))
    parts.append(tag(roi[ratio >= 1.0], "roi_area_ratio_ge_1"))
    parts.append(tag(roi[roi["labels_present"].eq("[1]")], "label1_only"))
    corr = pd.to_numeric(roi["mapping_score"], errors="coerce")
    parts.append(tag(roi.assign(_corr=corr).nsmallest(min(30, len(roi)), "_corr").drop(columns="_corr"), "lowest_mapping_correlation"))

    task_root = Path(
        "/root/autodl-tmp/aneurysm/outputs/"
        "api_gtmask_roi_cave_v5_fullmask_fullseq/adverse_prepost_series_task_v3"
    )
    for split in ("Train", "Valid"):
        path = task_root / f"{split.casefold()}_series_samples.csv"
        group = roi[roi["split"] == split]
        if path.is_file():
            samples = pd.read_csv(path, dtype=str, keep_default_na=False)
            target = dict(zip(samples["series_uid"], pd.to_numeric(samples["target"])))
            values = group["series_uid"].map(target)
            for outcome in (0, 1):
                candidates = group[values == outcome]
                if len(candidates):
                    parts.append(tag(
                        candidates.sample(min(8, len(candidates)), random_state=seed + outcome),
                        f"{split.casefold()}_target_{outcome}",
                    ))
        if len(group):
            parts.append(tag(group.sample(min(10, len(group)), random_state=seed + 20), f"{split.casefold()}_random"))
    return pd.concat(parts, ignore_index=True).drop_duplicates("phase_uid")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cfg = load_config(args.config)
    code_dir = Path(cfg["paths"]["code"])
    manifests = Path(cfg["paths"]["manifests"])
    reports = Path(cfg["paths"]["reports"])
    qa_root = reports / "roi_qa"
    qa_root.mkdir(parents=True, exist_ok=True)
    roi_module = load_roi_module(code_dir)
    roi = pd.read_csv(
        manifests / "roi_phase_manifest_eligible.csv",
        dtype=str,
        keep_default_na=False,
    )
    views = pd.read_csv(
        manifests / "whole_temporal_views_eligible.csv",
        dtype=str,
        keep_default_na=False,
    )
    blocks_by_hash = dict(zip(views["frame_list_hash"], views["blocks_json"]))
    sample = select_samples(roi, reports, args.seed)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for record in sample.to_dict("records"):
        try:
            paths = parse_pipe(record["frame_paths"])
            if not paths:
                raise ValueError("empty frame list")
            reference = roi_module.read_gray(record["reference_image_path"])
            labels = roi_module.read_gray(record["mask_path"])
            frame_shape = (int(record["frame_height"]), int(record["frame_width"]))
            labels_frame, _ = roi_module.effective_mask(labels, frame_shape, cfg["roi"])
            reference_frame = cv2.resize(
                reference,
                (frame_shape[1], frame_shape[0]),
                interpolation=(
                    cv2.INTER_AREA
                    if reference.shape[0] > frame_shape[0]
                    else cv2.INTER_CUBIC
                ),
            ) if reference.shape != frame_shape else reference
            mean = source_mean(paths)
            tight = bbox_from_text(record["original_bbox"])
            primary = bbox_from_text(record["expanded_bbox"])
            fallback = bbox_from_text(record["fallback_bbox"])
            extended = bbox_from_text(record["extended_fallback_bbox"])
            panels = [
                draw_overlay(reference_frame, labels_frame, tight, primary, fallback, extended, "uploaded 2D mean + mapped GT"),
                draw_overlay(mean, labels_frame, tight, primary, fallback, extended, "frozen JPG arithmetic mean + mapped GT"),
            ]
            selected = frame_indices(
                blocks_by_hash.get(record["frame_list_hash"], ""),
                len(paths),
            )
            names = ("early", "contrast-core", "late")
            for name, index in zip(names, selected):
                frame = cv2.imread(paths[index], cv2.IMREAD_GRAYSCALE)
                if frame is None:
                    raise FileNotFoundError(paths[index])
                full = draw_overlay(frame, labels_frame, tight, primary, fallback, extended, f"{name} frame={index}")
                local = cv2.cvtColor(normalize_u8(crop_frame(frame, primary)), cv2.COLOR_GRAY2BGR)
                local = cv2.resize(
                    local, (full.shape[1], full.shape[0]), interpolation=cv2.INTER_AREA
                )
                cv2.putText(
                    local,
                    f"{name} local ROI",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                panels.extend([full, local])
            width = 640
            resized_panels = []
            for panel in panels:
                height = max(1, int(round(panel.shape[0] * width / panel.shape[1])))
                resized_panels.append(cv2.resize(panel, (width, height), interpolation=cv2.INTER_AREA))
            target_height = max(panel.shape[0] for panel in resized_panels)
            padded = [
                cv2.copyMakeBorder(panel, 0, target_height - panel.shape[0], 0, 0, cv2.BORDER_CONSTANT)
                for panel in resized_panels
            ]
            lines = [
                np.concatenate(padded[index:index + 2], axis=1)
                for index in range(0, len(padded), 2)
            ]
            mosaic = np.concatenate(lines, axis=0)
            output = qa_root / f"{record['phase_uid'].replace('::', '__')}.jpg"
            if not cv2.imwrite(str(output), mosaic):
                raise OSError(f"failed to write {output}")
            rows.append({
                "phase_uid": record["phase_uid"],
                "split": record["split"],
                "series_uid": record["series_uid"],
                "phase": record["phase"],
                "qa_reason": record["qa_reason"],
                "mapping_score": record["mapping_score"],
                "annotation_shape": record["annotation_shape"],
                "frame_shape": f"{record['frame_height']}x{record['frame_width']}",
                "mask_resized_to_frame": record["mask_resized_to_frame"],
                "roi_area_ratio": record["roi_area_ratio"],
                "qa_path": str(output),
            })
        except Exception as exc:
            failures.append({
                "phase_uid": str(record.get("phase_uid", "")),
                "reason": f"{type(exc).__name__}:{exc}",
            })

    index = pd.DataFrame(rows)
    failure_frame = pd.DataFrame(failures, columns=["phase_uid", "reason"])
    atomic_csv(index, reports / "04_roi_qa_index.csv")
    atomic_csv(failure_frame, reports / "04_roi_qa_failures.csv")

    rejected = pd.read_csv(
        manifests / "local_phase_coverage_all.csv",
        dtype=str,
        keep_default_na=False,
    )
    rejected = rejected[
        rejected["local_exclusion_reason"].str.startswith("nonuniform_scale_not_allowed")
    ]
    atomic_csv(rejected, reports / "04_nonuniform_scale_rejected.csv")
    summary = {
        "status": "success" if not failures else "failed",
        "qa_requested": len(sample),
        "qa_generated": len(index),
        "qa_failures": len(failures),
        "weak_mapping_qas": int(index["qa_reason"].eq("weak_identity_all_8_checked").sum()) if len(index) else 0,
        "resized_mask_qas": int(index["mask_resized_to_frame"].astype(str).str.casefold().isin({"1", "true"}).sum()) if len(index) else 0,
        "nonuniform_scale_rejected": len(rejected),
        "qa_content": [
            "uploaded_2d_mean_with_frame_grid_gt_mask",
            "frozen_jpg_arithmetic_mean_with_same_gt_mask",
            "early_contrast_core_late_full_frames",
            "corresponding_local_roi_crops",
        ],
    }
    atomic_json(summary, reports / "04_roi_qa_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise AssertionError(f"{len(failures)} PNG2D ROI QA failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


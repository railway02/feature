#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def parse_box(value: str) -> tuple[int, int, int, int]:
    values = tuple(int(item) for item in str(value).split("|"))
    if len(values) != 4:
        raise ValueError(value)
    return values


def clipped_box_coverage(
    mask: np.ndarray, box: tuple[int, int, int, int]
) -> tuple[float, float, float]:
    x0, y0, x1, y1 = box
    height, width = mask.shape
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(width, x1), min(height, y1)
    inside = (
        int(mask[sy0:sy1, sx0:sx1].sum())
        if sx0 < sx1 and sy0 < sy1
        else 0
    )
    total = max(int(mask.sum()), 1)
    coverage = float(inside / total)
    area_ratio = float((x1 - x0) * (y1 - y0) / (height * width))
    return coverage, float(inside > 0), area_ratio


def local_contrast(image: np.ndarray, mask: np.ndarray) -> float:
    binary = (mask > 0).astype(np.uint8)
    points = np.argwhere(binary > 0)
    if not len(points):
        return float("nan")
    y0, x0 = points.min(axis=0)
    y1, x1 = points.max(axis=0) + 1
    height, width = binary.shape
    side = max(int(y1 - y0), int(x1 - x0))
    margin = max(8, side)
    ax0, ay0 = max(0, int(x0) - margin), max(0, int(y0) - margin)
    ax1, ay1 = min(width, int(x1) + margin), min(height, int(y1) + margin)
    annulus = np.zeros_like(binary, dtype=bool)
    annulus[ay0:ay1, ax0:ax1] = True
    annulus &= ~binary.astype(bool)
    foreground = image[binary > 0].astype(np.float64)
    background = image[annulus].astype(np.float64)
    if not len(foreground) or len(background) < 16:
        return float("nan")
    scale = float(np.std(background))
    if scale < 1e-8:
        return float("nan")
    return float(abs(np.mean(foreground) - np.mean(background)) / scale)


def quantiles(values: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {}
    return {
        "min": float(numeric.min()),
        "p10": float(numeric.quantile(0.10)),
        "median": float(numeric.median()),
        "p90": float(numeric.quantile(0.90)),
        "max": float(numeric.max()),
        "mean": float(numeric.mean()),
    }


def grouped_summary(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(frame)),
        "roi_1p5_coverage": quantiles(frame["roi_1p5_coverage"]),
        "roi_3p0_coverage": quantiles(frame["roi_3p0_coverage"]),
        "roi_3p0_area_ratio": quantiles(frame["roi_3p0_area_ratio"]),
        "roi_1p5_on_target": float(frame["roi_1p5_on_target"].mean()),
        "roi_3p0_on_target": float(frame["roi_3p0_on_target"].mean()),
        "roi_3p0_coverage_ge_0p95": float(
            (frame["roi_3p0_coverage"] >= 0.95).mean()
        ),
        "roi_3p0_full_coverage": float(
            (frame["roi_3p0_coverage"] >= 0.999).mean()
        ),
        "roi_3p0_zero_coverage": int(
            (frame["roi_3p0_coverage"] == 0.0).sum()
        ),
        "component_selection_changed": int(
            frame["component_selection_changed"].sum()
        ),
        "component_iou": quantiles(frame["component_iou"]),
        "lesion_area_ratio": quantiles(frame["lesion_area_ratio"]),
        "context_area_ratio": quantiles(frame["context_area_ratio"]),
        "mean_channel_local_contrast_z": quantiles(
            frame["mean_channel_local_contrast_z"]
        ),
        "median_channel_local_contrast_z": quantiles(
            frame["median_channel_local_contrast_z"]
        ),
        "max_enhancement_local_contrast_z": quantiles(
            frame["max_enhancement_local_contrast_z"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("/root/autodl-tmp/aneurysm")
    )
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()

    root = args.root.resolve()
    upstream_code = root / "code/api_adverse_lesion_cave_v1"
    import sys

    sys.path.insert(0, str(upstream_code))
    from segmentation import restore_model_probability

    report_root = root / "reports/api_adverse_lesion_cave_fast_v1"
    roi = pd.read_csv(
        root / "manifests/api_adverse_lesion_cave_fast_v1/roi_manifest_pred.csv",
        dtype=str,
        keep_default_na=False,
    )
    roi = roi[
        (roi["roi_branch"] == "pred")
        & (~roi["duplicate_excluded"].str.casefold().eq("true"))
    ].copy()
    if args.max_rows:
        roi = roi.head(args.max_rows).copy()
    index = pd.read_csv(
        root / "manifests/api_adverse_lesion_cave_v1/segmentation_dataset_index.csv",
        dtype=str,
        keep_default_na=False,
    )
    predictions = pd.read_csv(
        root / "manifests/api_adverse_lesion_cave_v1/segmentation_prediction_index.csv",
        dtype=str,
        keep_default_na=False,
    )
    metrics = pd.read_csv(
        root / "reports/api_adverse_lesion_cave_v1/segmentation_metrics.csv",
        dtype=str,
        keep_default_na=False,
    )
    merged = (
        roi.merge(
            index[
                [
                    "phase_uid",
                    "sample_uid",
                    "sample_path",
                    "lesion_area_ratio",
                ]
            ],
            on="phase_uid",
            how="left",
            validate="one_to_one",
        )
        .merge(
            predictions[["phase_uid", "mask_path"]].rename(
                columns={"mask_path": "saved_component_mask_path"}
            ),
            on="phase_uid",
            how="left",
            validate="one_to_one",
        )
        .merge(
            metrics[
                [
                    "sample_uid",
                    "coverage",
                    "on_target",
                    "dice",
                    "centroid_distance_pixels",
                    "bbox_iou",
                ]
            ].rename(
                columns={
                    "coverage": "reported_1p5_coverage",
                    "on_target": "reported_on_target",
                    "dice": "reported_dice",
                    "centroid_distance_pixels": "reported_centroid_distance",
                    "bbox_iou": "reported_bbox_iou",
                }
            ),
            on="sample_uid",
            how="left",
            validate="one_to_one",
        )
    )

    rows: list[dict[str, object]] = []
    for record in merged.to_dict("records"):
        raw = np.load(record["sample_path"], allow_pickle=False)
        model_mask = raw["mask"].astype(np.float32)
        gt = (
            restore_model_probability(record["sample_path"], model_mask) >= 0.5
        ).astype(np.uint8)
        context = raw["context"].astype(np.uint8)
        image = raw["image"].astype(np.float32) / 255.0

        coverage_1p5, target_1p5, area_1p5 = clipped_box_coverage(
            gt, parse_box(record["source_expanded_bbox_1p5"])
        )
        coverage_3p0, target_3p0, area_3p0 = clipped_box_coverage(
            gt, parse_box(record["expanded_bbox"])
        )

        roi_component = cv2.imread(
            record["roi_mask_path"], cv2.IMREAD_GRAYSCALE
        )
        saved_component = cv2.imread(
            record["saved_component_mask_path"], cv2.IMREAD_GRAYSCALE
        )
        if roi_component is None or saved_component is None:
            raise FileNotFoundError(
                record["roi_mask_path"]
                if roi_component is None
                else record["saved_component_mask_path"]
            )
        roi_component = roi_component > 0
        saved_component = saved_component > 0
        intersection = int(np.logical_and(roi_component, saved_component).sum())
        union = int(np.logical_or(roi_component, saved_component).sum())

        rows.append(
            {
                "phase_uid": record["phase_uid"],
                "sample_uid": record["sample_uid"],
                "patient_id": record["patient_id"],
                "split": record["split"],
                "phase": record["phase"],
                "annotation_grade": record["annotation_grade"],
                "annotation_layout": record["annotation_layout"],
                "prediction_kind": record["prediction_kind"],
                "fallback_type": record["fallback_type"],
                "segmentation_fold": record["segmentation_fold"],
                "lesion_area_ratio": float(record["lesion_area_ratio"]),
                "context_area_ratio": float(context.sum() / context.size),
                "roi_1p5_coverage": coverage_1p5,
                "roi_1p5_on_target": target_1p5,
                "roi_1p5_area_ratio": area_1p5,
                "roi_3p0_coverage": coverage_3p0,
                "roi_3p0_on_target": target_3p0,
                "roi_3p0_area_ratio": area_3p0,
                "reported_1p5_coverage": float(record["reported_1p5_coverage"]),
                "reported_on_target": float(record["reported_on_target"]),
                "reported_dice": float(record["reported_dice"]),
                "reported_centroid_distance": pd.to_numeric(
                    record["reported_centroid_distance"], errors="coerce"
                ),
                "reported_bbox_iou": float(record["reported_bbox_iou"]),
                "component_selection_changed": int(
                    not np.array_equal(roi_component, saved_component)
                ),
                "component_iou": float(intersection / max(union, 1)),
                "mean_channel_local_contrast_z": local_contrast(
                    image[0], model_mask
                ),
                "median_channel_local_contrast_z": local_contrast(
                    image[1], model_mask
                ),
                "max_enhancement_local_contrast_z": local_contrast(
                    image[2], model_mask
                ),
                "probability_path": record["probability_path"],
                "roi_mask_path": record["roi_mask_path"],
                "saved_component_mask_path": record[
                    "saved_component_mask_path"
                ],
                "sample_path": record["sample_path"],
                "segmentation_path": record["segmentation_path"],
                "expanded_bbox": record["expanded_bbox"],
                "source_expanded_bbox_1p5": record[
                    "source_expanded_bbox_1p5"
                ],
            }
        )

    frame = pd.DataFrame(rows)
    phase_path = report_root / "failure_mode_phase_audit.csv"
    frame.to_csv(phase_path, index=False)

    summary: dict[str, object] = {
        "version": "api_adverse_lesion_cave_fast_v1_failure_diagnosis_1",
        "rows": int(len(frame)),
        "all": grouped_summary(frame),
        "by_split_phase": {
            f"{split}|{phase}": grouped_summary(group)
            for (split, phase), group in frame.groupby(["split", "phase"])
        },
        "by_annotation_grade": {
            str(grade): grouped_summary(group)
            for grade, group in frame.groupby("annotation_grade")
        },
        "by_layout": {
            str(layout): grouped_summary(group)
            for layout, group in frame.groupby("annotation_layout")
        },
        "component_iou_below_0p5": int((frame["component_iou"] < 0.5).sum()),
        "component_iou_zero": int((frame["component_iou"] == 0.0).sum()),
        "reported_vs_recomputed_1p5_max_abs": float(
            np.max(
                np.abs(
                    frame["reported_1p5_coverage"]
                    - frame["roi_1p5_coverage"]
                )
            )
        ),
        "catastrophic_3p0_zero_coverage_examples": frame.loc[
            frame["roi_3p0_coverage"] == 0.0,
            [
                "patient_id",
                "phase_uid",
                "split",
                "phase",
                "annotation_grade",
                "annotation_layout",
                "reported_dice",
                "reported_centroid_distance",
                "lesion_area_ratio",
                "max_enhancement_local_contrast_z",
                "fallback_type",
            ],
        ]
        .head(100)
        .to_dict("records"),
        "outputs": {"phase_audit": str(phase_path)},
    }
    summary_path = report_root / "failure_mode_data_audit.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

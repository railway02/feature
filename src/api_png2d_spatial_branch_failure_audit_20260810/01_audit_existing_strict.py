#!/usr/bin/env python3
"""Read-only failure audit for the frozen v5 strict SegResNet models.

This script never writes into the v5 code/output/report roots.  It merges the
existing strict phase-level audit with the frozen mapping metadata, selects
informative cases, reruns only those cases with their correct outer-fold model,
and saves overlays plus derived localization diagnostics in a new report root.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch


DEFAULT_PROJECT = Path("/root/autodl-tmp/aneurysm")
DEFAULT_V5_CODE = DEFAULT_PROJECT / "code/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready"
DEFAULT_CONFIG = DEFAULT_PROJECT / "configs/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready_strict.json"
DEFAULT_OUTPUT = DEFAULT_PROJECT / "outputs/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready_strict"
DEFAULT_PHASE_CSV = DEFAULT_PROJECT / "reports/api_png2d_segresnet_cave_fusion_v5_strict_factorial_predroi_scalar/05_strict_outer_segmentation_phase_predictions.csv"
DEFAULT_MAPPING = DEFAULT_PROJECT / "manifests/api_png2d_gtmask_roi_cave_v2_complete_mapping_fullseq/roi_phase_manifest_eligible.csv"
DEFAULT_REPORT = DEFAULT_PROJECT / "reports/api_png2d_spatial_branch_failure_audit_20260810"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--source-output-root", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--phase-csv", default=str(DEFAULT_PHASE_CSV))
    ap.add_argument("--mapping-csv", default=str(DEFAULT_MAPPING))
    ap.add_argument("--v5-code-root", default=str(DEFAULT_V5_CODE))
    ap.add_argument("--report-root", default=str(DEFAULT_REPORT))
    ap.add_argument("--device", default="cuda:0")
    return ap.parse_args()


def add_category(store, frame, category, n, ascending=True):
    chosen = frame.sort_values(
        "dice" if category not in {"largest_overseg", "largest_underseg", "smallest_gt"} else {
            "largest_overseg": "pred_gt_area_ratio",
            "largest_underseg": "pred_gt_area_ratio",
            "smallest_gt": "gt_pixels",
        }[category],
        ascending=ascending,
    ).head(n)
    for idx in chosen.index:
        store[int(idx)].add(category)


def select_cases(frame: pd.DataFrame) -> pd.DataFrame:
    categories: dict[int, set[str]] = defaultdict(set)
    add_category(categories, frame, "lowest_dice", 14, ascending=True)
    add_category(categories, frame, "highest_dice", 8, ascending=False)
    add_category(categories, frame, "largest_overseg", 10, ascending=False)
    add_category(categories, frame, "largest_underseg", 8, ascending=True)
    add_category(categories, frame, "smallest_gt", 10, ascending=True)

    fold4_post = frame[(frame["fold"] == 4) & (frame["phase"] == "Post")]
    add_category(categories, fold4_post, "fold4_post_low", 10, ascending=True)

    large_cut = frame["gt_pixels"].quantile(0.75)
    large_post_fail = frame[
        (frame["phase"] == "Post")
        & (frame["gt_pixels"] >= large_cut)
        & (frame["dice"] < 0.2)
    ]
    add_category(categories, large_post_fail, "large_post_failure", 10, ascending=True)

    # Pair the lowest cases with their opposite phase to expose phase-specific
    # success/failure within exactly the same patient/series.
    lowest = frame.sort_values("dice").head(12)
    pair_lookup = set(zip(frame["series_uid"], frame["phase"]))
    for row in lowest.itertuples():
        other_phase = "Post" if row.phase == "Pre" else "Pre"
        key = (row.series_uid, other_phase)
        if key in pair_lookup:
            other_idx = int(frame[
                (frame["series_uid"] == row.series_uid) & (frame["phase"] == other_phase)
            ].index[0])
            categories[other_idx].add("counterpart_of_low")

    selected = frame.loc[sorted(categories)].copy()
    selected["selection_categories"] = [
        "|".join(sorted(categories[int(idx)])) for idx in selected.index
    ]
    return selected.reset_index(names="source_row")


def connected_components(binary: np.ndarray):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return 0, np.zeros_like(binary, dtype=np.uint8), 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    largest = (labels == largest_label).astype(np.uint8)
    return int(n - 1), largest, float(areas.max() / max(1, areas.sum()))


def dice(binary: np.ndarray, target: np.ndarray) -> float:
    inter = int(np.logical_and(binary, target).sum())
    denom = int(binary.sum()) + int(target.sum())
    return float((2.0 * inter + 1e-6) / (denom + 1e-6))


def centroid_distance(binary: np.ndarray, target: np.ndarray) -> float:
    def center(x):
        ys, xs = np.where(x > 0)
        if len(xs) == 0:
            return None
        return np.asarray([xs.mean(), ys.mean()], dtype=np.float64)

    a, b = center(binary), center(target)
    if a is None or b is None:
        return float("nan")
    h, w = target.shape
    return float(np.linalg.norm(a - b) / np.hypot(h, w))


def contour(image, binary, color, thickness=3):
    cs, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, cs, -1, color, thickness, lineType=cv2.LINE_AA)


def put_lines(image, lines, color=(255, 255, 255)):
    y = 28
    for line in lines:
        cv2.putText(
            image,
            str(line),
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(line),
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 25


def make_overlay(image, raw_mask, gt, prob, pred, row, derived):
    base = np.repeat((np.clip(image, 0, 1) * 255).astype(np.uint8)[..., None], 3, axis=2)

    panel_gt = base.copy()
    contour(panel_gt, raw_mask == 1, (0, 255, 255), 3)
    contour(panel_gt, raw_mask == 2, (255, 255, 0), 3)
    put_lines(panel_gt, ["GT: label1 yellow, label2 cyan"])

    panel_error = base.copy()
    alpha = np.zeros_like(panel_error)
    tp = (pred > 0) & (gt > 0)
    fp = (pred > 0) & (gt == 0)
    fn = (pred == 0) & (gt > 0)
    alpha[tp] = (0, 220, 0)
    alpha[fp] = (0, 0, 255)
    alpha[fn] = (255, 80, 0)
    active = tp | fp | fn
    panel_error[active] = (0.42 * panel_error[active] + 0.58 * alpha[active]).astype(np.uint8)
    contour(panel_error, gt, (0, 255, 0), 2)
    contour(panel_error, pred, (0, 0, 255), 2)
    put_lines(panel_error, ["TP green / FP red / FN blue"])

    heat = cv2.applyColorMap((np.clip(prob, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    panel_prob = cv2.addWeighted(base, 0.48, heat, 0.52, 0)
    contour(panel_prob, gt, (0, 255, 0), 3)
    put_lines(panel_prob, ["Pred probability + GT contour"])

    union = (gt > 0) | (pred > 0)
    ys, xs = np.where(union)
    panel_crop = panel_error.copy()
    if len(xs):
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        side = max(x1 - x0, y1 - y0, 192)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        half = int(side * 0.7)
        x0, x1 = max(0, cx - half), min(base.shape[1], cx + half)
        y0, y1 = max(0, cy - half), min(base.shape[0], cy + half)
        crop = panel_error[y0:y1, x0:x1]
        if crop.size:
            panel_crop = cv2.resize(crop, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
    put_lines(panel_crop, ["Union crop"])

    title = np.zeros((112, base.shape[1] * 2, 3), dtype=np.uint8)
    put_lines(
        title,
        [
            f"fold={int(row.fold)} patient={row.patient_id} phase={row.phase} categories={row.selection_categories}",
            f"Dice={derived['dice_recomputed']:.4f} LCC={derived['dice_lcc']:.4f} GT={int(gt.sum())} Pred={int(pred.sum())} ratio={pred.sum()/max(1,gt.sum()):.3f}",
            f"pred_components={derived['pred_components']} largest_ratio={derived['pred_largest_component_ratio']:.3f} centroid_dist={derived['centroid_distance_norm']:.3f}",
        ],
    )
    top = np.concatenate([panel_gt, panel_error], axis=1)
    bottom = np.concatenate([panel_prob, panel_crop], axis=1)
    return np.concatenate([title, top, bottom], axis=0)


def main():
    args = parse_args()
    v5_code = Path(args.v5_code_root).resolve()
    sys.path.insert(0, str(v5_code))
    from common import load_config  # pylint: disable=import-error,import-outside-toplevel
    from data import prepare_pair  # pylint: disable=import-error,import-outside-toplevel
    from segresnet_model import build_segresnet  # pylint: disable=import-error,import-outside-toplevel

    cfg = load_config(args.config)
    source = Path(args.source_output_root).resolve()
    report = Path(args.report_root).resolve()
    overlay_root = report / "overlays"
    overlay_root.mkdir(parents=True, exist_ok=True)

    phase = pd.read_csv(
        args.phase_csv,
        encoding="utf-8-sig",
        dtype={"patient_id": str, "series_uid": str},
    )
    mapping = pd.read_csv(
        args.mapping_csv,
        dtype={"patient_id": str, "series_uid": str},
        low_memory=False,
    )
    mapping["phase"] = mapping["phase"].str.capitalize()
    metadata_columns = [
        "series_uid", "phase", "png_key", "reference_image_path", "mask_path",
        "frame_height", "frame_width", "n_frames", "mapping_score",
        "orientation_status", "mask_resized_to_frame", "resize_scale_x", "resize_scale_y",
        "mask_area_ratio", "bbox_width_ratio", "bbox_height_ratio", "bbox_aspect_ratio",
        "bbox_fill_ratio", "centroid_x_ratio", "centroid_y_ratio", "circularity",
        "solidity", "component_count", "largest_component_ratio", "labels_present",
    ]
    enriched = phase.merge(
        mapping[metadata_columns],
        on=["series_uid", "phase"],
        how="left",
        validate="one_to_one",
    )
    if enriched["png_key"].isna().any():
        raise RuntimeError("Strict phase rows did not merge one-to-one with mapping metadata")
    enriched["aspect_ratio"] = enriched["frame_width"] / enriched["frame_height"]
    enriched["dice_band"] = pd.cut(
        enriched["dice"], [-np.inf, 0.2, 0.5, 0.8, np.inf],
        labels=["<0.2", "0.2-0.5", "0.5-0.8", ">=0.8"], right=False,
    )
    enriched["gt_area_quartile"] = pd.qcut(
        enriched["gt_pixels"], 4, labels=["Q1-smallest", "Q2", "Q3", "Q4-largest"]
    )
    enriched.to_csv(report / "01_enriched_phase_audit.csv", index=False)

    phase_area = enriched.groupby(
        ["phase", "gt_area_quartile"], observed=True
    ).agg(
        n=("dice", "size"),
        dice_mean=("dice", "mean"),
        dice_median=("dice", "median"),
        failure_lt_02=("dice", lambda x: float((x < 0.2).mean())),
        failure_lt_05=("dice", lambda x: float((x < 0.5).mean())),
        area_ratio_mean=("pred_gt_area_ratio", "mean"),
        area_ratio_median=("pred_gt_area_ratio", "median"),
        gt_pixels_median=("gt_pixels", "median"),
    ).reset_index()
    phase_area.to_csv(report / "02_phase_by_gt_area_quartile.csv", index=False)

    shape_summary = enriched.groupby(
        ["frame_height", "frame_width", "aspect_ratio"], observed=True
    ).agg(n=("dice", "size"), dice_mean=("dice", "mean")).reset_index()
    shape_summary.to_csv(report / "03_image_shape_summary.csv", index=False)

    correlations = enriched.select_dtypes(include=[np.number]).corr()["dice"].sort_values()
    correlations.rename("pearson_with_dice").to_csv(report / "04_numeric_correlations_with_dice.csv")

    selected = select_cases(enriched)
    selected.to_csv(report / "05_selected_cases_before_inference.csv", index=False)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    models = {}
    inference_rows = []
    for row in selected.itertuples(index=False):
        fold = int(row.fold)
        if fold not in models:
            checkpoint = source / "segmentation" / f"fold_{fold}" / "model.pt"
            raw = torch.load(checkpoint, map_location="cpu")
            model = build_segresnet(cfg)
            model.load_state_dict(raw["state_dict"], strict=True)
            model.to(device).eval()
            models[fold] = model

        image, gt_float = prepare_pair(row.reference_image_path, row.mask_path, cfg)
        gt = (gt_float > 0).astype(np.uint8)
        original_mask = cv2.imread(row.mask_path, cv2.IMREAD_GRAYSCALE)
        original_mask = cv2.resize(
            original_mask,
            (gt.shape[1], gt.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        x = torch.from_numpy(image[None, None]).float().to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            prob = torch.sigmoid(models[fold](x))[0, 0].float().cpu().numpy()
        pred = (prob >= 0.5).astype(np.uint8)
        pred_components, largest, largest_ratio = connected_components(pred)
        gt_components, _, _ = connected_components(gt)
        derived = {
            "dice_recomputed": dice(pred, gt),
            "dice_lcc": dice(largest, gt),
            "pred_components": pred_components,
            "gt_components": gt_components,
            "pred_largest_component_ratio": largest_ratio,
            "centroid_distance_norm": centroid_distance(pred, gt),
            "mean_probability_inside_gt": float(prob[gt > 0].mean()),
            "mean_probability_outside_gt": float(prob[gt == 0].mean()),
            "soft_area_gt_ratio": float(prob.sum() / max(1, gt.sum())),
        }
        for threshold in np.arange(0.1, 1.0, 0.1):
            derived[f"dice_t{threshold:.1f}"] = dice(prob >= threshold, gt)

        inference_rows.append({**row._asdict(), **derived})
        overlay = make_overlay(image, original_mask, gt, prob, pred, row, derived)
        safe_categories = row.selection_categories.replace("|", "-")
        name = f"fold{fold}_{row.patient_id}_{row.phase}_{safe_categories}.png"
        if not cv2.imwrite(str(overlay_root / name), overlay):
            raise RuntimeError(f"Failed to write overlay {name}")

    inference = pd.DataFrame(inference_rows)
    inference.to_csv(report / "06_selected_case_inference_diagnostics.csv", index=False)
    if not np.allclose(inference["dice"], inference["dice_recomputed"], atol=2e-3):
        mismatch = inference.loc[
            ~np.isclose(inference["dice"], inference["dice_recomputed"], atol=2e-3),
            ["patient_id", "phase", "dice", "dice_recomputed"],
        ]
        raise RuntimeError(f"Recomputed selected-case Dice mismatch:\n{mismatch}")

    summary = {
        "status": "success",
        "read_only_source": True,
        "source_v5_output": str(source),
        "n_strict_phase_rows": int(len(enriched)),
        "n_selected_overlays": int(len(inference)),
        "all_aspect_ratios_equal_one": bool(np.allclose(enriched["aspect_ratio"], 1.0)),
        "dice_macro": float(enriched["dice"].mean()),
        "pre_dice_macro": float(enriched.loc[enriched.phase == "Pre", "dice"].mean()),
        "post_dice_macro": float(enriched.loc[enriched.phase == "Post", "dice"].mean()),
        "failure_lt_02": float((enriched["dice"] < 0.2).mean()),
        "failure_lt_05": float((enriched["dice"] < 0.5).mean()),
        "median_selected_centroid_distance_norm": float(inference["centroid_distance_norm"].median()),
        "selected_lcc_improved_fraction": float((inference["dice_lcc"] > inference["dice_recomputed"] + 1e-6).mean()),
    }
    (report / "07_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

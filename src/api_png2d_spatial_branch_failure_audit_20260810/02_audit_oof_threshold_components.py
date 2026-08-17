#!/usr/bin/env python3
"""Diagnostic-only threshold/component audit on frozen strict OOF predictions.

The outer-holdout rows are used only to explain the historical model's failure
modes.  Results from this script must not be used to select a future threshold,
loss, architecture, or post-processing rule.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT = Path("/root/autodl-tmp/aneurysm")
V5_CODE = PROJECT / "code/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready"
CONFIG = PROJECT / "configs/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready_strict.json"
SOURCE = PROJECT / "outputs/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready_strict"
REPORT = PROJECT / "reports/api_png2d_spatial_branch_failure_audit_20260810"
THRESHOLDS = tuple(np.round(np.arange(0.1, 1.0, 0.1), 1))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--source-output-root", default=str(SOURCE))
    ap.add_argument("--v5-code-root", default=str(V5_CODE))
    ap.add_argument("--report-root", default=str(REPORT))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--checkpoint-file", default="model.pt")
    ap.add_argument("--output-tag", default="final_refit")
    return ap.parse_args()


def largest_component(binary: np.ndarray):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return np.zeros_like(binary, dtype=np.uint8), 0, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    label = 1 + int(np.argmax(areas))
    return (
        (labels == label).astype(np.uint8),
        int(n - 1),
        float(areas.max() / max(1, areas.sum())),
    )


def stats(binary: np.ndarray, gt: np.ndarray):
    inter = int(np.logical_and(binary, gt).sum())
    pred_area = int(binary.sum())
    gt_area = int(gt.sum())
    dice = float((2 * inter + 1e-6) / (pred_area + gt_area + 1e-6))
    iou = float((inter + 1e-6) / (pred_area + gt_area - inter + 1e-6))
    return dice, iou, inter, pred_area, gt_area


def prepare_label_map(mask_path: str, cfg) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)
    target = int(cfg["spatial"]["input_size"])
    h, w = mask.shape
    if bool(cfg["spatial"].get("letterbox", True)):
        scale = min(target / h, target / w)
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        resized = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        top = (target - nh) // 2
        bottom = target - nh - top
        left = (target - nw) // 2
        right = target - nw - left
        return cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
        )
    return cv2.resize(mask, (target, target), interpolation=cv2.INTER_NEAREST)


def audit_fold(model, cases, cfg, device, fold, SegPhaseDataset):
    ds = SegPhaseDataset(cases, cfg, augment=False)
    dl = DataLoader(
        ds,
        batch_size=int(cfg["feature_extraction"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["feature_extraction"]["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(cfg["feature_extraction"]["num_workers"]) > 0,
    )
    rows = []
    amp = bool(cfg["feature_extraction"]["amp"] and device.type == "cuda")
    for x, y, indices in dl:
        image_batch = x.numpy()[:, 0]
        x = x.to(device, non_blocking=True)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
            probability = torch.sigmoid(model(x)).float().cpu().numpy()[:, 0]
        gt_batch = y.numpy()[:, 0] > 0
        for j, source_index in enumerate(indices.tolist()):
            meta = ds.rows.iloc[int(source_index)]
            prob = probability[j]
            gt = gt_batch[j]
            image = image_batch[j]
            label_map = prepare_label_map(str(meta["mask_path"]), cfg)
            label1 = label_map == 1
            label2 = label_map == 2
            if not np.array_equal(gt, label1 | label2):
                raise RuntimeError(f"Nonzero-union mismatch for {meta['mask_path']}")
            ring = cv2.dilate(
                gt.astype(np.uint8), np.ones((25, 25), np.uint8), iterations=1
            ).astype(bool) & ~gt
            pred05 = prob >= 0.5
            row = {
                "fold": int(fold),
                "series_uid": str(meta["series_uid"]),
                "patient_id": str(meta["patient_id"]),
                "phase": str(meta["phase"]),
                "mean_probability_inside_gt": float(prob[gt].mean()),
                "mean_probability_outside_gt": float(prob[~gt].mean()),
                "soft_area_gt_ratio": float(prob.sum() / max(1, gt.sum())),
                "label1_pixels": int(label1.sum()),
                "label2_pixels": int(label2.sum()),
                "label2_fraction_of_union": float(label2.sum() / max(1, gt.sum())),
                "recall_label1_t0.5": float(np.logical_and(pred05, label1).sum() / max(1, label1.sum())),
                "recall_label2_t0.5": float(np.logical_and(pred05, label2).sum() / max(1, label2.sum())),
                "mean_probability_label1": float(prob[label1].mean()) if label1.any() else float("nan"),
                "mean_probability_label2": float(prob[label2].mean()) if label2.any() else float("nan"),
                "mean_intensity_union": float(image[gt].mean()),
                "std_intensity_union": float(image[gt].std()),
                "mean_intensity_label1": float(image[label1].mean()) if label1.any() else float("nan"),
                "mean_intensity_label2": float(image[label2].mean()) if label2.any() else float("nan"),
                "mean_intensity_local_ring": float(image[ring].mean()) if ring.any() else float("nan"),
                "local_dark_contrast_union": float(image[ring].mean() - image[gt].mean()) if ring.any() else float("nan"),
            }
            for threshold in THRESHOLDS:
                d, iou, inter, pred_area, gt_area = stats(prob >= threshold, gt)
                tag = f"t{threshold:.1f}"
                row[f"dice_{tag}"] = d
                row[f"iou_{tag}"] = iou
                row[f"inter_{tag}"] = inter
                row[f"pred_pixels_{tag}"] = pred_area
                row[f"gt_pixels_{tag}"] = gt_area

            hard = prob >= 0.5
            lcc, components, largest_ratio = largest_component(hard)
            d, iou, inter, pred_area, gt_area = stats(lcc, gt)
            row.update({
                "pred_components_t0.5": components,
                "pred_largest_component_ratio_t0.5": largest_ratio,
                "dice_lcc_t0.5": d,
                "iou_lcc_t0.5": iou,
                "inter_lcc_t0.5": inter,
                "pred_pixels_lcc_t0.5": pred_area,
                "gt_pixels_lcc_t0.5": gt_area,
            })
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame):
    records = []
    group_defs = [("Overall", "Overall", rows)]
    group_defs.extend((f"fold_{fold}", "Overall", rows[rows.fold == fold]) for fold in sorted(rows.fold.unique()))
    group_defs.extend(("All_folds", phase, rows[rows.phase == phase]) for phase in ("Pre", "Post"))
    for fold in sorted(rows.fold.unique()):
        for phase in ("Pre", "Post"):
            group_defs.append((f"fold_{fold}", phase, rows[(rows.fold == fold) & (rows.phase == phase)]))

    for group, phase, subset in group_defs:
        for threshold in THRESHOLDS:
            tag = f"t{threshold:.1f}"
            inter = subset[f"inter_{tag}"].sum()
            pred = subset[f"pred_pixels_{tag}"].sum()
            gt = subset[f"gt_pixels_{tag}"].sum()
            records.append({
                "group": group,
                "phase": phase,
                "method": "threshold",
                "threshold": threshold,
                "n": int(len(subset)),
                "macro_dice": float(subset[f"dice_{tag}"].mean()),
                "micro_dice": float((2 * inter + 1e-6) / (pred + gt + 1e-6)),
                "macro_iou": float(subset[f"iou_{tag}"].mean()),
                "pred_gt_area_ratio_total": float(pred / max(1, gt)),
                "failure_lt_02": float((subset[f"dice_{tag}"] < 0.2).mean()),
                "failure_lt_05": float((subset[f"dice_{tag}"] < 0.5).mean()),
            })

        inter = subset["inter_lcc_t0.5"].sum()
        pred = subset["pred_pixels_lcc_t0.5"].sum()
        gt = subset["gt_pixels_lcc_t0.5"].sum()
        records.append({
            "group": group,
            "phase": phase,
            "method": "largest_component",
            "threshold": 0.5,
            "n": int(len(subset)),
            "macro_dice": float(subset["dice_lcc_t0.5"].mean()),
            "micro_dice": float((2 * inter + 1e-6) / (pred + gt + 1e-6)),
            "macro_iou": float(subset["iou_lcc_t0.5"].mean()),
            "pred_gt_area_ratio_total": float(pred / max(1, gt)),
            "failure_lt_02": float((subset["dice_lcc_t0.5"] < 0.2).mean()),
            "failure_lt_05": float((subset["dice_lcc_t0.5"] < 0.5).mean()),
        })
    return pd.DataFrame(records)


def main():
    args = parse_args()
    sys.path.insert(0, str(Path(args.v5_code_root).resolve()))
    from common import load_config  # pylint: disable=import-error,import-outside-toplevel
    from data import SegPhaseDataset  # pylint: disable=import-error,import-outside-toplevel
    from segresnet_model import build_segresnet  # pylint: disable=import-error,import-outside-toplevel

    cfg = load_config(args.config)
    source = Path(args.source_output_root).resolve()
    report = Path(args.report_root).resolve()
    report.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(
        source / "case_manifest.csv", dtype={"patient_id": str, "series_uid": str}
    )
    train = manifest[manifest.split == "Train"].copy()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    fold_rows = []
    for fold in range(1, 6):
        checkpoint = source / "segmentation" / f"fold_{fold}" / args.checkpoint_file
        raw = torch.load(checkpoint, map_location="cpu")
        model = build_segresnet(cfg)
        model.load_state_dict(raw["state_dict"], strict=True)
        model.to(device).eval()
        holdout = train[train.fold.astype(int) == fold].copy()
        fold_rows.append(audit_fold(model, holdout, cfg, device, fold, SegPhaseDataset))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows = pd.concat(fold_rows, ignore_index=True)
    rows.to_csv(report / f"08_{args.output_tag}_oof_threshold_component_case_diagnostics.csv", index=False)
    summary = summarize(rows)
    summary.to_csv(report / f"09_{args.output_tag}_oof_threshold_component_summary.csv", index=False)

    overall = summary[(summary.group == "Overall") & (summary.phase == "Overall")]
    baseline = overall[(overall.method == "threshold") & (overall.threshold == 0.5)].iloc[0]
    diagnostic_best = overall[overall.method == "threshold"].sort_values("macro_dice", ascending=False).iloc[0]
    lcc = overall[overall.method == "largest_component"].iloc[0]
    result = {
        "status": "success",
        "diagnostic_only_do_not_select_on_outer_holdout": True,
        "checkpoint_file": args.checkpoint_file,
        "output_tag": args.output_tag,
        "n": int(len(rows)),
        "baseline_threshold_0.5_macro_dice": float(baseline.macro_dice),
        "diagnostic_best_threshold": float(diagnostic_best.threshold),
        "diagnostic_best_macro_dice": float(diagnostic_best.macro_dice),
        "diagnostic_threshold_gain": float(diagnostic_best.macro_dice - baseline.macro_dice),
        "largest_component_macro_dice": float(lcc.macro_dice),
        "largest_component_gain": float(lcc.macro_dice - baseline.macro_dice),
        "largest_component_failure_lt_02": float(lcc.failure_lt_02),
        "baseline_failure_lt_02": float(baseline.failure_lt_02),
    }
    (report / f"10_{args.output_tag}_oof_threshold_component_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

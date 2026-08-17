#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path("/root/autodl-tmp/aneurysm")
IMAGE_SIZE = 1024
METRIC_SIZE = 512
CONTEXT_FRACTION = 0.40
THRESHOLDS = [round(value, 2) for value in np.arange(0.05, 0.56, 0.05)]
VIEW_ID = "reference_upper_exact_frame_median_enhancement_phase_v1"


def load_legacy_module():
    path = ROOT / "code/api_adverse_lesion_record_v2/11_train_segmentation_pilot.py"
    spec = importlib.util.spec_from_file_location("record_v2_segmentation_pilot_legacy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


LEGACY = load_legacy_module()


def atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(temporary, path)


def atomic_torch(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


class ReferenceLocalizer(LEGACY.UNetPilot):
    def __init__(self, in_channels: int = 4, base: int = 32):
        super().__init__(in_channels=in_channels, base=base)
        self.out = nn.Conv2d(base, 2, 1)


def gaussian_center_targets(mask: torch.Tensor, output_size: int) -> torch.Tensor:
    batch, _channels, height, width = mask.shape
    device, dtype = mask.device, mask.dtype
    y_axis = torch.arange(output_size, device=device, dtype=dtype)
    x_axis = torch.arange(output_size, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y_axis, x_axis, indexing="ij")
    output = torch.zeros((batch, 1, output_size, output_size), device=device, dtype=dtype)
    scale_x = output_size / float(width)
    scale_y = output_size / float(height)
    for index in range(batch):
        points = torch.nonzero(mask[index, 0] > 0.5, as_tuple=False)
        if points.numel() == 0:
            continue
        y0, x0 = points.min(dim=0).values
        y1, x1 = points.max(dim=0).values
        center_y = points[:, 0].float().mean() * scale_y
        center_x = points[:, 1].float().mean() * scale_x
        lesion_side = torch.maximum((x1 - x0 + 1).float() * scale_x, (y1 - y0 + 1).float() * scale_y)
        sigma = torch.clamp(0.45 * lesion_side, min=1.5, max=10.0)
        heatmap = torch.exp(-((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2) / (2.0 * sigma ** 2))
        peak_y = int(torch.clamp(torch.round(center_y), 0, output_size - 1).item())
        peak_x = int(torch.clamp(torch.round(center_x), 0, output_size - 1).item())
        heatmap[peak_y, peak_x] = 1.0
        output[index, 0] = heatmap
    return output


class MaskCenterLoss(nn.Module):
    def __init__(self, pos_weight: float, center_weight: float = 1.0):
        super().__init__()
        self.mask_loss = LEGACY.FocalTverskyBCE(
            pos_weight=pos_weight,
            alpha=0.35,
            beta=0.65,
        )
        self.center_weight = float(center_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        mask_logits = logits[:, :1]
        center_logits = F.avg_pool2d(logits[:, 1:2], kernel_size=8, stride=8)
        center_target = gaussian_center_targets(target, center_logits.shape[-1])
        probability = torch.sigmoid(center_logits).clamp(1e-5, 1.0 - 1e-5)
        center_bce = F.binary_cross_entropy_with_logits(center_logits, center_target, reduction="none")
        focal_weight = center_target * 64.0 * torch.pow(1.0 - probability, 2.0)
        focal_weight = focal_weight + (1.0 - center_target) * torch.pow(probability, 2.0)
        center_loss = (center_bce * focal_weight).mean()
        mask_loss = self.mask_loss(mask_logits, target)
        total = mask_loss + self.center_weight * center_loss
        return total, {
            "mask_loss": float(mask_loss.detach().item()),
            "center_loss": float(center_loss.detach().item()),
        }


def primary_component_guided(
    mask_probability: np.ndarray,
    center_probability: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, bool]:
    binary = (np.asarray(mask_probability, dtype=np.float32) >= threshold).astype(np.uint8)
    count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return np.zeros_like(binary), True
    best_label, best_score = 1, -math.inf
    for label in range(1, count):
        component = labels == label
        center_score = float(center_probability[component].max(initial=0.0))
        mask_mass = float(mask_probability[component].sum(dtype=np.float64))
        score = center_score + 1e-4 * math.log1p(mask_mass)
        if score > best_score:
            best_label, best_score = label, score
    return (labels == best_label).astype(np.uint8), False


def one_metric(
    mask_probability: np.ndarray,
    center_probability: np.ndarray,
    target: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    mask_probability = np.asarray(mask_probability, dtype=np.float32)
    center_probability = np.asarray(center_probability, dtype=np.float32)
    target = np.asarray(target, dtype=np.uint8)
    if target.shape != (METRIC_SIZE, METRIC_SIZE):
        target = cv2.resize(target, (METRIC_SIZE, METRIC_SIZE), interpolation=cv2.INTER_NEAREST)
    if mask_probability.shape != target.shape:
        mask_probability = cv2.resize(mask_probability, (METRIC_SIZE, METRIC_SIZE), interpolation=cv2.INTER_LINEAR)
    if center_probability.shape != target.shape:
        center_probability = cv2.resize(center_probability, (METRIC_SIZE, METRIC_SIZE), interpolation=cv2.INTER_LINEAR)

    center_y, center_x = np.unravel_index(int(np.argmax(center_probability)), center_probability.shape)
    component, empty = primary_component_guided(mask_probability, center_probability, threshold)
    intersection = int(np.logical_and(component > 0, target > 0).sum())
    lesion_total = max(int(target.sum()), 1)
    on_target = float(intersection > 0)
    center_in_target = float(target[center_y, center_x] > 0)

    height, width = target.shape
    side = max(1, int(round(CONTEXT_FRACTION * min(height, width))))
    x0 = int(round(center_x - side / 2.0))
    y0 = int(round(center_y - side / 2.0))
    x1, y1 = x0 + side, y0 + side
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    coverage = float(target[sy0:sy1, sx0:sx1].sum() / lesion_total)

    target_y, target_x = np.where(target > 0)
    target_cx, target_cy = float(target_x.mean()), float(target_y.mean())
    centroid_distance = math.hypot(float(center_x) - target_cx, float(center_y) - target_cy)
    centroid_distance /= math.hypot(width, height)
    dice = (2.0 * intersection + 1.0) / (int(component.sum()) + lesion_total + 1.0)
    return {
        "empty": float(empty),
        "on_target": on_target,
        "center_in_target": center_in_target,
        "coverage": coverage,
        "coverage95": float(coverage >= 0.95),
        "zero_coverage": float(coverage <= 0.0),
        "roi_area_ratio": float(side * side / (height * width)),
        "centroid_distance_normalized": float(centroid_distance),
        "dice": float(dice),
    }


def predict(
    model: nn.Module,
    loader,
    device: torch.device,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    model.eval()
    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with torch.inference_mode():
        for image, _target, uids in loader:
            image = image.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(image)
                probability = torch.sigmoid(logits)
                probability = F.interpolate(
                    probability,
                    size=(METRIC_SIZE, METRIC_SIZE),
                    mode="bilinear",
                    align_corners=False,
                )
            values = probability.detach().cpu().numpy().astype(np.float16)
            for uid, value in zip(uids, values):
                output[str(uid)] = (value[0], value[1])
    return output


def evaluate_thresholds(
    frame: pd.DataFrame,
    probabilities: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, float], pd.DataFrame]:
    targets = {}
    for row in frame.itertuples(index=False):
        target = np.load(row.sample_path, allow_pickle=False)["mask"].astype(np.uint8)
        targets[str(row.sample_uid)] = target
    rows = []
    for threshold in THRESHOLDS:
        metrics = []
        for uid in frame.sample_uid.astype(str):
            mask_probability, center_probability = probabilities[uid]
            metrics.append(one_metric(mask_probability, center_probability, targets[uid], threshold))
        average = {key: float(np.mean([item[key] for item in metrics])) for key in metrics[0]}
        score = (
            3.0 * average["coverage95"]
            + average["coverage"]
            + average["on_target"]
            + 0.5 * average["center_in_target"]
            - 3.0 * average["zero_coverage"]
            - average["empty"]
        )
        rows.append({"threshold": float(threshold), "score": float(score), **average})
    table = pd.DataFrame(rows)
    best = table.sort_values(
        ["score", "coverage95", "zero_coverage", "on_target", "empty", "dice"],
        ascending=[False, False, True, False, True, False],
    ).iloc[0].to_dict()
    return best, table


def select_balanced(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    if maximum <= 0 or len(frame) <= maximum:
        return frame.reset_index(drop=True)
    ordered = frame.sort_values(
        ["phase", "lesion_area_ratio_model", "patient_id", "series_uid"]
    ).reset_index(drop=True)
    positions = np.linspace(0, len(ordered) - 1, maximum, dtype=int)
    return ordered.iloc[np.unique(positions)].reset_index(drop=True)


def subgroup_summary(frame: pd.DataFrame) -> dict:
    metrics = [
        "on_target",
        "center_in_target",
        "coverage",
        "coverage95",
        "zero_coverage",
        "empty",
        "centroid_distance_normalized",
        "dice",
    ]
    result = {}
    for group_name in ["phase", "lesion_size_quartile"]:
        result[group_name] = {}
        for value, group in frame.groupby(group_name, dropna=False):
            result[group_name][str(value)] = {
                "n": int(len(group)),
                **{column: float(group[column].mean()) for column in metrics},
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-train", type=int, default=400)
    args = parser.parse_args()

    LEGACY.set_seed(42)
    manifest_root = ROOT / "manifests/api_adverse_lesion_record_v2"
    report_root = ROOT / "reports/api_adverse_lesion_record_v2"
    index = pd.read_csv(
        manifest_root / "reference_frame_upper_pilot_index.csv",
        dtype={"patient_id": str, "sample_uid": str},
    )
    index["fold"] = pd.to_numeric(index["fold"], errors="raise").astype(int)
    train = index[index.fold != 1].copy().reset_index(drop=True)
    holdout = index[index.fold == 1].copy().reset_index(drop=True)
    if set(train.patient_id) & set(holdout.patient_id):
        raise AssertionError("Patient leakage in reference-frame Pilot")

    if args.smoke:
        train = train.head(8).copy()
        holdout = holdout.head(4).copy()
        epochs = 1
        workers = 0
        output_root = ROOT / f"outputs/api_adverse_lesion_record_v2/reference_localizer_upper_smoke_b{args.batch_size}"
        report_prefix = f"reference_localizer_upper_smoke_b{args.batch_size}"
    else:
        train = select_balanced(train, int(args.max_train))
        epochs = int(args.epochs)
        workers = int(args.workers)
        output_root = ROOT / "outputs/api_adverse_lesion_record_v2/reference_localizer_upper_model"
        report_prefix = "reference_localizer_upper"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ReferenceLocalizer(4, 32).to(device)
    pos_weight = LEGACY.compute_pos_weight(train, cap=40.0)
    criterion = MaskCenterLoss(pos_weight=pos_weight, center_weight=1.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    train_loader = LEGACY.make_loader(train, int(args.batch_size), True, 42, True, workers)
    holdout_loader = LEGACY.make_loader(holdout, int(args.batch_size), False, 42, False, workers)

    best_score = -math.inf
    best_epoch = 0
    best_threshold = 0.20
    wait = 0
    history = []
    checkpoint = output_root / "best.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        losses, mask_losses, center_losses = [], [], []
        for image, target, _uids in train_loader:
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                loss, parts = criterion(model(image), target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().item()))
            mask_losses.append(parts["mask_loss"])
            center_losses.append(parts["center_loss"])

        probabilities = predict(model, holdout_loader, device)
        best, threshold_table = evaluate_thresholds(holdout, probabilities)
        row = {
            "epoch": int(epoch),
            "train_loss": float(np.mean(losses)),
            "train_mask_loss": float(np.mean(mask_losses)),
            "train_center_loss": float(np.mean(center_losses)),
            **{key: float(value) for key, value in best.items()},
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if row["score"] > best_score + 1e-6:
            best_score = row["score"]
            best_epoch = epoch
            best_threshold = row["threshold"]
            wait = 0
            atomic_torch(
                {
                    "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "epoch": int(epoch),
                    "threshold": float(best_threshold),
                    "metrics": row,
                    "base_channels": 32,
                    "in_channels": 4,
                    "output_channels": ["mask", "center_heatmap"],
                    "view_id": VIEW_ID,
                    "pos_weight": float(pos_weight),
                },
                checkpoint,
            )
            atomic_csv(threshold_table, output_root / "best_threshold_search.csv")
        else:
            wait += 1
        atomic_csv(pd.DataFrame(history), output_root / "history.csv")
        if not args.smoke and wait >= 5:
            break

    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    probabilities = predict(model, holdout_loader, device)
    best, threshold_table = evaluate_thresholds(holdout, probabilities)

    per_sample = []
    for row in holdout.itertuples(index=False):
        target = np.load(row.sample_path, allow_pickle=False)["mask"].astype(np.uint8)
        mask_probability, center_probability = probabilities[str(row.sample_uid)]
        metrics = one_metric(mask_probability, center_probability, target, float(best["threshold"]))
        per_sample.append(
            {
                "sample_uid": row.sample_uid,
                "record_uid": row.record_uid,
                "patient_id": row.patient_id,
                "series_uid": row.series_uid,
                "phase": row.phase,
                "annotation_grade": row.annotation_grade,
                "lesion_area_ratio": float(row.lesion_area_ratio_model),
                **metrics,
            }
        )
    per_frame = pd.DataFrame(per_sample)
    try:
        per_frame["lesion_size_quartile"] = pd.qcut(
            per_frame.lesion_area_ratio,
            4,
            labels=["Q1_smallest", "Q2", "Q3", "Q4_largest"],
            duplicates="drop",
        ).astype(str)
    except ValueError:
        per_frame["lesion_size_quartile"] = "unavailable"

    gate = bool(
        best["on_target"] >= 0.93
        and best["zero_coverage"] <= 0.02
        and best["empty"] <= 0.03
        and best["coverage"] >= 0.95
        and best["coverage95"] >= 0.95
        and best["roi_area_ratio"] <= 0.20
    )
    summary = {
        "status": "complete",
        "smoke": bool(args.smoke),
        "purpose": "A-grade exact annotation-reference localization upper bound",
        "model": "shared_pre_post_unet_base32_groupnorm_mask_plus_center_heatmap",
        "input": "exact_annotation_reference+temporal_median+reference_enhancement+phase_channel",
        "image_size": IMAGE_SIZE,
        "metric_size": METRIC_SIZE,
        "context_fraction": CONTEXT_FRACTION,
        "train_samples": int(len(train)),
        "train_patients": int(train.patient_id.nunique()),
        "holdout_samples": int(len(holdout)),
        "holdout_patients": int(holdout.patient_id.nunique()),
        "holdout_fold": 1,
        "best_epoch": int(best_epoch),
        "epochs_ran": int(len(history)),
        "physical_batch_size": int(args.batch_size),
        "workers": int(workers),
        "pos_weight": float(pos_weight),
        "selected_threshold": float(best["threshold"]),
        "metrics": {key: float(value) for key, value in best.items()},
        "subgroups": subgroup_summary(per_frame),
        "pilot_gate_passed": gate,
        "gate_targets": {
            "on_target_min": 0.93,
            "zero_coverage_max": 0.02,
            "empty_max": 0.03,
            "mean_coverage_min": 0.95,
            "coverage95_rate_min": 0.95,
            "roi_area_ratio_max": 0.20,
        },
        "label_semantics_status": "engineering_assumption_labels_2_to_6_pending_hospital_confirmation",
        "checkpoint": str(checkpoint),
    }
    atomic_csv(pd.DataFrame(history), report_root / f"{report_prefix}_history.csv")
    atomic_csv(threshold_table, report_root / f"{report_prefix}_threshold_search.csv")
    atomic_csv(per_frame, report_root / f"{report_prefix}_holdout_metrics.csv")
    atomic_json(summary, report_root / f"{report_prefix}_summary.json")
    if not args.smoke:
        marker = report_root / (".REFERENCE_FRAME_UPPER_PASS" if gate else ".REFERENCE_FRAME_UPPER_NO_PASS")
        marker.write_text("pass\n" if gate else "no_pass\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

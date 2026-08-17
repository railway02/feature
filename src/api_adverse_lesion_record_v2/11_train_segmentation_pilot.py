#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path("/root/autodl-tmp/aneurysm")
IMAGE_SIZE = 1024
CONTEXT_FRACTION = 0.40
THRESHOLDS = [round(value, 2) for value in np.arange(0.10, 0.51, 0.10)]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


class PilotDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, augment: bool, seed: int):
        self.frame = frame.reset_index(drop=True)
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        raw = np.load(row.sample_path, allow_pickle=False)
        image = raw["image"].astype(np.float32) / 255.0
        mask = raw["mask"].astype(np.float32)
        phase_value = 0.0 if str(row.phase).casefold() == "pre" else 1.0
        phase = np.full((1, image.shape[1], image.shape[2]), phase_value, dtype=np.float32)
        image = np.concatenate([image, phase], axis=0)
        if self.augment:
            rng = np.random.default_rng(self.seed + index + random.randint(0, 1_000_000))
            if rng.random() < 0.5:
                image = image[:, :, ::-1].copy()
                mask = mask[:, ::-1].copy()
            gamma = float(rng.uniform(0.90, 1.10))
            image[:3] = np.clip(image[:3] ** gamma, 0.0, 1.0)
            if rng.random() < 0.3:
                image[:3] = np.clip(
                    image[:3] + rng.normal(0.0, 0.01, image[:3].shape),
                    0.0,
                    1.0,
                )
        return (
            torch.from_numpy(image),
            torch.from_numpy(mask[None]),
            str(row.sample_uid),
        )


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class UNetPilot(nn.Module):
    def __init__(self, in_channels: int = 4, base: int = 32):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.e1 = ConvBlock(in_channels, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.e4 = ConvBlock(base * 4, base * 8)
        self.b = ConvBlock(base * 8, base * 16)
        self.u4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)
        self.d4 = ConvBlock(base * 16, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.d3 = ConvBlock(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.d2 = ConvBlock(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.d1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    @staticmethod
    def join(up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([skip, up], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.b(self.pool(e4))
        d4 = self.d4(self.join(self.u4(b), e4))
        d3 = self.d3(self.join(self.u3(d4), e3))
        d2 = self.d2(self.join(self.u2(d3), e2))
        d1 = self.d1(self.join(self.u1(d2), e1))
        return self.out(d1)


class FocalTverskyBCE(nn.Module):
    def __init__(
        self,
        pos_weight: float,
        alpha: float = 0.30,
        beta: float = 0.70,
        centroid_weight: float = 0.0,
        area_weight: float = 0.0,
    ):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.centroid_weight = float(centroid_weight)
        self.area_weight = float(area_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)
        probability = torch.sigmoid(logits)
        dims = (1, 2, 3)
        tp = (probability * target).sum(dims)
        fp = (probability * (1.0 - target)).sum(dims)
        fn = ((1.0 - probability) * target).sum(dims)
        tversky = (tp + 1.0) / (tp + self.alpha * fp + self.beta * fn + 1.0)
        focal = torch.pow(1.0 - tversky, 0.75).mean()
        loss = 0.5 * bce + 0.5 * focal
        if self.centroid_weight > 0.0:
            pooled_logits = F.avg_pool2d(logits, kernel_size=8, stride=8)
            pooled_target = F.max_pool2d(target, kernel_size=8, stride=8)
            batch, _channels, height, width = pooled_logits.shape
            spatial = torch.softmax(pooled_logits.flatten(2) * 4.0, dim=-1)
            target_mass = pooled_target.flatten(2)
            target_mass = target_mass / target_mass.sum(dim=-1, keepdim=True).clamp_min(1.0)
            x = torch.linspace(0.0, 1.0, width, device=logits.device, dtype=logits.dtype)
            y = torch.linspace(0.0, 1.0, height, device=logits.device, dtype=logits.dtype)
            grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
            coordinates = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
            pred_center = torch.matmul(spatial, coordinates).reshape(batch, 2)
            target_center = torch.matmul(target_mass, coordinates).reshape(batch, 2)
            centroid = F.smooth_l1_loss(pred_center, target_center)
            loss = loss + self.centroid_weight * centroid
        if self.area_weight > 0.0:
            predicted_area = probability.mean(dims).clamp_min(1e-6)
            target_area = target.mean(dims).clamp_min(1e-6)
            area = F.smooth_l1_loss(torch.log(predicted_area), torch.log(target_area))
            loss = loss + self.area_weight * area
        return loss


def make_loader(
    frame: pd.DataFrame,
    batch_size: int,
    augment: bool,
    seed: int,
    shuffle: bool,
    workers: int,
) -> DataLoader:
    return DataLoader(
        PilotDataset(frame, augment, seed),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )


def primary_component(probability: np.ndarray, threshold: float) -> tuple[np.ndarray, bool]:
    probability = np.asarray(probability, dtype=np.float32)
    binary = (probability >= threshold).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return np.zeros_like(binary), True
    best_label, best_mass = 1, -math.inf
    for label in range(1, count):
        component = labels == label
        mass = float(probability[component].sum(dtype=np.float64))
        if mass > best_mass:
            best_label, best_mass = label, mass
    return (labels == best_label).astype(np.uint8), False


def one_metric(probability: np.ndarray, target: np.ndarray, threshold: float) -> dict[str, float]:
    probability = np.asarray(probability, dtype=np.float32)
    target = np.asarray(target, dtype=np.uint8)
    if max(target.shape) > 512:
        probability = cv2.resize(probability, (512, 512), interpolation=cv2.INTER_AREA)
        target = cv2.resize(target, (512, 512), interpolation=cv2.INTER_NEAREST)
    component, empty = primary_component(probability, threshold)
    height, width = target.shape
    if empty:
        y, x = np.unravel_index(int(np.argmax(probability)), probability.shape)
        bbox_w = bbox_h = 1
        cx, cy = float(x), float(y)
        on_target = 0.0
    else:
        ys, xs = np.where(component > 0)
        weights = probability[ys, xs].astype(np.float64)
        denominator = max(float(weights.sum()), 1e-8)
        cx = float((xs * weights).sum() / denominator)
        cy = float((ys * weights).sum() / denominator)
        bbox_w = int(xs.max() - xs.min() + 1)
        bbox_h = int(ys.max() - ys.min() + 1)
        on_target = float(np.logical_and(component > 0, target > 0).any())

    side = max(
        int(round(CONTEXT_FRACTION * min(height, width))),
        int(math.ceil(1.5 * max(bbox_w, bbox_h))),
    )
    side = min(side, max(height, width))
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x1, y1 = x0 + side, y0 + side
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    lesion_total = max(int(target.sum()), 1)
    covered = int(target[sy0:sy1, sx0:sx1].sum())
    coverage = covered / lesion_total
    target_y, target_x = np.where(target > 0)
    target_cx, target_cy = float(target_x.mean()), float(target_y.mean())
    centroid_distance = math.hypot(cx - target_cx, cy - target_cy) / math.hypot(width, height)
    intersection = int(np.logical_and(component > 0, target > 0).sum())
    dice = (2.0 * intersection + 1.0) / (int(component.sum()) + lesion_total + 1.0)
    return {
        "empty": float(empty),
        "on_target": on_target,
        "coverage": float(coverage),
        "coverage95": float(coverage >= 0.95),
        "zero_coverage": float(coverage <= 0.0),
        "roi_area_ratio": float(side * side / (height * width)),
        "centroid_distance_normalized": float(centroid_distance),
        "dice": float(dice),
    }


def evaluate_thresholds(
    frame: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    thresholds: list[float],
) -> tuple[dict[str, float], pd.DataFrame]:
    rows = []
    target_cache = {}
    for row in frame.itertuples(index=False):
        raw = np.load(row.sample_path, allow_pickle=False)
        target_cache[str(row.sample_uid)] = raw["mask"].astype(np.uint8)
    for threshold in thresholds:
        metrics = [
            one_metric(probabilities[uid], target_cache[uid], threshold)
            for uid in frame.sample_uid.astype(str)
        ]
        average = {key: float(np.mean([item[key] for item in metrics])) for key in metrics[0]}
        score = (
            2.0 * average["coverage95"]
            + average["on_target"]
            + average["coverage"]
            - 2.0 * average["zero_coverage"]
            - average["empty"]
            - 2.0 * max(average["roi_area_ratio"] - 0.20, 0.0)
        )
        rows.append({"threshold": float(threshold), "score": float(score), **average})
    table = pd.DataFrame(rows)
    best = table.sort_values(
        ["score", "coverage95", "on_target", "zero_coverage", "empty"],
        ascending=[False, False, False, True, True],
    ).iloc[0].to_dict()
    return best, table


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    output: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for image, _target, uids in loader:
            image = image.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                probability = torch.sigmoid(model(image))
            values = probability[:, 0].detach().cpu().numpy().astype(np.float16)
            for uid, value in zip(uids, values):
                output[str(uid)] = value
    return output


def compute_pos_weight(frame: pd.DataFrame, cap: float = 40.0) -> float:
    positive = float(frame.lesion_pixels_model.astype(float).sum())
    total = float(len(frame) * IMAGE_SIZE * IMAGE_SIZE)
    return float(min(max((total - positive) / max(positive, 1.0), 1.0), cap))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-holdout", type=int, default=166)
    parser.add_argument("--max-train", type=int, default=400)
    parser.add_argument("--variant", choices=["baseline", "centroid_fp"], default="centroid_fp")
    args = parser.parse_args()

    set_seed(42)
    manifest_root = ROOT / "manifests/api_adverse_lesion_record_v2"
    reports = ROOT / "reports/api_adverse_lesion_record_v2"
    index = pd.read_csv(
        manifest_root / "segmentation_pilot_p2_index.csv",
        dtype={"patient_id": str, "sample_uid": str},
    )
    index["fold"] = pd.to_numeric(index["fold"], errors="raise").astype(int)
    train = index[index.fold != 1].copy().reset_index(drop=True)
    holdout = index[index.fold == 1].copy().reset_index(drop=True)
    if set(train.patient_id) & set(holdout.patient_id):
        raise AssertionError("Patient leakage in segmentation pilot")
    variant = str(args.variant)
    if args.smoke:
        train = train.head(8).copy()
        holdout = holdout.head(4).copy()
        epochs = 1
        output_root = ROOT / f"outputs/api_adverse_lesion_record_v2/segmentation_pilot_p2_smoke_{variant}_b{args.batch_size}"
        report_prefix = f"segmentation_pilot_p2_smoke_{variant}_b{args.batch_size}"
        workers = 0
    else:
        epochs = int(args.epochs)
        suffix = "" if variant == "baseline" else f"_{variant}"
        output_root = ROOT / f"outputs/api_adverse_lesion_record_v2/segmentation_pilot_p2{suffix}_model"
        report_prefix = f"segmentation_pilot_p2{suffix}"
        workers = int(args.workers)

    if not args.smoke and args.max_train and len(train) > int(args.max_train):
        ordered = train.sort_values(
            ["phase", "annotation_grade", "lesion_area_ratio_model", "patient_id"]
        ).reset_index(drop=True)
        positions = np.linspace(0, len(ordered) - 1, int(args.max_train), dtype=int)
        train = ordered.iloc[np.unique(positions)].reset_index(drop=True)
    if not args.smoke and args.max_holdout and len(holdout) > int(args.max_holdout):
        ordered = holdout.sort_values(
            ["phase", "annotation_grade", "lesion_area_ratio_model", "patient_id"]
        ).reset_index(drop=True)
        positions = np.linspace(0, len(ordered) - 1, int(args.max_holdout), dtype=int)
        holdout = ordered.iloc[np.unique(positions)].reset_index(drop=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UNetPilot(4, 32).to(device)
    if variant == "centroid_fp":
        pos_weight = compute_pos_weight(train, cap=20.0)
        criterion = FocalTverskyBCE(
            pos_weight,
            alpha=0.60,
            beta=0.40,
            centroid_weight=0.50,
            area_weight=0.05,
        ).to(device)
    else:
        pos_weight = compute_pos_weight(train, cap=40.0)
        criterion = FocalTverskyBCE(pos_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    train_loader = make_loader(train, int(args.batch_size), True, 42, True, workers)
    holdout_loader = make_loader(holdout, int(args.batch_size), False, 42, False, workers)
    accumulation = max(1, math.ceil(4 / int(args.batch_size)))
    best_score = -math.inf
    best_epoch = 0
    best_threshold = 0.2
    wait = 0
    history = []
    checkpoint = output_root / "best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, (image, target, _uids) in enumerate(train_loader, 1):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                loss = criterion(model(image), target) / accumulation
            scaler.scale(loss).backward()
            losses.append(float(loss.item() * accumulation))
            if step % accumulation == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        probabilities = predict(model, holdout_loader, device)
        best, threshold_table = evaluate_thresholds(holdout, probabilities, THRESHOLDS)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "threshold": float(best["threshold"]),
            "score": float(best["score"]),
            "on_target": float(best["on_target"]),
            "coverage": float(best["coverage"]),
            "coverage95": float(best["coverage95"]),
            "zero_coverage": float(best["zero_coverage"]),
            "empty": float(best["empty"]),
            "roi_area_ratio": float(best["roi_area_ratio"]),
            "dice": float(best["dice"]),
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
                    "epoch": epoch,
                    "base_channels": 32,
                    "in_channels": 4,
                    "threshold": best_threshold,
                    "metrics": row,
                    "pos_weight": pos_weight,
                    "view_id": "p2_polarity_minip_median_q95q05_phase",
                    "loss_variant": variant,
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
    best, table = evaluate_thresholds(
        holdout,
        probabilities,
        [round(value, 2) for value in np.arange(0.05, 0.61, 0.05)],
    )
    per_sample = []
    for row in holdout.itertuples(index=False):
        target = np.load(row.sample_path, allow_pickle=False)["mask"].astype(np.uint8)
        metrics = one_metric(probabilities[str(row.sample_uid)], target, float(best["threshold"]))
        per_sample.append({
            "sample_uid": row.sample_uid,
            "record_uid": row.record_uid,
            "patient_id": row.patient_id,
            "series_uid": row.series_uid,
            "phase": row.phase,
            "annotation_grade": row.annotation_grade,
            "lesion_area_ratio": float(row.lesion_area_ratio_model),
            **metrics,
        })
    per_frame = pd.DataFrame(per_sample)
    gate = bool(
        best["on_target"] >= 0.93
        and best["zero_coverage"] <= 0.02
        and best["empty"] <= 0.03
        and best["coverage"] >= 0.95
        and best["roi_area_ratio"] <= 0.25
    )
    summary = {
        "status": "complete",
        "smoke": bool(args.smoke),
        "model": "shared_pre_post_unet_base32_groupnorm",
        "loss_variant": variant,
        "input": "polarity_minip+median+q95-q05+phase_channel",
        "image_size": IMAGE_SIZE,
        "metric_resolution": 512,
        "train_samples": len(train),
        "train_patients": int(train.patient_id.nunique()),
        "holdout_samples": len(holdout),
        "holdout_patients": int(holdout.patient_id.nunique()),
        "holdout_fold": 1,
        "best_epoch": int(best_epoch),
        "epochs_ran": len(history),
        "physical_batch_size": int(args.batch_size),
        "effective_batch_size": 4,
        "workers": workers,
        "pos_weight": float(pos_weight),
        "selected_threshold": float(best["threshold"]),
        "metrics": {key: float(value) for key, value in best.items()},
        "pilot_gate_passed": gate,
        "gate_targets": {
            "on_target_min": 0.93,
            "zero_coverage_max": 0.02,
            "empty_max": 0.03,
            "mean_coverage_min": 0.95,
            "roi_area_ratio_max": 0.25,
        },
        "checkpoint": str(checkpoint),
    }
    atomic_csv(pd.DataFrame(history), reports / f"{report_prefix}_history.csv")
    atomic_csv(table, reports / f"{report_prefix}_threshold_search.csv")
    atomic_csv(per_frame, reports / f"{report_prefix}_holdout_metrics.csv")
    atomic_json(summary, reports / f"{report_prefix}_summary.json")
    if not args.smoke:
        if variant == "baseline":
            marker_name = ".SEG_PILOT_PASS" if gate else ".SEG_PILOT_NO_PASS"
        else:
            marker_name = ".SEG_PILOT_CENTROID_PASS" if gate else ".SEG_PILOT_CENTROID_NO_PASS"
        marker = reports / marker_name
        marker.write_text("pass\n" if gate else "no_pass\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

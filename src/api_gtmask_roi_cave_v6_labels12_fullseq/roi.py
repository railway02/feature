from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def bbox_to_text(box: tuple[int, int, int, int]) -> str:
    return "|".join(str(int(value)) for value in box)


def bbox_from_text(value: str) -> tuple[int, int, int, int]:
    parts = tuple(int(float(item)) for item in str(value).split("|"))
    if len(parts) != 4:
        raise ValueError(f"无效 bbox：{value}")
    x0, y0, x1, y1 = parts
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"退化 bbox：{value}")
    return x0, y0, x1, y1


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    points = np.argwhere(np.asarray(mask, dtype=bool))
    if points.size == 0:
        raise ValueError("Mask 没有非零前景")
    y0, x0 = points.min(axis=0)
    y1, x1 = points.max(axis=0) + 1
    return int(x0), int(y0), int(x1), int(y1)


def round_up(value: float, multiple: int) -> int:
    if multiple <= 1:
        return int(math.ceil(value))
    return int(math.ceil(value / multiple) * multiple)


def square_from_center(cx: float, cy: float, side: int) -> tuple[int, int, int, int]:
    side = max(2, int(side))
    x0 = int(math.floor(cx - side / 2.0))
    y0 = int(math.floor(cy - side / 2.0))
    return x0, y0, x0 + side, y0 + side


def context_square_bbox(
    object_bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, int],
    *,
    bbox_factor: float = 1.5,
    min_side_pixels: int = 128,
    min_margin_pixels: int = 8,
    round_multiple: int = 32,
) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
    if bbox_factor < 1.0:
        raise ValueError("bbox_factor 必须 >= 1")
    h, w = map(int, frame_shape)
    x0, y0, x1, y1 = object_bbox
    bw, bh = x1 - x0, y1 - y0
    if bw <= 0 or bh <= 0:
        raise ValueError(f"无效目标框：{object_bbox}")
    object_side = max(bw, bh)
    desired = max(
        float(object_side) * float(bbox_factor),
        float(min_side_pixels),
        float(object_side + 2 * min_margin_pixels),
    )
    side = round_up(desired, int(round_multiple))
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    roi = square_from_center(cx, cy, side)
    rx0, ry0, rx1, ry1 = roi
    if rx0 > x0 or ry0 > y0 or rx1 < x1 or ry1 < y1:
        raise AssertionError(f"ROI 裁掉了 Mask：mask={object_bbox}, roi={roi}")
    audit = {
        "object_width": int(bw),
        "object_height": int(bh),
        "object_side": int(object_side),
        "roi_side": int(side),
        "bbox_factor": float(bbox_factor),
        "roi_area_ratio": float(side * side / max(h * w, 1)),
        "roi_exceeds_frame": bool(rx0 < 0 or ry0 < 0 or rx1 > w or ry1 > h),
    }
    return roi, audit


def crop_padding(box: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    h, w = map(int, shape)
    return max(0, -x0), max(0, -y0), max(0, x1 - w), max(0, y1 - h)


def crop_frame(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    if frame.ndim != 2:
        raise ValueError(f"单帧必须为二维，shape={frame.shape}")
    x0, y0, x1, y1 = box
    h, w = frame.shape
    left, top, right, bottom = crop_padding(box, frame.shape)
    crop = frame[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if left or top or right or bottom:
        border = np.concatenate((frame[0], frame[-1], frame[:, 0], frame[:, -1]))
        fill = float(np.median(border))
        crop = np.pad(crop, ((top, bottom), (left, right)), mode="constant", constant_values=fill)
    expected = (y1 - y0, x1 - x0)
    if crop.shape != expected:
        raise AssertionError(f"裁切尺寸错误：got={crop.shape}, expected={expected}, box={box}")
    return np.ascontiguousarray(crop)


def crop_frames(frames: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    frames = np.asarray(frames)
    if frames.ndim != 3:
        raise ValueError(f"序列必须为 [T,H,W]，shape={frames.shape}")
    return np.stack([crop_frame(frame, box) for frame in frames], axis=0)


def mask_statistics(labels: np.ndarray, foreground: np.ndarray) -> dict[str, Any]:
    binary = np.asarray(foreground, dtype=np.uint8)
    h, w = binary.shape
    x0, y0, x1, y1 = bbox_from_mask(binary)
    area = int(binary.sum())
    components, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    component_sizes = stats[1:, cv2.CC_STAT_AREA] if components > 1 else np.asarray([], dtype=int)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
    hull_area = 0.0
    if contours:
        points = np.concatenate(contours, axis=0)
        hull_area = float(cv2.contourArea(cv2.convexHull(points)))
    ys, xs = np.where(binary > 0)
    positive_labels, counts = np.unique(labels[foreground], return_counts=True)
    return {
        "foreground_pixels": area,
        "mask_area_ratio": float(area / max(h * w, 1)),
        "bbox_width_ratio": float((x1 - x0) / max(w, 1)),
        "bbox_height_ratio": float((y1 - y0) / max(h, 1)),
        "bbox_aspect_ratio": float((x1 - x0) / max(y1 - y0, 1)),
        "bbox_fill_ratio": float(area / max((x1 - x0) * (y1 - y0), 1)),
        "centroid_x_ratio": float(xs.mean() / max(w - 1, 1)),
        "centroid_y_ratio": float(ys.mean() / max(h - 1, 1)),
        "circularity": float(4.0 * math.pi * area / max(perimeter * perimeter, 1e-8)),
        "solidity": float(area / max(hull_area, 1.0)),
        "component_count": int(max(components - 1, 0)),
        "largest_component_ratio": float(component_sizes.max() / max(area, 1)) if len(component_sizes) else 0.0,
        "positive_labels": "|".join(map(str, positive_labels.astype(int).tolist())),
        "label_pixel_counts": "|".join(f"{int(v)}:{int(c)}" for v, c in zip(positive_labels, counts)),
    }

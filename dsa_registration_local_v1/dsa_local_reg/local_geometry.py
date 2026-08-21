from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class BBox:
    """Exclusive-end x/y box in the native phase canvas."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def as_text(self) -> str:
        return f"{self.x0}|{self.y0}|{self.x1}|{self.y1}"

    @classmethod
    def from_text(cls, value: str) -> "BBox":
        parts = tuple(int(float(item)) for item in str(value).split("|"))
        if len(parts) != 4:
            raise ValueError(f"Invalid bbox: {value!r}")
        box = cls(*parts)
        if box.width <= 0 or box.height <= 0:
            raise ValueError(f"Degenerate bbox: {value!r}")
        return box


@dataclass(frozen=True)
class CropResult:
    image: np.ndarray
    bbox: BBox
    padding_left: int
    padding_top: int
    padding_right: int
    padding_bottom: int
    valid_support: np.ndarray


def bbox_padding(box: BBox, shape_yx: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = map(int, shape_yx)
    return (
        max(0, -box.x0),
        max(0, -box.y0),
        max(0, box.x1 - width),
        max(0, box.y1 - height),
    )


def crop_with_border_median_padding(image: np.ndarray, box: BBox) -> CropResult:
    """Old Local-CAVE compatible crop: crop a phase bbox with border-median padding."""
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"Expected 2-D grayscale image, got {image.shape}")
    height, width = image.shape
    left, top, right, bottom = bbox_padding(box, image.shape)
    clipped = image[max(0, box.y0):min(height, box.y1), max(0, box.x0):min(width, box.x1)]
    support = np.ones_like(clipped, dtype=bool)
    if left or top or right or bottom:
        border = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]))
        fill = float(np.median(border))
        clipped = np.pad(
            clipped,
            ((top, bottom), (left, right)),
            mode="constant",
            constant_values=fill,
        )
        support = np.pad(
            support,
            ((top, bottom), (left, right)),
            mode="constant",
            constant_values=False,
        )
    expected = (box.height, box.width)
    if clipped.shape != expected:
        raise AssertionError(f"Crop shape {clipped.shape} != {expected} for {box}")
    return CropResult(
        image=np.ascontiguousarray(clipped),
        bbox=box,
        padding_left=left,
        padding_top=top,
        padding_right=right,
        padding_bottom=bottom,
        valid_support=np.ascontiguousarray(support),
    )


def scale_bbox(box: BBox, source_shape_yx: tuple[int, int], target_shape_yx: tuple[int, int]) -> BBox:
    """Map a bbox after whole-canvas scaling; never use this for lesion-only scaling."""
    source_h, source_w = map(float, source_shape_yx)
    target_h, target_w = map(float, target_shape_yx)
    if min(source_h, source_w, target_h, target_w) <= 0:
        raise ValueError("Canvas dimensions must be positive")
    sx, sy = target_w / source_w, target_h / source_h
    mapped = BBox(
        int(round(box.x0 * sx)),
        int(round(box.y0 * sy)),
        int(round(box.x1 * sx)),
        int(round(box.y1 * sy)),
    )
    if mapped.width <= 0 or mapped.height <= 0:
        raise AssertionError(f"Whole-canvas scaling made bbox degenerate: {box} -> {mapped}")
    return mapped


def resize_whole_canvas(image: np.ndarray, target_shape_yx: tuple[int, int], *, is_mask: bool = False) -> np.ndarray:
    """G1 audit-only whole-canvas normalization.  It is not a local-crop resize."""
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"Expected 2-D image, got {image.shape}")
    target_h, target_w = map(int, target_shape_yx)
    if (target_h, target_w) == image.shape:
        return image.copy()
    if is_mask:
        interpolation = cv2.INTER_NEAREST
    elif target_h * target_w < image.shape[0] * image.shape[1]:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR
    return cv2.resize(image, (target_w, target_h), interpolation=interpolation)


def source_spacing_from_manifest(resize_scale_x: float, resize_scale_y: float) -> tuple[float, float]:
    """Return (x,y) source pixels represented by one native temporal pixel.

    Old preprocessing records frame_pixels / PNG2D_reference_pixels as resize_scale.
    In G0 this spacing makes quantities from unequal export resolutions comparable without
    resizing either lesion crop.
    """
    sx, sy = float(resize_scale_x), float(resize_scale_y)
    if sx <= 0 or sy <= 0:
        raise ValueError(f"Invalid manifest resize scale: {(sx, sy)}")
    return 1.0 / sx, 1.0 / sy


def draw_bbox(image: np.ndarray, box: BBox, color: int = 255, thickness: int = 2) -> np.ndarray:
    out = np.asarray(image).copy()
    if out.ndim != 2:
        raise ValueError("draw_bbox expects grayscale image")
    cv2.rectangle(out, (box.x0, box.y0), (box.x1 - 1, box.y1 - 1), int(color), int(thickness))
    return out


def stack_phase_crops(images: Iterable[np.ndarray], box: BBox) -> np.ndarray:
    return np.stack([crop_with_border_median_padding(image, box).image for image in images], axis=0)

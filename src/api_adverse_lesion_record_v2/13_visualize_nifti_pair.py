#!/usr/bin/env python3
"""Render one Image.nii.gz / Segmentation.nii.gz pair without modifying it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image


LABEL_COLORS = {
    1: (0, 220, 255),
    2: (255, 40, 40),
    3: (255, 150, 0),
    4: (255, 235, 0),
    5: (230, 30, 255),
    6: (40, 255, 80),
}


def load_nifti_rgb(path: Path) -> np.ndarray:
    array = np.squeeze(np.asarray(nib.load(str(path)).dataobj))
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"Unsupported Image NIfTI shape after squeeze: {array.shape}")
    # NIfTI stores x,y while image viewers use row(y),column(x).
    array = np.swapaxes(array[..., :3], 0, 1)
    return np.clip(array, 0, 255).astype(np.uint8)


def load_nifti_mask(path: Path) -> np.ndarray:
    array = np.squeeze(np.asarray(nib.load(str(path)).dataobj))
    if array.ndim != 2:
        raise ValueError(f"Unsupported segmentation NIfTI shape after squeeze: {array.shape}")
    return np.swapaxes(array, 0, 1).astype(np.int16)


def colorize(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label, color in LABEL_COLORS.items():
        rgb[mask == label] = color
    return rgb


def overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.58) -> np.ndarray:
    result = image.astype(np.float32).copy()
    colored = colorize(mask).astype(np.float32)
    selected = mask > 0
    result[selected] = (1.0 - alpha) * result[selected] + alpha * colored[selected]
    return np.clip(result, 0, 255).astype(np.uint8)


def bbox(mask: np.ndarray, margin_fraction: float = 0.35) -> tuple[int, int, int, int]:
    yy, xx = np.where(mask)
    if len(xx) == 0:
        return 0, mask.shape[1], 0, mask.shape[0]
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    margin = max(24, int(max(x1 - x0, y1 - y0) * margin_fraction))
    return (
        max(0, x0 - margin),
        min(mask.shape[1], x1 + margin),
        max(0, y0 - margin),
        min(mask.shape[0], y1 + margin),
    )


def save_rgb(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array).save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-nifti", required=True, type=Path)
    parser.add_argument("--segmentation-nifti", required=True, type=Path)
    parser.add_argument("--matching-jpeg", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    nifti_image = load_nifti_rgb(args.image_nifti)
    mask = load_nifti_mask(args.segmentation_nifti)
    jpeg = np.asarray(Image.open(args.matching_jpeg).convert("RGB"))
    if jpeg.shape != nifti_image.shape:
        raise ValueError(f"JPEG {jpeg.shape} and NIfTI {nifti_image.shape} do not match")

    difference = np.abs(jpeg.astype(np.int16) - nifti_image.astype(np.int16)).astype(np.uint8)
    mask_rgb = colorize(mask)
    full_overlay = overlay(nifti_image, mask)

    # Labels 2-6 are shown as the current engineering lesion-target assumption,
    # not as a hospital-confirmed medical definition.
    lesion_candidate = mask >= 2
    x0, x1, y0, y1 = bbox(lesion_candidate)
    zoom = full_overlay[y0:y1, x0:x1]

    save_rgb(output_dir / "01_matching_jpeg.png", jpeg)
    save_rgb(output_dir / "02_image_nifti_render.png", nifti_image)
    save_rgb(output_dir / "03_segmentation_labels.png", mask_rgb)
    save_rgb(output_dir / "04_label_overlay.png", full_overlay)
    save_rgb(output_dir / "05_lesion_candidate_zoom.png", zoom)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
    panels = [
        (jpeg, "Matching DSA JPEG frame"),
        (nifti_image, "Image.nii.gz rendered as RGB"),
        (difference, f"Absolute pixel difference (max={int(difference.max())})"),
        (mask_rgb, "Segmentation.nii.gz labels 1-6"),
        (full_overlay, "All labels over Image.nii.gz"),
        (zoom, "Zoom: labels 2-6 candidate region"),
    ]
    for axis, (panel, title) in zip(axes.flat, panels):
        axis.imshow(panel)
        axis.set_title(title, fontsize=15)
        axis.axis("off")
    fig.suptitle("NIfTI image and segmentation example: 726527 / C6-1 / Pre", fontsize=19)
    fig.savefig(output_dir / "NIFTI_GZ_VISUAL_EXAMPLE.png", dpi=150, facecolor="white")
    plt.close(fig)

    metadata = {
        "image_nifti": str(args.image_nifti.resolve()),
        "segmentation_nifti": str(args.segmentation_nifti.resolve()),
        "matching_jpeg": str(args.matching_jpeg.resolve()),
        "image_shape_rendered": list(nifti_image.shape),
        "mask_shape_rendered": list(mask.shape),
        "mask_labels": [int(v) for v in np.unique(mask)],
        "pixel_difference_max": int(difference.max()),
        "pixel_difference_mean": float(difference.mean()),
        "lesion_candidate_bbox_xyxy": [x0, y0, x1, y1],
        "label_note": "Labels 2-6 are a current engineering assumption and require hospital confirmation.",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

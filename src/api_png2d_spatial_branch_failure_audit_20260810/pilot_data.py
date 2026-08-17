from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def unroll_phase_rows(case_frame: pd.DataFrame, phases) -> pd.DataFrame:
    rows = []
    for row in case_frame.itertuples(index=False):
        for phase in phases:
            key = str(phase).lower()
            rows.append({
                "patient_id": str(row.patient_id),
                "series_uid": str(row.series_uid),
                "phase": str(phase),
                "image_path": str(getattr(row, f"{key}_image")),
                "mask_path": str(getattr(row, f"{key}_mask")),
            })
    return pd.DataFrame(rows)


def synchronized_affine(image: np.ndarray, mask: np.ndarray, aug: dict[str, Any]):
    probability = float(aug.get("geometry_probability", 0.0))
    if probability <= 0 or np.random.random() >= probability:
        return image, mask

    h, w = image.shape
    angle = np.random.uniform(
        -float(aug.get("rotation_degrees", 0.0)),
        float(aug.get("rotation_degrees", 0.0)),
    )
    scale_delta = float(aug.get("scale_delta", 0.0))
    scale = np.random.uniform(1.0 - scale_delta, 1.0 + scale_delta)
    translate_fraction = float(aug.get("translate_fraction", 0.0))
    tx = np.random.uniform(-translate_fraction, translate_fraction) * w
    ty = np.random.uniform(-translate_fraction, translate_fraction) * h

    matrix = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), angle, scale)
    matrix[:, 2] += (tx, ty)
    fill = float(np.median(image))
    transformed_image = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill,
    )
    transformed_mask = cv2.warpAffine(
        mask.astype(np.uint8),
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if not np.any(transformed_mask > 0):
        return image, mask
    return transformed_image.astype(np.float32), (transformed_mask > 0).astype(np.float32)


class PilotSegmentationDataset(Dataset):
    def __init__(
        self,
        case_frame: pd.DataFrame,
        cfg: dict[str, Any],
        augment: bool,
        prepare_pair,
        geometry_enabled: bool,
    ):
        self.cfg = cfg
        self.augment = bool(augment)
        self.prepare_pair = prepare_pair
        self.geometry_enabled = bool(geometry_enabled)
        self.rows = unroll_phase_rows(case_frame, cfg["data"]["phases"])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[int(index)]
        image, mask = self.prepare_pair(row.image_path, row.mask_path, self.cfg)
        aug = self.cfg["segresnet"].get("augmentation", {})

        if self.augment and bool(aug.get("enabled", True)):
            if self.geometry_enabled:
                image, mask = synchronized_affine(image, mask, aug)

            contrast = float(aug.get("contrast", 0.0))
            brightness = float(aug.get("brightness", 0.0))
            noise_std = float(aug.get("noise_std", 0.0))
            if contrast > 0:
                factor = np.random.uniform(1.0 - contrast, 1.0 + contrast)
                mean = float(image.mean())
                image = (image - mean) * factor + mean
            if brightness > 0:
                image = image + np.random.uniform(-brightness, brightness)
            if noise_std > 0:
                image = image + np.random.normal(0, noise_std, image.shape).astype(np.float32)
            image = np.clip(image, 0.0, 1.0)

        return (
            torch.from_numpy(image[None]).float(),
            torch.from_numpy(mask[None]).float(),
            int(index),
        )

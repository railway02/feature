from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def synchronized_affine(image: np.ndarray, mask: np.ndarray, aug: dict[str, Any]):
    probability = float(aug.get("geometry_probability", 0.0))
    if probability <= 0 or np.random.random() >= probability:
        return image, mask, False, False

    height, width = image.shape
    angle = np.random.uniform(-float(aug["rotation_degrees"]), float(aug["rotation_degrees"]))
    delta = float(aug["scale_delta"])
    scale = np.random.uniform(1.0 - delta, 1.0 + delta)
    translation = float(aug["translate_fraction"])
    tx = np.random.uniform(-translation, translation) * width
    ty = np.random.uniform(-translation, translation) * height
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2.0, (height - 1) / 2.0), angle, scale)
    matrix[:, 2] += (tx, ty)

    transformed_image = cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=float(np.median(image)),
    )
    transformed_mask = cv2.warpAffine(
        mask.astype(np.uint8), matrix, (width, height), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    if not np.any(transformed_mask > 0):
        return image, mask, False, True
    return (
        transformed_image.astype(np.float32),
        (transformed_mask > 0).astype(np.float32),
        True,
        False,
    )


def augment_pair(image: np.ndarray, mask: np.ndarray, aug: dict[str, Any]):
    image, mask, geometry_applied, geometry_fallback = synchronized_affine(image, mask, aug)
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
        image = image + np.random.normal(0.0, noise_std, image.shape).astype(np.float32)
    return np.clip(image, 0.0, 1.0).astype(np.float32), mask, geometry_applied, geometry_fallback

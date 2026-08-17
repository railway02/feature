from __future__ import annotations

import numpy as np


def bbox_from_text(value: str) -> tuple[int, int, int, int]:
    parts = tuple(int(item) for item in str(value).split("|"))
    if len(parts) != 4:
        raise ValueError(value)
    return parts


def crop_frames(frames: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    if not (x0 < x1 and y0 < y1):
        raise AssertionError(f"Invalid ROI box {box}")
    requested_h, requested_w = y1 - y0, x1 - x0
    if requested_h != requested_w:
        raise AssertionError(f"ROI is not square: {box}")
    height, width = frames.shape[1:]
    left, top = max(0, -x0), max(0, -y0)
    right, bottom = max(0, x1 - width), max(0, y1 - height)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    if not (sx0 < sx1 and sy0 < sy1):
        raise AssertionError(f"ROI does not intersect frames: {box}")
    cropped = frames[:, sy0:sy1, sx0:sx1]
    if any((left, top, right, bottom)):
        padded = np.empty((len(frames), requested_h, requested_w), dtype=frames.dtype)
        for index, frame in enumerate(frames):
            border = np.concatenate((frame[0], frame[-1], frame[:, 0], frame[:, -1]))
            padded[index].fill(np.asarray(np.median(border), dtype=frames.dtype))
        padded[:, top:top + cropped.shape[1], left:left + cropped.shape[2]] = cropped
        cropped = padded
    if cropped.shape[1:] != (requested_h, requested_w):
        raise AssertionError(f"Unexpected crop shape: {cropped.shape}")
    return cropped

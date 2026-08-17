from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from common import atomic_directory, hash_lines, sha256_file, sha256_json, write_json_atomic


@dataclass(frozen=True)
class SquareTransform:
    original_h: int
    original_w: int
    side: int
    top: int
    bottom: int
    left: int
    right: int
    output_size: int
    border_value: int

    def to_json(self) -> dict[str, int]:
        return asdict(self)


def read_grayscale(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"OpenCV could not read frozen frame: {path}")
    return image


def load_gray_frames(paths: Sequence[str], num_workers: int = 4) -> np.ndarray:
    if not paths:
        raise ValueError("No frame paths")
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
        frames = list(executor.map(read_grayscale, [str(p) for p in paths]))
    shapes = {frame.shape for frame in frames}
    if len(shapes) != 1:
        raise AssertionError(f"Mixed dimensions in phase: {sorted(shapes)}")
    return np.stack(frames, axis=0)


def _outer_border_values(frames: np.ndarray, width_fraction: float = 0.03) -> np.ndarray:
    _, height, width = frames.shape
    by = max(1, int(round(height * width_fraction)))
    bx = max(1, int(round(width * width_fraction)))
    return np.concatenate([
        frames[:, :by, :].ravel(), frames[:, -by:, :].ravel(),
        frames[:, :, :bx].ravel(), frames[:, :, -bx:].ravel(),
    ])


def make_square_transform(frames: np.ndarray, output_size: int = 512) -> SquareTransform:
    _, height, width = frames.shape
    side = max(height, width)
    top = (side - height) // 2
    bottom = side - height - top
    left = (side - width) // 2
    right = side - width - left
    border_value = int(round(float(np.median(_outer_border_values(frames)))))
    return SquareTransform(height, width, side, top, bottom, left, right, output_size, border_value)


def frames_to_model(frames: np.ndarray, transform: SquareTransform) -> np.ndarray:
    outputs: list[np.ndarray] = []
    interpolation = cv2.INTER_AREA if transform.side >= transform.output_size else cv2.INTER_CUBIC
    for frame in frames:
        padded = cv2.copyMakeBorder(
            frame, transform.top, transform.bottom, transform.left, transform.right,
            cv2.BORDER_CONSTANT, value=transform.border_value,
        )
        outputs.append(cv2.resize(padded, (transform.output_size, transform.output_size), interpolation=interpolation))
    return np.stack(outputs).astype(np.float32) / 255.0


def map_original_to_model(
    image: np.ndarray,
    transform: SquareTransform,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    padded = cv2.copyMakeBorder(
        image.astype(np.float32), transform.top, transform.bottom, transform.left, transform.right,
        cv2.BORDER_CONSTANT, value=0.0,
    )
    return cv2.resize(padded, (transform.output_size, transform.output_size), interpolation=interpolation)


def map_model_to_original(
    image: np.ndarray,
    transform: SquareTransform,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    square = cv2.resize(image.astype(np.float32), (transform.side, transform.side), interpolation=interpolation)
    crop = square[
        transform.top:transform.top + transform.original_h,
        transform.left:transform.left + transform.original_w,
    ]
    if crop.shape != (transform.original_h, transform.original_w):
        raise RuntimeError(f"Inverse transform shape mismatch: {crop.shape}")
    return crop


def strict_contiguous_blocks(indices: Sequence[int]) -> list[np.ndarray]:
    if not indices:
        return []
    values = np.asarray(indices, dtype=np.int64)
    if not np.all(values[1:] > values[:-1]):
        raise AssertionError("Frame indices must be strictly increasing")
    split_points = np.flatnonzero(np.diff(values) != 1) + 1
    return [block for block in np.split(np.arange(len(values), dtype=np.int64), split_points) if len(block)]


def uniform_positions(length: int, max_len: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("length must be positive")
    if length <= max_len:
        return np.arange(length, dtype=np.int64)
    positions = np.rint(np.linspace(0, length - 1, max_len)).astype(np.int64)
    if len(np.unique(positions)) != max_len:
        raise AssertionError("Uniform sampling generated duplicate positions")
    return positions


def contrast_core_positions(frames01: np.ndarray, max_len: int) -> np.ndarray:
    positions = list(range(len(frames01)))
    while len(positions) > max_len:
        left = float(frames01[positions[0]].sum())
        right = float(frames01[positions[-1]].sum())
        if left >= right:
            positions.pop(0)
        else:
            positions.pop()
    return np.asarray(positions, dtype=np.int64)


def temporal_views(frames01: np.ndarray, max_len: int = 20) -> dict[str, np.ndarray]:
    uniform = uniform_positions(len(frames01), max_len)
    core = contrast_core_positions(frames01, max_len)
    return {"uniform_full20": uniform, "contrast_core20": core}


def save_npz(path: Path, values: dict[str, np.ndarray], compressed: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        np.savez_compressed(path, **values)
    else:
        np.savez(path, **values)


def probability_to_u8(probability: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(probability, 0.0, 1.0) * 255.0).astype(np.uint8)


def save_probability_png(path: Path, probability: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), probability_to_u8(probability)):
        raise RuntimeError(f"Could not write {path}")


def _normalize_gray(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    low, high = np.percentile(image, [1, 99])
    return np.rint(np.clip((image - low) / max(high - low, 1e-6), 0, 1) * 255).astype(np.uint8)


def save_av_overlay(path: Path, background: np.ndarray, artery: np.ndarray, vein: np.ndarray) -> None:
    gray = _normalize_gray(background)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    overlay = np.zeros_like(base)
    overlay[..., 2] = np.clip(artery, 0, 1) * 255.0  # red
    overlay[..., 0] = np.clip(vein, 0, 1) * 255.0    # blue
    alpha = np.clip(np.maximum(artery, vein)[..., None] * 0.65, 0, 0.65)
    output = base * (1 - alpha) + overlay * alpha
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.rint(output).astype(np.uint8)):
        raise RuntimeError(f"Could not write {path}")


def save_input_mosaic(
    path: Path,
    frames: np.ndarray,
    frame_indices: Sequence[int],
    maximum_tiles: int = 12,
    tile_size: int = 256,
) -> None:
    positions = uniform_positions(len(frames), min(maximum_tiles, len(frames)))
    tiles: list[np.ndarray] = []
    for position in positions:
        tile = cv2.resize(_normalize_gray(frames[int(position)]), (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            tile, f"frame={frame_indices[int(position)]}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 255, 0), 1, cv2.LINE_AA,
        )
        tiles.append(tile)
    columns = min(4, len(tiles))
    rows = int(np.ceil(len(tiles) / columns))
    blank = np.zeros_like(tiles[0])
    tiles.extend([blank] * (rows * columns - len(tiles)))
    grid = np.vstack([np.hstack(tiles[row * columns:(row + 1) * columns]) for row in range(rows)])
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), grid):
        raise RuntimeError(f"Could not write {path}")


__all__ = [
    "SquareTransform", "atomic_directory", "frames_to_model", "hash_lines", "load_gray_frames",
    "make_square_transform", "map_model_to_original", "map_original_to_model", "save_av_overlay",
    "save_input_mosaic", "save_npz", "save_probability_png", "sha256_file", "sha256_json",
    "strict_contiguous_blocks", "temporal_views", "write_json_atomic",
]

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_pipe_strings(value: object) -> list[str]:
    if value is None or str(value).strip().lower() in {"", "nan", "none"}:
        return []
    return [part for part in str(value).split("|") if part]


def parse_pipe_ints(value: object) -> list[int]:
    return [int(part) for part in parse_pipe_strings(value)]


def hash_lines(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def load_gray_frames(paths: Sequence[str]) -> np.ndarray:
    frames = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Cannot read frozen frame: {path}")
        frames.append(image)
    if not frames:
        raise ValueError("No frames")
    shapes = {frame.shape for frame in frames}
    if len(shapes) != 1:
        raise ValueError(f"Mixed dimensions in phase: {sorted(shapes)}")
    return np.stack(frames, axis=0)


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
        return self.__dict__.copy()


def _outer_border_values(frames: np.ndarray, width_fraction: float = 0.03) -> np.ndarray:
    _, h, w = frames.shape
    by = max(1, int(round(h * width_fraction)))
    bx = max(1, int(round(w * width_fraction)))
    return np.concatenate([
        frames[:, :by, :].ravel(), frames[:, -by:, :].ravel(),
        frames[:, :, :bx].ravel(), frames[:, :, -bx:].ravel(),
    ])


def make_square_transform(frames: np.ndarray, output_size: int = 512) -> SquareTransform:
    _, h, w = frames.shape
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    border_value = int(round(float(np.median(_outer_border_values(frames)))))
    return SquareTransform(h, w, side, top, bottom, left, right, output_size, border_value)


def frames_to_model(frames: np.ndarray, transform: SquareTransform) -> np.ndarray:
    outputs = []
    for frame in frames:
        padded = cv2.copyMakeBorder(
            frame, transform.top, transform.bottom, transform.left, transform.right,
            cv2.BORDER_CONSTANT, value=transform.border_value,
        )
        outputs.append(cv2.resize(padded, (transform.output_size, transform.output_size), interpolation=cv2.INTER_AREA))
    return np.stack(outputs).astype(np.float32) / 255.0


def map_original_to_model(image: np.ndarray, transform: SquareTransform, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    padded = cv2.copyMakeBorder(
        image.astype(np.float32), transform.top, transform.bottom, transform.left, transform.right,
        cv2.BORDER_CONSTANT, value=0.0,
    )
    return cv2.resize(padded, (transform.output_size, transform.output_size), interpolation=interpolation)


def map_model_to_original(image: np.ndarray, transform: SquareTransform, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    square = cv2.resize(image.astype(np.float32), (transform.side, transform.side), interpolation=interpolation)
    crop = square[
        transform.top:transform.top + transform.original_h,
        transform.left:transform.left + transform.original_w,
    ]
    if crop.shape != (transform.original_h, transform.original_w):
        raise RuntimeError(f"Inverse transform shape mismatch: {crop.shape}")
    return crop


def uniform_positions(length: int, max_len: int) -> np.ndarray:
    if length <= max_len:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.rint(np.linspace(0, length - 1, max_len)).astype(np.int64))


def contrast_core_positions(frames01: np.ndarray, max_len: int) -> np.ndarray:
    positions = list(range(len(frames01)))
    while len(positions) > max_len:
        if float(frames01[positions[0]].sum()) >= float(frames01[positions[-1]].sum()):
            positions.pop(0)
        else:
            positions.pop()
    return np.asarray(positions, dtype=np.int64)


def temporal_views(frames01: np.ndarray, max_len: int = 20) -> dict[str, np.ndarray]:
    uniform = uniform_positions(len(frames01), max_len)
    core = contrast_core_positions(frames01, max_len)
    return {"uniform_full20": uniform, "contrast_core20": core}


def split_large_gap_blocks(
    indices: Sequence[int], paths: Sequence[str], max_missing_frames_within_block: int = 2
) -> list[tuple[list[int], list[str], list[int]]]:
    if len(indices) != len(paths):
        raise ValueError("indices/paths length mismatch")
    order = np.argsort(np.asarray(indices), kind="stable")
    idx = [int(indices[i]) for i in order]
    pth = [str(paths[i]) for i in order]
    blocks: list[tuple[list[int], list[str], list[int]]] = []
    start = 0
    for pos in range(1, len(idx)):
        missing = idx[pos] - idx[pos - 1] - 1
        if missing > max_missing_frames_within_block:
            blocks.append((idx[start:pos], pth[start:pos], list(range(start, pos))))
            start = pos
    blocks.append((idx[start:], pth[start:], list(range(start, len(idx)))))
    return blocks


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=True)
    os.replace(temp, path)


@contextmanager
def atomic_directory(final_dir: Path, overwrite: bool = False) -> Iterator[Path]:
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = final_dir.with_name(final_dir.name + f".tmp.{uuid.uuid4().hex}")
    temp_dir.mkdir(parents=True)
    try:
        yield temp_dir
        backup = None
        if final_dir.exists():
            if not overwrite:
                raise FileExistsError(final_dir)
            backup = final_dir.with_name(final_dir.name + f".old.{uuid.uuid4().hex}")
            os.rename(final_dir, backup)
        os.rename(temp_dir, final_dir)
        if backup is not None:
            shutil.rmtree(backup)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

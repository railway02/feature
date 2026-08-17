from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from nifti_io import apply_orientation


@dataclass(frozen=True)
class MatchResult:
    score: float
    intensity_score: float
    gradient_score: float
    transform: str
    plane_name: str
    frame_path: str
    frame_index: int
    series_uid: str
    phase: str


def read_gray(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image.astype(np.float32)


def _normalize(image: np.ndarray, size: int) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"匹配图必须二维，shape={image.shape}")
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise ValueError("图像无有限像素")
    low, high = np.percentile(finite, [1.0, 99.0])
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros((size, size), dtype=np.float32)
    image = np.clip(image, low, high)
    image = (image - low) / (high - low)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    image -= np.float32(image.mean())
    std = np.float32(image.std())
    if std > 1e-8:
        image /= std
    return np.asarray(image, dtype=np.float32)


def _prepare(image: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    intensity = _normalize(image, size)
    gx = cv2.Sobel(intensity, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(intensity, cv2.CV_32F, 0, 1)
    gradient = cv2.magnitude(gx, gy).astype(np.float32)
    gradient -= np.float32(gradient.mean())
    norm = np.float32(gradient.std())
    if norm > 1e-8:
        gradient /= norm
    return intensity, gradient


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    av = a.ravel().astype(np.float64)
    bv = b.ravel().astype(np.float64)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1e-12:
        return 0.0
    return float(abs(np.dot(av, bv) / denom))


def prepared_similarity(reference: tuple[np.ndarray, np.ndarray], frame: tuple[np.ndarray, np.ndarray]) -> tuple[float, float, float]:
    intensity = _corr(reference[0], frame[0])
    gradient = _corr(reference[1], frame[1])
    return 0.80 * intensity + 0.20 * gradient, intensity, gradient


def similarity(reference: np.ndarray, frame: np.ndarray, size: int = 128) -> tuple[float, float, float]:
    return prepared_similarity(_prepare(reference, size), _prepare(frame, size))


def match_reference(
    reference_planes: list[tuple[str, np.ndarray]],
    candidates: list[dict[str, object]],
    transforms: Iterable[str],
    *,
    downsample: int = 128,
    frame_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] | None = None,
) -> list[MatchResult]:
    cache = frame_cache if frame_cache is not None else {}
    reference_variants: list[tuple[str, str, tuple[np.ndarray, np.ndarray]]] = []
    for plane_name, plane in reference_planes:
        for transform in transforms:
            reference_variants.append((plane_name, transform, _prepare(apply_orientation(plane, transform), downsample)))

    results: list[MatchResult] = []
    for candidate in candidates:
        paths = list(candidate["frame_paths"])
        for frame_index, frame_path in enumerate(paths):
            key = (str(frame_path), downsample)
            if key not in cache:
                cache[key] = _prepare(read_gray(key[0]), downsample)
            prepared_frame = cache[key]
            best: MatchResult | None = None
            for plane_name, transform, prepared_reference in reference_variants:
                score, intensity, gradient = prepared_similarity(prepared_reference, prepared_frame)
                item = MatchResult(
                    score=score, intensity_score=intensity, gradient_score=gradient,
                    transform=transform, plane_name=plane_name, frame_path=key[0], frame_index=frame_index,
                    series_uid=str(candidate["series_uid"]), phase=str(candidate["phase"]),
                )
                if best is None or item.score > best.score:
                    best = item
            if best is not None:
                results.append(best)
    return sorted(results, key=lambda item: (-item.score, item.series_uid, item.phase, item.frame_index))

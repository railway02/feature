from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
try:
    import nibabel as nib
except ModuleNotFoundError:
    nib=None
import numpy as np
import pandas as pd

from common import bool_value, hash_lines, parse_pipe, safe_uid, sha256_file


TRANSFORMS = (
    "identity", "transpose", "flip_x", "flip_y", "flip_xy",
    "transpose_flip_x", "transpose_flip_y", "transpose_flip_xy",
)


@dataclass(frozen=True)
class ResizeTransform:
    original_h: int
    original_w: int
    side: int
    top: int
    bottom: int
    left: int
    right: int
    output_size: int

    def to_json(self) -> dict[str, int]:
        return {
            "original_h": self.original_h, "original_w": self.original_w,
            "side": self.side, "top": self.top, "bottom": self.bottom,
            "left": self.left, "right": self.right, "output_size": self.output_size,
        }


def apply_orientation(array: np.ndarray, name: str) -> np.ndarray:
    if name == "identity":
        return array
    if name == "transpose":
        return np.swapaxes(array, 0, 1)
    if name == "flip_x":
        return np.flip(array, axis=1)
    if name == "flip_y":
        return np.flip(array, axis=0)
    if name == "flip_xy":
        return np.flip(np.flip(array, axis=0), axis=1)
    transposed = np.swapaxes(array, 0, 1)
    if name == "transpose_flip_x":
        return np.flip(transposed, axis=1)
    if name == "transpose_flip_y":
        return np.flip(transposed, axis=0)
    if name == "transpose_flip_xy":
        return np.flip(np.flip(transposed, axis=0), axis=1)
    raise ValueError(name)


def _squeeze_nifti_image(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    value = np.squeeze(value)
    if value.ndim == 2:
        return value.astype(np.float32)
    if value.ndim == 3 and value.shape[-1] in {3, 4}:
        rgb = value[..., :3].astype(np.float32)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if value.ndim == 3 and value.shape[0] in {3, 4}:
        rgb = np.moveaxis(value[:3], 0, -1).astype(np.float32)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    raise AssertionError(f"Unsupported NIfTI image shape: {array.shape} -> {value.shape}")


def _squeeze_nifti_mask(array: np.ndarray) -> np.ndarray:
    value = np.squeeze(np.asarray(array))
    if value.ndim != 2:
        raise AssertionError(f"Unsupported NIfTI mask shape: {array.shape} -> {value.shape}")
    return value.astype(np.int16)


def load_nifti_image(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    if nib is None: raise ModuleNotFoundError("nibabel is required only for NIfTI asset stages")
    image = nib.load(str(path))
    array = _squeeze_nifti_image(np.asanyarray(image.dataobj))
    return array, {
        "raw_shape": list(image.shape), "shape": list(array.shape),
        "zooms": list(image.header.get_zooms()),
        "axcodes": list(nib.aff2axcodes(image.affine)),
        "affine": image.affine.tolist(),
    }


def load_nifti_mask(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    if nib is None: raise ModuleNotFoundError("nibabel is required only for NIfTI asset stages")
    image = nib.load(str(path))
    array = _squeeze_nifti_mask(np.asanyarray(image.dataobj))
    labels = sorted(int(value) for value in np.unique(array))
    return array, {
        "raw_shape": list(image.shape), "shape": list(array.shape),
        "labels": labels, "nonzero_pixels": int((array != 0).sum()),
        "zooms": list(image.header.get_zooms()),
        "axcodes": list(nib.aff2axcodes(image.affine)),
        "affine": image.affine.tolist(),
    }


def read_frames(paths: Iterable[str]) -> np.ndarray:
    frames: list[np.ndarray] = []
    for value in paths:
        frame = cv2.imread(str(value), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise RuntimeError(f"Could not read frame: {value}")
        frames.append(frame)
    if not frames:
        raise ValueError("No frames")
    shapes = {item.shape for item in frames}
    if len(shapes) != 1:
        raise AssertionError(f"Mixed frame dimensions: {sorted(shapes)}")
    return np.stack(frames)


def normalize01(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image, dtype=np.float32)
    finite = value[np.isfinite(value)]
    if not len(finite):
        return np.zeros_like(value)
    low, high = np.percentile(finite, [1, 99])
    return np.clip((value - low) / max(float(high - low), 1e-6), 0, 1)


def normalized_ncc(first: np.ndarray, second: np.ndarray) -> float:
    a = normalize01(first).ravel().astype(np.float64)
    b = normalize01(second).ravel().astype(np.float64)
    a -= a.mean(); b -= b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / max(denominator, 1e-12))


def global_ssim(first: np.ndarray, second: np.ndarray) -> float:
    a = normalize01(first).astype(np.float64)
    b = normalize01(second).astype(np.float64)
    mean_a, mean_b = float(a.mean()), float(b.mean())
    var_a, var_b = float(a.var()), float(b.var())
    covariance = float(((a - mean_a) * (b - mean_b)).mean())
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    numerator = (2 * mean_a * mean_b + c1) * (2 * covariance + c2)
    denominator = (mean_a ** 2 + mean_b ** 2 + c1) * (var_a + var_b + c2)
    return float(numerator / max(denominator, 1e-12))


def image_match_score(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    if first.shape != second.shape:
        first = cv2.resize(first.astype(np.float32), (second.shape[1], second.shape[0]), interpolation=cv2.INTER_LINEAR)
    a, b = normalize01(first), normalize01(second)
    ncc = normalized_ncc(a, b)
    ssim = global_ssim(a, b)
    mae = float(np.mean(np.abs(a - b)))
    score = 0.55 * ((ncc + 1.0) / 2.0) + 0.30 * ((ssim + 1.0) / 2.0) + 0.15 * (1.0 - mae)
    return {"score": float(score), "ncc": ncc, "ssim": ssim, "mae": mae}


def build_summary_channels(frames: np.ndarray) -> dict[str, np.ndarray]:
    values = frames.astype(np.float32)
    mean = values.mean(axis=0)
    median = np.median(values, axis=0)
    early_count = max(1, min(3, len(values) // 4 if len(values) >= 4 else 1))
    baseline = np.median(values[:early_count], axis=0)
    enhancement = np.maximum(baseline[None] - values, 0.0)
    max_enhancement = enhancement.max(axis=0)
    temporal_range = values.max(axis=0) - values.min(axis=0)
    minip = values.min(axis=0)
    maxip = values.max(axis=0)
    frame_scores = enhancement.mean(axis=(1, 2))
    peak = values[int(np.argmax(frame_scores))]
    return {
        "temporal_mean": mean, "temporal_median": median,
        "max_enhancement": max_enhancement, "temporal_range": temporal_range,
        "minip": minip, "maxip": maxip, "peak_contrast_frame": peak,
        "baseline": baseline, "enhancement_sequence": enhancement,
    }


def summary_stack_u8(frames: np.ndarray, channels: Iterable[str]) -> np.ndarray:
    summaries = build_summary_channels(frames)
    output = []
    for name in channels:
        value = normalize01(summaries[name])
        output.append(np.rint(value * 255).astype(np.uint8))
    return np.stack(output, axis=0)


def make_resize_transform(height: int, width: int, output_size: int) -> ResizeTransform:
    side = max(height, width)
    top = (side - height) // 2
    bottom = side - height - top
    left = (side - width) // 2
    right = side - width - left
    return ResizeTransform(height, width, side, top, bottom, left, right, output_size)


def resize_stack_to_model(stack: np.ndarray, transform: ResizeTransform) -> np.ndarray:
    outputs = []
    for channel in stack:
        padded = cv2.copyMakeBorder(channel, transform.top, transform.bottom, transform.left, transform.right, cv2.BORDER_CONSTANT, value=0)
        interpolation = cv2.INTER_AREA if transform.side >= transform.output_size else cv2.INTER_CUBIC
        outputs.append(cv2.resize(padded, (transform.output_size, transform.output_size), interpolation=interpolation))
    return np.stack(outputs)


def resize_mask_to_model(mask: np.ndarray, transform: ResizeTransform) -> np.ndarray:
    padded = cv2.copyMakeBorder(mask.astype(np.uint8), transform.top, transform.bottom, transform.left, transform.right, cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(padded, (transform.output_size, transform.output_size), interpolation=cv2.INTER_NEAREST)


def restore_probability(probability: np.ndarray, transform: ResizeTransform) -> np.ndarray:
    square = cv2.resize(probability.astype(np.float32), (transform.side, transform.side), interpolation=cv2.INTER_LINEAR)
    return square[transform.top:transform.top + transform.original_h, transform.left:transform.left + transform.original_w]


def resize_transform_from_json(payload: dict[str, Any]) -> ResizeTransform:
    return ResizeTransform(**{key: int(payload[key]) for key in ResizeTransform.__dataclass_fields__})


def phase_from_segmentation_path(path: Path) -> str:
    text = str(path).casefold()
    if re.search(r"(^|[/_-])(pre|pro)([/_.=-]|$)", text):
        return "pre"
    if re.search(r"(^|[/_-])(post|pos|pot)([/_.=-]|$)", text):
        return "post"
    return "unknown"


def resolve_mask(series_path: Path, phase: str) -> dict[str, Any]:
    title = phase.capitalize()
    direct = series_path / f"{title}-Segmentation.nii.gz"
    biaozhu = series_path / f"{title}-biaozhu" / "Segmentation.nii.gz"
    exact_paths=[path for path in (direct,biaozhu) if path.is_file()]
    if direct.is_file():
        image_path=direct.parent/"Image.nii.gz"
        return {"status":"exact_direct_preferred" if biaozhu.is_file() else "exact","path":str(direct),"layout":"direct","candidate_count":len(exact_paths),"candidate_paths":"|".join(map(str,exact_paths)),"image_path":str(image_path) if image_path.is_file() else ""}
    if biaozhu.is_file():
        image_path=biaozhu.parent/"Image.nii.gz"
        return {"status":"exact","path":str(biaozhu),"layout":"nested_biaozhu","candidate_count":1,"candidate_paths":str(biaozhu),"image_path":str(image_path) if image_path.is_file() else ""}
    candidates = []
    if series_path.is_dir():
        for path in series_path.rglob("*.nii.gz"):
            if "segmentation" not in path.name.casefold():
                continue
            if phase_from_segmentation_path(path) == phase:
                candidates.append(path)
    candidates = sorted(set(candidates))
    if len(candidates) == 1:
        path = candidates[0]
        image_path = path.parent / "Image.nii.gz"
        return {"status": "recursive_unique", "path": str(path), "layout": "recursive", "candidate_count": 1, "candidate_paths": str(path), "image_path": str(image_path) if image_path.is_file() else ""}
    return {
        "status": "missing" if not candidates else "recursive_ambiguous",
        "path": "", "layout": "", "candidate_count": len(candidates),
        "candidate_paths": "|".join(str(path) for path in candidates), "image_path": "",
    }


def manifest_phase_rows(manifest_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    rows: list[dict[str, Any]] = []
    manifest_sha = sha256_file(manifest_path)
    for record in frame.to_dict("records"):
        for phase in ("pre", "post"):
            paths = parse_pipe(record.get(f"{phase}_frame_paths", ""))
            indices = [int(value) for value in parse_pipe(record.get(f"{phase}_frame_indices", ""))]
            can_run = bool_value(record.get(f"can_run_{phase}", False))
            if not can_run:
                continue
            if len(paths) != len(indices) or not paths:
                raise AssertionError(f"{record.get('series_uid')} {phase}: invalid frozen frame list")
            stored = str(record.get(f"{phase}_frame_list_hash", ""))
            if hash_lines(paths) != stored:
                raise AssertionError(f"{record.get('series_uid')} {phase}: frozen frame hash mismatch")
            series_path = Path(record["series_path"])
            resolved = resolve_mask(series_path, phase)
            rows.append({
                "phase_uid": safe_uid(record["split"], record["series_uid"], phase),
                "patient_id": str(record["patient_id"]), "split": str(record["split"]),
                "source_type": str(record.get("source_type", "")),
                "source_medical_record_root": str(record.get("source_medical_record_root", "")),
                "series_uid": str(record["series_uid"]), "series_id": str(record.get("series_id", "")),
                "series_path": str(series_path), "fixed_mapping_series": str(record.get("fixed_mapping_series", "")),
                "phase": phase, "frame_paths": "|".join(paths), "frame_indices": "|".join(map(str, indices)),
                "frame_list_hash": stored, "n_frames": len(paths), "manifest_path": str(manifest_path),
                "manifest_sha256": manifest_sha,
                "mask_resolution_status": resolved["status"], "segmentation_path": resolved["path"],
                "annotation_layout": resolved["layout"], "mask_candidate_count": resolved["candidate_count"],
                "mask_candidate_paths": resolved["candidate_paths"], "reference_image_path": resolved["image_path"],
            })
    output = pd.DataFrame(rows)
    if output["phase_uid"].duplicated().any():
        raise AssertionError("Duplicate phase_uid")
    return output


def lesion_and_context_masks(mask: np.ndarray, lesion_labels: Iterable[int], context_labels: Iterable[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lesion = np.isin(mask, list(lesion_labels))
    context = np.isin(mask, list(context_labels))
    all_nonzero = mask != 0
    return lesion.astype(np.uint8), context.astype(np.uint8), all_nonzero.astype(np.uint8)


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return mask.astype(np.uint8)
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == index).astype(np.uint8)


def mask_contrast_score(summary: np.ndarray, lesion: np.ndarray, context: np.ndarray, kernel_size:int=31) -> dict[str, float]:
    enhancement = normalize01(summary)
    foreground = (lesion > 0) | (context > 0)
    if not foreground.any():
        return {"score": float("-inf"), "lesion_z": float("nan"), "context_z": float("nan")}
    kernel = np.ones((int(kernel_size), int(kernel_size)), np.uint8)
    neighborhood = cv2.dilate(foreground.astype(np.uint8), kernel, iterations=1).astype(bool)
    background = neighborhood & ~foreground
    if not background.any():
        background = ~foreground
    bg = enhancement[background]
    bg_mean, bg_std = float(bg.mean()), max(float(bg.std()), 1e-5)
    lesion_z = float((enhancement[lesion > 0].mean() - bg_mean) / bg_std) if lesion.any() else float("nan")
    context_z = float((enhancement[context > 0].mean() - bg_mean) / bg_std) if context.any() else float("nan")
    values = [value for value in (lesion_z, context_z) if math.isfinite(value)]
    return {"score": float(np.mean(values)) if values else float("-inf"), "lesion_z": lesion_z, "context_z": context_z}


def draw_overlay(background: np.ndarray, mask: np.ndarray, path: Path, title: str = "") -> None:
    gray = np.rint(normalize01(background) * 255).astype(np.uint8)
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    colors = np.asarray([
        [0, 0, 0], [0, 0, 255], [0, 255, 0], [255, 0, 0],
        [0, 255, 255], [255, 0, 255], [255, 255, 0],
    ], dtype=np.uint8)
    color = colors[np.clip(mask.astype(int), 0, 6)]
    selected = mask > 0
    canvas[selected] = np.rint(0.45 * canvas[selected] + 0.55 * color[selected]).astype(np.uint8)
    if title:
        cv2.putText(canvas, title[:90], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"Could not write {path}")


def find_reference_match(reference: np.ndarray, frames: np.ndarray, transforms: Iterable[str] = TRANSFORMS) -> dict[str, Any]:
    target_side = 160
    frame_thumbnails = [
        cv2.resize(frame.astype(np.float32), (target_side, target_side), interpolation=cv2.INTER_AREA)
        for frame in frames
    ]
    candidates: list[dict[str, Any]] = []
    for transform in transforms:
        oriented = apply_orientation(reference, transform)
        oriented = cv2.resize(oriented.astype(np.float32), (target_side, target_side), interpolation=cv2.INTER_AREA)
        for index, frame in enumerate(frame_thumbnails):
            metrics = image_match_score(oriented, frame)
            candidates.append({"orientation_transform": transform, "frame_position": index, **metrics})
    candidates.sort(key=lambda item: item["score"], reverse=True)
    best = dict(candidates[0])
    overall_runner = candidates[1] if len(candidates) > 1 else None
    orientation_runner = next((item for item in candidates if item["orientation_transform"] != best["orientation_transform"]), None)
    frame_runner = next((item for item in candidates if item["orientation_transform"] == best["orientation_transform"] and item["frame_position"] != best["frame_position"]), None)
    best["frame_match_margin"] = float(best["score"] - frame_runner["score"]) if frame_runner else float("inf")
    best["orientation_margin"] = float(best["score"] - orientation_runner["score"]) if orientation_runner else float("inf")
    best["margin"] = best["orientation_margin"]
    best["runner_up"] = overall_runner
    best["orientation_runner_up"] = orientation_runner
    return best


def affine_equal(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return bool(np.allclose(np.asarray(first["affine"]), np.asarray(second["affine"]), rtol=0, atol=1e-6))

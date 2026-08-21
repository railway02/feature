from __future__ import annotations
from pathlib import Path
from typing import Sequence, Tuple
import cv2
import numpy as np


def read_gray(path: str | Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(path)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return arr.astype(np.float32)


def read_mask(path: str | Path) -> np.ndarray:
    arr = read_gray(path)
    return arr > 0


def read_sequence(paths: Sequence[str]) -> np.ndarray:
    frames = [read_gray(p) for p in paths]
    if not frames:
        raise ValueError("Empty frame list")
    shapes = {x.shape for x in frames}
    if len(shapes) != 1:
        raise ValueError(f"Frame shapes are inconsistent: {shapes}")
    return np.stack(frames, axis=0)


def read_sequence_canvas(paths: Sequence[str], target_hw: Tuple[int, int],
                         aspect_tolerance=0.01) -> tuple[np.ndarray, dict]:
    """Read a temporal sequence onto an explicit whole-FOV reference canvas.

    The operation normalises export resolution only.  It does not crop around the
    lesion and therefore preserves Pre/Post translation, rotation and in-image
    magnification for the subsequent global registration.  Coordinates are canvas
    pixels; the returned metadata records the source-to-canvas scale.
    """
    frames = []
    source_hw = None
    for path in paths:
        frame = read_gray(path)
        if source_hw is None:
            source_hw = frame.shape
            _validate_uniform_resize(source_hw, target_hw, aspect_tolerance)
        elif frame.shape != source_hw:
            raise ValueError(f"Frame shapes are inconsistent: first={source_hw}, current={frame.shape}")
        frames.append(resize_image_to_canvas(frame, target_hw, is_mask=False))
    if not frames:
        raise ValueError("Empty frame list")
    return np.stack(frames, axis=0).astype(np.float32), {
        "temporal_source_height": int(source_hw[0]),
        "temporal_source_width": int(source_hw[1]),
        "canvas_height": int(target_hw[0]),
        "canvas_width": int(target_hw[1]),
        "temporal_to_canvas_scale_y": float(target_hw[0] / source_hw[0]),
        "temporal_to_canvas_scale_x": float(target_hw[1] / source_hw[1]),
    }


def mask_center(mask: np.ndarray) -> Tuple[float, float]:
    yy, xx = np.nonzero(mask)
    if len(xx) == 0:
        return ((mask.shape[0] - 1) / 2.0, (mask.shape[1] - 1) / 2.0)
    return (float(np.mean(yy)), float(np.mean(xx)))


def joint_mask_center(mask_a: np.ndarray, mask_b: np.ndarray) -> Tuple[float, float]:
    """One shared crop centre for both phases, preserving their relative displacement.

    The centre is the centre of the union bounding box in the common full-image canvas.
    Unlike two independent lesion-centred crops, the exact same crop origin is used for
    Pre and Post.
    """
    union = np.asarray(mask_a, dtype=bool) | np.asarray(mask_b, dtype=bool)
    yy, xx = np.nonzero(union)
    if len(xx) == 0:
        return ((union.shape[0] - 1) / 2.0, (union.shape[1] - 1) / 2.0)
    return ((float(yy.min()) + float(yy.max())) / 2.0,
            (float(xx.min()) + float(xx.max())) / 2.0)


def crop_origin_yx(center_yx: Tuple[float, float], out_hw: Tuple[int, int]) -> Tuple[int, int]:
    oh, ow = int(out_hw[0]), int(out_hw[1])
    cy, cx = center_yx
    return (int(round(cy - (oh - 1) / 2.0)), int(round(cx - (ow - 1) / 2.0)))


def _validate_uniform_resize(source_hw: Tuple[int, int], target_hw: Tuple[int, int], tolerance=0.01):
    sy = float(target_hw[0]) / float(source_hw[0])
    sx = float(target_hw[1]) / float(source_hw[1])
    if abs(sy - sx) / max(abs(sy), abs(sx), 1e-8) > float(tolerance):
        raise ValueError(
            f"Non-uniform full-image resize is not allowed: source={source_hw}, target={target_hw}, "
            f"scale_y={sy:.6f}, scale_x={sx:.6f}"
        )
    return sy, sx


def resize_image_to_canvas(arr: np.ndarray, target_hw: Tuple[int, int], is_mask=False) -> np.ndarray:
    """Uniformly resize a full image/sequence to the verified reference canvas.

    This is whole-FOV coordinate normalisation, not lesion bbox size normalisation.
    """
    target_hw = tuple(int(x) for x in target_hw)
    source_hw = tuple(int(x) for x in arr.shape[-2:])
    _validate_uniform_resize(source_hw, target_hw)
    if source_hw == target_hw:
        return np.asarray(arr, dtype=bool if is_mask else arr.dtype).copy()
    interp = cv2.INTER_NEAREST if is_mask else (
        cv2.INTER_AREA if target_hw[0] < source_hw[0] else cv2.INTER_LINEAR
    )
    if arr.ndim == 2:
        out = cv2.resize(arr.astype(np.uint8) if is_mask else arr,
                         (target_hw[1], target_hw[0]), interpolation=interp)
    elif arr.ndim == 3:
        out = np.stack([
            cv2.resize(x, (target_hw[1], target_hw[0]), interpolation=interp) for x in arr
        ], axis=0)
    else:
        raise ValueError(f"Expected 2-D image or [T,H,W] sequence, got {arr.shape}")
    return out.astype(bool) if is_mask else out.astype(arr.dtype, copy=False)


def phase_to_reference_canvas(sequence: np.ndarray, reference: np.ndarray, mask: np.ndarray,
                              target_hw: Tuple[int, int], aspect_tolerance=0.01):
    """Map temporal/reference/mask data into one explicit full-image canvas."""
    if reference.shape != mask.shape:
        raise ValueError(f"Reference/mask shape mismatch: {reference.shape}/{mask.shape}")
    old_tol = aspect_tolerance
    _validate_uniform_resize(sequence.shape[1:], target_hw, old_tol)
    _validate_uniform_resize(reference.shape, target_hw, old_tol)
    seq_out = resize_image_to_canvas(sequence, target_hw, is_mask=False)
    ref_out = resize_image_to_canvas(reference, target_hw, is_mask=False)
    mask_out = resize_image_to_canvas(mask, target_hw, is_mask=True)
    return seq_out, ref_out, mask_out, {
        "temporal_source_height": int(sequence.shape[1]),
        "temporal_source_width": int(sequence.shape[2]),
        "reference_source_height": int(reference.shape[0]),
        "reference_source_width": int(reference.shape[1]),
        "canvas_height": int(target_hw[0]),
        "canvas_width": int(target_hw[1]),
        "temporal_to_canvas_scale_y": float(target_hw[0] / sequence.shape[1]),
        "temporal_to_canvas_scale_x": float(target_hw[1] / sequence.shape[2]),
        "reference_to_canvas_scale_y": float(target_hw[0] / reference.shape[0]),
        "reference_to_canvas_scale_x": float(target_hw[1] / reference.shape[1]),
    }


def read_sequence_canvas_crop(paths: Sequence[str], target_hw: Tuple[int, int],
                              center_yx: Tuple[float, float], out_hw: Tuple[int, int],
                              pad_value=0.0, aspect_tolerance=0.01):
    """Memory-safe whole-FOV resize followed immediately by the shared crop."""
    frames = []
    source_hw = None
    for path in paths:
        frame = read_gray(path)
        if source_hw is None:
            source_hw = frame.shape
            _validate_uniform_resize(source_hw, target_hw, aspect_tolerance)
        elif frame.shape != source_hw:
            raise ValueError(f"Frame shapes are inconsistent: first={source_hw}, current={frame.shape}")
        resized = resize_image_to_canvas(frame, target_hw, is_mask=False)
        frames.append(crop_fixed_2d(resized, center_yx, out_hw, pad_value))
    if not frames:
        raise ValueError("Empty frame list")
    return np.stack(frames, axis=0).astype(np.float32), {
        "temporal_source_height": int(source_hw[0]),
        "temporal_source_width": int(source_hw[1]),
        "canvas_height": int(target_hw[0]),
        "canvas_width": int(target_hw[1]),
        "temporal_to_canvas_scale_y": float(target_hw[0] / source_hw[0]),
        "temporal_to_canvas_scale_x": float(target_hw[1] / source_hw[1]),
    }


def crop_fixed_2d(arr: np.ndarray, center_yx: Tuple[float, float], out_hw: Tuple[int, int], pad_value=0) -> np.ndarray:
    h, w = arr.shape[-2:]
    oh, ow = int(out_hw[0]), int(out_hw[1])
    cy, cx = center_yx
    y0 = int(round(cy - (oh - 1) / 2.0))
    x0 = int(round(cx - (ow - 1) / 2.0))
    y1, x1 = y0 + oh, x0 + ow

    sy0, sx0 = max(0, y0), max(0, x0)
    sy1, sx1 = min(h, y1), min(w, x1)
    dy0, dx0 = sy0 - y0, sx0 - x0
    dy1, dx1 = dy0 + (sy1 - sy0), dx0 + (sx1 - sx0)

    out_shape = arr.shape[:-2] + (oh, ow)
    out = np.full(out_shape, pad_value, dtype=arr.dtype)
    out[..., dy0:dy1, dx0:dx1] = arr[..., sy0:sy1, sx0:sx1]
    return out


def crop_valid_mask(source_hw: Tuple[int, int], center_yx: Tuple[float, float], out_hw: Tuple[int, int]) -> np.ndarray:
    src = np.ones(tuple(int(x) for x in source_hw), dtype=np.uint8)
    return crop_fixed_2d(src, center_yx, out_hw, 0).astype(bool)


def crop_phase(sequence: np.ndarray, mask: np.ndarray, out_hw: Tuple[int, int], pad_value=0.0,
               center_yx: Tuple[float, float] | None = None):
    if sequence.shape[1:] != mask.shape:
        raise ValueError(f"Mask/frame mismatch: sequence={sequence.shape[1:]}, mask={mask.shape}")
    center = mask_center(mask) if center_yx is None else tuple(float(x) for x in center_yx)
    valid = crop_valid_mask(mask.shape, center, out_hw)
    return (
        crop_fixed_2d(sequence, center, out_hw, pad_value),
        crop_fixed_2d(mask.astype(np.uint8), center, out_hw, 0).astype(bool),
        valid,
        center,
    )

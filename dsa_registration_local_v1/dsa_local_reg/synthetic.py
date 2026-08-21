from __future__ import annotations

from typing import Any

import numpy as np

from .local_geometry import BBox, crop_with_border_median_padding, resize_whole_canvas, scale_bbox


def jacobian_det_from_displacement_yx(displacement_yx: np.ndarray, spacing_yx: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    """Analytic-test oracle only; clinical Jacobian stays in V5 registration_ants.py.

    ``displacement_yx[..., 0]`` is u_y and ``[..., 1]`` is u_x.
    """
    field = np.asarray(displacement_yx, dtype=np.float64)
    if field.ndim != 3 or field.shape[-1] != 2:
        raise ValueError(f"Expected [H,W,2] field, got {field.shape}")
    sy, sx = map(float, spacing_yx)
    uy, ux = field[..., 0], field[..., 1]
    duy_dy, duy_dx = np.gradient(uy, sy, sx)
    dux_dy, dux_dx = np.gradient(ux, sy, sx)
    return (1.0 + dux_dx) * (1.0 + duy_dy) - dux_dy * duy_dx


def _grid(shape_yx: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape_yx
    y, x = np.mgrid[:height, :width].astype(np.float64)
    return y, x


def translation_field(shape_yx: tuple[int, int], tx: float, ty: float) -> np.ndarray:
    field = np.zeros((*shape_yx, 2), dtype=np.float64)
    field[..., 0] = float(ty)
    field[..., 1] = float(tx)
    return field


def rotation_field(shape_yx: tuple[int, int], angle_rad: float) -> np.ndarray:
    y, x = _grid(shape_yx)
    cy, cx = (shape_yx[0] - 1) / 2.0, (shape_yx[1] - 1) / 2.0
    yy, xx = y - cy, x - cx
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    x_new, y_new = c * xx - s * yy, s * xx + c * yy
    field = np.empty((*shape_yx, 2), dtype=np.float64)
    field[..., 0] = y_new - yy
    field[..., 1] = x_new - xx
    return field


def local_radial_field(shape_yx: tuple[int, int], amplitude: float, sigma: float) -> np.ndarray:
    y, x = _grid(shape_yx)
    cy, cx = (shape_yx[0] - 1) / 2.0, (shape_yx[1] - 1) / 2.0
    dy, dx = y - cy, x - cx
    weight = float(amplitude) * np.exp(-(dx * dx + dy * dy) / (2.0 * float(sigma) ** 2))
    field = np.empty((*shape_yx, 2), dtype=np.float64)
    field[..., 0] = weight * dy
    field[..., 1] = weight * dx
    return field


def folding_field(shape_yx: tuple[int, int]) -> np.ndarray:
    _, x = _grid(shape_yx)
    field = np.zeros((*shape_yx, 2), dtype=np.float64)
    field[..., 1] = -1.5 * x
    return field


def uniform_scale_field(shape_yx: tuple[int, int], scale: float) -> np.ndarray:
    y, x = _grid(shape_yx)
    cy, cx = (shape_yx[0] - 1) / 2.0, (shape_yx[1] - 1) / 2.0
    field = np.empty((*shape_yx, 2), dtype=np.float64)
    field[..., 0] = (float(scale) - 1.0) * (y - cy)
    field[..., 1] = (float(scale) - 1.0) * (x - cx)
    return field


def _result(name: str, measurement: str, value: float, comparison: str, tolerance: float, passed: bool, **extra: Any) -> dict[str, Any]:
    return {
        "name": name,
        "measurement": measurement,
        "value": float(value),
        "comparison": comparison,
        "tolerance": float(tolerance),
        "passed": bool(passed),
        **extra,
    }


def run_synthetic_suite() -> list[dict[str, Any]]:
    shape = (129, 131)
    center = (shape[0] // 2, shape[1] // 2)
    results: list[dict[str, Any]] = []

    translation_j = jacobian_det_from_displacement_yx(translation_field(shape, tx=17.0, ty=-9.0))
    translation_error = float(np.max(np.abs(translation_j - 1.0)))
    results.append(_result("translation_residual_logj_zero", "max_abs_J_minus_1", translation_error, "<=", 1e-10, translation_error <= 1e-10))

    rotation_j = jacobian_det_from_displacement_yx(rotation_field(shape, angle_rad=np.deg2rad(12.0)))
    interior = rotation_j[2:-2, 2:-2]
    rotation_error = float(np.max(np.abs(interior - 1.0)))
    results.append(_result("rotation_residual_logj_zero", "interior_max_abs_J_minus_1", rotation_error, "<=", 1e-10, rotation_error <= 1e-10))

    raw_scale = 1.20
    raw_scale_j = jacobian_det_from_displacement_yx(uniform_scale_field(shape, raw_scale))
    raw_scale_error = float(abs(np.median(raw_scale_j) - raw_scale ** 2))
    results.append(_result("raw_global_scale_has_s_squared_determinant", "abs_median_J_minus_s_squared", raw_scale_error, "<=", 1e-10, raw_scale_error <= 1e-10, expected=float(raw_scale ** 2)))

    residual_identity = jacobian_det_from_displacement_yx(np.zeros((*shape, 2), dtype=np.float64))
    residual_scale_error = float(np.max(np.abs(residual_identity - 1.0)))
    results.append(_result("global_scale_removed_residual_identity", "max_abs_J_minus_1", residual_scale_error, "<=", 1e-10, residual_scale_error <= 1e-10))

    expansion_j = jacobian_det_from_displacement_yx(local_radial_field(shape, amplitude=0.08, sigma=20.0))
    contraction_j = jacobian_det_from_displacement_yx(local_radial_field(shape, amplitude=-0.08, sigma=20.0))
    expansion_logj = float(np.log(expansion_j[center]))
    contraction_logj = float(np.log(contraction_j[center]))
    results.append(_result("local_expansion_positive_logj", "center_logJ", expansion_logj, ">", 0.0, expansion_logj > 0.0))
    results.append(_result("local_contraction_negative_logj", "center_logJ", contraction_logj, "<", 0.0, contraction_logj < 0.0))

    fold_j = jacobian_det_from_displacement_yx(folding_field(shape))
    fold_fraction = float(np.mean(fold_j <= 0.0))
    results.append(_result("folding_detectable", "fold_fraction", fold_fraction, ">", 0.0, fold_fraction > 0.0))

    native = np.arange(80 * 120, dtype=np.uint8).reshape(80, 120)
    box = BBox(20, 10, 100, 70)
    independent = crop_with_border_median_padding(native, box)
    normalized = resize_whole_canvas(native, (160, 240))
    normalized_box = scale_bbox(box, native.shape, normalized.shape)
    normalized_crop = crop_with_border_median_padding(normalized, normalized_box)
    extent_error = float(max(abs(normalized_box.width - 2 * box.width), abs(normalized_box.height - 2 * box.height)))
    results.append(_result(
        "g1_whole_canvas_scale_preserves_bbox_extent", "max_bbox_extent_error_pixels", extent_error, "<=", 0.0,
        extent_error <= 0.0, g0_shape=list(independent.image.shape), g1_shape=list(normalized_crop.image.shape),
    ))

    same_size = float(independent.image.shape == normalized_crop.image.shape)
    results.append(_result(
        "independent_fixed_moving_matrix_sizes_allowed", "same_matrix_size_indicator", same_size, "==", 0.0,
        same_size == 0.0, moving_shape=list(independent.image.shape), fixed_shape=list(normalized_crop.image.shape),
    ))
    return results

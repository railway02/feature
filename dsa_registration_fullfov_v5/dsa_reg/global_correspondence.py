"""Deterministic whole-FOV vascular correspondence for Pre→Post DSA registration."""
from __future__ import annotations

import math
import cv2
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label

from .registration_sitk import resample


def _sitk():
    import SimpleITK as sitk
    return sitk


def canonical_similarity_matrix(shape, angle_deg: float, scale: float,
                                tx: float = 0.0, ty: float = 0.0) -> np.ndarray:
    """Return an OpenCV moving→fixed similarity matrix in canvas pixels."""
    h, w = (int(shape[0]), int(shape[1]))
    center = ((w - 1.0) / 2.0, (h - 1.0) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(angle_deg), float(scale)).astype(np.float64)
    matrix[0, 2] += float(tx)
    matrix[1, 2] += float(ty)
    return matrix


def canonical_matrix_to_resampling_transform(matrix: np.ndarray, kind: str):
    """Convert moving→fixed ``y=A x+b`` into a SimpleITK fixed→moving transform.

    SimpleITK resampling transforms consume output/fixed coordinates and return
    input/moving coordinates.  The inverse of the canonical matrix is therefore stored.
    The input and output domains are the same 2-D canvas and translations are pixels.
    Singular or reflected transforms are rejected.
    """
    sitk = _sitk()
    m = np.asarray(matrix, dtype=float)
    a = m[:, :2]
    b = m[:, 2]
    det = float(np.linalg.det(a))
    if not np.isfinite(det) or det <= 0:
        raise ValueError(f"Invalid canonical linear determinant: {det}")
    ai = np.linalg.inv(a)
    bi = -ai @ b
    if kind == "affine":
        out = sitk.AffineTransform(2)
        out.SetMatrix(tuple(ai.reshape(-1)))
        out.SetTranslation(tuple(float(x) for x in bi))
        return out
    scale_i = float(math.sqrt(np.linalg.det(ai)))
    angle_i = float(math.atan2(ai[1, 0], ai[0, 0]))
    if kind == "rigid":
        out = sitk.Euler2DTransform()
        out.SetCenter((0.0, 0.0))
        out.SetAngle(angle_i)
        out.SetTranslation(tuple(float(x) for x in bi))
        return out
    if kind == "similarity":
        out = sitk.Similarity2DTransform()
        out.SetCenter((0.0, 0.0))
        out.SetScale(scale_i)
        out.SetAngle(angle_i)
        out.SetTranslation(tuple(float(x) for x in bi))
        return out
    raise ValueError(kind)


def _resize_for_coarse(arr: np.ndarray, max_dim: int, is_mask=False):
    h, w = arr.shape
    factor = min(1.0, float(max_dim) / float(max(h, w)))
    if factor == 1.0:
        return np.asarray(arr).copy(), factor
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    out = cv2.resize(np.asarray(arr, dtype=np.uint8 if is_mask else np.float32),
                     (max(1, int(round(w * factor))), max(1, int(round(h * factor)))),
                     interpolation=interp)
    return (out > 0 if is_mask else out.astype(np.float32)), factor


def _largest_component(mask: np.ndarray) -> np.ndarray:
    lab, n = label(np.asarray(mask, dtype=bool))
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    return lab == int(np.argmax(sizes))


def _aligned_metrics(a: np.ndarray, b: np.ndarray, overlap: np.ndarray,
                     tolerances=(3, 5, 8), trim_percentile=90.0) -> dict:
    a = np.asarray(a, bool) & np.asarray(overlap, bool)
    b = np.asarray(b, bool) & np.asarray(overlap, bool)
    if not np.any(a) or not np.any(b):
        return {"score": -1e6, "fov_overlap": float(np.mean(overlap)),
                "trimmed_chamfer": np.inf, "major_trunk_coverage_5": 0.0,
                **{f"coverage_{t}_moving": 0.0 for t in tolerances},
                **{f"coverage_{t}_fixed": 0.0 for t in tolerances}}
    d_to_b = distance_transform_edt(~b)
    d_to_a = distance_transform_edt(~a)
    da, db = d_to_b[a], d_to_a[b]
    trim = float(np.mean(np.concatenate([
        np.minimum(da, np.percentile(da, trim_percentile)),
        np.minimum(db, np.percentile(db, trim_percentile)),
    ])))
    out = {"fov_overlap": float(np.mean(overlap)), "trimmed_chamfer": trim}
    for t in tolerances:
        out[f"coverage_{t}_moving"] = float(np.mean(da <= float(t)))
        out[f"coverage_{t}_fixed"] = float(np.mean(db <= float(t)))
    major_a = _largest_component(a)
    major_b = _largest_component(b)
    if np.any(major_a) and np.any(major_b):
        ma = distance_transform_edt(~major_b)[major_a]
        mb = distance_transform_edt(~major_a)[major_b]
        out["major_trunk_coverage_5"] = float(0.5 * (np.mean(ma <= 5) + np.mean(mb <= 5)))
    else:
        out["major_trunk_coverage_5"] = 0.0
    cov3 = .5 * (out.get("coverage_3_moving", 0) + out.get("coverage_3_fixed", 0))
    cov5 = .5 * (out.get("coverage_5_moving", 0) + out.get("coverage_5_fixed", 0))
    cov8 = .5 * (out.get("coverage_8_moving", 0) + out.get("coverage_8_fixed", 0))
    out["score"] = float(.25 * cov3 + .30 * cov5 + .20 * cov8 +
                         .15 * out["major_trunk_coverage_5"] + .10 * out["fov_overlap"] -
                         .01 * min(trim, 30.0))
    return out


def compatibility_metrics(moving_skeleton: np.ndarray, fixed_skeleton: np.ndarray,
                          moving_valid: np.ndarray, fixed_valid: np.ndarray,
                          transform, tolerances=(3, 5, 8), trim_percentile=90.0) -> dict:
    """Measure whole-FOV vascular correspondence after a candidate global transform.

    Coverage is reported bidirectionally because one DSA acquisition can contain branches
    absent from the other.  Distances are canvas pixels.  These are technical registration
    metrics, not disease features.
    """
    warped = resample(np.asarray(moving_skeleton, np.uint8), np.asarray(fixed_skeleton, np.uint8),
                      transform, is_mask=True)
    warped_valid = resample(np.asarray(moving_valid, np.uint8), np.asarray(fixed_valid, np.uint8),
                            transform, is_mask=True)
    overlap = np.asarray(fixed_valid, bool) & warped_valid
    out = _aligned_metrics(warped, fixed_skeleton, overlap, tolerances, trim_percentile)
    out["fov_overlap"] = float(np.sum(overlap) / max(1, np.sum(fixed_valid)))
    return out


def coarse_similarity_candidates(fixed_structure: np.ndarray, moving_structure: np.ndarray,
                                 fixed_skeleton: np.ndarray, moving_skeleton: np.ndarray,
                                 fixed_valid: np.ndarray, moving_valid: np.ndarray,
                                 cfg: dict, kind: str) -> list[dict]:
    """Generate deterministic full-FOV rotation/scale/translation initialisations.

    Rotation and scale are enumerated on a downsampled canvas.  For every hypothesis,
    phase correlation estimates the large translation, after which vascular centreline
    correspondence ranks candidates.  Returned transforms are fixed→moving SimpleITK
    resampling transforms ready for production refinement.
    """
    max_dim = int(cfg.get("coarse_max_dim", 256))
    fi, factor = _resize_for_coarse(fixed_structure, max_dim)
    mi, factor_m = _resize_for_coarse(moving_structure, max_dim)
    if abs(factor - factor_m) > 1e-6 or fi.shape != mi.shape:
        raise ValueError("Fixed/moving full-FOV canvases must have one shared coarse geometry")
    fs, _ = _resize_for_coarse(fixed_skeleton, max_dim, is_mask=True)
    ms, _ = _resize_for_coarse(moving_skeleton, max_dim, is_mask=True)
    fv, _ = _resize_for_coarse(fixed_valid, max_dim, is_mask=True)
    mv, _ = _resize_for_coarse(moving_valid, max_dim, is_mask=True)
    angles = [float(x) for x in cfg.get("coarse_rotation_degrees", [-12, -8, -4, 0, 4, 8, 12])]
    scales = [1.0] if kind == "rigid" else [float(x) for x in cfg.get(
        "coarse_scales", [0.80, 0.90, 1.0, 1.10, 1.20]
    )]
    window = cv2.createHanningWindow((fi.shape[1], fi.shape[0]), cv2.CV_32F)
    candidates = []
    for angle in angles:
        for scale in scales:
            mc = canonical_similarity_matrix(mi.shape, angle, scale)
            rotated = cv2.warpAffine(mi, mc, (fi.shape[1], fi.shape[0]), flags=cv2.INTER_LINEAR)
            shift, response = cv2.phaseCorrelate(rotated.astype(np.float32), fi.astype(np.float32), window)
            # Also retain a no-shift hypothesis; phase correlation can be distracted by
            # partial projection mismatch.  The geometric scorer chooses deterministically.
            for dx, dy, source in ((shift[0], shift[1], "phase_correlation"), (0.0, 0.0, "zero_shift")):
                coarse_matrix = mc.copy()
                coarse_matrix[0, 2] += float(dx)
                coarse_matrix[1, 2] += float(dy)
                warped_ms = cv2.warpAffine(ms.astype(np.uint8), coarse_matrix,
                                           (fi.shape[1], fi.shape[0]), flags=cv2.INTER_NEAREST) > 0
                warped_mv = cv2.warpAffine(mv.astype(np.uint8), coarse_matrix,
                                           (fi.shape[1], fi.shape[0]), flags=cv2.INTER_NEAREST) > 0
                metrics = _aligned_metrics(
                    warped_ms, fs, fv & warped_mv, (3, 5, 8), trim_percentile=90.0
                )
                full = canonical_similarity_matrix(
                    fixed_structure.shape, angle, scale, dx / factor, dy / factor
                )
                tx = canonical_matrix_to_resampling_transform(full, kind)
                candidates.append({
                    "transform": tx, "canonical_matrix": full,
                    "angle_deg": angle, "scale": scale,
                    "translation_x": float(dx / factor), "translation_y": float(dy / factor),
                    "translation_source": source, "phase_response": float(response),
                    "coarse_metrics": metrics,
                })
    candidates.sort(key=lambda x: (x["coarse_metrics"]["score"], x["phase_response"]), reverse=True)
    keep = max(1, int(cfg.get("coarse_keep_candidates", 4)))
    return candidates[:keep]


def broad_fixed_metric_support(vessel_mask: np.ndarray, valid_mask: np.ndarray,
                               dilation_px: int = 32) -> np.ndarray:
    """Broad fixed support for local refinement after coarse capture."""
    return binary_dilation(np.asarray(vessel_mask, bool), iterations=max(0, int(dilation_px))) & np.asarray(valid_mask, bool)


def global_compatibility_status(metrics: dict, params: dict, cfg: dict) -> tuple[str, list[str]]:
    """Return GLOBAL_PASS/PASS_WITH_CAUTION/FAIL without using outcomes."""
    fail, caution = [], []
    cov5 = .5 * (metrics.get("coverage_5_moving", 0) + metrics.get("coverage_5_fixed", 0))
    cov8 = .5 * (metrics.get("coverage_8_moving", 0) + metrics.get("coverage_8_fixed", 0))
    if cov8 < float(cfg.get("global_min_bidirectional_coverage_8", 0.20)):
        fail.append("low_bidirectional_coverage_8")
    elif cov5 < float(cfg.get("global_caution_bidirectional_coverage_5", 0.25)):
        caution.append("low_bidirectional_coverage_5")
    if metrics.get("trimmed_chamfer", np.inf) > float(cfg.get("global_max_trimmed_chamfer", 18.0)):
        # DSA projections can contain many phase-specific distal branches.  Whole-tree
        # Chamfer is therefore a caution signal; a true hard failure additionally needs
        # low matched coverage or several transform-plausibility warnings.
        caution.append("partial_projection_unmatched_branches")
    if metrics.get("fov_overlap", 0) < float(cfg.get("global_min_fov_overlap", 0.55)):
        fail.append("fov_overlap_low")
    elif metrics.get("fov_overlap", 0) < float(cfg.get("global_caution_fov_overlap", 0.60)):
        caution.append("fov_overlap_reduced")
    rotation = abs(float(params.get("rotation_deg", 0)))
    scale = float(params.get("scale", 1.0))
    if rotation > float(cfg.get("global_max_abs_rotation_deg", 18.0)):
        fail.append("rotation_implausible")
    elif rotation > float(cfg.get("global_caution_abs_rotation_deg", 12.0)):
        caution.append("rotation_large")
    if not (float(cfg.get("global_min_scale", 0.70)) <= scale <= float(cfg.get("global_max_scale", 1.35))):
        fail.append("scale_implausible")
    elif not (float(cfg.get("global_caution_min_scale", 0.80)) <= scale <=
              float(cfg.get("global_caution_max_scale", 1.25))):
        caution.append("scale_large")
    if metrics.get("score", -np.inf) < float(cfg.get("global_caution_min_score", 0.20)):
        caution.append("global_score_low")
    # Several individually borderline acquisition parameters together are a strong
    # projection-incompatibility signature, as seen in the Train low-mapping pilot.
    if len(set(caution)) >= int(cfg.get("global_max_caution_reasons_before_fail", 3)):
        fail.append("joint_global_plausibility_failure")
    if fail:
        return "GLOBAL_FAIL", fail + caution
    if caution:
        return "GLOBAL_PASS_WITH_CAUTION", caution
    return "GLOBAL_PASS", []

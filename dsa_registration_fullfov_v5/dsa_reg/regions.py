from __future__ import annotations
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt


def _iters(mask: np.ndarray, n: int, op: str) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if n <= 0:
        return mask.copy()
    if op == "dilate":
        return binary_dilation(mask, iterations=int(n))
    if op == "erode":
        return binary_erosion(mask, iterations=int(n), border_value=0)
    raise ValueError(op)


def build_anchor(vessel_mask: np.ndarray, lesion_mask: np.ndarray, exclusion_px: int,
                 valid_mask: np.ndarray | None = None, max_distance_px: int | None = None) -> np.ndarray:
    """Build a pseudo/automatic stable-vascular global-registration anchor.

    The source is vesselness-derived rather than an annotated parent-vessel segmentation.
    Lesion pixels and an exclusion rim are removed.  When ``max_distance_px`` is set,
    only vascular pixels within that canvas-pixel distance of the lesion are retained;
    this suppresses text, edge artefacts and unrelated remote vessels without aligning by
    lesion centre.  The resulting anchor is for registration/QC, never a neck annotation.
    """
    excluded = _iters(lesion_mask, exclusion_px, "dilate")
    anchor = np.asarray(vessel_mask, dtype=bool) & ~excluded
    if max_distance_px is not None:
        lesion = np.asarray(lesion_mask, dtype=bool)
        if np.any(lesion):
            anchor &= distance_transform_edt(~lesion) <= float(max_distance_px)
    if valid_mask is not None:
        anchor &= np.asarray(valid_mask, dtype=bool)
    return anchor


def measurement_regions(pre_global_mask: np.ndarray, post_mask: np.ndarray,
                        pre_global_vessel: np.ndarray | None = None,
                        post_vessel: np.ndarray | None = None,
                        valid_mask: np.ndarray | None = None,
                        boundary_inner=3, boundary_outer=5,
                        peri_inner=8, peri_outer=24,
                        roi_margin=40, vessel_roi_dilate=5):
    lesion = np.asarray(pre_global_mask, dtype=bool) | np.asarray(post_mask, dtype=bool)
    er = _iters(lesion, boundary_inner, "erode")
    di = _iters(lesion, boundary_outer, "dilate")
    boundary = di & ~er
    peri = _iters(lesion, peri_outer, "dilate") & ~_iters(lesion, peri_inner, "dilate")

    vessel_union = np.zeros_like(lesion, dtype=bool)
    if pre_global_vessel is not None:
        vessel_union |= np.asarray(pre_global_vessel, dtype=bool)
    if post_vessel is not None:
        vessel_union |= np.asarray(post_vessel, dtype=bool)

    # "ROI" here means the expanded local measurement support, not every pixel in a
    # padded 320x320 crop. This prevents edge/padding deformation from dominating the
    # Jacobian summary or largest-component statistic.
    roi = _iters(lesion, roi_margin, "dilate")
    if np.any(vessel_union):
        roi |= _iters(vessel_union, vessel_roi_dilate, "dilate")
    if valid_mask is not None:
        roi &= np.asarray(valid_mask, dtype=bool)

    return {
        "roi": roi,
        "lesion": lesion & roi,
        "boundary": boundary & roi,
        "peri": peri & roi,
    }

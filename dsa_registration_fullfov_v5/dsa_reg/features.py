from __future__ import annotations
import numpy as np
from scipy.ndimage import label, binary_erosion, distance_transform_edt
from skimage.measure import regionprops


def _stats(arr: np.ndarray, mask: np.ndarray, prefix: str, quantiles=(.10,.25,.75,.90,.95)) -> dict:
    vals = np.asarray(arr)[np.asarray(mask, dtype=bool)]
    vals = vals[np.isfinite(vals)]
    keys = ["mean", "median", "std", "min", "max"] + [f"p{int(q*100):02d}" for q in quantiles]
    if vals.size == 0:
        return {f"{prefix}_{k}": np.nan for k in keys}
    out = {
        f"{prefix}_mean": float(np.mean(vals)),
        f"{prefix}_median": float(np.median(vals)),
        f"{prefix}_std": float(np.std(vals)),
        f"{prefix}_min": float(np.min(vals)),
        f"{prefix}_max": float(np.max(vals)),
    }
    for q in quantiles:
        out[f"{prefix}_p{int(q*100):02d}"] = float(np.quantile(vals, q))
    return out


def _boundary(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    return m & ~binary_erosion(m, iterations=1, border_value=0)


def boundary_distance_features(pre_mask: np.ndarray, post_mask: np.ndarray, prefix="morph_boundary") -> dict:
    a, b = _boundary(pre_mask), _boundary(post_mask)
    if not np.any(a) or not np.any(b):
        return {f"{prefix}_{k}": np.nan for k in ["mean", "median", "p90", "p95", "max"]}
    db = distance_transform_edt(~b)
    da = distance_transform_edt(~a)
    vals = np.concatenate([db[a], da[b]]).astype(np.float64)
    return {
        f"{prefix}_mean": float(np.mean(vals)),
        f"{prefix}_median": float(np.median(vals)),
        f"{prefix}_p90": float(np.quantile(vals, .90)),
        f"{prefix}_p95": float(np.quantile(vals, .95)),
        f"{prefix}_max": float(np.max(vals)),
    }


def morphology_features(pre_mask: np.ndarray, post_mask: np.ndarray, prefix="morph") -> dict:
    def one(m):
        m = np.asarray(m, dtype=bool)
        props = regionprops(m.astype(np.uint8))
        total_area = float(m.sum())
        if not props:
            return {
                "area": 0.0, "largest_area": 0.0, "perimeter": np.nan,
                "equiv_diameter": np.nan, "eccentricity": np.nan, "solidity": np.nan,
                "centroid_y": np.nan, "centroid_x": np.nan,
            }
        p = max(props, key=lambda x: x.area)
        # Lesion masks should normally be one component. Retain total area separately so
        # that small valid components are not silently discarded.
        perimeter_total = float(sum(float(q.perimeter) for q in props))
        return {
            "area": total_area,
            "largest_area": float(p.area),
            "perimeter": perimeter_total,
            "equiv_diameter": float(p.equivalent_diameter_area),
            "eccentricity": float(p.eccentricity),
            "solidity": float(p.solidity),
            "centroid_y": float(p.centroid[0]),
            "centroid_x": float(p.centroid[1]),
        }

    a, b = one(pre_mask), one(post_mask)
    out = {}
    for k in a:
        out[f"{prefix}_{k}_pre"] = a[k]
        out[f"{prefix}_{k}_post"] = b[k]
        out[f"{prefix}_{k}_delta"] = b[k] - a[k] if np.isfinite(a[k]) and np.isfinite(b[k]) else np.nan
    out[f"{prefix}_area_logratio"] = float(np.log((b["area"] + 1e-6) / (a["area"] + 1e-6)))
    if np.isfinite(a["equiv_diameter"]) and np.isfinite(b["equiv_diameter"]):
        out[f"{prefix}_equiv_diameter_logratio"] = float(
            np.log((b["equiv_diameter"] + 1e-6) / (a["equiv_diameter"] + 1e-6))
        )
    else:
        out[f"{prefix}_equiv_diameter_logratio"] = np.nan
    if all(np.isfinite([a["centroid_x"], a["centroid_y"], b["centroid_x"], b["centroid_y"]])):
        out[f"{prefix}_centroid_distance"] = float(
            np.hypot(b["centroid_x"] - a["centroid_x"], b["centroid_y"] - a["centroid_y"])
        )
    else:
        out[f"{prefix}_centroid_distance"] = np.nan
    out.update(boundary_distance_features(pre_mask, post_mask))
    return out


def deformation_features(displacement: np.ndarray, canonical_logjac: np.ndarray,
                         canonical_jac: np.ndarray, folding_mask: np.ndarray,
                         valid_mask: np.ndarray, regions: dict,
                         tau=0.05, quantiles=(.10,.25,.75,.90,.95)) -> dict:
    out = {}
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(canonical_logjac)
    for name, mask in regions.items():
        m = np.asarray(mask, dtype=bool) & valid
        out.update(_stats(displacement, m, f"disp_{name}", quantiles))
        out.update(_stats(canonical_logjac, m, f"logjac_{name}", quantiles))
        vals = canonical_logjac[m]
        out[f"logjac_{name}_expansion_ratio"] = float(np.mean(vals > tau)) if vals.size else np.nan
        out[f"logjac_{name}_contraction_ratio"] = float(np.mean(vals < -tau)) if vals.size else np.nan

    roi = np.asarray(regions["roi"], dtype=bool) & valid
    exp = roi & (canonical_logjac > tau)
    con = roi & (canonical_logjac < -tau)
    for key, binary in [("expansion", exp), ("contraction", con)]:
        lab, n = label(binary)
        largest = int(np.bincount(lab.ravel())[1:].max()) if n else 0
        denom = int(np.sum(roi))
        out[f"logjac_largest_{key}_area"] = float(largest)
        out[f"logjac_largest_{key}_ratio"] = float(largest / denom) if denom else np.nan

    roi_all = np.asarray(regions["roi"], dtype=bool)
    deformation_support = (np.asarray(valid_mask, dtype=bool) | np.asarray(folding_mask, dtype=bool)) & roi_all
    fold_roi = np.asarray(folding_mask, dtype=bool) & deformation_support
    denom_support = int(np.sum(deformation_support))
    denom_roi = int(np.sum(roi_all))
    out["folding_rate"] = float(np.sum(fold_roi) / denom_support) if denom_support else np.nan
    vals_abs = np.abs(canonical_logjac[roi])
    out["abs_logjac_p99"] = float(np.quantile(vals_abs, .99)) if vals_abs.size else np.nan
    out["canonical_valid_fraction_roi"] = float(denom_support / denom_roi) if denom_roi else np.nan

    if "lesion" in regions and "stable" in regions:
        lesion_mask = np.asarray(regions["lesion"], dtype=bool) & valid
        stable_mask = np.asarray(regions["stable"], dtype=bool) & valid
        a = canonical_logjac[lesion_mask]
        b = canonical_logjac[stable_mask]
        out["logjac_lesion_minus_stable_median"] = (
            float(np.median(a) - np.median(b)) if a.size and b.size else np.nan
        )
        da = displacement[lesion_mask & np.isfinite(displacement)]
        db = displacement[stable_mask & np.isfinite(displacement)]
        out["disp_lesion_minus_stable_median"] = (
            float(np.median(da) - np.median(db)) if da.size and db.size else np.nan
        )
    return out


def pure_numpy_jacobian(displacement_yx: np.ndarray, spacing_yx=(1.0, 1.0)):
    """Jacobian determinant of phi(x,y)=x+u for analytic/unit-test fields.

    displacement_yx[...,0] = u_y and displacement_yx[...,1] = u_x.
    """
    uy, ux = displacement_yx[..., 0], displacement_yx[..., 1]
    sy, sx = float(spacing_yx[0]), float(spacing_yx[1])
    duy_dy, duy_dx = np.gradient(uy, sy, sx)
    dux_dy, dux_dx = np.gradient(ux, sy, sx)
    return (1 + dux_dx) * (1 + duy_dy) - dux_dy * duy_dx

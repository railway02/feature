from __future__ import annotations

import math
from typing import Iterable

import cv2
import numpy as np
from scipy.signal import correlate
from skimage.morphology import skeletonize


REGIONS = ("artery", "vein", "vessel", "active_vessel")
CURVE_NAMES = (
    "peak", "auc", "onset10", "onset50", "ttp", "rise_duration",
    "washout_half_time", "fwhm", "max_up_slope", "max_down_slope",
    "early_mean", "middle_mean", "late_mean", "temporal_variation",
    "local_peak_count", "baseline_contamination",
)


def _safe_log_ratio(a: float, b: float, eps: float = 1e-8) -> float:
    return float(math.log(max(a, eps) / max(b, eps)))


def weighted_curve(enhancement: np.ndarray, weight: np.ndarray) -> np.ndarray:
    w = np.clip(weight.astype(np.float64), 0, None)
    denominator = max(float(w.sum()), 1e-8)
    return np.asarray([(frame * w).sum() / denominator for frame in enhancement], dtype=np.float64)


def curve_summary(curve: np.ndarray) -> dict[str, float]:
    curve = np.asarray(curve, dtype=np.float64)
    n = len(curve)
    if n == 0 or not np.isfinite(curve).all():
        return {name: float("nan") for name in CURVE_NAMES}
    times = np.linspace(0.0, 1.0, n)
    peak_position = int(np.argmax(curve))
    peak = float(curve[peak_position])
    onset10_hits = np.flatnonzero(curve >= 0.1 * max(peak, 1e-8))
    onset50_hits = np.flatnonzero(curve >= 0.5 * max(peak, 1e-8))
    onset10_pos = int(onset10_hits[0]) if len(onset10_hits) else 0
    onset50_pos = int(onset50_hits[0]) if len(onset50_hits) else 0
    after = np.flatnonzero((np.arange(n) >= peak_position) & (curve <= 0.5 * max(peak, 1e-8)))
    washout_half = float(times[int(after[0])]) if len(after) else float("nan")
    half = np.flatnonzero(curve >= 0.5 * max(peak, 1e-8))
    fwhm = float(times[half[-1]] - times[half[0]]) if len(half) else float("nan")
    slopes = np.diff(curve) / np.maximum(np.diff(times), 1e-8)
    early = curve[times < 1 / 3]
    middle = curve[(times >= 1 / 3) & (times < 2 / 3)]
    late = curve[times >= 2 / 3]
    peaks = sum(curve[i] > curve[i - 1] and curve[i] >= curve[i + 1] for i in range(1, n - 1))
    return {
        "peak": peak,
        "auc": float(np.trapz(curve, times)),
        "onset10": float(times[onset10_pos]),
        "onset50": float(times[onset50_pos]),
        "ttp": float(times[peak_position]),
        "rise_duration": float(max(times[peak_position] - times[onset10_pos], 0.0)),
        "washout_half_time": washout_half,
        "fwhm": fwhm,
        "max_up_slope": float(slopes.max()) if len(slopes) else float("nan"),
        "max_down_slope": float(slopes.min()) if len(slopes) else float("nan"),
        "early_mean": float(early.mean()) if len(early) else float("nan"),
        "middle_mean": float(middle.mean()) if len(middle) else float("nan"),
        "late_mean": float(late.mean()) if len(late) else float("nan"),
        "temporal_variation": float(curve.std()),
        "local_peak_count": float(peaks),
        "baseline_contamination": float(curve[: min(3, n)].mean() / max(peak, 1e-8)),
    }


def _hard_mask(prob: np.ndarray, fov: np.ndarray, threshold: float = 0.5, minimum_pixels: int = 64):
    mask = fov & (prob >= threshold)
    fallback = False
    if int(mask.sum()) < minimum_pixels:
        values = prob[fov]
        if values.size:
            quantile = np.percentile(values, max(0.0, 100.0 * (1.0 - minimum_pixels / values.size)))
            mask = fov & (prob >= quantile)
            fallback = True
    return mask, fallback


def _skeleton_counts(mask: np.ndarray) -> tuple[int, int, int]:
    skeleton = skeletonize(mask)
    if not skeleton.any():
        return 0, 0, 0
    neighbors = cv2.filter2D(skeleton.astype(np.uint8), -1, np.ones((3, 3), np.uint8)) - skeleton.astype(np.uint8)
    endpoints = int(np.sum(skeleton & (neighbors == 1)))
    branches = int(np.sum(skeleton & (neighbors >= 3)))
    return int(skeleton.sum()), branches, endpoints


def region_morphology(prob: np.ndarray, fov: np.ndarray, prefix: str) -> tuple[dict[str, float], dict[str, float]]:
    fov_pixels = max(int(fov.sum()), 1)
    hard, fallback = _hard_mask(prob, fov)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(hard.astype(np.uint8), 8)
    components = max(count - 1, 0)
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if components else 0
    skeleton_length, branches, endpoints = _skeleton_counts(hard)
    y, x = np.nonzero(hard)
    if len(x):
        xn = x / max(prob.shape[1] - 1, 1)
        yn = y / max(prob.shape[0] - 1, 1)
        cx, cy = float(xn.mean()), float(yn.mean())
        sx, sy = float(xn.std()), float(yn.std())
    else:
        cx = cy = sx = sy = float("nan")
    midx, midy = prob.shape[1] // 2, prob.shape[0] // 2
    left, right = hard[:, :midx].sum(), hard[:, midx:].sum()
    upper, lower = hard[:midy].sum(), hard[midy:].sum()
    features = {
        f"{prefix}_soft_area_ratio_fov": float(prob[fov].mean()) if fov.any() else 0.0,
        f"{prefix}_hard_area_ratio_03_fov": float((fov & (prob >= 0.3)).sum() / fov_pixels),
        f"{prefix}_hard_area_ratio_05_fov": float((fov & (prob >= 0.5)).sum() / fov_pixels),
        f"{prefix}_hard_area_ratio_07_fov": float((fov & (prob >= 0.7)).sum() / fov_pixels),
        f"{prefix}_mean_probability_fov": float(prob[fov].mean()) if fov.any() else 0.0,
        f"{prefix}_p90_probability_fov": float(np.percentile(prob[fov], 90)) if fov.any() else 0.0,
        f"{prefix}_components_per_10k_fov": float(components * 10000 / fov_pixels),
        f"{prefix}_largest_component_ratio": float(largest / max(int(hard.sum()), 1)),
        f"{prefix}_skeleton_length_ratio_fov": float(skeleton_length / fov_pixels),
        f"{prefix}_branch_density_skeleton": float(branches / max(skeleton_length, 1)),
        f"{prefix}_endpoint_density_skeleton": float(endpoints / max(skeleton_length, 1)),
        f"{prefix}_centroid_x": cx,
        f"{prefix}_centroid_y": cy,
        f"{prefix}_spread_x": sx,
        f"{prefix}_spread_y": sy,
        f"{prefix}_left_right_balance": float((right - left) / max(right + left, 1)),
        f"{prefix}_upper_lower_balance": float((lower - upper) / max(lower + upper, 1)),
    }
    return features, {f"{prefix}_hard_mask_fallback": float(fallback)}


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    y, x = np.nonzero(mask)
    if not len(x):
        return float("nan"), float("nan")
    return float(x.mean() / max(mask.shape[1] - 1, 1)), float(y.mean() / max(mask.shape[0] - 1, 1))


def spatial_relations(artery: np.ndarray, vein: np.ndarray, vessel: np.ndarray, vessel_or: np.ndarray, fov: np.ndarray) -> dict[str, float]:
    a_mask, _ = _hard_mask(artery, fov)
    v_mask, _ = _hard_mask(vein, fov)
    ac = _centroid(a_mask)
    vc = _centroid(v_mask)
    valid = fov & np.isfinite(artery) & np.isfinite(vein)
    corr = float(np.corrcoef(artery[valid], vein[valid])[0, 1]) if valid.sum() > 2 else float("nan")
    a_soft, v_soft, vs = float(artery[fov].mean()), float(vein[fov].mean()), float(vessel[fov].mean())
    return {
        "av_soft_area_log_ratio": _safe_log_ratio(a_soft, v_soft),
        "artery_fraction_of_vessel": float(a_soft / max(vs, 1e-8)),
        "vein_fraction_of_vessel": float(v_soft / max(vs, 1e-8)),
        "av_soft_overlap_ratio": float(np.minimum(artery, vein)[fov].mean()),
        "av_hard_overlap_ratio_fov": float((a_mask & v_mask).sum() / max(int(fov.sum()), 1)),
        "av_centroid_distance": float(math.hypot(ac[0] - vc[0], ac[1] - vc[1])) if np.isfinite(ac + vc).all() else float("nan"),
        "av_centroid_dx": float(ac[0] - vc[0]),
        "av_centroid_dy": float(ac[1] - vc[1]),
        "av_probability_correlation": corr,
        "vessel_or_minus_max_mean": float((vessel_or - vessel)[fov].mean()),
    }


def best_lag(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = (a - a.mean()) / max(a.std(), 1e-8)
    b = (b - b.mean()) / max(b.std(), 1e-8)
    values = correlate(a, b, mode="full") / max(len(a), 1)
    lags = np.arange(-len(b) + 1, len(a))
    position = int(np.argmax(values))
    return float(lags[position] / max(len(a) - 1, 1)), float(values[position])


def av_temporal_relations(a_curve: np.ndarray, v_curve: np.ndarray, a_stats: dict[str, float], v_stats: dict[str, float]) -> dict[str, float]:
    lag, lag_corr = best_lag(a_curve, v_curve)
    return {
        "av_onset10_difference": a_stats["onset10"] - v_stats["onset10"],
        "av_onset50_difference": a_stats["onset50"] - v_stats["onset50"],
        "av_ttp_difference": a_stats["ttp"] - v_stats["ttp"],
        "av_washout_half_difference": a_stats["washout_half_time"] - v_stats["washout_half_time"],
        "av_fwhm_difference": a_stats["fwhm"] - v_stats["fwhm"],
        "av_log_peak_ratio": _safe_log_ratio(a_stats["peak"], v_stats["peak"]),
        "av_log_auc_ratio": _safe_log_ratio(a_stats["auc"], v_stats["auc"]),
        "av_late_to_peak_difference": (a_stats["late_mean"] / max(a_stats["peak"], 1e-8)) - (v_stats["late_mean"] / max(v_stats["peak"], 1e-8)),
        "av_best_lag": lag,
        "av_best_lag_correlation": lag_corr,
    }


def build_scalar_bank(
    enhancement: np.ndarray,
    fov: np.ndarray,
    activity: np.ndarray,
    artery: np.ndarray,
    vein: np.ndarray,
    vessel: np.ndarray,
    vessel_or: np.ndarray,
    v3_bridge,
    frame_indices: list[int],
) -> tuple[dict[str, float], dict[str, np.ndarray], dict[str, float]]:
    activity_norm = activity.astype(np.float32)
    values = activity_norm[fov]
    if values.size:
        lo, hi = np.percentile(values, [5, 99])
        activity_norm = np.clip((activity_norm - lo) / max(float(hi - lo), 1e-8), 0, 1)
    active_vessel = vessel * activity_norm
    region_weights = {
        "artery": artery * fov,
        "vein": vein * fov,
        "vessel": vessel * fov,
        "active_vessel": active_vessel * fov,
    }
    features: dict[str, float] = {}
    qc: dict[str, float] = {}
    for name in ("artery", "vein", "vessel"):
        morph, region_qc = region_morphology(region_weights[name], fov, name)
        features.update(morph)
        qc.update(region_qc)
    features.update(spatial_relations(artery, vein, vessel, vessel_or, fov))

    curves: dict[str, np.ndarray] = {}
    summaries: dict[str, dict[str, float]] = {}
    for name, weight in region_weights.items():
        curve = weighted_curve(enhancement, weight)
        summary = curve_summary(curve)
        curves[name] = curve.astype(np.float32)
        summaries[name] = summary
        features.update({f"{name}_tdc_{key}": value for key, value in summary.items()})
    features.update(av_temporal_relations(curves["artery"], curves["vein"], summaries["artery"], summaries["vein"]))

    hard_active, fallback = _hard_mask(active_vessel, fov, threshold=0.25, minimum_pixels=64)
    qc["active_vessel_hard_mask_fallback"] = float(fallback)
    maps, kinetic, _valid, filling_curves, filling, _visible = v3_bridge.kinetic_and_filling(
        enhancement, frame_indices, fov, hard_active, summaries["vessel"]["peak"]
    )
    features.update({f"cave_{key}": float(value) for key, value in kinetic.items() if np.isscalar(value)})
    features.update({f"cave_{key}": float(value) for key, value in filling.items() if np.isscalar(value)})
    curves["filling_visible_area"] = filling_curves["visible_area_fraction"].to_numpy(np.float32)
    curves["filling_new_area"] = filling_curves["new_area_fraction"].to_numpy(np.float32)
    curves["filling_washout_area"] = filling_curves["washout_area_fraction"].to_numpy(np.float32)
    return features, curves, qc


def expected_scalar_count() -> int:
    return 3 * 17 + 10 + 4 * 16 + 10 + 57 + 14

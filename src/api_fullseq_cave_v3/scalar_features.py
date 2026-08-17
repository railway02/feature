from __future__ import annotations

import math
from collections import OrderedDict

import cv2
import numpy as np
from scipy.signal import correlate
from skimage.morphology import skeletonize

REGIONS = ("artery", "vein", "vessel", "active_vessel")
CURVE_NAMES = (
    "peak", "auc_per_observed_span", "onset10_relative", "onset50_relative",
    "ttp_relative", "rise_duration_frames", "washout_half_relative", "fwhm_relative",
    "max_up_slope_per_frame", "max_down_slope_per_frame", "early_mean", "middle_mean",
    "late_mean", "temporal_variation", "local_peak_count", "baseline_contamination",
)
MORPH_NAMES = (
    "soft_area_ratio_fov", "hard_area_ratio_03_fov", "hard_area_ratio_05_fov",
    "hard_area_ratio_07_fov", "mean_probability_fov", "p90_probability_fov",
    "components_per_10k_fov", "largest_component_ratio", "skeleton_length_ratio_fov",
    "branch_density_skeleton", "endpoint_density_skeleton", "centroid_x", "centroid_y",
    "spread_x", "spread_y", "left_right_balance", "upper_lower_balance",
)
SPATIAL_NAMES = (
    "av_soft_area_log_ratio", "artery_fraction_of_vessel", "vein_fraction_of_vessel",
    "av_soft_overlap_ratio", "av_hard_overlap_ratio_fov", "av_centroid_distance",
    "av_centroid_dx", "av_centroid_dy", "av_probability_correlation",
    "vessel_union_minus_max_mean",
)
AV_TEMPORAL_NAMES = (
    "av_onset10_difference", "av_onset50_difference", "av_ttp_difference",
    "av_washout_half_difference", "av_fwhm_difference", "av_log_peak_ratio",
    "av_log_auc_ratio", "av_late_to_peak_difference", "av_best_lag",
    "av_best_lag_correlation",
)
EXPECTED_KINETIC_COUNT = 57
EXPECTED_FILLING_COUNT = 14


def expected_scalar_count() -> int:
    return 3 * len(MORPH_NAMES) + len(SPATIAL_NAMES) + 4 * len(CURVE_NAMES) + len(AV_TEMPORAL_NAMES) + EXPECTED_KINETIC_COUNT + EXPECTED_FILLING_COUNT


def _safe_log_ratio(a: float, b: float, eps: float = 1e-8) -> float:
    return float(math.log(max(float(a), eps) / max(float(b), eps)))


def weighted_curve(enhancement: np.ndarray, weight: np.ndarray) -> np.ndarray:
    w = np.clip(weight.astype(np.float64), 0.0, None)
    denominator = max(float(w.sum()), 1e-8)
    return np.asarray([(frame * w).sum() / denominator for frame in enhancement], dtype=np.float64)


def _blocks(indices: np.ndarray) -> np.ndarray:
    result = np.zeros(len(indices), dtype=np.int64)
    current = 0
    for position in range(1, len(indices)):
        if indices[position] - indices[position - 1] != 1:
            current += 1
        result[position] = current
    return result


def _observed_auc(curve: np.ndarray, indices: np.ndarray, blocks: np.ndarray) -> float:
    area, span = 0.0, 0.0
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    for block in np.unique(blocks):
        valid = (blocks == block) & np.isfinite(curve)
        if valid.sum() < 2:
            continue
        x, y = indices[valid].astype(np.float64), curve[valid]
        local_span = float(x[-1] - x[0])
        if local_span <= 0:
            continue
        area += float(trap(y, x))
        span += local_span
    return float(area / span) if span > 0 else float("nan")


def _local_peak_count(curve: np.ndarray, blocks: np.ndarray) -> int:
    total = 0
    for block in np.unique(blocks):
        values = curve[blocks == block]
        total += sum(values[i] > values[i - 1] and values[i] >= values[i + 1] for i in range(1, len(values) - 1))
    return int(total)


def curve_summary(curve: np.ndarray, frame_indices: list[int]) -> OrderedDict[str, float]:
    curve = np.asarray(curve, dtype=np.float64)
    indices = np.asarray(frame_indices, dtype=np.float64)
    output = OrderedDict((name, float("nan")) for name in CURVE_NAMES)
    if len(curve) != len(indices) or len(curve) == 0 or not np.isfinite(curve).all():
        return output
    first, last = float(indices[0]), float(indices[-1])
    relative = (indices - first) / max(last - first, 1.0)
    blocks = _blocks(indices)
    peak_position = int(np.argmax(curve))
    peak = float(curve[peak_position])
    threshold_peak = max(peak, 1e-8)
    peak_block = blocks[peak_position]
    before = np.flatnonzero((blocks == peak_block) & (np.arange(len(curve)) <= peak_position))
    onset10_hits = before[curve[before] >= 0.1 * threshold_peak]
    onset50_hits = before[curve[before] >= 0.5 * threshold_peak]
    onset10 = int(onset10_hits[0]) if len(onset10_hits) else peak_position
    onset50 = int(onset50_hits[0]) if len(onset50_hits) else peak_position
    after = np.flatnonzero((blocks == peak_block) & (np.arange(len(curve)) >= peak_position) & (curve <= 0.5 * threshold_peak))
    washout_half = int(after[0]) if len(after) else None
    half = np.flatnonzero((blocks == peak_block) & (curve >= 0.5 * threshold_peak))
    fwhm = float(relative[half[-1]] - relative[half[0]]) if len(half) else float("nan")
    slopes: list[float] = []
    for block in np.unique(blocks):
        positions = np.flatnonzero(blocks == block)
        if len(positions) > 1:
            slopes.extend((np.diff(curve[positions]) / np.diff(indices[positions])).tolist())
    slopes_array = np.asarray(slopes, dtype=np.float64)
    early = curve[relative < 1 / 3]
    middle = curve[(relative >= 1 / 3) & (relative < 2 / 3)]
    late = curve[relative >= 2 / 3]
    output.update({
        "peak": peak,
        "auc_per_observed_span": _observed_auc(curve, indices, blocks),
        "onset10_relative": float(relative[onset10]),
        "onset50_relative": float(relative[onset50]),
        "ttp_relative": float(relative[peak_position]),
        "rise_duration_frames": float(indices[peak_position] - indices[onset10]),
        "washout_half_relative": float(relative[washout_half]) if washout_half is not None else float("nan"),
        "fwhm_relative": fwhm,
        "max_up_slope_per_frame": float(slopes_array.max()) if len(slopes_array) else float("nan"),
        "max_down_slope_per_frame": float(slopes_array.min()) if len(slopes_array) else float("nan"),
        "early_mean": float(early.mean()) if len(early) else float("nan"),
        "middle_mean": float(middle.mean()) if len(middle) else float("nan"),
        "late_mean": float(late.mean()) if len(late) else float("nan"),
        "temporal_variation": float(curve.std()),
        "local_peak_count": float(_local_peak_count(curve, blocks)),
        "baseline_contamination": float(curve[: min(3, len(curve))].mean() / threshold_peak),
    })
    return output


def _hard_mask(probability: np.ndarray, fov: np.ndarray, threshold: float = 0.5, minimum_pixels: int = 64) -> tuple[np.ndarray, bool]:
    mask = fov & (probability >= threshold)
    fallback = False
    if int(mask.sum()) < minimum_pixels:
        values = probability[fov]
        if len(values):
            target = min(minimum_pixels, len(values))
            quantile = np.percentile(values, max(0.0, 100.0 * (1.0 - target / len(values))))
            mask = fov & (probability >= quantile)
            fallback = True
    return mask, fallback


def _skeleton_counts(mask: np.ndarray) -> tuple[int, int, int]:
    skeleton = skeletonize(mask)
    if not skeleton.any():
        return 0, 0, 0
    kernel = np.ones((3, 3), np.uint8)
    neighbors = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel) - skeleton.astype(np.uint8)
    return int(skeleton.sum()), int(np.sum(skeleton & (neighbors >= 3))), int(np.sum(skeleton & (neighbors == 1)))


def region_morphology(probability: np.ndarray, fov: np.ndarray, prefix: str) -> tuple[OrderedDict[str, float], dict[str, float]]:
    fov_pixels = max(int(fov.sum()), 1)
    hard, fallback = _hard_mask(probability, fov)
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(hard.astype(np.uint8), 8)
    components = max(count - 1, 0)
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if components else 0
    skeleton_length, branches, endpoints = _skeleton_counts(hard)
    y, x = np.nonzero(hard)
    if len(x):
        xn, yn = x / max(probability.shape[1] - 1, 1), y / max(probability.shape[0] - 1, 1)
        cx, cy, sx, sy = float(xn.mean()), float(yn.mean()), float(xn.std()), float(yn.std())
    else:
        cx = cy = sx = sy = float("nan")
    midx, midy = probability.shape[1] // 2, probability.shape[0] // 2
    left, right = int(hard[:, :midx].sum()), int(hard[:, midx:].sum())
    upper, lower = int(hard[:midy].sum()), int(hard[midy:].sum())
    values = probability[fov]
    result = OrderedDict([
        (f"{prefix}_soft_area_ratio_fov", float(values.mean()) if len(values) else 0.0),
        (f"{prefix}_hard_area_ratio_03_fov", float((fov & (probability >= 0.3)).sum() / fov_pixels)),
        (f"{prefix}_hard_area_ratio_05_fov", float((fov & (probability >= 0.5)).sum() / fov_pixels)),
        (f"{prefix}_hard_area_ratio_07_fov", float((fov & (probability >= 0.7)).sum() / fov_pixels)),
        (f"{prefix}_mean_probability_fov", float(values.mean()) if len(values) else 0.0),
        (f"{prefix}_p90_probability_fov", float(np.percentile(values, 90)) if len(values) else 0.0),
        (f"{prefix}_components_per_10k_fov", float(components * 10000 / fov_pixels)),
        (f"{prefix}_largest_component_ratio", float(largest / max(int(hard.sum()), 1))),
        (f"{prefix}_skeleton_length_ratio_fov", float(skeleton_length / fov_pixels)),
        (f"{prefix}_branch_density_skeleton", float(branches / max(skeleton_length, 1))),
        (f"{prefix}_endpoint_density_skeleton", float(endpoints / max(skeleton_length, 1))),
        (f"{prefix}_centroid_x", cx), (f"{prefix}_centroid_y", cy),
        (f"{prefix}_spread_x", sx), (f"{prefix}_spread_y", sy),
        (f"{prefix}_left_right_balance", float((right - left) / max(right + left, 1))),
        (f"{prefix}_upper_lower_balance", float((lower - upper) / max(lower + upper, 1))),
    ])
    return result, {f"{prefix}_hard_mask_fallback": float(fallback)}


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    y, x = np.nonzero(mask)
    if not len(x):
        return float("nan"), float("nan")
    return float(x.mean() / max(mask.shape[1] - 1, 1)), float(y.mean() / max(mask.shape[0] - 1, 1))


def spatial_relations(artery: np.ndarray, vein: np.ndarray, vessel: np.ndarray, vessel_union: np.ndarray, fov: np.ndarray) -> OrderedDict[str, float]:
    artery_mask, _ = _hard_mask(artery, fov)
    vein_mask, _ = _hard_mask(vein, fov)
    ac, vc = _centroid(artery_mask), _centroid(vein_mask)
    valid = fov & np.isfinite(artery) & np.isfinite(vein)
    corr = float(np.corrcoef(artery[valid], vein[valid])[0, 1]) if valid.sum() > 2 and np.std(artery[valid]) > 0 and np.std(vein[valid]) > 0 else float("nan")
    a_soft = float(artery[fov].mean()) if fov.any() else 0.0
    v_soft = float(vein[fov].mean()) if fov.any() else 0.0
    vessel_soft = float(vessel[fov].mean()) if fov.any() else 0.0
    finite_centroids = all(math.isfinite(value) for value in (*ac, *vc))
    return OrderedDict([
        ("av_soft_area_log_ratio", _safe_log_ratio(a_soft, v_soft)),
        ("artery_fraction_of_vessel", float(a_soft / max(vessel_soft, 1e-8))),
        ("vein_fraction_of_vessel", float(v_soft / max(vessel_soft, 1e-8))),
        ("av_soft_overlap_ratio", float(np.minimum(artery, vein)[fov].mean()) if fov.any() else 0.0),
        ("av_hard_overlap_ratio_fov", float((artery_mask & vein_mask).sum() / max(int(fov.sum()), 1))),
        ("av_centroid_distance", float(math.hypot(ac[0] - vc[0], ac[1] - vc[1])) if finite_centroids else float("nan")),
        ("av_centroid_dx", float(ac[0] - vc[0])), ("av_centroid_dy", float(ac[1] - vc[1])),
        ("av_probability_correlation", corr),
        ("vessel_union_minus_max_mean", float((vessel_union - vessel)[fov].mean()) if fov.any() else 0.0),
    ])


def best_lag(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if len(a) != len(b) or len(a) < 2 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan"), float("nan")
    aa, bb = (a - a.mean()) / a.std(), (b - b.mean()) / b.std()
    values = correlate(aa, bb, mode="full") / max(len(a), 1)
    lags = np.arange(-len(b) + 1, len(a))
    position = int(np.argmax(values))
    return float(lags[position] / max(len(a) - 1, 1)), float(values[position])


def av_temporal_relations(a_curve: np.ndarray, v_curve: np.ndarray, a_stats: dict[str, float], v_stats: dict[str, float]) -> OrderedDict[str, float]:
    lag, lag_corr = best_lag(a_curve, v_curve)
    return OrderedDict([
        ("av_onset10_difference", a_stats["onset10_relative"] - v_stats["onset10_relative"]),
        ("av_onset50_difference", a_stats["onset50_relative"] - v_stats["onset50_relative"]),
        ("av_ttp_difference", a_stats["ttp_relative"] - v_stats["ttp_relative"]),
        ("av_washout_half_difference", a_stats["washout_half_relative"] - v_stats["washout_half_relative"]),
        ("av_fwhm_difference", a_stats["fwhm_relative"] - v_stats["fwhm_relative"]),
        ("av_log_peak_ratio", _safe_log_ratio(a_stats["peak"], v_stats["peak"])),
        ("av_log_auc_ratio", _safe_log_ratio(a_stats["auc_per_observed_span"], v_stats["auc_per_observed_span"])),
        ("av_late_to_peak_difference", a_stats["late_mean"] / max(a_stats["peak"], 1e-8) - v_stats["late_mean"] / max(v_stats["peak"], 1e-8)),
        ("av_best_lag", lag), ("av_best_lag_correlation", lag_corr),
    ])


def build_scalar_bank(
    enhancement: np.ndarray,
    fov: np.ndarray,
    activity: np.ndarray,
    artery: np.ndarray,
    vein: np.ndarray,
    vessel: np.ndarray,
    vessel_union: np.ndarray,
    v3_bridge,
    frame_indices: list[int],
) -> tuple[OrderedDict[str, float], dict[str, np.ndarray], dict[str, float]]:
    activity_norm = activity.astype(np.float32)
    values = activity_norm[fov]
    if len(values):
        low, high = np.percentile(values, [5, 99])
        activity_norm = np.clip((activity_norm - low) / max(float(high - low), 1e-8), 0, 1)
    active_vessel = vessel * activity_norm
    region_weights = OrderedDict([
        ("artery", artery * fov), ("vein", vein * fov),
        ("vessel", vessel * fov), ("active_vessel", active_vessel * fov),
    ])
    features: OrderedDict[str, float] = OrderedDict()
    qc: dict[str, float] = {}
    for name in ("artery", "vein", "vessel"):
        morphology, region_qc = region_morphology(region_weights[name], fov, name)
        features.update(morphology)
        qc.update(region_qc)
    features.update(spatial_relations(artery, vein, vessel, vessel_union, fov))

    curves: dict[str, np.ndarray] = {}
    summaries: dict[str, OrderedDict[str, float]] = {}
    for name, weight in region_weights.items():
        curve = weighted_curve(enhancement, weight)
        summary = curve_summary(curve, frame_indices)
        curves[name] = curve.astype(np.float32)
        summaries[name] = summary
        features.update((f"{name}_tdc_{key}", value) for key, value in summary.items())
    features.update(av_temporal_relations(curves["artery"], curves["vein"], summaries["artery"], summaries["vein"]))

    hard_active, fallback = _hard_mask(active_vessel, fov, threshold=0.25, minimum_pixels=64)
    qc["active_vessel_hard_mask_fallback"] = float(fallback)
    _maps, kinetic, _valid, filling_curves, filling, _visible = v3_bridge.kinetic_and_filling(
        enhancement, frame_indices, fov, hard_active, summaries["vessel"]["peak"]
    )
    kinetic_scalars = OrderedDict((f"cave_{key}", float(value)) for key, value in kinetic.items() if np.isscalar(value))
    filling_scalars = OrderedDict((f"cave_{key}", float(value)) for key, value in filling.items() if np.isscalar(value))
    if len(kinetic_scalars) != EXPECTED_KINETIC_COUNT:
        raise AssertionError(f"Expected {EXPECTED_KINETIC_COUNT} kinetic scalars, got {len(kinetic_scalars)}")
    if len(filling_scalars) != EXPECTED_FILLING_COUNT:
        raise AssertionError(f"Expected {EXPECTED_FILLING_COUNT} filling scalars, got {len(filling_scalars)}")
    features.update(kinetic_scalars)
    features.update(filling_scalars)
    curves["filling_visible_area"] = filling_curves["visible_area_fraction"].to_numpy(np.float32)
    curves["filling_new_area"] = filling_curves["new_area_fraction"].to_numpy(np.float32)
    curves["filling_washout_area"] = filling_curves["washout_area_fraction"].to_numpy(np.float32)
    if len(features) != expected_scalar_count():
        raise AssertionError(f"Scalar schema count {len(features)} != {expected_scalar_count()}")
    return features, curves, qc

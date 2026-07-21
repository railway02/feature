#!/usr/bin/env python3
"""Build stable series- and patient-level features from api_fullseq_v3 pairdata.

The extractor remains label-blind. Time is represented by actual frame indices
and relative position; it never claims seconds or physical blood velocity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PAIR_CURVES = [
    "active_res_mag_norm_median",
    "active_res_mag_norm_p90",
    "active_weighted_mag_norm_mean",
    "active_direction_coherence",
    "active_direction_entropy",
    "vessel_res_mag_norm_median",
    "vessel_res_mag_norm_p90",
    "vessel_weighted_mag_norm_mean",
    "vessel_direction_coherence",
    "vessel_direction_entropy",
    "filling_front_coverage_fov",
    "persistent_coverage_fov",
]
PAIR_STATS = [
    "median", "iqr", "p90", "auc_per_observed_span",
    "peak_relative_position", "late_minus_early",
]
STAGES = ("precontrast", "washin", "peak", "washout")
STAGE_METRICS = (
    "active_weighted_mag_norm_mean",
    "vessel_weighted_mag_norm_mean",
)
TDC_FEATURES = [
    "tdc_peak", "tdc_auc_per_observed_span", "tdc_onset10_frames",
    "tdc_onset10_relative", "tdc_ttp_frames", "tdc_ttp_relative",
    "tdc_rise_duration_frames", "tdc_robust_rise_slope_per_frame",
    "tdc_early_median", "tdc_middle_median", "tdc_late_median",
    "tdc_late_retention_ratio", "tdc_temporal_variation",
    "tdc_local_peak_count",
]
FILLING_FEATURES = [
    "filling_maximum_area_fraction", "filling_area_auc_per_observed_span",
    "filling_time_to_50_frames", "filling_peak_relative_position",
    "filling_late_area_ratio", "filling_largest_component_ratio_at_peak",
    "filling_spatial_spread_at_peak", "filling_new_area_auc_per_observed_span",
]
COUPLING_FEATURES = [
    "coupling_zero_lag_correlation", "coupling_best_lag_correlation",
    "coupling_best_lag_frames", "coupling_flow_peak_minus_tdc_peak_frames",
]
CORE_PHASE_FEATURES = (
    [f"pair_{metric}_{stat}" for metric in PAIR_CURVES for stat in PAIR_STATS]
    + [f"stage_{stage}_{metric}_median" for metric in STAGE_METRICS for stage in STAGES]
    + TDC_FEATURES + FILLING_FEATURES + COUPLING_FEATURES
)
CORRELATION_FEATURES = {
    "coupling_zero_lag_correlation", "coupling_best_lag_correlation",
}
RELATIVE_FEATURES = {
    name for name in CORE_PHASE_FEATURES
    if name.endswith("_relative") or name.endswith("_relative_position")
}
QC_PHASE_FEATURES = [
    "qc_n_frames", "qc_n_pairs", "qc_frame_span", "qc_temporal_block_count",
    "qc_gap_count", "qc_fov_ratio", "qc_active_ratio_fov",
    "qc_vessel_ratio_fov", "qc_background_ratio_fov",
    "qc_polarity_margin", "qc_polarity_ambiguous",
    "qc_baseline_start_position", "qc_baseline_frame_count",
    "qc_vessel_fallback_to_active", "qc_background_fallback",
    "qc_hard_valid_ratio_median", "qc_fb_relative_median",
    "qc_fb_relative_p90", "qc_uncertainty_log_median",
    "qc_uncertainty_log_p90", "qc_global_motion_mag_norm_median",
    "qc_soft_weight_median", "qc_tdc_rise_valid",
    "qc_washout_observation_adequate", "qc_tdc_washout_observed",
    "qc_washout_right_censored", "qc_coupling_lag_search_valid",
    "qc_pair_runtime_seconds_sum",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().casefold() in {"true", "1", "yes", "y"}


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(sanitize_json(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(temporary, path)


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return sanitize_json(value.tolist())
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is pd.NA:
        return None
    return value


def numeric(values: Iterable[Any]) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)


def finite(values: Iterable[Any]) -> np.ndarray:
    array = numeric(values)
    return array[np.isfinite(array)]


def finite_median(values: Iterable[Any]) -> float:
    array = finite(values)
    return float(np.median(array)) if array.size else float("nan")


def finite_percentile(values: Iterable[Any], percentile: float) -> float:
    array = finite(values)
    return float(np.percentile(array, percentile)) if array.size else float("nan")


def blocks_from_indices(indices: np.ndarray) -> np.ndarray:
    if indices.size == 0:
        return np.asarray([], dtype=int)
    blocks = np.zeros(indices.size, dtype=int)
    current = 0
    for i in range(1, indices.size):
        if indices[i] - indices[i - 1] != 1:
            current += 1
        blocks[i] = current
    return blocks


def blocks_from_pairs(pair: pd.DataFrame) -> np.ndarray:
    blocks = np.zeros(len(pair), dtype=int)
    current = 0
    for i in range(1, len(pair)):
        if int(pair.iloc[i]["frame_index_t"]) != int(pair.iloc[i - 1]["frame_index_t1"]):
            current += 1
        blocks[i] = current
    return blocks


def observed_auc(values: np.ndarray, times: np.ndarray, block_ids: np.ndarray) -> tuple[float, float]:
    total_area = 0.0
    total_span = 0.0
    for block in np.unique(block_ids):
        valid = (block_ids == block) & np.isfinite(values) & np.isfinite(times)
        if int(valid.sum()) < 2:
            continue
        x = times[valid]
        y = values[valid]
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        span = float(x[-1] - x[0])
        if span <= 0:
            continue
        trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        total_area += float(trap(y, x))
        total_span += span
    return (float(total_area / total_span), total_span) if total_span > 0 else (float("nan"), 0.0)


def thirds(values: np.ndarray, relative: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(values) & np.isfinite(relative)
    return (
        values[valid & (relative < 1 / 3)],
        values[valid & (relative >= 1 / 3) & (relative < 2 / 3)],
        values[valid & (relative >= 2 / 3)],
    )


def curve_summary(values: np.ndarray, times: np.ndarray, relative: np.ndarray, blocks: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(values) & np.isfinite(times) & np.isfinite(relative)
    if not valid.any():
        return {name: float("nan") for name in PAIR_STATS}
    selected = values[valid]
    p25, p75 = np.percentile(selected, [25, 75])
    auc, _ = observed_auc(values, times, blocks)
    valid_indices = np.flatnonzero(valid)
    peak_index = int(valid_indices[np.argmax(values[valid])])
    early, _, late = thirds(values, relative)
    return {
        "median": float(np.median(selected)),
        "iqr": float(p75 - p25),
        "p90": float(np.percentile(selected, 90)),
        "auc_per_observed_span": auc,
        "peak_relative_position": float(relative[peak_index]),
        "late_minus_early": (
            float(np.median(late) - np.median(early))
            if early.size and late.size else float("nan")
        ),
    }


def local_peak_count_by_block(values: np.ndarray, blocks: np.ndarray) -> int:
    count = 0
    for block in np.unique(blocks):
        curve = values[(blocks == block) & np.isfinite(values)]
        count += sum(
            curve[i] > curve[i - 1] and curve[i] >= curve[i + 1]
            for i in range(1, len(curve) - 1)
        )
    return int(count)


def robust_tdc(frame: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    table = frame.sort_values(["frame_index", "sequence_position"]).reset_index(drop=True)
    indices = numeric(table["frame_index"])
    curve = numeric(table["tdc_active_median"])
    valid = np.isfinite(indices) & np.isfinite(curve)
    indices = indices[valid]
    curve = curve[valid]
    if curve.size == 0:
        return (
            {name: float("nan") for name in TDC_FEATURES},
            {
                "qc_tdc_rise_valid": 0,
                "qc_washout_observation_adequate": 0,
                "qc_tdc_washout_observed": 0,
                "qc_washout_right_censored": 1,
            },
        )
    first, last = float(indices.min()), float(indices.max())
    relative = (indices - first) / max(last - first, 1.0)
    blocks = blocks_from_indices(indices)
    peak_position = int(np.argmax(curve))
    peak = float(curve[peak_position])
    peak_block = int(blocks[peak_position])
    same_block_before = np.flatnonzero((blocks == peak_block) & (np.arange(len(curve)) <= peak_position))
    onset_candidates = same_block_before[curve[same_block_before] >= 0.10 * peak] if peak > 0 else np.asarray([], dtype=int)
    onset_position = int(onset_candidates[0]) if onset_candidates.size else None

    rise_valid = 0
    rise_slope = float("nan")
    rise_duration = float("nan")
    onset_frames = float("nan")
    onset_relative = float("nan")
    if onset_position is not None:
        onset_frames = float(indices[onset_position] - first)
        onset_relative = float(relative[onset_position])
        segment = np.arange(onset_position, peak_position + 1)
        segment = segment[blocks[segment] == peak_block]
        if len(segment) >= 3 and np.all(np.diff(indices[segment]) == 1):
            slopes = np.diff(curve[segment]) / np.diff(indices[segment])
            slopes = slopes[np.isfinite(slopes)]
            if slopes.size >= 2:
                rise_slope = float(np.median(slopes))
                rise_duration = float(indices[peak_position] - indices[onset_position])
                rise_valid = 1

    post_positions = np.flatnonzero((blocks == peak_block) & (np.arange(len(curve)) > peak_position))
    adequate = int(len(post_positions) >= 4)
    washout_observed = 0
    if adequate and peak > 0:
        post = np.r_[peak_position, post_positions]
        slopes = np.diff(curve[post]) / np.diff(indices[post])
        median_decay = float(np.median(slopes[np.isfinite(slopes)])) if np.isfinite(slopes).any() else float("nan")
        reaches_half = bool(np.any(curve[post_positions] <= 0.5 * peak))
        meaningful_drop = bool(peak - curve[post_positions[-1]] >= 0.20 * peak)
        washout_observed = int(math.isfinite(median_decay) and median_decay < 0 and reaches_half and meaningful_drop)
    right_censored = int(not adequate)

    auc, _ = observed_auc(curve, indices, blocks)
    early, middle, late = thirds(curve, relative)
    ratio_valid = peak >= 1e-4 and late.size > 0
    features = {
        "tdc_peak": peak,
        "tdc_auc_per_observed_span": auc,
        "tdc_onset10_frames": onset_frames,
        "tdc_onset10_relative": onset_relative,
        "tdc_ttp_frames": float(indices[peak_position] - first),
        "tdc_ttp_relative": float(relative[peak_position]),
        "tdc_rise_duration_frames": rise_duration,
        "tdc_robust_rise_slope_per_frame": rise_slope,
        "tdc_early_median": float(np.median(early)) if early.size else float("nan"),
        "tdc_middle_median": float(np.median(middle)) if middle.size else float("nan"),
        "tdc_late_median": float(np.median(late)) if late.size else float("nan"),
        "tdc_late_retention_ratio": float(np.median(late) / peak) if ratio_valid else float("nan"),
        "tdc_temporal_variation": float(np.std(curve)),
        "tdc_local_peak_count": float(local_peak_count_by_block(curve, blocks)),
    }
    qc = {
        "qc_tdc_rise_valid": rise_valid,
        "qc_washout_observation_adequate": adequate,
        "qc_tdc_washout_observed": washout_observed,
        "qc_washout_right_censored": right_censored,
    }
    return features, qc


def filling_features(curves: pd.DataFrame) -> dict[str, float]:
    table = curves.sort_values(["frame_index", "sequence_position"]).reset_index(drop=True)
    indices = numeric(table["frame_index"])
    area = numeric(table["visible_area_fraction"])
    new_area = numeric(table["new_area_fraction"])
    largest = numeric(table["largest_component_ratio"])
    spread = numeric(table["spatial_spread"])
    valid = np.isfinite(indices) & np.isfinite(area)
    indices, area, new_area, largest, spread = (
        values[valid] for values in (indices, area, new_area, largest, spread)
    )
    if area.size == 0:
        return {name: float("nan") for name in FILLING_FEATURES}
    first, last = float(indices.min()), float(indices.max())
    relative = (indices - first) / max(last - first, 1.0)
    blocks = blocks_from_indices(indices)
    peak_pos = int(np.argmax(area))
    peak = float(area[peak_pos])
    area_auc, _ = observed_auc(area, indices, blocks)
    new_auc, _ = observed_auc(new_area, indices, blocks)
    threshold = np.flatnonzero(area >= 0.5 * peak) if peak > 0 else np.asarray([], dtype=int)
    _, _, late = thirds(area, relative)
    return {
        "filling_maximum_area_fraction": peak,
        "filling_area_auc_per_observed_span": area_auc,
        "filling_time_to_50_frames": float(indices[threshold[0]] - first) if threshold.size else float("nan"),
        "filling_peak_relative_position": float(relative[peak_pos]),
        "filling_late_area_ratio": float(np.median(late) / peak) if late.size and peak >= 1e-8 else float("nan"),
        "filling_largest_component_ratio_at_peak": float(largest[peak_pos]) if np.isfinite(largest[peak_pos]) else float("nan"),
        "filling_spatial_spread_at_peak": float(spread[peak_pos]) if np.isfinite(spread[peak_pos]) else float("nan"),
        "filling_new_area_auc_per_observed_span": new_auc,
    }


def safe_corr(x: np.ndarray, y: np.ndarray, minimum_n: int = 3) -> tuple[float, int]:
    valid = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[valid], y[valid]
    if len(xv) < minimum_n or np.std(xv) <= 1e-12 or np.std(yv) <= 1e-12:
        return float("nan"), int(len(xv))
    return float(np.clip(np.corrcoef(xv, yv)[0, 1], -1.0, 1.0)), int(len(xv))


def lagged_corr(flow: np.ndarray, derivative: np.ndarray, blocks: np.ndarray, lag: int) -> tuple[float, int]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for block in np.unique(blocks):
        x = flow[blocks == block]
        y = derivative[blocks == block]
        if lag > 0 and len(x) > lag:
            xs.append(x[:-lag]); ys.append(y[lag:])
        elif lag < 0 and len(x) > -lag:
            xs.append(x[-lag:]); ys.append(y[:lag])
        elif lag == 0:
            xs.append(x); ys.append(y)
    if not xs:
        return float("nan"), 0
    return safe_corr(np.concatenate(xs), np.concatenate(ys), minimum_n=3)


def coupling_features(pair: pd.DataFrame, frame: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    table = pair.sort_values(["frame_index_t", "frame_index_t1", "pair_order"]).reset_index(drop=True)
    flow_column = "vessel_weighted_mag_norm_mean" if "vessel_weighted_mag_norm_mean" in table.columns else "active_weighted_mag_norm_mean"
    flow = numeric(table[flow_column])
    derivative = numeric(table["tdc_derivative_pair"])
    mid = 0.5 * (numeric(table["frame_index_t"]) + numeric(table["frame_index_t1"]))
    blocks = blocks_from_pairs(table)
    zero, _ = lagged_corr(flow, derivative, blocks, 0)
    best_corr = float("nan")
    best_lag = float("nan")
    lag_valid = 0
    if len(table) >= 10:
        candidates: list[tuple[float, int, float, int]] = []
        for lag in range(-3, 4):
            value, overlap = lagged_corr(flow, derivative, blocks, lag)
            if overlap >= 6 and math.isfinite(value):
                candidates.append((abs(value), lag, value, overlap))
        if candidates:
            _, lag, value, _ = max(candidates, key=lambda item: (item[0], -abs(item[1])))
            best_corr = float(value)
            best_lag = float(lag)
            lag_valid = 1

    valid_flow = np.isfinite(flow) & np.isfinite(mid)
    flow_peak = float(mid[np.flatnonzero(valid_flow)[np.argmax(flow[valid_flow])]]) if valid_flow.any() else float("nan")
    frame_table = frame.sort_values(["frame_index", "sequence_position"]).reset_index(drop=True)
    tdc = numeric(frame_table["tdc_active_median"])
    frame_indices = numeric(frame_table["frame_index"])
    valid_tdc = np.isfinite(tdc) & np.isfinite(frame_indices)
    tdc_peak = float(frame_indices[np.flatnonzero(valid_tdc)[np.argmax(tdc[valid_tdc])]]) if valid_tdc.any() else float("nan")
    return ({
        "coupling_zero_lag_correlation": zero,
        "coupling_best_lag_correlation": best_corr,
        "coupling_best_lag_frames": best_lag,
        "coupling_flow_peak_minus_tdc_peak_frames": flow_peak - tdc_peak if math.isfinite(flow_peak) and math.isfinite(tdc_peak) else float("nan"),
    }, {"qc_coupling_lag_search_valid": lag_valid})


@dataclass(frozen=True)
class PhasePlan:
    patient_id: str
    series_uid: str
    split: str
    phase: str
    selected_series_id: str
    selected_internal_series: str
    expected_pairs: int
    expected_frames: int
    frame_list_hash: str


def plans_from_manifest(manifest: pd.DataFrame) -> list[PhasePlan]:
    plans: list[PhasePlan] = []
    for row in manifest.to_dict("records"):
        for phase in ("pre", "post"):
            if not as_bool(row[f"can_run_{phase}"]):
                continue
            plans.append(PhasePlan(
                patient_id=str(row["patient_id"]), series_uid=str(row["series_uid"]),
                split=str(row["split"]), phase=phase,
                selected_series_id=str(row["selected_series_id"]),
                selected_internal_series=str(row[f"selected_{phase}_internal_series"]),
                expected_pairs=int(row[f"n_{phase}_contiguous_pairs"]),
                expected_frames=int(row[f"n_{phase}_frames"]),
                frame_list_hash=str(row[f"{phase}_frame_list_hash"]),
            ))
    return plans


def read_phase(root: Path, plan: PhasePlan) -> dict[str, Any]:
    directory = root / plan.patient_id / plan.series_uid / plan.phase
    required = [
        directory / "selected_frames.csv", directory / "pair_features.csv.gz",
        directory / "frame_kinetics.csv.gz", directory / "temporal_curves.csv.gz",
        directory / "phase_summary.json", directory / "metadata.json",
        directory / "pair_maps.npz", directory / "masks_and_kinetics.npz",
        directory / ".SUCCESS",
    ]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{directory}: missing {missing}")
    selected = pd.read_csv(directory / "selected_frames.csv", dtype={"patient_id": str, "series_uid": str})
    pair = pd.read_csv(directory / "pair_features.csv.gz", dtype={"patient_id": str, "series_uid": str})
    frame = pd.read_csv(directory / "frame_kinetics.csv.gz", dtype={"patient_id": str, "series_uid": str})
    curves = pd.read_csv(directory / "temporal_curves.csv.gz", dtype={"patient_id": str, "series_uid": str})
    summary = json.loads((directory / "phase_summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))

    if len(selected) != plan.expected_frames or len(pair) != plan.expected_pairs:
        raise AssertionError(f"{directory}: frame/pair count mismatch")
    if len(pair) and not (pd.to_numeric(pair["delta_frame"], errors="coerce") == 1).all():
        raise AssertionError(f"{directory}: cross-gap pair")
    for table_name, table in (("selected", selected), ("pair", pair), ("frame", frame), ("curves", curves)):
        if "patient_id" not in table or table["patient_id"].astype(str).nunique() != 1 or str(table["patient_id"].iloc[0]) != plan.patient_id:
            raise AssertionError(f"{directory}: {table_name} patient contamination")
        if "series_uid" not in table or table["series_uid"].astype(str).nunique() != 1 or str(table["series_uid"].iloc[0]) != plan.series_uid:
            raise AssertionError(f"{directory}: {table_name} series contamination")
        numeric_table = table.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
        if np.isinf(numeric_table).any():
            raise AssertionError(f"{directory}: infinity in {table_name}")
    if metadata["frame_list_hash"] != plan.frame_list_hash:
        raise AssertionError(f"{directory}: frame-list hash changed")
    if not metadata.get("cuda_actually_used", False) or metadata.get("cpu_fallback", True):
        raise AssertionError(f"{directory}: CUDA assertion failed")
    if summary.get("labels_read", False) or summary.get("model_trained", False) or summary.get("manifest_rescanned", False):
        raise AssertionError(f"{directory}: forbidden upstream behavior")
    return {"directory": directory, "selected": selected, "pair": pair, "frame": frame, "curves": curves, "summary": summary, "metadata": metadata}


def aggregate_phase(plan: PhasePlan, artifacts: dict[str, Any]) -> dict[str, Any]:
    pair = artifacts["pair"].sort_values(["frame_index_t", "frame_index_t1", "pair_order"]).reset_index(drop=True)
    selected = artifacts["selected"].sort_values(["frame_index", "sequence_position"]).reset_index(drop=True)
    frame = artifacts["frame"]
    curves = artifacts["curves"]
    summary = artifacts["summary"]
    indices = numeric(selected["frame_index"])
    first, last = float(indices.min()), float(indices.max())
    mid = 0.5 * (numeric(pair["frame_index_t"]) + numeric(pair["frame_index_t1"]))
    relative = (mid - first) / max(last - first, 1.0)
    blocks = blocks_from_pairs(pair)
    row: dict[str, Any] = {
        "patient_id": plan.patient_id, "series_uid": plan.series_uid,
        "split": plan.split, "phase": plan.phase,
        "selected_series_id": plan.selected_series_id,
        "selected_internal_series": plan.selected_internal_series,
    }
    for metric in PAIR_CURVES:
        if metric not in pair.columns:
            # Compat profile maps vessel mask to active, but older partial data may lack aliases.
            fallback = metric.replace("vessel_", "active_")
            if fallback not in pair.columns:
                raise ValueError(f"{artifacts['directory']}: missing pair curve {metric}")
            values = numeric(pair[fallback])
        else:
            values = numeric(pair[metric])
        for stat, value in curve_summary(values, mid, relative, blocks).items():
            row[f"pair_{metric}_{stat}"] = value
    for metric in STAGE_METRICS:
        source = metric if metric in pair.columns else metric.replace("vessel_", "active_")
        values = numeric(pair[source])
        for stage in STAGES:
            selected_values = values[pair["stage"].astype(str).to_numpy() == stage]
            row[f"stage_{stage}_{metric}_median"] = float(np.median(selected_values[np.isfinite(selected_values)])) if np.isfinite(selected_values).any() else float("nan")

    tdc, tdc_qc = robust_tdc(frame)
    row.update(tdc)
    row.update(filling_features(curves))
    coupling, coupling_qc = coupling_features(pair, frame)
    row.update(coupling)

    gap_count = int(np.sum(np.diff(indices) != 1))
    polarity = summary.get("polarity", {})
    activity = summary.get("activity_qc", {})
    fov = summary.get("fov_qc", {})
    row.update({
        "qc_n_frames": int(len(selected)), "qc_n_pairs": int(len(pair)),
        "qc_frame_span": int(last - first),
        "qc_temporal_block_count": gap_count + 1, "qc_gap_count": gap_count,
        "qc_fov_ratio": fov.get("fov_ratio"),
        "qc_active_ratio_fov": activity.get("active_ratio_fov"),
        "qc_vessel_ratio_fov": activity.get("vessel_ratio_fov"),
        "qc_background_ratio_fov": activity.get("background_ratio_fov"),
        "qc_polarity_margin": polarity.get("polarity_margin"),
        "qc_polarity_ambiguous": int(bool(polarity.get("polarity_ambiguous", False))),
        "qc_baseline_start_position": polarity.get("baseline_start_position"),
        "qc_baseline_frame_count": polarity.get("baseline_frame_count"),
        "qc_vessel_fallback_to_active": int(bool(activity.get("vessel_fallback_to_active", False))),
        "qc_background_fallback": int(bool(activity.get("background_fallback", False))),
        "qc_hard_valid_ratio_median": finite_median(pair["hard_valid_ratio_fov"]),
        "qc_fb_relative_median": finite_median(pair["fb_relative_mean"]),
        "qc_fb_relative_p90": finite_percentile(pair["fb_relative_mean"], 90),
        "qc_uncertainty_log_median": finite_median(pair["uncertainty_log_mean"]),
        "qc_uncertainty_log_p90": finite_percentile(pair["uncertainty_log_mean"], 90),
        "qc_global_motion_mag_norm_median": finite_median(pair["global_motion_mag_norm"]),
        "qc_soft_weight_median": finite_median(pair["soft_weight_mean_fov"]),
        "qc_pair_runtime_seconds_sum": float(np.nansum(numeric(pair["runtime_seconds"]))),
        **tdc_qc, **coupling_qc,
    })
    return row


def build_series_features(phase: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    all_features = CORE_PHASE_FEATURES + QC_PHASE_FEATURES
    for record in manifest.to_dict("records"):
        patient_id, series_uid = str(record["patient_id"]), str(record["series_uid"])
        subset = phase[(phase["patient_id"] == patient_id) & (phase["series_uid"] == series_uid)]
        row: dict[str, Any] = {
            "patient_id": patient_id, "series_uid": series_uid,
            "split": str(record["split"]), "source_type": str(record["source_type"]),
            "series_id": str(record["series_id"]),
            "missing_pre": int(not as_bool(record["can_run_pre"])),
            "missing_post": int(not as_bool(record["can_run_post"])),
        }
        phase_values: dict[str, dict[str, float]] = {}
        for phase_name in ("pre", "post"):
            phase_row = subset[subset["phase"] == phase_name]
            phase_values[phase_name] = {}
            for feature in all_features:
                value = float("nan") if phase_row.empty else pd.to_numeric(pd.Series([phase_row.iloc[0][feature]]), errors="coerce").iloc[0]
                value = float(value) if pd.notna(value) else float("nan")
                row[f"{phase_name}_{feature}"] = value
                phase_values[phase_name][feature] = value
        pre_dims = str(record.get("pre_dimensions", ""))
        post_dims = str(record.get("post_dimensions", ""))
        row["delta_shape_compatible"] = int(bool(pre_dims and post_dims and pre_dims == post_dims))
        for feature in CORE_PHASE_FEATURES:
            pre, post = phase_values["pre"][feature], phase_values["post"][feature]
            row[f"delta_{feature}"] = post - pre if math.isfinite(pre) and math.isfinite(post) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def build_patient_median(series: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [f"{prefix}_{feature}" for prefix in ("pre", "post") for feature in CORE_PHASE_FEATURES]
    rows: list[dict[str, Any]] = []
    for patient_id, group in series.groupby("patient_id", sort=True):
        row: dict[str, Any] = {
            "patient_id": str(patient_id), "split": str(group["split"].iloc[0]),
            "series_count": int(len(group)),
            "pre_series_count": int((group["missing_pre"] == 0).sum()),
            "post_series_count": int((group["missing_post"] == 0).sum()),
            "missing_pre": int((group["missing_pre"] == 0).sum() == 0),
            "missing_post": int((group["missing_post"] == 0).sum() == 0),
        }
        for column in feature_columns:
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            row[column] = float(np.median(values)) if values.size else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def feature_schema() -> dict[str, Any]:
    definitions = []
    for name in CORE_PHASE_FEATURES:
        if name.startswith("pair_") or name.startswith("stage_"):
            group, unit = "sea_raft_dynamic", "normalized_apparent_displacement_or_unitless"
        elif name.startswith("tdc_"):
            group = "tdc"
            unit = "frames" if name.endswith("_frames") else "relative_or_normalized_intensity"
        elif name.startswith("filling_"):
            group, unit = "filling", "frames_or_unitless"
        else:
            group, unit = "coupling", "frames_or_unitless"
        definitions.append({
            "feature_name": name, "group": group, "unit": unit,
            "default_model_candidate": True,
            "delta_default": False,
            "missing_policy": "NaN when mathematically undefined; never replaced by zero here",
        })
    return {
        "version": "api_fullseq_v3_core_schema_v1",
        "created_utc": utc_now(),
        "phase_core_feature_count": len(CORE_PHASE_FEATURES),
        "default_series_prepost_feature_count": 2 * len(CORE_PHASE_FEATURES),
        "phase_core_features": definitions,
        "phase_qc_features": QC_PHASE_FEATURES,
        "default_patient_aggregation": "median across selected valid series, label-blind",
        "scientific_scope": "SEA-RAFT apparent displacement per frame plus DSA intensity/filling dynamics; not physical blood velocity",
    }


def audit_numeric(frame: pd.DataFrame, core_columns: list[str]) -> dict[str, Any]:
    values = frame[core_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    inf_count = int(np.isinf(values).sum())
    extreme = int((np.abs(values[np.isfinite(values)]) > 100000).sum()) if np.isfinite(values).any() else 0
    correlation_ok = True
    relative_ok = True
    for column in core_columns:
        numeric_values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        finite_values = numeric_values[np.isfinite(numeric_values)]
        bare = column.removeprefix("pre_").removeprefix("post_").removeprefix("delta_")
        if bare in CORRELATION_FEATURES and finite_values.size and ((finite_values < -1.000001) | (finite_values > 1.000001)).any():
            correlation_ok = False
        if bare in RELATIVE_FEATURES and finite_values.size and ((finite_values < -1e-8) | (finite_values > 1.000001)).any():
            relative_ok = False
    return {
        "inf_count": inf_count, "values_abs_gt_100000": extreme,
        "correlations_inside_minus1_plus1": correlation_ok,
        "relative_positions_inside_0_1": relative_ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pairdata-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", required=True, choices=["pilot_train", "full_train", "full_valid"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    pairdata_root = Path(args.pairdata_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    success = output / ".FEATURES_SUCCESS"
    targets = [
        output / "phase_features.csv", output / "series_features.csv",
        output / "patient_median_features.csv", output / "feature_schema.json",
        output / "audit.json", success,
    ]
    if any(path.exists() for path in targets) and not args.overwrite:
        raise FileExistsError("Refusing to overwrite existing v3 feature outputs")
    if not (pairdata_root / ".SUCCESS").is_file():
        raise FileNotFoundError(f"Pairdata root incomplete: {pairdata_root}")

    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str, "series_uid": str})
    plans = plans_from_manifest(manifest)
    phase_rows = [aggregate_phase(plan, read_phase(pairdata_root, plan)) for plan in plans]
    phase = pd.DataFrame(phase_rows)
    series = build_series_features(phase, manifest)
    patient = build_patient_median(series)
    schema = feature_schema()

    expected = {
        "full_train": {"series": 1147, "patients": 1055, "phases": 2087, "pairs": 43364},
        "full_valid": {"series": 287, "patients": 264, "phases": 535, "pairs": 11040},
    }
    actual = {
        "series": int(len(series)), "patients": int(len(patient)),
        "phases": int(len(phase)), "pairs": int(pd.to_numeric(phase["qc_n_pairs"], errors="coerce").sum()),
    }
    if args.mode in expected and actual != expected[args.mode]:
        raise AssertionError(f"Formal feature size mismatch expected={expected[args.mode]}, actual={actual}")

    phase_core = CORE_PHASE_FEATURES
    series_core = [f"{prefix}_{feature}" for prefix in ("pre", "post") for feature in CORE_PHASE_FEATURES]
    phase_audit = audit_numeric(phase, phase_core)
    series_audit = audit_numeric(series, series_core)
    hard_ok = all([
        phase_audit["inf_count"] == 0, phase_audit["values_abs_gt_100000"] == 0,
        phase_audit["correlations_inside_minus1_plus1"], phase_audit["relative_positions_inside_0_1"],
        series_audit["inf_count"] == 0, series_audit["values_abs_gt_100000"] == 0,
        series_audit["correlations_inside_minus1_plus1"], series_audit["relative_positions_inside_0_1"],
    ])
    if not hard_ok:
        raise AssertionError(f"Feature hard audit failed: phase={phase_audit}, series={series_audit}")

    write_csv_atomic(phase, output / "phase_features.csv")
    write_csv_atomic(series, output / "series_features.csv")
    write_csv_atomic(patient, output / "patient_median_features.csv")
    write_json_atomic(output / "feature_schema.json", schema)
    audit = {
        "version": "api_fullseq_v3_feature_audit_v1", "created_utc": utc_now(),
        "mode": args.mode, "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path), "pairdata_root": str(pairdata_root),
        "actual": actual, "phase_core_features": len(CORE_PHASE_FEATURES),
        "series_default_prepost_features": 2 * len(CORE_PHASE_FEATURES),
        "phase_numeric": phase_audit, "series_numeric": series_audit,
        "labels_read": False, "model_trained": False,
    }
    write_json_atomic(output / "audit.json", audit)
    write_json_atomic(success, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fail-closed HEMO extraction from motion-corrected native DSA signal."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .temporal_contract import FrozenPhaseContract
from .temporal_motion import CorrectedPhaseSequence
from .v5_adapter import load_v5_module


REGIONS = ("lesion", "peri")
COMPACT_METRICS = (
    "ttp_from_arrival", "mtt", "curve_width", "auc_peaknorm", "washin_peaknorm", "washout_peaknorm",
)
RAW_METRICS = (
    "peak", "arrival", "ttp", "ttp_from_arrival", "auc", "washin", "washout", "mtt", "curve_width",
    "auc_peaknorm", "washin_peaknorm", "washout_peaknorm",
)


def compact36_columns() -> list[str]:
    return [f"hemo_{region}_{metric}_{representation}" for region in REGIONS for metric in COMPACT_METRICS
            for representation in ("pre", "post", "post_minus_pre")]


def native_peri_radii(phase: FrozenPhaseContract) -> tuple[int, int]:
    scale = float(np.mean((phase.record.resize_scale_x, phase.record.resize_scale_y)))
    inner = max(1, int(round(8 * scale)))
    outer = max(inner + 1, int(round(24 * scale)))
    return inner, outer


def build_hemo_regions(lesion_mask: np.ndarray, stable_valid: np.ndarray, phase: FrozenPhaseContract) -> dict[str, np.ndarray]:
    lesion = np.asarray(lesion_mask, dtype=bool) & np.asarray(stable_valid, dtype=bool)
    inner, outer = native_peri_radii(phase)
    source = np.asarray(lesion_mask, dtype=np.uint8)
    outer_mask = cv2.dilate(source, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * outer + 1, 2 * outer + 1))) > 0
    inner_mask = cv2.dilate(source, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inner + 1, 2 * inner + 1))) > 0
    peri = outer_mask & ~inner_mask & np.asarray(stable_valid, dtype=bool)
    return {"lesion": lesion, "peri": peri}


def _phase_invalid(reason: str) -> dict[str, Any]:
    return {
        "valid": False, "reason": reason,
        "curves": {region: np.full(0, np.nan, dtype=np.float32) for region in REGIONS},
        "phase_features": {region: {metric: np.nan for metric in RAW_METRICS} for region in REGIONS},
        "normalized": {region: np.full(32, np.nan, dtype=np.float32) for region in REGIONS},
        "qc": {},
    }


def extract_phase_hemo(corrected: CorrectedPhaseSequence, phase: FrozenPhaseContract, cfg: dict[str, Any]) -> dict[str, Any]:
    hcfg = cfg["jacobian_hemo"]["hemo"]
    regions = build_hemo_regions(corrected.local.lesion_mask, corrected.stable_valid, phase)
    if any(int(mask.sum()) < int(hcfg["min_region_pixels"]) for mask in regions.values()):
        missing = [name for name, mask in regions.items() if int(mask.sum()) < int(hcfg["min_region_pixels"])]
        return _phase_invalid("insufficient_stable_region_pixels:" + ",".join(missing))
    hemo = load_v5_module(cfg, "hemodynamics.py")
    curves: dict[str, np.ndarray] = {}
    features: dict[str, dict[str, float]] = {}
    normalized: dict[str, np.ndarray] = {}
    norm_meta: dict[str, Any] = {}
    polarities: dict[str, float] = {}
    for name, support in regions.items():
        curve, polarity = hemo.time_density_curve(corrected.corrected_signal, support, baseline_n_frames=hcfg["baseline_n_frames"])
        feature = hemo.curve_features(curve, frame_interval_seconds=None)
        _, sampled, nmeta = hemo.normalized_phase_features(curve, n_samples=hcfg["normalized_curve_samples"])
        curves[name], features[name], normalized[name], norm_meta[name], polarities[name] = curve, feature, sampled, nmeta, float(polarity)
    compact_finite = all(np.isfinite(features[region][metric]) for region in REGIONS for metric in COMPACT_METRICS)
    return {
        "valid": bool(compact_finite),
        "reason": "" if compact_finite else "nonfinite_compact_descriptor",
        "curves": curves, "phase_features": features, "normalized": normalized,
        "qc": {
            "stable_valid_fraction": float(np.mean(corrected.stable_valid)),
            "lesion_pixels": int(regions["lesion"].sum()), "peri_pixels": int(regions["peri"].sum()),
            "peri_inner_native_px": native_peri_radii(phase)[0], "peri_outer_native_px": native_peri_radii(phase)[1],
            "polarities": polarities, "normalized_phase": norm_meta,
            "n_frames": int(len(corrected.corrected_signal)),
            "source_frame_indices": corrected.source_frame_indices.astype(int).tolist(),
        },
    }


def build_prepost_compact36(pre_result: dict[str, Any], post_result: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for region in REGIONS:
        for metric in COMPACT_METRICS:
            pre = float(pre_result["phase_features"][region][metric])
            post = float(post_result["phase_features"][region][metric])
            out[f"hemo_{region}_{metric}_pre"] = pre
            out[f"hemo_{region}_{metric}_post"] = post
            out[f"hemo_{region}_{metric}_post_minus_pre"] = post - pre if np.isfinite(pre) and np.isfinite(post) else np.nan
    if list(out) != compact36_columns():
        raise AssertionError("HEMO compact36 schema order changed")
    return out


def build_raw_hemo_row(pre_result: dict[str, Any], post_result: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for phase_name, result in (("pre", pre_result), ("post", post_result)):
        for region in REGIONS:
            for metric in RAW_METRICS:
                out[f"hemo_{region}_{metric}_{phase_name}"] = float(result["phase_features"][region][metric])
    return out


def write_hemo_artifacts(series_uid: str, pre_result: dict[str, Any], post_result: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for phase_name, result in (("pre", pre_result), ("post", post_result)):
        for region in REGIONS:
            arrays[f"{phase_name}_{region}_tdc"] = np.asarray(result["curves"][region], dtype=np.float32)
            arrays[f"{phase_name}_{region}_normalized_phase"] = np.asarray(result["normalized"][region], dtype=np.float32)
    np.savez_compressed(root / "hemodynamic_curves.npz", **arrays)
    compact = build_prepost_compact36(pre_result, post_result)
    raw = build_raw_hemo_row(pre_result, post_result)
    valid = bool(pre_result["valid"] and post_result["valid"] and all(np.isfinite(v) for v in compact.values()))
    reasons = [f"pre:{pre_result['reason']}" for _ in [0] if pre_result["reason"]] + [f"post:{post_result['reason']}" for _ in [0] if post_result["reason"]]
    payload = {
        "series_uid": series_uid, "pre_hemo_valid": bool(pre_result["valid"]), "post_hemo_valid": bool(post_result["valid"]),
        "hemo_valid": valid, "hemo_invalid_reasons": ";".join(reasons), "compact36": compact, "raw": raw,
        "pre_qc": pre_result["qc"], "post_qc": post_result["qc"],
    }
    (root / "hemodynamic_features.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    (root / "hemodynamic_qc.json").write_text(json.dumps({
        key: payload[key] for key in ("series_uid", "pre_hemo_valid", "post_hemo_valid", "hemo_valid", "hemo_invalid_reasons", "pre_qc", "post_qc")
    }, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    return payload

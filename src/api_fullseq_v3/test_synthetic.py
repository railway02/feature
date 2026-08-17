#!/usr/bin/env python3
"""Synthetic semantic tests for the v3 re-extraction pipeline."""
from __future__ import annotations

import importlib.util
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


extractor = load("v3_extract", "extract_pairdata.py")
builder = load("v3_build", "build_features.py")


def config() -> dict:
    return {
        "normalization": {"baseline_frame_count": 3},
        "activity": {
            "activity_temporal_percentile": 90,
            "active_percentile": 70,
            "active_mad_multiplier": 1.0,
            "high_activity_percentile": 90,
            "high_activity_mad_multiplier": 2.0,
            "background_percentile": 35,
            "background_mad_multiplier": 0.5,
            "minimum_active_pixels": 8,
            "minimum_background_pixels": 8,
        },
        "kinetics": {
            "washout_front_fraction_of_peak": 0.10,
        },
        "flow_qc": {
            "minimum_region_pixels": 4,
            "fb_relative_hard_max": 1.0,
            "uncertainty_log_hard_max": 10.0,
            "fb_soft_tau": 1.0,
            "uncertainty_soft_tau": 10.0,
            "direction_bins": 8,
            "high_flow_percentile": 80,
            "high_change_percentile": 80,
        },
        "v3": {
            "science_profile": "improved",
            "adaptive_baseline": {"window_frames": 3, "max_start_position": 3},
            "polarity_ambiguity_margin": 0.08,
            "vessel_mask": {"vesselness_percentile": 65.0, "minimum_component_pixels": 2},
        },
    }


def test_adaptive_baseline() -> None:
    frames = np.zeros((8, 24, 24), dtype=np.float32)
    frames[:3] = 0.20
    frames[3:] = 0.20
    frames[4:, 8:16, 8:16] += np.linspace(0.1, 0.5, 4)[:, None, None]
    fov = np.ones((24, 24), dtype=bool)
    _, enhancement, qc = extractor.baseline_polarity_enhancement(frames, fov, config())
    assert qc["baseline_start_position"] == 0
    assert qc["baseline_frame_count"] == 3
    assert qc["polarity_label"] == "brightening"
    assert enhancement.max() > 0.4


def test_vessel_mask_nonempty() -> None:
    enhancement = np.zeros((8, 48, 48), dtype=np.float32)
    for i in range(8):
        enhancement[i, 20:23, 8:40] = i / 7
        enhancement[i, 8:40, 28:31] = i / 7
    fov = np.ones((48, 48), dtype=bool)
    masks, qc, _ = extractor.build_activity_masks(enhancement, fov, config())
    assert masks["active"].sum() >= 8
    assert masks["vessel"].sum() >= 4
    assert 0 < qc["vessel_ratio_fov"] <= qc["active_ratio_fov"]


def test_flow_pair_features() -> None:
    h = w = 32
    forward = np.zeros((h, w, 2), dtype=np.float32)
    forward[..., 0] = 1.0
    backward = np.zeros_like(forward)
    backward[..., 0] = -1.0
    uncertainty = np.zeros((h, w), dtype=np.float32)
    fov = np.ones((h, w), dtype=bool)
    active = np.zeros((h, w), dtype=bool); active[8:24, 8:24] = True
    vessel = np.zeros((h, w), dtype=bool); vessel[14:18, 8:24] = True
    background = fov & ~active
    masks = {
        "fov": fov, "active": active, "vessel": vessel,
        "high_activity": vessel, "background": background,
    }
    e1 = np.zeros((h, w), dtype=np.float32)
    e2 = np.zeros((h, w), dtype=np.float32); e2[8:24, 8:24] = 1.0
    peak = np.ones((h, w), dtype=np.float32)
    visible1 = np.zeros((h, w), dtype=bool)
    visible2 = active.copy()
    row, cache = extractor.analyze_flow_pair(
        forward, backward, uncertainty, e1, e2, peak, visible1, visible2, masks, config()
    )
    assert math.isfinite(row["vessel_res_mag_norm_median"])
    assert row["hard_valid_ratio_fov"] > 0.8
    assert cache["hard_valid"].shape == (h, w)


def tdc_frame(indices: list[int], values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "frame_index": indices,
        "sequence_position": range(len(indices)),
        "tdc_active_median": values,
    })


def test_gap_does_not_form_rise() -> None:
    features, qc = builder.robust_tdc(tdc_frame([0, 1, 2, 10, 11], [0.0, 0.1, 0.2, 0.8, 1.0]))
    assert qc["qc_tdc_rise_valid"] == 0
    assert math.isnan(features["tdc_robust_rise_slope_per_frame"])


def test_plateau_is_not_censored() -> None:
    _, qc = builder.robust_tdc(tdc_frame(list(range(8)), [0, .2, .6, 1, .95, .94, .93, .92]))
    assert qc["qc_washout_observation_adequate"] == 1
    assert qc["qc_tdc_washout_observed"] == 0
    assert qc["qc_washout_right_censored"] == 0


def test_last_peak_is_censored() -> None:
    _, qc = builder.robust_tdc(tdc_frame(list(range(6)), [0, .1, .2, .4, .7, 1.0]))
    assert qc["qc_washout_observation_adequate"] == 0
    assert qc["qc_washout_right_censored"] == 1


def test_coupling_uses_true_tdc_peak() -> None:
    pair = pd.DataFrame({
        "pair_order": range(10),
        "frame_index_t": range(10),
        "frame_index_t1": range(1, 11),
        "vessel_weighted_mag_norm_mean": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        "tdc_derivative_pair": [0, .2, .4, .2, 0, 0, 0, 0, 0, 0],
    })
    frame = tdc_frame(list(range(11)), [0, .1, .3, .6, .8, 1.0, .9, .8, .7, .6, .5])
    features, _ = builder.coupling_features(pair, frame)
    # Flow peak is pair 7 midpoint 7.5; true TDC peak is frame 5.
    assert abs(features["coupling_flow_peak_minus_tdc_peak_frames"] - 2.5) < 1e-9


def test_schema_size() -> None:
    assert len(builder.CORE_PHASE_FEATURES) == 106
    assert builder.feature_schema()["default_series_prepost_feature_count"] == 212



def test_phase_artifact_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        directory = root / "P1" / "S1" / "pre"
        directory.mkdir(parents=True)
        indices = [0, 1, 2, 3, 4]
        selected = pd.DataFrame({
            "patient_id": ["P1"] * 5, "series_uid": ["S1"] * 5,
            "phase": ["pre"] * 5, "sequence_position": range(5),
            "frame_index": indices, "absolute_path": [f"/x/{i}.jpg" for i in indices],
        })
        pair_data = {
            "patient_id": ["P1"] * 4, "series_uid": ["S1"] * 4, "phase": ["pre"] * 4,
            "pair_order": range(4), "frame_index_t": [0, 1, 2, 3],
            "frame_index_t1": [1, 2, 3, 4], "delta_frame": [1] * 4,
            "stage": ["precontrast", "washin", "peak", "washout"],
            "tdc_derivative_pair": [0.1, 0.3, 0.4, -0.2],
            "hard_valid_ratio_fov": [0.9] * 4, "fb_relative_mean": [0.1] * 4,
            "uncertainty_log_mean": [0.0] * 4, "global_motion_mag_norm": [0.01] * 4,
            "soft_weight_mean_fov": [0.8] * 4, "runtime_seconds": [0.01] * 4,
        }
        for index, metric in enumerate(builder.PAIR_CURVES):
            pair_data[metric] = np.linspace(0.1 + index * 0.001, 0.4 + index * 0.001, 4)
        pair = pd.DataFrame(pair_data)
        frame = pd.DataFrame({
            "patient_id": ["P1"] * 5, "series_uid": ["S1"] * 5, "phase": ["pre"] * 5,
            "sequence_position": range(5), "frame_index": indices,
            "tdc_active_median": [0.0, 0.2, 0.7, 1.0, 0.8],
        })
        curves = pd.DataFrame({
            "patient_id": ["P1"] * 5, "series_uid": ["S1"] * 5, "phase": ["pre"] * 5,
            "sequence_position": range(5), "frame_index": indices,
            "visible_area_fraction": [0.0, 0.1, 0.4, 0.8, 0.7],
            "new_area_fraction": [0.0, 0.1, 0.3, 0.4, 0.0],
            "largest_component_ratio": [0.0, 0.5, 0.7, 0.9, 0.9],
            "spatial_spread": [0.0, 0.1, 0.2, 0.3, 0.3],
        })
        selected.to_csv(directory / "selected_frames.csv", index=False)
        pair.to_csv(directory / "pair_features.csv.gz", index=False, compression="gzip")
        frame.to_csv(directory / "frame_kinetics.csv.gz", index=False, compression="gzip")
        curves.to_csv(directory / "temporal_curves.csv.gz", index=False, compression="gzip")
        summary = {
            "labels_read": False, "model_trained": False, "manifest_rescanned": False,
            "polarity": {"polarity_margin": 0.5, "polarity_ambiguous": False, "baseline_start_position": 0, "baseline_frame_count": 3},
            "activity_qc": {"active_ratio_fov": 0.1, "vessel_ratio_fov": 0.05, "background_ratio_fov": 0.5, "vessel_fallback_to_active": False, "background_fallback": False},
            "fov_qc": {"fov_ratio": 0.8},
        }
        (directory / "phase_summary.json").write_text(json.dumps(summary))
        frame_hash = hashlib.sha256("\n".join(f"/x/{i}.jpg" for i in indices).encode()).hexdigest()
        metadata = {"frame_list_hash": frame_hash, "cuda_actually_used": True, "cpu_fallback": False}
        (directory / "metadata.json").write_text(json.dumps(metadata))
        np.savez_compressed(directory / "pair_maps.npz", pair_order=np.arange(4))
        np.savez_compressed(directory / "masks_and_kinetics.npz", fov=np.ones((8, 8), np.uint8))
        (directory / ".SUCCESS").write_text("{}")
        plan = builder.PhasePlan(
            patient_id="P1", series_uid="S1", split="Train", phase="pre",
            selected_series_id="main", selected_internal_series="1",
            expected_pairs=4, expected_frames=5, frame_list_hash=frame_hash,
        )
        artifacts = builder.read_phase(root, plan)
        aggregated = builder.aggregate_phase(plan, artifacts)
        assert all(name in aggregated for name in builder.CORE_PHASE_FEATURES)
        assert aggregated["qc_n_pairs"] == 4

def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"[PASS] tests={len(tests)} core_phase_features={len(builder.CORE_PHASE_FEATURES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

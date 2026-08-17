#!/usr/bin/env python3
"""Synthetic semantic tests for the CAVE feature-bank production pipeline."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from common import as_bool, hash_lines
from io_ops import (
    frames_to_model, make_square_transform, map_model_to_original,
    map_original_to_model, strict_contiguous_blocks, temporal_views,
)
from manifest import expected_pair_count
from pooling import build_embedding_bank, embedding_feature_names, resample_trajectory, weighted_mean
from scalar_features import build_scalar_bank, curve_summary, expected_scalar_count


class FakeBridge:
    def kinetic_and_filling(self, enhancement, indices, fov, active_mask, tdc_peak):
        height, width = fov.shape
        maps = {"peak": np.where(active_mask, 1.0, np.nan).astype(np.float32)}
        kinetic = {f"kinetic_test_{index:02d}": float(index) for index in range(57)}
        filling = {f"filling_test_{index:02d}": float(index) for index in range(14)}
        curves = pd.DataFrame({
            "visible_area_fraction": np.linspace(0, 1, len(indices)),
            "new_area_fraction": np.linspace(0, 0.2, len(indices)),
            "washout_area_fraction": np.linspace(0, 0.1, len(indices)),
        })
        return maps, kinetic, active_mask, curves, filling, np.zeros_like(enhancement, dtype=bool)


def test_bool_parser() -> None:
    assert as_bool(True) and as_bool("True") and as_bool("1") and as_bool("yes")
    assert not as_bool(False) and not as_bool("False") and not as_bool("0") and not as_bool("")


def test_strict_blocks() -> None:
    blocks = strict_contiguous_blocks([1, 2, 3, 8, 9, 12])
    assert [block.tolist() for block in blocks] == [[0, 1, 2], [3, 4], [5]]
    assert expected_pair_count([1, 2, 3, 8, 9, 12]) == 3


def test_temporal_views() -> None:
    frames = np.zeros((30, 16, 16), dtype=np.float32)
    frames[5:25] = np.arange(20, dtype=np.float32)[:, None, None]
    views = temporal_views(frames, 20)
    assert tuple(views) == ("uniform_full20", "contrast_core20")
    assert len(views["uniform_full20"]) == 20
    assert views["uniform_full20"][0] == 0 and views["uniform_full20"][-1] == 29
    assert len(views["contrast_core20"]) == 20


def test_square_roundtrip() -> None:
    frames = np.zeros((3, 80, 120), dtype=np.uint8)
    frames[:, 20:60, 30:90] = 200
    transform = make_square_transform(frames, 64)
    model = frames_to_model(frames, transform)
    assert model.shape == (3, 64, 64)
    original = np.zeros((80, 120), dtype=np.float32)
    original[20:60, 30:90] = 1.0
    mapped = map_original_to_model(original, transform)
    restored = map_model_to_original(mapped, transform)
    assert restored.shape == original.shape
    assert np.mean(np.abs(restored - original)) < 0.08


def test_pooling() -> None:
    features = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(1, 2, 4, 4)
    weight = torch.zeros(1, 1, 4, 4)
    weight[:, :, :2] = 1
    pooled = weighted_mean(features, weight)
    assert pooled.shape == (1, 2)
    assert torch.allclose(pooled[0, 0], features[0, 0, :2].mean())

    f4 = torch.randn(1, 512, 64, 64)
    f5 = torch.randn(1, 512, 32, 32)
    artery = torch.rand(1, 1, 512, 512)
    vein = torch.rand(1, 1, 512, 512)
    activity = torch.rand(1, 1, 512, 512)
    fov = torch.ones(1, 1, 512, 512)
    primary, auxiliary, qc = build_embedding_bank(f4, f5, artery, vein, activity, fov)
    assert primary.shape == (1, 5120)
    assert torch.isfinite(primary).all()
    assert len(auxiliary) == 5
    assert qc["vessel_weight_sum"] > 0
    assert len(embedding_feature_names()) == 5120


def test_curve_gap_awareness() -> None:
    summary = curve_summary(np.asarray([0, 0.1, 0.2, 0.8, 1.0]), [0, 1, 2, 10, 11])
    # The slope between frame 2 and 10 must not be treated as an adjacent-frame slope.
    assert summary["max_up_slope_per_frame"] <= 0.2 + 1e-6
    assert summary["local_peak_count"] == 0


def test_scalar_schema() -> None:
    frames = np.zeros((8, 48, 48), dtype=np.float32)
    for index in range(8):
        frames[index, 20:24, 8:40] = index / 7
    fov = np.ones((48, 48), dtype=bool)
    artery = np.zeros((48, 48), dtype=np.float32); artery[20:24, 8:28] = 0.9
    vein = np.zeros((48, 48), dtype=np.float32); vein[20:24, 24:40] = 0.8
    vessel = np.maximum(artery, vein)
    union = 1 - (1 - artery) * (1 - vein)
    activity = frames.max(axis=0)
    scalar, curves, qc = build_scalar_bank(
        frames, fov, activity, artery, vein, vessel, union, FakeBridge(), list(range(8))
    )
    assert len(scalar) == expected_scalar_count() == 206
    assert len(set(scalar)) == 206
    assert set(curves) >= {"artery", "vein", "vessel", "active_vessel"}
    assert "active_vessel_hard_mask_fallback" in qc


def test_trajectory_resampling() -> None:
    trajectory = np.arange(5 * 3, dtype=np.float32).reshape(5, 3)
    output = resample_trajectory(trajectory, 16)
    assert output.shape == (16, 3)
    assert np.allclose(output[0], trajectory[0]) and np.allclose(output[-1], trajectory[-1])


def main() -> int:
    tests = [
        test_bool_parser, test_strict_blocks, test_temporal_views, test_square_roundtrip,
        test_pooling, test_curve_gap_awareness, test_scalar_schema, test_trajectory_resampling,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"[PASS] synthetic tests={len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

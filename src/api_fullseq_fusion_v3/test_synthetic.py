#!/usr/bin/env python3
"""Small deterministic checks for the multimodal fusion preprocessor."""
from __future__ import annotations

import numpy as np

from train_fusion_prediction_models import (
    MultimodalPreprocessor,
    VARIANTS,
    grouped_splits,
    metric_row,
)


def main() -> int:
    rng = np.random.default_rng(42)
    groups = np.repeat(np.arange(30), 2).astype(str)
    y = np.tile(np.array([0, 1], dtype=int), 30)
    data = {
        "cave_deep": rng.normal(size=(60, 128)).astype(np.float32),
        "cave_scalar": rng.normal(size=(60, 24)).astype(np.float32),
        "searaft": rng.normal(size=(60, 30)).astype(np.float32),
        "missing": np.zeros((60, 2), dtype=np.float32),
        "target": y,
    }
    data["cave_scalar"][::7, 0] = np.nan
    data["searaft"][::5, 1] = np.nan
    splits = grouped_splits(y, groups, requested=5, seed=42)
    fit_index, holdout_index = splits[0]
    assert not (set(groups[fit_index]) & set(groups[holdout_index]))
    fit_data = {key: value[fit_index] for key, value in data.items()}
    holdout_data = {key: value[holdout_index] for key, value in data.items()}
    preprocessor = MultimodalPreprocessor().fit(fit_data, seed=42)
    transformed = preprocessor.transform_all(holdout_data)
    assert set(transformed) == set(VARIANTS)
    assert all(len(value) == len(holdout_index) for value in transformed.values())
    assert all(np.isfinite(value).all() for value in transformed.values())
    metrics = metric_row(
        "synthetic", "Dummy", "holdout", y[holdout_index],
        np.full(len(holdout_index), 0.5), 0.5,
    )
    assert metrics["rows"] == len(holdout_index)
    print({"status": "ok", "dimensions": preprocessor.audit()["final_dimensions"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
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


builder = load("formal_builder", "10_build_adverse_prepost_record_task.py")
trainer = load("formal_trainer", "11_train_adverse_prepost_formal.py")


def test_strict_phase_exclusions() -> None:
    frame = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, np.nan],
            "mapping_accepted_final": [True, True, True, True, True],
            "series_row": [0, 1, 2, 3, 4],
            "series_patient_id": ["1", "2", "3", "4", "5"],
            "patient_id": ["1", "2", "3", "4", "5"],
            "pre_finite": [True, True, False, False, True],
            "post_finite": [True, False, True, False, True],
            "both_available": [True, False, False, False, True],
        }
    )
    reasons = builder.assign_exclusion_reason(frame)
    assert reasons.tolist() == [
        "",
        "strict_prepost_exclusion_only_one_phase_available",
        "strict_prepost_exclusion_only_one_phase_available",
        "strict_prepost_exclusion_no_phase_available",
        "invalid_or_missing_adverse_label",
    ]


def test_fold_grouping() -> None:
    rows = []
    for patient in range(30):
        for record in range(2):
            rows.append(
                {
                    "record_uid": f"r{patient}_{record}",
                    "patient_id": str(patient),
                    "series_uid": f"s{patient}",
                    "target": int((patient + record) % 5 == 0),
                }
            )
    table = builder.assign_grouped_folds(pd.DataFrame(rows))
    assert sorted(table["fold"].unique().tolist()) == [1, 2, 3, 4, 5]
    assert table.groupby("patient_id")["fold"].nunique().max() == 1


def test_preprocessor() -> None:
    rng = np.random.default_rng(7)
    deep = rng.normal(size=(70, 256)).astype(np.float32)
    scalar = rng.normal(size=(70, 25)).astype(np.float32)
    scalar[rng.random(scalar.shape) < 0.15] = np.nan

    preprocessor = trainer.FusionPreprocessor(
        deep_pca_dim=24,
        use_scalar=True,
        scalar_pca_dim=10,
        seed=7,
    ).fit(deep[:50], scalar[:50])
    transformed = preprocessor.transform(deep[50:], scalar[50:])
    assert transformed.shape == (20, 34)
    assert np.isfinite(transformed).all()


def main() -> int:
    test_strict_phase_exclusions()
    test_fold_grouping()
    test_preprocessor()
    print("FORMAL_ADVERSE_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

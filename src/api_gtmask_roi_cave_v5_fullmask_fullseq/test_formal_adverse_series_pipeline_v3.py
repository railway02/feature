#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = load(
    "formal_builder_v3",
    "10_build_adverse_prepost_series_task_v3.py",
)
trainer = load(
    "formal_trainer_v3",
    "11_train_adverse_prepost_series_formal_v3.py",
)


def test_record_phase_exclusions() -> None:
    frame = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, np.nan],
            "mapping_accepted_final": [True] * 5,
            "series_row": [0, 1, 2, 3, 4],
            "series_patient_id": ["1", "2", "3", "4", "5"],
            "patient_id": ["1", "2", "3", "4", "5"],
            "pre_finite": [True, True, False, False, True],
            "post_finite": [True, False, True, False, True],
            "both_available": [True, False, False, False, True],
        }
    )
    reasons = builder.assign_record_exclusion_reason(frame).tolist()
    assert reasons == [
        "",
        "strict_prepost_exclusion_only_one_phase_available",
        "strict_prepost_exclusion_only_one_phase_available",
        "strict_prepost_exclusion_no_phase_available",
        "invalid_or_missing_adverse_label",
    ]


def test_series_level_collapse_and_conflict() -> None:
    # Same patient has two distinct series with different labels: both survive.
    # Same series with repeated equal labels: one sample survives.
    # Same series with conflicting labels: only that series is excluded.
    records = pd.DataFrame(
        {
            "record_uid": ["r1", "r2", "r3", "r4", "r5", "r6"],
            "patient_id": ["10", "10", "20", "20", "30", "30"],
            "split": ["Train"] * 6,
            "excel_row_number": [2, 3, 4, 5, 6, 7],
            "target": [0, 1, 0, 0, 0, 1],
            "records_for_patient_source": [2, 2, 2, 2, 2, 2],
        }
    )
    mapping = pd.DataFrame(
        {
            "record_uid": records["record_uid"],
            "patient_id": records["patient_id"],
            "split": records["split"],
            "mapped_series_uid": ["s1", "s2", "s3", "s3", "s4", "s4"],
            "mapping_accepted_final": [True] * 6,
        }
    )
    series_uid = np.asarray(["s1", "s2", "s3", "s4"], dtype=str)
    embeddings = np.arange(4 * 2 * 5120, dtype=np.float32).reshape(4, 2, 5120)
    scalar = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)
    store = {
        "split": "Train",
        "series_uid": series_uid,
        "patient_id": np.asarray(["10", "10", "20", "30"], dtype=str),
        "embeddings": embeddings,
        "pre_finite": np.asarray([True] * 4),
        "post_finite": np.asarray([True] * 4),
        "both_available": np.asarray([True] * 4),
    }

    with tempfile.TemporaryDirectory() as tmp:
        metadata, summary = builder.build_split(
            records,
            mapping,
            store,
            scalar,
            ["a", "b", "c"],
            Path(tmp),
        )
        assert set(metadata["series_uid"]) == {"s1", "s2", "s3"}
        assert dict(zip(metadata["series_uid"], metadata["target"])) == {
            "s1": 0,
            "s2": 1,
            "s3": 0,
        }
        assert summary["included_series"] == 3
        assert summary["duplicate_same_label_records_collapsed"] == 1
        assert summary["excluded_conflicting_series"] == 1
        assert summary["patients_with_mixed_series_labels_preserved"] == 1

        with np.load(Path(tmp) / "train_features.npz", allow_pickle=False) as raw:
            assert raw["deep"].shape == (3, 10240)
            assert raw["series_uid"].astype(str).tolist() == ["s1", "s2", "s3"]


def test_train_only_scalar_schema() -> None:
    train = pd.DataFrame(
        {
            "series_uid": ["a", "b", "c"],
            "patient_id": ["1", "2", "3"],
            "train_signal": [1.0, 2.0, 3.0],
            "all_missing": ["", "", ""],
        }
    )
    valid = pd.DataFrame(
        {
            "series_uid": ["d", "e"],
            "patient_id": ["4", "5"],
            "train_signal": [4.0, 5.0],
            "valid_extra": [10.0, 11.0],
        }
    )
    train_matrix, valid_matrix, names, audit = (
        builder.train_only_scalar_schema(
            {"scalar_frame": train, "scalar_path": Path("train.csv")},
            {"scalar_frame": valid, "scalar_path": Path("valid.csv")},
        )
    )
    assert names == ["train_signal"]
    assert train_matrix.shape == (3, 1)
    assert valid_matrix.shape == (2, 1)
    assert audit["valid_used_to_select_schema"] is False


def test_mixed_label_group_splits() -> None:
    # Patients may contain both labels because different series are valid samples.
    groups = np.asarray(
        [str(patient) for patient in range(30) for _ in range(2)],
        dtype=str,
    )
    y = np.asarray(
        [(patient + series) % 4 == 0 for patient in range(30) for series in range(2)],
        dtype=np.int64,
    )
    splits = trainer.grouped_splits(y, groups, requested=3, seed=17, retries=20)
    seen = set()
    for development, holdout in splits:
        assert len(np.unique(y[development])) == 2
        assert len(np.unique(y[holdout])) == 2
        development_groups = set(groups[development])
        holdout_groups = set(groups[holdout])
        assert not (development_groups & holdout_groups)
        assert not (seen & holdout_groups)
        seen |= holdout_groups


def test_preprocessor() -> None:
    rng = np.random.default_rng(7)
    deep = rng.normal(size=(70, 64)).astype(np.float32)
    scalar = rng.normal(size=(70, 20)).astype(np.float32)
    scalar[rng.random(scalar.shape) < 0.15] = np.nan
    processor = trainer.FusionPreprocessor(
        deep_pca_dim=16,
        use_scalar=True,
        scalar_pca_dim=8,
        seed=7,
    ).fit(deep[:50], scalar[:50])
    transformed = processor.transform(deep[50:], scalar[50:])
    assert transformed.shape == (20, 24)
    assert np.isfinite(transformed).all()


def test_nested_mlp_inner_cv_cpu_and_cache() -> None:
    rng = np.random.default_rng(11)
    groups = np.asarray([str(i // 2) for i in range(60)])
    y = np.asarray([(i // 2 + i % 2) % 4 == 0 for i in range(60)], dtype=np.int64)
    deep = rng.normal(size=(60, 32)).astype(np.float32)
    scalar = rng.normal(size=(60, 8)).astype(np.float32)
    config = trainer.MLPConfig(
        deep_pca_dim=8,
        hidden1=12,
        hidden2=4,
        dropout1=0.1,
        dropout2=0.1,
        learning_rate=1e-3,
        batch_size=16,
        max_epochs=4,
        patience=2,
    )
    cache: dict[tuple[int, int], dict] = {}
    pooled, epochs, rows = trainer.evaluate_mlp_config_inner_cv(
        deep,
        scalar,
        y,
        groups,
        config,
        use_scalar=False,
        device=torch.device("cpu"),
        seed=13,
        search_seeds=1,
        amp_enabled=False,
        transform_cache=cache,
    )
    assert pooled.shape == (60,)
    assert np.isfinite(pooled).all()
    assert len(epochs) == 3
    assert len(rows) == 3
    assert len(cache) == 6

    # A second architecture with the same PCA dimension must reuse transforms.
    config2 = trainer.MLPConfig(
        deep_pca_dim=8,
        hidden1=16,
        hidden2=4,
        dropout1=0.2,
        dropout2=0.1,
        learning_rate=1e-3,
        batch_size=16,
        max_epochs=3,
        patience=2,
    )
    trainer.evaluate_mlp_config_inner_cv(
        deep,
        scalar,
        y,
        groups,
        config2,
        use_scalar=False,
        device=torch.device("cpu"),
        seed=13,
        search_seeds=1,
        amp_enabled=False,
        transform_cache=cache,
    )
    assert len(cache) == 6


def main() -> int:
    test_record_phase_exclusions()
    test_series_level_collapse_and_conflict()
    test_train_only_scalar_schema()
    test_mixed_label_group_splits()
    test_preprocessor()
    test_nested_mlp_inner_cv_cpu_and_cache()
    print("FORMAL_ADVERSE_SERIES_V3_TESTS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
CORE_PATH = HERE.parent / "11_train_adverse_prepost_series_formal_v3.py"


def import_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = import_path(HERE / "01_build_matched_tasks.py", "matched_task_builder_tests")
core = import_path(CORE_PATH, "matched_core_tests")
adapter = import_path(HERE / "training_adapter.py", "matched_adapter_tests")
driver = import_path(HERE / "02_train_matched_experiment.py", "matched_driver_tests")


class MatchedAblationTests(unittest.TestCase):
    def test_morphology_schema_is_fixed_and_safe(self) -> None:
        names = builder.morphology_feature_names()
        self.assertEqual(len(names), 36)
        builder.assert_feature_names_safe(names)
        self.assertFalse(any("target" in name for name in names))
        self.assertFalse(any("label" in name for name in names))

    def test_morphology_pre_post_delta_order(self) -> None:
        pre = np.arange(24, dtype=np.float32).reshape(2, 12)
        post = pre + 3.0
        output = builder.build_morphology_matrix(pre, post)
        self.assertEqual(output.shape, (2, 36))
        np.testing.assert_array_equal(output[:, :12], pre)
        np.testing.assert_array_equal(output[:, 12:24], post)
        np.testing.assert_array_equal(output[:, 24:], np.full((2, 12), 3.0))

    def test_temporal_signature_ignores_non_temporal_metadata(self) -> None:
        left = {
            "blocks": [{"indices": [1, 2], "view_indices": {"uniform_full20": [1, 2]}}],
            "roi": {"used_bbox": "1|2|3|4"},
        }
        right = {
            "blocks": [{"indices": [1, 2], "view_indices": {"uniform_full20": [1, 2]}}],
            "other": True,
        }
        self.assertEqual(builder.temporal_signature(left), builder.temporal_signature(right))

    def test_wl_fits_independent_branches(self) -> None:
        rng = np.random.default_rng(7)
        whole = rng.normal(0.0, 1.0, size=(40, 20)).astype(np.float32)
        local = rng.normal(100.0, 3.0, size=(40, 20)).astype(np.float32)
        adapter.install(core, "WL")
        preprocessor = adapter.MatchedPreprocessor(
            deep_pca_dim=8,
            use_scalar=True,
            seed=11,
        ).fit(whole, local)
        self.assertIsNot(preprocessor.primary.scaler, preprocessor.secondary.scaler)
        self.assertLess(float(np.mean(preprocessor.primary.scaler.mean_)), 1.0)
        self.assertGreater(float(np.mean(preprocessor.secondary.scaler.mean_)), 90.0)
        output = preprocessor.transform(whole[:5], local[:5])
        self.assertEqual(output.shape, (5, 16))
        self.assertTrue(np.isfinite(output).all())

    def test_wl_slice_keeps_equal_requested_components(self) -> None:
        adapter.install(core, "WL")
        values = np.arange(2 * 20, dtype=np.float32).reshape(2, 20)
        sliced = adapter.matched_slice_components(values, 4, 10, True)
        expected = np.concatenate([values[:, :4], values[:, 10:14]], axis=1)
        np.testing.assert_array_equal(sliced, expected)

    def test_m0_uses_no_image_branch(self) -> None:
        rng = np.random.default_rng(9)
        mask = rng.uniform(size=(40, 36)).astype(np.float32)
        adapter.install(core, "M0")
        preprocessor = adapter.MatchedPreprocessor(
            deep_pca_dim=64,
            use_scalar=False,
            seed=13,
        ).fit(mask, np.empty((40, 0), dtype=np.float32))
        output = preprocessor.transform(mask[:3], np.empty((3, 0), dtype=np.float32))
        self.assertEqual(output.shape[0], 3)
        self.assertLessEqual(output.shape[1], 36)
        self.assertTrue(np.isfinite(output).all())


    def test_generic_bootstrap_is_patient_clustered(self) -> None:
        y = np.asarray([0, 0, 1, 1] * 10, dtype=np.int64)
        patient = np.asarray(
            [f"p{index // 2:02d}" for index in range(len(y))],
            dtype=str,
        )
        probabilities = {
            "logistic": np.linspace(0.1, 0.9, len(y), dtype=np.float64),
            "mlp": np.linspace(0.9, 0.1, len(y), dtype=np.float64),
        }
        ci, paired, effective = driver.patient_cluster_bootstrap_generic(
            core,
            y,
            patient,
            probabilities,
            {"logistic": 0.5, "mlp": 0.5},
            repeats=50,
            seed=17,
        )
        self.assertEqual(effective, 50)
        self.assertEqual(len(ci), 6)
        self.assertEqual(len(paired), 3)
        self.assertTrue((ci["bootstrap_unit"] == "patient_id").all())
        self.assertTrue(
            (paired["difference_definition"] == "comparison - reference").all()
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)


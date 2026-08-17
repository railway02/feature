from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from common import sha256_file


def import_builder():
    path = CODE / "03_build_png2d_roi_manifests.py"
    spec = importlib.util.spec_from_file_location("png2d_roi_builder_test", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PNG2DStage1Tests(unittest.TestCase):
    def setUp(self):
        self.module = import_builder()
        self.cfg = {
            "target_rule": "png2d_gt_labels_1_2_nonzero",
            "foreground_labels": [1, 2],
            "bbox_factor": 1.5,
            "min_side_pixels": 16,
            "min_margin_pixels": 2,
            "round_multiple": 4,
            "fallback_bbox_factor": 2.0,
            "fallback_min_side_pixels": 24,
            "extended_fallback_bbox_factor": 3.0,
            "extended_fallback_min_side_pixels": 32,
            "allow_mask_resize": True,
            "resize_interpolation": "nearest",
            "require_uniform_xy_scale": True,
            "max_resize_aspect_ratio_error": 0.001,
        }

    def test_exact_shape_does_not_resize(self):
        labels = np.zeros((64, 64), dtype=np.uint8)
        labels[10:20, 20:30] = 2
        effective, audit = self.module.effective_mask(labels, (64, 64), self.cfg)
        self.assertTrue(np.array_equal(labels, effective))
        self.assertEqual(audit["mask_resized_to_frame"], 0)
        self.assertEqual(audit["mask_resize_interpolation"], "none")

    def test_uniform_nearest_resize_preserves_label_set(self):
        labels = np.zeros((64, 64), dtype=np.uint8)
        labels[8:16, 10:20] = 1
        labels[30:48, 32:52] = 2
        effective, audit = self.module.effective_mask(labels, (32, 32), self.cfg)
        self.assertEqual(effective.shape, (32, 32))
        self.assertEqual(set(np.unique(effective)), {0, 1, 2})
        self.assertEqual(audit["mask_resized_to_frame"], 1)
        self.assertEqual(audit["mask_resize_interpolation"], "nearest")
        self.assertAlmostEqual(audit["resize_scale_x"], audit["resize_scale_y"])

    def test_nonuniform_resize_is_rejected(self):
        labels = np.zeros((64, 64), dtype=np.uint8)
        labels[8:16, 10:20] = 1
        with self.assertRaisesRegex(ValueError, "nonuniform_scale_not_allowed"):
            self.module.effective_mask(labels, (80, 64), self.cfg)

    def test_label_one_only_is_valid_nonzero_gt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = np.zeros((64, 64), dtype=np.uint8)
            frame[10:30, 20:40] = 120
            reference = frame.copy()
            labels = np.zeros((64, 64), dtype=np.uint8)
            labels[15:22, 25:33] = 1
            frame_path = root / "frame.jpg"
            reference_path = root / "mean.png"
            mask_path = root / "mask.png"
            self.assertTrue(cv2.imwrite(str(frame_path), frame))
            self.assertTrue(cv2.imwrite(str(reference_path), reference))
            self.assertTrue(cv2.imwrite(str(mask_path), labels))
            whole = root / "whole" / "train" / "p1" / "s1" / "pre"
            whole.mkdir(parents=True)
            metadata = {"blocks": [{"indices": [0], "view_indices": {
                "uniform_full20": [0], "contrast_core20": [0]
            }}]}
            (whole / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            mapped = {
                "phase_uid": "s1::pre",
                "split": "Train",
                "source_series_order": "0",
                "source_phase_order": "0",
                "patient_id": "p1",
                "series_uid": "s1",
                "series_id": "",
                "phase": "pre",
                "frame_paths": str(frame_path),
                "frame_list_hash": "hash1",
                "phase_mapping_status": "accepted",
                "mapping_method": "png2d_mean_identity_verified",
                "mapping_reason": "",
                "png_key": "p1_Pre",
                "reference_image_path": str(reference_path),
                "reference_sha256": sha256_file(reference_path),
                "mask_path": str(mask_path),
                "mask_sha256": sha256_file(mask_path),
                "identity_pearson_correlation": "1.0",
                "orientation_transform": "identity",
                "orientation_status": "identity_mean_verified",
            }
            coverage, roi, morphology, temporal = self.module.evaluate_phase(
                mapped, self.cfg, root / "whole"
            )
            self.assertEqual(coverage["local_eligible"], 1)
            self.assertEqual(coverage["labels_present_effective"], "[1]")
            self.assertEqual(coverage["labels_ignored"], "[]")
            self.assertIsNotNone(roi)
            self.assertEqual(roi["foreground_rule"], "labels_in_1_2_equivalent_nonzero")
            self.assertEqual(roi["selected_labels"], "[1,2]")
            self.assertGreater(roi["selected_foreground_pixels"], 0)
            self.assertIsNotNone(morphology)
            self.assertIsNotNone(temporal)

    def test_unresolved_mapping_is_excluded_without_io(self):
        mapped = {
            "phase_uid": "s1::pre",
            "split": "Train",
            "source_series_order": "0",
            "source_phase_order": "0",
            "patient_id": "p1",
            "series_uid": "s1",
            "phase": "pre",
            "frame_list_hash": "hash1",
            "phase_mapping_status": "needs_review",
            "mapping_method": "png2d_mean_below_correlation_threshold",
        }
        coverage, roi, morphology, temporal = self.module.evaluate_phase(
            mapped, self.cfg, Path("/nonexistent")
        )
        self.assertEqual(coverage["local_exclusion_reason"], "mapping_needs_review")
        self.assertIsNone(roi)
        self.assertIsNone(morphology)
        self.assertIsNone(temporal)


if __name__ == "__main__":
    unittest.main()


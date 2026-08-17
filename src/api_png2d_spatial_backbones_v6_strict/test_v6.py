from __future__ import annotations

import unittest

import cv2
import numpy as np
import torch

from augmentations import synchronized_affine
from data import percentile_normalize
from featurebank_utils import select_oof_by_fold
from model_interface import roi_pool


class V6UnitTests(unittest.TestCase):
    def test_mask_positive_values_are_foreground(self):
        raw = np.asarray([[0, 1], [2, 255]], dtype=np.uint8)
        expected = np.asarray([[0, 1], [1, 1]], dtype=np.float32)
        np.testing.assert_array_equal((raw > 0).astype(np.float32), expected)

    def test_percentile_normalization_is_finite_and_bounded(self):
        image = np.arange(100, dtype=np.float32).reshape(10, 10)
        result = percentile_normalize(image, 1.0, 99.0)
        self.assertTrue(np.isfinite(result).all())
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)

    def test_synchronized_affine_preserves_alignment(self):
        image = np.zeros((128, 128), dtype=np.float32)
        mask = np.zeros((128, 128), dtype=np.float32)
        cv2.circle(image, (64, 64), 15, 1.0, -1)
        cv2.circle(mask, (64, 64), 15, 1.0, -1)
        np.random.seed(20260810)
        transformed_image, transformed_mask, applied, fallback = synchronized_affine(
            image,
            mask,
            {
                "geometry_probability": 1.0,
                "rotation_degrees": 10.0,
                "translate_fraction": 0.06,
                "scale_delta": 0.10,
            },
        )
        self.assertTrue(applied)
        self.assertFalse(fallback)
        image_binary = transformed_image > 0.5
        mask_binary = transformed_mask > 0
        intersection = np.logical_and(image_binary, mask_binary).sum()
        union = np.logical_or(image_binary, mask_binary).sum()
        self.assertGreater(intersection / max(1, union), 0.90)

    def test_roi_pool_accepts_arbitrary_spatial_size(self):
        feature = torch.arange(2 * 256 * 17 * 19, dtype=torch.float32).reshape(2, 256, 17, 19)
        mask = torch.ones(2, 1, 71, 83)
        pooled, mass = roi_pool(feature, mask, mode="bilinear")
        self.assertEqual(tuple(pooled.shape), (2, 256))
        self.assertEqual(tuple(mass.shape), (2, 1))
        self.assertTrue(torch.isfinite(pooled).all())

    def test_featurebank_fold_selection(self):
        values = np.arange(3 * 5 * 2).reshape(3, 5, 2)
        folds = np.asarray([1, 3, 5])
        selected = select_oof_by_fold(values, folds)
        np.testing.assert_array_equal(selected, np.stack([values[0, 0], values[1, 2], values[2, 4]]))


if __name__ == "__main__":
    unittest.main()

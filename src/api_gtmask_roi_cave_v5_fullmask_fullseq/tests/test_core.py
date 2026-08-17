from __future__ import annotations

import gzip
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

PACKAGE = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(PACKAGE))

from nifti_io import load_label_mask
from roi import bbox_from_mask, context_square_bbox, crop_frames


def import_script(name: str):
    path = PACKAGE / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_nifti(path: Path, array: np.ndarray) -> None:
    array = np.asarray(array)
    code, bitpix = (2, 8) if array.dtype == np.uint8 else (4, 16)
    header = bytearray(352)
    struct.pack_into("<i", header, 0, 348)
    dimensions = [array.ndim, *array.shape] + [1] * (7 - array.ndim)
    struct.pack_into("<8h", header, 40, *dimensions[:8])
    struct.pack_into("<h", header, 70, code)
    struct.pack_into("<h", header, 72, bitpix)
    struct.pack_into("<f", header, 108, 352.0)
    header[344:348] = b"n+1\0"
    with gzip.open(path, "wb") as handle:
        handle.write(header)
        handle.write(array.tobytes(order="F"))


class CoreTests(unittest.TestCase):
    def test_minimal_nifti_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Post-Segmentation.nii.gz"
            labels = np.zeros((12, 10), dtype=np.int16)
            labels[3:8, 4:9] = 6
            write_nifti(path, labels)
            loaded, meta = load_label_mask(path)
            np.testing.assert_array_equal(loaded, labels)
            self.assertEqual(meta["collapsed_shape"], "12x10")

    def test_roi_covers_all_nonzero_labels(self):
        labels = np.zeros((64, 64), dtype=np.int16)
        labels[20:30, 10:25] = 1
        labels[25:35, 24:32] = 6
        bbox = bbox_from_mask(labels != 0)
        roi, audit = context_square_bbox(
            bbox, labels.shape, bbox_factor=1.5, min_side_pixels=16,
            min_margin_pixels=4, round_multiple=8,
        )
        self.assertLessEqual(roi[0], bbox[0])
        self.assertLessEqual(roi[1], bbox[1])
        self.assertGreaterEqual(roi[2], bbox[2])
        self.assertGreaterEqual(roi[3], bbox[3])
        self.assertEqual(audit["roi_side"] % 8, 0)

    def test_same_crop_is_applied_to_every_frame(self):
        frames = np.stack([np.full((10, 10), value, np.uint8) for value in (1, 2, 3)])
        cropped = crop_frames(frames, (-2, 2, 6, 10))
        self.assertEqual(cropped.shape, (3, 8, 8))
        self.assertEqual(int(cropped[0, 4, 4]), 1)
        self.assertEqual(int(cropped[2, 4, 4]), 3)

    def test_phase_hint_for_standalone_mask(self):
        module = import_script("01_discover_masks.py")
        self.assertEqual(module.phase_hint(Path("/x/y/Post-Segmentation.nii.gz")), "post")
        self.assertEqual(module.phase_hint(Path("/x/Pre-biaozhu/Segmentation.nii.gz")), "pre")


def _tiny_phase_index() -> pd.DataFrame:
    return pd.DataFrame([{
        "phase_uid": "Train__p1__main::post",
        "series_uid": "Train__p1__main",
        "phase": "post",
        "frame_paths": "a.jpg|b.jpg",
    }])


class OverrideResolutionTests(unittest.TestCase):
    def test_opaque_upstream_phase_uid_falls_back_to_series_phase(self):
        module = import_script("02_map_masks_to_phases.py")
        row = {"phase_uid": "b43d687870297fb650fd9881", "series_uid": "Train__p1__main", "phase": "Post"}
        uid, phase_row = module.resolve_override_phase(row, _tiny_phase_index())
        self.assertEqual(uid, "Train__p1__main::post")
        self.assertIsNotNone(phase_row)
        self.assertEqual(phase_row["series_uid"], "Train__p1__main")

    def test_explicit_valid_phase_uid_is_preferred(self):
        module = import_script("02_map_masks_to_phases.py")
        row = {"phase_uid": "Train__p1__main::post", "series_uid": "Train__p1__main", "phase": "post"}
        uid, phase_row = module.resolve_override_phase(row, _tiny_phase_index())
        self.assertEqual(uid, "Train__p1__main::post")
        self.assertIsNotNone(phase_row)

    def test_unresolvable_override_returns_empty_row(self):
        module = import_script("02_map_masks_to_phases.py")
        row = {"phase_uid": "deadbeef", "series_uid": "Train__unknown", "phase": "post"}
        uid, phase_row = module.resolve_override_phase(row, _tiny_phase_index())
        self.assertEqual(uid, "deadbeef")
        self.assertIsNone(phase_row)


class QaOverlayTests(unittest.TestCase):
    def test_overlay_marks_contour_tight_and_roi_boxes(self):
        module = import_script("04_make_roi_qa.py")
        frame = np.full((64, 64), 40, np.uint8)
        labels = np.zeros((64, 64), np.int32)
        # L 形前景：轮廓不会与 tight bbox 边框完全重合，三种标记都应可见
        labels[20:30, 10:15] = 2
        labels[25:30, 15:25] = 2
        image = module.overlay(frame, labels, (10, 20, 25, 30), (0, 8, 40, 42))
        self.assertEqual(image.shape, (64, 64, 3))
        # 黄色轮廓/绿色 ROI 边框/红色 tight bbox 都必须真实画上
        self.assertTrue(((image[..., 0] == 0) & (image[..., 1] == 255) & (image[..., 2] == 255)).any())
        self.assertTrue(((image[..., 0] == 0) & (image[..., 1] == 255) & (image[..., 2] == 0)).any())
        self.assertTrue(((image[..., 0] == 0) & (image[..., 1] == 0) & (image[..., 2] == 255)).any())

class EligibilityTests(unittest.TestCase):
    def _fixture(self, directory: str):
        root = Path(directory)
        frame = np.full((64, 64), 40, np.uint8)
        frame[20:30, 10:25] = 220
        frame_path = root / "f0.png"
        cv2.imwrite(str(frame_path), frame)
        mask_path = root / "mask.nii.gz"
        labels = np.zeros((64, 64), dtype=np.int16)
        labels[20:30, 10:25] = 2
        write_nifti(mask_path, labels)
        whole = root / "whole" / "train" / "p1" / "s1" / "post"
        whole.mkdir(parents=True)
        (whole / "metadata.json").write_text(
            json.dumps({"blocks": [{"indices": [0], "view_indices": {"uniform_full20": [0], "contrast_core20": [0]}}]})
        )
        mapped = {
            "phase_uid": "s1::post", "split": "Train", "source_series_order": "0",
            "source_phase_order": "1", "patient_id": "p1", "series_uid": "s1",
            "phase": "post", "frame_list_hash": "h1",
            "frame_paths": str(frame_path), "phase_mapping_status": "accepted",
            "mapping_method": "upstream", "mask_path": str(mask_path), "mask_sha256": "",
            "orientation_transform": "identity",
        }
        return mapped, root / "whole"

    def test_eligible_phase_produces_roi(self):
        module = import_script("03_build_full_roi_manifests.py")
        with tempfile.TemporaryDirectory() as directory:
            mapped, whole = self._fixture(directory)
            cov, roi_row, morph, view = module.evaluate_phase(mapped, {}, whole)
            self.assertEqual(cov["local_eligible"], 1)
            self.assertEqual(cov["local_exclusion_reason"], "")
            self.assertIsNotNone(roi_row)
            self.assertGreaterEqual(int(roi_row["roi_side"]), 23)

    def test_missing_whole_metadata_excludes(self):
        module = import_script("03_build_full_roi_manifests.py")
        with tempfile.TemporaryDirectory() as directory:
            mapped, whole = self._fixture(directory)
            cov, roi_row, _, _ = module.evaluate_phase(mapped, {}, whole / "nonexistent")
            self.assertEqual(cov["local_eligible"], 0)
            self.assertTrue(cov["local_exclusion_reason"].startswith("whole_metadata_missing"))
            self.assertIsNone(roi_row)

    def test_unmapped_phase_excluded_without_io(self):
        module = import_script("03_build_full_roi_manifests.py")
        cov, roi_row, _, _ = module.evaluate_phase(
            {"phase_uid": "s1::post", "split": "Train", "source_series_order": "0",
             "source_phase_order": "1", "patient_id": "p1", "series_uid": "s1",
             "phase": "post", "frame_list_hash": "h1", "phase_mapping_status": "missing",
             "mapping_method": "", "mask_path": "", "mask_sha256": "", "orientation_transform": ""},
            {}, Path("/nonexistent"),
        )
        self.assertEqual(cov["local_exclusion_reason"], "no_mask_mapping")
        self.assertIsNone(roi_row)


if __name__ == "__main__":
    unittest.main()

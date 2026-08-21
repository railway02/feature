from pathlib import Path

import numpy as np
import pandas as pd

from dsa_reg.io import phase_to_reference_canvas, joint_mask_center, crop_phase, crop_origin_yx
from dsa_reg.manifest import load_manifest
from dsa_reg.pipeline import identity_mapping_is_independently_verified, _direct_reference_to_peak
from dsa_reg.global_correspondence import (
    canonical_similarity_matrix, coarse_similarity_candidates, compatibility_metrics,
    global_compatibility_status,
)
import cv2


def test_whole_fov_resize_and_shared_crop_preserve_relative_translation():
    pre_seq = np.zeros((2, 64, 64), np.float32)
    post_seq = np.zeros((2, 64, 64), np.float32)
    pre_ref = np.zeros((128, 128), np.float32)
    post_ref = np.zeros((128, 128), np.float32)
    pre_mask = np.zeros((128, 128), bool); pre_mask[55:65, 50:60] = 1
    post_mask = np.zeros((128, 128), bool); post_mask[59:69, 57:67] = 1
    pre_seq, _, pre_mask, _ = phase_to_reference_canvas(pre_seq, pre_ref, pre_mask, (128, 128))
    post_seq, _, post_mask, _ = phase_to_reference_canvas(post_seq, post_ref, post_mask, (128, 128))
    center = joint_mask_center(pre_mask, post_mask)
    _, pre_crop, _, _ = crop_phase(pre_seq, pre_mask, (64, 64), center_yx=center)
    _, post_crop, _, _ = crop_phase(post_seq, post_mask, (64, 64), center_yx=center)
    py, px = np.mean(np.nonzero(pre_crop), axis=1)
    qy, qx = np.mean(np.nonzero(post_crop), axis=1)
    assert np.allclose([qy - py, qx - px], [4.0, 7.0])
    assert crop_origin_yx(center, (64, 64)) == crop_origin_yx(center, (64, 64))


def test_unified_manifest_schema_adapter(tmp_path: Path):
    row = {
        "split": "Train", "patient_id": 1, "series_uid": "u", "series_id": "s",
        "png2d_image_path_pre": "a", "png2d_image_path_post": "b",
        "png2d_mask_path_pre": "c", "png2d_mask_path_post": "d",
        "pre_frame_paths": "e", "post_frame_paths": "f",
        "n_pre_frames": 1, "n_post_frames": 1,
    }
    p = tmp_path / "manifest.csv"; pd.DataFrame([row]).to_csv(p, index=False)
    out = load_manifest(p)
    assert out.loc[0, "pre_reference_image_path"] == "a"
    assert out.loc[0, "post_mask_path"] == "d"
    assert out.loc[0, "pre_n_frames"] == 1


def test_canvas_preserves_content_scale_for_similarity_to_correct():
    # Both full-FOV exports map to the canvas, but their actual object magnification
    # remains 1.2 rather than being normalised away by preprocessing.
    pre_ref = np.zeros((100, 100), np.float32); post_ref = np.zeros((100, 100), np.float32)
    yy, xx = np.mgrid[:100, :100]
    pre_ref[(yy-50)**2 + (xx-50)**2 <= 10**2] = 1
    post_ref[(yy-50)**2 + (xx-50)**2 <= 12**2] = 1
    pre_mask, post_mask = pre_ref > 0, post_ref > 0
    pre_seq = pre_ref[None, ...]; post_seq = post_ref[None, ...]
    _, pre_canvas, _, _ = phase_to_reference_canvas(pre_seq, pre_ref, pre_mask, (120, 120))
    _, post_canvas, _, _ = phase_to_reference_canvas(post_seq, post_ref, post_mask, (120, 120))
    # Area ratio remains approximately s^2=1.44, demonstrating that canvas export
    # normalisation does not erase true in-image magnification.
    ratio = float((post_canvas > .5).sum() / (pre_canvas > .5).sum())
    assert np.isclose(ratio, 1.2**2, rtol=.12)


def test_full_fov_coarse_similarity_captures_large_translation_and_scale():
    """293121-style acquisition change must be captured before local refinement/SyN."""
    moving = np.zeros((384, 384), np.float32)
    moving[60:320, 120:124] = 1
    moving[115:120, 55:285] = 1
    moving[195:300, 220:225] = 1
    moving[250:255, 180:335] = 1
    truth = canonical_similarity_matrix(moving.shape, 4.0, 1.20, 128.0, -72.0)
    fixed = cv2.warpAffine(moving, truth, (384, 384))
    cfg = {
        "coarse_max_dim": 192,
        "coarse_rotation_degrees": [-4, 0, 4],
        "coarse_scales": [1.0, 1.1, 1.2],
        "coarse_keep_candidates": 3,
    }
    candidates = coarse_similarity_candidates(
        fixed, moving, fixed > .2, moving > .2,
        np.ones_like(fixed, bool), np.ones_like(moving, bool), cfg, "similarity",
    )
    best = candidates[0]
    metrics = compatibility_metrics(
        moving > .2, fixed > .2, np.ones_like(moving, bool),
        np.ones_like(fixed, bool), best["transform"],
    )
    assert best["scale"] == 1.2
    assert abs(best["translation_x"] - 128.0) < 6
    assert abs(best["translation_y"] + 72.0) < 6
    assert metrics["trimmed_chamfer"] < 1.5


def test_global_gate_distinguishes_partial_projection_from_joint_implausibility():
    cfg = {
        "global_min_bidirectional_coverage_8": .15,
        "global_caution_bidirectional_coverage_5": .20,
        "global_max_trimmed_chamfer": 22,
        "global_min_coverage_5_for_partial_projection": .30,
        "global_min_fov_overlap": .45,
        "global_caution_fov_overlap": .60,
        "global_max_abs_rotation_deg": 20,
        "global_caution_abs_rotation_deg": 12,
        "global_min_scale": .65, "global_max_scale": 1.45,
        "global_caution_min_scale": .80, "global_caution_max_scale": 1.25,
        "global_caution_min_score": .20,
        "global_max_caution_reasons_before_fail": 3,
    }
    partial = {
        "coverage_5_moving": .34, "coverage_5_fixed": .36,
        "coverage_8_moving": .42, "coverage_8_fixed": .44,
        "trimmed_chamfer": 48, "fov_overlap": 1.0, "score": .06,
    }
    status, _ = global_compatibility_status(
        partial, {"rotation_deg": .5, "scale": 1.05}, cfg
    )
    assert status == "GLOBAL_PASS_WITH_CAUTION"
    joint_bad = {
        "coverage_5_moving": .38, "coverage_5_fixed": .36,
        "coverage_8_moving": .49, "coverage_8_fixed": .42,
        "trimmed_chamfer": 16, "fov_overlap": .48, "score": .15,
    }
    status, reasons = global_compatibility_status(
        joint_bad, {"rotation_deg": -17.7, "scale": .77}, cfg
    )
    assert status == "GLOBAL_FAIL" and "joint_global_plausibility_failure" in reasons


def test_reference_mapping_rejection_is_fail_closed_without_independent_evidence(monkeypatch, tmp_path):
    import dsa_reg.pipeline as pipeline
    def fail(*args, **kwargs):
        raise RuntimeError('forced reference registration rejection')
    monkeypatch.setattr(pipeline, 'register_pair', fail)
    cfg = {
        'geometry': {'min_mapping_score': .75, 'identity_mapping_method_tokens': ['identity']},
        'preprocess': {'percentile_low': 1, 'percentile_high': 99, 'frangi_sigmas': [1, 2],
                       'vessel_threshold_percentile': 82, 'vessel_min_size': 2, 'centerline_sigma_px': 2},
        'intra_registration': {'reference_transform': 'similarity', 'metric': 'correlation',
                               'shrink_factors': [1], 'smoothing_sigmas': [0], 'learning_rate': 1,
                               'min_step': .01, 'iterations': 1, 'gradient_tolerance': 1e-6,
                               'reference_min_ncc': .15, 'reference_min_ncc_gain': .01},
    }
    ref = np.zeros((32, 32), np.float32); ref[12:20, 15:17] = 1
    mask = ref > 0
    mapped, meta = _direct_reference_to_peak(ref, mask, ref, np.ones_like(mask), cfg, tmp_path,
                                              mapping_score=.99, mapping_method='unverified_alias')
    assert mapped is None and not meta['phase_geometry_valid']
    assert identity_mapping_is_independently_verified(.99, 'identity_verified', cfg)

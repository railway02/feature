#!/usr/bin/env python
"""Production-path validation of residual Pre→Post Jacobian semantics.

Each case passes through JPG I/O, canvas/crop preprocessing, peak selection, intra-phase
rigid correction, SimpleITK global registration, actual ANTs SyNOnly Warp/InverseWarp,
and the production canonical Jacobian function.  It intentionally does not validate a
standalone NumPy determinant.  The global-scale case verifies that a 1.20 scale is
absorbed by Similarity and excluded from the residual clinical Jacobian.
"""
import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import gaussian_filter

from dsa_reg.pipeline import process_series


def base_image(radius=9, shape=(128, 128)):
    h, w = shape
    img = np.zeros((h, w), np.float32)
    cv2.line(img, (16, 72), (112, 72), 1.0, 3)
    cv2.line(img, (64, 72), (90, 35), .9, 2)
    cv2.line(img, (78, 72), (105, 98), .8, 2)
    cv2.circle(img, (64, 62), radius, 1.0, -1)
    return gaussian_filter(img, 1.0)


def lesion_mask(radius):
    m = np.zeros((128, 128), np.uint8)
    cv2.circle(m, (64, 62), radius, 255, -1)
    return m


def affine(img, angle=0, scale=1, tx=0, ty=0, nearest=False):
    h, w = img.shape
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    mat[:, 2] += np.array([tx, ty])
    return cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                          borderValue=0)


def sequence(img):
    amplitudes = np.array([0, .05, .18, .45, .75, 1, .82, .55, .28, .08, .02], np.float32)
    return np.stack([img * a for a in amplitudes])


def write_case(root: Path, name: str, pre, post, pre_mask, post_mask):
    pre_dir, post_dir = root / name / "pre", root / name / "post"
    pre_dir.mkdir(parents=True); post_dir.mkdir(parents=True)
    pre_paths, post_paths = [], []
    for phase, seq, directory, paths, start in (
        ("pre", sequence(pre), pre_dir, pre_paths, 1), ("post", sequence(post), post_dir, post_paths, 30)
    ):
        for i, frame in enumerate(seq, start):
            p = directory / f"IMG-{phase}-{i:05d}.jpg"
            cv2.imwrite(str(p), np.clip(frame * 255, 0, 255).astype(np.uint8)); paths.append(str(p))
    refs = root / name / "refs"; refs.mkdir()
    pre_ref, post_ref = refs / "pre.png", refs / "post.png"
    pre_m, post_m = refs / "pre_mask.png", refs / "post_mask.png"
    cv2.imwrite(str(pre_ref), (pre * 255).astype(np.uint8)); cv2.imwrite(str(post_ref), (post * 255).astype(np.uint8))
    cv2.imwrite(str(pre_m), pre_mask); cv2.imwrite(str(post_m), post_mask)
    return pd.Series({
        "split": "Train", "patient_id": 990000 + len(name), "series_uid": name, "series_id": name,
        "pre_reference_image_path": str(pre_ref), "post_reference_image_path": str(post_ref),
        "pre_mask_path": str(pre_m), "post_mask_path": str(post_m),
        "pre_frame_paths": "|".join(pre_paths), "post_frame_paths": "|".join(post_paths),
        "pre_n_frames": len(pre_paths), "post_n_frames": len(post_paths),
        "pre_mapping_method": "synthetic_identity_verified", "post_mapping_method": "synthetic_identity_verified",
        "pre_mapping_score": 1.0, "post_mapping_score": 1.0,
    })


def median_region(feature_root: Path, case: str, region: str):
    z = np.load(feature_root / "Train" / str(990000 + len(case)) / case / "change_maps.npz")
    valid = z["canonical_valid"].astype(bool)
    return float(np.nanmedian(z["canonical_logjac"][z[region].astype(bool) & valid]))


def main():
    cfg0 = yaml.safe_load((Path(__file__).resolve().parents[1] / "config/default.yaml").read_text())
    outcomes = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); out = root / "out"
        cfg = cfg0
        cfg["paths"].update({"output_root": str(out), "remap": {}})
        cfg["geometry"]["canvas_size"] = [128, 128]; cfg["roi"]["size"] = [128, 128]
        cfg["intra_registration"]["iterations"] = 50
        cfg["global_registration"].update({"iterations": 120, "run_methods": ["rigid", "similarity", "affine"]})
        cfg["nonrigid"]["reg_iterations"] = [40, 20, 10]
        pre = base_image(9)
        cases = {
            "translation": (pre, affine(pre, tx=4, ty=-3), lesion_mask(9), affine(lesion_mask(9), tx=4, ty=-3, nearest=True)),
            "expansion": (pre, base_image(12), lesion_mask(9), lesion_mask(12)),
            "contraction": (base_image(12), pre, lesion_mask(12), lesion_mask(9)),
            "global_scale": (pre, affine(pre, scale=1.20), lesion_mask(9), affine(lesion_mask(9), scale=1.20, nearest=True)),
        }
        for name, spec in cases.items():
            f = process_series(write_case(root, name, *spec), cfg)
            metrics = {
                "registration_valid": int(f["registration_valid"]),
                "q_reg": float(f["q_reg"]),
                "global_similarity_scale": float(f["global_similarity_scale"]),
                "lesion_logjac_median": median_region(out, name, "lesion"),
                "stable_logjac_median": median_region(out, name, "stable"),
            }
            outcomes[name] = metrics
        assertions = {
            "translation": abs(outcomes["translation"]["stable_logjac_median"]) < .15,
            "expansion": outcomes["expansion"]["lesion_logjac_median"] > .05,
            "contraction": outcomes["contraction"]["lesion_logjac_median"] < -.05,
            "global_scale_parameter": abs(outcomes["global_scale"]["global_similarity_scale"] - 1.20) < .15,
            "global_scale_residual": abs(outcomes["global_scale"]["stable_logjac_median"]) < .15,
        }
    report = {"outcomes": outcomes, "assertions": assertions, "all_pass": bool(all(assertions.values()))}
    print(json.dumps(report, indent=2))
    if not report["all_pass"]:
        raise SystemExit("production Jacobian validation failed")


if __name__ == "__main__":
    main()

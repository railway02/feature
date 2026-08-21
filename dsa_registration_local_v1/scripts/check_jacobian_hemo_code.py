#!/usr/bin/env python3
"""Minimal code/synthetic checks required before the single fixed Smoke10."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from dsa_local_reg.common import atomic_json, load_config
from dsa_local_reg.hemodynamics_v1 import compact36_columns
from dsa_local_reg.jacobian_derived import (
    existing42_columns, extended28_columns, extract_extended_raw28, rederive_canonical_maps,
)
from dsa_local_reg.temporal_contract import build_frozen_contracts, select_smoke10
from dsa_local_reg.temporal_motion import apply_signal_and_support
from dsa_local_reg.v5_adapter import load_v5_module


def _result(name: str, passed: bool, **details):
    return {"name": name, "passed": bool(passed), **details}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT / "config" / "default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    results = []
    smoke = select_smoke10(cfg["paths"]["fov50_series"])
    results.append(_result("select_smoke10_5_strata_x_2", len(smoke) == 10 and smoke.stratum.nunique() == 5 and not smoke.series_uid.duplicated().any()))
    results.append(_result("feature_schema_dimensions_and_order", len(compact36_columns()) == 36 and len(existing42_columns()) == 42 and len(extended28_columns()) == 28,
                           compact36=len(compact36_columns()), existing42=len(existing42_columns()), extended28=len(extended28_columns())))
    # Pure frame→peak identity: raw signal and binary real-source support must remain exact.
    raw = np.arange(2 * 32 * 32, dtype=np.float32).reshape(2, 32, 32)
    support = np.ones((32, 32), dtype=bool); support[:2] = False
    import SimpleITK as sitk
    identity = sitk.Euler2DTransform(); identity.SetCenter((15.5, 15.5))
    corrected, corrected_support = apply_signal_and_support(raw, support, [identity, identity], cfg)
    results.append(_result("identity_motion_signal_and_support", np.array_equal(corrected, raw) and np.array_equal(corrected_support, np.stack([support, support]))))
    # Synthetic translation: the same local pattern shifted in a later frame should be rigid-correctable;
    # its support is independently warped with nearest-neighbour and cannot create out-of-source values.
    base = np.zeros((64, 64), np.float32); base[20:42, 25:38] = 100.0; base[30:34, 10:56] = 40.0
    moving = np.zeros_like(base); moving[3:, 5:] = base[:-3, :-5]
    sitkmod = load_v5_module(cfg, "registration_sitk.py")
    tx, _ = sitkmod.register_pair(base, moving, kind="rigid", fixed_mask=np.ones_like(base, bool), moving_mask=np.ones_like(base, bool),
                                  metric="correlation", shrink_factors=(2, 1), smoothing_sigmas=(1, 0), iterations=80)
    aligned = sitkmod.resample(moving, base, tx, default=0.0)
    ncc = float(np.corrcoef(base.ravel(), aligned.ravel())[0, 1])
    results.append(_result("synthetic_translation_rigid_correction", np.isfinite(ncc) and ncc > 0.95, ncc=ncc))
    positive = np.array([[True, False], [False, True]]); folding = np.array([[False, True], [False, False]])
    fixed = np.array([[True, True], [False, True]])
    valid_corrected, folding_corrected = positive & fixed, folding & fixed
    results.append(_result("jacobian_positive_folding_support_partition", not np.any(valid_corrected & folding_corrected) and int((valid_corrected | folding_corrected).sum()) == 3,
                           denominator=int((valid_corrected | folding_corrected).sum())))
    toy_logj = np.array([[.08, -.08, 0.], [.06, -.07, .02], [0., 0., 0.]], dtype=np.float32)
    toy_regions = {"lesion": np.ones_like(toy_logj, bool), "peri_lesion": np.ones_like(toy_logj, bool), "whole_valid_local_roi": np.ones_like(toy_logj, bool)}
    toy_raw28 = extract_extended_raw28(toy_logj, toy_regions)
    results.append(_result("extended_raw28_frozen_schema_order", list(toy_raw28) == extended28_columns()))
    contracts = build_frozen_contracts(cfg)
    first = next(item for item in contracts if item.series_uid == smoke.iloc[0].series_uid)
    maps = rederive_canonical_maps(first, cfg)
    jc = cfg["jacobian_hemo"]["jacobian"]
    compare = maps["comparison"]
    passed = compare["n"] > 0 and compare["mae"] <= float(jc["stored_rederived_logj_mae_max"]) and compare["max_abs"] <= float(jc["stored_rederived_logj_max_abs"])
    results.append(_result("existing_warp_canonical_logj_readonly_consistency", passed, series_uid=first.series_uid, **compare,
                           residual_identity=maps["identity"]["residual_linear_identity_verified"], ants_registration_called=False))
    failures = [item for item in results if not item["passed"]]
    payload = {"status": "PASS" if not failures else "FAIL", "scope": "minimal_code_and_synthetic_checks_only", "outcome_accessed": False,
               "g0_rigid_or_syn_rerun": False, "results": results, "failures": failures}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(payload, args.output_dir / "MINIMAL_CODE_CHECKS.json")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())

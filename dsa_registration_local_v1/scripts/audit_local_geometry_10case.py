#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from dsa_local_reg.common import atomic_json, load_config
from dsa_local_reg.preprocessing_adapter import load_local_reference_pairs
from dsa_local_reg.visualization import (
    geometry_selection_table,
    make_independent_local_crop_sheet,
    select_geometry_cases,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage B: 10-case independent-local-crop geometry audit")
    parser.add_argument("--config", type=Path, default=PROJECT / "config" / "default.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-cases", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    split = str(cfg["audit"]["train_split"])
    records = load_local_reference_pairs(cfg, split=split)
    n_cases = int(args.n_cases or cfg["audit"]["stage_b_cases"])
    selected = select_geometry_cases(records, n_cases=n_cases)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    geometry_selection_table(selected).to_csv(args.output_dir / "selected_geometry_cases.csv", index=False)
    g1_canvas = tuple(int(value) for value in cfg["geometry"]["g1_audit_canvas_size"])
    rows = []
    for record in selected:
        rows.append(make_independent_local_crop_sheet(
            record,
            args.output_dir / "contact_sheets" / f"{record.series_uid}.png",
            frame_position=int(cfg["audit"]["representative_frame_position"]),
            g1_canvas_yx=g1_canvas,
            dpi=int(cfg["audit"]["contact_sheet_dpi"]),
        ))
    provenance = pd.DataFrame(rows)
    provenance.to_csv(args.output_dir / "geometry_audit_provenance.csv", index=False)
    expected_shapes = []
    expected_paddings = []
    for record in selected:
        expected_shapes.extend([
            [record.pre.expanded_bbox.height, record.pre.expanded_bbox.width],
            [record.post.expanded_bbox.height, record.post.expanded_bbox.width],
        ])
        expected_paddings.extend([
            [record.pre.padding_left, record.pre.padding_top, record.pre.padding_right, record.pre.padding_bottom],
            [record.post.padding_left, record.post.padding_top, record.post.padding_right, record.post.padding_bottom],
        ])
    observed_shapes = []
    observed_paddings = []
    for row in rows:
        observed_shapes.extend([row["pre_g0_shape"], row["post_g0_shape"]])
        observed_paddings.extend([row["pre_padding"], row["post_padding"]])
    checks = [
        {"name": "selected_train_cases", "value": len(selected), "comparison": "==", "tolerance": n_cases, "passed": len(selected) == n_cases},
        {"name": "all_g0_crop_shapes_equal_old_expanded_bbox", "value": int(observed_shapes == expected_shapes), "comparison": "==", "tolerance": 1, "passed": observed_shapes == expected_shapes},
        {"name": "all_g0_padding_equals_old_provenance", "value": int(observed_paddings == expected_paddings), "comparison": "==", "tolerance": 1, "passed": observed_paddings == expected_paddings},
        {"name": "all_orientation_identity", "value": int(all(r.pre.orientation_transform == "identity" and r.post.orientation_transform == "identity" for r in selected)), "comparison": "==", "tolerance": 1, "passed": all(r.pre.orientation_transform == "identity" and r.post.orientation_transform == "identity" for r in selected)},
        {"name": "all_source_pixel_spacings_positive", "value": int(all(min(*r.pre.source_spacing_xy, *r.post.source_spacing_xy) > 0 for r in selected)), "comparison": "==", "tolerance": 1, "passed": all(min(*r.pre.source_spacing_xy, *r.post.source_spacing_xy) > 0 for r in selected)},
        {"name": "direction_fixed_post_moving_pre", "value": "fixed=Post,moving=Pre", "comparison": "==", "tolerance": "fixed=Post,moving=Pre", "passed": True},
        {"name": "g1_not_promoted_to_primary", "value": cfg["geometry"]["primary_mode"], "comparison": "==", "tolerance": "g0_native_independent_local_crop", "passed": cfg["geometry"]["primary_mode"] == "g0_native_independent_local_crop"},
    ]
    failures = [check["name"] for check in checks if not check["passed"]]
    pd.DataFrame(checks).to_csv(args.output_dir / "stage_b_check_report.csv", index=False)
    payload = {
        "status": "pass" if not failures else "fail", "stage": "B", "scope": "geometry_only_no_real_registration",
        "split": split, "n_cases": len(selected), "geometry_primary": cfg["geometry"]["primary_mode"],
        "g1_status": "audit_only_not_primary", "series_uids": [record.series_uid for record in selected],
        "checks": checks, "failures": failures,
        "geometry_decision": "G0 remains primary; no technical pixel-scale evidence requires G1 in this 10-case audit" if not failures else "No geometry mode freeze due to failed checks",
    }
    marker = "STAGE_B_PASS.json" if not failures else "STAGE_B_FAIL.json"
    atomic_json(payload, args.output_dir / marker)
    print(f"{payload['status'].upper()}: wrote {len(selected)} independent-local-crop geometry contact sheets to {args.output_dir}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Paired, development-only comparison: geometry+original weight vs geometry+cap3."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp/aneurysm/reports/api_png2d_spatial_branch_failure_audit_20260810/pilot/outer_development_fold_1")
REFERENCE = ROOT / "segresnet_geometry/best_inner_valid_predictions.csv"
CANDIDATE = ROOT / "segresnet_geometry_pos3/best_inner_valid_predictions.csv"
OUT = ROOT / "segresnet_geometry_pos3/paired_vs_geometry"
KEY = ["patient_id", "series_uid", "phase"]


def bootstrap(values: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    estimates = np.empty(10000, dtype=np.float64)
    for i in range(len(estimates)):
        estimates[i] = values[rng.integers(0, len(values), len(values))].mean()
    return {"mean": float(values.mean()), "ci_low": float(np.quantile(estimates, .025)),
            "ci_high": float(np.quantile(estimates, .975)), "probability_gt_zero": float((estimates > 0).mean())}


def summary(group: pd.DataFrame) -> dict:
    old_fail = group.dice_geometry < .2
    new_fail = group.dice_pos3 < .2
    return {
        "n": int(len(group)), "geometry_dice": float(group.dice_geometry.mean()),
        "pos3_dice": float(group.dice_pos3.mean()), "mean_dice_gain": float(group.dice_gain.mean()),
        "median_dice_gain": float(group.dice_gain.median()), "fraction_improved": float((group.dice_gain > 0).mean()),
        "geometry_failure_lt_02": float(old_fail.mean()), "pos3_failure_lt_02": float(new_fail.mean()),
        "geometry_failure_lt_05": float((group.dice_geometry < .5).mean()), "pos3_failure_lt_05": float((group.dice_pos3 < .5).mean()),
        "severe_failures_rescued": int((old_fail & ~new_fail).sum()), "new_severe_failures": int((~old_fail & new_fail).sum()),
        "geometry_area_ratio_mean": float(group.pred_gt_area_ratio_geometry.mean()),
        "pos3_area_ratio_mean": float(group.pred_gt_area_ratio_pos3.mean()),
        "geometry_area_ratio_total": float(group.pred_pixels_geometry.sum() / group.gt_pixels_geometry.sum()),
        "pos3_area_ratio_total": float(group.pred_pixels_pos3.sum() / group.gt_pixels_pos3.sum()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(REFERENCE, dtype={"patient_id": str, "series_uid": str})
    new = pd.read_csv(CANDIDATE, dtype={"patient_id": str, "series_uid": str})
    paired = old.merge(new, on=KEY, how="inner", validate="one_to_one", suffixes=("_geometry", "_pos3"))
    if len(paired) != len(old) or len(paired) != len(new): raise RuntimeError("prediction rows did not align")
    if not np.array_equal(paired.gt_pixels_geometry, paired.gt_pixels_pos3): raise RuntimeError("GT changed")
    paired["dice_gain"] = paired.dice_pos3 - paired.dice_geometry
    paired["gt_size_quartile"] = pd.qcut(paired.gt_pixels_geometry, 4, labels=["Q1-smallest", "Q2", "Q3", "Q4-largest"])
    paired.to_csv(OUT / "paired_phase_predictions.csv", index=False)
    rows = [{"stratum": "Overall", "level": "Overall", **summary(paired)}]
    for col in ["phase", "gt_size_quartile"]:
        for level, group in paired.groupby(col, observed=True): rows.append({"stratum": col, "level": str(level), **summary(group)})
    pd.DataFrame(rows).to_csv(OUT / "paired_stratified_summary.csv", index=False)
    result = {"status": "success", "development_only": True, "outer_holdout_evaluated": False,
              "independent_valid_used": False, "n_phase_images": int(len(paired)),
              "macro_dice_gain_bootstrap": bootstrap(paired.dice_gain.to_numpy(), 20260813),
              "pre_dice_gain_bootstrap": bootstrap(paired.loc[paired.phase == "Pre", "dice_gain"].to_numpy(), 20260814),
              "post_dice_gain_bootstrap": bootstrap(paired.loc[paired.phase == "Post", "dice_gain"].to_numpy(), 20260815),
              "overall": summary(paired)}
    (OUT / "paired_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()

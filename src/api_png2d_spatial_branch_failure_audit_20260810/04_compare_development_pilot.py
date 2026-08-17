#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/root/autodl-tmp/aneurysm/reports/api_png2d_spatial_branch_failure_audit_20260810/pilot/outer_development_fold_1")
BASELINE = ROOT / "frozen_baseline/inner_valid_predictions.csv"
GEOMETRY = ROOT / "segresnet_geometry/best_inner_valid_predictions.csv"
OUT = ROOT / "paired_comparison"


def bootstrap_mean_difference(values: np.ndarray, seed=20260810, draws=10000):
    rng = np.random.default_rng(seed)
    n = len(values)
    estimates = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        estimates[i] = values[rng.integers(0, n, size=n)].mean()
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "probability_gt_zero": float((estimates > 0).mean()),
    }


def summarize(group):
    return pd.Series({
        "n": int(len(group)),
        "baseline_dice": float(group.dice_baseline.mean()),
        "geometry_dice": float(group.dice_geometry.mean()),
        "mean_dice_gain": float(group.dice_gain.mean()),
        "median_dice_gain": float(group.dice_gain.median()),
        "fraction_improved": float((group.dice_gain > 0).mean()),
        "baseline_failure_lt_02": float((group.dice_baseline < 0.2).mean()),
        "geometry_failure_lt_02": float((group.dice_geometry < 0.2).mean()),
        "severe_failures_rescued": int(((group.dice_baseline < 0.2) & (group.dice_geometry >= 0.2)).sum()),
        "new_severe_failures": int(((group.dice_baseline >= 0.2) & (group.dice_geometry < 0.2)).sum()),
        "baseline_area_ratio_mean": float(group.pred_gt_area_ratio_baseline.mean()),
        "geometry_area_ratio_mean": float(group.pred_gt_area_ratio_geometry.mean()),
    })


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    key = ["patient_id", "series_uid", "phase"]
    baseline = pd.read_csv(BASELINE, dtype={"patient_id": str, "series_uid": str})
    geometry = pd.read_csv(GEOMETRY, dtype={"patient_id": str, "series_uid": str})
    paired = baseline.merge(
        geometry,
        on=key,
        how="inner",
        validate="one_to_one",
        suffixes=("_baseline", "_geometry"),
    )
    if len(paired) != len(baseline) or len(paired) != len(geometry):
        raise RuntimeError("Pilot prediction rows did not align one-to-one")
    if not np.array_equal(paired.gt_pixels_baseline, paired.gt_pixels_geometry):
        raise RuntimeError("GT pixel counts changed between pilot variants")

    paired["dice_gain"] = paired.dice_geometry - paired.dice_baseline
    paired["gt_size_quartile"] = pd.qcut(
        paired.gt_pixels_baseline,
        4,
        labels=["Q1-smallest", "Q2", "Q3", "Q4-largest"],
    )
    paired["baseline_dice_band"] = pd.cut(
        paired.dice_baseline,
        [-np.inf, 0.2, 0.5, 0.8, np.inf],
        labels=["<0.2", "0.2-0.5", "0.5-0.8", ">=0.8"],
        right=False,
    )
    paired.to_csv(OUT / "paired_phase_predictions.csv", index=False)

    tables = []
    tables.append(pd.DataFrame([{"stratum": "Overall", **summarize(paired).to_dict()}]))
    for column in ["phase", "gt_size_quartile", "baseline_dice_band"]:
        table = paired.groupby(column, observed=True).apply(summarize).reset_index()
        table.insert(0, "stratum", column)
        tables.append(table.rename(columns={column: "level"}))
    summary = pd.concat(tables, ignore_index=True, sort=False)
    summary.to_csv(OUT / "paired_stratified_summary.csv", index=False)

    result = {
        "status": "success",
        "development_only": True,
        "outer_holdout_evaluated": False,
        "independent_valid_used": False,
        "n_phase_images": int(len(paired)),
        "macro_dice_gain_bootstrap": bootstrap_mean_difference(paired.dice_gain.to_numpy()),
        "pre_dice_gain_bootstrap": bootstrap_mean_difference(
            paired.loc[paired.phase == "Pre", "dice_gain"].to_numpy(), seed=20260811
        ),
        "post_dice_gain_bootstrap": bootstrap_mean_difference(
            paired.loc[paired.phase == "Post", "dice_gain"].to_numpy(), seed=20260812
        ),
        "fraction_improved": float((paired.dice_gain > 0).mean()),
        "severe_failures_rescued": int(((paired.dice_baseline < 0.2) & (paired.dice_geometry >= 0.2)).sum()),
        "new_severe_failures": int(((paired.dice_baseline >= 0.2) & (paired.dice_geometry < 0.2)).sum()),
        "largest_improvements": paired.nlargest(10, "dice_gain")[
            key + ["dice_baseline", "dice_geometry", "dice_gain", "gt_pixels_baseline"]
        ].to_dict("records"),
        "largest_regressions": paired.nsmallest(10, "dice_gain")[
            key + ["dice_baseline", "dice_geometry", "dice_gain", "gt_pixels_baseline"]
        ].to_dict("records"),
    }
    (OUT / "paired_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

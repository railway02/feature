#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def metrics(
    frame: pd.DataFrame, whole_column: str, roi_column: str
) -> dict[str, object]:
    result: dict[str, object] = {
        "rows": int(len(frame)),
        "positive": int(frame["target"].sum()) if len(frame) else 0,
    }
    if frame.empty:
        return result
    target = frame["target"].to_numpy(np.int64)
    result["positive_fraction"] = float(target.mean())
    if len(np.unique(target)) != 2:
        return result
    whole = frame[whole_column].to_numpy(np.float64)
    roi = frame[roi_column].to_numpy(np.float64)
    whole_result = {
        "auroc": float(roc_auc_score(target, whole)),
        "auprc": float(average_precision_score(target, whole)),
        "brier": float(brier_score_loss(target, whole)),
    }
    roi_result = {
        "auroc": float(roc_auc_score(target, roi)),
        "auprc": float(average_precision_score(target, roi)),
        "brier": float(brier_score_loss(target, roi)),
    }
    result.update(
        {
            "whole": whole_result,
            "roi": roi_result,
            "roi_minus_whole": {
                key: roi_result[key] - whole_result[key]
                for key in ("auroc", "auprc", "brier")
            },
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("/root/autodl-tmp/aneurysm")
    )
    args = parser.parse_args()
    root = args.root.resolve()
    reports = root / "reports/api_adverse_lesion_cave_fast_v1"

    phase = pd.read_csv(
        reports / "failure_mode_phase_audit.csv",
        dtype={"patient_id": str},
    )
    roi_manifest = pd.read_csv(
        root / "manifests/api_adverse_lesion_cave_fast_v1/roi_manifest_pred.csv",
        dtype=str,
        keep_default_na=False,
    )
    roi_manifest = roi_manifest[
        (roi_manifest["roi_branch"] == "pred")
        & (~roi_manifest["duplicate_excluded"].str.casefold().eq("true"))
    ][["phase_uid", "series_uid"]]
    phase = phase.merge(
        roi_manifest, on="phase_uid", how="left", validate="one_to_one"
    )

    quality = (
        phase.groupby(["split", "patient_id"], as_index=False)
        .agg(
            phase_rows=("phase_uid", "size"),
            series_count=("series_uid", "nunique"),
            pre_phase_rows=("phase", lambda values: int((values == "pre").sum())),
            post_phase_rows=("phase", lambda values: int((values == "post").sum())),
            median_3p0_coverage=("roi_3p0_coverage", "median"),
            minimum_3p0_coverage=("roi_3p0_coverage", "min"),
            mean_3p0_coverage=("roi_3p0_coverage", "mean"),
            zero_coverage_fraction=(
                "roi_3p0_coverage",
                lambda values: float((values == 0.0).mean()),
            ),
            coverage_ge_0p95_fraction=(
                "roi_3p0_coverage",
                lambda values: float((values >= 0.95).mean()),
            ),
            median_roi_area_ratio=("roi_3p0_area_ratio", "median"),
            segmentation_argmax_fraction=(
                "fallback_type",
                lambda values: float((values != "none").mean()),
            ),
        )
    )
    quality["all_phases_ge_0p95"] = quality["minimum_3p0_coverage"] >= 0.95
    quality["median_coverage_ge_0p95"] = (
        quality["median_3p0_coverage"] >= 0.95
    )
    quality["any_zero_coverage"] = quality["zero_coverage_fraction"] > 0
    quality["multi_series"] = quality["series_count"] > 1

    inputs = {
        "Train": (
            reports / "train_oof_predictions.csv",
            "whole_probability",
            "pred_roi_probability",
        ),
        "Valid": (
            reports / "valid_predictions.csv",
            "whole_probability",
            "pred_roi_probability",
        ),
    }
    patient_frames = []
    summary: dict[str, object] = {
        "version": "api_adverse_lesion_cave_fast_v1_patient_subgroups_1"
    }
    for split, (path, whole_column, roi_column) in inputs.items():
        prediction = pd.read_csv(path, dtype={"patient_id": str})
        combined = prediction.merge(
            quality[quality["split"] == split],
            on="patient_id",
            how="inner",
            validate="one_to_one",
        )
        patient_frames.append(combined.assign(split=split))
        subgroups = {
            "all": np.ones(len(combined), dtype=bool),
            "all_phases_ge_0p95": combined["all_phases_ge_0p95"],
            "not_all_phases_ge_0p95": ~combined["all_phases_ge_0p95"],
            "median_coverage_ge_0p95": combined[
                "median_coverage_ge_0p95"
            ],
            "median_coverage_lt_0p95": ~combined[
                "median_coverage_ge_0p95"
            ],
            "no_zero_coverage": ~combined["any_zero_coverage"],
            "any_zero_coverage": combined["any_zero_coverage"],
            "single_series": ~combined["multi_series"],
            "multi_series": combined["multi_series"],
        }
        split_summary = {
            name: metrics(
                combined[np.asarray(mask, dtype=bool)],
                whole_column,
                roi_column,
            )
            for name, mask in subgroups.items()
        }
        split_summary["coverage_by_target"] = {
            str(int(target)): {
                "patients": int(len(group)),
                "median_3p0_coverage_mean": float(
                    group["median_3p0_coverage"].mean()
                ),
                "minimum_3p0_coverage_mean": float(
                    group["minimum_3p0_coverage"].mean()
                ),
                "any_zero_coverage_fraction": float(
                    group["any_zero_coverage"].mean()
                ),
                "multi_series_fraction": float(group["multi_series"].mean()),
                "series_count_mean": float(group["series_count"].mean()),
            }
            for target, group in combined.groupby("target")
        }
        summary[split] = split_summary

    patient = pd.concat(patient_frames, ignore_index=True)
    patient_path = reports / "failure_mode_patient_audit.csv"
    summary_path = reports / "failure_mode_patient_subgroups.json"
    patient.to_csv(patient_path, index=False)
    summary["outputs"] = {
        "patient_audit": str(patient_path),
        "phase_audit": str(reports / "failure_mode_phase_audit.csv"),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Summarize the fixed strict ROI 2D/oracle experiments without retraining."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common import atomic_csv, atomic_json


ROOT = Path("/root/autodl-tmp/aneurysm")
REPORT = ROOT / "reports/api_png2d_segresnet_cave_fusion_v5_strict_2d_roi_oracles"


def read_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def row(name: str, metrics: dict) -> dict:
    return {
        "model": name,
        "OOF_AUROC": metrics["train_oof"]["AUROC"],
        "OOF_AUPRC": metrics["train_oof"]["AUPRC"],
        "OOF_Brier": metrics["train_oof"]["Brier"],
        "Valid_AUROC": metrics["valid"]["AUROC"],
        "Valid_AUPRC": metrics["valid"]["AUPRC"],
        "Valid_Brier": metrics["valid"]["Brier"],
    }


def verify_new_output(root: Path, mode: str) -> dict:
    success = json.loads((root / ".SUCCESS.json").read_text(encoding="utf-8"))
    assert success["status"] == "success"
    base = root / "fusion" / mode
    oof = pd.read_csv(base / "train_oof_predictions.csv")
    valid = pd.read_csv(base / "valid_predictions.csv")
    folds = pd.read_csv(base / "fold_metrics.csv")
    assert len(oof) == 781 and oof["series_uid"].astype(str).nunique() == 781
    assert len(valid) == 207 and valid["series_uid"].astype(str).nunique() == 207
    assert oof["probability"].notna().all() and valid["probability"].notna().all()
    assert set(folds["fold"].astype(int)) == {1, 2, 3, 4, 5}
    return read_metrics(base / "metrics.json")


def delta(a: dict, b: dict, label: str) -> dict:
    out = {"comparison": label}
    for split in ("OOF", "Valid"):
        for metric in ("AUROC", "AUPRC", "Brier"):
            out[f"delta_{split}_{metric}"] = a[f"{split}_{metric}"] - b[f"{split}_{metric}"]
    return out


def main() -> None:
    pred2d_root = ROOT / "outputs/api_png2d_segresnet_cave_fusion_v5_strict_predroi_2d_only"
    gt2d_root = ROOT / "outputs/api_png2d_segresnet_cave_fusion_v5_strict_gtroi_2d_only"
    gtcave_root = ROOT / "outputs/api_png2d_segresnet_cave_fusion_v5_strict_gtroi_cave_gated_oracle"

    pred2d = row("PredROI_2D_only", verify_new_output(pred2d_root, "spatial_only"))
    gt2d = row("GTROI_2D_only", verify_new_output(gt2d_root, "spatial_only"))
    gtcave = row("GTROI_CAVE_gated_oracle", verify_new_output(gtcave_root, "gated_interaction"))

    b0 = row(
        "Historical CAVE-Deep Logistic baseline (B0)",
        read_metrics(
            ROOT
            / "outputs/api_png2d_gtmask_roi_cave_v2_complete_mapping_fullseq"
            / "adverse_prepost_series_formal_models_v31/metrics.json"
        ),
    )
    c = row(
        "Global + CAVE Deep + gated (C)",
        read_metrics(
            ROOT
            / "outputs/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready_strict"
            / "fusion/gated_interaction/metrics.json"
        ),
    )
    a = row(
        "PredROI + CAVE Deep + gated (A primary)",
        read_metrics(
            ROOT
            / "outputs/api_png2d_segresnet_cave_fusion_v5_strict_cell_a_predroi_deep"
            / "fusion/gated_interaction/metrics.json"
        ),
    )
    logistic = row(
        "PredROI + CAVE Deep + Logistic",
        read_metrics(
            ROOT
            / "outputs/api_png2d_segresnet_cave_fusion_v5_strict_predroi_cave_logistic"
            / "fusion/spatial_only/metrics.json"
        ) if False else read_metrics(
            ROOT
            / "reports/api_png2d_segresnet_cave_fusion_v5_strict_predroi_cave_logistic/metrics.json"
        ),
    )

    rows = [b0, c, a, logistic, pred2d, gt2d, gtcave]
    metrics = pd.DataFrame(rows)
    comparisons = pd.DataFrame([
        delta(gt2d, pred2d, "GTROI_2D_only minus PredROI_2D_only"),
        delta(gtcave, a, "GTROI_CAVE_gated_oracle minus PredROI_CAVE_gated_primary"),
        delta(a, pred2d, "PredROI_CAVE_gated_primary minus PredROI_2D_only"),
        delta(gtcave, gt2d, "GTROI_CAVE_gated_oracle minus GTROI_2D_only"),
    ])

    REPORT.mkdir(parents=True, exist_ok=True)
    atomic_csv(metrics, REPORT / "strict_roi_oracle_model_comparison.csv")
    atomic_csv(comparisons, REPORT / "strict_roi_oracle_requested_comparisons.csv")
    atomic_json(
        {
            "status": "success",
            "new_experiments": ["PredROI_2D_only", "GTROI_2D_only", "GTROI_CAVE_gated_oracle"],
            "strict_oof_rows": 781,
            "valid_rows": 207,
            "models": rows,
            "comparisons": comparisons.to_dict("records"),
            "valid_used_for_selection": False,
            "segresnet_retrained": False,
            "spatial_features_reextracted": False,
            "cave_rerun": False,
        },
        REPORT / "strict_roi_oracle_summary.json",
    )

    lines = [
        "# Strict ROI 2D and GTROI Oracle Experiments",
        "",
        "## Scope",
        "",
        "Only the requested three independent experiments were trained. All reuse the existing strict five-fold SegResNet featurebank and fixed CAVE deep NPZ; no SegResNet retraining, spatial re-extraction, CAVE rerun, cohort/fold change, or gate-necessity comparison was performed.",
        "",
        "- `PredROI_2D_only`: fold-k `pred_combined` (1024-D) -> Linear(1024,256) -> LayerNorm -> GELU -> Dropout(0.2) -> main head.",
        "- `GTROI_2D_only`: fold-k `gt_combined` (1024-D) -> same 2D-only head. This is a spatial oracle because GT masks are used in feature extraction.",
        "- `GTROI_CAVE_gated_oracle`: fold-k `gt_combined` (1024-D) plus CAVE deep (10240-D), using the frozen bidirectional gated four-way-interaction teacher architecture.",
        "",
        "Each new run has a success marker and exactly 781 unique OOF series, 207 unique Valid series, and all five outer folds.",
        "",
        "## Requested Seven-Model Table",
        "",
        metrics.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Requested Comparisons (first model minus second model)",
        "",
        comparisons.to_markdown(index=False, floatfmt="+.6f"),
        "",
        "## Interpretation Boundary",
        "",
        "All representation and primary-model selection remains based exclusively on strict Train OOF. Valid values are descriptive only. GTROI models are oracle analyses and cannot be used as deployable inference models because their spatial features require original GT masks.",
        "",
        "The teacher gated structure remains fixed; this report does not compare or test whether gating is necessary.",
    ]
    (REPORT / "STRICT_ROI_2D_AND_GTORI_ORACLE_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import atomic_csv, atomic_json


CELLS = {
    "C_existing_global_deep": {
        "spatial": "Global",
        "temporal": "Deep",
        "root": "/root/autodl-tmp/aneurysm/outputs/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready_strict",
    },
    "A_new_predroi_deep": {
        "spatial": "Global+PredROI",
        "temporal": "Deep",
        "root": "/root/autodl-tmp/aneurysm/outputs/api_png2d_segresnet_cave_fusion_v5_strict_cell_a_predroi_deep",
    },
    "B_new_global_deepscalar": {
        "spatial": "Global",
        "temporal": "Deep+Scalar",
        "root": "/root/autodl-tmp/aneurysm/outputs/api_png2d_segresnet_cave_fusion_v5_strict_cell_b_global_deepscalar",
    },
    "D_existing_main_predroi_deepscalar": {
        "spatial": "Global+PredROI",
        "temporal": "Deep+Scalar",
        "root": "/root/autodl-tmp/aneurysm/outputs/api_png2d_segresnet_cave_fusion_v5_strict_main_predroi_deepscalar",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-root", required=True)
    args = ap.parse_args()
    report = Path(args.report_root).resolve()

    rows = []
    fold_rows = []
    for cell, spec in CELLS.items():
        root = Path(spec["root"])
        mode_root = root / "fusion" / "gated_interaction"
        metrics = json.loads((mode_root / "metrics.json").read_text(encoding="utf-8"))
        row = {
            "cell": cell,
            "spatial": spec["spatial"],
            "temporal": spec["temporal"],
            "spatial_representation": metrics["spatial_representation"],
            "temporal_representation": metrics.get("temporal_representation", "deep_only"),
            "OOF_AUROC": metrics["train_oof"]["AUROC"],
            "OOF_AUPRC": metrics["train_oof"]["AUPRC"],
            "OOF_Brier": metrics["train_oof"]["Brier"],
            "Valid_AUROC": metrics["valid"]["AUROC"],
            "Valid_AUPRC": metrics["valid"]["AUPRC"],
            "Valid_Brier": metrics["valid"]["Brier"],
        }
        rows.append(row)
        folds = pd.read_csv(mode_root / "fold_metrics.csv", encoding="utf-8-sig")
        folds.insert(0, "cell", cell)
        folds.insert(1, "spatial", spec["spatial"])
        folds.insert(2, "temporal", spec["temporal"])
        fold_rows.append(folds)

    table = pd.DataFrame(rows)
    by_cell = table.set_index("cell")
    effects = []
    for metric in ("OOF_AUROC", "OOF_AUPRC", "OOF_Brier"):
        c = float(by_cell.loc["C_existing_global_deep", metric])
        a = float(by_cell.loc["A_new_predroi_deep", metric])
        b = float(by_cell.loc["B_new_global_deepscalar", metric])
        d = float(by_cell.loc["D_existing_main_predroi_deepscalar", metric])
        effects.append({
            "metric": metric,
            "roi_main_effect": ((a + d) - (c + b)) / 2.0,
            "scalar_main_effect": ((b + d) - (c + a)) / 2.0,
            "roi_x_scalar_interaction": d - a - b + c,
        })
    effects_df = pd.DataFrame(effects)
    folds_df = pd.concat(fold_rows, ignore_index=True)

    atomic_csv(table, report / "06_strict_factorial_2x2_metrics.csv")
    atomic_csv(folds_df, report / "06_strict_factorial_2x2_fold_metrics.csv")
    atomic_csv(effects_df, report / "06_strict_factorial_2x2_oof_effects.csv")
    atomic_json({
        "status": "success",
        "selection_basis": "strict_train_oof_only",
        "valid_used_for_selection": False,
        "model": "gated_interaction_only",
        "cells": table.to_dict("records"),
        "oof_effects": effects_df.to_dict("records"),
    }, report / "06_strict_factorial_2x2.json")

    text = [
        "# Strict 2x2 Factorial: Global/PredROI x Deep/Deep+Scalar",
        "",
        "All cells use strict folds, gated_interaction only, raw-to-256 projections, bidirectional gating, four-way interaction, 1024->512->256 fusion, and the same main head.",
        "",
        "All main-effect and interaction quantities below use pooled strict Train OOF only. Valid is reported but was not used for selection.",
        "",
        "## Cell Metrics",
        "",
        table.to_markdown(index=False),
        "",
        "## Strict OOF Effects",
        "",
        effects_df.to_markdown(index=False),
        "",
        "Definitions: ROI main effect = mean(A,D) - mean(C,B); scalar main effect = mean(B,D) - mean(C,A); interaction = D - A - B + C.",
        "",
        "## Per-Fold Strict OOF",
        "",
        folds_df.to_markdown(index=False),
        "",
    ]
    (report / "06_strict_factorial_2x2.md").write_text("\n".join(text), encoding="utf-8")


if __name__ == "__main__":
    main()

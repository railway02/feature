#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common import atomic_csv


ROOT = Path("/root/autodl-tmp/aneurysm")
OUT = ROOT / "outputs/api_png2d_segresnet_cave_fusion_v5_strict_predroi_cave_logistic"
REP = ROOT / "reports/api_png2d_segresnet_cave_fusion_v5_strict_predroi_cave_logistic"


def main():
    logistic = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
    primary = json.loads((ROOT / "outputs/api_png2d_segresnet_cave_fusion_v5_strict_cell_a_predroi_deep/fusion/gated_interaction/metrics.json").read_text(encoding="utf-8"))
    old = pd.read_csv(ROOT / "outputs/api_png2d_gtmask_roi_cave_v2_complete_mapping_fullseq/adverse_prepost_series_formal_models_v31/metrics.csv", encoding="utf-8-sig")
    b0_oof = old[(old["model"] == "Logistic_Deep") & (old["split"] == "Train_OOF")].iloc[0]
    b0_valid = old[(old["model"] == "Logistic_Deep") & (old["split"] == "Valid")].iloc[0]
    rows = [
        {"model": "Historical CAVE-Deep Logistic (B0)", "OOF_AUROC": b0_oof.AUROC, "OOF_AUPRC": b0_oof.AUPRC, "OOF_Brier": b0_oof.Brier, "Valid_AUROC": b0_valid.AUROC, "Valid_AUPRC": b0_valid.AUPRC, "Valid_Brier": b0_valid.Brier},
        {"model": "PredROI-CAVE-Deep Gated Fusion (A primary)", "OOF_AUROC": primary["train_oof"]["AUROC"], "OOF_AUPRC": primary["train_oof"]["AUPRC"], "OOF_Brier": primary["train_oof"]["Brier"], "Valid_AUROC": primary["valid"]["AUROC"], "Valid_AUPRC": primary["valid"]["AUPRC"], "Valid_Brier": primary["valid"]["Brier"]},
        {"model": "PredROI_CAVE_Logistic", "OOF_AUROC": logistic["train_oof"]["AUROC"], "OOF_AUPRC": logistic["train_oof"]["AUPRC"], "OOF_Brier": logistic["train_oof"]["Brier"], "Valid_AUROC": logistic["valid"]["AUROC"], "Valid_AUPRC": logistic["valid"]["AUPRC"], "Valid_Brier": logistic["valid"]["Brier"]},
    ]
    table = pd.DataFrame(rows)
    atomic_csv(table, REP / "three_model_comparison.csv")
    text = [
        "# PredROI_CAVE_Logistic Strict Comparison",
        "",
        "`PredROI_CAVE_Logistic` uses concat(pred_combined[1024], CAVE_deep[10240]) = 11264-D.",
        "It reuses the formal v31 Logistic core: grouped outer folds, 3-fold Train-only inner CV, PCA dimensions 32/64/128, C values 0.01/0.1/1/10, inner pooled AUPRC selection, and five-fold Valid averaging.",
        "",
        table.to_markdown(index=False),
        "",
        "Valid was not used for parameter selection. This Logistic result is an additional classifier reference; it does not modify the frozen teacher gated primary model.",
        "",
    ]
    (REP / "three_model_comparison.md").write_text("\n".join(text), encoding="utf-8")


if __name__ == "__main__":
    main()

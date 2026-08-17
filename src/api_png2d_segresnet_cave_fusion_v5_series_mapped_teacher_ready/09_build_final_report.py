#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common import atomic_csv


ROOT = Path("/root/autodl-tmp/aneurysm")
FINAL_DIR = ROOT / "reports/api_png2d_segresnet_cave_fusion_v5_final_strict_predroi_cave_deep_gated"
FINAL_MD = ROOT / "reports/FINAL_STRICT_PREDROI_CAVE_DEEP_GATED_REPORT_ZH.md"


def metrics(root: Path):
    return json.loads((root / "fusion/gated_interaction/metrics.json").read_text(encoding="utf-8"))


def metric_row(name, m):
    return {
        "model": name,
        "OOF_AUROC": m["train_oof"]["AUROC"],
        "OOF_AUPRC": m["train_oof"]["AUPRC"],
        "OOF_Brier": m["train_oof"]["Brier"],
        "Valid_AUROC": m["valid"]["AUROC"],
        "Valid_AUPRC": m["valid"]["AUPRC"],
        "Valid_Brier": m["valid"]["Brier"],
    }


def main():
    a_root = ROOT / "outputs/api_png2d_segresnet_cave_fusion_v5_strict_cell_a_predroi_deep"
    c_root = ROOT / "outputs/api_png2d_segresnet_cave_fusion_v5_series_mapped_teacher_ready_strict"
    b1_root = c_root
    seed_roots = {
        20260818: a_root,
        20260819: ROOT / "outputs/api_png2d_segresnet_cave_fusion_v5_strict_primary_predroi_deep_seed_20260819",
        20260820: ROOT / "outputs/api_png2d_segresnet_cave_fusion_v5_strict_primary_predroi_deep_seed_20260820",
    }
    a = metrics(a_root)
    c = metrics(c_root)
    b1 = json.loads((b1_root / "fusion/cave_only/metrics.json").read_text(encoding="utf-8"))
    old = pd.read_csv(ROOT / "outputs/api_png2d_gtmask_roi_cave_v2_complete_mapping_fullseq/adverse_prepost_series_formal_models_v31/metrics.csv", encoding="utf-8-sig")
    b0_oof = old[(old["model"] == "Logistic_Deep") & (old["split"] == "Train_OOF")].iloc[0]
    b0_valid = old[(old["model"] == "Logistic_Deep") & (old["split"] == "Valid")].iloc[0]

    stability_rows = []
    for seed, root in seed_roots.items():
        m = metrics(root)
        stability_rows.append({"seed": seed, **metric_row("PredROI-CAVE-Deep Gated Fusion", m)})
    stability = pd.DataFrame(stability_rows)
    stability_metrics = ["OOF_AUROC", "OOF_AUPRC", "OOF_Brier", "Valid_AUROC", "Valid_AUPRC", "Valid_Brier"]
    summary = stability[stability_metrics].agg(["mean", "std"]).T.reset_index().rename(columns={"index": "metric"})
    atomic_csv(stability, FINAL_DIR / "09_primary_seed_stability.csv")
    atomic_csv(summary, FINAL_DIR / "09_primary_seed_stability_summary.csv")

    primary = metric_row("A Primary: PredROI-CAVE-Deep Gated Fusion", a)
    c_row = metric_row("C: Global+Deep Gated", c)
    b1_row = {
        "model": "B1: CAVE-Deep Teacher Neural Head",
        "OOF_AUROC": b1["train_oof"]["AUROC"], "OOF_AUPRC": b1["train_oof"]["AUPRC"], "OOF_Brier": b1["train_oof"]["Brier"],
        "Valid_AUROC": b1["valid"]["AUROC"], "Valid_AUPRC": b1["valid"]["AUPRC"], "Valid_Brier": b1["valid"]["Brier"],
    }
    b0_row = {
        "model": "B0: Historical CAVE-Deep Logistic",
        "OOF_AUROC": float(b0_oof["AUROC"]), "OOF_AUPRC": float(b0_oof["AUPRC"]), "OOF_Brier": float(b0_oof["Brier"]),
        "Valid_AUROC": float(b0_valid["AUROC"]), "Valid_AUPRC": float(b0_valid["AUPRC"]), "Valid_Brier": float(b0_valid["Brier"]),
    }
    comparison = pd.DataFrame([primary, b0_row, b1_row, c_row])
    a_minus_c = {
        "OOF_AUROC": primary["OOF_AUROC"] - c_row["OOF_AUROC"],
        "OOF_AUPRC": primary["OOF_AUPRC"] - c_row["OOF_AUPRC"],
        "OOF_Brier": primary["OOF_Brier"] - c_row["OOF_Brier"],
    }
    fold_a = pd.read_csv(a_root / "fusion/gated_interaction/fold_metrics.csv", encoding="utf-8-sig")
    fold_c = pd.read_csv(c_root / "fusion/gated_interaction/fold_metrics.csv", encoding="utf-8-sig")
    improved = int((fold_a.set_index("fold")["OOF_AUPRC"] > fold_c.set_index("fold")["OOF_AUPRC"]).sum())

    bootstrap = pd.read_csv(FINAL_DIR / "07_paired_patient_bootstrap.csv", encoding="utf-8-sig")
    self_ci = pd.read_csv(FINAL_DIR / "07_primary_patient_bootstrap_ci.csv", encoding="utf-8-sig")
    factorial = pd.read_csv(ROOT / "reports/api_png2d_segresnet_cave_fusion_v5_strict_factorial_predroi_scalar/06_strict_factorial_2x2_metrics.csv", encoding="utf-8-sig")
    effects = pd.read_csv(ROOT / "reports/api_png2d_segresnet_cave_fusion_v5_strict_factorial_predroi_scalar/06_strict_factorial_2x2_oof_effects.csv", encoding="utf-8-sig")
    seg = pd.read_csv(ROOT / "reports/api_png2d_segresnet_cave_fusion_v5_strict_factorial_predroi_scalar/05_strict_outer_segmentation_metrics.csv", encoding="utf-8-sig")
    seg_overall = seg[seg["phase"] == "Overall"]
    gate = json.loads((FINAL_DIR / "08_primary_gate_audit.json").read_text(encoding="utf-8"))
    gate_group = pd.read_csv(FINAL_DIR / "08_primary_gate_case_mean_by_target.csv", encoding="utf-8-sig")

    text = [
        "# 最终严格 PredROI-CAVE-Deep Gated Fusion 报告",
        "",
        "## 1. 最终方法",
        "",
        "最终主模型：**PredROI-CAVE-Deep Gated Fusion**（亦称 SegResNet ROI-CAVE Gated Fusion）。",
        "",
        "```text",
        "z2D_raw = pred_combined = [G_pre, ROIpred_pre, G_post, ROIpred_post] = 1024-D",
        "zT_raw  = CAVE deep = [Pre 5120, Post 5120] = 10240-D",
        "z2D_raw -> Linear(1024,256) -> LayerNorm -> GELU -> Dropout(0.2)",
        "zT_raw  -> Linear(10240,256) -> LayerNorm -> GELU -> Dropout(0.2)",
        "bidirectional gates -> four-way interaction -> 1024->512->256 fusion -> main head",
        "```",
        "",
        "主模型不包含 scalar、PCA、GT ROI、形态学、3D、临床特征或额外 fusion mode。",
        "",
        "## 2. 数据与泄漏控制",
        "",
        "- Train：781 个 series，133 阳性；Valid：207 个 series，37 阳性。",
        "- 严格五折患者分组；series_uid 是预测主键。",
        "- 每个 strict fold 的 SegResNet 未使用其 outer holdout，也未使用 Valid 的图像、Mask 或 outcome。",
        "- 主融合仅使用相应 fold 的 pred_combined featurebank 与 CAVE deep。",
        "- Valid 从未用于架构、epoch、表示或 seed 选择。",
        "",
        "Primary architecture selection was based exclusively on Train strict OOF performance; OOF bootstrap intervals are conditional on the selected architecture and do not account for architecture-selection uncertainty. The independent Valid cohort was not used for architecture or epoch selection.",
        "",
        "## 3. Outer Segmentation 审计",
        "",
        "复用既有 strict outer-holdout 审计；该审计仅用于评估。",
        "",
        seg_overall.to_markdown(index=False),
        "",
        "五个 Overall 的 empty_pred_rate 均为 0。完整 Pre/Post/Overall 结果保存在 `05_strict_outer_segmentation_metrics.csv`。",
        "",
        "## 4. 主结果",
        "",
        comparison.to_markdown(index=False),
        "",
        "A 相对 C 的 strict Train OOF：" + ", ".join(f"{k}={v:+.6f}" for k, v in a_minus_c.items()) + "。",
        f"A 相对 C 在 {improved}/5 个 strict outer folds 的 AUPRC 均提高。",
        "",
        "## 5. 配对患者 Bootstrap",
        "",
        "5,000 次按患者聚类、有放回 bootstrap。每次抽中的患者均保留其全部 series，重复抽中时保留其重复次数；未计算 p-value。",
        "",
        bootstrap.to_markdown(index=False),
        "",
        "主模型 A 自身置信区间：",
        "",
        self_ci.to_markdown(index=False),
        "",
        "## 6. Scalar 2×2 消融",
        "",
        factorial.to_markdown(index=False),
        "",
        effects.to_markdown(index=False),
        "",
        "最终 temporal 表示为 Deep-only。2×2 结果仅作为消融报告；scalar descriptors 不属于最终主模型。",
        "",
        "## 7. 三 Seed 稳定性",
        "",
        "Seed 20260818 是保留的 primary single-run。20260819 与 20260820 是仅 fusion 的独立运行，fold、featurebank、架构、batch size、AMP 与超参数完全一致；预测未平均成新的主模型。",
        "",
        stability.to_markdown(index=False),
        "",
        summary.to_markdown(index=False),
        "",
        "## 8. 完整 OOF Gate 审计",
        "",
        "审计加载 A 的原始 config 与每折 checkpoint，并以 eval/no_grad 模式仅 forward 该 checkpoint 未见过的 outer holdout。保存的 `a2D` 与 `aT` 均为 [781,256]，每个唯一 series_uid 恰一行。",
        "",
        pd.DataFrame([{"gate": "a2D", **gate["a2D"]}, {"gate": "aT", **gate["aT"]}]).to_markdown(index=False),
        "",
        "熵计算前将 gate clip 至 [1e-7, 1-1e-7]。按 target 的 case-level gate-mean 分布保存在 `08_primary_gate_case_mean_by_target.csv`。",
        "",
        gate_group.to_markdown(index=False),
        "",
        "## 9. 最终结论",
        "",
        "在当前 strict cross-fit adverse cohort 中，在保持 CAVE deep temporal representation 和 teacher-aligned gated fusion 结构不变的条件下，引入 SegResNet probability-mask-guided ROI spatial representation 后，相较 global-only spatial representation，Train OOF AUPRC 在全部五个 outer folds 中均提高；独立 Valid 亦观察到同方向变化。",
        "",
        "在当前 gated 结构中，engineered CAVE scalar descriptors 未进一步改善 ROI-aware 模型，并在 strict 2×2 消融中观察到负 ROI×scalar interaction；因此 scalar 不进入最终主模型。",
        "",
    ]
    FINAL_MD.write_text("\n".join(text), encoding="utf-8")


if __name__ == "__main__":
    main()

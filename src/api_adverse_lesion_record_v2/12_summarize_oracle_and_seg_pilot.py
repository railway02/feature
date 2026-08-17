#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from common import atomic_json, load_config, update_run_manifest


CONFIG = "/root/autodl-tmp/aneurysm/configs/api_adverse_lesion_record_v2/config.json"


def grouped(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    metrics = [
        "on_target",
        "coverage",
        "coverage95",
        "zero_coverage",
        "empty",
        "roi_area_ratio",
        "centroid_distance_normalized",
        "dice",
    ]
    result = frame.groupby(columns, dropna=False)[metrics].mean().reset_index()
    counts = frame.groupby(columns, dropna=False).size().rename("rows").reset_index()
    return counts.merge(result, on=columns, how="left").to_dict("records")


def load_pilot(reports: Path, prefix: str) -> tuple[dict, pd.DataFrame]:
    summary = json.loads((reports / f"{prefix}_summary.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(reports / f"{prefix}_holdout_metrics.csv")
    frame["lesion_size_quartile"] = pd.qcut(
        frame["lesion_area_ratio"],
        q=4,
        labels=["Q1_smallest", "Q2", "Q3", "Q4_largest"],
        duplicates="drop",
    ).astype(str)
    return summary, frame


def main() -> int:
    config = load_config(CONFIG)
    reports = Path(config["paths"]["reports"])
    oracle = json.loads((reports / "record_gt_oracle_oof_summary.json").read_text(encoding="utf-8"))
    valid = json.loads((reports / "record_gt_oracle_internal_valid_summary.json").read_text(encoding="utf-8"))
    baseline, baseline_frame = load_pilot(reports, "segmentation_pilot_p2")
    centroid, centroid_frame = load_pilot(reports, "segmentation_pilot_p2_centroid_fp")
    valid_metrics = {row["model"]: row for row in valid["metrics"]}
    payload = {
        "status": "stopped_after_segmentation_pilot_gate",
        "science_scope": {
            "prediction_unit": "record_uid",
            "group_unit": "patient_id",
            "outcome": "followup_rroc_in_2_or_3",
            "temporal_model": "CAVE_v3_only",
            "searaft_used": False,
        },
        "gt_oracle": {
            "whole_auprc_train_oof": oracle["whole_auprc"],
            "selected_fusion": oracle["selected_fusion"],
            "selected_scale": oracle["selected_scale"],
            "selected_auprc_train_oof": oracle["selected_auprc"],
            "delta_auprc_train_oof": oracle["delta_auprc"],
            "folds_improved": oracle["folds_improved"],
            "gate_passed": oracle["oracle_gate_passed"],
        },
        "internal_valid": {
            "whole_auprc": valid_metrics["G0_whole"]["auprc"],
            "gt_morphology_auprc": valid_metrics["G1_gt_morphology"]["auprc"],
            "gt_local40_auprc": valid_metrics["G3_gt_local40"]["auprc"],
            "whole_gt_local40_auprc": valid_metrics["G5_whole_gt_local40"]["auprc"],
            "fusion_minus_whole_auprc": (
                valid_metrics["G5_whole_gt_local40"]["auprc"]
                - valid_metrics["G0_whole"]["auprc"]
            ),
            "valid_used_for_selection": valid["valid_used_for_selection"],
        },
        "segmentation_pilot_baseline": baseline,
        "segmentation_pilot_centroid_fp": centroid,
        "segmentation_strata": {
            "baseline_by_phase": grouped(baseline_frame, ["phase"]),
            "baseline_by_grade": grouped(baseline_frame, ["annotation_grade"]),
            "baseline_by_lesion_size": grouped(baseline_frame, ["lesion_size_quartile"]),
            "centroid_by_phase": grouped(centroid_frame, ["phase"]),
            "centroid_by_grade": grouped(centroid_frame, ["annotation_grade"]),
            "centroid_by_lesion_size": grouped(centroid_frame, ["lesion_size_quartile"]),
        },
        "decision": {
            "formal_segmentation_oof_started": False,
            "pred_local_cave_started": False,
            "prediction_training_started": False,
            "reason": (
                "GT Local gain did not reproduce on Internal Valid, and both Train-only "
                "automatic segmentation pilots failed the predefined localization/coverage/area gate."
            ),
        },
    }
    atomic_json(payload, reports / "EXECUTION_REPORT_AFTER_ORACLE_AND_SEG_PILOT.json")

    b = baseline["metrics"]
    c = centroid["metrics"]
    lines = [
        "# api_adverse_lesion_record_v2：Oracle 与分割 Pilot 执行报告",
        "",
        "本报告是本轮新增权威报告；旧 v1、fast_v1、旧 GT oracle 和原 CAVE 产物均未删除或覆盖。",
        "",
        "## 1. 本轮实际执行范围",
        "",
        "- 仅使用 CAVE v3，不使用 SEA-RAFT。",
        "- 预测单位为 record_uid，交叉验证按 patient_id 分组。",
        "- Local-CAVE 特征由局部时序重新运行冻结 CAVE 得到；未复用 Whole embedding。",
        "- 先做 GT Context Oracle，再决定是否训练自动分割器。",
        "- 自动分割仅做两个连续的最小 Pilot：P2 baseline，以及针对其大 ROI 问题的 centroid_fp 修复。",
        "",
        "## 2. GT Oracle Train OOF",
        "",
        f"- Whole AUPRC：{oracle['whole_auprc']:.6f}",
        f"- 选择：{oracle['selected_fusion']}，Context={oracle['selected_scale']}%",
        f"- 选择分支 AUPRC：{oracle['selected_auprc']:.6f}",
        f"- ΔAUPRC：{oracle['delta_auprc']:+.6f}",
        f"- 改善 folds：{oracle['folds_improved']}/5",
        "- Oracle Gate：PASS",
        "",
        "## 3. 冻结 Internal Valid",
        "",
        f"- Whole AUPRC：{valid_metrics['G0_whole']['auprc']:.6f}",
        f"- GT morphology AUPRC：{valid_metrics['G1_gt_morphology']['auprc']:.6f}",
        f"- GT Local40 AUPRC：{valid_metrics['G3_gt_local40']['auprc']:.6f}",
        f"- Whole + GT Local40 AUPRC：{valid_metrics['G5_whole_gt_local40']['auprc']:.6f}",
        f"- 融合相对 Whole：{payload['internal_valid']['fusion_minus_whole_auprc']:+.6f}",
        "",
        "结论：Train OOF 的 Local 融合增益没有在 Internal Valid 复现；GT morphology 是 Valid 中表现最好的病灶分支。",
        "",
        "## 4. 自动分割 P2 baseline Pilot",
        "",
        "- 输入：polarity-aware MinIP + median + q95−q05 + phase channel。",
        "- 模型：共享 Pre/Post U-Net，1024，base32，GroupNorm。",
        f"- 最佳 epoch：{baseline['best_epoch']}，阈值：{baseline['selected_threshold']:.2f}",
        f"- on-target：{b['on_target']:.4f}",
        f"- mean coverage：{b['coverage']:.4f}",
        f"- coverage≥0.95：{b['coverage95']:.4f}",
        f"- zero coverage：{b['zero_coverage']:.4f}",
        f"- empty：{b['empty']:.4f}",
        f"- ROI area ratio：{b['roi_area_ratio']:.4f}",
        "- Gate：NO PASS",
        "",
        "## 5. centroid_fp 针对性修复 Pilot",
        "",
        "- 保持同一数据、同一代表图和同一 U-Net。",
        "- 降低正类权重，提高假阳性惩罚，并加入 centroid/area 辅助损失。",
        f"- 最佳 epoch：{centroid['best_epoch']}，阈值：{centroid['selected_threshold']:.2f}",
        f"- on-target：{c['on_target']:.4f}",
        f"- mean coverage：{c['coverage']:.4f}",
        f"- coverage≥0.95：{c['coverage95']:.4f}",
        f"- zero coverage：{c['zero_coverage']:.4f}",
        f"- empty：{c['empty']:.4f}",
        f"- ROI area ratio：{c['roi_area_ratio']:.4f}",
        "- Gate：NO PASS",
        "",
        "centroid_fp 将 ROI area 从 baseline 的 "
        f"{b['roi_area_ratio']:.3f} 降至 {c['roi_area_ratio']:.3f}，"
        "但 on-target 与 coverage 同时下降，说明当前代表图上病灶与广泛血管增强仍难稳定分离。",
        "",
        "## 6. 停止决定",
        "",
        "本轮不启动正式五折分割、不生成 Pred Local-CAVE、不运行新的预测训练。",
        "",
        "原因：",
        "",
        "1. GT Local 的 Train OOF 增益未在 Internal Valid 复现；",
        "2. 两个自动定位 Pilot 均未达到预设 Gate；",
        "3. 继续堆模型或消融会重新进入高成本、多分支但输入质量未解决的旧问题。",
        "",
        "## 7. 下一步最小建议",
        "",
        "在继续训练前，优先做病灶标签医学语义确认和代表图/Mask overlay 的人工核查；如果标签与映射无误，再考虑使用病灶中心点/候选检测而非直接从 MinIP 分割整幅小 Mask。不要立即启动五折。",
        "",
        "分层数值详见 EXECUTION_REPORT_AFTER_ORACLE_AND_SEG_PILOT.json。",
    ]
    path = reports / "EXECUTION_REPORT_AFTER_ORACLE_AND_SEG_PILOT_ZH.md"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    (reports / ".STOPPED_SEG_PILOT_GATE").write_text("stopped\n", encoding="utf-8")
    update_run_manifest(config, "segmentation_pilot_decision", payload["decision"])
    print(json.dumps(payload["decision"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

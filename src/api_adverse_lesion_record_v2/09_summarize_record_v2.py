#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from common import atomic_json, load_config, update_run_manifest


def main() -> int:
    config = load_config("/root/autodl-tmp/aneurysm/configs/api_adverse_lesion_record_v2/config.json")
    reports = Path(config["paths"]["reports"])
    mapping = json.loads((reports / "record_manifest_audit.json").read_text(encoding="utf-8"))
    temporal = json.loads((reports / "frozen_temporal_view_audit.json").read_text(encoding="utf-8"))
    oof = json.loads((reports / "record_gt_oracle_oof_summary.json").read_text(encoding="utf-8")) if (reports / "record_gt_oracle_oof_summary.json").is_file() else None
    valid = json.loads((reports / "record_gt_oracle_internal_valid_summary.json").read_text(encoding="utf-8")) if (reports / "record_gt_oracle_internal_valid_summary.json").is_file() else None
    payload = {"version": config["version"], "mapping": mapping, "temporal": temporal, "gt_oracle_oof": oof, "internal_valid": valid}
    atomic_json(payload, reports / "FINAL_SUMMARY.json")
    lines = [
        "# api_adverse_lesion_record_v2 权威运行总结", "",
        "本文件由运行产物自动生成；旧 v1/fast_v1/GT oracle 均未覆盖。", "",
        "## 记录级主队列", "",
        f"- Train：{mapping['primary_oracle']['Train']['records']} 条记录，{mapping['primary_oracle']['Train']['patients']} 名患者，{mapping['primary_oracle']['Train']['positive']} 阳性。",
        f"- Valid：{mapping['primary_oracle']['Valid']['records']} 条记录，{mapping['primary_oracle']['Valid']['patients']} 名患者，{mapping['primary_oracle']['Valid']['positive']} 阳性。",
        f"- 冻结 record×phase：{temporal['rows']}。", "",
        "## 特征提取原则", "",
        "- 仅使用 CAVE v3，不使用 SEA-RAFT。",
        "- GT Local 使用固定 30%/40% Context。",
        "- temporal views 与 Whole-CAVE 逐 phase 完全一致。",
        "- Whole 全图确定归一化、极性和 activity 后再裁剪。",
        "- 预测单位为 record_uid，CV 分组单位为 patient_id。", "",
    ]
    if oof:
        lines.extend([
            "## Train OOF GT Oracle", "",
            f"- Whole AUPRC：{oof['whole_auprc']:.6f}",
            f"- 最佳融合：{oof['selected_fusion']}",
            f"- 最佳融合 AUPRC：{oof['selected_auprc']:.6f}",
            f"- ΔAUPRC：{oof['delta_auprc']:.6f}",
            f"- 改善 folds：{oof['folds_improved']}/5",
            f"- Oracle Gate：{'PASS' if oof['oracle_gate_passed'] else 'NO GAIN'}", "",
        ])
    if valid:
        lines.extend(["## Internal Valid", "", f"- 冻结选择的 Context：{valid['selected_scale']}%", "", "详见 `record_gt_oracle_internal_valid_metrics.csv`。", ""])
    if oof and not oof["oracle_gate_passed"]:
        lines.extend(["## 自动停止结论", "", "Whole + GT Local 在 Train OOF 未达到预设增益，因此不启动分割 v2 五折训练。", ""])
    (reports / "FINAL_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    update_run_manifest(config, "summarize", {"status": "complete", "oracle_gate_passed": None if not oof else oof["oracle_gate_passed"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common import atomic_csv, atomic_json, load_config, resolve_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = resolve_path(cfg["output_root"], cfg["project_root"])
    rep = resolve_path(cfg["report_root"], cfg["project_root"])
    rep.mkdir(parents=True, exist_ok=True)

    rows = []
    fold_tables = {}

    for mode in cfg["fusion"]["modes"]:
        path = out / "fusion" / mode / "metrics.json"
        if not path.is_file():
            continue

        m = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "mode": mode,
            "strategy": m["strategy"],
            "spatial_representation": m.get("spatial_representation", "legacy_roi"),
            "representation_oof_status": m.get("representation_oof_status", "unknown"),
            "segmentation_population": m.get("segmentation_population", "unknown"),
            "valid_representation_status": m.get("valid_representation_status", "unknown"),
            "OOF_AUROC": m["train_oof"]["AUROC"],
            "OOF_AUPRC": m["train_oof"]["AUPRC"],
            "OOF_Brier": m["train_oof"]["Brier"],
            "Valid_AUROC": m["valid"]["AUROC"],
            "Valid_AUPRC": m["valid"]["AUPRC"],
            "Valid_Brier": m["valid"]["Brier"],
            "pilot_representation_warning": m["pilot_representation_warning"],
        })

        fold_path = out / "fusion" / mode / "fold_metrics.csv"
        if fold_path.is_file():
            fold_tables[mode] = pd.read_csv(fold_path)

    if not rows:
        raise RuntimeError("No fusion metrics found")

    df = pd.DataFrame(rows)
    base = df[df["mode"] == "cave_only"]

    if len(base) == 1:
        df["delta_OOF_AUPRC_vs_CAVE"] = (
            df["OOF_AUPRC"] - float(base.iloc[0]["OOF_AUPRC"])
        )
        df["delta_OOF_AUROC_vs_CAVE"] = (
            df["OOF_AUROC"] - float(base.iloc[0]["OOF_AUROC"])
        )
    else:
        df["delta_OOF_AUPRC_vs_CAVE"] = np.nan
        df["delta_OOF_AUROC_vs_CAVE"] = np.nan

    df["folds_AUPRC_improved_vs_CAVE"] = np.nan

    if "cave_only" in fold_tables:
        b = fold_tables["cave_only"].set_index("fold")["OOF_AUPRC"]
        for i, row in df.iterrows():
            mode = row["mode"]
            if mode in fold_tables:
                c = fold_tables[mode].set_index("fold")["OOF_AUPRC"]
                idx = b.index.intersection(c.index)
                df.loc[i, "folds_AUPRC_improved_vs_CAVE"] = int(
                    (c.loc[idx] > b.loc[idx]).sum()
                )

    gate = cfg["fusion"]["gate"]
    df["OOF_gate_pass"] = (
        (df["delta_OOF_AUPRC_vs_CAVE"] >= float(gate["delta_auprc_min"]))
        &
        (df["folds_AUPRC_improved_vs_CAVE"] >= int(gate["folds_improved_min"]))
    )
    df.loc[df["mode"] == "cave_only", "OOF_gate_pass"] = False

    atomic_csv(df, rep / "04_summary_metrics.csv")
    atomic_json(
        {
            "status": "success",
            "teacher_aligned_main_path": True,
            "uses_pca": False,
        "strategy": cfg["spatial"]["strategy"],
        "spatial_representation": cfg["spatial"].get("representation", "global_only"),
        "temporal_representation": cfg.get("temporal", {}).get("representation", "deep_only"),
            "results": df.to_dict("records"),
        },
        rep / "04_summary.json",
    )

    text = [
        "# SegResNet + CAVE v4（Teacher-Aligned）",
        "",
        "- 直接读取现成均值图 PNG + 对应 Mask PNG。",
        "- 不从 DSA 序列重新计算均值图。",
        f"- 2D representation: `{cfg['spatial'].get('representation', 'global_only')}`。",
        f"- CAVE representation: `{cfg.get('temporal', {}).get('representation', 'deep_only')}`。",
        "- 主融合严格使用 raw feature -> Linear(...,256) -> LayerNorm -> GELU -> Dropout。",
        "- 非 strict_crossfit 的 Train 结果标记为 pilot，不得称为严格 representation OOF。",
        "- 主融合不使用 PCA。",
        "",
        df.to_markdown(index=False),
        "",
    ]
    (rep / "04_summary.md").write_text("\n".join(text), encoding="utf-8")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

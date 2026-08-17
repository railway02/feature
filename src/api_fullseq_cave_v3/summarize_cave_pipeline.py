#!/usr/bin/env python3
"""Summarize CAVE feature extraction and prediction outputs."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def atomic_json(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(frame: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--table-root", required=True)
    parser.add_argument("--task-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--report-root", required=True)
    args = parser.parse_args()
    project = Path(args.project).resolve()
    feature_root = Path(args.feature_root).resolve()
    table_root = Path(args.table_root).resolve()
    task_root = Path(args.task_root).resolve()
    model_root = Path(args.model_root).resolve()
    report_root = Path(args.report_root).resolve()
    required = [
        report_root / "train_audit.json", report_root / "valid_audit.json",
        table_root / "train/build_audit.json", table_root / "valid/build_audit.json",
        task_root / ".TASKS_SUCCESS", model_root / ".MODELS_SUCCESS",
        model_root / "all_task_metrics.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing pipeline outputs:\n" + "\n".join(missing))
    metrics = pd.read_csv(model_root / "all_task_metrics.csv")
    oof = metrics[(metrics["split"] == "Train_OOF") & (metrics["model"] != "Dummy")].copy()
    best = oof.sort_values(["task", "auprc", "auroc"], ascending=[True, False, False]).groupby("task", as_index=False).first()
    selected_rows = []
    for _, row in best.iterrows():
        valid = metrics[(metrics["task"] == row["task"]) & (metrics["model"] == row["model"]) & (metrics["split"] == "Valid")]
        if len(valid) != 1:
            raise AssertionError(f"Missing Valid row for {row['task']} {row['model']}")
        v = valid.iloc[0]
        selected_rows.append({
            "task": row["task"], "selected_model_by_train_oof": row["model"],
            "train_oof_auroc": row["auroc"], "train_oof_auprc": row["auprc"], "train_oof_brier": row["brier"],
            "valid_auroc": v["auroc"], "valid_auprc": v["auprc"], "valid_brier": v["brier"],
            "valid_balanced_accuracy": v["balanced_accuracy"],
        })
    selected = pd.DataFrame(selected_rows)
    atomic_csv(selected, report_root / "selected_model_metrics.csv")

    sea_candidates = [
        project / "outputs/api_fullseq_v3_models/all_task_metrics.csv",
        project / "outputs/api_fullseq_v3_models/summary/all_task_metrics.csv",
        project / "outputs/api_fullseq_v3_models/full/all_task_metrics.csv",
    ]
    sea_path = next((path for path in sea_candidates if path.is_file()), None)
    comparison_path = None
    if sea_path is not None:
        sea = pd.read_csv(sea_path)
        cave_valid = metrics[metrics["split"] == "Valid"].copy()
        cave_valid.insert(0, "backbone", "CAVE")
        sea_valid = sea[sea["split"] == "Valid"].copy()
        sea_valid.insert(0, "backbone", "SEA-RAFT_v3")
        comparison = pd.concat([sea_valid, cave_valid], ignore_index=True, sort=False)
        comparison_path = report_root / "cave_vs_searaft_valid_metrics.csv"
        atomic_csv(comparison, comparison_path)

    payload = {
        "version": "api_fullseq_cave_v3_full_auto_models_1",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "feature_root": str(feature_root), "table_root": str(table_root),
        "task_root": str(task_root), "model_root": str(model_root),
        "selected_models": selected.to_dict("records"),
        "all_metrics": str(model_root / "all_task_metrics.csv"),
        "sea_raft_comparison": str(comparison_path) if comparison_path else None,
        "valid_used_for_training_or_model_selection": False,
    }
    atomic_json(payload, report_root / "full_auto_summary.json")
    atomic_json(payload, report_root / ".FULL_AUTO_WITH_MODELS_SUCCESS")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

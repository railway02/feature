#!/usr/bin/env python3
"""Summarize the formal strict-Pre+Post series-level adverse V3 run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    args = parser.parse_args()

    task_root = args.task_root.resolve()
    model_root = args.model_root.resolve()

    print("===== SERIES COHORT =====")
    task_path = task_root / "task_summary.json"
    if task_path.is_file():
        summary = json.loads(task_path.read_text(encoding="utf-8"))
        for split in ("train", "valid"):
            item = summary[split]
            print(
                f"{split.upper()}: "
                f"source_records={item['source_excel_records']} "
                f"included_series={item['included_series']} "
                f"patients={item['included_patients']} "
                f"positive_series={item['positive_series']} "
                f"negative_series={item['negative_series']}"
            )
            print(
                "  duplicate_same_label_records_collapsed:",
                item["duplicate_same_label_records_collapsed"],
            )
            print(
                "  excluded_conflicting_series:",
                item["excluded_conflicting_series"],
            )
            print(
                "  record exclusions:",
                item["record_exclusion_reasons"],
            )
            print(
                "  series exclusions:",
                item["series_exclusion_reasons"],
            )
    else:
        print("Task not built:", task_path)

    print("\n===== METRICS =====")
    metrics_path = model_root / "metrics.csv"
    if metrics_path.is_file():
        metrics = pd.read_csv(metrics_path)
        columns = [
            "split",
            "model",
            "rows",
            "patients",
            "positive",
            "AUROC",
            "AUPRC",
            "Brier",
            "Balanced Accuracy",
            "Sensitivity",
            "Specificity",
            "threshold_from_train_oof",
        ]
        print(metrics[columns].to_string(index=False))
    else:
        print("Models not complete:", metrics_path)

    print("\n===== TRAIN-OOF SELECTED MODEL =====")
    selected_path = model_root / "selected_model_by_train_oof.json"
    if selected_path.is_file():
        print(selected_path.read_text(encoding="utf-8"))
    else:
        print("Selection not complete:", selected_path)

    print("\n===== VALID PATIENT-CLUSTER 95% CI =====")
    ci_path = model_root / "valid_patient_cluster_bootstrap_ci.csv"
    if ci_path.is_file():
        ci = pd.read_csv(ci_path)
        print(
            ci[ci["metric"].isin(["AUROC", "AUPRC", "Brier"])]
            .to_string(index=False)
        )
    else:
        print("Bootstrap not complete:", ci_path)

    print("\n===== COMPLETED BASE MODEL FOLDS =====")
    fold_success = sorted(model_root.glob("folds/*/fold_*/.SUCCESS.json"))
    print(f"{len(fold_success)}/20")
    for path in fold_success:
        print(path)

    stack_path = model_root / "stacked_ensemble/stack_summary.json"
    print("\n===== STACKED ENSEMBLE =====")
    if stack_path.is_file():
        print(stack_path.read_text(encoding="utf-8"))
    else:
        print("not complete:", stack_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

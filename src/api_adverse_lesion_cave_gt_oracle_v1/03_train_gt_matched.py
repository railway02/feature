#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


VARIANTS = ("deep", "scalar", "fusion")


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def subset(data: dict[str, np.ndarray], indices: np.ndarray):
    return {key: value[indices] for key, value in data.items()}


def atomic_joblib(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    joblib.dump(value, temporary)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/root/autodl-tmp/aneurysm/configs/api_adverse_lesion_cave_fast_v1.json",
    )
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root = Path(config["project_root"])
    source_reports = Path(config["paths"]["reports"])
    reports = root / "reports/api_adverse_lesion_cave_gt_oracle_v1"
    outputs = root / "outputs/api_adverse_lesion_cave_gt_oracle_v1"
    reports.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)

    fixed = import_module(
        Path(config["fixed_trainer"]), "matched_variants_fixed"
    )
    builder = import_module(
        root / "code/api_fullseq_cave_v3/build_cave_prediction_tasks.py",
        "matched_variants_builder",
    )

    train_meta = pd.read_csv(
        source_reports / "train_oof_predictions.csv", dtype={"patient_id": str}
    )
    valid_meta = pd.read_csv(
        source_reports / "valid_predictions.csv", dtype={"patient_id": str}
    )
    train_ids = train_meta["patient_id"].astype(str).tolist()
    valid_ids = valid_meta["patient_id"].astype(str).tolist()
    y_train = train_meta["target"].to_numpy(np.int64)
    y_valid = valid_meta["target"].to_numpy(np.int64)
    folds = train_meta["fold"].to_numpy(np.int64)
    groups = np.asarray(train_ids, dtype=str)

    roots = {
        "whole": {
            "train": Path(config["whole_train_tables"]),
            "valid": Path(config["whole_valid_tables"]),
        },
        "gt_roi": {
            "train": outputs / "cave_gt_roi_tables/train",
            "valid": outputs / "cave_gt_roi_tables/valid",
        },
    }
    stores: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for representation, paths in roots.items():
        train_store = builder.FeatureStore(paths["train"], "patient")
        valid_store = builder.FeatureStore(paths["valid"], "patient")
        if train_store.scalar_columns != valid_store.scalar_columns:
            raise AssertionError(f"{representation}: scalar schema mismatch")
        train_deep, train_scalar, train_missing = train_store.extract(train_ids)
        valid_deep, valid_scalar, valid_missing = valid_store.extract(valid_ids)
        stores[representation] = {
            "train": {
                "deep": train_deep,
                "scalar": train_scalar,
                "missing": train_missing,
            },
            "valid": {
                "deep": valid_deep,
                "scalar": valid_scalar,
                "missing": valid_missing,
            },
        }

    audit_rows: list[dict[str, object]] = []
    audit_path = reports / "matched_logistic_variants_convergence_audit.csv"
    model_root = outputs / "matched_logistic_variants"
    metric_rows = []
    fold_rows = []
    train_predictions = train_meta[["patient_id", "split", "target", "fold"]].copy()
    valid_predictions = valid_meta[["patient_id", "split", "target"]].copy()
    seed = int(config["prediction"]["seed"])
    c_grid = [float(value) for value in config["prediction"]["c_grid"]]
    expected_folds = sorted(np.unique(folds).tolist())

    for representation, split_data in stores.items():
        train_data = split_data["train"]
        valid_data = split_data["valid"]
        for variant in VARIANTS:
            oof = np.full(len(train_ids), np.nan, dtype=np.float64)
            valid_fold_probabilities = []
            for fold in expected_folds:
                holdout = np.flatnonzero(folds == fold)
                development = np.flatnonzero(folds != fold)
                development_data = subset(train_data, development)
                inner_predictions = {
                    c_value: np.full(len(development), np.nan, dtype=np.float64)
                    for c_value in c_grid
                }
                inner_splits = fixed.grouped_splits(
                    y_train[development],
                    groups[development],
                    int(config["prediction"]["inner_folds"]),
                    seed + fold * 1000,
                )
                for inner_fold, (fit_index, inner_holdout) in enumerate(
                    inner_splits, 1
                ):
                    fit_data = subset(development_data, fit_index)
                    inner_holdout_data = subset(
                        development_data, inner_holdout
                    )
                    preprocessor = fixed.FusionPreprocessor().fit(
                        fit_data, seed + fold * 100 + inner_fold
                    )
                    fit_x = preprocessor.transform_all(fit_data)[variant]
                    inner_holdout_x = preprocessor.transform_all(
                        inner_holdout_data
                    )[variant]
                    for c_value in c_grid:
                        model = fixed.fit_logistic_checked(
                            fit_x,
                            y_train[development][fit_index],
                            c_value,
                            {
                                "task": "adverse_patient",
                                "representation": representation,
                                "outer_fold": fold,
                                "stage": "inner_cv",
                                "variant": variant,
                                "inner_fold": inner_fold,
                            },
                            audit_rows,
                            audit_path,
                            inner_holdout_x,
                            y_train[development][inner_holdout],
                        )
                        inner_predictions[c_value][inner_holdout] = (
                            model.predict_proba(inner_holdout_x)[:, 1]
                        )
                inner_scores = {
                    str(c_value): fixed.safe_ap(
                        y_train[development], probability
                    )
                    for c_value, probability in inner_predictions.items()
                }
                selected_c = max(
                    c_grid,
                    key=lambda value: (
                        inner_scores[str(value)],
                        -float(value),
                    ),
                )
                preprocessor = fixed.FusionPreprocessor().fit(
                    development_data, seed + fold * 100
                )
                development_x = preprocessor.transform_all(
                    development_data
                )[variant]
                holdout_x = preprocessor.transform_all(
                    subset(train_data, holdout)
                )[variant]
                valid_x = preprocessor.transform_all(valid_data)[variant]
                model = fixed.fit_logistic_checked(
                    development_x,
                    y_train[development],
                    selected_c,
                    {
                        "task": "adverse_patient",
                        "representation": representation,
                        "outer_fold": fold,
                        "stage": "outer_development_refit",
                        "variant": variant,
                        "inner_fold": 0,
                    },
                    audit_rows,
                    audit_path,
                    holdout_x,
                    y_train[holdout],
                )
                holdout_probability = model.predict_proba(holdout_x)[:, 1]
                valid_probability = model.predict_proba(valid_x)[:, 1]
                oof[holdout] = holdout_probability
                valid_fold_probabilities.append(valid_probability)
                atomic_joblib(
                    {
                        "preprocessor": preprocessor,
                        "model": model,
                        "selected_c": float(selected_c),
                        "inner_ap_by_c": inner_scores,
                        "development_patient_ids": groups[development].tolist(),
                        "holdout_patient_ids": groups[holdout].tolist(),
                    },
                    model_root
                    / representation
                    / variant
                    / f"fold_{fold}.joblib",
                )
                fold_rows.append(
                    {
                        "representation": representation,
                        "variant": variant,
                        "fold": fold,
                        "selected_c": float(selected_c),
                        "holdout_rows": int(len(holdout)),
                        "holdout_positive": int(y_train[holdout].sum()),
                        "holdout_auroc": fixed.safe_auc(
                            y_train[holdout], holdout_probability
                        ),
                        "holdout_auprc": fixed.safe_ap(
                            y_train[holdout], holdout_probability
                        ),
                    }
                )
            if not np.isfinite(oof).all():
                raise AssertionError(
                    f"{representation} {variant}: incomplete OOF"
                )
            valid_probability = np.mean(
                np.stack(valid_fold_probabilities), axis=0
            )
            threshold = float(fixed.youden_threshold(y_train, oof))
            model_name = f"{representation}_Logistic_{variant}"
            metric_rows.append(
                fixed.metric_row(
                    "adverse_patient",
                    model_name,
                    "Train_OOF",
                    y_train,
                    oof,
                    threshold,
                )
            )
            metric_rows.append(
                fixed.metric_row(
                    "adverse_patient",
                    model_name,
                    "Valid",
                    y_valid,
                    valid_probability,
                    threshold,
                )
            )
            train_predictions[f"{representation}_{variant}_probability"] = oof
            valid_predictions[
                f"{representation}_{variant}_probability"
            ] = valid_probability

    metric_frame = pd.DataFrame(metric_rows)
    fold_frame = pd.DataFrame(fold_rows)
    metric_path = reports / "matched_logistic_variants_metrics.csv"
    fold_path = reports / "matched_logistic_variants_fold_metrics.csv"
    train_path = reports / "matched_logistic_variants_train_oof_predictions.csv"
    valid_path = reports / "matched_logistic_variants_valid_predictions.csv"
    metric_frame.to_csv(metric_path, index=False)
    fold_frame.to_csv(fold_path, index=False)
    train_predictions.to_csv(train_path, index=False)
    valid_predictions.to_csv(valid_path, index=False)

    comparisons = {}
    for variant in VARIANTS:
        comparisons[variant] = {}
        for split in ("Train_OOF", "Valid"):
            whole = metric_frame[
                (metric_frame["model"] == f"whole_Logistic_{variant}")
                & (metric_frame["split"] == split)
            ].iloc[0]
            gt_roi = metric_frame[
                (metric_frame["model"] == f"gt_roi_Logistic_{variant}")
                & (metric_frame["split"] == split)
            ].iloc[0]
            comparisons[variant][split] = {
                f"whole_{key}": float(whole[key])
                for key in ("auroc", "auprc", "brier")
            }
            comparisons[variant][split].update(
                {
                    f"gt_roi_{key}": float(gt_roi[key])
                    for key in ("auroc", "auprc", "brier")
                }
            )
            comparisons[variant][split].update(
                {
                    f"gt_roi_minus_whole_{key}": float(gt_roi[key] - whole[key])
                    for key in ("auroc", "auprc", "brier")
                }
            )
    summary = {
        "version": "gt_oracle_1p5_matched_logistic_variants_1",
        "task": "adverse_patient",
        "representations": list(roots),
        "variants": list(VARIANTS),
        "same_patient_folds_as_primary_logistic": True,
        "train_rows": int(len(train_ids)),
        "train_positive": int(y_train.sum()),
        "valid_rows": int(len(valid_ids)),
        "valid_positive": int(y_valid.sum()),
        "valid_used_for_selection": False,
        "convergence_warning_count": int(
            pd.DataFrame(audit_rows)["convergence_warning"].sum()
        ),
        "comparisons": comparisons,
        "outputs": {
            "metrics": str(metric_path),
            "fold_metrics": str(fold_path),
            "train_predictions": str(train_path),
            "valid_predictions": str(valid_path),
            "convergence_audit": str(audit_path),
        },
    }
    summary_path = reports / "matched_logistic_variants_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

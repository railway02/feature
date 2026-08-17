#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from common import (
    atomic_csv,
    atomic_json,
    configure_runtime,
    hash_lines,
    load_config,
    sha256_file,
    write_success,
)


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def subset(data: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[indices] for key, value in data.items()}


def atomic_joblib(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    joblib.dump(value, temporary)
    os.replace(temporary, path)


def atomic_torch(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def quarantine_partial(path: Path) -> None:
    if not path.exists() or not any(path.iterdir()):
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = path.with_name(f"{path.name}.quarantine_{stamp}")
    counter = 1
    while destination.exists():
        destination = path.with_name(f"{path.name}.quarantine_{stamp}_{counter}")
        counter += 1
    path.replace(destination)


def metric(fixed, model: str, split: str, y: np.ndarray, probability: np.ndarray, threshold: float):
    return fixed.metric_row(
        "adverse_patient",
        model,
        split,
        y,
        probability,
        threshold,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    configure_runtime(config)
    project = Path(config["project_root"])
    outputs = Path(config["paths"]["outputs"])
    reports = Path(config["paths"]["reports"])
    marker = reports / ".MLP_FUSION_SUCCESS"
    summary_path = reports / "mlp_fusion_summary.json"
    if marker.is_file() and summary_path.is_file() and not args.overwrite:
        print(summary_path.read_text(encoding="utf-8"))
        return 0

    fixed_path = Path(config["fixed_trainer"])
    if sha256_file(fixed_path) != config["fixed_trainer_sha256"]:
        raise AssertionError("Fixed trainer hash mismatch")
    fixed = import_module(fixed_path, "fast_v1_mlp_fixed")
    builder = import_module(
        project / "code/api_fullseq_cave_v3/build_cave_prediction_tasks.py",
        "fast_v1_mlp_task_builder",
    )

    train_meta = pd.read_csv(
        reports / "train_oof_predictions.csv",
        dtype={"patient_id": str},
    )
    valid_meta = pd.read_csv(
        reports / "valid_predictions.csv",
        dtype={"patient_id": str},
    )
    required_train = {"patient_id", "target", "fold"}
    required_valid = {"patient_id", "target"}
    if not required_train.issubset(train_meta.columns):
        raise KeyError(f"Train metadata missing {sorted(required_train - set(train_meta.columns))}")
    if not required_valid.issubset(valid_meta.columns):
        raise KeyError(f"Valid metadata missing {sorted(required_valid - set(valid_meta.columns))}")
    if train_meta["patient_id"].duplicated().any() or valid_meta["patient_id"].duplicated().any():
        raise AssertionError("Duplicate patient IDs")
    if set(train_meta["patient_id"]) & set(valid_meta["patient_id"]):
        raise AssertionError("Train/Valid patient overlap")

    train_ids = train_meta["patient_id"].astype(str).tolist()
    valid_ids = valid_meta["patient_id"].astype(str).tolist()
    y_train = pd.to_numeric(train_meta["target"], errors="raise").to_numpy(np.int64)
    y_valid = pd.to_numeric(valid_meta["target"], errors="raise").to_numpy(np.int64)
    fold_assignment = pd.to_numeric(train_meta["fold"], errors="raise").to_numpy(np.int64)
    expected_folds = list(range(1, int(config["prediction"]["folds"]) + 1))
    if sorted(np.unique(fold_assignment).tolist()) != expected_folds:
        raise AssertionError("Unexpected patient fold assignments")
    if not set(np.unique(y_train)).issubset({0, 1}) or not set(np.unique(y_valid)).issubset({0, 1}):
        raise AssertionError("Non-binary target")

    table_roots = {
        "whole": {
            "train": Path(config["whole_train_tables"]),
            "valid": Path(config["whole_valid_tables"]),
        },
        "pred_roi": {
            "train": outputs / "cave_pred_roi_tables/train",
            "valid": outputs / "cave_pred_roi_tables/valid",
        },
    }
    stores = {}
    data = {}
    scalar_schema = None
    for representation, roots in table_roots.items():
        train_store = builder.FeatureStore(roots["train"], "patient")
        valid_store = builder.FeatureStore(roots["valid"], "patient")
        if train_store.scalar_columns != valid_store.scalar_columns:
            raise AssertionError(f"{representation}: Train/Valid scalar schema differs")
        if scalar_schema is None:
            scalar_schema = list(train_store.scalar_columns)
        elif scalar_schema != list(train_store.scalar_columns):
            raise AssertionError("Whole/Pred ROI scalar schema differs")
        stores[representation] = (train_store, valid_store)
        train_deep, train_scalar, train_missing = train_store.extract(train_ids)
        valid_deep, valid_scalar, valid_missing = valid_store.extract(valid_ids)
        data[representation] = {
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = int(config["prediction"]["seed"])
    model_root = outputs / "mlp_fusion_models"
    model_root.mkdir(parents=True, exist_ok=True)
    all_metric_rows = []
    all_fold_rows = []
    train_predictions = train_meta[["patient_id", "split", "target", "fold"]].copy()
    valid_columns = [column for column in ("patient_id", "split", "target") if column in valid_meta.columns]
    valid_predictions = valid_meta[valid_columns].copy()
    representation_results = {}

    for representation in ("whole", "pred_roi"):
        train_data = data[representation]["train"]
        valid_data = data[representation]["valid"]
        oof_probability = np.full(len(train_meta), np.nan, dtype=np.float64)
        valid_fold_probability = []

        for fold in expected_folds:
            holdout = np.flatnonzero(fold_assignment == fold)
            development = np.flatnonzero(fold_assignment != fold)
            fold_dir = model_root / representation / f"fold_{fold}"
            success_path = fold_dir / ".SUCCESS.json"
            prediction_path = fold_dir / "predictions.npz"
            expected_holdout_ids = np.asarray([train_ids[index] for index in holdout], dtype=str)

            resumed = False
            if success_path.is_file() and prediction_path.is_file() and not args.overwrite:
                with np.load(prediction_path) as raw:
                    saved_holdout_ids = raw["holdout_patient_id"].astype(str)
                    saved_valid_ids = raw["valid_patient_id"].astype(str)
                    holdout_probability = raw["holdout_probability"].astype(np.float64)
                    valid_probability = raw["valid_probability"].astype(np.float64)
                if (
                    np.array_equal(saved_holdout_ids, expected_holdout_ids)
                    and np.array_equal(saved_valid_ids, np.asarray(valid_ids, dtype=str))
                    and holdout_probability.shape == (len(holdout),)
                    and valid_probability.shape == (len(valid_ids),)
                    and np.isfinite(holdout_probability).all()
                    and np.isfinite(valid_probability).all()
                ):
                    payload = json.loads(success_path.read_text(encoding="utf-8"))
                    best_epoch = int(payload["best_epoch"])
                    stop_ap = float(payload["early_stop_validation_ap"])
                    pre_audit = payload["preprocessor"]
                    resumed = True
                else:
                    quarantine_partial(fold_dir)

            if not resumed:
                if fold_dir.exists() and any(fold_dir.iterdir()):
                    quarantine_partial(fold_dir)
                fold_dir.mkdir(parents=True, exist_ok=True)
                development_data = subset(train_data, development)
                holdout_data = subset(train_data, holdout)
                preprocessor = fixed.FusionPreprocessor().fit(
                    development_data,
                    seed + fold * 100,
                )
                development_x = preprocessor.transform_all(development_data)["fusion"]
                holdout_x = preprocessor.transform_all(holdout_data)["fusion"]
                valid_x = preprocessor.transform_all(valid_data)["fusion"]
                model, best_epoch, stop_ap = fixed.fit_mlp(
                    development_x,
                    y_train[development],
                    np.asarray(train_ids, dtype=str)[development],
                    device,
                    seed + fold,
                )
                holdout_probability = fixed.mlp_predict(model, holdout_x, device)
                valid_probability = fixed.mlp_predict(model, valid_x, device)
                pre_audit = preprocessor.audit()
                atomic_joblib(preprocessor, fold_dir / "preprocessor.joblib")
                atomic_torch(
                    {
                        "input_dim": int(development_x.shape[1]),
                        "state_dict": {
                            key: value.detach().cpu() for key, value in model.state_dict().items()
                        },
                        "best_epoch": int(best_epoch),
                        "best_validation_ap": float(stop_ap),
                    },
                    fold_dir / "mlp_fusion.pt",
                )
                atomic_npz(
                    prediction_path,
                    holdout_patient_id=expected_holdout_ids,
                    holdout_probability=holdout_probability,
                    valid_patient_id=np.asarray(valid_ids, dtype=str),
                    valid_probability=valid_probability,
                )
                atomic_json(
                    {
                        "representation": representation,
                        "fold": fold,
                        "best_epoch": int(best_epoch),
                        "early_stop_validation_ap": float(stop_ap),
                        "development_rows": int(len(development)),
                        "holdout_rows": int(len(holdout)),
                        "valid_rows": int(len(valid_ids)),
                        "holdout_patient_hash": hash_lines(expected_holdout_ids.tolist()),
                        "valid_patient_hash": hash_lines(valid_ids),
                        "fixed_trainer_sha256": config["fixed_trainer_sha256"],
                        "preprocessor": pre_audit,
                    },
                    success_path,
                )

            oof_probability[holdout] = holdout_probability
            valid_fold_probability.append(valid_probability)
            all_fold_rows.append(
                {
                    "representation": representation,
                    "fold": fold,
                    "development_rows": int(len(development)),
                    "holdout_rows": int(len(holdout)),
                    "best_epoch": int(best_epoch),
                    "early_stop_validation_ap": float(stop_ap),
                    "holdout_auroc": fixed.safe_auc(y_train[holdout], holdout_probability),
                    "holdout_auprc": fixed.safe_ap(y_train[holdout], holdout_probability),
                    "resumed": resumed,
                    "preprocessor": json.dumps(pre_audit, sort_keys=True),
                }
            )
            print(
                f"[MLP FOLD PASS] representation={representation} fold={fold} "
                f"best_epoch={best_epoch} stop_ap={stop_ap:.6f} resumed={resumed}",
                flush=True,
            )

        if not np.isfinite(oof_probability).all():
            raise AssertionError(f"{representation}: incomplete OOF probability")
        valid_probability = np.mean(np.stack(valid_fold_probability), axis=0)
        if not np.isfinite(valid_probability).all():
            raise AssertionError(f"{representation}: nonfinite Valid probability")
        threshold = float(fixed.youden_threshold(y_train, oof_probability))
        model_name = f"{representation}_MLP_fusion"
        oof_metrics = metric(fixed, model_name, "Train_OOF", y_train, oof_probability, threshold)
        valid_metrics = metric(fixed, model_name, "Valid", y_valid, valid_probability, threshold)
        all_metric_rows.extend([oof_metrics, valid_metrics])
        train_predictions[f"{representation}_mlp_fusion_probability"] = oof_probability
        valid_predictions[f"{representation}_mlp_fusion_probability"] = valid_probability
        representation_results[representation] = {
            "threshold_from_train_oof": threshold,
            "train_oof": oof_metrics,
            "valid": valid_metrics,
        }

    whole = representation_results["whole"]
    roi = representation_results["pred_roi"]
    summary = {
        "version": "api_adverse_lesion_cave_fast_v1_mlp_fusion_1",
        "task": "adverse_patient",
        "analysis_timing": "post_hoc_after_primary_logistic_valid_review",
        "scientific_role": "supplementary_exploratory_model_family",
        "representations": ["whole", "pred_roi"],
        "input": "patient_median deep_10240 + scalar + missing_pre_post",
        "train_rows": int(len(train_meta)),
        "train_positive": int(y_train.sum()),
        "valid_rows": int(len(valid_meta)),
        "valid_positive": int(y_valid.sum()),
        "outer_folds": len(expected_folds),
        "same_patient_folds_as_primary_logistic": True,
        "valid_used_for_preprocessing_training_early_stopping_or_threshold": False,
        "valid_evaluated_after_all_train_oof_predictions_complete": True,
        "device": str(device),
        "fixed_trainer": str(fixed_path),
        "fixed_trainer_sha256": config["fixed_trainer_sha256"],
        "scalar_feature_count": int(len(scalar_schema or [])),
        "results": representation_results,
        "pred_roi_minus_whole": {
            "train_oof_auroc": roi["train_oof"]["auroc"] - whole["train_oof"]["auroc"],
            "train_oof_auprc": roi["train_oof"]["auprc"] - whole["train_oof"]["auprc"],
            "train_oof_brier": roi["train_oof"]["brier"] - whole["train_oof"]["brier"],
            "valid_auroc": roi["valid"]["auroc"] - whole["valid"]["auroc"],
            "valid_auprc": roi["valid"]["auprc"] - whole["valid"]["auprc"],
            "valid_brier": roi["valid"]["brier"] - whole["valid"]["brier"],
        },
        "segmentation_quality_warning_inherited": True,
    }
    atomic_csv(pd.DataFrame(all_metric_rows), reports / "mlp_fusion_metrics.csv")
    atomic_csv(pd.DataFrame(all_fold_rows), reports / "mlp_fusion_fold_audit.csv")
    atomic_csv(train_predictions, reports / "mlp_fusion_train_oof_predictions.csv")
    atomic_csv(valid_predictions, reports / "mlp_fusion_valid_predictions.csv")
    atomic_json(summary, summary_path)
    write_success(marker, "11_train_mlp_fusion", config, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

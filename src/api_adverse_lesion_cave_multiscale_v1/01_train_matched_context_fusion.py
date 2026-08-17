#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


ROOT = Path("/root/autodl-tmp/aneurysm")
FAST_CONFIG = ROOT / "configs/api_adverse_lesion_cave_fast_v1.json"
REPORTS = ROOT / "reports/api_adverse_lesion_cave_multiscale_v1"
OUTPUTS = ROOT / "outputs/api_adverse_lesion_cave_multiscale_v1"


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def subset(data: dict[str, object], index: np.ndarray) -> dict[str, object]:
    return {
        "clinical": np.asarray(data["clinical"])[index],
        "images": {
            name: {
                key: np.asarray(value)[index]
                for key, value in branch.items()
            }
            for name, branch in data["images"].items()
        },
    }


def make_imputer() -> SimpleImputer:
    try:
        return SimpleImputer(strategy="median", keep_empty_features=True)
    except TypeError:
        return SimpleImputer(strategy="median")


@dataclass
class MultiScalePreprocessor:
    fixed: object
    optimizer: object
    image_names: tuple[str, ...]
    deep: dict[str, object] = field(default_factory=dict)
    clinical_imputer: SimpleImputer | None = None
    clinical_scaler: StandardScaler | None = None
    final_scaler: StandardScaler | None = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["fixed"] = None
        state["optimizer"] = None
        return state

    def fit(self, data: dict[str, object], seed: int) -> "MultiScalePreprocessor":
        self.deep = {}
        for offset, name in enumerate(self.image_names):
            values = self.optimizer.transform_deep_representation(
                data["images"][name]["deep"], "post_delta"
            )
            self.deep[name] = self.fixed.DeepBranch().fit(
                values, seed + offset * 10000
            )
        self.clinical_imputer = make_imputer()
        clinical = self.clinical_imputer.fit_transform(
            np.asarray(data["clinical"], dtype=np.float64)
        )
        self.clinical_scaler = StandardScaler().fit(clinical)
        base = self._base(data)
        self.final_scaler = StandardScaler().fit(base)
        transformed = self.final_scaler.transform(base)
        if not np.isfinite(transformed).all():
            raise AssertionError("Nonfinite fitted multiscale matrix")
        return self

    def _base(self, data: dict[str, object]) -> np.ndarray:
        pieces: list[np.ndarray] = []
        for name in self.image_names:
            values = self.optimizer.transform_deep_representation(
                data["images"][name]["deep"], "post_delta"
            )
            pieces.append(self.deep[name].transform(values))
            pieces.append(
                np.asarray(data["images"][name]["missing"], dtype=np.float64)
            )
        if self.clinical_imputer is None or self.clinical_scaler is None:
            raise RuntimeError("Clinical branch is not fitted")
        clinical = self.clinical_imputer.transform(
            np.asarray(data["clinical"], dtype=np.float64)
        )
        pieces.append(self.clinical_scaler.transform(clinical))
        base = np.concatenate(pieces, axis=1)
        if not np.isfinite(base).all():
            raise AssertionError("Nonfinite multiscale base matrix")
        return base

    def transform(self, data: dict[str, object]) -> np.ndarray:
        if self.final_scaler is None:
            raise RuntimeError("Final scaler is not fitted")
        value = self.final_scaler.transform(self._base(data)).astype(np.float64)
        if not np.isfinite(value).all():
            raise AssertionError("Nonfinite multiscale transformed matrix")
        return value


def atomic_joblib(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    joblib.dump(value, temporary)
    os.replace(temporary, path)


def main() -> int:
    config = json.loads(FAST_CONFIG.read_text(encoding="utf-8"))
    REPORTS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    fixed = import_file(Path(config["fixed_trainer"]), "multiscale_fixed")
    builder = import_file(
        ROOT / "code/api_fullseq_cave_v3/build_cave_prediction_tasks.py",
        "multiscale_builder",
    )
    optimizer = import_file(
        ROOT / "tools/api_fullseq_cave_v3_prediction_audit/optimize_existing_features_cv.py",
        "multiscale_optimizer",
    )

    source_reports = Path(config["paths"]["reports"])
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
        "pred_roi": {
            "train": Path(config["paths"]["outputs"]) / "cave_pred_roi_tables/train",
            "valid": Path(config["paths"]["outputs"]) / "cave_pred_roi_tables/valid",
        },
        "gt_roi": {
            "train": ROOT / "outputs/api_adverse_lesion_cave_gt_oracle_v1/cave_gt_roi_tables/train",
            "valid": ROOT / "outputs/api_adverse_lesion_cave_gt_oracle_v1/cave_gt_roi_tables/valid",
        },
    }
    images: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for name, paths in roots.items():
        train_store = builder.FeatureStore(paths["train"], "patient")
        valid_store = builder.FeatureStore(paths["valid"], "patient")
        train_deep, _, train_missing = train_store.extract(train_ids)
        valid_deep, _, valid_missing = valid_store.extract(valid_ids)
        images[name] = {
            "train": {"deep": train_deep, "missing": train_missing},
            "valid": {"deep": valid_deep, "missing": valid_missing},
        }

    v1_config = json.loads(
        (ROOT / "configs/api_adverse_lesion_cave_v1.json").read_text(encoding="utf-8")
    )
    train_records = optimizer.excel_records(Path(v1_config["train_excel"]), "Train")
    valid_records = optimizer.excel_records(Path(v1_config["valid_excel"]), "Valid")
    train_clinical, clinical_names = optimizer.clinical_matrix(
        "adverse_patient", train_meta, train_records, "context"
    )
    valid_clinical, valid_clinical_names = optimizer.clinical_matrix(
        "adverse_patient", valid_meta, valid_records, "context"
    )
    if clinical_names != valid_clinical_names:
        raise AssertionError("Train/Valid clinical schema mismatch")

    train_data = {
        "clinical": train_clinical,
        "images": {name: value["train"] for name, value in images.items()},
    }
    valid_data = {
        "clinical": valid_clinical,
        "images": {name: value["valid"] for name, value in images.items()},
    }
    representations = {
        "clinical_context": (),
        "whole_context": ("whole",),
        "pred_roi_context": ("pred_roi",),
        "whole_pred_roi_context": ("whole", "pred_roi"),
        "gt_roi_context": ("gt_roi",),
        "whole_gt_roi_context": ("whole", "gt_roi"),
    }

    seed = int(config["prediction"]["seed"])
    c_grid = [float(value) for value in config["prediction"]["c_grid"]]
    expected_folds = sorted(np.unique(folds).tolist())
    audit_rows: list[dict[str, object]] = []
    audit_path = REPORTS / "context_multiscale_convergence_audit.csv"
    metric_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    train_predictions = train_meta[["patient_id", "split", "target", "fold"]].copy()
    valid_predictions = valid_meta[["patient_id", "split", "target"]].copy()

    for representation, image_names in representations.items():
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
                y_train[development], groups[development],
                int(config["prediction"]["inner_folds"]), seed + fold * 1000,
            )
            for inner_fold, (fit_index, inner_holdout) in enumerate(inner_splits, 1):
                fit_data = subset(development_data, fit_index)
                inner_data = subset(development_data, inner_holdout)
                preprocessor = MultiScalePreprocessor(
                    fixed, optimizer, image_names
                ).fit(fit_data, seed + fold * 100 + inner_fold)
                fit_x = preprocessor.transform(fit_data)
                inner_x = preprocessor.transform(inner_data)
                for c_value in c_grid:
                    model = fixed.fit_logistic_checked(
                        fit_x, y_train[development][fit_index], c_value,
                        {
                            "task": "adverse_patient",
                            "representation": representation,
                            "outer_fold": fold,
                            "stage": "inner_cv",
                            "variant": "deep_post_delta_plus_context",
                            "inner_fold": inner_fold,
                        },
                        audit_rows, audit_path, inner_x,
                        y_train[development][inner_holdout],
                    )
                    inner_predictions[c_value][inner_holdout] = model.predict_proba(inner_x)[:, 1]
            inner_scores = {
                str(c_value): fixed.safe_ap(y_train[development], probability)
                for c_value, probability in inner_predictions.items()
            }
            selected_c = max(
                c_grid, key=lambda value: (inner_scores[str(value)], -float(value))
            )
            preprocessor = MultiScalePreprocessor(
                fixed, optimizer, image_names
            ).fit(development_data, seed + fold * 100)
            development_x = preprocessor.transform(development_data)
            holdout_x = preprocessor.transform(subset(train_data, holdout))
            valid_x = preprocessor.transform(valid_data)
            model = fixed.fit_logistic_checked(
                development_x, y_train[development], selected_c,
                {
                    "task": "adverse_patient",
                    "representation": representation,
                    "outer_fold": fold,
                    "stage": "outer_development_refit",
                    "variant": "deep_post_delta_plus_context",
                    "inner_fold": 0,
                },
                audit_rows, audit_path, holdout_x, y_train[holdout],
            )
            holdout_probability = model.predict_proba(holdout_x)[:, 1]
            oof[holdout] = holdout_probability
            valid_fold_probabilities.append(model.predict_proba(valid_x)[:, 1])
            atomic_joblib(
                {
                    "preprocessor": preprocessor,
                    "model": model,
                    "selected_c": selected_c,
                    "clinical_names": clinical_names,
                    "development_patient_ids": groups[development].tolist(),
                    "holdout_patient_ids": groups[holdout].tolist(),
                },
                OUTPUTS / "models" / representation / f"fold_{fold}.joblib",
            )
            fold_rows.append({
                "representation": representation,
                "fold": fold,
                "selected_c": selected_c,
                "holdout_rows": int(len(holdout)),
                "holdout_positive": int(y_train[holdout].sum()),
                "holdout_auroc": fixed.safe_auc(y_train[holdout], holdout_probability),
                "holdout_auprc": fixed.safe_ap(y_train[holdout], holdout_probability),
            })
        if not np.isfinite(oof).all():
            raise AssertionError(f"Incomplete OOF: {representation}")
        valid_probability = np.mean(np.stack(valid_fold_probabilities), axis=0)
        threshold = float(fixed.youden_threshold(y_train, oof))
        metric_rows.append(fixed.metric_row(
            "adverse_patient", representation, "Train_OOF",
            y_train, oof, threshold,
        ))
        metric_rows.append(fixed.metric_row(
            "adverse_patient", representation, "Valid",
            y_valid, valid_probability, threshold,
        ))
        train_predictions[f"{representation}_probability"] = oof
        valid_predictions[f"{representation}_probability"] = valid_probability

    metrics = pd.DataFrame(metric_rows)
    folds_frame = pd.DataFrame(fold_rows)
    metrics.to_csv(REPORTS / "context_multiscale_metrics.csv", index=False)
    folds_frame.to_csv(REPORTS / "context_multiscale_fold_metrics.csv", index=False)
    train_predictions.to_csv(REPORTS / "context_multiscale_train_oof_predictions.csv", index=False)
    valid_predictions.to_csv(REPORTS / "context_multiscale_valid_predictions.csv", index=False)
    summary = {
        "version": "api_adverse_lesion_cave_multiscale_v1",
        "task": "adverse_patient",
        "cave_version": "api_fullseq_cave_v3",
        "image_representation": "post_plus_delta",
        "clinical_feature_set": "context",
        "representations": {key: list(value) for key, value in representations.items()},
        "same_current_patient_cohort_and_folds": True,
        "train_rows": len(train_ids),
        "train_positive": int(y_train.sum()),
        "valid_rows": len(valid_ids),
        "valid_positive": int(y_valid.sum()),
        "valid_used_for_selection": False,
        "convergence_warning_count": int(pd.DataFrame(audit_rows)["convergence_warning"].sum()),
        "metrics": metrics.to_dict("records"),
    }
    (REPORTS / "context_multiscale_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

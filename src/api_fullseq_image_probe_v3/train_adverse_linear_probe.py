#!/usr/bin/env python3
"""Train controlled linear probes for pure image-derived adverse features.

The same nested, regularized Logistic protocol is used for every feature set.
No PCA, random projection, clinical predictor, or Valid fitting is allowed.
This makes OOF/Valid performance a direct test of how linearly predictive each
frozen image representation is.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler


SEED = 42
C_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0)
VARIANTS = (
    "sea_prepost",
    "sea_full",
    "cave_embedding",
    "cave_scalar",
    "cave_all",
    "cave_embedding_sea_full",
    "cave_all_sea_full",
)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(temporary, path)


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def safe_auc(y: np.ndarray, probability: np.ndarray) -> float:
    return float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else float("nan")


def safe_ap(y: np.ndarray, probability: np.ndarray) -> float:
    return float(average_precision_score(y, probability)) if len(np.unique(y)) == 2 else float("nan")


def youden_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, probability)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    return float(thresholds[finite][int(np.argmax(tpr[finite] - fpr[finite]))])


def metric_row(
    model: str,
    split: str,
    y: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "task": "adverse_patient",
        "model": model,
        "split": split,
        "rows": int(len(y)),
        "positive": int((y == 1).sum()),
        "negative": int((y == 0).sum()),
        "auroc": safe_auc(y, probability),
        "auprc": safe_ap(y, probability),
        "brier": float(brier_score_loss(y, probability)),
        "threshold": float(threshold),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


@dataclass
class Bundle:
    patient_id: np.ndarray
    sea_prepost: np.ndarray
    sea_full: np.ndarray
    cave_embedding: np.ndarray
    cave_scalar: np.ndarray
    missing: np.ndarray
    target: np.ndarray

    def subset(self, indices: np.ndarray) -> "Bundle":
        return Bundle(**{
            field_name: getattr(self, field_name)[indices]
            for field_name in self.__dataclass_fields__
        })


def load_bundle(path: Path) -> Bundle:
    with np.load(path) as raw:
        bundle = Bundle(**{
            "patient_id": raw["patient_id"].astype(str),
            "sea_prepost": np.array(raw["sea_prepost"], dtype=np.float32, copy=True),
            "sea_full": np.array(raw["sea_full"], dtype=np.float32, copy=True),
            "cave_embedding": np.array(raw["cave_embedding"], dtype=np.float32, copy=True),
            "cave_scalar": np.array(raw["cave_scalar"], dtype=np.float32, copy=True),
            "missing": np.array(raw["missing"], dtype=np.float32, copy=True),
            "target": np.array(raw["target"], dtype=np.int64, copy=True),
        })
    expected = {
        "sea_prepost": 212,
        "sea_full": 319,
        "cave_embedding": 10240,
    }
    for name, dimension in expected.items():
        if getattr(bundle, name).shape != (len(bundle.target), dimension):
            raise AssertionError(f"Unexpected {name} shape {getattr(bundle, name).shape}")
    if bundle.missing.shape != (len(bundle.target), 2):
        raise AssertionError("Unexpected missing shape")
    if len(set(bundle.patient_id.tolist())) != len(bundle.patient_id):
        raise AssertionError("Duplicate patient_id")
    return bundle


@dataclass
class EmbeddingBranch:
    scaler: StandardScaler | None = None

    def fit(self, values: np.ndarray) -> "EmbeddingBranch":
        x = np.nan_to_num(
            np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
        )
        self.scaler = StandardScaler().fit(x)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("Embedding branch not fitted")
        x = np.nan_to_num(
            np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
        )
        return self.scaler.transform(x).astype(np.float64)

    def audit(self) -> dict[str, Any]:
        return {
            "input_columns": int(len(self.scaler.mean_)) if self.scaler else 0,
            "pca": False,
            "random_projection": False,
        }


def make_imputer() -> SimpleImputer:
    try:
        return SimpleImputer(
            strategy="median", add_indicator=True, keep_empty_features=True
        )
    except TypeError:  # pragma: no cover
        return SimpleImputer(strategy="median", add_indicator=True)


@dataclass
class NumericBranch:
    minimum_finite_fraction: float = 0.25
    keep: np.ndarray | None = None
    imputer: SimpleImputer | None = None
    robust: RobustScaler | None = None
    standard: StandardScaler | None = None

    def fit(self, values: np.ndarray) -> "NumericBranch":
        x = np.asarray(values, dtype=np.float64)
        finite_fraction = np.isfinite(x).mean(axis=0)
        with np.errstate(all="ignore"):
            variance = np.nanvar(x, axis=0)
        self.keep = (
            (finite_fraction >= self.minimum_finite_fraction)
            & np.isfinite(variance)
            & (variance > 1e-12)
        )
        if not self.keep.any():
            raise AssertionError("No usable numeric features")
        self.imputer = make_imputer()
        transformed = self.imputer.fit_transform(x[:, self.keep])
        self.robust = RobustScaler().fit(transformed)
        transformed = self.robust.transform(transformed)
        self.standard = StandardScaler().fit(transformed)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.keep is None or self.imputer is None or self.robust is None or self.standard is None:
            raise RuntimeError("Numeric branch not fitted")
        transformed = self.imputer.transform(
            np.asarray(values, dtype=np.float64)[:, self.keep]
        )
        transformed = self.robust.transform(transformed)
        return self.standard.transform(transformed).astype(np.float64)

    def audit(self) -> dict[str, Any]:
        indicators = 0
        if self.imputer is not None and getattr(self.imputer, "indicator_", None) is not None:
            indicators = int(len(self.imputer.indicator_.features_))
        return {
            "kept_input_columns": int(self.keep.sum()) if self.keep is not None else 0,
            "missing_indicator_columns": indicators,
            "output_columns": (
                int(len(self.standard.mean_)) if self.standard is not None else 0
            ),
            "pca": False,
        }


@dataclass
class ProbePreprocessor:
    cave_embedding: EmbeddingBranch | None = None
    cave_scalar: NumericBranch | None = None
    sea_prepost: NumericBranch | None = None
    sea_full: NumericBranch | None = None
    final_scalers: dict[str, StandardScaler] = field(default_factory=dict)

    def fit(self, data: Bundle) -> "ProbePreprocessor":
        self.cave_embedding = EmbeddingBranch().fit(data.cave_embedding)
        self.cave_scalar = NumericBranch().fit(data.cave_scalar)
        self.sea_prepost = NumericBranch().fit(data.sea_prepost)
        self.sea_full = NumericBranch().fit(data.sea_full)
        base = self._base_transform(data)
        self.final_scalers = {
            variant: StandardScaler().fit(base[variant]) for variant in VARIANTS
        }
        self.transform_all(data)
        return self

    def _base_transform(self, data: Bundle) -> dict[str, np.ndarray]:
        if any(branch is None for branch in (
            self.cave_embedding, self.cave_scalar, self.sea_prepost, self.sea_full
        )):
            raise RuntimeError("Probe preprocessor not fitted")
        embedding = self.cave_embedding.transform(data.cave_embedding)
        scalar = self.cave_scalar.transform(data.cave_scalar)
        sea_prepost = self.sea_prepost.transform(data.sea_prepost)
        sea_full = self.sea_full.transform(data.sea_full)
        missing = np.asarray(data.missing, dtype=np.float64)
        result = {
            "sea_prepost": np.concatenate([sea_prepost, missing], axis=1),
            "sea_full": np.concatenate([sea_full, missing], axis=1),
            "cave_embedding": np.concatenate([embedding, missing], axis=1),
            "cave_scalar": np.concatenate([scalar, missing], axis=1),
            "cave_all": np.concatenate([embedding, scalar, missing], axis=1),
            "cave_embedding_sea_full": np.concatenate(
                [embedding, sea_full, missing], axis=1
            ),
            "cave_all_sea_full": np.concatenate(
                [embedding, scalar, sea_full, missing], axis=1
            ),
        }
        if any(not np.isfinite(value).all() for value in result.values()):
            raise AssertionError("Probe matrices contain nonfinite values")
        return result

    def transform_all(self, data: Bundle) -> dict[str, np.ndarray]:
        if set(self.final_scalers) != set(VARIANTS):
            raise RuntimeError("Final scalers not fitted")
        base = self._base_transform(data)
        return {
            variant: self.final_scalers[variant].transform(base[variant]).astype(
                np.float64, copy=False
            )
            for variant in VARIANTS
        }

    def audit(self) -> dict[str, Any]:
        return {
            "cave_embedding": self.cave_embedding.audit() if self.cave_embedding else None,
            "cave_scalar": self.cave_scalar.audit() if self.cave_scalar else None,
            "sea_prepost": self.sea_prepost.audit() if self.sea_prepost else None,
            "sea_full": self.sea_full.audit() if self.sea_full else None,
            "final_dimensions": {
                variant: int(len(scaler.mean_))
                for variant, scaler in self.final_scalers.items()
            },
            "pca": False,
            "clinical_predictors": False,
        }


def logistic_model(c_value: float, rows: int, columns: int) -> LogisticRegression:
    return LogisticRegression(
        C=float(c_value),
        class_weight="balanced",
        penalty="l2",
        solver="liblinear",
        dual=bool(columns > rows),
        max_iter=10000,
        tol=1e-4,
        random_state=SEED,
    )


def fit_checked(
    x: np.ndarray,
    y: np.ndarray,
    c_value: float,
    context: dict[str, Any],
    audit_rows: list[dict[str, Any]],
    audit_path: Path,
    evaluation_x: np.ndarray | None = None,
    evaluation_y: np.ndarray | None = None,
) -> LogisticRegression:
    model = logistic_model(c_value, x.shape[0], x.shape[1])
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x, y)
    elapsed = time.perf_counter() - start
    convergence = [item for item in caught if issubclass(item.category, ConvergenceWarning)]
    evaluation_probability = (
        model.predict_proba(evaluation_x)[:, 1] if evaluation_x is not None else None
    )
    row = {
        **context,
        "C": float(c_value),
        "rows": int(x.shape[0]),
        "columns": int(x.shape[1]),
        "dual": bool(model.dual),
        "n_iter": int(np.asarray(model.n_iter_).max()),
        "fit_seconds": float(elapsed),
        "convergence_warning": bool(convergence),
        "warning_text": " || ".join(str(item.message) for item in caught),
        "coefficient_l2_norm": float(np.linalg.norm(model.coef_)),
        "coefficient_max_abs": float(np.abs(model.coef_).max()),
        "evaluation_auroc": (
            safe_auc(evaluation_y, evaluation_probability)
            if evaluation_probability is not None and evaluation_y is not None else float("nan")
        ),
        "evaluation_auprc": (
            safe_ap(evaluation_y, evaluation_probability)
            if evaluation_probability is not None and evaluation_y is not None else float("nan")
        ),
    }
    audit_rows.append(row)
    atomic_csv(pd.DataFrame(audit_rows), audit_path)
    if convergence:
        raise RuntimeError(f"Logistic convergence failure: {context}")
    return model


def splits(y: np.ndarray, requested: int, seed: int):
    return list(
        StratifiedKFold(n_splits=requested, shuffle=True, random_state=seed).split(
            np.zeros(len(y)), y
        )
    )


def select_c_nested(
    data: Bundle,
    task: str,
    outer_fold: int,
    audit_rows: list[dict[str, Any]],
    audit_path: Path,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    predictions = {
        variant: {
            c_value: np.full(len(data.target), np.nan, dtype=np.float64)
            for c_value in C_GRID
        }
        for variant in VARIANTS
    }
    for inner_fold, (fit_index, holdout_index) in enumerate(
        splits(data.target, 3, SEED + outer_fold * 1000), start=1
    ):
        fit_data = data.subset(fit_index)
        holdout_data = data.subset(holdout_index)
        preprocessor = ProbePreprocessor().fit(fit_data)
        fit_x = preprocessor.transform_all(fit_data)
        holdout_x = preprocessor.transform_all(holdout_data)
        for variant in VARIANTS:
            for c_value in C_GRID:
                model = fit_checked(
                    fit_x[variant], fit_data.target, c_value,
                    {
                        "task": task,
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "stage": "inner_cv",
                        "variant": variant,
                    },
                    audit_rows, audit_path,
                    holdout_x[variant], holdout_data.target,
                )
                predictions[variant][c_value][holdout_index] = model.predict_proba(
                    holdout_x[variant]
                )[:, 1]
    scores: dict[str, dict[str, float]] = {}
    selected: dict[str, float] = {}
    for variant in VARIANTS:
        scores[variant] = {
            str(c_value): safe_ap(data.target, predictions[variant][c_value])
            for c_value in C_GRID
        }
        selected[variant] = float(
            max(C_GRID, key=lambda value: (scores[variant][str(value)], -value))
        )
    return selected, scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)
    train = load_bundle(dataset_dir / "train.npz")
    valid = load_bundle(dataset_dir / "valid.npz")
    if set(train.patient_id) & set(valid.patient_id):
        raise AssertionError("Train/Valid patient overlap")
    schema = json.loads((dataset_dir / "feature_schema.json").read_text(encoding="utf-8"))
    if schema["metadata_predictor_columns"]:
        raise AssertionError("Metadata predictors entered image probe")
    outer_splits = splits(train.target, 5, SEED)
    oof = {
        variant: np.full(len(train.target), np.nan, dtype=np.float64)
        for variant in VARIANTS
    }
    valid_folds = {variant: [] for variant in VARIANTS}
    audit_rows: list[dict[str, Any]] = []
    audit_path = output / "logistic_convergence_audit.csv"
    fold_rows: list[dict[str, Any]] = []
    for fold, (development_index, holdout_index) in enumerate(outer_splits, start=1):
        print(f"[FOLD START] {fold}/5", flush=True)
        development = train.subset(development_index)
        holdout = train.subset(holdout_index)
        selected_c, c_scores = select_c_nested(
            development, "adverse_patient", fold, audit_rows, audit_path
        )
        preprocessor = ProbePreprocessor().fit(development)
        development_x = preprocessor.transform_all(development)
        holdout_x = preprocessor.transform_all(holdout)
        valid_x = preprocessor.transform_all(valid)
        fold_dir = output / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, fold_dir / "preprocessor.joblib")
        for variant in VARIANTS:
            model = fit_checked(
                development_x[variant], development.target, selected_c[variant],
                {
                    "task": "adverse_patient",
                    "outer_fold": fold,
                    "inner_fold": 0,
                    "stage": "outer_development_refit",
                    "variant": variant,
                },
                audit_rows, audit_path,
                holdout_x[variant], holdout.target,
            )
            oof[variant][holdout_index] = model.predict_proba(holdout_x[variant])[:, 1]
            valid_folds[variant].append(model.predict_proba(valid_x[variant])[:, 1])
            joblib.dump(model, fold_dir / f"linear_{variant}.joblib")
            fold_rows.append({
                "fold": fold,
                "variant": variant,
                "development_rows": int(len(development_index)),
                "holdout_rows": int(len(holdout_index)),
                "selected_c": selected_c[variant],
                "inner_oof_auprc_by_c": json.dumps(c_scores[variant], sort_keys=True),
                "holdout_auroc": safe_auc(
                    holdout.target, oof[variant][holdout_index]
                ),
                "holdout_auprc": safe_ap(
                    holdout.target, oof[variant][holdout_index]
                ),
                "preprocessor": json.dumps(preprocessor.audit(), sort_keys=True),
            })
        print(f"[FOLD DONE] {fold}/5", flush=True)
    probabilities: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "Dummy": (
            np.full(len(train.target), train.target.mean()),
            np.full(len(valid.target), train.target.mean()),
        )
    }
    for variant in VARIANTS:
        if not np.isfinite(oof[variant]).all():
            raise AssertionError(f"Incomplete OOF {variant}")
        probabilities[f"LinearProbe_{variant}"] = (
            oof[variant], np.mean(np.stack(valid_folds[variant]), axis=0)
        )

    full_selected_c, full_scores = select_c_nested(
        train, "adverse_patient", 0, audit_rows, audit_path
    )
    full_preprocessor = ProbePreprocessor().fit(train)
    full_x = full_preprocessor.transform_all(train)
    joblib.dump(full_preprocessor, output / "full_train_preprocessor.joblib")
    for variant in VARIANTS:
        model = fit_checked(
            full_x[variant], train.target, full_selected_c[variant],
            {
                "task": "adverse_patient",
                "outer_fold": 0,
                "inner_fold": 0,
                "stage": "full_train_refit",
                "variant": variant,
            },
            audit_rows, audit_path,
        )
        joblib.dump({
            "model": model,
            "selected_c": full_selected_c[variant],
            "inner_oof_auprc_by_c": full_scores[variant],
        }, output / f"full_train_linear_{variant}.joblib")

    metrics: list[dict[str, Any]] = []
    thresholds: dict[str, float] = {}
    train_predictions = pd.DataFrame({
        "patient_id": train.patient_id, "target": train.target
    })
    valid_predictions = pd.DataFrame({
        "patient_id": valid.patient_id, "target": valid.target
    })
    for model_name, (train_probability, valid_probability) in probabilities.items():
        threshold = youden_threshold(train.target, train_probability)
        thresholds[model_name] = threshold
        metrics.extend([
            metric_row(model_name, "Train_OOF", train.target, train_probability, threshold),
            metric_row(model_name, "Valid", valid.target, valid_probability, threshold),
        ])
        train_predictions[f"{model_name.lower()}_probability"] = train_probability
        valid_predictions[f"{model_name.lower()}_probability"] = valid_probability
    metrics_frame = pd.DataFrame(metrics)
    atomic_csv(metrics_frame, output / "metrics.csv")
    atomic_csv(train_predictions, output / "train_oof_predictions.csv")
    atomic_csv(valid_predictions, output / "valid_predictions.csv")
    atomic_csv(pd.DataFrame(fold_rows), output / "fold_audit.csv")
    learned_oof = metrics_frame[
        (metrics_frame["split"] == "Train_OOF") & (metrics_frame["model"] != "Dummy")
    ]
    best_model = str(
        learned_oof.sort_values(["auprc", "auroc"], ascending=False).iloc[0]["model"]
    )
    summary = {
        "version": "api_fullseq_image_probe_v3_linear_1",
        "task": "adverse_patient",
        "predictors": "image-derived only",
        "models": list(probabilities),
        "best_model_selected_by_train_oof_auprc": best_model,
        "thresholds_from_train_oof": thresholds,
        "outer_folds": 5,
        "inner_folds": 3,
        "c_grid": list(C_GRID),
        "pca": False,
        "random_projection": False,
        "clinical_predictors": False,
        "valid_used_for_preprocessing_training_selection_or_threshold": False,
        "valid_labels_used_for": "metrics after predictions only",
        "seed": SEED,
    }
    atomic_json(summary, output / "summary.json")
    atomic_json(summary, output / ".MODELS_SUCCESS")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

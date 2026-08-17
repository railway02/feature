#!/root/autodl-tmp/envs/aneurysm-ml/bin/python
"""Train and evaluate the adverse_pre baselines under a locked protocol.

All development decisions are made from the training set. The validation set is
read at startup for invariant checks, then used exactly once per frozen model to
obtain probabilities after model family, hyperparameters, and thresholds have
already been fixed.
"""

from __future__ import annotations

import json
import logging
import math
import os
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PYTHON = Path("/root/autodl-tmp/envs/aneurysm-ml/bin/python")
TRAIN_PATH = PROJECT_ROOT / "outputs/task_datasets/adverse_pre_train.csv"
VALID_PATH = PROJECT_ROOT / "outputs/task_datasets/adverse_pre_valid.csv"
FINAL_OUTPUT_DIR = PROJECT_ROOT / "outputs/baselines/adverse_pre_v1"
FINAL_REPORT_DIR = PROJECT_ROOT / "reports/adverse_pre_v1"

if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
    raise RuntimeError(
        f"Wrong Python interpreter: {sys.executable}. Required: {EXPECTED_PYTHON}"
    )

try:
    import joblib
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy
    import sklearn
    from catboost import CatBoostClassifier, __version__ as catboost_version
    from sklearn.base import clone
    from sklearn.calibration import calibration_curve
    from sklearn.dummy import DummyClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    print(
        f"Required dependency import failed under {sys.executable}: {exc}",
        file=sys.stderr,
        flush=True,
    )
    raise


SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 5
BOOTSTRAP_REPEATS = 2000
MODEL_ORDER = ["Dummy", "Logistic", "CatBoost"]
METRIC_ORDER = [
    "auroc",
    "auprc",
    "balanced_accuracy",
    "f1",
    "precision",
    "sensitivity",
    "specificity",
    "brier_score",
]

CONFIGURATION: dict[str, Any] = {
    "experiment": "adverse_pre_v1",
    "task": "Predict binary adverse outcome from 48 preoperative 2D DSA SEA-RAFT patient-level features",
    "python_interpreter": str(EXPECTED_PYTHON),
    "random_seed": SEED,
    "inputs": {
        "train": str(TRAIN_PATH),
        "valid": str(VALID_PATH),
    },
    "expected_invariants": {
        "train_rows": 794,
        "train_positive": 132,
        "valid_rows": 209,
        "valid_positive": 36,
        "feature_count": 48,
        "train_valid_patient_id_intersection": 0,
        "allowed_labels": [0, 1],
        "excluded_exact_columns": [
            "patient_id",
            "split",
            "adverse",
            "runtime_s",
            "n_pairs",
        ],
        "excluded_feature_prefixes": ["post_", "delta_"],
    },
    "protocol": {
        "outer_cv": {
            "class": "StratifiedKFold",
            "n_splits": OUTER_FOLDS,
            "shuffle": True,
            "random_state": SEED,
        },
        "inner_cv": {
            "class": "StratifiedKFold",
            "n_splits": INNER_FOLDS,
            "shuffle": True,
            "random_state": SEED,
        },
        "inner_scoring": "average_precision",
        "model_selection": "highest Train nested OOF AUPRC only; ties follow MODEL_ORDER",
        "threshold_selection": (
            "maximize Train OOF balanced accuracy; ties minimize distance to 0.5, "
            "then choose the lower threshold"
        ),
        "validation_use": (
            "one predict_proba call per frozen final model after model family, "
            "hyperparameters, and OOF threshold are frozen"
        ),
        "bootstrap": {
            "unit": "validation patient",
            "valid_repeats": BOOTSTRAP_REPEATS,
            "random_seed_per_model": SEED,
            "discard_single_class_resamples": True,
            "confidence_interval": "2.5th and 97.5th percentiles",
            "threshold": "frozen Train OOF threshold",
        },
    },
    "models": {
        "Dummy": {
            "estimator": "DummyClassifier",
            "fixed_params": {"strategy": "prior"},
            "param_grid": [{}],
        },
        "Logistic": {
            "estimator": "Pipeline(SimpleImputer(median), StandardScaler, LogisticRegression)",
            "fixed_params": {
                "imputer__strategy": "median",
                "model__solver": "liblinear",
                "model__penalty": "l2",
                "model__max_iter": 5000,
                "model__random_state": SEED,
            },
            "param_grid": {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
        },
        "CatBoost": {
            "estimator": "CatBoostClassifier",
            "fixed_params": {
                "task_type": "CPU",
                "thread_count": 8,
                "random_seed": SEED,
                "verbose": False,
                "allow_writing_files": False,
                "loss_function": "Logloss",
            },
            "param_grid": [
                {
                    "iterations": [300],
                    "depth": [3, 5],
                    "learning_rate": [0.03, 0.1],
                    "l2_leaf_reg": [5.0],
                },
                {
                    "iterations": [300],
                    "depth": [3, 5],
                    "learning_rate": [0.03, 0.1],
                    "l2_leaf_reg": [5.0],
                    "auto_class_weights": ["Balanced"],
                },
            ],
        },
    },
}


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
            default=json_default,
        )
        handle.write("\n")


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("adverse_pre_v1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        fmt="%(asctime)sZ | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    formatter.converter = time_gmtime
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def time_gmtime(*args: Any) -> Any:
    import time

    return time.gmtime(*args)


def assert_input_invariants(
    train_df: pd.DataFrame, valid_df: pd.DataFrame
) -> list[str]:
    required = {"patient_id", "split", "adverse"}
    assert required.issubset(train_df.columns), "Train is missing required metadata columns"
    assert required.issubset(valid_df.columns), "Valid is missing required metadata columns"

    assert len(train_df) == 794, f"Train row count is {len(train_df)}, expected 794"
    assert len(valid_df) == 209, f"Valid row count is {len(valid_df)}, expected 209"
    assert train_df["adverse"].notna().all(), "Train labels contain missing values"
    assert valid_df["adverse"].notna().all(), "Valid labels contain missing values"

    train_labels = train_df["adverse"].astype(int)
    valid_labels = valid_df["adverse"].astype(int)
    assert np.array_equal(
        train_df["adverse"].to_numpy(dtype=float), train_labels.to_numpy(dtype=float)
    ), "Train labels are not integer-valued"
    assert np.array_equal(
        valid_df["adverse"].to_numpy(dtype=float), valid_labels.to_numpy(dtype=float)
    ), "Valid labels are not integer-valued"
    assert set(train_labels.unique()).issubset({0, 1}), "Train labels are not binary 0/1"
    assert set(valid_labels.unique()).issubset({0, 1}), "Valid labels are not binary 0/1"
    assert set(train_labels.unique()) == {0, 1}, "Train does not contain both label classes"
    assert set(valid_labels.unique()) == {0, 1}, "Valid does not contain both label classes"
    assert int(train_labels.sum()) == 132, (
        f"Train positive count is {int(train_labels.sum())}, expected 132"
    )
    assert int(valid_labels.sum()) == 36, (
        f"Valid positive count is {int(valid_labels.sum())}, expected 36"
    )

    assert train_df["patient_id"].notna().all(), "Train patient_id contains missing values"
    assert valid_df["patient_id"].notna().all(), "Valid patient_id contains missing values"
    assert train_df["patient_id"].is_unique, "Train patient_id is not unique"
    assert valid_df["patient_id"].is_unique, "Valid patient_id is not unique"
    overlap = set(train_df["patient_id"]).intersection(set(valid_df["patient_id"]))
    assert len(overlap) == 0, f"Train/Valid patient_id overlap is {len(overlap)}, expected 0"
    assert set(train_df["split"].astype(str).unique()) == {"train"}, (
        "Train split column contains values other than 'train'"
    )
    assert set(valid_df["split"].astype(str).unique()) == {"valid"}, (
        "Valid split column contains values other than 'valid'"
    )

    excluded = {"patient_id", "split", "adverse"}
    train_features = [column for column in train_df.columns if column not in excluded]
    valid_features = [column for column in valid_df.columns if column not in excluded]
    assert train_features == valid_features, (
        "Feature columns or their order differ between Train and Valid"
    )
    assert len(train_features) == 48, (
        f"Model feature count is {len(train_features)}, expected exactly 48"
    )
    assert excluded.isdisjoint(train_features), "Metadata/label columns entered features"
    forbidden_exact = {"runtime_s", "n_pairs"}
    forbidden_found = [
        column
        for column in train_features
        if column in forbidden_exact
        or column.startswith("post_")
        or column.startswith("delta_")
    ]
    assert not forbidden_found, f"Forbidden model features found: {forbidden_found}"

    non_numeric_train = [
        column
        for column in train_features
        if not pd.api.types.is_numeric_dtype(train_df[column])
    ]
    non_numeric_valid = [
        column
        for column in valid_features
        if not pd.api.types.is_numeric_dtype(valid_df[column])
    ]
    assert not non_numeric_train, f"Non-numeric Train features: {non_numeric_train}"
    assert not non_numeric_valid, f"Non-numeric Valid features: {non_numeric_valid}"
    assert not np.isinf(train_df[train_features].to_numpy(dtype=float)).any(), (
        "Train features contain positive or negative infinity"
    )
    assert not np.isinf(valid_df[valid_features].to_numpy(dtype=float)).any(), (
        "Valid features contain positive or negative infinity"
    )
    return train_features


def build_estimator_and_grid(model_name: str) -> tuple[Any, Any]:
    if model_name == "Dummy":
        return DummyClassifier(strategy="prior"), [{}]
    if model_name == "Logistic":
        estimator = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        solver="liblinear",
                        penalty="l2",
                        max_iter=5000,
                        random_state=SEED,
                    ),
                ),
            ]
        )
        return estimator, CONFIGURATION["models"]["Logistic"]["param_grid"]
    if model_name == "CatBoost":
        estimator = CatBoostClassifier(
            task_type="CPU",
            thread_count=8,
            random_seed=SEED,
            verbose=False,
            allow_writing_files=False,
            loss_function="Logloss",
        )
        return estimator, CONFIGURATION["models"]["CatBoost"]["param_grid"]
    raise ValueError(f"Unknown model: {model_name}")


def safe_auroc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    try:
        if np.unique(y_true).size < 2:
            return math.nan
        return float(roc_auc_score(y_true, probabilities))
    except (ValueError, TypeError):
        return math.nan


def safe_auprc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    try:
        if np.unique(y_true).size < 2:
            return math.nan
        return float(average_precision_score(y_true, probabilities))
    except (ValueError, TypeError):
        return math.nan


def calculate_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else math.nan
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else math.nan
    try:
        brier = float(brier_score_loss(y_true, probabilities))
    except (ValueError, TypeError):
        brier = math.nan
    return {
        "auroc": safe_auroc(y_true, probabilities),
        "auprc": safe_auprc(y_true, probabilities),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "brier_score": brier,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def append_cv_results(
    rows: list[dict[str, Any]],
    model_name: str,
    stage: str,
    outer_fold: int | None,
    search: GridSearchCV,
) -> None:
    results = search.cv_results_
    for index, params in enumerate(results["params"]):
        row: dict[str, Any] = {
            "model": model_name,
            "stage": stage,
            "outer_fold": outer_fold,
            "candidate_index": index,
            "params_json": json.dumps(
                params, ensure_ascii=False, sort_keys=True, default=json_default
            ),
            "mean_test_average_precision": float(results["mean_test_score"][index]),
            "std_test_average_precision": float(results["std_test_score"][index]),
            "rank_test_average_precision": int(results["rank_test_score"][index]),
            "mean_fit_time_s": float(results["mean_fit_time"][index]),
            "std_fit_time_s": float(results["std_fit_time"][index]),
            "mean_score_time_s": float(results["mean_score_time"][index]),
            "std_score_time_s": float(results["std_score_time"][index]),
        }
        if "mean_train_score" in results:
            row["mean_train_average_precision"] = float(
                results["mean_train_score"][index]
            )
            row["std_train_average_precision"] = float(
                results["std_train_score"][index]
            )
        rows.append(row)


def run_nested_oof(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    logger: logging.Logger,
    cv_rows: list[dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    estimator, param_grid = build_estimator_and_grid(model_name)
    outer_cv = StratifiedKFold(
        n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED
    )
    oof_probabilities = np.full(len(y_train), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []

    for outer_fold, (development_indices, holdout_indices) in enumerate(
        outer_cv.split(x_train, y_train), start=1
    ):
        inner_cv = StratifiedKFold(
            n_splits=INNER_FOLDS, shuffle=True, random_state=SEED
        )
        search = GridSearchCV(
            estimator=clone(estimator),
            param_grid=param_grid,
            scoring="average_precision",
            cv=inner_cv,
            refit=True,
            n_jobs=1,
            return_train_score=True,
            error_score="raise",
        )
        search.fit(
            x_train.iloc[development_indices], y_train[development_indices]
        )
        fold_probabilities = search.best_estimator_.predict_proba(
            x_train.iloc[holdout_indices]
        )[:, 1]
        oof_probabilities[holdout_indices] = fold_probabilities
        fold_auroc = safe_auroc(y_train[holdout_indices], fold_probabilities)
        fold_auprc = safe_auprc(y_train[holdout_indices], fold_probabilities)
        fold_brier = float(
            brier_score_loss(y_train[holdout_indices], fold_probabilities)
        )
        fold_row = {
            "model": model_name,
            "outer_fold": outer_fold,
            "development_n": int(len(development_indices)),
            "development_positive": int(y_train[development_indices].sum()),
            "holdout_n": int(len(holdout_indices)),
            "holdout_positive": int(y_train[holdout_indices].sum()),
            "outer_auroc": fold_auroc,
            "outer_auprc": fold_auprc,
            "outer_brier_score": fold_brier,
            "inner_best_average_precision": float(search.best_score_),
            "best_params_json": json.dumps(
                search.best_params_,
                ensure_ascii=False,
                sort_keys=True,
                default=json_default,
            ),
        }
        fold_rows.append(fold_row)
        append_cv_results(
            cv_rows, model_name, "nested_outer_inner_search", outer_fold, search
        )
        logger.info(
            "%s outer fold %d/%d | development=%d (positive=%d) | "
            "holdout=%d (positive=%d) | AUROC=%.6f | AUPRC=%.6f | "
            "Brier=%.6f | inner best AP=%.6f | params=%s",
            model_name,
            outer_fold,
            OUTER_FOLDS,
            len(development_indices),
            int(y_train[development_indices].sum()),
            len(holdout_indices),
            int(y_train[holdout_indices].sum()),
            fold_auroc,
            fold_auprc,
            fold_brier,
            search.best_score_,
            search.best_params_,
        )

    assert np.isfinite(oof_probabilities).all(), (
        f"{model_name} nested CV did not produce a finite OOF probability for every Train row"
    )
    return oof_probabilities, fold_rows


def threshold_search(
    model_name: str, y_true: np.ndarray, probabilities: np.ndarray
) -> tuple[float, pd.DataFrame]:
    candidates = np.unique(
        np.concatenate(
            [
                np.array([0.0, 1.0], dtype=float),
                np.asarray(probabilities, dtype=float),
            ]
        )
    )
    rows: list[dict[str, Any]] = []
    for threshold in candidates:
        metrics = calculate_metrics(y_true, probabilities, float(threshold))
        rows.append(
            {
                "model": model_name,
                "threshold": float(threshold),
                "balanced_accuracy": metrics["balanced_accuracy"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
            }
        )
    frame = pd.DataFrame(rows)
    best_balanced_accuracy = frame["balanced_accuracy"].max()
    tied = frame[
        np.isclose(
            frame["balanced_accuracy"].to_numpy(dtype=float),
            best_balanced_accuracy,
            rtol=0.0,
            atol=1e-12,
        )
    ].copy()
    tied["distance_to_0_5"] = (tied["threshold"] - 0.5).abs()
    tied = tied.sort_values(
        ["distance_to_0_5", "threshold"], ascending=[True, True]
    )
    selected_threshold = float(tied.iloc[0]["threshold"])
    frame["selected"] = np.isclose(
        frame["threshold"].to_numpy(dtype=float),
        selected_threshold,
        rtol=0.0,
        atol=1e-15,
    )
    return selected_threshold, frame


def run_final_train_only_search_and_fit(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    logger: logging.Logger,
    cv_rows: list[dict[str, Any]],
) -> tuple[Any, dict[str, Any], float]:
    estimator, param_grid = build_estimator_and_grid(model_name)
    inner_cv = StratifiedKFold(
        n_splits=INNER_FOLDS, shuffle=True, random_state=SEED
    )
    search = GridSearchCV(
        estimator=clone(estimator),
        param_grid=param_grid,
        scoring="average_precision",
        cv=inner_cv,
        refit=False,
        n_jobs=1,
        return_train_score=True,
        error_score="raise",
    )
    search.fit(x_train, y_train)
    append_cv_results(cv_rows, model_name, "final_full_train_inner_search", None, search)
    final_params = dict(search.best_params_)
    final_inner_score = float(search.best_score_)
    frozen_model = clone(estimator).set_params(**final_params)
    frozen_model.fit(x_train, y_train)
    logger.info(
        "%s final Train-only inner CV | best AP=%.6f | params=%s | "
        "refit once on all Train rows",
        model_name,
        final_inner_score,
        final_params,
    )
    return frozen_model, final_params, final_inner_score


def bootstrap_confidence_intervals(
    model_name: str,
    y_valid: np.ndarray,
    valid_probabilities: np.ndarray,
    threshold: float,
    point_metrics: dict[str, float | int],
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED)
    distributions: dict[str, list[float]] = {metric: [] for metric in METRIC_ORDER}
    attempts = 0
    valid_repeats = 0
    n_patients = len(y_valid)
    while valid_repeats < BOOTSTRAP_REPEATS:
        attempts += 1
        sampled_indices = rng.integers(0, n_patients, size=n_patients)
        sampled_y = y_valid[sampled_indices]
        if np.unique(sampled_y).size < 2:
            continue
        sampled_probabilities = valid_probabilities[sampled_indices]
        metrics = calculate_metrics(sampled_y, sampled_probabilities, threshold)
        for metric in METRIC_ORDER:
            distributions[metric].append(float(metrics[metric]))
        valid_repeats += 1

    rows: list[dict[str, Any]] = []
    for metric in METRIC_ORDER:
        values = np.asarray(distributions[metric], dtype=float)
        rows.append(
            {
                "model": model_name,
                "metric": metric,
                "estimate": float(point_metrics[metric]),
                "ci_lower_2_5": float(np.nanpercentile(values, 2.5)),
                "ci_upper_97_5": float(np.nanpercentile(values, 97.5)),
                "valid_bootstrap_repeats": BOOTSTRAP_REPEATS,
                "total_sampling_attempts": attempts,
                "discarded_single_class_repeats": attempts - BOOTSTRAP_REPEATS,
                "random_seed": SEED,
                "frozen_threshold": threshold,
            }
        )
    logger.info(
        "%s validation bootstrap complete | valid repeats=%d | attempts=%d | "
        "discarded single-class=%d",
        model_name,
        BOOTSTRAP_REPEATS,
        attempts,
        attempts - BOOTSTRAP_REPEATS,
    )
    return rows


def save_environment(path: Path, started_at: datetime) -> None:
    lines = [
        f"experiment=adverse_pre_v1",
        f"generated_at_utc={datetime.now(timezone.utc).isoformat()}",
        f"run_started_at_utc={started_at.isoformat()}",
        f"sys.executable={sys.executable}",
        f"required_interpreter={EXPECTED_PYTHON}",
        f"python_version={platform.python_version()}",
        f"python_implementation={platform.python_implementation()}",
        f"platform={platform.platform()}",
        f"numpy={np.__version__}",
        f"pandas={pd.__version__}",
        f"scipy={scipy.__version__}",
        f"scikit-learn={sklearn.__version__}",
        f"joblib={joblib.__version__}",
        f"matplotlib={matplotlib.__version__}",
        f"catboost={catboost_version}",
        f"catboost_task_type=CPU",
        f"catboost_thread_count=8",
        f"random_seed={SEED}",
        f"command={sys.executable} {Path(__file__).resolve()}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_roc_curves(
    y_train: np.ndarray,
    y_valid: np.ndarray,
    oof_probabilities: dict[str, np.ndarray],
    valid_probabilities: dict[str, np.ndarray],
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, y_true, probability_map, title in [
        (axes[0], y_train, oof_probabilities, "Train nested OOF ROC"),
        (axes[1], y_valid, valid_probabilities, "Frozen-model Valid ROC"),
    ]:
        for model_name in MODEL_ORDER:
            probabilities = probability_map[model_name]
            auc = safe_auroc(y_true, probabilities)
            if np.unique(y_true).size >= 2:
                false_positive_rate, true_positive_rate, _ = roc_curve(
                    y_true, probabilities
                )
                axis.plot(
                    false_positive_rate,
                    true_positive_rate,
                    linewidth=2,
                    label=f"{model_name} (AUROC={auc:.3f})",
                )
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        axis.set(xlabel="False positive rate", ylabel="True positive rate", title=title)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_pr_curves(
    y_train: np.ndarray,
    y_valid: np.ndarray,
    oof_probabilities: dict[str, np.ndarray],
    valid_probabilities: dict[str, np.ndarray],
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, y_true, probability_map, title in [
        (axes[0], y_train, oof_probabilities, "Train nested OOF PR"),
        (axes[1], y_valid, valid_probabilities, "Frozen-model Valid PR"),
    ]:
        prevalence = float(np.mean(y_true))
        for model_name in MODEL_ORDER:
            probabilities = probability_map[model_name]
            precision, recall, _ = precision_recall_curve(y_true, probabilities)
            auprc = safe_auprc(y_true, probabilities)
            axis.plot(
                recall,
                precision,
                linewidth=2,
                label=f"{model_name} (AUPRC={auprc:.3f})",
            )
        axis.axhline(
            prevalence,
            linestyle="--",
            color="gray",
            linewidth=1,
            label=f"Prevalence={prevalence:.3f}",
        )
        axis.set(xlabel="Recall", ylabel="Precision", title=title)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_calibration_curves(
    y_train: np.ndarray,
    y_valid: np.ndarray,
    oof_probabilities: dict[str, np.ndarray],
    valid_probabilities: dict[str, np.ndarray],
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, y_true, probability_map, title in [
        (axes[0], y_train, oof_probabilities, "Train nested OOF calibration"),
        (axes[1], y_valid, valid_probabilities, "Frozen-model Valid calibration"),
    ]:
        for model_name in MODEL_ORDER:
            observed, predicted = calibration_curve(
                y_true,
                probability_map[model_name],
                n_bins=10,
                strategy="quantile",
            )
            axis.plot(predicted, observed, marker="o", linewidth=1.5, label=model_name)
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        axis.set(
            xlabel="Mean predicted probability",
            ylabel="Observed event fraction",
            title=title,
        )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrices(
    train_metrics: dict[str, dict[str, float | int]],
    valid_metrics: dict[str, dict[str, float | int]],
    path: Path,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(12, 8))
    for row_index, (dataset_name, metrics_map) in enumerate(
        [("Train nested OOF", train_metrics), ("Frozen-model Valid", valid_metrics)]
    ):
        for column_index, model_name in enumerate(MODEL_ORDER):
            metrics = metrics_map[model_name]
            matrix = np.array(
                [
                    [metrics["tn"], metrics["fp"]],
                    [metrics["fn"], metrics["tp"]],
                ],
                dtype=int,
            )
            axis = axes[row_index, column_index]
            axis.imshow(matrix, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    axis.text(j, i, str(matrix[i, j]), ha="center", va="center")
            axis.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
            axis.set_yticks([0, 1], labels=["True 0", "True 1"])
            axis.set_title(f"{dataset_name}\n{model_name}")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def markdown_metric_table(
    metrics_map: dict[str, dict[str, float | int]],
) -> str:
    headers = [
        "Model",
        "AUROC",
        "AUPRC",
        "Balanced Acc.",
        "F1",
        "Precision",
        "Sensitivity",
        "Specificity",
        "Brier",
        "TN/FP/FN/TP",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for model_name in MODEL_ORDER:
        metrics = metrics_map[model_name]
        values = [
            model_name,
            f"{float(metrics['auroc']):.4f}",
            f"{float(metrics['auprc']):.4f}",
            f"{float(metrics['balanced_accuracy']):.4f}",
            f"{float(metrics['f1']):.4f}",
            f"{float(metrics['precision']):.4f}",
            f"{float(metrics['sensitivity']):.4f}",
            f"{float(metrics['specificity']):.4f}",
            f"{float(metrics['brier_score']):.4f}",
            f"{metrics['tn']}/{metrics['fp']}/{metrics['fn']}/{metrics['tp']}",
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def markdown_ci_table(ci_frame: pd.DataFrame) -> str:
    lines = [
        "| Model | Metric | Estimate | 95% bootstrap CI |",
        "| --- | --- | ---: | ---: |",
    ]
    for model_name in MODEL_ORDER:
        model_rows = ci_frame[ci_frame["model"] == model_name]
        for metric in METRIC_ORDER:
            row = model_rows[model_rows["metric"] == metric].iloc[0]
            lines.append(
                f"| {model_name} | {metric} | {row['estimate']:.4f} | "
                f"[{row['ci_lower_2_5']:.4f}, {row['ci_upper_97_5']:.4f}] |"
            )
    return "\n".join(lines)


def generate_report(
    path: Path,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_names: list[str],
    train_metrics: dict[str, dict[str, float | int]],
    valid_metrics: dict[str, dict[str, float | int]],
    ci_frame: pd.DataFrame,
    selected_model: str,
    final_params: dict[str, dict[str, Any]],
    thresholds: dict[str, float],
    final_inner_scores: dict[str, float],
    started_at: datetime,
    ended_at: datetime,
) -> None:
    selected_oof = float(train_metrics[selected_model]["auprc"])
    selected_valid = float(valid_metrics[selected_model]["auprc"])
    selected_gap = selected_valid - selected_oof
    dummy_oof = float(train_metrics["Dummy"]["auprc"])
    dummy_valid = float(valid_metrics["Dummy"]["auprc"])
    report = f"""# adverse_pre_v1 experiment report

## Protocol and data isolation

- Train: {len(train_df)} patients, {int(train_df['adverse'].sum())} positive.
- Valid: {len(valid_df)} patients, {int(valid_df['adverse'].sum())} positive.
- Features: {len(feature_names)} preoperative patient-level SEA-RAFT features.
- Train and Valid patient identifiers are disjoint. No sample merging, re-splitting, or exchange was performed.
- All preprocessing, nested cross-validation, hyperparameter selection, model selection, and threshold selection used Train only.
- The primary model-selection criterion was Train nested OOF AUPRC. Each model threshold maximized Train OOF balanced accuracy.
- After freezing model type, final Train-only inner-CV hyperparameters, and OOF thresholds, each final model was fitted on all Train rows and `predict_proba` was called on Valid exactly once.
- Valid was not used for tuning, feature selection, preprocessing estimates, calibration, model selection, or threshold selection.

## Train nested OOF performance

{markdown_metric_table(train_metrics)}

## Frozen-model Valid performance

{markdown_metric_table(valid_metrics)}

## Valid patient-level bootstrap confidence intervals

Each interval uses 2,000 valid two-class patient bootstrap resamples with seed 42. The Train OOF threshold remains frozen in every repeat.

{markdown_ci_table(ci_frame)}

## Frozen decisions

The selected model is **{selected_model}**, based only on the highest Train nested OOF AUPRC ({selected_oof:.6f}). Its Valid AUPRC is {selected_valid:.6f}, for a Valid-minus-OOF difference of {selected_gap:+.6f}.

| Model | Final Train-only inner-CV AP | Frozen OOF threshold | Final hyperparameters |
| --- | ---: | ---: | --- |
"""
    for model_name in MODEL_ORDER:
        report += (
            f"| {model_name} | {final_inner_scores[model_name]:.6f} | "
            f"{thresholds[model_name]:.12g} | "
            f"`{json.dumps(final_params[model_name], ensure_ascii=False, sort_keys=True, default=json_default)}` |\n"
        )
    report += f"""

## Comparison with Dummy and generalization note

- Selected model Train OOF AUPRC minus Dummy: {selected_oof - dummy_oof:+.6f}.
- Selected model Valid AUPRC minus Dummy: {selected_valid - dummy_valid:+.6f}.
- The OOF-to-Valid difference should be interpreted with the bootstrap intervals and the fixed external split; no Valid-driven changes were made.

## Reproducibility

- Required interpreter: `{EXPECTED_PYTHON}`
- Random seed: {SEED}
- Outer CV: 5-fold stratified, shuffled, seed 42.
- Inner CV: 5-fold stratified, shuffled, seed 42; scoring=`average_precision`.
- CatBoost: CPU only, `thread_count=8`, `random_seed=42`, `verbose=False`.
- Started (UTC): {started_at.isoformat()}
- Ended (UTC): {ended_at.isoformat()}
- Elapsed: {(ended_at - started_at).total_seconds():.1f} seconds.

The exact feature list, parameter grids, fold-level search results, threshold search, predictions, fitted models, package versions, plots, and feature interpretation tables are stored in `outputs/baselines/adverse_pre_v1/`.
"""
    path.write_text(report, encoding="utf-8")


def close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def main() -> int:
    if FINAL_OUTPUT_DIR.exists() or FINAL_REPORT_DIR.exists():
        raise FileExistsError(
            "Formal result path already exists; refusing to overwrite: "
            f"{FINAL_OUTPUT_DIR} or {FINAL_REPORT_DIR}"
        )

    FINAL_OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    FINAL_REPORT_DIR.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    run_token = f"{started_at.strftime('%Y%m%dT%H%M%SZ')}_{os.getpid()}"
    work_output_dir = FINAL_OUTPUT_DIR.parent / f".adverse_pre_v1_work_{run_token}"
    work_report_dir = FINAL_REPORT_DIR.parent / f".adverse_pre_v1_work_{run_token}"
    work_output_dir.mkdir(parents=False, exist_ok=False)
    work_report_dir.mkdir(parents=False, exist_ok=False)
    logger = setup_logging(work_output_dir / "run.log")

    try:
        logger.info("Experiment started")
        logger.info("Command: %s %s", sys.executable, Path(__file__).resolve())
        logger.info("Configuration: %s", json.dumps(CONFIGURATION, default=json_default))
        write_json(work_output_dir / "configuration.json", CONFIGURATION)
        save_environment(work_output_dir / "environment.txt", started_at)

        logger.info("Reading Train input: %s", TRAIN_PATH)
        logger.info("Reading Valid input: %s", VALID_PATH)
        train_df = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
        valid_df = pd.read_csv(VALID_PATH, encoding="utf-8-sig")
        feature_names = assert_input_invariants(train_df, valid_df)
        logger.info(
            "Input invariants passed | Train=%d positive=%d | Valid=%d positive=%d | features=%d",
            len(train_df),
            int(train_df["adverse"].sum()),
            len(valid_df),
            int(valid_df["adverse"].sum()),
            len(feature_names),
        )
        logger.info("Feature list: %s", feature_names)
        write_json(
            work_output_dir / "feature_names.json",
            {
                "feature_count": len(feature_names),
                "feature_names": feature_names,
                "excluded_columns": ["patient_id", "split", "adverse"],
                "forbidden_model_columns": [
                    "runtime_s",
                    "n_pairs",
                    "post_*",
                    "delta_*",
                ],
            },
        )

        x_train = train_df.loc[:, feature_names]
        x_valid = valid_df.loc[:, feature_names]
        y_train = train_df["adverse"].astype(int).to_numpy()
        y_valid = valid_df["adverse"].astype(int).to_numpy()

        cv_rows: list[dict[str, Any]] = []
        fold_rows: list[dict[str, Any]] = []
        oof_probabilities: dict[str, np.ndarray] = {}
        thresholds: dict[str, float] = {}
        train_metrics: dict[str, dict[str, float | int]] = {}
        threshold_frames: list[pd.DataFrame] = []

        for model_name in MODEL_ORDER:
            logger.info("Starting nested OOF procedure for %s", model_name)
            probabilities, model_fold_rows = run_nested_oof(
                model_name, x_train, y_train, logger, cv_rows
            )
            oof_probabilities[model_name] = probabilities
            fold_rows.extend(model_fold_rows)
            threshold, search_frame = threshold_search(
                model_name, y_train, probabilities
            )
            thresholds[model_name] = threshold
            threshold_frames.append(search_frame)
            train_metrics[model_name] = calculate_metrics(
                y_train, probabilities, threshold
            )
            logger.info(
                "%s OOF complete | AUPRC=%.6f | AUROC=%.6f | threshold=%.12g | "
                "balanced accuracy=%.6f",
                model_name,
                train_metrics[model_name]["auprc"],
                train_metrics[model_name]["auroc"],
                threshold,
                train_metrics[model_name]["balanced_accuracy"],
            )

        selected_model = max(
            MODEL_ORDER,
            key=lambda name: (
                float(train_metrics[name]["auprc"]),
                -MODEL_ORDER.index(name),
            ),
        )
        logger.info(
            "Model family frozen from Train nested OOF AUPRC only: %s (AUPRC=%.6f)",
            selected_model,
            train_metrics[selected_model]["auprc"],
        )
        for model_name in MODEL_ORDER:
            logger.info(
                "%s threshold frozen from Train OOF only: %.12g",
                model_name,
                thresholds[model_name],
            )

        final_models: dict[str, Any] = {}
        final_params: dict[str, dict[str, Any]] = {}
        final_inner_scores: dict[str, float] = {}
        for model_name in MODEL_ORDER:
            model, params, inner_score = run_final_train_only_search_and_fit(
                model_name, x_train, y_train, logger, cv_rows
            )
            final_models[model_name] = model
            final_params[model_name] = params
            final_inner_scores[model_name] = inner_score

        logger.info(
            "All model families, hyperparameters, and thresholds are frozen; "
            "beginning the single validation probability prediction per model"
        )
        valid_probabilities: dict[str, np.ndarray] = {}
        valid_metrics: dict[str, dict[str, float | int]] = {}
        for model_name in MODEL_ORDER:
            probabilities = final_models[model_name].predict_proba(x_valid)[:, 1]
            assert np.isfinite(probabilities).all(), (
                f"{model_name} produced non-finite Valid probabilities"
            )
            valid_probabilities[model_name] = probabilities
            valid_metrics[model_name] = calculate_metrics(
                y_valid, probabilities, thresholds[model_name]
            )
            logger.info(
                "%s Valid evaluation | AUROC=%.6f | AUPRC=%.6f | "
                "balanced accuracy=%.6f | F1=%.6f | threshold=%.12g",
                model_name,
                valid_metrics[model_name]["auroc"],
                valid_metrics[model_name]["auprc"],
                valid_metrics[model_name]["balanced_accuracy"],
                valid_metrics[model_name]["f1"],
                thresholds[model_name],
            )

        bootstrap_rows: list[dict[str, Any]] = []
        for model_name in MODEL_ORDER:
            bootstrap_rows.extend(
                bootstrap_confidence_intervals(
                    model_name,
                    y_valid,
                    valid_probabilities[model_name],
                    thresholds[model_name],
                    valid_metrics[model_name],
                    logger,
                )
            )
        bootstrap_frame = pd.DataFrame(bootstrap_rows)

        train_oof_frame = pd.DataFrame(
            {
                "patient_id": train_df["patient_id"],
                "split": train_df["split"],
                "adverse": y_train,
            }
        )
        valid_prediction_frame = pd.DataFrame(
            {
                "patient_id": valid_df["patient_id"],
                "split": valid_df["split"],
                "adverse": y_valid,
            }
        )
        for model_name in MODEL_ORDER:
            slug = model_name.lower()
            train_oof_frame[f"{slug}_probability"] = oof_probabilities[model_name]
            train_oof_frame[f"{slug}_prediction"] = (
                oof_probabilities[model_name] >= thresholds[model_name]
            ).astype(int)
            valid_prediction_frame[f"{slug}_probability"] = valid_probabilities[
                model_name
            ]
            valid_prediction_frame[f"{slug}_prediction"] = (
                valid_probabilities[model_name] >= thresholds[model_name]
            ).astype(int)

        train_metric_rows: list[dict[str, Any]] = []
        valid_metric_rows: list[dict[str, Any]] = []
        comparison_rows: list[dict[str, Any]] = []
        oof_rank_order = sorted(
            MODEL_ORDER,
            key=lambda name: (
                -float(train_metrics[name]["auprc"]),
                MODEL_ORDER.index(name),
            ),
        )
        oof_ranks = {name: rank for rank, name in enumerate(oof_rank_order, start=1)}
        for model_name in MODEL_ORDER:
            train_metric_rows.append(
                {
                    "dataset": "train_nested_oof",
                    "model": model_name,
                    "threshold": thresholds[model_name],
                    **train_metrics[model_name],
                }
            )
            valid_metric_rows.append(
                {
                    "dataset": "valid_frozen_once",
                    "model": model_name,
                    "threshold": thresholds[model_name],
                    **valid_metrics[model_name],
                }
            )
            comparison_rows.append(
                {
                    "model": model_name,
                    "train_oof_auprc": train_metrics[model_name]["auprc"],
                    "train_oof_auroc": train_metrics[model_name]["auroc"],
                    "train_oof_balanced_accuracy": train_metrics[model_name][
                        "balanced_accuracy"
                    ],
                    "train_oof_brier_score": train_metrics[model_name]["brier_score"],
                    "oof_auprc_rank": oof_ranks[model_name],
                    "selected_by_train_oof_auprc": model_name == selected_model,
                    "frozen_threshold": thresholds[model_name],
                    "final_train_inner_cv_auprc": final_inner_scores[model_name],
                    "final_params_json": json.dumps(
                        final_params[model_name],
                        ensure_ascii=False,
                        sort_keys=True,
                        default=json_default,
                    ),
                    "valid_auprc_report_only": valid_metrics[model_name]["auprc"],
                    "valid_auroc_report_only": valid_metrics[model_name]["auroc"],
                    "valid_minus_oof_auprc": float(valid_metrics[model_name]["auprc"])
                    - float(train_metrics[model_name]["auprc"]),
                }
            )

        train_oof_frame.to_csv(
            work_output_dir / "train_oof_predictions.csv", index=False, encoding="utf-8"
        )
        valid_prediction_frame.to_csv(
            work_output_dir / "valid_predictions.csv", index=False, encoding="utf-8"
        )
        pd.DataFrame(comparison_rows).to_csv(
            work_output_dir / "model_comparison.csv", index=False, encoding="utf-8"
        )
        pd.DataFrame(valid_metric_rows).to_csv(
            work_output_dir / "valid_metrics.csv", index=False, encoding="utf-8"
        )
        pd.DataFrame(train_metric_rows).to_csv(
            work_output_dir / "train_oof_metrics.csv", index=False, encoding="utf-8"
        )
        bootstrap_frame.to_csv(
            work_output_dir / "bootstrap_confidence_intervals.csv",
            index=False,
            encoding="utf-8",
        )
        pd.DataFrame(cv_rows).to_csv(
            work_output_dir / "cv_results.csv", index=False, encoding="utf-8"
        )
        pd.DataFrame(fold_rows).to_csv(
            work_output_dir / "outer_fold_results.csv", index=False, encoding="utf-8"
        )
        pd.concat(threshold_frames, ignore_index=True).to_csv(
            work_output_dir / "threshold_search.csv", index=False, encoding="utf-8"
        )

        joblib.dump(final_models["Dummy"], work_output_dir / "dummy_prior.joblib")
        joblib.dump(
            final_models["Logistic"], work_output_dir / "logistic_regression.joblib"
        )
        final_models["CatBoost"].save_model(
            str(work_output_dir / "catboost_classifier.cbm")
        )

        logistic_coefficients = pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": final_models["Logistic"]
                .named_steps["model"]
                .coef_[0],
            }
        )
        logistic_coefficients["absolute_coefficient"] = logistic_coefficients[
            "coefficient"
        ].abs()
        logistic_coefficients = logistic_coefficients.sort_values(
            "absolute_coefficient", ascending=False
        )
        logistic_coefficients.to_csv(
            work_output_dir / "logistic_coefficients.csv",
            index=False,
            encoding="utf-8",
        )

        catboost_importance = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": final_models["CatBoost"].get_feature_importance(),
            }
        ).sort_values("importance", ascending=False)
        catboost_importance.to_csv(
            work_output_dir / "catboost_feature_importance.csv",
            index=False,
            encoding="utf-8",
        )

        plot_roc_curves(
            y_train,
            y_valid,
            oof_probabilities,
            valid_probabilities,
            work_output_dir / "roc_curves.png",
        )
        plot_pr_curves(
            y_train,
            y_valid,
            oof_probabilities,
            valid_probabilities,
            work_output_dir / "pr_curves.png",
        )
        plot_calibration_curves(
            y_train,
            y_valid,
            oof_probabilities,
            valid_probabilities,
            work_output_dir / "calibration_curves.png",
        )
        plot_confusion_matrices(
            train_metrics,
            valid_metrics,
            work_output_dir / "confusion_matrices.png",
        )

        ended_at = datetime.now(timezone.utc)
        generate_report(
            work_report_dir / "adverse_pre_report.md",
            train_df,
            valid_df,
            feature_names,
            train_metrics,
            valid_metrics,
            bootstrap_frame,
            selected_model,
            final_params,
            thresholds,
            final_inner_scores,
            started_at,
            ended_at,
        )
        logger.info(
            "Experiment completed successfully | selected model=%s | elapsed=%.1f seconds",
            selected_model,
            (ended_at - started_at).total_seconds(),
        )
        logger.info("Formal output directory: %s", FINAL_OUTPUT_DIR)
        logger.info("Formal report directory: %s", FINAL_REPORT_DIR)
        close_logger(logger)

        if FINAL_OUTPUT_DIR.exists() or FINAL_REPORT_DIR.exists():
            raise FileExistsError(
                "Formal result path appeared during execution; refusing to overwrite"
            )
        work_output_dir.rename(FINAL_OUTPUT_DIR)
        work_report_dir.rename(FINAL_REPORT_DIR)
        print(f"Completed: {FINAL_OUTPUT_DIR}", flush=True)
        print(f"Report: {FINAL_REPORT_DIR / 'adverse_pre_report.md'}", flush=True)
        return 0
    except Exception:
        logger.exception("Critical experiment failure")
        logger.error("Incomplete staging output retained at: %s", work_output_dir)
        logger.error("Incomplete staging report retained at: %s", work_report_dir)
        close_logger(logger)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

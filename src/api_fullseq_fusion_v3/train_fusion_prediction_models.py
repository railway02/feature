#!/usr/bin/env python3
"""Train aligned CAVE + SEA-RAFT downstream prediction models.

The script reads the frozen task artifacts produced by ``api_fullseq_v3`` and
``api_fullseq_cave_v3``.  It never rebuilds labels or image features.  Rows are
accepted only when stable identifiers, row order, targets, and Train/Valid
patient isolation agree exactly between the two sources.

Reported Logistic OOF predictions use patient-grouped outer folds.  Every
imputer, scaler, random projection, PCA, and C selection is fitted strictly
inside the relevant development fold.  Official Valid is transform/evaluate
only.  Any Logistic convergence warning fails the formal run.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.random_projection import SparseRandomProjection

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover
    StratifiedGroupKFold = None


SEED = 42
C_GRID = (0.01, 0.1, 1.0, 10.0)
VARIANTS = (
    "searaft",
    "cave_deep",
    "cave_scalar",
    "cave_fusion",
    "multimodal_fusion",
)
LOGISTIC_SOLVER = "newton-cg"
LOGISTIC_MAX_ITER = 1000
LOGISTIC_TOL = 1e-4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


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
    if len(np.unique(y)) != 2:
        return float("nan")
    return float(roc_auc_score(y, probability))


def safe_ap(y: np.ndarray, probability: np.ndarray) -> float:
    if len(np.unique(y)) != 2:
        return float("nan")
    return float(average_precision_score(y, probability))


def youden_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, probability)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    return float(thresholds[finite][int(np.argmax(tpr[finite] - fpr[finite]))])


def metric_row(
    task: str,
    model: str,
    split: str,
    y: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "task": task,
        "model": model,
        "split": split,
        "rows": int(len(y)),
        "positive": int((y == 1).sum()),
        "negative": int((y == 0).sum()),
        "positive_fraction": float(np.mean(y == 1)),
        "auroc": safe_auc(y, probability),
        "auprc": safe_ap(y, probability),
        "brier": float(brier_score_loss(y, probability)),
        "threshold": float(threshold),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def n_splits_for(y: np.ndarray, groups: np.ndarray, requested: int) -> int:
    return max(
        2,
        min(requested, int(np.bincount(y, minlength=2).min()), len(np.unique(groups))),
    )


def grouped_splits(
    y: np.ndarray, groups: np.ndarray, requested: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    if StratifiedGroupKFold is None:
        raise RuntimeError("StratifiedGroupKFold is required")
    splitter = StratifiedGroupKFold(
        n_splits=n_splits_for(y, groups, requested),
        shuffle=True,
        random_state=seed,
    )
    splits = list(splitter.split(np.zeros(len(y)), y, groups))
    for fit_index, holdout_index in splits:
        overlap = set(groups[fit_index]) & set(groups[holdout_index])
        if overlap:
            raise AssertionError(f"Grouped split leaked {len(overlap)} patients")
    return splits


def make_imputer(add_indicator: bool) -> SimpleImputer:
    try:
        return SimpleImputer(
            strategy="median", add_indicator=add_indicator, keep_empty_features=True
        )
    except TypeError:  # pragma: no cover
        return SimpleImputer(strategy="median", add_indicator=add_indicator)


@dataclass
class NumericBranch:
    requested_components: int
    minimum_finite_fraction: float = 0.25
    keep: np.ndarray | None = None
    imputer: SimpleImputer | None = None
    scaler: RobustScaler | None = None
    pca: PCA | None = None

    def fit(self, values: np.ndarray, seed: int) -> "NumericBranch":
        values = np.asarray(values, dtype=np.float64)
        finite_fraction = np.isfinite(values).mean(axis=0)
        with np.errstate(all="ignore"):
            variances = np.nanvar(values, axis=0)
        self.keep = (
            (finite_fraction >= self.minimum_finite_fraction)
            & np.isfinite(variances)
            & (variances > 1e-12)
        )
        if not self.keep.any():
            raise AssertionError("No usable numeric columns")
        self.imputer = make_imputer(add_indicator=True)
        transformed = self.imputer.fit_transform(values[:, self.keep])
        self.scaler = RobustScaler()
        transformed = self.scaler.fit_transform(transformed)
        n_components = min(
            self.requested_components, transformed.shape[0] - 1, transformed.shape[1]
        )
        if n_components < 1:
            raise AssertionError("Insufficient data for numeric PCA")
        solver = "randomized" if n_components < min(transformed.shape) else "full"
        self.pca = PCA(n_components=n_components, svd_solver=solver, random_state=seed)
        self.pca.fit(transformed)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.keep is None or self.imputer is None or self.scaler is None or self.pca is None:
            raise RuntimeError("Numeric branch not fitted")
        transformed = self.imputer.transform(
            np.asarray(values, dtype=np.float64)[:, self.keep]
        )
        transformed = self.scaler.transform(transformed)
        return self.pca.transform(transformed).astype(np.float64)

    def audit(self) -> dict[str, Any]:
        indicators = 0
        if self.imputer is not None and getattr(self.imputer, "indicator_", None) is not None:
            indicators = int(len(self.imputer.indicator_.features_))
        return {
            "kept_columns": int(self.keep.sum()) if self.keep is not None else 0,
            "imputer_indicator_columns": indicators,
            "components": int(self.pca.n_components_) if self.pca is not None else 0,
            "explained_variance": (
                float(self.pca.explained_variance_ratio_.sum())
                if self.pca is not None else None
            ),
        }


@dataclass
class DeepBranch:
    projection_components: int = 512
    pca_components: int = 64
    projector: SparseRandomProjection | None = None
    scaler: StandardScaler | None = None
    pca: PCA | None = None

    def fit(self, values: np.ndarray, seed: int) -> "DeepBranch":
        x = np.nan_to_num(
            np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        self.projector = SparseRandomProjection(
            n_components=min(self.projection_components, x.shape[1]),
            density="auto",
            dense_output=True,
            random_state=seed,
        )
        projected = self.projector.fit_transform(x).astype(np.float32)
        self.scaler = StandardScaler()
        projected = self.scaler.fit_transform(projected)
        n_components = min(
            self.pca_components, projected.shape[0] - 1, projected.shape[1]
        )
        if n_components < 1:
            raise AssertionError("Insufficient data for deep PCA")
        self.pca = PCA(
            n_components=n_components, svd_solver="randomized", random_state=seed + 1
        )
        self.pca.fit(projected)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.projector is None or self.scaler is None or self.pca is None:
            raise RuntimeError("Deep branch not fitted")
        x = np.nan_to_num(
            np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        projected = self.projector.transform(x).astype(np.float32)
        projected = self.scaler.transform(projected)
        return self.pca.transform(projected).astype(np.float64)

    def audit(self) -> dict[str, Any]:
        return {
            "input_dimension": int(self.projector.n_features_in_) if self.projector else 0,
            "random_projection_components": (
                int(self.projector.n_components) if self.projector else 0
            ),
            "pca_components": int(self.pca.n_components_) if self.pca else 0,
            "explained_variance": (
                float(self.pca.explained_variance_ratio_.sum()) if self.pca else None
            ),
        }


@dataclass
class MultimodalPreprocessor:
    cave_deep: DeepBranch | None = None
    cave_scalar: NumericBranch | None = None
    searaft: NumericBranch | None = None
    final_scalers: dict[str, StandardScaler] = field(default_factory=dict)

    def fit(self, data: dict[str, np.ndarray], seed: int) -> "MultimodalPreprocessor":
        self.cave_deep = DeepBranch().fit(data["cave_deep"], seed)
        self.cave_scalar = NumericBranch(32).fit(data["cave_scalar"], seed + 1000)
        self.searaft = NumericBranch(64).fit(data["searaft"], seed + 2000)
        base = self._base_transform_all(data)
        self.final_scalers = {
            variant: StandardScaler().fit(np.asarray(base[variant], dtype=np.float64))
            for variant in VARIANTS
        }
        self.transform_all(data)
        return self

    def _base_transform_all(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.cave_deep is None or self.cave_scalar is None or self.searaft is None:
            raise RuntimeError("Multimodal preprocessor not fitted")
        deep = self.cave_deep.transform(data["cave_deep"])
        cave_scalar = self.cave_scalar.transform(data["cave_scalar"])
        searaft = self.searaft.transform(data["searaft"])
        missing = np.asarray(data["missing"], dtype=np.float64)
        result = {
            "searaft": np.concatenate([searaft, missing], axis=1),
            "cave_deep": np.concatenate([deep, missing], axis=1),
            "cave_scalar": np.concatenate([cave_scalar, missing], axis=1),
            "cave_fusion": np.concatenate([deep, cave_scalar, missing], axis=1),
            "multimodal_fusion": np.concatenate(
                [deep, cave_scalar, searaft, missing], axis=1
            ),
        }
        if any(not np.isfinite(value).all() for value in result.values()):
            raise AssertionError("Reduced multimodal features contain nonfinite values")
        return result

    def transform_all(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if set(self.final_scalers) != set(VARIANTS):
            raise RuntimeError("Final variant scalers not fitted")
        base = self._base_transform_all(data)
        result = {
            variant: self.final_scalers[variant].transform(base[variant]).astype(
                np.float64, copy=False
            )
            for variant in VARIANTS
        }
        if any(not np.isfinite(value).all() for value in result.values()):
            raise AssertionError("Final scaled multimodal features contain nonfinite values")
        return result

    def audit(self) -> dict[str, Any]:
        return {
            "cave_deep": self.cave_deep.audit() if self.cave_deep else None,
            "cave_scalar": self.cave_scalar.audit() if self.cave_scalar else None,
            "searaft": self.searaft.audit() if self.searaft else None,
            "post_concatenation_standard_scaler": True,
            "final_dimensions": {
                variant: int(len(scaler.mean_))
                for variant, scaler in self.final_scalers.items()
            },
            "constant_columns_mapped_to_zero": {
                variant: int(np.sum(np.asarray(scaler.var_) == 0.0))
                for variant, scaler in self.final_scalers.items()
            },
        }


def subset_data(data: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {
        key: value[indices]
        for key, value in data.items()
        if key in {"cave_deep", "cave_scalar", "searaft", "missing", "target"}
    }


def stable_keys(searaft: pd.DataFrame, cave: pd.DataFrame) -> list[str]:
    if "record_uid" in searaft.columns and "record_uid" in cave.columns:
        keys = ["record_uid", "series_uid", "patient_id"]
    else:
        keys = ["patient_id"]
    missing = [key for key in keys if key not in searaft.columns or key not in cave.columns]
    if missing:
        raise KeyError(f"Missing stable alignment keys: {missing}")
    return keys


def load_split(
    searaft_task_dir: Path,
    cave_task_dir: Path,
    split: str,
    searaft_features: list[str],
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    searaft_frame = pd.read_csv(
        searaft_task_dir / f"{split}.csv", dtype={"patient_id": str}
    )
    cave_meta = pd.read_csv(
        cave_task_dir / f"{split}_meta.csv", dtype={"patient_id": str}
    )
    with np.load(cave_task_dir / f"{split}_features.npz") as raw:
        cave_arrays = {
            key: np.array(raw[key], copy=True)
            for key in ("deep", "scalar", "missing", "target")
        }
    if len(searaft_frame) != len(cave_meta) or len(cave_meta) != len(cave_arrays["target"]):
        raise AssertionError(f"{split}: source row counts differ")
    keys = stable_keys(searaft_frame, cave_meta)
    for key in keys:
        left = searaft_frame[key].fillna("").astype(str).to_numpy()
        right = cave_meta[key].fillna("").astype(str).to_numpy()
        if not np.array_equal(left, right):
            mismatch = int(np.flatnonzero(left != right)[0])
            raise AssertionError(
                f"{split}: row alignment mismatch for {key} at row {mismatch}"
            )
    searaft_target = pd.to_numeric(
        searaft_frame["target"], errors="raise"
    ).astype(int).to_numpy()
    cave_meta_target = pd.to_numeric(
        cave_meta["target"], errors="raise"
    ).astype(int).to_numpy()
    cave_target = cave_arrays["target"].astype(int)
    if not (
        np.array_equal(searaft_target, cave_meta_target)
        and np.array_equal(searaft_target, cave_target)
    ):
        raise AssertionError(f"{split}: source targets differ")
    missing_features = set(searaft_features) - set(searaft_frame.columns)
    if missing_features:
        raise KeyError(f"{split}: missing SEA-RAFT features: {sorted(missing_features)[:5]}")
    searaft_values = searaft_frame[searaft_features].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float32)
    data = {
        "cave_deep": cave_arrays["deep"].astype(np.float32, copy=False),
        "cave_scalar": cave_arrays["scalar"].astype(np.float32, copy=False),
        "searaft": searaft_values,
        "missing": cave_arrays["missing"].astype(np.float32, copy=False),
        "target": cave_target,
    }
    audit = {
        "split": split,
        "rows": int(len(cave_meta)),
        "patients": int(cave_meta["patient_id"].astype(str).nunique()),
        "positive": int((cave_target == 1).sum()),
        "negative": int((cave_target == 0).sum()),
        "stable_keys": keys,
        "row_order_identical": True,
        "targets_identical": True,
        "cave_deep_shape": list(data["cave_deep"].shape),
        "cave_scalar_shape": list(data["cave_scalar"].shape),
        "searaft_shape": list(data["searaft"].shape),
        "missing_shape": list(data["missing"].shape),
    }
    return cave_meta, data, audit


def load_aligned_task(
    searaft_task_dir: Path, cave_task_dir: Path
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    searaft_config = json.loads(
        (searaft_task_dir / "task_config.json").read_text(encoding="utf-8")
    )
    cave_config = json.loads(
        (cave_task_dir / "task_config.json").read_text(encoding="utf-8")
    )
    if searaft_config["task_name"] != cave_config["task_name"]:
        raise AssertionError("Task names differ between sources")
    features = list(searaft_config["feature_columns"])
    train_meta, train, train_audit = load_split(
        searaft_task_dir, cave_task_dir, "train", features
    )
    valid_meta, valid, valid_audit = load_split(
        searaft_task_dir, cave_task_dir, "valid", features
    )
    overlap = set(train_meta["patient_id"].astype(str)) & set(
        valid_meta["patient_id"].astype(str)
    )
    if overlap:
        raise AssertionError(f"Train/Valid patient overlap={len(overlap)}")
    config = {
        "task_name": searaft_config["task_name"],
        "task_level": searaft_config.get("task_level"),
        "label_definition": searaft_config.get("label_definition"),
        "searaft_feature_count": len(features),
        "cave_scalar_feature_count": int(train["cave_scalar"].shape[1]),
        "cave_deep_feature_count": int(train["cave_deep"].shape[1]),
    }
    audit = {
        "task_name": config["task_name"],
        "train": train_audit,
        "valid": valid_audit,
        "train_valid_patient_overlap": 0,
    }
    return config, train_meta, valid_meta, train, valid, audit


def fit_logistic_checked(
    x_train: np.ndarray,
    y_train: np.ndarray,
    c_value: float,
    context: dict[str, Any],
    audit_rows: list[dict[str, Any]],
    audit_path: Path,
    x_evaluation: np.ndarray | None = None,
    y_evaluation: np.ndarray | None = None,
) -> LogisticRegression:
    x_train = np.asarray(x_train, dtype=np.float64)
    x_evaluation = (
        None if x_evaluation is None else np.asarray(x_evaluation, dtype=np.float64)
    )
    model = LogisticRegression(
        C=float(c_value),
        class_weight="balanced",
        penalty="l2",
        solver=LOGISTIC_SOLVER,
        max_iter=LOGISTIC_MAX_ITER,
        tol=LOGISTIC_TOL,
    )
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x_train, y_train)
    elapsed = time.perf_counter() - start
    convergence = [item for item in caught if issubclass(item.category, ConvergenceWarning)]
    train_probability = model.predict_proba(x_train)[:, 1]
    evaluation_probability = (
        model.predict_proba(x_evaluation)[:, 1] if x_evaluation is not None else None
    )
    row = {
        **context,
        "C": float(c_value),
        "solver": LOGISTIC_SOLVER,
        "tol": LOGISTIC_TOL,
        "max_iter": LOGISTIC_MAX_ITER,
        "n_iter_": json.dumps(np.asarray(model.n_iter_).astype(int).tolist()),
        "convergence_warning": bool(convergence),
        "warning_text": " || ".join(
            f"{item.category.__name__}: {item.message}" for item in caught
        ),
        "fit_seconds": float(elapsed),
        "coefficient_max_abs": float(np.max(np.abs(model.coef_))),
        "coefficient_l2_norm": float(np.linalg.norm(model.coef_)),
        "train_auroc": safe_auc(y_train, train_probability),
        "train_auprc": safe_ap(y_train, train_probability),
        "evaluation_auroc": (
            safe_auc(y_evaluation, evaluation_probability)
            if evaluation_probability is not None and y_evaluation is not None
            else float("nan")
        ),
        "evaluation_auprc": (
            safe_ap(y_evaluation, evaluation_probability)
            if evaluation_probability is not None and y_evaluation is not None
            else float("nan")
        ),
        "input_rows": int(x_train.shape[0]),
        "input_columns": int(x_train.shape[1]),
        "input_abs_max": float(np.max(np.abs(x_train))),
        "input_column_std_max": float(np.std(x_train, axis=0).max()),
    }
    audit_rows.append(row)
    atomic_csv(pd.DataFrame(audit_rows), audit_path)
    print("[LOGISTIC] " + json.dumps(row, sort_keys=True), flush=True)
    if convergence:
        raise RuntimeError(
            "Logistic convergence failure: "
            + " ".join(f"{key}={value}" for key, value in context.items())
        )
    return model


def select_c_nested(
    raw_data: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    split_seed: int,
    preprocess_seed: int,
    task: str,
    outer_fold: int,
    audit_rows: list[dict[str, Any]],
    audit_path: Path,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    predictions = {
        variant: {
            float(c_value): np.full(len(y), np.nan, dtype=np.float64)
            for c_value in C_GRID
        }
        for variant in VARIANTS
    }
    inner_splits = grouped_splits(y, groups, requested=3, seed=split_seed)
    for inner_fold, (fit_index, holdout_index) in enumerate(inner_splits, start=1):
        fit_data = subset_data(raw_data, fit_index)
        holdout_data = subset_data(raw_data, holdout_index)
        preprocessor = MultimodalPreprocessor().fit(
            fit_data, preprocess_seed + inner_fold * 10
        )
        fit_x = preprocessor.transform_all(fit_data)
        holdout_x = preprocessor.transform_all(holdout_data)
        for variant in VARIANTS:
            for c_value in C_GRID:
                model = fit_logistic_checked(
                    fit_x[variant],
                    y[fit_index],
                    float(c_value),
                    {
                        "task": task,
                        "outer_fold": outer_fold,
                        "stage": "inner_cv",
                        "variant": variant,
                        "inner_fold": inner_fold,
                    },
                    audit_rows,
                    audit_path,
                    holdout_x[variant],
                    y[holdout_index],
                )
                predictions[variant][float(c_value)][holdout_index] = model.predict_proba(
                    holdout_x[variant]
                )[:, 1]
    scores: dict[str, dict[str, float]] = {}
    selected: dict[str, float] = {}
    for variant in VARIANTS:
        if any(
            not np.isfinite(predictions[variant][float(c_value)]).all()
            for c_value in C_GRID
        ):
            raise AssertionError(f"Incomplete inner OOF predictions for {variant}")
        scores[variant] = {
            str(c_value): safe_ap(y, predictions[variant][float(c_value)])
            for c_value in C_GRID
        }
        selected[variant] = float(
            max(C_GRID, key=lambda value: (scores[variant][str(value)], -float(value)))
        )
    return selected, scores


class MLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(1)


def early_stop_split(
    y: np.ndarray, groups: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    for offset in range(80):
        fit_index, stop_index = next(
            GroupShuffleSplit(
                n_splits=1, test_size=0.15, random_state=seed + offset
            ).split(np.zeros(len(y)), y, groups)
        )
        if len(np.unique(y[fit_index])) == 2 and len(np.unique(y[stop_index])) == 2:
            if set(groups[fit_index]) & set(groups[stop_index]):
                raise AssertionError("MLP early-stop patient leakage")
            return fit_index, stop_index
    raise RuntimeError("Could not create a grouped MLP early-stop split with both classes")


@dataclass
class MLPResult:
    model: MLP
    preprocessor: MultimodalPreprocessor
    holdout_probability: np.ndarray
    valid_probability: np.ndarray
    best_epoch: int
    best_validation_ap: float


def fit_mlp_fold(
    development_data: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    holdout_data: dict[str, np.ndarray],
    valid_data: dict[str, np.ndarray],
    device: torch.device,
    seed: int,
) -> MLPResult:
    set_seed(seed)
    fit_index, stop_index = early_stop_split(y, groups, seed)
    fit_data = subset_data(development_data, fit_index)
    stop_data = subset_data(development_data, stop_index)
    preprocessor = MultimodalPreprocessor().fit(fit_data, seed + 10000)
    x_fit = preprocessor.transform_all(fit_data)["multimodal_fusion"].astype(np.float32)
    x_stop = preprocessor.transform_all(stop_data)["multimodal_fusion"].astype(np.float32)
    x_holdout = preprocessor.transform_all(holdout_data)["multimodal_fusion"].astype(np.float32)
    x_valid = preprocessor.transform_all(valid_data)["multimodal_fusion"].astype(np.float32)
    y_fit = y[fit_index].astype(np.float32)
    y_stop = y[stop_index].astype(int)
    model = MLP(x_fit.shape[1]).to(device)
    positive = max(int((y_fit == 1).sum()), 1)
    negative = max(int((y_fit == 0).sum()), 1)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [negative / positive], dtype=torch.float32, device=device
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(x_fit), torch.from_numpy(y_fit)
        ),
        batch_size=min(32, len(x_fit)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    stop_tensor = torch.from_numpy(x_stop).to(device)
    best_state: dict[str, torch.Tensor] | None = None
    best_ap = -math.inf
    best_epoch = 0
    stale = 0
    for epoch in range(1, 121):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            probability = torch.sigmoid(model(stop_tensor)).cpu().numpy()
        score = safe_ap(y_stop, probability)
        if score > best_ap + 1e-6:
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_ap = score
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= 15:
            break
    if best_state is None:
        raise RuntimeError("MLP did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    with torch.no_grad():
        holdout_probability = torch.sigmoid(
            model(torch.from_numpy(x_holdout).to(device))
        ).cpu().numpy().astype(np.float64)
        valid_probability = torch.sigmoid(
            model(torch.from_numpy(x_valid).to(device))
        ).cpu().numpy().astype(np.float64)
    return MLPResult(
        model=model,
        preprocessor=preprocessor,
        holdout_probability=holdout_probability,
        valid_probability=valid_probability,
        best_epoch=best_epoch,
        best_validation_ap=float(best_ap),
    )


def train_task(
    searaft_task_dir: Path,
    cave_task_dir: Path,
    output_root: Path,
    device: torch.device,
    skip_mlp: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config, train_meta, valid_meta, train, valid, alignment_audit = load_aligned_task(
        searaft_task_dir, cave_task_dir
    )
    task_name = config["task_name"]
    output = output_root / task_name
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(alignment_audit, output / "alignment_audit.json")
    y_train = train["target"].astype(int)
    y_valid = valid["target"].astype(int)
    groups = train_meta["patient_id"].astype(str).to_numpy()
    if len(np.unique(y_train)) != 2 or len(np.unique(y_valid)) != 2:
        raise AssertionError(f"{task_name}: both splits require both classes")
    folds = grouped_splits(y_train, groups, requested=5, seed=SEED)
    logistic_audit_path = output / "logistic_convergence_audit.csv"
    logistic_audits: list[dict[str, Any]] = []
    logistic_oof = {
        variant: np.full(len(y_train), np.nan, dtype=np.float64)
        for variant in VARIANTS
    }
    logistic_valid = {variant: [] for variant in VARIANTS}
    mlp_oof = np.full(len(y_train), np.nan, dtype=np.float64)
    mlp_valid: list[np.ndarray] = []
    fold_audits: list[dict[str, Any]] = []

    for fold, (development, holdout) in enumerate(folds, start=1):
        print(
            f"[FOLD START] task={task_name} fold={fold}/{len(folds)} "
            f"development={len(development)} holdout={len(holdout)}",
            flush=True,
        )
        development_data = subset_data(train, development)
        holdout_data = subset_data(train, holdout)
        selected_c, c_scores = select_c_nested(
            development_data,
            y_train[development],
            groups[development],
            split_seed=SEED + fold * 1000,
            preprocess_seed=SEED + fold * 100,
            task=task_name,
            outer_fold=fold,
            audit_rows=logistic_audits,
            audit_path=logistic_audit_path,
        )
        preprocessor = MultimodalPreprocessor().fit(
            development_data, SEED + fold * 100
        )
        development_x = preprocessor.transform_all(development_data)
        holdout_x = preprocessor.transform_all(holdout_data)
        valid_x = preprocessor.transform_all(valid)
        fold_dir = output / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, fold_dir / "preprocessor.joblib")
        for variant in VARIANTS:
            model = fit_logistic_checked(
                development_x[variant],
                y_train[development],
                selected_c[variant],
                {
                    "task": task_name,
                    "outer_fold": fold,
                    "stage": "outer_development_refit",
                    "variant": variant,
                    "inner_fold": 0,
                },
                logistic_audits,
                logistic_audit_path,
                holdout_x[variant],
                y_train[holdout],
            )
            logistic_oof[variant][holdout] = model.predict_proba(
                holdout_x[variant]
            )[:, 1]
            logistic_valid[variant].append(
                model.predict_proba(valid_x[variant])[:, 1]
            )
            joblib.dump(model, fold_dir / f"logistic_{variant}.joblib")
            fold_audits.append({
                "task": task_name,
                "fold": fold,
                "model": f"Logistic_{variant}",
                "development_rows": int(len(development)),
                "holdout_rows": int(len(holdout)),
                "best_c": selected_c[variant],
                "inner_ap_by_c": json.dumps(c_scores[variant], sort_keys=True),
                "preprocessor": json.dumps(preprocessor.audit(), sort_keys=True),
            })
        if not skip_mlp:
            mlp_result = fit_mlp_fold(
                development_data,
                y_train[development],
                groups[development],
                holdout_data,
                valid,
                device,
                SEED + fold,
            )
            mlp_oof[holdout] = mlp_result.holdout_probability
            mlp_valid.append(mlp_result.valid_probability)
            joblib.dump(
                mlp_result.preprocessor, fold_dir / "mlp_preprocessor.joblib"
            )
            torch.save({
                "input_dim": int(
                    mlp_result.model.network[0].in_features
                ),
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in mlp_result.model.state_dict().items()
                },
                "best_epoch": mlp_result.best_epoch,
                "best_validation_ap": mlp_result.best_validation_ap,
            }, fold_dir / "mlp_multimodal_fusion.pt")
            fold_audits.append({
                "task": task_name,
                "fold": fold,
                "model": "MLP_multimodal_fusion",
                "development_rows": int(len(development)),
                "holdout_rows": int(len(holdout)),
                "best_epoch": mlp_result.best_epoch,
                "early_stop_validation_ap": mlp_result.best_validation_ap,
                "preprocessor_scope": "grouped early-stop fit subset only",
                "preprocessor": json.dumps(
                    mlp_result.preprocessor.audit(), sort_keys=True
                ),
            })

    probabilities: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "Dummy": (
            np.full(len(y_train), y_train.mean(), dtype=np.float64),
            np.full(len(y_valid), y_train.mean(), dtype=np.float64),
        )
    }
    for variant in VARIANTS:
        if not np.isfinite(logistic_oof[variant]).all():
            raise AssertionError(f"Incomplete Logistic_{variant} OOF")
        probabilities[f"Logistic_{variant}"] = (
            logistic_oof[variant],
            np.mean(np.stack(logistic_valid[variant]), axis=0),
        )
    if not skip_mlp:
        if not np.isfinite(mlp_oof).all():
            raise AssertionError("Incomplete MLP OOF")
        probabilities["MLP_multimodal_fusion"] = (
            mlp_oof,
            np.mean(np.stack(mlp_valid), axis=0),
        )

    full_selected_c, full_scores = select_c_nested(
        train,
        y_train,
        groups,
        split_seed=SEED + 9100,
        preprocess_seed=SEED + 9000,
        task=task_name,
        outer_fold=0,
        audit_rows=logistic_audits,
        audit_path=logistic_audit_path,
    )
    full_preprocessor = MultimodalPreprocessor().fit(train, SEED + 9000)
    full_x = full_preprocessor.transform_all(train)
    joblib.dump(full_preprocessor, output / "full_train_preprocessor.joblib")
    for variant in VARIANTS:
        model = fit_logistic_checked(
            full_x[variant],
            y_train,
            full_selected_c[variant],
            {
                "task": task_name,
                "outer_fold": 0,
                "stage": "full_train_refit",
                "variant": variant,
                "inner_fold": 0,
            },
            logistic_audits,
            logistic_audit_path,
        )
        joblib.dump({
            "model": model,
            "selected_c": full_selected_c[variant],
            "cv_ap_by_c": full_scores[variant],
        }, output / f"full_train_logistic_{variant}.joblib")

    metrics: list[dict[str, Any]] = []
    thresholds: dict[str, float] = {}
    train_predictions = train_meta.copy()
    valid_predictions = valid_meta.copy()
    for model_name, (oof_probability, valid_probability) in probabilities.items():
        threshold = youden_threshold(y_train, oof_probability)
        thresholds[model_name] = threshold
        metrics.extend([
            metric_row(
                task_name, model_name, "Train_OOF", y_train, oof_probability, threshold
            ),
            metric_row(
                task_name, model_name, "Valid", y_valid, valid_probability, threshold
            ),
        ])
        train_predictions[f"{model_name.lower()}_probability"] = oof_probability
        valid_predictions[f"{model_name.lower()}_probability"] = valid_probability
    metrics_frame = pd.DataFrame(metrics)
    atomic_csv(metrics_frame, output / "metrics.csv")
    atomic_csv(train_predictions, output / "train_oof_predictions.csv")
    atomic_csv(valid_predictions, output / "valid_predictions.csv")
    atomic_csv(pd.DataFrame(fold_audits), output / "fold_audit.csv")
    learned_oof = metrics_frame[
        (metrics_frame["split"] == "Train_OOF") & (metrics_frame["model"] != "Dummy")
    ]
    best_model = str(
        learned_oof.sort_values(["auprc", "auroc"], ascending=False).iloc[0]["model"]
    )
    run_config = {
        "version": "api_fullseq_fusion_v3_models_1",
        **config,
        "train_rows": int(len(y_train)),
        "valid_rows": int(len(y_valid)),
        "outer_folds": len(folds),
        "models": list(probabilities),
        "best_model_selected_by_train_oof_auprc": best_model,
        "thresholds_from_train_oof": thresholds,
        "c_grid": list(C_GRID),
        "cave_deep_random_projection_components": 512,
        "cave_deep_pca_components": 64,
        "cave_scalar_pca_components": 32,
        "searaft_pca_components": 64,
        "post_concatenation_standard_scaler": True,
        "logistic_solver": LOGISTIC_SOLVER,
        "logistic_dtype": "float64",
        "logistic_convergence_warning_policy": "record_then_fail",
        "inner_c_search_preprocessing_scope": "inner-development only",
        "outer_oof_preprocessing_scope": "outer-development only",
        "mlp_early_stop_preprocessing_scope": (
            None if skip_mlp else "grouped early-stop fit subset only"
        ),
        "valid_used_for_fitting_selection_early_stopping_or_threshold": False,
        "device": str(device),
        "seed": SEED,
    }
    atomic_json(run_config, output / "run_config.json")
    atomic_json(run_config, output / ".SUCCESS")
    return run_config, alignment_audit


def fusion_gain_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (task, split), group in metrics.groupby(["task", "split"], sort=True):
        indexed = group.set_index("model")
        fusion_name = "Logistic_multimodal_fusion"
        if fusion_name not in indexed.index:
            continue
        fusion = indexed.loc[fusion_name]
        row: dict[str, Any] = {
            "task": task,
            "split": split,
            "fusion_model": fusion_name,
            "fusion_auroc": float(fusion["auroc"]),
            "fusion_auprc": float(fusion["auprc"]),
            "fusion_brier": float(fusion["brier"]),
        }
        for baseline in ("Logistic_searaft", "Logistic_cave_fusion"):
            if baseline in indexed.index:
                reference = indexed.loc[baseline]
                suffix = baseline.removeprefix("Logistic_")
                row[f"delta_auroc_vs_{suffix}"] = float(
                    fusion["auroc"] - reference["auroc"]
                )
                row[f"delta_auprc_vs_{suffix}"] = float(
                    fusion["auprc"] - reference["auprc"]
                )
                row[f"delta_brier_vs_{suffix}"] = float(
                    fusion["brier"] - reference["brier"]
                )
        rows.append(row)
    return pd.DataFrame(rows)


def discover_tasks(searaft_root: Path, cave_root: Path) -> list[str]:
    searaft_tasks = {
        path.name
        for path in searaft_root.iterdir()
        if path.is_dir() and (path / "task_config.json").is_file()
    }
    cave_tasks = {
        path.name
        for path in cave_root.iterdir()
        if path.is_dir() and (path / "task_config.json").is_file()
    }
    tasks = sorted(searaft_tasks & cave_tasks)
    if not tasks:
        raise FileNotFoundError("No shared task directories")
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--searaft-task-root", required=True)
    parser.add_argument("--cave-task-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--skip-mlp", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    searaft_root = Path(args.searaft_task_root).resolve()
    cave_root = Path(args.cave_task_root).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)
    available_tasks = discover_tasks(searaft_root, cave_root)
    tasks = available_tasks if not args.tasks else list(args.tasks)
    missing_tasks = set(tasks) - set(available_tasks)
    if missing_tasks:
        raise KeyError(f"Tasks unavailable in both sources: {sorted(missing_tasks)}")
    device = torch.device(args.device)
    if not args.audit_only and not args.skip_mlp:
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")

    alignment: dict[str, Any] = {}
    if args.audit_only:
        for task in tasks:
            *_, audit = load_aligned_task(searaft_root / task, cave_root / task)
            alignment[task] = audit
        summary = {
            "version": "api_fullseq_fusion_v3_alignment_1",
            "tasks": tasks,
            "all_sources_aligned": True,
            "alignment": alignment,
        }
        atomic_json(summary, output / "alignment_audit.json")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    results: dict[str, Any] = {}
    all_metrics: list[pd.DataFrame] = []
    for task in tasks:
        result, task_alignment = train_task(
            searaft_root / task,
            cave_root / task,
            output,
            device,
            args.skip_mlp,
        )
        results[task] = result
        alignment[task] = task_alignment
        all_metrics.append(pd.read_csv(output / task / "metrics.csv"))
    combined = pd.concat(all_metrics, ignore_index=True)
    atomic_csv(combined, output / "all_task_metrics.csv")
    atomic_csv(fusion_gain_table(combined), output / "fusion_gains.csv")
    atomic_json(alignment, output / "alignment_audit.json")
    summary = {
        "version": "api_fullseq_fusion_v3_models_1",
        "tasks": tasks,
        "models": results,
        "valid_used_for_training": False,
        "device": str(device),
        "seed": SEED,
        "skip_mlp": bool(args.skip_mlp),
        "all_sources_aligned": True,
        "post_concatenation_standard_scaler": True,
        "logistic_solver": LOGISTIC_SOLVER,
        "logistic_convergence_warning_policy": "record_then_fail",
    }
    atomic_json(summary, output / "summary.json")
    atomic_json(summary, output / ".MODELS_SUCCESS")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


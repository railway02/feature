#!/usr/bin/env python3
"""Formal grouped-CV training for strict Pre+Post Local-CAVE adverse outcome.

Models
------
- Dummy prior
- Logistic Deep: fold-local PCA on Pre+Post 10240-D embeddings
- Logistic Fusion: fold-local deep PCA + fold-local scalar preprocessing/PCA
- MLP Deep: grouped inner early stopping, 3-seed outer-fold ensemble
- MLP Fusion: grouped inner early stopping, 3-seed outer-fold ensemble

Leakage controls
----------------
- Outer folds are frozen in the task builder and grouped by patient_id.
- Every imputer, clip bound, scaler, PCA and hyperparameter decision is fitted
  inside the relevant Train development subset.
- Official Valid is prediction/evaluation only.
- Decision thresholds are selected from pooled Train OOF predictions only.
- Valid uncertainty uses patient-cluster bootstrap because multiple lesion
  records may belong to one patient.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

if not os.environ.get("OMP_NUM_THREADS", "").isdigit():
    os.environ["OMP_NUM_THREADS"] = "8"
if not os.environ.get("MKL_NUM_THREADS", "").isdigit():
    os.environ["MKL_NUM_THREADS"] = "8"
if not os.environ.get("OPENBLAS_NUM_THREADS", "").isdigit():
    os.environ["OPENBLAS_NUM_THREADS"] = "8"
if not os.environ.get("NUMEXPR_NUM_THREADS", "").isdigit():
    os.environ["NUMEXPR_NUM_THREADS"] = "8"

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 20260803
C_GRID = (0.01, 0.1, 1.0, 10.0)
LOGISTIC_DEEP_DIMS = (32, 64, 128)
MLP_DEEP_DIMS = (64, 128)
SCALAR_PCA_DIM = 48
MAX_MISSING_RATE = 0.90
LOWER_QUANTILE = 0.005
UPPER_QUANTILE = 0.995
MIN_VARIANCE = 1e-12
DEFAULT_MLP_SEEDS = 3
DEFAULT_BOOTSTRAP_REPEATS = 2000
MODEL_ORDER = (
    "Dummy",
    "Logistic_Deep",
    "Logistic_Fusion",
    "MLP_Deep",
    "MLP_Fusion",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(
        temporary,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )
    os.replace(temporary, path)


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            return
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def safe_auc(y: np.ndarray, probability: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, probability))


def safe_ap(y: np.ndarray, probability: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, probability))


def youden_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y, probability)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    score = tpr[finite] - fpr[finite]
    best = np.flatnonzero(score == np.max(score))
    # Stable tie-break: threshold closest to 0.5, then lower threshold.
    candidates = thresholds[finite][best]
    return float(sorted(candidates, key=lambda value: (abs(value - 0.5), value))[0])


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
        "task": "adverse_outcome_record_strict_prepost",
        "model": model,
        "split": split,
        "rows": int(len(y)),
        "patients": None,
        "positive": int((y == 1).sum()),
        "negative": int((y == 0).sum()),
        "positive_fraction": float(np.mean(y == 1)),
        "AUROC": safe_auc(y, probability),
        "AUPRC": safe_ap(y, probability),
        "Balanced Accuracy": float(balanced_accuracy_score(y, prediction)),
        "F1": float(f1_score(y, prediction, zero_division=0)),
        "Precision": float(precision_score(y, prediction, zero_division=0)),
        "Sensitivity": float(recall_score(y, prediction, zero_division=0)),
        "Specificity": float(tn / max(tn + fp, 1)),
        "Brier": float(brier_score_loss(y, probability)),
        "threshold_from_train_oof": float(threshold),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def load_task(task_root: Path) -> dict[str, dict[str, np.ndarray]]:
    success = task_root / ".TASK_SUCCESS.json"
    if not success.is_file():
        raise RuntimeError(f"Task builder success lock missing: {success}")

    result: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "valid"):
        path = task_root / f"{split}_features.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as raw:
            required = {
                "deep",
                "scalar",
                "target",
                "record_uid",
                "patient_id",
                "series_uid",
                "scalar_feature_names",
            }
            if split == "train":
                required.add("fold")
            missing = required - set(raw.files)
            if missing:
                raise KeyError(f"{path}: missing arrays {sorted(missing)}")
            result[split] = {name: np.asarray(raw[name]) for name in raw.files}

    train = result["train"]
    valid = result["valid"]
    for name, data in result.items():
        n = len(data["target"])
        for key in ("deep", "scalar", "record_uid", "patient_id", "series_uid"):
            if len(data[key]) != n:
                raise AssertionError(f"{name}: {key} row mismatch")
        if data["deep"].shape[1] != 10240:
            raise AssertionError(f"{name}: deep dimension changed")
        if not np.isfinite(data["deep"]).all():
            raise AssertionError(f"{name}: strict Pre+Post deep contains nonfinite")
        if not set(np.unique(data["target"]).tolist()).issubset({0, 1}):
            raise AssertionError(f"{name}: non-binary target")

    if not np.array_equal(
        train["scalar_feature_names"].astype(str),
        valid["scalar_feature_names"].astype(str),
    ):
        raise AssertionError("Train/Valid scalar feature names differ")
    if set(train["patient_id"].astype(str)) & set(valid["patient_id"].astype(str)):
        raise AssertionError("Train/Valid patient overlap")
    if sorted(np.unique(train["fold"]).tolist()) != [1, 2, 3, 4, 5]:
        raise AssertionError("Train fold assignment must be 1..5")
    fold_frame = pd.DataFrame(
        {
            "patient_id": train["patient_id"].astype(str),
            "fold": train["fold"].astype(int),
        }
    )
    if int(fold_frame.groupby("patient_id")["fold"].nunique().max()) != 1:
        raise AssertionError("Patient leakage across outer folds")
    return result


@dataclass
class ScalarPreprocessor:
    max_missing_rate: float = MAX_MISSING_RATE
    lower_quantile: float = LOWER_QUANTILE
    upper_quantile: float = UPPER_QUANTILE
    min_variance: float = MIN_VARIANCE

    kept_after_missing: np.ndarray | None = None
    kept_after_variance: np.ndarray | None = None
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    median: np.ndarray | None = None
    scaler: RobustScaler | None = None
    pca: PCA | None = None

    def fit(self, x: np.ndarray, pca_dim: int, seed: int) -> "ScalarPreprocessor":
        x = np.asarray(x, dtype=np.float64)
        finite = np.isfinite(x)
        missing_rate = 1.0 - finite.mean(axis=0)
        median = np.full(x.shape[1], np.nan, dtype=np.float64)
        for index in np.flatnonzero(missing_rate <= self.max_missing_rate):
            values = x[finite[:, index], index]
            if values.size:
                median[index] = np.median(values)
        kept = np.flatnonzero(
            (missing_rate <= self.max_missing_rate) & np.isfinite(median)
        )
        if kept.size == 0:
            raise RuntimeError("All scalar features removed by missing filter")
        selected = x[:, kept]
        lower = np.nanquantile(selected, self.lower_quantile, axis=0)
        upper = np.nanquantile(selected, self.upper_quantile, axis=0)
        med = median[kept]
        clipped = np.clip(selected, lower, upper)
        clipped = np.where(np.isfinite(clipped), clipped, med[None, :])
        variance = np.var(clipped, axis=0)
        keep_variance = np.flatnonzero(variance > self.min_variance)
        if keep_variance.size == 0:
            raise RuntimeError("All scalar features removed as constant")

        self.kept_after_missing = kept
        self.kept_after_variance = keep_variance
        self.lower = lower[keep_variance]
        self.upper = upper[keep_variance]
        self.median = med[keep_variance]

        values = selected[:, keep_variance]
        values = np.clip(values, self.lower, self.upper)
        values = np.where(np.isfinite(values), values, self.median[None, :])
        self.scaler = RobustScaler(quantile_range=(25.0, 75.0))
        values = self.scaler.fit_transform(values)

        n_components = max(
            1,
            min(int(pca_dim), values.shape[0] - 1, values.shape[1]),
        )
        self.pca = PCA(
            n_components=n_components,
            svd_solver="randomized",
            random_state=seed,
        )
        self.pca.fit(values)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if any(
            value is None
            for value in (
                self.kept_after_missing,
                self.kept_after_variance,
                self.lower,
                self.upper,
                self.median,
                self.scaler,
                self.pca,
            )
        ):
            raise RuntimeError("Scalar preprocessor is not fitted")
        selected = np.asarray(x, dtype=np.float64)[:, self.kept_after_missing]
        selected = selected[:, self.kept_after_variance]
        selected = np.clip(selected, self.lower, self.upper)
        selected = np.where(
            np.isfinite(selected),
            selected,
            self.median[None, :],
        )
        selected = self.scaler.transform(selected)
        result = self.pca.transform(selected).astype(np.float32)
        if not np.isfinite(result).all():
            raise RuntimeError("Nonfinite scalar transform")
        return result

    def audit(self) -> dict[str, Any]:
        if self.pca is None:
            return {}
        return {
            "kept_after_missing": int(len(self.kept_after_missing)),
            "kept_after_variance": int(len(self.kept_after_variance)),
            "pca_components": int(self.pca.n_components_),
            "pca_explained_variance_ratio_sum": float(
                np.sum(self.pca.explained_variance_ratio_)
            ),
        }


@dataclass
class FusionPreprocessor:
    deep_pca_dim: int
    use_scalar: bool
    scalar_pca_dim: int = SCALAR_PCA_DIM
    seed: int = SEED

    deep_scaler: StandardScaler | None = None
    deep_pca: PCA | None = None
    scalar_preprocessor: ScalarPreprocessor | None = None

    def fit(self, deep: np.ndarray, scalar: np.ndarray) -> "FusionPreprocessor":
        deep = np.asarray(deep, dtype=np.float32)
        if not np.isfinite(deep).all():
            raise RuntimeError("Nonfinite deep input")
        self.deep_scaler = StandardScaler()
        deep_scaled = self.deep_scaler.fit_transform(deep)
        n_components = max(
            1,
            min(int(self.deep_pca_dim), deep.shape[0] - 1, deep.shape[1]),
        )
        self.deep_pca = PCA(
            n_components=n_components,
            svd_solver="randomized",
            random_state=self.seed,
        )
        self.deep_pca.fit(deep_scaled)
        if self.use_scalar:
            self.scalar_preprocessor = ScalarPreprocessor().fit(
                scalar,
                self.scalar_pca_dim,
                self.seed + 17,
            )
        return self

    def transform(self, deep: np.ndarray, scalar: np.ndarray) -> np.ndarray:
        if self.deep_scaler is None or self.deep_pca is None:
            raise RuntimeError("Deep preprocessor is not fitted")
        deep_scaled = self.deep_scaler.transform(deep)
        deep_pca = self.deep_pca.transform(deep_scaled).astype(np.float32)
        if not self.use_scalar:
            return deep_pca
        if self.scalar_preprocessor is None:
            raise RuntimeError("Scalar preprocessor missing")
        scalar_pca = self.scalar_preprocessor.transform(scalar)
        result = np.concatenate([deep_pca, scalar_pca], axis=1)
        if not np.isfinite(result).all():
            raise RuntimeError("Nonfinite fusion transform")
        return result

    def audit(self) -> dict[str, Any]:
        if self.deep_pca is None:
            return {}
        return {
            "deep_pca_requested": int(self.deep_pca_dim),
            "deep_pca_components": int(self.deep_pca.n_components_),
            "deep_pca_explained_variance_ratio_sum": float(
                np.sum(self.deep_pca.explained_variance_ratio_)
            ),
            "use_scalar": bool(self.use_scalar),
            "scalar": (
                self.scalar_preprocessor.audit()
                if self.scalar_preprocessor is not None
                else {}
            ),
        }


def inner_splits(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    splits = list(splitter.split(np.zeros(len(y)), y, groups))
    for train_index, valid_index in splits:
        if len(np.unique(y[train_index])) < 2 or len(np.unique(y[valid_index])) < 2:
            raise RuntimeError("Inner grouped split lacks a class")
    return splits


def select_logistic(
    deep: np.ndarray,
    scalar: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    use_scalar: bool,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    splits = inner_splits(y, groups, n_splits=3, seed=seed)
    rows: list[dict[str, Any]] = []
    best: tuple[float, int, float] | None = None
    best_config: dict[str, Any] = {}

    for deep_dim in LOGISTIC_DEEP_DIMS:
        predictions_by_c = {
            float(c): np.full(len(y), np.nan, dtype=np.float64)
            for c in C_GRID
        }
        for inner_fold, (development, holdout) in enumerate(splits, start=1):
            preprocessor = FusionPreprocessor(
                deep_pca_dim=deep_dim,
                use_scalar=use_scalar,
                seed=seed + deep_dim * 10 + inner_fold,
            ).fit(deep[development], scalar[development])
            x_development = preprocessor.transform(
                deep[development], scalar[development]
            )
            x_holdout = preprocessor.transform(deep[holdout], scalar[holdout])
            for c_value in C_GRID:
                model = LogisticRegression(
                    C=float(c_value),
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=10000,
                    random_state=seed,
                )
                model.fit(x_development, y[development])
                predictions_by_c[float(c_value)][holdout] = model.predict_proba(
                    x_holdout
                )[:, 1]

        for c_value, probability in predictions_by_c.items():
            if not np.isfinite(probability).all():
                raise RuntimeError("Incomplete inner logistic predictions")
            ap = safe_ap(y, probability)
            auc = safe_auc(y, probability)
            row = {
                "deep_pca_dim": int(deep_dim),
                "use_scalar": bool(use_scalar),
                "C": float(c_value),
                "inner_AUPRC": ap,
                "inner_AUROC": auc,
            }
            rows.append(row)
            key = (ap, auc, -deep_dim, -math.log10(c_value))
            if best is None or key > best:
                best = key
                best_config = row.copy()
    return best_config, rows


@dataclass(frozen=True)
class MLPConfig:
    deep_pca_dim: int
    hidden1: int
    hidden2: int
    dropout1: float
    dropout2: float
    learning_rate: float
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 180
    patience: int = 22


MLP_CONFIGS = (
    MLPConfig(64, 128, 32, 0.30, 0.15, 1e-3),
    MLPConfig(128, 128, 32, 0.30, 0.15, 1e-3),
    MLPConfig(128, 256, 64, 0.40, 0.20, 5e-4),
    MLPConfig(64, 256, 64, 0.45, 0.25, 5e-4),
)


class AdverseMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden1: int,
        hidden2: int,
        dropout1: float,
        dropout2: float,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.GELU(),
            nn.Dropout(dropout1),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.GELU(),
            nn.Dropout(dropout2),
            nn.Linear(hidden2, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(1)


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)),
    )
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )


@torch.inference_mode()
def mlp_predict(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    probabilities: list[np.ndarray] = []
    for start in range(0, len(x), 1024):
        tensor = torch.from_numpy(x[start : start + 1024]).to(device)
        probabilities.append(torch.sigmoid(model(tensor)).cpu().numpy())
    return np.concatenate(probabilities).astype(np.float64)


def fit_mlp_early_stop(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    config: MLPConfig,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, torch.Tensor], int, float, list[dict[str, Any]]]:
    set_seed(seed)
    model = AdverseMLP(
        x_train.shape[1],
        config.hidden1,
        config.hidden2,
        config.dropout1,
        config.dropout2,
    ).to(device)
    positive = max(int((y_train == 1).sum()), 1)
    negative = max(int((y_train == 0).sum()), 1)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            negative / positive,
            dtype=torch.float32,
            device=device,
        )
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loader = make_loader(
        x_train,
        y_train,
        config.batch_size,
        seed,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_ap = -math.inf
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_y)
            total_rows += len(batch_y)

        probability = mlp_predict(model, x_validation, device)
        ap = safe_ap(y_validation, probability)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(total_rows, 1),
                "validation_AUPRC": ap,
                "validation_AUROC": safe_auc(y_validation, probability),
            }
        )
        if ap > best_ap + 1e-6:
            best_ap = ap
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("MLP did not produce a checkpoint")
    return best_state, best_epoch, float(best_ap), history


def fit_mlp_fixed_epochs(
    x: np.ndarray,
    y: np.ndarray,
    config: MLPConfig,
    epochs: int,
    device: torch.device,
    seed: int,
) -> AdverseMLP:
    set_seed(seed)
    model = AdverseMLP(
        x.shape[1],
        config.hidden1,
        config.hidden2,
        config.dropout1,
        config.dropout2,
    ).to(device)
    positive = max(int((y == 1).sum()), 1)
    negative = max(int((y == 0).sum()), 1)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            negative / positive,
            dtype=torch.float32,
            device=device,
        )
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loader = make_loader(x, y, config.batch_size, seed)
    for _ in range(max(1, int(epochs))):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
    return model


def choose_inner_holdout(
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    splits = inner_splits(y, groups, n_splits=5, seed=seed)
    target_size = len(y) / 5.0
    global_rate = float(np.mean(y))
    candidates = []
    for train_index, valid_index in splits:
        score = abs(len(valid_index) - target_size) + 50.0 * abs(
            float(np.mean(y[valid_index])) - global_rate
        )
        candidates.append((score, train_index, valid_index))
    _, train_index, valid_index = min(candidates, key=lambda item: item[0])
    return train_index, valid_index


def select_mlp_config(
    deep: np.ndarray,
    scalar: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    use_scalar: bool,
    device: torch.device,
    seed: int,
) -> tuple[MLPConfig, int, list[dict[str, Any]]]:
    inner_train, inner_valid = choose_inner_holdout(y, groups, seed)
    rows: list[dict[str, Any]] = []
    best_key: tuple[float, float, int] | None = None
    best_config = MLP_CONFIGS[0]
    best_epoch = 1

    preprocessor_cache: dict[int, tuple[FusionPreprocessor, np.ndarray, np.ndarray]] = {}
    for config_index, config in enumerate(MLP_CONFIGS):
        if config.deep_pca_dim not in preprocessor_cache:
            preprocessor = FusionPreprocessor(
                deep_pca_dim=config.deep_pca_dim,
                use_scalar=use_scalar,
                seed=seed + config.deep_pca_dim,
            ).fit(deep[inner_train], scalar[inner_train])
            x_train = preprocessor.transform(
                deep[inner_train], scalar[inner_train]
            )
            x_valid = preprocessor.transform(
                deep[inner_valid], scalar[inner_valid]
            )
            preprocessor_cache[config.deep_pca_dim] = (
                preprocessor,
                x_train,
                x_valid,
            )
        preprocessor, x_train, x_valid = preprocessor_cache[config.deep_pca_dim]
        state, epoch, ap, history = fit_mlp_early_stop(
            x_train,
            y[inner_train],
            x_valid,
            y[inner_valid],
            config,
            device,
            seed + 1000 + config_index,
        )
        del state
        auc = safe_auc(
            y[inner_valid],
            # Retrain the selected state is unnecessary for ranking because
            # fit_mlp_early_stop already reports best AP; AUROC is taken from
            # the best-AP epoch history as a deterministic tie-break.
            np.zeros(len(inner_valid), dtype=np.float64),
        )
        best_history = max(
            history,
            key=lambda row: (
                row["validation_AUPRC"],
                row["validation_AUROC"],
                -row["epoch"],
            ),
        )
        auc = float(best_history["validation_AUROC"])
        rows.append(
            {
                **asdict(config),
                "use_scalar": bool(use_scalar),
                "best_epoch": int(epoch),
                "inner_AUPRC": float(ap),
                "inner_AUROC": auc,
                "preprocessor": json.dumps(
                    preprocessor.audit(),
                    sort_keys=True,
                ),
            }
        )
        key = (float(ap), auc, -config.deep_pca_dim)
        if best_key is None or key > best_key:
            best_key = key
            best_config = config
            best_epoch = int(epoch)
    return best_config, best_epoch, rows


def save_fold_cache(
    path: Path,
    holdout_ids: np.ndarray,
    valid_ids: np.ndarray,
    holdout_probability: np.ndarray,
    valid_probability: np.ndarray,
    payload: dict[str, Any],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    atomic_npz(
        path / "predictions.npz",
        holdout_record_uid=holdout_ids.astype(str),
        valid_record_uid=valid_ids.astype(str),
        holdout_probability=holdout_probability.astype(np.float64),
        valid_probability=valid_probability.astype(np.float64),
    )
    atomic_json(payload, path / ".SUCCESS.json")


def load_fold_cache(
    path: Path,
    holdout_ids: np.ndarray,
    valid_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    success = path / ".SUCCESS.json"
    predictions = path / "predictions.npz"
    if not success.is_file() or not predictions.is_file():
        return None
    try:
        with np.load(predictions, allow_pickle=False) as raw:
            saved_holdout = raw["holdout_record_uid"].astype(str)
            saved_valid = raw["valid_record_uid"].astype(str)
            holdout_probability = raw["holdout_probability"].astype(np.float64)
            valid_probability = raw["valid_probability"].astype(np.float64)
        if not np.array_equal(saved_holdout, holdout_ids.astype(str)):
            return None
        if not np.array_equal(saved_valid, valid_ids.astype(str)):
            return None
        if holdout_probability.shape != (len(holdout_ids),):
            return None
        if valid_probability.shape != (len(valid_ids),):
            return None
        if not np.isfinite(holdout_probability).all():
            return None
        if not np.isfinite(valid_probability).all():
            return None
        payload = json.loads(success.read_text(encoding="utf-8"))
        return holdout_probability, valid_probability, payload
    except Exception:
        return None


def train_logistic_outer(
    model_name: str,
    train: dict[str, np.ndarray],
    valid: dict[str, np.ndarray],
    use_scalar: bool,
    output_root: Path,
    overwrite: bool,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    y = train["target"].astype(np.int64)
    folds = train["fold"].astype(int)
    groups = train["patient_id"].astype(str)
    record_ids = train["record_uid"].astype(str)
    valid_ids = valid["record_uid"].astype(str)
    oof = np.full(len(y), np.nan, dtype=np.float64)
    valid_folds: list[np.ndarray] = []
    audit_rows: list[dict[str, Any]] = []

    for fold in range(1, 6):
        development = np.flatnonzero(folds != fold)
        holdout = np.flatnonzero(folds == fold)
        fold_dir = output_root / "folds" / model_name / f"fold_{fold}"
        cached = (
            None
            if overwrite
            else load_fold_cache(
                fold_dir,
                record_ids[holdout],
                valid_ids,
            )
        )
        if cached is not None:
            holdout_probability, valid_probability, payload = cached
            resumed = True
        else:
            best, inner_rows = select_logistic(
                train["deep"][development],
                train["scalar"][development],
                y[development],
                groups[development],
                use_scalar,
                SEED + fold * 100,
            )
            preprocessor = FusionPreprocessor(
                deep_pca_dim=int(best["deep_pca_dim"]),
                use_scalar=use_scalar,
                seed=SEED + fold * 1000,
            ).fit(
                train["deep"][development],
                train["scalar"][development],
            )
            x_development = preprocessor.transform(
                train["deep"][development],
                train["scalar"][development],
            )
            x_holdout = preprocessor.transform(
                train["deep"][holdout],
                train["scalar"][holdout],
            )
            x_valid = preprocessor.transform(
                valid["deep"],
                valid["scalar"],
            )
            model = LogisticRegression(
                C=float(best["C"]),
                class_weight="balanced",
                solver="liblinear",
                max_iter=10000,
                random_state=SEED + fold,
            )
            model.fit(x_development, y[development])
            holdout_probability = model.predict_proba(x_holdout)[:, 1]
            valid_probability = model.predict_proba(x_valid)[:, 1]
            fold_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    "preprocessor": preprocessor,
                    "model": model,
                },
                fold_dir / "model.joblib",
            )
            atomic_csv(
                pd.DataFrame(inner_rows),
                fold_dir / "inner_search.csv",
            )
            payload = {
                "model": model_name,
                "fold": fold,
                "use_scalar": use_scalar,
                "best_inner_config": best,
                "preprocessor": preprocessor.audit(),
                "development_rows": int(len(development)),
                "holdout_rows": int(len(holdout)),
            }
            save_fold_cache(
                fold_dir,
                record_ids[holdout],
                valid_ids,
                holdout_probability,
                valid_probability,
                payload,
            )
            resumed = False

        oof[holdout] = holdout_probability
        valid_folds.append(valid_probability)
        audit_rows.append(
            {
                "model": model_name,
                "fold": fold,
                "development_rows": int(len(development)),
                "holdout_rows": int(len(holdout)),
                "holdout_patients": int(
                    len(np.unique(groups[holdout]))
                ),
                "holdout_AUPRC": safe_ap(y[holdout], holdout_probability),
                "holdout_AUROC": safe_auc(y[holdout], holdout_probability),
                "resumed": resumed,
                "selection": json.dumps(payload, sort_keys=True),
            }
        )
        print(
            f"[PASS] {model_name} fold={fold} "
            f"AUPRC={audit_rows[-1]['holdout_AUPRC']:.6f} resumed={resumed}",
            flush=True,
        )
    if not np.isfinite(oof).all():
        raise RuntimeError(f"{model_name}: incomplete OOF")
    return oof, np.mean(np.stack(valid_folds), axis=0), audit_rows


def train_mlp_outer(
    model_name: str,
    train: dict[str, np.ndarray],
    valid: dict[str, np.ndarray],
    use_scalar: bool,
    device: torch.device,
    output_root: Path,
    overwrite: bool,
    ensemble_seeds: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    y = train["target"].astype(np.int64)
    folds = train["fold"].astype(int)
    groups = train["patient_id"].astype(str)
    record_ids = train["record_uid"].astype(str)
    valid_ids = valid["record_uid"].astype(str)
    oof = np.full(len(y), np.nan, dtype=np.float64)
    valid_folds: list[np.ndarray] = []
    audit_rows: list[dict[str, Any]] = []

    for fold in range(1, 6):
        development = np.flatnonzero(folds != fold)
        holdout = np.flatnonzero(folds == fold)
        fold_dir = output_root / "folds" / model_name / f"fold_{fold}"
        cached = (
            None
            if overwrite
            else load_fold_cache(
                fold_dir,
                record_ids[holdout],
                valid_ids,
            )
        )
        if cached is not None:
            holdout_probability, valid_probability, payload = cached
            resumed = True
        else:
            config, selected_epoch, search_rows = select_mlp_config(
                train["deep"][development],
                train["scalar"][development],
                y[development],
                groups[development],
                use_scalar,
                device,
                SEED + fold * 100,
            )
            preprocessor = FusionPreprocessor(
                deep_pca_dim=config.deep_pca_dim,
                use_scalar=use_scalar,
                seed=SEED + fold * 1000,
            ).fit(
                train["deep"][development],
                train["scalar"][development],
            )
            x_development = preprocessor.transform(
                train["deep"][development],
                train["scalar"][development],
            )
            x_holdout = preprocessor.transform(
                train["deep"][holdout],
                train["scalar"][holdout],
            )
            x_valid = preprocessor.transform(
                valid["deep"],
                valid["scalar"],
            )

            holdout_seed_predictions: list[np.ndarray] = []
            valid_seed_predictions: list[np.ndarray] = []
            fold_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(preprocessor, fold_dir / "preprocessor.joblib")
            for seed_index in range(ensemble_seeds):
                model_seed = SEED + fold * 10000 + seed_index
                model = fit_mlp_fixed_epochs(
                    x_development,
                    y[development],
                    config,
                    selected_epoch,
                    device,
                    model_seed,
                )
                holdout_seed_predictions.append(
                    mlp_predict(model, x_holdout, device)
                )
                valid_seed_predictions.append(
                    mlp_predict(model, x_valid, device)
                )
                torch.save(
                    {
                        "model_name": model_name,
                        "fold": fold,
                        "seed": model_seed,
                        "input_dim": int(x_development.shape[1]),
                        "config": asdict(config),
                        "selected_epoch": int(selected_epoch),
                        "state_dict": {
                            key: value.detach().cpu()
                            for key, value in model.state_dict().items()
                        },
                    },
                    fold_dir / f"model_seed_{seed_index}.pt",
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            holdout_probability = np.mean(
                np.stack(holdout_seed_predictions),
                axis=0,
            )
            valid_probability = np.mean(
                np.stack(valid_seed_predictions),
                axis=0,
            )
            atomic_csv(
                pd.DataFrame(search_rows),
                fold_dir / "inner_search.csv",
            )
            payload = {
                "model": model_name,
                "fold": fold,
                "use_scalar": use_scalar,
                "selected_config": asdict(config),
                "selected_epoch": int(selected_epoch),
                "ensemble_seeds": int(ensemble_seeds),
                "preprocessor": preprocessor.audit(),
                "development_rows": int(len(development)),
                "holdout_rows": int(len(holdout)),
            }
            save_fold_cache(
                fold_dir,
                record_ids[holdout],
                valid_ids,
                holdout_probability,
                valid_probability,
                payload,
            )
            resumed = False

        oof[holdout] = holdout_probability
        valid_folds.append(valid_probability)
        audit_rows.append(
            {
                "model": model_name,
                "fold": fold,
                "development_rows": int(len(development)),
                "holdout_rows": int(len(holdout)),
                "holdout_patients": int(
                    len(np.unique(groups[holdout]))
                ),
                "holdout_AUPRC": safe_ap(y[holdout], holdout_probability),
                "holdout_AUROC": safe_auc(y[holdout], holdout_probability),
                "resumed": resumed,
                "selection": json.dumps(payload, sort_keys=True),
            }
        )
        print(
            f"[PASS] {model_name} fold={fold} "
            f"AUPRC={audit_rows[-1]['holdout_AUPRC']:.6f} resumed={resumed}",
            flush=True,
        )

    if not np.isfinite(oof).all():
        raise RuntimeError(f"{model_name}: incomplete OOF")
    return oof, np.mean(np.stack(valid_folds), axis=0), audit_rows


def patient_cluster_bootstrap(
    y: np.ndarray,
    groups: np.ndarray,
    probabilities: dict[str, np.ndarray],
    thresholds: dict[str, float],
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_groups = np.unique(groups.astype(str))
    group_indices = {
        group: np.flatnonzero(groups.astype(str) == group)
        for group in unique_groups
    }
    rng = np.random.default_rng(seed)
    metrics_by_model: dict[str, dict[str, list[float]]] = {
        model: {
            metric: []
            for metric in (
                "AUROC",
                "AUPRC",
                "Balanced Accuracy",
                "Sensitivity",
                "Specificity",
                "Brier",
            )
        }
        for model in MODEL_ORDER
    }
    paired: dict[tuple[str, str], dict[str, list[float]]] = {}
    for reference in ("Dummy", "Logistic_Deep", "Logistic_Fusion"):
        for comparison in ("MLP_Deep", "MLP_Fusion"):
            paired[(reference, comparison)] = {
                "AUROC": [],
                "AUPRC": [],
                "Brier improvement": [],
            }

    valid_repeats = 0
    for _ in range(repeats):
        sampled_groups = rng.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )
        indices = np.concatenate(
            [group_indices[group] for group in sampled_groups]
        )
        y_sample = y[indices]
        if len(np.unique(y_sample)) < 2:
            continue
        valid_repeats += 1
        current: dict[str, dict[str, Any]] = {}
        for model in MODEL_ORDER:
            current[model] = metric_row(
                model,
                "bootstrap",
                y_sample,
                probabilities[model][indices],
                thresholds[model],
            )
            for metric in metrics_by_model[model]:
                metrics_by_model[model][metric].append(
                    float(current[model][metric])
                )
        for (reference, comparison), values in paired.items():
            values["AUROC"].append(
                current[comparison]["AUROC"] - current[reference]["AUROC"]
            )
            values["AUPRC"].append(
                current[comparison]["AUPRC"] - current[reference]["AUPRC"]
            )
            values["Brier improvement"].append(
                current[reference]["Brier"] - current[comparison]["Brier"]
            )

    ci_rows: list[dict[str, Any]] = []
    for model, metrics in metrics_by_model.items():
        for metric, values_list in metrics.items():
            values = np.asarray(values_list, dtype=np.float64)
            ci_rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "bootstrap_mean": float(np.mean(values)),
                    "ci_lower": float(np.percentile(values, 2.5)),
                    "ci_upper": float(np.percentile(values, 97.5)),
                    "effective_repeats": int(valid_repeats),
                    "bootstrap_unit": "patient_id",
                }
            )

    paired_rows: list[dict[str, Any]] = []
    for (reference, comparison), metrics in paired.items():
        for metric, values_list in metrics.items():
            values = np.asarray(values_list, dtype=np.float64)
            lower = float(np.percentile(values, 2.5))
            upper = float(np.percentile(values, 97.5))
            paired_rows.append(
                {
                    "reference": reference,
                    "comparison": comparison,
                    "metric": metric,
                    "difference_mean": float(np.mean(values)),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "crosses_zero": bool(lower <= 0 <= upper),
                    "effective_repeats": int(valid_repeats),
                    "bootstrap_unit": "patient_id",
                }
            )
    return pd.DataFrame(ci_rows), pd.DataFrame(paired_rows)


def write_report(
    path: Path,
    metrics: pd.DataFrame,
    fold_audit: pd.DataFrame,
    task_summary: dict[str, Any],
) -> None:
    train_metrics = metrics[metrics["split"] == "Train_OOF"]
    valid_metrics = metrics[metrics["split"] == "Valid"]
    lines = [
        "# Strict Pre+Post Local-CAVE adverse-outcome report",
        "",
        "## Cohort",
        "",
        f"- Train records: {task_summary['train']['included_records']}",
        f"- Valid records: {task_summary['valid']['included_records']}",
        "- Prediction unit: record_uid.",
        "- Grouping/bootstrap unit: patient_id.",
        "- Pre-only, Post-only and no-phase records are excluded.",
        "- Official Valid is never used for preprocessing, model selection, "
        "early stopping or threshold selection.",
        "",
        "## Train pooled OOF",
        "",
        train_metrics.to_markdown(index=False),
        "",
        "## Independent Valid",
        "",
        valid_metrics.to_markdown(index=False),
        "",
        "## Fold audit",
        "",
        fold_audit[
            [
                "model",
                "fold",
                "development_rows",
                "holdout_rows",
                "holdout_patients",
                "holdout_AUROC",
                "holdout_AUPRC",
                "resumed",
            ]
        ].to_markdown(index=False),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mlp-seeds", type=int, default=DEFAULT_MLP_SEEDS)
    parser.add_argument(
        "--bootstrap-repeats",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPEATS,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    task_root = args.task_root.resolve()
    output_dir = args.output_dir.resolve()
    prepare_output(output_dir, args.overwrite)
    start = time.time()
    set_seed(SEED)

    task_summary = json.loads(
        (task_root / "task_summary.json").read_text(encoding="utf-8")
    )
    data = load_task(task_root)
    train = data["train"]
    valid = data["valid"]
    y_train = train["target"].astype(np.int64)
    y_valid = valid["target"].astype(np.int64)

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)

    probability_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fold_rows: list[dict[str, Any]] = []

    prior = float(np.mean(y_train))
    probability_pairs["Dummy"] = (
        np.full(len(y_train), prior, dtype=np.float64),
        np.full(len(y_valid), prior, dtype=np.float64),
    )

    for model_name, use_scalar in (
        ("Logistic_Deep", False),
        ("Logistic_Fusion", True),
    ):
        oof, valid_probability, rows = train_logistic_outer(
            model_name,
            train,
            valid,
            use_scalar,
            output_dir,
            args.overwrite,
        )
        probability_pairs[model_name] = (oof, valid_probability)
        fold_rows.extend(rows)

    for model_name, use_scalar in (
        ("MLP_Deep", False),
        ("MLP_Fusion", True),
    ):
        oof, valid_probability, rows = train_mlp_outer(
            model_name,
            train,
            valid,
            use_scalar,
            device,
            output_dir,
            args.overwrite,
            args.mlp_seeds,
        )
        probability_pairs[model_name] = (oof, valid_probability)
        fold_rows.extend(rows)

    thresholds: dict[str, float] = {}
    metric_rows: list[dict[str, Any]] = []
    train_predictions = pd.DataFrame(
        {
            "record_uid": train["record_uid"].astype(str),
            "patient_id": train["patient_id"].astype(str),
            "series_uid": train["series_uid"].astype(str),
            "target": y_train,
            "fold": train["fold"].astype(int),
        }
    )
    valid_predictions = pd.DataFrame(
        {
            "record_uid": valid["record_uid"].astype(str),
            "patient_id": valid["patient_id"].astype(str),
            "series_uid": valid["series_uid"].astype(str),
            "target": y_valid,
        }
    )

    for model in MODEL_ORDER:
        oof_probability, valid_probability = probability_pairs[model]
        threshold = youden_threshold(y_train, oof_probability)
        thresholds[model] = threshold
        train_row = metric_row(
            model,
            "Train_OOF",
            y_train,
            oof_probability,
            threshold,
        )
        valid_row = metric_row(
            model,
            "Valid",
            y_valid,
            valid_probability,
            threshold,
        )
        train_row["patients"] = int(
            len(np.unique(train["patient_id"].astype(str)))
        )
        valid_row["patients"] = int(
            len(np.unique(valid["patient_id"].astype(str)))
        )
        metric_rows.extend([train_row, valid_row])
        train_predictions[f"{model}_probability"] = oof_probability
        valid_predictions[f"{model}_probability"] = valid_probability
        train_predictions[f"{model}_prediction"] = (
            oof_probability >= threshold
        ).astype(int)
        valid_predictions[f"{model}_prediction"] = (
            valid_probability >= threshold
        ).astype(int)

    metrics = pd.DataFrame(metric_rows)
    fold_audit = pd.DataFrame(fold_rows)
    atomic_csv(metrics, output_dir / "metrics.csv")
    atomic_csv(fold_audit, output_dir / "fold_audit.csv")
    atomic_csv(train_predictions, output_dir / "train_oof_predictions.csv")
    atomic_csv(valid_predictions, output_dir / "valid_predictions.csv")

    valid_probability_map = {
        model: probability_pairs[model][1]
        for model in MODEL_ORDER
    }
    bootstrap_ci, paired_bootstrap = patient_cluster_bootstrap(
        y_valid,
        valid["patient_id"].astype(str),
        valid_probability_map,
        thresholds,
        args.bootstrap_repeats,
        SEED + 50000,
    )
    atomic_csv(bootstrap_ci, output_dir / "valid_patient_cluster_bootstrap_ci.csv")
    atomic_csv(
        paired_bootstrap,
        output_dir / "valid_paired_model_differences.csv",
    )

    summary = {
        "status": "success",
        "version": "formal_adverse_prepost_local_cave_models_v1",
        "task": "adverse_outcome_record_strict_prepost",
        "models": list(MODEL_ORDER),
        "train_rows": int(len(y_train)),
        "train_patients": int(
            len(np.unique(train["patient_id"].astype(str)))
        ),
        "train_positive": int(y_train.sum()),
        "valid_rows": int(len(y_valid)),
        "valid_patients": int(
            len(np.unique(valid["patient_id"].astype(str)))
        ),
        "valid_positive": int(y_valid.sum()),
        "outer_folds": 5,
        "grouping_unit": "patient_id",
        "threshold_source": "pooled Train OOF only",
        "valid_used_for_fitting_or_selection": False,
        "strict_phase_policy": "both Pre and Post required",
        "thresholds": thresholds,
        "mlp_ensemble_seeds_per_outer_fold": int(args.mlp_seeds),
        "bootstrap_repeats_requested": int(args.bootstrap_repeats),
        "bootstrap_unit": "patient_id",
        "device": str(device),
        "seed": SEED,
        "task_success_lock_sha256": sha256_file(
            task_root / ".TASK_SUCCESS.json"
        ),
        "elapsed_seconds": float(time.time() - start),
    }
    atomic_json(summary, output_dir / "summary.json")
    atomic_json(summary, output_dir / ".MODELS_SUCCESS.json")
    write_report(
        output_dir / "report.md",
        metrics,
        fold_audit,
        task_summary,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Train and validate a simple Pre+Post MLP adverse-outcome classifier.

Protocol
--------
- Input: 147 Pre + 147 Post phase model-candidate features (294 raw columns).
- Target: patient-level adverse outcome, 0/1.
- Outer stratified 5-fold OOF on Train.
- Inner stratified holdout for epoch selection only.
- Fold-local preprocessing: high-missing removal, winsorization, median imputation,
  constant removal, RobustScaler.
- Final Valid probability: mean of five outer-fold models.
- Threshold: selected only from pooled Train OOF predictions.
- Main model: simple fully connected MLP (d -> 64 -> 16 -> 1).
- Dummy and Logistic are low-cost sanity baselines, not alternative research branches.
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
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from sklearn.dummy import DummyClassifier
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
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
N_SPLITS = 5
INNER_VALID_FRACTION = 0.15
BATCH_SIZE = 32
MAX_EPOCHS = 200
PATIENCE = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BOOTSTRAP_REPEATS = 2000
MAX_MISSING_RATE = 0.80
LOWER_CLIP_QUANTILE = 0.005
UPPER_CLIP_QUANTILE = 0.995
MIN_VARIANCE = 1e-12
MODEL_ORDER = ["Dummy", "Logistic", "Pre+Post MLP"]
EXPECTED_TRAIN_ROWS = 855
EXPECTED_VALID_ROWS = 226
EXPECTED_TRAIN_POSITIVE = 137
EXPECTED_VALID_POSITIVE = 38
EXPECTED_RAW_FEATURES = 294


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


class RunLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = path.open("x", encoding="utf-8")

    def log(self, message: str) -> None:
        line = f"[{utc_now()}] {message}"
        print(line, flush=True)
        self.handle.write(line + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


@dataclass
class FoldPreprocessor:
    raw_feature_names: list[str]
    max_missing_rate: float = MAX_MISSING_RATE
    lower_quantile: float = LOWER_CLIP_QUANTILE
    upper_quantile: float = UPPER_CLIP_QUANTILE
    min_variance: float = MIN_VARIANCE

    def __post_init__(self) -> None:
        self.kept_after_missing: list[str] = []
        self.kept_features: list[str] = []
        self.lower_bounds: pd.Series | None = None
        self.upper_bounds: pd.Series | None = None
        self.medians: pd.Series | None = None
        self.scaler: RobustScaler | None = None
        self.dropped_high_missing: list[str] = []
        self.dropped_constant: list[str] = []

    def fit(self, frame: pd.DataFrame) -> "FoldPreprocessor":
        raw = frame[self.raw_feature_names].apply(pd.to_numeric, errors="coerce")
        missing_rate = raw.isna().mean()
        self.kept_after_missing = missing_rate[missing_rate <= self.max_missing_rate].index.tolist()
        self.dropped_high_missing = missing_rate[missing_rate > self.max_missing_rate].index.tolist()
        if not self.kept_after_missing:
            raise RuntimeError("All features removed by missing-rate filter")

        selected = raw[self.kept_after_missing]
        self.lower_bounds = selected.quantile(self.lower_quantile)
        self.upper_bounds = selected.quantile(self.upper_quantile)
        clipped = selected.clip(lower=self.lower_bounds, upper=self.upper_bounds, axis=1)
        self.medians = clipped.median(axis=0, skipna=True)
        valid_medians = self.medians[np.isfinite(self.medians.to_numpy(dtype=np.float64))]
        self.kept_after_missing = valid_medians.index.tolist()
        self.medians = valid_medians
        self.lower_bounds = self.lower_bounds[self.kept_after_missing]
        self.upper_bounds = self.upper_bounds[self.kept_after_missing]

        clipped = raw[self.kept_after_missing].clip(
            lower=self.lower_bounds, upper=self.upper_bounds, axis=1
        )
        imputed = clipped.fillna(self.medians)
        variances = imputed.var(axis=0, ddof=0)
        self.kept_features = variances[variances > self.min_variance].index.tolist()
        self.dropped_constant = variances[variances <= self.min_variance].index.tolist()
        if not self.kept_features:
            raise RuntimeError("All features removed by constant-feature filter")

        self.lower_bounds = self.lower_bounds[self.kept_features]
        self.upper_bounds = self.upper_bounds[self.kept_features]
        self.medians = self.medians[self.kept_features]
        imputed = raw[self.kept_features].clip(
            lower=self.lower_bounds, upper=self.upper_bounds, axis=1
        ).fillna(self.medians)

        values = imputed.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError("Non-finite values remain after preprocessing fit")
        self.scaler = RobustScaler(quantile_range=(25.0, 75.0))
        self.scaler.fit(values)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.scaler is None or self.lower_bounds is None or self.upper_bounds is None or self.medians is None:
            raise RuntimeError("Preprocessor is not fitted")
        selected = frame[self.kept_features].apply(pd.to_numeric, errors="coerce")
        clipped = selected.clip(lower=self.lower_bounds, upper=self.upper_bounds, axis=1)
        imputed = clipped.fillna(self.medians)
        values = imputed.to_numpy(dtype=np.float64)
        transformed = self.scaler.transform(values).astype(np.float32)
        if not np.isfinite(transformed).all():
            raise RuntimeError("Non-finite values after preprocessing transform")
        return transformed

    def audit_dict(self) -> dict[str, Any]:
        return {
            "raw_feature_count": len(self.raw_feature_names),
            "kept_feature_count": len(self.kept_features),
            "dropped_high_missing_count": len(self.dropped_high_missing),
            "dropped_constant_count": len(self.dropped_constant),
            "kept_features": self.kept_features,
            "dropped_high_missing": self.dropped_high_missing,
            "dropped_constant": self.dropped_constant,
            "max_missing_rate": self.max_missing_rate,
            "lower_clip_quantile": self.lower_quantile,
            "upper_clip_quantile": self.upper_quantile,
            "scaler": "RobustScaler(25,75)",
        }


class AdverseMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(16, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )


def positive_weight(y: np.ndarray, device: torch.device) -> torch.Tensor:
    positive = int(np.sum(y == 1))
    negative = int(np.sum(y == 0))
    if positive <= 0 or negative <= 0:
        raise RuntimeError("Training subset must contain both classes")
    return torch.tensor([negative / positive], dtype=torch.float32, device=device)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_count = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_count
        total_count += batch_count
    return total_loss / max(total_count, 1)


@torch.inference_mode()
def predict_mlp(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    tensor = torch.from_numpy(x.astype(np.float32)).to(device)
    probabilities: list[np.ndarray] = []
    for start in range(0, len(tensor), 1024):
        logits = model(tensor[start : start + 1024])
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities).astype(np.float64)


def train_with_early_stopping(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    device: torch.device,
    seed: int,
    fold: int,
) -> tuple[int, float, list[dict[str, Any]]]:
    set_seed(seed)
    model = AdverseMLP(x_train.shape[1]).to(device)
    pos_weight = positive_weight(y_train, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loader = make_loader(x_train, y_train, BATCH_SIZE, True, seed)

    best_epoch = 1
    best_auprc = -math.inf
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        loss = train_epoch(model, loader, criterion, optimizer, device)
        probabilities = predict_mlp(model, x_valid, device)
        auprc = float(average_precision_score(y_valid, probabilities))
        history.append(
            {
                "fold": fold,
                "stage": "inner_early_stopping",
                "epoch": epoch,
                "train_loss": loss,
                "inner_valid_auprc": auprc,
                "input_dim": x_train.shape[1],
            }
        )
        if auprc > best_auprc + 1e-12:
            best_auprc = auprc
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            break
    return best_epoch, best_auprc, history


def train_fixed_epochs(
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    device: torch.device,
    seed: int,
    fold: int,
) -> tuple[AdverseMLP, float, list[dict[str, Any]]]:
    set_seed(seed)
    model = AdverseMLP(x.shape[1]).to(device)
    pos_weight = positive_weight(y, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loader = make_loader(x, y, BATCH_SIZE, True, seed)
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, loader, criterion, optimizer, device)
        history.append(
            {
                "fold": fold,
                "stage": "outer_development_refit",
                "epoch": epoch,
                "train_loss": loss,
                "inner_valid_auprc": np.nan,
                "input_dim": x.shape[1],
            }
        )
    return model, float(pos_weight.item()), history


def metric_dict(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    sensitivity = recall_score(y_true, predictions, zero_division=0)
    specificity = tn / max(tn + fp, 1)
    return {
        "threshold": float(threshold),
        "AUROC": float(roc_auc_score(y_true, probabilities)),
        "AUPRC": float(average_precision_score(y_true, probabilities)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "F1": float(f1_score(y_true, predictions, zero_division=0)),
        "Precision": float(precision_score(y_true, predictions, zero_division=0)),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "Brier": float(brier_score_loss(y_true, probabilities)),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, pd.DataFrame]:
    unique = np.unique(probabilities)
    candidates = np.unique(
        np.concatenate(
            [
                np.array([0.0, 0.5, 1.0]),
                unique,
                np.nextafter(unique, np.inf),
            ]
        )
    )
    rows: list[dict[str, float]] = []
    best_threshold = 0.5
    best_key = (-math.inf, -math.inf, -math.inf)
    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(int)
        score = float(balanced_accuracy_score(y_true, predictions))
        key = (score, -abs(float(threshold) - 0.5), -float(threshold))
        rows.append({"threshold": float(threshold), "balanced_accuracy": score})
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold, pd.DataFrame(rows)


def fast_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    n_positive = int(y_true.sum())
    n_negative = n - n_positive
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(n, dtype=np.float64)
    position = 0
    while position < n:
        stop = position + 1
        while stop < n and sorted_scores[stop] == sorted_scores[position]:
            stop += 1
        average_rank = 0.5 * ((position + 1) + stop)
        ranks[order[position:stop]] = average_rank
        position = stop
    positive_rank_sum = float(ranks[y_true == 1].sum())
    return (
        positive_rank_sum - n_positive * (n_positive + 1) / 2.0
    ) / (n_positive * n_negative)


def fast_average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    n_positive = int(y_true.sum())
    if n_positive == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")[::-1]
    y_sorted = y_true[order]
    score_sorted = scores[order]
    distinct = np.where(np.diff(score_sorted))[0]
    threshold_indices = np.r_[distinct, len(scores) - 1]
    true_positives = np.cumsum(y_sorted)[threshold_indices].astype(np.float64)
    false_positives = (threshold_indices + 1).astype(np.float64) - true_positives
    precision = true_positives / np.maximum(true_positives + false_positives, 1.0)
    recall = true_positives / n_positive
    return float(-np.sum(np.diff(np.r_[0.0, recall]) * precision)) * -1.0


def fast_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int8)
    predictions = probabilities >= threshold
    positive = y_true == 1
    negative = ~positive
    tp = int(np.sum(predictions & positive))
    tn = int(np.sum((~predictions) & negative))
    fp = int(np.sum(predictions & negative))
    fn = int(np.sum((~predictions) & positive))
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2.0 * precision * sensitivity / max(precision + sensitivity, 1e-15)
    return {
        "AUROC": fast_auroc(y_true, probabilities),
        "AUPRC": fast_average_precision(y_true, probabilities),
        "Balanced Accuracy": 0.5 * (sensitivity + specificity),
        "F1": f1,
        "Precision": precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Brier": float(np.mean((probabilities - y_true) ** 2)),
    }


def bootstrap_metrics(
    y_true: np.ndarray,
    model_probabilities: dict[str, np.ndarray],
    thresholds: dict[str, float],
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    rng = np.random.default_rng(seed)
    metric_names = [
        "AUROC",
        "AUPRC",
        "Balanced Accuracy",
        "F1",
        "Precision",
        "Sensitivity",
        "Specificity",
        "Brier",
    ]
    samples: dict[str, dict[str, list[float]]] = {
        model: {metric: [] for metric in metric_names} for model in MODEL_ORDER
    }
    paired: dict[str, dict[str, list[float]]] = {
        reference: {"AUROC": [], "AUPRC": [], "Brier improvement": []}
        for reference in ("Dummy", "Logistic")
    }
    valid_repeats = 0
    n = len(y_true)
    for _ in range(repeats):
        indices = rng.integers(0, n, size=n)
        y_sample = y_true[indices]
        if len(np.unique(y_sample)) < 2:
            continue
        valid_repeats += 1
        current: dict[str, dict[str, float]] = {}
        for model in MODEL_ORDER:
            current[model] = fast_metrics(
                y_sample,
                model_probabilities[model][indices],
                thresholds[model],
            )
            for metric in metric_names:
                samples[model][metric].append(float(current[model][metric]))
        for reference in ("Dummy", "Logistic"):
            paired[reference]["AUROC"].append(
                current["Pre+Post MLP"]["AUROC"] - current[reference]["AUROC"]
            )
            paired[reference]["AUPRC"].append(
                current["Pre+Post MLP"]["AUPRC"] - current[reference]["AUPRC"]
            )
            paired[reference]["Brier improvement"].append(
                current[reference]["Brier"] - current["Pre+Post MLP"]["Brier"]
            )

    ci_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for metric in metric_names:
            values = np.asarray(samples[model][metric], dtype=np.float64)
            ci_rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "bootstrap_mean": float(np.mean(values)),
                    "ci_lower": float(np.percentile(values, 2.5)),
                    "ci_upper": float(np.percentile(values, 97.5)),
                    "valid_repeats": valid_repeats,
                }
            )
    pair_rows: list[dict[str, Any]] = []
    for reference, metrics in paired.items():
        for metric, values_list in metrics.items():
            values = np.asarray(values_list, dtype=np.float64)
            lower = float(np.percentile(values, 2.5))
            upper = float(np.percentile(values, 97.5))
            pair_rows.append(
                {
                    "reference": reference,
                    "comparison": "Pre+Post MLP",
                    "metric": metric,
                    "difference_mean": float(np.mean(values)),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "crosses_zero": bool(lower <= 0 <= upper),
                    "valid_repeats": valid_repeats,
                }
            )
    return pd.DataFrame(ci_rows), pd.DataFrame(pair_rows), valid_repeats

def load_task(path: Path, expected_split: str) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(path, dtype={"patient_id": str})
    required_first = ["patient_id", "task_split", "adverse"]
    if frame.columns[:3].tolist() != required_first:
        raise AssertionError(
            f"{path}: first columns must be {required_first}, found {frame.columns[:3].tolist()}"
        )
    if frame["patient_id"].duplicated().any():
        raise AssertionError(f"{path}: duplicate patient IDs")
    if not frame["task_split"].astype(str).eq(expected_split).all():
        raise AssertionError(f"{path}: split mismatch")
    y = pd.to_numeric(frame["adverse"], errors="raise").astype(int)
    if not set(y.unique()).issubset({0, 1}):
        raise AssertionError(f"{path}: non-binary label")
    feature_names = frame.columns[3:].tolist()
    if len(feature_names) != EXPECTED_RAW_FEATURES:
        raise AssertionError(
            f"{path}: expected {EXPECTED_RAW_FEATURES} raw features, found {len(feature_names)}"
        )
    for column in feature_names:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if np.isinf(frame[column].to_numpy(dtype=np.float64)).any():
            raise AssertionError(f"{path}: infinity in {column}")
    return frame, feature_names


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            if any(path.iterdir()):
                raise FileExistsError(f"Refusing to overwrite non-empty output directory: {path}")
        else:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    for subdir in ("models", "preprocessors", "reports"):
        (path / subdir).mkdir(parents=True, exist_ok=True)


def save_checkpoint(
    path: Path,
    model: AdverseMLP,
    fold: int,
    best_epoch: int,
    input_dim: int,
    selected_features: list[str],
    pos_weight: float,
) -> None:
    torch.save(
        {
            "fold": fold,
            "state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "input_dim": input_dim,
            "selected_features": selected_features,
            "pos_weight": pos_weight,
            "architecture": "input->64->16->1, ReLU, Dropout(0.30/0.20)",
        },
        path,
    )


def write_report(
    path: Path,
    train_metrics: pd.DataFrame,
    valid_metrics: pd.DataFrame,
    fold_frame: pd.DataFrame,
    valid_repeats: int,
) -> None:
    lines = [
        "# adverse_prepost_fullseq_v2_mlp report",
        "",
        "## Protocol",
        "",
        "- Input: 147 Pre + 147 Post full-sequence model-candidate features.",
        "- Model: simple MLP classification head, d→64→16→1.",
        "- Loss: BCEWithLogitsLoss with fold-local positive weight.",
        "- Outer 5-fold OOF; inner holdout used only to select epoch.",
        "- Fold-local preprocessing; independent Valid never fits preprocessing or thresholds.",
        "- Valid prediction is the mean probability from five outer models.",
        "",
        "## Fold summary",
        "",
        "```text\n" + fold_frame.to_string(index=False) + "\n```",
        "",
        "## Train pooled OOF metrics",
        "",
        "```text\n" + train_metrics.to_string(index=False) + "\n```",
        "",
        "## Independent Valid metrics",
        "",
        "```text\n" + valid_metrics.to_string(index=False) + "\n```",
        "",
        f"- Valid bootstrap effective repeats: {valid_repeats}/{BOOTSTRAP_REPEATS}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--valid-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--skip-expected-assertions", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prepare_output(args.output_dir, args.overwrite)
    logger = RunLogger(args.output_dir / "run.log")
    running_path = args.output_dir / ".RUNNING"
    running_path.write_text(f"pid={os.getpid()}\nstarted={utc_now()}\n", encoding="utf-8")
    start_time = time.time()

    try:
        set_seed(SEED)
        train_frame, feature_names = load_task(args.train_csv, "Train")
        valid_frame, valid_feature_names = load_task(args.valid_csv, "Valid")
        if feature_names != valid_feature_names:
            raise AssertionError("Train/Valid feature names or order differ")
        overlap = set(train_frame["patient_id"]) & set(valid_frame["patient_id"])
        if overlap:
            raise AssertionError(f"Train/Valid patient overlap: {sorted(overlap)[:10]}")

        y_train = train_frame["adverse"].to_numpy(dtype=np.int64)
        y_valid = valid_frame["adverse"].to_numpy(dtype=np.int64)
        if not args.skip_expected_assertions:
            assertions = {
                "train_rows": (len(train_frame), EXPECTED_TRAIN_ROWS),
                "valid_rows": (len(valid_frame), EXPECTED_VALID_ROWS),
                "train_positive": (int(y_train.sum()), EXPECTED_TRAIN_POSITIVE),
                "valid_positive": (int(y_valid.sum()), EXPECTED_VALID_POSITIVE),
            }
            failures = [
                f"{name}: expected={expected} actual={actual}"
                for name, (actual, expected) in assertions.items()
                if actual != expected
            ]
            if failures:
                raise AssertionError("Expected cohort assertion failure:\n" + "\n".join(failures))

        if args.device.startswith("cuda"):
            if not torch.cuda.is_available():
                if not args.allow_cpu:
                    raise RuntimeError("CUDA requested but unavailable")
                device = torch.device("cpu")
            else:
                device = torch.device(args.device)
        else:
            if not args.allow_cpu:
                raise RuntimeError("CPU mode requires --allow-cpu")
            device = torch.device("cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
        logger.log(
            f"Loaded Train={len(train_frame)} positive={int(y_train.sum())}; "
            f"Valid={len(valid_frame)} positive={int(y_valid.sum())}; raw_features={len(feature_names)}"
        )
        logger.log(f"Device={device}; CUDA available={torch.cuda.is_available()}")

        outer = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        oof_probabilities = {model: np.full(len(train_frame), np.nan) for model in MODEL_ORDER}
        oof_assignments = {model: np.zeros(len(train_frame), dtype=int) for model in MODEL_ORDER}
        valid_fold_probabilities = {
            model: np.full((len(valid_frame), N_SPLITS), np.nan) for model in MODEL_ORDER
        }
        fold_rows: list[dict[str, Any]] = []
        training_history: list[dict[str, Any]] = []

        for fold, (development_index, holdout_index) in enumerate(
            outer.split(np.zeros(len(y_train)), y_train), start=1
        ):
            y_development = y_train[development_index]
            y_holdout = y_train[holdout_index]
            inner_train_index, inner_valid_index = train_test_split(
                development_index,
                test_size=INNER_VALID_FRACTION,
                random_state=SEED + fold,
                stratify=y_train[development_index],
            )

            inner_preprocessor = FoldPreprocessor(feature_names).fit(
                train_frame.iloc[inner_train_index]
            )
            x_inner_train = inner_preprocessor.transform(train_frame.iloc[inner_train_index])
            x_inner_valid = inner_preprocessor.transform(train_frame.iloc[inner_valid_index])
            best_epoch, best_inner_auprc, inner_history = train_with_early_stopping(
                x_inner_train,
                y_train[inner_train_index],
                x_inner_valid,
                y_train[inner_valid_index],
                device,
                seed=SEED + fold,
                fold=fold,
            )
            training_history.extend(inner_history)

            full_preprocessor = FoldPreprocessor(feature_names).fit(
                train_frame.iloc[development_index]
            )
            x_development = full_preprocessor.transform(train_frame.iloc[development_index])
            x_holdout = full_preprocessor.transform(train_frame.iloc[holdout_index])
            x_valid = full_preprocessor.transform(valid_frame)

            model, pos_weight_value, refit_history = train_fixed_epochs(
                x_development,
                y_development,
                best_epoch,
                device,
                seed=SEED + 1000 + fold,
                fold=fold,
            )
            training_history.extend(refit_history)
            mlp_holdout = predict_mlp(model, x_holdout, device)
            mlp_valid = predict_mlp(model, x_valid, device)

            logistic = LogisticRegression(
                solver="liblinear",
                C=1.0,
                class_weight="balanced",
                max_iter=2000,
                random_state=SEED + fold,
            )
            logistic.fit(x_development, y_development)
            logistic_holdout = logistic.predict_proba(x_holdout)[:, 1]
            logistic_valid = logistic.predict_proba(x_valid)[:, 1]

            dummy = DummyClassifier(strategy="prior")
            dummy.fit(np.zeros((len(development_index), 1)), y_development)
            dummy_holdout = dummy.predict_proba(np.zeros((len(holdout_index), 1)))[:, 1]
            dummy_valid = dummy.predict_proba(np.zeros((len(valid_frame), 1)))[:, 1]

            fold_predictions = {
                "Dummy": (dummy_holdout, dummy_valid),
                "Logistic": (logistic_holdout, logistic_valid),
                "Pre+Post MLP": (mlp_holdout, mlp_valid),
            }
            for model_name, (holdout_prob, valid_prob) in fold_predictions.items():
                oof_probabilities[model_name][holdout_index] = holdout_prob
                oof_assignments[model_name][holdout_index] += 1
                valid_fold_probabilities[model_name][:, fold - 1] = valid_prob

            torch.save(
                {"strategy": "prior", "class_prior": dummy.class_prior_.tolist()},
                args.output_dir / "models" / f"dummy_fold_{fold}.pt",
            )
            joblib.dump(logistic, args.output_dir / "models" / f"logistic_fold_{fold}.joblib")
            joblib.dump(
                full_preprocessor,
                args.output_dir / "preprocessors" / f"fold_{fold}_preprocessor.joblib",
            )
            save_checkpoint(
                args.output_dir / "models" / f"mlp_fold_{fold}.pt",
                model,
                fold,
                best_epoch,
                x_development.shape[1],
                full_preprocessor.kept_features,
                pos_weight_value,
            )
            (args.output_dir / "preprocessors" / f"fold_{fold}_feature_audit.json").write_text(
                json.dumps(full_preprocessor.audit_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            fold_row = {
                "fold": fold,
                "development_n": len(development_index),
                "development_positive": int(y_development.sum()),
                "holdout_n": len(holdout_index),
                "holdout_positive": int(y_holdout.sum()),
                "inner_train_n": len(inner_train_index),
                "inner_valid_n": len(inner_valid_index),
                "best_epoch": best_epoch,
                "inner_best_AUPRC": best_inner_auprc,
                "selected_feature_count": len(full_preprocessor.kept_features),
                "dropped_high_missing": len(full_preprocessor.dropped_high_missing),
                "dropped_constant": len(full_preprocessor.dropped_constant),
                "pos_weight": pos_weight_value,
            }
            fold_rows.append(fold_row)
            logger.log(
                f"Fold {fold}/{N_SPLITS} complete | dev={len(development_index)} "
                f"holdout={len(holdout_index)} best_epoch={best_epoch} "
                f"features={len(full_preprocessor.kept_features)} inner_AP={best_inner_auprc:.6f}"
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        for model_name in MODEL_ORDER:
            if not np.all(oof_assignments[model_name] == 1):
                raise AssertionError(f"{model_name}: each Train patient must have exactly one OOF prediction")
            if not np.isfinite(oof_probabilities[model_name]).all():
                raise AssertionError(f"{model_name}: non-finite OOF probabilities")
            if not np.isfinite(valid_fold_probabilities[model_name]).all():
                raise AssertionError(f"{model_name}: non-finite Valid fold probabilities")

        valid_probabilities = {
            model_name: valid_fold_probabilities[model_name].mean(axis=1)
            for model_name in MODEL_ORDER
        }
        thresholds: dict[str, float] = {}
        threshold_frames: list[pd.DataFrame] = []
        for model_name in MODEL_ORDER:
            threshold, search = choose_threshold(y_train, oof_probabilities[model_name])
            thresholds[model_name] = threshold
            search.insert(0, "model", model_name)
            threshold_frames.append(search)
        pd.concat(threshold_frames, ignore_index=True).to_csv(
            args.output_dir / "threshold_search.csv", index=False
        )
        (args.output_dir / "thresholds.json").write_text(
            json.dumps(thresholds, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        frozen_protocol = {
            "created_utc": utc_now(),
            "train_input_sha256": sha256_file(args.train_csv),
            "valid_input_sha256": sha256_file(args.valid_csv),
            "raw_feature_count": len(feature_names),
            "feature_names": feature_names,
            "outer_folds": N_SPLITS,
            "inner_valid_fraction": INNER_VALID_FRACTION,
            "threshold_source": "Train pooled OOF only",
            "valid_ensemble": "mean probability from five outer-fold models",
            "preprocessing": {
                "max_missing_rate": MAX_MISSING_RATE,
                "winsorization": [LOWER_CLIP_QUANTILE, UPPER_CLIP_QUANTILE],
                "imputation": "fold-training median",
                "constant_filter": MIN_VARIANCE,
                "scaler": "RobustScaler(25,75)",
            },
            "model": "d->64->16->1",
            "loss": "BCEWithLogitsLoss with fold-local pos_weight",
            "thresholds": thresholds,
        }
        (args.output_dir / "frozen_protocol.json").write_text(
            json.dumps(frozen_protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        train_metric_rows = []
        valid_metric_rows = []
        for model_name in MODEL_ORDER:
            train_metric_rows.append(
                {"model": model_name, **metric_dict(y_train, oof_probabilities[model_name], thresholds[model_name])}
            )
            valid_metric_rows.append(
                {"model": model_name, **metric_dict(y_valid, valid_probabilities[model_name], thresholds[model_name])}
            )
        train_metrics = pd.DataFrame(train_metric_rows)
        valid_metrics = pd.DataFrame(valid_metric_rows)

        oof_output = train_frame[["patient_id", "task_split", "adverse"]].copy()
        valid_output = valid_frame[["patient_id", "task_split", "adverse"]].copy()
        for model_name in MODEL_ORDER:
            safe = model_name.lower().replace("+", "plus").replace(" ", "_")
            oof_output[f"{safe}_probability"] = oof_probabilities[model_name]
            oof_output[f"{safe}_prediction"] = (
                oof_probabilities[model_name] >= thresholds[model_name]
            ).astype(int)
            valid_output[f"{safe}_probability"] = valid_probabilities[model_name]
            valid_output[f"{safe}_prediction"] = (
                valid_probabilities[model_name] >= thresholds[model_name]
            ).astype(int)
            for fold in range(1, N_SPLITS + 1):
                valid_output[f"{safe}_fold_{fold}_probability"] = valid_fold_probabilities[model_name][
                    :, fold - 1
                ]

        oof_output.to_csv(args.output_dir / "train_oof_predictions.csv", index=False)
        valid_output.to_csv(args.output_dir / "valid_predictions.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(args.output_dir / "fold_metrics.csv", index=False)
        pd.DataFrame(training_history).to_csv(args.output_dir / "training_history.csv", index=False)
        train_metrics.to_csv(args.output_dir / "train_oof_metrics.csv", index=False)
        valid_metrics.to_csv(args.output_dir / "valid_metrics.csv", index=False)

        bootstrap_ci, paired_bootstrap, valid_repeats = bootstrap_metrics(
            y_valid,
            valid_probabilities,
            thresholds,
            BOOTSTRAP_REPEATS,
            seed=SEED + 9000,
        )
        bootstrap_ci.to_csv(args.output_dir / "bootstrap_confidence_intervals.csv", index=False)
        paired_bootstrap.to_csv(args.output_dir / "paired_bootstrap_comparisons.csv", index=False)

        configuration = {
            "seed": SEED,
            "n_splits": N_SPLITS,
            "inner_valid_fraction": INNER_VALID_FRACTION,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            "runtime_seconds": time.time() - start_time,
        }
        (args.output_dir / "configuration.json").write_text(
            json.dumps(configuration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        write_report(
            args.output_dir / "reports" / "report.md",
            train_metrics,
            valid_metrics,
            pd.DataFrame(fold_rows),
            valid_repeats,
        )

        running_path.unlink(missing_ok=True)
        (args.output_dir / ".SUCCESS").write_text(utc_now() + "\n", encoding="utf-8")
        (args.output_dir / "exit_status.txt").write_text("0\n", encoding="utf-8")
        logger.log(f"Experiment completed successfully in {time.time() - start_time:.3f}s")
        logger.close()
        return 0
    except Exception:
        (args.output_dir / "exit_status.txt").write_text("1\n", encoding="utf-8")
        logger.log(traceback.format_exc())
        logger.close()
        raise


if __name__ == "__main__":
    raise SystemExit(main())

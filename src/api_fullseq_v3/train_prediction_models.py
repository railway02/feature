#!/usr/bin/env python3
"""Train Dummy, Logistic and MLP baselines for api_fullseq_v3 tasks.

All fitting, imputation, scaling, early stopping and threshold selection use the
provided Train split only. The official Valid split is prediction/evaluation
only. Record-level tasks use patient_id as the grouping variable.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover
    StratifiedGroupKFold = None


SEED = 42
C_GRID = (0.01, 0.1, 1.0, 10.0)


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
    return float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else float("nan")


def safe_ap(y: np.ndarray, probability: np.ndarray) -> float:
    return float(average_precision_score(y, probability)) if len(np.unique(y)) == 2 else float("nan")


def youden_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y, probability)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    score = tpr[finite] - fpr[finite]
    return float(thresholds[finite][int(np.argmax(score))])


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
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
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
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "brier": float(brier_score_loss(y, probability)),
        "threshold_from_train_oof": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def n_splits_for(y: np.ndarray, groups: np.ndarray, requested: int = 5) -> int:
    class_min = int(np.bincount(y, minlength=2).min())
    group_count = len(np.unique(groups))
    return max(2, min(requested, class_min, group_count))


def outer_splits(y: np.ndarray, groups: np.ndarray, requested: int = 5):
    n_splits = n_splits_for(y, groups, requested)
    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        return list(splitter.split(np.zeros(len(y)), y, groups))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return list(splitter.split(np.zeros(len(y)), y))


def logistic_pipeline(c_value: float) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", RobustScaler()),
        (
            "model",
            LogisticRegression(
                C=float(c_value),
                class_weight="balanced",
                solver="liblinear",
                max_iter=5000,
                random_state=SEED,
            ),
        ),
    ])


def select_logistic_c(
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    candidates: Iterable[float] = C_GRID,
) -> tuple[float, dict[str, float]]:
    splits = outer_splits(y, groups, requested=4)
    scores: dict[str, float] = {}
    for c_value in candidates:
        predictions = np.full(len(y), np.nan, dtype=np.float64)
        for train_index, holdout_index in splits:
            model = logistic_pipeline(float(c_value))
            model.fit(x.iloc[train_index], y[train_index])
            predictions[holdout_index] = model.predict_proba(x.iloc[holdout_index])[:, 1]
        valid = np.isfinite(predictions)
        score = safe_ap(y[valid], predictions[valid])
        scores[str(c_value)] = score
    best = max(candidates, key=lambda value: (scores[str(value)], -float(value)))
    return float(best), scores


class MLP(nn.Module):
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(1)


@dataclass
class MLPFit:
    model: MLP
    imputer: SimpleImputer
    scaler: RobustScaler
    best_epoch: int
    best_validation_ap: float


def group_early_stop_split(y: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    for offset in range(40):
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed + offset)
        train_index, validation_index = next(splitter.split(np.zeros(len(y)), y, groups))
        if len(np.unique(y[train_index])) == 2 and len(np.unique(y[validation_index])) == 2:
            return train_index, validation_index
    # Deterministic fallback for unusually small/imbalanced tasks.
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    train_index, validation_index = next(splitter.split(np.zeros(len(y)), y))
    return train_index, validation_index


def preprocess_fit(x: pd.DataFrame) -> tuple[np.ndarray, SimpleImputer, RobustScaler]:
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    transformed = imputer.fit_transform(x)
    scaler = RobustScaler()
    transformed = scaler.fit_transform(transformed)
    return transformed.astype(np.float32), imputer, scaler


def preprocess_transform(x: pd.DataFrame, imputer: SimpleImputer, scaler: RobustScaler) -> np.ndarray:
    return scaler.transform(imputer.transform(x)).astype(np.float32)


def fit_mlp(
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    device: torch.device,
    seed: int,
    max_epochs: int = 120,
    patience: int = 15,
    batch_size: int = 32,
) -> MLPFit:
    set_seed(seed)
    train_index, validation_index = group_early_stop_split(y, groups, seed)
    x_train, imputer, scaler = preprocess_fit(x.iloc[train_index])
    x_validation = preprocess_transform(x.iloc[validation_index], imputer, scaler)
    y_train = y[train_index].astype(np.float32)
    y_validation = y[validation_index].astype(np.float32)

    model = MLP(x_train.shape[1]).to(device)
    positive = max(int((y_train == 1).sum()), 1)
    negative = max(int((y_train == 0).sum()), 1)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negative / positive], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(y_train)
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
        drop_last=False,
    )
    validation_tensor = torch.from_numpy(x_validation).to(device)

    best_state: dict[str, torch.Tensor] | None = None
    best_ap = -math.inf
    best_epoch = 0
    stale = 0
    for epoch in range(1, max_epochs + 1):
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
            probability = torch.sigmoid(model(validation_tensor)).cpu().numpy()
        score = safe_ap(y_validation.astype(int), probability)
        if score > best_ap + 1e-6:
            best_ap = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("MLP failed to produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    return MLPFit(model, imputer, scaler, best_epoch, float(best_ap))


def mlp_probability(fit: MLPFit, x: pd.DataFrame, device: torch.device) -> np.ndarray:
    array = preprocess_transform(x, fit.imputer, fit.scaler)
    fit.model.eval()
    with torch.no_grad():
        logits = fit.model(torch.from_numpy(array).to(device))
        return torch.sigmoid(logits).cpu().numpy().astype(np.float64)


def train_task(task_dir: Path, output_dir: Path, device: torch.device) -> dict[str, Any]:
    config = json.loads((task_dir / "task_config.json").read_text(encoding="utf-8"))
    task_name = config["task_name"]
    features = list(config["feature_columns"])
    train = pd.read_csv(task_dir / "train.csv", dtype={"patient_id": str})
    valid = pd.read_csv(task_dir / "valid.csv", dtype={"patient_id": str})
    if set(features) - set(train.columns) or set(features) - set(valid.columns):
        raise KeyError(f"{task_name}: feature columns changed")
    x_train = train[features]
    x_valid = valid[features]
    y_train = pd.to_numeric(train["target"], errors="raise").astype(int).to_numpy()
    y_valid = pd.to_numeric(valid["target"], errors="raise").astype(int).to_numpy()
    groups = train["patient_id"].astype(str).to_numpy()
    if len(np.unique(y_train)) != 2 or len(np.unique(y_valid)) != 2:
        raise AssertionError(f"{task_name}: both splits must contain both classes")

    task_output = output_dir / task_name
    task_output.mkdir(parents=True, exist_ok=True)
    folds = outer_splits(y_train, groups, requested=5)

    # Dummy prior.
    prior = float(np.mean(y_train))
    dummy_oof = np.full(len(train), prior, dtype=np.float64)
    dummy_valid = np.full(len(valid), prior, dtype=np.float64)

    # Nested-ish Logistic: C chosen within each outer development split.
    logistic_oof = np.full(len(train), np.nan, dtype=np.float64)
    logistic_valid_fold: list[np.ndarray] = []
    logistic_fold_rows: list[dict[str, Any]] = []
    for fold, (development, holdout) in enumerate(folds, start=1):
        best_c, inner_scores = select_logistic_c(
            x_train.iloc[development], y_train[development], groups[development]
        )
        model = logistic_pipeline(best_c)
        model.fit(x_train.iloc[development], y_train[development])
        logistic_oof[holdout] = model.predict_proba(x_train.iloc[holdout])[:, 1]
        logistic_valid_fold.append(model.predict_proba(x_valid)[:, 1])
        logistic_fold_rows.append({
            "fold": fold,
            "development_rows": int(len(development)),
            "holdout_rows": int(len(holdout)),
            "best_c": best_c,
            "inner_ap_by_c": json.dumps(inner_scores, sort_keys=True),
        })
    logistic_valid = np.mean(np.stack(logistic_valid_fold, axis=0), axis=0)
    global_c, global_c_scores = select_logistic_c(x_train, y_train, groups)
    final_logistic = logistic_pipeline(global_c)
    final_logistic.fit(x_train, y_train)
    joblib.dump(final_logistic, task_output / "logistic_full_train.joblib")

    # MLP outer OOF and official-Valid ensemble.
    mlp_oof = np.full(len(train), np.nan, dtype=np.float64)
    mlp_valid_fold: list[np.ndarray] = []
    mlp_fold_rows: list[dict[str, Any]] = []
    model_dir = task_output / "mlp_folds"
    model_dir.mkdir(parents=True, exist_ok=True)
    for fold, (development, holdout) in enumerate(folds, start=1):
        fit = fit_mlp(
            x_train.iloc[development],
            y_train[development],
            groups[development],
            device,
            SEED + fold,
        )
        mlp_oof[holdout] = mlp_probability(fit, x_train.iloc[holdout], device)
        mlp_valid_fold.append(mlp_probability(fit, x_valid, device))
        torch.save(
            {
                "input_dim": int(next(fit.model.parameters()).shape[1]),
                "state_dict": {key: value.detach().cpu() for key, value in fit.model.state_dict().items()},
                "best_epoch": fit.best_epoch,
                "best_validation_ap": fit.best_validation_ap,
            },
            model_dir / f"fold_{fold}.pt",
        )
        joblib.dump(
            {"imputer": fit.imputer, "scaler": fit.scaler},
            model_dir / f"fold_{fold}_preprocess.joblib",
        )
        mlp_fold_rows.append({
            "fold": fold,
            "development_rows": int(len(development)),
            "holdout_rows": int(len(holdout)),
            "best_epoch": int(fit.best_epoch),
            "early_stop_validation_ap": float(fit.best_validation_ap),
        })
    mlp_valid = np.mean(np.stack(mlp_valid_fold, axis=0), axis=0)

    if not np.isfinite(logistic_oof).all() or not np.isfinite(mlp_oof).all():
        raise AssertionError(f"{task_name}: incomplete OOF predictions")

    probabilities = {
        "Dummy": (dummy_oof, dummy_valid),
        "Logistic": (logistic_oof, logistic_valid),
        "MLP": (mlp_oof, mlp_valid),
    }
    metrics: list[dict[str, Any]] = []
    thresholds: dict[str, float] = {}
    train_predictions = train[[column for column in ["record_uid", "series_uid", "patient_id", "split", "target"] if column in train.columns]].copy()
    valid_predictions = valid[[column for column in ["record_uid", "series_uid", "patient_id", "split", "target"] if column in valid.columns]].copy()
    for model_name, (oof_probability, valid_probability) in probabilities.items():
        threshold = youden_threshold(y_train, oof_probability)
        thresholds[model_name] = threshold
        metrics.append(metric_row(task_name, model_name, "Train_OOF", y_train, oof_probability, threshold))
        metrics.append(metric_row(task_name, model_name, "Valid", y_valid, valid_probability, threshold))
        train_predictions[f"{model_name.lower()}_probability"] = oof_probability
        valid_predictions[f"{model_name.lower()}_probability"] = valid_probability

    atomic_csv(pd.DataFrame(metrics), task_output / "metrics.csv")
    atomic_csv(train_predictions, task_output / "train_oof_predictions.csv")
    atomic_csv(valid_predictions, task_output / "valid_predictions.csv")
    atomic_csv(pd.DataFrame(logistic_fold_rows), task_output / "logistic_fold_audit.csv")
    atomic_csv(pd.DataFrame(mlp_fold_rows), task_output / "mlp_fold_audit.csv")
    run_config = {
        "task_name": task_name,
        "feature_count": len(features),
        "train_rows": len(train),
        "valid_rows": len(valid),
        "train_positive": int((y_train == 1).sum()),
        "valid_positive": int((y_valid == 1).sum()),
        "outer_folds": len(folds),
        "thresholds_from_train_oof": thresholds,
        "logistic_global_c": global_c,
        "logistic_global_cv_ap_by_c": global_c_scores,
        "mlp": {
            "hidden": [64, 16],
            "dropout": [0.30, 0.20],
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "max_epochs": 120,
            "patience": 15,
            "batch_size": 32,
            "class_weight": "fold-specific pos_weight",
        },
        "valid_used_for_fitting_or_threshold_selection": False,
        "device": str(device),
        "seed": SEED,
    }
    atomic_json(run_config, task_output / "run_config.json")
    atomic_json(run_config, task_output / ".SUCCESS")
    return run_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    task_root = Path(args.task_root).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for MLP but unavailable")

    task_dirs = sorted(
        path for path in task_root.iterdir()
        if path.is_dir() and (path / "task_config.json").is_file()
    )
    if not task_dirs:
        raise FileNotFoundError(f"No task directories under {task_root}")
    results: dict[str, Any] = {}
    all_metrics: list[pd.DataFrame] = []
    for task_dir in task_dirs:
        config = train_task(task_dir, output, device)
        results[config["task_name"]] = config
        all_metrics.append(pd.read_csv(output / config["task_name"] / "metrics.csv"))
    combined = pd.concat(all_metrics, ignore_index=True)
    atomic_csv(combined, output / "all_task_metrics.csv")
    summary = {
        "version": "api_fullseq_v3_prediction_models_v1",
        "tasks": sorted(results),
        "valid_used_for_training": False,
        "device": str(device),
        "seed": SEED,
    }
    atomic_json(summary, output / "summary.json")
    atomic_json(summary, output / ".MODELS_SUCCESS")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Train CAVE Dummy/Logistic/MLP models with fold-only branch reduction.

For each outer Train fold, the 10,240-dimensional CAVE embedding branch is
reduced to PCA64 and the scalar branch to PCA32 using outer-development data
only. The same reduced representation is then used for deep, scalar and fusion
Logistic models and a fusion MLP. Official Valid is prediction/evaluation only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.random_projection import SparseRandomProjection

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover
    StratifiedGroupKFold = None

SEED = 42
C_GRID = (0.01, 0.1, 1.0, 10.0)
VARIANTS = ("deep", "scalar", "fusion")


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(tmp, path)


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2: return 0.5
    fpr, tpr, thresholds = roc_curve(y, p)
    finite = np.isfinite(thresholds)
    return float(thresholds[finite][int(np.argmax(tpr[finite] - fpr[finite]))]) if finite.any() else 0.5


def metric_row(task: str, model: str, split: str, y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "task": task, "model": model, "split": split, "rows": int(len(y)),
        "positive": int((y == 1).sum()), "negative": int((y == 0).sum()),
        "positive_fraction": float(np.mean(y == 1)), "auroc": safe_auc(y, p),
        "auprc": safe_ap(y, p), "brier": float(brier_score_loss(y, p)),
        "threshold": float(threshold), "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "sensitivity": float(tp / max(tp + fn, 1)), "specificity": float(tn / max(tn + fp, 1)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def n_splits_for(y: np.ndarray, groups: np.ndarray, requested: int) -> int:
    return max(2, min(requested, int(np.bincount(y, minlength=2).min()), len(np.unique(groups))))


def grouped_splits(y: np.ndarray, groups: np.ndarray, requested: int, seed: int):
    n_splits = n_splits_for(y, groups, requested)
    if StratifiedGroupKFold is not None:
        return list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups))
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def make_imputer(add_indicator: bool) -> SimpleImputer:
    try:
        return SimpleImputer(strategy="median", add_indicator=add_indicator, keep_empty_features=True)
    except TypeError:
        return SimpleImputer(strategy="median", add_indicator=add_indicator)


@dataclass
class NumericBranch:
    requested_components: int
    scaler_kind: str
    minimum_finite_fraction: float
    add_indicator: bool
    keep: np.ndarray | None = None
    imputer: SimpleImputer | None = None
    scaler: Any = None
    pca: PCA | None = None

    def fit(self, values: np.ndarray, seed: int) -> "NumericBranch":
        values = np.asarray(values, dtype=np.float64)
        finite_fraction = np.isfinite(values).mean(axis=0)
        with np.errstate(all="ignore"):
            variances = np.nanvar(values, axis=0)
        self.keep = (finite_fraction >= self.minimum_finite_fraction) & np.isfinite(variances) & (variances > 1e-12)
        if not self.keep.any(): raise AssertionError("No usable columns in branch")
        self.imputer = make_imputer(self.add_indicator)
        transformed = self.imputer.fit_transform(values[:, self.keep])
        self.scaler = StandardScaler() if self.scaler_kind == "standard" else RobustScaler()
        transformed = self.scaler.fit_transform(transformed)
        n_components = min(self.requested_components, transformed.shape[0] - 1, transformed.shape[1])
        if n_components < 1: raise AssertionError("Insufficient data for PCA")
        solver = "randomized" if n_components < min(transformed.shape) else "full"
        self.pca = PCA(n_components=n_components, svd_solver=solver, random_state=seed)
        self.pca.fit(transformed)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.keep is None or self.imputer is None or self.scaler is None or self.pca is None:
            raise RuntimeError("Branch not fitted")
        x = self.imputer.transform(np.asarray(values, dtype=np.float64)[:, self.keep])
        return self.pca.transform(self.scaler.transform(x)).astype(np.float32)

    def audit(self) -> dict[str, Any]:
        return {
            "kept_columns": int(self.keep.sum()) if self.keep is not None else 0,
            "components": int(self.pca.n_components_) if self.pca is not None else 0,
            "explained_variance": float(self.pca.explained_variance_ratio_.sum()) if self.pca is not None else None,
        }


@dataclass
class DeepBranch:
    """Fast deterministic compression for the 10,240-D frozen embedding bank.

    Missing phase embeddings are zero-filled and explicitly represented by the
    two missing flags. A sparse Johnson-Lindenstrauss projection reduces the
    branch to 512 dimensions before fold-only scaling and PCA64.
    """
    projection_components: int = 512
    pca_components: int = 64
    projector: SparseRandomProjection | None = None
    scaler: StandardScaler | None = None
    pca: PCA | None = None

    def fit(self, values: np.ndarray, seed: int) -> "DeepBranch":
        x = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        n_projected = min(self.projection_components, x.shape[1])
        self.projector = SparseRandomProjection(
            n_components=n_projected, density="auto", dense_output=True, random_state=seed
        )
        projected = self.projector.fit_transform(x).astype(np.float32)
        self.scaler = StandardScaler()
        projected = self.scaler.fit_transform(projected)
        n_components = min(self.pca_components, projected.shape[0] - 1, projected.shape[1])
        if n_components < 1:
            raise AssertionError("Insufficient data for deep PCA")
        self.pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed + 1)
        self.pca.fit(projected)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.projector is None or self.scaler is None or self.pca is None:
            raise RuntimeError("Deep branch not fitted")
        x = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        projected = self.projector.transform(x).astype(np.float32)
        return self.pca.transform(self.scaler.transform(projected)).astype(np.float32)

    def audit(self) -> dict[str, Any]:
        return {
            "input_dimension": 10240,
            "random_projection_components": int(self.projector.n_components) if self.projector else 0,
            "pca_components": int(self.pca.n_components_) if self.pca else 0,
            "explained_variance": float(self.pca.explained_variance_ratio_.sum()) if self.pca else None,
        }


@dataclass
class FusionPreprocessor:
    deep: DeepBranch | None = None
    scalar: NumericBranch | None = None

    def fit(self, data: dict[str, np.ndarray], seed: int) -> "FusionPreprocessor":
        self.deep = DeepBranch().fit(data["deep"], seed)
        self.scalar = NumericBranch(32, "robust", 0.25, True).fit(data["scalar"], seed + 1000)
        self.transform_all(data)
        return self

    def transform_all(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self.deep is None or self.scalar is None: raise RuntimeError("Preprocessor not fitted")
        d = self.deep.transform(data["deep"])
        s = self.scalar.transform(data["scalar"])
        m = np.asarray(data["missing"], dtype=np.float32)
        result = {
            "deep": np.concatenate([d, m], axis=1).astype(np.float32),
            "scalar": np.concatenate([s, m], axis=1).astype(np.float32),
            "fusion": np.concatenate([d, s, m], axis=1).astype(np.float32),
        }
        if any(not np.isfinite(value).all() for value in result.values()):
            raise AssertionError("Reduced features contain nonfinite values")
        return result

    def audit(self) -> dict[str, Any]:
        return {"deep": self.deep.audit() if self.deep else None, "scalar": self.scalar.audit() if self.scalar else None}


def load_task(task_dir: Path):
    config = json.loads((task_dir / "task_config.json").read_text(encoding="utf-8"))
    train_meta = pd.read_csv(task_dir / "train_meta.csv", dtype={"patient_id": str})
    valid_meta = pd.read_csv(task_dir / "valid_meta.csv", dtype={"patient_id": str})
    train_raw, valid_raw = np.load(task_dir / "train_features.npz"), np.load(task_dir / "valid_features.npz")
    train = {key: train_raw[key] for key in ("deep", "scalar", "missing", "target")}
    valid = {key: valid_raw[key] for key in ("deep", "scalar", "missing", "target")}
    if len(train_meta) != len(train["target"]) or len(valid_meta) != len(valid["target"]):
        raise AssertionError("Task metadata/array row mismatch")
    return config, train_meta, valid_meta, train, valid


def select_c(x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> tuple[float, dict[str, float]]:
    splits = grouped_splits(y, groups, requested=3, seed=seed)
    scores: dict[str, float] = {}
    for c_value in C_GRID:
        predictions = np.full(len(y), np.nan, dtype=np.float64)
        for fit_index, holdout_index in splits:
            model = LogisticRegression(C=float(c_value), class_weight="balanced", solver="liblinear", max_iter=5000, random_state=seed)
            model.fit(x[fit_index], y[fit_index])
            predictions[holdout_index] = model.predict_proba(x[holdout_index])[:, 1]
        scores[str(c_value)] = safe_ap(y, predictions)
    best = max(C_GRID, key=lambda value: (scores[str(value)], -float(value)))
    return float(best), scores


class MLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.30),
            nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.20), nn.Linear(32, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.network(x).squeeze(1)


def early_stop_split(y: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    for offset in range(40):
        train_index, valid_index = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed + offset).split(np.zeros(len(y)), y, groups))
        if len(np.unique(y[train_index])) == 2 and len(np.unique(y[valid_index])) == 2:
            return train_index, valid_index
    return next(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def fit_mlp(x: np.ndarray, y: np.ndarray, groups: np.ndarray, device: torch.device, seed: int):
    set_seed(seed)
    fit_index, stop_index = early_stop_split(y, groups, seed)
    x_fit, x_stop = x[fit_index].astype(np.float32), x[stop_index].astype(np.float32)
    y_fit, y_stop = y[fit_index].astype(np.float32), y[stop_index].astype(int)
    model = MLP(x.shape[1]).to(device)
    positive, negative = max(int((y_fit == 1).sum()), 1), max(int((y_fit == 0).sum()), 1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative / positive], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(x_fit), torch.from_numpy(y_fit)),
        batch_size=min(32, len(x_fit)), shuffle=True, generator=torch.Generator().manual_seed(seed),
    )
    stop_tensor = torch.from_numpy(x_stop).to(device)
    best_state, best_ap, best_epoch, stale = None, -math.inf, 0, 0
    for epoch in range(1, 121):
        model.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): p = torch.sigmoid(model(stop_tensor)).cpu().numpy()
        score = safe_ap(y_stop, p)
        if score > best_ap + 1e-6:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_ap, best_epoch, stale = score, epoch, 0
        else: stale += 1
        if stale >= 15: break
    if best_state is None: raise RuntimeError("MLP failed")
    model.load_state_dict(best_state); model.to(device)
    return model, best_epoch, float(best_ap)


def mlp_predict(model: MLP, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(x.astype(np.float32)).to(device))).cpu().numpy().astype(np.float64)


def train_task(task_dir: Path, output_root: Path, device: torch.device) -> dict[str, Any]:
    config, train_meta, valid_meta, train, valid = load_task(task_dir)
    task_name = config["task_name"]
    out = output_root / task_name; out.mkdir(parents=True, exist_ok=True)
    y_train, y_valid = train["target"].astype(int), valid["target"].astype(int)
    groups = train_meta["patient_id"].astype(str).to_numpy()
    if len(np.unique(y_train)) != 2 or len(np.unique(y_valid)) != 2: raise AssertionError("Both splits need both classes")
    folds = grouped_splits(y_train, groups, 5, SEED)
    probabilities: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "Dummy": (np.full(len(y_train), y_train.mean()), np.full(len(y_valid), y_train.mean()))
    }
    logistic_oof = {variant: np.full(len(y_train), np.nan, dtype=np.float64) for variant in VARIANTS}
    logistic_valid = {variant: [] for variant in VARIANTS}
    mlp_oof = np.full(len(y_train), np.nan, dtype=np.float64); mlp_valid: list[np.ndarray] = []
    audits: list[dict[str, Any]] = []

    for fold, (development, holdout) in enumerate(folds, start=1):
        dev_data = {key: train[key][development] for key in ("deep", "scalar", "missing")}
        hold_data = {key: train[key][holdout] for key in ("deep", "scalar", "missing")}
        pre = FusionPreprocessor().fit(dev_data, SEED + fold * 100)
        dev_x, hold_x, valid_x = pre.transform_all(dev_data), pre.transform_all(hold_data), pre.transform_all(valid)
        fold_dir = out / f"fold_{fold}"; fold_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pre, fold_dir / "preprocessor.joblib")
        for variant in VARIANTS:
            best_c, c_scores = select_c(dev_x[variant], y_train[development], groups[development], SEED + fold * 1000)
            model = LogisticRegression(C=best_c, class_weight="balanced", solver="liblinear", max_iter=5000, random_state=SEED)
            model.fit(dev_x[variant], y_train[development])
            logistic_oof[variant][holdout] = model.predict_proba(hold_x[variant])[:, 1]
            logistic_valid[variant].append(model.predict_proba(valid_x[variant])[:, 1])
            joblib.dump(model, fold_dir / f"logistic_{variant}.joblib")
            audits.append({
                "task": task_name, "fold": fold, "model": f"Logistic_{variant}",
                "development_rows": int(len(development)), "holdout_rows": int(len(holdout)),
                "best_c": best_c, "inner_ap_by_c": json.dumps(c_scores, sort_keys=True),
                "preprocessor": json.dumps(pre.audit(), sort_keys=True),
            })
        mlp, best_epoch, stop_ap = fit_mlp(dev_x["fusion"], y_train[development], groups[development], device, SEED + fold)
        mlp_oof[holdout] = mlp_predict(mlp, hold_x["fusion"], device)
        mlp_valid.append(mlp_predict(mlp, valid_x["fusion"], device))
        torch.save({"input_dim": dev_x["fusion"].shape[1], "state_dict": {k: v.detach().cpu() for k, v in mlp.state_dict().items()}, "best_epoch": best_epoch, "best_validation_ap": stop_ap}, fold_dir / "mlp_fusion.pt")
        audits.append({"task": task_name, "fold": fold, "model": "MLP_fusion", "development_rows": int(len(development)), "holdout_rows": int(len(holdout)), "best_epoch": best_epoch, "early_stop_validation_ap": stop_ap, "preprocessor": json.dumps(pre.audit(), sort_keys=True)})

    for variant in VARIANTS:
        if not np.isfinite(logistic_oof[variant]).all(): raise AssertionError(f"Incomplete Logistic_{variant} OOF")
        probabilities[f"Logistic_{variant}"] = (logistic_oof[variant], np.mean(np.stack(logistic_valid[variant]), axis=0))
    if not np.isfinite(mlp_oof).all(): raise AssertionError("Incomplete MLP OOF")
    probabilities["MLP_fusion"] = (mlp_oof, np.mean(np.stack(mlp_valid), axis=0))

    # Full-Train deployable Logistic models. These do not affect OOF/Valid metrics above.
    full_pre = FusionPreprocessor().fit({key: train[key] for key in ("deep", "scalar", "missing")}, SEED + 9000)
    full_x = full_pre.transform_all(train)
    joblib.dump(full_pre, out / "full_train_preprocessor.joblib")
    for variant in VARIANTS:
        best_c, scores = select_c(full_x[variant], y_train, groups, SEED + 9100)
        model = LogisticRegression(C=best_c, class_weight="balanced", solver="liblinear", max_iter=5000, random_state=SEED)
        model.fit(full_x[variant], y_train)
        joblib.dump({"model": model, "selected_c": best_c, "cv_ap_by_c": scores}, out / f"full_train_logistic_{variant}.joblib")

    metrics, thresholds = [], {}
    train_predictions, valid_predictions = train_meta.copy(), valid_meta.copy()
    for model_name, (oof_p, valid_p) in probabilities.items():
        threshold = youden_threshold(y_train, oof_p); thresholds[model_name] = threshold
        metrics.extend([metric_row(task_name, model_name, "Train_OOF", y_train, oof_p, threshold), metric_row(task_name, model_name, "Valid", y_valid, valid_p, threshold)])
        train_predictions[f"{model_name.lower()}_probability"] = oof_p
        valid_predictions[f"{model_name.lower()}_probability"] = valid_p
    metrics_frame = pd.DataFrame(metrics)
    atomic_csv(metrics_frame, out / "metrics.csv"); atomic_csv(train_predictions, out / "train_oof_predictions.csv"); atomic_csv(valid_predictions, out / "valid_predictions.csv"); atomic_csv(pd.DataFrame(audits), out / "fold_audit.csv")
    learned_oof = metrics_frame[(metrics_frame["split"] == "Train_OOF") & (metrics_frame["model"] != "Dummy")]
    best_model = str(learned_oof.sort_values(["auprc", "auroc"], ascending=False).iloc[0]["model"])
    run_config = {
        "version": "api_fullseq_cave_v3_prediction_models_2", "task_name": task_name,
        "train_rows": int(len(y_train)), "valid_rows": int(len(y_valid)), "outer_folds": len(folds),
        "models": list(probabilities), "best_model_selected_by_train_oof_auprc": best_model,
        "thresholds_from_train_oof": thresholds, "deep_random_projection_components": 512, "deep_pca_components": 64, "scalar_pca_components": 32,
        "reduction_fit_scope": "outer-development only for reported OOF/Valid ensemble",
        "valid_used_for_fitting_selection_early_stopping_or_threshold": False,
        "device": str(device), "seed": SEED,
    }
    atomic_json(run_config, out / "run_config.json"); atomic_json(run_config, out / ".SUCCESS")
    return run_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu"); parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(); task_root, output = Path(args.task_root).resolve(), Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        if not args.overwrite: raise FileExistsError(f"Refusing to overwrite non-empty {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True); device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    task_dirs = sorted(p for p in task_root.iterdir() if p.is_dir() and (p / "task_config.json").is_file())
    if not task_dirs: raise FileNotFoundError(f"No tasks under {task_root}")
    all_metrics, results = [], {}
    for task_dir in task_dirs:
        result = train_task(task_dir, output, device); results[result["task_name"]] = result
        all_metrics.append(pd.read_csv(output / result["task_name"] / "metrics.csv"))
    atomic_csv(pd.concat(all_metrics, ignore_index=True), output / "all_task_metrics.csv")
    summary = {"version": "api_fullseq_cave_v3_prediction_models_2", "tasks": sorted(results), "valid_used_for_training": False, "device": str(device), "seed": SEED}
    atomic_json(summary, output / "summary.json"); atomic_json(summary, output / ".MODELS_SUCCESS")
    print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())

#!/usr/bin/env python3
"""Formal efficient nested grouped-CV training for strict Pre+Post SERIES-level Local-CAVE adverse outcome.

Models
------
- Dummy
- Logistic_Deep
- Logistic_Fusion
- MLP_Deep
- MLP_Fusion

Key safeguards
--------------
- Fixed outer 5-fold assignment grouped by patient_id.
- All preprocessing, PCA, model selection and early stopping are Train-only.
- Logistic uses 3-fold grouped inner CV.
- MLP uses 3-fold grouped inner CV. Within each inner development split, a
  separate grouped early-stopping split selects the epoch; the model is then
  refitted on the full inner development and evaluated on the untouched inner
  holdout.
- Official Valid is prediction/evaluation only.
- Thresholds are selected from pooled Train OOF predictions only.
- Valid uncertainty uses patient-cluster bootstrap.
- Fold caches are reused only when a code/task/config signature matches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_CPU_THREADS = max(1, min(8, os.cpu_count() or 1))
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    if not os.environ.get(_name, "").isdigit():
        os.environ[_name] = str(DEFAULT_CPU_THREADS)

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


SEED = 20260804
C_GRID = (0.01, 0.1, 1.0, 10.0)
LOGISTIC_DEEP_DIMS = (32, 64, 128)
SCALAR_PCA_DIM = 48
MAX_MISSING_RATE = 0.90
LOWER_QUANTILE = 0.005
UPPER_QUANTILE = 0.995
MIN_VARIANCE = 1e-12
DEFAULT_MLP_SEEDS = 3
DEFAULT_MLP_SEARCH_SEEDS = 2
DEFAULT_BOOTSTRAP_REPEATS = 2000
BASE_MODEL_ORDER = (
    "Logistic_Deep",
    "Logistic_Fusion",
    "MLP_Deep",
    "MLP_Fusion",
)
MODEL_ORDER = (
    "Dummy",
    *BASE_MODEL_ORDER,
    "Stacked_Ensemble",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_runtime(cpu_threads: int, device: torch.device) -> dict[str, Any]:
    cpu_threads = max(1, int(cpu_threads))
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    runtime = {
        "cpu_threads": cpu_threads,
        "cuda": device.type == "cuda",
        "tf32": False,
        "amp": False,
    }
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        runtime["tf32"] = True
    return runtime



def make_grad_scaler(enabled: bool):
    """Use the modern AMP API when available, with a compatibility fallback."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled: bool):
    try:
        return torch.amp.autocast("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    material = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


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
    if path.exists() and any(path.iterdir()) and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def safe_auc(y: np.ndarray, probability: np.ndarray) -> float:
    return (
        float(roc_auc_score(y, probability))
        if len(np.unique(y)) == 2
        else float("nan")
    )


def safe_ap(y: np.ndarray, probability: np.ndarray) -> float:
    return (
        float(average_precision_score(y, probability))
        if len(np.unique(y)) == 2
        else float("nan")
    )


def youden_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y, probability)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    score = tpr[finite] - fpr[finite]
    candidates = thresholds[finite][score == np.max(score)]
    return float(
        sorted(candidates, key=lambda value: (abs(value - 0.5), value))[0]
    )


def metric_row(
    model: str,
    split: str,
    y: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y,
        prediction,
        labels=[0, 1],
    ).ravel()
    return {
        "task": "adverse_outcome_series_strict_prepost_v3",
        "model": model,
        "split": split,
        "rows": int(len(y)),
        "positive": int((y == 1).sum()),
        "negative": int((y == 0).sum()),
        "positive_fraction": float(np.mean(y == 1)),
        "AUROC": safe_auc(y, probability),
        "AUPRC": safe_ap(y, probability),
        "Balanced Accuracy": float(
            balanced_accuracy_score(y, prediction)
        ),
        "F1": float(f1_score(y, prediction, zero_division=0)),
        "Precision": float(
            precision_score(y, prediction, zero_division=0)
        ),
        "Sensitivity": float(
            recall_score(y, prediction, zero_division=0)
        ),
        "Specificity": float(tn / max(tn + fp, 1)),
        "Brier": float(brier_score_loss(y, probability)),
        "threshold_from_train_oof": float(threshold),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def load_task(task_root: Path) -> dict[str, dict[str, np.ndarray]]:
    success_path = task_root / ".TASK_SUCCESS.json"
    if not success_path.is_file():
        raise RuntimeError(f"Missing task success lock: {success_path}")
    summary = json.loads(success_path.read_text(encoding="utf-8"))
    if summary.get("version") != "formal_adverse_prepost_local_cave_series_v3":
        raise AssertionError(
            f"Unexpected task version: {summary.get('version')}"
        )

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
                "patient_id",
                "series_uid",
                "scalar_feature_names",
            }
            if split == "train":
                required.add("fold")
            missing = required - set(raw.files)
            if missing:
                raise KeyError(f"{path}: missing arrays {sorted(missing)}")
            result[split] = {
                name: np.asarray(raw[name]) for name in raw.files
            }

    train = result["train"]
    valid = result["valid"]
    for split, data in result.items():
        n = len(data["target"])
        for key in (
            "deep",
            "scalar",
            "patient_id",
            "series_uid",
        ):
            if len(data[key]) != n:
                raise AssertionError(f"{split}: {key} row mismatch")
        if data["deep"].shape != (n, 10240):
            raise AssertionError(
                f"{split}: unexpected deep shape {data['deep'].shape}"
            )
        if not np.isfinite(data["deep"]).all():
            raise AssertionError(f"{split}: nonfinite strict deep values")
        if not set(np.unique(data["target"]).tolist()).issubset({0, 1}):
            raise AssertionError(f"{split}: non-binary target")

    if not np.array_equal(
        train["scalar_feature_names"].astype(str),
        valid["scalar_feature_names"].astype(str),
    ):
        raise AssertionError("Train/Valid scalar schema differs")
    if set(train["patient_id"].astype(str)) & set(
        valid["patient_id"].astype(str)
    ):
        raise AssertionError("Train/Valid patient overlap")
    if sorted(np.unique(train["fold"]).astype(int).tolist()) != [1, 2, 3, 4, 5]:
        raise AssertionError("Train folds must equal 1..5")
    fold_check = pd.DataFrame(
        {
            "patient_id": train["patient_id"].astype(str),
            "fold": train["fold"].astype(int),
        }
    )
    if int(fold_check.groupby("patient_id")["fold"].nunique().max()) != 1:
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

    def fit(
        self,
        x: np.ndarray,
        pca_dim: int,
        seed: int,
    ) -> "ScalarPreprocessor":
        x = np.asarray(x, dtype=np.float64)
        finite = np.isfinite(x)
        missing_rate = 1.0 - finite.mean(axis=0)
        median = np.full(x.shape[1], np.nan, dtype=np.float64)
        for index in np.flatnonzero(
            missing_rate <= self.max_missing_rate
        ):
            values = x[finite[:, index], index]
            if values.size:
                median[index] = np.median(values)

        kept = np.flatnonzero(
            (missing_rate <= self.max_missing_rate)
            & np.isfinite(median)
        )
        if kept.size == 0:
            raise RuntimeError("All scalar columns removed by missing filter")
        selected = x[:, kept]
        lower = np.nanquantile(
            selected,
            self.lower_quantile,
            axis=0,
        )
        upper = np.nanquantile(
            selected,
            self.upper_quantile,
            axis=0,
        )
        med = median[kept]
        clipped = np.clip(selected, lower, upper)
        clipped = np.where(
            np.isfinite(clipped),
            clipped,
            med[None, :],
        )
        variance = np.var(clipped, axis=0)
        keep_variance = np.flatnonzero(
            variance > self.min_variance
        )
        if keep_variance.size == 0:
            raise RuntimeError("All scalar columns removed as constant")

        self.kept_after_missing = kept
        self.kept_after_variance = keep_variance
        self.lower = lower[keep_variance]
        self.upper = upper[keep_variance]
        self.median = med[keep_variance]

        values = selected[:, keep_variance]
        values = np.clip(values, self.lower, self.upper)
        values = np.where(
            np.isfinite(values),
            values,
            self.median[None, :],
        )
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
        required = (
            self.kept_after_missing,
            self.kept_after_variance,
            self.lower,
            self.upper,
            self.median,
            self.scaler,
            self.pca,
        )
        if any(value is None for value in required):
            raise RuntimeError("Scalar preprocessor is not fitted")
        selected = np.asarray(x, dtype=np.float64)[
            :,
            self.kept_after_missing,
        ]
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

    def fit(
        self,
        deep: np.ndarray,
        scalar: np.ndarray,
    ) -> "FusionPreprocessor":
        deep = np.asarray(deep, dtype=np.float32)
        if not np.isfinite(deep).all():
            raise RuntimeError("Nonfinite deep input")
        self.deep_scaler = StandardScaler()
        deep_scaled = self.deep_scaler.fit_transform(deep)
        n_components = max(
            1,
            min(
                int(self.deep_pca_dim),
                deep.shape[0] - 1,
                deep.shape[1],
            ),
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

    def transform(
        self,
        deep: np.ndarray,
        scalar: np.ndarray,
    ) -> np.ndarray:
        if self.deep_scaler is None or self.deep_pca is None:
            raise RuntimeError("Deep preprocessor is not fitted")
        deep_pca = self.deep_pca.transform(
            self.deep_scaler.transform(deep)
        ).astype(np.float32)
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


def slice_fusion_components(
    transformed: np.ndarray,
    requested_deep_dim: int,
    fitted_deep_dim: int,
    use_scalar: bool,
) -> np.ndarray:
    deep_dim = min(int(requested_deep_dim), int(fitted_deep_dim))
    if not use_scalar:
        return transformed[:, :deep_dim]
    scalar = transformed[:, fitted_deep_dim:]
    return np.concatenate(
        [transformed[:, :deep_dim], scalar],
        axis=1,
    ).astype(np.float32, copy=False)


def feasible_group_splits(
    y: np.ndarray,
    groups: np.ndarray,
    requested: int,
) -> int:
    """Find a safe fold count when one patient may own mixed-label series."""
    frame = pd.DataFrame(
        {"group": groups.astype(str), "target": y.astype(int)}
    )
    grouped = frame.groupby("group")["target"]
    group_has_negative = grouped.apply(lambda values: bool((values == 0).any()))
    group_has_positive = grouped.apply(lambda values: bool((values == 1).any()))
    n_splits = min(
        int(requested),
        int(len(group_has_positive)),
        int(group_has_positive.sum()),
        int(group_has_negative.sum()),
    )
    if n_splits < 2:
        raise RuntimeError(
            f"Insufficient class-containing patient groups for {requested} folds"
        )
    return n_splits


def grouped_splits(
    y: np.ndarray,
    groups: np.ndarray,
    requested: int,
    seed: int,
    retries: int = 12,
) -> list[tuple[np.ndarray, np.ndarray]]:
    n_splits = feasible_group_splits(y, groups, requested)
    global_rate = float(np.mean(y))
    target_size = len(y) / n_splits
    best: tuple[float, list[tuple[np.ndarray, np.ndarray]]] | None = None

    for offset in range(retries):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed + offset,
        )
        splits = list(
            splitter.split(np.zeros(len(y)), y, groups)
        )
        valid = True
        score = 0.0
        seen_groups: set[str] = set()
        for development, holdout in splits:
            if (
                len(np.unique(y[development])) != 2
                or len(np.unique(y[holdout])) != 2
            ):
                valid = False
                break
            holdout_groups = set(groups[holdout].astype(str))
            if seen_groups & holdout_groups:
                valid = False
                break
            seen_groups |= holdout_groups
            score += abs(len(holdout) - target_size) / max(target_size, 1)
            score += 5.0 * abs(float(np.mean(y[holdout])) - global_rate)
        if not valid:
            continue
        if best is None or score < best[0]:
            best = (score, splits)
    if best is None:
        raise RuntimeError("Unable to create valid grouped splits")
    return best[1]


def choose_early_stop_split(
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    splits = grouped_splits(
        y,
        groups,
        requested=5,
        seed=seed,
        retries=8,
    )
    target_size = len(y) / len(splits)
    global_rate = float(np.mean(y))
    return min(
        splits,
        key=lambda pair: (
            abs(len(pair[1]) - target_size)
            + 50.0
            * abs(float(np.mean(y[pair[1]])) - global_rate)
        ),
    )


def select_logistic(
    deep: np.ndarray,
    scalar: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    use_scalar: bool,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Nested grouped selection with one max-dimensional PCA per inner fold."""
    splits = grouped_splits(
        y,
        groups,
        requested=3,
        seed=seed,
    )
    max_requested_dim = max(LOGISTIC_DEEP_DIMS)
    predictions = {
        (int(deep_dim), float(c_value)): np.full(
            len(y),
            np.nan,
            dtype=np.float64,
        )
        for deep_dim in LOGISTIC_DEEP_DIMS
        for c_value in C_GRID
    }

    for inner_fold, (development, holdout) in enumerate(
        splits,
        start=1,
    ):
        preprocessor = FusionPreprocessor(
            deep_pca_dim=max_requested_dim,
            use_scalar=use_scalar,
            seed=seed + inner_fold,
        ).fit(deep[development], scalar[development])
        fitted_deep_dim = int(preprocessor.deep_pca.n_components_)
        x_development_full = preprocessor.transform(
            deep[development],
            scalar[development],
        )
        x_holdout_full = preprocessor.transform(
            deep[holdout],
            scalar[holdout],
        )

        for deep_dim in LOGISTIC_DEEP_DIMS:
            x_development = slice_fusion_components(
                x_development_full,
                deep_dim,
                fitted_deep_dim,
                use_scalar,
            )
            x_holdout = slice_fusion_components(
                x_holdout_full,
                deep_dim,
                fitted_deep_dim,
                use_scalar,
            )
            for c_value in C_GRID:
                model = LogisticRegression(
                    C=float(c_value),
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=10000,
                    random_state=seed,
                )
                model.fit(x_development, y[development])
                predictions[(int(deep_dim), float(c_value))][holdout] = (
                    model.predict_proba(x_holdout)[:, 1]
                )

    rows: list[dict[str, Any]] = []
    best_key: tuple[float, float, int, float] | None = None
    best_config: dict[str, Any] = {}
    for (deep_dim, c_value), probability in predictions.items():
        if not np.isfinite(probability).all():
            raise RuntimeError("Incomplete inner logistic OOF")
        ap = safe_ap(y, probability)
        auc = safe_auc(y, probability)
        row = {
            "deep_pca_dim": int(deep_dim),
            "use_scalar": bool(use_scalar),
            "C": float(c_value),
            "inner_pooled_AUPRC": ap,
            "inner_pooled_AUROC": auc,
            "pca_reuse_policy": (
                f"fit max {max_requested_dim} once per inner fold; "
                "slice leading components"
            ),
        }
        rows.append(row)
        key = (ap, auc, -int(deep_dim), -float(c_value))
        if best_key is None or key > best_key:
            best_key = key
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
        pin_memory=torch.cuda.is_available(),
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
        tensor = torch.from_numpy(
            x[start : start + 1024].astype(np.float32)
        ).to(device, non_blocking=True)
        logits = model(tensor)
        probabilities.append(
            torch.sigmoid(logits).cpu().numpy()
        )
    return np.concatenate(probabilities).astype(np.float64)


def fit_mlp_early_stop(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    config: MLPConfig,
    device: torch.device,
    seed: int,
    amp_enabled: bool,
) -> tuple[int, float, float, list[dict[str, Any]]]:
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
    scaler = make_grad_scaler(amp_enabled)
    loader = make_loader(
        x_train,
        y_train,
        config.batch_size,
        seed,
    )

    best_epoch = 0
    best_ap = -math.inf
    best_auc = -math.inf
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(amp_enabled):
                loss = criterion(model(batch_x), batch_y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * len(batch_y)
            total_rows += len(batch_y)

        probability = mlp_predict(model, x_validation, device)
        ap = safe_ap(y_validation, probability)
        auc = safe_auc(y_validation, probability)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(total_rows, 1),
                "validation_AUPRC": ap,
                "validation_AUROC": auc,
            }
        )
        current_key = (ap, auc, -epoch)
        best_key = (best_ap, best_auc, -best_epoch)
        improved = (
            ap > best_ap + 1e-6
            or (
                abs(ap - best_ap) <= 1e-6
                and auc > best_auc + 1e-6
            )
        )
        if improved:
            best_epoch = epoch
            best_ap = ap
            best_auc = auc
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break

    if best_epoch <= 0:
        raise RuntimeError("MLP early stopping failed")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_epoch, float(best_ap), float(best_auc), history


def fit_mlp_fixed_epochs(
    x: np.ndarray,
    y: np.ndarray,
    config: MLPConfig,
    epochs: int,
    device: torch.device,
    seed: int,
    amp_enabled: bool,
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
    scaler = make_grad_scaler(amp_enabled)
    loader = make_loader(x, y, config.batch_size, seed)
    for _ in range(max(1, int(epochs))):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(amp_enabled):
                loss = criterion(model(batch_x), batch_y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )
            scaler.step(optimizer)
            scaler.update()
    return model


def evaluate_mlp_config_inner_cv(
    deep: np.ndarray,
    scalar: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    config: MLPConfig,
    use_scalar: bool,
    device: torch.device,
    seed: int,
    search_seeds: int,
    amp_enabled: bool,
    transform_cache: dict[Any, Any] | None = None,
    shared_splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, list[int], list[dict[str, Any]]]:
    splits = (
        shared_splits
        if shared_splits is not None
        else grouped_splits(
            y,
            groups,
            requested=3,
            seed=seed,
        )
    )
    pooled = np.full(len(y), np.nan, dtype=np.float64)
    selected_epochs: list[int] = []
    rows: list[dict[str, Any]] = []

    for inner_fold, (inner_development, inner_holdout) in enumerate(
        splits,
        start=1,
    ):
        max_mlp_deep_dim = max(item.deep_pca_dim for item in MLP_CONFIGS)
        cache_key = ("transform_max", inner_fold, int(max_mlp_deep_dim))
        cached_transform = (
            transform_cache.get(cache_key)
            if transform_cache is not None
            else None
        )
        if cached_transform is None:
            early_key = ("early_indices", inner_fold)
            cached_early = (
                transform_cache.get(early_key)
                if transform_cache is not None
                else None
            )
            if cached_early is None:
                early_train_rel, early_valid_rel = choose_early_stop_split(
                    y[inner_development],
                    groups[inner_development],
                    seed + inner_fold * 100,
                )
                early_train = inner_development[early_train_rel]
                early_valid = inner_development[early_valid_rel]
                if transform_cache is not None:
                    transform_cache[early_key] = (
                        early_train,
                        early_valid,
                    )
            else:
                early_train, early_valid = cached_early

            early_preprocessor = FusionPreprocessor(
                deep_pca_dim=max_mlp_deep_dim,
                use_scalar=use_scalar,
                seed=seed + inner_fold * 1000 + 1,
            ).fit(deep[early_train], scalar[early_train])
            x_early_train = early_preprocessor.transform(
                deep[early_train],
                scalar[early_train],
            )
            x_early_valid = early_preprocessor.transform(
                deep[early_valid],
                scalar[early_valid],
            )

            refit_preprocessor = FusionPreprocessor(
                deep_pca_dim=max_mlp_deep_dim,
                use_scalar=use_scalar,
                seed=seed + inner_fold * 1000 + 2,
            ).fit(
                deep[inner_development],
                scalar[inner_development],
            )
            x_inner_development = refit_preprocessor.transform(
                deep[inner_development],
                scalar[inner_development],
            )
            x_inner_holdout = refit_preprocessor.transform(
                deep[inner_holdout],
                scalar[inner_holdout],
            )
            cached_transform = {
                "early_train": early_train,
                "early_valid": early_valid,
                "x_early_train_full": x_early_train,
                "x_early_valid_full": x_early_valid,
                "x_inner_development_full": x_inner_development,
                "x_inner_holdout_full": x_inner_holdout,
                "fitted_deep_dim": int(refit_preprocessor.deep_pca.n_components_),
                "preprocessor_audit": refit_preprocessor.audit(),
            }
            if transform_cache is not None:
                transform_cache[cache_key] = cached_transform
        else:
            early_train = cached_transform["early_train"]
            early_valid = cached_transform["early_valid"]

        fitted_deep_dim = int(cached_transform["fitted_deep_dim"])
        x_early_train = slice_fusion_components(
            cached_transform["x_early_train_full"],
            config.deep_pca_dim,
            fitted_deep_dim,
            use_scalar,
        )
        x_early_valid = slice_fusion_components(
            cached_transform["x_early_valid_full"],
            config.deep_pca_dim,
            fitted_deep_dim,
            use_scalar,
        )
        x_inner_development = slice_fusion_components(
            cached_transform["x_inner_development_full"],
            config.deep_pca_dim,
            fitted_deep_dim,
            use_scalar,
        )
        x_inner_holdout = slice_fusion_components(
            cached_transform["x_inner_holdout_full"],
            config.deep_pca_dim,
            fitted_deep_dim,
            use_scalar,
        )
        best_epoch, stop_ap, stop_auc, history = fit_mlp_early_stop(
            x_early_train,
            y[early_train],
            x_early_valid,
            y[early_valid],
            config,
            device,
            seed + inner_fold * 10000,
            amp_enabled,
        )
        selected_epochs.append(best_epoch)

        seed_predictions: list[np.ndarray] = []
        for seed_index in range(search_seeds):
            model = fit_mlp_fixed_epochs(
                x_inner_development,
                y[inner_development],
                config,
                best_epoch,
                device,
                seed + inner_fold * 100000 + seed_index,
                amp_enabled,
            )
            seed_predictions.append(
                mlp_predict(model, x_inner_holdout, device)
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        probability = np.mean(np.stack(seed_predictions), axis=0)
        pooled[inner_holdout] = probability
        rows.append(
            {
                "inner_fold": inner_fold,
                "inner_development_rows": int(len(inner_development)),
                "inner_holdout_rows": int(len(inner_holdout)),
                "early_train_rows": int(len(early_train)),
                "early_valid_rows": int(len(early_valid)),
                "selected_epoch": int(best_epoch),
                "early_stop_AUPRC": float(stop_ap),
                "early_stop_AUROC": float(stop_auc),
                "untouched_inner_holdout_AUPRC": safe_ap(
                    y[inner_holdout],
                    probability,
                ),
                "untouched_inner_holdout_AUROC": safe_auc(
                    y[inner_holdout],
                    probability,
                ),
                "search_seeds": int(search_seeds),
                "preprocessor": json.dumps(
                    cached_transform["preprocessor_audit"],
                    sort_keys=True,
                ),
                "early_history_rows": int(len(history)),
            }
        )

    if not np.isfinite(pooled).all():
        raise RuntimeError("Incomplete nested inner MLP predictions")
    return pooled, selected_epochs, rows


def select_mlp_config(
    deep: np.ndarray,
    scalar: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    use_scalar: bool,
    device: torch.device,
    seed: int,
    search_seeds: int,
    amp_enabled: bool,
) -> tuple[MLPConfig, int, list[dict[str, Any]], list[dict[str, Any]]]:
    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    # Reuse grouped splits and expensive PCA/scalar transformations for
    # configurations sharing the same representation dimension.
    shared_splits = grouped_splits(
        y,
        groups,
        requested=3,
        seed=seed,
    )
    transform_cache: dict[Any, Any] = {}
    best_key: tuple[float, float, int, int] | None = None
    best_config = MLP_CONFIGS[0]
    best_epoch = 1

    for config_index, config in enumerate(MLP_CONFIGS):
        pooled, epochs, rows = evaluate_mlp_config_inner_cv(
            deep,
            scalar,
            y,
            groups,
            config,
            use_scalar,
            device,
            seed,
            search_seeds,
            amp_enabled,
            transform_cache,
            shared_splits,
        )
        for row in rows:
            fold_rows.append(
                {
                    "config_index": config_index,
                    **asdict(config),
                    "use_scalar": bool(use_scalar),
                    **row,
                }
            )
        pooled_ap = safe_ap(y, pooled)
        pooled_auc = safe_auc(y, pooled)
        median_epoch = max(
            1,
            int(round(float(np.median(epochs)))),
        )
        epoch_iqr = float(
            np.percentile(epochs, 75)
            - np.percentile(epochs, 25)
        )
        summary = {
            "config_index": config_index,
            **asdict(config),
            "use_scalar": bool(use_scalar),
            "inner_pooled_AUPRC": pooled_ap,
            "inner_pooled_AUROC": pooled_auc,
            "selected_epoch_median": median_epoch,
            "selected_epoch_values": json.dumps(epochs),
            "selected_epoch_iqr": epoch_iqr,
            "search_seeds": int(search_seeds),
        }
        summary_rows.append(summary)
        key = (
            pooled_ap,
            pooled_auc,
            -config.deep_pca_dim,
            -(config.hidden1 + config.hidden2),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_config = config
            best_epoch = median_epoch

    return best_config, best_epoch, fold_rows, summary_rows


def save_fold_cache(
    directory: Path,
    expected_signature: str,
    holdout_ids: np.ndarray,
    valid_ids: np.ndarray,
    holdout_probability: np.ndarray,
    valid_probability: np.ndarray,
    payload: dict[str, Any],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    atomic_npz(
        directory / "predictions.npz",
        holdout_series_uid=holdout_ids.astype(str),
        valid_series_uid=valid_ids.astype(str),
        holdout_probability=holdout_probability.astype(np.float64),
        valid_probability=valid_probability.astype(np.float64),
    )
    atomic_json(
        {
            **payload,
            "cache_signature": expected_signature,
        },
        directory / ".SUCCESS.json",
    )


def load_fold_cache(
    directory: Path,
    expected_signature: str,
    holdout_ids: np.ndarray,
    valid_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    success_path = directory / ".SUCCESS.json"
    prediction_path = directory / "predictions.npz"
    if not success_path.is_file() or not prediction_path.is_file():
        return None
    try:
        payload = json.loads(success_path.read_text(encoding="utf-8"))
        if payload.get("cache_signature") != expected_signature:
            return None
        with np.load(prediction_path, allow_pickle=False) as raw:
            saved_holdout = raw["holdout_series_uid"].astype(str)
            saved_valid = raw["valid_series_uid"].astype(str)
            holdout_probability = raw[
                "holdout_probability"
            ].astype(np.float64)
            valid_probability = raw[
                "valid_probability"
            ].astype(np.float64)
        checks = (
            np.array_equal(saved_holdout, holdout_ids.astype(str)),
            np.array_equal(saved_valid, valid_ids.astype(str)),
            holdout_probability.shape == (len(holdout_ids),),
            valid_probability.shape == (len(valid_ids),),
            np.isfinite(holdout_probability).all(),
            np.isfinite(valid_probability).all(),
        )
        if not all(checks):
            return None
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
    task_hash: str,
    trainer_hash: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    y = train["target"].astype(np.int64)
    folds = train["fold"].astype(int)
    groups = train["patient_id"].astype(str)
    sample_ids = train["series_uid"].astype(str)
    valid_ids = valid["series_uid"].astype(str)
    oof = np.full(len(y), np.nan, dtype=np.float64)
    valid_folds: list[np.ndarray] = []
    audit_rows: list[dict[str, Any]] = []

    for fold in range(1, 6):
        development = np.flatnonzero(folds != fold)
        holdout = np.flatnonzero(folds == fold)
        fold_dir = output_root / "folds" / model_name / f"fold_{fold}"
        signature = stable_hash(
            {
                "task_hash": task_hash,
                "trainer_hash": trainer_hash,
                "model": model_name,
                "fold": fold,
                "use_scalar": use_scalar,
                "C_GRID": C_GRID,
                "PCA_DIMS": LOGISTIC_DEEP_DIMS,
            }
        )
        cached = (
            None
            if overwrite
            else load_fold_cache(
                fold_dir,
                signature,
                sample_ids[holdout],
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
            holdout_probability = model.predict_proba(
                x_holdout
            )[:, 1]
            valid_probability = model.predict_proba(x_valid)[:, 1]
            fold_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {"preprocessor": preprocessor, "model": model},
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
                signature,
                sample_ids[holdout],
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
                "holdout_AUPRC": safe_ap(
                    y[holdout],
                    holdout_probability,
                ),
                "holdout_AUROC": safe_auc(
                    y[holdout],
                    holdout_probability,
                ),
                "resumed": resumed,
                "selection": json.dumps(payload, sort_keys=True),
            }
        )
        print(
            f"[PASS] {model_name} fold={fold} "
            f"AUPRC={audit_rows[-1]['holdout_AUPRC']:.6f} "
            f"resumed={resumed}",
            flush=True,
        )

    if not np.isfinite(oof).all():
        raise RuntimeError(f"{model_name}: incomplete OOF")
    return (
        oof,
        np.mean(np.stack(valid_folds), axis=0),
        audit_rows,
    )


def train_mlp_outer(
    model_name: str,
    train: dict[str, np.ndarray],
    valid: dict[str, np.ndarray],
    use_scalar: bool,
    device: torch.device,
    output_root: Path,
    overwrite: bool,
    final_seeds: int,
    search_seeds: int,
    task_hash: str,
    trainer_hash: str,
    amp_enabled: bool,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    y = train["target"].astype(np.int64)
    folds = train["fold"].astype(int)
    groups = train["patient_id"].astype(str)
    sample_ids = train["series_uid"].astype(str)
    valid_ids = valid["series_uid"].astype(str)
    oof = np.full(len(y), np.nan, dtype=np.float64)
    valid_folds: list[np.ndarray] = []
    audit_rows: list[dict[str, Any]] = []

    for fold in range(1, 6):
        development = np.flatnonzero(folds != fold)
        holdout = np.flatnonzero(folds == fold)
        fold_dir = output_root / "folds" / model_name / f"fold_{fold}"
        signature = stable_hash(
            {
                "task_hash": task_hash,
                "trainer_hash": trainer_hash,
                "model": model_name,
                "fold": fold,
                "use_scalar": use_scalar,
                "MLP_CONFIGS": [asdict(config) for config in MLP_CONFIGS],
                "final_seeds": final_seeds,
                "search_seeds": search_seeds,
                "inner_folds": 3,
                "early_stop_policy": (
                    "grouped sub-split then full-inner-development refit"
                ),
            }
        )
        cached = (
            None
            if overwrite
            else load_fold_cache(
                fold_dir,
                signature,
                sample_ids[holdout],
                valid_ids,
            )
        )
        if cached is not None:
            holdout_probability, valid_probability, payload = cached
            resumed = True
        else:
            (
                selected_config,
                selected_epoch,
                inner_fold_rows,
                config_summary_rows,
            ) = select_mlp_config(
                train["deep"][development],
                train["scalar"][development],
                y[development],
                groups[development],
                use_scalar,
                device,
                SEED + fold * 100,
                search_seeds,
                amp_enabled,
            )
            preprocessor = FusionPreprocessor(
                deep_pca_dim=selected_config.deep_pca_dim,
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
            joblib.dump(
                preprocessor,
                fold_dir / "preprocessor.joblib",
            )
            for seed_index in range(final_seeds):
                model_seed = SEED + fold * 10000 + seed_index
                model = fit_mlp_fixed_epochs(
                    x_development,
                    y[development],
                    selected_config,
                    selected_epoch,
                    device,
                    model_seed,
                    amp_enabled,
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
                        "config": asdict(selected_config),
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
                pd.DataFrame(inner_fold_rows),
                fold_dir / "inner_fold_search.csv",
            )
            atomic_csv(
                pd.DataFrame(config_summary_rows),
                fold_dir / "inner_config_summary.csv",
            )
            payload = {
                "model": model_name,
                "fold": fold,
                "use_scalar": use_scalar,
                "selected_config": asdict(selected_config),
                "selected_epoch": int(selected_epoch),
                "final_ensemble_seeds": int(final_seeds),
                "inner_search_seeds": int(search_seeds),
                "inner_cv_folds": 3,
                "early_stop_validation_is_separate_from_inner_holdout": True,
                "preprocessor": preprocessor.audit(),
                "development_rows": int(len(development)),
                "holdout_rows": int(len(holdout)),
            }
            save_fold_cache(
                fold_dir,
                signature,
                sample_ids[holdout],
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
                "holdout_AUPRC": safe_ap(
                    y[holdout],
                    holdout_probability,
                ),
                "holdout_AUROC": safe_auc(
                    y[holdout],
                    holdout_probability,
                ),
                "resumed": resumed,
                "selection": json.dumps(payload, sort_keys=True),
            }
        )
        print(
            f"[PASS] {model_name} fold={fold} "
            f"AUPRC={audit_rows[-1]['holdout_AUPRC']:.6f} "
            f"resumed={resumed}",
            flush=True,
        )

    if not np.isfinite(oof).all():
        raise RuntimeError(f"{model_name}: incomplete OOF")
    return (
        oof,
        np.mean(np.stack(valid_folds), axis=0),
        audit_rows,
    )


def probability_logit_features(
    probability_map: dict[str, np.ndarray],
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for model in BASE_MODEL_ORDER:
        probability = np.clip(
            np.asarray(probability_map[model], dtype=np.float64),
            1e-5,
            1.0 - 1e-5,
        )
        columns.append(np.log(probability / (1.0 - probability)))
    result = np.column_stack(columns)
    if not np.isfinite(result).all():
        raise RuntimeError("Nonfinite stack features")
    return result


def stack_pipeline(c_value: float, seed: int):
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=float(c_value),
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=10000,
                    random_state=seed,
                ),
            ),
        ]
    )


def select_stack_c(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> tuple[float, list[dict[str, Any]]]:
    splits = grouped_splits(
        y,
        groups,
        requested=3,
        seed=seed,
    )
    rows: list[dict[str, Any]] = []
    best_key: tuple[float, float, float] | None = None
    best_c = 0.1
    for c_value in C_GRID:
        prediction = np.full(len(y), np.nan, dtype=np.float64)
        for development, holdout in splits:
            model = stack_pipeline(float(c_value), seed)
            model.fit(x[development], y[development])
            prediction[holdout] = model.predict_proba(x[holdout])[:, 1]
        if not np.isfinite(prediction).all():
            raise RuntimeError("Incomplete stack inner OOF")
        ap = safe_ap(y, prediction)
        auc = safe_auc(y, prediction)
        row = {
            "C": float(c_value),
            "inner_pooled_AUPRC": ap,
            "inner_pooled_AUROC": auc,
        }
        rows.append(row)
        key = (ap, auc, -float(c_value))
        if best_key is None or key > best_key:
            best_key = key
            best_c = float(c_value)
    return best_c, rows


def train_stacked_ensemble(
    train: dict[str, np.ndarray],
    valid: dict[str, np.ndarray],
    probability_pairs: dict[str, tuple[np.ndarray, np.ndarray]],
    output_root: Path,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    y = train["target"].astype(np.int64)
    groups = train["patient_id"].astype(str)
    folds = train["fold"].astype(int)
    train_features = probability_logit_features(
        {
            model: probability_pairs[model][0]
            for model in BASE_MODEL_ORDER
        }
    )
    valid_features = probability_logit_features(
        {
            model: probability_pairs[model][1]
            for model in BASE_MODEL_ORDER
        }
    )
    oof = np.full(len(y), np.nan, dtype=np.float64)
    valid_fold_probabilities: list[np.ndarray] = []
    audit_rows: list[dict[str, Any]] = []
    model_dir = output_root / "stacked_ensemble"
    model_dir.mkdir(parents=True, exist_ok=True)

    for fold in range(1, 6):
        development = np.flatnonzero(folds != fold)
        holdout = np.flatnonzero(folds == fold)
        best_c, inner_rows = select_stack_c(
            train_features[development],
            y[development],
            groups[development],
            SEED + 70000 + fold,
        )
        model = stack_pipeline(best_c, SEED + 71000 + fold)
        model.fit(train_features[development], y[development])
        oof[holdout] = model.predict_proba(
            train_features[holdout]
        )[:, 1]
        valid_fold_probabilities.append(
            model.predict_proba(valid_features)[:, 1]
        )
        joblib.dump(model, model_dir / f"fold_{fold}.joblib")
        atomic_csv(
            pd.DataFrame(inner_rows),
            model_dir / f"fold_{fold}_inner_search.csv",
        )
        audit_rows.append(
            {
                "model": "Stacked_Ensemble",
                "fold": fold,
                "development_rows": int(len(development)),
                "holdout_rows": int(len(holdout)),
                "holdout_patients": int(
                    len(np.unique(groups[holdout]))
                ),
                "holdout_AUPRC": safe_ap(y[holdout], oof[holdout]),
                "holdout_AUROC": safe_auc(y[holdout], oof[holdout]),
                "resumed": False,
                "selection": json.dumps(
                    {
                        "best_c": best_c,
                        "base_models": list(BASE_MODEL_ORDER),
                        "meta_features": "clipped probability logits",
                    },
                    sort_keys=True,
                ),
            }
        )

    if not np.isfinite(oof).all():
        raise RuntimeError("Stacked_Ensemble: incomplete OOF")
    valid_probability = np.mean(
        np.stack(valid_fold_probabilities),
        axis=0,
    )
    atomic_json(
        {
            "base_models": list(BASE_MODEL_ORDER),
            "meta_model": "fold-ensemble balanced LogisticRegression",
            "meta_features": "clipped probability logits",
            "selection_source": "Train OOF only",
            "valid_used_for_selection": False,
        },
        model_dir / "stack_summary.json",
    )
    print(
        f"[PASS] Stacked_Ensemble pooled OOF "
        f"AUPRC={safe_ap(y, oof):.6f}",
        flush=True,
    )
    return oof, valid_probability, audit_rows


def patient_cluster_bootstrap(
    y: np.ndarray,
    groups: np.ndarray,
    probabilities: dict[str, np.ndarray],
    thresholds: dict[str, float],
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = groups.astype(str)
    unique_groups = np.unique(groups)
    group_indices = {
        group: np.flatnonzero(groups == group)
        for group in unique_groups
    }
    rng = np.random.default_rng(seed)
    metric_names = (
        "AUROC",
        "AUPRC",
        "Balanced Accuracy",
        "Sensitivity",
        "Specificity",
        "Brier",
    )
    distributions = {
        model: {metric: [] for metric in metric_names}
        for model in MODEL_ORDER
    }
    comparisons: dict[tuple[str, str], dict[str, list[float]]] = {}
    for reference in ("Dummy", "Logistic_Deep", "Logistic_Fusion"):
        for comparison in ("MLP_Deep", "MLP_Fusion", "Stacked_Ensemble"):
            comparisons[(reference, comparison)] = {
                "AUROC": [],
                "AUPRC": [],
                "Brier improvement": [],
            }

    effective = 0
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
        effective += 1
        current: dict[str, dict[str, Any]] = {}
        for model in MODEL_ORDER:
            current[model] = metric_row(
                model,
                "bootstrap",
                y_sample,
                probabilities[model][indices],
                thresholds[model],
            )
            for metric in metric_names:
                distributions[model][metric].append(
                    float(current[model][metric])
                )
        for (reference, comparison), values in comparisons.items():
            values["AUROC"].append(
                current[comparison]["AUROC"]
                - current[reference]["AUROC"]
            )
            values["AUPRC"].append(
                current[comparison]["AUPRC"]
                - current[reference]["AUPRC"]
            )
            values["Brier improvement"].append(
                current[reference]["Brier"]
                - current[comparison]["Brier"]
            )

    if effective == 0:
        raise RuntimeError("No valid bootstrap repeat")

    ci_rows: list[dict[str, Any]] = []
    for model, metrics in distributions.items():
        for metric, values_list in metrics.items():
            values = np.asarray(values_list, dtype=np.float64)
            ci_rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "bootstrap_mean": float(np.mean(values)),
                    "ci_lower": float(np.percentile(values, 2.5)),
                    "ci_upper": float(np.percentile(values, 97.5)),
                    "effective_repeats": int(effective),
                    "bootstrap_unit": "patient_id",
                }
            )

    comparison_rows: list[dict[str, Any]] = []
    for (reference, comparison), metrics in comparisons.items():
        for metric, values_list in metrics.items():
            values = np.asarray(values_list, dtype=np.float64)
            lower = float(np.percentile(values, 2.5))
            upper = float(np.percentile(values, 97.5))
            comparison_rows.append(
                {
                    "reference": reference,
                    "comparison": comparison,
                    "metric": metric,
                    "difference_mean": float(np.mean(values)),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "crosses_zero": bool(lower <= 0 <= upper),
                    "effective_repeats": int(effective),
                    "bootstrap_unit": "patient_id",
                }
            )
    return pd.DataFrame(ci_rows), pd.DataFrame(comparison_rows)


def write_report(
    path: Path,
    metrics: pd.DataFrame,
    fold_audit: pd.DataFrame,
    task_summary: dict[str, Any],
) -> None:
    train_metrics = metrics[metrics["split"] == "Train_OOF"]
    valid_metrics = metrics[metrics["split"] == "Valid"]
    lines = [
        "# Formal strict Pre+Post series-level Local-CAVE adverse-outcome V3",
        "",
        "## Cohort",
        "",
        f"- Train series: {task_summary['train']['included_series']}",
        f"- Train patients: {task_summary['train']['included_patients']}",
        f"- Valid series: {task_summary['valid']['included_series']}",
        f"- Valid patients: {task_summary['valid']['included_patients']}",
        "- Pre-only, Post-only and no-phase series are excluded.",
        "- Different series from one patient may retain different adverse labels.",
        "- Scalar feature names are selected from Train only.",
        "- Official Valid is never used for fitting, preprocessing, model "
        "selection, early stopping or threshold selection.",
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
    parser.add_argument("--cpu-threads", type=int, default=DEFAULT_CPU_THREADS)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument(
        "--mlp-seeds",
        type=int,
        default=DEFAULT_MLP_SEEDS,
    )
    parser.add_argument(
        "--mlp-search-seeds",
        type=int,
        default=DEFAULT_MLP_SEARCH_SEEDS,
    )
    parser.add_argument(
        "--bootstrap-repeats",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPEATS,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.mlp_seeds < 1 or args.mlp_search_seeds < 1:
        raise ValueError("MLP seed counts must be >= 1")
    task_root = args.task_root.resolve()
    output_dir = args.output_dir.resolve()
    prepare_output(output_dir, args.overwrite)
    set_seed(SEED)
    start = time.time()

    task_success_path = task_root / ".TASK_SUCCESS.json"
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
    runtime = configure_runtime(args.cpu_threads, device)
    amp_enabled = bool(device.type == "cuda" and not args.disable_amp)
    runtime["amp"] = amp_enabled

    task_hash = sha256_file(task_success_path)
    trainer_hash = sha256_file(Path(__file__).resolve())
    run_manifest = {
        "version": "formal_adverse_prepost_local_cave_models_v3",
        "task_hash": task_hash,
        "trainer_hash": trainer_hash,
        "device": str(device),
        "runtime": runtime,
        "mlp_final_seeds": int(args.mlp_seeds),
        "mlp_search_seeds": int(args.mlp_search_seeds),
        "bootstrap_repeats": int(args.bootstrap_repeats),
        "models": list(MODEL_ORDER),
        "runtime": runtime,
    }
    atomic_json(run_manifest, output_dir / "run_manifest.json")

    probability_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fold_rows: list[dict[str, Any]] = []
    dummy_oof = np.full(len(y_train), np.nan, dtype=np.float64)
    dummy_valid_folds: list[np.ndarray] = []
    for fold in range(1, 6):
        development = np.flatnonzero(train["fold"].astype(int) != fold)
        holdout = np.flatnonzero(train["fold"].astype(int) == fold)
        development_prior = float(np.mean(y_train[development]))
        dummy_oof[holdout] = development_prior
        dummy_valid_folds.append(
            np.full(len(y_valid), development_prior, dtype=np.float64)
        )
    if not np.isfinite(dummy_oof).all():
        raise AssertionError("Dummy OOF is incomplete")
    probability_pairs["Dummy"] = (
        dummy_oof,
        np.mean(np.stack(dummy_valid_folds), axis=0),
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
            task_hash,
            trainer_hash,
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
            args.mlp_search_seeds,
            task_hash,
            trainer_hash,
            amp_enabled,
        )
        probability_pairs[model_name] = (oof, valid_probability)
        fold_rows.extend(rows)

    stack_oof, stack_valid, stack_rows = train_stacked_ensemble(
        train,
        valid,
        probability_pairs,
        output_dir,
    )
    probability_pairs["Stacked_Ensemble"] = (stack_oof, stack_valid)
    fold_rows.extend(stack_rows)

    thresholds: dict[str, float] = {}
    metric_rows: list[dict[str, Any]] = []
    train_predictions = pd.DataFrame(
        {
            "series_uid": train["series_uid"].astype(str),
            "patient_id": train["patient_id"].astype(str),
            "target": y_train,
            "fold": train["fold"].astype(int),
        }
    )
    valid_predictions = pd.DataFrame(
        {
            "series_uid": valid["series_uid"].astype(str),
            "patient_id": valid["patient_id"].astype(str),
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

    oof_candidates = metrics[
        (metrics["split"] == "Train_OOF")
        & (metrics["model"] != "Dummy")
    ].copy()
    oof_candidates = oof_candidates.sort_values(
        ["AUPRC", "AUROC", "Brier", "model"],
        ascending=[False, False, True, True],
    )
    selected_model = str(oof_candidates.iloc[0]["model"])
    selected_valid_row = metrics[
        (metrics["split"] == "Valid")
        & (metrics["model"] == selected_model)
    ].iloc[0]
    selected_payload = {
        "selected_model": selected_model,
        "selection_source": "pooled Train OOF only",
        "selection_primary_metric": "AUPRC",
        "selection_tiebreakers": ["AUROC descending", "Brier ascending"],
        "valid_used_for_selection": False,
        "selected_valid_metrics": {
            key: (
                value.item()
                if hasattr(value, "item")
                else value
            )
            for key, value in selected_valid_row.to_dict().items()
        },
    }
    atomic_json(
        selected_payload,
        output_dir / "selected_model_by_train_oof.json",
    )
    atomic_csv(
        train_predictions,
        output_dir / "train_oof_predictions.csv",
    )
    atomic_csv(
        valid_predictions,
        output_dir / "valid_predictions.csv",
    )

    valid_probability_map = {
        model: probability_pairs[model][1] for model in MODEL_ORDER
    }
    bootstrap_ci, paired = patient_cluster_bootstrap(
        y_valid,
        valid["patient_id"].astype(str),
        valid_probability_map,
        thresholds,
        args.bootstrap_repeats,
        SEED + 50000,
    )
    atomic_csv(
        bootstrap_ci,
        output_dir / "valid_patient_cluster_bootstrap_ci.csv",
    )
    atomic_csv(
        paired,
        output_dir / "valid_paired_model_differences.csv",
    )

    summary = {
        "status": "success",
        "version": "formal_adverse_prepost_local_cave_models_v3",
        "task": "adverse_outcome_series_strict_prepost",
        "models": list(MODEL_ORDER),
        "selected_model_by_train_oof": selected_model,
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
        "logistic_inner_folds": 3,
        "mlp_inner_folds": 3,
        "mlp_early_stop_split_separate_from_inner_holdout": True,
        "grouping_unit": "patient_id",
        "threshold_source": "pooled Train OOF only",
        "valid_used_for_fitting_or_selection": False,
        "strict_phase_policy": "both Pre and Post required",
        "same_series_label_conflicts_excluded_in_task_builder": True,
        "mixed_labels_across_distinct_series_allowed": True,
        "scalar_schema_train_only": True,
        "thresholds": thresholds,
        "mlp_final_seeds_per_outer_fold": int(args.mlp_seeds),
        "mlp_search_seeds_per_inner_fold": int(
            args.mlp_search_seeds
        ),
        "bootstrap_repeats_requested": int(
            args.bootstrap_repeats
        ),
        "bootstrap_unit": "patient_id",
        "device": str(device),
        "seed": SEED,
        "task_success_lock_sha256": task_hash,
        "trainer_sha256": trainer_hash,
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

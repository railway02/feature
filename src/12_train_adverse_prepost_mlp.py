#!/root/autodl-tmp/envs/aneurysm-ml/bin/python
"""Train the frozen adverse Pre+Post MLP experiment.

The independent Valid set is never used for early stopping, threshold
selection, model selection, or tuning. It only receives predictions from the
five Train-derived outer-fold models and is evaluated after the protocol is
durably frozen.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path("/root/autodl-tmp/aneurysm")
EXPECTED_PYTHON = Path("/root/autodl-tmp/envs/aneurysm-ml/bin/python")
TRAIN_PATH = PROJECT_ROOT / "outputs/task_datasets/adverse_prepost_train.csv"
VALID_PATH = PROJECT_ROOT / "outputs/task_datasets/adverse_prepost_valid.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs/baselines/adverse_prepost_mlp_v1"
REPORT_DIR = PROJECT_ROOT / "reports/adverse_prepost_mlp_v1"
BUILD_REPORT_PATH = REPORT_DIR / "data_build_report.md"
MODEL_DIR = OUTPUT_DIR / "models"
SCALER_DIR = OUTPUT_DIR / "scalers"
PLOT_DIR = OUTPUT_DIR / "plots"
RUNNING_PATH = OUTPUT_DIR / ".RUNNING"
SUCCESS_PATH = OUTPUT_DIR / ".SUCCESS"

if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
    raise RuntimeError(
        f"Wrong Python interpreter: {sys.executable}; required: {EXPECTED_PYTHON}"
    )

try:
    import joblib
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import sklearn
    import torch
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
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    print(f"Dependency import failed under {sys.executable}: {exc}", file=sys.stderr)
    raise


EXPERIMENT = "adverse_prepost_mlp_v1"
SEED = 42
N_SPLITS = 5
INNER_VALID_FRACTION = 0.15
BATCH_SIZE = 32
MAX_EPOCHS = 200
PATIENCE = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BOOTSTRAP_REPEATS = 2000
EXPECTED_ROWS = {"train": 794, "valid": 209}
EXPECTED_POSITIVES = {"train": 132, "valid": 36}
EXPECTED_PRE = 48
EXPECTED_POST = 48
EXPECTED_FEATURES = 96
EXPECTED_COLUMNS = 99
MODEL_ORDER = ["Dummy", "Logistic", "Pre+Post MLP"]
MODEL_SLUG = {
    "Dummy": "dummy",
    "Logistic": "logistic",
    "Pre+Post MLP": "mlp",
}
METRIC_NAMES = [
    "auroc",
    "auprc",
    "balanced_accuracy",
    "f1",
    "precision",
    "sensitivity",
    "specificity",
    "brier_score",
    "tp",
    "tn",
    "fp",
    "fn",
]
NETWORK_DESCRIPTION = [
    "Linear(96, 64)",
    "ReLU",
    "Dropout(0.30)",
    "Linear(64, 16)",
    "ReLU",
    "Dropout(0.20)",
    "Linear(16, 1)",
]

REQUIRED_OUTPUT_FILES = [
    "train_oof_predictions.csv",
    "valid_predictions.csv",
    "fold_metrics.csv",
    "train_oof_metrics.csv",
    "valid_metrics.csv",
    "bootstrap_confidence_intervals.csv",
    "paired_bootstrap_comparisons.csv",
    "training_history.csv",
    "thresholds.json",
    "feature_names.json",
    "configuration.json",
    "frozen_protocol.json",
    "environment.txt",
    "gpu_verification.txt",
    "run.log",
    "exit_status.txt",
    "models/mlp_fold_1.pt",
    "models/mlp_fold_2.pt",
    "models/mlp_fold_3.pt",
    "models/mlp_fold_4.pt",
    "models/mlp_fold_5.pt",
    "models/dummy.joblib",
    "models/logistic.joblib",
    "scalers/mlp_fold_1_scaler.joblib",
    "scalers/mlp_fold_2_scaler.joblib",
    "scalers/mlp_fold_3_scaler.joblib",
    "scalers/mlp_fold_4_scaler.joblib",
    "scalers/mlp_fold_5_scaler.joblib",
    "plots/train_oof_roc.png",
    "plots/train_oof_pr.png",
    "plots/valid_roc.png",
    "plots/valid_pr.png",
    "plots/valid_calibration.png",
    "plots/valid_confusion_matrices.png",
    "plots/mlp_training_curves.png",
]


class ExperimentError(RuntimeError):
    """Raised when a locked protocol assertion fails."""


class MLP(nn.Module):
    def __init__(self, input_dim: int = EXPECTED_FEATURES) -> None:
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


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if pd.isna(value):
        return None
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_text_sync(path: Path, text: str, mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def write_json_sync(path: Path, payload: Any) -> None:
    write_text_sync(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
            default=json_default,
        )
        + "\n",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preflight_output_directories() -> str:
    if OUTPUT_DIR.exists():
        existing_outputs = sorted(str(path) for path in OUTPUT_DIR.rglob("*") if path.exists())
        if existing_outputs:
            raise ExperimentError(
                "Model output directory is not empty; refusing to overwrite: "
                + json.dumps(existing_outputs, ensure_ascii=False)
            )

    if not REPORT_DIR.is_dir():
        raise ExperimentError(f"Required report directory is absent: {REPORT_DIR}")
    report_entries = sorted(path.name for path in REPORT_DIR.iterdir())
    if report_entries != [BUILD_REPORT_PATH.name]:
        raise ExperimentError(
            "Report directory may contain only data_build_report.md before training; "
            f"found={report_entries}"
        )
    if not BUILD_REPORT_PATH.is_file() or BUILD_REPORT_PATH.stat().st_size <= 0:
        raise ExperimentError(f"Read-only data-build report is missing or empty: {BUILD_REPORT_PATH}")
    return sha256_file(BUILD_REPORT_PATH)


def initialize_run_directories(started_at: datetime) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=False)
    SCALER_DIR.mkdir(parents=True, exist_ok=False)
    PLOT_DIR.mkdir(parents=True, exist_ok=False)
    write_text_sync(
        RUNNING_PATH,
        f"status=RUNNING\nstarted_at_utc={started_at.isoformat()}\npid={os.getpid()}\n",
        mode="x",
    )


def setup_logging(path: Path) -> logging.Logger:
    logger = logging.getLogger(EXPERIMENT)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)sZ | %(levelname)s | %(message)s", "%Y-%m-%dT%H:%M:%S"
    )
    formatter.converter = time.gmtime
    file_handler = logging.FileHandler(path, mode="x", encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def flush_logger(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        handler.flush()


def close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalized_patient_ids(series: pd.Series, context: str) -> pd.Series:
    if series.isna().any():
        raise ExperimentError(f"{context}: patient_id contains missing values")

    def normalize(value: object) -> str:
        text = str(value).strip()
        if not text:
            raise ExperimentError(f"{context}: patient_id is empty after trimming")
        if text.endswith(".0") and text[:-2].lstrip("+-").isdigit():
            text = text[:-2]
        return text[1:] if text.startswith("+") else text

    result = series.map(normalize)
    duplicates = result[result.duplicated(keep=False)]
    if not duplicates.empty:
        raise ExperimentError(
            f"{context}: duplicate normalized patient_id values: "
            f"{sorted(duplicates.unique().tolist())[:10]}"
        )
    return result


def core_pre_columns(columns: Iterable[str]) -> list[str]:
    return [
        column
        for column in columns
        if column.startswith("pre_")
        and column != "pre_n_pairs"
        and "runtime_s" not in column
    ]


def core_post_columns(columns: Iterable[str]) -> list[str]:
    return [
        column
        for column in columns
        if column.startswith("post_")
        and column != "post_n_pairs"
        and "runtime_s" not in column
    ]


def validate_inputs(
    train_df: pd.DataFrame, valid_df: pd.DataFrame
) -> tuple[list[str], pd.Series, pd.Series]:
    if train_df.columns.tolist() != valid_df.columns.tolist():
        raise ExperimentError("Train and Valid column names/order are not identical")
    if train_df.columns.duplicated().any() or valid_df.columns.duplicated().any():
        raise ExperimentError("Input CSV contains duplicate column names")
    if train_df.columns[:3].tolist() != ["patient_id", "split", "adverse"]:
        raise ExperimentError("First three columns must be patient_id, split, adverse")

    pre_columns = core_pre_columns(train_df.columns)
    post_columns = core_post_columns(train_df.columns)
    feature_names = [*pre_columns, *post_columns]
    if len(pre_columns) != EXPECTED_PRE or len(post_columns) != EXPECTED_POST:
        raise ExperimentError(
            f"Expected 48 Pre and 48 Post features; found {len(pre_columns)} and "
            f"{len(post_columns)}"
        )
    if len(feature_names) != EXPECTED_FEATURES:
        raise ExperimentError(f"Expected 96 model features; found {len(feature_names)}")
    expected_order = ["patient_id", "split", "adverse", *feature_names]
    if train_df.columns.tolist() != expected_order:
        raise ExperimentError(
            "Input columns are not exactly metadata + 48 core Pre + 48 core Post"
        )

    forbidden = [
        column
        for column in train_df.columns
        if column.startswith("delta_")
        or "runtime_s" in column
        or "n_pairs" in column
        or "missing" in column.lower()
        or column.startswith("Unnamed:")
    ]
    if forbidden:
        raise ExperimentError(f"Forbidden input/model columns found: {forbidden}")
    if set(feature_names) & {"patient_id", "split", "adverse"}:
        raise ExperimentError("Metadata or label columns entered the model feature list")

    train_ids = normalized_patient_ids(train_df["patient_id"], "Train")
    valid_ids = normalized_patient_ids(valid_df["patient_id"], "Valid")
    overlap = sorted(set(train_ids) & set(valid_ids))
    if overlap:
        raise ExperimentError(
            f"Train/Valid patient intersection is {len(overlap)}; examples={overlap[:10]}"
        )

    for split, frame, expected_ids in (
        ("train", train_df, train_ids),
        ("valid", valid_df, valid_ids),
    ):
        if frame.shape != (EXPECTED_ROWS[split], EXPECTED_COLUMNS):
            raise ExperimentError(
                f"{split}: expected shape {EXPECTED_ROWS[split]}x{EXPECTED_COLUMNS}, "
                f"found {frame.shape[0]}x{frame.shape[1]}"
            )
        if expected_ids.nunique(dropna=False) != EXPECTED_ROWS[split]:
            raise ExperimentError(f"{split}: patient_id is not unique")
        if set(frame["split"].astype(str)) != {split}:
            raise ExperimentError(f"{split}: split column is not uniformly {split}")
        try:
            labels = pd.to_numeric(frame["adverse"], errors="raise")
        except Exception as exc:
            raise ExperimentError(f"{split}: adverse labels are not numeric: {exc}") from exc
        if set(labels.unique().tolist()) != {0, 1}:
            raise ExperimentError(f"{split}: adverse must contain only 0 and 1")
        positives = int((labels == 1).sum())
        if positives != EXPECTED_POSITIVES[split]:
            raise ExperimentError(
                f"{split}: expected {EXPECTED_POSITIVES[split]} positives, found {positives}"
            )
        try:
            values = frame.loc[:, feature_names].apply(pd.to_numeric, errors="raise").to_numpy(float)
        except Exception as exc:
            raise ExperimentError(f"{split}: feature conversion failed: {exc}") from exc
        if not np.isfinite(values).all():
            raise ExperimentError(f"{split}: model features contain NaN or infinity")
    return feature_names, train_ids, valid_ids


def configuration_payload() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "python": str(EXPECTED_PYTHON),
        "inputs": {"train": str(TRAIN_PATH), "valid": str(VALID_PATH)},
        "model_features": {
            "pre": EXPECTED_PRE,
            "post": EXPECTED_POST,
            "total": EXPECTED_FEATURES,
            "excluded": ["patient_id", "split", "adverse"],
            "forbidden_tokens": ["delta_", "runtime_s", "n_pairs", "missing"],
        },
        "network": NETWORK_DESCRIPTION,
        "loss": "BCEWithLogitsLoss",
        "pos_weight": "n_negative / n_positive, recalculated for each MLP training subset",
        "optimizer": {
            "name": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
        },
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "inner_validation_fraction": INNER_VALID_FRACTION,
        "outer_cv": {
            "name": "StratifiedKFold",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": SEED,
        },
        "baselines": {
            "Dummy": {"strategy": "prior"},
            "Logistic": {
                "pipeline": ["SimpleImputer(median)", "StandardScaler", "LogisticRegression"],
                "penalty": "l2",
                "solver": "liblinear",
                "C": 1.0,
                "class_weight": "balanced",
                "max_iter": 5000,
                "random_state": SEED,
            },
        },
        "threshold_rule": [
            "maximize Train pooled OOF balanced accuracy",
            "tie: closest to 0.5",
            "tie: smaller threshold",
        ],
        "bootstrap": {
            "dataset": "independent Valid",
            "patient_level_repeats": BOOTSTRAP_REPEATS,
            "random_seed": SEED,
            "shared_indices_across_models": True,
            "single_class_samples": "skip",
            "retrain": False,
            "reselect_threshold": False,
        },
        "valid_role": "five-fold probability averaging and frozen-protocol evaluation only",
        "random_seed": SEED,
        "gpu_required": True,
    }


def save_environment(path: Path, started_at: datetime, input_hashes: dict[str, str]) -> None:
    lines = [
        f"experiment={EXPERIMENT}",
        f"started_at_utc={started_at.isoformat()}",
        f"pid={os.getpid()}",
        f"python_executable={sys.executable}",
        f"python_version={sys.version.replace(os.linesep, ' ')}",
        f"platform={platform.platform()}",
        f"torch={torch.__version__}",
        f"torch_cuda_version={torch.version.cuda}",
        f"sklearn={sklearn.__version__}",
        f"pandas={pd.__version__}",
        f"numpy={np.__version__}",
        f"matplotlib={matplotlib.__version__}",
        f"joblib={joblib.__version__}",
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}",
        f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')}",
        f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}",
        f"NUMEXPR_NUM_THREADS={os.environ.get('NUMEXPR_NUM_THREADS')}",
        f"PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')}",
        f"train_sha256={input_hashes['train']}",
        f"valid_sha256={input_hashes['valid']}",
    ]
    write_text_sync(path, "\n".join(lines) + "\n")


def verify_gpu(path: Path, logger: logging.Logger) -> torch.device:
    if not torch.cuda.is_available():
        raise ExperimentError("CUDA is required but torch.cuda.is_available() is False")
    if torch.cuda.device_count() < 1:
        raise ExperimentError("CUDA is required but no CUDA device is visible")
    device = torch.device("cuda:0")
    try:
        nvidia_smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception as exc:
        raise ExperimentError(f"nvidia-smi verification failed: {exc}") from exc
    seed_everything(SEED)
    left = torch.randn((128, 96), device=device)
    right = torch.randn((96, 32), device=device)
    smoke = (left @ right).mean()
    torch.cuda.synchronize(device)
    if not bool(torch.isfinite(smoke).item()):
        raise ExperimentError("CUDA tensor smoke test produced a non-finite value")
    properties = torch.cuda.get_device_properties(device)
    evidence = [
        f"verified_at_utc={utc_now().isoformat()}",
        f"torch_cuda_available={torch.cuda.is_available()}",
        f"torch_cuda_version={torch.version.cuda}",
        f"cuda_device_count={torch.cuda.device_count()}",
        f"selected_device={device}",
        f"device_name={torch.cuda.get_device_name(device)}",
        f"device_capability={torch.cuda.get_device_capability(device)}",
        f"device_total_memory_bytes={properties.total_memory}",
        f"cuda_smoke_test_mean={float(smoke.item())}",
        "nvidia_smi:",
        nvidia_smi,
    ]
    write_text_sync(path, "\n".join(evidence) + "\n")
    logger.info(
        "CUDA verified | device=%s | name=%s | torch_cuda=%s",
        device,
        torch.cuda.get_device_name(device),
        torch.version.cuda,
    )
    del left, right, smoke
    torch.cuda.empty_cache()
    return device


def pos_weight_value(labels: np.ndarray) -> float:
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives <= 0 or negatives <= 0:
        raise ExperimentError("MLP training subset must contain both classes")
    return float(negatives / positives)


def train_epoch(
    model: MLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch_features, batch_labels in loader:
        batch_features = batch_features.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_features)
        loss = loss_function(logits, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().item()) * len(batch_labels)
        total_examples += len(batch_labels)
    return total_loss / total_examples


def predict_mlp(
    model: MLP, features: np.ndarray, device: torch.device, batch_size: int = 256
) -> np.ndarray:
    model.eval()
    dataset = TensorDataset(torch.from_numpy(features.astype(np.float32, copy=False)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for (batch_features,) in loader:
            logits = model(batch_features.to(device, non_blocking=True))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    result = np.concatenate(probabilities).astype(float)
    if not np.isfinite(result).all():
        raise ExperimentError("MLP produced non-finite probabilities")
    return result


def make_training_loader(
    features: np.ndarray, labels: np.ndarray, seed: int
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(features.astype(np.float32, copy=False)),
        torch.from_numpy(labels.astype(np.float32, copy=False)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        generator=generator,
    )


def train_with_early_stopping(
    fold: int,
    x_inner_train: np.ndarray,
    y_inner_train: np.ndarray,
    x_inner_valid: np.ndarray,
    y_inner_valid: np.ndarray,
    device: torch.device,
    history_rows: list[dict[str, Any]],
    logger: logging.Logger,
) -> tuple[int, float, float]:
    stage_seed = SEED + fold
    seed_everything(stage_seed)
    model = MLP().to(device)
    weight = pos_weight_value(y_inner_train)
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(weight, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loader = make_training_loader(x_inner_train, y_inner_train, stage_seed)
    best_epoch = 0
    best_auprc = -math.inf
    epochs_without_improvement = 0

    logger.info(
        "MLP fold %d/%d early-stopping stage started | inner_train=%d positive=%d | "
        "inner_valid=%d positive=%d | pos_weight=%.6f",
        fold,
        N_SPLITS,
        len(y_inner_train),
        int(y_inner_train.sum()),
        len(y_inner_valid),
        int(y_inner_valid.sum()),
        weight,
    )
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_epoch(model, loader, optimizer, loss_function, device)
        inner_probabilities = predict_mlp(model, x_inner_valid, device)
        inner_auprc = float(average_precision_score(y_inner_valid, inner_probabilities))
        improved = inner_auprc > best_auprc + 1e-12
        if improved:
            best_auprc = inner_auprc
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history_rows.append(
            {
                "fold": fold,
                "stage": "inner_early_stopping",
                "epoch": epoch,
                "train_loss": train_loss,
                "inner_validation_auprc": inner_auprc,
                "best_so_far": improved,
                "best_epoch_so_far": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "training_subset_size": len(y_inner_train),
                "training_positive": int(y_inner_train.sum()),
                "pos_weight": weight,
                "seed": stage_seed,
            }
        )
        if improved or epoch == 1 or epoch % 10 == 0:
            logger.info(
                "MLP fold %d early epoch=%d | loss=%.6f | inner_AUPRC=%.6f | "
                "best_epoch=%d best_AUPRC=%.6f | patience=%d/%d",
                fold,
                epoch,
                train_loss,
                inner_auprc,
                best_epoch,
                best_auprc,
                epochs_without_improvement,
                PATIENCE,
            )
        if epochs_without_improvement >= PATIENCE:
            logger.info(
                "MLP fold %d early stopping triggered at epoch %d | best_epoch=%d | "
                "best_inner_AUPRC=%.6f",
                fold,
                epoch,
                best_epoch,
                best_auprc,
            )
            break
    if best_epoch < 1 or best_epoch > MAX_EPOCHS:
        raise ExperimentError(f"Fold {fold}: invalid best_epoch={best_epoch}")
    del model, optimizer, loss_function, loader
    torch.cuda.empty_cache()
    gc.collect()
    return best_epoch, best_auprc, weight


def train_fixed_epochs(
    fold: int,
    scaled_features: np.ndarray,
    labels: np.ndarray,
    best_epoch: int,
    device: torch.device,
    history_rows: list[dict[str, Any]],
    logger: logging.Logger,
) -> tuple[MLP, float]:
    stage_seed = SEED + 1000 + fold
    seed_everything(stage_seed)
    model = MLP().to(device)
    weight = pos_weight_value(labels)
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(weight, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loader = make_training_loader(scaled_features, labels, stage_seed)
    logger.info(
        "MLP fold %d reinitialized; full outer-development retrain for exactly %d epochs | "
        "n=%d positive=%d | pos_weight=%.6f",
        fold,
        best_epoch,
        len(labels),
        int(labels.sum()),
        weight,
    )
    for epoch in range(1, best_epoch + 1):
        train_loss = train_epoch(model, loader, optimizer, loss_function, device)
        history_rows.append(
            {
                "fold": fold,
                "stage": "full_outer_retrain",
                "epoch": epoch,
                "train_loss": train_loss,
                "inner_validation_auprc": math.nan,
                "best_so_far": False,
                "best_epoch_so_far": best_epoch,
                "epochs_without_improvement": math.nan,
                "training_subset_size": len(labels),
                "training_positive": int(labels.sum()),
                "pos_weight": weight,
                "seed": stage_seed,
            }
        )
        if epoch == 1 or epoch == best_epoch or epoch % 20 == 0:
            logger.info(
                "MLP fold %d fixed retrain epoch=%d/%d | loss=%.6f",
                fold,
                epoch,
                best_epoch,
                train_loss,
            )
    return model, weight


def logistic_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    solver="liblinear",
                    C=1.0,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def safe_auroc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return math.nan
    return float(roc_auc_score(labels, probabilities))


def safe_auprc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return math.nan
    return float(average_precision_score(labels, probabilities))


def calculate_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int | str]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if tp + fn else math.nan
    specificity = float(tn / (tn + fp)) if tn + fp else math.nan
    return {
        "auroc": safe_auroc(labels, probabilities),
        "auprc": safe_auprc(labels, probabilities),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "confusion_matrix": json.dumps([[int(tn), int(fp)], [int(fn), int(tp)]]),
    }


def select_threshold(
    model_name: str, labels: np.ndarray, probabilities: np.ndarray
) -> tuple[float, pd.DataFrame]:
    unique = np.unique(np.asarray(probabilities, dtype=float))
    just_above = np.nextafter(unique, np.inf)
    just_above = just_above[(just_above >= 0.0) & (just_above <= 1.0)]
    candidates = np.unique(
        np.concatenate([np.array([0.0, 0.5, 1.0]), unique, just_above])
    )
    rows: list[dict[str, Any]] = []
    for threshold in candidates:
        metrics = calculate_metrics(labels, probabilities, float(threshold))
        rows.append(
            {
                "model": model_name,
                "threshold": float(threshold),
                "balanced_accuracy": metrics["balanced_accuracy"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "tp": metrics["tp"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
            }
        )
    frame = pd.DataFrame(rows)
    best = float(frame["balanced_accuracy"].max())
    tied = frame[
        np.isclose(frame["balanced_accuracy"], best, rtol=0.0, atol=1e-12)
    ].copy()
    tied["distance_to_0_5"] = (tied["threshold"] - 0.5).abs()
    tied = tied.sort_values(["distance_to_0_5", "threshold"], ascending=[True, True])
    selected = float(tied.iloc[0]["threshold"])
    frame["distance_to_0_5"] = (frame["threshold"] - 0.5).abs()
    frame["selected"] = np.isclose(frame["threshold"], selected, rtol=0.0, atol=0.0)
    return selected, frame


def bootstrap_valid(
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
    thresholds: dict[str, float],
    point_metrics: dict[str, dict[str, float | int | str]],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    accepted_indices: list[np.ndarray] = []
    attempts = 0
    while len(accepted_indices) < BOOTSTRAP_REPEATS:
        attempts += 1
        indices = rng.integers(0, len(labels), size=len(labels))
        if np.unique(labels[indices]).size < 2:
            continue
        accepted_indices.append(indices)

    distributions = {
        model: {metric: [] for metric in METRIC_NAMES} for model in MODEL_ORDER
    }
    paired_distributions: dict[tuple[str, str], list[float]] = {}
    for reference in ["Dummy", "Logistic"]:
        for metric in ["auroc_difference", "auprc_difference", "brier_improvement"]:
            paired_distributions[(reference, metric)] = []

    for indices in accepted_indices:
        sampled_labels = labels[indices]
        sampled: dict[str, dict[str, float | int | str]] = {}
        for model in MODEL_ORDER:
            metrics = calculate_metrics(
                sampled_labels, probabilities[model][indices], thresholds[model]
            )
            sampled[model] = metrics
            for metric in METRIC_NAMES:
                distributions[model][metric].append(float(metrics[metric]))
        for reference in ["Dummy", "Logistic"]:
            paired_distributions[(reference, "auroc_difference")].append(
                float(sampled["Pre+Post MLP"]["auroc"])
                - float(sampled[reference]["auroc"])
            )
            paired_distributions[(reference, "auprc_difference")].append(
                float(sampled["Pre+Post MLP"]["auprc"])
                - float(sampled[reference]["auprc"])
            )
            paired_distributions[(reference, "brier_improvement")].append(
                float(sampled[reference]["brier_score"])
                - float(sampled["Pre+Post MLP"]["brier_score"])
            )

    ci_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for metric in METRIC_NAMES:
            values = np.asarray(distributions[model][metric], dtype=float)
            ci_rows.append(
                {
                    "dataset": "valid_patient_bootstrap",
                    "model": model,
                    "metric": metric,
                    "estimate": float(point_metrics[model][metric]),
                    "ci_lower_2_5": float(np.percentile(values, 2.5)),
                    "ci_upper_97_5": float(np.percentile(values, 97.5)),
                    "valid_repeats": BOOTSTRAP_REPEATS,
                    "sampling_attempts": attempts,
                    "skipped_single_class": attempts - BOOTSTRAP_REPEATS,
                    "random_seed": SEED,
                    "shared_indices_across_models": True,
                    "frozen_threshold": thresholds[model],
                }
            )

    paired_rows: list[dict[str, Any]] = []
    for reference in ["Dummy", "Logistic"]:
        point_values = {
            "auroc_difference": float(point_metrics["Pre+Post MLP"]["auroc"])
            - float(point_metrics[reference]["auroc"]),
            "auprc_difference": float(point_metrics["Pre+Post MLP"]["auprc"])
            - float(point_metrics[reference]["auprc"]),
            "brier_improvement": float(point_metrics[reference]["brier_score"])
            - float(point_metrics["Pre+Post MLP"]["brier_score"]),
        }
        for metric, estimate in point_values.items():
            values = np.asarray(paired_distributions[(reference, metric)], dtype=float)
            lower = float(np.percentile(values, 2.5))
            upper = float(np.percentile(values, 97.5))
            crosses_zero = bool(lower <= 0.0 <= upper)
            if lower > 0.0:
                evidence = "95% CI is entirely positive; MLP improvement is supported"
            elif upper < 0.0:
                evidence = "95% CI is entirely negative; reference outperforms MLP"
            else:
                evidence = "95% CI crosses zero; no clear improvement may be claimed"
            paired_rows.append(
                {
                    "model": "Pre+Post MLP",
                    "reference_model": reference,
                    "metric": metric,
                    "positive_means_mlp_better": True,
                    "estimate": estimate,
                    "ci_lower_2_5": lower,
                    "ci_upper_97_5": upper,
                    "ci_crosses_zero": crosses_zero,
                    "evidence_statement": evidence,
                    "valid_repeats": BOOTSTRAP_REPEATS,
                    "sampling_attempts": attempts,
                    "random_seed": SEED,
                    "shared_indices": True,
                    "thresholds_frozen": True,
                    "models_retrained": False,
                }
            )
    logger.info(
        "Valid paired bootstrap complete | accepted=%d attempts=%d skipped_single_class=%d",
        BOOTSTRAP_REPEATS,
        attempts,
        attempts - BOOTSTRAP_REPEATS,
    )
    return pd.DataFrame(ci_rows), pd.DataFrame(paired_rows)


def plot_roc_curves(
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
    title: str,
    path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(7, 6))
    for model in MODEL_ORDER:
        false_positive, true_positive, _ = roc_curve(labels, probabilities[model])
        axis.plot(
            false_positive,
            true_positive,
            linewidth=2,
            label=f"{model} (AUROC={roc_auc_score(labels, probabilities[model]):.3f})",
        )
    axis.plot([0, 1], [0, 1], "k--", linewidth=1)
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title(title)
    axis.legend(loc="lower right")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_pr_curves(
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
    title: str,
    path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(7, 6))
    for model in MODEL_ORDER:
        precision, recall, _ = precision_recall_curve(labels, probabilities[model])
        axis.plot(
            recall,
            precision,
            linewidth=2,
            label=(
                f"{model} (AUPRC="
                f"{average_precision_score(labels, probabilities[model]):.3f})"
            ),
        )
    axis.axhline(float(labels.mean()), color="k", linestyle="--", linewidth=1)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title(title)
    axis.legend(loc="best")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_calibration(
    labels: np.ndarray, probabilities: dict[str, np.ndarray], path: Path
) -> None:
    fig, axis = plt.subplots(figsize=(7, 6))
    for model in MODEL_ORDER:
        observed, predicted = calibration_curve(
            labels, probabilities[model], n_bins=10, strategy="quantile"
        )
        axis.plot(predicted, observed, marker="o", linewidth=1.8, label=model)
    axis.plot([0, 1], [0, 1], "k--", linewidth=1)
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed event rate")
    axis.set_title("Valid calibration")
    axis.legend(loc="best")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_confusion_matrices(
    valid_metrics: dict[str, dict[str, float | int | str]],
    thresholds: dict[str, float],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for axis, model in zip(axes, MODEL_ORDER, strict=True):
        metrics = valid_metrics[model]
        matrix = np.array(
            [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
            dtype=int,
        )
        image = axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        axis.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
        axis.set_yticks([0, 1], labels=["True 0", "True 1"])
        axis.set_title(f"{model}\nthreshold={thresholds[model]:.4g}")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle("Valid confusion matrices (frozen OOF thresholds)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_training_curves(history: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    early = history[history["stage"] == "inner_early_stopping"]
    retrain = history[history["stage"] == "full_outer_retrain"]
    for fold in range(1, N_SPLITS + 1):
        early_fold = early[early["fold"] == fold]
        retrain_fold = retrain[retrain["fold"] == fold]
        axes[0].plot(early_fold["epoch"], early_fold["train_loss"], label=f"Fold {fold}")
        axes[1].plot(
            early_fold["epoch"],
            early_fold["inner_validation_auprc"],
            label=f"Fold {fold}",
        )
        axes[2].plot(
            retrain_fold["epoch"], retrain_fold["train_loss"], label=f"Fold {fold}"
        )
    axes[0].set_title("Inner-stage training loss")
    axes[1].set_title("Inner validation AUPRC")
    axes[2].set_title("Full outer retrain loss")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("BCEWithLogitsLoss")
    axes[1].set_ylabel("AUPRC")
    axes[2].set_ylabel("BCEWithLogitsLoss")
    axes[2].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = [header, separator]
    for _, row in frame.loc[:, columns].iterrows():
        cells: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                cells.append("NA" if math.isnan(float(value)) else f"{float(value):.6f}")
            else:
                cells.append(str(value).replace("|", "\\|"))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def evidence_statement(paired: pd.DataFrame, reference: str) -> str:
    subset = paired[paired["reference_model"] == reference].set_index("metric")
    positive_metrics = [
        metric
        for metric in ["auroc_difference", "auprc_difference", "brier_improvement"]
        if float(subset.loc[metric, "ci_lower_2_5"]) > 0.0
    ]
    crossing_metrics = [
        metric
        for metric in ["auroc_difference", "auprc_difference", "brier_improvement"]
        if bool(subset.loc[metric, "ci_crosses_zero"])
    ]
    if len(positive_metrics) == 3:
        return (
            f"Against {reference}, all three paired 95% CIs are positive; the Valid data "
            "support a clear MLP improvement for discrimination and Brier score."
        )
    if positive_metrics:
        return (
            f"Against {reference}, positive paired evidence exists for {positive_metrics}, "
            f"but {crossing_metrics} cross zero and cannot be claimed as clear gains."
        )
    return (
        f"Against {reference}, the paired 95% CIs do not establish a clear positive MLP "
        "gain; no definitive improvement should be claimed."
    )


def generate_reports(
    train_metrics: pd.DataFrame,
    valid_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    paired: pd.DataFrame,
    fold_metadata: list[dict[str, Any]],
    thresholds: dict[str, float],
    started_at: datetime,
    ended_at: datetime,
    input_hashes: dict[str, str],
    build_report_hash: str,
    bootstrap_attempts: int,
) -> None:
    best_epochs = [int(item["best_epoch"]) for item in fold_metadata]
    valid_ci = bootstrap[bootstrap["metric"].isin(["auroc", "auprc", "brier_score"])].copy()
    dummy_evidence = evidence_statement(paired, "Dummy")
    logistic_evidence = evidence_statement(paired, "Logistic")
    warnings = []
    for _, row in paired.iterrows():
        if bool(row["ci_crosses_zero"]):
            warnings.append(
                f"{row['metric']} for MLP vs {row['reference_model']} has a 95% CI "
                "crossing zero; no clear gain is claimed."
            )
    if not warnings:
        warnings.append("No paired comparison CI crossed zero.")
    elapsed = (ended_at - started_at).total_seconds()
    report = f"""# Adverse Pre+Post MLP v1 report

## Protocol status

- Experiment: `{EXPERIMENT}`
- Status: SUCCESS
- Independent Valid used for early stopping, threshold selection, model selection, or tuning: **No**
- Valid probabilities: mean of five Train-derived outer-fold model probabilities
- Thresholds: selected only from pooled Train OOF probabilities
- MLP loss: `BCEWithLogitsLoss`
- GPU required and used: Yes
- Read-only data-build report SHA256 preserved: `{build_report_hash}`

## Inputs

- Train: `{TRAIN_PATH}` — 794 patients, 132 positive, 96 model features
- Valid: `{VALID_PATH}` — 209 patients, 36 positive, 96 model features
- Train SHA256: `{input_hashes['train']}`
- Valid SHA256: `{input_hashes['valid']}`

## MLP architecture and folds

- Architecture: `96 → 64 → 16 → 1`, ReLU, dropout 0.30/0.20, no terminal sigmoid in the model
- Best epochs: {best_epochs}

## Train pooled OOF metrics

{markdown_table(train_metrics, ['model', 'threshold', 'auroc', 'auprc', 'balanced_accuracy', 'f1', 'precision', 'sensitivity', 'specificity', 'brier_score', 'tp', 'tn', 'fp', 'fn'])}

## Independent Valid metrics

{markdown_table(valid_metrics, ['model', 'threshold', 'auroc', 'auprc', 'balanced_accuracy', 'f1', 'precision', 'sensitivity', 'specificity', 'brier_score', 'tp', 'tn', 'fp', 'fn'])}

## Valid 95% patient-bootstrap confidence intervals

{markdown_table(valid_ci, ['model', 'metric', 'estimate', 'ci_lower_2_5', 'ci_upper_97_5'])}

## Paired Bootstrap comparisons

Positive AUROC/AUPRC differences mean MLP is higher. Positive Brier improvement means reference Brier minus MLP Brier is positive.

{markdown_table(paired, ['reference_model', 'metric', 'estimate', 'ci_lower_2_5', 'ci_upper_97_5', 'ci_crosses_zero'])}

## Evidence assessment

- Predictive value relative to Dummy: {dummy_evidence}
- Nonlinear MLP value relative to Logistic: {logistic_evidence}
- These are out-of-sample associations on the frozen Valid cohort, not causal claims.

## Warnings

{os.linesep.join(f'- {warning}' for warning in warnings)}

- Bootstrap accepted repeats: {BOOTSTRAP_REPEATS}; attempts: {bootstrap_attempts}; single-class samples skipped: {bootstrap_attempts - BOOTSTRAP_REPEATS}.
- Total runtime: {elapsed:.1f} seconds.
"""
    summary = f"""# Execution summary

- Experiment: `{EXPERIMENT}`
- Exit code: 0
- Started UTC: {started_at.isoformat()}
- Completed UTC: {ended_at.isoformat()}
- Total runtime seconds: {elapsed:.3f}
- Python PID: {os.getpid()}
- Python: `{sys.executable}`
- CUDA device: `{torch.cuda.get_device_name(0)}`
- Train assertions: 794 rows, 132 positives, 48 Pre + 48 Post = 96 finite features
- Valid assertions: 209 rows, 36 positives, 48 Pre + 48 Post = 96 finite features
- Train/Valid patient intersection: 0
- Fold best epochs: {best_epochs}
- Frozen thresholds: `{json.dumps(thresholds, ensure_ascii=False, default=json_default)}`
- Valid was not used for early stopping, tuning, model selection, or threshold selection.
- Existing `data_build_report.md` was preserved unchanged.
"""
    write_text_sync(REPORT_DIR / "report.md", report, mode="x")
    write_text_sync(REPORT_DIR / "execution_summary.md", summary, mode="x")


def verify_required_artifacts(build_report_hash: str, logger: logging.Logger) -> None:
    missing_or_empty = []
    for relative in REQUIRED_OUTPUT_FILES:
        path = OUTPUT_DIR / relative
        if not path.is_file() or path.stat().st_size <= 0:
            missing_or_empty.append(str(path))
    for name in ["report.md", "execution_summary.md", "data_build_report.md"]:
        path = REPORT_DIR / name
        if not path.is_file() or path.stat().st_size <= 0:
            missing_or_empty.append(str(path))
    if missing_or_empty:
        raise ExperimentError(f"Required artifacts missing or empty: {missing_or_empty}")
    if sha256_file(BUILD_REPORT_PATH) != build_report_hash:
        raise ExperimentError("data_build_report.md changed during training")
    flush_logger(logger)


def main() -> int:
    logger: logging.Logger | None = None
    started_at = utc_now()
    build_report_hash = ""
    try:
        build_report_hash = preflight_output_directories()
        initialize_run_directories(started_at)
        logger = setup_logging(OUTPUT_DIR / "run.log")
        logger.info("%s started | pid=%d", EXPERIMENT, os.getpid())
        logger.info("Command interpreter=%s | script=%s", sys.executable, Path(__file__).resolve())
        logger.info(
            "Read-only data build report accepted and preserved | path=%s | sha256=%s",
            BUILD_REPORT_PATH,
            build_report_hash,
        )
        write_json_sync(OUTPUT_DIR / "configuration.json", configuration_payload())

        input_hashes_before = {
            "train": sha256_file(TRAIN_PATH),
            "valid": sha256_file(VALID_PATH),
        }
        save_environment(OUTPUT_DIR / "environment.txt", started_at, input_hashes_before)
        logger.info("Reading frozen Pre+Post CSV inputs")
        train_df = pd.read_csv(TRAIN_PATH, dtype={"patient_id": "string"})
        valid_df = pd.read_csv(VALID_PATH, dtype={"patient_id": "string"})
        feature_names, train_ids, valid_ids = validate_inputs(train_df, valid_df)
        logger.info(
            "All input assertions passed | Train=%d positive=%d | Valid=%d positive=%d | "
            "Pre=48 Post=48 model_features=%d | patient_overlap=0",
            len(train_df),
            int(pd.to_numeric(train_df["adverse"]).sum()),
            len(valid_df),
            int(pd.to_numeric(valid_df["adverse"]).sum()),
            len(feature_names),
        )
        write_json_sync(
            OUTPUT_DIR / "feature_names.json",
            {
                "feature_count": len(feature_names),
                "pre_feature_count": EXPECTED_PRE,
                "post_feature_count": EXPECTED_POST,
                "feature_names": feature_names,
                "pre_feature_names": core_pre_columns(feature_names),
                "post_feature_names": core_post_columns(feature_names),
                "excluded_from_model": ["patient_id", "split", "adverse"],
                "forbidden_tokens": ["delta_", "runtime_s", "n_pairs", "missing"],
            },
        )

        device = verify_gpu(OUTPUT_DIR / "gpu_verification.txt", logger)
        torch.cuda.reset_peak_memory_stats(device)

        x_train = train_df.loc[:, feature_names].apply(pd.to_numeric, errors="raise")
        x_valid = valid_df.loc[:, feature_names].apply(pd.to_numeric, errors="raise")
        y_train = pd.to_numeric(train_df["adverse"], errors="raise").astype(int).to_numpy()
        y_valid = pd.to_numeric(valid_df["adverse"], errors="raise").astype(int).to_numpy()
        outer_splits = list(
            StratifiedKFold(
                n_splits=N_SPLITS,
                shuffle=True,
                random_state=SEED,
            ).split(x_train, y_train)
        )

        oof_probabilities = {
            model: np.full(len(train_df), np.nan, dtype=float) for model in MODEL_ORDER
        }
        oof_assignments = {
            model: np.zeros(len(train_df), dtype=int) for model in MODEL_ORDER
        }
        valid_fold_probabilities = {
            model: np.full((len(valid_df), N_SPLITS), np.nan, dtype=float)
            for model in MODEL_ORDER
        }
        outer_fold_assignment = np.zeros(len(train_df), dtype=int)
        dummy_models: list[DummyClassifier] = []
        logistic_models: list[Pipeline] = []
        history_rows: list[dict[str, Any]] = []
        fold_metadata: list[dict[str, Any]] = []

        for fold, (development_index, holdout_index) in enumerate(outer_splits, start=1):
            logger.info(
                "Outer fold %d/%d started | development=%d positive=%d | holdout=%d positive=%d",
                fold,
                N_SPLITS,
                len(development_index),
                int(y_train[development_index].sum()),
                len(holdout_index),
                int(y_train[holdout_index].sum()),
            )
            outer_fold_assignment[holdout_index] = fold

            dummy = DummyClassifier(strategy="prior")
            dummy.fit(x_train.iloc[development_index], y_train[development_index])
            oof_probabilities["Dummy"][holdout_index] = dummy.predict_proba(
                x_train.iloc[holdout_index]
            )[:, 1]
            valid_fold_probabilities["Dummy"][:, fold - 1] = dummy.predict_proba(x_valid)[:, 1]
            oof_assignments["Dummy"][holdout_index] += 1
            dummy_models.append(dummy)

            logistic = logistic_pipeline()
            logistic.fit(x_train.iloc[development_index], y_train[development_index])
            oof_probabilities["Logistic"][holdout_index] = logistic.predict_proba(
                x_train.iloc[holdout_index]
            )[:, 1]
            valid_fold_probabilities["Logistic"][:, fold - 1] = logistic.predict_proba(x_valid)[:, 1]
            oof_assignments["Logistic"][holdout_index] += 1
            logistic_models.append(logistic)
            logger.info("Outer fold %d baselines complete", fold)

            inner_train_index, inner_valid_index = train_test_split(
                development_index,
                test_size=INNER_VALID_FRACTION,
                random_state=SEED,
                shuffle=True,
                stratify=y_train[development_index],
            )
            inner_scaler = StandardScaler()
            x_inner_train_scaled = inner_scaler.fit_transform(x_train.iloc[inner_train_index])
            x_inner_valid_scaled = inner_scaler.transform(x_train.iloc[inner_valid_index])
            best_epoch, best_inner_auprc, inner_pos_weight = train_with_early_stopping(
                fold,
                x_inner_train_scaled,
                y_train[inner_train_index],
                x_inner_valid_scaled,
                y_train[inner_valid_index],
                device,
                history_rows,
                logger,
            )

            full_scaler = StandardScaler()
            x_development_scaled = full_scaler.fit_transform(x_train.iloc[development_index])
            x_holdout_scaled = full_scaler.transform(x_train.iloc[holdout_index])
            x_valid_scaled = full_scaler.transform(x_valid)
            joblib.dump(full_scaler, SCALER_DIR / f"mlp_fold_{fold}_scaler.joblib")
            model, full_pos_weight = train_fixed_epochs(
                fold,
                x_development_scaled,
                y_train[development_index],
                best_epoch,
                device,
                history_rows,
                logger,
            )
            oof_probabilities["Pre+Post MLP"][holdout_index] = predict_mlp(
                model, x_holdout_scaled, device
            )
            valid_fold_probabilities["Pre+Post MLP"][:, fold - 1] = predict_mlp(
                model, x_valid_scaled, device
            )
            oof_assignments["Pre+Post MLP"][holdout_index] += 1
            checkpoint = {
                "experiment": EXPERIMENT,
                "fold": fold,
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "feature_names": feature_names,
                "architecture": NETWORK_DESCRIPTION,
                "best_epoch": best_epoch,
                "best_inner_validation_auprc": best_inner_auprc,
                "outer_development_size": len(development_index),
                "outer_holdout_size": len(holdout_index),
                "pos_weight_full_outer_development": full_pos_weight,
                "base_random_seed": SEED,
                "training_seed": SEED + 1000 + fold,
            }
            torch.save(checkpoint, MODEL_DIR / f"mlp_fold_{fold}.pt")
            fold_metadata.append(
                {
                    "fold": fold,
                    "outer_development_size": len(development_index),
                    "outer_development_positive": int(y_train[development_index].sum()),
                    "outer_holdout_size": len(holdout_index),
                    "outer_holdout_positive": int(y_train[holdout_index].sum()),
                    "inner_train_size": len(inner_train_index),
                    "inner_train_positive": int(y_train[inner_train_index].sum()),
                    "inner_early_stopping_size": len(inner_valid_index),
                    "inner_early_stopping_positive": int(y_train[inner_valid_index].sum()),
                    "best_epoch": best_epoch,
                    "best_inner_validation_auprc": best_inner_auprc,
                    "inner_pos_weight": inner_pos_weight,
                    "full_outer_pos_weight": full_pos_weight,
                    "valid_used_for_early_stopping": False,
                    "valid_probability_generated": True,
                }
            )
            logger.info(
                "Outer fold %d/%d complete | best_epoch=%d | best_inner_AUPRC=%.6f | "
                "OOF_holdout=%d | Valid_probability_saved_for_later_mean=%d",
                fold,
                N_SPLITS,
                best_epoch,
                best_inner_auprc,
                len(holdout_index),
                len(valid_df),
            )
            del model, inner_scaler, full_scaler
            torch.cuda.empty_cache()
            gc.collect()

        for model in MODEL_ORDER:
            if not np.all(oof_assignments[model] == 1):
                raise ExperimentError(f"{model}: each Train patient must have exactly one OOF prediction")
            if not np.isfinite(oof_probabilities[model]).all():
                raise ExperimentError(f"{model}: OOF probabilities are incomplete or non-finite")
            if not np.isfinite(valid_fold_probabilities[model]).all():
                raise ExperimentError(f"{model}: Valid fold probabilities are incomplete or non-finite")
        if not np.all(outer_fold_assignment >= 1):
            raise ExperimentError("Outer-fold assignment is incomplete")
        logger.info("All five folds complete; OOF assignment exactly once per Train patient")

        joblib.dump(
            {"experiment": EXPERIMENT, "outer_cv": configuration_payload()["outer_cv"], "fold_models": dummy_models},
            MODEL_DIR / "dummy.joblib",
        )
        joblib.dump(
            {"experiment": EXPERIMENT, "outer_cv": configuration_payload()["outer_cv"], "fold_models": logistic_models},
            MODEL_DIR / "logistic.joblib",
        )
        history_frame = pd.DataFrame(history_rows)
        history_frame.to_csv(OUTPUT_DIR / "training_history.csv", index=False)

        thresholds: dict[str, float] = {}
        threshold_frames: list[pd.DataFrame] = []
        train_metric_records: dict[str, dict[str, float | int | str]] = {}
        for model in MODEL_ORDER:
            threshold, threshold_frame = select_threshold(
                model, y_train, oof_probabilities[model]
            )
            thresholds[model] = threshold
            threshold_frames.append(threshold_frame)
            train_metric_records[model] = calculate_metrics(
                y_train, oof_probabilities[model], threshold
            )
            logger.info(
                "%s Train pooled OOF | threshold=%.12g | AUROC=%.6f AUPRC=%.6f "
                "balanced_accuracy=%.6f Brier=%.6f",
                model,
                threshold,
                train_metric_records[model]["auroc"],
                train_metric_records[model]["auprc"],
                train_metric_records[model]["balanced_accuracy"],
                train_metric_records[model]["brier_score"],
            )
        write_json_sync(OUTPUT_DIR / "thresholds.json", thresholds)
        pd.concat(threshold_frames, ignore_index=True).to_csv(
            OUTPUT_DIR / "threshold_search.csv", index=False
        )

        train_oof_frame = pd.DataFrame(
            {
                "patient_id": train_ids,
                "split": train_df["split"],
                "adverse": y_train,
                "outer_fold": outer_fold_assignment,
            }
        )
        for model in MODEL_ORDER:
            slug = MODEL_SLUG[model]
            train_oof_frame[f"{slug}_probability"] = oof_probabilities[model]
            train_oof_frame[f"{slug}_prediction"] = (
                oof_probabilities[model] >= thresholds[model]
            ).astype(int)
            train_oof_frame[f"{slug}_threshold"] = thresholds[model]
        train_oof_frame.to_csv(OUTPUT_DIR / "train_oof_predictions.csv", index=False)

        train_metrics_frame = pd.DataFrame(
            [
                {
                    "dataset": "train_pooled_oof",
                    "model": model,
                    "threshold": thresholds[model],
                    **train_metric_records[model],
                }
                for model in MODEL_ORDER
            ]
        )
        train_metrics_frame.to_csv(OUTPUT_DIR / "train_oof_metrics.csv", index=False)

        fold_rows: list[dict[str, Any]] = []
        metadata_by_fold = {int(item["fold"]): item for item in fold_metadata}
        for fold, (_, holdout_index) in enumerate(outer_splits, start=1):
            for model in MODEL_ORDER:
                row = {
                    "fold": fold,
                    "model": model,
                    "threshold_from_pooled_train_oof": thresholds[model],
                    **calculate_metrics(
                        y_train[holdout_index],
                        oof_probabilities[model][holdout_index],
                        thresholds[model],
                    ),
                }
                row.update(metadata_by_fold[fold])
                fold_rows.append(row)
        pd.DataFrame(fold_rows).to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)

        frozen_at = utc_now()
        frozen_protocol = {
            "experiment": EXPERIMENT,
            "frozen_at_utc": frozen_at.isoformat(),
            "feature_names": feature_names,
            "feature_count": len(feature_names),
            "network_structure": NETWORK_DESCRIPTION,
            "fold_best_epochs": [int(item["best_epoch"]) for item in fold_metadata],
            "fold_best_inner_validation_auprc": [
                float(item["best_inner_validation_auprc"]) for item in fold_metadata
            ],
            "model_thresholds": thresholds,
            "threshold_source": "Train pooled OOF only",
            "threshold_rule": configuration_payload()["threshold_rule"],
            "mlp_scaler_rule": (
                "Internal StandardScaler fit only on inner training subset for early stopping; "
                "new StandardScaler fit on complete outer development fold for exact best-epoch retrain"
            ),
            "loss": "BCEWithLogitsLoss",
            "pos_weight_strategy": "n_negative / n_positive for each MLP training subset",
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "random_seed": SEED,
            "valid_probability_rule": "arithmetic mean of five outer-fold model probabilities",
            "valid_used_before_freeze_for_metrics": False,
            "valid_used_for_early_stopping": False,
            "valid_used_for_threshold_selection": False,
            "valid_used_for_model_selection_or_tuning": False,
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
        }
        write_json_sync(OUTPUT_DIR / "frozen_protocol.json", frozen_protocol)
        if (OUTPUT_DIR / "frozen_protocol.json").stat().st_size <= 0:
            raise ExperimentError("frozen_protocol.json was not durably written")
        logger.info(
            "frozen_protocol.json written before any Valid metric | frozen_at=%s",
            frozen_at.isoformat(),
        )

        valid_probabilities = {
            model: valid_fold_probabilities[model].mean(axis=1) for model in MODEL_ORDER
        }
        valid_metric_records: dict[str, dict[str, float | int | str]] = {}
        for model in MODEL_ORDER:
            valid_metric_records[model] = calculate_metrics(
                y_valid, valid_probabilities[model], thresholds[model]
            )
            logger.info(
                "%s official Valid | AUROC=%.6f AUPRC=%.6f balanced_accuracy=%.6f "
                "Brier=%.6f | frozen_threshold=%.12g",
                model,
                valid_metric_records[model]["auroc"],
                valid_metric_records[model]["auprc"],
                valid_metric_records[model]["balanced_accuracy"],
                valid_metric_records[model]["brier_score"],
                thresholds[model],
            )

        valid_prediction_frame = pd.DataFrame(
            {"patient_id": valid_ids, "split": valid_df["split"], "adverse": y_valid}
        )
        for model in MODEL_ORDER:
            slug = MODEL_SLUG[model]
            for fold in range(1, N_SPLITS + 1):
                valid_prediction_frame[f"{slug}_fold_{fold}_probability"] = (
                    valid_fold_probabilities[model][:, fold - 1]
                )
            valid_prediction_frame[f"{slug}_probability"] = valid_probabilities[model]
            valid_prediction_frame[f"{slug}_prediction"] = (
                valid_probabilities[model] >= thresholds[model]
            ).astype(int)
            valid_prediction_frame[f"{slug}_threshold"] = thresholds[model]
        valid_prediction_frame.to_csv(OUTPUT_DIR / "valid_predictions.csv", index=False)

        valid_metrics_frame = pd.DataFrame(
            [
                {
                    "dataset": "valid_frozen_five_fold_mean",
                    "model": model,
                    "threshold": thresholds[model],
                    **valid_metric_records[model],
                }
                for model in MODEL_ORDER
            ]
        )
        valid_metrics_frame.to_csv(OUTPUT_DIR / "valid_metrics.csv", index=False)

        bootstrap_frame, paired_frame = bootstrap_valid(
            y_valid,
            valid_probabilities,
            thresholds,
            valid_metric_records,
            logger,
        )
        bootstrap_frame.to_csv(
            OUTPUT_DIR / "bootstrap_confidence_intervals.csv", index=False
        )
        paired_frame.to_csv(
            OUTPUT_DIR / "paired_bootstrap_comparisons.csv", index=False
        )

        plot_roc_curves(
            y_train,
            oof_probabilities,
            "Train pooled OOF ROC",
            PLOT_DIR / "train_oof_roc.png",
        )
        plot_pr_curves(
            y_train,
            oof_probabilities,
            "Train pooled OOF precision-recall",
            PLOT_DIR / "train_oof_pr.png",
        )
        plot_roc_curves(
            y_valid,
            valid_probabilities,
            "Independent Valid ROC",
            PLOT_DIR / "valid_roc.png",
        )
        plot_pr_curves(
            y_valid,
            valid_probabilities,
            "Independent Valid precision-recall",
            PLOT_DIR / "valid_pr.png",
        )
        plot_calibration(y_valid, valid_probabilities, PLOT_DIR / "valid_calibration.png")
        plot_confusion_matrices(
            valid_metric_records, thresholds, PLOT_DIR / "valid_confusion_matrices.png"
        )
        plot_training_curves(history_frame, PLOT_DIR / "mlp_training_curves.png")

        peak_memory = torch.cuda.max_memory_allocated(device)
        write_text_sync(
            OUTPUT_DIR / "gpu_verification.txt",
            f"formal_training_peak_memory_allocated_bytes={peak_memory}\n"
            f"formal_training_completed_on_device={device}\n",
            mode="a",
        )

        input_hashes_after = {
            "train": sha256_file(TRAIN_PATH),
            "valid": sha256_file(VALID_PATH),
        }
        if input_hashes_after != input_hashes_before:
            raise ExperimentError("Frozen input CSV hash changed during training")
        if sha256_file(BUILD_REPORT_PATH) != build_report_hash:
            raise ExperimentError("Read-only data_build_report.md changed during training")

        ended_at = utc_now()
        bootstrap_attempts = int(bootstrap_frame["sampling_attempts"].iloc[0])
        generate_reports(
            train_metrics_frame,
            valid_metrics_frame,
            bootstrap_frame,
            paired_frame,
            fold_metadata,
            thresholds,
            started_at,
            ended_at,
            input_hashes_before,
            build_report_hash,
            bootstrap_attempts,
        )
        write_text_sync(OUTPUT_DIR / "exit_status.txt", "0\n")
        verify_required_artifacts(build_report_hash, logger)
        logger.info(
            "All required artifacts verified | elapsed_seconds=%.3f | exit_code=0",
            (ended_at - started_at).total_seconds(),
        )
        flush_logger(logger)
        RUNNING_PATH.unlink()
        write_text_sync(
            SUCCESS_PATH,
            f"status=SUCCESS\ncompleted_at_utc={ended_at.isoformat()}\nexit_status=0\n",
            mode="x",
        )
        return 0
    except Exception as exc:
        if OUTPUT_DIR.exists():
            try:
                if SUCCESS_PATH.exists():
                    SUCCESS_PATH.unlink()
                if not RUNNING_PATH.exists():
                    write_text_sync(
                        RUNNING_PATH,
                        f"status=FAILED_RUNNING_RETAINED\nfailed_at_utc={utc_now().isoformat()}\n",
                    )
                write_text_sync(
                    OUTPUT_DIR / "exit_status.txt",
                    f"1\n{type(exc).__name__}: {exc}\n",
                )
            except Exception:
                pass
        if logger is not None:
            logger.exception("%s failed; .RUNNING retained", EXPERIMENT)
        else:
            traceback.print_exc()
        return 1
    finally:
        if logger is not None and logger.handlers:
            close_logger(logger)


if __name__ == "__main__":
    sys.exit(main())

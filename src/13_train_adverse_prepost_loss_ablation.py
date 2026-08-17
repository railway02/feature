#!/root/autodl-tmp/envs/aneurysm-ml/bin/python
"""Exploratory loss-function ablation for the frozen Pre+Post MLP dataset.

This is explicitly an exploratory loss-function ablation after observing the
original Valid results. It is not a new independent confirmation experiment.
All three losses use identical folds, inner early-stopping splits, scalers,
network architecture, optimizer settings, initialization seeds, and batch
ordering. The independent Valid set is never used for epoch or threshold
selection.
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
PROTECTED_OUTPUT_DIR = PROJECT_ROOT / "outputs/baselines/adverse_prepost_mlp_v1"
PROTECTED_REPORT_DIR = PROJECT_ROOT / "reports/adverse_prepost_mlp_v1"
PROTECTED_SCRIPT = PROJECT_ROOT / "code/12_train_adverse_prepost_mlp.py"
OUTPUT_DIR = PROJECT_ROOT / "outputs/baselines/adverse_prepost_mlp_loss_ablation_v1"
REPORT_DIR = PROJECT_ROOT / "reports/adverse_prepost_mlp_loss_ablation_v1"
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
    from sklearn.preprocessing import StandardScaler
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    print(f"Dependency import failed under {sys.executable}: {exc}", file=sys.stderr)
    raise


EXPERIMENT = "adverse_prepost_mlp_loss_ablation_v1"
EXPLORATORY_LABEL = (
    "exploratory loss-function ablation after observing the original Valid results"
)
INDEPENDENCE_DISCLAIMER = "not a new independent confirmation experiment"
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
EXPECTED_COLUMNS = 99
EXPECTED_PRE = 48
EXPECTED_POST = 48
EXPECTED_FEATURES = 96
LOSS_ORDER = ["weighted_bce", "unweighted_bce", "mse_brier"]
LOSS_DEFINITIONS = {
    "weighted_bce": (
        "BCEWithLogitsLoss(pos_weight=n_negative/n_positive); loss applied to logits"
    ),
    "unweighted_bce": "BCEWithLogitsLoss(); no pos_weight; loss applied to logits",
    "mse_brier": (
        "MSELoss(torch.sigmoid(logits), labels); no pos_weight; MSE is never applied "
        "directly to logits"
    ),
}
NETWORK_DESCRIPTION = [
    "Linear(96, 64)",
    "ReLU",
    "Dropout(0.30)",
    "Linear(64, 16)",
    "ReLU",
    "Dropout(0.20)",
    "Linear(16, 1)",
]
METRIC_NAMES = [
    "auroc",
    "auprc",
    "brier_score",
    "balanced_accuracy",
    "f1",
    "precision",
    "sensitivity",
    "specificity",
]
PAIRED_COMPARISONS = [
    ("unweighted_bce", "weighted_bce"),
    ("mse_brier", "weighted_bce"),
    ("mse_brier", "unweighted_bce"),
]
REQUIRED_OUTPUTS = [
    "train_oof_predictions.csv",
    "valid_predictions.csv",
    "fold_assignments.csv",
    "fold_metrics.csv",
    "train_oof_metrics.csv",
    "valid_metrics.csv",
    "bootstrap_confidence_intervals.csv",
    "paired_bootstrap_comparisons.csv",
    "training_history.csv",
    "threshold_search.csv",
    "thresholds.json",
    "feature_names.json",
    "configuration.json",
    "frozen_protocol.json",
    "protected_state.json",
    "environment.txt",
    "gpu_verification.txt",
    "run.log",
    "exit_status.txt",
    "scalers/fold_1_scaler.joblib",
    "scalers/fold_2_scaler.joblib",
    "scalers/fold_3_scaler.joblib",
    "scalers/fold_4_scaler.joblib",
    "scalers/fold_5_scaler.joblib",
    "plots/train_oof_roc.png",
    "plots/train_oof_pr.png",
    "plots/valid_roc.png",
    "plots/valid_pr.png",
    "plots/valid_calibration.png",
    "plots/training_curves.png",
]
for loss_name in LOSS_ORDER:
    for fold in range(1, N_SPLITS + 1):
        REQUIRED_OUTPUTS.append(f"models/{loss_name}_fold_{fold}.pt")


class ExperimentError(RuntimeError):
    """Raised when a locked ablation assertion fails."""


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(EXPECTED_FEATURES, 64),
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


def directory_manifest(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise ExperimentError(f"Protected directory is missing: {path}")
    rows: list[str] = []
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        rows.append(
            f"{item.relative_to(path)}\t{item.stat().st_size}\t{sha256_file(item)}"
        )
    digest = hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()
    return {
        "path": str(path.resolve()),
        "file_count": len(files),
        "manifest_sha256": digest,
        "files": rows,
    }


def protected_state() -> dict[str, Any]:
    if not PROTECTED_SCRIPT.is_file():
        raise ExperimentError(f"Protected script is missing: {PROTECTED_SCRIPT}")
    return {
        "protected_script": {
            "path": str(PROTECTED_SCRIPT.resolve()),
            "sha256": sha256_file(PROTECTED_SCRIPT),
            "size": PROTECTED_SCRIPT.stat().st_size,
        },
        "protected_output_directory": directory_manifest(PROTECTED_OUTPUT_DIR),
        "protected_report_directory": directory_manifest(PROTECTED_REPORT_DIR),
    }


def preflight_new_paths() -> dict[str, Any]:
    for path, description in (
        (OUTPUT_DIR, "new model output directory"),
        (REPORT_DIR, "new report directory"),
    ):
        if path.exists():
            entries = sorted(str(item) for item in path.rglob("*") if item.exists())
            if entries:
                raise ExperimentError(
                    f"{description} is not empty; refusing to overwrite: {entries}"
                )
    return protected_state()


def initialize_directories(started_at: datetime) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
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


def normalize_patient_ids(series: pd.Series, context: str) -> pd.Series:
    if series.isna().any():
        raise ExperimentError(f"{context}: patient_id contains missing values")

    def normalize(value: object) -> str:
        text = str(value).strip()
        if not text:
            raise ExperimentError(f"{context}: empty patient_id")
        if text.endswith(".0") and text[:-2].lstrip("+-").isdigit():
            text = text[:-2]
        return text[1:] if text.startswith("+") else text

    normalized = series.map(normalize)
    if normalized.duplicated().any():
        duplicates = normalized[normalized.duplicated(keep=False)].unique().tolist()
        raise ExperimentError(f"{context}: duplicate patient_id values: {duplicates[:10]}")
    return normalized


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
        raise ExperimentError("Train and Valid columns/order differ")
    if train_df.columns[:3].tolist() != ["patient_id", "split", "adverse"]:
        raise ExperimentError("Metadata columns are not in the required order")
    pre_columns = core_pre_columns(train_df.columns)
    post_columns = core_post_columns(train_df.columns)
    features = [*pre_columns, *post_columns]
    if len(pre_columns) != EXPECTED_PRE or len(post_columns) != EXPECTED_POST:
        raise ExperimentError(
            f"Expected 48 Pre and 48 Post features; found {len(pre_columns)} and "
            f"{len(post_columns)}"
        )
    if train_df.columns.tolist() != ["patient_id", "split", "adverse", *features]:
        raise ExperimentError("Input is not exactly metadata plus 96 core Pre+Post features")
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
        raise ExperimentError(f"Forbidden columns found: {forbidden}")
    if set(features) & {"patient_id", "split", "adverse"}:
        raise ExperimentError("Metadata or labels entered model features")

    train_ids = normalize_patient_ids(train_df["patient_id"], "Train")
    valid_ids = normalize_patient_ids(valid_df["patient_id"], "Valid")
    overlap = set(train_ids) & set(valid_ids)
    if overlap:
        raise ExperimentError(f"Train/Valid patient overlap is {len(overlap)}")
    for split, frame, ids in (
        ("train", train_df, train_ids),
        ("valid", valid_df, valid_ids),
    ):
        if frame.shape != (EXPECTED_ROWS[split], EXPECTED_COLUMNS):
            raise ExperimentError(
                f"{split}: expected {EXPECTED_ROWS[split]}x{EXPECTED_COLUMNS}, "
                f"found {frame.shape[0]}x{frame.shape[1]}"
            )
        if ids.nunique() != EXPECTED_ROWS[split]:
            raise ExperimentError(f"{split}: patient_id is not unique")
        if set(frame["split"].astype(str)) != {split}:
            raise ExperimentError(f"{split}: invalid split values")
        labels = pd.to_numeric(frame["adverse"], errors="raise")
        if set(labels.unique().tolist()) != {0, 1}:
            raise ExperimentError(f"{split}: adverse is not binary 0/1")
        if int(labels.sum()) != EXPECTED_POSITIVES[split]:
            raise ExperimentError(f"{split}: positive count assertion failed")
        values = frame.loc[:, features].apply(pd.to_numeric, errors="raise").to_numpy(float)
        if not np.isfinite(values).all():
            raise ExperimentError(f"{split}: feature matrix contains NaN or infinity")
    if len(features) != EXPECTED_FEATURES:
        raise ExperimentError(f"Expected 96 features, found {len(features)}")
    return features, train_ids, valid_ids


def configuration_payload() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "experiment_label": EXPLORATORY_LABEL,
        "independence_disclaimer": INDEPENDENCE_DISCLAIMER,
        "inputs": {"train": str(TRAIN_PATH), "valid": str(VALID_PATH)},
        "feature_count": EXPECTED_FEATURES,
        "network": NETWORK_DESCRIPTION,
        "losses": LOSS_DEFINITIONS,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
        },
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "early_stopping_metric": "inner AUPRC for every loss",
        "inner_validation_fraction": INNER_VALID_FRACTION,
        "outer_cv": {
            "name": "StratifiedKFold",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": SEED,
        },
        "fairness_controls": [
            "same outer folds",
            "same inner early-stopping indices",
            "same inner and outer scalers",
            "same architecture",
            "same initialization seed within each fold/stage",
            "same DataLoader shuffle seed and batch order within each fold/stage",
            "same optimizer and training limits",
        ],
        "threshold_rule": [
            "maximize Train pooled OOF balanced accuracy",
            "tie: closest to 0.5",
            "tie: smaller threshold",
        ],
        "valid_role": "five-fold probability mean and frozen-protocol evaluation only",
        "bootstrap": {
            "repeats": BOOTSTRAP_REPEATS,
            "random_seed": SEED,
            "shared_indices": True,
            "paired_comparisons": PAIRED_COMPARISONS,
            "single_class_samples": "skip",
            "retrain": False,
            "threshold_reselection": False,
        },
        "random_seed": SEED,
        "gpu_required": True,
    }


def save_environment(path: Path, started_at: datetime, input_hashes: dict[str, str]) -> None:
    lines = [
        f"experiment={EXPERIMENT}",
        f"experiment_label={EXPLORATORY_LABEL}",
        f"independence_disclaimer={INDEPENDENCE_DISCLAIMER}",
        f"started_at_utc={started_at.isoformat()}",
        f"pid={os.getpid()}",
        f"python={sys.executable}",
        f"python_version={sys.version.replace(os.linesep, ' ')}",
        f"platform={platform.platform()}",
        f"torch={torch.__version__}",
        f"torch_cuda={torch.version.cuda}",
        f"sklearn={sklearn.__version__}",
        f"pandas={pd.__version__}",
        f"numpy={np.__version__}",
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
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise ExperimentError("CUDA GPU is required but unavailable")
    device = torch.device("cuda:0")
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    seed_everything(SEED)
    test_a = torch.randn((128, 96), device=device)
    test_b = torch.randn((96, 32), device=device)
    test_value = (test_a @ test_b).mean()
    torch.cuda.synchronize(device)
    if not bool(torch.isfinite(test_value).item()):
        raise ExperimentError("CUDA smoke test failed")
    props = torch.cuda.get_device_properties(device)
    write_text_sync(
        path,
        "\n".join(
            [
                f"verified_at_utc={utc_now().isoformat()}",
                f"torch_cuda_available={torch.cuda.is_available()}",
                f"torch_cuda_version={torch.version.cuda}",
                f"cuda_device_count={torch.cuda.device_count()}",
                f"selected_device={device}",
                f"device_name={torch.cuda.get_device_name(device)}",
                f"device_capability={torch.cuda.get_device_capability(device)}",
                f"device_total_memory_bytes={props.total_memory}",
                f"cuda_smoke_test_mean={float(test_value.item())}",
                "nvidia_smi:",
                result.stdout.strip(),
            ]
        )
        + "\n",
    )
    logger.info("CUDA verified | device=%s | name=%s", device, torch.cuda.get_device_name(0))
    del test_a, test_b, test_value
    torch.cuda.empty_cache()
    return device


def positive_weight(labels: np.ndarray) -> float:
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives <= 0 or negatives <= 0:
        raise ExperimentError("Training subset must contain both classes")
    return float(negatives / positives)


def loss_components(
    loss_name: str, labels: np.ndarray, device: torch.device
) -> tuple[nn.Module, float | None]:
    if loss_name == "weighted_bce":
        weight = positive_weight(labels)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(weight, dtype=torch.float32, device=device)
        )
        return criterion, weight
    if loss_name == "unweighted_bce":
        return nn.BCEWithLogitsLoss(), None
    if loss_name == "mse_brier":
        return nn.MSELoss(), None
    raise ExperimentError(f"Unknown loss: {loss_name}")


def compute_loss(
    loss_name: str,
    criterion: nn.Module,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    if loss_name == "mse_brier":
        probabilities = torch.sigmoid(logits)
        return criterion(probabilities, labels)
    return criterion(logits, labels)


def make_loader(features: np.ndarray, labels: np.ndarray, seed: int) -> DataLoader:
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


def train_epoch(
    model: MLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    loss_name: str,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    count = 0
    for batch_features, batch_labels in loader:
        batch_features = batch_features.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_features)
        loss = compute_loss(loss_name, criterion, logits, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().item()) * len(batch_labels)
        count += len(batch_labels)
    return total_loss / count


def predict(model: MLP, features: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features.astype(np.float32, copy=False))),
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )
    values: list[np.ndarray] = []
    with torch.no_grad():
        for (batch_features,) in loader:
            logits = model(batch_features.to(device, non_blocking=True))
            values.append(torch.sigmoid(logits).cpu().numpy())
    result = np.concatenate(values).astype(float)
    if not np.isfinite(result).all():
        raise ExperimentError("Non-finite model probabilities")
    return result


def early_stopping_train(
    loss_name: str,
    fold: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_early: np.ndarray,
    y_early: np.ndarray,
    device: torch.device,
    history: list[dict[str, Any]],
    logger: logging.Logger,
) -> tuple[int, float, float | None]:
    stage_seed = SEED + fold
    seed_everything(stage_seed)
    model = MLP().to(device)
    criterion, pos_weight = loss_components(loss_name, y_train, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loader = make_loader(x_train, y_train, stage_seed)
    best_epoch = 0
    best_auprc = -math.inf
    no_improvement = 0
    logger.info(
        "Fold %d/%d | %s early-stopping started | train=%d positive=%d | "
        "early=%d positive=%d | seed=%d | pos_weight=%s",
        fold,
        N_SPLITS,
        loss_name,
        len(y_train),
        int(y_train.sum()),
        len(y_early),
        int(y_early.sum()),
        stage_seed,
        "None" if pos_weight is None else f"{pos_weight:.6f}",
    )
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_epoch(
            model, loader, optimizer, criterion, loss_name, device
        )
        early_probabilities = predict(model, x_early, device)
        early_auprc = float(average_precision_score(y_early, early_probabilities))
        improved = early_auprc > best_auprc + 1e-12
        if improved:
            best_epoch = epoch
            best_auprc = early_auprc
            no_improvement = 0
        else:
            no_improvement += 1
        history.append(
            {
                "fold": fold,
                "loss": loss_name,
                "stage": "inner_early_stopping",
                "epoch": epoch,
                "train_loss": train_loss,
                "inner_auprc": early_auprc,
                "improved": improved,
                "best_epoch_so_far": best_epoch,
                "epochs_without_improvement": no_improvement,
                "training_size": len(y_train),
                "training_positive": int(y_train.sum()),
                "pos_weight": pos_weight,
                "seed": stage_seed,
            }
        )
        if improved or epoch == 1 or epoch % 10 == 0:
            logger.info(
                "Fold %d %s early epoch=%d | loss=%.6f | AUPRC=%.6f | "
                "best_epoch=%d best_AUPRC=%.6f | patience=%d/%d",
                fold,
                loss_name,
                epoch,
                train_loss,
                early_auprc,
                best_epoch,
                best_auprc,
                no_improvement,
                PATIENCE,
            )
        if no_improvement >= PATIENCE:
            logger.info(
                "Fold %d %s early stopping at epoch=%d | best_epoch=%d | "
                "best_AUPRC=%.6f",
                fold,
                loss_name,
                epoch,
                best_epoch,
                best_auprc,
            )
            break
    if best_epoch < 1:
        raise ExperimentError(f"{loss_name} fold {fold}: invalid best epoch")
    del model, criterion, optimizer, loader
    torch.cuda.empty_cache()
    gc.collect()
    return best_epoch, best_auprc, pos_weight


def fixed_epoch_train(
    loss_name: str,
    fold: int,
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    device: torch.device,
    history: list[dict[str, Any]],
    logger: logging.Logger,
) -> tuple[MLP, float | None]:
    stage_seed = SEED + 1000 + fold
    seed_everything(stage_seed)
    model = MLP().to(device)
    criterion, pos_weight = loss_components(loss_name, labels, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loader = make_loader(features, labels, stage_seed)
    logger.info(
        "Fold %d %s reinitialized for exactly %d full-development epochs | "
        "seed=%d | pos_weight=%s",
        fold,
        loss_name,
        epochs,
        stage_seed,
        "None" if pos_weight is None else f"{pos_weight:.6f}",
    )
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model, loader, optimizer, criterion, loss_name, device
        )
        history.append(
            {
                "fold": fold,
                "loss": loss_name,
                "stage": "full_outer_retrain",
                "epoch": epoch,
                "train_loss": train_loss,
                "inner_auprc": math.nan,
                "improved": False,
                "best_epoch_so_far": epochs,
                "epochs_without_improvement": math.nan,
                "training_size": len(labels),
                "training_positive": int(labels.sum()),
                "pos_weight": pos_weight,
                "seed": stage_seed,
            }
        )
        if epoch == 1 or epoch == epochs or epoch % 20 == 0:
            logger.info(
                "Fold %d %s fixed epoch=%d/%d | loss=%.6f",
                fold,
                loss_name,
                epoch,
                epochs,
                train_loss,
            )
    return model, pos_weight


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
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "confusion_matrix": json.dumps([[int(tn), int(fp)], [int(fn), int(tp)]]),
    }


def select_threshold(
    loss_name: str, labels: np.ndarray, probabilities: np.ndarray
) -> tuple[float, pd.DataFrame]:
    unique = np.unique(probabilities)
    above = np.nextafter(unique, np.inf)
    above = above[(above >= 0.0) & (above <= 1.0)]
    candidates = np.unique(
        np.concatenate([np.array([0.0, 0.5, 1.0]), unique, above])
    )
    rows = []
    for threshold in candidates:
        metrics = calculate_metrics(labels, probabilities, float(threshold))
        rows.append(
            {
                "loss": loss_name,
                "threshold": float(threshold),
                "balanced_accuracy": metrics["balanced_accuracy"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
            }
        )
    frame = pd.DataFrame(rows)
    best = float(frame["balanced_accuracy"].max())
    tied = frame[np.isclose(frame["balanced_accuracy"], best, rtol=0.0, atol=1e-12)].copy()
    tied["distance_to_0_5"] = (tied["threshold"] - 0.5).abs()
    tied = tied.sort_values(["distance_to_0_5", "threshold"])
    selected = float(tied.iloc[0]["threshold"])
    frame["distance_to_0_5"] = (frame["threshold"] - 0.5).abs()
    frame["selected"] = np.isclose(frame["threshold"], selected, rtol=0.0, atol=0.0)
    return selected, frame


def bootstrap_analysis(
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
    thresholds: dict[str, float],
    point_metrics: dict[str, dict[str, float | int | str]],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    accepted: list[np.ndarray] = []
    attempts = 0
    while len(accepted) < BOOTSTRAP_REPEATS:
        attempts += 1
        indices = rng.integers(0, len(labels), size=len(labels))
        if np.unique(labels[indices]).size < 2:
            continue
        accepted.append(indices)

    distributions = {
        loss_name: {metric: [] for metric in METRIC_NAMES}
        for loss_name in LOSS_ORDER
    }
    paired = {
        (comparison, reference, metric): []
        for comparison, reference in PAIRED_COMPARISONS
        for metric in ["auroc_difference", "auprc_difference", "brier_improvement"]
    }
    for indices in accepted:
        sampled_labels = labels[indices]
        sampled_metrics = {}
        for loss_name in LOSS_ORDER:
            metrics = calculate_metrics(
                sampled_labels,
                probabilities[loss_name][indices],
                thresholds[loss_name],
            )
            sampled_metrics[loss_name] = metrics
            for metric in METRIC_NAMES:
                distributions[loss_name][metric].append(float(metrics[metric]))
        for comparison, reference in PAIRED_COMPARISONS:
            paired[(comparison, reference, "auroc_difference")].append(
                float(sampled_metrics[comparison]["auroc"])
                - float(sampled_metrics[reference]["auroc"])
            )
            paired[(comparison, reference, "auprc_difference")].append(
                float(sampled_metrics[comparison]["auprc"])
                - float(sampled_metrics[reference]["auprc"])
            )
            paired[(comparison, reference, "brier_improvement")].append(
                float(sampled_metrics[reference]["brier_score"])
                - float(sampled_metrics[comparison]["brier_score"])
            )

    ci_rows: list[dict[str, Any]] = []
    for loss_name in LOSS_ORDER:
        for metric in METRIC_NAMES:
            values = np.asarray(distributions[loss_name][metric], dtype=float)
            ci_rows.append(
                {
                    "loss": loss_name,
                    "metric": metric,
                    "estimate": float(point_metrics[loss_name][metric]),
                    "ci_lower_2_5": float(np.percentile(values, 2.5)),
                    "ci_upper_97_5": float(np.percentile(values, 97.5)),
                    "valid_repeats": BOOTSTRAP_REPEATS,
                    "sampling_attempts": attempts,
                    "skipped_single_class": attempts - BOOTSTRAP_REPEATS,
                    "shared_indices": True,
                    "threshold": thresholds[loss_name],
                }
            )

    paired_rows: list[dict[str, Any]] = []
    for comparison, reference in PAIRED_COMPARISONS:
        estimates = {
            "auroc_difference": float(point_metrics[comparison]["auroc"])
            - float(point_metrics[reference]["auroc"]),
            "auprc_difference": float(point_metrics[comparison]["auprc"])
            - float(point_metrics[reference]["auprc"]),
            "brier_improvement": float(point_metrics[reference]["brier_score"])
            - float(point_metrics[comparison]["brier_score"]),
        }
        for metric, estimate in estimates.items():
            values = np.asarray(paired[(comparison, reference, metric)], dtype=float)
            lower = float(np.percentile(values, 2.5))
            upper = float(np.percentile(values, 97.5))
            crosses_zero = bool(lower <= 0.0 <= upper)
            if lower > 0.0:
                interpretation = "comparison loss is better; CI entirely positive"
            elif upper < 0.0:
                interpretation = "reference loss is better; CI entirely negative"
            else:
                interpretation = "CI crosses zero; no clear difference"
            paired_rows.append(
                {
                    "comparison_loss": comparison,
                    "reference_loss": reference,
                    "metric": metric,
                    "positive_means_comparison_better": True,
                    "estimate": estimate,
                    "ci_lower_2_5": lower,
                    "ci_upper_97_5": upper,
                    "ci_crosses_zero": crosses_zero,
                    "interpretation": interpretation,
                    "valid_repeats": BOOTSTRAP_REPEATS,
                    "sampling_attempts": attempts,
                    "shared_indices": True,
                    "thresholds_frozen": True,
                    "models_retrained": False,
                }
            )
    logger.info(
        "Valid bootstrap complete | accepted=%d attempts=%d skipped_single_class=%d",
        BOOTSTRAP_REPEATS,
        attempts,
        attempts - BOOTSTRAP_REPEATS,
    )
    return pd.DataFrame(ci_rows), pd.DataFrame(paired_rows)


def plot_roc_set(
    labels: np.ndarray, probabilities: dict[str, np.ndarray], title: str, path: Path
) -> None:
    fig, axis = plt.subplots(figsize=(7, 6))
    for loss_name in LOSS_ORDER:
        false_positive, true_positive, _ = roc_curve(labels, probabilities[loss_name])
        axis.plot(
            false_positive,
            true_positive,
            linewidth=2,
            label=f"{loss_name} (AUROC={roc_auc_score(labels, probabilities[loss_name]):.3f})",
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


def plot_pr_set(
    labels: np.ndarray, probabilities: dict[str, np.ndarray], title: str, path: Path
) -> None:
    fig, axis = plt.subplots(figsize=(7, 6))
    for loss_name in LOSS_ORDER:
        precision, recall, _ = precision_recall_curve(labels, probabilities[loss_name])
        axis.plot(
            recall,
            precision,
            linewidth=2,
            label=(
                f"{loss_name} (AUPRC="
                f"{average_precision_score(labels, probabilities[loss_name]):.3f})"
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


def plot_calibration_set(
    labels: np.ndarray, probabilities: dict[str, np.ndarray], path: Path
) -> None:
    fig, axis = plt.subplots(figsize=(7, 6))
    for loss_name in LOSS_ORDER:
        observed, predicted = calibration_curve(
            labels, probabilities[loss_name], n_bins=10, strategy="quantile"
        )
        axis.plot(predicted, observed, marker="o", label=loss_name)
    axis.plot([0, 1], [0, 1], "k--", linewidth=1)
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed event rate")
    axis.set_title("Valid calibration by loss")
    axis.legend(loc="best")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_training_history(history: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    for row, loss_name in enumerate(LOSS_ORDER):
        early = history[
            (history["loss"] == loss_name)
            & (history["stage"] == "inner_early_stopping")
        ]
        for fold in range(1, N_SPLITS + 1):
            fold_frame = early[early["fold"] == fold]
            axes[row, 0].plot(
                fold_frame["epoch"], fold_frame["train_loss"], label=f"Fold {fold}"
            )
            axes[row, 1].plot(
                fold_frame["epoch"], fold_frame["inner_auprc"], label=f"Fold {fold}"
            )
        axes[row, 0].set_title(f"{loss_name}: inner training loss")
        axes[row, 1].set_title(f"{loss_name}: inner validation AUPRC")
        axes[row, 0].set_ylabel("Loss")
        axes[row, 1].set_ylabel("AUPRC")
        for column in range(2):
            axes[row, column].set_xlabel("Epoch")
            axes[row, column].grid(alpha=0.25)
    axes[0, 1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    rows = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for _, row in frame.loc[:, columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append("NA" if math.isnan(float(value)) else f"{float(value):.6f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def generate_reports(
    train_metrics: pd.DataFrame,
    valid_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    paired: pd.DataFrame,
    fold_metadata: list[dict[str, Any]],
    thresholds: dict[str, float],
    started_at: datetime,
    ended_at: datetime,
) -> None:
    best_epochs = {
        loss_name: [
            int(item["best_epoch"])
            for item in fold_metadata
            if item["loss"] == loss_name
        ]
        for loss_name in LOSS_ORDER
    }
    focused_ci = bootstrap[bootstrap["metric"].isin(["auroc", "auprc", "brier_score"])]
    elapsed = (ended_at - started_at).total_seconds()
    report = f"""# Pre+Post MLP loss-function ablation

## Experiment status

- Label: **{EXPLORATORY_LABEL}**
- This is **{INDEPENDENCE_DISCLAIMER}**.
- Status: SUCCESS
- Architecture, folds, internal early-stopping splits, scalers, optimizer, initialization seeds, and batch order were identical across losses.
- Valid was not used for early stopping, epoch selection, or threshold selection.

## Losses

- `weighted_bce`: {LOSS_DEFINITIONS['weighted_bce']}
- `unweighted_bce`: {LOSS_DEFINITIONS['unweighted_bce']}
- `mse_brier`: {LOSS_DEFINITIONS['mse_brier']}

## Fold best epochs

`{json.dumps(best_epochs, ensure_ascii=False)}`

## Train pooled OOF metrics

{markdown_table(train_metrics, ['loss', 'threshold', 'auroc', 'auprc', 'brier_score', 'balanced_accuracy', 'f1', 'precision', 'sensitivity', 'specificity'])}

## Valid five-fold-mean metrics

{markdown_table(valid_metrics, ['loss', 'threshold', 'auroc', 'auprc', 'brier_score', 'balanced_accuracy', 'f1', 'precision', 'sensitivity', 'specificity'])}

## Valid 95% bootstrap confidence intervals

{markdown_table(focused_ci, ['loss', 'metric', 'estimate', 'ci_lower_2_5', 'ci_upper_97_5'])}

## Shared-index paired bootstrap comparisons

For AUROC/AUPRC, estimate = comparison minus reference. For Brier improvement, estimate = reference Brier minus comparison Brier. Positive values always favor the comparison loss.

{markdown_table(paired, ['comparison_loss', 'reference_loss', 'metric', 'estimate', 'ci_lower_2_5', 'ci_upper_97_5', 'ci_crosses_zero'])}

## Interpretation guardrail

This analysis was initiated after the original Valid results were observed. Any apparent loss-function advantage is exploratory and requires prospective or otherwise independent confirmation.

- Runtime: {elapsed:.3f} seconds
"""
    summary = f"""# Execution summary

- Experiment: `{EXPERIMENT}`
- Label: {EXPLORATORY_LABEL}
- Independence: {INDEPENDENCE_DISCLAIMER}
- Exit code: 0
- Started UTC: {started_at.isoformat()}
- Completed UTC: {ended_at.isoformat()}
- Runtime seconds: {elapsed:.3f}
- PID: {os.getpid()}
- Python: `{sys.executable}`
- GPU: `{torch.cuda.get_device_name(0)}`
- Train: 794 patients, 132 positive, 96 finite Pre+Post features
- Valid: 209 patients, 36 positive, 96 finite Pre+Post features
- Best epochs: `{json.dumps(best_epochs, ensure_ascii=False)}`
- Thresholds: `{json.dumps(thresholds, ensure_ascii=False)}`
- Valid used for epoch or threshold selection: No
"""
    write_text_sync(REPORT_DIR / "report.md", report, mode="x")
    write_text_sync(REPORT_DIR / "execution_summary.md", summary, mode="x")


def verify_outputs(protected_before: dict[str, Any], logger: logging.Logger) -> None:
    missing = [
        str(OUTPUT_DIR / relative)
        for relative in REQUIRED_OUTPUTS
        if not (OUTPUT_DIR / relative).is_file()
        or (OUTPUT_DIR / relative).stat().st_size <= 0
    ]
    for name in ["report.md", "execution_summary.md"]:
        path = REPORT_DIR / name
        if not path.is_file() or path.stat().st_size <= 0:
            missing.append(str(path))
    if missing:
        raise ExperimentError(f"Missing or empty required outputs: {missing}")
    protected_after = protected_state()
    if protected_after != protected_before:
        raise ExperimentError("Protected original experiment changed during ablation")
    flush_logger(logger)


def main() -> int:
    logger: logging.Logger | None = None
    started_at = utc_now()
    protected_before: dict[str, Any] = {}
    try:
        protected_before = preflight_new_paths()
        initialize_directories(started_at)
        logger = setup_logging(OUTPUT_DIR / "run.log")
        logger.info("%s started | pid=%d", EXPERIMENT, os.getpid())
        logger.info("Experiment label: %s", EXPLORATORY_LABEL)
        logger.info("Independence disclaimer: %s", INDEPENDENCE_DISCLAIMER)
        write_json_sync(
            OUTPUT_DIR / "protected_state.json",
            {"before": protected_before, "after": None},
        )
        write_json_sync(OUTPUT_DIR / "configuration.json", configuration_payload())

        input_hashes_before = {
            "train": sha256_file(TRAIN_PATH),
            "valid": sha256_file(VALID_PATH),
        }
        save_environment(OUTPUT_DIR / "environment.txt", started_at, input_hashes_before)
        train_df = pd.read_csv(TRAIN_PATH, dtype={"patient_id": "string"})
        valid_df = pd.read_csv(VALID_PATH, dtype={"patient_id": "string"})
        feature_names, train_ids, valid_ids = validate_inputs(train_df, valid_df)
        logger.info(
            "Input assertions passed | Train=794 positive=132 | Valid=209 positive=36 | "
            "Pre=48 Post=48 features=96 | overlap=0"
        )
        write_json_sync(
            OUTPUT_DIR / "feature_names.json",
            {
                "feature_count": len(feature_names),
                "pre_count": EXPECTED_PRE,
                "post_count": EXPECTED_POST,
                "feature_names": feature_names,
                "excluded": ["patient_id", "split", "adverse"],
            },
        )
        device = verify_gpu(OUTPUT_DIR / "gpu_verification.txt", logger)
        torch.cuda.reset_peak_memory_stats(device)

        x_train = train_df.loc[:, feature_names].apply(pd.to_numeric, errors="raise")
        x_valid = valid_df.loc[:, feature_names].apply(pd.to_numeric, errors="raise")
        y_train = pd.to_numeric(train_df["adverse"]).astype(int).to_numpy()
        y_valid = pd.to_numeric(valid_df["adverse"]).astype(int).to_numpy()
        outer_splits = list(
            StratifiedKFold(
                n_splits=N_SPLITS, shuffle=True, random_state=SEED
            ).split(x_train, y_train)
        )

        oof_probabilities = {
            loss_name: np.full(len(train_df), np.nan) for loss_name in LOSS_ORDER
        }
        oof_assignments = {
            loss_name: np.zeros(len(train_df), dtype=int) for loss_name in LOSS_ORDER
        }
        valid_fold_probabilities = {
            loss_name: np.full((len(valid_df), N_SPLITS), np.nan)
            for loss_name in LOSS_ORDER
        }
        outer_fold_assignment = np.zeros(len(train_df), dtype=int)
        history_rows: list[dict[str, Any]] = []
        fold_metadata: list[dict[str, Any]] = []
        fold_assignment_rows: list[dict[str, Any]] = []

        for fold, (development_index, holdout_index) in enumerate(outer_splits, start=1):
            inner_train_index, inner_early_index = train_test_split(
                development_index,
                test_size=INNER_VALID_FRACTION,
                random_state=SEED,
                shuffle=True,
                stratify=y_train[development_index],
            )
            outer_fold_assignment[holdout_index] = fold
            for index in development_index:
                role = "inner_train" if index in set(inner_train_index) else "inner_early_stopping"
                fold_assignment_rows.append(
                    {
                        "fold": fold,
                        "train_row_index": int(index),
                        "patient_id": train_ids.iloc[index],
                        "outer_role": "development",
                        "inner_role": role,
                    }
                )
            for index in holdout_index:
                fold_assignment_rows.append(
                    {
                        "fold": fold,
                        "train_row_index": int(index),
                        "patient_id": train_ids.iloc[index],
                        "outer_role": "holdout",
                        "inner_role": "not_applicable",
                    }
                )

            inner_scaler = StandardScaler()
            x_inner_train = inner_scaler.fit_transform(x_train.iloc[inner_train_index])
            x_inner_early = inner_scaler.transform(x_train.iloc[inner_early_index])
            outer_scaler = StandardScaler()
            x_development = outer_scaler.fit_transform(x_train.iloc[development_index])
            x_holdout = outer_scaler.transform(x_train.iloc[holdout_index])
            x_valid_fold = outer_scaler.transform(x_valid)
            joblib.dump(outer_scaler, SCALER_DIR / f"fold_{fold}_scaler.joblib")

            logger.info(
                "Outer fold %d/%d shared split ready | development=%d holdout=%d | "
                "inner_train=%d inner_early=%d",
                fold,
                N_SPLITS,
                len(development_index),
                len(holdout_index),
                len(inner_train_index),
                len(inner_early_index),
            )
            for loss_name in LOSS_ORDER:
                best_epoch, best_auprc, inner_pos_weight = early_stopping_train(
                    loss_name,
                    fold,
                    x_inner_train,
                    y_train[inner_train_index],
                    x_inner_early,
                    y_train[inner_early_index],
                    device,
                    history_rows,
                    logger,
                )
                model, outer_pos_weight = fixed_epoch_train(
                    loss_name,
                    fold,
                    x_development,
                    y_train[development_index],
                    best_epoch,
                    device,
                    history_rows,
                    logger,
                )
                oof_probabilities[loss_name][holdout_index] = predict(
                    model, x_holdout, device
                )
                valid_fold_probabilities[loss_name][:, fold - 1] = predict(
                    model, x_valid_fold, device
                )
                oof_assignments[loss_name][holdout_index] += 1
                torch.save(
                    {
                        "experiment": EXPERIMENT,
                        "experiment_label": EXPLORATORY_LABEL,
                        "loss": loss_name,
                        "loss_definition": LOSS_DEFINITIONS[loss_name],
                        "fold": fold,
                        "state_dict": {
                            key: value.detach().cpu()
                            for key, value in model.state_dict().items()
                        },
                        "feature_names": feature_names,
                        "architecture": NETWORK_DESCRIPTION,
                        "best_epoch": best_epoch,
                        "best_inner_auprc": best_auprc,
                        "inner_pos_weight": inner_pos_weight,
                        "outer_pos_weight": outer_pos_weight,
                        "initialization_seed_early": SEED + fold,
                        "initialization_seed_retrain": SEED + 1000 + fold,
                    },
                    MODEL_DIR / f"{loss_name}_fold_{fold}.pt",
                )
                fold_metadata.append(
                    {
                        "fold": fold,
                        "loss": loss_name,
                        "development_size": len(development_index),
                        "holdout_size": len(holdout_index),
                        "inner_train_size": len(inner_train_index),
                        "inner_early_size": len(inner_early_index),
                        "best_epoch": best_epoch,
                        "best_inner_auprc": best_auprc,
                        "inner_pos_weight": inner_pos_weight,
                        "outer_pos_weight": outer_pos_weight,
                        "valid_used_for_epoch_selection": False,
                    }
                )
                logger.info(
                    "Outer fold %d %s complete | best_epoch=%d | best_inner_AUPRC=%.6f",
                    fold,
                    loss_name,
                    best_epoch,
                    best_auprc,
                )
                del model
                torch.cuda.empty_cache()
                gc.collect()
            del inner_scaler, outer_scaler

        pd.DataFrame(fold_assignment_rows).to_csv(
            OUTPUT_DIR / "fold_assignments.csv", index=False
        )
        for loss_name in LOSS_ORDER:
            if not np.all(oof_assignments[loss_name] == 1):
                raise ExperimentError(f"{loss_name}: OOF assignment is not exactly once")
            if not np.isfinite(oof_probabilities[loss_name]).all():
                raise ExperimentError(f"{loss_name}: incomplete OOF probabilities")
            if not np.isfinite(valid_fold_probabilities[loss_name]).all():
                raise ExperimentError(f"{loss_name}: incomplete Valid fold probabilities")
        logger.info("All losses and folds complete; paired OOF/Valid probabilities are complete")

        history_frame = pd.DataFrame(history_rows)
        history_frame.to_csv(OUTPUT_DIR / "training_history.csv", index=False)
        thresholds: dict[str, float] = {}
        threshold_frames = []
        train_metric_records = {}
        for loss_name in LOSS_ORDER:
            threshold, search = select_threshold(
                loss_name, y_train, oof_probabilities[loss_name]
            )
            thresholds[loss_name] = threshold
            threshold_frames.append(search)
            train_metric_records[loss_name] = calculate_metrics(
                y_train, oof_probabilities[loss_name], threshold
            )
            logger.info(
                "%s Train OOF | threshold=%.12g | AUROC=%.6f AUPRC=%.6f Brier=%.6f",
                loss_name,
                threshold,
                train_metric_records[loss_name]["auroc"],
                train_metric_records[loss_name]["auprc"],
                train_metric_records[loss_name]["brier_score"],
            )
        write_json_sync(OUTPUT_DIR / "thresholds.json", thresholds)
        pd.concat(threshold_frames, ignore_index=True).to_csv(
            OUTPUT_DIR / "threshold_search.csv", index=False
        )

        train_predictions = pd.DataFrame(
            {
                "patient_id": train_ids,
                "split": train_df["split"],
                "adverse": y_train,
                "outer_fold": outer_fold_assignment,
            }
        )
        for loss_name in LOSS_ORDER:
            train_predictions[f"{loss_name}_probability"] = oof_probabilities[loss_name]
            train_predictions[f"{loss_name}_prediction"] = (
                oof_probabilities[loss_name] >= thresholds[loss_name]
            ).astype(int)
            train_predictions[f"{loss_name}_threshold"] = thresholds[loss_name]
        train_predictions.to_csv(OUTPUT_DIR / "train_oof_predictions.csv", index=False)
        train_metrics_frame = pd.DataFrame(
            [
                {
                    "dataset": "train_pooled_oof",
                    "loss": loss_name,
                    "threshold": thresholds[loss_name],
                    **train_metric_records[loss_name],
                }
                for loss_name in LOSS_ORDER
            ]
        )
        train_metrics_frame.to_csv(OUTPUT_DIR / "train_oof_metrics.csv", index=False)

        fold_rows = []
        metadata_lookup = {
            (int(item["fold"]), item["loss"]): item for item in fold_metadata
        }
        for fold, (_, holdout_index) in enumerate(outer_splits, start=1):
            for loss_name in LOSS_ORDER:
                row = {
                    "fold": fold,
                    "loss": loss_name,
                    "threshold": thresholds[loss_name],
                    **calculate_metrics(
                        y_train[holdout_index],
                        oof_probabilities[loss_name][holdout_index],
                        thresholds[loss_name],
                    ),
                }
                row.update(metadata_lookup[(fold, loss_name)])
                fold_rows.append(row)
        pd.DataFrame(fold_rows).to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)

        best_epochs = {
            loss_name: [
                int(item["best_epoch"])
                for item in fold_metadata
                if item["loss"] == loss_name
            ]
            for loss_name in LOSS_ORDER
        }
        frozen_at = utc_now()
        frozen_protocol = {
            "experiment": EXPERIMENT,
            "experiment_label": EXPLORATORY_LABEL,
            "independence_disclaimer": INDEPENDENCE_DISCLAIMER,
            "frozen_at_utc": frozen_at.isoformat(),
            "feature_names": feature_names,
            "network": NETWORK_DESCRIPTION,
            "loss_definitions": LOSS_DEFINITIONS,
            "best_epochs": best_epochs,
            "thresholds": thresholds,
            "threshold_source": "Train pooled OOF only",
            "common_outer_folds": True,
            "common_inner_splits": True,
            "common_scalers": True,
            "common_initialization_and_batch_seeds": True,
            "early_stopping_metric": "inner AUPRC for every loss",
            "valid_probability_rule": "five-fold arithmetic mean",
            "valid_used_for_epoch_selection": False,
            "valid_used_for_threshold_selection": False,
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "paired_comparisons": PAIRED_COMPARISONS,
        }
        write_json_sync(OUTPUT_DIR / "frozen_protocol.json", frozen_protocol)
        logger.info("Frozen protocol written before any Valid metric")

        valid_probabilities = {
            loss_name: valid_fold_probabilities[loss_name].mean(axis=1)
            for loss_name in LOSS_ORDER
        }
        valid_metric_records = {
            loss_name: calculate_metrics(
                y_valid, valid_probabilities[loss_name], thresholds[loss_name]
            )
            for loss_name in LOSS_ORDER
        }
        for loss_name in LOSS_ORDER:
            logger.info(
                "%s Valid | AUROC=%.6f AUPRC=%.6f Brier=%.6f BA=%.6f",
                loss_name,
                valid_metric_records[loss_name]["auroc"],
                valid_metric_records[loss_name]["auprc"],
                valid_metric_records[loss_name]["brier_score"],
                valid_metric_records[loss_name]["balanced_accuracy"],
            )

        valid_predictions = pd.DataFrame(
            {"patient_id": valid_ids, "split": valid_df["split"], "adverse": y_valid}
        )
        for loss_name in LOSS_ORDER:
            for fold in range(1, N_SPLITS + 1):
                valid_predictions[f"{loss_name}_fold_{fold}_probability"] = (
                    valid_fold_probabilities[loss_name][:, fold - 1]
                )
            valid_predictions[f"{loss_name}_probability"] = valid_probabilities[loss_name]
            valid_predictions[f"{loss_name}_prediction"] = (
                valid_probabilities[loss_name] >= thresholds[loss_name]
            ).astype(int)
            valid_predictions[f"{loss_name}_threshold"] = thresholds[loss_name]
        valid_predictions.to_csv(OUTPUT_DIR / "valid_predictions.csv", index=False)
        valid_metrics_frame = pd.DataFrame(
            [
                {
                    "dataset": "valid_frozen_five_fold_mean",
                    "loss": loss_name,
                    "threshold": thresholds[loss_name],
                    **valid_metric_records[loss_name],
                }
                for loss_name in LOSS_ORDER
            ]
        )
        valid_metrics_frame.to_csv(OUTPUT_DIR / "valid_metrics.csv", index=False)

        bootstrap_frame, paired_frame = bootstrap_analysis(
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

        plot_roc_set(
            y_train, oof_probabilities, "Train pooled OOF ROC", PLOT_DIR / "train_oof_roc.png"
        )
        plot_pr_set(
            y_train, oof_probabilities, "Train pooled OOF PR", PLOT_DIR / "train_oof_pr.png"
        )
        plot_roc_set(
            y_valid, valid_probabilities, "Valid ROC", PLOT_DIR / "valid_roc.png"
        )
        plot_pr_set(
            y_valid, valid_probabilities, "Valid PR", PLOT_DIR / "valid_pr.png"
        )
        plot_calibration_set(
            y_valid, valid_probabilities, PLOT_DIR / "valid_calibration.png"
        )
        plot_training_history(history_frame, PLOT_DIR / "training_curves.png")

        peak_memory = torch.cuda.max_memory_allocated(device)
        write_text_sync(
            OUTPUT_DIR / "gpu_verification.txt",
            f"formal_training_peak_memory_allocated_bytes={peak_memory}\n"
            f"formal_training_device={device}\n",
            mode="a",
        )
        input_hashes_after = {
            "train": sha256_file(TRAIN_PATH),
            "valid": sha256_file(VALID_PATH),
        }
        if input_hashes_after != input_hashes_before:
            raise ExperimentError("Input CSV hash changed during ablation")
        protected_after = protected_state()
        if protected_after != protected_before:
            raise ExperimentError("Protected original experiment changed during ablation")
        write_json_sync(
            OUTPUT_DIR / "protected_state.json",
            {"before": protected_before, "after": protected_after, "unchanged": True},
        )

        ended_at = utc_now()
        generate_reports(
            train_metrics_frame,
            valid_metrics_frame,
            bootstrap_frame,
            paired_frame,
            fold_metadata,
            thresholds,
            started_at,
            ended_at,
        )
        write_text_sync(OUTPUT_DIR / "exit_status.txt", "0\n")
        verify_outputs(protected_before, logger)
        logger.info(
            "All ablation artifacts verified | elapsed_seconds=%.3f | exit_code=0",
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

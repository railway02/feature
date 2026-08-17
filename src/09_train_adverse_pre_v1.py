#!/root/autodl-tmp/envs/aneurysm-ml/bin/python
"""Locked adverse_pre baseline experiment.

This implementation preserves the original 09_train_adverse_pre.py and runs a
strict Train-only development protocol. Valid is used only after model family,
hyperparameters, and thresholds have been frozen to disk.
"""


from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import os
import platform

import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PYTHON = Path("/root/autodl-tmp/envs/aneurysm-ml/bin/python")
TRAIN_PATH = PROJECT_ROOT / "outputs/task_datasets/adverse_pre_train.csv"
VALID_PATH = PROJECT_ROOT / "outputs/task_datasets/adverse_pre_valid.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs/baselines/adverse_pre_v1"
REPORT_DIR = PROJECT_ROOT / "reports/adverse_pre_v1"
MODEL_DIR = OUTPUT_DIR / "models"
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
    import scipy
    import sklearn
    from catboost import CatBoostClassifier, __version__ as catboost_version
    from catboost.utils import get_gpu_device_count
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
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import GridSearchCV, ParameterSampler, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    print(f"Dependency import failed under {sys.executable}: {exc}", file=sys.stderr)
    raise


SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 5
CATBOOST_CANDIDATE_COUNT = 12
BOOTSTRAP_REPEATS = 2000
MODEL_ORDER = ["Dummy", "Logistic", "CatBoost"]
PREDICTION_SLUG = {"Dummy": "dummy", "Logistic": "logistic", "CatBoost": "catboost"}
METRIC_COLUMNS = [
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

LOGISTIC_C_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]
LOGISTIC_CLASS_WEIGHTS = [None, "balanced"]
CATBOOST_PARAMETER_DISTRIBUTIONS = {
    "iterations": [300, 600, 1000],
    "depth": [3, 5, 7],
    "learning_rate": [0.02, 0.05, 0.1],
    "l2_leaf_reg": [3, 10],
    "auto_class_weights": [None, "Balanced"],
}


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Cannot JSON serialize {type(value).__name__}")


def write_text_sync(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def write_json_sync(path: Path, payload: Any) -> None:
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=False,
        default=json_default,
    )
    write_text_sync(path, text + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def setup_logging(path: Path) -> logging.Logger:
    logger = logging.getLogger("adverse_pre_v1")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)sZ | %(levelname)s | %(message)s", "%Y-%m-%dT%H:%M:%S"
    )
    formatter.converter = time.gmtime
    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
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


def catboost_sampled_candidates() -> list[dict[str, Any]]:
    candidates = list(
        ParameterSampler(
            CATBOOST_PARAMETER_DISTRIBUTIONS,
            n_iter=CATBOOST_CANDIDATE_COUNT,
            random_state=SEED,
        )
    )
    assert len(candidates) == CATBOOST_CANDIDATE_COUNT
    signatures = {json.dumps(item, sort_keys=True, default=json_default) for item in candidates}
    assert len(signatures) == CATBOOST_CANDIDATE_COUNT, "CatBoost candidates are not unique"
    return candidates


CATBOOST_CANDIDATES = catboost_sampled_candidates()
LOGISTIC_PARAM_GRIDS = [
    {"model__C": [c_value], "model__class_weight": [class_weight]}
    for c_value in LOGISTIC_C_VALUES
    for class_weight in LOGISTIC_CLASS_WEIGHTS
]
CATBOOST_PARAM_GRIDS = [
    {
        f"model__{key}": [value]
        for key, value in candidate.items()
        if not (key == "auto_class_weights" and value is None)
    }
    for candidate in CATBOOST_CANDIDATES
]

CONFIGURATION: dict[str, Any] = {
    "experiment": "adverse_pre_v1",
    "script": str(Path(__file__).resolve()),
    "preserved_original_script": str(PROJECT_ROOT / "code/09_train_adverse_pre.py"),
    "python_interpreter": str(EXPECTED_PYTHON),
    "random_seed": SEED,
    "inputs": {"train": str(TRAIN_PATH), "valid": str(VALID_PATH)},
    "expected_data": {
        "train_rows": 794,
        "train_positive": 132,
        "valid_rows": 209,
        "valid_positive": 36,
        "feature_count": 48,
    },
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
        "scoring": "average_precision",
    },
    "model_selection": [
        "Train pooled OOF AUPRC descending",
        "Train pooled OOF AUROC descending",
        "Train pooled OOF Brier Score ascending",
    ],
    "threshold_selection": (
        "Train pooled OOF balanced accuracy maximum; ties closest to 0.5; "
        "remaining ties choose lower threshold"
    ),
    "dummy": {"strategy": "prior"},
    "logistic": {
        "pipeline": ["SimpleImputer(median)", "StandardScaler", "LogisticRegression"],
        "fixed": {
            "penalty": "l2",
            "solver": "liblinear",
            "max_iter": 5000,
            "random_state": SEED,
        },
        "grid": {
            "C": LOGISTIC_C_VALUES,
            "class_weight": LOGISTIC_CLASS_WEIGHTS,
        },
    },
    "catboost": {
        "pipeline": ["SimpleImputer(median)", "CatBoostClassifier"],
        "fixed": {
            "task_type": "GPU",
            "devices": "0",
            "random_seed": SEED,
            "loss_function": "Logloss",
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": 8,
        },
        "parameter_distributions": CATBOOST_PARAMETER_DISTRIBUTIONS,
        "parameter_sampler_n_iter": CATBOOST_CANDIDATE_COUNT,
        "parameter_sampler_random_state": SEED,
        "sampled_candidates": CATBOOST_CANDIDATES,
        "search_n_jobs": 1,
        "valid_eval_set": False,
        "early_stopping": False,
        "cpu_fallback": False,
    },
    "bootstrap": {
        "valid_patient_repeats": BOOTSTRAP_REPEATS,
        "random_seed": SEED,
        "shared_indices_across_models": True,
        "skip_single_class": True,
        "retrain": False,
        "thresholds": "frozen Train OOF thresholds",
    },
}


def initialize_run_directories() -> None:
    if SUCCESS_PATH.exists():
        raise FileExistsError(f"Existing success marker; refusing overwrite: {SUCCESS_PATH}")
    if OUTPUT_DIR.exists():
        existing = list(OUTPUT_DIR.iterdir())
        raise FileExistsError(
            f"Output directory already exists; refusing overwrite ({len(existing)} entries): {OUTPUT_DIR}"
        )
    if REPORT_DIR.exists():
        existing = list(REPORT_DIR.iterdir())
        raise FileExistsError(
            f"Report directory already exists; refusing overwrite ({len(existing)} entries): {REPORT_DIR}"
        )
    MODEL_DIR.mkdir(parents=True, exist_ok=False)
    PLOT_DIR.mkdir(parents=True, exist_ok=False)
    REPORT_DIR.mkdir(parents=True, exist_ok=False)
    write_text_sync(
        RUNNING_PATH,
        f"status=RUNNING\nstarted_at_utc={utc_now().isoformat()}\npid={os.getpid()}\n",
    )


def save_environment(path: Path, started_at: datetime) -> None:
    lines = [
        "experiment=adverse_pre_v1",
        f"generated_at_utc={utc_now().isoformat()}",
        f"run_started_at_utc={started_at.isoformat()}",
        f"sys.executable={sys.executable}",
        f"required_interpreter={EXPECTED_PYTHON}",
        f"python_version={platform.python_version()}",
        f"python_implementation={platform.python_implementation()}",
        f"platform={platform.platform()}",
        f"numpy={np.__version__}",
        f"pandas={pd.__version__}",
        f"scipy={scipy.__version__}",
        f"scikit_learn={sklearn.__version__}",
        f"joblib={joblib.__version__}",
        f"matplotlib={matplotlib.__version__}",
        f"catboost={catboost_version}",
        "catboost_task_type=GPU",
        "catboost_devices=0",
        "catboost_thread_count=8",
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '<unset>')}",
        f"random_seed={SEED}",
        f"command={sys.executable} {Path(__file__).resolve()}",
    ]
    write_text_sync(path, "\n".join(lines) + "\n")


def verify_gpu(path: Path, logger: logging.Logger) -> None:
    lines = [
        f"verified_at_utc={utc_now().isoformat()}",
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
        f"catboost_version={catboost_version}",
        "task_type=GPU",
        "devices=0",
        "allow_writing_files=False",
        "cpu_fallback=False",
    ]
    try:
        command = subprocess.run(
            ["nvidia-smi"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        lines.extend(
            [
                f"nvidia_smi_returncode={command.returncode}",
                "--- nvidia-smi stdout ---",
                command.stdout.rstrip(),
                "--- nvidia-smi stderr ---",
                command.stderr.rstrip(),
            ]
        )
        if command.returncode != 0:
            raise RuntimeError(f"nvidia-smi failed with return code {command.returncode}")

        gpu_count = int(get_gpu_device_count())
        lines.append(f"get_gpu_device_count()={gpu_count}")
        if gpu_count < 1:
            raise RuntimeError("CatBoost reported no GPU devices")

        test_x = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.2, 0.8],
                [0.8, 0.2],
                [0.1, 0.1],
                [0.9, 0.9],
            ],
            dtype=float,
        )
        test_y = np.array([0, 0, 0, 1, 0, 1, 0, 1], dtype=int)
        test_model = CatBoostClassifier(
            iterations=5,
            depth=2,
            learning_rate=0.1,
            task_type="GPU",
            devices="0",
            random_seed=SEED,
            loss_function="Logloss",
            verbose=False,
            allow_writing_files=False,
            thread_count=8,
        )
        test_model.fit(test_x, test_y)
        test_probabilities = test_model.predict_proba(test_x)[:, 1]
        if not np.isfinite(test_probabilities).all():
            raise RuntimeError("Minimal CatBoost GPU test returned non-finite probabilities")
        lines.extend(
            [
                "minimal_catboost_gpu_training_test=PASS",
                f"minimal_test_iterations={test_model.tree_count_}",
                "minimal_test_probabilities="
                + ",".join(f"{value:.8f}" for value in test_probabilities),
            ]
        )
        write_text_sync(path, "\n".join(lines) + "\n")
        logger.info("GPU verification passed | devices=%d | minimal CatBoost test passed", gpu_count)
        del test_model
        gc.collect()
    except Exception as exc:
        lines.extend(
            [
                "minimal_catboost_gpu_training_test=FAIL",
                f"error_type={type(exc).__name__}",
                f"error={exc}",
                traceback.format_exc(),
            ]
        )
        write_text_sync(path, "\n".join(lines) + "\n")
        raise RuntimeError("CatBoost GPU verification failed; CPU fallback is disabled") from exc


def validate_inputs(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> list[str]:
    required = ["patient_id", "split", "adverse"]
    assert all(column in train_df.columns for column in required), "Train metadata columns missing"
    assert all(column in valid_df.columns for column in required), "Valid metadata columns missing"
    assert len(train_df) == 794, f"Train rows={len(train_df)}, expected 794"
    assert len(valid_df) == 209, f"Valid rows={len(valid_df)}, expected 209"

    assert train_df["patient_id"].notna().all(), "Train patient_id contains missing values"
    assert valid_df["patient_id"].notna().all(), "Valid patient_id contains missing values"
    assert train_df["patient_id"].is_unique, "Train patient_id is not unique"
    assert valid_df["patient_id"].is_unique, "Valid patient_id is not unique"
    overlap = set(train_df["patient_id"]).intersection(set(valid_df["patient_id"]))
    assert len(overlap) == 0, f"Train/Valid patient overlap={len(overlap)}, expected 0"

    assert set(train_df["split"].astype(str)) == {"train"}, "Train split values are invalid"
    assert set(valid_df["split"].astype(str)) == {"valid"}, "Valid split values are invalid"
    assert train_df["adverse"].notna().all(), "Train labels contain missing values"
    assert valid_df["adverse"].notna().all(), "Valid labels contain missing values"
    train_y_float = train_df["adverse"].to_numpy(dtype=float)
    valid_y_float = valid_df["adverse"].to_numpy(dtype=float)
    assert np.array_equal(train_y_float, train_y_float.astype(int)), "Train labels are not integer-valued"
    assert np.array_equal(valid_y_float, valid_y_float.astype(int)), "Valid labels are not integer-valued"
    assert set(train_y_float.astype(int)) == {0, 1}, "Train labels are not exactly {0,1}"
    assert set(valid_y_float.astype(int)) == {0, 1}, "Valid labels are not exactly {0,1}"
    assert int(train_y_float.sum()) == 132, f"Train positives={int(train_y_float.sum())}, expected 132"
    assert int(valid_y_float.sum()) == 36, f"Valid positives={int(valid_y_float.sum())}, expected 36"

    excluded = set(required)
    train_features = [column for column in train_df.columns if column not in excluded]
    valid_features = [column for column in valid_df.columns if column not in excluded]
    assert train_features == valid_features, "Train/Valid feature names or order differ"
    assert len(train_features) == 48, f"Feature count={len(train_features)}, expected 48"
    assert excluded.isdisjoint(train_features), "patient_id, split, or adverse entered model features"
    assert all(column.startswith("pre_") for column in train_features), (
        "Every model feature must be a pre_ preoperative feature"
    )
    forbidden_tokens = ["post_", "delta_", "runtime_s", "n_pairs"]
    forbidden = [
        column
        for column in train_features
        if any(token in column.lower() for token in forbidden_tokens)
    ]
    assert not forbidden, f"Forbidden feature names found: {forbidden}"

    train_non_numeric = [
        column for column in train_features if not pd.api.types.is_numeric_dtype(train_df[column])
    ]
    valid_non_numeric = [
        column for column in valid_features if not pd.api.types.is_numeric_dtype(valid_df[column])
    ]
    assert not train_non_numeric, f"Non-numeric Train features: {train_non_numeric}"
    assert not valid_non_numeric, f"Non-numeric Valid features: {valid_non_numeric}"
    assert not np.isinf(train_df[train_features].to_numpy(dtype=float)).any(), (
        "Train numeric features contain positive/negative infinity"
    )
    assert not np.isinf(valid_df[valid_features].to_numpy(dtype=float)).any(), (
        "Valid numeric features contain positive/negative infinity"
    )
    return train_features


def write_data_validation_report(
    path: Path,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_names: list[str],
    input_hashes: dict[str, str],
) -> None:
    text = f"""# adverse_pre_v1 数据验证

- 结论：PASS
- Train：{len(train_df)} 行，正样本 {int(train_df['adverse'].sum())}
- Valid：{len(valid_df)} 行，正样本 {int(valid_df['adverse'].sum())}
- Train/Valid 患者交集：0
- patient_id：两个集合内均唯一
- 标签：仅包含 0 和 1
- 模型特征：恰好 {len(feature_names)} 个，Train/Valid 名称与顺序完全一致
- 特征范围：全部以 `pre_` 开头；无 `post_`、`delta_`、`runtime_s`、`n_pairs`
- 结构列：`patient_id`、`split`、`adverse` 未进入模型
- 数值质量：未发现正无穷或负无穷；缺失值仅允许由管道内中位数插补处理
- 数据处理：输入 CSV 只读加载，未修改、覆盖或重建
- Train SHA256：`{input_hashes['train']}`
- Valid SHA256：`{input_hashes['valid']}`

## 特征（固定顺序）

"""
    text += "\n".join(f"{index}. `{name}`" for index, name in enumerate(feature_names, 1))
    text += "\n"
    write_text_sync(path, text)


def build_estimator(model_name: str) -> Any:
    if model_name == "Dummy":
        return DummyClassifier(strategy="prior")
    if model_name == "Logistic":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty="l2",
                        solver="liblinear",
                        max_iter=5000,
                        random_state=SEED,
                    ),
                ),
            ]
        )
    if model_name == "CatBoost":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    CatBoostClassifier(
                        task_type="GPU",
                        devices="0",
                        random_seed=SEED,
                        loss_function="Logloss",
                        verbose=False,
                        allow_writing_files=False,
                        thread_count=8,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown model: {model_name}")


def param_grids(model_name: str) -> list[dict[str, list[Any]]]:
    if model_name == "Logistic":
        return LOGISTIC_PARAM_GRIDS
    if model_name == "CatBoost":
        return CATBOOST_PARAM_GRIDS
    raise ValueError(f"No search grid for {model_name}")


def scalar_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in params.items()}


def unprefix_model_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        key.removeprefix("model__"): value
        for key, value in params.items()
    }


def reported_model_params(model_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Restore protocol-level None when CatBoost omits the corresponding option."""
    reported = unprefix_model_params(params)
    if model_name == "CatBoost":
        reported.setdefault("auto_class_weights", None)
    return reported


def append_search_results(
    rows: list[dict[str, Any]],
    search: GridSearchCV,
    model_name: str,
    stage: str,
    outer_fold: int | None,
) -> None:
    results = search.cv_results_
    actual_params = [scalar_params(item) for item in results["params"]]
    if model_name == "CatBoost":
        expected_params = [
            {
                key: values[0]
                for key, values in candidate.items()
                if not (key == "model__auto_class_weights" and values[0] is None)
            }
            for candidate in CATBOOST_PARAM_GRIDS
        ]
        assert actual_params == expected_params, (
            "CatBoost candidate set/order changed across search calls"
        )
    for index, params in enumerate(actual_params):
        rows.append(
            {
                "model": model_name,
                "stage": stage,
                "outer_fold": outer_fold,
                "candidate_id": index + 1,
                "params_json": json.dumps(
                    reported_model_params(model_name, params),
                    ensure_ascii=False,
                    sort_keys=True,
                    default=json_default,
                ),
                "mean_test_average_precision": float(results["mean_test_score"][index]),
                "std_test_average_precision": float(results["std_test_score"][index]),
                "rank_test_average_precision": int(results["rank_test_score"][index]),
                "mean_fit_time_s": float(results["mean_fit_time"][index]),
                "std_fit_time_s": float(results["std_fit_time"][index]),
                "mean_score_time_s": float(results["mean_score_time"][index]),
                "std_score_time_s": float(results["std_score_time"][index]),
                "search_n_jobs": 1,
                "inner_folds": INNER_FOLDS,
                "inner_shuffle": True,
                "inner_random_state": SEED,
                "scoring": "average_precision",
            }
        )


def run_inner_search(
    model_name: str,
    x: pd.DataFrame,
    y: np.ndarray,
    stage: str,
    outer_fold: int | None,
    cv_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], float]:
    search = GridSearchCV(
        estimator=build_estimator(model_name),
        param_grid=param_grids(model_name),
        scoring="average_precision",
        cv=StratifiedKFold(
            n_splits=INNER_FOLDS,
            shuffle=True,
            random_state=SEED,
        ),
        refit=False,
        n_jobs=1,
        error_score="raise",
        return_train_score=False,
    )
    search.fit(x, y)
    append_search_results(cv_rows, search, model_name, stage, outer_fold)
    best_params = scalar_params(search.cv_results_["params"][search.best_index_])
    best_score = float(search.best_score_)
    del search
    gc.collect()
    return best_params, best_score


def safe_auroc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return math.nan
    return float(roc_auc_score(y_true, probabilities))


def safe_auprc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return math.nan
    return float(average_precision_score(y_true, probabilities))


def calculate_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int | str]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if tp + fn else math.nan
    specificity = float(tn / (tn + fp)) if tn + fp else math.nan
    return {
        "auroc": safe_auroc(y_true, probabilities),
        "auprc": safe_auprc(y_true, probabilities),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "confusion_matrix": json.dumps([[int(tn), int(fp)], [int(fn), int(tp)]]),
    }


def run_nested_oof(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    outer_splits: list[tuple[np.ndarray, np.ndarray]],
    cv_rows: list[dict[str, Any]],
    logger: logging.Logger,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    probabilities = np.full(len(y_train), np.nan, dtype=float)
    assignment_count = np.zeros(len(y_train), dtype=int)
    fold_search: list[dict[str, Any]] = []
    for fold, (development_index, holdout_index) in enumerate(outer_splits, start=1):
        if model_name == "Dummy":
            best_params: dict[str, Any] = {"strategy": "prior"}
            best_score = math.nan
            fitted = build_estimator(model_name)
        else:
            best_params, best_score = run_inner_search(
                model_name,
                x_train.iloc[development_index],
                y_train[development_index],
                "outer_inner_search",
                fold,
                cv_rows,
            )
            fitted = clone(build_estimator(model_name)).set_params(**best_params)
        fitted.fit(x_train.iloc[development_index], y_train[development_index])
        fold_probabilities = fitted.predict_proba(x_train.iloc[holdout_index])[:, 1]
        assert np.isfinite(fold_probabilities).all(), (
            f"{model_name} outer fold {fold} produced non-finite OOF probabilities"
        )
        probabilities[holdout_index] = fold_probabilities
        assignment_count[holdout_index] += 1
        fold_search.append(
            {
                "fold": fold,
                "best_params": best_params,
                "inner_best_average_precision": best_score,
            }
        )
        logger.info(
            "%s outer fold %d/%d complete | development=%d positive=%d | "
            "holdout=%d positive=%d | inner_best_AP=%s | params=%s",
            model_name,
            fold,
            OUTER_FOLDS,
            len(development_index),
            int(y_train[development_index].sum()),
            len(holdout_index),
            int(y_train[holdout_index].sum()),
            "NA" if math.isnan(best_score) else f"{best_score:.6f}",
            reported_model_params(model_name, best_params),
        )
        del fitted
        gc.collect()
    assert np.all(assignment_count == 1), (
        f"{model_name} OOF assignment counts are not exactly one per Train patient"
    )
    assert np.isfinite(probabilities).all(), f"{model_name} OOF probabilities are incomplete"
    return probabilities, fold_search


def select_threshold(
    model_name: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    unique_probabilities = np.unique(np.asarray(probabilities, dtype=float))
    just_above = np.nextafter(unique_probabilities, np.inf)
    just_above = just_above[(just_above >= 0.0) & (just_above <= 1.0)]
    candidates = np.unique(
        np.concatenate(
            [
                np.array([0.0, 0.5, 1.0], dtype=float),
                unique_probabilities,
                just_above,
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
                "tp": metrics["tp"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
            }
        )
    frame = pd.DataFrame(rows)
    best_value = float(frame["balanced_accuracy"].max())
    tied = frame[
        np.isclose(frame["balanced_accuracy"], best_value, rtol=0.0, atol=1e-12)
    ].copy()
    tied["distance_to_0_5"] = (tied["threshold"] - 0.5).abs()
    tied = tied.sort_values(["distance_to_0_5", "threshold"], ascending=[True, True])
    threshold = float(tied.iloc[0]["threshold"])
    frame["distance_to_0_5"] = (frame["threshold"] - 0.5).abs()
    frame["selected"] = np.isclose(frame["threshold"], threshold, rtol=0.0, atol=0.0)
    return threshold, frame


def model_ranking(
    train_metrics: dict[str, dict[str, float | int | str]]
) -> list[str]:
    return sorted(
        MODEL_ORDER,
        key=lambda name: (
            -float(train_metrics[name]["auprc"]),
            -float(train_metrics[name]["auroc"]),
            float(train_metrics[name]["brier_score"]),
            MODEL_ORDER.index(name),
        ),
    )


def fit_final_models(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    cv_rows: list[dict[str, Any]],
    logger: logging.Logger,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, float | None]]:
    models: dict[str, Any] = {}
    selected_params: dict[str, dict[str, Any]] = {}
    inner_scores: dict[str, float | None] = {}

    dummy = build_estimator("Dummy")
    dummy.fit(x_train, y_train)
    models["Dummy"] = dummy
    selected_params["Dummy"] = {"strategy": "prior"}
    inner_scores["Dummy"] = None
    logger.info("Dummy fitted once on full Train | strategy=prior")

    for model_name in ["Logistic", "CatBoost"]:
        best_params, best_score = run_inner_search(
            model_name,
            x_train,
            y_train,
            "final_full_train_inner_search",
            None,
            cv_rows,
        )
        fitted = clone(build_estimator(model_name)).set_params(**best_params)
        fitted.fit(x_train, y_train)
        models[model_name] = fitted
        selected_params[model_name] = reported_model_params(model_name, best_params)
        inner_scores[model_name] = best_score
        logger.info(
            "%s final Train-only search and refit complete | AP=%.6f | params=%s",
            model_name,
            best_score,
            selected_params[model_name],
        )
    return models, selected_params, inner_scores


def complete_hyperparameters(selected_params: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "Dummy": {"strategy": "prior"},
        "Logistic": {
            "pipeline": {
                "imputer": {"strategy": "median"},
                "scaler": "StandardScaler",
                "model": {
                    "penalty": "l2",
                    "solver": "liblinear",
                    "max_iter": 5000,
                    "random_state": SEED,
                    **selected_params["Logistic"],
                },
            }
        },
        "CatBoost": {
            "pipeline": {
                "imputer": {"strategy": "median"},
                "model": {
                    "task_type": "GPU",
                    "devices": "0",
                    "random_seed": SEED,
                    "loss_function": "Logloss",
                    "verbose": False,
                    "allow_writing_files": False,
                    "thread_count": 8,
                    **selected_params["CatBoost"],
                },
            }
        },
    }


def build_outer_fold_metrics(
    y_train: np.ndarray,
    oof_probabilities: dict[str, np.ndarray],
    thresholds: dict[str, float],
    outer_splits: list[tuple[np.ndarray, np.ndarray]],
    fold_search: dict[str, list[dict[str, Any]]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        for fold, (development_index, holdout_index) in enumerate(outer_splits, start=1):
            metrics = calculate_metrics(
                y_train[holdout_index],
                oof_probabilities[model_name][holdout_index],
                thresholds[model_name],
            )
            search_info = fold_search[model_name][fold - 1]
            rows.append(
                {
                    "model": model_name,
                    "outer_fold": fold,
                    "development_n": len(development_index),
                    "development_positive": int(y_train[development_index].sum()),
                    "holdout_n": len(holdout_index),
                    "holdout_positive": int(y_train[holdout_index].sum()),
                    "frozen_train_oof_threshold": thresholds[model_name],
                    "inner_best_average_precision": search_info["inner_best_average_precision"],
                    "best_params_json": json.dumps(
                        reported_model_params(model_name, search_info["best_params"]),
                        ensure_ascii=False,
                        sort_keys=True,
                        default=json_default,
                    ),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def create_bootstrap_indices(y_valid: np.ndarray) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(SEED)
    accepted: list[np.ndarray] = []
    attempts = 0
    while len(accepted) < BOOTSTRAP_REPEATS:
        attempts += 1
        indices = rng.integers(0, len(y_valid), size=len(y_valid))
        if np.unique(y_valid[indices]).size < 2:
            continue
        accepted.append(indices)
    return np.stack(accepted), attempts


def bootstrap_results(
    y_valid: np.ndarray,
    valid_probabilities: dict[str, np.ndarray],
    thresholds: dict[str, float],
    valid_metrics: dict[str, dict[str, float | int | str]],
    selected_model: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    indices_matrix, attempts = create_bootstrap_indices(y_valid)
    distributions: dict[str, dict[str, list[float]]] = {
        model_name: {metric: [] for metric in METRIC_COLUMNS}
        for model_name in MODEL_ORDER
    }
    paired = {"auroc_improvement": [], "auprc_improvement": [], "brier_improvement": []}

    for indices in indices_matrix:
        sampled_y = y_valid[indices]
        repeat_metrics: dict[str, dict[str, float | int | str]] = {}
        for model_name in MODEL_ORDER:
            metrics = calculate_metrics(
                sampled_y,
                valid_probabilities[model_name][indices],
                thresholds[model_name],
            )
            repeat_metrics[model_name] = metrics
            for metric in METRIC_COLUMNS:
                distributions[model_name][metric].append(float(metrics[metric]))
        paired["auroc_improvement"].append(
            float(repeat_metrics[selected_model]["auroc"])
            - float(repeat_metrics["Dummy"]["auroc"])
        )
        paired["auprc_improvement"].append(
            float(repeat_metrics[selected_model]["auprc"])
            - float(repeat_metrics["Dummy"]["auprc"])
        )
        paired["brier_improvement"].append(
            float(repeat_metrics["Dummy"]["brier_score"])
            - float(repeat_metrics[selected_model]["brier_score"])
        )

    ci_rows: list[dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        for metric in METRIC_COLUMNS:
            values = np.asarray(distributions[model_name][metric], dtype=float)
            ci_rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "estimate": float(valid_metrics[model_name][metric]),
                    "ci_lower_2_5": float(np.percentile(values, 2.5)),
                    "ci_upper_97_5": float(np.percentile(values, 97.5)),
                    "valid_repeats": BOOTSTRAP_REPEATS,
                    "sampling_attempts": attempts,
                    "skipped_single_class": attempts - BOOTSTRAP_REPEATS,
                    "random_seed": SEED,
                    "shared_indices_across_models": True,
                    "frozen_threshold": thresholds[model_name],
                }
            )

    point_differences = {
        "auroc_improvement": float(valid_metrics[selected_model]["auroc"])
        - float(valid_metrics["Dummy"]["auroc"]),
        "auprc_improvement": float(valid_metrics[selected_model]["auprc"])
        - float(valid_metrics["Dummy"]["auprc"]),
        "brier_improvement": float(valid_metrics["Dummy"]["brier_score"])
        - float(valid_metrics[selected_model]["brier_score"]),
    }
    paired_rows: list[dict[str, Any]] = []
    for metric, values_list in paired.items():
        values = np.asarray(values_list, dtype=float)
        lower = float(np.percentile(values, 2.5))
        upper = float(np.percentile(values, 97.5))
        excludes_zero = bool(lower > 0.0 or upper < 0.0)
        if lower > 0.0:
            evidence = "CI不跨0且为正：入选模型优于Dummy的差异证据较明确"
        elif upper < 0.0:
            evidence = "CI不跨0且为负：Dummy优于入选模型的差异证据较明确"
        else:
            evidence = "CI跨0：不能表述为较明确的差异证据"
        paired_rows.append(
            {
                "selected_model": selected_model,
                "reference_model": "Dummy",
                "metric": metric,
                "positive_means_selected_better": True,
                "estimate": point_differences[metric],
                "ci_lower_2_5": lower,
                "ci_upper_97_5": upper,
                "ci_excludes_zero": excludes_zero,
                "evidence_statement": evidence,
                "valid_repeats": BOOTSTRAP_REPEATS,
                "sampling_attempts": attempts,
                "random_seed": SEED,
                "shared_indices": True,
            }
        )
    logger.info(
        "Valid bootstrap complete | effective=%d | attempts=%d | skipped_single_class=%d",
        BOOTSTRAP_REPEATS,
        attempts,
        attempts - BOOTSTRAP_REPEATS,
    )
    return pd.DataFrame(ci_rows), pd.DataFrame(paired_rows)


def plot_roc(
    y_true: np.ndarray,
    probability_map: dict[str, np.ndarray],
    title: str,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    for model_name in MODEL_ORDER:
        fpr, tpr, _ = roc_curve(y_true, probability_map[model_name])
        axis.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{model_name} (AUROC={safe_auroc(y_true, probability_map[model_name]):.3f})",
        )
    axis.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title=title)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_pr(
    y_true: np.ndarray,
    probability_map: dict[str, np.ndarray],
    title: str,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    for model_name in MODEL_ORDER:
        precision, recall, _ = precision_recall_curve(y_true, probability_map[model_name])
        axis.plot(
            recall,
            precision,
            linewidth=2,
            label=f"{model_name} (AUPRC={safe_auprc(y_true, probability_map[model_name]):.3f})",
        )
    prevalence = float(np.mean(y_true))
    axis.axhline(prevalence, linestyle="--", color="gray", label=f"Prevalence={prevalence:.3f}")
    axis.set(xlabel="Recall", ylabel="Precision", title=title)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_valid_calibration(
    y_valid: np.ndarray,
    valid_probabilities: dict[str, np.ndarray],
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    for model_name in MODEL_ORDER:
        observed, predicted = calibration_curve(
            y_valid,
            valid_probabilities[model_name],
            n_bins=10,
            strategy="quantile",
        )
        axis.plot(predicted, observed, marker="o", linewidth=1.5, label=model_name)
    axis.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axis.set(
        xlabel="Mean predicted probability",
        ylabel="Observed event fraction",
        title="Valid calibration (frozen models)",
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_confusion(
    metrics: dict[str, float | int | str], model_name: str, threshold: float, path: Path
) -> None:
    matrix = np.array(
        [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]], dtype=int
    )
    figure, axis = plt.subplots(figsize=(5, 4.5))
    axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=14)
    axis.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    axis.set_yticks([0, 1], labels=["True 0", "True 1"])
    axis.set_title(f"Valid confusion: {model_name}\nFrozen threshold={threshold:.6g}")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_interpretation(
    final_models: dict[str, Any], feature_names: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logistic_coefficients = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": final_models["Logistic"].named_steps["model"].coef_[0],
        }
    )
    logistic_coefficients["absolute_coefficient"] = logistic_coefficients["coefficient"].abs()
    logistic_coefficients = logistic_coefficients.sort_values(
        ["absolute_coefficient", "feature"], ascending=[False, True]
    )
    logistic_coefficients.to_csv(
        OUTPUT_DIR / "logistic_coefficients.csv", index=False, encoding="utf-8"
    )

    catboost_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": final_models["CatBoost"].named_steps["model"].get_feature_importance(),
        }
    ).sort_values(["importance", "feature"], ascending=[False, True])
    catboost_importance.to_csv(
        OUTPUT_DIR / "catboost_feature_importance.csv", index=False, encoding="utf-8"
    )
    return logistic_coefficients, catboost_importance


def plot_feature_tables(
    logistic_coefficients: pd.DataFrame, catboost_importance: pd.DataFrame
) -> None:
    logistic_top = logistic_coefficients.head(20).sort_values("coefficient")
    figure, axis = plt.subplots(figsize=(9, 7))
    colors = ["#b2182b" if value < 0 else "#2166ac" for value in logistic_top["coefficient"]]
    axis.barh(logistic_top["feature"], logistic_top["coefficient"], color=colors)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_title("Logistic coefficients (top 20 by absolute value)")
    axis.set_xlabel("Standardized coefficient")
    figure.tight_layout()
    figure.savefig(PLOT_DIR / "logistic_coefficients.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    cat_top = catboost_importance.head(20).sort_values("importance")
    figure, axis = plt.subplots(figsize=(9, 7))
    axis.barh(cat_top["feature"], cat_top["importance"], color="#2c7fb8")
    axis.set_title("CatBoost feature importance (top 20)")
    axis.set_xlabel("Feature importance")
    figure.tight_layout()
    figure.savefig(PLOT_DIR / "catboost_feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def markdown_metrics(frame: pd.DataFrame) -> str:
    headers = [
        "Model", "AUROC", "AUPRC", "Bal.Acc", "F1", "Precision",
        "Sensitivity", "Specificity", "Brier", "TP/TN/FP/FN",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for model_name in MODEL_ORDER:
        row = frame[frame["model"] == model_name].iloc[0]
        lines.append(
            "| "
            + " | ".join(
                [
                    model_name,
                    f"{row['auroc']:.4f}",
                    f"{row['auprc']:.4f}",
                    f"{row['balanced_accuracy']:.4f}",
                    f"{row['f1']:.4f}",
                    f"{row['precision']:.4f}",
                    f"{row['sensitivity']:.4f}",
                    f"{row['specificity']:.4f}",
                    f"{row['brier_score']:.4f}",
                    f"{int(row['tp'])}/{int(row['tn'])}/{int(row['fp'])}/{int(row['fn'])}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def generate_reports(
    train_metrics_frame: pd.DataFrame,
    valid_metrics_frame: pd.DataFrame,
    bootstrap_frame: pd.DataFrame,
    paired_frame: pd.DataFrame,
    ranking: list[str],
    thresholds: dict[str, float],
    hyperparameters: dict[str, Any],
    final_inner_scores: dict[str, float | None],
    started_at: datetime,
    ended_at: datetime,
) -> None:
    selected_model = ranking[0]
    paired_lines = []
    for _, row in paired_frame.iterrows():
        paired_lines.append(
            f"- {row['metric']}: {row['estimate']:+.4f}, 95% CI "
            f"[{row['ci_lower_2_5']:+.4f}, {row['ci_upper_97_5']:+.4f}]；"
            f"{row['evidence_statement']}。"
        )
    selected_ci = bootstrap_frame[
        (bootstrap_frame["model"] == selected_model)
        & (bootstrap_frame["metric"].isin(["auroc", "auprc", "brier_score"]))
    ]
    ci_lines = [
        f"- {row.metric}: {row.estimate:.4f} "
        f"[{row.ci_lower_2_5:.4f}, {row.ci_upper_97_5:.4f}]"
        for row in selected_ci.itertuples()
    ]
    hyper_lines = []
    for model_name in MODEL_ORDER:
        score = final_inner_scores[model_name]
        score_text = "不适用（固定模型）" if score is None else f"{score:.6f}"
        hyper_lines.append(
            f"- {model_name}: Train-only 内层 AP={score_text}；阈值={thresholds[model_name]:.12g}；"
            f"参数=`{json.dumps(hyperparameters[model_name], ensure_ascii=False, default=json_default)}`"
        )
    report = f"""# adverse_pre_v1 正式报告

## 冻结协议

- Train 仅用于 5×5 分层嵌套交叉验证、模型选择、超参数选择和阈值选择。
- 每个 Train 患者对每个模型恰好获得一次外层 OOF 概率。
- 模型排序依次使用 pooled OOF AUPRC、AUROC、Brier Score。
- 三个阈值分别由各自 Train pooled OOF 概率最大化 Balanced Accuracy 得到。
- 完整 Train 上再次执行 Train-only 5 折内层搜索并拟合最终模型。
- `frozen_selection.json` 写入并落盘后，才对 Valid 各模型执行一次 `predict_proba`。
- Valid 未用于调参、模型选择、阈值选择、特征选择、early stopping 或 eval_set。

## Train pooled OOF

{markdown_metrics(train_metrics_frame)}

## Valid（冻结后正式评估）

{markdown_metrics(valid_metrics_frame)}

## 模型选择与冻结值

- OOF 排名：{' > '.join(ranking)}
- 入选模型：**{selected_model}**
"""
    report += "\n".join(hyper_lines)
    report += "\n\n## Valid 患者级 Bootstrap（2000 次有效重复）\n\n"
    report += "入选模型主要概率指标的 95% 百分位置信区间：\n\n"
    report += "\n".join(ci_lines)
    report += "\n\n入选模型相对 Dummy 的配对差值（正值均表示入选模型更好）：\n\n"
    report += "\n".join(paired_lines)
    report += f"""

## GPU 与复现说明

- CatBoost 固定为 `task_type="GPU"`、`devices="0"`，无 CPU 自动回退。
- 正式训练前已保存 `gpu_verification.txt`，包含 nvidia-smi、CUDA_VISIBLE_DEVICES、CatBoost 版本、GPU 数量和最小 GPU 训练测试。
- CatBoost 即使固定随机种子，GPU 并行计算仍可能产生轻微非确定性。
- 随机种子：42。
- 开始时间（UTC）：{started_at.isoformat()}
- 结束时间（UTC）：{ended_at.isoformat()}
- 总运行时间：{(ended_at - started_at).total_seconds():.1f} 秒。
"""
    write_text_sync(REPORT_DIR / "report.md", report)

    summary = f"""# adverse_pre_v1 执行摘要

- 状态：SUCCESS
- 入选模型：{selected_model}
- OOF 排名：{' > '.join(ranking)}
- 冻结阈值：{json.dumps(thresholds, ensure_ascii=False, default=json_default)}
- Bootstrap：2000 次有效患者级有放回抽样，所有模型共享索引，seed=42
- CatBoost：GPU 0；失败不回退 CPU
- 开始：{started_at.isoformat()}
- 结束：{ended_at.isoformat()}
- 运行时间：{(ended_at - started_at).total_seconds():.1f} 秒
- 退出状态：0
"""
    write_text_sync(REPORT_DIR / "execution_summary.md", summary)


REQUIRED_RELATIVE_FILES = [
    "train_oof_predictions.csv",
    "valid_predictions.csv",
    "model_comparison.csv",
    "train_oof_metrics.csv",
    "valid_metrics.csv",
    "bootstrap_confidence_intervals.csv",
    "selected_vs_dummy_bootstrap.csv",
    "cv_results.csv",
    "outer_fold_metrics.csv",
    "threshold_search.csv",
    "final_hyperparameters.json",
    "thresholds.json",
    "feature_names.json",
    "configuration.json",
    "frozen_selection.json",
    "environment.txt",
    "gpu_verification.txt",
    "run.log",
    "exit_status.txt",
    "models/dummy.joblib",
    "models/logistic.joblib",
    "models/catboost_pipeline.joblib",
    "models/catboost.cbm",
    "logistic_coefficients.csv",
    "catboost_feature_importance.csv",
    "plots/train_oof_roc.png",
    "plots/train_oof_pr.png",
    "plots/valid_roc.png",
    "plots/valid_pr.png",
    "plots/valid_calibration.png",
    "plots/valid_confusion_dummy.png",
    "plots/valid_confusion_logistic.png",
    "plots/valid_confusion_catboost.png",
    "plots/logistic_coefficients.png",
    "plots/catboost_feature_importance.png",
]


def verify_required_artifacts(logger: logging.Logger) -> None:
    flush_logger(logger)
    missing_or_empty: list[str] = []
    for relative in REQUIRED_RELATIVE_FILES:
        path = OUTPUT_DIR / relative
        if not path.is_file() or path.stat().st_size <= 0:
            missing_or_empty.append(str(path))
    for name in ["report.md", "data_validation.md", "execution_summary.md"]:
        path = REPORT_DIR / name
        if not path.is_file() or path.stat().st_size <= 0:
            missing_or_empty.append(str(path))
    if missing_or_empty:
        raise RuntimeError(f"Missing or empty required artifacts: {missing_or_empty}")


def main() -> int:
    logger: logging.Logger | None = None
    started_at = utc_now()
    try:
        initialize_run_directories()
        logger = setup_logging(OUTPUT_DIR / "run.log")
        logger.info("adverse_pre_v1 started")
        logger.info("Interpreter: %s", sys.executable)
        logger.info("Script: %s", Path(__file__).resolve())
        logger.info("Original script preserved at: %s", PROJECT_ROOT / "code/09_train_adverse_pre.py")
        write_json_sync(OUTPUT_DIR / "configuration.json", CONFIGURATION)
        save_environment(OUTPUT_DIR / "environment.txt", started_at)

        input_hashes_before = {
            "train": sha256_file(TRAIN_PATH),
            "valid": sha256_file(VALID_PATH),
        }
        logger.info("Reading immutable input CSV files")
        train_df = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
        valid_df = pd.read_csv(VALID_PATH, encoding="utf-8-sig")
        feature_names = validate_inputs(train_df, valid_df)
        logger.info(
            "Input assertions passed | Train=%d positive=%d | Valid=%d positive=%d | features=%d",
            len(train_df),
            int(train_df["adverse"].sum()),
            len(valid_df),
            int(valid_df["adverse"].sum()),
            len(feature_names),
        )
        write_json_sync(
            OUTPUT_DIR / "feature_names.json",
            {
                "feature_count": len(feature_names),
                "feature_names": feature_names,
                "excluded_from_model": ["patient_id", "split", "adverse"],
                "all_preoperative": True,
                "forbidden_tokens": ["post_", "delta_", "runtime_s", "n_pairs"],
            },
        )
        write_data_validation_report(
            REPORT_DIR / "data_validation.md",
            train_df,
            valid_df,
            feature_names,
            input_hashes_before,
        )

        verify_gpu(OUTPUT_DIR / "gpu_verification.txt", logger)
        logger.info("GPU evidence saved before formal model training")

        x_train = train_df.loc[:, feature_names]
        x_valid = valid_df.loc[:, feature_names]
        y_train = train_df["adverse"].astype(int).to_numpy()
        y_valid = valid_df["adverse"].astype(int).to_numpy()
        outer_splits = list(
            StratifiedKFold(
                n_splits=OUTER_FOLDS,
                shuffle=True,
                random_state=SEED,
            ).split(x_train, y_train)
        )

        cv_rows: list[dict[str, Any]] = []
        oof_probabilities: dict[str, np.ndarray] = {}
        fold_search: dict[str, list[dict[str, Any]]] = {}
        for model_name in MODEL_ORDER:
            logger.info("Starting true nested OOF procedure: %s", model_name)
            probabilities, search_info = run_nested_oof(
                model_name,
                x_train,
                y_train,
                outer_splits,
                cv_rows,
                logger,
            )
            oof_probabilities[model_name] = probabilities
            fold_search[model_name] = search_info

        thresholds: dict[str, float] = {}
        threshold_frames: list[pd.DataFrame] = []
        train_metrics: dict[str, dict[str, float | int | str]] = {}
        for model_name in MODEL_ORDER:
            threshold, frame = select_threshold(
                model_name, y_train, oof_probabilities[model_name]
            )
            thresholds[model_name] = threshold
            threshold_frames.append(frame)
            train_metrics[model_name] = calculate_metrics(
                y_train, oof_probabilities[model_name], threshold
            )
            logger.info(
                "%s pooled OOF | AUPRC=%.6f AUROC=%.6f Brier=%.6f | threshold=%.12g",
                model_name,
                train_metrics[model_name]["auprc"],
                train_metrics[model_name]["auroc"],
                train_metrics[model_name]["brier_score"],
                threshold,
            )

        ranking = model_ranking(train_metrics)
        selected_model = ranking[0]
        logger.info("Train pooled OOF ranking frozen: %s", " > ".join(ranking))
        logger.info("Selected model from Train only: %s", selected_model)

        final_models, selected_params, final_inner_scores = fit_final_models(
            x_train, y_train, cv_rows, logger
        )
        hyperparameters = complete_hyperparameters(selected_params)

        train_oof_frame = pd.DataFrame(
            {
                "patient_id": train_df["patient_id"],
                "split": train_df["split"],
                "adverse": y_train,
            }
        )
        for model_name in MODEL_ORDER:
            slug = PREDICTION_SLUG[model_name]
            train_oof_frame[f"{slug}_probability"] = oof_probabilities[model_name]
            train_oof_frame[f"{slug}_prediction"] = (
                oof_probabilities[model_name] >= thresholds[model_name]
            ).astype(int)
            train_oof_frame[f"{slug}_threshold"] = thresholds[model_name]
        train_oof_frame.to_csv(
            OUTPUT_DIR / "train_oof_predictions.csv", index=False, encoding="utf-8"
        )

        train_metric_rows = [
            {
                "dataset": "train_pooled_nested_oof",
                "model": model_name,
                "threshold": thresholds[model_name],
                **train_metrics[model_name],
            }
            for model_name in MODEL_ORDER
        ]
        train_metrics_frame = pd.DataFrame(train_metric_rows)
        train_metrics_frame.to_csv(
            OUTPUT_DIR / "train_oof_metrics.csv", index=False, encoding="utf-8"
        )
        pd.concat(threshold_frames, ignore_index=True).to_csv(
            OUTPUT_DIR / "threshold_search.csv", index=False, encoding="utf-8"
        )
        build_outer_fold_metrics(
            y_train,
            oof_probabilities,
            thresholds,
            outer_splits,
            fold_search,
        ).to_csv(OUTPUT_DIR / "outer_fold_metrics.csv", index=False, encoding="utf-8")
        pd.DataFrame(cv_rows).to_csv(
            OUTPUT_DIR / "cv_results.csv", index=False, encoding="utf-8"
        )
        write_json_sync(OUTPUT_DIR / "thresholds.json", thresholds)
        write_json_sync(
            OUTPUT_DIR / "final_hyperparameters.json",
            {
                "models": hyperparameters,
                "final_train_only_inner_average_precision": final_inner_scores,
                "catboost_sampled_candidates": CATBOOST_CANDIDATES,
            },
        )

        joblib.dump(final_models["Dummy"], MODEL_DIR / "dummy.joblib")
        joblib.dump(final_models["Logistic"], MODEL_DIR / "logistic.joblib")
        joblib.dump(final_models["CatBoost"], MODEL_DIR / "catboost_pipeline.joblib")
        final_models["CatBoost"].named_steps["model"].save_model(
            str(MODEL_DIR / "catboost.cbm")
        )
        logistic_coefficients, catboost_importance = save_interpretation(
            final_models, feature_names
        )

        frozen_at = utc_now()
        ranking_records = []
        for rank, model_name in enumerate(ranking, start=1):
            ranking_records.append(
                {
                    "rank": rank,
                    "model": model_name,
                    "train_oof_auprc": train_metrics[model_name]["auprc"],
                    "train_oof_auroc": train_metrics[model_name]["auroc"],
                    "train_oof_brier_score": train_metrics[model_name]["brier_score"],
                }
            )
        frozen_selection = {
            "experiment": "adverse_pre_v1",
            "frozen_at_utc": frozen_at.isoformat(),
            "random_seed": SEED,
            "train_oof_model_ranking": ranking_records,
            "selected_model": selected_model,
            "final_hyperparameters": hyperparameters,
            "final_train_only_inner_average_precision": final_inner_scores,
            "frozen_thresholds": thresholds,
            "selection_rule": [
                "AUPRC descending",
                "AUROC descending",
                "Brier Score ascending",
            ],
            "valid_used_before_freeze": False,
        }
        write_json_sync(OUTPUT_DIR / "frozen_selection.json", frozen_selection)
        assert (OUTPUT_DIR / "frozen_selection.json").stat().st_size > 0
        logger.info(
            "frozen_selection.json durably written before any Valid prediction | frozen_at=%s",
            frozen_at.isoformat(),
        )

        valid_probabilities: dict[str, np.ndarray] = {}
        valid_metrics: dict[str, dict[str, float | int | str]] = {}
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
                "%s official Valid prediction complete | AUROC=%.6f AUPRC=%.6f "
                "Brier=%.6f threshold=%.12g",
                model_name,
                valid_metrics[model_name]["auroc"],
                valid_metrics[model_name]["auprc"],
                valid_metrics[model_name]["brier_score"],
                thresholds[model_name],
            )

        valid_prediction_frame = pd.DataFrame(
            {
                "patient_id": valid_df["patient_id"],
                "split": valid_df["split"],
                "adverse": y_valid,
            }
        )
        for model_name in MODEL_ORDER:
            slug = PREDICTION_SLUG[model_name]
            valid_prediction_frame[f"{slug}_probability"] = valid_probabilities[model_name]
            valid_prediction_frame[f"{slug}_prediction"] = (
                valid_probabilities[model_name] >= thresholds[model_name]
            ).astype(int)
            valid_prediction_frame[f"{slug}_threshold"] = thresholds[model_name]
        valid_prediction_frame.to_csv(
            OUTPUT_DIR / "valid_predictions.csv", index=False, encoding="utf-8"
        )

        valid_metrics_frame = pd.DataFrame(
            [
                {
                    "dataset": "valid_frozen_official",
                    "model": model_name,
                    "threshold": thresholds[model_name],
                    **valid_metrics[model_name],
                }
                for model_name in MODEL_ORDER
            ]
        )
        valid_metrics_frame.to_csv(
            OUTPUT_DIR / "valid_metrics.csv", index=False, encoding="utf-8"
        )

        bootstrap_frame, paired_frame = bootstrap_results(
            y_valid,
            valid_probabilities,
            thresholds,
            valid_metrics,
            selected_model,
            logger,
        )
        bootstrap_frame.to_csv(
            OUTPUT_DIR / "bootstrap_confidence_intervals.csv", index=False, encoding="utf-8"
        )
        paired_frame.to_csv(
            OUTPUT_DIR / "selected_vs_dummy_bootstrap.csv", index=False, encoding="utf-8"
        )

        rank_lookup = {model_name: rank for rank, model_name in enumerate(ranking, start=1)}
        comparison_rows = []
        for model_name in MODEL_ORDER:
            comparison_rows.append(
                {
                    "model": model_name,
                    "train_oof_rank": rank_lookup[model_name],
                    "selected": model_name == selected_model,
                    "train_oof_auprc": train_metrics[model_name]["auprc"],
                    "train_oof_auroc": train_metrics[model_name]["auroc"],
                    "train_oof_brier_score": train_metrics[model_name]["brier_score"],
                    "frozen_threshold": thresholds[model_name],
                    "final_train_only_inner_average_precision": final_inner_scores[model_name],
                    "final_hyperparameters_json": json.dumps(
                        hyperparameters[model_name],
                        ensure_ascii=False,
                        sort_keys=True,
                        default=json_default,
                    ),
                    "valid_auprc_report_only": valid_metrics[model_name]["auprc"],
                    "valid_auroc_report_only": valid_metrics[model_name]["auroc"],
                    "valid_brier_score_report_only": valid_metrics[model_name]["brier_score"],
                }
            )
        pd.DataFrame(comparison_rows).sort_values("train_oof_rank").to_csv(
            OUTPUT_DIR / "model_comparison.csv", index=False, encoding="utf-8"
        )

        plot_roc(
            y_train,
            oof_probabilities,
            "Train pooled nested OOF ROC",
            PLOT_DIR / "train_oof_roc.png",
        )
        plot_pr(
            y_train,
            oof_probabilities,
            "Train pooled nested OOF precision-recall",
            PLOT_DIR / "train_oof_pr.png",
        )
        plot_roc(
            y_valid,
            valid_probabilities,
            "Valid ROC (frozen models)",
            PLOT_DIR / "valid_roc.png",
        )
        plot_pr(
            y_valid,
            valid_probabilities,
            "Valid precision-recall (frozen models)",
            PLOT_DIR / "valid_pr.png",
        )
        plot_valid_calibration(
            y_valid, valid_probabilities, PLOT_DIR / "valid_calibration.png"
        )
        for model_name in MODEL_ORDER:
            plot_confusion(
                valid_metrics[model_name],
                model_name,
                thresholds[model_name],
                PLOT_DIR / f"valid_confusion_{PREDICTION_SLUG[model_name]}.png",
            )
        plot_feature_tables(logistic_coefficients, catboost_importance)

        input_hashes_after = {
            "train": sha256_file(TRAIN_PATH),
            "valid": sha256_file(VALID_PATH),
        }
        assert input_hashes_after == input_hashes_before, "Input CSV hash changed during run"

        ended_at = utc_now()
        generate_reports(
            train_metrics_frame,
            valid_metrics_frame,
            bootstrap_frame,
            paired_frame,
            ranking,
            thresholds,
            hyperparameters,
            final_inner_scores,
            started_at,
            ended_at,
        )
        write_text_sync(OUTPUT_DIR / "exit_status.txt", "0\n")
        verify_required_artifacts(logger)
        logger.info(
            "All required artifacts verified non-empty | selected=%s | elapsed=%.1f seconds",
            selected_model,
            (ended_at - started_at).total_seconds(),
        )
        flush_logger(logger)
        RUNNING_PATH.unlink()
        write_text_sync(
            SUCCESS_PATH,
            f"status=SUCCESS\ncompleted_at_utc={ended_at.isoformat()}\nexit_status=0\n",
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
            logger.exception("adverse_pre_v1 failed; .RUNNING retained")
            close_logger(logger)
        else:
            traceback.print_exc()
        return 1
    finally:
        if logger is not None and logger.handlers:
            close_logger(logger)


if __name__ == "__main__":
    sys.exit(main())

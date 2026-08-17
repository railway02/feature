#!/usr/bin/env python3
"""Train a controlled pure-image multi-branch MLP adverse-outcome probe.

The input features are screened inside each Train fitting fold with the same
documented CAVE/SEA groups used by the stable sparse Logistic experiment.  A
single predeclared medium candidate size and compact MLP architecture are used
to avoid a broad neural-network hyperparameter search on only 172 positives.

Inner Train folds determine only the number of epochs.  Outer holdouts produce
OOF predictions.  Valid labels are not accessed until final metrics, after all
Valid probabilities have been generated.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_adverse_stable_sparse_probe import (  # noqa: E402
    FeatureCatalog,
    VARIANT_SOURCES,
    atomic_csv,
    atomic_json,
    compact_feature_metadata,
    compact_matrix,
    cv_splits,
    fit_preprocessors,
    load_bundle,
    metric_row,
    rank_groups,
    safe_ap,
    safe_auc,
    selections_for_size,
    sha256,
    transform_sources,
    youden_threshold,
)


SEED = 42
SIZE_NAME = "medium"
OUTER_SEEDS = (11, 29, 47)
MAX_EPOCHS = 120
MIN_REFIT_EPOCHS = 10
PATIENCE = 18
BATCH_SIZE = 2048
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
DROPOUT = 0.35
BRANCH_WIDTH = {
    "cave_embedding": 48,
    "cave_scalar": 24,
    "sea_full": 24,
}
HEAD_WIDTH = 48
PERMUTATION_REPEATS = 3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


class BranchEncoder(nn.Module):
    def __init__(self, input_dimension: int, output_dimension: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimension, output_dimension),
            nn.LayerNorm(output_dimension),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class MultiBranchMLP(nn.Module):
    def __init__(self, source_dimensions: dict[str, int]) -> None:
        super().__init__()
        self.source_order = tuple(source_dimensions)
        self.encoders = nn.ModuleDict({
            source: BranchEncoder(dimension, BRANCH_WIDTH[source])
            for source, dimension in source_dimensions.items()
        })
        fusion_dimension = sum(BRANCH_WIDTH[source] for source in self.source_order) + 2
        self.head = nn.Sequential(
            nn.Linear(fusion_dimension, HEAD_WIDTH),
            nn.LayerNorm(HEAD_WIDTH),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HEAD_WIDTH, 1),
        )

    def forward(self, values: torch.Tensor, slices: dict[str, tuple[int, int]]) -> torch.Tensor:
        encoded = [
            self.encoders[source](values[:, slices[source][0]:slices[source][1]])
            for source in self.source_order
        ]
        missing_start, missing_end = slices["missing"]
        encoded.append(values[:, missing_start:missing_end])
        return self.head(torch.cat(encoded, dim=1)).squeeze(1)


def design_slices(
    selections: dict[str, np.ndarray], variant: str
) -> tuple[dict[str, int], dict[str, tuple[int, int]]]:
    dimensions = {
        source: int(len(selections[source])) for source in VARIANT_SOURCES[variant]
    }
    slices: dict[str, tuple[int, int]] = {}
    offset = 0
    for source in VARIANT_SOURCES[variant]:
        end = offset + dimensions[source]
        slices[source] = (offset, end)
        offset = end
    slices["missing"] = (offset, offset + 2)
    return dimensions, slices


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def predict_probability(
    model: MultiBranchMLP,
    values: np.ndarray,
    slices: dict[str, tuple[int, int]],
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
        probability = torch.sigmoid(model(tensor, slices)).cpu().numpy()
    return np.asarray(probability, dtype=np.float64)


def ensemble_probability(
    models: list[MultiBranchMLP],
    values: np.ndarray,
    slices: dict[str, tuple[int, int]],
    device: torch.device,
) -> np.ndarray:
    return np.mean(
        np.stack([predict_probability(model, values, slices, device) for model in models]),
        axis=0,
    )


@dataclass
class TrainResult:
    model: MultiBranchMLP
    epochs_run: int
    best_epoch: int
    best_validation_auprc: float
    best_validation_auroc: float
    final_training_loss: float
    seconds: float


def train_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    source_dimensions: dict[str, int],
    slices: dict[str, tuple[int, int]],
    device: torch.device,
    seed: int,
    epochs: int,
    validation_x: np.ndarray | None = None,
    validation_y: np.ndarray | None = None,
) -> TrainResult:
    set_seed(seed)
    model = MultiBranchMLP(source_dimensions).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    positives = int(np.sum(train_y == 1))
    negatives = int(np.sum(train_y == 0))
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / max(positives, 1), dtype=torch.float32, device=device)
    )
    x_tensor = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 100003)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_ap = -np.inf
    best_auc = -np.inf
    epochs_without_improvement = 0
    final_loss = float("nan")
    start_time = time.perf_counter()
    epochs_run = 0
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_y), generator=generator)
        loss_sum = 0.0
        rows_seen = 0
        for batch_start in range(0, len(train_y), BATCH_SIZE):
            positions = permutation[batch_start:batch_start + BATCH_SIZE].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_tensor[positions], slices)
            loss = loss_function(logits, y_tensor[positions])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite MLP loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batch_rows = int(len(positions))
            loss_sum += float(loss.detach().cpu()) * batch_rows
            rows_seen += batch_rows
        final_loss = loss_sum / max(rows_seen, 1)
        epochs_run = epoch
        if validation_x is None or validation_y is None:
            continue
        probability = predict_probability(model, validation_x, slices, device)
        validation_ap = safe_ap(validation_y, probability)
        validation_auc = safe_auc(validation_y, probability)
        improved = (
            validation_ap > best_ap + 1e-6
            or (abs(validation_ap - best_ap) <= 1e-6 and validation_auc > best_auc + 1e-6)
        )
        if improved:
            best_ap = validation_ap
            best_auc = validation_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch >= MIN_REFIT_EPOCHS and epochs_without_improvement >= PATIENCE:
            break
    if validation_x is not None:
        if best_state is None:
            raise AssertionError("Early stopping never captured a model")
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs_run
        best_ap = float("nan")
        best_auc = float("nan")
    return TrainResult(
        model=model,
        epochs_run=epochs_run,
        best_epoch=best_epoch,
        best_validation_auprc=float(best_ap),
        best_validation_auroc=float(best_auc),
        final_training_loss=float(final_loss),
        seconds=float(time.perf_counter() - start_time),
    )


def prepare_designs(
    fit_data: Any,
    evaluation_data: Any,
    valid_data: Any | None,
    catalog: FeatureCatalog,
) -> tuple[
    Any,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray] | None,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    preprocessors = fit_preprocessors(fit_data)
    fit_transformed = transform_sources(preprocessors, fit_data)
    evaluation_transformed = transform_sources(preprocessors, evaluation_data)
    valid_transformed = (
        transform_sources(preprocessors, valid_data) if valid_data is not None else None
    )
    scores, rankings = rank_groups(catalog, fit_transformed, preprocessors, fit_data.target)
    selections = selections_for_size(rankings, SIZE_NAME)
    fit_designs = {
        variant: compact_matrix(fit_transformed, selections, variant)
        for variant in VARIANT_SOURCES
    }
    evaluation_designs = {
        variant: compact_matrix(evaluation_transformed, selections, variant)
        for variant in VARIANT_SOURCES
    }
    valid_designs = (
        {
            variant: compact_matrix(valid_transformed, selections, variant)
            for variant in VARIANT_SOURCES
        }
        if valid_transformed is not None else None
    )
    metadata = {
        variant: compact_feature_metadata(catalog, scores, selections, variant)
        for variant in VARIANT_SOURCES
    }
    audit = {
        "preprocessors": {source: item.audit() for source, item in preprocessors.items()},
        "group_counts": {source: len(group) for source, group in rankings.items()},
    }
    return (
        preprocessors,
        fit_designs,
        evaluation_designs,
        valid_designs,
        selections,
        scores,
        {"metadata": metadata, "audit": audit},
    )


def inner_epoch_selection(
    data: Any,
    catalog: FeatureCatalog,
    outer_fold: int,
    device: torch.device,
    training_rows: list[dict[str, Any]],
) -> tuple[dict[str, int], dict[str, list[dict[str, Any]]]]:
    best_epochs = {variant: [] for variant in VARIANT_SOURCES}
    inner_details = {variant: [] for variant in VARIANT_SOURCES}
    for inner_fold, (fit_index, holdout_index) in enumerate(
        cv_splits(data.target, 3, SEED + outer_fold * 1000), start=1
    ):
        fit_data = data.subset(fit_index)
        holdout_data = data.subset(holdout_index)
        (
            _, fit_designs, holdout_designs, _, selections, _, prepared,
        ) = prepare_designs(fit_data, holdout_data, None, catalog)
        for variant in VARIANT_SOURCES:
            source_dimensions, slices = design_slices(selections, variant)
            seed = SEED + outer_fold * 10000 + inner_fold * 100 + list(VARIANT_SOURCES).index(variant)
            result = train_model(
                fit_designs[variant],
                fit_data.target,
                source_dimensions,
                slices,
                device,
                seed,
                MAX_EPOCHS,
                holdout_designs[variant],
                holdout_data.target,
            )
            best_epochs[variant].append(result.best_epoch)
            detail = {
                "inner_fold": inner_fold,
                "best_epoch": result.best_epoch,
                "epochs_run": result.epochs_run,
                "holdout_auroc": result.best_validation_auroc,
                "holdout_auprc": result.best_validation_auprc,
                "compact_dimension": int(fit_designs[variant].shape[1]),
            }
            inner_details[variant].append(detail)
            training_rows.append({
                "stage": "inner_epoch_selection",
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "variant": variant,
                "seed": seed,
                "requested_epochs": MAX_EPOCHS,
                "epochs_run": result.epochs_run,
                "best_epoch": result.best_epoch,
                "validation_auroc": result.best_validation_auroc,
                "validation_auprc": result.best_validation_auprc,
                "final_training_loss": result.final_training_loss,
                "fit_seconds": result.seconds,
                "parameters": parameter_count(result.model),
                "compact_dimension": int(fit_designs[variant].shape[1]),
                "preprocessing_audit": json.dumps(prepared["audit"], sort_keys=True),
            })
    selected_epochs = {
        variant: max(
            MIN_REFIT_EPOCHS,
            int(round(float(np.median(values)))),
        )
        for variant, values in best_epochs.items()
    }
    return selected_epochs, inner_details


def save_model_artifact(
    model: MultiBranchMLP,
    path: Path,
    source_dimensions: dict[str, int],
    slices: dict[str, tuple[int, int]],
    feature_metadata: list[dict[str, Any]],
    epochs: int,
    seed: int,
) -> None:
    torch.save({
        "state_dict": model.state_dict(),
        "source_dimensions": source_dimensions,
        "slices": slices,
        "features": feature_metadata,
        "epochs": epochs,
        "seed": seed,
        "architecture": {
            "branch_width": BRANCH_WIDTH,
            "head_width": HEAD_WIDTH,
            "dropout": DROPOUT,
        },
    }, path)


def group_permutation_importance(
    models: list[MultiBranchMLP],
    evaluation_x: np.ndarray,
    evaluation_y: np.ndarray,
    slices: dict[str, tuple[int, int]],
    metadata: list[dict[str, Any]],
    device: torch.device,
    fold: int,
    variant: str,
) -> list[dict[str, Any]]:
    baseline_probability = ensemble_probability(models, evaluation_x, slices, device)
    baseline_ap = safe_ap(evaluation_y, baseline_probability)
    baseline_auc = safe_auc(evaluation_y, baseline_probability)
    group_columns: dict[str, list[int]] = {}
    group_sources: dict[str, str] = {}
    for column, item in enumerate(metadata):
        group = str(item["group"])
        group_columns.setdefault(group, []).append(column)
        group_sources[group] = str(item["source"])
    rows: list[dict[str, Any]] = []
    for group_index, group in enumerate(sorted(group_columns)):
        columns = np.asarray(group_columns[group], dtype=np.int64)
        for repeat in range(PERMUTATION_REPEATS):
            rng = np.random.default_rng(SEED + fold * 100000 + group_index * 100 + repeat)
            row_order = rng.permutation(len(evaluation_y))
            permuted = evaluation_x.copy()
            permuted[:, columns] = evaluation_x[row_order][:, columns]
            probability = ensemble_probability(models, permuted, slices, device)
            permuted_ap = safe_ap(evaluation_y, probability)
            permuted_auc = safe_auc(evaluation_y, probability)
            rows.append({
                "fold": fold,
                "variant": variant,
                "source": group_sources[group],
                "group": group,
                "group_columns": int(len(columns)),
                "repeat": repeat + 1,
                "baseline_auprc": baseline_ap,
                "permuted_auprc": permuted_ap,
                "auprc_decrease": baseline_ap - permuted_ap,
                "baseline_auroc": baseline_auc,
                "permuted_auroc": permuted_auc,
                "auroc_decrease": baseline_auc - permuted_auc,
            })
    return rows


def aggregate_group_importance(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    aggregated = (
        frame.groupby(["variant", "source", "group"], as_index=False)
        .agg(
            outer_folds=("fold", "nunique"),
            group_columns_median=("group_columns", "median"),
            mean_auprc_decrease=("auprc_decrease", "mean"),
            median_auprc_decrease=("auprc_decrease", "median"),
            positive_auprc_decrease_fraction=("auprc_decrease", lambda value: float((value > 0).mean())),
            mean_auroc_decrease=("auroc_decrease", "mean"),
            median_auroc_decrease=("auroc_decrease", "median"),
            positive_auroc_decrease_fraction=("auroc_decrease", lambda value: float((value > 0).mean())),
        )
        .sort_values(
            ["variant", "mean_auprc_decrease", "mean_auroc_decrease"],
            ascending=[True, False, False],
        )
    )
    aggregated["importance_rank_within_variant"] = aggregated.groupby("variant").cumcount() + 1
    return aggregated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--cave-task-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = Path(args.dataset_dir).resolve()
    cave_task_config = Path(args.cave_task_config).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    train = load_bundle(dataset_dir / "train.npz")
    valid = load_bundle(dataset_dir / "valid.npz")
    if set(train.patient_id) & set(valid.patient_id):
        raise AssertionError("Train/Valid patient overlap")
    schema = json.loads((dataset_dir / "feature_schema.json").read_text(encoding="utf-8"))
    if schema.get("metadata_predictor_columns"):
        raise AssertionError("Clinical predictors entered MLP probe")
    catalog = FeatureCatalog.load(dataset_dir, cave_task_config)

    training_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []
    oof = {
        variant: np.full(len(train.target), np.nan, dtype=np.float64)
        for variant in VARIANT_SOURCES
    }
    valid_fold_predictions = {variant: [] for variant in VARIANT_SOURCES}

    for fold, (development_index, holdout_index) in enumerate(
        cv_splits(train.target, 5, SEED), start=1
    ):
        print(f"[MLP OUTER FOLD START] {fold}/5", flush=True)
        development = train.subset(development_index)
        holdout = train.subset(holdout_index)
        selected_epochs, inner_details = inner_epoch_selection(
            development, catalog, fold, device, training_rows
        )
        (
            preprocessors,
            development_designs,
            holdout_designs,
            valid_designs,
            selections,
            _,
            prepared,
        ) = prepare_designs(development, holdout, valid, catalog)
        if valid_designs is None:
            raise AssertionError("Valid designs missing")
        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessors, fold_dir / "source_preprocessors.joblib")
        for variant_index, variant in enumerate(VARIANT_SOURCES):
            epochs = selected_epochs[variant]
            source_dimensions, slices = design_slices(selections, variant)
            models: list[MultiBranchMLP] = []
            seed_probabilities_holdout: list[np.ndarray] = []
            seed_probabilities_valid: list[np.ndarray] = []
            for seed_offset in OUTER_SEEDS:
                seed = SEED + fold * 10000 + variant_index * 100 + seed_offset
                result = train_model(
                    development_designs[variant],
                    development.target,
                    source_dimensions,
                    slices,
                    device,
                    seed,
                    epochs,
                )
                models.append(result.model)
                seed_probabilities_holdout.append(
                    predict_probability(result.model, holdout_designs[variant], slices, device)
                )
                seed_probabilities_valid.append(
                    predict_probability(result.model, valid_designs[variant], slices, device)
                )
                training_rows.append({
                    "stage": "outer_development_refit",
                    "outer_fold": fold,
                    "inner_fold": 0,
                    "variant": variant,
                    "seed": seed,
                    "requested_epochs": epochs,
                    "epochs_run": result.epochs_run,
                    "best_epoch": result.best_epoch,
                    "validation_auroc": float("nan"),
                    "validation_auprc": float("nan"),
                    "final_training_loss": result.final_training_loss,
                    "fit_seconds": result.seconds,
                    "parameters": parameter_count(result.model),
                    "compact_dimension": int(development_designs[variant].shape[1]),
                    "preprocessing_audit": json.dumps(prepared["audit"], sort_keys=True),
                })
                save_model_artifact(
                    result.model,
                    fold_dir / f"mlp_{variant}_seed{seed_offset}.pt",
                    source_dimensions,
                    slices,
                    prepared["metadata"][variant],
                    epochs,
                    seed,
                )
            holdout_probability = np.mean(np.stack(seed_probabilities_holdout), axis=0)
            valid_probability = np.mean(np.stack(seed_probabilities_valid), axis=0)
            oof[variant][holdout_index] = holdout_probability
            valid_fold_predictions[variant].append(valid_probability)
            permutation_rows.extend(group_permutation_importance(
                models,
                holdout_designs[variant],
                holdout.target,
                slices,
                prepared["metadata"][variant],
                device,
                fold,
                variant,
            ))
            fold_rows.append({
                "fold": fold,
                "variant": variant,
                "development_rows": int(len(development_index)),
                "holdout_rows": int(len(holdout_index)),
                "selected_epochs": epochs,
                "inner_best_epochs": json.dumps(
                    [row["best_epoch"] for row in inner_details[variant]]
                ),
                "inner_holdout_auprc": json.dumps(
                    [row["holdout_auprc"] for row in inner_details[variant]]
                ),
                "compact_dimension": int(development_designs[variant].shape[1]),
                "parameters_per_seed_model": parameter_count(models[0]),
                "holdout_auroc": safe_auc(holdout.target, holdout_probability),
                "holdout_auprc": safe_ap(holdout.target, holdout_probability),
                "seed_probability_mean_std": float(
                    np.mean(np.std(np.stack(seed_probabilities_holdout), axis=0))
                ),
            })
        atomic_csv(pd.DataFrame(training_rows), output_dir / "training_audit.csv")
        atomic_csv(pd.DataFrame(fold_rows), output_dir / "fold_audit.csv")
        atomic_csv(pd.DataFrame(permutation_rows), output_dir / "group_permutation_occurrences.csv")
        print(f"[MLP OUTER FOLD DONE] {fold}/5", flush=True)

    print("[MLP FULL TRAIN START]", flush=True)
    full_epochs, full_inner_details = inner_epoch_selection(
        train, catalog, 0, device, training_rows
    )
    # Use a harmless Train subset as the required evaluation-design argument;
    # it is not used for fitting, selecting, or reporting the final models.
    train_reference = train.subset(np.arange(min(2, len(train.target))))
    (
        full_preprocessors,
        full_designs,
        _,
        _,
        full_selections,
        _,
        full_prepared,
    ) = prepare_designs(train, train_reference, None, catalog)
    full_dir = output_dir / "full_train"
    full_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(full_preprocessors, full_dir / "source_preprocessors.joblib")
    for variant_index, variant in enumerate(VARIANT_SOURCES):
        source_dimensions, slices = design_slices(full_selections, variant)
        for seed_offset in OUTER_SEEDS:
            seed = SEED + 900000 + variant_index * 100 + seed_offset
            result = train_model(
                full_designs[variant],
                train.target,
                source_dimensions,
                slices,
                device,
                seed,
                full_epochs[variant],
            )
            training_rows.append({
                "stage": "full_train_refit",
                "outer_fold": 0,
                "inner_fold": 0,
                "variant": variant,
                "seed": seed,
                "requested_epochs": full_epochs[variant],
                "epochs_run": result.epochs_run,
                "best_epoch": result.best_epoch,
                "validation_auroc": float("nan"),
                "validation_auprc": float("nan"),
                "final_training_loss": result.final_training_loss,
                "fit_seconds": result.seconds,
                "parameters": parameter_count(result.model),
                "compact_dimension": int(full_designs[variant].shape[1]),
                "preprocessing_audit": json.dumps(full_prepared["audit"], sort_keys=True),
            })
            save_model_artifact(
                result.model,
                full_dir / f"mlp_{variant}_seed{seed_offset}.pt",
                source_dimensions,
                slices,
                full_prepared["metadata"][variant],
                full_epochs[variant],
                seed,
            )

    probability_sets: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "Dummy": (
            np.full(len(train.target), train.target.mean(), dtype=np.float64),
            np.full(len(valid.target), train.target.mean(), dtype=np.float64),
        )
    }
    for variant in VARIANT_SOURCES:
        if not np.isfinite(oof[variant]).all():
            raise AssertionError(f"Incomplete MLP OOF: {variant}")
        valid_probability = np.mean(np.stack(valid_fold_predictions[variant]), axis=0)
        if not np.isfinite(valid_probability).all():
            raise AssertionError(f"Nonfinite MLP Valid: {variant}")
        probability_sets[f"KeyMLP_{variant}"] = (oof[variant], valid_probability)

    thresholds: dict[str, float] = {}
    metrics: list[dict[str, Any]] = []
    train_predictions = pd.DataFrame({"patient_id": train.patient_id, "target": train.target})
    valid_predictions = pd.DataFrame({"patient_id": valid.patient_id})
    for model_name, (train_probability, valid_probability) in probability_sets.items():
        threshold = youden_threshold(train.target, train_probability)
        thresholds[model_name] = threshold
        metrics.append(metric_row(
            model_name, "Train_OOF", train.target, train_probability, threshold
        ))
        train_predictions[f"{model_name.lower()}_probability"] = train_probability
        valid_predictions[f"{model_name.lower()}_probability"] = valid_probability
    # Valid labels enter only after every probability and Train threshold exists.
    valid_predictions["target"] = valid.target
    for model_name, (_, valid_probability) in probability_sets.items():
        metrics.append(metric_row(
            model_name, "Valid", valid.target, valid_probability, thresholds[model_name]
        ))

    metrics_frame = pd.DataFrame(metrics)
    atomic_csv(metrics_frame, output_dir / "metrics.csv")
    atomic_csv(train_predictions, output_dir / "train_oof_predictions.csv")
    atomic_csv(valid_predictions, output_dir / "valid_predictions.csv")
    atomic_csv(pd.DataFrame(training_rows), output_dir / "training_audit.csv")
    atomic_csv(pd.DataFrame(fold_rows), output_dir / "fold_audit.csv")
    atomic_csv(pd.DataFrame(permutation_rows), output_dir / "group_permutation_occurrences.csv")
    aggregate_group_importance(permutation_rows).to_csv(
        output_dir / "group_permutation_importance.csv", index=False, encoding="utf-8"
    )

    learned_oof = metrics_frame[
        (metrics_frame["split"] == "Train_OOF") & (metrics_frame["model"] != "Dummy")
    ].sort_values(["auprc", "auroc"], ascending=False)
    summary = {
        "version": "api_fullseq_image_probe_v3_key_mlp_1",
        "task": "patient-level adverse outcome",
        "predictors": "frozen image-derived CAVE and SEA-RAFT features only",
        "device": str(device),
        "torch_version": torch.__version__,
        "train_rows": int(len(train.target)),
        "train_positive": int(train.target.sum()),
        "valid_rows": int(len(valid.target)),
        "valid_positive": int(valid.target.sum()),
        "outer_folds": 5,
        "inner_folds_for_epochs": 3,
        "outer_seed_ensemble": list(OUTER_SEEDS),
        "screening_size": SIZE_NAME,
        "architecture": {
            "branch_width": BRANCH_WIDTH,
            "head_width": HEAD_WIDTH,
            "dropout": DROPOUT,
            "activation": "GELU",
            "normalization": "LayerNorm",
        },
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "max_inner_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "full_train_epochs": full_epochs,
        "full_train_inner_details": full_inner_details,
        "best_model_selected_by_train_oof_auprc": str(learned_oof.iloc[0]["model"]),
        "thresholds_from_train_oof": thresholds,
        "pca": False,
        "random_projection": False,
        "clinical_predictors": False,
        "valid_used_for_preprocessing_screening_training_epochs_or_threshold": False,
        "valid_labels_used_for": "final metrics only after predictions",
        "input_sha256": {
            "train.npz": sha256(dataset_dir / "train.npz"),
            "valid.npz": sha256(dataset_dir / "valid.npz"),
            "feature_schema.json": sha256(dataset_dir / "feature_schema.json"),
            "cave_task_config.json": sha256(cave_task_config),
        },
        "seed": SEED,
    }
    atomic_json(summary, output_dir / "summary.json")
    atomic_json(summary, output_dir / ".MODELS_SUCCESS")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

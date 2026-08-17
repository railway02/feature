#!/usr/bin/env python3
"""Train leakage-controlled stable sparse probes on frozen CAVE/SEA features.

The model is intentionally a transparent two-level probe:

1. Within every Train fitting fold, rank features inside documented source
   groups by a univariate standardized effect score and retain a balanced
   quota from every group.
2. Fit an L1-regularized Logistic model on only those compact candidates.

Candidate quota and Logistic C are selected by inner Train CV.  Valid is never
used for imputation, scaling, screening, tuning, thresholding, or early
stopping; its labels are touched only after all predictions have been made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold


SEED = 42
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0)
SIZE_CONFIGS = {
    "small": {"cave_embedding": 4, "cave_scalar": 1, "sea_full": 1},
    "medium": {"cave_embedding": 8, "cave_scalar": 2, "sea_full": 2},
    "large": {"cave_embedding": 12, "cave_scalar": 3, "sea_full": 3},
}
SIZE_ORDER = {name: index for index, name in enumerate(SIZE_CONFIGS)}
VARIANT_SOURCES = {
    "cave_embedding_key": ("cave_embedding",),
    "cave_scalar_key": ("cave_scalar",),
    "sea_full_key": ("sea_full",),
    "cave_embedding_sea_key": ("cave_embedding", "sea_full"),
    "cave_all_sea_key": ("cave_embedding", "cave_scalar", "sea_full"),
}
CAVE_PRIMARY_BLOCKS = (
    "f5_global_mean",
    "f5_vessel_mean",
    "f5_artery_mean",
    "f5_vein_mean",
    "f5_active_vessel_mean",
    "f5_vessel_top10_abs_magnitude",
    "f4_vessel_mean",
    "f4_artery_mean",
    "f4_active_vessel_mean",
    "f4_vessel_top10_abs_magnitude",
)
NONZERO_EPSILON = 1e-10


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_auc(y: np.ndarray, probability: np.ndarray) -> float:
    return float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else float("nan")


def safe_ap(y: np.ndarray, probability: np.ndarray) -> float:
    return float(average_precision_score(y, probability)) if len(np.unique(y)) == 2 else float("nan")


def youden_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    false_positive, true_positive, thresholds = roc_curve(y, probability)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    return float(thresholds[finite][int(np.argmax(true_positive[finite] - false_positive[finite]))])


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
        "task": "adverse_patient",
        "model": model,
        "split": split,
        "rows": int(len(y)),
        "positive": int((y == 1).sum()),
        "negative": int((y == 0).sum()),
        "prevalence": float(np.mean(y)),
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


def cv_splits(y: np.ndarray, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), y))


@dataclass
class Bundle:
    patient_id: np.ndarray
    sea_full: np.ndarray
    cave_embedding: np.ndarray
    cave_scalar: np.ndarray
    missing: np.ndarray
    target: np.ndarray

    def subset(self, indices: np.ndarray) -> "Bundle":
        return Bundle(**{
            field_name: getattr(self, field_name)[indices]
            for field_name in self.__dataclass_fields__
        })


def load_bundle(path: Path) -> Bundle:
    with np.load(path) as raw:
        result = Bundle(
            patient_id=raw["patient_id"].astype(str),
            sea_full=np.array(raw["sea_full"], dtype=np.float32, copy=True),
            cave_embedding=np.array(raw["cave_embedding"], dtype=np.float32, copy=True),
            cave_scalar=np.array(raw["cave_scalar"], dtype=np.float32, copy=True),
            missing=np.array(raw["missing"], dtype=np.float32, copy=True),
            target=np.array(raw["target"], dtype=np.int64, copy=True),
        )
    expected = {
        "sea_full": 319,
        "cave_embedding": 10240,
        "cave_scalar": 658,
        "missing": 2,
    }
    for name, dimension in expected.items():
        if getattr(result, name).shape != (len(result.target), dimension):
            raise AssertionError(f"Unexpected {name} shape: {getattr(result, name).shape}")
    if len(set(result.patient_id.tolist())) != len(result.patient_id):
        raise AssertionError(f"Duplicate patient ID in {path}")
    if not set(np.unique(result.target)).issubset({0, 1}):
        raise AssertionError(f"Nonbinary target in {path}")
    return result


def sea_group(feature_name: str) -> str:
    phase, remainder = feature_name.split("_", 1)
    if remainder == "shape_compatible":
        return f"sea/{phase}/shape_compatible"
    if remainder.startswith("pair_"):
        body = remainder[len("pair_"):]
        families = (
            "active_res",
            "active_weighted",
            "active_direction_coherence",
            "active_direction_entropy",
            "vessel_res",
            "vessel_weighted",
            "vessel_direction_coherence",
            "vessel_direction_entropy",
            "filling_front",
            "persistent",
        )
        family = next((item for item in families if body.startswith(item)), "other")
        return f"sea/{phase}/pair_{family}"
    family = remainder.split("_", 1)[0]
    if family not in {"stage", "tdc", "filling", "coupling"}:
        family = "other"
    return f"sea/{phase}/{family}"


def cave_scalar_group(feature_name: str) -> str:
    if feature_name.startswith(("f4_", "f5_")):
        block = next(
            (item for item in CAVE_PRIMARY_BLOCKS if feature_name.startswith(f"{item}_prepost_")),
            "other_embedding_distance",
        )
        return f"cave_scalar/prepost_distance/{block}"
    phase, remainder = feature_name.split("_", 1)
    if remainder.startswith("active_vessel_tdc_"):
        family = "active_vessel_tdc"
    elif remainder.startswith("artery_tdc_"):
        family = "artery_tdc"
    elif remainder.startswith("artery_"):
        family = "artery_morphology"
    elif remainder.startswith("vein_tdc_"):
        family = "vein_tdc"
    elif remainder.startswith("vein_"):
        family = "vein_morphology"
    elif remainder.startswith("vessel_tdc_"):
        family = "vessel_tdc"
    elif remainder.startswith("vessel_"):
        family = "vessel_morphology"
    elif remainder.startswith("av_"):
        family = "artery_vein_coupling"
    elif remainder.startswith("cave_filling_"):
        family = "cave_filling"
    elif remainder.startswith("cave_kinetic_"):
        family = "cave_kinetic"
    else:
        family = "other"
    return f"cave_scalar/{phase}/{family}"


@dataclass(frozen=True)
class FeatureCatalog:
    names: dict[str, tuple[str, ...]]
    groups: dict[str, tuple[str, ...]]

    @classmethod
    def load(cls, dataset_dir: Path, cave_task_config: Path) -> "FeatureCatalog":
        schema = json.loads((dataset_dir / "feature_schema.json").read_text(encoding="utf-8"))
        sea_names = tuple(schema["feature_sets"]["sea_full"]["columns"])
        task = json.loads(cave_task_config.read_text(encoding="utf-8"))
        scalar_names = tuple(name for name in task["scalar_columns"] if name != "series_count")
        embedding_names = tuple(
            f"cave_{phase}_{block}_ch{channel:03d}"
            for phase in ("pre", "post")
            for block in CAVE_PRIMARY_BLOCKS
            for channel in range(512)
        )
        embedding_groups = tuple(
            f"cave_embedding/{phase}/{block}"
            for phase in ("pre", "post")
            for block in CAVE_PRIMARY_BLOCKS
            for _ in range(512)
        )
        names = {
            "cave_embedding": embedding_names,
            "cave_scalar": scalar_names,
            "sea_full": sea_names,
            "missing": ("missing_pre", "missing_post"),
        }
        groups = {
            "cave_embedding": embedding_groups,
            "cave_scalar": tuple(cave_scalar_group(name) for name in scalar_names),
            "sea_full": tuple(sea_group(name) for name in sea_names),
            "missing": ("missing/phase", "missing/phase"),
        }
        expected = {"cave_embedding": 10240, "cave_scalar": 658, "sea_full": 319, "missing": 2}
        for source, dimension in expected.items():
            if len(names[source]) != dimension or len(groups[source]) != dimension:
                raise AssertionError(f"Unexpected catalog dimension for {source}")
            if len(set(names[source])) != dimension:
                raise AssertionError(f"Duplicate feature names for {source}")
        if len(set(groups["cave_embedding"])) != 20:
            raise AssertionError("CAVE embedding must map to Pre/Post x 10 documented blocks")
        return cls(names=names, groups=groups)

    def frame(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for source in ("cave_embedding", "cave_scalar", "sea_full", "missing"):
            for index, (name, group) in enumerate(zip(self.names[source], self.groups[source])):
                rows.append({
                    "source": source,
                    "source_index": index,
                    "feature_name": name,
                    "group": group,
                })
        return pd.DataFrame(rows)


@dataclass
class SourcePreprocessor:
    kind: str
    fill_values: np.ndarray | None = None
    means: np.ndarray | None = None
    scales: np.ndarray | None = None
    usable: np.ndarray | None = None
    finite_fraction: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "SourcePreprocessor":
        raw = np.asarray(values, dtype=np.float32)
        finite = np.isfinite(raw)
        self.finite_fraction = finite.mean(axis=0).astype(np.float32)
        if self.kind == "embedding":
            fill = np.zeros(raw.shape[1], dtype=np.float32)
            filled = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
            coverage = np.ones(raw.shape[1], dtype=bool)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                fill = np.nanmedian(np.where(finite, raw, np.nan), axis=0).astype(np.float32)
            fill[~np.isfinite(fill)] = 0.0
            filled = np.where(finite, raw, fill[None, :]).astype(np.float32)
            coverage = self.finite_fraction >= 0.25
        means = np.mean(filled, axis=0, dtype=np.float64).astype(np.float32)
        scales = np.std(filled, axis=0, dtype=np.float64).astype(np.float32)
        usable = coverage & np.isfinite(scales) & (scales > 1e-6)
        safe_scales = scales.copy()
        safe_scales[~usable] = 1.0
        self.fill_values = fill
        self.means = means
        self.scales = safe_scales
        self.usable = usable
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if any(item is None for item in (self.fill_values, self.means, self.scales, self.usable)):
            raise RuntimeError("Source preprocessor not fitted")
        raw = np.asarray(values, dtype=np.float32)
        finite = np.isfinite(raw)
        filled = np.where(finite, raw, self.fill_values[None, :]).astype(np.float32)
        transformed = ((filled - self.means[None, :]) / self.scales[None, :]).astype(np.float32)
        transformed[:, ~self.usable] = 0.0
        if not np.isfinite(transformed).all():
            raise AssertionError("Nonfinite transformed feature")
        return transformed

    def audit(self) -> dict[str, Any]:
        if self.usable is None or self.finite_fraction is None:
            raise RuntimeError("Source preprocessor not fitted")
        return {
            "kind": self.kind,
            "input_columns": int(len(self.usable)),
            "usable_columns": int(self.usable.sum()),
            "constant_or_low_coverage_columns": int((~self.usable).sum()),
            "minimum_finite_fraction": float(np.min(self.finite_fraction)),
            "pca": False,
            "random_projection": False,
        }


def source_values(data: Bundle, source: str) -> np.ndarray:
    return np.asarray(getattr(data, source), dtype=np.float32)


def fit_preprocessors(data: Bundle) -> dict[str, SourcePreprocessor]:
    return {
        "cave_embedding": SourcePreprocessor("embedding").fit(data.cave_embedding),
        "cave_scalar": SourcePreprocessor("numeric").fit(data.cave_scalar),
        "sea_full": SourcePreprocessor("numeric").fit(data.sea_full),
        "missing": SourcePreprocessor("numeric").fit(data.missing),
    }


def transform_sources(
    preprocessors: dict[str, SourcePreprocessor], data: Bundle
) -> dict[str, np.ndarray]:
    return {
        source: preprocessor.transform(source_values(data, source))
        for source, preprocessor in preprocessors.items()
    }


def standardized_effect_scores(
    transformed: np.ndarray, y: np.ndarray, usable: np.ndarray
) -> np.ndarray:
    negative = transformed[y == 0]
    positive = transformed[y == 1]
    mean_difference = np.mean(positive, axis=0) - np.mean(negative, axis=0)
    pooled_variance = 0.5 * (np.var(positive, axis=0) + np.var(negative, axis=0))
    score = np.abs(mean_difference) / np.sqrt(pooled_variance + 1e-8)
    score = np.asarray(score, dtype=np.float64)
    score[~usable] = -np.inf
    score[~np.isfinite(score)] = -np.inf
    return score


def rank_groups(
    catalog: FeatureCatalog,
    transformed: dict[str, np.ndarray],
    preprocessors: dict[str, SourcePreprocessor],
    y: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, tuple[int, ...]]]]:
    scores: dict[str, np.ndarray] = {}
    rankings: dict[str, dict[str, tuple[int, ...]]] = {}
    for source in ("cave_embedding", "cave_scalar", "sea_full"):
        usable = preprocessors[source].usable
        if usable is None:
            raise RuntimeError("Missing usable mask")
        source_scores = standardized_effect_scores(transformed[source], y, usable)
        scores[source] = source_scores
        grouped: dict[str, list[int]] = {}
        for index, group in enumerate(catalog.groups[source]):
            if np.isfinite(source_scores[index]):
                grouped.setdefault(group, []).append(index)
        rankings[source] = {}
        for group, indices in grouped.items():
            ordered = sorted(indices, key=lambda index: (-source_scores[index], index))
            rankings[source][group] = tuple(ordered)
    return scores, rankings


def select_group_quota(
    rankings: dict[str, tuple[int, ...]], quota: int
) -> np.ndarray:
    selected: list[int] = []
    for group in sorted(rankings):
        selected.extend(rankings[group][:quota])
    return np.asarray(selected, dtype=np.int64)


def selections_for_size(
    rankings: dict[str, dict[str, tuple[int, ...]]], size_name: str
) -> dict[str, np.ndarray]:
    quotas = SIZE_CONFIGS[size_name]
    return {
        source: select_group_quota(rankings[source], quotas[source])
        for source in ("cave_embedding", "cave_scalar", "sea_full")
    }


def compact_matrix(
    transformed: dict[str, np.ndarray],
    selections: dict[str, np.ndarray],
    variant: str,
) -> np.ndarray:
    parts = [transformed[source][:, selections[source]] for source in VARIANT_SOURCES[variant]]
    parts.append(transformed["missing"])
    result = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise AssertionError(f"Nonfinite compact matrix for {variant}")
    return result


def compact_feature_metadata(
    catalog: FeatureCatalog,
    scores: dict[str, np.ndarray],
    selections: dict[str, np.ndarray],
    variant: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in VARIANT_SOURCES[variant]:
        for index in selections[source]:
            rows.append({
                "feature_name": catalog.names[source][int(index)],
                "source": source,
                "group": catalog.groups[source][int(index)],
                "source_index": int(index),
                "screening_score": float(scores[source][int(index)]),
            })
    for index in range(2):
        rows.append({
            "feature_name": catalog.names["missing"][index],
            "source": "missing",
            "group": catalog.groups["missing"][index],
            "source_index": index,
            "screening_score": float("nan"),
        })
    return rows


def logistic_model(c_value: float) -> LogisticRegression:
    return LogisticRegression(
        C=float(c_value),
        class_weight="balanced",
        penalty="l1",
        solver="liblinear",
        max_iter=10000,
        tol=1e-4,
        random_state=SEED,
    )


class FitAudit:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows: list[dict[str, Any]] = []

    def add(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        if len(self.rows) % 25 == 0:
            self.flush()

    def flush(self) -> None:
        if self.rows:
            atomic_csv(pd.DataFrame(self.rows), self.path)


def fit_checked(
    x: np.ndarray,
    y: np.ndarray,
    c_value: float,
    context: dict[str, Any],
    audit: FitAudit,
    evaluation_x: np.ndarray | None = None,
    evaluation_y: np.ndarray | None = None,
) -> LogisticRegression:
    model = logistic_model(c_value)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x, y)
    elapsed = time.perf_counter() - start
    convergence = [item for item in caught if issubclass(item.category, ConvergenceWarning)]
    coefficient = np.asarray(model.coef_[0], dtype=np.float64)
    probability = model.predict_proba(evaluation_x)[:, 1] if evaluation_x is not None else None
    audit.add({
        **context,
        "C": float(c_value),
        "rows": int(x.shape[0]),
        "columns": int(x.shape[1]),
        "n_iter": int(np.asarray(model.n_iter_).max()),
        "fit_seconds": float(elapsed),
        "nonzero_coefficients": int((np.abs(coefficient) > NONZERO_EPSILON).sum()),
        "coefficient_l1_norm": float(np.abs(coefficient).sum()),
        "coefficient_max_abs": float(np.abs(coefficient).max(initial=0.0)),
        "convergence_warning": bool(convergence),
        "warning_text": " || ".join(str(item.message) for item in caught),
        "evaluation_auroc": (
            safe_auc(evaluation_y, probability)
            if probability is not None and evaluation_y is not None else float("nan")
        ),
        "evaluation_auprc": (
            safe_ap(evaluation_y, probability)
            if probability is not None and evaluation_y is not None else float("nan")
        ),
    })
    if convergence:
        audit.flush()
        raise RuntimeError(f"Logistic convergence warning: {context}")
    return model


def select_hyperparameters(
    data: Bundle,
    catalog: FeatureCatalog,
    outer_fold: int,
    audit: FitAudit,
    inner_score_rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    predictions = {
        variant: {
            (size_name, c_value): np.full(len(data.target), np.nan, dtype=np.float64)
            for size_name in SIZE_CONFIGS
            for c_value in C_GRID
        }
        for variant in VARIANT_SOURCES
    }
    inner_audits: list[dict[str, Any]] = []
    for inner_fold, (fit_index, holdout_index) in enumerate(
        cv_splits(data.target, 3, SEED + outer_fold * 1000), start=1
    ):
        fit_data = data.subset(fit_index)
        holdout_data = data.subset(holdout_index)
        preprocessors = fit_preprocessors(fit_data)
        fit_transformed = transform_sources(preprocessors, fit_data)
        holdout_transformed = transform_sources(preprocessors, holdout_data)
        scores, rankings = rank_groups(catalog, fit_transformed, preprocessors, fit_data.target)
        inner_audits.append({
            "inner_fold": inner_fold,
            "preprocessors": {source: item.audit() for source, item in preprocessors.items()},
            "group_counts": {source: len(value) for source, value in rankings.items()},
        })
        for size_name in SIZE_CONFIGS:
            selections = selections_for_size(rankings, size_name)
            for variant in VARIANT_SOURCES:
                fit_x = compact_matrix(fit_transformed, selections, variant)
                holdout_x = compact_matrix(holdout_transformed, selections, variant)
                for c_value in C_GRID:
                    model = fit_checked(
                        fit_x,
                        fit_data.target,
                        c_value,
                        {
                            "stage": "inner_cv",
                            "outer_fold": outer_fold,
                            "inner_fold": inner_fold,
                            "variant": variant,
                            "size_name": size_name,
                        },
                        audit,
                        holdout_x,
                        holdout_data.target,
                    )
                    predictions[variant][(size_name, c_value)][holdout_index] = model.predict_proba(
                        holdout_x
                    )[:, 1]
        del fit_transformed, holdout_transformed

    selected: dict[str, dict[str, Any]] = {}
    for variant in VARIANT_SOURCES:
        candidates: list[dict[str, Any]] = []
        for size_name in SIZE_CONFIGS:
            for c_value in C_GRID:
                probability = predictions[variant][(size_name, c_value)]
                if not np.isfinite(probability).all():
                    raise AssertionError(f"Incomplete inner prediction for {variant}")
                row = {
                    "outer_fold": outer_fold,
                    "variant": variant,
                    "size_name": size_name,
                    "C": float(c_value),
                    "inner_oof_auprc": safe_ap(data.target, probability),
                    "inner_oof_auroc": safe_auc(data.target, probability),
                    "selected": False,
                }
                candidates.append(row)
        best = max(
            candidates,
            key=lambda row: (
                row["inner_oof_auprc"],
                row["inner_oof_auroc"],
                -SIZE_ORDER[row["size_name"]],
                -row["C"],
            ),
        )
        best["selected"] = True
        selected[variant] = {"size_name": best["size_name"], "C": best["C"]}
        inner_score_rows.extend(candidates)
    return selected, {"inner_folds": inner_audits}


def fit_outer_or_full(
    development: Bundle,
    evaluation: Bundle | None,
    valid: Bundle | None,
    catalog: FeatureCatalog,
    hyperparameters: dict[str, dict[str, Any]],
    stage: str,
    fold: int,
    output_dir: Path,
    audit: FitAudit,
    occurrence_rows: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    preprocessors = fit_preprocessors(development)
    development_transformed = transform_sources(preprocessors, development)
    evaluation_transformed = transform_sources(preprocessors, evaluation) if evaluation is not None else None
    valid_transformed = transform_sources(preprocessors, valid) if valid is not None else None
    scores, rankings = rank_groups(catalog, development_transformed, preprocessors, development.target)
    artifact_dir = output_dir / (f"fold_{fold}" if stage == "outer_refit" else "full_train")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessors, artifact_dir / "source_preprocessors.joblib")
    evaluation_predictions: dict[str, np.ndarray] = {}
    valid_predictions: dict[str, np.ndarray] = {}
    variant_audit: dict[str, Any] = {}
    selection_cache = {
        size_name: selections_for_size(rankings, size_name) for size_name in SIZE_CONFIGS
    }
    for variant, selected in hyperparameters.items():
        size_name = str(selected["size_name"])
        c_value = float(selected["C"])
        selections = selection_cache[size_name]
        development_x = compact_matrix(development_transformed, selections, variant)
        evaluation_x = (
            compact_matrix(evaluation_transformed, selections, variant)
            if evaluation_transformed is not None else None
        )
        valid_x = (
            compact_matrix(valid_transformed, selections, variant)
            if valid_transformed is not None else None
        )
        model = fit_checked(
            development_x,
            development.target,
            c_value,
            {
                "stage": stage,
                "outer_fold": fold,
                "inner_fold": 0,
                "variant": variant,
                "size_name": size_name,
            },
            audit,
            evaluation_x,
            evaluation.target if evaluation is not None else None,
        )
        metadata = compact_feature_metadata(catalog, scores, selections, variant)
        coefficient = np.asarray(model.coef_[0], dtype=np.float64)
        if len(metadata) != len(coefficient):
            raise AssertionError(f"Coefficient mapping mismatch for {variant}")
        for item, value in zip(metadata, coefficient):
            occurrence_rows.append({
                "stage": stage,
                "fold": fold,
                "variant": variant,
                **item,
                "coefficient": float(value),
                "nonzero": bool(abs(value) > NONZERO_EPSILON),
                "positive": bool(value > NONZERO_EPSILON),
                "negative": bool(value < -NONZERO_EPSILON),
            })
        if evaluation_x is not None:
            evaluation_predictions[variant] = model.predict_proba(evaluation_x)[:, 1]
        if valid_x is not None:
            # Deliberately do not access valid.target here.
            valid_predictions[variant] = model.predict_proba(valid_x)[:, 1]
        artifact = {
            "model": model,
            "variant": variant,
            "sources": VARIANT_SOURCES[variant],
            "size_name": size_name,
            "C": c_value,
            "features": metadata,
            "valid_labels_used": False,
        }
        joblib.dump(artifact, artifact_dir / f"sparse_{variant}.joblib")
        variant_audit[variant] = {
            "size_name": size_name,
            "C": c_value,
            "compact_dimension": len(metadata),
            "nonzero_coefficients": int((np.abs(coefficient) > NONZERO_EPSILON).sum()),
        }
    fit_audit = {
        "stage": stage,
        "fold": fold,
        "preprocessors": {source: item.audit() for source, item in preprocessors.items()},
        "groups_with_usable_features": {source: len(value) for source, value in rankings.items()},
        "variants": variant_audit,
    }
    atomic_json(fit_audit, artifact_dir / "fit_audit.json")
    return evaluation_predictions, valid_predictions, fit_audit


def aggregate_stability(
    occurrence_frame: pd.DataFrame,
    output_dir: Path,
    outer_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outer = occurrence_frame[occurrence_frame["stage"] == "outer_refit"].copy()
    full = occurrence_frame[occurrence_frame["stage"] == "full_train_refit"].copy()
    rows: list[dict[str, Any]] = []
    for (variant, feature_name), group in outer.groupby(["variant", "feature_name"], sort=False):
        nonzero = group[group["nonzero"]]
        positive = int(group["positive"].sum())
        negative = int(group["negative"].sum())
        nonzero_count = positive + negative
        full_match = full[(full["variant"] == variant) & (full["feature_name"] == feature_name)]
        consistency = max(positive, negative) / nonzero_count if nonzero_count else 0.0
        row = {
            "variant": variant,
            "feature_name": feature_name,
            "source": str(group.iloc[0]["source"]),
            "group": str(group.iloc[0]["group"]),
            "outer_selection_count": int(group["fold"].nunique()),
            "outer_selection_fraction": float(group["fold"].nunique() / outer_folds),
            "outer_nonzero_count": nonzero_count,
            "outer_positive_count": positive,
            "outer_negative_count": negative,
            "direction_consistency": float(consistency),
            "mean_screening_score_when_selected": float(group["screening_score"].mean()),
            "mean_abs_coefficient_when_selected": float(group["coefficient"].abs().mean()),
            "mean_abs_coefficient_across_outer_folds": float(group["coefficient"].abs().sum() / outer_folds),
            "full_train_selected": bool(len(full_match)),
            "full_train_nonzero": bool(full_match["nonzero"].any()) if len(full_match) else False,
            "full_train_coefficient": float(full_match.iloc[0]["coefficient"]) if len(full_match) else 0.0,
        }
        row["stable_key_feature"] = bool(
            row["outer_selection_count"] >= 4
            and row["outer_nonzero_count"] >= 3
            and row["direction_consistency"] >= 0.8
        )
        rows.append(row)
    stability = pd.DataFrame(rows)
    if not stability.empty:
        stability = stability.sort_values(
            [
                "variant",
                "stable_key_feature",
                "outer_nonzero_count",
                "outer_selection_count",
                "mean_abs_coefficient_across_outer_folds",
            ],
            ascending=[True, False, False, False, False],
        ).reset_index(drop=True)
        stability["importance_rank_within_variant"] = (
            stability.groupby("variant").cumcount() + 1
        )
    atomic_csv(stability, output_dir / "stable_feature_importance.csv")

    group_rows: list[dict[str, Any]] = []
    stable_lookup = stability.set_index(["variant", "feature_name"])["stable_key_feature"] if len(stability) else None
    for (variant, source, feature_group), group in outer.groupby(
        ["variant", "source", "group"], sort=False
    ):
        stable_count = 0
        if stable_lookup is not None:
            for name in group["feature_name"].unique():
                stable_count += int(bool(stable_lookup.get((variant, name), False)))
        group_rows.append({
            "variant": variant,
            "source": source,
            "group": feature_group,
            "outer_folds_present": int(group["fold"].nunique()),
            "unique_screened_features": int(group["feature_name"].nunique()),
            "screened_occurrences": int(len(group)),
            "nonzero_occurrences": int(group["nonzero"].sum()),
            "positive_occurrences": int(group["positive"].sum()),
            "negative_occurrences": int(group["negative"].sum()),
            "mean_abs_coefficient_across_occurrences": float(group["coefficient"].abs().mean()),
            "stable_key_feature_count": stable_count,
        })
    group_importance = pd.DataFrame(group_rows).sort_values(
        ["variant", "stable_key_feature_count", "nonzero_occurrences", "mean_abs_coefficient_across_occurrences"],
        ascending=[True, False, False, False],
    )
    atomic_csv(group_importance, output_dir / "group_importance.csv")
    return stability, group_importance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--cave-task-config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
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
    dataset_schema = json.loads((dataset_dir / "feature_schema.json").read_text(encoding="utf-8"))
    if dataset_schema.get("metadata_predictor_columns"):
        raise AssertionError("Clinical metadata predictors entered the image probe")
    catalog = FeatureCatalog.load(dataset_dir, cave_task_config)
    atomic_csv(catalog.frame(), output_dir / "feature_catalog.csv")

    audit = FitAudit(output_dir / "logistic_fit_audit.csv")
    inner_score_rows: list[dict[str, Any]] = []
    occurrence_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    outer_splits = cv_splits(train.target, 5, SEED)
    oof = {
        variant: np.full(len(train.target), np.nan, dtype=np.float64)
        for variant in VARIANT_SOURCES
    }
    valid_fold_predictions = {variant: [] for variant in VARIANT_SOURCES}

    for fold, (development_index, holdout_index) in enumerate(outer_splits, start=1):
        print(f"[OUTER FOLD START] {fold}/5", flush=True)
        development = train.subset(development_index)
        holdout = train.subset(holdout_index)
        selected, inner_audit = select_hyperparameters(
            development, catalog, fold, audit, inner_score_rows
        )
        holdout_prediction, valid_prediction, fit_audit = fit_outer_or_full(
            development=development,
            evaluation=holdout,
            valid=valid,
            catalog=catalog,
            hyperparameters=selected,
            stage="outer_refit",
            fold=fold,
            output_dir=output_dir,
            audit=audit,
            occurrence_rows=occurrence_rows,
        )
        for variant in VARIANT_SOURCES:
            oof[variant][holdout_index] = holdout_prediction[variant]
            valid_fold_predictions[variant].append(valid_prediction[variant])
            fold_rows.append({
                "fold": fold,
                "variant": variant,
                "development_rows": int(len(development_index)),
                "holdout_rows": int(len(holdout_index)),
                "selected_size": selected[variant]["size_name"],
                "selected_C": selected[variant]["C"],
                "compact_dimension": fit_audit["variants"][variant]["compact_dimension"],
                "nonzero_coefficients": fit_audit["variants"][variant]["nonzero_coefficients"],
                "holdout_auroc": safe_auc(holdout.target, holdout_prediction[variant]),
                "holdout_auprc": safe_ap(holdout.target, holdout_prediction[variant]),
                "inner_preprocessing_audit": json.dumps(inner_audit, sort_keys=True),
            })
        audit.flush()
        atomic_csv(pd.DataFrame(inner_score_rows), output_dir / "inner_model_selection.csv")
        atomic_csv(pd.DataFrame(occurrence_rows), output_dir / "feature_occurrences.csv")
        atomic_csv(pd.DataFrame(fold_rows), output_dir / "fold_audit.csv")
        print(f"[OUTER FOLD DONE] {fold}/5", flush=True)

    full_selected, full_inner_audit = select_hyperparameters(
        train, catalog, 0, audit, inner_score_rows
    )
    fit_outer_or_full(
        development=train,
        evaluation=None,
        valid=None,
        catalog=catalog,
        hyperparameters=full_selected,
        stage="full_train_refit",
        fold=0,
        output_dir=output_dir,
        audit=audit,
        occurrence_rows=occurrence_rows,
    )
    audit.flush()
    atomic_csv(pd.DataFrame(inner_score_rows), output_dir / "inner_model_selection.csv")
    occurrence_frame = pd.DataFrame(occurrence_rows)
    atomic_csv(occurrence_frame, output_dir / "feature_occurrences.csv")
    stability, group_importance = aggregate_stability(occurrence_frame, output_dir, 5)

    probability_sets: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "Dummy": (
            np.full(len(train.target), train.target.mean(), dtype=np.float64),
            np.full(len(valid.target), train.target.mean(), dtype=np.float64),
        )
    }
    for variant in VARIANT_SOURCES:
        if not np.isfinite(oof[variant]).all():
            raise AssertionError(f"Incomplete OOF predictions: {variant}")
        valid_probability = np.mean(np.stack(valid_fold_predictions[variant]), axis=0)
        if not np.isfinite(valid_probability).all():
            raise AssertionError(f"Nonfinite Valid predictions: {variant}")
        probability_sets[f"StableSparse_{variant}"] = (oof[variant], valid_probability)

    metrics: list[dict[str, Any]] = []
    thresholds: dict[str, float] = {}
    train_predictions = pd.DataFrame({"patient_id": train.patient_id, "target": train.target})
    valid_predictions = pd.DataFrame({"patient_id": valid.patient_id})
    for model_name, (train_probability, valid_probability) in probability_sets.items():
        threshold = youden_threshold(train.target, train_probability)
        thresholds[model_name] = threshold
        train_predictions[f"{model_name.lower()}_probability"] = train_probability
        valid_predictions[f"{model_name.lower()}_probability"] = valid_probability
        metrics.append(metric_row(model_name, "Train_OOF", train.target, train_probability, threshold))
    # Valid targets enter only here, after preprocessing, screening, tuning,
    # thresholds, models, and prediction files have all been determined.
    valid_predictions["target"] = valid.target
    for model_name, (_, valid_probability) in probability_sets.items():
        metrics.append(metric_row(model_name, "Valid", valid.target, valid_probability, thresholds[model_name]))

    metrics_frame = pd.DataFrame(metrics)
    atomic_csv(metrics_frame, output_dir / "metrics.csv")
    atomic_csv(train_predictions, output_dir / "train_oof_predictions.csv")
    atomic_csv(valid_predictions, output_dir / "valid_predictions.csv")
    atomic_csv(pd.DataFrame(fold_rows), output_dir / "fold_audit.csv")

    learned_oof = metrics_frame[
        (metrics_frame["split"] == "Train_OOF") & (metrics_frame["model"] != "Dummy")
    ].sort_values(["auprc", "auroc"], ascending=False)
    best_model = str(learned_oof.iloc[0]["model"])
    fit_audit_frame = pd.DataFrame(audit.rows)
    convergence_warnings = int(fit_audit_frame["convergence_warning"].sum())
    if convergence_warnings:
        raise AssertionError("Convergence warnings present in successful run")
    stable_counts = (
        stability.groupby("variant")["stable_key_feature"].sum().astype(int).to_dict()
        if len(stability) else {}
    )
    summary = {
        "version": "api_fullseq_image_probe_v3_stable_sparse_1",
        "task": "patient-level adverse outcome",
        "predictors": "frozen image-derived CAVE and SEA-RAFT features only",
        "train_rows": int(len(train.target)),
        "train_positive": int(train.target.sum()),
        "valid_rows": int(len(valid.target)),
        "valid_positive": int(valid.target.sum()),
        "train_valid_patient_overlap": 0,
        "outer_folds": 5,
        "inner_folds": 3,
        "screening": "within-fit-fold balanced group quota by absolute standardized effect",
        "size_configs": SIZE_CONFIGS,
        "c_grid": list(C_GRID),
        "classifier": "class-balanced L1 LogisticRegression",
        "best_model_selected_by_train_oof_auprc": best_model,
        "thresholds_from_train_oof": thresholds,
        "full_train_hyperparameters": full_selected,
        "full_train_inner_audit": full_inner_audit,
        "stable_key_feature_counts": stable_counts,
        "logistic_fit_count": int(len(fit_audit_frame)),
        "convergence_warning_count": convergence_warnings,
        "pca": False,
        "random_projection": False,
        "mlp": False,
        "clinical_predictors": False,
        "valid_used_for_preprocessing_screening_tuning_threshold_or_early_stopping": False,
        "valid_labels_used_for": "final metrics only after predictions",
        "seed": SEED,
        "input_sha256": {
            "train.npz": sha256(dataset_dir / "train.npz"),
            "valid.npz": sha256(dataset_dir / "valid.npz"),
            "feature_schema.json": sha256(dataset_dir / "feature_schema.json"),
            "cave_task_config.json": sha256(cave_task_config),
        },
    }
    atomic_json(summary, output_dir / "summary.json")
    atomic_json(summary, output_dir / ".MODELS_SUCCESS")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

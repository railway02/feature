#!/usr/bin/env python3
"""Train adverse-outcome models from CAVE/SEA key representations and metadata.

No unsupervised PCA or random projection is used.  The complete CAVE embedding
blocks are consumed by a shared supervised block encoder, and SEA dense maps by
a supervised spatial CNN.  Clinical preprocessing and neural early stopping
are fitted inside each Train development fold.  Official Valid is prediction
only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


SEED = 42
VARIANTS = (
    "clinical",
    "cave_clinical",
    "searaft_clinical",
    "cave_searaft",
    "full_keyfusion",
)


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
    fpr, tpr, thresholds = roc_curve(y, probability)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    return float(thresholds[finite][int(np.argmax(tpr[finite] - fpr[finite]))])


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


@dataclass
class ClinicalPreprocessor:
    imputer: SimpleImputer | None = None
    scaler: StandardScaler | None = None

    def fit(self, values: np.ndarray) -> "ClinicalPreprocessor":
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        transformed = self.imputer.fit_transform(np.asarray(values, dtype=np.float64))
        self.scaler = StandardScaler()
        self.scaler.fit(transformed)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.imputer is None or self.scaler is None:
            raise RuntimeError("Clinical preprocessor not fitted")
        transformed = self.imputer.transform(np.asarray(values, dtype=np.float64))
        output = self.scaler.transform(transformed).astype(np.float32)
        if not np.isfinite(output).all():
            raise AssertionError("Clinical features contain nonfinite values")
        return output


@dataclass
class DataBundle:
    patient_id: np.ndarray
    cave: np.ndarray
    sea: np.ndarray
    clinical: np.ndarray
    missing: np.ndarray
    target: np.ndarray

    def subset(self, indices: np.ndarray) -> "DataBundle":
        return DataBundle(
            patient_id=self.patient_id[indices],
            cave=self.cave[indices],
            sea=self.sea[indices],
            clinical=self.clinical[indices],
            missing=self.missing[indices],
            target=self.target[indices],
        )


def load_bundle(path: Path) -> DataBundle:
    with np.load(path) as raw:
        bundle = DataBundle(
            patient_id=raw["patient_id"].astype(str),
            cave=np.array(raw["cave"], dtype=np.float32, copy=True),
            sea=np.array(raw["sea"], dtype=np.float32, copy=True),
            clinical=np.array(raw["clinical"], dtype=np.float32, copy=True),
            missing=np.array(raw["missing"], dtype=np.float32, copy=True),
            target=np.array(raw["target"], dtype=np.int64, copy=True),
        )
    if bundle.cave.ndim != 4 or bundle.cave.shape[1:] != (2, 10, 512):
        raise AssertionError(f"Unexpected CAVE tensor {bundle.cave.shape}")
    if bundle.sea.ndim != 5 or bundle.sea.shape[1:] != (2, 70, 16, 16):
        raise AssertionError(f"Unexpected SEA tensor {bundle.sea.shape}")
    if bundle.missing.shape != (len(bundle.target), 4):
        raise AssertionError(f"Unexpected missing tensor {bundle.missing.shape}")
    if bundle.patient_id.shape[0] != len(bundle.target):
        raise AssertionError("Patient/target row mismatch")
    if len(set(bundle.patient_id.tolist())) != len(bundle.patient_id):
        raise AssertionError("Duplicate patient_id")
    if not set(np.unique(bundle.target)).issubset({0, 1}):
        raise AssertionError("Non-binary target")
    bundle.cave = np.nan_to_num(bundle.cave, nan=0.0, posinf=0.0, neginf=0.0)
    bundle.sea = np.nan_to_num(bundle.sea, nan=0.0, posinf=0.0, neginf=0.0)
    return bundle


class CaveEncoder(nn.Module):
    """Shared supervised projection over all 20 phase/block CAVE vectors."""

    def __init__(self) -> None:
        super().__init__()
        self.channel_norm = nn.LayerNorm(512)
        self.shared_projection = nn.Sequential(
            nn.Linear(512, 16),
            nn.GELU(),
            nn.Dropout(0.20),
        )
        self.block_fusion = nn.Sequential(
            nn.Linear(2 * 10 * 16, 128),
            nn.GELU(),
            nn.Dropout(0.35),
        )

    def forward(self, cave: torch.Tensor) -> torch.Tensor:
        projected = self.shared_projection(self.channel_norm(cave))
        return self.block_fusion(projected.flatten(1))


class SeaEncoder(nn.Module):
    """Supervised CNN over all key SEA-RAFT map summaries."""

    def __init__(self) -> None:
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(2 * 70, 32, kernel_size=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 128),
            nn.GELU(),
            nn.Dropout(0.35),
        )

    def forward(self, sea: torch.Tensor) -> torch.Tensor:
        batch = sea.shape[0]
        merged = sea.reshape(batch, 2 * 70, 16, 16)
        return self.output(self.spatial(merged))


class ClinicalEncoder(nn.Module):
    def __init__(self, input_dimension: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimension, 32),
            nn.GELU(),
            nn.Dropout(0.20),
        )

    def forward(self, clinical: torch.Tensor) -> torch.Tensor:
        return self.network(clinical)


class KeyFusionNet(nn.Module):
    def __init__(self, variant: str, clinical_dimension: int) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.use_cave = variant in {"cave_clinical", "cave_searaft", "full_keyfusion"}
        self.use_sea = variant in {"searaft_clinical", "cave_searaft", "full_keyfusion"}
        self.use_clinical = variant in {
            "clinical", "cave_clinical", "searaft_clinical", "full_keyfusion"
        }
        self.cave_encoder = CaveEncoder() if self.use_cave else None
        self.sea_encoder = SeaEncoder() if self.use_sea else None
        self.clinical_encoder = (
            ClinicalEncoder(clinical_dimension) if self.use_clinical else None
        )
        dimension = 128 * int(self.use_cave) + 128 * int(self.use_sea)
        dimension += 32 * int(self.use_clinical)
        if self.use_cave and self.use_sea:
            self.missing_indices = (0, 1, 2, 3)
        elif self.use_cave:
            self.missing_indices = (0, 1)
        elif self.use_sea:
            self.missing_indices = (2, 3)
        else:
            self.missing_indices = ()
        dimension += len(self.missing_indices)
        self.head = nn.Sequential(
            nn.Linear(dimension, 64),
            nn.GELU(),
            nn.Dropout(0.40),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        cave: torch.Tensor,
        sea: torch.Tensor,
        clinical: torch.Tensor,
        missing: torch.Tensor,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if self.cave_encoder is not None:
            parts.append(self.cave_encoder(cave))
        if self.sea_encoder is not None:
            parts.append(self.sea_encoder(sea))
        if self.clinical_encoder is not None:
            parts.append(self.clinical_encoder(clinical))
        if self.missing_indices:
            parts.append(missing[:, self.missing_indices])
        return self.head(torch.cat(parts, dim=1)).squeeze(1)


class BundleDataset(torch.utils.data.Dataset):
    def __init__(self, bundle: DataBundle, clinical: np.ndarray) -> None:
        self.cave = torch.from_numpy(bundle.cave.astype(np.float32, copy=False))
        self.sea = torch.from_numpy(bundle.sea.astype(np.float32, copy=False))
        self.clinical = torch.from_numpy(clinical.astype(np.float32, copy=False))
        self.missing = torch.from_numpy(bundle.missing.astype(np.float32, copy=False))
        self.target = torch.from_numpy(bundle.target.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int):
        return (
            self.cave[index],
            self.sea[index],
            self.clinical[index],
            self.missing[index],
            self.target[index],
        )


def predict(
    model: KeyFusionNet,
    bundle: DataBundle,
    clinical: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    loader = torch.utils.data.DataLoader(
        BundleDataset(bundle, clinical), batch_size=batch_size, shuffle=False
    )
    output: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for cave, sea, clinical_x, missing, _ in loader:
            logits = model(
                cave.to(device),
                sea.to(device),
                clinical_x.to(device),
                missing.to(device),
            )
            output.append(torch.sigmoid(logits).cpu().numpy())
    probability = np.concatenate(output).astype(np.float64)
    if not np.isfinite(probability).all():
        raise AssertionError("Nonfinite model probabilities")
    return probability


def early_stop_indices(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    fit_index, stop_index = next(splitter.split(np.zeros(len(y)), y))
    return fit_index, stop_index


@dataclass
class FitResult:
    model: KeyFusionNet
    preprocessor: ClinicalPreprocessor
    best_epoch: int
    best_ap: float
    history: list[dict[str, float]]


def train_with_early_stopping(
    variant: str,
    development: DataBundle,
    device: torch.device,
    seed: int,
    max_epochs: int = 200,
    patience: int = 20,
) -> FitResult:
    set_seed(seed)
    fit_index, stop_index = early_stop_indices(development.target, seed)
    fit = development.subset(fit_index)
    stop = development.subset(stop_index)
    preprocessor = ClinicalPreprocessor().fit(fit.clinical)
    fit_clinical = preprocessor.transform(fit.clinical)
    stop_clinical = preprocessor.transform(stop.clinical)
    model = KeyFusionNet(variant, fit_clinical.shape[1]).to(device)
    positive = max(int((fit.target == 1).sum()), 1)
    negative = max(int((fit.target == 0).sum()), 1)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negative / positive], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-3
    )
    loader = torch.utils.data.DataLoader(
        BundleDataset(fit, fit_clinical),
        batch_size=min(32, len(fit.target)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_ap = -math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses: list[float] = []
        for cave, sea, clinical, missing, target in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                cave.to(device),
                sea.to(device),
                clinical.to(device),
                missing.to(device),
            )
            loss = loss_fn(logits, target.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        stop_probability = predict(model, stop, stop_clinical, device)
        stop_ap = safe_ap(stop.target, stop_probability)
        history.append({
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)),
            "early_stop_auprc": stop_ap,
            "early_stop_auroc": safe_auc(stop.target, stop_probability),
        })
        if stop_ap > best_ap + 1e-5:
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_ap = stop_ap
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError(f"{variant}: no MLP checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    return FitResult(model, preprocessor, best_epoch, float(best_ap), history)


def train_fixed_epochs(
    variant: str,
    train: DataBundle,
    preprocessor: ClinicalPreprocessor,
    epochs: int,
    device: torch.device,
    seed: int,
) -> KeyFusionNet:
    set_seed(seed)
    clinical = preprocessor.transform(train.clinical)
    model = KeyFusionNet(variant, clinical.shape[1]).to(device)
    positive = max(int((train.target == 1).sum()), 1)
    negative = max(int((train.target == 0).sum()), 1)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negative / positive], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-3
    )
    loader = torch.utils.data.DataLoader(
        BundleDataset(train, clinical),
        batch_size=min(32, len(train.target)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    for _ in range(max(1, epochs)):
        model.train()
        for cave, sea, clinical_x, missing, target in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                cave.to(device),
                sea.to(device),
                clinical_x.to(device),
                missing.to(device),
            )
            loss = loss_fn(logits, target.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
    return model


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def previous_comparison(current: pd.DataFrame, previous_path: Path) -> pd.DataFrame:
    rows = current[current["split"] == "Valid"].copy()
    rows["source_run"] = "keyfusion_v3"
    if previous_path.is_file():
        previous = pd.read_csv(previous_path)
        previous = previous[
            (previous["task"] == "adverse_patient")
            & (previous["split"] == "Valid")
            & previous["model"].isin([
                "Logistic_searaft",
                "Logistic_cave_deep",
                "Logistic_cave_fusion",
                "Logistic_multimodal_fusion",
                "MLP_multimodal_fusion",
            ])
        ].copy()
        previous["source_run"] = "previous_feature_fusion"
        rows = pd.concat([rows, previous[rows.columns]], ignore_index=True)
    return rows.sort_values(["auprc", "auroc"], ascending=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    train = load_bundle(dataset_dir / "train.npz")
    valid = load_bundle(dataset_dir / "valid.npz")
    overlap = set(train.patient_id) & set(valid.patient_id)
    if overlap:
        raise AssertionError(f"Train/Valid patient overlap={len(overlap)}")
    schema = json.loads((dataset_dir / "feature_schema.json").read_text(encoding="utf-8"))
    if schema["clinical"]["valid_categories_used_to_build_vocabulary"]:
        raise AssertionError("Valid categories entered clinical vocabulary")

    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(
            np.zeros(len(train.target)), train.target
        )
    )
    probabilities: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "Dummy": (
            np.full(len(train.target), train.target.mean(), dtype=np.float64),
            np.full(len(valid.target), train.target.mean(), dtype=np.float64),
        )
    }
    fold_audits: list[dict[str, Any]] = []
    best_epochs: dict[str, list[int]] = {variant: [] for variant in VARIANTS}
    for variant in VARIANTS:
        print(f"[VARIANT START] {variant}", flush=True)
        oof = np.full(len(train.target), np.nan, dtype=np.float64)
        valid_folds: list[np.ndarray] = []
        for fold, (development_index, holdout_index) in enumerate(folds, start=1):
            development = train.subset(development_index)
            holdout = train.subset(holdout_index)
            result = train_with_early_stopping(
                variant, development, device, SEED + fold * 100 + VARIANTS.index(variant)
            )
            holdout_clinical = result.preprocessor.transform(holdout.clinical)
            valid_clinical = result.preprocessor.transform(valid.clinical)
            holdout_probability = predict(
                result.model, holdout, holdout_clinical, device
            )
            valid_probability = predict(result.model, valid, valid_clinical, device)
            oof[holdout_index] = holdout_probability
            valid_folds.append(valid_probability)
            best_epochs[variant].append(result.best_epoch)
            fold_dir = output / variant / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "variant": variant,
                "state_dict": result.model.state_dict(),
                "clinical_dimension": int(train.clinical.shape[1]),
                "best_epoch": result.best_epoch,
                "best_early_stop_auprc": result.best_ap,
                "parameter_count": parameter_count(result.model),
            }, fold_dir / "model.pt")
            joblib.dump(result.preprocessor, fold_dir / "clinical_preprocessor.joblib")
            atomic_csv(pd.DataFrame(result.history), fold_dir / "training_history.csv")
            fold_audits.append({
                "variant": variant,
                "fold": fold,
                "development_rows": int(len(development_index)),
                "holdout_rows": int(len(holdout_index)),
                "development_positive": int(train.target[development_index].sum()),
                "holdout_positive": int(train.target[holdout_index].sum()),
                "best_epoch": result.best_epoch,
                "best_early_stop_auprc": result.best_ap,
                "holdout_auroc": safe_auc(holdout.target, holdout_probability),
                "holdout_auprc": safe_ap(holdout.target, holdout_probability),
                "parameter_count": parameter_count(result.model),
                "valid_used_for_training": False,
            })
            print(
                f"[FOLD DONE] variant={variant} fold={fold} "
                f"epoch={result.best_epoch} holdout_ap={safe_ap(holdout.target, holdout_probability):.4f}",
                flush=True,
            )
        if not np.isfinite(oof).all():
            raise AssertionError(f"Incomplete OOF for {variant}")
        probabilities[f"MLP_{variant}"] = (
            oof,
            np.mean(np.stack(valid_folds), axis=0),
        )

        deploy_epochs = max(1, int(round(float(np.median(best_epochs[variant])))))
        full_preprocessor = ClinicalPreprocessor().fit(train.clinical)
        full_model = train_fixed_epochs(
            variant,
            train,
            full_preprocessor,
            deploy_epochs,
            device,
            SEED + 9000 + VARIANTS.index(variant),
        )
        deploy_dir = output / variant
        torch.save({
            "variant": variant,
            "state_dict": full_model.state_dict(),
            "clinical_dimension": int(train.clinical.shape[1]),
            "epochs": deploy_epochs,
            "epoch_source": "rounded median outer-fold best epoch",
            "parameter_count": parameter_count(full_model),
        }, deploy_dir / "full_train_model.pt")
        joblib.dump(full_preprocessor, deploy_dir / "full_train_clinical_preprocessor.joblib")

    metrics: list[dict[str, Any]] = []
    thresholds: dict[str, float] = {}
    train_predictions = pd.DataFrame({
        "patient_id": train.patient_id,
        "target": train.target,
    })
    valid_predictions = pd.DataFrame({
        "patient_id": valid.patient_id,
        "target": valid.target,
    })
    for model_name, (oof_probability, valid_probability) in probabilities.items():
        threshold = youden_threshold(train.target, oof_probability)
        thresholds[model_name] = threshold
        metrics.extend([
            metric_row(model_name, "Train_OOF", train.target, oof_probability, threshold),
            metric_row(model_name, "Valid", valid.target, valid_probability, threshold),
        ])
        train_predictions[f"{model_name.lower()}_probability"] = oof_probability
        valid_predictions[f"{model_name.lower()}_probability"] = valid_probability
    metrics_frame = pd.DataFrame(metrics)
    atomic_csv(metrics_frame, output / "metrics.csv")
    atomic_csv(train_predictions, output / "train_oof_predictions.csv")
    atomic_csv(valid_predictions, output / "valid_predictions.csv")
    atomic_csv(pd.DataFrame(fold_audits), output / "fold_audit.csv")
    comparison = previous_comparison(
        metrics_frame,
        dataset_dir.parent / "api_fullseq_fusion_v3_models/all_task_metrics.csv",
    )
    atomic_csv(comparison, output / "comparison_previous_fusion.csv")
    learned_oof = metrics_frame[
        (metrics_frame["split"] == "Train_OOF") & (metrics_frame["model"] != "Dummy")
    ]
    best_model = str(
        learned_oof.sort_values(["auprc", "auroc"], ascending=False).iloc[0]["model"]
    )
    summary = {
        "version": "api_fullseq_keyfusion_v3_adverse_models_1",
        "task": "adverse_patient",
        "train_rows": len(train.target),
        "train_positive": int(train.target.sum()),
        "valid_rows": len(valid.target),
        "valid_positive": int(valid.target.sum()),
        "models": list(probabilities),
        "best_model_selected_by_train_oof_auprc": best_model,
        "thresholds_from_train_oof": thresholds,
        "outer_folds": 5,
        "cave_primary_input": "complete 2x10x512 frozen embedding; supervised shared projection; no PCA",
        "searaft_primary_input": "2x70x16x16 dense key flow-map summaries; supervised CNN; no PCA",
        "clinical_input": "leakage-screened Train.xlsx/valid.xlsx fields",
        "valid_used_for_training_selection_early_stopping_or_threshold": False,
        "device": str(device),
        "seed": SEED,
        "best_epochs_by_variant": best_epochs,
    }
    atomic_json(summary, output / "summary.json")
    atomic_json(summary, output / ".MODELS_SUCCESS")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


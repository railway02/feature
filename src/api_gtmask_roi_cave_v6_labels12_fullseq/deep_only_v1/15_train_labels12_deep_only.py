#!/usr/bin/env python3
"""Train only Logistic/MLP on Labels 1+2 Local-CAVE deep features."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from matched_common import atomic_csv, atomic_json, atomic_npz, read_npz, sha256_file, stable_hash


PROJECT = Path("/root/autodl-tmp/aneurysm")
TASK_ROOT = (
    PROJECT
    / "outputs/api_gtmask_roi_cave_v6_labels12_fullseq/"
    "adverse_prepost_series_task_v3_fixed908"
)
MODEL_ROOT = (
    PROJECT
    / "outputs/api_gtmask_roi_cave_v6_labels12_fullseq/"
    "labels12_deep_only_models_v1"
)
CORE_TRAINER = (
    PROJECT
    / "code/api_gtmask_roi_cave_v6_labels12_fullseq/"
    "11_train_adverse_prepost_series_formal_v3.py"
)


def import_core(path: Path):
    spec = importlib.util.spec_from_file_location("matched_ablation_core_trainer", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_experiment(task_root: Path, experiment: str) -> dict[str, dict[str, np.ndarray]]:
    if experiment != "L12":
        raise ValueError(f"Only L12 is supported, got {experiment}")
    directory = task_root
    if not (directory / ".TASK_SUCCESS.json").is_file():
        raise FileNotFoundError(directory / ".TASK_SUCCESS.json")
    result: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "valid"):
        raw = read_npz(directory / f"{split}_features.npz")
        common = {
            key: raw[key]
            for key in ("series_uid", "patient_id", "target")
        }
        if split == "train":
            common["fold"] = raw["fold"]
        common["deep"] = np.asarray(raw["deep"], dtype=np.float32)
        common["scalar"] = np.empty((len(common["target"]), 0), dtype=np.float32)
        for name in ("deep", "scalar"):
            if not np.isfinite(common[name]).all():
                raise AssertionError(f"{experiment}/{split}/{name} is nonfinite")
        result[split] = common
    train, valid = result["train"], result["valid"]
    if len(train["series_uid"]) != 908 or len(valid["series_uid"]) != 239:
        raise AssertionError("Labels12 fixed cohort size changed")
    if set(train["patient_id"].astype(str)) & set(valid["patient_id"].astype(str)):
        raise AssertionError("Train/Valid patient overlap")
    fold_frame = pd.DataFrame(
        {
            "patient_id": train["patient_id"].astype(str),
            "fold": train["fold"].astype(int),
        }
    )
    if fold_frame.groupby("patient_id")["fold"].nunique().max() != 1:
        raise AssertionError("Patient crosses fixed outer folds")
    return result


def task_signature(task_root: Path, experiment: str) -> str:
    directory = task_root
    return stable_hash(
        {
            "success": sha256_file(directory / ".TASK_SUCCESS.json"),
            "train_npz": sha256_file(directory / "train_features.npz"),
            "valid_npz": sha256_file(directory / "valid_features.npz"),
        }
    )


def trainer_signature(core_path: Path) -> str:
    here = Path(__file__).resolve()
    return stable_hash(
        {
            "core": sha256_file(core_path),
            "driver": sha256_file(here),
        }
    )


def fit_outer_logistic_smoke(
    core: Any,
    train: dict[str, np.ndarray],
    valid: dict[str, np.ndarray],
    use_second: bool,
    fold: int,
) -> dict[str, Any]:
    y = train["target"].astype(np.int64)
    folds = train["fold"].astype(int)
    groups = train["patient_id"].astype(str)
    development = np.flatnonzero(folds != fold)
    holdout = np.flatnonzero(folds == fold)
    best, search = core.select_logistic(
        train["deep"][development],
        train["scalar"][development],
        y[development],
        groups[development],
        use_second,
        core.SEED + fold * 100,
    )
    preprocessor = core.FusionPreprocessor(
        deep_pca_dim=int(best["deep_pca_dim"]),
        use_scalar=use_second,
        seed=core.SEED + fold * 1000,
    ).fit(train["deep"][development], train["scalar"][development])
    x_development = preprocessor.transform(
        train["deep"][development], train["scalar"][development]
    )
    x_holdout = preprocessor.transform(
        train["deep"][holdout], train["scalar"][holdout]
    )
    x_valid = preprocessor.transform(valid["deep"], valid["scalar"])
    transformed = {
        "development": x_development,
        "holdout": x_holdout,
        "valid": x_valid,
    }
    if any(not np.isfinite(values).all() for values in transformed.values()):
        raise AssertionError("Logistic smoke transformed features are nonfinite")

    model = core.LogisticRegression(
        C=float(best["C"]),
        class_weight="balanced",
        solver="liblinear",
        max_iter=10000,
        random_state=core.SEED + fold,
    )
    model.fit(x_development, y[development])
    holdout_probability = model.predict_proba(x_holdout)[:, 1]
    valid_probability = model.predict_proba(x_valid)[:, 1]
    if not np.isfinite(holdout_probability).all() or not np.isfinite(
        valid_probability
    ).all():
        raise AssertionError("Logistic smoke prediction is nonfinite")
    return {
        "best": best,
        "search_rows": len(search),
        "preprocessor": preprocessor.audit(),
        "transformed_shapes": {
            name: list(values.shape) for name, values in transformed.items()
        },
        "transformed_finite": True,
        "prediction_finite": True,
        "holdout_index": holdout,
        "holdout_probability": holdout_probability,
        "valid_probability": valid_probability,
    }


def fit_outer_mlp_smoke(
    core: Any,
    train: dict[str, np.ndarray],
    valid: dict[str, np.ndarray],
    use_second: bool,
    fold: int,
    device: torch.device,
    search_seeds: int,
    amp_enabled: bool,
) -> dict[str, Any]:
    y = train["target"].astype(np.int64)
    folds = train["fold"].astype(int)
    groups = train["patient_id"].astype(str)
    development = np.flatnonzero(folds != fold)
    holdout = np.flatnonzero(folds == fold)
    config, epoch, inner_rows, summaries = core.select_mlp_config(
        train["deep"][development],
        train["scalar"][development],
        y[development],
        groups[development],
        use_second,
        device,
        core.SEED + fold * 100,
        search_seeds,
        amp_enabled,
    )
    preprocessor = core.FusionPreprocessor(
        deep_pca_dim=config.deep_pca_dim,
        use_scalar=use_second,
        seed=core.SEED + fold * 1000,
    ).fit(train["deep"][development], train["scalar"][development])
    x_development = preprocessor.transform(
        train["deep"][development], train["scalar"][development]
    )
    x_holdout = preprocessor.transform(
        train["deep"][holdout], train["scalar"][holdout]
    )
    x_valid = preprocessor.transform(valid["deep"], valid["scalar"])
    transformed = {
        "development": x_development,
        "holdout": x_holdout,
        "valid": x_valid,
    }
    if any(not np.isfinite(values).all() for values in transformed.values()):
        raise AssertionError("MLP smoke transformed features are nonfinite")

    model = core.fit_mlp_fixed_epochs(
        x_development,
        y[development],
        config,
        epoch,
        device,
        core.SEED + fold * 10000,
        amp_enabled,
    )
    holdout_probability = core.mlp_predict(model, x_holdout, device)
    valid_probability = core.mlp_predict(model, x_valid, device)
    if not np.isfinite(holdout_probability).all() or not np.isfinite(
        valid_probability
    ).all():
        raise AssertionError("MLP smoke prediction is nonfinite")
    return {
        "config": core.asdict(config),
        "selected_epoch": int(epoch),
        "inner_fold_rows": len(inner_rows),
        "config_summary_rows": len(summaries),
        "preprocessor": preprocessor.audit(),
        "transformed_shapes": {
            name: list(values.shape) for name, values in transformed.items()
        },
        "transformed_finite": True,
        "prediction_finite": True,
        "holdout_index": holdout,
        "holdout_probability": holdout_probability,
        "valid_probability": valid_probability,
    }


def run_smoke(
    core: Any,
    data: dict[str, dict[str, np.ndarray]],
    output: Path,
    experiment: str,
    fold: int,
    device: torch.device,
    search_seeds: int,
    amp_enabled: bool,
) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite smoke output: {output}")
    train, valid = data["train"], data["valid"]
    folds = train["fold"].astype(int)
    development = np.flatnonzero(folds != fold)
    expected_holdout = np.flatnonzero(folds == fold)
    development_patients = set(train["patient_id"].astype(str)[development])
    holdout_patients = set(train["patient_id"].astype(str)[expected_holdout])
    patient_overlap = sorted(development_patients & holdout_patients)
    if patient_overlap:
        raise AssertionError(
            f"Development/holdout patient leakage: {patient_overlap[:5]}"
        )

    if experiment == "L12":
        expected_shapes = {
            "train_labels12_local_deep": [908, 10240],
            "valid_labels12_local_deep": [239, 10240],
        }
        actual_shapes = {
            "train_labels12_local_deep": list(train["deep"].shape),
            "valid_labels12_local_deep": list(valid["deep"].shape),
        }
        model_input_components = ["labels12_local_pre5120_post5120"]
    elif experiment == "W0":
        expected_shapes = {
            "train_whole_deep": [908, 10240],
            "valid_whole_deep": [239, 10240],
        }
        actual_shapes = {
            "train_whole_deep": list(train["deep"].shape),
            "valid_whole_deep": list(valid["deep"].shape),
        }
        model_input_components = ["whole_pre5120_post5120"]
    else:
        expected_shapes = {
            "train_whole_deep": [908, 10240],
            "train_local_deep": [908, 10240],
            "valid_whole_deep": [239, 10240],
            "valid_local_deep": [239, 10240],
        }
        actual_shapes = {
            "train_whole_deep": list(train["deep"].shape),
            "train_local_deep": list(train["scalar"].shape),
            "valid_whole_deep": list(valid["deep"].shape),
            "valid_local_deep": list(valid["scalar"].shape),
        }
        model_input_components = ["whole_prepost_10240", "local_prepost_10240"]
    if actual_shapes != expected_shapes:
        raise AssertionError(
            f"{experiment} smoke input shapes {actual_shapes}, expected {expected_shapes}"
        )
    use_second = experiment == "WL"
    logistic = fit_outer_logistic_smoke(core, train, valid, use_second, fold)
    mlp = fit_outer_mlp_smoke(
        core,
        train,
        valid,
        use_second,
        fold,
        device,
        search_seeds,
        amp_enabled,
    )
    holdout = logistic.pop("holdout_index")
    if not np.array_equal(holdout, mlp.pop("holdout_index")):
        raise AssertionError("Smoke Logistic/MLP holdout differs")
    y_holdout = train["target"].astype(np.int64)[holdout]
    probabilities = {
        "logistic": logistic.pop("holdout_probability"),
        "mlp": mlp.pop("holdout_probability"),
    }
    valid_probabilities = {
        "logistic": logistic.pop("valid_probability"),
        "mlp": mlp.pop("valid_probability"),
    }
    if not np.array_equal(holdout, expected_holdout):
        raise AssertionError("Smoke holdout does not match the fixed outer fold")
    all_probabilities = [*probabilities.values(), *valid_probabilities.values()]
    if any(not np.isfinite(values).all() for values in all_probabilities):
        raise AssertionError("Smoke predictions are nonfinite")
    if experiment == "WL":
        for model_name, model_audit in (("logistic", logistic), ("mlp", mlp)):
            if not model_audit["preprocessor"].get("separate_branch_fit", False):
                raise AssertionError(
                    f"WL {model_name} did not report separate Whole/Local fit"
                )
    summary = {
        "status": "success",
        "mode": "one_outer_fold_smoke_full_inner_protocol",
        "experiment": experiment,
        "fold": fold,
        "fixed_patient_grouped_fold": True,
        "development_patients": int(len(development_patients)),
        "holdout_patients": int(len(holdout_patients)),
        "development_holdout_patient_overlap": 0,
        "patient_leakage_check": "PASS",
        "model_input_components": model_input_components,
        "model_input_shapes": actual_shapes,
        "model_input_finite": True,
        "labels_used_as_features": False,
        "fold_used_as_feature": False,
        "predictions_used_as_features": False,
        "preprocessing_fit_scope": "outer development rows only",
        "preprocessing_fit_rows": int(len(development)),
        "valid_fit_rows": 0,
        "valid_used_for_fit": False,
        "valid_used_for_selection": False,
        "transformed_features_finite": bool(
            logistic["transformed_finite"] and mlp["transformed_finite"]
        ),
        "predictions_finite": bool(
            logistic["prediction_finite"] and mlp["prediction_finite"]
        ),
        "whole_local_separate_preprocessing": experiment == "WL",
        "development_rows": int(len(development)),
        "holdout_rows": int(len(holdout)),
        "valid_rows": int(len(valid["target"])),
        "logistic_holdout_AUROC": core.safe_auc(y_holdout, probabilities["logistic"]),
        "logistic_holdout_AUPRC": core.safe_ap(y_holdout, probabilities["logistic"]),
        "mlp_holdout_AUROC": core.safe_auc(y_holdout, probabilities["mlp"]),
        "mlp_holdout_AUPRC": core.safe_ap(y_holdout, probabilities["mlp"]),
        "logistic": logistic,
        "mlp": mlp,
    }
    output.mkdir(parents=True, exist_ok=False)
    atomic_npz(
        output / "smoke_predictions.npz",
        holdout_series_uid=train["series_uid"].astype(str)[holdout],
        holdout_target=y_holdout,
        logistic_holdout_probability=probabilities["logistic"],
        mlp_holdout_probability=probabilities["mlp"],
        valid_series_uid=valid["series_uid"].astype(str),
        logistic_valid_probability=valid_probabilities["logistic"],
        mlp_valid_probability=valid_probabilities["mlp"],
    )
    atomic_json(summary, output / "smoke_summary.json")
    atomic_json(summary, output / ".SMOKE_SUCCESS.json")


def patient_cluster_bootstrap_generic(
    core: Any,
    y: np.ndarray,
    groups: np.ndarray,
    probabilities: dict[str, np.ndarray],
    thresholds: dict[str, float],
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Bootstrap arbitrary model names at the patient level."""
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups).astype(str)
    if repeats < 1:
        raise ValueError("Bootstrap repeats must be positive")
    if set(probabilities) != set(thresholds):
        raise AssertionError("Bootstrap probability/threshold model names differ")
    for model, values in probabilities.items():
        values = np.asarray(values, dtype=np.float64)
        if values.shape != y.shape or not np.isfinite(values).all():
            raise AssertionError(f"Bad bootstrap probability vector for {model}")

    unique_groups = np.unique(groups)
    group_indices = {
        group: np.flatnonzero(groups == group) for group in unique_groups
    }
    rng = np.random.default_rng(seed)
    metric_names = ("AUROC", "AUPRC", "Brier")
    distributions = {
        model: {metric: [] for metric in metric_names}
        for model in probabilities
    }
    model_names = list(probabilities)
    reference = model_names[0]
    comparisons = {
        model: {metric: [] for metric in metric_names}
        for model in model_names[1:]
    }
    effective = 0
    for _ in range(repeats):
        sampled_groups = rng.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )
        indices = np.concatenate([group_indices[group] for group in sampled_groups])
        y_sample = y[indices]
        if len(np.unique(y_sample)) < 2:
            continue
        effective += 1
        current: dict[str, dict[str, Any]] = {}
        for model, values in probabilities.items():
            current[model] = core.metric_row(
                model,
                "bootstrap",
                y_sample,
                np.asarray(values)[indices],
                thresholds[model],
            )
            for metric in metric_names:
                distributions[model][metric].append(float(current[model][metric]))
        for model, metric_values in comparisons.items():
            metric_values["AUROC"].append(
                current[model]["AUROC"] - current[reference]["AUROC"]
            )
            metric_values["AUPRC"].append(
                current[model]["AUPRC"] - current[reference]["AUPRC"]
            )
            metric_values["Brier"].append(
                current[model]["Brier"] - current[reference]["Brier"]
            )
    if effective == 0:
        raise RuntimeError("No valid patient-cluster bootstrap repeat")

    ci_rows: list[dict[str, Any]] = []
    for model, metrics in distributions.items():
        for metric, values in metrics.items():
            array = np.asarray(values, dtype=np.float64)
            ci_rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "bootstrap_mean": float(np.mean(array)),
                    "ci_lower": float(np.percentile(array, 2.5)),
                    "ci_upper": float(np.percentile(array, 97.5)),
                    "effective_repeats": effective,
                    "bootstrap_unit": "patient_id",
                }
            )
    paired_rows: list[dict[str, Any]] = []
    for model, metrics in comparisons.items():
        for metric, values in metrics.items():
            array = np.asarray(values, dtype=np.float64)
            paired_rows.append(
                {
                    "reference": reference,
                    "comparison": model,
                    "metric": metric,
                    "difference_definition": "comparison - reference",
                    "difference_mean": float(np.mean(array)),
                    "ci_lower": float(np.percentile(array, 2.5)),
                    "ci_upper": float(np.percentile(array, 97.5)),
                    "effective_repeats": effective,
                    "bootstrap_unit": "patient_id",
                }
            )
    return pd.DataFrame(ci_rows), pd.DataFrame(paired_rows), effective


def run_formal(
    core: Any,
    data: dict[str, dict[str, np.ndarray]],
    output: Path,
    experiment: str,
    device: torch.device,
    mlp_seeds: int,
    search_seeds: int,
    amp_enabled: bool,
    task_hash: str,
    trainer_hash: str,
    bootstrap_repeats: int,
) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite formal output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    train, valid = data["train"], data["valid"]
    use_second = experiment == "WL"
    logistic_name = f"{experiment}_Logistic"
    mlp_name = f"{experiment}_MLP"
    logistic_oof, logistic_valid, logistic_rows = core.train_logistic_outer(
        logistic_name,
        train,
        valid,
        use_second,
        output,
        False,
        task_hash,
        trainer_hash,
    )
    mlp_oof, mlp_valid, mlp_rows = core.train_mlp_outer(
        mlp_name,
        train,
        valid,
        use_second,
        device,
        output,
        False,
        mlp_seeds,
        search_seeds,
        task_hash,
        trainer_hash,
        amp_enabled,
    )
    y_train = train["target"].astype(np.int64)
    y_valid = valid["target"].astype(np.int64)
    pairs = {
        logistic_name: (logistic_oof, logistic_valid),
        mlp_name: (mlp_oof, mlp_valid),
    }
    metrics: list[dict[str, Any]] = []
    predictions_train = pd.DataFrame(
        {
            "series_uid": train["series_uid"].astype(str),
            "patient_id": train["patient_id"].astype(str),
            "target": y_train,
            "fold": train["fold"].astype(int),
        }
    )
    predictions_valid = pd.DataFrame(
        {
            "series_uid": valid["series_uid"].astype(str),
            "patient_id": valid["patient_id"].astype(str),
            "target": y_valid,
        }
    )
    thresholds: dict[str, float] = {}
    for model, (oof, valid_probability) in pairs.items():
        threshold = core.youden_threshold(y_train, oof)
        thresholds[model] = threshold
        metrics.append(core.metric_row(model, "Train_OOF", y_train, oof, threshold))
        metrics.append(
            core.metric_row(model, "Valid", y_valid, valid_probability, threshold)
        )
        predictions_train[f"{model}_probability"] = oof
        predictions_valid[f"{model}_probability"] = valid_probability
    metrics_frame = pd.DataFrame(metrics)
    candidates = metrics_frame[metrics_frame["split"] == "Train_OOF"].sort_values(
        ["AUPRC", "AUROC", "Brier", "model"],
        ascending=[False, False, True, True],
    )
    selected_model = str(candidates.iloc[0]["model"])
    bootstrap_ci, bootstrap_paired, effective_repeats = (
        patient_cluster_bootstrap_generic(
            core,
            y_valid,
            valid["patient_id"].astype(str),
            {model: values[1] for model, values in pairs.items()},
            thresholds,
            bootstrap_repeats,
            core.SEED + 91000,
        )
    )
    summary = {
        "status": "success",
        "experiment": experiment,
        "selected_model": selected_model,
        "selection_source": "pooled Train OOF only",
        "valid_used_for_selection": False,
        "threshold_source": "pooled Train OOF only",
        "thresholds": thresholds,
        "outer_folds": 5,
        "inner_folds": 3,
        "mlp_final_seeds": mlp_seeds,
        "mlp_search_seeds": search_seeds,
        "separate_whole_local_preprocessing": experiment == "WL",
        "bootstrap_repeats_requested": bootstrap_repeats,
        "bootstrap_effective_repeats": effective_repeats,
        "bootstrap_unit": "patient_id",
        "bootstrap_split": "Valid",
        "task_signature": task_hash,
        "trainer_signature": trainer_hash,
    }
    atomic_csv(metrics_frame, output / "metrics.csv")
    atomic_csv(pd.DataFrame(logistic_rows + mlp_rows), output / "fold_audit.csv")
    atomic_csv(predictions_train, output / "train_oof_predictions.csv")
    atomic_csv(predictions_valid, output / "valid_predictions.csv")
    atomic_csv(bootstrap_ci, output / "valid_patient_cluster_bootstrap_ci.csv")
    atomic_csv(bootstrap_paired, output / "valid_logistic_mlp_paired_bootstrap.csv")
    atomic_json(summary, output / "summary.json")
    atomic_json(summary, output / ".MODELS_SUCCESS.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, default=TASK_ROOT)
    parser.add_argument("--output-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--core-trainer", type=Path, default=CORE_TRAINER)
    parser.add_argument("--experiment", choices=("L12",), default="L12")
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--mlp-seeds", type=int, default=3)
    parser.add_argument("--mlp-search-seeds", type=int, default=2)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--disable-amp", action="store_true")
    args = parser.parse_args()
    if args.mode == "smoke" and args.fold not in range(1, 6):
        raise ValueError("Smoke fold must be 1..5")
    if args.mlp_seeds < 1 or args.mlp_search_seeds < 1:
        raise ValueError("MLP seed counts must be positive")
    if args.bootstrap_repeats < 1:
        raise ValueError("Bootstrap repeat count must be positive")

    core_path = args.core_trainer.resolve()
    core = import_core(core_path)
    core.set_seed(core.SEED)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
    core.configure_runtime(args.cpu_threads, device)
    amp_enabled = bool(device.type == "cuda" and not args.disable_amp)
    data = load_experiment(args.task_root.resolve(), args.experiment)
    task_hash = task_signature(args.task_root.resolve(), args.experiment)
    trainer_hash = trainer_signature(core_path)
    started = time.time()
    if args.mode == "smoke":
        output = args.output_root.resolve() / "smoke" / args.experiment
        run_smoke(
            core,
            data,
            output,
            args.experiment,
            args.fold,
            device,
            args.mlp_search_seeds,
            amp_enabled,
        )
    else:
        output = args.output_root.resolve() / "formal" / args.experiment
        run_formal(
            core,
            data,
            output,
            args.experiment,
            device,
            args.mlp_seeds,
            args.mlp_search_seeds,
            amp_enabled,
            task_hash,
            trainer_hash,
            args.bootstrap_repeats,
        )
    print(
        json.dumps(
            {
                "status": "success",
                "mode": args.mode,
                "experiment": args.experiment,
                "output": str(output),
                "elapsed_seconds": time.time() - started,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


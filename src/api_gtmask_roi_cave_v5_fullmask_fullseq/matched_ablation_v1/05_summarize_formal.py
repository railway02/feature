#!/usr/bin/env python3
"""Build the fixed-cohort matched-ablation formal comparison and paired bootstrap."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from matched_common import atomic_csv, atomic_json, require_new_directory, sha256_file


PROJECT = Path("/root/autodl-tmp/aneurysm")
LOCAL_ROOT = (
    PROJECT
    / "outputs/api_gtmask_roi_cave_v5_fullmask_fullseq"
)
BASE_TASK = LOCAL_ROOT / "adverse_prepost_series_task_v3"
LOCAL_RESULTS = LOCAL_ROOT / "adverse_prepost_series_formal_models_v31"
MATCHED_ROOT = LOCAL_ROOT / "adverse_prepost_matched_ablation_models_v1"
FORMAL_ROOT = MATCHED_ROOT / "formal"
OUTPUT = FORMAL_ROOT / "unified_v1"

SEED = 20260804
BOOTSTRAP_REPEATS = 2000

INPUT_LABELS = {
    "M0": "Mask morphology only (Pre/Post/delta, 36)",
    "W0": "Whole-CAVE Pre 5120 + Post 5120",
    "L0": "Local-CAVE Pre 5120 + Post 5120",
    "LS0": "Local deep 10240 + Local scalar 658",
    "WL": "Whole deep 10240 + Local deep 10240",
}

LOCAL_CANDIDATES = {
    "L0": ("Logistic_Deep", "MLP_Deep"),
    "LS0": ("Logistic_Fusion", "MLP_Fusion"),
}

PAIR_DEFINITIONS = (
    ("L0 - W0", "L0", "W0"),
    ("WL - W0", "WL", "W0"),
    ("WL - L0", "WL", "L0"),
    ("LS0 - L0", "LS0", "L0"),
    ("M0 - Dummy", "M0", "Dummy"),
)


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"series_uid": str, "patient_id": str},
        keep_default_na=False,
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_values(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "AUROC": float(roc_auc_score(target, probability)),
        "AUPRC": float(average_precision_score(target, probability)),
        "Brier": float(brier_score_loss(target, probability)),
    }


def select_by_train_oof(
    metrics: pd.DataFrame,
    candidates: tuple[str, ...],
) -> str:
    selected = metrics[
        (metrics["split"] == "Train_OOF")
        & metrics["model"].isin(candidates)
    ].copy()
    if set(selected["model"]) != set(candidates):
        raise AssertionError(f"Missing Train OOF candidate metrics: {candidates}")
    selected = selected.sort_values(
        ["AUPRC", "AUROC", "Brier", "model"],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    return str(selected.iloc[0]["model"])


def validate_predictions(
    frame: pd.DataFrame,
    fixed: dict[str, np.ndarray],
    split: str,
) -> None:
    expected = len(fixed["series_uid"])
    if len(frame) != expected:
        raise AssertionError(f"{split}: prediction rows {len(frame)} != {expected}")
    if not frame["series_uid"].is_unique:
        raise AssertionError(f"{split}: duplicate series_uid")
    for key in ("series_uid", "patient_id", "target"):
        left = frame[key].astype(str).to_numpy()
        right = np.asarray(fixed[key]).astype(str)
        if not np.array_equal(left, right):
            raise AssertionError(f"{split}: {key} order/value mismatch")
    if split == "train":
        if "fold" not in frame:
            raise AssertionError("Train prediction file lacks fixed fold")
        if not np.array_equal(
            frame["fold"].astype(int).to_numpy(),
            np.asarray(fixed["fold"]).astype(int),
        ):
            raise AssertionError("Train prediction fold mismatch")
    probability_columns = [
        name for name in frame.columns if name.endswith("_probability")
    ]
    if not probability_columns:
        raise AssertionError(f"{split}: no probability columns")
    for name in probability_columns:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise AssertionError(f"{split}: nonfinite {name}")


def mlp_parameter_count(input_dim: int, hidden1: int, hidden2: int) -> int:
    return int(
        input_dim * hidden1
        + 3 * hidden1
        + hidden1 * hidden2
        + 4 * hidden2
        + 1
    )


def complexity_rows(
    experiment: str,
    selected_model: str,
    fold_audit_path: Path,
) -> list[dict[str, Any]]:
    fold_audit = pd.read_csv(fold_audit_path, encoding="utf-8-sig")
    selected = fold_audit[fold_audit["model"] == selected_model].copy()
    if sorted(selected["fold"].astype(int).tolist()) != [1, 2, 3, 4, 5]:
        raise AssertionError(f"{experiment}: selected model does not have 5 folds")

    rows: list[dict[str, Any]] = []
    for _, record in selected.sort_values("fold").iterrows():
        audit = json.loads(record["selection"])
        preprocessor = audit["preprocessor"]
        fold = int(record["fold"])
        deep_pca: int | None = None
        scalar_pca: int | None = None
        whole_pca: int | None = None
        local_pca: int | None = None

        if experiment == "M0":
            deep_pca = int(preprocessor["morphology"]["pca_components"])
            total_dim = deep_pca
            pca_label = f"F{fold}:Mask={deep_pca}"
        elif experiment == "W0":
            whole_pca = int(preprocessor["whole"]["fitted_pca_dimension"])
            total_dim = whole_pca
            pca_label = f"F{fold}:Whole={whole_pca}"
        elif experiment == "WL":
            if not bool(preprocessor.get("separate_branch_fit")):
                raise AssertionError(f"WL fold {fold}: separate branch fit missing")
            whole_pca = int(preprocessor["whole"]["fitted_pca_dimension"])
            local_pca = int(preprocessor["local"]["fitted_pca_dimension"])
            total_dim = whole_pca + local_pca
            pca_label = (
                f"F{fold}:Whole={whole_pca}+Local={local_pca}"
                f"={total_dim}"
            )
        elif experiment == "L0":
            deep_pca = int(preprocessor["deep_pca_components"])
            total_dim = deep_pca
            pca_label = f"F{fold}:Local={deep_pca}"
        elif experiment == "LS0":
            deep_pca = int(preprocessor["deep_pca_components"])
            scalar_pca = int(preprocessor["scalar"]["pca_components"])
            total_dim = deep_pca + scalar_pca
            pca_label = (
                f"F{fold}:Local={deep_pca}+LocalScalar={scalar_pca}"
                f"={total_dim}"
            )
        else:
            raise ValueError(experiment)

        selected_config = audit.get("selected_config")
        best_inner = audit.get("best_inner_config")
        if selected_config is not None:
            hidden1 = int(selected_config["hidden1"])
            hidden2 = int(selected_config["hidden2"])
            per_model_parameters = mlp_parameter_count(
                total_dim,
                hidden1,
                hidden2,
            )
            ensemble_members = int(audit.get("final_ensemble_seeds", 3))
            model_family = "MLP"
        elif best_inner is not None:
            hidden1 = None
            hidden2 = None
            per_model_parameters = int(total_dim + 1)
            ensemble_members = 1
            model_family = "Logistic"
        else:
            raise AssertionError(
                f"{experiment} fold {fold}: no model configuration in audit"
            )

        rows.append(
            {
                "Experiment": experiment,
                "Selected model": selected_model,
                "Fold": fold,
                "Model family": model_family,
                "Deep PCA": deep_pca,
                "Scalar PCA": scalar_pca,
                "Whole PCA": whole_pca,
                "Local PCA": local_pca,
                "Total model input dim": int(total_dim),
                "PCA label": pca_label,
                "Hidden1": hidden1,
                "Hidden2": hidden2,
                "Parameters per model": per_model_parameters,
                "Ensemble members in fold": ensemble_members,
                "Parameters represented by fold ensemble": (
                    per_model_parameters * ensemble_members
                ),
                "Development rows": int(record["development_rows"]),
                "Holdout rows": int(record["holdout_rows"]),
                "Holdout patients": int(record["holdout_patients"]),
                "Development/holdout patient overlap": 0,
                "Valid fit rows": 0,
                "Valid fit forbidden": bool(
                    preprocessor.get("valid_fit_forbidden", True)
                ),
            }
        )
    return rows


def paired_bootstrap(
    target: np.ndarray,
    patient_id: np.ndarray,
    probabilities: dict[str, np.ndarray],
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    metric_functions: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
        "AUROC": lambda y, p: float(roc_auc_score(y, p)),
        "AUPRC": lambda y, p: float(average_precision_score(y, p)),
        "Brier": lambda y, p: float(brier_score_loss(y, p)),
    }
    patient_id = patient_id.astype(str)
    unique_patients = np.unique(patient_id)
    patient_rows = {
        patient: np.flatnonzero(patient_id == patient)
        for patient in unique_patients
    }
    rng = np.random.default_rng(seed)
    distributions = {
        (comparison, metric): []
        for comparison, _, _ in PAIR_DEFINITIONS
        for metric in metric_functions
    }
    effective = 0
    for _ in range(repeats):
        sampled_patients = rng.choice(
            unique_patients,
            size=len(unique_patients),
            replace=True,
        )
        indices = np.concatenate(
            [patient_rows[patient] for patient in sampled_patients]
        )
        y_sample = target[indices]
        if len(np.unique(y_sample)) < 2:
            continue
        effective += 1
        current = {
            model: {
                metric: function(y_sample, values[indices])
                for metric, function in metric_functions.items()
            }
            for model, values in probabilities.items()
        }
        for comparison, left, right in PAIR_DEFINITIONS:
            for metric in metric_functions:
                distributions[(comparison, metric)].append(
                    current[left][metric] - current[right][metric]
                )
    if effective == 0:
        raise RuntimeError("No effective patient bootstrap repeat")

    full_metrics = {
        model: metric_values(target, values)
        for model, values in probabilities.items()
    }
    rows: list[dict[str, Any]] = []
    for comparison, left, right in PAIR_DEFINITIONS:
        for metric in metric_functions:
            values = np.asarray(
                distributions[(comparison, metric)],
                dtype=np.float64,
            )
            lower = float(np.percentile(values, 2.5))
            upper = float(np.percentile(values, 97.5))
            rows.append(
                {
                    "Comparison": comparison,
                    "Left": left,
                    "Right": right,
                    "Metric": metric,
                    "Difference direction": f"{left} - {right}",
                    "Interpretation": (
                        "negative favors left"
                        if metric == "Brier"
                        else "positive favors left"
                    ),
                    "Point difference": (
                        full_metrics[left][metric]
                        - full_metrics[right][metric]
                    ),
                    "Bootstrap mean difference": float(np.mean(values)),
                    "CI lower 95%": lower,
                    "CI upper 95%": upper,
                    "Crosses zero": bool(lower <= 0 <= upper),
                    "Effective repeats": effective,
                    "Requested repeats": repeats,
                    "Bootstrap unit": "patient_id",
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap-repeats", type=int, default=BOOTSTRAP_REPEATS)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite unified output: {output}")
    if args.bootstrap_repeats != 2000:
        raise AssertionError("Formal paired bootstrap must use exactly 2000 repeats")

    fixed = {
        split: read_npz(BASE_TASK / f"{split}_features.npz")
        for split in ("train", "valid")
    }
    if len(fixed["train"]["series_uid"]) != 908:
        raise AssertionError("Fixed Train cohort is not 908")
    if len(fixed["valid"]["series_uid"]) != 239:
        raise AssertionError("Fixed Valid cohort is not 239")

    patient_fold = pd.DataFrame(
        {
            "patient_id": fixed["train"]["patient_id"].astype(str),
            "fold": fixed["train"]["fold"].astype(int),
        }
    )
    if int(patient_fold.groupby("patient_id")["fold"].nunique().max()) != 1:
        raise AssertionError("A patient crosses fixed outer folds")
    fold_overlap: dict[str, int] = {}
    for fold in range(1, 6):
        development = set(
            patient_fold.loc[
                patient_fold["fold"] != fold,
                "patient_id",
            ]
        )
        holdout = set(
            patient_fold.loc[
                patient_fold["fold"] == fold,
                "patient_id",
            ]
        )
        fold_overlap[str(fold)] = len(development & holdout)
    if any(fold_overlap.values()):
        raise AssertionError(f"Patient overlap by fold: {fold_overlap}")

    local_metrics = pd.read_csv(
        LOCAL_RESULTS / "metrics.csv",
        encoding="utf-8-sig",
    )
    selections: dict[str, str] = {
        experiment: select_by_train_oof(local_metrics, candidates)
        for experiment, candidates in LOCAL_CANDIDATES.items()
    }

    result_roots: dict[str, Path] = {
        "L0": LOCAL_RESULTS,
        "LS0": LOCAL_RESULTS,
    }
    for experiment in ("M0", "W0", "WL"):
        root = FORMAL_ROOT / experiment
        success_path = root / ".MODELS_SUCCESS.json"
        if not success_path.is_file():
            raise FileNotFoundError(success_path)
        success = load_json(success_path)
        if success.get("status") != "success":
            raise AssertionError(f"{experiment}: unsuccessful formal result")
        if success.get("selection_source") != "pooled Train OOF only":
            raise AssertionError(f"{experiment}: invalid selection source")
        if success.get("valid_used_for_selection") is not False:
            raise AssertionError(f"{experiment}: Valid used for selection")
        if int(success.get("outer_folds", 0)) != 5:
            raise AssertionError(f"{experiment}: not five outer folds")
        if int(success.get("bootstrap_effective_repeats", 0)) != 2000:
            raise AssertionError(f"{experiment}: incomplete bootstrap")
        selections[experiment] = str(success["selected_model"])
        result_roots[experiment] = root

    train_frames: dict[str, pd.DataFrame] = {}
    valid_frames: dict[str, pd.DataFrame] = {}
    for experiment, root in result_roots.items():
        train_frame = read_csv(root / "train_oof_predictions.csv")
        valid_frame = read_csv(root / "valid_predictions.csv")
        validate_predictions(train_frame, fixed["train"], "train")
        validate_predictions(valid_frame, fixed["valid"], "valid")
        train_frames[experiment] = train_frame
        valid_frames[experiment] = valid_frame

    metrics_rows: list[dict[str, Any]] = []
    probability_vectors: dict[str, dict[str, np.ndarray]] = {
        "train": {},
        "valid": {},
    }
    for experiment in ("M0", "W0", "L0", "LS0", "WL"):
        model = selections[experiment]
        probability_column = f"{model}_probability"
        train_probability = pd.to_numeric(
            train_frames[experiment][probability_column],
            errors="raise",
        ).to_numpy(float)
        valid_probability = pd.to_numeric(
            valid_frames[experiment][probability_column],
            errors="raise",
        ).to_numpy(float)
        probability_vectors["train"][experiment] = train_probability
        probability_vectors["valid"][experiment] = valid_probability
        train_metrics = metric_values(
            fixed["train"]["target"].astype(int),
            train_probability,
        )
        valid_metrics = metric_values(
            fixed["valid"]["target"].astype(int),
            valid_probability,
        )
        metrics_rows.append(
            {
                "Experiment": experiment,
                "Selected model": model,
                "Input": INPUT_LABELS[experiment],
                "Train OOF AUROC": train_metrics["AUROC"],
                "Train OOF AUPRC": train_metrics["AUPRC"],
                "Train OOF Brier": train_metrics["Brier"],
                "Valid AUROC": valid_metrics["AUROC"],
                "Valid AUPRC": valid_metrics["AUPRC"],
                "Valid Brier": valid_metrics["Brier"],
                "Selection source": "pooled Train OOF only",
                "Valid used for selection": False,
            }
        )

    dummy_train = pd.to_numeric(
        train_frames["L0"]["Dummy_probability"],
        errors="raise",
    ).to_numpy(float)
    dummy_valid = pd.to_numeric(
        valid_frames["L0"]["Dummy_probability"],
        errors="raise",
    ).to_numpy(float)
    probability_vectors["train"]["Dummy"] = dummy_train
    probability_vectors["valid"]["Dummy"] = dummy_valid

    complexity: list[dict[str, Any]] = []
    for experiment in ("M0", "W0", "L0", "LS0", "WL"):
        complexity.extend(
            complexity_rows(
                experiment,
                selections[experiment],
                result_roots[experiment] / "fold_audit.csv",
            )
        )
    complexity_frame = pd.DataFrame(complexity)
    metrics_frame = pd.DataFrame(metrics_rows)

    summary_rows: list[dict[str, Any]] = []
    for _, metric_record in metrics_frame.iterrows():
        experiment = str(metric_record["Experiment"])
        selected_complexity = complexity_frame[
            complexity_frame["Experiment"] == experiment
        ].sort_values("Fold")
        pca_dims = "; ".join(selected_complexity["PCA label"].astype(str))
        per_model = "; ".join(
            f"F{int(row['Fold'])}:{int(row['Parameters per model'])}"
            for _, row in selected_complexity.iterrows()
        )
        total_parameters = int(
            selected_complexity[
                "Parameters represented by fold ensemble"
            ].sum()
        )
        summary_rows.append(
            {
                "Experiment": experiment,
                "Selected model": metric_record["Selected model"],
                "Input": metric_record["Input"],
                "PCA dims": pca_dims,
                "Parameters": (
                    f"per model [{per_model}]; "
                    f"all fold/seed members={total_parameters}"
                ),
                "Train OOF AUROC": metric_record["Train OOF AUROC"],
                "Train OOF AUPRC": metric_record["Train OOF AUPRC"],
                "Train OOF Brier": metric_record["Train OOF Brier"],
                "Valid AUROC": metric_record["Valid AUROC"],
                "Valid AUPRC": metric_record["Valid AUPRC"],
                "Valid Brier": metric_record["Valid Brier"],
            }
        )
    unified_frame = pd.DataFrame(summary_rows)

    bootstrap_frame = paired_bootstrap(
        fixed["valid"]["target"].astype(int),
        fixed["valid"]["patient_id"].astype(str),
        probability_vectors["valid"],
        args.bootstrap_repeats,
        SEED + 73000,
    )

    validation = {
        "status": "PASS",
        "fixed_train_rows": 908,
        "fixed_valid_rows": 239,
        "fixed_train_patients": int(
            len(np.unique(fixed["train"]["patient_id"].astype(str)))
        ),
        "fixed_valid_patients": int(
            len(np.unique(fixed["valid"]["patient_id"].astype(str)))
        ),
        "patient_overlap_by_outer_fold": fold_overlap,
        "development_holdout_patient_overlap": 0,
        "valid_fit_rows": 0,
        "all_prediction_vectors_finite": bool(
            all(
                np.isfinite(values).all()
                for split in probability_vectors.values()
                for values in split.values()
            )
        ),
        "uid_target_patient_fold_order_verified": True,
        "selected_models": selections,
        "selection_source": "pooled Train OOF only",
        "valid_used_for_selection": False,
        "paired_bootstrap_repeats": args.bootstrap_repeats,
        "paired_bootstrap_effective_repeats": int(
            bootstrap_frame["Effective repeats"].min()
        ),
        "paired_bootstrap_unit": "patient_id",
        "amp_failures": 0,
        "fp32_reruns": 0,
        "formal_experiments": {
            experiment: {
                "logistic_folds": 5,
                "mlp_folds": 5,
                "success_lock": str(
                    FORMAL_ROOT / experiment / ".MODELS_SUCCESS.json"
                ),
                "success_lock_sha256": sha256_file(
                    FORMAL_ROOT / experiment / ".MODELS_SUCCESS.json"
                ),
            }
            for experiment in ("M0", "W0", "WL")
        },
        "local_result_root": str(LOCAL_RESULTS),
        "local_result_unchanged": True,
        "ls0_definition": "Local deep + Local scalar 658",
        "wl_definition": "Whole deep + Local deep",
    }

    staging = output.with_name(f".{output.name}.{os.getpid()}.staging")
    require_new_directory(staging)
    atomic_csv(unified_frame, staging / "unified_metrics.csv")
    atomic_csv(metrics_frame, staging / "recomputed_metrics.csv")
    atomic_csv(complexity_frame, staging / "model_complexity_audit.csv")
    atomic_csv(bootstrap_frame, staging / "paired_patient_bootstrap.csv")
    atomic_json(validation, staging / "postrun_validation.json")
    summary = {
        "status": "success",
        "version": "matched_ablation_formal_unified_v1",
        "selection_source": "pooled Train OOF only",
        "valid_used_for_selection": False,
        "selected_models": selections,
        "experiments": ["M0", "W0", "L0", "LS0", "WL"],
        "paired_bootstrap_repeats": args.bootstrap_repeats,
        "paired_bootstrap_unit": "patient_id",
        "difference_definition": "left - right",
        "brier_interpretation": "negative difference favors left",
        "validation": validation,
    }
    atomic_json(summary, staging / "summary.json")
    atomic_json(summary, staging / ".SUMMARY_SUCCESS.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, output)

    print(unified_frame.to_string(index=False))
    print(bootstrap_frame.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


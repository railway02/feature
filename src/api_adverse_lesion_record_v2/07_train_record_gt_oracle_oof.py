#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from common import atomic_csv, atomic_json, load_config, update_run_manifest


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def post_delta(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.shape[1:] != (2, 5120):
        raise AssertionError(f"Unexpected series embedding shape {values.shape}")
    pre, post = values[:, 0], values[:, 1]
    delta = np.asarray(post - pre, dtype=np.float32)
    delta[~np.isfinite(pre).all(axis=1)] = np.nan
    return np.concatenate([post, delta], axis=1)


def load_series_embedding(path: Path, series_uids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    raw = np.load(path)
    index = {str(uid): idx for idx, uid in enumerate(raw["series_uid"].astype(str))}
    missing = [uid for uid in series_uids if uid not in index]
    if missing:
        raise KeyError(f"Missing series embeddings: {missing[:5]}")
    order = np.asarray([index[uid] for uid in series_uids], dtype=np.int64)
    embeddings = np.asarray(raw["embeddings"][order], dtype=np.float32)
    flags = np.stack([raw["missing_pre"][order], raw["missing_post"][order]], axis=1).astype(np.float64)
    return embeddings, flags


def load_morphology(
    path: Path,
    series_uids: list[str],
    columns: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    frame = pd.read_csv(path, dtype={"series_uid": str})
    frame = frame.set_index("series_uid", drop=False)
    missing = [uid for uid in series_uids if uid not in frame.index]
    if missing:
        raise KeyError(f"Missing GT morphology: {missing[:5]}")
    selected = frame.loc[series_uids]
    excluded = {
        "record_uid", "patient_id", "series_uid", "series_id", "split", "source_type",
        "phase", "roi_branch", "mask_source", "annotation_grade", "annotation_layout",
    }
    if columns is None:
        candidate_columns = [column for column in selected.columns if column not in excluded]
        numeric = selected[candidate_columns].apply(pd.to_numeric, errors="coerce")
        useful = numeric.notna().any(axis=0)
        columns = numeric.columns[useful].tolist()
    else:
        missing_columns = [column for column in columns if column not in selected.columns]
        if missing_columns:
            raise KeyError(f"Missing GT morphology columns: {missing_columns}")
        numeric = selected[columns].apply(pd.to_numeric, errors="coerce")
    return numeric.loc[:, columns].to_numpy(np.float64), columns


def subset(data: dict, indices: np.ndarray) -> dict:
    return {
        "images": {name: {key: value[indices] for key, value in branch.items()} for name, branch in data["images"].items()},
        "morphology": data["morphology"][indices],
    }


@dataclass
class OraclePreprocessor:
    fixed: object
    image_names: tuple[str, ...]
    use_morphology: bool
    deep: dict[str, object] = field(default_factory=dict)
    morphology: object | None = None
    final_scaler: StandardScaler | None = None

    def __getstate__(self):
        state = self.__dict__.copy(); state["fixed"] = None; return state

    def fit(self, data: dict, seed: int) -> "OraclePreprocessor":
        self.deep = {
            name: self.fixed.DeepBranch().fit(post_delta(data["images"][name]["embedding"]), seed + index * 10000)
            for index, name in enumerate(self.image_names)
        }
        if self.use_morphology:
            self.morphology = self.fixed.NumericBranch(24, "robust", 0.25, True).fit(data["morphology"], seed + 50000)
        base = self._base(data)
        self.final_scaler = StandardScaler().fit(base)
        return self

    def _base(self, data: dict) -> np.ndarray:
        pieces = []
        for name in self.image_names:
            pieces.append(self.deep[name].transform(post_delta(data["images"][name]["embedding"])))
            pieces.append(np.asarray(data["images"][name]["missing"], dtype=np.float64))
        if self.use_morphology:
            pieces.append(self.morphology.transform(data["morphology"]))
        if not pieces:
            raise AssertionError("Empty oracle representation")
        value = np.concatenate(pieces, axis=1)
        if not np.isfinite(value).all():
            raise AssertionError("Nonfinite oracle base features")
        return value

    def transform(self, data: dict) -> np.ndarray:
        if self.final_scaler is None:
            raise RuntimeError("Oracle preprocessor not fitted")
        value = self.final_scaler.transform(self._base(data)).astype(np.float64)
        if not np.isfinite(value).all():
            raise AssertionError("Nonfinite oracle features")
        return value


def atomic_joblib(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    joblib.dump(value, temporary)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/root/autodl-tmp/aneurysm/configs/api_adverse_lesion_record_v2/config.json")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["project_root"])
    reports = Path(config["paths"]["reports"])
    outputs = Path(config["paths"]["outputs"])
    fixed = import_file(Path(config["fixed_trainer"]), "record_oracle_fixed")
    record = pd.read_csv(Path(config["paths"]["manifests"]) / "oracle_record_manifest.csv", dtype={"patient_id": str, "series_uid": str})
    train = record[record.split == "Train"].sort_values("record_uid").reset_index(drop=True)
    series = train.series_uid.astype(str).tolist()
    morphology, morphology_columns = load_morphology(Path(config["gt_morphology_series"]), series)
    data = {"images": {}, "morphology": morphology}
    atomic_json(
        {"columns": morphology_columns, "count": len(morphology_columns)},
        reports / "record_gt_oracle_morphology_columns.json",
    )
    roots = {
        "whole": Path(config["whole_train_tables"]),
        "local30": outputs / "cave_gt_context30_tables/train",
        "local40": outputs / "cave_gt_context40_tables/train",
    }
    for name, directory in roots.items():
        embedding, missing = load_series_embedding(directory / "series_embeddings_5120.npz", series)
        data["images"][name] = {"embedding": embedding, "missing": missing}
    representations = {
        "G0_whole": (("whole",), False),
        "G1_gt_morphology": ((), True),
        "G2_gt_local30": (("local30",), False),
        "G3_gt_local40": (("local40",), False),
        "G4_whole_gt_local30": (("whole", "local30"), False),
        "G5_whole_gt_local40": (("whole", "local40"), False),
    }
    y = train.target.to_numpy(np.int64)
    folds = train.fold.to_numpy(np.int64)
    groups = train.patient_id.astype(str).to_numpy()
    c_grid = [float(value) for value in config["prediction"]["c_grid"]]
    seed = int(config["prediction"]["seed"])
    audit_rows = []
    audit_path = reports / "record_gt_oracle_convergence_audit.csv"
    metrics, fold_metrics = [], []
    predictions = train[["record_uid", "patient_id", "series_uid", "target", "fold"]].copy()

    for representation, (image_names, use_morphology) in representations.items():
        oof = np.full(len(train), np.nan, dtype=np.float64)
        for fold in sorted(np.unique(folds)):
            holdout = np.flatnonzero(folds == fold); development = np.flatnonzero(folds != fold)
            development_data = subset(data, development)
            inner_predictions = {c: np.full(len(development), np.nan) for c in c_grid}
            inner = fixed.grouped_splits(y[development], groups[development], int(config["prediction"]["inner_folds"]), seed + fold * 1000)
            for inner_fold, (fit_index, inner_holdout) in enumerate(inner, 1):
                fit_data = subset(development_data, fit_index); inner_data = subset(development_data, inner_holdout)
                pre = OraclePreprocessor(fixed, image_names, use_morphology).fit(fit_data, seed + fold * 100 + inner_fold)
                fit_x, inner_x = pre.transform(fit_data), pre.transform(inner_data)
                for c_value in c_grid:
                    model = fixed.fit_logistic_checked(
                        fit_x, y[development][fit_index], c_value,
                        {"task": "followup_rroc23_record_v2", "representation": representation, "outer_fold": int(fold), "stage": "inner_cv", "variant": "logistic", "inner_fold": int(inner_fold)},
                        audit_rows, audit_path, inner_x, y[development][inner_holdout],
                    )
                    inner_predictions[c_value][inner_holdout] = model.predict_proba(inner_x)[:, 1]
            scores = {str(c): fixed.safe_ap(y[development], probability) for c, probability in inner_predictions.items()}
            selected_c = max(c_grid, key=lambda value: (scores[str(value)], -value))
            pre = OraclePreprocessor(fixed, image_names, use_morphology).fit(development_data, seed + fold * 100)
            development_x = pre.transform(development_data); holdout_x = pre.transform(subset(data, holdout))
            model = fixed.fit_logistic_checked(
                development_x, y[development], selected_c,
                {"task": "followup_rroc23_record_v2", "representation": representation, "outer_fold": int(fold), "stage": "outer_refit", "variant": "logistic", "inner_fold": 0},
                audit_rows, audit_path, holdout_x, y[holdout],
            )
            probability = model.predict_proba(holdout_x)[:, 1]; oof[holdout] = probability
            atomic_joblib(
                {
                    "preprocessor": pre,
                    "model": model,
                    "selected_c": selected_c,
                    "image_names": image_names,
                    "use_morphology": use_morphology,
                    "morphology_columns": morphology_columns,
                },
                outputs / "record_gt_oracle_models" / representation / f"fold_{fold}.joblib",
            )
            fold_metrics.append({"representation": representation, "fold": fold, "selected_c": selected_c, "rows": len(holdout), "positive": int(y[holdout].sum()), "auroc": fixed.safe_auc(y[holdout], probability), "auprc": fixed.safe_ap(y[holdout], probability)})
        threshold = float(fixed.youden_threshold(y, oof))
        row = fixed.metric_row("followup_rroc23_record_v2", representation, "Train_OOF", y, oof, threshold)
        metrics.append(row); predictions[f"{representation}_probability"] = oof
    metric_frame = pd.DataFrame(metrics); fold_frame = pd.DataFrame(fold_metrics)
    atomic_csv(metric_frame, reports / "record_gt_oracle_oof_metrics.csv")
    atomic_csv(fold_frame, reports / "record_gt_oracle_fold_metrics.csv")
    atomic_csv(predictions, reports / "record_gt_oracle_oof_predictions.csv")
    whole = metric_frame.set_index("model").loc["G0_whole"]
    candidates = metric_frame[metric_frame.model.isin(["G4_whole_gt_local30", "G5_whole_gt_local40"])].copy()
    best = candidates.sort_values(["auprc", "auroc"], ascending=False).iloc[0]
    selected_scale = "30" if best.model.endswith("30") else "40"
    pivot = fold_frame.pivot(index="fold", columns="representation", values="auprc")
    consistent = int((pivot[best.model] > pivot["G0_whole"]).sum())
    delta_ap = float(best.auprc - whole.auprc)
    gate = bool(delta_ap >= float(config["prediction"]["oracle_gain_ap"]) and consistent >= 3)
    summary = {
        "status": "complete", "records": len(train), "patients": int(train.patient_id.nunique()), "positive": int(y.sum()),
        "fold_ids": [int(value) for value in sorted(np.unique(folds))],
        "morphology_columns": morphology_columns,
        "selected_fusion": best.model, "selected_scale": selected_scale,
        "whole_auprc": float(whole.auprc), "selected_auprc": float(best.auprc), "delta_auprc": delta_ap,
        "folds_improved": consistent, "oracle_gate_passed": gate,
        "convergence_warning_count": int(pd.DataFrame(audit_rows).convergence_warning.sum()),
        "metrics": metric_frame.to_dict("records"),
    }
    atomic_json(summary, reports / "record_gt_oracle_oof_summary.json")
    marker = reports / (".GT_ORACLE_PASS" if gate else ".GT_ORACLE_NO_GAIN")
    marker.write_text("pass\n" if gate else "no_gain\n")
    update_run_manifest(config, "record_gt_oracle_oof", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

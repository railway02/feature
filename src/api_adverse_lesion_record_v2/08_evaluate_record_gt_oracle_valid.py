#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from common import atomic_csv, atomic_json, load_config, update_run_manifest


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module); return module


def load_series(path: Path, series_uids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    raw = np.load(path); index = {str(uid): i for i, uid in enumerate(raw["series_uid"].astype(str))}
    if any(uid not in index for uid in series_uids):
        raise KeyError("Missing Valid series embedding")
    order = np.asarray([index[uid] for uid in series_uids])
    return np.asarray(raw["embeddings"][order], np.float32), np.stack([raw["missing_pre"][order], raw["missing_post"][order]], axis=1).astype(np.float64)


def load_morphology(path: Path, series_uids: list[str], columns: list[str]) -> np.ndarray:
    frame = pd.read_csv(path, dtype={"series_uid": str}).set_index("series_uid", drop=False).loc[series_uids]
    missing_columns = [column for column in columns if column not in frame.columns]
    if missing_columns:
        raise KeyError(f"Missing frozen GT morphology columns: {missing_columns}")
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    return numeric.to_numpy(np.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/root/autodl-tmp/aneurysm/configs/api_adverse_lesion_record_v2/config.json")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["project_root"]); reports = Path(config["paths"]["reports"]); outputs = Path(config["paths"]["outputs"])
    summary = json.loads((reports / "record_gt_oracle_oof_summary.json").read_text(encoding="utf-8"))
    if not summary["oracle_gate_passed"]:
        raise RuntimeError("GT Oracle OOF gate did not pass; refusing Valid evaluation")
    scale = str(summary["selected_scale"])
    # Train OOF serialized branch instances from this exact dynamic module
    # name. Register it before joblib.load in this new process.
    fixed = import_file(Path(config["fixed_trainer"]), "record_oracle_fixed")
    trainer_module = import_file(root / "code/api_adverse_lesion_record_v2/07_train_record_gt_oracle_oof.py", "record_oracle_trainer_classes")
    sys.modules["__main__"].OraclePreprocessor = trainer_module.OraclePreprocessor
    record = pd.read_csv(Path(config["paths"]["manifests"]) / "oracle_record_manifest.csv", dtype={"patient_id": str, "series_uid": str})
    valid = record[record.split == "Valid"].sort_values("record_uid").reset_index(drop=True)
    series = valid.series_uid.astype(str).tolist(); y = valid.target.to_numpy(np.int64)
    whole_e, whole_m = load_series(Path(config["whole_valid_tables"]) / "series_embeddings_5120.npz", series)
    local_e, local_m = load_series(outputs / f"cave_gt_context{scale}_tables/valid/series_embeddings_5120.npz", series)
    morphology_columns = list(summary["morphology_columns"])
    morphology = load_morphology(Path(config["gt_morphology_series"]), series, morphology_columns)
    data = {"images": {"whole": {"embedding": whole_e, "missing": whole_m}, f"local{scale}": {"embedding": local_e, "missing": local_m}}, "morphology": morphology}
    representations = {
        "G0_whole": (("whole",), False),
        "G1_gt_morphology": ((), True),
        f"G{2 if scale == '30' else 3}_gt_local{scale}": ((f"local{scale}",), False),
        summary["selected_fusion"]: (("whole", f"local{scale}"), False),
    }
    train_metrics = pd.read_csv(reports / "record_gt_oracle_oof_metrics.csv").set_index("model")
    rows = []; predictions = valid[["record_uid", "patient_id", "series_uid", "target"]].copy()
    for representation, _spec in representations.items():
        fold_probabilities = []
        fold_ids = [int(value) for value in summary["fold_ids"]]
        if len(fold_ids) != int(config["prediction"]["outer_folds"]):
            raise AssertionError(f"Unexpected frozen fold ids: {fold_ids}")
        for fold in fold_ids:
            artifact = joblib.load(outputs / "record_gt_oracle_models" / representation / f"fold_{fold}.joblib")
            fold_probabilities.append(artifact["model"].predict_proba(artifact["preprocessor"].transform(data))[:, 1])
        probability = np.mean(np.stack(fold_probabilities), axis=0)
        threshold = float(train_metrics.loc[representation, "threshold"])
        rows.append(fixed.metric_row("followup_rroc23_record_v2", representation, "Internal_Valid", y, probability, threshold))
        predictions[f"{representation}_probability"] = probability
    metrics = pd.DataFrame(rows)
    atomic_csv(metrics, reports / "record_gt_oracle_internal_valid_metrics.csv")
    atomic_csv(predictions, reports / "record_gt_oracle_internal_valid_predictions.csv")
    payload = {"status": "complete", "selected_scale": scale, "records": len(valid), "patients": int(valid.patient_id.nunique()), "positive": int(y.sum()), "valid_used_for_selection": False, "metrics": metrics.to_dict("records")}
    atomic_json(payload, reports / "record_gt_oracle_internal_valid_summary.json")
    (reports / ".INTERNAL_VALID_COMPLETE").write_text("pass\n")
    update_run_manifest(config, "record_gt_oracle_internal_valid", payload)
    print(json.dumps(payload, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())

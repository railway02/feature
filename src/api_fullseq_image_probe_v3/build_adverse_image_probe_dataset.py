#!/usr/bin/env python3
"""Build a pure image-feature adverse-outcome probe dataset.

Excel is read only for patient IDs and the adverse-outcome label.  No clinical,
treatment, anatomic, RROC, date, or follow-up column enters model features.

SEA-RAFT feature sets
---------------------
* prepost: the 212 default scientific core features (Pre106 + Post106).
* full: prepost plus all 106 scientific Pre-to-Post delta features and the
  delta-shape-compatible flag, aggregated label-blind from series to patient.
  QC/runtime columns are intentionally excluded.

CAVE feature sets
-----------------
* embedding: complete frozen Pre/Post 5120-D embeddings (10240 total).
* scalar: all frozen image-derived task scalar features except series_count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def normalize_patient_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\.0$", "", str(value).strip())


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_excel_labels(path: Path) -> dict[str, int]:
    frame = pd.read_excel(
        path,
        usecols=["病案号", "不良转归：1是；0否"],
        dtype=object,
    )
    frame["patient_id"] = frame["病案号"].map(normalize_patient_id)
    frame["target"] = pd.to_numeric(
        frame["不良转归：1是；0否"], errors="coerce"
    )
    labels: dict[str, int] = {}
    for patient_id, group in frame.groupby("patient_id"):
        values = group["target"].dropna().astype(int).unique().tolist()
        if len(values) == 1:
            labels[str(patient_id)] = int(values[0])
    return labels


def task_rows(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path, dtype={"patient_id": str})
    if frame["patient_id"].duplicated().any():
        raise AssertionError(f"Duplicate patient_id: {path}")
    target = pd.to_numeric(frame["target"], errors="raise").astype(int).to_numpy()
    return frame, target


def sea_feature_columns(schema_path: Path) -> tuple[list[str], list[str]]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    core = [item["feature_name"] for item in schema["phase_core_features"]]
    if len(core) != 106 or len(set(core)) != 106:
        raise AssertionError("Unexpected SEA core schema")
    prepost = [f"{phase}_{name}" for phase in ("pre", "post") for name in core]
    full = [*prepost, "delta_shape_compatible", *[f"delta_{name}" for name in core]]
    if len(prepost) != 212 or len(full) != 319:
        raise AssertionError("Unexpected SEA feature counts")
    return prepost, full


def aggregate_sea_patient(
    series_path: Path,
    patient_ids: list[str],
    prepost_columns: list[str],
    full_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    usecols = ["patient_id", "series_uid", "missing_pre", "missing_post", *full_columns]
    series = pd.read_csv(
        series_path,
        usecols=usecols,
        dtype={"patient_id": str, "series_uid": str},
    )
    relevant = series[series["patient_id"].astype(str).isin(patient_ids)].copy()
    if relevant["series_uid"].duplicated().any():
        raise AssertionError(f"Duplicate series_uid in {series_path}")
    grouped = relevant.groupby("patient_id", sort=False)
    missing_patients = sorted(set(patient_ids) - set(grouped.groups))
    if missing_patients:
        raise KeyError(f"Patients without SEA series: {missing_patients[:10]}")
    prepost_rows: list[np.ndarray] = []
    full_rows: list[np.ndarray] = []
    missing_rows: list[np.ndarray] = []
    series_counts: list[int] = []
    for patient_id in patient_ids:
        group = grouped.get_group(patient_id)
        numeric = group[full_columns].apply(pd.to_numeric, errors="coerce")
        median = numeric.median(axis=0, skipna=True)
        prepost_rows.append(median[prepost_columns].to_numpy(dtype=np.float32))
        full_rows.append(median[full_columns].to_numpy(dtype=np.float32))
        missing_rows.append(np.asarray([
            float(pd.to_numeric(group["missing_pre"], errors="coerce").fillna(1).min()),
            float(pd.to_numeric(group["missing_post"], errors="coerce").fillna(1).min()),
        ], dtype=np.float32))
        series_counts.append(len(group))
    missing = np.stack(missing_rows)
    if missing[:, 1].any():
        raise AssertionError("SEA Post missing at patient level")
    audit = {
        "series_rows": int(len(relevant)),
        "patients": len(patient_ids),
        "series_per_patient_min": int(min(series_counts)),
        "series_per_patient_max": int(max(series_counts)),
        "patient_missing_pre": int(missing[:, 0].sum()),
        "patient_missing_post": int(missing[:, 1].sum()),
        "qc_columns_used": 0,
        "runtime_columns_used": 0,
    }
    return (
        np.stack(prepost_rows).astype(np.float32),
        np.stack(full_rows).astype(np.float32),
        missing.astype(np.float32),
        audit,
    )


def load_cave_task(
    task_dir: Path,
    split: str,
    patient_ids: list[str],
    expected_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    meta = pd.read_csv(task_dir / f"{split}_meta.csv", dtype={"patient_id": str})
    if meta["patient_id"].astype(str).tolist() != patient_ids:
        raise AssertionError(f"CAVE {split} metadata row order changed")
    with np.load(task_dir / f"{split}_features.npz") as raw:
        deep = np.array(raw["deep"], dtype=np.float32, copy=True)
        scalar = np.array(raw["scalar"], dtype=np.float32, copy=True)
        missing = np.array(raw["missing"], dtype=np.float32, copy=True)
        target = np.array(raw["target"], dtype=np.int64, copy=True)
    if not np.array_equal(target, expected_target):
        raise AssertionError(f"CAVE {split} targets changed")
    config = json.loads((task_dir / "task_config.json").read_text(encoding="utf-8"))
    scalar_names = list(config["scalar_columns"])
    if scalar.shape[1] != len(scalar_names):
        raise AssertionError("CAVE scalar schema mismatch")
    keep = [index for index, name in enumerate(scalar_names) if name != "series_count"]
    kept_names = [scalar_names[index] for index in keep]
    scalar = scalar[:, keep]
    if deep.shape != (len(patient_ids), 10240):
        raise AssertionError(f"Unexpected CAVE deep shape {deep.shape}")
    if missing.shape != (len(patient_ids), 2) or missing[:, 1].any():
        raise AssertionError("Unexpected CAVE missing flags")
    return deep, scalar, missing, kept_names


def build_split(
    project: Path,
    split: str,
    labels: dict[str, int],
    output: Path,
    prepost_columns: list[str],
    full_columns: list[str],
) -> dict[str, Any]:
    task_dir = project / "outputs/api_fullseq_cave_v3_tasks/adverse_patient"
    meta, target = task_rows(task_dir / f"{split}_meta.csv")
    patient_ids = meta["patient_id"].astype(str).tolist()
    excel_target = np.asarray([labels[patient_id] for patient_id in patient_ids], dtype=np.int64)
    if not np.array_equal(target, excel_target):
        raise AssertionError(f"{split}: task labels differ from Excel")
    sea_prepost, sea_full, sea_missing, sea_audit = aggregate_sea_patient(
        project / f"outputs/api_fullseq_v3_features/full/{split}/series_features.csv",
        patient_ids,
        prepost_columns,
        full_columns,
    )
    cave_embedding, cave_scalar, cave_missing, cave_scalar_names = load_cave_task(
        task_dir, split, patient_ids, target
    )
    if not np.array_equal(sea_missing, cave_missing):
        raise AssertionError(f"{split}: CAVE/SEA patient missing flags differ")
    atomic_npz(
        output / f"{split}.npz",
        patient_id=np.asarray(patient_ids),
        sea_prepost=sea_prepost,
        sea_full=sea_full,
        cave_embedding=cave_embedding,
        cave_scalar=cave_scalar,
        missing=cave_missing,
        target=target,
    )
    return {
        "rows": len(patient_ids),
        "positive": int(target.sum()),
        "negative": int((target == 0).sum()),
        "sea_prepost_shape": list(sea_prepost.shape),
        "sea_full_shape": list(sea_full.shape),
        "cave_embedding_shape": list(cave_embedding.shape),
        "cave_scalar_shape": list(cave_scalar.shape),
        "missing_shape": list(cave_missing.shape),
        "sea_audit": sea_audit,
        "cave_scalar_feature_count": len(cave_scalar_names),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="/root/autodl-tmp/aneurysm")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    project = Path(args.project).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)
    train_excel = project / "metadata/Train.xlsx"
    valid_excel = project / "metadata/valid.xlsx"
    train_labels = read_excel_labels(train_excel)
    valid_labels = read_excel_labels(valid_excel)
    prepost_columns, full_columns = sea_feature_columns(
        project / "outputs/api_fullseq_v3_features/full/train/feature_schema.json"
    )
    train_summary = build_split(
        project, "train", train_labels, output, prepost_columns, full_columns
    )
    valid_summary = build_split(
        project, "valid", valid_labels, output, prepost_columns, full_columns
    )
    with np.load(output / "train.npz") as train_raw, np.load(output / "valid.npz") as valid_raw:
        overlap = set(train_raw["patient_id"].astype(str)) & set(valid_raw["patient_id"].astype(str))
    if overlap:
        raise AssertionError(f"Train/Valid patient overlap={len(overlap)}")
    schema = {
        "version": "api_fullseq_image_probe_v3_dataset_1",
        "task": "patient-level adverse outcome",
        "predictor_scope": "image-derived features only",
        "metadata_columns_read": ["病案号", "不良转归：1是；0否"],
        "metadata_predictor_columns": [],
        "valid_labels_used_for": "final metrics only",
        "feature_sets": {
            "sea_prepost": {"dimension": 212, "columns": prepost_columns},
            "sea_full": {
                "dimension": 319,
                "columns": full_columns,
                "includes": "Pre106 + Post106 + Delta106 + delta_shape_compatible",
                "qc_excluded": True,
                "runtime_excluded": True,
            },
            "cave_embedding": {"dimension": 10240, "pca": False},
            "cave_scalar": {
                "dimension": train_summary["cave_scalar_shape"][1],
                "series_count_excluded": True,
            },
        },
        "train_valid_patient_overlap": 0,
    }
    atomic_json(schema, output / "feature_schema.json")
    summary = {
        "version": "api_fullseq_image_probe_v3_dataset_1",
        "train": train_summary,
        "valid": valid_summary,
        "input_sha256": {
            "Train.xlsx": sha256(train_excel),
            "valid.xlsx": sha256(valid_excel),
            "sea_train_series_features": sha256(
                project / "outputs/api_fullseq_v3_features/full/train/series_features.csv"
            ),
            "sea_valid_series_features": sha256(
                project / "outputs/api_fullseq_v3_features/full/valid/series_features.csv"
            ),
        },
        "output_sha256": {
            "train.npz": sha256(output / "train.npz"),
            "valid.npz": sha256(output / "valid.npz"),
        },
    }
    atomic_json(summary, output / "build_summary.json")
    atomic_json(summary, output / ".DATASET_SUCCESS")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

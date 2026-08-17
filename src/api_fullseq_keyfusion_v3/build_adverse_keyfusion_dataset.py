#!/usr/bin/env python3
"""Build adverse-outcome key-representation fusion datasets.

Primary inputs
--------------
* CAVE: complete frozen Pre/Post 5120-D embeddings, reshaped into the ten
  documented 512-channel blocks per phase.
* SEA-RAFT: dense cached flow/output maps from pair_maps.npz.  Fourteen key
  model-output channels are summarized over early/middle/late time, temporal
  standard deviation, and temporal maximum while retaining a 16x16 spatial
  grid.  Series representations are aggregated to patient level by label-blind
  median.
* Clinical: leakage-screened demographics, anatomy, multiplicity, and treatment
  descriptors from Train.xlsx/valid.xlsx.

Outcome, immediate/follow-up RROC, dates, and follow-up duration never enter
the feature arrays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MAP_CHANNELS = (
    "forward_u_norm",
    "forward_v_norm",
    "backward_u_norm",
    "backward_v_norm",
    "residual_u_norm",
    "residual_v_norm",
    "residual_mag_norm",
    "fb_relative",
    "uncertainty_log",
    "soft_weight",
    "filling_front",
    "persistent",
    "washout_front",
    "hard_valid",
)
MASK_CHANNELS = {"filling_front", "persistent", "washout_front", "hard_valid"}
TEMPORAL_SUMMARIES = ("early_mean", "middle_mean", "late_mean", "std", "max")
SPATIAL_SIZE = 16
PHASES = ("pre", "post")

SAFE_NUMERIC = {
    "age": "年龄",
    "record_count": None,
    "multiple": "是否多发",
    "flow_diverter": "是否密网支架",
    "coil": "是否使用弹簧圈",
}
SAFE_CATEGORICAL = {
    "sex": "性别",
    "procedure": "手术方式：0单栓；1SAC；2密网；3密网+弹簧圈",
    "stent": "支架类型",
    "side": "侧别",
    "location": "部位",
}
FORBIDDEN_METADATA = {
    "病案号",
    "姓名",
    "术后即刻RROC",
    "随访RROC123",
    "不良转归：1是；0否",
    "DSA时间",
    "最后一次随访时间",
    "随访间隔（月）",
    "随访时间",
    "随访时间.1",
}


def normalize_patient_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\.0$", "", str(value).strip())


def normalize_category(value: Any) -> str:
    if value is None or pd.isna(value) or not str(value).strip():
        return "__MISSING__"
    return re.sub(r"\s+", " ", str(value).strip().lower())


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


def load_excel(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, dtype=object).copy()
    frame["patient_id"] = frame["病案号"].map(normalize_patient_id)
    if (frame["patient_id"] == "").any():
        raise AssertionError(f"Blank patient IDs in {path}")
    return frame


def category_vocab(train: pd.DataFrame) -> dict[str, list[str]]:
    vocabulary: dict[str, list[str]] = {}
    for logical, source in SAFE_CATEGORICAL.items():
        values = train[source].map(normalize_category)
        counts = Counter(values.tolist())
        if logical == "stent":
            kept = sorted(value for value, count in counts.items() if count >= 5)
            extras = ["__RARE__", "__UNKNOWN__"]
        else:
            kept = sorted(counts)
            extras = ["__UNKNOWN__"]
        vocabulary[logical] = list(dict.fromkeys([*kept, *extras]))
    return vocabulary


def mapped_category(logical: str, value: Any, vocabulary: dict[str, list[str]]) -> str:
    normalized = normalize_category(value)
    if normalized in vocabulary[logical]:
        return normalized
    if logical == "stent" and "__RARE__" in vocabulary[logical]:
        return "__RARE__"
    return "__UNKNOWN__"


def clinical_feature_names(vocabulary: dict[str, list[str]]) -> list[str]:
    names = list(SAFE_NUMERIC)
    for logical in SAFE_CATEGORICAL:
        names.extend(f"{logical}::{value}" for value in vocabulary[logical])
    return names


def clinical_matrix(
    excel: pd.DataFrame,
    patient_ids: list[str],
    vocabulary: dict[str, list[str]],
) -> tuple[np.ndarray, list[str]]:
    names = clinical_feature_names(vocabulary)
    index = {name: position for position, name in enumerate(names)}
    rows: list[np.ndarray] = []
    grouped = {patient_id: group for patient_id, group in excel.groupby("patient_id")}
    for patient_id in patient_ids:
        if patient_id not in grouped:
            raise KeyError(f"Missing clinical rows for patient {patient_id}")
        group = grouped[patient_id]
        vector = np.zeros(len(names), dtype=np.float32)
        ages = pd.to_numeric(group[SAFE_NUMERIC["age"]], errors="coerce").dropna()
        vector[index["age"]] = float(ages.median()) if len(ages) else np.nan
        vector[index["record_count"]] = float(len(group))
        for logical in ("multiple", "flow_diverter", "coil"):
            values = pd.to_numeric(group[SAFE_NUMERIC[logical]], errors="coerce").dropna()
            vector[index[logical]] = float(values.max()) if len(values) else np.nan
        for logical, source in SAFE_CATEGORICAL.items():
            observed = {
                mapped_category(logical, value, vocabulary)
                for value in group[source].tolist()
            }
            for value in observed:
                vector[index[f"{logical}::{value}"]] = 1.0
        rows.append(vector)
    return np.stack(rows).astype(np.float32), names


def spatial_pool(values: np.ndarray) -> np.ndarray:
    if values.shape != (96, 96):
        raise AssertionError(f"Unexpected SEA map shape {values.shape}")
    factor = 96 // SPATIAL_SIZE
    return values.reshape(SPATIAL_SIZE, factor, SPATIAL_SIZE, factor).mean(axis=(1, 3))


def temporal_bins(length: int) -> list[np.ndarray]:
    if length < 1:
        raise AssertionError("SEA map has no temporal pairs")
    bins = np.array_split(np.arange(length), 3)
    fallback = np.arange(length)
    return [part if len(part) else fallback for part in bins]


def phase_key_maps(path: Path) -> np.ndarray:
    with np.load(path) as raw:
        missing = set(MAP_CHANNELS) - set(raw.files)
        if missing:
            raise KeyError(f"{path}: missing map channels {sorted(missing)}")
        channel_outputs: list[np.ndarray] = []
        temporal_length: int | None = None
        for channel in MAP_CHANNELS:
            values = np.asarray(raw[channel], dtype=np.float32)
            if values.ndim != 3 or values.shape[1:] != (96, 96):
                raise AssertionError(f"{path}: {channel} shape={values.shape}")
            if temporal_length is None:
                temporal_length = int(values.shape[0])
            elif temporal_length != int(values.shape[0]):
                raise AssertionError(f"{path}: inconsistent temporal length")
            if channel in MASK_CHANNELS:
                values = values / 255.0
            values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
            bins = temporal_bins(len(values))
            summaries = [values[part].mean(axis=0) for part in bins]
            summaries.extend([values.std(axis=0), values.max(axis=0)])
            channel_outputs.extend(spatial_pool(summary) for summary in summaries)
    output = np.stack(channel_outputs).astype(np.float32)
    expected = (len(MAP_CHANNELS) * len(TEMPORAL_SUMMARIES), SPATIAL_SIZE, SPATIAL_SIZE)
    if output.shape != expected or not np.isfinite(output).all():
        raise AssertionError(f"Invalid SEA key tensor {output.shape}: {path}")
    return output


def series_key_tensor(
    pairdata_root: Path, patient_id: str, series_uid: str
) -> tuple[np.ndarray, np.ndarray]:
    tensors: list[np.ndarray] = []
    missing: list[float] = []
    for phase in PHASES:
        path = pairdata_root / patient_id / series_uid / phase / "pair_maps.npz"
        if path.is_file():
            tensors.append(phase_key_maps(path))
            missing.append(0.0)
        else:
            tensors.append(np.full(
                (len(MAP_CHANNELS) * len(TEMPORAL_SUMMARIES), SPATIAL_SIZE, SPATIAL_SIZE),
                np.nan,
                dtype=np.float32,
            ))
            missing.append(1.0)
    if missing[1] != 0.0:
        raise AssertionError(f"Post SEA maps missing for {patient_id}/{series_uid}")
    return np.stack(tensors), np.asarray(missing, dtype=np.float32)


def build_patient_sea(
    patient_ids: list[str],
    series_features: pd.DataFrame,
    pairdata_root: Path,
    workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    relevant = series_features[
        series_features["patient_id"].astype(str).isin(patient_ids)
    ].copy()
    grouped = {
        patient_id: group["series_uid"].astype(str).tolist()
        for patient_id, group in relevant.groupby("patient_id", sort=False)
    }
    missing_patients = sorted(set(patient_ids) - set(grouped))
    if missing_patients:
        raise KeyError(f"Patients without SEA series: {missing_patients[:10]}")
    jobs = [
        (patient_id, series_uid)
        for patient_id in patient_ids
        for series_uid in grouped[patient_id]
    ]

    def run(job: tuple[str, str]) -> tuple[str, str, np.ndarray, np.ndarray]:
        patient_id, series_uid = job
        tensor, missing = series_key_tensor(pairdata_root, patient_id, series_uid)
        return patient_id, series_uid, tensor, missing

    results: dict[str, list[tuple[str, np.ndarray, np.ndarray]]] = {
        patient_id: [] for patient_id in patient_ids
    }
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for count, (patient_id, series_uid, tensor, missing) in enumerate(
            executor.map(run, jobs), start=1
        ):
            results[patient_id].append((series_uid, tensor, missing))
            if count % 100 == 0 or count == len(jobs):
                print(f"[SEA MAPS] {count}/{len(jobs)} series", flush=True)

    patient_tensors: list[np.ndarray] = []
    patient_missing: list[np.ndarray] = []
    series_counts: list[int] = []
    for patient_id in patient_ids:
        ordered = sorted(results[patient_id], key=lambda item: item[0])
        stack = np.stack([item[1] for item in ordered]).astype(np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            patient = np.nanmedian(stack, axis=0).astype(np.float32)
        missing = np.isnan(patient).all(axis=(1, 2, 3)).astype(np.float32)
        if missing[1] != 0.0:
            raise AssertionError(f"Patient {patient_id} has no Post SEA maps")
        patient_tensors.append(patient)
        patient_missing.append(missing)
        series_counts.append(len(ordered))
    return (
        np.stack(patient_tensors).astype(np.float32),
        np.stack(patient_missing).astype(np.float32),
        np.asarray(series_counts, dtype=np.int16),
    )


def load_cave_embeddings(
    path: Path, patient_ids: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as raw:
        ids = raw["patient_id"].astype(str)
        embeddings = np.asarray(raw["embeddings"], dtype=np.float32)
    if embeddings.ndim != 3 or embeddings.shape[1:] != (2, 5120):
        raise AssertionError(f"Unexpected CAVE embedding shape {embeddings.shape}")
    lookup = {patient_id: index for index, patient_id in enumerate(ids)}
    missing = sorted(set(patient_ids) - set(lookup))
    if missing:
        raise KeyError(f"Patients without CAVE embeddings: {missing[:10]}")
    aligned = np.stack([embeddings[lookup[patient_id]] for patient_id in patient_ids])
    phase_missing = np.isnan(aligned).all(axis=2).astype(np.float32)
    if phase_missing[:, 1].any():
        raise AssertionError("CAVE Post embeddings missing")
    return aligned.reshape(len(aligned), 2, 10, 512), phase_missing


def task_meta(path: Path) -> tuple[list[str], np.ndarray]:
    frame = pd.read_csv(path, dtype={"patient_id": str})
    if frame["patient_id"].duplicated().any():
        raise AssertionError(f"Duplicate patient_id in {path}")
    target = pd.to_numeric(frame["target"], errors="raise").astype(int).to_numpy()
    if not set(np.unique(target)).issubset({0, 1}):
        raise AssertionError(f"Non-binary target in {path}")
    return frame["patient_id"].astype(str).tolist(), target


def build_split(
    split: str,
    project: Path,
    excel: pd.DataFrame,
    vocabulary: dict[str, list[str]],
    output: Path,
    workers: int,
) -> dict[str, Any]:
    task_root = project / "outputs/api_fullseq_cave_v3_tasks/adverse_patient"
    patient_ids, target = task_meta(task_root / f"{split}_meta.csv")
    cave_path = project / f"outputs/api_fullseq_cave_v3_tables/{split}/patient_median_embeddings_5120.npz"
    cave, cave_missing = load_cave_embeddings(cave_path, patient_ids)
    series_path = project / f"outputs/api_fullseq_v3_features/full/{split}/series_features.csv"
    series = pd.read_csv(series_path, dtype={"patient_id": str, "series_uid": str})
    sea, sea_missing, series_counts = build_patient_sea(
        patient_ids,
        series,
        project / f"outputs/api_fullseq_v3_pairdata/full/{split}",
        workers,
    )
    clinical, clinical_names = clinical_matrix(excel, patient_ids, vocabulary)
    if len(patient_ids) != len(cave) or len(cave) != len(sea) or len(sea) != len(clinical):
        raise AssertionError("Split arrays are not row aligned")
    missing = np.concatenate([cave_missing, sea_missing], axis=1).astype(np.float32)
    atomic_npz(
        output / f"{split}.npz",
        patient_id=np.asarray(patient_ids),
        cave=cave.astype(np.float32),
        sea=sea.astype(np.float16),
        clinical=clinical.astype(np.float32),
        missing=missing,
        series_count=series_counts,
        target=target.astype(np.int64),
    )
    return {
        "split": split,
        "rows": len(patient_ids),
        "positive": int(target.sum()),
        "negative": int((target == 0).sum()),
        "cave_shape": list(cave.shape),
        "sea_shape": list(sea.shape),
        "clinical_shape": list(clinical.shape),
        "missing_shape": list(missing.shape),
        "series_total": int(series_counts.sum()),
        "series_per_patient_min": int(series_counts.min()),
        "series_per_patient_max": int(series_counts.max()),
        "missing_cave_pre": int(cave_missing[:, 0].sum()),
        "missing_sea_pre": int(sea_missing[:, 0].sum()),
        "clinical_feature_count": len(clinical_names),
        "all_post_present": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="/root/autodl-tmp/aneurysm")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    project = Path(args.project).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)
    train_excel_path = project / "metadata/Train.xlsx"
    valid_excel_path = project / "metadata/valid.xlsx"
    train_excel = load_excel(train_excel_path)
    valid_excel = load_excel(valid_excel_path)
    if set(train_excel["patient_id"]) & set(valid_excel["patient_id"]):
        raise AssertionError("Train/Valid patient overlap in Excel")
    vocabulary = category_vocab(train_excel)
    names = clinical_feature_names(vocabulary)
    train_summary = build_split(
        "train", project, train_excel, vocabulary, output, args.workers
    )
    valid_summary = build_split(
        "valid", project, valid_excel, vocabulary, output, args.workers
    )
    with np.load(output / "train.npz") as train_raw, np.load(output / "valid.npz") as valid_raw:
        overlap = set(train_raw["patient_id"].astype(str)) & set(
            valid_raw["patient_id"].astype(str)
        )
    if overlap:
        raise AssertionError(f"Final Train/Valid patient overlap={len(overlap)}")
    schema = {
        "version": "api_fullseq_keyfusion_v3_dataset_1",
        "target": "patient-level adverse outcome 1 vs 0",
        "primary_features": {
            "cave": {
                "source": "patient_median_embeddings_5120.npz",
                "shape_per_patient": [2, 10, 512],
                "phases": list(PHASES),
                "blocks_per_phase": 10,
                "channels_per_block": 512,
                "reduction_before_training": "none",
            },
            "searaft": {
                "source": "series phase pair_maps.npz, label-blind patient median",
                "shape_per_patient": [
                    2,
                    len(MAP_CHANNELS) * len(TEMPORAL_SUMMARIES),
                    SPATIAL_SIZE,
                    SPATIAL_SIZE,
                ],
                "map_channels": list(MAP_CHANNELS),
                "temporal_summaries": list(TEMPORAL_SUMMARIES),
                "spatial_pool": f"96x96 average pooled to {SPATIAL_SIZE}x{SPATIAL_SIZE}",
                "pca": False,
            },
        },
        "clinical": {
            "feature_names": names,
            "feature_count": len(names),
            "safe_source_columns": [
                value for value in [*SAFE_NUMERIC.values(), *SAFE_CATEGORICAL.values()]
                if value is not None
            ],
            "forbidden_columns": sorted(FORBIDDEN_METADATA),
            "vocabulary": vocabulary,
            "valid_categories_used_to_build_vocabulary": False,
        },
        "train_valid_patient_overlap": 0,
        "labels_read_only_after_frozen_image_features": True,
    }
    atomic_json(schema, output / "feature_schema.json")
    summary = {
        "version": "api_fullseq_keyfusion_v3_dataset_1",
        "train": train_summary,
        "valid": valid_summary,
        "feature_schema_sha256": sha256(output / "feature_schema.json"),
        "input_sha256": {
            "Train.xlsx": sha256(train_excel_path),
            "valid.xlsx": sha256(valid_excel_path),
            "cave_train_embeddings": sha256(
                project / "outputs/api_fullseq_cave_v3_tables/train/patient_median_embeddings_5120.npz"
            ),
            "cave_valid_embeddings": sha256(
                project / "outputs/api_fullseq_cave_v3_tables/valid/patient_median_embeddings_5120.npz"
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

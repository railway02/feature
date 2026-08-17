#!/usr/bin/env python3
"""Build phase-, series-, and patient-level tables from the frozen CAVE feature bank."""
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import sha256_file, write_csv_atomic, write_json_atomic, write_parquet_atomic
from manifest import load_manifest
from pooling import PRIMARY_BLOCKS


def _load_json_numbers(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): (float("nan") if value is None else float(value)) for key, value in payload.items()}


def _phase_dir(root: Path, split: str, patient_id: str, series_uid: str, phase: str) -> Path:
    return root / split.casefold() / patient_id / series_uid / phase


def _load_phase(root: Path, row: pd.Series, phase: str, schema: dict[str, Any]):
    directory = _phase_dir(root, str(row["split"]), str(row["patient_id"]), str(row["series_uid"]), phase)
    if not (directory / ".SUCCESS.json").is_file():
        return None
    vector = np.load(directory / "embedding_5120.npy").astype(np.float32)
    if vector.shape != (schema["embedding_dimension"],) or not np.isfinite(vector).all():
        raise AssertionError(f"Invalid embedding: {directory}")
    scalar = _load_json_numbers(directory / "scalar_features.json")
    if list(scalar) != schema["scalar_feature_names"]:
        raise AssertionError(f"Scalar schema/order mismatch: {directory}")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    qc = json.loads((directory / "qc.json").read_text(encoding="utf-8"))
    return {"embedding": vector, "scalar": scalar, "metadata": metadata, "qc": qc}


def _group_distances(pre: np.ndarray, post: np.ndarray) -> dict[str, float]:
    output: dict[str, float] = {}
    for index, name in enumerate(PRIMARY_BLOCKS):
        a = pre[index * 512:(index + 1) * 512].astype(np.float64)
        b = post[index * 512:(index + 1) * 512].astype(np.float64)
        norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        cosine = float(np.dot(a, b) / max(norm_a * norm_b, 1e-8))
        output[f"{name}_prepost_cosine"] = cosine
        output[f"{name}_prepost_cosine_distance"] = float(1.0 - cosine)
        output[f"{name}_prepost_normalized_l2"] = float(
            np.linalg.norm(a / max(norm_a, 1e-8) - b / max(norm_b, 1e-8))
        )
        output[f"{name}_prepost_log_norm_ratio"] = float(
            np.log(max(norm_b, 1e-8) / max(norm_a, 1e-8))
        )
    return output


def _nanmedian_stack(values: list[np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
    if not values:
        return np.full(shape, np.nan, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(np.stack(values).astype(np.float32), axis=0).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-split")
    parser.add_argument("--expected-series", type=int)
    parser.add_argument("--expected-patients", type=int)
    parser.add_argument("--verify-files", action="store_true")
    args = parser.parse_args()

    bundle = load_manifest(
        args.manifest,
        expected_split=args.expected_split,
        verify_files=args.verify_files,
        expected_counts=None,
    )
    schema_path = args.feature_root / "feature_schema.json"
    if not schema_path.is_file():
        raise FileNotFoundError(schema_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    phase_rows: list[dict[str, Any]] = []
    series_rows: list[dict[str, Any]] = []
    series_embeddings: list[np.ndarray] = []
    series_uids: list[str] = []
    patient_ids: list[str] = []

    for _, row in bundle.frame.iterrows():
        patient_id = str(row["patient_id"])
        series_uid = str(row["series_uid"])
        loaded = {phase: _load_phase(args.feature_root, row, phase, schema) for phase in ("pre", "post")}
        for phase in ("pre", "post"):
            item = loaded[phase]
            if item is None:
                continue
            phase_row = {
                "patient_id": patient_id,
                "series_uid": series_uid,
                "split": str(row["split"]),
                "source_type": str(row.get("source_type", "")),
                "series_id": str(row.get("series_id", "")),
                "phase": phase,
                "frame_list_hash": str(row.get(f"{phase}_frame_list_hash", "")),
                **item["scalar"],
            }
            phase_rows.append(phase_row)

        missing_pre = loaded["pre"] is None
        missing_post = loaded["post"] is None
        pre_vector = np.full(5120, np.nan, dtype=np.float32) if missing_pre else loaded["pre"]["embedding"]
        post_vector = np.full(5120, np.nan, dtype=np.float32) if missing_post else loaded["post"]["embedding"]
        series_embeddings.append(np.stack([pre_vector, post_vector]))
        series_uids.append(series_uid)
        patient_ids.append(patient_id)
        scalar_row: dict[str, Any] = {
            "patient_id": patient_id,
            "series_uid": series_uid,
            "split": str(row["split"]),
            "source_type": str(row.get("source_type", "")),
            "series_id": str(row.get("series_id", "")),
            "missing_pre": int(missing_pre),
            "missing_post": int(missing_post),
        }
        for phase in ("pre", "post"):
            item = loaded[phase]
            if item is None:
                scalar_row.update({f"{phase}_{name}": np.nan for name in schema["scalar_feature_names"]})
            else:
                scalar_row.update({f"{phase}_{name}": value for name, value in item["scalar"].items()})
        if not missing_pre and not missing_post:
            scalar_row.update(_group_distances(pre_vector, post_vector))
            for name in schema["scalar_feature_names"]:
                pre_value = scalar_row[f"pre_{name}"]
                post_value = scalar_row[f"post_{name}"]
                scalar_row[f"delta_{name}"] = (
                    float(post_value - pre_value)
                    if math.isfinite(float(pre_value)) and math.isfinite(float(post_value))
                    else np.nan
                )
        else:
            scalar_row.update({f"{name}_prepost_{suffix}": np.nan for name in PRIMARY_BLOCKS for suffix in (
                "cosine", "cosine_distance", "normalized_l2", "log_norm_ratio"
            )})
            scalar_row.update({f"delta_{name}": np.nan for name in schema["scalar_feature_names"]})
        series_rows.append(scalar_row)

    embeddings_array = np.stack(series_embeddings).astype(np.float32)
    series_frame = pd.DataFrame(series_rows)
    phase_frame = pd.DataFrame(phase_rows)
    if args.expected_series is not None and len(series_frame) != args.expected_series:
        raise AssertionError(f"Expected {args.expected_series} series, got {len(series_frame)}")
    if args.expected_patients is not None and series_frame["patient_id"].nunique() != args.expected_patients:
        raise AssertionError(
            f"Expected {args.expected_patients} patients, got {series_frame['patient_id'].nunique()}"
        )
    if series_frame["series_uid"].duplicated().any():
        raise AssertionError("Duplicate series_uid in feature table")

    np.savez_compressed(
        args.output_dir / "series_embeddings_5120.npz",
        series_uid=np.asarray(series_uids, dtype=str),
        patient_id=np.asarray(patient_ids, dtype=str),
        embeddings=embeddings_array,
        missing_pre=series_frame["missing_pre"].to_numpy(np.uint8),
        missing_post=series_frame["missing_post"].to_numpy(np.uint8),
    )
    write_csv_atomic(phase_frame, args.output_dir / "phase_scalar_features.csv")
    write_parquet_atomic(phase_frame, args.output_dir / "phase_scalar_features.parquet")
    write_csv_atomic(series_frame, args.output_dir / "series_scalar_features.csv")
    write_parquet_atomic(series_frame, args.output_dir / "series_scalar_features.parquet")

    patient_embedding_rows: list[np.ndarray] = []
    patient_scalar_rows: list[dict[str, Any]] = []
    patient_order = sorted(series_frame["patient_id"].astype(str).unique())
    series_index = {uid: index for index, uid in enumerate(series_uids)}
    numeric_columns = [
        column for column in series_frame.columns
        if column not in {"patient_id", "series_uid", "split", "source_type", "series_id"}
    ]
    for patient_id in patient_order:
        group = series_frame[series_frame["patient_id"].astype(str) == patient_id]
        indices = [series_index[str(uid)] for uid in group["series_uid"]]
        patient_embedding_rows.append(_nanmedian_stack([embeddings_array[index] for index in indices], (2, 5120)))
        row: dict[str, Any] = {
            "patient_id": patient_id,
            "split": str(group["split"].iloc[0]),
            "series_count": int(len(group)),
            "missing_pre_all": int(group["missing_pre"].all()),
            "missing_post_all": int(group["missing_post"].all()),
        }
        numeric = group[numeric_columns].apply(pd.to_numeric, errors="coerce")
        medians = numeric.median(axis=0, skipna=True)
        row.update({column: medians[column] for column in numeric_columns})
        patient_scalar_rows.append(row)
    patient_embeddings = np.stack(patient_embedding_rows).astype(np.float32)
    patient_frame = pd.DataFrame(patient_scalar_rows)
    np.savez_compressed(
        args.output_dir / "patient_median_embeddings_5120.npz",
        patient_id=np.asarray(patient_order, dtype=str),
        embeddings=patient_embeddings,
        missing_pre=np.isnan(patient_embeddings[:, 0]).all(axis=1).astype(np.uint8),
        missing_post=np.isnan(patient_embeddings[:, 1]).all(axis=1).astype(np.uint8),
    )
    write_csv_atomic(patient_frame, args.output_dir / "patient_median_scalar_features.csv")
    write_parquet_atomic(patient_frame, args.output_dir / "patient_median_scalar_features.parquet")

    audit = {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "feature_schema_sha256": schema["schema_sha256"],
        "series": int(len(series_frame)),
        "patients": int(series_frame["patient_id"].nunique()),
        "phases": int(len(phase_frame)),
        "pre_phases": int((phase_frame["phase"] == "pre").sum()),
        "post_phases": int((phase_frame["phase"] == "post").sum()),
        "series_embeddings_shape": list(embeddings_array.shape),
        "patient_embeddings_shape": list(patient_embeddings.shape),
        "series_scalar_columns": int(len(series_frame.columns)),
        "phase_scalar_columns": int(len(phase_frame.columns)),
        "inf_count_series_scalar": int(np.isinf(series_frame.select_dtypes(include=[np.number]).to_numpy()).sum()),
        "inf_count_phase_scalar": int(np.isinf(phase_frame.select_dtypes(include=[np.number]).to_numpy()).sum()),
        "train_valid_labels_read": False,
    }
    write_json_atomic(args.output_dir / "build_audit.json", audit)
    write_json_atomic(args.output_dir / "feature_schema.json", schema)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

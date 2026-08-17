#!/usr/bin/env python3
"""Hard audit a CAVE feature bank against its frozen manifest and schema."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import write_csv_atomic, write_json_atomic
from manifest import load_manifest

REQUIRED_PHASE_FILES = (
    "embedding_5120.npy", "embedding_views_5120.npz", "f4_last_ensemble.fp16.npy",
    "f5_last_ensemble.fp16.npy", "phase_trajectories_16.fp16.npz",
    "probabilities_original.fp16.npz", "curves.npz", "scalar_features.json",
    "metadata.json", "qc.json", ".SUCCESS.json", "input_mosaic.jpg",
    "artery_vein_overlay.png", "artery_probability.png", "vein_probability.png",
    "vessel_probability.png", "vessel_union_probability.png",
)


def _phase_dir(root: Path, plan) -> Path:
    return root / plan.split.casefold() / plan.patient_id / plan.series_uid / plan.phase


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-split")
    parser.add_argument("--expected-series", type=int, required=True)
    parser.add_argument("--expected-patients", type=int, required=True)
    parser.add_argument("--expected-pre", type=int, required=True)
    parser.add_argument("--expected-post", type=int, required=True)
    parser.add_argument("--verify-source-files", action="store_true")
    args = parser.parse_args()

    expected_counts = {
        "series": args.expected_series, "patients": args.expected_patients,
        "pre": args.expected_pre, "post": args.expected_post,
        "phases": args.expected_pre + args.expected_post,
    }
    bundle = load_manifest(
        args.manifest, expected_split=args.expected_split,
        verify_files=args.verify_source_files, expected_counts=expected_counts,
    )
    schema_path = args.feature_root / "feature_schema.json"
    if not schema_path.is_file():
        raise FileNotFoundError(schema_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    scalar_names = schema["scalar_feature_names"]
    details: list[dict[str, Any]] = []
    embeddings: list[np.ndarray] = []
    report: dict[str, Any] = {
        **bundle.summary,
        "expected": expected_counts,
        "missing_phase_directories": 0,
        "missing_required_files": 0,
        "wrong_embedding_shape": 0,
        "nonfinite_embeddings": 0,
        "all_zero_embeddings": 0,
        "wrong_scalar_schema": 0,
        "wrong_provenance": 0,
        "wrong_probability_range": 0,
        "wrong_probability_shape": 0,
        "success_pre": 0,
        "success_post": 0,
        "empty_vessel_probability_count": 0,
        "low_vessel_probability_count": 0,
    }
    for plan in bundle.plans:
        directory = _phase_dir(args.feature_root, plan)
        row = {"patient_id": plan.patient_id, "series_uid": plan.series_uid, "phase": plan.phase}
        if not directory.is_dir():
            report["missing_phase_directories"] += 1
            row["issue"] = "missing_phase_directory"
            details.append(row)
            continue
        missing = [name for name in REQUIRED_PHASE_FILES if not (directory / name).is_file()]
        if missing:
            report["missing_required_files"] += len(missing)
            row["issue"] = "missing_files"
            row["values"] = "|".join(missing)
            details.append(row)
            continue
        success = json.loads((directory / ".SUCCESS.json").read_text(encoding="utf-8"))
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if (
            success.get("feature_schema_sha256") != schema["schema_sha256"]
            or metadata.get("feature_schema_sha256") != schema["schema_sha256"]
            or success.get("manifest_sha256") != plan.manifest_sha256
            or metadata.get("manifest_sha256") != plan.manifest_sha256
            or success.get("frame_list_hash") != plan.frame_list_hash
            or metadata.get("frame_list_hash") != plan.frame_list_hash
            or metadata.get("labels_read") is not False
            or metadata.get("model_trained") is not False
            or metadata.get("manifest_rescanned") is not False
        ):
            report["wrong_provenance"] += 1
            row["issue"] = "provenance_mismatch"
            details.append(row)
        vector = np.load(directory / "embedding_5120.npy")
        report["wrong_embedding_shape"] += int(vector.shape != (5120,))
        report["nonfinite_embeddings"] += int(not np.isfinite(vector).all())
        report["all_zero_embeddings"] += int(np.allclose(vector, 0))
        if vector.shape == (5120,) and np.isfinite(vector).all():
            embeddings.append(vector.astype(np.float32))
        scalar = json.loads((directory / "scalar_features.json").read_text(encoding="utf-8"))
        report["wrong_scalar_schema"] += int(list(scalar) != scalar_names)
        probabilities = np.load(directory / "probabilities_original.fp16.npz")
        original_shape = tuple(metadata["original_shape"][-2:])
        for key in ("artery", "vein", "vessel", "vessel_union"):
            array = probabilities[key].astype(np.float32)
            report["wrong_probability_shape"] += int(array.shape != original_shape)
            report["wrong_probability_range"] += int(
                not np.isfinite(array).all() or array.min() < -1e-4 or array.max() > 1.0001
            )
        vessel = probabilities["vessel"].astype(np.float32)
        report["empty_vessel_probability_count"] += int(np.allclose(vessel, 0))
        report["low_vessel_probability_count"] += int(float(vessel.mean()) < 1e-4)
        report[f"success_{plan.phase}"] += 1

    if embeddings:
        stack = np.stack(embeddings)
        report["embedding_rows"] = int(len(stack))
        report["embedding_median_channel_variance"] = float(np.median(np.var(stack, axis=0)))
        report["embedding_zero_variance_channels"] = int((np.var(stack, axis=0) < 1e-12).sum())
        norms = np.linalg.norm(stack, axis=1)
        report["embedding_norm_min"] = float(norms.min())
        report["embedding_norm_median"] = float(np.median(norms))
        report["embedding_norm_max"] = float(norms.max())
    else:
        report["embedding_rows"] = 0
        report["embedding_median_channel_variance"] = 0.0
        report["embedding_zero_variance_channels"] = 5120

    hard_keys = (
        "missing_phase_directories", "missing_required_files", "wrong_embedding_shape",
        "nonfinite_embeddings", "all_zero_embeddings", "wrong_scalar_schema",
        "wrong_provenance", "wrong_probability_range", "wrong_probability_shape",
    )
    hard_fail = (
        report["success_pre"] != args.expected_pre
        or report["success_post"] != args.expected_post
        or any(int(report[key]) for key in hard_keys)
        or report["embedding_median_channel_variance"] <= 0
    )
    report["hard_fail"] = bool(hard_fail)
    write_json_atomic(args.output, report)
    write_csv_atomic(pd.DataFrame(details), args.output.with_suffix(".csv"))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 2 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

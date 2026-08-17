#!/usr/bin/env python3
"""Audit api_fullseq_v2 promotion equivalence and completed Full Train outputs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT = Path("/root/autodl-tmp/aneurysm")
CODE15 = PROJECT / "code/15_extract_api_fullseq_v2_pairdata.py"
CODE16 = PROJECT / "code/16_build_api_fullseq_v2_features.py"
CODE17 = PROJECT / "code/17_extract_api_fullseq_v2_full_pairdata.py"
CODE18 = PROJECT / "code/18_build_api_fullseq_v2_full_features.py"
PILOT_CONFIG = PROJECT / "configs/api_fullseq_v2_pairdata_config.json"
FULL_CONFIG = PROJECT / "configs/api_fullseq_v2_full_config.json"
SCHEMA = PROJECT / "configs/api_fullseq_v2_feature_schema.json"
PILOT_FREEZE = PROJECT / "manifests/api_fullseq_v2_gpu_pilot_frozen.csv"
FULL_MANIFEST = PROJECT / "manifests/api_fullseq_v2_full_frozen.csv"
VALID_MANIFEST = PROJECT / "manifests/api_fullseq_v2_valid_manifest.csv"
PILOT_PAIRDATA = PROJECT / "outputs/api_fullseq_v2_pairdata/pilot/train"
PROMOTION_PAIRDATA = PROJECT / "outputs/api_fullseq_v2_pairdata/promotion_equivalence"
PILOT_FEATURES = PROJECT / "outputs/api_fullseq_v2_features/pilot"
PROMOTION_FEATURES = PROJECT / "outputs/api_fullseq_v2_features/promotion_equivalence"
FULL_PAIRDATA = PROJECT / "outputs/api_fullseq_v2_pairdata/full/train"
FULL_FEATURES = PROJECT / "outputs/api_fullseq_v2_features/full"
REPORT_ROOT = PROJECT / "reports/api_fullseq_v2_feature_full"

SCIENTIFIC_FUNCTIONS = {
    "stage1": [
        "normalize_phase",
        "build_fov_mask",
        "baseline_polarity_enhancement",
        "build_activity_masks",
        "tdc_and_stage_features",
        "build_kinetic_maps",
        "build_filling_features",
        "uncertainty_log_from_info",
        "infer_flow",
        "warp_backward_flow",
        "direction_statistics",
        "region_flow_statistics",
        "legacy_pair_features",
        "pair_stage",
        "analyze_flow_pair",
        "normalized_cache_map",
    ],
    "stage2": [
        "flow_tdc_coupling",
        "aggregate_phase",
        "definition_for_feature",
        "build_schema",
        "build_patient_features",
    ],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    material = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def sha256_lines(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize_json(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize_json(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    write_json(temporary, value)
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def thresholds_hash(config: dict[str, Any]) -> str:
    return sha256_json({
        "normalization": config["normalization"],
        "fov": config["fov"],
        "activity": config["activity"],
        "kinetics": config["kinetics"],
        "flow_qc": config["flow_qc"],
        "aggregation": config["aggregation"],
    })


def sea_raft_code_tree_hash(repo_root: Path) -> str:
    paths = sorted(
        [
            *repo_root.joinpath("core").rglob("*.py"),
            *repo_root.joinpath("config").rglob("*.json"),
            *repo_root.joinpath("config").rglob("*.py"),
        ],
        key=lambda path: path.relative_to(repo_root).as_posix(),
    )
    return sha256_lines([
        f"{path.relative_to(repo_root).as_posix()}\t{sha256_file(path)}"
        for path in paths
    ])


def function_hashes(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            material = ast.dump(node, annotate_fields=True, include_attributes=False)
            result[node.name] = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return result


def run_hash_audit() -> dict[str, Any]:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    pairs = [("stage1", CODE15, CODE17), ("stage2", CODE16, CODE18)]
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for stage, pilot_path, full_path in pairs:
        pilot_hashes = function_hashes(pilot_path)
        full_hashes = function_hashes(full_path)
        for name in SCIENTIFIC_FUNCTIONS[stage]:
            pilot_hash = pilot_hashes.get(name, "")
            full_hash = full_hashes.get(name, "")
            if not pilot_hash or not full_hash:
                missing.append(f"{stage}:{name}")
            rows.append({
                "stage": stage,
                "function_name": name,
                "pilot_path": str(pilot_path),
                "full_path": str(full_path),
                "pilot_ast_sha256": pilot_hash,
                "full_ast_sha256": full_hash,
                "changed": pilot_hash != full_hash,
            })
    function_frame = pd.DataFrame(rows)
    function_frame.to_csv(
        REPORT_ROOT / "promotion_function_hashes.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    pilot_config = load_json(PILOT_CONFIG)
    full_config = load_json(FULL_CONFIG)
    repo_root = Path(pilot_config["model"]["repo_root"])
    checks = {
        "scientific_function_count_positive": len(rows) > 0,
        "scientific_functions_present": not missing,
        "changed_scientific_functions_zero": not function_frame["changed"].any(),
        "feature_schema_hash_unchanged": sha256_file(SCHEMA)
        == pilot_config["frozen_inputs"].get(
            "configs/api_fullseq_v2_feature_schema.json", sha256_file(SCHEMA)
        ),
        "scientific_thresholds_hash_unchanged": thresholds_hash(pilot_config)
        == thresholds_hash(full_config),
        "model_hash_unchanged": sha256_file(Path(full_config["model"]["model_file"]))
        == pilot_config["model"]["model_sha256"],
        "model_config_hash_unchanged": sha256_file(Path(full_config["model"]["config"]))
        == pilot_config["model"]["model_config_sha256"],
        "sea_raft_code_tree_hash_unchanged": sea_raft_code_tree_hash(repo_root)
        == pilot_config["model"]["sea_raft_code_tree_sha256"],
        "pilot_stage1_hash_unchanged": sha256_file(CODE15)
        == "3ced2bb4e2a98e605a03e6757b254625c8f77f88f5e340577c5eb23da3c6b275",
        "pilot_stage2_hash_unchanged": sha256_file(CODE16)
        == "af2f8da9af5a63d924fe5deb7f27c0c61cbdbf96bb2c4be89e551edd4da1d817",
        "pilot_schema_hash_unchanged": sha256_file(SCHEMA)
        == "6e011922eff8afdac064a9763f530c84b6a9603ddb0a399bd7875a38529d32c9",
        "pilot_freeze_hash_unchanged": sha256_file(PILOT_FREEZE)
        == "5bf4007126a14ea9480d64f64a75d90e96e771af5e8feb1574dd375ef61de849",
    }
    snapshot = {
        "created_utc": utc_now(),
        "scientific_function_count": len(rows),
        "changed_scientific_functions": int(function_frame["changed"].sum()),
        "missing_scientific_functions": missing,
        "checks": checks,
        "hashes": {
            "pilot_stage1": sha256_file(CODE15),
            "pilot_stage2": sha256_file(CODE16),
            "full_stage1": sha256_file(CODE17),
            "full_stage2": sha256_file(CODE18),
            "pilot_config": sha256_file(PILOT_CONFIG),
            "full_config": sha256_file(FULL_CONFIG),
            "feature_schema": sha256_file(SCHEMA),
            "scientific_thresholds": thresholds_hash(full_config),
            "model": sha256_file(Path(full_config["model"]["model_file"])),
            "model_config": sha256_file(Path(full_config["model"]["config"])),
            "sea_raft_code_tree": sea_raft_code_tree_hash(repo_root),
            "full_frozen_manifest": sha256_file(FULL_MANIFEST),
        },
    }
    lines = [
        "# api_fullseq_v2 Pilot → Full promotion diff",
        "",
        f"- Generated: {snapshot['created_utc']}",
        f"- Scientific functions audited: {len(rows)}",
        f"- Changed scientific functions: {snapshot['changed_scientific_functions']}",
        f"- Missing scientific functions: {missing or 'none'}",
        "- Allowed changes: script name, mode/Manifest selection, expected counts, paths, resume, progress, fixed QC, and Train/Valid execution gates.",
        "- Labels read: no",
        "- Model training: no",
        "",
        "## Hash gates",
        "",
        *[f"- {'PASS' if passed else 'FAIL'} — {name}" for name, passed in checks.items()],
        "",
    ]
    (REPORT_ROOT / "promotion_diff.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(REPORT_ROOT / "hash_freeze.json", snapshot)
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError("Promotion hash audit failed: " + "|".join(failed))
    return snapshot


def record_result(
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    scope: str,
    artifact: str,
    check: str,
    passed: bool,
    detail: str,
    max_abs_error: float | None = None,
    max_rel_error: float | None = None,
) -> None:
    row = {
        "scope": scope,
        "artifact": artifact,
        "check": check,
        "passed": bool(passed),
        "detail": detail,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
    }
    results.append(row)
    if not passed:
        failures.append(dict(row))


def compare_table(
    pilot_path: Path,
    full_path: Path,
    scope: str,
    artifact: str,
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    rtol: float,
    atol: float,
) -> None:
    pilot = pd.read_csv(pilot_path, dtype={"patient_id": str})
    full = pd.read_csv(full_path, dtype={"patient_id": str})
    columns_equal = list(pilot.columns) == list(full.columns)
    record_result(
        results, failures, scope, artifact, "column_order", columns_equal,
        f"pilot_columns={len(pilot.columns)} full_columns={len(full.columns)}",
    )
    rows_equal = len(pilot) == len(full)
    record_result(
        results, failures, scope, artifact, "row_count", rows_equal,
        f"pilot_rows={len(pilot)} full_rows={len(full)}",
    )
    if not columns_equal or not rows_equal:
        return
    for column in pilot.columns:
        left = pilot[column]
        right = full[column]
        left_nan = left.isna().to_numpy()
        right_nan = right.isna().to_numpy()
        nan_equal = np.array_equal(left_nan, right_nan)
        record_result(
            results, failures, scope, artifact, f"{column}:nan_positions", nan_equal,
            f"pilot_nan={int(left_nan.sum())} full_nan={int(right_nan.sum())}",
        )
        if not nan_equal:
            continue
        valid = ~left_nan
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            left_values = pd.to_numeric(left, errors="coerce").to_numpy(dtype=np.float64)
            right_values = pd.to_numeric(right, errors="coerce").to_numpy(dtype=np.float64)
            finite = valid & np.isfinite(left_values) & np.isfinite(right_values)
            if finite.any():
                difference = np.abs(left_values[finite] - right_values[finite])
                denominator = np.maximum(np.abs(left_values[finite]), atol)
                max_abs = float(difference.max())
                max_rel = float((difference / denominator).max())
            else:
                max_abs = 0.0
                max_rel = 0.0
            if pd.api.types.is_integer_dtype(left) and pd.api.types.is_integer_dtype(right):
                passed = np.array_equal(left_values[valid], right_values[valid])
            else:
                passed = bool(np.allclose(
                    left_values, right_values, rtol=rtol, atol=atol, equal_nan=True
                ))
            detail = f"valid={int(valid.sum())}"
            if not passed:
                bad = np.flatnonzero(
                    ~np.isclose(left_values, right_values, rtol=rtol, atol=atol, equal_nan=True)
                )
                if len(bad):
                    index = int(bad[0])
                    detail += f" first_row={index} pilot={left.iloc[index]!r} full={right.iloc[index]!r}"
            record_result(
                results, failures, scope, artifact, f"{column}:values", passed,
                detail, max_abs, max_rel,
            )
        else:
            left_values = left.astype("string").fillna("<NA>").to_numpy()
            right_values = right.astype("string").fillna("<NA>").to_numpy()
            passed = np.array_equal(left_values, right_values)
            detail = f"rows={len(left_values)}"
            if not passed:
                bad = np.flatnonzero(left_values != right_values)
                if len(bad):
                    index = int(bad[0])
                    detail += f" first_row={index} pilot={left_values[index]!r} full={right_values[index]!r}"
            record_result(results, failures, scope, artifact, f"{column}:values", passed, detail)


def compare_cache(
    pilot_path: Path,
    full_path: Path,
    scope: str,
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    rtol: float,
    atol: float,
) -> None:
    with np.load(pilot_path) as pilot, np.load(full_path) as full:
        keys_equal = pilot.files == full.files
        record_result(
            results, failures, scope, "phase_cache", "key_order", keys_equal,
            f"pilot={pilot.files} full={full.files}",
        )
        if not keys_equal:
            return
        for key in pilot.files:
            left = pilot[key]
            right = full[key]
            shape_equal = left.shape == right.shape
            dtype_equal = left.dtype == right.dtype
            record_result(results, failures, scope, "phase_cache", f"{key}:shape", shape_equal, f"pilot={left.shape} full={right.shape}")
            record_result(results, failures, scope, "phase_cache", f"{key}:dtype", dtype_equal, f"pilot={left.dtype} full={right.dtype}")
            if not shape_equal or not dtype_equal:
                continue
            if np.issubdtype(left.dtype, np.floating):
                passed = bool(np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=True))
                finite = np.isfinite(left) & np.isfinite(right)
                max_abs = float(np.max(np.abs(left[finite].astype(np.float64) - right[finite].astype(np.float64)))) if finite.any() else 0.0
                denominator = np.maximum(np.abs(left[finite].astype(np.float64)), atol)
                max_rel = float(np.max(np.abs(left[finite].astype(np.float64) - right[finite].astype(np.float64)) / denominator)) if finite.any() else 0.0
            else:
                passed = np.array_equal(left, right)
                max_abs = float(np.max(np.abs(left.astype(np.int64) - right.astype(np.int64)))) if left.size else 0.0
                max_rel = 0.0
            record_result(results, failures, scope, "phase_cache", f"{key}:values", passed, f"elements={left.size}", max_abs, max_rel)


def compare_nested(
    left: Any,
    right: Any,
    path: str,
    scope: str,
    artifact: str,
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    rtol: float,
    atol: float,
) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        keys_equal = list(left.keys()) == list(right.keys())
        record_result(results, failures, scope, artifact, f"{path}:keys", keys_equal, f"left={list(left)} right={list(right)}")
        if keys_equal:
            for key in left:
                compare_nested(left[key], right[key], f"{path}.{key}", scope, artifact, results, failures, rtol, atol)
        return
    if isinstance(left, list) and isinstance(right, list):
        length_equal = len(left) == len(right)
        record_result(results, failures, scope, artifact, f"{path}:length", length_equal, f"left={len(left)} right={len(right)}")
        if length_equal:
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                compare_nested(left_item, right_item, f"{path}[{index}]", scope, artifact, results, failures, rtol, atol)
        return
    if isinstance(left, (int, float, bool)) and isinstance(right, (int, float, bool)):
        passed = bool(np.isclose(float(left), float(right), rtol=rtol, atol=atol, equal_nan=True))
        max_abs = abs(float(left) - float(right)) if math.isfinite(float(left)) and math.isfinite(float(right)) else 0.0
        max_rel = max_abs / max(abs(float(left)), atol) if math.isfinite(float(left)) and math.isfinite(float(right)) else 0.0
        record_result(results, failures, scope, artifact, path, passed, f"pilot={left!r} full={right!r}", max_abs, max_rel)
        return
    passed = left == right
    record_result(results, failures, scope, artifact, path, passed, f"pilot={left!r} full={right!r}")


def compare_phase_json(
    pilot_dir: Path,
    full_dir: Path,
    scope: str,
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    rtol: float,
    atol: float,
) -> None:
    pilot_summary = load_json(pilot_dir / "phase_summary.json")
    full_summary = load_json(full_dir / "phase_summary.json")
    summary_keys = [
        "patient_id", "split", "source_type", "source_medical_record_root",
        "selected_series_id", "selected_series_path", "phase",
        "selected_internal_series", "n_frames", "manifest_expected_pairs",
        "processed_pairs", "complete_phase", "frame_list_hash", "polarity",
        "fov_qc", "activity_qc", "tdc_stages", "base_features", "qc_features",
        "numeric_audits", "cache_scaling", "table_format", "parquet_fallback",
        "labels_read", "model_trained", "manifest_rescanned",
    ]
    compare_nested(
        {key: pilot_summary[key] for key in summary_keys},
        {key: full_summary[key] for key in summary_keys},
        "summary", scope, "phase_summary", results, failures, rtol, atol,
    )
    pilot_metadata = load_json(pilot_dir / "metadata.json")
    full_metadata = load_json(full_dir / "metadata.json")
    metadata_keys = [
        "patient_id", "split", "phase", "frame_paths", "frame_indices",
        "frame_list_hash", "source_metadata_signature_before",
        "source_metadata_signature_after", "thresholds_sha256", "model_sha256",
        "model_config_sha256", "sea_raft_code_tree_sha256", "cache_scaling",
        "cache_dtype", "table_storage", "cuda_actually_used", "cpu_fallback",
    ]
    compare_nested(
        {key: pilot_metadata[key] for key in metadata_keys},
        {key: full_metadata[key] for key in metadata_keys},
        "metadata", scope, "metadata", results, failures, rtol, atol,
    )


def run_equivalence() -> None:
    hash_snapshot = run_hash_audit()
    config = load_json(FULL_CONFIG)
    rtol = float(config["promotion_equivalence"]["rtol"])
    atol = float(config["promotion_equivalence"]["atol"])
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    manifest = pd.read_csv(FULL_MANIFEST, dtype={"patient_id": str})
    gpu_ids = config["promotion_equivalence"]["gpu_patient_ids"]
    selected = manifest[manifest["patient_id"].isin(gpu_ids)].copy()
    order = {patient_id: index for index, patient_id in enumerate(gpu_ids)}
    selected["_order"] = selected["patient_id"].map(order)
    selected = selected.sort_values("_order")
    expected_phases: list[tuple[str, str, int]] = []
    for row in selected.itertuples(index=False):
        for phase in ("pre", "post"):
            if bool(getattr(row, f"can_run_{phase}")):
                expected_phases.append((row.patient_id, phase, int(getattr(row, f"n_{phase}_contiguous_pairs"))))
    record_result(results, failures, "gpu", "root", "expected_patient_ids", selected["patient_id"].tolist() == gpu_ids, f"actual={selected['patient_id'].tolist()}")
    record_result(results, failures, "gpu", "root", "expected_phase_count", len(expected_phases) == int(config["promotion_equivalence"]["expected_gpu_phases"]), f"actual={len(expected_phases)}")
    record_result(results, failures, "gpu", "root", "expected_pair_count", sum(item[2] for item in expected_phases) == int(config["promotion_equivalence"]["expected_gpu_pairs"]), f"actual={sum(item[2] for item in expected_phases)}")
    for patient_id, phase, expected_pairs in expected_phases:
        scope = f"gpu:{patient_id}:{phase}"
        pilot_dir = PILOT_PAIRDATA / patient_id / phase
        full_dir = PROMOTION_PAIRDATA / patient_id / phase
        for name in ["selected_frames.csv", "frame_kinetics.csv.gz", "pair_features.csv.gz", "temporal_curves.csv.gz"]:
            compare_table(pilot_dir / name, full_dir / name, scope, name, results, failures, rtol, atol)
        compare_cache(pilot_dir / "phase_cache.npz", full_dir / "phase_cache.npz", scope, results, failures, rtol, atol)
        compare_phase_json(pilot_dir, full_dir, scope, results, failures, rtol, atol)
        pair = pd.read_csv(full_dir / "pair_features.csv.gz", dtype={"patient_id": str})
        metadata = load_json(full_dir / "metadata.json")
        record_result(results, failures, scope, "pair_features", "expected_pairs", len(pair) == expected_pairs, f"expected={expected_pairs} actual={len(pair)}")
        record_result(results, failures, scope, "pair_features", "all_delta_frame_one", bool((pair["delta_frame"] == 1).all()), f"unique={sorted(pair['delta_frame'].unique().tolist())}")
        record_result(results, failures, scope, "metadata", "cuda_actually_used", metadata["cuda_actually_used"] is True, str(metadata["cuda_actually_used"]))
        record_result(results, failures, scope, "metadata", "no_cpu_fallback", metadata["cpu_fallback"] is False, str(metadata["cpu_fallback"]))
    record_result(results, failures, "gpu", "case_348817", "pre_post_present", (PROMOTION_PAIRDATA / "348817/pre/.SUCCESS").is_file() and (PROMOTION_PAIRDATA / "348817/post/.SUCCESS").is_file(), "Pre and Post required")
    record_result(results, failures, "gpu", "case_458123", "post_only", not (PROMOTION_PAIRDATA / "458123/pre").exists() and (PROMOTION_PAIRDATA / "458123/post/.SUCCESS").is_file(), "Pre absent; Post present")
    pair571 = pd.read_csv(PROMOTION_PAIRDATA / "571569/post/pair_features.csv.gz")
    record_result(results, failures, "gpu", "case_571569", "post_pairs_14", len(pair571) == 14, f"actual={len(pair571)}")

    compare_table(
        PILOT_FEATURES / "train_phase_features.csv",
        PROMOTION_FEATURES / "train_phase_features.csv",
        "cpu", "train_phase_features", results, failures, rtol, atol,
    )
    compare_table(
        PILOT_FEATURES / "train_patient_features.csv",
        PROMOTION_FEATURES / "train_patient_features.csv",
        "cpu", "train_patient_features", results, failures, rtol, atol,
    )
    phase = pd.read_csv(PROMOTION_FEATURES / "train_phase_features.csv", dtype={"patient_id": str})
    patient = pd.read_csv(PROMOTION_FEATURES / "train_patient_features.csv", dtype={"patient_id": str})
    schema = load_json(SCHEMA)
    record_result(results, failures, "cpu", "root", "patient_count_15", patient["patient_id"].nunique() == 15 and len(patient) == 15, f"rows={len(patient)} unique={patient['patient_id'].nunique()}")
    record_result(results, failures, "cpu", "root", "phase_count_27", len(phase) == 27, f"rows={len(phase)}")
    post_only = patient[patient["missing_pre"] == 1]
    pre_columns = [column for column in patient.columns if column.startswith("pre_")]
    delta_columns = [column for column in patient.columns if column.startswith("delta_")]
    record_result(results, failures, "cpu", "missingness", "post_only_pre_delta_nan", bool(post_only[pre_columns + delta_columns].isna().all().all()), f"post_only={post_only['patient_id'].tolist()}")
    safe_delta = {
        f"delta_{item['feature_name']}"
        for item in schema["phase_features"] if item["delta_policy"] == "safe"
    }
    record_result(results, failures, "cpu", "schema", "delta_columns_safe_only", set(delta_columns) == safe_delta, f"expected={len(safe_delta)} actual={len(delta_columns)}")
    delta_valid = True
    delta_detail = "all finite Pre/Post delta-safe values equal Post-Pre"
    for row in patient.itertuples(index=False):
        if row.missing_pre or row.missing_post:
            continue
        phase_rows = phase[phase["patient_id"] == row.patient_id].set_index("phase")
        for item in schema["phase_features"]:
            if item["delta_policy"] != "safe":
                continue
            name = item["feature_name"]
            expected = phase_rows.loc["post", name] - phase_rows.loc["pre", name]
            actual = getattr(row, f"delta_{name}")
            if not np.isclose(expected, actual, rtol=rtol, atol=atol, equal_nan=True):
                delta_valid = False
                delta_detail = f"patient={row.patient_id} feature={name} expected={expected} actual={actual}"
                break
        if not delta_valid:
            break
    record_result(results, failures, "cpu", "delta", "delta_values_post_minus_pre", delta_valid, delta_detail)
    record_result(results, failures, "hash", "promotion", "all_hash_gates", all(hash_snapshot["checks"].values()), f"changed={hash_snapshot['changed_scientific_functions']}")

    result_frame = pd.DataFrame(results)
    failure_frame = pd.DataFrame(failures, columns=result_frame.columns)
    result_frame.to_csv(REPORT_ROOT / "equivalence_results.csv", index=False, encoding="utf-8", lineterminator="\n")
    failure_frame.to_csv(REPORT_ROOT / "equivalence_failures.csv", index=False, encoding="utf-8", lineterminator="\n")
    report = [
        "# api_fullseq_v2 promotion equivalence report",
        "",
        f"- Generated: {utc_now()}",
        f"- Checks: {len(results)}",
        f"- Passed: {sum(item['passed'] for item in results)}",
        f"- Failed: {len(failures)}",
        f"- Tolerance: rtol={rtol}, atol={atol}, equal_nan=True",
        "- GPU cases: 348817, 458123, 571569",
        "- CPU aggregation: all 15 frozen Train Pilot patients",
        "- CUDA actually used: required",
        "- CPU fallback: forbidden",
        "- Labels read: no",
        "- Model training: no",
        "",
    ]
    if failures:
        report.extend(["## Failures", "", *[f"- {item['scope']} / {item['artifact']} / {item['check']}: {item['detail']}" for item in failures], ""])
    else:
        report.extend(["## Result", "", "All promotion hash, GPU pairdata, and CPU aggregation equivalence gates PASS.", ""])
    (REPORT_ROOT / "equivalence_report.md").write_text("\n".join(report), encoding="utf-8")
    if failures:
        raise AssertionError(f"Promotion equivalence failed with {len(failures)} checks")
    write_json_atomic(REPORT_ROOT / ".PROMOTION_EQUIVALENCE_SUCCESS", {
        "created_utc": utc_now(),
        "gpu_patients": gpu_ids,
        "gpu_phases": len(expected_phases),
        "gpu_pairs": sum(item[2] for item in expected_phases),
        "cpu_patients": int(patient["patient_id"].nunique()),
        "cpu_phases": len(phase),
        "cpu_pairs": int(phase["processed_pairs"].sum()),
        "checks": len(results),
        "rtol": rtol,
        "atol": atol,
    })


def feature_distribution(frame: pd.DataFrame, table_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        rows.append({
            "table": table_name,
            "feature": column,
            "rows": len(values),
            "missing_count": int(np.isnan(values).sum()),
            "inf_count": int(np.isinf(values).sum()),
            "finite_ratio": float(len(finite) / max(len(values), 1)),
            "min": float(np.min(finite)) if len(finite) else np.nan,
            "p1": float(np.percentile(finite, 1)) if len(finite) else np.nan,
            "p50": float(np.percentile(finite, 50)) if len(finite) else np.nan,
            "p99": float(np.percentile(finite, 99)) if len(finite) else np.nan,
            "max": float(np.max(finite)) if len(finite) else np.nan,
            "skewness": float(pd.Series(finite).skew()) if len(finite) >= 3 else np.nan,
            "zero_count": int((finite == 0).sum()),
            "negative_count": int((finite < 0).sum()),
        })
    return pd.DataFrame(rows)


def run_full_audit() -> None:
    if not (REPORT_ROOT / ".PROMOTION_EQUIVALENCE_SUCCESS").is_file():
        raise FileNotFoundError("Full Train is gated by .PROMOTION_EQUIVALENCE_SUCCESS")
    hash_snapshot = run_hash_audit()
    config = load_json(FULL_CONFIG)
    manifest = pd.read_csv(FULL_MANIFEST, dtype={"patient_id": str})
    valid = pd.read_csv(VALID_MANIFEST, dtype={"patient_id": str})
    schema = load_json(SCHEMA)
    phase_features = pd.read_csv(FULL_FEATURES / "train_phase_features.csv", dtype={"patient_id": str})
    patient_features = pd.read_csv(FULL_FEATURES / "train_patient_features.csv", dtype={"patient_id": str})
    root_summary = load_json(FULL_PAIRDATA / "run_summary.json")
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    expected_patients = int(config["full"]["expected_train_patients"])
    expected_phases = int(config["full"]["expected_train_phases"])
    expected_pairs = int(config["full"]["expected_train_pairs"])
    check("patients_1055", len(manifest) == expected_patients and manifest["patient_id"].nunique() == expected_patients, f"rows={len(manifest)} unique={manifest['patient_id'].nunique()}")
    check("phases_1921_manifest", int(manifest["can_run_pre"].sum() + manifest["can_run_post"].sum()) == expected_phases, f"actual={int(manifest['can_run_pre'].sum() + manifest['can_run_post'].sum())}")
    check("pairs_39906_manifest", int(manifest["n_pre_contiguous_pairs"].sum() + manifest["n_post_contiguous_pairs"].sum()) == expected_pairs, f"actual={int(manifest['n_pre_contiguous_pairs'].sum() + manifest['n_post_contiguous_pairs'].sum())}")
    check("prepost_866", int(manifest["can_run_prepost"].sum()) == int(config["full"]["expected_train_prepost_patients"]), f"actual={int(manifest['can_run_prepost'].sum())}")
    post_only_manifest = (~manifest["can_run_pre"] & manifest["can_run_post"])
    check("post_only_189", int(post_only_manifest.sum()) == int(config["full"]["expected_train_post_only_patients"]), f"actual={int(post_only_manifest.sum())}")
    check("train_valid_overlap_zero", not (set(manifest["patient_id"]) & set(valid["patient_id"])), f"overlap={len(set(manifest['patient_id']) & set(valid['patient_id']))}")
    check("root_success_present", (FULL_PAIRDATA / ".SUCCESS").is_file(), str(FULL_PAIRDATA / ".SUCCESS"))
    check("root_summary_counts", root_summary["patients"] == expected_patients and len(root_summary["phase_summaries"]) == expected_phases and root_summary["processed_pairs"] == expected_pairs, f"patients={root_summary['patients']} phases={len(root_summary['phase_summaries'])} pairs={root_summary['processed_pairs']}")
    check("root_cuda_only", root_summary["cuda_actually_used"] is True and root_summary["cpu_fallback"] is False, f"cuda={root_summary['cuda_actually_used']} cpu_fallback={root_summary['cpu_fallback']}")
    check("root_scope_train_only", root_summary["full_train_started"] is True and root_summary["full_valid_started"] is False, f"full_train={root_summary['full_train_started']} full_valid={root_summary['full_valid_started']}")
    check("root_labels_training_false", root_summary["labels_read"] is False and root_summary["model_trained"] is False, f"labels={root_summary['labels_read']} trained={root_summary['model_trained']}")

    positions = config["full"]["qc_manifest_positions_one_based"]
    patient_order = manifest["patient_id"].tolist()
    qc_ids = {patient_order[int(position) - 1] for position in positions}
    phase_rows: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    total_pairs = 0
    selected_png = 0
    selected_parameter = 0
    cross_patient = 0
    cross_series = 0
    cross_phase = 0
    cross_internal = 0
    cross_gap = 0
    source_changed = 0
    labels_read = 0
    model_trained = 0
    cuda_fail = 0
    qc_fail = 0
    for manifest_row in manifest.to_dict("records"):
        patient_id = manifest_row["patient_id"]
        for phase in ("pre", "post"):
            if not bool(manifest_row[f"can_run_{phase}"]):
                continue
            expected = int(manifest_row[f"n_{phase}_contiguous_pairs"])
            phase_dir = FULL_PAIRDATA / patient_id / phase
            required = [".SUCCESS", "selected_frames.csv", "pair_features.csv.gz", "phase_summary.json", "metadata.json"]
            missing = [name for name in required if not (phase_dir / name).is_file()]
            if missing:
                unresolved.append({"patient_id": patient_id, "phase": phase, "failure": f"missing:{'|'.join(missing)}"})
                phase_rows.append({"patient_id": patient_id, "phase": phase, "expected_pairs": expected, "actual_pairs": np.nan, "success": False, "failure": f"missing:{'|'.join(missing)}"})
                continue
            selected = pd.read_csv(phase_dir / "selected_frames.csv", dtype={"patient_id": str})
            pair = pd.read_csv(phase_dir / "pair_features.csv.gz", dtype={"patient_id": str})
            summary = load_json(phase_dir / "phase_summary.json")
            metadata = load_json(phase_dir / "metadata.json")
            actual = len(pair)
            total_pairs += actual
            selected_png += int(selected["absolute_path"].str.lower().str.endswith(".png").sum())
            selected_parameter += int(selected["absolute_path"].str.upper().str.contains("CBF|CBV|MTT|TTP", regex=True).sum())
            cross_patient += int(pair["patient_id"].nunique() != 1 or pair["patient_id"].iloc[0] != patient_id)
            cross_series += int(pair["selected_series_id"].nunique() != 1 or str(pair["selected_series_id"].iloc[0]) != str(manifest_row["selected_series_id"]))
            cross_phase += int(pair["phase"].nunique() != 1 or pair["phase"].iloc[0] != phase)
            cross_internal += int(pair["selected_internal_series"].nunique() != 1)
            cross_gap += int((pair["delta_frame"] != 1).sum())
            source_changed += int(metadata["source_metadata_signature_before"] != metadata["source_metadata_signature_after"])
            labels_read += int(bool(summary["labels_read"]))
            model_trained += int(bool(summary["model_trained"]))
            cuda_fail += int(not metadata["cuda_actually_used"] or metadata["cpu_fallback"])
            visual_count = len(list((phase_dir / "visualizations").glob("*_residual_flow.jpg"))) if (phase_dir / "visualizations").is_dir() else 0
            expected_visual = 1 if patient_id in qc_ids else 0
            qc_fail += int(visual_count != expected_visual)
            failure = "" if actual == expected else f"pair_mismatch:{expected}!={actual}"
            if failure:
                unresolved.append({"patient_id": patient_id, "phase": phase, "failure": failure})
            phase_rows.append({
                "patient_id": patient_id,
                "split": manifest_row["split"],
                "source_type": manifest_row["source_type"],
                "selected_series_id": manifest_row["selected_series_id"],
                "phase": phase,
                "n_frames": len(selected),
                "expected_pairs": expected,
                "actual_pairs": actual,
                "delta_frame_one": bool((pair["delta_frame"] == 1).all()),
                "cuda_actually_used": metadata["cuda_actually_used"],
                "cpu_fallback": metadata["cpu_fallback"],
                "source_unchanged": metadata["source_metadata_signature_before"] == metadata["source_metadata_signature_after"],
                "visual_pairs": visual_count,
                "fixed_qc_patient": patient_id in qc_ids,
                "success": not failure,
                "failure": failure,
            })
    phase_audit = pd.DataFrame(phase_rows)
    phase_audit.to_csv(REPORT_ROOT / "train_phase_audit.csv", index=False, encoding="utf-8", lineterminator="\n")
    check("phase_success_1921", len(phase_audit) == expected_phases and bool(phase_audit["success"].all()), f"rows={len(phase_audit)} failed={int((~phase_audit['success']).sum())}")
    check("pairs_actual_39906", total_pairs == expected_pairs, f"actual={total_pairs}")
    check("expected_equals_actual_every_phase", bool((phase_audit["expected_pairs"] == phase_audit["actual_pairs"]).all()), f"phases={len(phase_audit)}")
    check("all_delta_frame_one", cross_gap == 0 and bool(phase_audit["delta_frame_one"].all()), f"cross_gap={cross_gap}")
    check("selected_png_zero", selected_png == 0, f"count={selected_png}")
    check("selected_parameter_maps_zero", selected_parameter == 0, f"count={selected_parameter}")
    check("cross_patient_zero", cross_patient == 0, f"count={cross_patient}")
    check("cross_series_zero", cross_series == 0, f"count={cross_series}")
    check("cross_phase_zero", cross_phase == 0, f"count={cross_phase}")
    check("cross_internal_series_zero", cross_internal == 0, f"count={cross_internal}")
    check("source_files_unchanged", source_changed == 0, f"count={source_changed}")
    check("labels_read_no", labels_read == 0, f"count={labels_read}")
    check("model_training_no", model_trained == 0, f"count={model_trained}")
    check("cuda_all_phases", cuda_fail == 0, f"count={cuda_fail}")
    check("fixed_qc_exact", qc_fail == 0, f"mismatched_phases={qc_fail} qc_patients={sorted(qc_ids)}")
    check("unresolved_failed_phases_zero", len(unresolved) == 0, f"count={len(unresolved)}")

    expected_phase_keys: list[tuple[str, str]] = []
    for row in manifest.to_dict("records"):
        for phase in ("pre", "post"):
            if bool(row[f"can_run_{phase}"]):
                expected_phase_keys.append((row["patient_id"], phase))
    actual_phase_keys = list(zip(phase_features["patient_id"], phase_features["phase"]))
    check("phase_feature_rows_1921", len(phase_features) == expected_phases, f"actual={len(phase_features)}")
    check("phase_feature_order", actual_phase_keys == expected_phase_keys, f"keys={len(actual_phase_keys)}")
    check("phase_feature_pairs_39906", int(phase_features["processed_pairs"].sum()) == expected_pairs, f"actual={int(phase_features['processed_pairs'].sum())}")
    check("patient_feature_rows_1055", len(patient_features) == expected_patients and patient_features["patient_id"].nunique() == expected_patients, f"rows={len(patient_features)} unique={patient_features['patient_id'].nunique()}")
    check("patient_feature_order", patient_features["patient_id"].tolist() == patient_order, f"rows={len(patient_features)}")
    phase_numeric = phase_features.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    patient_numeric = patient_features.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    check("inf_zero", not np.isinf(phase_numeric).any() and not np.isinf(patient_numeric).any(), f"phase_inf={int(np.isinf(phase_numeric).sum())} patient_inf={int(np.isinf(patient_numeric).sum())}")
    post_only = patient_features[patient_features["missing_pre"] == 1]
    pre_columns = [column for column in patient_features.columns if column.startswith("pre_")]
    delta_columns = [column for column in patient_features.columns if column.startswith("delta_")]
    check("post_only_feature_count_189", len(post_only) == int(config["full"]["expected_train_post_only_patients"]), f"actual={len(post_only)}")
    check("post_only_pre_delta_nan", bool(post_only[pre_columns + delta_columns].isna().all().all()), f"patients={len(post_only)}")
    model_patient = [item["feature_name"] for item in schema["patient_features"] if item["model_candidate"]]
    model_values = patient_features[model_patient].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    check("model_candidate_nonmissing_finite", not np.isinf(model_values).any(), f"finite={int(np.isfinite(model_values).sum())} inf={int(np.isinf(model_values).sum())}")
    check("schema_hash_unchanged", sha256_file(SCHEMA) == "6e011922eff8afdac064a9763f530c84b6a9603ddb0a399bd7875a38529d32c9", sha256_file(SCHEMA))
    check("scientific_threshold_hash_unchanged", thresholds_hash(config) == hash_snapshot["hashes"]["scientific_thresholds"], thresholds_hash(config))
    check("model_hash_unchanged", sha256_file(Path(config["model"]["model_file"])) == hash_snapshot["hashes"]["model"], sha256_file(Path(config["model"]["model_file"])))
    check("scientific_function_hashes_unchanged", hash_snapshot["changed_scientific_functions"] == 0 and all(hash_snapshot["checks"].values()), f"changed={hash_snapshot['changed_scientific_functions']}")

    patient_audit_rows: list[dict[str, Any]] = []
    phase_by_patient = phase_audit.groupby("patient_id", sort=False)
    for row in manifest.to_dict("records"):
        group = phase_by_patient.get_group(row["patient_id"])
        feature_row = patient_features[patient_features["patient_id"] == row["patient_id"]].iloc[0]
        patient_audit_rows.append({
            "patient_id": row["patient_id"],
            "split": row["split"],
            "source_type": row["source_type"],
            "expected_phases": int(row["can_run_pre"]) + int(row["can_run_post"]),
            "actual_phases": len(group),
            "expected_pairs": int(row["n_pre_contiguous_pairs"] + row["n_post_contiguous_pairs"]),
            "actual_pairs": int(group["actual_pairs"].sum()),
            "can_run_pre": bool(row["can_run_pre"]),
            "can_run_post": bool(row["can_run_post"]),
            "missing_pre": int(feature_row["missing_pre"]),
            "missing_post": int(feature_row["missing_post"]),
            "all_phases_success": bool(group["success"].all()),
        })
    patient_audit = pd.DataFrame(patient_audit_rows)
    patient_audit.to_csv(REPORT_ROOT / "train_patient_audit.csv", index=False, encoding="utf-8", lineterminator="\n")
    distribution = pd.concat([
        feature_distribution(phase_features, "phase"),
        feature_distribution(patient_features, "patient"),
    ], ignore_index=True)
    distribution.to_csv(REPORT_ROOT / "train_feature_distribution.csv", index=False, encoding="utf-8", lineterminator="\n")
    failure_rows = [item for item in checks if not item["passed"]] + unresolved
    pd.DataFrame(failure_rows, columns=["name", "passed", "detail"] if not unresolved else None).to_csv(
        REPORT_ROOT / "train_failures.csv", index=False, encoding="utf-8", lineterminator="\n"
    )
    failed = [item for item in checks if not item["passed"]]
    source_distribution = patient_features["source_type"].value_counts(dropna=False).to_dict()
    phase_distribution = manifest.assign(
        phase_pattern=np.where(manifest["can_run_prepost"], "Pre+Post", "Post-only")
    )["phase_pattern"].value_counts().to_dict()
    audit_lines = [
        "# api_fullseq_v2 Full Train independent audit",
        "",
        f"- Generated: {utc_now()}",
        f"- Patients: {len(patient_features)}",
        f"- Phases: {len(phase_features)}",
        f"- Pairs: {total_pairs}",
        f"- Unresolved failed phases: {len(unresolved)}",
        f"- Source type distribution: {json.dumps(source_distribution, ensure_ascii=False, sort_keys=True)}",
        f"- Pre/Post distribution: {json.dumps(phase_distribution, ensure_ascii=False, sort_keys=True)}",
        f"- Fixed QC patient IDs: {', '.join(sorted(qc_ids))}",
        "- Labels read: no",
        "- Model training: no",
        "- Full Valid started: no",
        "",
        "## Hard assertions",
        "",
        *[f"- {'PASS' if item['passed'] else 'FAIL'} — {item['name']}: {item['detail']}" for item in checks],
        "",
        "## Distribution audit",
        "",
        f"- Numeric feature rows audited: {len(distribution)}",
        f"- Features containing Inf: {int((distribution['inf_count'] > 0).sum())}",
        f"- Active ratio mean: {pd.to_numeric(phase_features['qc_active_ratio_fov'], errors='coerce').mean():.6g}",
        f"- FB error mean: {pd.to_numeric(phase_features['qc_pair_fb_relative_mean_mean'], errors='coerce').mean():.6g}",
        f"- Soft weight mean: {pd.to_numeric(phase_features['qc_pair_soft_weight_mean_fov_mean'], errors='coerce').mean():.6g}",
        f"- Global motion mean: {pd.to_numeric(phase_features['qc_pair_global_motion_mag_norm_mean'], errors='coerce').mean():.6g}",
        f"- Residual motion mean: {pd.to_numeric(phase_features['flow_active_res_mag_norm_median_mean'], errors='coerce').mean():.6g}",
        "- Distribution values are recorded for audit only; extraction thresholds and science were not changed.",
        "",
    ]
    (REPORT_ROOT / "train_audit.md").write_text("\n".join(audit_lines), encoding="utf-8")
    hash_snapshot["full_train_audit"] = {
        "completed_utc": utc_now(),
        "patients": len(patient_features),
        "phases": len(phase_features),
        "pairs": total_pairs,
        "failed_checks": [item["name"] for item in failed],
        "unresolved_failed_phases": len(unresolved),
        "full_pairdata_root_summary_sha256": sha256_file(FULL_PAIRDATA / "run_summary.json"),
        "train_phase_features_sha256": sha256_file(FULL_FEATURES / "train_phase_features.csv"),
        "train_patient_features_sha256": sha256_file(FULL_FEATURES / "train_patient_features.csv"),
    }
    write_json(REPORT_ROOT / "hash_freeze.json", hash_snapshot)
    if failed or unresolved:
        raise AssertionError(f"Full Train audit failed: checks={len(failed)} unresolved={len(unresolved)}")
    write_json_atomic(REPORT_ROOT / ".FULL_TRAIN_SUCCESS", {
        "created_utc": utc_now(),
        "patients": len(patient_features),
        "phases": len(phase_features),
        "pairs": total_pairs,
        "unresolved_failed_phases": 0,
        "hard_checks": len(checks),
        "labels_read": False,
        "model_trained": False,
        "full_valid_started": False,
    })


def write_failure(stage: str, exc: BaseException) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    path = REPORT_ROOT / "failure.md"
    if path.exists():
        return
    path.write_text(
        "\n".join([
            "# api_fullseq_v2 Full Train failure",
            "",
            f"- Generated: {utc_now()}",
            f"- Stage: {stage}",
            f"- Exception: {type(exc).__name__}: {exc}",
            "- Existing successful phase markers were preserved.",
            "- Full Train may be resumed with --resume.",
            "- Full Valid started: no.",
            "- Labels read: no.",
            "- Model training: no.",
            "",
            "## Traceback",
            "",
            traceback.format_exc(),
            "",
        ]),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["hash-audit", "equivalence", "full-audit"])
    args = parser.parse_args()
    stage = args.mode
    try:
        if args.mode == "hash-audit":
            snapshot = run_hash_audit()
            print(json.dumps({
                "scientific_function_count": snapshot["scientific_function_count"],
                "changed_scientific_functions": snapshot["changed_scientific_functions"],
                "checks": snapshot["checks"],
            }, ensure_ascii=False, indent=2))
        elif args.mode == "equivalence":
            run_equivalence()
            print(str(REPORT_ROOT / ".PROMOTION_EQUIVALENCE_SUCCESS"))
        else:
            run_full_audit()
            print(str(REPORT_ROOT / ".FULL_TRAIN_SUCCESS"))
        return 0
    except BaseException as exc:
        write_failure(stage, exc)
        raise


if __name__ == "__main__":
    sys.exit(main())

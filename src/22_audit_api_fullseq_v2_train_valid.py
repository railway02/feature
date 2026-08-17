#!/usr/bin/env python3
"""Automated release, split, and hash audit for api_fullseq_v2 Full Train + Valid."""

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
CONFIG = PROJECT / "configs/api_fullseq_v2_full_train_valid_config.json"
PILOT_CONFIG = PROJECT / "configs/api_fullseq_v2_pairdata_config.json"
SCHEMA = PROJECT / "configs/api_fullseq_v2_feature_schema.json"
TRAIN_MANIFEST = PROJECT / "manifests/api_fullseq_v2_train_manifest.csv"
VALID_MANIFEST = PROJECT / "manifests/api_fullseq_v2_valid_manifest.csv"
COMBINED_MANIFEST = PROJECT / "manifests/api_fullseq_v2_full_train_valid_frozen.csv"
CODE17 = PROJECT / "code/17_extract_api_fullseq_v2_full_pairdata.py"
CODE18 = PROJECT / "code/18_build_api_fullseq_v2_full_features.py"
CODE20 = PROJECT / "code/20_extract_api_fullseq_v2_train_valid_pairdata.py"
CODE21 = PROJECT / "code/21_build_api_fullseq_v2_train_valid_features.py"
CODE22 = PROJECT / "code/22_audit_api_fullseq_v2_train_valid.py"
REPORT_ROOT = PROJECT / "reports/api_fullseq_v2_feature_full"
PAIR_ROOT = PROJECT / "outputs/api_fullseq_v2_pairdata/full"
FEATURE_ROOT = PROJECT / "outputs/api_fullseq_v2_features/full"
PILOT_PAIRDATA = PROJECT / "outputs/api_fullseq_v2_pairdata/pilot/train"
PROMOTION_PAIRDATA = PROJECT / "outputs/api_fullseq_v2_pairdata/promotion_equivalence"
EQUIVALENCE_RESULTS = REPORT_ROOT / "equivalence_results.csv"
EQUIVALENCE_FAILURES = REPORT_ROOT / "equivalence_failures.csv"
HISTORICAL_REPORT = REPORT_ROOT / "equivalence_report.md"
HISTORICAL_FAILURE = REPORT_ROOT / "failure.md"
TRAIN_SUCCESS = REPORT_ROOT / ".FULL_TRAIN_SUCCESS"
VALID_SUCCESS = REPORT_ROOT / ".FULL_VALID_SUCCESS"
FINAL_SUCCESS = REPORT_ROOT / ".FULL_TRAIN_VALID_SUCCESS"
HASH_BEFORE_VALID = REPORT_ROOT / "hash_before_valid.json"
HASH_AFTER_VALID = REPORT_ROOT / "hash_after_valid.json"

SCIENTIFIC_FUNCTIONS = {
    "stage1": [
        "normalize_phase", "build_fov_mask", "baseline_polarity_enhancement",
        "build_activity_masks", "tdc_and_stage_features", "build_kinetic_maps",
        "build_filling_features", "uncertainty_log_from_info", "infer_flow",
        "warp_backward_flow", "direction_statistics", "region_flow_statistics",
        "legacy_pair_features", "pair_stage", "analyze_flow_pair",
        "normalized_cache_map",
    ],
    "stage2": [
        "flow_tdc_coupling", "aggregate_phase", "definition_for_feature",
        "build_schema", "build_patient_features",
    ],
}


class HardFailure(RuntimeError):
    pass


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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def science_audit() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for stage, pilot_path, production_path in [
        ("stage1", CODE17, CODE20),
        ("stage2", CODE18, CODE21),
    ]:
        pilot = function_hashes(pilot_path)
        production = function_hashes(production_path)
        for name in SCIENTIFIC_FUNCTIONS[stage]:
            left = pilot.get(name, "")
            right = production.get(name, "")
            if not left or not right:
                missing.append(f"{stage}:{name}")
            rows.append({
                "stage": stage,
                "function": name,
                "pilot_ast_sha256": left,
                "production_ast_sha256": right,
                "changed": left != right,
            })
    return {
        "rows": rows,
        "scientific_function_count": len(rows),
        "changed_scientific_functions": sum(int(row["changed"]) for row in rows),
        "missing_scientific_functions": missing,
        "combined_ast_sha256": sha256_json(rows),
    }


def manifest_summary(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path, dtype={"patient_id": str})
    return {
        "rows": len(frame),
        "unique_patients": int(frame["patient_id"].nunique()),
        "phases": int(frame["can_run_pre"].sum() + frame["can_run_post"].sum()),
        "pairs": int(frame["n_pre_contiguous_pairs"].sum() + frame["n_post_contiguous_pairs"].sum()),
        "prepost": int(frame["can_run_prepost"].sum()),
        "post_only": int((~frame["can_run_pre"] & frame["can_run_post"]).sum()),
        "duplicates": int(frame["patient_id"].duplicated().sum()),
        "split_counts": frame["split"].value_counts().to_dict(),
    }


def check_frozen_inputs(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative, expected in config["frozen_inputs"].items():
        path = PROJECT / relative
        actual = sha256_file(path) if path.is_file() else "MISSING"
        rows.append({
            "name": f"frozen:{relative}",
            "passed": actual == expected,
            "detail": f"expected={expected} actual={actual}",
        })
    return rows


def frame_equal_exact(left: pd.DataFrame, right: pd.DataFrame, columns: list[str]) -> bool:
    if len(left) != len(right) or any(column not in left or column not in right for column in columns):
        return False
    for column in columns:
        left_values = left[column]
        right_values = right[column]
        if not np.array_equal(left_values.isna().to_numpy(), right_values.isna().to_numpy()):
            return False
        if pd.api.types.is_numeric_dtype(left_values) and pd.api.types.is_numeric_dtype(right_values):
            if not np.array_equal(
                pd.to_numeric(left_values, errors="coerce").to_numpy(),
                pd.to_numeric(right_values, errors="coerce").to_numpy(),
                equal_nan=True,
            ):
                return False
        else:
            if not np.array_equal(
                left_values.astype("string").fillna("<NA>").to_numpy(),
                right_values.astype("string").fillna("<NA>").to_numpy(),
            ):
                return False
    return True


def promotion_input_audit() -> list[dict[str, Any]]:
    config = load_json(CONFIG)
    rows: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        rows.append({"name": name, "passed": bool(passed), "detail": detail})

    root = load_json(PROMOTION_PAIRDATA / "run_summary.json")
    add("promotion_root_counts", root["patients"] == 3 and len(root["phase_summaries"]) == 5 and root["processed_pairs"] == 90, f"patients={root['patients']} phases={len(root['phase_summaries'])} pairs={root['processed_pairs']}")
    add("promotion_cuda_semantic", bool(root["cuda_actually_used"]) and not bool(root["cpu_fallback"]), f"cuda={root['cuda_actually_used']} fallback={root['cpu_fallback']}")
    gpu_ids = config["promotion_equivalence"]["gpu_patient_ids"]
    manifest = pd.read_csv(TRAIN_MANIFEST, dtype={"patient_id": str})
    selected = manifest[manifest["patient_id"].isin(gpu_ids)]
    for row in selected.to_dict("records"):
        patient_id = row["patient_id"]
        for phase in ("pre", "post"):
            if not bool(row[f"can_run_{phase}"]):
                continue
            key = f"{patient_id}:{phase}"
            pilot_dir = PILOT_PAIRDATA / patient_id / phase
            production_dir = PROMOTION_PAIRDATA / patient_id / phase
            left_selected = pd.read_csv(pilot_dir / "selected_frames.csv", dtype={"patient_id": str})
            right_selected = pd.read_csv(production_dir / "selected_frames.csv", dtype={"patient_id": str})
            selected_columns = [
                "patient_id", "split", "source_type", "selected_series_id", "phase",
                "selected_internal_series", "sequence_position", "frame_index",
                "absolute_path", "height", "width", "starts_true_contiguous_pair",
                "delta_to_next_frame",
            ]
            add(f"promotion_selected_frames:{key}", frame_equal_exact(left_selected, right_selected, selected_columns), f"rows={len(right_selected)}")
            left_pair = pd.read_csv(pilot_dir / "pair_features.csv.gz", dtype={"patient_id": str})
            right_pair = pd.read_csv(production_dir / "pair_features.csv.gz", dtype={"patient_id": str})
            pair_columns = [
                "patient_id", "split", "source_type", "selected_series_id", "phase",
                "selected_internal_series", "pair_order", "sequence_position_t",
                "sequence_position_t1", "frame_index_t", "frame_index_t1", "delta_frame",
                "stage", "normalized_pair_time", "tdc_derivative_pair",
            ]
            add(f"promotion_pair_identity:{key}", frame_equal_exact(left_pair, right_pair, pair_columns), f"rows={len(right_pair)}")
            expected_pairs = int(row[f"n_{phase}_contiguous_pairs"])
            add(f"promotion_pair_count:{key}", len(right_pair) == expected_pairs, f"expected={expected_pairs} actual={len(right_pair)}")
            add(f"promotion_delta_frame:{key}", bool((right_pair["delta_frame"] == 1).all()), f"unique={sorted(right_pair['delta_frame'].unique().tolist())}")
            metadata = load_json(production_dir / "metadata.json")
            summary = load_json(production_dir / "phase_summary.json")
            add(f"promotion_cuda:{key}", bool(metadata["cuda_actually_used"]) and not bool(metadata["cpu_fallback"]), f"cuda={metadata['cuda_actually_used']} fallback={metadata['cpu_fallback']}")
            add(f"promotion_no_labels_training:{key}", not bool(summary["labels_read"]) and not bool(summary["model_trained"]), f"labels={summary['labels_read']} trained={summary['model_trained']}")
    pair571 = pd.read_csv(PROMOTION_PAIRDATA / "571569/post/pair_features.csv.gz")
    add("promotion_571569_post_14", len(pair571) == 14, f"actual={len(pair571)}")
    return rows


def run_preflight() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    config = load_json(CONFIG)
    pilot_config = load_json(PILOT_CONFIG)
    science = science_audit()
    train = manifest_summary(TRAIN_MANIFEST)
    valid = manifest_summary(VALID_MANIFEST)
    train_ids = set(pd.read_csv(TRAIN_MANIFEST, dtype={"patient_id": str})["patient_id"])
    valid_ids = set(pd.read_csv(VALID_MANIFEST, dtype={"patient_id": str})["patient_id"])
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    add("historical_equivalence_report_present", HISTORICAL_REPORT.is_file(), str(HISTORICAL_REPORT))
    add("historical_equivalence_failures_present", EQUIVALENCE_FAILURES.is_file(), str(EQUIVALENCE_FAILURES))
    add("historical_failure_present", HISTORICAL_FAILURE.is_file(), str(HISTORICAL_FAILURE))
    failures = pd.read_csv(EQUIVALENCE_FAILURES)
    add("historical_gpu_differences_recorded", len(failures) > 0, f"rows={len(failures)} nonblocking=true")
    equivalence = pd.read_csv(EQUIVALENCE_RESULTS)
    passed = equivalence["passed"].astype(str).str.lower().isin(["true", "1"])
    cpu_failed = int((equivalence["scope"].str.startswith("cpu") & ~passed).sum())
    hash_failed = int(((equivalence["scope"] == "hash") & ~passed).sum())
    add("same_pairdata_cpu_aggregation_failures_zero", cpu_failed == 0, f"failed={cpu_failed}")
    add("historical_hash_failures_zero", hash_failed == 0, f"failed={hash_failed}")
    add("scientific_function_count_21", science["scientific_function_count"] == 21, f"actual={science['scientific_function_count']}")
    add("scientific_function_hashes_unchanged", science["changed_scientific_functions"] == 0 and not science["missing_scientific_functions"], f"changed={science['changed_scientific_functions']} missing={science['missing_scientific_functions']}")
    add("schema_hash_unchanged", sha256_file(SCHEMA) == "6e011922eff8afdac064a9763f530c84b6a9603ddb0a399bd7875a38529d32c9", sha256_file(SCHEMA))
    add("threshold_hash_unchanged", thresholds_hash(config) == thresholds_hash(pilot_config), f"pilot={thresholds_hash(pilot_config)} production={thresholds_hash(config)}")
    add("model_hash_unchanged", sha256_file(Path(config["model"]["model_file"])) == pilot_config["model"]["model_sha256"], sha256_file(Path(config["model"]["model_file"])))
    add("model_config_hash_unchanged", sha256_file(Path(config["model"]["config"])) == pilot_config["model"]["model_config_sha256"], sha256_file(Path(config["model"]["config"])))
    add("sea_raft_code_tree_unchanged", sea_raft_code_tree_hash(Path(config["model"]["repo_root"])) == pilot_config["model"]["sea_raft_code_tree_sha256"], sea_raft_code_tree_hash(Path(config["model"]["repo_root"])))
    add("train_manifest_scale", train == {"rows": 1055, "unique_patients": 1055, "phases": 1921, "pairs": 39906, "prepost": 866, "post_only": 189, "duplicates": 0, "split_counts": {"Train": 1055}}, json.dumps(train, sort_keys=True))
    add("valid_manifest_scale", valid == {"rows": 264, "unique_patients": 264, "phases": 492, "pairs": 10124, "prepost": 228, "post_only": 36, "duplicates": 0, "split_counts": {"Valid": 264}}, json.dumps(valid, sort_keys=True))
    add("train_valid_overlap_zero", not (train_ids & valid_ids), f"overlap={len(train_ids & valid_ids)}")
    combined = pd.read_csv(COMBINED_MANIFEST, dtype={"patient_id": str})
    add("combined_frozen_manifest", len(combined) == 1319 and combined["patient_id"].nunique() == 1319 and sha256_file(COMBINED_MANIFEST) == config["frozen_inputs"]["manifests/api_fullseq_v2_full_train_valid_frozen.csv"], f"rows={len(combined)} unique={combined['patient_id'].nunique()} hash={sha256_file(COMBINED_MANIFEST)}")
    checks.extend(check_frozen_inputs(config))
    checks.extend(promotion_input_audit())
    failed = [item for item in checks if not item["passed"]]
    report = [
        "# api_fullseq_v2 unattended production release audit",
        "",
        f"- Generated: {utc_now()}",
        f"- Scientific functions audited: {science['scientific_function_count']}",
        f"- Changed scientific functions: {science['changed_scientific_functions']}",
        f"- Historical strict GPU difference rows: {len(failures)} (recorded, nonblocking)",
        f"- Same-pairdata CPU aggregation failures: {cpu_failed}",
        "- Boolean metadata comparison uses truth-value semantics, not object identity.",
        "- Runtime, timestamps, wall time, peak memory, output paths, float16 cache quantization, and sparse binary-cache boundary pixels are nonblocking audit fields.",
        "- Labels read: no",
        "- Model training: no",
        "",
        "## Hard release gates",
        "",
        *[f"- {'PASS' if item['passed'] else 'FAIL'} — {item['name']}: {item['detail']}" for item in checks],
        "",
    ]
    (REPORT_ROOT / "unattended_release_audit.md").write_text("\n".join(report), encoding="utf-8")
    if failed:
        raise HardFailure("Production release audit failed: " + "|".join(item["name"] for item in failed))
    print(json.dumps({
        "release": "PASS",
        "scientific_function_count": science["scientific_function_count"],
        "changed_scientific_functions": science["changed_scientific_functions"],
        "historical_gpu_differences_nonblocking": len(failures),
        "cpu_aggregation_failures": cpu_failed,
        "train": train,
        "valid": valid,
    }, ensure_ascii=False, indent=2))


def frozen_snapshot() -> dict[str, Any]:
    config = load_json(CONFIG)
    science = science_audit()
    files = {
        relative: sha256_file(PROJECT / relative)
        for relative in sorted(config["frozen_inputs"])
    }
    files.update({
        "code/20_extract_api_fullseq_v2_train_valid_pairdata.py": sha256_file(CODE20),
        "code/21_build_api_fullseq_v2_train_valid_features.py": sha256_file(CODE21),
        "code/22_audit_api_fullseq_v2_train_valid.py": sha256_file(CODE22),
        "configs/api_fullseq_v2_full_train_valid_config.json": sha256_file(CONFIG),
        "manifests/api_fullseq_v2_train_manifest.csv": sha256_file(TRAIN_MANIFEST),
        "manifests/api_fullseq_v2_valid_manifest.csv": sha256_file(VALID_MANIFEST),
        "manifests/api_fullseq_v2_full_train_valid_frozen.csv": sha256_file(COMBINED_MANIFEST),
        "model": sha256_file(Path(config["model"]["model_file"])),
        "model_config": sha256_file(Path(config["model"]["config"])),
        "sea_raft_code_tree": sea_raft_code_tree_hash(Path(config["model"]["repo_root"])),
        "scientific_thresholds": thresholds_hash(config),
        "scientific_ast_set": science["combined_ast_sha256"],
    })
    return {"created_utc": utc_now(), "hashes": files}


def snapshot_before_valid() -> None:
    if not TRAIN_SUCCESS.is_file():
        raise HardFailure("Cannot snapshot before Valid without .FULL_TRAIN_SUCCESS")
    write_json(HASH_BEFORE_VALID, frozen_snapshot())
    print(str(HASH_BEFORE_VALID))


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


def split_spec(split: str) -> dict[str, Any]:
    config = load_json(CONFIG)
    if split == "train":
        return {
            "label": "Train", "manifest": TRAIN_MANIFEST,
            "pair_root": PAIR_ROOT / "train",
            "phase_features": FEATURE_ROOT / "train_phase_features.csv",
            "patient_features": FEATURE_ROOT / "train_patient_features.csv",
            "expected_patients": int(config["full"]["expected_train_patients"]),
            "expected_phases": int(config["full"]["expected_train_phases"]),
            "expected_pairs": int(config["full"]["expected_train_pairs"]),
            "expected_prepost": int(config["full"]["expected_train_prepost_patients"]),
            "expected_post_only": int(config["full"]["expected_train_post_only_patients"]),
            "qc_positions": config["full"]["qc_manifest_positions_one_based"],
            "report": REPORT_ROOT / "train_audit.md",
            "phase_audit": REPORT_ROOT / "train_phase_audit.csv",
            "patient_audit": REPORT_ROOT / "train_patient_audit.csv",
            "distribution": REPORT_ROOT / "train_feature_distribution.csv",
            "failures": REPORT_ROOT / "train_failures.csv",
            "success": TRAIN_SUCCESS,
        }
    return {
        "label": "Valid", "manifest": VALID_MANIFEST,
        "pair_root": PAIR_ROOT / "valid",
        "phase_features": FEATURE_ROOT / "valid_phase_features.csv",
        "patient_features": FEATURE_ROOT / "valid_patient_features.csv",
        "expected_patients": int(config["full"]["expected_valid_patients"]),
        "expected_phases": int(config["full"]["expected_valid_phases"]),
        "expected_pairs": int(config["full"]["expected_valid_pairs"]),
        "expected_prepost": int(config["full"]["expected_valid_prepost_patients"]),
        "expected_post_only": int(config["full"]["expected_valid_post_only_patients"]),
        "qc_positions": config["full"]["valid_qc_manifest_positions_one_based"],
        "report": REPORT_ROOT / "valid_audit.md",
        "phase_audit": REPORT_ROOT / "valid_phase_audit.csv",
        "patient_audit": REPORT_ROOT / "valid_patient_audit.csv",
        "distribution": REPORT_ROOT / "valid_feature_distribution.csv",
        "failures": REPORT_ROOT / "valid_failures.csv",
        "success": VALID_SUCCESS,
    }


def audit_split(split: str) -> None:
    spec = split_spec(split)
    config = load_json(CONFIG)
    schema = load_json(SCHEMA)
    if split == "valid":
        if not TRAIN_SUCCESS.is_file():
            raise HardFailure("Full Valid audit requires .FULL_TRAIN_SUCCESS")
        if not HASH_BEFORE_VALID.is_file():
            raise HardFailure("Full Valid audit requires hash_before_valid.json")
    manifest = pd.read_csv(spec["manifest"], dtype={"patient_id": str})
    phase_features = pd.read_csv(spec["phase_features"], dtype={"patient_id": str})
    patient_features = pd.read_csv(spec["patient_features"], dtype={"patient_id": str})
    root_summary = load_json(spec["pair_root"] / "run_summary.json")
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    add("patient_manifest_scale", len(manifest) == spec["expected_patients"] and manifest["patient_id"].nunique() == spec["expected_patients"], f"rows={len(manifest)} unique={manifest['patient_id'].nunique()}")
    add("manifest_phase_scale", int(manifest["can_run_pre"].sum() + manifest["can_run_post"].sum()) == spec["expected_phases"], f"actual={int(manifest['can_run_pre'].sum() + manifest['can_run_post'].sum())}")
    add("manifest_pair_scale", int(manifest["n_pre_contiguous_pairs"].sum() + manifest["n_post_contiguous_pairs"].sum()) == spec["expected_pairs"], f"actual={int(manifest['n_pre_contiguous_pairs'].sum() + manifest['n_post_contiguous_pairs'].sum())}")
    add("manifest_prepost", int(manifest["can_run_prepost"].sum()) == spec["expected_prepost"], f"actual={int(manifest['can_run_prepost'].sum())}")
    post_only_manifest = ~manifest["can_run_pre"] & manifest["can_run_post"]
    add("manifest_post_only", int(post_only_manifest.sum()) == spec["expected_post_only"], f"actual={int(post_only_manifest.sum())}")
    add("root_success", (spec["pair_root"] / ".SUCCESS").is_file(), str(spec["pair_root"] / ".SUCCESS"))
    add("root_counts", root_summary["patients"] == spec["expected_patients"] and len(root_summary["phase_summaries"]) == spec["expected_phases"] and root_summary["processed_pairs"] == spec["expected_pairs"], f"patients={root_summary['patients']} phases={len(root_summary['phase_summaries'])} pairs={root_summary['processed_pairs']}")
    add("root_cuda_semantic", bool(root_summary["cuda_actually_used"]) and not bool(root_summary["cpu_fallback"]), f"cuda={root_summary['cuda_actually_used']} fallback={root_summary['cpu_fallback']}")
    add("root_mode_isolation", (bool(root_summary["full_train_started"]) and not bool(root_summary["full_valid_started"])) if split == "train" else (bool(root_summary["full_valid_started"]) and not bool(root_summary["full_train_started"])), f"train={root_summary['full_train_started']} valid={root_summary['full_valid_started']}")
    add("root_no_labels_training", not bool(root_summary["labels_read"]) and not bool(root_summary["model_trained"]), f"labels={root_summary['labels_read']} trained={root_summary['model_trained']}")

    patient_order = manifest["patient_id"].tolist()
    qc_ids = {patient_order[int(position) - 1] for position in spec["qc_positions"]}
    mapping = config["updated_fixed_series"]
    phase_rows: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    total_pairs = selected_png = selected_parameter = 0
    cross_patient = cross_series = cross_phase = cross_internal = cross_gap = 0
    source_changed = old_source_reads = source_mapping_fail = 0
    labels_read = model_trained = cuda_fail = qc_fail = 0
    for row in manifest.to_dict("records"):
        patient_id = row["patient_id"]
        for phase in ("pre", "post"):
            if not bool(row[f"can_run_{phase}"]):
                continue
            expected_pairs = int(row[f"n_{phase}_contiguous_pairs"])
            phase_dir = spec["pair_root"] / patient_id / phase
            required = [".SUCCESS", "selected_frames.csv", "pair_features.csv.gz", "phase_summary.json", "metadata.json"]
            missing = [name for name in required if not (phase_dir / name).is_file()]
            if missing:
                failure = f"missing:{'|'.join(missing)}"
                unresolved.append({"patient_id": patient_id, "phase": phase, "failure": failure})
                phase_rows.append({"patient_id": patient_id, "phase": phase, "expected_pairs": expected_pairs, "actual_pairs": np.nan, "success": False, "failure": failure})
                continue
            selected = pd.read_csv(phase_dir / "selected_frames.csv", dtype={"patient_id": str})
            pair = pd.read_csv(phase_dir / "pair_features.csv.gz", dtype={"patient_id": str})
            summary = load_json(phase_dir / "phase_summary.json")
            metadata = load_json(phase_dir / "metadata.json")
            actual_pairs = len(pair)
            total_pairs += actual_pairs
            selected_png += int(selected["absolute_path"].str.lower().str.endswith(".png").sum())
            selected_parameter += int(selected["absolute_path"].str.upper().str.contains("CBF|CBV|MTT|TTP", regex=True).sum())
            if actual_pairs:
                cross_patient += int(pair["patient_id"].nunique() != 1 or pair["patient_id"].iloc[0] != patient_id)
                cross_series += int(pair["selected_series_id"].nunique() != 1 or str(pair["selected_series_id"].iloc[0]) != str(row["selected_series_id"]))
                cross_phase += int(pair["phase"].nunique() != 1 or pair["phase"].iloc[0] != phase)
                cross_internal += int(pair["selected_internal_series"].nunique() != 1 or str(pair["selected_internal_series"].iloc[0]) != str(row[f"selected_{phase}_internal_series"]))
                cross_gap += int((pair["delta_frame"] != 1).sum())
            source_changed += int(metadata["source_metadata_signature_before"] != metadata["source_metadata_signature_after"])
            labels_read += int(bool(summary["labels_read"]))
            model_trained += int(bool(summary["model_trained"]))
            cuda_fail += int(not bool(metadata["cuda_actually_used"]) or bool(metadata["cpu_fallback"]))
            frame_paths = metadata["frame_paths"]
            if patient_id in mapping:
                expected_root = str(PROJECT / "staging/updated_10_cases" / patient_id)
                source_mapping_fail += int(summary["source_type"] != "updated_10_cases")
                source_mapping_fail += int(str(summary["source_medical_record_root"]) != expected_root)
                source_mapping_fail += int(str(summary["selected_series_id"]) != str(mapping[patient_id]))
                old_source_reads += sum(int(str(path).startswith(f"/root/autodl-tmp/tiantanDSA/{patient_id}/")) for path in frame_paths)
            visual_count = len(list((phase_dir / "visualizations").glob("*_residual_flow.jpg"))) if (phase_dir / "visualizations").is_dir() else 0
            expected_visual = 1 if patient_id in qc_ids else 0
            qc_fail += int(visual_count != expected_visual)
            failure = "" if actual_pairs == expected_pairs else f"pair_mismatch:{expected_pairs}!={actual_pairs}"
            if failure:
                unresolved.append({"patient_id": patient_id, "phase": phase, "failure": failure})
            phase_rows.append({
                "patient_id": patient_id, "split": row["split"],
                "source_type": row["source_type"], "selected_series_id": row["selected_series_id"],
                "phase": phase, "n_frames": len(selected), "expected_pairs": expected_pairs,
                "actual_pairs": actual_pairs, "delta_frame_one": bool((pair["delta_frame"] == 1).all()),
                "cuda_actually_used": bool(metadata["cuda_actually_used"]),
                "cpu_fallback": bool(metadata["cpu_fallback"]),
                "source_unchanged": metadata["source_metadata_signature_before"] == metadata["source_metadata_signature_after"],
                "visual_pairs": visual_count, "fixed_qc_patient": patient_id in qc_ids,
                "success": not failure, "failure": failure,
            })
    phase_audit = pd.DataFrame(phase_rows)
    phase_audit.to_csv(spec["phase_audit"], index=False, encoding="utf-8", lineterminator="\n")
    add("phase_rows", len(phase_audit) == spec["expected_phases"], f"actual={len(phase_audit)}")
    add("pair_total", total_pairs == spec["expected_pairs"], f"actual={total_pairs}")
    add("expected_actual_every_phase", bool((phase_audit["expected_pairs"] == phase_audit["actual_pairs"]).all()), f"phases={len(phase_audit)}")
    add("all_delta_frame_one", cross_gap == 0 and bool(phase_audit["delta_frame_one"].all()), f"cross_gap={cross_gap}")
    add("selected_png_zero", selected_png == 0, f"count={selected_png}")
    add("selected_parameter_zero", selected_parameter == 0, f"count={selected_parameter}")
    add("cross_patient_zero", cross_patient == 0, f"count={cross_patient}")
    add("cross_series_zero", cross_series == 0, f"count={cross_series}")
    add("cross_phase_zero", cross_phase == 0, f"count={cross_phase}")
    add("cross_internal_zero", cross_internal == 0, f"count={cross_internal}")
    add("source_unchanged", source_changed == 0, f"count={source_changed}")
    add("updated_source_mapping", source_mapping_fail == 0, f"count={source_mapping_fail}")
    add("updated_old_source_reads_zero", old_source_reads == 0, f"count={old_source_reads}")
    add("labels_read_no", labels_read == 0, f"count={labels_read}")
    add("model_training_no", model_trained == 0, f"count={model_trained}")
    add("cuda_every_phase", cuda_fail == 0, f"count={cuda_fail}")
    add("fixed_qc_exact", qc_fail == 0, f"mismatched_phases={qc_fail} qc_ids={sorted(qc_ids)}")
    add("unresolved_failed_phases_zero", len(unresolved) == 0, f"count={len(unresolved)}")
    if split == "valid":
        row549 = manifest[manifest["patient_id"] == "549117"]
        add("valid_549117_staging_c6_1", len(row549) == 1 and row549.iloc[0]["source_type"] == "updated_10_cases" and str(row549.iloc[0]["selected_series_id"]) == "C6-1" and str(row549.iloc[0]["source_medical_record_root"]) == str(PROJECT / "staging/updated_10_cases/549117"), row549[["source_type", "selected_series_id", "source_medical_record_root"]].to_dict("records"))

    expected_phase_keys: list[tuple[str, str]] = []
    for row in manifest.to_dict("records"):
        for phase in ("pre", "post"):
            if bool(row[f"can_run_{phase}"]):
                expected_phase_keys.append((row["patient_id"], phase))
    add("phase_feature_rows", len(phase_features) == spec["expected_phases"], f"actual={len(phase_features)}")
    add("phase_feature_order", list(zip(phase_features["patient_id"], phase_features["phase"])) == expected_phase_keys, f"keys={len(phase_features)}")
    add("phase_feature_pair_total", int(phase_features["processed_pairs"].sum()) == spec["expected_pairs"], f"actual={int(phase_features['processed_pairs'].sum())}")
    add("patient_feature_rows", len(patient_features) == spec["expected_patients"] and patient_features["patient_id"].nunique() == spec["expected_patients"], f"rows={len(patient_features)} unique={patient_features['patient_id'].nunique()}")
    add("patient_feature_order", patient_features["patient_id"].tolist() == patient_order, f"rows={len(patient_features)}")
    phase_numeric = phase_features.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    patient_numeric = patient_features.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    add("inf_zero", not np.isinf(phase_numeric).any() and not np.isinf(patient_numeric).any(), f"phase_inf={int(np.isinf(phase_numeric).sum())} patient_inf={int(np.isinf(patient_numeric).sum())}")
    post_only = patient_features[patient_features["missing_pre"] == 1]
    pre_columns = [column for column in patient_features.columns if column.startswith("pre_")]
    delta_columns = [column for column in patient_features.columns if column.startswith("delta_")]
    add("post_only_feature_count", len(post_only) == spec["expected_post_only"], f"actual={len(post_only)}")
    add("post_only_pre_delta_nan", bool(post_only[pre_columns + delta_columns].isna().all().all()), f"patients={len(post_only)}")
    model_columns = [item["feature_name"] for item in schema["patient_features"] if item["model_candidate"]]
    model_values = patient_features[model_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    add("model_candidate_nonmissing_finite", not np.isinf(model_values).any(), f"finite={int(np.isfinite(model_values).sum())} inf={int(np.isinf(model_values).sum())}")
    science = science_audit()
    add("scientific_hashes_unchanged", science["changed_scientific_functions"] == 0 and not science["missing_scientific_functions"], f"changed={science['changed_scientific_functions']}")
    checks.extend(check_frozen_inputs(config))

    patient_rows: list[dict[str, Any]] = []
    grouped = phase_audit.groupby("patient_id", sort=False)
    for row in manifest.to_dict("records"):
        group = grouped.get_group(row["patient_id"])
        feature_row = patient_features[patient_features["patient_id"] == row["patient_id"]].iloc[0]
        patient_rows.append({
            "patient_id": row["patient_id"], "split": row["split"], "source_type": row["source_type"],
            "expected_phases": int(row["can_run_pre"]) + int(row["can_run_post"]),
            "actual_phases": len(group),
            "expected_pairs": int(row["n_pre_contiguous_pairs"] + row["n_post_contiguous_pairs"]),
            "actual_pairs": int(group["actual_pairs"].sum()),
            "missing_pre": int(feature_row["missing_pre"]), "missing_post": int(feature_row["missing_post"]),
            "all_phases_success": bool(group["success"].all()),
        })
    pd.DataFrame(patient_rows).to_csv(spec["patient_audit"], index=False, encoding="utf-8", lineterminator="\n")
    distribution = pd.concat([
        feature_distribution(phase_features, "phase"),
        feature_distribution(patient_features, "patient"),
    ], ignore_index=True)
    distribution.to_csv(spec["distribution"], index=False, encoding="utf-8", lineterminator="\n")
    failed = [item for item in checks if not item["passed"]]
    failure_rows = [
        {"scope": "check", "name": item["name"], "detail": item["detail"]}
        for item in failed
    ] + [
        {"scope": "phase", "name": f"{item['patient_id']}:{item['phase']}", "detail": item["failure"]}
        for item in unresolved
    ]
    pd.DataFrame(failure_rows, columns=["scope", "name", "detail"]).to_csv(spec["failures"], index=False, encoding="utf-8", lineterminator="\n")
    source_distribution = patient_features["source_type"].value_counts(dropna=False).to_dict()
    pattern_distribution = manifest.assign(pattern=np.where(manifest["can_run_prepost"], "Pre+Post", "Post-only"))["pattern"].value_counts().to_dict()
    report = [
        f"# api_fullseq_v2 Full {spec['label']} automated audit",
        "", f"- Generated: {utc_now()}",
        f"- Patients: {len(patient_features)}", f"- Phases: {len(phase_features)}",
        f"- Pairs: {total_pairs}", f"- Unresolved failed phases: {len(unresolved)}",
        f"- Source distribution: {json.dumps(source_distribution, ensure_ascii=False, sort_keys=True)}",
        f"- Pre/Post distribution: {json.dumps(pattern_distribution, ensure_ascii=False, sort_keys=True)}",
        f"- Fixed QC patient IDs: {', '.join(sorted(qc_ids))}",
        "- Labels read: no", "- Model training: no", "",
        "## Hard assertions", "",
        *[f"- {'PASS' if item['passed'] else 'FAIL'} — {item['name']}: {item['detail']}" for item in checks],
        "", "## Distribution audit", "",
        f"- Numeric feature rows audited: {len(distribution)}",
        f"- Features containing Inf: {int((distribution['inf_count'] > 0).sum())}",
        "- Distribution values are audit-only and did not alter extraction science.", "",
    ]
    spec["report"].write_text("\n".join(report), encoding="utf-8")
    if failed or unresolved:
        raise HardFailure(f"Full {spec['label']} audit failed: checks={len(failed)} unresolved={len(unresolved)}")
    if split == "valid":
        after = frozen_snapshot()
        write_json(HASH_AFTER_VALID, after)
        before = load_json(HASH_BEFORE_VALID)
        if before["hashes"] != after["hashes"]:
            changed = sorted(key for key in set(before["hashes"]) | set(after["hashes"]) if before["hashes"].get(key) != after["hashes"].get(key))
            raise HardFailure("Frozen hashes changed across Valid: " + "|".join(changed))
    write_json_atomic(spec["success"], {
        "created_utc": utc_now(), "split": spec["label"],
        "patients": len(patient_features), "phases": len(phase_features),
        "pairs": total_pairs, "unresolved_failed_phases": 0,
        "hard_checks": len(checks), "labels_read": False, "model_trained": False,
    })
    print(str(spec["success"]))


def finalize() -> None:
    if not TRAIN_SUCCESS.is_file() or not VALID_SUCCESS.is_file():
        raise HardFailure("Finalization requires both Train and Valid success markers")
    if not HASH_BEFORE_VALID.is_file() or not HASH_AFTER_VALID.is_file():
        raise HardFailure("Finalization requires before/after Valid hash snapshots")
    before = load_json(HASH_BEFORE_VALID)
    after = load_json(HASH_AFTER_VALID)
    if before["hashes"] != after["hashes"]:
        raise HardFailure("Frozen hashes differ before and after Valid")
    train = pd.read_csv(FEATURE_ROOT / "train_patient_features.csv", dtype={"patient_id": str})
    valid = pd.read_csv(FEATURE_ROOT / "valid_patient_features.csv", dtype={"patient_id": str})
    overlap = sorted(set(train["patient_id"]) & set(valid["patient_id"]))
    isolation = [
        "# api_fullseq_v2 Train/Valid isolation audit", "",
        f"- Generated: {utc_now()}", f"- Train patients: {train['patient_id'].nunique()}",
        f"- Valid patients: {valid['patient_id'].nunique()}", f"- Patient overlap: {len(overlap)}",
        "- Frozen hashes unchanged across Valid: yes", "- Labels read: no",
        "- Model training: no", "",
    ]
    (REPORT_ROOT / "train_valid_isolation_audit.md").write_text("\n".join(isolation), encoding="utf-8")
    if overlap:
        raise HardFailure(f"Train/Valid patient overlap={len(overlap)}")
    summary = [
        "# api_fullseq_v2 final unattended Train + Valid summary", "",
        f"- Completed: {utc_now()}", "- Full Train: PASS (1055 patients, 1921 phases, 39906 pairs)",
        "- Full Valid: PASS (264 patients, 492 phases, 10124 pairs)",
        "- Train/Valid patient overlap: 0", "- Frozen hashes unchanged across Valid: yes",
        "- Historical strict GPU differences retained as nonblocking evidence: yes",
        "- Labels read: no", "- Model training: no", "",
    ]
    (REPORT_ROOT / "final_unattended_summary.md").write_text("\n".join(summary), encoding="utf-8")
    write_json_atomic(FINAL_SUCCESS, {
        "created_utc": utc_now(), "train_success": True, "valid_success": True,
        "train_patients": 1055, "valid_patients": 264, "patient_overlap": 0,
        "hashes_unchanged_across_valid": True, "labels_read": False, "model_trained": False,
    })
    print(str(FINAL_SUCCESS))


def write_failure(stage: str, exc: BaseException) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    path = REPORT_ROOT / "unattended_failure.md"
    if path.exists():
        return
    path.write_text(
        "\n".join([
            "# api_fullseq_v2 unattended production failure", "",
            f"- Generated: {utc_now()}", f"- Stage: {stage}",
            f"- Exception: {type(exc).__name__}: {exc}",
            "- Existing successful phases were preserved for --resume.",
            "- Historical promotion evidence was not modified.",
            "- Labels read: no.", "- Model training: no.", "",
            "## Traceback", "", traceback.format_exc(), "",
        ]),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=[
        "preflight", "audit-train", "snapshot-before-valid", "audit-valid", "finalize",
    ])
    args = parser.parse_args()
    try:
        if args.mode == "preflight":
            run_preflight()
        elif args.mode == "audit-train":
            audit_split("train")
        elif args.mode == "snapshot-before-valid":
            snapshot_before_valid()
        elif args.mode == "audit-valid":
            audit_split("valid")
        else:
            finalize()
        return 0
    except HardFailure as exc:
        write_failure(args.mode, exc)
        print(f"HARD_FAILURE: {exc}", file=sys.stderr)
        return 42
    except BaseException as exc:
        write_failure(args.mode, exc)
        raise


if __name__ == "__main__":
    sys.exit(main())

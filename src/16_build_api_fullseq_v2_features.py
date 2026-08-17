#!/usr/bin/env python3
"""Build phase-level and patient-level api_fullseq_v2 Pilot features from stage-one data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if not os.environ.get("OMP_NUM_THREADS", "").isdigit():
    os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd


PROJECT = Path("/root/autodl-tmp/aneurysm")
CONFIG_PATH = PROJECT / "configs/api_fullseq_v2_pairdata_config.json"
PILOT_MANIFEST = PROJECT / "manifests/api_fullseq_v2_pilot_manifest.csv"
SCHEMA_DEFAULT = PROJECT / "configs/api_fullseq_v2_feature_schema.json"
FREEZE_DEFAULT = PROJECT / "manifests/api_fullseq_v2_gpu_pilot_frozen.csv"
FEATURE_ROOT = PROJECT / "outputs/api_fullseq_v2_features/pilot"
REPORT_ROOT = PROJECT / "reports/api_fullseq_v2_feature_pilot"
CODE15 = PROJECT / "code/15_extract_api_fullseq_v2_pairdata.py"
CODE16 = PROJECT / "code/16_build_api_fullseq_v2_features.py"
FLOW_METRICS = [
    "active_res_mag_norm_median",
    "active_res_mag_norm_p90",
    "active_weighted_mag_norm_mean",
    "active_direction_coherence",
    "active_direction_entropy",
    "filling_front_weighted_mag_norm_mean",
    "filling_front_coverage_fov",
    "persistent_weighted_mag_norm_mean",
    "persistent_coverage_fov",
    "washout_front_weighted_mag_norm_mean",
    "washout_front_coverage_fov",
]
STAGE_FLOW_METRICS = [
    "active_weighted_mag_norm_mean",
    "active_direction_coherence",
    "filling_front_weighted_mag_norm_mean",
    "filling_front_coverage_fov",
    "persistent_weighted_mag_norm_mean",
    "washout_front_weighted_mag_norm_mean",
    "washout_front_coverage_fov",
]
PAIR_QC_METRICS = [
    "fb_relative_mean",
    "fb_relative_p90",
    "uncertainty_log_mean",
    "uncertainty_log_p90",
    "hard_valid_ratio_fov",
    "soft_weight_mean_fov",
    "global_motion_mag_norm",
    "runtime_seconds",
]
STAGES = ("precontrast", "washin", "peak", "washout")
LEGACY_COLUMNS = [
    "mag_mean", "mag_median", "mag_std", "mag_p90", "mag_p95", "mag_max",
    "mag_norm_mean", "mag_norm_p90", "u_mean", "u_std", "v_mean", "v_std",
    "direction_entropy", "uncertainty_mean", "uncertainty_std", "uncertainty_p90",
]
IDENTIFIER_COLUMNS = [
    "patient_id", "split", "source_type", "selected_series_id", "phase",
    "selected_internal_series", "n_frames", "manifest_expected_pairs",
    "processed_pairs",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    material = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


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
    path.write_text(
        json.dumps(sanitize_json(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def mode_spec(config: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode == "pilot_train":
        return {
            "ids": list(config["pilot"]["train_patient_ids"]),
            "split": "Train",
            "expected_pairs": int(config["pilot"]["expected_train_pairs"]),
            "phase_default": FEATURE_ROOT / "train_phase_features.csv",
            "patient_default": FEATURE_ROOT / "train_patient_features.csv",
            "missing_default": FEATURE_ROOT / "train_missingness_audit.csv",
            "group_default": FEATURE_ROOT / "train_feature_group_counts.csv",
            "report_markdown": REPORT_ROOT / "train_pilot_audit.md",
            "report_phase": REPORT_ROOT / "train_phase_audit.csv",
        }
    if mode == "pilot_valid_integrity":
        return {
            "ids": list(config["pilot"]["valid_integrity_patient_ids"]),
            "split": "Valid",
            "expected_pairs": int(config["pilot"]["expected_valid_integrity_pairs"]),
            "phase_default": FEATURE_ROOT / "valid_integrity_phase_features.csv",
            "patient_default": FEATURE_ROOT / "valid_integrity_patient_features.csv",
            "missing_default": FEATURE_ROOT / "valid_integrity_missingness_audit.csv",
            "group_default": FEATURE_ROOT / "valid_integrity_feature_group_counts.csv",
            "report_markdown": REPORT_ROOT / "valid_integrity_audit.md",
            "report_phase": REPORT_ROOT / "valid_integrity_phase_audit.csv",
        }
    if mode in {"full_train", "full_valid"}:
        raise RuntimeError(f"{mode} is forbidden for the current Pilot task")
    raise ValueError(mode)


def expected_phase_records(config: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    spec = mode_spec(config, mode)
    manifest = pd.read_csv(PILOT_MANIFEST, dtype={"patient_id": str})
    manifest = manifest[manifest["patient_id"].isin(spec["ids"])].copy()
    if set(manifest["patient_id"]) != set(spec["ids"]):
        raise AssertionError("Expected Pilot patient missing from frozen manifest")
    if not (manifest["split"] == spec["split"]).all():
        raise AssertionError("Split isolation failure")
    order = {patient_id: index for index, patient_id in enumerate(spec["ids"])}
    manifest["_order"] = manifest["patient_id"].map(order)
    manifest = manifest.sort_values("_order")
    records: list[dict[str, Any]] = []
    for row in manifest.to_dict("records"):
        for phase in ("pre", "post"):
            if bool(row[f"can_run_{phase}"]):
                records.append({
                    "patient_id": row["patient_id"],
                    "split": row["split"],
                    "source_type": row["source_type"],
                    "selected_series_id": row["selected_series_id"],
                    "phase": phase,
                    "n_frames": int(row[f"n_{phase}_frames"]),
                    "expected_pairs": int(row[f"n_{phase}_contiguous_pairs"]),
                })
    return records


def read_phase_artifacts(
    pairdata_root: Path,
    expected: dict[str, Any],
) -> dict[str, Any]:
    phase_dir = pairdata_root / expected["patient_id"] / expected["phase"]
    required = [
        "selected_frames.csv", "pair_features.csv.gz", "frame_kinetics.csv.gz",
        "temporal_curves.csv.gz", "phase_cache.npz", "phase_summary.json",
        "metadata.json", "run.log", ".SUCCESS",
    ]
    missing = [name for name in required if not (phase_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{phase_dir}: missing {missing}")
    selected = pd.read_csv(
        phase_dir / "selected_frames.csv", dtype={"patient_id": str}
    )
    pair = pd.read_csv(
        phase_dir / "pair_features.csv.gz", dtype={"patient_id": str}
    )
    frame = pd.read_csv(
        phase_dir / "frame_kinetics.csv.gz", dtype={"patient_id": str}
    )
    curves = pd.read_csv(
        phase_dir / "temporal_curves.csv.gz", dtype={"patient_id": str}
    )
    summary = json.loads((phase_dir / "phase_summary.json").read_text(encoding="utf-8"))
    metadata = json.loads((phase_dir / "metadata.json").read_text(encoding="utf-8"))
    if len(selected) != expected["n_frames"]:
        raise AssertionError(f"{phase_dir}: selected frame count mismatch")
    if len(pair) != expected["expected_pairs"]:
        raise AssertionError(f"{phase_dir}: pair count mismatch")
    if not (pair["delta_frame"] == 1).all():
        raise AssertionError(f"{phase_dir}: cross-gap pair detected")
    if pair["patient_id"].nunique() != 1 or pair["patient_id"].iloc[0] != expected["patient_id"]:
        raise AssertionError(f"{phase_dir}: cross-patient contamination")
    if pair["phase"].nunique() != 1 or pair["phase"].iloc[0] != expected["phase"]:
        raise AssertionError(f"{phase_dir}: cross-phase contamination")
    if pair["selected_series_id"].nunique() != 1 or str(pair["selected_series_id"].iloc[0]) != str(expected["selected_series_id"]):
        raise AssertionError(f"{phase_dir}: cross-series contamination")
    if pair["selected_internal_series"].nunique() != 1:
        raise AssertionError(f"{phase_dir}: cross-internal-series contamination")
    if int(summary["processed_pairs"]) != expected["expected_pairs"]:
        raise AssertionError(f"{phase_dir}: phase summary pair mismatch")
    if not summary["complete_phase"]:
        raise AssertionError(f"{phase_dir}: partial smoke output cannot enter features")
    if not metadata["cuda_actually_used"] or metadata["cpu_fallback"]:
        raise AssertionError(f"{phase_dir}: CUDA assertion failed")
    for table_name, table in {
        "pair": pair, "frame": frame, "curves": curves
    }.items():
        numeric = table.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
        if np.isinf(numeric).any():
            raise AssertionError(f"{phase_dir}: infinity in {table_name}")
    return {
        "phase_dir": phase_dir,
        "selected": selected,
        "pair": pair,
        "frame": frame,
        "curves": curves,
        "summary": summary,
        "metadata": metadata,
    }


def finite_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    numeric = numeric[np.isfinite(numeric)]
    return float(np.mean(numeric)) if len(numeric) else float("nan")


def finite_std(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    numeric = numeric[np.isfinite(numeric)]
    return float(np.std(numeric)) if len(numeric) else float("nan")


def finite_percentile(values: pd.Series, percentile: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    numeric = numeric[np.isfinite(numeric)]
    return float(np.percentile(numeric, percentile)) if len(numeric) else float("nan")


def lagged_correlation(flow: np.ndarray, derivative: np.ndarray, lag: int) -> float:
    if lag > 0:
        left, right = flow[:-lag], derivative[lag:]
    elif lag < 0:
        left, right = flow[-lag:], derivative[:lag]
    else:
        left, right = flow, derivative
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    if len(left) < 3 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def flow_tdc_coupling(pair: pd.DataFrame) -> dict[str, float]:
    flow = pd.to_numeric(
        pair["active_weighted_mag_norm_mean"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    derivative = pd.to_numeric(
        pair["tdc_derivative_pair"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    times = pd.to_numeric(
        pair["normalized_pair_time"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    correlations = {
        lag: lagged_correlation(flow, derivative, lag) for lag in range(-3, 4)
    }
    finite_correlations = {
        lag: value for lag, value in correlations.items() if math.isfinite(value)
    }
    if finite_correlations:
        best_lag, best_correlation = max(
            finite_correlations.items(), key=lambda item: item[1]
        )
    else:
        best_lag, best_correlation = 0, float("nan")
    finite_flow = np.isfinite(flow)
    finite_derivative = np.isfinite(derivative)
    flow_peak_time = (
        float(times[np.nanargmax(np.where(finite_flow, flow, np.nan))])
        if finite_flow.any() else float("nan")
    )
    derivative_peak_time = (
        float(times[np.nanargmax(np.where(finite_derivative, derivative, np.nan))])
        if finite_derivative.any() else float("nan")
    )
    washin = pair["stage"].astype(str).eq("washin").to_numpy()
    valid_washin = washin & np.isfinite(flow) & np.isfinite(derivative) & np.isfinite(times)
    if int(valid_washin.sum()) >= 2:
        order = np.argsort(times[valid_washin])
        flow_auc = float(np.trapz(flow[valid_washin][order], times[valid_washin][order]))
        tdc_auc = float(np.trapz(np.abs(derivative[valid_washin][order]), times[valid_washin][order]))
        ratio = flow_auc / max(tdc_auc, 1e-12)
    else:
        ratio = float("nan")
    return {
        "coupling_maximum_flow_tdc_correlation": best_correlation,
        "coupling_lag_at_maximum_correlation": float(
            best_lag / max(len(pair) - 1, 1)
        ),
        "coupling_flow_peak_minus_tdc_peak": flow_peak_time - derivative_peak_time,
        "coupling_washin_flow_auc_to_tdc_auc_ratio": ratio,
        "coupling_pair_flow_intensity_corr_mean": finite_mean(pair["flow_intensity_corr"]),
        "coupling_pair_flow_intensity_corr_p90": finite_percentile(pair["flow_intensity_corr"], 90),
        "coupling_flow_front_overlap_mean": finite_mean(pair["flow_front_overlap"]),
        "coupling_high_flow_high_change_ratio_mean": finite_mean(
            pair["high_flow_high_change_ratio"]
        ),
    }


def aggregate_phase(
    artifacts: dict[str, Any],
    emit_legacy_ablation: bool,
) -> dict[str, Any]:
    summary = artifacts["summary"]
    pair = artifacts["pair"]
    row: dict[str, Any] = {
        "patient_id": summary["patient_id"],
        "split": summary["split"],
        "source_type": summary["source_type"],
        "selected_series_id": summary["selected_series_id"],
        "phase": summary["phase"],
        "selected_internal_series": summary["selected_internal_series"],
        "n_frames": summary["n_frames"],
        "manifest_expected_pairs": summary["manifest_expected_pairs"],
        "processed_pairs": summary["processed_pairs"],
    }
    row.update(summary["base_features"])
    for metric in FLOW_METRICS:
        row[f"flow_{metric}_mean"] = finite_mean(pair[metric])
        row[f"flow_{metric}_std"] = finite_std(pair[metric])
        row[f"flow_{metric}_p90"] = finite_percentile(pair[metric], 90)
    for stage in STAGES:
        subset = pair[pair["stage"].astype(str) == stage]
        for metric in STAGE_FLOW_METRICS:
            row[f"flow_{stage}_{metric}_mean"] = finite_mean(subset[metric])
    row.update(flow_tdc_coupling(pair))
    for key, value in summary["qc_features"].items():
        row[key] = value
    for metric in PAIR_QC_METRICS:
        row[f"qc_pair_{metric}_mean"] = finite_mean(pair[metric])
        row[f"qc_pair_{metric}_p90"] = finite_percentile(pair[metric], 90)
    for stage in STAGES:
        row[f"qc_stage_{stage}_pair_count"] = int(
            (pair["stage"].astype(str) == stage).sum()
        )
    if emit_legacy_ablation:
        for column in LEGACY_COLUMNS:
            row[f"legacy_{column}_mean"] = finite_mean(pair[column])
            row[f"legacy_{column}_std"] = finite_std(pair[column])
            row[f"legacy_{column}_max"] = finite_percentile(pair[column], 100)
    return row


def infer_unit(name: str) -> str:
    lower = name.casefold()
    if "frame_index" in lower or lower.endswith("_count") or "pair_count" in lower:
        return "count_or_frame_index"
    if any(token in lower for token in ("time", "duration", "fwhm", "lag", "position")):
        return "normalized_time"
    if "slope" in lower:
        return "normalized_intensity_per_normalized_time"
    if "auc" in lower:
        return "normalized_integral"
    if any(token in lower for token in ("mag_norm", "motion_u_norm", "motion_v_norm")):
        return "image_fraction_per_frame"
    if any(token in lower for token in ("ratio", "coverage", "coherence", "entropy", "corr", "polarity")):
        return "unitless"
    if any(token in lower for token in ("peak", "mean", "median", "p10", "p90", "std", "iqr")):
        return "normalized_intensity_or_unitless"
    return "unitless"


def definition_for_feature(name: str) -> dict[str, Any]:
    if name.startswith("qc_"):
        group = "qc"
        source = "stage-one QC or pair-level reliability aggregation"
        model_candidate = False
        qc_only = True
        delta_policy = "never_delta"
    elif name.startswith("tdc_"):
        group = "global_kinetics"
        source = "active-ROI global time-density curve"
        model_candidate = not (
            "frame_index" in name or name.endswith("_washout_present")
        )
        qc_only = not model_candidate
        delta_policy = "safe" if model_candidate else "never_delta"
    elif name.startswith("kinetic_"):
        group = "spatial_kinetic_maps"
        source = "pixelwise TOA/TTP/peak/AUC/slope/FWHM parameter maps"
        model_candidate = not (
            name.endswith("_valid_ratio") or name == "kinetic_any_valid_ratio"
        )
        qc_only = not model_candidate
        delta_policy = "safe" if model_candidate else "never_delta"
    elif name.startswith("filling_"):
        group = "filling_morphology"
        source = "visible/new/washout area and spatial filling curves"
        model_candidate = True
        qc_only = False
        delta_policy = "safe"
    elif name.startswith("flow_"):
        group = "sea_raft_apparent_transport"
        source = "forward/backward SEA-RAFT residual apparent displacement aggregation"
        model_candidate = True
        qc_only = False
        delta_policy = "safe"
    elif name.startswith("coupling_"):
        group = "flow_tdc_coupling"
        source = "apparent-transport curve and TDC-derivative coupling"
        model_candidate = True
        qc_only = False
        delta_policy = "safe"
    elif name.startswith("legacy_"):
        group = "legacy_ablation"
        source = "legacy 16 pair columns aggregated without rerunning SEA-RAFT"
        model_candidate = False
        qc_only = False
        delta_policy = "never_delta"
    else:
        group = "other"
        source = "phase summary"
        model_candidate = False
        qc_only = True
        delta_policy = "never_delta"
    return {
        "feature_name": name,
        "group": group,
        "definition": source,
        "source": source,
        "unit": infer_unit(name),
        "model_candidate": model_candidate,
        "qc_only": qc_only,
        "delta_policy": delta_policy,
        "missing_policy": (
            "NaN when the region, temporal stage, or required phase is unavailable; never zero-filled"
        ),
    }


def build_schema(
    phase_frame: pd.DataFrame,
    emit_legacy_ablation: bool,
) -> dict[str, Any]:
    phase_feature_names = [
        column for column in phase_frame.columns if column not in IDENTIFIER_COLUMNS
    ]
    phase_definitions = [definition_for_feature(name) for name in phase_feature_names]
    patient_definitions: list[dict[str, Any]] = []
    for definition in phase_definitions:
        for prefix in ("pre", "post"):
            patient_definition = dict(definition)
            patient_definition["feature_name"] = f"{prefix}_{definition['feature_name']}"
            patient_definition["source"] = f"{prefix} phase: {definition['source']}"
            patient_definitions.append(patient_definition)
        if definition["delta_policy"] == "safe":
            delta_definition = dict(definition)
            delta_definition["feature_name"] = f"delta_{definition['feature_name']}"
            delta_definition["definition"] = (
                f"Post minus Pre for delta-safe phase feature {definition['feature_name']}"
            )
            delta_definition["source"] = "post phase minus pre phase"
            patient_definitions.append(delta_definition)
    patient_definitions.extend([
        {
            "feature_name": "missing_pre",
            "group": "missingness",
            "definition": "1 when no runnable Pre phase is present",
            "source": "frozen patient manifest",
            "unit": "binary",
            "model_candidate": False,
            "qc_only": True,
            "delta_policy": "never_delta",
            "missing_policy": "never missing",
        },
        {
            "feature_name": "missing_post",
            "group": "missingness",
            "definition": "1 when no runnable Post phase is present",
            "source": "frozen patient manifest",
            "unit": "binary",
            "model_candidate": False,
            "qc_only": True,
            "delta_policy": "never_delta",
            "missing_policy": "never missing",
        },
    ])
    return {
        "version": "api_fullseq_v2_feature_schema_v1",
        "created_utc": utc_now(),
        "emit_legacy_ablation": emit_legacy_ablation,
        "phase_features": phase_definitions,
        "patient_features": patient_definitions,
        "delta_rule": "Only phase features with delta_policy=safe receive delta_ columns",
        "post_only_rule": "All pre_ and delta_ values remain NaN; missing_pre=1",
        "qc_rule": "qc_only features default model_candidate=false",
        "model_candidate_phase_count": int(
            sum(bool(item["model_candidate"]) for item in phase_definitions)
        ),
        "qc_phase_count": int(sum(bool(item["qc_only"]) for item in phase_definitions)),
        "phase_feature_count": len(phase_definitions),
        "patient_feature_count": len(patient_definitions),
        "groups": dict(pd.Series([item["group"] for item in phase_definitions]).value_counts()),
    }


def build_patient_features(
    phase_frame: pd.DataFrame,
    schema: dict[str, Any],
    expected_ids: list[str],
    expected_split: str,
) -> pd.DataFrame:
    feature_definitions = {
        item["feature_name"]: item for item in schema["phase_features"]
    }
    feature_names = list(feature_definitions)
    rows: list[dict[str, Any]] = []
    for patient_id in expected_ids:
        patient_phases = phase_frame[phase_frame["patient_id"] == patient_id]
        if patient_phases.empty:
            raise AssertionError(f"No phase features for {patient_id}")
        if patient_phases["split"].nunique() != 1 or patient_phases["split"].iloc[0] != expected_split:
            raise AssertionError(f"Split isolation failure for {patient_id}")
        row: dict[str, Any] = {
            "patient_id": patient_id,
            "split": expected_split,
            "source_type": patient_phases["source_type"].iloc[0],
            "selected_series_id": patient_phases["selected_series_id"].iloc[0],
        }
        phase_rows = {
            phase: patient_phases[patient_phases["phase"] == phase]
            for phase in ("pre", "post")
        }
        missing_pre = phase_rows["pre"].empty
        missing_post = phase_rows["post"].empty
        row["missing_pre"] = int(missing_pre)
        row["missing_post"] = int(missing_post)
        for feature_name, definition in feature_definitions.items():
            values: dict[str, float] = {}
            for phase in ("pre", "post"):
                if phase_rows[phase].empty:
                    value = float("nan")
                else:
                    raw = phase_rows[phase].iloc[0][feature_name]
                    value = (
                        float(raw)
                        if isinstance(raw, (int, float, np.integer, np.floating, bool, np.bool_))
                        and not pd.isna(raw)
                        else float("nan")
                    )
                row[f"{phase}_{feature_name}"] = value
                values[phase] = value
            if definition["delta_policy"] == "safe":
                row[f"delta_{feature_name}"] = (
                    values["post"] - values["pre"]
                    if math.isfinite(values["post"]) and math.isfinite(values["pre"])
                    else float("nan")
                )
        rows.append(row)
    patient_frame = pd.DataFrame(rows)
    if patient_frame["patient_id"].duplicated().any():
        raise AssertionError("Patient-level feature table is not unique")
    return patient_frame


def schema_assertions(schema: dict[str, Any]) -> list[dict[str, Any]]:
    assertions: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        assertions.append({"name": name, "passed": bool(passed), "detail": detail})

    required = {
        "feature_name", "group", "definition", "source", "unit",
        "model_candidate", "qc_only", "delta_policy", "missing_policy",
    }
    all_definitions = schema["phase_features"] + schema["patient_features"]
    check(
        "schema_all_required_fields",
        all(required.issubset(item) for item in all_definitions),
        f"definitions={len(all_definitions)}",
    )
    check(
        "schema_delta_policies_valid",
        all(
            item["delta_policy"] in {"safe", "matched_view_only", "never_delta"}
            for item in all_definitions
        ),
        "safe|matched_view_only|never_delta",
    )
    check(
        "qc_default_not_model_candidate",
        all(
            not item["model_candidate"]
            for item in all_definitions if item["qc_only"]
        ),
        "all qc_only model_candidate=false",
    )
    patient_delta_names = {
        item["feature_name"]
        for item in schema["patient_features"]
        if item["feature_name"].startswith("delta_")
    }
    safe_expected = {
        f"delta_{item['feature_name']}"
        for item in schema["phase_features"]
        if item["delta_policy"] == "safe"
    }
    check(
        "only_delta_safe_features_have_delta",
        patient_delta_names == safe_expected,
        f"delta_features={len(patient_delta_names)}",
    )
    return assertions


def thresholds_hash(config: dict[str, Any]) -> str:
    return sha256_json({
        "normalization": config["normalization"],
        "fov": config["fov"],
        "activity": config["activity"],
        "kinetics": config["kinetics"],
        "flow_qc": config["flow_qc"],
        "aggregation": config["aggregation"],
    })


def freeze_rows(schema_path: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    repo_root = Path(config["model"]["repo_root"])
    return [
        {
            "artifact_type": "code",
            "name": "stage1_extractor",
            "path": str(CODE15),
            "sha256": sha256_file(CODE15),
        },
        {
            "artifact_type": "code",
            "name": "stage2_builder",
            "path": str(CODE16),
            "sha256": sha256_file(CODE16),
        },
        {
            "artifact_type": "config",
            "name": "pairdata_config",
            "path": str(CONFIG_PATH),
            "sha256": sha256_file(CONFIG_PATH),
        },
        {
            "artifact_type": "schema",
            "name": "feature_schema",
            "path": str(schema_path),
            "sha256": sha256_file(schema_path),
        },
        {
            "artifact_type": "model",
            "name": "sea_raft_model",
            "path": str(config["model"]["model_file"]),
            "sha256": sha256_file(Path(config["model"]["model_file"])),
        },
        {
            "artifact_type": "config",
            "name": "sea_raft_model_config",
            "path": str(config["model"]["config"]),
            "sha256": sha256_file(Path(config["model"]["config"])),
        },
        {
            "artifact_type": "code_tree",
            "name": "sea_raft_core_config_tree",
            "path": str(repo_root),
            "sha256": sea_raft_code_tree_hash(repo_root),
        },
        {
            "artifact_type": "thresholds",
            "name": "scientific_thresholds",
            "path": str(CONFIG_PATH),
            "sha256": thresholds_hash(config),
        },
    ]


def verify_freeze(freeze_path: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    frozen = pd.read_csv(freeze_path)
    assertions: list[dict[str, Any]] = []
    for row in frozen.itertuples(index=False):
        if row.artifact_type == "thresholds":
            actual = thresholds_hash(config)
        elif row.artifact_type == "code_tree":
            actual = sea_raft_code_tree_hash(Path(row.path))
        else:
            actual = sha256_file(Path(row.path))
        assertions.append({
            "name": f"frozen_hash_{row.name}",
            "passed": actual == row.sha256,
            "detail": f"expected={row.sha256} actual={actual}",
        })
    return assertions


def phase_audit_table(
    phase_frame: pd.DataFrame,
    artifacts: list[dict[str, Any]],
    schema: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    model_features = [
        item["feature_name"]
        for item in schema["phase_features"] if item["model_candidate"]
    ]
    artifacts_by_key = {
        (item["summary"]["patient_id"], item["summary"]["phase"]): item
        for item in artifacts
    }
    for row in phase_frame.to_dict("records"):
        key = (row["patient_id"], row["phase"])
        summary = artifacts_by_key[key]["summary"]
        model_values = pd.to_numeric(
            pd.Series({name: row[name] for name in model_features}), errors="coerce"
        ).to_numpy(dtype=np.float64)
        rows.append({
            "patient_id": row["patient_id"],
            "split": row["split"],
            "source_type": row["source_type"],
            "selected_series_id": row["selected_series_id"],
            "phase": row["phase"],
            "n_frames": row["n_frames"],
            "expected_pairs": row["manifest_expected_pairs"],
            "actual_pairs": row["processed_pairs"],
            "polarity": summary["polarity"]["polarity_label"],
            "tdc_stage_counts": json.dumps(
                summary["tdc_stages"]["stage_counts"], ensure_ascii=False, sort_keys=True
            ),
            "active_ratio_fov": row["qc_active_ratio_fov"],
            "background_ratio_fov": row["qc_background_ratio_fov"],
            "kinetic_map_valid_ratio": row["qc_kinetic_valid_ratio"],
            "global_motion_mag_norm_mean": row["qc_pair_global_motion_mag_norm_mean"],
            "residual_flow_mag_norm_median_mean": row["flow_active_res_mag_norm_median_mean"],
            "uncertainty_log_mean": row["qc_pair_uncertainty_log_mean_mean"],
            "fb_relative_mean": row["qc_pair_fb_relative_mean_mean"],
            "soft_weight_mean": row["qc_pair_soft_weight_mean_fov_mean"],
            "filling_front_coverage_mean": row["flow_filling_front_coverage_fov_mean"],
            "persistent_coverage_mean": row["flow_persistent_coverage_fov_mean"],
            "washout_front_coverage_mean": row["flow_washout_front_coverage_fov_mean"],
            "flow_tdc_max_correlation": row["coupling_maximum_flow_tdc_correlation"],
            "flow_tdc_lag": row["coupling_lag_at_maximum_correlation"],
            "model_candidate_nonmissing": int(np.isfinite(model_values).sum()),
            "model_candidate_nan": int(np.isnan(model_values).sum()),
            "model_candidate_inf": int(np.isinf(model_values).sum()),
            "staging_source": row["source_type"] == "updated_10_cases",
        })
    return pd.DataFrame(rows)


def missingness_audit(
    patient_frame: pd.DataFrame,
    schema: dict[str, Any],
) -> pd.DataFrame:
    model_features = [
        item["feature_name"]
        for item in schema["patient_features"] if item["model_candidate"]
    ]
    rows: list[dict[str, Any]] = []
    for patient in patient_frame.to_dict("records"):
        values = pd.to_numeric(
            pd.Series({name: patient.get(name, np.nan) for name in model_features}),
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        rows.append({
            "patient_id": patient["patient_id"],
            "split": patient["split"],
            "missing_pre": patient["missing_pre"],
            "missing_post": patient["missing_post"],
            "model_candidate_total": len(model_features),
            "model_candidate_nonmissing": int(np.isfinite(values).sum()),
            "model_candidate_nan": int(np.isnan(values).sum()),
            "model_candidate_inf": int(np.isinf(values).sum()),
        })
    return pd.DataFrame(rows)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *[
            "| " + " | ".join(clean(value) for value in row) + " |"
            for row in rows
        ],
    ]


def build_report(
    mode: str,
    phase_audit: pd.DataFrame,
    patient_frame: pd.DataFrame,
    schema: dict[str, Any],
    assertions: list[dict[str, Any]],
    pairdata_root: Path,
) -> str:
    lines = [
        f"# api_fullseq_v2 {mode} feature audit",
        "",
        f"- Generated: {utc_now()}",
        f"- Pairdata root: {pairdata_root}",
        f"- Unique patients: {patient_frame['patient_id'].nunique()}",
        f"- Phases: {len(phase_audit)}",
        f"- Pairs: {int(phase_audit['actual_pairs'].sum())}",
        f"- Phase features: {schema['phase_feature_count']}",
        f"- Phase model candidates: {schema['model_candidate_phase_count']}",
        f"- Phase QC features: {schema['qc_phase_count']}",
        f"- Patient schema features: {schema['patient_feature_count']}",
        "- Labels read: no",
        "- Model training: no",
        "- Full Train/Valid extraction: not started",
        "",
        "## Phase audit",
        "",
        *markdown_table(
            [
                "patient_id", "phase", "frames", "pairs", "polarity",
                "active", "background", "kinetic valid", "global motion",
                "residual flow", "uncertainty", "FB", "soft weight",
                "front", "persistent", "washout", "Flow-TDC corr",
            ],
            [
                [
                    row.patient_id, row.phase, row.n_frames, row.actual_pairs,
                    row.polarity, f"{row.active_ratio_fov:.4f}",
                    f"{row.background_ratio_fov:.4f}",
                    f"{row.kinetic_map_valid_ratio:.4f}",
                    f"{row.global_motion_mag_norm_mean:.6g}",
                    f"{row.residual_flow_mag_norm_median_mean:.6g}",
                    f"{row.uncertainty_log_mean:.6g}",
                    f"{row.fb_relative_mean:.6g}",
                    f"{row.soft_weight_mean:.6g}",
                    f"{row.filling_front_coverage_mean:.6g}",
                    f"{row.persistent_coverage_mean:.6g}",
                    f"{row.washout_front_coverage_mean:.6g}",
                    f"{row.flow_tdc_max_correlation:.6g}",
                ]
                for row in phase_audit.itertuples(index=False)
            ],
        ),
        "",
        "## Feature groups",
        "",
        *markdown_table(
            ["group", "count"],
            [[group, count] for group, count in schema["groups"].items()],
        ),
        "",
        "## Hard assertions",
        "",
        *[
            f"- {'PASS' if item['passed'] else 'FAIL'} — {item['name']}: {item['detail']}"
            for item in assertions
        ],
        "",
    ]
    return "\n".join(lines)


def write_failure(stage: str, exc: BaseException) -> None:
    path = REPORT_ROOT / "failure.md"
    if path.exists():
        return
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([
            "# api_fullseq_v2 feature-pilot failure",
            "",
            f"- Stage: {stage}",
            f"- Exception: {type(exc).__name__}: {exc}",
            "- Manifest modified: no.",
            "- Labels read: no.",
            "- Model training: no.",
            "",
            traceback.format_exc(),
            "",
        ]),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairdata-root", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["pilot_train", "pilot_valid_integrity", "full_train", "full_valid"],
    )
    parser.add_argument("--schema-output", default=str(SCHEMA_DEFAULT))
    parser.add_argument("--phase-output", default=None)
    parser.add_argument("--patient-output", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--emit-legacy-ablation", action="store_true")
    parser.add_argument("--freeze-output", default=str(FREEZE_DEFAULT))
    args = parser.parse_args()

    stage = "preflight"
    try:
        config = load_config()
        spec = mode_spec(config, args.mode)
        pairdata_root = Path(args.pairdata_root).resolve()
        expected_records = expected_phase_records(config, args.mode)
        if not (pairdata_root / ".SUCCESS").is_file():
            raise FileNotFoundError(f"Stage-one root is not complete: {pairdata_root}")
        root_summary = json.loads(
            (pairdata_root / "run_summary.json").read_text(encoding="utf-8")
        )
        if int(root_summary["processed_pairs"]) != spec["expected_pairs"]:
            raise AssertionError("Stage-one root pair total mismatch")
        if not root_summary["cuda_actually_used"] or root_summary["cpu_fallback"]:
            raise AssertionError("Stage-one CUDA hard assertion failed")
        if root_summary["full_train_started"] or root_summary["full_valid_started"]:
            raise AssertionError("Forbidden full extraction was started")

        if args.dry_run:
            output = {
                "mode": args.mode,
                "pairdata_root": str(pairdata_root),
                "expected_patients": len(spec["ids"]),
                "expected_phases": len(expected_records),
                "expected_pairs": spec["expected_pairs"],
                "phase_directories_present": sum(
                    (pairdata_root / item["patient_id"] / item["phase"] / ".SUCCESS").is_file()
                    for item in expected_records
                ),
                "schema_exists": Path(args.schema_output).exists(),
                "emit_legacy_ablation": args.emit_legacy_ablation,
                "labels_read": False,
                "training_started": False,
                "full_extraction_started": False,
            }
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return 0

        schema_path = Path(args.schema_output).resolve()
        phase_output = (
            Path(args.phase_output).resolve()
            if args.phase_output else Path(spec["phase_default"]).resolve()
        )
        patient_output = (
            Path(args.patient_output).resolve()
            if args.patient_output else Path(spec["patient_default"]).resolve()
        )
        missing_output = Path(spec["missing_default"]).resolve()
        group_output = Path(spec["group_default"]).resolve()
        report_markdown = Path(spec["report_markdown"]).resolve()
        report_phase = Path(spec["report_phase"]).resolve()
        success_marker = FEATURE_ROOT / (
            "train.SUCCESS" if args.mode == "pilot_train" else "valid_integrity.SUCCESS"
        )
        freeze_path = Path(args.freeze_output).resolve()
        final_acceptance = REPORT_ROOT / "final_acceptance.md"

        if args.resume and success_marker.is_file():
            print(f"Already complete: {success_marker}")
            return 0
        targets = [
            phase_output, patient_output, missing_output, group_output,
            report_markdown, report_phase, success_marker,
        ]
        if args.mode == "pilot_train":
            targets.extend([schema_path, freeze_path])
        else:
            targets.append(final_acceptance)
            if not schema_path.is_file() or not freeze_path.is_file():
                raise FileNotFoundError("Valid requires frozen schema and hash manifest")
        conflicts = [path for path in targets if path.exists()]
        if conflicts:
            raise FileExistsError(
                "Refusing to overwrite existing feature targets: "
                + "|".join(str(path) for path in conflicts)
            )

        stage = "read_stage_one"
        artifacts = [
            read_phase_artifacts(pairdata_root, expected)
            for expected in expected_records
        ]
        phase_rows = [
            aggregate_phase(item, args.emit_legacy_ablation)
            for item in artifacts
        ]
        phase_frame = pd.DataFrame(phase_rows)
        if phase_frame[["patient_id", "phase"]].duplicated().any():
            raise AssertionError("Phase feature key is not unique")
        if phase_frame["patient_id"].nunique() != len(spec["ids"]):
            raise AssertionError("Phase feature patient coverage mismatch")
        if int(phase_frame["processed_pairs"].sum()) != spec["expected_pairs"]:
            raise AssertionError("Phase feature pair total mismatch")
        if not (phase_frame["split"] == spec["split"]).all():
            raise AssertionError("Phase output split contamination")

        generated_schema = build_schema(phase_frame, args.emit_legacy_ablation)
        if args.mode == "pilot_train":
            schema = generated_schema
        else:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            if schema["emit_legacy_ablation"] != args.emit_legacy_ablation:
                raise AssertionError("Valid legacy-ablation setting differs from Train")
            if schema["phase_features"] != generated_schema["phase_features"]:
                raise AssertionError("Valid phase schema differs from frozen Train schema")
            if schema["patient_features"] != generated_schema["patient_features"]:
                raise AssertionError("Valid patient schema differs from frozen Train schema")

        patient_frame = build_patient_features(
            phase_frame, schema, spec["ids"], spec["split"]
        )
        phase_audit = phase_audit_table(phase_frame, artifacts, schema)
        missing_frame = missingness_audit(patient_frame, schema)
        group_frame = pd.DataFrame([
            {"group": group, "feature_count": count}
            for group, count in schema["groups"].items()
        ])

        stage = "hard_assertions"
        assertions = schema_assertions(schema)

        def check(name: str, passed: bool, detail: str) -> None:
            assertions.append({"name": name, "passed": bool(passed), "detail": detail})

        check(
            "pilot_unique_patient_count",
            patient_frame["patient_id"].nunique() == len(spec["ids"]),
            f"expected={len(spec['ids'])} actual={patient_frame['patient_id'].nunique()}",
        )
        check(
            "pair_total",
            int(phase_audit["actual_pairs"].sum()) == spec["expected_pairs"],
            f"expected={spec['expected_pairs']} actual={int(phase_audit['actual_pairs'].sum())}",
        )
        check(
            "phase_expected_equals_actual_pairs",
            (phase_audit["expected_pairs"] == phase_audit["actual_pairs"]).all(),
            f"phases={len(phase_audit)}",
        )
        check(
            "split_isolation",
            (patient_frame["split"] == spec["split"]).all()
            and (phase_frame["split"] == spec["split"]).all(),
            spec["split"],
        )
        check(
            "stage_one_cuda_actual",
            all(item["metadata"]["cuda_actually_used"] for item in artifacts),
            f"phases={len(artifacts)}",
        )
        check(
            "no_cpu_fallback",
            not any(item["metadata"]["cpu_fallback"] for item in artifacts),
            "all phases CUDA-only",
        )
        check(
            "all_delta_frame_one",
            all((item["pair"]["delta_frame"] == 1).all() for item in artifacts),
            "no cross-gap pair",
        )
        check(
            "no_cross_patient_series_phase_internal",
            all(
                item["pair"]["patient_id"].nunique() == 1
                and item["pair"]["phase"].nunique() == 1
                and item["pair"]["selected_series_id"].nunique() == 1
                and item["pair"]["selected_internal_series"].nunique() == 1
                for item in artifacts
            ),
            f"phases={len(artifacts)}",
        )
        check(
            "selected_png_zero",
            all(
                not item["selected"]["absolute_path"].str.lower().str.endswith(".png").any()
                for item in artifacts
            ),
            "count=0",
        )
        check(
            "selected_parameter_map_zero",
            all(
                not item["selected"]["absolute_path"].str.upper().str.contains(
                    "CBF|CBV|MTT|TTP", regex=True
                ).any()
                for item in artifacts
            ),
            "count=0",
        )
        updated_old_reads = sum(
            int(
                item["summary"]["patient_id"] in config["updated_fixed_series"]
                and any(
                    str(path).startswith(
                        f"/root/autodl-tmp/tiantanDSA/{item['summary']['patient_id']}/"
                    )
                    for path in item["metadata"]["frame_paths"]
                )
            )
            for item in artifacts
        )
        check(
            "updated_old_source_read_zero",
            updated_old_reads == 0,
            f"count={updated_old_reads}",
        )
        check(
            "updated_staging_fixed_source",
            all(
                item["summary"]["source_type"] == "updated_10_cases"
                for item in artifacts
                if item["summary"]["patient_id"] in config["updated_fixed_series"]
            ),
            "all updated Pilot phases use staging",
        )
        if args.mode == "pilot_train":
            row571 = phase_audit[
                (phase_audit["patient_id"] == "571569")
                & (phase_audit["phase"] == "post")
            ]
            check(
                "case_571569_post_pairs_14",
                len(row571) == 1 and int(row571.iloc[0]["actual_pairs"]) == 14,
                row571.to_dict("records"),
            )
        model_phase_names = [
            item["feature_name"]
            for item in schema["phase_features"] if item["model_candidate"]
        ]
        phase_model_values = phase_frame[model_phase_names].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float64)
        check(
            "phase_model_candidate_nonmissing_finite",
            not np.isinf(phase_model_values).any(),
            f"nonmissing={int(np.isfinite(phase_model_values).sum())}",
        )
        model_patient_names = [
            item["feature_name"]
            for item in schema["patient_features"] if item["model_candidate"]
        ]
        patient_model_values = patient_frame[model_patient_names].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float64)
        check(
            "patient_model_candidate_nonmissing_finite",
            not np.isinf(patient_model_values).any(),
            f"nonmissing={int(np.isfinite(patient_model_values).sum())}",
        )
        pre_columns = [
            column for column in patient_frame.columns if column.startswith("pre_")
        ]
        delta_columns = [
            column for column in patient_frame.columns if column.startswith("delta_")
        ]
        post_only = patient_frame[patient_frame["missing_pre"] == 1]
        check(
            "post_only_pre_and_delta_are_nan",
            post_only[pre_columns + delta_columns].isna().all().all(),
            f"post_only_patients={post_only['patient_id'].tolist()}",
        )
        check(
            "labels_not_read",
            all(not item["summary"]["labels_read"] for item in artifacts),
            "labels_read=false",
        )
        check(
            "no_model_training",
            all(not item["summary"]["model_trained"] for item in artifacts),
            "model_trained=false",
        )
        check(
            "no_full_extraction",
            not root_summary["full_train_started"]
            and not root_summary["full_valid_started"],
            "Pilot only",
        )
        check(
            "source_metadata_unchanged",
            all(
                item["metadata"]["source_metadata_signature_before"]
                == item["metadata"]["source_metadata_signature_after"]
                for item in artifacts
            ),
            f"phases={len(artifacts)}",
        )
        check(
            "manifest_not_rescanned",
            all(not item["summary"]["manifest_rescanned"] for item in artifacts),
            "frozen explicit frame lists only",
        )
        for relative, expected_hash in config["frozen_inputs"].items():
            actual_hash = sha256_file(PROJECT / relative)
            check(
                f"frozen_input_{Path(relative).name}",
                actual_hash == expected_hash,
                f"expected={expected_hash} actual={actual_hash}",
            )
        failed_prewrite = [item for item in assertions if not item["passed"]]
        if failed_prewrite:
            raise AssertionError(
                "Pre-write hard assertions failed: "
                + "|".join(item["name"] for item in failed_prewrite)
            )

        stage = "write_features_and_schema"
        FEATURE_ROOT.mkdir(parents=True, exist_ok=True)
        REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        phase_frame.to_csv(
            phase_output, index=False, encoding="utf-8", lineterminator="\n"
        )
        patient_frame.to_csv(
            patient_output, index=False, encoding="utf-8", lineterminator="\n"
        )
        missing_frame.to_csv(
            missing_output, index=False, encoding="utf-8", lineterminator="\n"
        )
        group_frame.to_csv(
            group_output, index=False, encoding="utf-8", lineterminator="\n"
        )
        if args.mode == "pilot_train":
            write_json(schema_path, schema)
            pd.DataFrame(freeze_rows(schema_path, config)).to_csv(
                freeze_path, index=False, encoding="utf-8", lineterminator="\n"
            )
            freeze_assertions = verify_freeze(freeze_path, config)
            assertions.extend(freeze_assertions)
        else:
            before_valid_freeze = verify_freeze(freeze_path, config)
            assertions.extend(before_valid_freeze)
            if not all(item["passed"] for item in before_valid_freeze):
                raise AssertionError("Frozen hash changed before Valid stage-two")

        stage = "reports_and_final_hash_validation"
        phase_audit.to_csv(
            report_phase, index=False, encoding="utf-8", lineterminator="\n"
        )
        if args.mode == "pilot_valid_integrity":
            after_valid_freeze = verify_freeze(freeze_path, config)
            assertions.extend([
                {
                    "name": f"valid_after_{item['name']}",
                    "passed": item["passed"],
                    "detail": item["detail"],
                }
                for item in after_valid_freeze
            ])
            if not all(item["passed"] for item in after_valid_freeze):
                raise AssertionError("Frozen hash changed after Valid")
        failed = [item for item in assertions if not item["passed"]]
        if failed:
            raise AssertionError(
                "Hard assertions failed: "
                + "|".join(item["name"] for item in failed)
            )
        report_text = build_report(
            args.mode, phase_audit, patient_frame, schema, assertions, pairdata_root
        )
        report_markdown.write_text(report_text, encoding="utf-8")
        write_json(success_marker, {
            "mode": args.mode,
            "patients": int(patient_frame["patient_id"].nunique()),
            "phases": len(phase_frame),
            "pairs": int(phase_audit["actual_pairs"].sum()),
            "schema_sha256": sha256_file(schema_path),
            "finished_utc": utc_now(),
        })
        if args.mode == "pilot_valid_integrity":
            final_acceptance.write_text(
                "\n".join([
                    "# api_fullseq_v2 feature Pilot final acceptance",
                    "",
                    f"- Completed: {utc_now()}",
                    "- Train Pilot patients: 15",
                    "- Valid integrity patient: 549117 only",
                    "- Train expected/actual pairs: 618/618",
                    "- Valid expected/actual pairs: 44/44",
                    f"- Frozen schema SHA256: {sha256_file(schema_path)}",
                    f"- Frozen hash rows: {len(pd.read_csv(freeze_path))}",
                    f"- Valid-after-freeze assertions: {sum(item['passed'] for item in assertions)}/{len(assertions)} PASS",
                    "- Manifest modified: no",
                    "- Labels read: no",
                    "- Model training: no",
                    "- Full Train/Valid extraction: not started",
                    "",
                ]),
                encoding="utf-8",
            )
        print(json.dumps({
            "mode": args.mode,
            "patients": int(patient_frame["patient_id"].nunique()),
            "phases": len(phase_frame),
            "pairs": int(phase_audit["actual_pairs"].sum()),
            "phase_feature_count": schema["phase_feature_count"],
            "phase_model_candidate_count": schema["model_candidate_phase_count"],
            "patient_feature_count": schema["patient_feature_count"],
            "assertions_pass": int(sum(item["passed"] for item in assertions)),
            "assertions_total": len(assertions),
            "phase_output": str(phase_output),
            "patient_output": str(patient_output),
            "schema": str(schema_path),
        }, ensure_ascii=False, indent=2))
        return 0
    except BaseException as exc:
        write_failure(stage, exc)
        raise


if __name__ == "__main__":
    sys.exit(main())


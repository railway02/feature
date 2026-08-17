#!/usr/bin/env python3
"""Extract api_fullseq_v2 Full Train then Full Valid pairdata with frozen science."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

if not os.environ.get("OMP_NUM_THREADS", "").isdigit():
    os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


PROJECT = Path("/root/autodl-tmp/aneurysm")
CONFIG_DEFAULT = PROJECT / "configs/api_fullseq_v2_full_train_valid_config.json"
PILOT_MANIFEST = PROJECT / "manifests/api_fullseq_v2_pilot_manifest.csv"
FULL_TRAIN_MANIFEST = PROJECT / "manifests/api_fullseq_v2_train_manifest.csv"
FULL_VALID_MANIFEST = PROJECT / "manifests/api_fullseq_v2_valid_manifest.csv"
FRAME_AUDIT = PROJECT / "manifests/api_fullseq_v2_frame_audit.csv"
REPORT_ROOT = PROJECT / "reports/api_fullseq_v2_feature_full"
LOG_ROOT = PROJECT / "logs"
TRAIN_OUTPUT_DEFAULT = PROJECT / "outputs/api_fullseq_v2_pairdata/full/train"
VALID_OUTPUT_DEFAULT = PROJECT / "outputs/api_fullseq_v2_pairdata/full/valid"
FROZEN_HASH_MANIFEST = PROJECT / "manifests/api_fullseq_v2_gpu_pilot_frozen.csv"
TRAIN_SUCCESS_MARKER = REPORT_ROOT / ".FULL_TRAIN_SUCCESS"
CACHE_SIZE = (64, 64)
JPEG_SUFFIXES = {".jpg", ".jpeg"}
PARAMETER_TOKENS = ("CBF", "CBV", "MTT", "TTP")
LEGACY_PAIR_COLUMNS = [
    "mag_mean", "mag_median", "mag_std", "mag_p90", "mag_p95", "mag_max",
    "mag_norm_mean", "mag_norm_p90", "u_mean", "u_std", "v_mean", "v_std",
    "direction_entropy", "uncertainty_mean", "uncertainty_std", "uncertainty_p90",
]
REGIONS = ("active", "filling_front", "persistent", "washout_front")


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


def hash_lines(values: Iterable[str]) -> str:
    material = "\n".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(material).hexdigest() if material else ""


def parse_pipe_strings(value: Any) -> list[str]:
    if value is None or pd.isna(value) or str(value) == "":
        return []
    return [part for part in str(value).split("|") if part != ""]


def parse_pipe_ints(value: Any) -> list[int]:
    return [int(part) for part in parse_pipe_strings(value)]


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
    if value is pd.NA:
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(sanitize_json(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    write_json(temporary, value)
    os.replace(temporary, path)


def source_metadata_signature(paths: list[Path]) -> str:
    rows: list[str] = []
    for path in paths:
        stat_result = path.stat()
        rows.append(f"{path}\t{stat_result.st_size}\t{stat_result.st_mtime_ns}")
    return hash_lines(rows)


def dataframe_numeric_audit(frame: pd.DataFrame) -> dict[str, Any]:
    nonfinite: dict[str, int] = {}
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
            bad = np.isinf(values).sum()
            if bad:
                nonfinite[column] = int(bad)
    return {
        "rows": len(frame),
        "numeric_columns": int(sum(pd.api.types.is_numeric_dtype(frame[column]) for column in frame.columns)),
        "inf_count": int(sum(nonfinite.values())),
        "inf_by_column": nonfinite,
    }


def verify_frozen_inputs(config: dict[str, Any]) -> None:
    failures: list[str] = []
    for relative, expected in config["frozen_inputs"].items():
        path = PROJECT / relative
        actual = sha256_file(path)
        if actual != expected:
            failures.append(f"{relative}: expected={expected} actual={actual}")
    model_path = Path(config["model"]["model_file"])
    model_hash = sha256_file(model_path)
    if model_hash != config["model"]["model_sha256"]:
        failures.append(
            f"model: expected={config['model']['model_sha256']} actual={model_hash}"
        )
    model_config_path = Path(config["model"]["config"])
    model_config_hash = sha256_file(model_config_path)
    if model_config_hash != config["model"]["model_config_sha256"]:
        failures.append(
            f"model_config: expected={config['model']['model_config_sha256']} actual={model_config_hash}"
        )
    if failures:
        raise AssertionError("Frozen input hash failure:\n" + "\n".join(failures))


def verify_valid_frozen_hashes() -> None:
    if not FROZEN_HASH_MANIFEST.is_file():
        raise FileNotFoundError(
            f"Valid integrity requires frozen hash manifest: {FROZEN_HASH_MANIFEST}"
        )
    frozen = pd.read_csv(FROZEN_HASH_MANIFEST)
    failures: list[str] = []
    for row in frozen.itertuples(index=False):
        if row.artifact_type not in {"code", "config", "schema", "model", "thresholds", "code_tree"}:
            continue
        if row.artifact_type == "thresholds":
            config = json.loads(CONFIG_DEFAULT.read_text(encoding="utf-8"))
            actual = sha256_json({
                "normalization": config["normalization"],
                "fov": config["fov"],
                "activity": config["activity"],
                "kinetics": config["kinetics"],
                "flow_qc": config["flow_qc"],
                "aggregation": config["aggregation"],
            })
        elif row.artifact_type == "code_tree":
            actual = sea_raft_code_tree_hash(Path(row.path))
        else:
            actual = sha256_file(Path(row.path))
        if actual != row.sha256:
            failures.append(
                f"{row.artifact_type}:{row.path}: expected={row.sha256} actual={actual}"
            )
    if failures:
        raise AssertionError("Frozen hash changed before Valid:\n" + "\n".join(failures))


class RunLogger:
    def __init__(self, mode: str) -> None:
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = LOG_ROOT / f"api_fullseq_v2_full_train_valid_{mode}_{stamp}_{os.getpid()}.log"
        self.handle = self.path.open("x", encoding="utf-8")

    def log(self, message: str) -> None:
        line = f"[{utc_now()}] {message}"
        print(line, flush=True)
        self.handle.write(line + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["execution_limits"]["labels_forbidden"] is not True:
        raise AssertionError("labels_forbidden must remain true")
    if config["execution_limits"]["training_forbidden"] is not True:
        raise AssertionError("training_forbidden must remain true")
    return config


def selected_ids_for_mode(
    config: dict[str, Any], mode: str, manifest_path: Path
) -> tuple[list[str], str]:
    if mode == "pilot_train":
        return list(config["pilot"]["train_patient_ids"]), "Train"
    if mode == "pilot_valid_integrity":
        return list(config["pilot"]["valid_integrity_patient_ids"]), "Valid"
    if mode == "full_train":
        if config["execution_limits"]["full_train_forbidden_for_current_task"]:
            raise RuntimeError("full_train is forbidden by the Full config")
        if manifest_path.resolve() != FULL_TRAIN_MANIFEST.resolve():
            raise AssertionError("Full Train requires the frozen Full Train manifest")
        manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})
        return manifest["patient_id"].tolist(), "Train"
    if mode == "full_valid":
        if config["execution_limits"]["full_valid_forbidden_for_current_task"]:
            raise RuntimeError("Full Valid is forbidden for the current task")
        if manifest_path.resolve() != FULL_VALID_MANIFEST.resolve():
            raise AssertionError("Full Valid requires the authoritative Valid manifest")
        manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})
        return manifest["patient_id"].tolist(), "Valid"
    raise ValueError(f"Unsupported mode: {mode}")


def parse_requested_patient_ids(values: list[str] | None) -> list[str]:
    if not values:
        return []
    result: list[str] = []
    for value in values:
        result.extend(part.strip() for part in value.split(",") if part.strip())
    return result


def manifest_phase_plans(
    manifest_path: Path,
    config: dict[str, Any],
    mode: str,
    requested_ids: list[str],
    phase_filter: str | None,
    limit: int | None,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    allowed_ids, expected_split = selected_ids_for_mode(config, mode, manifest_path)
    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})
    required_columns = {
        "patient_id", "split", "source_type", "source_medical_record_root",
        "selected_series_id", "selected_series_path",
        "selected_pre_internal_series", "selected_post_internal_series",
        "pre_frame_indices", "post_frame_indices", "pre_frame_paths", "post_frame_paths",
        "pre_frame_list_hash", "post_frame_list_hash",
        "n_pre_contiguous_pairs", "n_post_contiguous_pairs",
        "can_run_pre", "can_run_post", "patient_status",
    }
    missing = required_columns - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    manifest = manifest[manifest["patient_id"].isin(allowed_ids)].copy()
    if set(manifest["patient_id"]) != set(allowed_ids):
        raise AssertionError(
            f"Mode IDs missing from manifest: {sorted(set(allowed_ids) - set(manifest['patient_id']))}"
        )
    if manifest["patient_id"].duplicated().any():
        raise AssertionError("Selected manifest patient_id is not unique")
    if not (manifest["split"] == expected_split).all():
        raise AssertionError(f"Split isolation failure for {mode}")
    order = {patient_id: index for index, patient_id in enumerate(allowed_ids)}
    manifest["_order"] = manifest["patient_id"].map(order)
    manifest = manifest.sort_values("_order").drop(columns="_order")

    if requested_ids:
        unknown = set(requested_ids) - set(allowed_ids)
        if unknown:
            raise ValueError(f"Requested patient IDs outside mode: {sorted(unknown)}")
        manifest = manifest[manifest["patient_id"].isin(requested_ids)].copy()
        requested_order = {patient_id: index for index, patient_id in enumerate(requested_ids)}
        manifest["_order"] = manifest["patient_id"].map(requested_order)
        manifest = manifest.sort_values("_order").drop(columns="_order")
    if limit is not None:
        manifest = manifest.head(limit)

    plans: list[dict[str, Any]] = []
    fixed_mapping = config["updated_fixed_series"]
    for row in manifest.to_dict("records"):
        patient_id = str(row["patient_id"])
        if patient_id in fixed_mapping:
            expected_root = PROJECT / "staging/updated_10_cases" / patient_id
            if row["source_type"] != "updated_10_cases":
                raise AssertionError(f"{patient_id}: updated source_type changed")
            if Path(row["source_medical_record_root"]).resolve() != expected_root.resolve():
                raise AssertionError(f"{patient_id}: updated source root changed")
            if str(row["selected_series_id"]) != fixed_mapping[patient_id]:
                raise AssertionError(f"{patient_id}: fixed selected series changed")
            if str(row["selected_series_path"]).startswith(
                f"/root/autodl-tmp/tiantanDSA/{patient_id}"
            ):
                raise AssertionError(f"{patient_id}: old source fallback detected")

        for phase in ("pre", "post"):
            if phase_filter and phase != phase_filter:
                continue
            can_run = bool(row[f"can_run_{phase}"])
            paths = [Path(value) for value in parse_pipe_strings(row[f"{phase}_frame_paths"])]
            indices = parse_pipe_ints(row[f"{phase}_frame_indices"])
            if not can_run:
                if paths or indices:
                    raise AssertionError(f"{patient_id} {phase}: non-runnable phase has frames")
                continue
            if len(paths) != len(indices) or len(paths) < 2:
                raise AssertionError(f"{patient_id} {phase}: invalid frozen frame list")
            actual_hash = hash_lines(str(path) for path in paths)
            if actual_hash != str(row[f"{phase}_frame_list_hash"]):
                raise AssertionError(f"{patient_id} {phase}: frame-list hash mismatch")
            pairs = [
                (position, position + 1)
                for position in range(len(indices) - 1)
                if indices[position + 1] - indices[position] == 1
            ]
            expected_pairs = int(row[f"n_{phase}_contiguous_pairs"])
            if len(pairs) != expected_pairs:
                raise AssertionError(
                    f"{patient_id} {phase}: expected {expected_pairs}, reconstructed {len(pairs)}"
                )
            for path in paths:
                if path.suffix.casefold() not in JPEG_SUFFIXES:
                    raise AssertionError(f"Selected non-JPEG path: {path}")
                upper = path.name.upper()
                if any(token in upper for token in PARAMETER_TOKENS):
                    raise AssertionError(f"Selected parameter-map path: {path}")
                if not path.is_file():
                    raise FileNotFoundError(path)
            plans.append({
                "patient_id": patient_id,
                "split": row["split"],
                "source_type": row["source_type"],
                "source_medical_record_root": row["source_medical_record_root"],
                "selected_series_id": row["selected_series_id"],
                "selected_series_path": row["selected_series_path"],
                "phase": phase,
                "selected_internal_series": row[f"selected_{phase}_internal_series"],
                "frame_paths": paths,
                "frame_indices": indices,
                "frame_list_hash": row[f"{phase}_frame_list_hash"],
                "manifest_expected_pairs": expected_pairs,
                "pairs": pairs,
                "patient_status": row["patient_status"],
            })
    return plans, manifest


def dry_run_summary(
    plans: list[dict[str, Any]],
    manifest: pd.DataFrame,
    config: dict[str, Any],
    mode: str,
    filtered: bool,
) -> dict[str, Any]:
    pair_total = sum(len(plan["pairs"]) for plan in plans)
    phase_rows = [
        {
            "patient_id": plan["patient_id"],
            "split": plan["split"],
            "phase": plan["phase"],
            "n_frames": len(plan["frame_paths"]),
            "manifest_expected_pairs": plan["manifest_expected_pairs"],
            "reconstructed_pairs": len(plan["pairs"]),
            "source_type": plan["source_type"],
            "selected_series_id": plan["selected_series_id"],
        }
        for plan in plans
    ]
    if not filtered:
        if mode == "pilot_train":
            expected = config["pilot"]["expected_train_pairs"]
        elif mode == "pilot_valid_integrity":
            expected = config["pilot"]["expected_valid_integrity_pairs"]
        elif mode == "full_train":
            expected = config["full"]["expected_train_pairs"]
        elif mode == "full_valid":
            expected = config["full"]["expected_valid_pairs"]
        else:
            raise RuntimeError(f"Unsupported full mode: {mode}")
        if pair_total != expected:
            raise AssertionError(f"{mode}: expected {expected} pairs, found {pair_total}")
    if mode == "pilot_train" and not filtered:
        if manifest["patient_id"].nunique() != 15:
            raise AssertionError("Train Pilot must contain 15 unique patient IDs")
    if mode == "pilot_valid_integrity" and not filtered:
        if manifest["patient_id"].tolist() != ["549117"]:
            raise AssertionError("Valid integrity must contain only 549117")
    if mode == "full_train" and not filtered:
        expected_patients = int(config["full"]["expected_train_patients"])
        expected_phases = int(config["full"]["expected_train_phases"])
        if manifest["patient_id"].nunique() != expected_patients:
            raise AssertionError(
                f"Full Train patient count expected {expected_patients}, "
                f"found {manifest['patient_id'].nunique()}"
            )
        if len(plans) != expected_phases:
            raise AssertionError(
                f"Full Train phase count expected {expected_phases}, found {len(plans)}"
            )
        if int(manifest["can_run_prepost"].sum()) != int(
            config["full"]["expected_train_prepost_patients"]
        ):
            raise AssertionError("Full Train Pre+Post count changed")
        if int((~manifest["can_run_pre"] & manifest["can_run_post"]).sum()) != int(
            config["full"]["expected_train_post_only_patients"]
        ):
            raise AssertionError("Full Train Post-only count changed")
    if mode == "full_valid" and not filtered:
        expected_patients = int(config["full"]["expected_valid_patients"])
        expected_phases = int(config["full"]["expected_valid_phases"])
        if manifest["patient_id"].nunique() != expected_patients:
            raise AssertionError(
                f"Full Valid patient count expected {expected_patients}, "
                f"found {manifest['patient_id'].nunique()}"
            )
        if len(plans) != expected_phases:
            raise AssertionError(
                f"Full Valid phase count expected {expected_phases}, found {len(plans)}"
            )
        if int(manifest["can_run_prepost"].sum()) != int(
            config["full"]["expected_valid_prepost_patients"]
        ):
            raise AssertionError("Full Valid Pre+Post count changed")
        if int((~manifest["can_run_pre"] & manifest["can_run_post"]).sum()) != int(
            config["full"]["expected_valid_post_only_patients"]
        ):
            raise AssertionError("Full Valid Post-only count changed")
    prepost = int(manifest["can_run_prepost"].sum())
    post_only = int((~manifest["can_run_pre"] & manifest["can_run_post"]).sum())
    return {
        "mode": mode,
        "unique_patients": int(manifest["patient_id"].nunique()),
        "phases": len(plans),
        "pairs": pair_total,
        "prepost_patients": prepost,
        "post_only_patients": post_only,
        "phase_rows": phase_rows,
        "manifest_rescanned": False,
        "labels_read": False,
        "full_extraction_started": False,
    }


def read_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"OpenCV could not read frozen frame: {path}")
    return image


def load_phase_frames(paths: list[Path], num_workers: int) -> np.ndarray:
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
        frames = list(executor.map(read_grayscale, paths))
    shapes = {frame.shape for frame in frames}
    if len(shapes) != 1:
        raise AssertionError(f"Frozen phase frames have mixed dimensions: {shapes}")
    return np.stack(frames, axis=0)


def normalize_phase(frames: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, float, float]:
    low = float(np.percentile(frames, config["normalization"]["phase_low_percentile"]))
    high = float(np.percentile(frames, config["normalization"]["phase_high_percentile"]))
    scale = max(high - low, 1e-6)
    normalized = np.clip(frames.astype(np.float32), low, high)
    normalized = (normalized - low) / scale
    return normalized.astype(np.float32), low, high


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return mask.astype(bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def build_fov_mask(normalized: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    median_image = np.median(normalized, axis=0)
    fov_config = config["fov"]
    mask = (
        (median_image > float(fov_config["low_normalized_threshold"]))
        & (median_image < float(fov_config["high_normalized_threshold"]))
    )
    height, width = mask.shape
    border_y = max(1, int(round(height * float(fov_config["border_fraction"]))))
    border_x = max(1, int(round(width * float(fov_config["border_fraction"]))))
    border_mask = np.zeros_like(mask)
    border_mask[border_y:height - border_y, border_x:width - border_x] = True
    mask &= border_mask
    kernel_size = int(fov_config["morphology_kernel"])
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
    fallback = False
    if fov_config["largest_component"]:
        mask = largest_component(mask)
    if int(mask.sum()) < int(0.1 * height * width):
        mask = border_mask
        fallback = True
    return mask, {
        "fov_pixels": int(mask.sum()),
        "fov_ratio": float(mask.mean()),
        "fov_border_y": border_y,
        "fov_border_x": border_x,
        "fov_fallback_border_only": fallback,
    }


def baseline_polarity_enhancement(
    normalized: np.ndarray,
    fov_mask: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    baseline_count = min(
        int(config["normalization"]["baseline_frame_count"]), normalized.shape[0]
    )
    baseline = np.median(normalized[:baseline_count], axis=0)
    delta = normalized - baseline[None]
    values = delta[:, fov_mask]
    positive_score = float(np.percentile(values, 99))
    negative_score = float(np.percentile(-values, 99))
    polarity = 1.0 if positive_score >= negative_score else -1.0
    enhancement = np.maximum(polarity * delta, 0.0).astype(np.float32)
    return baseline.astype(np.float32), enhancement, {
        "polarity": polarity,
        "polarity_label": "brightening" if polarity > 0 else "darkening",
        "positive_score": positive_score,
        "negative_score": negative_score,
        "baseline_frame_count": baseline_count,
    }


def robust_mad(values: np.ndarray) -> tuple[float, float]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, mad


def build_activity_masks(
    enhancement: np.ndarray,
    fov_mask: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray]:
    activity = np.percentile(
        enhancement, config["activity"]["activity_temporal_percentile"], axis=0
    ).astype(np.float32)
    fov_values = activity[fov_mask]
    median, mad = robust_mad(fov_values)
    active_threshold = max(
        float(np.percentile(fov_values, config["activity"]["active_percentile"])),
        median + float(config["activity"]["active_mad_multiplier"]) * mad,
    )
    high_threshold = max(
        float(np.percentile(fov_values, config["activity"]["high_activity_percentile"])),
        median + float(config["activity"]["high_activity_mad_multiplier"]) * mad,
    )
    background_threshold = min(
        float(np.percentile(fov_values, config["activity"]["background_percentile"])),
        median + float(config["activity"]["background_mad_multiplier"]) * mad,
    )
    active = fov_mask & (activity >= active_threshold)
    high = fov_mask & (activity >= high_threshold)
    background = fov_mask & (activity <= background_threshold)
    active = cv2.morphologyEx(active.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    high = cv2.morphologyEx(high.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    if int(active.sum()) < int(config["activity"]["minimum_active_pixels"]):
        raise AssertionError(f"Activity ROI too small: {int(active.sum())} pixels")
    background_fallback = False
    if int(background.sum()) < int(config["activity"]["minimum_background_pixels"]):
        background = fov_mask & ~high
        background_fallback = True
    if int(background.sum()) < int(config["activity"]["minimum_background_pixels"]):
        raise AssertionError(f"Background ROI too small: {int(background.sum())} pixels")
    return {
        "active": active,
        "high_activity": high,
        "background": background,
        "fov": fov_mask,
    }, {
        "activity_median": median,
        "activity_mad": mad,
        "active_threshold": active_threshold,
        "high_activity_threshold": high_threshold,
        "background_threshold": background_threshold,
        "active_pixels": int(active.sum()),
        "high_activity_pixels": int(high.sum()),
        "background_pixels": int(background.sum()),
        "active_ratio_fov": float(active.sum() / max(fov_mask.sum(), 1)),
        "high_activity_ratio_fov": float(high.sum() / max(fov_mask.sum(), 1)),
        "background_ratio_fov": float(background.sum() / max(fov_mask.sum(), 1)),
        "background_fallback": background_fallback,
    }, activity


def trimmed_mean(values: np.ndarray, trim_fraction: float) -> float:
    if values.size == 0:
        return float("nan")
    sorted_values = np.sort(values)
    trim = int(math.floor(values.size * trim_fraction))
    if trim * 2 >= values.size:
        return float(np.mean(sorted_values))
    return float(np.mean(sorted_values[trim:values.size - trim]))


def first_crossing(curve: np.ndarray, threshold: float) -> int | None:
    positions = np.flatnonzero(curve >= threshold)
    return int(positions[0]) if len(positions) else None


def normalized_times(indices: list[int]) -> np.ndarray:
    values = np.asarray(indices, dtype=np.float32)
    span = max(float(values[-1] - values[0]), 1.0)
    return (values - values[0]) / span


def classify_frame_stages(
    curve: np.ndarray,
    onset_position: int,
    peak_position: int,
    peak_fraction: float,
) -> list[str]:
    peak_value = max(float(curve[peak_position]), 1e-8)
    stages: list[str] = []
    for position, value in enumerate(curve):
        if position < onset_position:
            stages.append("precontrast")
        elif value >= peak_fraction * peak_value:
            stages.append("peak")
        elif position <= peak_position:
            stages.append("washin")
        else:
            stages.append("washout")
    return stages


def tdc_and_stage_features(
    enhancement: np.ndarray,
    indices: list[int],
    masks: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    times = normalized_times(indices)
    active = masks["active"]
    high = masks["high_activity"]
    trim_fraction = float(config["kinetics"]["trim_fraction"])
    active_median = np.asarray(
        [np.median(frame[active]) for frame in enhancement], dtype=np.float32
    )
    active_trimmed = np.asarray(
        [trimmed_mean(frame[active], trim_fraction) for frame in enhancement], dtype=np.float32
    )
    high_median = np.asarray(
        [np.median(frame[high]) if high.any() else np.nan for frame in enhancement],
        dtype=np.float32,
    )
    curve = active_median
    peak_position = int(np.nanargmax(curve))
    peak = float(curve[peak_position])
    onset10 = first_crossing(curve, 0.1 * peak)
    onset50 = first_crossing(curve, 0.5 * peak)
    onset_position = onset10 if onset10 is not None else 0
    after_peak = np.arange(len(curve)) >= peak_position
    half_after = np.flatnonzero(after_peak & (curve <= 0.5 * peak))
    washout_half_position = int(half_after[0]) if len(half_after) else None
    half_mask = curve >= 0.5 * peak
    half_positions = np.flatnonzero(half_mask)
    fwhm = (
        float(times[half_positions[-1]] - times[half_positions[0]])
        if len(half_positions) else float("nan")
    )
    slopes = np.diff(curve) / np.maximum(np.diff(times), 1e-6)
    peak_count = int(
        sum(
            curve[i] > curve[i - 1] and curve[i] >= curve[i + 1]
            for i in range(1, len(curve) - 1)
        )
    )
    stages = classify_frame_stages(
        curve,
        onset_position,
        peak_position,
        float(config["kinetics"]["peak_stage_fraction"]),
    )
    frame_table = pd.DataFrame({
        "sequence_position": np.arange(len(indices), dtype=int),
        "frame_index": indices,
        "normalized_time": times,
        "tdc_active_median": active_median,
        "tdc_active_trimmed_mean": active_trimmed,
        "tdc_high_activity_median": high_median,
        "tdc_derivative": np.r_[slopes, np.nan],
        "stage": stages,
    })
    early = curve[times < 1 / 3]
    middle = curve[(times >= 1 / 3) & (times < 2 / 3)]
    late = curve[times >= 2 / 3]
    features = {
        "tdc_peak": peak,
        "tdc_onset10": float(times[onset10]) if onset10 is not None else float("nan"),
        "tdc_onset50": float(times[onset50]) if onset50 is not None else float("nan"),
        "tdc_time_to_peak": float(times[peak_position]),
        "tdc_rise_duration": (
            float(times[peak_position] - times[onset_position])
            if peak_position >= onset_position else 0.0
        ),
        "tdc_washout_half_time": (
            float(times[washout_half_position])
            if washout_half_position is not None else float("nan")
        ),
        "tdc_fwhm": fwhm,
        "tdc_normalized_auc": float(np.trapz(curve, times)),
        "tdc_max_up_slope": float(np.max(slopes)) if len(slopes) else float("nan"),
        "tdc_max_down_slope": float(np.min(slopes)) if len(slopes) else float("nan"),
        "tdc_early_mean": float(np.mean(early)) if len(early) else float("nan"),
        "tdc_middle_mean": float(np.mean(middle)) if len(middle) else float("nan"),
        "tdc_late_mean": float(np.mean(late)) if len(late) else float("nan"),
        "tdc_temporal_variation": float(np.std(curve)),
        "tdc_local_peak_count": peak_count,
        "tdc_baseline_contamination": float(
            np.mean(curve[: min(3, len(curve))]) / max(peak, 1e-8)
        ),
        "tdc_peak_frame_index": int(indices[peak_position]),
        "tdc_onset_frame_index": int(indices[onset_position]),
        "tdc_washout_present": washout_half_position is not None,
        "stage_counts": dict(pd.Series(stages).value_counts()),
    }
    return frame_table, features, stages


def summarize_map(values: np.ndarray, valid_mask: np.ndarray, active_pixels: int) -> dict[str, float]:
    selected = values[valid_mask]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return {
            "valid_ratio": 0.0,
            "median": float("nan"),
            "iqr": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
            "spatial_std": float("nan"),
            "spatial_heterogeneity": float("nan"),
        }
    median = float(np.median(selected))
    p25, p75 = np.percentile(selected, [25, 75])
    iqr = float(p75 - p25)
    return {
        "valid_ratio": float(selected.size / max(active_pixels, 1)),
        "median": median,
        "iqr": iqr,
        "p10": float(np.percentile(selected, 10)),
        "p90": float(np.percentile(selected, 90)),
        "spatial_std": float(np.std(selected)),
        "spatial_heterogeneity": float(iqr / max(abs(median), 1e-6)),
    }


def build_kinetic_maps(
    enhancement: np.ndarray,
    indices: list[int],
    active_mask: np.ndarray,
    tdc_peak: float,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray]:
    frame_count, height, width = enhancement.shape
    times = normalized_times(indices)
    flat_positions = np.flatnonzero(active_mask.ravel())
    values = enhancement.reshape(frame_count, -1)[:, flat_positions]
    peaks = np.max(values, axis=0)
    minimum_peak = max(
        float(tdc_peak) * float(config["kinetics"]["pixel_peak_min_fraction_of_global"]),
        1e-6,
    )
    valid = peaks >= minimum_peak
    peak_positions = np.argmax(values, axis=0)
    threshold10 = 0.1 * peaks
    threshold50 = 0.5 * peaks
    above10 = values >= threshold10[None]
    above50 = values >= threshold50[None]
    toa10_positions = np.argmax(above10, axis=0)
    toa50_positions = np.argmax(above50, axis=0)
    auc = np.trapz(values, times, axis=0)
    ttp = times[peak_positions]
    toa10 = times[toa10_positions]
    toa50 = times[toa50_positions]
    rise_duration = np.maximum(ttp - toa10, 1e-6)
    rise_slope = peaks / rise_duration
    last50_positions = frame_count - 1 - np.argmax(above50[::-1], axis=0)
    fwhm = times[last50_positions] - times[toa50_positions]
    slopes = np.diff(values, axis=0) / np.maximum(np.diff(times)[:, None], 1e-6)
    washout_slope = np.full(values.shape[1], np.nan, dtype=np.float32)
    for position in range(values.shape[1]):
        after = slopes[peak_positions[position]:, position]
        if after.size:
            washout_slope[position] = float(np.min(after))

    map_vectors = {
        "toa10": toa10,
        "toa50": toa50,
        "ttp": ttp,
        "peak": peaks,
        "normalized_auc": auc,
        "rise_slope": rise_slope,
        "washout_slope": washout_slope,
        "fwhm": fwhm,
    }
    maps: dict[str, np.ndarray] = {}
    features: dict[str, Any] = {}
    active_pixels = int(active_mask.sum())
    valid_full = np.zeros(height * width, dtype=bool)
    valid_full[flat_positions] = valid
    valid_map = valid_full.reshape(height, width)
    for name, vector in map_vectors.items():
        full = np.full(height * width, np.nan, dtype=np.float32)
        safe_vector = np.asarray(vector, dtype=np.float32)
        safe_vector[~valid] = np.nan
        full[flat_positions] = safe_vector
        map_image = full.reshape(height, width)
        maps[name] = map_image
        summary = summarize_map(map_image, valid_map & np.isfinite(map_image), active_pixels)
        for statistic, value in summary.items():
            features[f"kinetic_{name}_{statistic}"] = value
    features["kinetic_any_valid_ratio"] = float(valid.sum() / max(active_pixels, 1))
    return maps, features, valid_map


def component_and_spread(mask: np.ndarray) -> tuple[float, float]:
    count = int(mask.sum())
    if count == 0:
        return 0.0, float("nan")
    components, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if components > 1 else count
    y, x = np.nonzero(mask)
    height, width = mask.shape
    x_norm = x / max(width - 1, 1)
    y_norm = y / max(height - 1, 1)
    spread = math.sqrt(float(np.var(x_norm) + np.var(y_norm)))
    return float(largest / count), spread


def time_to_area_fraction(
    area_curve: np.ndarray,
    times: np.ndarray,
    fraction: float,
) -> float:
    maximum = float(np.max(area_curve))
    if maximum <= 0:
        return float("nan")
    positions = np.flatnonzero(area_curve >= fraction * maximum)
    return float(times[positions[0]]) if len(positions) else float("nan")


def build_filling_features(
    enhancement: np.ndarray,
    indices: list[int],
    fov_mask: np.ndarray,
    active_mask: np.ndarray,
    peak_map: np.ndarray,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray]:
    times = normalized_times(indices)
    valid_peak = active_mask & np.isfinite(peak_map) & (peak_map > 0)
    threshold = float(config["kinetics"]["visible_fraction_of_pixel_peak"])
    visible = np.zeros_like(enhancement, dtype=bool)
    for position in range(enhancement.shape[0]):
        visible[position] = (
            valid_peak
            & (enhancement[position] >= threshold * np.nan_to_num(peak_map, nan=np.inf))
        )
    new_masks = np.zeros_like(visible)
    washout_masks = np.zeros_like(visible)
    new_masks[0] = visible[0]
    for position in range(1, len(visible)):
        new_masks[position] = visible[position] & ~visible[position - 1]
        washout_masks[position] = visible[position - 1] & ~visible[position]
    fov_pixels = max(int(fov_mask.sum()), 1)
    area = visible.reshape(len(visible), -1).sum(axis=1) / fov_pixels
    new_area = new_masks.reshape(len(visible), -1).sum(axis=1) / fov_pixels
    washout_area = washout_masks.reshape(len(visible), -1).sum(axis=1) / fov_pixels
    largest_ratios: list[float] = []
    spreads: list[float] = []
    for mask in visible:
        largest, spread = component_and_spread(mask)
        largest_ratios.append(largest)
        spreads.append(spread)
    largest_array = np.asarray(largest_ratios, dtype=np.float32)
    spread_array = np.asarray(spreads, dtype=np.float32)
    growth = np.diff(area) / np.maximum(np.diff(times), 1e-6)
    peak_position = int(np.argmax(area))
    features = {
        "filling_time_to_10_percent_area": time_to_area_fraction(area, times, 0.1),
        "filling_time_to_50_percent_area": time_to_area_fraction(area, times, 0.5),
        "filling_time_to_90_percent_area": time_to_area_fraction(area, times, 0.9),
        "filling_maximum_area_fraction": float(np.max(area)),
        "filling_area_auc": float(np.trapz(area, times)),
        "filling_maximum_area_growth_rate": float(np.max(growth)) if len(growth) else float("nan"),
        "filling_growth_peak_position": (
            float(times[int(np.argmax(growth))]) if len(growth) else float("nan")
        ),
        "filling_washout_area_decay": float(
            (area[peak_position] - area[-1]) / max(area[peak_position], 1e-8)
        ),
        "filling_largest_component_ratio_peak": float(largest_array[peak_position]),
        "filling_largest_component_ratio_max": float(np.nanmax(largest_array)),
        "filling_spatial_spread_peak": float(spread_array[peak_position]),
        "filling_spatial_spread_max": float(np.nanmax(spread_array)),
        "filling_new_area_auc": float(np.trapz(new_area, times)),
        "filling_washout_area_auc": float(np.trapz(washout_area, times)),
    }
    curves = pd.DataFrame({
        "sequence_position": np.arange(len(indices), dtype=int),
        "frame_index": indices,
        "normalized_time": times,
        "visible_area_fraction": area,
        "new_area_fraction": new_area,
        "washout_area_fraction": washout_area,
        "largest_component_ratio": largest_array,
        "spatial_spread": spread_array,
    })
    return curves, features, visible


def sea_raft_code_tree_hash(repo_root: Path) -> str:
    rows: list[str] = []
    files = sorted(
        [
            *repo_root.joinpath("core").rglob("*.py"),
            *repo_root.joinpath("config").rglob("*.json"),
            *repo_root.joinpath("config").rglob("*.py"),
        ],
        key=lambda path: path.relative_to(repo_root).as_posix(),
    )
    for path in files:
        rows.append(f"{path.relative_to(repo_root).as_posix()}\t{sha256_file(path)}")
    return hash_lines(rows)


def load_sea_raft(config: dict[str, Any], device_text: str, logger: RunLogger):
    if not device_text.startswith("cuda"):
        raise RuntimeError("CUDA is mandatory; CPU fallback is forbidden")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is forbidden")
    device = torch.device(device_text)
    torch.cuda.set_device(device)
    repo_root = Path(config["model"]["repo_root"])
    sys.path.insert(0, str(repo_root / "core"))
    sys.path.insert(0, str(repo_root))
    from raft import RAFT

    model_config = json.loads(Path(config["model"]["config"]).read_text(encoding="utf-8"))
    model_args = argparse.Namespace(**model_config)
    local_dir = Path(config["model"]["local_pretrained_dir"])
    logger.log(f"Loading SEA-RAFT with RAFT.from_pretrained from {local_dir}")
    model = RAFT.from_pretrained(str(local_dir), args=model_args)
    model = model.to(device)
    model.eval()
    if next(model.parameters()).device.type != "cuda":
        raise AssertionError("SEA-RAFT parameters are not on CUDA")
    torch.backends.cudnn.benchmark = True
    torch.cuda.reset_peak_memory_stats(device)
    properties = torch.cuda.get_device_properties(device)
    gpu_metadata = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": str(device),
        "gpu_name": properties.name,
        "gpu_total_memory_bytes": int(properties.total_memory),
        "model_load_method": config["model"]["load_method"],
        "model_sha256": sha256_file(Path(config["model"]["model_file"])),
        "model_config_sha256": sha256_file(Path(config["model"]["config"])),
        "sea_raft_code_tree_sha256": sea_raft_code_tree_hash(repo_root),
        "iters": int(model_args.iters),
        "scale": int(model_args.scale),
    }
    if gpu_metadata["sea_raft_code_tree_sha256"] != config["model"]["sea_raft_code_tree_sha256"]:
        raise AssertionError("SEA-RAFT code-tree hash changed")
    return model, model_args, device, gpu_metadata


def image_tensor(normalized_frame: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = np.repeat(normalized_frame[..., None], 3, axis=-1) * 255.0
    return (
        torch.from_numpy(np.ascontiguousarray(rgb))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(device)
    )


def uncertainty_log_from_info(
    info: torch.Tensor,
    var_min: float,
    var_max: float,
) -> torch.Tensor:
    raw_b = info[:, 2:]
    log_b = torch.zeros_like(raw_b)
    weights = info[:, :2].softmax(dim=1)
    log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=var_max)
    log_b[:, 1] = torch.clamp(raw_b[:, 1], min=var_min, max=0)
    return (log_b * weights).sum(dim=1, keepdim=True)


@torch.inference_mode()
def infer_flow(
    model,
    model_args,
    frame1: np.ndarray,
    frame2: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    tensor1 = image_tensor(frame1, device)
    tensor2 = image_tensor(frame2, device)
    original_size = tensor1.shape[-2:]
    scale_factor = 2.0 ** int(model_args.scale)
    if scale_factor != 1.0:
        scaled_size = (
            max(8, int(round(original_size[0] * scale_factor))),
            max(8, int(round(original_size[1] * scale_factor))),
        )
        tensor1 = F.interpolate(
            tensor1, size=scaled_size, mode="bilinear", align_corners=False
        )
        tensor2 = F.interpolate(
            tensor2, size=scaled_size, mode="bilinear", align_corners=False
        )
    output = model(tensor1, tensor2, iters=model_args.iters, test_mode=True)
    flow = output["flow"][-1]
    info = output["info"][-1]
    if flow.shape[-2:] != original_size:
        flow = F.interpolate(
            flow, size=original_size, mode="bilinear", align_corners=False
        ) / scale_factor
        info = F.interpolate(info, size=original_size, mode="area")
    uncertainty_log = uncertainty_log_from_info(
        info, float(model_args.var_min), float(model_args.var_max)
    )
    if not flow.is_cuda or not uncertainty_log.is_cuda:
        raise AssertionError("SEA-RAFT inference output is not on CUDA")
    return (
        flow[0].permute(1, 2, 0).float().cpu().numpy(),
        uncertainty_log[0, 0].float().cpu().numpy(),
    )


def warp_backward_flow(
    backward: np.ndarray,
    forward: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = forward.shape[:2]
    x, y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    map_x = x + forward[..., 0].astype(np.float32)
    map_y = y + forward[..., 1].astype(np.float32)
    valid = (map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1)
    warped = np.empty_like(backward, dtype=np.float32)
    for channel in range(2):
        warped[..., channel] = cv2.remap(
            backward[..., channel].astype(np.float32),
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=np.nan,
        )
    return warped, valid


def direction_statistics(
    u: np.ndarray,
    v: np.ndarray,
    weights: np.ndarray | None,
    bins: int,
) -> tuple[float, float]:
    magnitude = np.hypot(u, v)
    valid = np.isfinite(magnitude) & (magnitude > 1e-12)
    if int(valid.sum()) < bins:
        return float("nan"), float("nan")
    angle = np.arctan2(v[valid], u[valid])
    if weights is None:
        selected_weights = np.ones_like(angle)
    else:
        selected_weights = np.asarray(weights[valid], dtype=np.float64)
    weight_sum = max(float(selected_weights.sum()), 1e-12)
    coherence = float(
        math.hypot(
            float(np.sum(selected_weights * np.cos(angle))),
            float(np.sum(selected_weights * np.sin(angle))),
        )
        / weight_sum
    )
    histogram, _ = np.histogram(
        angle, bins=bins, range=(-math.pi, math.pi), weights=selected_weights
    )
    if histogram.sum() <= 0:
        entropy = float("nan")
    else:
        probabilities = histogram[histogram > 0] / histogram.sum()
        entropy = float(
            -np.sum(probabilities * np.log(probabilities)) / math.log(bins)
        )
    return coherence, entropy


def region_flow_statistics(
    name: str,
    region: np.ndarray,
    fov_mask: np.ndarray,
    active_mask: np.ndarray,
    residual_u_norm: np.ndarray,
    residual_v_norm: np.ndarray,
    fb_relative: np.ndarray,
    uncertainty_log: np.ndarray,
    hard_valid: np.ndarray,
    soft_weight: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    region_count = int(region.sum())
    eligible = region & hard_valid
    eligible_count = int(eligible.sum())
    result: dict[str, Any] = {
        f"{name}_pixels": region_count,
        f"{name}_coverage_fov": float(region_count / max(int(fov_mask.sum()), 1)),
        f"{name}_coverage_active": float(region_count / max(int(active_mask.sum()), 1)),
        f"{name}_hard_valid_ratio": float(eligible_count / max(region_count, 1)),
    }
    metric_names = [
        "res_mag_norm_mean", "res_mag_norm_median", "res_mag_norm_iqr",
        "res_mag_norm_p90", "weighted_mag_norm_mean", "direction_coherence",
        "direction_entropy", "fb_relative_mean", "uncertainty_log_mean",
        "soft_weight_mean",
    ]
    if eligible_count < int(config["flow_qc"]["minimum_region_pixels"]):
        result.update({f"{name}_{metric}": float("nan") for metric in metric_names})
        return result
    u = residual_u_norm[eligible]
    v = residual_v_norm[eligible]
    magnitude = np.hypot(u, v)
    weights = soft_weight[eligible]
    p25, p75 = np.percentile(magnitude, [25, 75])
    coherence, entropy = direction_statistics(
        u, v, weights, int(config["flow_qc"]["direction_bins"])
    )
    result.update({
        f"{name}_res_mag_norm_mean": float(np.mean(magnitude)),
        f"{name}_res_mag_norm_median": float(np.median(magnitude)),
        f"{name}_res_mag_norm_iqr": float(p75 - p25),
        f"{name}_res_mag_norm_p90": float(np.percentile(magnitude, 90)),
        f"{name}_weighted_mag_norm_mean": float(
            np.sum(weights * magnitude) / max(float(weights.sum()), 1e-12)
        ),
        f"{name}_direction_coherence": coherence,
        f"{name}_direction_entropy": entropy,
        f"{name}_fb_relative_mean": float(np.mean(fb_relative[eligible])),
        f"{name}_uncertainty_log_mean": float(np.mean(uncertainty_log[eligible])),
        f"{name}_soft_weight_mean": float(np.mean(weights)),
    })
    return result


def legacy_pair_features(flow: np.ndarray, uncertainty_log: np.ndarray) -> dict[str, float]:
    u = flow[..., 0]
    v = flow[..., 1]
    magnitude = np.hypot(u, v)
    height, width = magnitude.shape
    diagonal = max(math.hypot(height, width), 1.0)
    normalized = magnitude / diagonal
    coherence, entropy = direction_statistics(u, v, None, 18)
    return {
        "mag_mean": float(np.mean(magnitude)),
        "mag_median": float(np.median(magnitude)),
        "mag_std": float(np.std(magnitude)),
        "mag_p90": float(np.percentile(magnitude, 90)),
        "mag_p95": float(np.percentile(magnitude, 95)),
        "mag_max": float(np.max(magnitude)),
        "mag_norm_mean": float(np.mean(normalized)),
        "mag_norm_p90": float(np.percentile(normalized, 90)),
        "u_mean": float(np.mean(u)),
        "u_std": float(np.std(u)),
        "v_mean": float(np.mean(v)),
        "v_std": float(np.std(v)),
        "direction_entropy": entropy,
        "uncertainty_mean": float(np.mean(uncertainty_log)),
        "uncertainty_std": float(np.std(uncertainty_log)),
        "uncertainty_p90": float(np.percentile(uncertainty_log, 90)),
    }


def pair_stage(frame_stages: list[str], first_position: int, second_position: int) -> str:
    first = frame_stages[first_position]
    second = frame_stages[second_position]
    if "peak" in {first, second}:
        return "peak"
    if second == "washout":
        return "washout"
    if second == "washin" or first == "washin":
        return "washin"
    return "precontrast"


def analyze_flow_pair(
    forward: np.ndarray,
    backward: np.ndarray,
    uncertainty_log: np.ndarray,
    enhancement1: np.ndarray,
    enhancement2: np.ndarray,
    peak_map: np.ndarray,
    visible1: np.ndarray,
    visible2: np.ndarray,
    masks: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    height, width = forward.shape[:2]
    warped_backward, in_bounds = warp_backward_flow(backward, forward)
    fb_vector = forward + warped_backward
    fb_error = np.linalg.norm(fb_vector, axis=-1)
    forward_mag = np.linalg.norm(forward, axis=-1)
    backward_mag = np.linalg.norm(warped_backward, axis=-1)
    fb_relative = fb_error / np.maximum(forward_mag + backward_mag + 0.5, 1e-6)
    finite = (
        np.isfinite(forward).all(axis=-1)
        & np.isfinite(warped_backward).all(axis=-1)
        & np.isfinite(uncertainty_log)
        & np.isfinite(fb_relative)
    )
    hard_valid = (
        finite
        & in_bounds
        & (fb_relative <= float(config["flow_qc"]["fb_relative_hard_max"]))
        & (
            uncertainty_log
            <= float(config["flow_qc"]["uncertainty_log_hard_max"])
        )
    )
    soft_weight = np.zeros((height, width), dtype=np.float32)
    soft_weight[hard_valid] = (
        np.exp(
            -fb_relative[hard_valid] / float(config["flow_qc"]["fb_soft_tau"])
        )
        * np.exp(
            -uncertainty_log[hard_valid]
            / float(config["flow_qc"]["uncertainty_soft_tau"])
        )
    ).astype(np.float32)
    background_valid = masks["background"] & hard_valid
    if int(background_valid.sum()) < int(config["activity"]["minimum_background_pixels"]):
        raise AssertionError("Too few reliable background pixels for global-motion removal")
    global_u = float(np.median(forward[..., 0][background_valid]))
    global_v = float(np.median(forward[..., 1][background_valid]))
    residual_u = forward[..., 0] - global_u
    residual_v = forward[..., 1] - global_v
    residual_u_norm = residual_u / max(width, 1)
    residual_v_norm = residual_v / max(height, 1)
    residual_mag_norm = np.hypot(residual_u_norm, residual_v_norm)

    active = masks["active"]
    filling_front = active & ~visible1 & visible2
    persistent = active & visible1 & visible2
    delta_enhancement = enhancement2 - enhancement1
    washout_front = (
        active
        & visible1
        & ~visible2
        & (
            delta_enhancement
            < -float(config["kinetics"]["washout_front_fraction_of_peak"])
            * np.nan_to_num(peak_map, nan=np.inf)
        )
    )
    regions = {
        "active": active,
        "filling_front": filling_front,
        "persistent": persistent,
        "washout_front": washout_front,
    }
    row: dict[str, Any] = {}
    row.update(legacy_pair_features(forward, uncertainty_log))
    row.update({
        "global_motion_u_pixels": global_u,
        "global_motion_v_pixels": global_v,
        "global_motion_u_norm": global_u / max(width, 1),
        "global_motion_v_norm": global_v / max(height, 1),
        "global_motion_mag_norm": math.hypot(
            global_u / max(width, 1), global_v / max(height, 1)
        ),
        "hard_valid_ratio_fov": float((hard_valid & masks["fov"]).sum() / max(masks["fov"].sum(), 1)),
        "fb_error_mean": float(np.nanmean(fb_error[masks["fov"]])),
        "fb_relative_mean": float(np.nanmean(fb_relative[masks["fov"]])),
        "fb_relative_p90": float(np.nanpercentile(fb_relative[masks["fov"]], 90)),
        "uncertainty_log_mean": float(np.mean(uncertainty_log[masks["fov"]])),
        "uncertainty_log_p90": float(np.percentile(uncertainty_log[masks["fov"]], 90)),
        "soft_weight_mean_fov": float(np.mean(soft_weight[masks["fov"]])),
    })
    for name, region in regions.items():
        row.update(
            region_flow_statistics(
                name, region, masks["fov"], active, residual_u_norm,
                residual_v_norm, fb_relative, uncertainty_log, hard_valid,
                soft_weight, config,
            )
        )

    coupling_valid = active & hard_valid
    magnitude_values = residual_mag_norm[coupling_valid]
    change_values = np.abs(delta_enhancement[coupling_valid])
    if len(magnitude_values) >= int(config["flow_qc"]["minimum_region_pixels"]):
        if np.std(magnitude_values) > 1e-12 and np.std(change_values) > 1e-12:
            correlation = float(np.corrcoef(magnitude_values, change_values)[0, 1])
        else:
            correlation = float("nan")
        high_flow_threshold = float(np.percentile(
            magnitude_values, config["flow_qc"]["high_flow_percentile"]
        ))
        high_change_threshold = float(np.percentile(
            change_values, config["flow_qc"]["high_change_percentile"]
        ))
        high_flow = coupling_valid & (residual_mag_norm >= high_flow_threshold)
        high_change = coupling_valid & (np.abs(delta_enhancement) >= high_change_threshold)
        row["flow_intensity_corr"] = correlation
        row["flow_front_overlap"] = float(
            (high_flow & filling_front).sum() / max(filling_front.sum(), 1)
        )
        row["high_flow_high_change_ratio"] = float(
            (high_flow & high_change).sum() / max(coupling_valid.sum(), 1)
        )
    else:
        row["flow_intensity_corr"] = float("nan")
        row["flow_front_overlap"] = float("nan")
        row["high_flow_high_change_ratio"] = float("nan")

    cache = {
        "residual_u_norm": residual_u_norm,
        "residual_v_norm": residual_v_norm,
        "residual_mag_norm": residual_mag_norm,
        "fb_relative": fb_relative,
        "uncertainty_log": uncertainty_log,
        "filling_front": filling_front,
        "persistent": persistent,
        "washout_front": washout_front,
        "hard_valid": hard_valid,
        "soft_weight": soft_weight,
    }
    return row, cache


def resize_float64(values: np.ndarray) -> np.ndarray:
    return cv2.resize(
        values.astype(np.float32), CACHE_SIZE[::-1], interpolation=cv2.INTER_AREA
    )


def resize_mask64(mask: np.ndarray) -> np.ndarray:
    resized = cv2.resize(
        mask.astype(np.float32), CACHE_SIZE[::-1], interpolation=cv2.INTER_AREA
    )
    return np.clip(np.rint(resized * 255.0), 0, 255).astype(np.uint8)


def normalized_cache_map(values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(CACHE_SIZE, dtype=np.float16), {
            "low": None, "high": None, "method": "no_finite_values"
        }
    low = float(np.percentile(finite, 1))
    high = float(np.percentile(finite, 99))
    if high <= low:
        normalized = np.zeros_like(values, dtype=np.float32)
    else:
        normalized = np.clip((values - low) / (high - low), 0, 1)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    return resize_float64(normalized).astype(np.float16), {
        "low": low, "high": high, "method": "phase_map_p1_p99_then_area_resize"
    }




def save_table_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8", compression="gzip", lineterminator="\n")


def save_flow_visualization(
    output_dir: Path,
    pair_name: str,
    forward: np.ndarray,
    cache: dict[str, np.ndarray],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    u = cache["residual_u_norm"]
    v = cache["residual_v_norm"]
    magnitude, angle = cv2.cartToPolar(u.astype(np.float32), v.astype(np.float32))
    hsv = np.zeros((*magnitude.shape, 3), dtype=np.uint8)
    hsv[..., 0] = np.mod(angle * 90 / math.pi, 180).astype(np.uint8)
    hsv[..., 1] = 255
    upper = max(float(np.percentile(magnitude, 99)), 1e-8)
    hsv[..., 2] = np.clip(magnitude / upper * 255, 0, 255).astype(np.uint8)
    flow_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(str(output_dir / f"{pair_name}_residual_flow.jpg"), flow_image)
    uncertainty = cache["uncertainty_log"]
    unc_upper = max(float(np.percentile(uncertainty, 99)), 1e-8)
    unc_image = np.clip(uncertainty / unc_upper * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(
        str(output_dir / f"{pair_name}_uncertainty.jpg"),
        cv2.applyColorMap(unc_image, cv2.COLORMAP_TURBO),
    )
    masks_image = np.zeros((*magnitude.shape, 3), dtype=np.uint8)
    masks_image[cache["persistent"]] = (0, 180, 0)
    masks_image[cache["filling_front"]] = (0, 0, 255)
    masks_image[cache["washout_front"]] = (255, 0, 0)
    cv2.imwrite(str(output_dir / f"{pair_name}_front_regions.jpg"), masks_image)


def selected_frames_table(plan: dict[str, Any]) -> pd.DataFrame:
    pair_starts = {first for first, _ in plan["pairs"]}
    rows: list[dict[str, Any]] = []
    frames = load_phase_frames(plan["frame_paths"], 1)
    for position, (path, frame_index) in enumerate(
        zip(plan["frame_paths"], plan["frame_indices"])
    ):
        height, width = frames[position].shape
        next_delta = (
            plan["frame_indices"][position + 1] - frame_index
            if position + 1 < len(plan["frame_indices"]) else np.nan
        )
        rows.append({
            "patient_id": plan["patient_id"],
            "split": plan["split"],
            "source_type": plan["source_type"],
            "selected_series_id": plan["selected_series_id"],
            "phase": plan["phase"],
            "selected_internal_series": plan["selected_internal_series"],
            "sequence_position": position,
            "frame_index": frame_index,
            "absolute_path": str(path),
            "height": height,
            "width": width,
            "starts_true_contiguous_pair": position in pair_starts,
            "delta_to_next_frame": next_delta,
        })
    return pd.DataFrame(rows)


def process_phase(
    plan: dict[str, Any],
    output_root: Path,
    model,
    model_args,
    device: torch.device,
    gpu_metadata: dict[str, Any],
    config: dict[str, Any],
    num_workers: int,
    max_visual_pairs: int,
    max_pairs_per_phase: int | None,
    resume: bool,
    logger: RunLogger,
) -> dict[str, Any]:
    patient_id = plan["patient_id"]
    phase = plan["phase"]
    phase_output = output_root / patient_id / phase
    if phase_output.exists():
        success_path = phase_output / ".SUCCESS"
        if resume and success_path.is_file():
            metadata = json.loads((phase_output / "metadata.json").read_text(encoding="utf-8"))
            if metadata["frame_list_hash"] != plan["frame_list_hash"]:
                raise AssertionError(f"{patient_id} {phase}: resume frame hash mismatch")
            logger.log(f"[RESUME SKIP] {patient_id} {phase}")
            return json.loads((phase_output / "phase_summary.json").read_text(encoding="utf-8"))
        if resume:
            failed_root = output_root / "_failed_attempts" / patient_id
            failed_root.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            archived = failed_root / f"{phase}_{stamp}_{os.getpid()}"
            os.replace(phase_output, archived)
            logger.log(
                f"[RESUME RETRY] archived incomplete {patient_id} {phase} to {archived}"
            )
        else:
            raise FileExistsError(f"Phase output already exists: {phase_output}")
    phase_output.mkdir(parents=True, exist_ok=False)
    running_marker = phase_output / ".RUNNING"
    running_marker.write_text(utc_now() + "\n", encoding="utf-8")
    phase_log_path = phase_output / "run.log"
    phase_log = phase_log_path.open("x", encoding="utf-8")

    def phase_message(message: str) -> None:
        line = f"[{utc_now()}] {message}"
        phase_log.write(line + "\n")
        phase_log.flush()
        logger.log(f"{patient_id} {phase}: {message}")

    source_signature_before = source_metadata_signature(plan["frame_paths"])
    start_phase = time.perf_counter()
    phase_message(
        f"loading {len(plan['frame_paths'])} frozen frames and "
        f"{plan['manifest_expected_pairs']} true contiguous pairs"
    )
    frames = load_phase_frames(plan["frame_paths"], num_workers)
    normalized, p1, p99 = normalize_phase(frames, config)
    fov_mask, fov_qc = build_fov_mask(normalized, config)
    baseline, enhancement, polarity_qc = baseline_polarity_enhancement(
        normalized, fov_mask, config
    )
    masks, activity_qc, activity_map = build_activity_masks(
        enhancement, fov_mask, config
    )
    frame_kinetics, tdc_features, frame_stages = tdc_and_stage_features(
        enhancement, plan["frame_indices"], masks, config
    )
    kinetic_maps, kinetic_features, kinetic_valid_mask = build_kinetic_maps(
        enhancement,
        plan["frame_indices"],
        masks["active"],
        float(tdc_features["tdc_peak"]),
        config,
    )
    filling_curves, filling_features, visible = build_filling_features(
        enhancement,
        plan["frame_indices"],
        fov_mask,
        masks["active"],
        kinetic_maps["peak"],
        config,
    )
    temporal_curves = frame_kinetics.merge(
        filling_curves,
        on=["sequence_position", "frame_index", "normalized_time"],
        how="inner",
        validate="one_to_one",
    )
    temporal_curves.insert(0, "phase", phase)
    temporal_curves.insert(0, "patient_id", patient_id)

    selected_table = pd.DataFrame({
        "patient_id": patient_id,
        "split": plan["split"],
        "source_type": plan["source_type"],
        "selected_series_id": plan["selected_series_id"],
        "phase": phase,
        "selected_internal_series": plan["selected_internal_series"],
        "sequence_position": np.arange(len(plan["frame_paths"]), dtype=int),
        "frame_index": plan["frame_indices"],
        "absolute_path": [str(path) for path in plan["frame_paths"]],
        "height": frames.shape[1],
        "width": frames.shape[2],
        "starts_true_contiguous_pair": [
            position in {first for first, _ in plan["pairs"]}
            for position in range(len(plan["frame_paths"]))
        ],
        "delta_to_next_frame": [
            (
                plan["frame_indices"][position + 1] - plan["frame_indices"][position]
                if position + 1 < len(plan["frame_indices"]) else np.nan
            )
            for position in range(len(plan["frame_paths"]))
        ],
    })

    pairs_to_process = list(plan["pairs"])
    if max_pairs_per_phase is not None:
        pairs_to_process = pairs_to_process[:max_pairs_per_phase]
    visual_positions: set[int] = set()
    if max_visual_pairs > 0 and pairs_to_process:
        visual_positions = set(
            int(value)
            for value in np.linspace(
                0, len(pairs_to_process) - 1,
                min(max_visual_pairs, len(pairs_to_process)),
            )
        )

    pair_rows: list[dict[str, Any]] = []
    cache_lists: dict[str, list[np.ndarray]] = {
        "residual_u_norm": [],
        "residual_v_norm": [],
        "residual_mag_norm": [],
        "fb_relative": [],
        "uncertainty_log": [],
        "filling_front": [],
        "persistent": [],
        "washout_front": [],
        "hard_valid": [],
        "soft_weight": [],
    }
    for pair_order, (first_position, second_position) in enumerate(pairs_to_process):
        frame_index1 = plan["frame_indices"][first_position]
        frame_index2 = plan["frame_indices"][second_position]
        if frame_index2 - frame_index1 != 1:
            raise AssertionError(f"{patient_id} {phase}: attempted cross-gap pair")
        pair_start = time.perf_counter()
        forward, uncertainty_forward = infer_flow(
            model, model_args, normalized[first_position], normalized[second_position], device
        )
        backward, _ = infer_flow(
            model, model_args, normalized[second_position], normalized[first_position], device
        )
        torch.cuda.synchronize(device)
        features, cache = analyze_flow_pair(
            forward,
            backward,
            uncertainty_forward,
            enhancement[first_position],
            enhancement[second_position],
            kinetic_maps["peak"],
            visible[first_position],
            visible[second_position],
            masks,
            config,
        )
        runtime = time.perf_counter() - pair_start
        features.update({
            "patient_id": patient_id,
            "split": plan["split"],
            "source_type": plan["source_type"],
            "selected_series_id": plan["selected_series_id"],
            "phase": phase,
            "selected_internal_series": plan["selected_internal_series"],
            "pair_order": pair_order,
            "sequence_position_t": first_position,
            "sequence_position_t1": second_position,
            "frame_index_t": frame_index1,
            "frame_index_t1": frame_index2,
            "delta_frame": frame_index2 - frame_index1,
            "stage": pair_stage(frame_stages, first_position, second_position),
            "normalized_pair_time": float(
                0.5
                * (
                    temporal_curves.loc[first_position, "normalized_time"]
                    + temporal_curves.loc[second_position, "normalized_time"]
                )
            ),
            "tdc_derivative_pair": float(
                temporal_curves.loc[second_position, "tdc_active_median"]
                - temporal_curves.loc[first_position, "tdc_active_median"]
            ),
            "runtime_seconds": runtime,
        })
        pair_rows.append(features)
        for key in cache_lists:
            if cache[key].dtype == bool:
                cache_lists[key].append(
                    cv2.resize(
                        cache[key].astype(np.float32),
                        CACHE_SIZE[::-1],
                        interpolation=cv2.INTER_AREA,
                    )
                )
            else:
                cache_lists[key].append(resize_float64(cache[key]))
        if pair_order in visual_positions:
            save_flow_visualization(
                phase_output / "visualizations",
                f"pair_{pair_order:03d}_{frame_index1}_{frame_index2}",
                forward,
                cache,
            )
        phase_message(
            f"pair {pair_order + 1}/{len(pairs_to_process)} "
            f"frames {frame_index1}->{frame_index2} runtime={runtime:.3f}s"
        )

    pair_frame = pd.DataFrame(pair_rows)
    if len(pair_frame):
        if not (pair_frame["delta_frame"] == 1).all():
            raise AssertionError(f"{patient_id} {phase}: delta_frame assertion failed")
        flow_energy = pair_frame.set_index("sequence_position_t")[
            "active_weighted_mag_norm_mean"
        ]
        temporal_curves["flow_energy_to_next"] = temporal_curves["sequence_position"].map(
            flow_energy
        )
    else:
        temporal_curves["flow_energy_to_next"] = np.nan

    frame_kinetics_output = temporal_curves[[
        "patient_id", "phase", "sequence_position", "frame_index", "normalized_time",
        "tdc_active_median", "tdc_active_trimmed_mean", "tdc_high_activity_median",
        "tdc_derivative", "stage",
    ]].copy()
    selected_table.to_csv(
        phase_output / "selected_frames.csv", index=False, encoding="utf-8", lineterminator="\n"
    )
    save_table_csv_gz(pair_frame, phase_output / "pair_features.csv.gz")
    save_table_csv_gz(frame_kinetics_output, phase_output / "frame_kinetics.csv.gz")
    save_table_csv_gz(temporal_curves, phase_output / "temporal_curves.csv.gz")

    cache_scaling: dict[str, Any] = {}
    cache_arrays: dict[str, np.ndarray] = {}
    for key in ("residual_u_norm", "residual_v_norm", "residual_mag_norm", "fb_relative", "uncertainty_log", "soft_weight"):
        stack = np.stack(cache_lists[key], axis=0) if cache_lists[key] else np.zeros((1, *CACHE_SIZE), np.float32)
        cache_arrays[f"{key}_mean_64"] = np.mean(stack, axis=0).astype(np.float16)
        cache_arrays[f"{key}_p90_64"] = np.percentile(stack, 90, axis=0).astype(np.float16)
    for key in ("filling_front", "persistent", "washout_front", "hard_valid"):
        stack = np.stack(cache_lists[key], axis=0) if cache_lists[key] else np.zeros((1, *CACHE_SIZE), np.float32)
        cache_arrays[f"{key}_frequency_64"] = np.clip(
            np.rint(np.mean(stack, axis=0) * 255.0), 0, 255
        ).astype(np.uint8)
    cache_arrays["active_mask_64"] = resize_mask64(masks["active"])
    for map_name in ("toa10", "toa50", "ttp", "peak", "normalized_auc"):
        cached, scaling = normalized_cache_map(kinetic_maps[map_name])
        cache_arrays[f"{map_name}_normalized_64"] = cached
        cache_scaling[map_name] = scaling
    np.savez_compressed(phase_output / "phase_cache.npz", **cache_arrays)

    numeric_audits = {
        "pair_features": dataframe_numeric_audit(pair_frame),
        "frame_kinetics": dataframe_numeric_audit(frame_kinetics_output),
        "temporal_curves": dataframe_numeric_audit(temporal_curves),
    }
    if any(audit["inf_count"] for audit in numeric_audits.values()):
        raise AssertionError(f"{patient_id} {phase}: non-finite infinity detected")
    processed_pairs = len(pair_frame)
    complete_phase = max_pairs_per_phase is None
    if complete_phase and processed_pairs != plan["manifest_expected_pairs"]:
        raise AssertionError(
            f"{patient_id} {phase}: expected {plan['manifest_expected_pairs']}, "
            f"processed {processed_pairs}"
        )
    source_signature_after = source_metadata_signature(plan["frame_paths"])
    if source_signature_after != source_signature_before:
        raise AssertionError(f"{patient_id} {phase}: source metadata changed")

    pair_qc_features: dict[str, Any] = {}
    if len(pair_frame):
        for column in [
            "global_motion_mag_norm", "active_res_mag_norm_median",
            "active_res_mag_norm_p90", "active_weighted_mag_norm_mean",
            "fb_relative_mean", "uncertainty_log_mean", "soft_weight_mean_fov",
            "filling_front_coverage_fov", "persistent_coverage_fov",
            "washout_front_coverage_fov", "flow_intensity_corr",
            "flow_front_overlap", "high_flow_high_change_ratio",
        ]:
            pair_qc_features[f"qc_pair_{column}_mean"] = float(
                pd.to_numeric(pair_frame[column], errors="coerce").mean()
            )
    base_features = {
        **{
            key: value
            for key, value in tdc_features.items()
            if isinstance(value, (int, float, bool, np.integer, np.floating))
        },
        **kinetic_features,
        **filling_features,
    }
    qc_features = {
        "qc_polarity": polarity_qc["polarity"],
        "qc_positive_score": polarity_qc["positive_score"],
        "qc_negative_score": polarity_qc["negative_score"],
        "qc_fov_ratio": fov_qc["fov_ratio"],
        "qc_fov_fallback_border_only": fov_qc["fov_fallback_border_only"],
        "qc_active_ratio_fov": activity_qc["active_ratio_fov"],
        "qc_high_activity_ratio_fov": activity_qc["high_activity_ratio_fov"],
        "qc_background_ratio_fov": activity_qc["background_ratio_fov"],
        "qc_background_fallback": activity_qc["background_fallback"],
        "qc_kinetic_valid_ratio": kinetic_features["kinetic_any_valid_ratio"],
        "qc_manifest_expected_pairs": plan["manifest_expected_pairs"],
        "qc_processed_pairs": processed_pairs,
        "qc_complete_phase": complete_phase,
        "qc_phase_p1": p1,
        "qc_phase_p99": p99,
        "qc_source_signature_unchanged": True,
        **pair_qc_features,
    }
    phase_summary = {
        "patient_id": patient_id,
        "split": plan["split"],
        "source_type": plan["source_type"],
        "source_medical_record_root": plan["source_medical_record_root"],
        "selected_series_id": plan["selected_series_id"],
        "selected_series_path": plan["selected_series_path"],
        "phase": phase,
        "selected_internal_series": plan["selected_internal_series"],
        "n_frames": len(plan["frame_paths"]),
        "manifest_expected_pairs": plan["manifest_expected_pairs"],
        "processed_pairs": processed_pairs,
        "complete_phase": complete_phase,
        "frame_list_hash": plan["frame_list_hash"],
        "polarity": polarity_qc,
        "fov_qc": fov_qc,
        "activity_qc": activity_qc,
        "tdc_stages": {
            "frame_stages": frame_stages,
            "stage_counts": tdc_features["stage_counts"],
            "onset_frame_index": tdc_features["tdc_onset_frame_index"],
            "peak_frame_index": tdc_features["tdc_peak_frame_index"],
        },
        "base_features": base_features,
        "qc_features": qc_features,
        "numeric_audits": numeric_audits,
        "cache_scaling": cache_scaling,
        "gpu": gpu_metadata,
        "runtime_seconds": time.perf_counter() - start_phase,
        "table_format": "csv.gz",
        "parquet_fallback": True,
        "labels_read": False,
        "model_trained": False,
        "manifest_rescanned": False,
        "created_utc": utc_now(),
    }
    metadata = {
        "patient_id": patient_id,
        "split": plan["split"],
        "phase": phase,
        "frame_paths": [str(path) for path in plan["frame_paths"]],
        "frame_indices": plan["frame_indices"],
        "frame_list_hash": plan["frame_list_hash"],
        "source_metadata_signature_before": source_signature_before,
        "source_metadata_signature_after": source_signature_after,
        "config_path": str(CONFIG_DEFAULT),
        "config_sha256": sha256_file(CONFIG_DEFAULT),
        "thresholds_sha256": sha256_json({
            "normalization": config["normalization"],
            "fov": config["fov"],
            "activity": config["activity"],
            "kinetics": config["kinetics"],
            "flow_qc": config["flow_qc"],
            "aggregation": config["aggregation"],
        }),
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "model_sha256": gpu_metadata["model_sha256"],
        "model_config_sha256": gpu_metadata["model_config_sha256"],
        "sea_raft_code_tree_sha256": gpu_metadata["sea_raft_code_tree_sha256"],
        "cache_scaling": cache_scaling,
        "cache_dtype": {"float": "float16", "mask": "uint8_0_to_255_frequency"},
        "table_storage": {
            "parquet_engine_available": False,
            "selected_format": "csv.gz",
            "schema_numeric_values_unchanged_by_fallback": True,
        },
        "cuda_actually_used": True,
        "cpu_fallback": False,
    }
    write_json(phase_output / "phase_summary.json", phase_summary)
    write_json(phase_output / "metadata.json", metadata)
    success_payload = {
        "patient_id": patient_id,
        "phase": phase,
        "processed_pairs": processed_pairs,
        "complete_phase": complete_phase,
        "finished_utc": utc_now(),
    }
    write_json_atomic(phase_output / ".SUCCESS", success_payload)
    running_marker.unlink()
    phase_message(
        f"SUCCESS frames={len(plan['frame_paths'])} pairs={processed_pairs} "
        f"polarity={polarity_qc['polarity_label']} "
        f"active_ratio={activity_qc['active_ratio_fov']:.6f}"
    )
    phase_log.close()
    return phase_summary


def write_failure(stage: str, exc: BaseException) -> None:
    failure_path = REPORT_ROOT / "unattended_failure.md"
    if failure_path.exists():
        return
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    failure_path.write_text(
        "\n".join([
            "# api_fullseq_v2 Full Train + Full Valid unattended failure",
            "",
            f"- Stage: {stage}",
            f"- Exception: {type(exc).__name__}: {exc}",
            "- Manifest modified: no.",
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


def output_root_for_mode(mode: str) -> Path:
    if mode in {"pilot_train", "full_train"}:
        return TRAIN_OUTPUT_DEFAULT
    if mode in {"pilot_valid_integrity", "full_valid"}:
        return VALID_OUTPUT_DEFAULT
    raise ValueError(mode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(FULL_TRAIN_MANIFEST))
    parser.add_argument(
        "--mode",
        required=True,
        choices=["pilot_train", "pilot_valid_integrity", "full_train", "full_valid"],
    )
    parser.add_argument("--patient-id", action="append", default=None)
    parser.add_argument("--phase", choices=["pre", "post"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--max-visual-pairs", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-pairs-per-phase", type=int, default=None)
    parser.add_argument("--config", default=str(CONFIG_DEFAULT))
    args = parser.parse_args()

    stage = "preflight"
    logger: RunLogger | None = None
    formal_run = False
    try:
        config_path = Path(args.config).resolve()
        config = load_config(config_path)
        if config_path != CONFIG_DEFAULT.resolve():
            raise AssertionError("Current task requires the Full Train + Valid config")
        if args.mode == "full_valid" and not args.dry_run and not TRAIN_SUCCESS_MARKER.is_file():
            raise AssertionError("Full Valid requires .FULL_TRAIN_SUCCESS")
        if args.mode in {"full_train", "full_valid"} and args.max_visual_pairs != 0:
            raise AssertionError(
                "Full extraction requires --max-visual-pairs 0; fixed QC is selected internally"
            )
        verify_frozen_inputs(config)
        requested_ids = parse_requested_patient_ids(args.patient_id)
        plans, selected_manifest = manifest_phase_plans(
            Path(args.manifest).resolve(),
            config,
            args.mode,
            requested_ids,
            args.phase,
            args.limit,
        )
        filtered = bool(requested_ids or args.phase or args.limit is not None)
        summary = dry_run_summary(
            plans, selected_manifest, config, args.mode, filtered
        )
        if args.dry_run:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0

        if args.mode == "pilot_valid_integrity":
            verify_valid_frozen_hashes()
        output_root = (
            Path(args.output_root).resolve()
            if args.output_root else output_root_for_mode(args.mode).resolve()
        )
        formal_run = not str(output_root).startswith("/tmp/")
        if output_root.exists():
            if not args.resume:
                raise FileExistsError(f"Output root already exists: {output_root}")
        else:
            output_root.mkdir(parents=True, exist_ok=False)

        logger = RunLogger(args.mode)
        logger.log(json.dumps({
            "event": "start",
            "mode": args.mode,
            "patients": int(selected_manifest["patient_id"].nunique()),
            "phases": len(plans),
            "manifest_pairs": sum(plan["manifest_expected_pairs"] for plan in plans),
            "output_root": str(output_root),
            "filtered": filtered,
            "parquet_engine": False,
            "table_fallback": "csv.gz",
        }, ensure_ascii=False, sort_keys=True))

        stage = "load_gpu_model"
        model, model_args, device, gpu_metadata = load_sea_raft(
            config, args.device, logger
        )
        stage = "phase_extraction"
        phase_summaries: list[dict[str, Any]] = []
        qc_patient_ids: set[str] = set()
        if args.mode in {"full_train", "full_valid"} and not filtered:
            patient_order = selected_manifest["patient_id"].astype(str).tolist()
            positions = (
                config["full"]["qc_manifest_positions_one_based"]
                if args.mode == "full_train"
                else config["full"]["valid_qc_manifest_positions_one_based"]
            )
            qc_patient_ids = {patient_order[int(position) - 1] for position in positions}
        for index, plan in enumerate(plans, start=1):
            logger.log(
                f"phase {index}/{len(plans)} {plan['patient_id']} {plan['phase']}"
            )
            phase_summaries.append(
                process_phase(
                    plan,
                    output_root,
                    model,
                    model_args,
                    device,
                    gpu_metadata,
                    config,
                    args.num_workers,
                    1 if plan["patient_id"] in qc_patient_ids else args.max_visual_pairs,
                    args.max_pairs_per_phase,
                    args.resume,
                    logger,
                )
            )

        stage = "root_audit"
        torch.cuda.synchronize(device)
        peak_memory = int(torch.cuda.max_memory_allocated(device))
        if peak_memory <= 0:
            raise AssertionError("CUDA was not actually used")
        total_processed = sum(int(item["processed_pairs"]) for item in phase_summaries)
        complete_run = args.max_pairs_per_phase is None
        if complete_run and not filtered:
            if args.mode == "pilot_train":
                expected_total = config["pilot"]["expected_train_pairs"]
            elif args.mode == "pilot_valid_integrity":
                expected_total = config["pilot"]["expected_valid_integrity_pairs"]
            elif args.mode == "full_train":
                expected_total = config["full"]["expected_train_pairs"]
            elif args.mode == "full_valid":
                expected_total = config["full"]["expected_valid_pairs"]
            else:
                raise RuntimeError(f"Unsupported full mode: {args.mode}")
            if total_processed != expected_total:
                raise AssertionError(
                    f"Root pair total expected {expected_total}, got {total_processed}"
                )
        if args.mode == "pilot_train" and not filtered:
            if selected_manifest["patient_id"].nunique() != 15:
                raise AssertionError("Train Pilot patient count changed")
        if args.mode == "pilot_valid_integrity" and not filtered:
            if selected_manifest["patient_id"].tolist() != ["549117"]:
                raise AssertionError("Valid integrity isolation changed")
        if args.mode == "full_train" and not filtered:
            if selected_manifest["patient_id"].nunique() != int(
                config["full"]["expected_train_patients"]
            ):
                raise AssertionError("Full Train patient count changed")
            if len(phase_summaries) != int(config["full"]["expected_train_phases"]):
                raise AssertionError("Full Train phase count changed")
        if args.mode == "full_valid" and not filtered:
            if selected_manifest["patient_id"].nunique() != int(
                config["full"]["expected_valid_patients"]
            ):
                raise AssertionError("Full Valid patient count changed")
            if len(phase_summaries) != int(config["full"]["expected_valid_phases"]):
                raise AssertionError("Full Valid phase count changed")
        old_source_reads = sum(
            int(
                plan["patient_id"] in config["updated_fixed_series"]
                and any(
                    str(path).startswith(
                        f"/root/autodl-tmp/tiantanDSA/{plan['patient_id']}/"
                    )
                    for path in plan["frame_paths"]
                )
            )
            for plan in plans
        )
        if old_source_reads:
            raise AssertionError(f"Updated old-source reads={old_source_reads}")
        root_summary = {
            "mode": args.mode,
            "patients": int(selected_manifest["patient_id"].nunique()),
            "phases": len(phase_summaries),
            "manifest_expected_pairs": sum(
                int(item["manifest_expected_pairs"]) for item in phase_summaries
            ),
            "processed_pairs": total_processed,
            "complete_run": complete_run,
            "filtered": filtered,
            "cuda_actually_used": True,
            "cpu_fallback": False,
            "cuda_peak_memory_bytes": peak_memory,
            "updated_old_source_reads": old_source_reads,
            "selected_png": 0,
            "selected_parameter_maps": 0,
            "fixed_qc_patient_ids": sorted(qc_patient_ids),
            "labels_read": False,
            "model_trained": False,
            "full_train_started": args.mode == "full_train",
            "full_valid_started": args.mode == "full_valid",
            "manifest_rescanned": False,
            "phase_summaries": phase_summaries,
            "config_sha256": sha256_file(CONFIG_DEFAULT),
            "code_sha256": sha256_file(Path(__file__).resolve()),
            "model_sha256": gpu_metadata["model_sha256"],
            "created_utc": utc_now(),
        }
        write_json(output_root / "run_summary.json", root_summary)
        pd.DataFrame([
            {
                "patient_id": item["patient_id"],
                "split": item["split"],
                "phase": item["phase"],
                "n_frames": item["n_frames"],
                "manifest_expected_pairs": item["manifest_expected_pairs"],
                "processed_pairs": item["processed_pairs"],
                "complete_phase": item["complete_phase"],
                "polarity": item["polarity"]["polarity_label"],
                "active_ratio_fov": item["activity_qc"]["active_ratio_fov"],
                "background_ratio_fov": item["activity_qc"]["background_ratio_fov"],
                "kinetic_valid_ratio": item["qc_features"]["qc_kinetic_valid_ratio"],
                "runtime_seconds": item["runtime_seconds"],
            }
            for item in phase_summaries
        ]).to_csv(
            output_root / "phase_audit.csv",
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )
        write_json_atomic(output_root / ".SUCCESS", {
            "mode": args.mode,
            "processed_pairs": total_processed,
            "finished_utc": utc_now(),
        })
        logger.log(
            f"SUCCESS mode={args.mode} patients={root_summary['patients']} "
            f"phases={len(phase_summaries)} pairs={total_processed} "
            f"cuda_peak_memory={peak_memory}"
        )
        return 0
    except BaseException as exc:
        if logger is not None:
            logger.log(f"FAIL stage={stage}: {type(exc).__name__}: {exc}")
        if formal_run:
            write_failure(stage, exc)
        if isinstance(exc, (AssertionError, ValueError)):
            return 42
        raise
    finally:
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    sys.exit(main())

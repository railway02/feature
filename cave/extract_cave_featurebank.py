#!/usr/bin/env python3
"""Extract a frozen, label-blind CAVE DSA feature bank from v3 manifests.

The program runs the official CAVE sequence AV ConvGRU once per deterministic
block/view, stores f4/f5 caches, 5120-D pooled embeddings, temporal trajectories,
artery/vein probabilities, CAVE-mask kinetics, visual QC, and full provenance.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if not os.environ.get("OMP_NUM_THREADS", "").isdigit():
    os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import numpy as np
import pandas as pd
import torch

from cave_model import (
    CAVEHookExtractor, cave_code_tree_hash, git_commit, git_is_dirty, load_cave_model,
)
from common import (
    RunLogger, sha256_file, sha256_json, sha256_tree, utc_now, write_csv_atomic,
    write_json_atomic,
)
from io_ops import (
    atomic_directory, frames_to_model, load_gray_frames, make_square_transform,
    map_model_to_original, map_original_to_model, save_av_overlay, save_input_mosaic,
    save_npz, save_probability_png, strict_contiguous_blocks, temporal_views,
)
from manifest import ManifestBundle, PhasePlan, load_manifest
from pooling import (
    PRIMARY_BLOCKS, TRAJECTORY_REGIONS, TRAJECTORY_SCALES, build_embedding_bank,
    pool_trajectory, resample_trajectory,
)
from release import verify_release
from scalar_features import build_scalar_bank, expected_scalar_count
from schema import ensure_schema
from v3_bridge import V3Bridge

EXPECTED_CKPT_SHA = "c90b7e066e32039cf61352993a9c57784caac6aa1fdb042dc4801df6dc729651"
EXPECTED_CKPT_SIZE = 332_731_061


def _to_tensor(frames01: np.ndarray, device: torch.device) -> torch.Tensor:
    array = np.ascontiguousarray(frames01[:, None], dtype=np.float32)
    return torch.from_numpy(array).unsqueeze(0).to(device=device, dtype=torch.float32)


def _weighted_average(arrays: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    if not arrays or len(arrays) != len(weights):
        raise ValueError("Invalid weighted-average inputs")
    stack = np.stack(arrays).astype(np.float32)
    shape = (-1,) + (1,) * arrays[0].ndim
    return np.sum(stack * weights.reshape(shape), axis=0, dtype=np.float32)


def _finite_count(values: dict[str, Any]) -> tuple[int, int]:
    finite, nonfinite = 0, 0
    for value in values.values():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            finite += 1
        else:
            nonfinite += 1
    return finite, nonfinite


def _validate_frozen_config(config: dict[str, Any]) -> None:
    required = {
        "project", "cave_commit", "image_size", "max_sequence_length", "temporal_views",
        "strict_contiguous_blocks", "primary_embedding_dimension", "scalar_feature_count",
        "amp", "labels_forbidden", "training_forbidden", "manifest_rescan_forbidden",
        "expected_counts",
    }
    missing = required - set(config)
    if missing:
        raise KeyError(f"Frozen config missing {sorted(missing)}")
    if int(config["image_size"]) != 512:
        raise AssertionError("CAVE checkpoint is frozen to image_size=512")
    if int(config["max_sequence_length"]) != 20:
        raise AssertionError("max_sequence_length must remain 20")
    if config["temporal_views"] != ["uniform_full20", "contrast_core20"]:
        raise AssertionError("Temporal views changed")
    if config["strict_contiguous_blocks"] is not True:
        raise AssertionError("Any frame gap must split a CAVE block")
    if int(config["primary_embedding_dimension"]) != 5120:
        raise AssertionError("Embedding dimension changed")
    if int(config["scalar_feature_count"]) != expected_scalar_count():
        raise AssertionError("Scalar feature count changed")
    for key in ("labels_forbidden", "training_forbidden", "manifest_rescan_forbidden"):
        if config[key] is not True:
            raise AssertionError(f"{key} must remain true")


def _expected_for_mode(config: dict[str, Any], mode: str) -> tuple[str | None, dict[str, int] | None]:
    if mode == "full_train":
        return "Train", dict(config["expected_counts"]["train"])
    if mode == "full_valid":
        return "Valid", dict(config["expected_counts"]["valid"])
    return None, None


def _existing_cache_is_compatible(
    directory: Path,
    frozen_config_hash: str,
    plan: PhasePlan,
    checkpoint_sha256: str,
) -> bool:
    success_path = directory / ".SUCCESS.json"
    metadata_path = directory / "metadata.json"
    if not success_path.is_file() or not metadata_path.is_file():
        return False
    success = json.loads(success_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checks = [
        success.get("frozen_config_hash") == frozen_config_hash,
        success.get("checkpoint_sha256") == checkpoint_sha256,
        metadata.get("frame_list_hash") == plan.frame_list_hash,
        metadata.get("manifest_sha256") == plan.manifest_sha256,
        metadata.get("embedding_dim") == 5120,
        metadata.get("scalar_dim") == expected_scalar_count(),
    ]
    if not all(checks):
        raise RuntimeError(f"Existing cache is incompatible and must not be silently reused: {directory}")
    required = (
        "embedding_5120.npy", "embedding_views_5120.npz", "f4_last_ensemble.fp16.npy",
        "f5_last_ensemble.fp16.npy", "phase_trajectories_16.fp16.npz",
        "probabilities_original.fp16.npz", "curves.npz", "scalar_features.json",
        "metadata.json", "qc.json", ".SUCCESS.json",
    )
    return all((directory / name).is_file() for name in required)


def process_model_view(
    extractor: CAVEHookExtractor,
    model_frames: np.ndarray,
    positions: np.ndarray,
    activity_model: torch.Tensor,
    fov_model: torch.Tensor,
    device: torch.device,
    amp: bool,
    trajectory_length: int,
) -> dict[str, Any]:
    tensor = _to_tensor(model_frames[positions], device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=amp
    ):
        output = extractor(tensor)
    torch.cuda.synchronize(device)
    runtime = time.perf_counter() - start
    artery = torch.sigmoid(output.logits[:, 0:1]).float() * fov_model
    vein = torch.sigmoid(output.logits[:, 1:2]).float() * fov_model
    vessel = torch.maximum(artery, vein)
    vessel_union = (1.0 - (1.0 - artery) * (1.0 - vein)).clamp(0, 1)
    primary, auxiliary, pool_qc = build_embedding_bank(
        output.f4_last, output.f5_last, artery, vein, activity_model, fov_model
    )
    trajectories: dict[str, np.ndarray] = {}
    for scale, sequence in (("f4", output.f4_sequence), ("f5", output.f5_sequence)):
        for region, weight in (
            ("global", None), ("vessel", vessel), ("artery", artery), ("vein", vein),
            ("active_vessel", vessel * activity_model),
        ):
            pooled = pool_trajectory(sequence, weight)[0].float().cpu().numpy()
            trajectories[f"{scale}_{region}_native"] = pooled.astype(np.float16)
            trajectories[f"{scale}_{region}_{trajectory_length}"] = resample_trajectory(
                pooled, trajectory_length
            ).astype(np.float16)
    result = {
        "embedding": primary[0].float().cpu().numpy().astype(np.float32),
        "auxiliary": {
            name: value[0].float().cpu().numpy().astype(np.float32)
            for name, value in auxiliary.items()
        },
        "f4_last": output.f4_last[0].detach().float().cpu().numpy().astype(np.float16),
        "f5_last": output.f5_last[0].detach().float().cpu().numpy().astype(np.float16),
        "artery": artery[0, 0].cpu().numpy().astype(np.float32),
        "vein": vein[0, 0].cpu().numpy().astype(np.float32),
        "vessel": vessel[0, 0].cpu().numpy().astype(np.float32),
        "vessel_union": vessel_union[0, 0].cpu().numpy().astype(np.float32),
        "trajectories": trajectories,
        "qc": {
            **pool_qc,
            "runtime_seconds": runtime,
            "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
            "n_input_frames": int(len(positions)),
            "probability_min": float(min(artery.min().item(), vein.min().item())),
            "probability_max": float(max(artery.max().item(), vein.max().item())),
        },
    }
    del tensor, output, artery, vein, vessel, vessel_union, primary, auxiliary
    torch.cuda.empty_cache()
    return result


def _aggregate_block_results(
    block_results: list[dict[str, Any]],
    view_name: str,
    trajectory_length: int,
) -> dict[str, Any]:
    weights = np.asarray([len(block["indices"]) for block in block_results], dtype=np.float64)
    weights /= weights.sum()
    view_results = [block["views"][view_name] for block in block_results]
    output: dict[str, Any] = {
        "embedding": _weighted_average([item["embedding"] for item in view_results], weights),
        "f4_last": _weighted_average([item["f4_last"] for item in view_results], weights),
        "f5_last": _weighted_average([item["f5_last"] for item in view_results], weights),
    }
    for map_name in ("artery", "vein", "vessel", "vessel_union"):
        output[map_name] = _weighted_average([item[map_name] for item in view_results], weights)
    output["trajectories"] = {}
    for scale in TRAJECTORY_SCALES:
        for region in TRAJECTORY_REGIONS:
            key = f"{scale}_{region}_{trajectory_length}"
            output["trajectories"][key] = _weighted_average(
                [item["trajectories"][key] for item in view_results], weights
            )
    return output


def _phase_output_dir(output_root: Path, plan: PhasePlan) -> Path:
    return output_root / plan.split.casefold() / plan.patient_id / plan.series_uid / plan.phase


def process_phase(
    args: argparse.Namespace,
    extractor: CAVEHookExtractor,
    v3: V3Bridge,
    plan: PhasePlan,
    provenance: dict[str, Any],
    schema_path: Path,
) -> dict[str, Any]:
    output_dir = _phase_output_dir(args.output_root, plan)
    if output_dir.exists() and not args.overwrite and _existing_cache_is_compatible(
        output_dir, args.frozen_config_hash, plan, provenance["checkpoint_sha256"]
    ):
        return {"status": "skipped", "series_uid": plan.series_uid, "phase": plan.phase}

    frames = load_gray_frames(plan.frame_paths, num_workers=args.io_workers)
    if frames.shape[0] != len(plan.frame_indices):
        raise AssertionError("Loaded frame count differs from frozen plan")
    preprocessing = v3.preprocess(frames)
    transform = make_square_transform(frames, args.image_size)
    model_frames = frames_to_model(frames, transform)
    activity_model_np = map_original_to_model(preprocessing["activity"], transform)
    activity_values = activity_model_np[map_original_to_model(
        preprocessing["fov"].astype(np.float32), transform, cv2.INTER_NEAREST
    ) >= 0.5]
    if len(activity_values):
        low, high = np.percentile(activity_values, [5, 99])
        activity_model_np = np.clip(
            (activity_model_np - low) / max(float(high - low), 1e-8), 0, 1
        )
    fov_model_np = map_original_to_model(
        preprocessing["fov"].astype(np.float32), transform, cv2.INTER_NEAREST
    )
    activity_model = torch.from_numpy(activity_model_np.astype(np.float32))[None, None].to(args.device)
    fov_model = torch.from_numpy((fov_model_np >= 0.5).astype(np.float32))[None, None].to(args.device)

    block_positions = strict_contiguous_blocks(plan.frame_indices)
    if not block_positions:
        raise AssertionError("No temporal blocks")
    block_results: list[dict[str, Any]] = []
    for block_number, positions_global in enumerate(block_positions):
        block_frames = model_frames[positions_global]
        views = temporal_views(block_frames, args.max_len)
        unique: dict[tuple[int, ...], dict[str, Any]] = {}
        view_results: dict[str, dict[str, Any]] = {}
        for view_name, positions_local in views.items():
            key = tuple(int(value) for value in positions_local)
            if key not in unique:
                unique[key] = process_model_view(
                    extractor=extractor,
                    model_frames=block_frames,
                    positions=positions_local,
                    activity_model=activity_model,
                    fov_model=fov_model,
                    device=args.device,
                    amp=args.amp,
                    trajectory_length=args.trajectory_length,
                )
            view_results[view_name] = unique[key]
        block_results.append({
            "block_number": block_number,
            "positions_global": positions_global.tolist(),
            "indices": [plan.frame_indices[int(position)] for position in positions_global],
            "view_positions_local": {name: values.tolist() for name, values in views.items()},
            "view_indices": {
                name: [plan.frame_indices[int(positions_global[int(local)])] for local in values]
                for name, values in views.items()
            },
            "views": view_results,
        })

    phase_views = {
        view_name: _aggregate_block_results(block_results, view_name, args.trajectory_length)
        for view_name in args.temporal_view_names
    }
    ensemble_embedding = np.mean(
        np.stack([phase_views[name]["embedding"] for name in args.temporal_view_names]), axis=0
    ).astype(np.float32)
    if ensemble_embedding.shape != (5120,) or not np.isfinite(ensemble_embedding).all():
        raise FloatingPointError("Invalid phase embedding")
    ensemble_f4 = np.mean(
        np.stack([phase_views[name]["f4_last"] for name in args.temporal_view_names]), axis=0
    ).astype(np.float16)
    ensemble_f5 = np.mean(
        np.stack([phase_views[name]["f5_last"] for name in args.temporal_view_names]), axis=0
    ).astype(np.float16)
    ensemble_maps_model = {
        key: np.mean(np.stack([phase_views[name][key] for name in args.temporal_view_names]), axis=0).astype(np.float32)
        for key in ("artery", "vein", "vessel", "vessel_union")
    }
    ensemble_maps_original = {
        key: np.clip(map_model_to_original(value, transform), 0, 1).astype(np.float32)
        for key, value in ensemble_maps_model.items()
    }
    phase_trajectories = {}
    for key in next(iter(phase_views.values()))["trajectories"]:
        phase_trajectories[key] = np.mean(
            np.stack([phase_views[name]["trajectories"][key] for name in args.temporal_view_names]),
            axis=0,
        ).astype(np.float16)
        for view_name in args.temporal_view_names:
            phase_trajectories[f"{view_name}__{key}"] = phase_views[view_name]["trajectories"][key].astype(np.float16)

    scalar, curves, scalar_qc = build_scalar_bank(
        preprocessing["enhancement"], preprocessing["fov"], preprocessing["activity"],
        ensemble_maps_original["artery"], ensemble_maps_original["vein"],
        ensemble_maps_original["vessel"], ensemble_maps_original["vessel_union"],
        v3, list(plan.frame_indices),
    )
    schema = ensure_schema(schema_path, sorted(scalar.keys()), args.frozen_config_hash)
    finite_scalars, nonfinite_scalars = _finite_count(scalar)

    view_a = phase_views[args.temporal_view_names[0]]["embedding"]
    view_b = phase_views[args.temporal_view_names[1]]["embedding"]
    norm_a, norm_b = float(np.linalg.norm(view_a)), float(np.linalg.norm(view_b))
    view_cosine = float(np.dot(view_a, view_b) / max(norm_a * norm_b, 1e-8))
    view_l2 = float(np.linalg.norm(view_a - view_b))
    all_runtimes = [
        item["qc"]["runtime_seconds"]
        for block in block_results for item in block["views"].values()
    ]
    all_peak_memory = [
        item["qc"]["peak_gpu_memory_mb"]
        for block in block_results for item in block["views"].values()
    ]

    with atomic_directory(output_dir, overwrite=args.overwrite) as temporary:
        np.save(temporary / "embedding_5120.npy", ensemble_embedding)
        save_npz(
            temporary / "embedding_views_5120.npz",
            {name: phase_views[name]["embedding"].astype(np.float32) for name in args.temporal_view_names},
        )
        np.save(temporary / "f4_last_ensemble.fp16.npy", ensemble_f4)
        np.save(temporary / "f5_last_ensemble.fp16.npy", ensemble_f5)
        save_npz(temporary / "phase_trajectories_16.fp16.npz", phase_trajectories)
        save_npz(
            temporary / "probabilities_original.fp16.npz",
            {key: value.astype(np.float16) for key, value in ensemble_maps_original.items()},
        )
        save_npz(temporary / "curves.npz", {key: value.astype(np.float32) for key, value in curves.items()})
        write_json_atomic(temporary / "scalar_features.json", scalar)

        blocks_dir = temporary / "blocks"
        for block in block_results:
            for view_name, result in block["views"].items():
                view_dir = blocks_dir / f"block{block['block_number']:02d}" / view_name
                view_dir.mkdir(parents=True, exist_ok=True)
                np.save(view_dir / "embedding_5120.npy", result["embedding"])
                save_npz(view_dir / "auxiliary_embeddings.npz", result["auxiliary"])
                np.save(view_dir / "f4_last.fp16.npy", result["f4_last"])
                np.save(view_dir / "f5_last.fp16.npy", result["f5_last"])
                save_npz(view_dir / "trajectories.fp16.npz", result["trajectories"])
                save_npz(
                    view_dir / "probabilities_512.fp16.npz",
                    {key: result[key].astype(np.float16) for key in ("artery", "vein", "vessel", "vessel_union")},
                )
                write_json_atomic(view_dir / "qc.json", result["qc"])

        save_input_mosaic(temporary / "input_mosaic.jpg", frames, plan.frame_indices)
        background = np.max(preprocessing["enhancement"], axis=0)
        save_av_overlay(
            temporary / "artery_vein_overlay.png", background,
            ensemble_maps_original["artery"], ensemble_maps_original["vein"],
        )
        for key in ("artery", "vein", "vessel", "vessel_union"):
            save_probability_png(temporary / f"{key}_probability.png", ensemble_maps_original[key])

        qc = {
            **preprocessing["qc"], **scalar_qc,
            "n_frames": len(plan.frame_indices),
            "n_blocks": len(block_positions),
            "n_singleton_blocks": int(sum(len(block) == 1 for block in block_positions)),
            "n_frame_gaps": int(sum(b - a > 1 for a, b in zip(plan.frame_indices[:-1], plan.frame_indices[1:]))),
            "max_frame_gap": int(max([b - a for a, b in zip(plan.frame_indices[:-1], plan.frame_indices[1:])] or [1])),
            "view_embedding_l2": view_l2,
            "view_embedding_cosine": view_cosine,
            "embedding_min": float(ensemble_embedding.min()),
            "embedding_max": float(ensemble_embedding.max()),
            "embedding_mean": float(ensemble_embedding.mean()),
            "embedding_std": float(ensemble_embedding.std()),
            "embedding_norm": float(np.linalg.norm(ensemble_embedding)),
            "nonfinite_embedding_count": int((~np.isfinite(ensemble_embedding)).sum()),
            "finite_scalar_count": finite_scalars,
            "nonfinite_scalar_count": nonfinite_scalars,
            "artery_probability_mean_fov": float(ensemble_maps_original["artery"][preprocessing["fov"]].mean()),
            "vein_probability_mean_fov": float(ensemble_maps_original["vein"][preprocessing["fov"]].mean()),
            "vessel_probability_mean_fov": float(ensemble_maps_original["vessel"][preprocessing["fov"]].mean()),
            "runtime_seconds_sum": float(sum(all_runtimes)),
            "peak_gpu_memory_mb_max": float(max(all_peak_memory)),
        }
        write_json_atomic(temporary / "qc.json", qc)
        metadata = {
            "patient_id": plan.patient_id,
            "series_uid": plan.series_uid,
            "series_id": plan.series_id,
            "source_type": plan.source_type,
            "split": plan.split,
            "phase": plan.phase,
            "frame_indices": list(plan.frame_indices),
            "frame_list_hash": plan.frame_list_hash,
            "manifest_sha256": plan.manifest_sha256,
            "original_shape": list(frames.shape),
            "transform": transform.to_json(),
            "blocks": [
                {
                    "block_number": block["block_number"],
                    "indices": block["indices"],
                    "view_indices": block["view_indices"],
                }
                for block in block_results
            ],
            "primary_embedding_blocks": list(PRIMARY_BLOCKS),
            "embedding_dim": 5120,
            "scalar_dim": len(scalar),
            "feature_schema_sha256": schema["schema_sha256"],
            "frozen_config_hash": args.frozen_config_hash,
            "provenance": provenance,
            "labels_read": False,
            "model_trained": False,
            "manifest_rescanned": False,
        }
        write_json_atomic(temporary / "metadata.json", metadata)
        embedding_sha = sha256_file(temporary / "embedding_5120.npy")
        write_json_atomic(temporary / ".SUCCESS.json", {
            "status": "success",
            "completed_at": utc_now(),
            "frozen_config_hash": args.frozen_config_hash,
            "feature_schema_sha256": schema["schema_sha256"],
            "checkpoint_sha256": provenance["checkpoint_sha256"],
            "manifest_sha256": plan.manifest_sha256,
            "frame_list_hash": plan.frame_list_hash,
            "embedding_sha256": embedding_sha,
        })
    return {
        "status": "ok", "patient_id": plan.patient_id, "series_uid": plan.series_uid,
        "phase": plan.phase, "frames": len(plan.frame_indices), "blocks": len(block_positions),
        "runtime_seconds": float(sum(all_runtimes)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full_train", "full_valid", "custom"), default="custom")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cave-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--v3-extractor", type=Path, required=True)
    parser.add_argument("--v3-base-config", type=Path, required=True)
    parser.add_argument("--v3-override-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--release-freeze", type=Path)
    parser.add_argument("--max-series", type=int)
    parser.add_argument("--series-uid", action="append", default=[])
    parser.add_argument("--io-workers", type=int, default=4)
    parser.add_argument("--no-verify-files", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.report_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logger = RunLogger(args.report_root / f"extract_{args.mode}_{stamp}_{os.getpid()}.log")
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required by the official CAVE ConvGRU implementation")
        args.device = torch.device("cuda:0")
        frozen = json.loads(args.frozen_config.read_text(encoding="utf-8"))
        _validate_frozen_config(frozen)
        args.frozen_config_hash = sha256_json(frozen)
        args.image_size = int(frozen["image_size"])
        args.max_len = int(frozen["max_sequence_length"])
        args.trajectory_length = int(frozen.get("trajectory_resampled_length", 16))
        args.temporal_view_names = tuple(frozen["temporal_views"])
        args.amp = bool(frozen["amp"]) and not args.no_amp

        if args.checkpoint.stat().st_size != EXPECTED_CKPT_SIZE:
            raise AssertionError("CAVE checkpoint size mismatch")
        if sha256_file(args.checkpoint) != EXPECTED_CKPT_SHA:
            raise AssertionError("CAVE checkpoint SHA256 mismatch")
        current_commit = git_commit(args.cave_repo)
        if current_commit != frozen["cave_commit"]:
            raise AssertionError(f"CAVE commit mismatch: expected={frozen['cave_commit']} actual={current_commit}")
        if git_is_dirty(args.cave_repo):
            raise AssertionError("CAVE repository is dirty; refuse non-reproducible extraction")

        expected_split, expected_counts = _expected_for_mode(frozen, args.mode)
        requested = set(args.series_uid) if args.series_uid else None
        bundle = load_manifest(
            args.manifest,
            expected_split=expected_split,
            verify_files=not args.no_verify_files,
            expected_counts=expected_counts,
            max_series=args.max_series,
            requested_series_uids=requested,
        )
        logger.log(f"Manifest summary: {bundle.summary}")

        package_root = Path(__file__).resolve().parent
        schema_path = args.output_root / "feature_schema.json"
        provenance = {
            "checkpoint_path": str(args.checkpoint.resolve()),
            "checkpoint_sha256": EXPECTED_CKPT_SHA,
            "checkpoint_size": EXPECTED_CKPT_SIZE,
            "cave_repo": str(args.cave_repo.resolve()),
            "cave_commit": current_commit,
            "cave_code_tree_sha256": cave_code_tree_hash(args.cave_repo),
            "feature_package_tree_sha256": sha256_tree(package_root),
            "frozen_config_path": str(args.frozen_config.resolve()),
            "frozen_config_hash": args.frozen_config_hash,
            "manifest_path": str(args.manifest.resolve()),
            "manifest_sha256": bundle.summary["manifest_sha256"],
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        }
        v3 = V3Bridge(
            args.v3_extractor, args.v3_base_config, args.v3_override_config,
            expected_hashes=frozen.get("v3_expected_hashes"),
        )
        provenance.update(v3.provenance)

        if args.mode == "full_valid":
            if args.release_freeze is None:
                raise ValueError("--release-freeze is required for full_valid")
            required = {
                "extractor": Path(__file__).resolve(),
                "package_tree": Path(__file__).resolve().parent,
                "frozen_config": args.frozen_config.resolve(),
                "train_manifest": Path(frozen["train_manifest"]).resolve(),
                "valid_manifest": args.manifest.resolve(),
                "feature_schema": schema_path.resolve(),
                "v3_extractor": args.v3_extractor.resolve(),
                "v3_base_config": args.v3_base_config.resolve(),
                "v3_override_config": args.v3_override_config.resolve(),
            }
            verify_release(args.release_freeze, required, args.cave_repo, args.checkpoint)
            logger.log(f"Valid release freeze verified: {args.release_freeze}")

        model, extractor = load_cave_model(args.cave_repo, args.checkpoint, args.device)
        del model
        args.output_root.mkdir(parents=True, exist_ok=True)
        run_rows: list[dict[str, Any]] = []
        failures = 0
        for number, plan in enumerate(bundle.plans, 1):
            logger.log(f"[{number}/{len(bundle.plans)}] {plan.series_uid} {plan.phase}")
            try:
                result = process_phase(args, extractor, v3, plan, provenance, schema_path)
            except Exception as exc:
                failures += 1
                result = {
                    "status": "failed", "patient_id": plan.patient_id,
                    "series_uid": plan.series_uid, "phase": plan.phase,
                    "error": repr(exc), "traceback": traceback.format_exc(),
                }
                failure_dir = args.output_root / "_failures"
                failure_dir.mkdir(parents=True, exist_ok=True)
                write_json_atomic(failure_dir / f"{plan.series_uid}__{plan.phase}.json", result)
                logger.log(f"FAILED {plan.series_uid} {plan.phase}: {exc!r}")
            run_rows.append(result)
            write_csv_atomic(pd.DataFrame(run_rows), args.output_root / "run_index.csv")
        extractor.close()

        status_counts = pd.Series([row["status"] for row in run_rows]).value_counts(dropna=False).to_dict()
        summary = {
            "mode": args.mode,
            "started_manifest_summary": bundle.summary,
            "planned_phases": len(bundle.plans),
            "status_counts": {str(k): int(v) for k, v in status_counts.items()},
            "failures": failures,
            "cuda_actually_used": True,
            "cpu_fallback": False,
            "labels_read": False,
            "model_trained": False,
            "manifest_rescanned": False,
            "provenance": provenance,
            "completed_at": utc_now(),
        }
        write_json_atomic(args.output_root / f"run_summary_{args.mode}.json", summary)
        logger.log(f"Run summary: {summary['status_counts']}")
        return 2 if failures else 0
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())

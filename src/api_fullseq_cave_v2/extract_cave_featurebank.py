from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from cave_model import git_commit, load_cave_model
from io_ops import (
    atomic_directory, frames_to_model, hash_lines, load_gray_frames,
    make_square_transform, map_model_to_original, map_original_to_model,
    parse_pipe_ints, parse_pipe_strings, sha256_file, sha256_json,
    split_large_gap_blocks, temporal_views, write_json_atomic,
)
from pooling import build_embedding_bank, pool_trajectory, resample_trajectory
from scalar_features import build_scalar_bank, expected_scalar_count
from v3_bridge import V3Bridge


EXPECTED_CKPT_SHA = "c90b7e066e32039cf61352993a9c57784caac6aa1fdb042dc4801df6dc729651"
EXPECTED_CKPT_SIZE = 332_731_061
PRIMARY_BLOCKS = (
    "f5_global_mean", "f5_vessel_mean", "f5_artery_mean", "f5_vein_mean",
    "f5_active_vessel_mean", "f5_vessel_top10_abs_magnitude",
    "f4_vessel_mean", "f4_artery_mean", "f4_active_vessel_mean",
    "f4_vessel_top10_abs_magnitude",
)


def to_tensor(frames01: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(frames01[:, None]).unsqueeze(0).to(device=device, dtype=torch.float32)


def _save_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **values)


def process_model_view(
    extractor,
    model_frames: np.ndarray,
    positions: np.ndarray,
    activity_model: torch.Tensor,
    fov_model: torch.Tensor,
    device: torch.device,
    amp: bool,
) -> dict:
    x = to_tensor(model_frames[positions], device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        output = extractor(x)
    runtime = time.perf_counter() - start
    artery = torch.sigmoid(output.logits[:, 0:1]).float() * fov_model
    vein = torch.sigmoid(output.logits[:, 1:2]).float() * fov_model
    vessel = torch.maximum(artery, vein)
    vessel_or = (1.0 - (1.0 - artery) * (1.0 - vein)).clamp(0, 1)
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
            trajectories[f"{scale}_{region}_16"] = resample_trajectory(pooled, 16).astype(np.float16)
    result = {
        "embedding": primary[0].float().cpu().numpy().astype(np.float32),
        "auxiliary": {name: value[0].float().cpu().numpy().astype(np.float32) for name, value in auxiliary.items()},
        "f4_last": output.f4_last[0].detach().cpu().numpy().astype(np.float16),
        "f5_last": output.f5_last[0].detach().cpu().numpy().astype(np.float16),
        "artery": artery[0, 0].cpu().numpy().astype(np.float32),
        "vein": vein[0, 0].cpu().numpy().astype(np.float32),
        "vessel": vessel[0, 0].cpu().numpy().astype(np.float32),
        "vessel_or": vessel_or[0, 0].cpu().numpy().astype(np.float32),
        "trajectories": trajectories,
        "qc": {
            **pool_qc,
            "runtime_seconds": runtime,
            "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        },
    }
    del x, output, artery, vein, vessel, vessel_or, primary, auxiliary
    return result


def _weighted_average(arrays: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    return np.sum(np.stack(arrays).astype(np.float32) * weights.reshape((-1,) + (1,) * arrays[0].ndim), axis=0)


def process_phase(args, extractor, v3, row: pd.Series, phase: str, provenance: dict) -> dict:
    paths = parse_pipe_strings(row.get(f"{phase}_frame_paths"))
    indices = parse_pipe_ints(row.get(f"{phase}_frame_indices"))
    if not paths:
        return {"status": "missing", "phase": phase}
    if len(paths) != len(indices):
        raise ValueError("Frozen frame path/index mismatch")
    if hash_lines(paths) != str(row.get(f"{phase}_frame_list_hash", "")):
        raise AssertionError(f"Frozen frame list hash mismatch: {row['series_uid']} {phase}")

    frames = load_gray_frames(paths)
    prep = v3.preprocess(frames)
    transform = make_square_transform(frames, args.image_size)
    model_frames = frames_to_model(frames, transform)
    activity_model_np = map_original_to_model(prep["activity"], transform)
    fov_model_np = map_original_to_model(prep["fov"].astype(np.float32), transform, cv2.INTER_NEAREST)
    activity_model = torch.from_numpy(activity_model_np)[None, None].to(args.device)
    fov_model = torch.from_numpy((fov_model_np >= 0.5).astype(np.float32))[None, None].to(args.device)

    blocks = split_large_gap_blocks(indices, paths, args.max_missing_frames)
    block_results = []
    for block_number, (block_indices, _block_paths, block_positions) in enumerate(blocks):
        block_frames = model_frames[block_positions]
        views = temporal_views(block_frames, args.max_len)
        unique: dict[tuple[int, ...], dict] = {}
        view_results: dict[str, dict] = {}
        for view_name, positions in views.items():
            key = tuple(int(v) for v in positions)
            if key not in unique:
                unique[key] = process_model_view(
                    extractor, block_frames, positions, activity_model, fov_model, args.device, args.amp
                )
            view_results[view_name] = unique[key]
        block_results.append({
            "block_number": block_number,
            "indices": block_indices,
            "positions": block_positions,
            "views": view_results,
            "view_positions": {name: value.tolist() for name, value in views.items()},
        })

    block_weights = np.asarray([len(block["indices"]) for block in block_results], dtype=np.float64)
    block_weights /= block_weights.sum()
    phase_view_embeddings: dict[str, np.ndarray] = {}
    phase_view_maps: dict[str, dict[str, np.ndarray]] = {}
    for view_name in ("uniform_full20", "contrast_core20"):
        phase_view_embeddings[view_name] = _weighted_average(
            [block["views"][view_name]["embedding"] for block in block_results], block_weights
        ).astype(np.float32)
        phase_view_maps[view_name] = {}
        for map_name in ("artery", "vein", "vessel", "vessel_or"):
            phase_view_maps[view_name][map_name] = _weighted_average(
                [block["views"][view_name][map_name] for block in block_results], block_weights
            ).astype(np.float32)
    ensemble_embedding = np.mean(np.stack(list(phase_view_embeddings.values())), axis=0).astype(np.float32)
    ensemble_maps_model = {
        name: np.mean(np.stack([phase_view_maps[v][name] for v in phase_view_maps]), axis=0).astype(np.float32)
        for name in ("artery", "vein", "vessel", "vessel_or")
    }
    ensemble_maps_original = {
        name: np.clip(map_model_to_original(value, transform), 0, 1)
        for name, value in ensemble_maps_model.items()
    }
    scalar, curves, scalar_qc = build_scalar_bank(
        prep["enhancement"], prep["fov"], prep["activity"],
        ensemble_maps_original["artery"], ensemble_maps_original["vein"],
        ensemble_maps_original["vessel"], ensemble_maps_original["vessel_or"],
        v3, indices,
    )
    if len(scalar) != expected_scalar_count():
        raise AssertionError(f"Scalar schema count {len(scalar)} != {expected_scalar_count()}")

    out_dir = args.output_root / str(row["split"]).lower() / str(row["patient_id"]) / str(row["series_uid"]) / phase
    with atomic_directory(out_dir, overwrite=args.overwrite) as temp:
        np.save(temp / "embedding_5120.npy", ensemble_embedding)
        _save_npz(temp / "embedding_views_5120.npz", phase_view_embeddings)
        _save_npz(temp / "probabilities_original.fp16.npz", {k: v.astype(np.float16) for k, v in ensemble_maps_original.items()})
        _save_npz(temp / "curves.npz", {k: v.astype(np.float32) for k, v in curves.items()})
        write_json_atomic(temp / "scalar_features.json", scalar)

        blocks_dir = temp / "blocks"
        for block in block_results:
            for view_name, result in block["views"].items():
                view_dir = blocks_dir / f"block{block['block_number']:02d}" / view_name
                view_dir.mkdir(parents=True, exist_ok=True)
                np.save(view_dir / "embedding_5120.npy", result["embedding"])
                _save_npz(view_dir / "auxiliary_embeddings.npz", result["auxiliary"])
                np.save(view_dir / "f4_last.fp16.npy", result["f4_last"])
                np.save(view_dir / "f5_last.fp16.npy", result["f5_last"])
                _save_npz(view_dir / "trajectories.fp16.npz", result["trajectories"])
                _save_npz(view_dir / "probabilities_512.fp16.npz", {
                    k: result[k].astype(np.float16) for k in ("artery", "vein", "vessel", "vessel_or")
                })
                write_json_atomic(view_dir / "qc.json", result["qc"])

        view_l2 = float(np.linalg.norm(
            phase_view_embeddings["uniform_full20"] - phase_view_embeddings["contrast_core20"]
        ))
        view_cos = float(np.dot(
            phase_view_embeddings["uniform_full20"], phase_view_embeddings["contrast_core20"]
        ) / max(
            np.linalg.norm(phase_view_embeddings["uniform_full20"]) *
            np.linalg.norm(phase_view_embeddings["contrast_core20"]), 1e-8
        ))
        qc = {
            **prep["qc"], **scalar_qc,
            "n_frames": len(indices),
            "n_blocks": len(blocks),
            "n_frame_gaps": int(sum(max(indices[i] - indices[i - 1] - 1, 0) > 0 for i in range(1, len(indices)))),
            "max_frame_gap": int(max([indices[i] - indices[i - 1] for i in range(1, len(indices))] or [1])),
            "view_embedding_l2": view_l2,
            "view_embedding_cosine": view_cos,
            "nonfinite_embedding_count": int((~np.isfinite(ensemble_embedding)).sum()),
            "nonfinite_scalar_count": int(sum(not np.isfinite(v) for v in scalar.values())),
        }
        write_json_atomic(temp / "qc.json", qc)
        metadata = {
            "patient_id": str(row["patient_id"]), "series_uid": str(row["series_uid"]),
            "split": str(row["split"]), "phase": phase,
            "frame_indices": indices, "frame_paths_hash": hash_lines(paths),
            "frame_list_hash": str(row.get(f"{phase}_frame_list_hash", "")),
            "transform": transform.to_json(),
            "blocks": [{
                "indices": b["indices"], "view_positions": b["view_positions"]
            } for b in block_results],
            "primary_embedding_blocks": list(PRIMARY_BLOCKS),
            "embedding_dim": 5120,
            "scalar_dim": len(scalar),
            "provenance": provenance,
            "frozen_config_hash": args.frozen_config_hash,
        }
        write_json_atomic(temp / "metadata.json", metadata)
        write_json_atomic(temp / ".SUCCESS.json", {
            "status": "success", "frozen_config_hash": args.frozen_config_hash,
            "embedding_sha256": sha256_file(temp / "embedding_5120.npy"),
        })
    return {"status": "ok", "series_uid": str(row["series_uid"]), "phase": phase}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cave-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--v3-extractor", type=Path, required=True)
    parser.add_argument("--v3-base-config", type=Path, required=True)
    parser.add_argument("--v3-override-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--max-series", type=int)
    parser.add_argument("--max-len", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-missing-frames", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.device = torch.device("cuda:0")
    args.amp = not args.no_amp
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the official CAVE ConvGRU implementation")
    if args.checkpoint.stat().st_size != EXPECTED_CKPT_SIZE or sha256_file(args.checkpoint) != EXPECTED_CKPT_SHA:
        raise AssertionError("CAVE checkpoint size/SHA mismatch")
    frozen = json.loads(args.frozen_config.read_text(encoding="utf-8"))
    args.frozen_config_hash = sha256_json(frozen)
    model, extractor = load_cave_model(args.cave_repo, args.checkpoint, args.device)
    v3 = V3Bridge(args.v3_extractor, args.v3_base_config, args.v3_override_config)
    provenance = {
        "checkpoint_sha256": EXPECTED_CKPT_SHA,
        "checkpoint_size": EXPECTED_CKPT_SIZE,
        "cave_commit": git_commit(args.cave_repo),
        "extractor_code_sha256": sha256_file(Path(__file__)),
        **v3.provenance,
    }
    manifest = pd.read_csv(args.manifest, dtype={"patient_id": str})
    manifest = manifest.loc[manifest["selected_for_extraction"].astype(bool)].copy()
    if args.max_series:
        manifest = manifest.head(args.max_series)
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_rows, failures = [], 0
    for number, (_, row) in enumerate(manifest.iterrows(), 1):
        print(f"[{number}/{len(manifest)}] {row['series_uid']}", flush=True)
        for phase in ("pre", "post"):
            final_dir = args.output_root / str(row["split"]).lower() / str(row["patient_id"]) / str(row["series_uid"]) / phase
            success = final_dir / ".SUCCESS.json"
            if success.exists() and not args.overwrite:
                payload = json.loads(success.read_text(encoding="utf-8"))
                if payload.get("frozen_config_hash") != args.frozen_config_hash:
                    raise RuntimeError(f"Existing cache uses another frozen config: {final_dir}")
                run_rows.append({"status": "skipped", "series_uid": row["series_uid"], "phase": phase})
                continue
            try:
                result = process_phase(args, extractor, v3, row, phase, provenance)
            except Exception as exc:
                failures += 1
                result = {"status": "failed", "series_uid": str(row["series_uid"]), "phase": phase, "error": repr(exc)}
                failure_dir = args.output_root / "_failures"
                failure_dir.mkdir(exist_ok=True)
                write_json_atomic(failure_dir / f"{row['series_uid']}__{phase}.json", result)
            run_rows.append(result)
            pd.DataFrame(run_rows).to_csv(args.output_root / "run_index.csv", index=False)
    extractor.close()
    print(pd.Series([r["status"] for r in run_rows]).value_counts(dropna=False))
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

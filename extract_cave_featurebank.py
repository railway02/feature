from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from cave_feature_model import load_cave_av_convgru, verify_wrapper_equivalence
from feature_ops import (
    adaptive_enhancement,
    build_embedding_bank,
    curve_features,
    load_gray_frames,
    make_temporal_views,
    normalize_activity,
    pad_square_resize,
    parse_pipe_ints,
    parse_pipe_strings,
    pool_trajectory,
    probability_shape_features,
    save_json,
    split_contiguous_blocks,
    weighted_curve,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resize_np(image: np.ndarray, size: int = 256) -> np.ndarray:
    return cv2.resize(image.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)


def to_model_tensor(frames01: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(frames01[:, None]).unsqueeze(0).to(device=device, dtype=torch.float32)


def process_block(
    wrapper,
    frames_u8: np.ndarray,
    device: torch.device,
    amp: bool,
    max_len: int,
) -> dict:
    frames01 = pad_square_resize(frames_u8, 512)
    enhancement, polarity_meta = adaptive_enhancement(frames01)
    activity_np = normalize_activity(enhancement)
    activity = torch.from_numpy(activity_np)[None, None].to(device)
    views = make_temporal_views(frames01, max_len=max_len)

    view_embeddings = []
    view_maps = []
    f4_maps = []
    f5_maps = []
    trajectories: dict[str, np.ndarray] = {}
    view_names = []

    for view_name, positions in views.items():
        view_names.append(view_name)
        x = to_model_tensor(frames01[positions], device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=amp
        ):
            outputs = wrapper(x)
        artery = torch.sigmoid(outputs.logits[:, 0:1]).float()
        vein = torch.sigmoid(outputs.logits[:, 1:2]).float()
        vessel = 1.0 - (1.0 - artery) * (1.0 - vein)

        embedding, block_names = build_embedding_bank(
            outputs.f4_last.float(),
            outputs.f5_last.float(),
            artery,
            vein,
            activity,
        )
        view_embeddings.append(embedding[0].cpu().numpy().astype(np.float32))
        view_maps.append(torch.cat([artery, vein, vessel], dim=1)[0].cpu().numpy().astype(np.float32))
        f4_maps.append(outputs.f4_last[0].float().cpu().numpy().astype(np.float16))
        f5_maps.append(outputs.f5_last[0].float().cpu().numpy().astype(np.float16))

        for scale_name, sequence in (("f4", outputs.f4_sequence), ("f5", outputs.f5_sequence)):
            for region_name, weight in (
                ("global", None),
                ("vessel", vessel),
                ("artery", artery),
                ("vein", vein),
            ):
                pooled = pool_trajectory(sequence.float(), weight)[0]
                trajectories[f"{view_name}_{scale_name}_{region_name}"] = (
                    pooled.cpu().numpy().astype(np.float16)
                )

        del x, outputs, artery, vein, vessel, embedding
        torch.cuda.empty_cache()

    embedding_views = np.stack(view_embeddings, axis=0)
    maps_views = np.stack(view_maps, axis=0)
    embedding = embedding_views.mean(axis=0)
    probabilities = maps_views.mean(axis=0)
    f4_last = np.mean(np.stack(f4_maps, axis=0).astype(np.float32), axis=0).astype(np.float16)
    f5_last = np.mean(np.stack(f5_maps, axis=0).astype(np.float32), axis=0).astype(np.float16)

    artery_np, vein_np, vessel_np = probabilities
    scalar = {}
    scalar.update(probability_shape_features(artery_np, "artery"))
    scalar.update(probability_shape_features(vein_np, "vein"))
    scalar.update(probability_shape_features(vessel_np, "vessel"))
    scalar.update(curve_features(weighted_curve(enhancement, artery_np), "artery_tdc"))
    scalar.update(curve_features(weighted_curve(enhancement, vein_np), "vein_tdc"))
    scalar.update(curve_features(weighted_curve(enhancement, vessel_np), "vessel_tdc"))
    scalar["view_embedding_l2"] = float(
        np.linalg.norm(embedding_views[0] - embedding_views[-1])
    ) if len(embedding_views) > 1 else 0.0
    scalar["view_probability_mae"] = float(
        np.mean(np.abs(maps_views[0] - maps_views[-1]))
    ) if len(maps_views) > 1 else 0.0

    return {
        "embedding": embedding,
        "embedding_views": embedding_views,
        "embedding_block_names": block_names,
        "probabilities": probabilities,
        "f4_last": f4_last,
        "f5_last": f5_last,
        "trajectories": trajectories,
        "scalar": scalar,
        "polarity": polarity_meta,
        "view_names": view_names,
        "view_positions": {key: value.tolist() for key, value in views.items()},
    }


def aggregate_blocks(results: list[dict], lengths: list[int]) -> dict:
    weights = np.asarray(lengths, dtype=np.float64)
    weights /= weights.sum()

    embedding = np.sum(
        np.stack([r["embedding"] for r in results], axis=0) * weights[:, None], axis=0
    ).astype(np.float32)
    probabilities = np.sum(
        np.stack([r["probabilities"] for r in results], axis=0) * weights[:, None, None, None],
        axis=0,
    ).astype(np.float32)
    f4_last = np.sum(
        np.stack([r["f4_last"] for r in results], axis=0).astype(np.float32) * weights[:, None, None, None],
        axis=0,
    ).astype(np.float16)
    f5_last = np.sum(
        np.stack([r["f5_last"] for r in results], axis=0).astype(np.float32) * weights[:, None, None, None],
        axis=0,
    ).astype(np.float16)

    scalar_keys = sorted(set().union(*(r["scalar"].keys() for r in results)))
    scalar = {}
    for key in scalar_keys:
        values = np.asarray([r["scalar"].get(key, np.nan) for r in results], dtype=np.float64)
        valid = np.isfinite(values)
        scalar[key] = float(np.average(values[valid], weights=weights[valid])) if valid.any() else float("nan")

    trajectories = {}
    for block_index, result in enumerate(results):
        for key, value in result["trajectories"].items():
            trajectories[f"block{block_index:02d}_{key}"] = value

    return {
        "embedding": embedding,
        "probabilities": probabilities,
        "f4_last": f4_last,
        "f5_last": f5_last,
        "scalar": scalar,
        "trajectories": trajectories,
    }


def process_phase(
    wrapper,
    row: pd.Series,
    phase: str,
    out_dir: Path,
    device: torch.device,
    amp: bool,
    max_len: int,
    checkpoint_sha: str,
) -> dict:
    paths = parse_pipe_strings(row[f"{phase}_frame_paths"])
    indices = parse_pipe_ints(row[f"{phase}_frame_indices"])
    if not paths:
        return {"status": "missing", "phase": phase}
    blocks = split_contiguous_blocks(indices, paths)
    block_results = []
    block_lengths = []
    for block_indices, block_paths in blocks:
        frames = load_gray_frames(block_paths)
        result = process_block(wrapper, frames, device, amp, max_len)
        result["indices"] = block_indices
        block_results.append(result)
        block_lengths.append(len(block_indices))
    combined = aggregate_blocks(block_results, block_lengths)

    temp_dir = out_dir.with_name(out_dir.name + ".tmp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    np.save(temp_dir / "embedding_5120.npy", combined["embedding"].astype(np.float32))
    raw_view_embeddings = {}
    for block_index, result in enumerate(block_results):
        for view_index, view_name in enumerate(result["view_names"]):
            raw_view_embeddings[f"block{block_index:02d}_{view_name}"] = (
                result["embedding_views"][view_index].astype(np.float32)
            )
    np.savez_compressed(temp_dir / "embedding_views_5120.npz", **raw_view_embeddings)
    np.save(temp_dir / "f4_last.fp16.npy", combined["f4_last"])
    np.save(temp_dir / "f5_last.fp16.npy", combined["f5_last"])
    np.savez_compressed(temp_dir / "temporal_trajectories.npz", **combined["trajectories"])

    artery, vein, vessel = combined["probabilities"]
    np.savez_compressed(
        temp_dir / "probabilities_256.npz",
        artery=resize_np(artery, 256).astype(np.float16),
        vein=resize_np(vein, 256).astype(np.float16),
        vessel=resize_np(vessel, 256).astype(np.float16),
    )
    cv2.imwrite(str(temp_dir / "artery_probability.png"), np.rint(255 * artery).astype(np.uint8))
    cv2.imwrite(str(temp_dir / "vein_probability.png"), np.rint(255 * vein).astype(np.uint8))
    cv2.imwrite(str(temp_dir / "vessel_probability.png"), np.rint(255 * vessel).astype(np.uint8))

    save_json(temp_dir / "scalar_features.json", combined["scalar"])
    metadata = {
        "patient_id": str(row["patient_id"]),
        "series_uid": str(row["series_uid"]),
        "split": str(row["split"]),
        "phase": phase,
        "n_frames": len(indices),
        "n_blocks": len(blocks),
        "frame_indices": indices,
        "frame_list_hash": str(row.get(f"{phase}_frame_list_hash", "")),
        "checkpoint_sha256": checkpoint_sha,
        "embedding_dim": 5120,
        "block_details": [
            {
                "indices": result["indices"],
                "view_names": result["view_names"],
                "view_positions": result["view_positions"],
                "polarity": result["polarity"],
            }
            for result in block_results
        ],
    }
    save_json(temp_dir / "metadata.json", metadata)
    (temp_dir / ".SUCCESS").write_text("ok\n", encoding="utf-8")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    temp_dir.replace(out_dir)
    return {"status": "ok", **metadata, **combined["scalar"]}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cave-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-series", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=20)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = get_args()
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CAVE extraction requires CUDA because the official ConvGRU initializes states on CUDA")
    checkpoint_sha = sha256_file(args.checkpoint)
    official, wrapper = load_cave_av_convgru(args.cave_repo, args.checkpoint, device)

    # Tiny equivalence test before any patient data are processed.
    sample = torch.rand(1, 2, 1, 64, 64, device=device)
    max_abs = verify_wrapper_equivalence(official, wrapper, sample)
    print(f"[PASS] wrapper equivalence max_abs={max_abs:.3e}")

    manifest = pd.read_csv(args.manifest, dtype={"patient_id": str})
    manifest = manifest.loc[manifest["selected_for_extraction"].astype(bool)].copy()
    if args.max_series is not None:
        manifest = manifest.head(args.max_series)

    rows = []
    args.output_root.mkdir(parents=True, exist_ok=True)
    for count, (_, row) in enumerate(manifest.iterrows(), start=1):
        series_dir = args.output_root / str(row["split"]).lower() / str(row["patient_id"]) / str(row["series_uid"])
        print(f"[{count}/{len(manifest)}] {row['series_uid']}")
        for phase in ("pre", "post"):
            phase_dir = series_dir / phase
            if (phase_dir / ".SUCCESS").exists() and not args.overwrite:
                rows.append({"status": "skipped", "series_uid": row["series_uid"], "phase": phase})
                continue
            try:
                result = process_phase(
                    wrapper, row, phase, phase_dir, device, not args.no_amp,
                    args.max_len, checkpoint_sha,
                )
            except Exception as exc:
                phase_dir.mkdir(parents=True, exist_ok=True)
                save_json(phase_dir / "failure.json", {"error": repr(exc)})
                result = {
                    "status": "failed",
                    "series_uid": str(row["series_uid"]),
                    "patient_id": str(row["patient_id"]),
                    "split": str(row["split"]),
                    "phase": phase,
                    "error": repr(exc),
                }
            rows.append(result)
            pd.DataFrame(rows).to_csv(args.output_root / "run_index.csv", index=False)

    print(pd.Series([row.get("status") for row in rows]).value_counts(dropna=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

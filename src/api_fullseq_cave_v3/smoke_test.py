#!/usr/bin/env python3
"""GPU smoke test for the pinned CAVE checkpoint and feature hooks."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cave_model import load_cave_model
from pooling import build_embedding_bank, pool_trajectory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cave-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device("cuda:0")
    model, extractor = load_cave_model(args.cave_repo, args.checkpoint, device)
    for frames, amp in ((4, False), (20, True)):
        tensor = torch.rand(1, frames, 1, 512, 512, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=amp
        ):
            direct_logits = model(tensor)
            output = extractor(tensor)
            fov = torch.ones(1, 1, 512, 512, device=device)
            activity = torch.rand(1, 1, 512, 512, device=device)
            artery = torch.sigmoid(output.logits[:, 0:1])
            vein = torch.sigmoid(output.logits[:, 1:2])
            embedding, auxiliary, qc = build_embedding_bank(
                output.f4_last, output.f5_last, artery, vein, activity, fov
            )
            trajectory = pool_trajectory(output.f5_sequence, torch.maximum(artery, vein))
        maximum_difference = float((direct_logits - output.logits).abs().max().item())
        assert maximum_difference == 0.0
        assert output.logits.shape == (1, 2, 512, 512)
        assert output.f4_sequence.shape == (1, frames, 512, 64, 64)
        assert output.f5_sequence.shape == (1, frames, 512, 32, 32)
        assert embedding.shape == (1, 5120)
        assert trajectory.shape == (1, frames, 512)
        assert torch.isfinite(output.logits).all() and torch.isfinite(embedding).all()
        print({
            "frames": frames,
            "amp": amp,
            "logits": tuple(output.logits.shape),
            "f4_sequence": tuple(output.f4_sequence.shape),
            "f5_sequence": tuple(output.f5_sequence.shape),
            "embedding": tuple(embedding.shape),
            "trajectory": tuple(trajectory.shape),
            "official_hook_max_abs_diff": maximum_difference,
            "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
            "pool_qc": qc,
            "auxiliary_blocks": sorted(auxiliary),
        })
        del tensor, direct_logits, output, embedding, trajectory
        torch.cuda.empty_cache()
    extractor.close()
    print("[PASS] CAVE checkpoint, official logits, f4/f5 hooks, 5120-D embedding")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

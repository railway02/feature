from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cave_model import load_cave_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cave-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda:0")
    model, extractor = load_cave_model(args.cave_repo, args.checkpoint, device)
    for frames, amp in ((4, False), (20, True)):
        x = torch.rand(1, frames, 1, 512, 512, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            output = extractor(x)
        assert output.logits.shape == (1, 2, 512, 512)
        assert output.f4_last.shape == (1, 512, 64, 64)
        assert output.f5_last.shape == (1, 512, 32, 32)
        assert torch.isfinite(output.logits).all()
        print({
            "frames": frames, "amp": amp,
            "logits": tuple(output.logits.shape),
            "f4": tuple(output.f4_last.shape), "f5": tuple(output.f5_last.shape),
            "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        })
    extractor.close()
    print("[PASS] CAVE T=4/T=20 smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

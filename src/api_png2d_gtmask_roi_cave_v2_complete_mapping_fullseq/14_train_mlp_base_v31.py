#!/usr/bin/env python3
"""Train one GPU MLP base model for the formal series V3 task.

Run two copies concurrently:
  --model MLP_Deep
  --model MLP_Fusion

Each process owns a disjoint model directory, so no fold file is written by
both workers.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import torch


def load_trainer(path: Path):
    spec = importlib.util.spec_from_file_location("formal_series_v3_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=("MLP_Deep", "MLP_Fusion"),
        required=True,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--mlp-seeds", type=int, default=3)
    parser.add_argument("--mlp-search-seeds", type=int, default=2)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--overwrite-model-cache", action="store_true")
    parser.add_argument("--quick-smoke", action="store_true")
    args = parser.parse_args()

    task_root = args.task_root.resolve()
    output_dir = args.output_dir.resolve()
    trainer_path = args.trainer.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    core = load_trainer(trainer_path)
    if args.quick_smoke:
        core.MLP_CONFIGS = (
            core.MLPConfig(
                deep_pca_dim=8,
                hidden1=12,
                hidden2=4,
                dropout1=0.1,
                dropout2=0.1,
                learning_rate=1e-3,
                batch_size=16,
                max_epochs=3,
                patience=2,
            ),
        )

    if args.mlp_seeds < 1 or args.mlp_search_seeds < 1:
        raise ValueError("MLP seed counts must be >= 1")

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)

    core.set_seed(core.SEED)
    runtime = core.configure_runtime(args.cpu_threads, device)
    amp_enabled = bool(device.type == "cuda" and not args.disable_amp)
    runtime["amp"] = amp_enabled

    data = core.load_task(task_root)
    train = data["train"]
    valid = data["valid"]
    task_success = task_root / ".TASK_SUCCESS.json"
    task_hash = core.sha256_file(task_success)
    trainer_hash = core.sha256_file(trainer_path)

    if args.overwrite_model_cache:
        shutil.rmtree(
            output_dir / "folds" / args.model,
            ignore_errors=True,
        )

    use_scalar = args.model == "MLP_Fusion"
    start = time.time()
    _, _, audit = core.train_mlp_outer(
        args.model,
        train,
        valid,
        use_scalar,
        device,
        output_dir,
        False,
        args.mlp_seeds,
        args.mlp_search_seeds,
        task_hash,
        trainer_hash,
        amp_enabled,
    )

    payload = {
        "status": "success",
        "worker": args.model,
        "model": args.model,
        "outer_folds": [1, 2, 3, 4, 5],
        "device": str(device),
        "cpu_threads": int(args.cpu_threads),
        "runtime": runtime,
        "mlp_final_seeds": int(args.mlp_seeds),
        "mlp_search_seeds": int(args.mlp_search_seeds),
        "task_hash": task_hash,
        "trainer_hash": trainer_hash,
        "fold_audit_rows": len(audit),
        "elapsed_seconds": float(time.time() - start),
    }
    filename = args.model.casefold() + "_worker.SUCCESS.json"
    success = output_dir / "workers" / filename
    core.atomic_json(payload, success)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

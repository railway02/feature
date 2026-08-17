#!/usr/bin/env python3
"""Train the two CPU Logistic base models for the formal series V3 task.

This worker writes only:
  folds/Logistic_Deep/fold_*/
  folds/Logistic_Fusion/fold_*/
  workers/logistic_worker.SUCCESS.json

It is safe to run concurrently with the two GPU MLP workers because all model
directories are disjoint.
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
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--overwrite-model-cache", action="store_true")
    parser.add_argument("--quick-smoke", action="store_true")
    args = parser.parse_args()

    task_root = args.task_root.resolve()
    output_dir = args.output_dir.resolve()
    trainer_path = args.trainer.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    core = load_trainer(trainer_path)
    if args.quick_smoke:
        core.C_GRID = (0.1,)
        core.LOGISTIC_DEEP_DIMS = (8,)

    core.set_seed(core.SEED)
    runtime = core.configure_runtime(
        args.cpu_threads,
        torch.device("cpu"),
    )
    data = core.load_task(task_root)
    train = data["train"]
    valid = data["valid"]

    task_success = task_root / ".TASK_SUCCESS.json"
    task_hash = core.sha256_file(task_success)
    trainer_hash = core.sha256_file(trainer_path)

    if args.overwrite_model_cache:
        for model_name in ("Logistic_Deep", "Logistic_Fusion"):
            shutil.rmtree(
                output_dir / "folds" / model_name,
                ignore_errors=True,
            )

    start = time.time()
    rows = []
    for model_name, use_scalar in (
        ("Logistic_Deep", False),
        ("Logistic_Fusion", True),
    ):
        _, _, audit = core.train_logistic_outer(
            model_name,
            train,
            valid,
            use_scalar,
            output_dir,
            False,
            task_hash,
            trainer_hash,
        )
        rows.extend(audit)

    payload = {
        "status": "success",
        "worker": "logistic_cpu",
        "models": ["Logistic_Deep", "Logistic_Fusion"],
        "outer_folds": [1, 2, 3, 4, 5],
        "cpu_threads": int(args.cpu_threads),
        "runtime": runtime,
        "task_hash": task_hash,
        "trainer_hash": trainer_hash,
        "fold_audit_rows": len(rows),
        "elapsed_seconds": float(time.time() - start),
    }
    success = output_dir / "workers/logistic_worker.SUCCESS.json"
    core.atomic_json(payload, success)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

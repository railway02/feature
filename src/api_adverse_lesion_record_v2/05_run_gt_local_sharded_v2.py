#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import atomic_csv, atomic_json, load_config, sha256_file, update_run_manifest


def balanced_shards(frame: pd.DataFrame, count: int) -> list[list[str]]:
    costs = {}
    for row in frame.to_dict("records"):
        cost = sum(
            max(int(float(row.get(f"n_{phase}_frames", 0) or 0)), 1)
            for phase in ("pre", "post")
            if str(row.get(f"can_run_{phase}", "")).casefold() == "true"
        )
        costs[str(row["series_uid"])] = max(cost, 1)
    shards = [[] for _ in range(count)]
    loads = [0 for _ in range(count)]
    for uid in sorted(costs, key=lambda value: (-costs[value], value)):
        index = int(np.argmin(loads))
        shards[index].append(uid)
        loads[index] += costs[uid]
    return shards


def consolidate(shard_roots: list[Path], output_root: Path, split: str) -> int:
    phases = 0
    schemas = []
    indexes = []
    for shard in shard_roots:
        schema = shard / "feature_schema.json"
        if schema.is_file():
            schemas.append(schema)
        index = shard / "run_index.csv"
        if index.is_file():
            indexes.append(pd.read_csv(index, dtype=str, keep_default_na=False))
        source = shard / split.casefold()
        if not source.is_dir():
            continue
        for success in list(source.rglob(".SUCCESS.json")):
            phase_dir = success.parent
            target = output_root / split.casefold() / phase_dir.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                source_payload = json.loads(success.read_text(encoding="utf-8"))
                target_success = target / ".SUCCESS.json"
                if not target_success.is_file():
                    raise RuntimeError(f"Incomplete target {target}")
                target_payload = json.loads(target_success.read_text(encoding="utf-8"))
                if source_payload.get("embedding_sha256") != target_payload.get("embedding_sha256"):
                    raise RuntimeError(f"Conflicting target {target}")
                shutil.rmtree(phase_dir)
            else:
                shutil.move(str(phase_dir), str(target))
            phases += 1
    if not schemas:
        raise RuntimeError("No feature schema from shards")
    first = schemas[0]
    target_schema = output_root / "feature_schema.json"
    if not target_schema.is_file():
        shutil.copy2(first, target_schema)
    if indexes:
        atomic_csv(pd.concat(indexes, ignore_index=True), output_root / f"run_index_{split.casefold()}.csv")
    return phases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/root/autodl-tmp/aneurysm/configs/api_adverse_lesion_record_v2/config.json")
    parser.add_argument("--scale", choices=["30", "40"], required=True)
    parser.add_argument("--split", choices=["Train", "Valid"], required=True)
    parser.add_argument("--gpu-processes", type=int)
    parser.add_argument("--io-workers", type=int)
    parser.add_argument("--max-series", type=int)
    parser.add_argument("--tag", default="formal")
    args = parser.parse_args()
    config = load_config(args.config)
    manifests = Path(config["paths"]["manifests"])
    outputs = Path(config["paths"]["outputs"])
    reports = Path(config["paths"]["reports"])
    manifest_path = manifests / f"cave_manifest_gt_context{args.scale}_{args.split.casefold()}.csv"
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    if args.max_series:
        manifest = manifest.head(args.max_series).copy()
    processes = min(args.gpu_processes or int(config["oracle"]["gpu_processes"]), len(manifest))
    io_workers = args.io_workers or int(config["oracle"]["io_workers_per_process"])
    shards = balanced_shards(manifest, processes)
    output_root = (outputs / f"cave_gt_context{args.scale}_featurebank" if args.tag == "formal" else outputs / "smoke" / f"cave_gt_context{args.scale}_featurebank")
    report_root = reports / f"{args.tag}_{args.split.casefold()}_context{args.scale}"
    report_root.mkdir(parents=True, exist_ok=True)
    shard_base = outputs / f"cave_gt_context{args.scale}_shards_{args.tag}_{args.split.casefold()}_{processes}p"
    children, handles, shard_roots = [], [], []
    for index, uids in enumerate(shards):
        series_file = report_root / f"shard_{index:02d}_series.txt"
        series_file.write_text("\n".join(uids) + "\n", encoding="utf-8")
        shard_root = shard_base / f"shard_{index:02d}"
        shard_report = report_root / f"shard_{index:02d}"
        log_path = report_root / f"shard_{index:02d}.log"
        handle = log_path.open("a", encoding="utf-8")
        handles.append(handle); shard_roots.append(shard_root)
        command = [
            config["cave_python"], str(Path(config["paths"]["code"]) / "04_extract_gt_local_worker_v2.py"),
            "--config", str(Path(args.config).resolve()), "--scale", args.scale,
            "--split", args.split, "--output-root", str(shard_root),
            "--report-root", str(shard_report), "--series-uids-file", str(series_file),
            "--io-workers", str(io_workers), "--disable-empty-cache",
        ]
        children.append(subprocess.Popen(command, cwd=config["project_root"], stdout=handle, stderr=subprocess.STDOUT))
    start = time.perf_counter()
    codes = [child.wait() for child in children]
    elapsed = time.perf_counter() - start
    for handle in handles:
        handle.close()
    if any(code != 0 for code in codes):
        raise RuntimeError(f"GT context shard failure: {codes}; see {report_root}")
    phases = consolidate(shard_roots, output_root, args.split)
    expected = sum(
        int(str(row.get("can_run_pre", "")).casefold() == "true")
        + int(str(row.get("can_run_post", "")).casefold() == "true")
        for row in manifest.to_dict("records")
    )
    if phases != expected:
        raise AssertionError(f"Consolidated phase mismatch {phases} != {expected}")
    summary = {
        "scale": args.scale, "split": args.split, "tag": args.tag,
        "series": len(manifest), "phases": phases, "wall_seconds": elapsed,
        "phases_per_second": phases / max(elapsed, 1e-8), "worker_codes": codes,
        "output_root": str(output_root), "manifest_sha256": sha256_file(manifest_path),
    }
    atomic_json(summary, reports / f"gt_context{args.scale}_{args.split.casefold()}_{args.tag}_extraction_summary.json")
    update_run_manifest(config, f"extract_gt_context{args.scale}_{args.split.casefold()}_{args.tag}", {"status": "complete", **summary})
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

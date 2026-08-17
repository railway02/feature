#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import atomic_csv, atomic_json, configure_runtime, load_config, sha256_file


def gpu_sample() -> dict[str, float] | None:
    try:
        output = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ], text=True).strip().splitlines()[0]
        util, used, total = [float(value.strip()) for value in output.split(",")]
        return {"gpu_utilization": util, "gpu_memory_used_mb": used, "gpu_memory_total_mb": total}
    except Exception:
        return None


def balanced_shards(frame: pd.DataFrame, series_uids: list[str], count: int) -> list[list[str]]:
    selected = frame[frame["series_uid"].astype(str).isin(series_uids)].copy()
    if selected["series_uid"].duplicated().any():
        raise AssertionError("Duplicate series_uid in CAVE manifest")
    costs = {}
    for row in selected.to_dict("records"):
        cost = 0
        for phase in ("pre", "post"):
            if str(row.get(f"can_run_{phase}", "")).casefold() == "true":
                cost += max(int(float(row.get(f"n_{phase}_frames", 0) or 0)), 1)
        costs[str(row["series_uid"])] = max(cost, 1)
    shards = [[] for _ in range(count)]
    loads = [0 for _ in range(count)]
    for uid in sorted(series_uids, key=lambda value: (-costs.get(value, 1), value)):
        index = int(np.argmin(loads))
        shards[index].append(uid)
        loads[index] += costs.get(uid, 1)
    return shards


def consolidate(shard_roots: list[Path], output_root: Path, split: str) -> dict:
    split_name = split.casefold()
    output_root.mkdir(parents=True, exist_ok=True)
    schema_hashes = {}
    run_frames = []
    phases = 0
    for shard_root in shard_roots:
        schema = shard_root / "feature_schema.json"
        if schema.is_file():
            schema_hashes[str(shard_root)] = sha256_file(schema)
        run_index = shard_root / "run_index.csv"
        if run_index.is_file():
            run_frames.append(pd.read_csv(run_index, dtype=str, keep_default_na=False))
        source_split = shard_root / split_name
        if not source_split.is_dir():
            continue
        for success in list(source_split.rglob(".SUCCESS.json")):
            source_phase = success.parent
            relative = source_phase.relative_to(source_split)
            target = output_root / split_name / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                source_payload = json.loads(success.read_text(encoding="utf-8"))
                target_success = target / ".SUCCESS.json"
                if not target_success.is_file():
                    raise RuntimeError(f"Incomplete existing target: {target}")
                target_payload = json.loads(target_success.read_text(encoding="utf-8"))
                if source_payload.get("embedding_sha256") != target_payload.get("embedding_sha256"):
                    raise RuntimeError(f"Conflicting existing target: {target}")
                shutil.rmtree(source_phase)
            else:
                shutil.move(str(source_phase), str(target))
            phases += 1
    unique_schema = set(schema_hashes.values())
    if len(unique_schema) != 1:
        raise AssertionError(f"Shard feature schemas differ: {schema_hashes}")
    first_schema = next((root / "feature_schema.json" for root in shard_roots if (root / "feature_schema.json").is_file()), None)
    if first_schema:
        target_schema = output_root / "feature_schema.json"
        if target_schema.is_file() and sha256_file(target_schema) != sha256_file(first_schema):
            # The worker schema records the split-specific frozen config hash.
            # Train and Valid therefore have different provenance hashes even
            # when their scientific feature definitions are identical. A
            # shared consolidated featurebank must compare the actual feature
            # contract while preserving per-phase frozen-config provenance.
            ignored_provenance = {"frozen_config_hash", "schema_sha256"}
            target_payload = json.loads(target_schema.read_text(encoding="utf-8"))
            source_payload = json.loads(first_schema.read_text(encoding="utf-8"))
            target_scientific = {
                key: value for key, value in target_payload.items() if key not in ignored_provenance
            }
            source_scientific = {
                key: value for key, value in source_payload.items() if key not in ignored_provenance
            }
            if target_scientific != source_scientific:
                raise AssertionError("Existing consolidated scientific feature schema differs")
        if not target_schema.is_file():
            shutil.copy2(first_schema, target_schema)
    if run_frames:
        run_index = pd.concat(run_frames, ignore_index=True)
        atomic_csv(run_index, output_root / f"run_index_{split_name}.csv")
    return {"consolidated_phases": phases, "feature_schema_sha256": next(iter(unique_schema))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--split", choices=["Train", "Valid"], required=True)
    parser.add_argument("--series-uids-file", type=Path)
    parser.add_argument("--gpu-processes", type=int, required=True)
    parser.add_argument("--io-workers", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--disable-empty-cache", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    configure_runtime(config)
    manifests = Path(config["paths"]["manifests"])
    manifest_path = manifests / f"cave_manifest_pred_{args.split.casefold()}.csv"
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    all_uids = manifest["series_uid"].astype(str).tolist()
    if args.series_uids_file:
        requested = [line.strip() for line in args.series_uids_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        requested = all_uids
    missing = sorted(set(requested) - set(all_uids))
    if missing:
        raise AssertionError(f"Requested series absent from manifest: {missing[:5]}")
    processes = min(max(1, args.gpu_processes), len(requested))
    shards = balanced_shards(manifest, requested, processes)

    args.report_root.mkdir(parents=True, exist_ok=True)
    version_tag = str(config["version"]).replace(".", "_")
    shard_base = args.output_root.parent / f"{args.output_root.name}_shards_{version_tag}_{args.split.casefold()}_{processes}p_{args.io_workers}io"
    shard_base.mkdir(parents=True, exist_ok=True)
    children = []
    shard_roots = []
    handles = []
    for index, uids in enumerate(shards):
        series_file = args.report_root / f"shard_{index:02d}_series.txt"
        series_file.write_text("\n".join(uids) + "\n", encoding="utf-8")
        shard_root = shard_base / f"shard_{index:02d}"
        shard_report = args.report_root / f"shard_{index:02d}"
        shard_roots.append(shard_root)
        log_path = args.report_root / f"shard_{index:02d}.log"
        handle = log_path.open("a", encoding="utf-8")
        handles.append(handle)
        command = [
            config["cave_python"],
            str(Path(config["paths"]["code"]) / "02_extract_pred_worker_fixed.py"),
            "--config", config["_config_path"],
            "--split", args.split,
            "--output-root", str(shard_root),
            "--report-root", str(shard_report),
            "--series-uids-file", str(series_file),
            "--io-workers", str(args.io_workers),
        ]
        if args.disable_empty_cache:
            command.append("--disable-empty-cache")
        children.append(subprocess.Popen(command, cwd=config["project_root"], stdout=handle, stderr=subprocess.STDOUT))

    samples = []
    stop = threading.Event()
    def monitor() -> None:
        while not stop.wait(0.5):
            sample = gpu_sample()
            if sample:
                sample["time_seconds"] = time.time()
                samples.append(sample)
    thread = threading.Thread(target=monitor, daemon=True)
    start = time.perf_counter()
    thread.start()
    codes = [child.wait() for child in children]
    wall = time.perf_counter() - start
    stop.set()
    thread.join(timeout=2)
    for handle in handles:
        handle.close()
    if any(code != 0 for code in codes):
        raise RuntimeError(f"CAVE shard failure exit_codes={codes}")

    consolidated = consolidate(shard_roots, args.output_root, args.split)
    sample_frame = pd.DataFrame(samples)
    if not sample_frame.empty:
        atomic_csv(sample_frame, args.report_root / "gpu_samples.csv")
    summary = {
        "split": args.split,
        "series": len(requested),
        "gpu_processes": processes,
        "io_workers_per_process": args.io_workers,
        "disable_empty_cache": args.disable_empty_cache,
        "wall_seconds": wall,
        "consolidated_phases": consolidated["consolidated_phases"],
        "phases_per_second": consolidated["consolidated_phases"] / max(wall, 1e-8),
        "gpu_utilization_mean": float(sample_frame["gpu_utilization"].mean()) if not sample_frame.empty else None,
        "gpu_utilization_p95": float(sample_frame["gpu_utilization"].quantile(0.95)) if not sample_frame.empty else None,
        "gpu_memory_used_peak_mb": float(sample_frame["gpu_memory_used_mb"].max()) if not sample_frame.empty else None,
        "feature_schema_sha256": consolidated["feature_schema_sha256"],
        "worker_exit_codes": codes,
    }
    atomic_json(summary, args.report_root / "run_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

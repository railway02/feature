#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/root/autodl-tmp/aneurysm/configs/api_adverse_lesion_cave_fast_v1.json",
    )
    parser.add_argument("--split", choices=["Train", "Valid"], required=True)
    parser.add_argument("--gpu-processes", type=int, default=2)
    parser.add_argument("--io-workers", type=int, default=8)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path(config["project_root"])
    fast_code = root / "code/api_adverse_lesion_cave_fast_v1"
    sys.path.insert(0, str(fast_code))
    import common as fast_common

    fast_shards = import_file(
        fast_code / "03_run_sharded_extraction.py", "gt_oracle_fast_shards"
    )
    code_root = root / "code/api_adverse_lesion_cave_gt_oracle_v1"
    output_root = (
        root
        / "outputs/api_adverse_lesion_cave_gt_oracle_v1/cave_gt_roi_featurebank"
    )
    report_root = (
        root
        / "reports/api_adverse_lesion_cave_gt_oracle_v1"
        / f"formal_{args.split.casefold()}"
    )
    manifest_path = (
        root
        / "manifests/api_adverse_lesion_cave_v1"
        / f"cave_manifest_gt_{args.split.casefold()}.csv"
    )
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    series_uids = manifest["series_uid"].astype(str).tolist()
    processes = min(max(1, args.gpu_processes), len(series_uids))
    shards = fast_shards.balanced_shards(manifest, series_uids, processes)

    report_root.mkdir(parents=True, exist_ok=True)
    shard_base = (
        output_root.parent
        / f"cave_gt_roi_featurebank_shards_{args.split.casefold()}_{processes}p_{args.io_workers}io"
    )
    shard_base.mkdir(parents=True, exist_ok=True)
    children = []
    handles = []
    shard_roots = []
    for index, uids in enumerate(shards):
        series_path = report_root / f"shard_{index:02d}_series.txt"
        series_path.write_text("\n".join(uids) + "\n", encoding="utf-8")
        shard_root = shard_base / f"shard_{index:02d}"
        shard_report = report_root / f"shard_{index:02d}"
        log_path = report_root / f"shard_{index:02d}.log"
        handle = log_path.open("a", encoding="utf-8")
        handles.append(handle)
        shard_roots.append(shard_root)
        command = [
            config["cave_python"],
            str(code_root / "01_extract_gt_worker.py"),
            "--config",
            str(config_path),
            "--split",
            args.split,
            "--output-root",
            str(shard_root),
            "--report-root",
            str(shard_report),
            "--series-uids-file",
            str(series_path),
            "--io-workers",
            str(args.io_workers),
            "--disable-empty-cache",
        ]
        children.append(
            subprocess.Popen(
                command,
                cwd=root,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        )

    started = time.perf_counter()
    codes = [child.wait() for child in children]
    wall_seconds = time.perf_counter() - started
    for handle in handles:
        handle.close()
    if any(code != 0 for code in codes):
        raise RuntimeError(f"GT oracle shard failure: {codes}")
    consolidated = fast_shards.consolidate(
        shard_roots, output_root, args.split
    )
    expected_phases = 0
    for record in manifest.to_dict("records"):
        expected_phases += int(
            str(record.get("can_run_pre", "")).casefold() == "true"
        )
        expected_phases += int(
            str(record.get("can_run_post", "")).casefold() == "true"
        )
    if consolidated["consolidated_phases"] != expected_phases:
        raise AssertionError(
            f"GT oracle phase mismatch {consolidated['consolidated_phases']} != {expected_phases}"
        )
    summary = {
        "version": "api_adverse_lesion_cave_gt_oracle_v1",
        "split": args.split,
        "series": int(len(series_uids)),
        "expected_phases": int(expected_phases),
        "consolidated_phases": int(consolidated["consolidated_phases"]),
        "gpu_processes": int(processes),
        "io_workers_per_process": int(args.io_workers),
        "wall_seconds": float(wall_seconds),
        "phases_per_second": float(
            expected_phases / max(wall_seconds, 1e-8)
        ),
        "worker_exit_codes": codes,
        "feature_schema_sha256": consolidated["feature_schema_sha256"],
        "output_root": str(output_root),
        "manifest": str(manifest_path),
        "manifest_sha256": fast_common.sha256_file(manifest_path),
    }
    fast_common.atomic_json(
        summary,
        root
        / "reports/api_adverse_lesion_cave_gt_oracle_v1"
        / f"gt_oracle_{args.split.casefold()}_extraction_summary.json",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

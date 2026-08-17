#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from common import as_bool, atomic_csv, atomic_json, load_config, sha256_file


def series_cost(row: dict[str, str]) -> int:
    cost = 0
    for phase in ("pre", "post"):
        if as_bool(row.get(f"can_run_{phase}")):
            cost += max(int(float(row.get(f"n_{phase}_frames", 0) or 0)), 1)
    return max(cost, 1)


def balanced(frame: pd.DataFrame, uids: list[str], n_shards: int) -> tuple[list[list[str]], dict[str, int]]:
    costs = {str(row["series_uid"]): series_cost(row) for row in frame.to_dict("records")}
    shards = [[] for _ in range(n_shards)]
    loads = [0] * n_shards
    for uid in sorted(uids, key=lambda value: (-costs.get(value, 1), value)):
        index = int(np.argmin(loads))
        shards[index].append(uid)
        loads[index] += costs.get(uid, 1)
    return shards, costs


def scientific_schema(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {key: value for key, value in payload.items() if key not in {"frozen_config_hash", "schema_sha256"}}


def consolidate(shard_roots: list[Path], output: Path, split: str) -> dict[str, int]:
    moved = duplicate = 0
    schemas: list[Path] = []
    indices: list[pd.DataFrame] = []
    for root in shard_roots:
        schema = root / "feature_schema.json"
        if schema.is_file():
            schemas.append(schema)
        index = root / "run_index.csv"
        if index.is_file():
            indices.append(pd.read_csv(index, dtype=str, keep_default_na=False))
        source = root / split.casefold()
        if not source.is_dir():
            continue
        for success in list(source.rglob(".SUCCESS.json")):
            phase_dir = success.parent
            relative = phase_dir.relative_to(source)
            target = output / split.casefold() / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                target_success = target / ".SUCCESS.json"
                if not target_success.is_file():
                    raise RuntimeError(f"目标目录存在但没有 SUCCESS：{target}")
                source_payload = json.loads(success.read_text(encoding="utf-8"))
                target_payload = json.loads(target_success.read_text(encoding="utf-8"))
                if source_payload.get("embedding_sha256") != target_payload.get("embedding_sha256"):
                    raise RuntimeError(f"重复 phase 输出不一致：{target}")
                shutil.rmtree(phase_dir)
                duplicate += 1
            else:
                shutil.move(str(phase_dir), str(target))
                moved += 1

    target_schema = output / "feature_schema.json"
    if schemas:
        reference = scientific_schema(schemas[0])
        if any(scientific_schema(path) != reference for path in schemas[1:]):
            raise AssertionError("不同 shard 的科学 feature schema 不一致")
        if target_schema.is_file() and scientific_schema(target_schema) != reference:
            raise AssertionError("已有 consolidated schema 与新 shard schema 不一致")
        if not target_schema.is_file():
            output.mkdir(parents=True, exist_ok=True)
            shutil.copy2(schemas[0], target_schema)
    elif not target_schema.is_file():
        raise RuntimeError("没有 shard schema，且 consolidated feature_schema.json 不存在")

    if indices:
        run_index = pd.concat(indices, ignore_index=True)
        keys = [column for column in ("series_uid", "phase") if column in run_index.columns]
        if keys:
            run_index = run_index.drop_duplicates(keys, keep="last")
        atomic_csv(run_index, output / f"run_index_{split.casefold()}.csv")
    return {"moved": moved, "duplicate": duplicate}



def series_is_complete(output: Path, split: str, row: dict[str, str]) -> bool:
    base = output / split.casefold() / str(row.get("patient_id", "")) / str(row["series_uid"])
    for phase in ("pre", "post"):
        if as_bool(row.get(f"can_run_{phase}")) and not (base / phase / ".SUCCESS.json").is_file():
            return False
    return True


def remove_series_output(output: Path, split: str, row: dict[str, str]) -> None:
    base = output / split.casefold() / str(row.get("patient_id", "")) / str(row["series_uid"])
    if base.exists():
        shutil.rmtree(base)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["Train", "Valid"], required=True)
    parser.add_argument("--max-series", type=int)
    parser.add_argument("--series-uids-file", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gpu-processes", type=int)
    parser.add_argument("--io-workers", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifests = Path(cfg["paths"]["manifests"])
    stage1 = manifests / ".STAGE1_ELIGIBLE_SUCCESS.json"
    if not stage1.is_file():
        raise RuntimeError(f"第一阶段未闭合：{stage1}")

    kind = "smoke" if args.smoke else "full"
    report_root = Path(cfg["paths"]["reports"]) / f"extract_{args.split.casefold()}_{kind}"
    base_outputs = Path(cfg["paths"]["outputs"])
    output = base_outputs / ("smoke_local_eligible_featurebank" if args.smoke else "cave_local_eligible_featurebank")
    report_root.mkdir(parents=True, exist_ok=True)

    manifest_path = manifests / f"cave_manifest_local_{args.split.casefold()}_eligible.csv"
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    if manifest["series_uid"].duplicated().any():
        raise AssertionError("提取 manifest 重复 series_uid")
    uids = manifest["series_uid"].astype(str).tolist()
    if args.series_uids_file:
        wanted = [line.strip() for line in args.series_uids_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        missing = sorted(set(wanted) - set(uids))
        if missing:
            raise ValueError(f"series-uids-file 中有 {len(missing)} 个 series 不在 eligible manifest：{missing[:5]}")
        uids = [uid for uid in uids if uid in set(wanted)]
    if args.max_series:
        uids = uids[: args.max_series]
    if not uids:
        raise RuntimeError("没有待提取 series")

    manifest_by_uid = {str(row["series_uid"]): row for row in manifest.to_dict("records")}
    if args.overwrite:
        for uid in uids:
            remove_series_output(output, args.split, manifest_by_uid[uid])

    completed_before = [uid for uid in uids if series_is_complete(output, args.split, manifest_by_uid[uid])]
    pending_uids = [uid for uid in uids if uid not in set(completed_before)]
    requested_processes = int(args.gpu_processes or cfg.get("runtime", {}).get("gpu_processes", 1))
    n_shards = max(1, min(requested_processes, max(len(pending_uids), 1)))
    io_workers = int(args.io_workers or cfg.get("runtime", {}).get("io_workers", 8))
    shards, costs = balanced(manifest, pending_uids, n_shards)

    shard_of = {uid: index for index, items in enumerate(shards) for uid in items}
    plan_rows = []
    for uid in uids:
        row = manifest_by_uid[uid]
        plan_rows.append({
            "split": args.split,
            "kind": kind,
            "status_before_run": "complete" if uid in set(completed_before) else "pending",
            "shard": shard_of.get(uid, ""),
            "series_uid": uid,
            "patient_id": row.get("patient_id", ""),
            "can_run_pre": row.get("can_run_pre", ""),
            "can_run_post": row.get("can_run_post", ""),
            "n_pre_frames": row.get("n_pre_frames", ""),
            "n_post_frames": row.get("n_post_frames", ""),
            "estimated_cost_frames": costs.get(uid, series_cost(row)),
        })
    atomic_csv(pd.DataFrame(plan_rows), report_root / "extraction_plan.csv")

    shard_base = base_outputs / f"shards_{args.split.casefold()}_{kind}"
    if args.overwrite and shard_base.exists():
        shutil.rmtree(shard_base)
    shard_base.mkdir(parents=True, exist_ok=True)

    visible = cfg.get("runtime", {}).get("cuda_visible_devices", [0])
    devices = [str(value) for value in visible] if isinstance(visible, list) else [value for value in str(visible).split(",") if value]
    if not devices:
        devices = ["0"]

    shard_roots: list[Path] = []
    children: list[subprocess.Popen] = []
    handles = []
    for shard_index, items in enumerate(shards):
        if not items:
            continue
        series_file = report_root / f"shard_{shard_index:02d}_series.txt"
        series_file.write_text("\n".join(items) + "\n", encoding="utf-8")
        shard_output = shard_base / f"shard_{shard_index:02d}"
        shard_roots.append(shard_output)
        log_path = report_root / f"shard_{shard_index:02d}.log"
        log = log_path.open("a", encoding="utf-8")
        handles.append(log)
        cmd = [
            cfg["cave_python"],
            str(Path(cfg["paths"]["code"]) / "06_extract_local_cave_worker.py"),
            "--config", cfg["_config_path"],
            "--split", args.split,
            "--output-root", str(shard_output),
            "--report-root", str(report_root / f"shard_{shard_index:02d}"),
            "--series-uids-file", str(series_file),
            "--io-workers", str(io_workers),
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = devices[shard_index % len(devices)]
        children.append(subprocess.Popen(cmd, cwd=cfg["project_root"], env=env, stdout=log, stderr=subprocess.STDOUT))

    exit_codes = [process.wait() for process in children]
    for handle in handles:
        handle.close()
    if any(code != 0 for code in exit_codes):
        atomic_json({"exit_codes": exit_codes, "status": "failed"}, report_root / "summary.json")
        raise RuntimeError(f"CAVE shard 失败：{exit_codes}；查看 {report_root}")

    consolidation = consolidate(shard_roots, output, args.split) if shard_roots else {"moved": 0, "duplicate": 0}
    if not (output / "feature_schema.json").is_file():
        raise RuntimeError("提取结束后缺少 consolidated feature_schema.json")
    expected_phases = 0
    for row in manifest[manifest["series_uid"].astype(str).isin(uids)].to_dict("records"):
        expected_phases += int(as_bool(row.get("can_run_pre"))) + int(as_bool(row.get("can_run_post")))
    summary = {
        "status": "success",
        "split": args.split,
        "kind": kind,
        "series_requested": len(uids),
        "series_complete_before_run": len(completed_before),
        "series_pending": len(pending_uids),
        "expected_phases": expected_phases,
        "shards": len(shard_roots),
        "exit_codes": exit_codes,
        "consolidation": consolidation,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "output_root": str(output),
        "feature_schema_sha256": sha256_file(output / "feature_schema.json"),
    }
    atomic_json(summary, report_root / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

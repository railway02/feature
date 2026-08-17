#!/usr/bin/env python3
"""Freeze the successful Full-Train v3 release before any Valid extraction."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("api_fullseq_v3_extractor", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="/root/autodl-tmp/aneurysm")
    parser.add_argument("--extractor", default="code/api_fullseq_v3/extract_pairdata.py")
    parser.add_argument("--builder", default="code/api_fullseq_v3/build_features.py")
    parser.add_argument("--base-config", default="configs/api_fullseq_v2_full_train_valid_config.json")
    parser.add_argument("--override-config", default="configs/api_fullseq_v3_improved_overrides.json")
    parser.add_argument("--train-manifest", default="manifests/api_fullseq_v3_train_all_series_frozen.csv")
    parser.add_argument("--valid-manifest", default="manifests/api_fullseq_v3_valid_all_series_frozen.csv")
    parser.add_argument("--train-pairdata", default="outputs/api_fullseq_v3_pairdata/full/train")
    parser.add_argument("--train-features", default="outputs/api_fullseq_v3_features/full/train")
    parser.add_argument("--report-dir", default="reports/api_fullseq_v3_reextract")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project = Path(args.project).resolve()
    resolve = lambda value: (project / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    extractor = resolve(args.extractor)
    builder = resolve(args.builder)
    base_config = resolve(args.base_config)
    override_config = resolve(args.override_config)
    train_manifest = resolve(args.train_manifest)
    valid_manifest = resolve(args.valid_manifest)
    train_pairdata = resolve(args.train_pairdata)
    train_features = resolve(args.train_features)
    report = resolve(args.report_dir)
    release = report / "train_release_freeze.json"
    marker = report / ".FULL_TRAIN_FEATURES_SUCCESS"
    if (release.exists() or marker.exists()) and not args.overwrite:
        raise FileExistsError("Release freeze already exists; use --overwrite only after deliberate invalidation")

    required = [
        extractor, builder, base_config, override_config, train_manifest, valid_manifest,
        train_pairdata / ".SUCCESS", train_pairdata / "run_summary.json",
        train_features / ".FEATURES_SUCCESS", train_features / "feature_schema.json",
        train_features / "audit.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing successful Full-Train artifacts:\n" + "\n".join(missing))

    pair_summary = json.loads((train_pairdata / "run_summary.json").read_text(encoding="utf-8"))
    feature_audit = json.loads((train_features / "audit.json").read_text(encoding="utf-8"))
    expected = {"patients": 1055, "series": 1147, "phases": 2087, "processed_pairs": 43364}
    actual = {key: int(pair_summary[key]) for key in expected}
    if actual != expected:
        raise AssertionError(f"Full-Train pairdata size mismatch: expected={expected}, actual={actual}")
    expected_features = {"series": 1147, "patients": 1055, "phases": 2087, "pairs": 43364}
    if feature_audit.get("actual") != expected_features:
        raise AssertionError(
            f"Full-Train feature size mismatch: expected={expected_features}, actual={feature_audit.get('actual')}"
        )
    if not pair_summary.get("cuda_actually_used") or pair_summary.get("cpu_fallback"):
        raise AssertionError("Full-Train pairdata did not pass CUDA assertion")
    if pair_summary.get("labels_read") or pair_summary.get("model_trained") or pair_summary.get("manifest_rescanned"):
        raise AssertionError("Forbidden label/training/rescan flag in Full-Train pairdata")

    module = load_module(extractor)
    merged = module.load_config(base_config, override_config)
    model = Path(merged["model"]["model_file"]).resolve()
    model_config = Path(merged["model"]["config"]).resolve()
    sea_raft_root = Path(merged["model"]["repo_root"]).resolve()
    for path in (model, model_config, sea_raft_root):
        if not path.exists():
            raise FileNotFoundError(path)

    file_artifacts = {
        "extractor": extractor,
        "builder": builder,
        "base_config": base_config,
        "override_config": override_config,
        "train_manifest": train_manifest,
        "valid_manifest": valid_manifest,
        "model": model,
        "model_config": model_config,
        "feature_schema": train_features / "feature_schema.json",
    }
    artifacts = [
        {"name": name, "path": str(path), "sha256": sha256_file(path), "kind": "file"}
        for name, path in file_artifacts.items()
    ]
    artifacts.append({
        "name": "sea_raft_code_tree",
        "path": str(sea_raft_root),
        "sha256": module.sea_raft_code_tree_hash(sea_raft_root),
        "kind": "code_tree",
    })
    payload = {
        "version": "api_fullseq_v3_train_release_freeze_v1",
        "created_utc": utc_now(),
        "science_profile": merged["v3"]["science_profile"],
        "full_train_pairdata": actual,
        "full_train_features": expected_features,
        "artifacts": artifacts,
        "labels_read": False,
        "model_trained": False,
    }
    write_json_atomic(release, payload)
    write_json_atomic(marker, {
        "created_utc": payload["created_utc"],
        "release_freeze": str(release),
        "release_freeze_sha256": sha256_file(release),
    })
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

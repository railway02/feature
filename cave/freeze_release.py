#!/usr/bin/env python3
"""Freeze successful Full-Train CAVE extraction before any Full-Valid run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cave_model import cave_code_tree_hash, git_commit
from common import sha256_file, sha256_tree, utc_now, write_json_atomic
from release import artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--cave-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--v3-extractor", type=Path, required=True)
    parser.add_argument("--v3-base-config", type=Path, required=True)
    parser.add_argument("--v3-override-config", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--train-table-root", type=Path, required=True)
    parser.add_argument("--train-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    schema = args.feature_root / "feature_schema.json"
    build_audit = args.train_table_root / "build_audit.json"
    run_summary = args.feature_root / "run_summary_full_train.json"
    required_files = [
        args.frozen_config, args.train_manifest, args.valid_manifest, args.checkpoint,
        args.v3_extractor, args.v3_base_config, args.v3_override_config,
        schema, build_audit, args.train_audit, run_summary,
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Full-Train freeze inputs:\n" + "\n".join(missing))
    audit_payload = json.loads(args.train_audit.read_text(encoding="utf-8"))
    build_payload = json.loads(build_audit.read_text(encoding="utf-8"))
    run_payload = json.loads(run_summary.read_text(encoding="utf-8"))
    if audit_payload.get("hard_fail"):
        raise AssertionError("Train feature-bank audit failed")
    if int(build_payload.get("series", -1)) != 1147 or int(build_payload.get("patients", -1)) != 1055:
        raise AssertionError("Train feature-table hard counts failed")
    if int(run_payload.get("failures", -1)) != 0:
        raise AssertionError("Train extraction contains failures")
    if not run_payload.get("cuda_actually_used") or run_payload.get("cpu_fallback"):
        raise AssertionError("Train extraction did not prove CUDA execution")
    if run_payload.get("labels_read") or run_payload.get("model_trained") or run_payload.get("manifest_rescanned"):
        raise AssertionError("Forbidden action flag in Train extraction")

    artifacts = [
        artifact((args.package_root / "extract_cave_featurebank.py").resolve(), "extractor"),
        artifact(args.package_root.resolve(), "package_tree", kind="tree"),
        artifact(args.frozen_config.resolve(), "frozen_config"),
        artifact(args.train_manifest.resolve(), "train_manifest"),
        artifact(args.valid_manifest.resolve(), "valid_manifest"),
        artifact(schema.resolve(), "feature_schema"),
        artifact(args.v3_extractor.resolve(), "v3_extractor"),
        artifact(args.v3_base_config.resolve(), "v3_base_config"),
        artifact(args.v3_override_config.resolve(), "v3_override_config"),
        artifact(args.train_audit.resolve(), "train_audit"),
        artifact(build_audit.resolve(), "train_build_audit"),
        artifact(run_summary.resolve(), "train_run_summary"),
    ]
    payload = {
        "release_name": "api_fullseq_cave_v3_full_train_release",
        "frozen_at": utc_now(),
        "artifacts": artifacts,
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_size": args.checkpoint.stat().st_size,
        "cave_repo": str(args.cave_repo.resolve()),
        "cave_commit": git_commit(args.cave_repo),
        "cave_code_tree_sha256": cave_code_tree_hash(args.cave_repo),
        "package_tree_sha256": sha256_tree(args.package_root),
        "train_counts": {
            "series": 1147, "patients": 1055, "pre": 940, "post": 1147, "phases": 2087,
        },
        "valid_counts": {
            "series": 287, "patients": 264, "pre": 248, "post": 287, "phases": 535,
        },
        "labels_read": False,
        "model_trained": False,
        "manifest_rescanned": False,
    }
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

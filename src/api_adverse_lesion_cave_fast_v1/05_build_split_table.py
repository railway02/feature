#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from common import atomic_json, configure_runtime, load_config, run, sha256_file, write_success


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--split", choices=["Train", "Valid"], required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    configure_runtime(config)
    root = Path(config["project_root"])
    outputs = Path(config["paths"]["outputs"])
    manifests = Path(config["paths"]["manifests"])
    reports = Path(config["paths"]["reports"])
    feature_root = outputs / "cave_pred_roi_featurebank"
    output_dir = outputs / "cave_pred_roi_tables" / args.split.casefold()
    manifest = manifests / f"cave_manifest_pred_{args.split.casefold()}.csv"
    env = os.environ.copy()
    env["PYTHONPATH"] = config["cave_code_root"]
    command = [
        config["cave_python"],
        str(Path(config["cave_code_root"]) / "build_feature_tables.py"),
        "--manifest", str(manifest),
        "--feature-root", str(feature_root),
        "--output-dir", str(output_dir),
        "--expected-split", args.split,
        "--verify-files",
    ]
    run(command, cwd=root, env=env)
    audit = json.loads((output_dir / "build_audit.json").read_text(encoding="utf-8"))
    payload = {
        "split": args.split,
        "manifest_sha256": sha256_file(manifest),
        "feature_schema_sha256": sha256_file(feature_root / "feature_schema.json"),
        "build_audit": audit,
    }
    atomic_json(payload, reports / f"pred_roi_{args.split.casefold()}_table_audit.json")
    write_success(reports / f".PRED_ROI_{args.split.upper()}_TABLE_SUCCESS", "05_build_split_table", config, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

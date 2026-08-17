#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from common import atomic_json, load_config, update_run_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/root/autodl-tmp/aneurysm/configs/api_adverse_lesion_record_v2/config.json")
    parser.add_argument("--scale", choices=["30", "40"], required=True)
    parser.add_argument("--split", choices=["Train", "Valid"], required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["project_root"])
    feature_root = Path(config["paths"]["outputs"]) / f"cave_gt_context{args.scale}_featurebank"
    output_dir = Path(config["paths"]["outputs"]) / f"cave_gt_context{args.scale}_tables" / args.split.casefold()
    manifest = Path(config["paths"]["manifests"]) / f"cave_manifest_gt_context{args.scale}_{args.split.casefold()}.csv"
    env = os.environ.copy()
    env["PYTHONPATH"] = config["cave_code_root"]
    command = [
        config["cave_python"], str(Path(config["cave_code_root"]) / "build_feature_tables.py"),
        "--manifest", str(manifest), "--feature-root", str(feature_root),
        "--output-dir", str(output_dir), "--expected-split", args.split, "--verify-files",
    ]
    subprocess.run(command, cwd=root, env=env, check=True)
    audit = json.loads((output_dir / "build_audit.json").read_text(encoding="utf-8"))
    payload = {"status": "complete", "scale": args.scale, "split": args.split, "output_dir": str(output_dir), "build_audit": audit}
    reports = Path(config["paths"]["reports"])
    atomic_json(payload, reports / f"gt_context{args.scale}_{args.split.casefold()}_table_audit.json")
    update_run_manifest(config, f"table_gt_context{args.scale}_{args.split.casefold()}", payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

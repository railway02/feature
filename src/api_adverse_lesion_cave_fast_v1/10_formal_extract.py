#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import configure_runtime, load_config, run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--split", choices=["Train", "Valid"], required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    configure_runtime(config)
    reports = Path(config["paths"]["reports"])
    outputs = Path(config["paths"]["outputs"])
    recommendation_path = reports / "recommended_runtime_config.json"
    if not recommendation_path.is_file():
        raise FileNotFoundError(recommendation_path)
    recommendation = json.loads(recommendation_path.read_text(encoding="utf-8"))
    command = [
        config["cave_python"],
        str(Path(config["paths"]["code"]) / "03_run_sharded_extraction.py"),
        "--config", config["_config_path"],
        "--split", args.split,
        "--gpu-processes", str(int(recommendation["gpu_processes"])),
        "--io-workers", str(int(recommendation["io_workers"])),
        "--output-root", str(outputs / "cave_pred_roi_featurebank"),
        "--report-root", str(reports / f"formal_{args.split.casefold()}"),
    ]
    if recommendation.get("disable_per_view_empty_cache"):
        command.append("--disable-empty-cache")
    run(command, cwd=config["project_root"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

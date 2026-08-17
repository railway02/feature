#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import configure_runtime, load_config, require_file, run_checked, stage_logger, write_marker


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--config"); parser.add_argument("--variant",default="pred_roi"); parser.add_argument("--overwrite",action="store_true"); args=parser.parse_args(); config=load_config(args.config); configure_runtime(config); finish=stage_logger(f"12_train_adverse_models_fixed:{args.variant}")
    evidence=require_file(config["fixed_trainer"],config["fixed_trainer_sha256"]); outputs=Path(config["paths"]["outputs"]); reports=Path(config["paths"]["reports"]); task_root=outputs/"adverse_tasks"/args.variant; model_root=outputs/"adverse_models"/args.variant
    if not (reports/".ADVERSE_TASKS_SUCCESS").is_file(): raise RuntimeError("Adverse task gate not passed")
    marker=model_root/".MODELS_SUCCESS"
    if marker.is_file() and not args.overwrite:
        summary=json.loads(marker.read_text(encoding="utf-8")); finish({"status":"skipped","variant":args.variant}); print(json.dumps(summary,indent=2)); return 0
    command=[config["prediction_python"],config["fixed_trainer"],"--task-root",str(task_root),"--output-dir",str(model_root),"--device","cuda:0"]
    if args.overwrite or model_root.exists(): command.append("--overwrite")
    run_checked(command,cwd=Path(config["project_root"])); summary=json.loads(marker.read_text(encoding="utf-8")); write_marker(reports/f".MODELS_{args.variant.upper()}_SUCCESS",f"12_train_adverse_models_fixed:{args.variant}",config,{"fixed_trainer":evidence},summary)
    if args.variant=="pred_roi": write_marker(reports/".MODELS_SUCCESS","12_train_adverse_models_fixed:pred_roi",config,{"fixed_trainer":evidence},summary)
    finish({"variant":args.variant}); return 0


if __name__=="__main__": raise SystemExit(main())

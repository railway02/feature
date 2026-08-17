#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from common import configure_runtime, load_config, run_checked, sha256_file, stage_logger, write_marker, atomic_json


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--config"); parser.add_argument("--branch",choices=["pred","gt","all_nonzero","all"],default="all"); args=parser.parse_args(); config=load_config(args.config); configure_runtime(config); finish=stage_logger("10_build_roi_cave_tables")
    root=Path(config["project_root"]); manifests=Path(config["paths"]["manifests"]); outputs=Path(config["paths"]["outputs"]); reports=Path(config["paths"]["reports"]); branches=["pred","gt","all_nonzero"] if args.branch=="all" else [args.branch]; summaries={}
    env=os.environ.copy(); env["PYTHONPATH"]=config["cave_code_root"]
    for branch in branches:
        summaries[branch]={}
        feature_root=outputs/f"cave_{branch}_roi_featurebank"
        for split in ("Train","Valid"):
            manifest=manifests/f"cave_manifest_{branch}_{split.casefold()}.csv"; output_dir=outputs/f"cave_{branch}_roi_tables"/split.casefold()
            command=[config["cave_python"],str(Path(config["cave_code_root"])/"build_feature_tables.py"),"--manifest",str(manifest),"--feature-root",str(feature_root),"--output-dir",str(output_dir),"--expected-split",split,"--verify-files"]
            run_checked(command,cwd=root,env=env)
            aggregation=[config["prediction_python"],str(Path(config["paths"]["code"])/"build_patient_aggregations.py"),"--source-dir",str(output_dir),"--output-dir",str(output_dir),"--rank-manifest",str(manifests/f"roi_manifest_{branch}.csv"),"--split",split]
            run_checked(aggregation,cwd=root)
            audit=json.loads((output_dir/"build_audit.json").read_text(encoding="utf-8")); audit["alternative_aggregations"]=json.loads((output_dir/"alternative_aggregation_audit.json").read_text(encoding="utf-8"))
            summaries[branch][split]=audit
    if "pred" in branches:
        summaries["whole_alternative"]={}
        for split in ("Train","Valid"):
            source=root/"outputs/api_fullseq_cave_v3_tables"/split.casefold(); output_dir=outputs/"whole_alternative_aggregation_tables"/split.casefold()
            aggregation=[config["prediction_python"],str(Path(config["paths"]["code"])/"build_patient_aggregations.py"),"--source-dir",str(source),"--output-dir",str(output_dir),"--rank-manifest",str(manifests/"roi_manifest_pred.csv"),"--split",split]
            run_checked(aggregation,cwd=root)
            summaries["whole_alternative"][split]=json.loads((output_dir/"alternative_aggregation_audit.json").read_text(encoding="utf-8"))
    atomic_json(summaries,reports/"roi_cave_table_summary.json"); write_marker(reports/".CAVE_FEATURES_SUCCESS","10_build_roi_cave_tables",config,{"manifests":{str(path):sha256_file(path) for branch in branches for path in [manifests/f"cave_manifest_{branch}_train.csv",manifests/f"cave_manifest_{branch}_valid.csv"]}},summaries); finish({"branches":branches}); print(json.dumps(summaries,ensure_ascii=False,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())

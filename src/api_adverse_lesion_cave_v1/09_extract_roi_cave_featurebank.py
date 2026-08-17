#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import common as roi_common
from assets import read_frames
from roi import bbox_from_text, crop_frames


def import_frozen_cave(code_root: Path):
    sys.path.insert(0,str(code_root.resolve()))
    for name in ["common","io_ops","manifest","pooling","release","scalar_features","schema","v3_bridge","cave_model","extract_cave_featurebank"]:
        sys.modules.pop(name,None)
    return importlib.import_module("extract_cave_featurebank"), importlib.import_module("io_ops")


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--config"); parser.add_argument("--branch",choices=["pred","gt","all_nonzero"],required=True); parser.add_argument("--split",choices=["Train","Valid"],required=True); parser.add_argument("--max-series",type=int); parser.add_argument("--overwrite",action="store_true"); args=parser.parse_args()
    config=roi_common.load_config(args.config); roi_common.configure_runtime(config); finish=roi_common.stage_logger(f"09_extract_roi_cave_featurebank:{args.branch}:{args.split}")
    manifests=Path(config["paths"]["manifests"]); outputs=Path(config["paths"]["outputs"]); reports=Path(config["paths"]["reports"])
    roi_path=manifests/f"roi_manifest_{args.branch}.csv"; manifest_path=manifests/f"cave_manifest_{args.branch}_{args.split.casefold()}.csv"
    roi_frame=pd.read_csv(roi_path,dtype=str,keep_default_na=False); roi_frame=roi_frame[(roi_frame.split==args.split)&(~roi_frame.duplicate_excluded.astype(str).str.casefold().eq("true"))].copy()
    by_hash={str(row.frame_list_hash):row._asdict() for row in roi_frame.itertuples(index=False)}
    if len(by_hash)!=len(roi_frame): raise AssertionError(f"ROI frame-list hash collision branch={args.branch} split={args.split}")
    base=json.loads(Path(config["base_cave_config"]).read_text(encoding="utf-8")); roi_sha=roi_common.sha256_file(roi_path); manifest_sha=roi_common.sha256_file(manifest_path)
    base.update({"roi_pipeline_version":config["version"],"roi_branch":args.branch,"roi_split":args.split,"roi_manifest_sha256":roi_sha,"roi_cave_manifest_sha256":manifest_sha,"roi_adapter":"post_load_gray_frames_pre_v3_preprocess_v1"})
    frozen_path=outputs/"cave_frozen_configs"/f"{args.branch}_{args.split.casefold()}.json"; roi_common.atomic_json(base,frozen_path)
    frozen_hash=roi_common.sha256_json(base); feature_root=outputs/f"cave_{args.branch}_roi_featurebank"
    required=("embedding_5120.npy","embedding_views_5120.npz","f4_last_ensemble.fp16.npy","f5_last_ensemble.fp16.npy","phase_trajectories_16.fp16.npz","probabilities_original.fp16.npz","curves.npz","scalar_features.json","metadata.json","qc.json",".SUCCESS.json")
    quarantined=[]
    for row in roi_frame.to_dict("records"):
        directory=feature_root/args.split.casefold()/str(row["patient_id"])/str(row["series_uid"])/str(row["phase"])
        if not directory.exists(): continue
        compatible=False
        try:
            success=json.loads((directory/".SUCCESS.json").read_text(encoding="utf-8"))
            compatible=bool(
                success.get("status")=="success"
                and success.get("frozen_config_hash")==frozen_hash
                and success.get("checkpoint_sha256")==config["checkpoint_sha256"]
                and success.get("manifest_sha256")==manifest_sha
                and success.get("frame_list_hash")==row["frame_list_hash"]
                and success.get("roi_manifest_sha256")==roi_sha
                and success.get("roi_branch")==args.branch
                and all((directory/name).is_file() for name in required)
            )
        except Exception: compatible=False
        if not compatible:
            target=roi_common.quarantine(directory,"stale_or_incomplete")
            quarantined.append({"source":str(directory),"target":str(target)})
    cave,io_ops=import_frozen_cave(Path(config["cave_code_root"])); original_load=io_ops.load_gray_frames; original_process=cave.process_phase

    def roi_load(paths, num_workers=4):
        frames=original_load(paths,num_workers=num_workers); key=io_ops.hash_lines([str(value) for value in paths])
        if key not in by_hash: raise KeyError(f"No ROI mapping for frame-list hash {key}")
        box=bbox_from_text(by_hash[key]["expanded_bbox"]); return crop_frames(frames,box)

    def roi_process(args_inner,extractor,v3,plan,provenance,schema_path):
        key=plan.frame_list_hash
        if key not in by_hash: raise KeyError(f"No ROI row for {plan.series_uid} {plan.phase}")
        row=by_hash[key]; result=original_process(args_inner,extractor,v3,plan,provenance,schema_path)
        directory=cave._phase_output_dir(args_inner.output_root,plan)
        if (directory/"metadata.json").is_file():
            metadata=json.loads((directory/"metadata.json").read_text(encoding="utf-8")); metadata["roi"]={
                "roi_pipeline_version":config["version"],"roi_branch":args.branch,"mask_source":"gt_all_nonzero" if args.branch=="all_nonzero" else "oof_pred" if args.branch=="pred" and args.split=="Train" else "valid_pred" if args.branch=="pred" else "gt",
                "segmentation_model_hash":row.get("segmentation_model_hash",""),"segmentation_fold":int(float(row.get("segmentation_fold",0) or 0)),"original_bbox":row.get("original_bbox",""),"expanded_bbox":row["expanded_bbox"],"crop_padding_factor":float(row["crop_padding_factor"]),"crop_padding":row.get("crop_padding","0|0|0|0"),"roi_area_ratio":float(row["roi_area_ratio"]),"mask_area_ratio":float(row["mask_area_ratio"]),"alignment_transform":row["orientation_transform"],"fallback_type":row["fallback_type"],"roi_manifest_sha256":roi_sha,
            }; roi_common.atomic_json(metadata,directory/"metadata.json")
            qc=json.loads((directory/"qc.json").read_text(encoding="utf-8")); qc.update({"roi_area_ratio":float(row["roi_area_ratio"]),"roi_mask_area_ratio":float(row["mask_area_ratio"]),"roi_fallback":int(row["fallback_type"]!="none")}); roi_common.atomic_json(qc,directory/"qc.json")
            success=json.loads((directory/".SUCCESS.json").read_text(encoding="utf-8")); success.update({"roi_manifest_sha256":roi_sha,"roi_branch":args.branch,"expanded_bbox":row["expanded_bbox"]}); roi_common.atomic_json(success,directory/".SUCCESS.json")
        return result

    cave.load_gray_frames=roi_load; cave.process_phase=roi_process
    cave_report=reports/f"cave_{args.branch}_roi"
    argv=["extract_cave_featurebank.py","--mode","custom","--manifest",str(manifest_path),"--cave-repo",config["cave_repo"],"--checkpoint",config["checkpoint"],"--v3-extractor",config["v3_extractor"],"--v3-base-config",config["v3_base_config"],"--v3-override-config",config["v3_override_config"],"--output-root",str(feature_root),"--report-root",str(cave_report),"--frozen-config",str(frozen_path),"--io-workers",str(config["runtime"]["io_workers"])]
    if args.max_series: argv.extend(["--max-series",str(args.max_series)])
    if args.overwrite: argv.append("--overwrite")
    old_argv=sys.argv; sys.argv=argv
    try:
        code=cave.main()
        if code:
            print(f"[RETRY] CAVE phase failures detected branch={args.branch} split={args.split}",flush=True)
            code=cave.main()
    finally: sys.argv=old_argv
    failure_rows=[]; failure_rate=0.0
    if code:
        run_index=pd.read_csv(feature_root/"run_index.csv",dtype=str,keep_default_na=False)
        failure_rows=run_index[run_index.status=="failed"].to_dict("records")
        failure_rate=len(failure_rows)/max(len(run_index),1)
        if failure_rate>float(config["gates"]["maximum_cave_phase_failure_rate"]):
            raise RuntimeError(f"CAVE failure rate {failure_rate:.4f} exceeds gate after retry branch={args.branch} split={args.split}")
        print(f"[EXCLUDE] persistent CAVE failures={len(failure_rows)} rate={failure_rate:.4f}",flush=True)
        code=0
    suffix="_SMOKE_SUCCESS" if args.max_series else "_SUCCESS"
    marker=reports/f".CAVE_{args.branch.upper()}_{args.split.upper()}{suffix}"; payload={"branch":args.branch,"split":args.split,"roi_manifest_sha256":roi_sha,"cave_manifest_sha256":manifest_sha,"feature_root":str(feature_root),"max_series":args.max_series,"quarantined":quarantined,"automatic_retry":True,"persistent_failures":failure_rows,"failure_rate":failure_rate}
    roi_common.write_marker(marker,f"09_extract_roi_cave_featurebank:{args.branch}:{args.split}",config,{"roi_manifest_sha256":roi_sha,"cave_manifest_sha256":manifest_sha},payload); finish(payload); return int(code)


if __name__=="__main__": raise SystemExit(main())

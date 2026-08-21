#!/usr/bin/env python
import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))
import argparse
from pathlib import Path
import pandas as pd
from dsa_reg.utils import load_yaml, ensure_dir
from dsa_reg.manifest import load_manifest, assert_no_cross_split_patient_leakage, select_series_policy, audit_manifest

p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--manifest",default=None); p.add_argument("--output-root",default=None); args=p.parse_args()
cfg=load_yaml(args.config); manifest=args.manifest or cfg["paths"]["manifest"]
if args.output_root: cfg["paths"]["output_root"]=args.output_root
df=load_manifest(manifest); assert_no_cross_split_patient_leakage(df)
print(f"rows={len(df)}, unique_patients={df.patient_id.nunique()}, split={df.split.value_counts().to_dict()}")
dups=df[df.duplicated("patient_id",keep=False)]
print(f"patients_with_multiple_series={dups.patient_id.nunique()}")
sel=select_series_policy(df,cfg["manifest"].get("patient_series_policy","all"))
print(f"selected_rows={len(sel)}, selected_patients={sel.patient_id.nunique()}")
out=ensure_dir(cfg["paths"]["output_root"])
a=audit_manifest(sel,cfg["paths"].get("remap",{})); a.to_csv(out/"manifest_audit.csv",index=False)
print(a[["frame_count_ok","pre_reference_exists","post_reference_exists","pre_mask_exists","post_mask_exists","pre_all_frames_exist","post_all_frames_exist"]].mean())
print("saved",out/"manifest_audit.csv")

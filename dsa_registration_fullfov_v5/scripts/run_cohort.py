#!/usr/bin/env python
import os
# Some hosted environments export OMP_NUM_THREADS=0, which libgomp rejects.  Correct it
# before importing NumPy/ANTs; launch_background_cohort.py supplies an explicit positive
# value for production throughput and bounded worker oversubscription.
if not os.environ.get("OMP_NUM_THREADS", "").strip().isdigit() or int(os.environ.get("OMP_NUM_THREADS", "0")) < 1:
    os.environ["OMP_NUM_THREADS"] = "1"
import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))
import argparse, traceback, json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from dsa_reg.utils import load_yaml, ensure_dir
from dsa_reg.manifest import load_manifest, assert_no_cross_split_patient_leakage, select_series_policy
from dsa_reg.pipeline import process_series

p=argparse.ArgumentParser()
p.add_argument("--config",required=True); p.add_argument("--manifest",default=None)
p.add_argument("--split",default=None); p.add_argument("--patient-id",type=int,default=None); p.add_argument("--limit",type=int,default=None)
p.add_argument("--series-uid",action="append",default=None,help="Exact manifest series UID; repeatable for controlled technical reruns.")
p.add_argument("--output-root",default=None)
p.add_argument("--pilot-fast",action="store_true",
               help="CPU pilot settings; requires --limit <= 5 and is not a full-cohort protocol.")
p.add_argument("--continue-on-error",action="store_true")
p.add_argument("--workers",type=int,default=1,
               help="Independent series workers. Use CPU/ANTs thread limits in the launcher to avoid oversubscription.")
p.add_argument("--resume",action="store_true",help="Reuse an existing per-series features.json; never overwrite completed cases.")
p.add_argument("--allow-exploratory-tau",action="store_true",
               help="Allow fallback tau. Never use this for locked Valid/full-cohort experiments.")
args=p.parse_args(); cfg=load_yaml(args.config); manifest=args.manifest or cfg["paths"]["manifest"]
if args.output_root:
    cfg["paths"]["output_root"] = args.output_root
if args.pilot_fast:
    if args.limit is None or args.limit > 5:
        raise ValueError("--pilot-fast requires --limit between 1 and 5")
    cfg["global_registration"]["run_methods"] = ["rigid", "similarity"]
    cfg["roi"]["size"] = [256, 256]
    cfg["global_registration"]["iterations"] = min(50, int(cfg["global_registration"]["iterations"]))
    cfg["intra_registration"]["iterations"] = min(25, int(cfg["intra_registration"]["iterations"]))
    cfg["intra_registration"]["min_vessel_score_ratio"] = max(
        0.50, float(cfg["intra_registration"].get("min_vessel_score_ratio", 0.15))
    )
    cfg["nonrigid"]["reg_iterations"] = [10, 5, 2]
    print("PILOT-FAST MODE: reduced CPU iterations; not valid for final cohort inference")
if cfg.get("nonrigid",{}).get("enabled",True):
    tau_artifact=cfg.get("features",{}).get("expansion_tau_artifact")
    if not tau_artifact and not args.allow_exploratory_tau:
        raise RuntimeError(
            "A Train-calibrated features.expansion_tau_artifact is required. "
            "Use --allow-exploratory-tau only for synthetic/Train pilot work."
        )
df=load_manifest(manifest); assert_no_cross_split_patient_leakage(df); df=select_series_policy(df,cfg["manifest"].get("patient_series_policy","all"))
if args.split: df=df[df.split==args.split]
if args.patient_id is not None: df=df[df.patient_id==args.patient_id]
if args.series_uid: df=df[df.series_uid.astype(str).isin(set(map(str,args.series_uid)))]
if args.limit: df=df.head(args.limit)
def invalid_result(row, e):
    return {
        "patient_id": int(row.patient_id), "split": str(row.split),
        "series_uid": str(row.series_uid), "series_id": str(row.series_id),
        "registration_valid": 0, "q_reg": 0.0, "neck_feature_available": False,
        "registration_invalid_reasons": "pipeline_exception:" + type(e).__name__,
        "pipeline_exception": repr(e),
    }

def completed_path(row, output_root):
    from dsa_reg.utils import sanitize_key
    return Path(output_root) / str(row.split) / str(row.patient_id) / sanitize_key(str(row.series_uid)) / "features.json"

def worker(payload):
    row_dict, worker_cfg = payload
    return process_series(pd.Series(row_dict), worker_cfg)

results=[]; errors=[]; pending=[]
for _, row in df.iterrows():
    done = completed_path(row, cfg["paths"]["output_root"])
    if args.resume and done.exists():
        try:
            results.append(json.loads(done.read_text()))
            continue
        except Exception as e:
            errors.append({"patient_id": row.patient_id, "series_uid": row.series_uid,
                           "error": "resume_read_failed:" + repr(e), "traceback": traceback.format_exc()})
    pending.append(row)

if args.workers < 1:
    raise ValueError("--workers must be >= 1")
if args.workers == 1:
    iterator = tqdm(pending, total=len(pending))
    for row in iterator:
        try: results.append(process_series(row,cfg))
        except Exception as e:
            errors.append({"patient_id":row.patient_id,"series_uid":row.series_uid,"error":repr(e),"traceback":traceback.format_exc()})
            # Retain every manifest series in the master table.  This is intentionally a
            # neutral/fail-closed representation, not imputation of an apparent deformation.
            results.append(invalid_result(row, e))
            if not args.continue_on_error: raise
else:
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, (row.to_dict(), cfg)): row for row in pending}
        for future in tqdm(as_completed(futures), total=len(futures)):
            row = futures[future]
            try:
                results.append(future.result())
            except Exception as e:
                errors.append({"patient_id":row.patient_id,"series_uid":row.series_uid,"error":repr(e),"traceback":traceback.format_exc()})
                results.append(invalid_result(row, e))
                if not args.continue_on_error:
                    for f in futures: f.cancel()
                    raise
out=ensure_dir(cfg["paths"]["output_root"])
if results:
    pd.DataFrame(results).sort_values(["split", "patient_id", "series_uid"]).to_csv(
        out/"registration_features_series.csv",index=False
    )
if errors: pd.DataFrame(errors).to_csv(out/"registration_errors.csv",index=False)
print(f"done: retained={len(results)} exception={len(errors)} resumed={len(df)-len(pending)}")

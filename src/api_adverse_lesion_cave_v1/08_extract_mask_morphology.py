#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common import atomic_csv, atomic_json, configure_runtime, load_config, sha256_file, stage_logger, write_marker
from roi import mask_features


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--config"); args=parser.parse_args(); config=load_config(args.config); configure_runtime(config); finish=stage_logger("08_extract_mask_morphology")
    manifests=Path(config["paths"]["manifests"]); outputs=Path(config["paths"]["outputs"]); reports=Path(config["paths"]["reports"])
    roi_path=manifests/"roi_manifest_all_branches.csv"; roi=pd.read_csv(roi_path,dtype=str,keep_default_na=False); feature_root=outputs/"mask_morphology"; summaries={}
    for branch in ("pred","gt","all_nonzero"):
        phase_rows=[]
        for row in roi[roi.roi_branch==branch].to_dict("records"):
            mask=cv2.imread(row["roi_mask_path"],cv2.IMREAD_GRAYSCALE)
            if mask is None: raise RuntimeError(row["roi_mask_path"])
            phase_rows.append({"patient_id":row["patient_id"],"series_uid":row["series_uid"],"split":row["split"],"phase":row["phase"],**mask_features(mask)})
        phase=pd.DataFrame(phase_rows); series_rows=[]
        feature_cols=[column for column in phase.columns if column not in {"patient_id","series_uid","split","phase"}]
        for (patient_id,series_uid,split),group in phase.groupby(["patient_id","series_uid","split"],sort=True):
            record={"patient_id":patient_id,"series_uid":series_uid,"split":split,"missing_pre":int(not (group.phase=="pre").any()),"missing_post":int(not (group.phase=="post").any())}
            for phase_name in ("pre","post"):
                item=group[group.phase==phase_name]
                for column in feature_cols: record[f"{phase_name}_{column}"]=float(item.iloc[0][column]) if len(item) else np.nan
            for column in feature_cols:
                a=record[f"pre_{column}"]; b=record[f"post_{column}"]; record[f"delta_{column}"]=float(b-a) if np.isfinite(a) and np.isfinite(b) else np.nan
            series_rows.append(record)
        series=pd.DataFrame(series_rows); numeric=[c for c in series.columns if c not in {"patient_id","series_uid","split"}]
        patients=[]
        for (patient_id,split),group in series.groupby(["patient_id","split"],sort=True):
            med=group[numeric].apply(pd.to_numeric,errors="coerce").median(axis=0,skipna=True); patients.append({"patient_id":patient_id,"split":split,"series_count":len(group),**med.to_dict()})
        patient=pd.DataFrame(patients); feature_root.mkdir(parents=True,exist_ok=True); atomic_csv(phase,feature_root/f"{branch}_phase.csv"); atomic_csv(series,feature_root/f"{branch}_series.csv"); atomic_csv(patient,feature_root/f"{branch}_patient_median.csv")
        summaries[branch]={"phases":len(phase),"series":len(series),"patients":len(patient),"patient_columns":len(patient.columns)}
    summary={"branches":summaries,"source_roi_manifest_sha256":sha256_file(roi_path)}; atomic_json(summary,reports/"mask_morphology_summary.json"); write_marker(reports/".MASK_FEATURES_SUCCESS","08_extract_mask_morphology",config,{"roi_manifest_sha256":sha256_file(roi_path)},summary); finish(summary); print(json.dumps(summary,ensure_ascii=False,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())

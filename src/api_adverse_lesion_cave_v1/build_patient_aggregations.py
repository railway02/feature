#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from common import atomic_csv, atomic_json, bool_value


META_COLUMNS={"patient_id","series_uid","split","source_type","series_id"}


def reduce_values(values:np.ndarray,kind:str)->np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",category=RuntimeWarning)
        if kind=="max": return np.nanmax(values,axis=0).astype(np.float32)
        if kind=="top2_mean": return np.nanmean(values,axis=0).astype(np.float32)
    raise ValueError(kind)


def build(source_dir:Path,output_dir:Path,rank_manifest:Path,split:str)->dict:
    scalar_path=source_dir/"series_scalar_features.parquet"
    series=pd.read_parquet(scalar_path) if scalar_path.is_file() else pd.read_csv(source_dir/"series_scalar_features.csv")
    series["patient_id"]=series.patient_id.astype(str); series["series_uid"]=series.series_uid.astype(str)
    raw=np.load(source_dir/"series_embeddings_5120.npz")
    uids=raw["series_uid"].astype(str); embeddings=raw["embeddings"].astype(np.float32)
    if embeddings.ndim!=3 or embeddings.shape[1:]!=(2,5120): raise AssertionError(embeddings.shape)
    embedding_index={uid:index for index,uid in enumerate(uids)}
    if set(series.series_uid)!=set(embedding_index): raise AssertionError("Series scalar/embedding mismatch")
    roi=pd.read_csv(rank_manifest,dtype=str,keep_default_na=False)
    roi=roi[(roi.split==split)&(~roi.duplicate_excluded.map(bool_value))].copy()
    roi["mask_area_ratio"]=pd.to_numeric(roi.mask_area_ratio,errors="coerce")
    ranks=roi.groupby("series_uid").mask_area_ratio.max().to_dict()
    numeric=[c for c in series.columns if c not in META_COLUMNS and pd.api.types.is_numeric_dtype(series[c])]
    output_dir.mkdir(parents=True,exist_ok=True); summaries={}
    for kind in ("max","top2_mean"):
        patient_ids=sorted(series.patient_id.unique()); patient_embeddings=[]; rows=[]
        for patient_id in patient_ids:
            group=series[series.patient_id==patient_id].copy()
            ordered=sorted(group.series_uid.tolist(),key=lambda uid:(-float(ranks.get(uid,float("-inf"))),uid))
            selected=ordered if kind=="max" else ordered[:2]
            values=np.stack([embeddings[embedding_index[uid]] for uid in selected])
            aggregate=reduce_values(values,kind); patient_embeddings.append(aggregate)
            numeric_values=group.set_index("series_uid").loc[selected,numeric].apply(pd.to_numeric,errors="coerce").to_numpy(np.float32)
            scalar=reduce_values(numeric_values,kind)
            row={"patient_id":patient_id,"split":str(group.split.iloc[0]),"series_count":len(group),"selected_series_count":len(selected),"selected_series_uids":"|".join(selected),"missing_pre_all":int(np.isnan(aggregate[0]).all()),"missing_post_all":int(np.isnan(aggregate[1]).all())}
            row.update({column:float(scalar[index]) for index,column in enumerate(numeric)})
            rows.append(row)
        array=np.stack(patient_embeddings).astype(np.float32); frame=pd.DataFrame(rows)
        prefix=f"patient_{kind}"
        np.savez_compressed(output_dir/f"{prefix}_embeddings_5120.npz",patient_id=np.asarray(patient_ids,dtype=str),embeddings=array,missing_pre=np.isnan(array[:,0]).all(1).astype(np.uint8),missing_post=np.isnan(array[:,1]).all(1).astype(np.uint8))
        atomic_csv(frame,output_dir/f"{prefix}_scalar_features.csv")
        try: frame.to_parquet(output_dir/f"{prefix}_scalar_features.parquet",index=False)
        except Exception: pass
        summaries[kind]={"patients":len(frame),"embedding_shape":list(array.shape),"ranked_series":int(sum(uid in ranks for uid in series.series_uid))}
    audit={"source_dir":str(source_dir),"output_dir":str(output_dir),"rank_manifest":str(rank_manifest),"split":split,"ranking":"descending mask_area_ratio, series_uid tie-break; outcome-blind","aggregations":summaries}
    atomic_json(audit,output_dir/"alternative_aggregation_audit.json")
    print(json.dumps(audit,ensure_ascii=False,indent=2)); return audit


def main()->int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--rank-manifest",type=Path,required=True)
    parser.add_argument("--split",choices=["Train","Valid"],required=True)
    args=parser.parse_args(); build(args.source_dir,args.output_dir,args.rank_manifest,args.split); return 0


if __name__=="__main__": raise SystemExit(main())

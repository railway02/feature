#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import atomic_csv, atomic_json, configure_runtime, load_config, normalize_patient_id, sha256_file, stage_logger, write_marker


META_COLUMNS={"patient_id","split","series_count","missing_pre","missing_post","missing_pre_all","missing_post_all"}


class PatientStore:
    def __init__(self,table_dir:Path,aggregation:str="median"):
        prefix=f"patient_{aggregation}"
        parquet=table_dir/f"{prefix}_scalar_features.parquet"; csv=table_dir/f"{prefix}_scalar_features.csv"; scalar_path=parquet if parquet.is_file() else csv
        if not scalar_path.is_file(): raise FileNotFoundError(scalar_path)
        self.scalar=pd.read_parquet(scalar_path) if scalar_path.suffix==".parquet" else pd.read_csv(scalar_path)
        self.scalar["patient_id"]=self.scalar.patient_id.astype(str); self.scalar=self.scalar.set_index("patient_id",drop=False)
        raw=np.load(table_dir/f"{prefix}_embeddings_5120.npz"); ids=raw["patient_id"].astype(str); emb=raw["embeddings"].astype(np.float32)
        if emb.ndim!=3 or emb.shape[1:]!=(2,5120): raise AssertionError(f"Unexpected embeddings {emb.shape}")
        self.emb={uid:emb[index] for index,uid in enumerate(ids)}
        excluded=META_COLUMNS|{"patient_id"}; self.columns=[c for c in self.scalar.columns if c not in excluded and pd.api.types.is_numeric_dtype(self.scalar[c])]
        if not self.columns: raise AssertionError(f"No scalar columns {table_dir}")

    def ids(self): return set(self.scalar.index.astype(str))&set(self.emb)

    def extract(self,ids:list[str],phase_mode:str="prepost"):
        deep=np.stack([self.emb[uid] for uid in ids]).astype(np.float32); rows=self.scalar.loc[ids].copy(); scalar=rows[self.columns].apply(pd.to_numeric,errors="coerce").to_numpy(np.float32); columns=list(self.columns)
        missing=np.stack([np.isnan(deep[:,0]).all(1),np.isnan(deep[:,1]).all(1)],axis=1).astype(np.float32)
        if phase_mode=="pre":
            deep[:,1]=np.nan; missing[:,1]=1
            for index,column in enumerate(columns):
                if column.startswith("post_") or column.startswith("delta_") or "prepost" in column: scalar[:,index]=np.nan
        elif phase_mode=="post":
            deep[:,0]=np.nan; missing[:,0]=1
            for index,column in enumerate(columns):
                if column.startswith("pre_") or column.startswith("delta_") or "prepost" in column: scalar[:,index]=np.nan
        return deep.reshape(len(ids),-1),scalar,missing,columns


def consistent_labels(path:Path,split:str):
    frame=pd.read_excel(path,dtype=object); frame["patient_id"]=frame["病案号"].map(normalize_patient_id); adverse=next(c for c in frame.columns if str(c).startswith("不良转归")); rows=[]; conflicts=[]
    for patient,group in frame.groupby("patient_id"):
        values=pd.to_numeric(group[adverse],errors="coerce").dropna().astype(int).unique().tolist()
        if len(values)==1 and values[0] in {0,1}: rows.append({"patient_id":patient,"split":split,"target":values[0]})
        elif len(values)>1: conflicts.append({"patient_id":patient,"split":split,"values":"|".join(map(str,sorted(values))),"rows":len(group)})
    follow=next((c for c in frame.columns if str(c).startswith("随访RROC")),None); relation={"rows_with_both":0,"exact_matches":0}
    if follow:
        a=pd.to_numeric(frame[adverse],errors="coerce"); f=pd.to_numeric(frame[follow],errors="coerce"); valid=a.isin([0,1])&f.isin([1,2,3]); derived=(f!=1).astype(float); relation={"rows_with_both":int(valid.sum()),"exact_matches":int((a[valid]==derived[valid]).sum()),"exact_fraction":float((a[valid]==derived[valid]).mean()) if valid.any() else None}
    return pd.DataFrame(rows),pd.DataFrame(conflicts),relation


def write_cave_task(task_root:Path,train_store:PatientStore,valid_store:PatientStore,train_meta:pd.DataFrame,valid_meta:pd.DataFrame,phase_mode:str,variant_name:str):
    task_dir=task_root/"adverse_patient"; task_dir.mkdir(parents=True,exist_ok=True); train_ids=train_meta.patient_id.astype(str).tolist(); valid_ids=valid_meta.patient_id.astype(str).tolist()
    tr_deep,tr_scalar,tr_missing,tr_cols=train_store.extract(train_ids,phase_mode); va_deep,va_scalar,va_missing,va_cols=valid_store.extract(valid_ids,phase_mode)
    if tr_cols!=va_cols: raise AssertionError("Train/Valid scalar schema mismatch")
    np.savez_compressed(task_dir/"train_features.npz",deep=tr_deep,scalar=tr_scalar,missing=tr_missing,target=train_meta.target.to_numpy(np.int64)); np.savez_compressed(task_dir/"valid_features.npz",deep=va_deep,scalar=va_scalar,missing=va_missing,target=valid_meta.target.to_numpy(np.int64))
    atomic_csv(train_meta,task_dir/"train_meta.csv"); atomic_csv(valid_meta,task_dir/"valid_meta.csv")
    config={"task_name":"adverse_patient","variant_name":variant_name,"phase_mode":phase_mode,"train":{"rows":len(train_meta),"positive":int(train_meta.target.sum())},"valid":{"rows":len(valid_meta),"positive":int(valid_meta.target.sum())},"deep_dimension":tr_deep.shape[1],"scalar_dimension":tr_scalar.shape[1],"missing_dimension":2,"scalar_columns":tr_cols,"valid_used_for_selection":False}; atomic_json(config,task_dir/"task_config.json"); atomic_json(config,task_root/".TASKS_SUCCESS")


def write_morphology_task(task_root:Path,train_path:Path,valid_path:Path,train_meta:pd.DataFrame,valid_meta:pd.DataFrame,branch:str):
    all_frame=pd.read_csv(train_path); all_frame["patient_id"]=all_frame.patient_id.astype(str); train=all_frame[all_frame.split=="Train"].set_index("patient_id"); valid=all_frame[all_frame.split=="Valid"].set_index("patient_id")
    columns=[c for c in all_frame.columns if c not in {"patient_id","split","series_count"} and pd.api.types.is_numeric_dtype(all_frame[c])]
    task=task_root/"adverse_patient"; task.mkdir(parents=True,exist_ok=True)
    def arrays(meta,store):
        values=store.loc[meta.patient_id.astype(str).tolist(),columns].apply(pd.to_numeric,errors="coerce").to_numpy(np.float32); return values,values.copy(),np.zeros((len(values),2),np.float32)
    td,ts,tm=arrays(train_meta,train); vd,vs,vm=arrays(valid_meta,valid)
    np.savez_compressed(task/"train_features.npz",deep=td,scalar=ts,missing=tm,target=train_meta.target.to_numpy(np.int64)); np.savez_compressed(task/"valid_features.npz",deep=vd,scalar=vs,missing=vm,target=valid_meta.target.to_numpy(np.int64)); atomic_csv(train_meta,task/"train_meta.csv"); atomic_csv(valid_meta,task/"valid_meta.csv"); cfg={"task_name":"adverse_patient","variant_name":f"{branch}_morphology","train_rows":len(train_meta),"valid_rows":len(valid_meta),"columns":columns}; atomic_json(cfg,task/"task_config.json"); atomic_json(cfg,task_root/".TASKS_SUCCESS")


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--config"); args=parser.parse_args(); config=load_config(args.config); configure_runtime(config); finish=stage_logger("11_build_adverse_tasks")
    outputs=Path(config["paths"]["outputs"]); reports=Path(config["paths"]["reports"]); task_base=outputs/"adverse_tasks"
    train_labels,train_conflicts,train_relation=consistent_labels(Path(config["train_excel"]),"Train"); valid_labels,valid_conflicts,valid_relation=consistent_labels(Path(config["valid_excel"]),"Valid")
    stores={
        "pred":(PatientStore(outputs/"cave_pred_roi_tables"/"train"),PatientStore(outputs/"cave_pred_roi_tables"/"valid")),
        "gt":(PatientStore(outputs/"cave_gt_roi_tables"/"train"),PatientStore(outputs/"cave_gt_roi_tables"/"valid")),
        "whole":(PatientStore(Path(config["project_root"])/"outputs/api_fullseq_cave_v3_tables/train"),PatientStore(Path(config["project_root"])/"outputs/api_fullseq_cave_v3_tables/valid")),
        "all_nonzero":(PatientStore(outputs/"cave_all_nonzero_roi_tables"/"train"),PatientStore(outputs/"cave_all_nonzero_roi_tables"/"valid")),
        "pred_max":(PatientStore(outputs/"cave_pred_roi_tables"/"train","max"),PatientStore(outputs/"cave_pred_roi_tables"/"valid","max")),
        "pred_top2":(PatientStore(outputs/"cave_pred_roi_tables"/"train","top2_mean"),PatientStore(outputs/"cave_pred_roi_tables"/"valid","top2_mean")),
        "whole_max":(PatientStore(outputs/"whole_alternative_aggregation_tables"/"train","max"),PatientStore(outputs/"whole_alternative_aggregation_tables"/"valid","max")),
        "whole_top2":(PatientStore(outputs/"whole_alternative_aggregation_tables"/"train","top2_mean"),PatientStore(outputs/"whole_alternative_aggregation_tables"/"valid","top2_mean")),
    }
    train_ids=stores["pred"][0].ids()&stores["gt"][0].ids()&stores["all_nonzero"][0].ids()&stores["whole"][0].ids()&set(train_labels.patient_id); valid_ids=stores["pred"][1].ids()&stores["gt"][1].ids()&stores["all_nonzero"][1].ids()&stores["whole"][1].ids()&set(valid_labels.patient_id)
    train_meta=train_labels[train_labels.patient_id.isin(train_ids)].sort_values("patient_id").reset_index(drop=True); valid_meta=valid_labels[valid_labels.patient_id.isin(valid_ids)].sort_values("patient_id").reset_index(drop=True)
    gate=config["gates"]; coverage={"train":len(train_meta)/max(len(train_labels),1),"valid":len(valid_meta)/max(len(valid_labels),1)}; eligible=coverage["train"]>=gate["minimum_train_patient_coverage"] and coverage["valid"]>=gate["minimum_valid_patient_coverage"] and int(train_meta.target.sum())>=gate["minimum_train_positive"] and int(valid_meta.target.sum())>=gate["minimum_valid_positive"]
    if task_base.exists(): shutil.rmtree(task_base)
    for name,(branch,phase) in {"pred_roi":("pred","prepost"),"pred_roi_pre":("pred","pre"),"pred_roi_post":("pred","post"),"pred_roi_max":("pred_max","prepost"),"pred_roi_top2_mean":("pred_top2","prepost"),"gt_roi":("gt","prepost"),"all_nonzero_roi":("all_nonzero","prepost"),"whole_same_cohort":("whole","prepost"),"whole_same_cohort_max":("whole_max","prepost"),"whole_same_cohort_top2_mean":("whole_top2","prepost")}.items(): write_cave_task(task_base/name,stores[branch][0],stores[branch][1],train_meta,valid_meta,phase,name)
    morphology_root=outputs/"mask_morphology"
    for branch in ("pred","gt"): write_morphology_task(task_base/f"{branch}_morphology",morphology_root/f"{branch}_patient_median.csv",morphology_root/f"{branch}_patient_median.csv",train_meta,valid_meta,branch)
    conflicts=pd.concat([train_conflicts,valid_conflicts],ignore_index=True); atomic_csv(conflicts,reports/"adverse_label_conflicts.csv")
    summary={"eligible_for_models":eligible,"train_rows":len(train_meta),"valid_rows":len(valid_meta),"train_positive":int(train_meta.target.sum()),"valid_positive":int(valid_meta.target.sum()),"coverage":coverage,"adverse_followup_relation":{"Train":train_relation,"Valid":valid_relation},"task_variants":[p.name for p in task_base.iterdir() if p.is_dir()],"train_valid_overlap":len(set(train_meta.patient_id)&set(valid_meta.patient_id))}
    atomic_json(summary,reports/"adverse_task_audit.json")
    stop_marker=reports/".STOPPED_INSUFFICIENT_COHORT"
    if not eligible:
        atomic_json(summary,stop_marker)
    else:
        if stop_marker.is_file(): stop_marker.unlink()
        write_marker(reports/".ADVERSE_TASKS_SUCCESS","11_build_adverse_tasks",config,{"pred_train_tables":sha256_file(outputs/"cave_pred_roi_tables"/"train"/"build_audit.json")},summary)
    finish(summary); print(json.dumps(summary,ensure_ascii=False,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())

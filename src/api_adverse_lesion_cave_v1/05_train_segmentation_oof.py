#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from common import atomic_csv, atomic_json, configure_runtime, load_config, quarantine, sha256_file, sha256_json, stage_logger, write_marker
from segmentation import predict_frame, restore_model_probability, threshold_metrics, train_model


def atomic_probability(path: Path, probability: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(f".{path.name}.{os.getpid()}.tmp.npz"); np.savez_compressed(tmp,probability=probability.astype(np.float16)); os.replace(tmp,path)


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--config"); parser.add_argument("--max-samples",type=int); parser.add_argument("--epochs",type=int); parser.add_argument("--output-root"); args=parser.parse_args()
    config=load_config(args.config); configure_runtime(config); finish=stage_logger("05_train_segmentation_oof")
    manifests=Path(config["paths"]["manifests"]); outputs=Path(args.output_root or config["paths"]["outputs"]); reports=Path(config["paths"]["reports"])
    index_path=manifests/"segmentation_dataset_index.csv"; index=pd.read_csv(index_path,dtype=str,keep_default_na=False)
    index=index[index["split"]=="Train"].copy(); index["lesion_area_ratio"]=pd.to_numeric(index["lesion_area_ratio"])
    if args.max_samples:
        keep=index["patient_id"].drop_duplicates().head(args.max_samples).tolist(); index=index[index["patient_id"].isin(keep)].copy()
    seg_cfg=dict(config["segmentation"]); seg_cfg["input_channels_count"]=len(seg_cfg["input_channels"])
    if args.epochs: seg_cfg["max_epochs"]=args.epochs; seg_cfg["early_stop"]=max(1,args.epochs)
    model_root=outputs/"segmentation_models"; prediction_root=outputs/"segmentation_oof_probabilities"
    all_predictions=[]; all_folds=[]; thresholds={}; phase_summaries={}
    for phase in ("pre","post"):
        phase_frame=index[index["phase"]==phase].reset_index(drop=True)
        if phase_frame.empty: continue
        unique_patients=phase_frame["patient_id"].nunique(); requested=min(int(seg_cfg["folds"]),unique_patients)
        if requested<2: raise AssertionError(f"Insufficient patients for {phase} folds")
        splitter=GroupKFold(n_splits=requested); groups=phase_frame["patient_id"].to_numpy()
        oof_model_prob={}; best_epochs=[]; fold_audits=[]
        for fold,(train_idx,hold_idx) in enumerate(splitter.split(phase_frame,groups=groups),1):
            train_frame=phase_frame.iloc[train_idx].copy(); hold_frame=phase_frame.iloc[hold_idx].copy()
            overlap=set(train_frame.patient_id)&set(hold_frame.patient_id)
            if overlap: raise AssertionError(f"Patient leakage fold={fold} phase={phase}")
            checkpoint=model_root/phase/f"fold_{fold}.pt"
            audit_path=model_root/phase/f"fold_{fold}_audit.json"
            resume_signature=sha256_json({"segmentation":seg_cfg,"phase":phase,"fold":fold,"train":sorted(train_frame.sample_uid.tolist()),"holdout":sorted(hold_frame.sample_uid.tolist())})
            compatible=False
            if checkpoint.is_file() and audit_path.is_file():
                try:
                    audit=json.loads(audit_path.read_text(encoding="utf-8"))
                    compatible=audit.get("resume_signature")==resume_signature and audit.get("checkpoint_sha256")==sha256_file(checkpoint)
                except Exception: compatible=False
            if not compatible:
                for item in (checkpoint,audit_path): quarantine(item,"resume_mismatch")
                audit=train_model(train_frame,hold_frame,seg_cfg,checkpoint,int(seg_cfg["seed"])+fold,args.epochs)
                audit["resume_signature"]=resume_signature; audit["checkpoint_sha256"]=sha256_file(checkpoint)
                atomic_json(audit,audit_path)
            from segmentation import load_model
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); model=load_model(checkpoint,device)
            probabilities=predict_frame(model,hold_frame,int(seg_cfg["batch_size"]))
            oof_model_prob.update(probabilities); best_epochs.append(int(audit["best_epoch"] or audit["epochs_ran"]))
            for row in hold_frame.to_dict("records"):
                all_folds.append({"sample_uid":row["sample_uid"],"patient_id":row["patient_id"],"phase":phase,"fold":fold})
            fold_audits.append({"fold":fold,"train_samples":len(train_frame),"holdout_samples":len(hold_frame),"train_patients":train_frame.patient_id.nunique(),"holdout_patients":hold_frame.patient_id.nunique(),"best_epoch":audit["best_epoch"],"best_score":audit["best_score"]})
            del model; torch.cuda.empty_cache()
        if set(oof_model_prob)!=set(phase_frame.sample_uid): raise AssertionError(f"Incomplete {phase} OOF predictions")
        threshold_rows=[]
        values=np.arange(float(seg_cfg["threshold_min"]),float(seg_cfg["threshold_max"])+1e-9,float(seg_cfg["threshold_step"]))
        for threshold in values:
            metrics=[]
            for row in phase_frame.to_dict("records"):
                raw=np.load(row["sample_path"],allow_pickle=False); gt=raw["mask"].astype(np.uint8)
                metrics.append(threshold_metrics(oof_model_prob[row["sample_uid"]],gt,float(threshold),config["roi"]))
            average={key:float(np.mean([item[key] for item in metrics])) for key in metrics[0]}
            threshold_rows.append({"phase":phase,"threshold":float(threshold),**average})
        threshold_frame=pd.DataFrame(threshold_rows); best=threshold_frame.sort_values(["score","coverage","roi_area_ratio"],ascending=[False,False,True]).iloc[0]
        threshold=float(best["threshold"]); thresholds[phase]=threshold
        threshold_frame.to_csv(model_root/phase/"threshold_search.csv",index=False)
        for row in phase_frame.to_dict("records"):
            original=restore_model_probability(row["sample_path"],oof_model_prob[row["sample_uid"]])
            path=prediction_root/phase/f"{row['sample_uid']}.npz"; atomic_probability(path,original)
            fold_value=next(item["fold"] for item in all_folds if item["sample_uid"]==row["sample_uid"])
            model_hash=sha256_file(model_root/phase/f"fold_{fold_value}.pt")
            all_predictions.append({"sample_uid":row["sample_uid"],"phase_uid":row["phase_uid"],"patient_id":row["patient_id"],"split":"Train","phase":phase,"series_uid":row["series_uid"],"prediction_kind":"oof","segmentation_fold":fold_value,"segmentation_model_hash":model_hash,"probability_path":str(path),"threshold":threshold})
        final_epochs=max(1,int(round(float(np.median(best_epochs)))))
        final_checkpoint=model_root/phase/"full_train.pt"; final_audit_path=model_root/phase/"full_train_audit.json"
        final_signature=sha256_json({"segmentation":seg_cfg,"phase":phase,"scope":"full_train","samples":sorted(phase_frame.sample_uid.tolist()),"epochs":final_epochs})
        final_compatible=False
        if final_checkpoint.is_file() and final_audit_path.is_file():
            try:
                audit=json.loads(final_audit_path.read_text(encoding="utf-8"))
                final_compatible=audit.get("resume_signature")==final_signature and audit.get("checkpoint_sha256")==sha256_file(final_checkpoint)
            except Exception: final_compatible=False
        if not final_compatible:
            for item in (final_checkpoint,final_audit_path): quarantine(item,"resume_mismatch")
            audit=train_model(phase_frame,None,seg_cfg,final_checkpoint,int(seg_cfg["seed"])+9000,final_epochs)
            audit["fixed_epochs_from_fold_median"]=final_epochs
            audit["resume_signature"]=final_signature; audit["checkpoint_sha256"]=sha256_file(final_checkpoint)
            atomic_json(audit,final_audit_path)
        phase_summaries[phase]={"samples":len(phase_frame),"patients":unique_patients,"folds":requested,"selected_threshold":threshold,"threshold_metrics":best.to_dict(),"fold_audits":fold_audits,"final_epochs":final_epochs,"final_checkpoint":str(final_checkpoint),"final_checkpoint_sha256":sha256_file(final_checkpoint)}
    predictions=pd.DataFrame(all_predictions); atomic_csv(predictions,manifests/"segmentation_train_oof_predictions.csv"); atomic_csv(pd.DataFrame(all_folds),manifests/"segmentation_fold_assignments.csv")
    summary={"phases":phase_summaries,"thresholds":thresholds,"train_oof_predictions":len(predictions),"source_index_sha256":sha256_file(index_path),"max_samples":args.max_samples,"epochs_override":args.epochs}
    atomic_json(summary,reports/"segmentation_oof_training_summary.json")
    if args.output_root is None: write_marker(reports/".SEGMENTATION_OOF_SUCCESS","05_train_segmentation_oof",config,{"source_index_sha256":sha256_file(index_path)},summary)
    finish({"predictions":len(predictions),"phases":list(phase_summaries)}); print(json.dumps(summary,ensure_ascii=False,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from common import atomic_csv, atomic_json, atomic_text, configure_runtime, load_config, sha256_file, stage_logger, write_marker
from segmentation import component_from_probability, load_model, predict_frame, restore_model_probability, threshold_metrics


def atomic_probability(path: Path, probability: np.ndarray) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(f".{path.name}.{os.getpid()}.tmp.npz"); np.savez_compressed(tmp,probability=probability.astype(np.float16)); os.replace(tmp,path)


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),(mask>0).astype(np.uint8)*255): raise RuntimeError(path)


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--config"); parser.add_argument("--max-samples",type=int); args=parser.parse_args()
    config=load_config(args.config); configure_runtime(config); finish=stage_logger("06_infer_train_oof_valid_masks")
    manifests=Path(config["paths"]["manifests"]); outputs=Path(config["paths"]["outputs"]); reports=Path(config["paths"]["reports"])
    index=pd.read_csv(manifests/"segmentation_dataset_index.csv",dtype=str,keep_default_na=False)
    train_predictions=pd.read_csv(manifests/"segmentation_train_oof_predictions.csv",dtype=str,keep_default_na=False)
    training_summary=json.loads((reports/"segmentation_oof_training_summary.json").read_text(encoding="utf-8")); thresholds={key:float(value) for key,value in training_summary["thresholds"].items()}
    rows=[]
    for row in train_predictions.to_dict("records"):
        probability=np.load(row["probability_path"])["probability"].astype(np.float32); threshold=float(row["threshold"]); mask=component_from_probability(probability,threshold)
        mask_path=outputs/"segmentation_masks"/"train"/row["phase"]/f"{row['sample_uid']}.png"; save_mask(mask_path,mask)
        rows.append({**row,"mask_path":str(mask_path),"empty_prediction":int(mask.sum()==0),"probability_max":float(probability.max())})
    valid=index[index["split"]=="Valid"].copy()
    if args.max_samples:
        keep=valid.patient_id.drop_duplicates().head(args.max_samples).tolist(); valid=valid[valid.patient_id.isin(keep)]
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for phase in ("pre","post"):
        phase_frame=valid[valid["phase"]==phase].reset_index(drop=True)
        if phase_frame.empty: continue
        checkpoint=outputs/"segmentation_models"/phase/"full_train.pt"; model=load_model(checkpoint,device); probabilities=predict_frame(model,phase_frame,int(config["segmentation"]["batch_size"])); threshold=thresholds[phase]
        for row in phase_frame.to_dict("records"):
            original=restore_model_probability(row["sample_path"],probabilities[row["sample_uid"]]); probability_path=outputs/"segmentation_valid_probabilities"/phase/f"{row['sample_uid']}.npz"; atomic_probability(probability_path,original)
            mask=component_from_probability(original,threshold); mask_path=outputs/"segmentation_masks"/"valid"/phase/f"{row['sample_uid']}.png"; save_mask(mask_path,mask)
            rows.append({"sample_uid":row["sample_uid"],"phase_uid":row["phase_uid"],"patient_id":row["patient_id"],"split":"Valid","phase":phase,"series_uid":row["series_uid"],"prediction_kind":"full_train_frozen","segmentation_fold":0,"segmentation_model_hash":sha256_file(checkpoint),"probability_path":str(probability_path),"threshold":threshold,"mask_path":str(mask_path),"empty_prediction":int(mask.sum()==0),"probability_max":float(original.max())})
        del model; torch.cuda.empty_cache()
    output=pd.DataFrame(rows).sort_values(["split","phase","patient_id","series_uid"]); atomic_csv(output,manifests/"segmentation_prediction_index.csv")
    index_lookup=index.set_index("sample_uid",drop=False)
    metric_rows=[]
    for record in output.to_dict("records"):
        sample=index_lookup.loc[record["sample_uid"]]
        probability=np.load(record["probability_path"])["probability"].astype(np.float32)
        raw=np.load(sample["sample_path"],allow_pickle=False)
        gt=(restore_model_probability(sample["sample_path"],raw["mask"].astype(np.float32))>=0.5).astype(np.uint8)
        values=threshold_metrics(probability,gt,float(record["threshold"]),config["roi"])
        metric_rows.append({"sample_uid":record["sample_uid"],"patient_id":record["patient_id"],"split":record["split"],"phase":record["phase"],"annotation_grade":sample["annotation_grade"],"prediction_kind":record["prediction_kind"],**values})
    metrics_frame=pd.DataFrame(metric_rows)
    atomic_csv(metrics_frame,reports/"segmentation_metrics.csv")
    numeric=[c for c in metrics_frame.columns if c not in {"sample_uid","patient_id","split","phase","annotation_grade","prediction_kind"}]
    grouped=metrics_frame.groupby(["split","phase","annotation_grade"],dropna=False)[numeric].mean().reset_index()
    atomic_csv(grouped,reports/"segmentation_metrics_summary.csv")
    lines=["# Segmentation QC summary","",f"Samples: {len(metrics_frame)}",f"Patients: {metrics_frame.patient_id.nunique()}","","## Split / phase / grade metrics",""]
    for record in grouped.to_dict("records"):
        lines.append(f"- {record['split']} {record['phase']} grade {record['annotation_grade']}: Dice={record['dice']:.4f}, IoU={record['iou']:.4f}, sensitivity={record['sensitivity']:.4f}, on-target={record['on_target']:.4f}, padded coverage={record['coverage']:.4f}, empty={record['empty']:.4f}")
    atomic_text("\n".join(lines)+"\n",reports/"segmentation_qc_summary.md")
    gate_cfg=config["gates"]; phase_gate={}
    for phase,payload in training_summary["phases"].items():
        values=payload["threshold_metrics"]
        phase_gate[phase]={"on_target":float(values["on_target"]),"coverage":float(values["coverage"]),"empty":float(values["empty"]),"passed":float(values["on_target"])>=gate_cfg["minimum_oof_on_target"] and float(values["coverage"])>=gate_cfg["minimum_padded_box_coverage"] and float(values["empty"])<=gate_cfg["maximum_empty_rate"]}
    gate_passed=bool(phase_gate) and all(item["passed"] for item in phase_gate.values())
    summary={"rows":len(output),"by_split_phase":{"|".join(key):int(value) for key,value in output.groupby(["split","phase"]).size().to_dict().items()},"empty_predictions":int(pd.to_numeric(output["empty_prediction"]).sum()),"thresholds":thresholds,"train_oof_only":True,"valid_model_scope":"all_train","segmentation_gate":{"passed":gate_passed,"phases":phase_gate}}
    atomic_json(summary,reports/"segmentation_inference_summary.json"); write_marker(reports/".SEGMENTATION_COMPLETE","06_infer_train_oof_valid_masks",config,{"oof_index_sha256":sha256_file(manifests/"segmentation_train_oof_predictions.csv")},summary)
    if gate_passed:
        if (reports/".FAILED_SEGMENTATION_GATE").is_file(): (reports/".FAILED_SEGMENTATION_GATE").unlink()
        write_marker(reports/".SEGMENTATION_SUCCESS","06_infer_train_oof_valid_masks",config,{"oof_index_sha256":sha256_file(manifests/"segmentation_train_oof_predictions.csv")},summary)
    else:
        if (reports/".SEGMENTATION_SUCCESS").is_file(): (reports/".SEGMENTATION_SUCCESS").unlink()
        atomic_json(summary,reports/".FAILED_SEGMENTATION_GATE")
    finish(summary); print(json.dumps(summary,ensure_ascii=False,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())

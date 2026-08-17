#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from assets import apply_orientation, build_summary_channels, largest_component, lesion_and_context_masks, load_nifti_mask, read_frames
from common import atomic_csv, atomic_json, bool_value, configure_runtime, load_config, parse_pipe, sha256_file, stage_logger, write_marker
from roi import bbox_to_text, box_padding, probability_mass_component, save_bbox_overlay, save_temporal_montage
from segmentation import bbox, expanded_square_box


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),(mask>0).astype(np.uint8)*255): raise RuntimeError(path)


def branch_manifest(original_path: Path, roi: pd.DataFrame, branch: str, split: str, output: Path) -> dict:
    original=pd.read_csv(original_path,dtype=str,keep_default_na=False)
    duplicate_flags=roi["duplicate_excluded"].map(bool_value)
    duplicate_excluded=set(roi.loc[duplicate_flags,"phase_uid"])
    keep_rows=[]
    for row in original.to_dict("records"):
        keep_any=False
        for phase in ("pre","post"):
            phase_rows=roi[(roi.series_uid==row["series_uid"])&(roi.phase==phase)&(~roi.phase_uid.isin(duplicate_excluded))]
            usable=not phase_rows.empty
            if usable: keep_any=True
            else:
                row[f"can_run_{phase}"]="False"; row[f"{phase}_frame_paths"]=""; row[f"{phase}_frame_indices"]=""; row[f"{phase}_frame_list_hash"]=""
                for column in (f"n_{phase}_frames",f"n_{phase}_contiguous_pairs"):
                    if column in row: row[column]="0"
        if keep_any: keep_rows.append(row)
    frame=pd.DataFrame(keep_rows)
    if frame.empty: raise RuntimeError(f"No rows for {branch} {split}")
    atomic_csv(frame,output)
    return {"rows":len(frame),"patients":frame.patient_id.nunique(),"pre":sum(bool_value(v) for v in frame.can_run_pre),"post":sum(bool_value(v) for v in frame.can_run_post),"sha256":sha256_file(output)}


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--config"); parser.add_argument("--max-rows",type=int); args=parser.parse_args()
    config=load_config(args.config); configure_runtime(config); finish=stage_logger("07_build_roi_manifests")
    manifests=Path(config["paths"]["manifests"]); outputs=Path(config["paths"]["outputs"]); reports=Path(config["paths"]["reports"])
    aligned=pd.read_csv(manifests/"authoritative_roi_manifest_primary.csv",dtype=str,keep_default_na=False); predictions=pd.read_csv(manifests/"segmentation_prediction_index.csv",dtype=str,keep_default_na=False)
    merged=aligned.merge(predictions[["phase_uid","prediction_kind","segmentation_fold","segmentation_model_hash","probability_path","threshold","mask_path","empty_prediction","probability_max"]],on="phase_uid",how="inner",validate="one_to_one")
    if args.max_rows: merged=merged.head(args.max_rows).copy()
    rows=[]; roi_cfg=config["roi"]; mask_cfg=config["mask"]; qc_root=reports/"roi_qc"
    for record in merged.to_dict("records"):
        frames=read_frames(parse_pipe(record["frame_paths"])); probability=np.load(record["probability_path"])["probability"].astype(np.float32); threshold=float(record["threshold"])
        if probability.shape!=frames.shape[1:]: probability=cv2.resize(probability,(frames.shape[2],frames.shape[1]),interpolation=cv2.INTER_LINEAR)
        pred=probability_mass_component(probability,threshold); pred_box=bbox(pred); fallback="none"; center=None
        if pred_box is None:
            y,x=np.unravel_index(int(np.argmax(probability)),probability.shape); center=(float(x),float(y)); fallback="probability_argmax"
        pred_expanded=expanded_square_box(pred_box,frames.shape[1:],float(roi_cfg["padding_factor"]),float(roi_cfg["minimum_side_fraction"]),float(roi_cfg["maximum_side_fraction"]),center,allow_outside=True)
        pred_context=expanded_square_box(pred_box,frames.shape[1:],float(roi_cfg["context_padding_factor"]),float(roi_cfg["minimum_side_fraction"]),float(roi_cfg["maximum_side_fraction"]),center,allow_outside=True)
        raw_mask,_=load_nifti_mask(Path(record["segmentation_path"])); oriented=apply_orientation(raw_mask,record["orientation_transform"])
        if oriented.shape!=frames.shape[1:]: oriented=cv2.resize(oriented,(frames.shape[2],frames.shape[1]),interpolation=cv2.INTER_NEAREST)
        lesion,context,all_nonzero=lesion_and_context_masks(oriented,mask_cfg["lesion_labels"],mask_cfg["context_labels"])
        gt_box=bbox(lesion)
        if int(lesion.sum())<int(mask_cfg["minimum_pixels"]):
            alternative=((oriented!=0)&(~np.isin(oriented,mask_cfg["context_labels"]))).astype(np.uint8)
            lesion=largest_component(alternative)
            if int(lesion.sum())<int(mask_cfg["minimum_pixels"]): raise AssertionError(f"no_lesion_target:{record['phase_uid']}")
        gt_expanded=expanded_square_box(
            gt_box,frames.shape[1:],float(roi_cfg["padding_factor"]),
            float(roi_cfg["minimum_side_fraction"]),float(roi_cfg["maximum_side_fraction"]),
            allow_outside=True,
        )
        all_box=bbox(all_nonzero)
        all_expanded=expanded_square_box(
            all_box,frames.shape[1:],float(roi_cfg["padding_factor"]),
            float(roi_cfg["minimum_side_fraction"]),float(roi_cfg["maximum_side_fraction"]),
            allow_outside=True,
        )
        masks={"pred":pred,"gt":lesion,"all_nonzero":all_nonzero}; boxes={"pred":pred_expanded,"gt":gt_expanded,"all_nonzero":all_expanded}; original_boxes={"pred":pred_box,"gt":gt_box,"all_nonzero":all_box}
        summary=build_summary_channels(frames)["max_enhancement"]
        for branch in ("pred","gt","all_nonzero"):
            mask_path=outputs/"roi_masks"/branch/record["split"].casefold()/record["phase"]/f"{record['phase_uid']}.png"; save_mask(mask_path,masks[branch])
            box_value=boxes[branch]; x0,y0,x1,y1=box_value
            padding=box_padding(box_value,frames.shape[1:])
            rows.append({**record,"roi_branch":branch,"roi_mask_path":str(mask_path),"original_bbox":bbox_to_text(original_boxes[branch]) if original_boxes[branch] else "","expanded_bbox":bbox_to_text(box_value),"context_bbox":bbox_to_text(pred_context) if branch=="pred" else "","crop_padding_factor":float(roi_cfg["padding_factor"]),"crop_padding":bbox_to_text(padding),"roi_area_ratio":((x1-x0)*(y1-y0))/(frames.shape[1]*frames.shape[2]),"mask_area_ratio":float(masks[branch].sum()/masks[branch].size),"fallback_type":fallback if branch=="pred" else "none"})
            if branch in {"pred","gt"} and len(rows)<=200:
                save_bbox_overlay(summary,masks[branch],box_value,qc_root/branch/f"{record['phase_uid']}.jpg",f"{record['patient_id']} {record['phase']} {branch} {fallback}")
        if len(rows)<=200: save_temporal_montage(frames,pred_expanded,qc_root/"temporal"/f"{record['phase_uid']}.jpg")
    roi=pd.DataFrame(rows)
    roi["mask_content_sha256"]=roi["roi_mask_path"].map(lambda value:sha256_file(Path(value)))
    roi["duplicate_key"]=roi["patient_id"].astype(str)+"|"+roi["phase"].astype(str)+"|"+roi["frame_list_hash"].astype(str)+"|"+roi["mask_content_sha256"]
    roi["duplicate_rank"]=roi.groupby(["roi_branch","duplicate_key"]).cumcount(); roi["duplicate_excluded"]=roi["duplicate_rank"]>0
    atomic_csv(roi,manifests/"roi_manifest_all_branches.csv")
    branch_summaries={}
    for branch in ("pred","gt","all_nonzero"):
        branch_frame=roi[roi.roi_branch==branch].copy(); atomic_csv(branch_frame,manifests/f"roi_manifest_{branch}.csv")
        branch_summaries[branch]={}
        for split,source_manifest in (("Train",Path(config["train_manifest"])),("Valid",Path(config["valid_manifest"]))):
            subset=branch_frame[branch_frame.split==split].copy(); output=manifests/f"cave_manifest_{branch}_{split.casefold()}.csv"; branch_summaries[branch][split]=branch_manifest(source_manifest,subset,branch,split,output)
    fallback_frame=roi[(roi.roi_branch=="pred")&(roi.fallback_type!="none")].copy(); atomic_csv(fallback_frame,reports/"roi_fallbacks.csv")
    summary={"roi_rows":len(roi),"pred_phases":int((roi.roi_branch=="pred").sum()),"gt_phases":int((roi.roi_branch=="gt").sum()),"fallbacks":len(fallback_frame),"duplicates_excluded":int(roi.duplicate_excluded.sum()),"branch_manifests":branch_summaries}
    atomic_json(summary,reports/"roi_manifest_summary.json"); write_marker(reports/".ROI_SUCCESS","07_build_roi_manifests",config,{"aligned_sha256":sha256_file(manifests/"authoritative_roi_manifest_primary.csv"),"prediction_sha256":sha256_file(manifests/"segmentation_prediction_index.csv")},summary)
    finish(summary); print(json.dumps(summary,ensure_ascii=False,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())

#!/usr/bin/env python3
"""Frozen five-fold SegResNet inference for every annotated /2D phase."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset

from loader import (CHECKPOINT_ROOT, DEFAULT_OUTPUT_ROOT, V6_CONFIG, checkpoint_seen_patients,
                    outcome_phase_fold_map, prepare_image, read_all2d_manifest, sha256,
                    write_json, write_npz)

V6_CODE = Path("/root/autodl-tmp/aneurysm/code/api_png2d_spatial_backbones_v6_strict")
sys.path.insert(0, str(V6_CODE))
from common import load_config  # noqa: E402
from model_interface import build_model, global_pool, roi_pool  # noqa: E402


class ImageOnlyDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, cfg: dict): self.frame, self.cfg = frame.reset_index(drop=True), cfg
    def __len__(self): return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]
        x = prepare_image(row.image_path, int(self.cfg["data"]["input_size"]), float(self.cfg["data"]["percentile_low"]), float(self.cfg["data"]["percentile_high"]))
        return torch.from_numpy(x).unsqueeze(0), index


@torch.inference_mode()
def infer_fold(model, frame, cfg, device, batch_size, workers):
    ds = ImageOnlyDataset(frame, cfg)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type=="cuda", persistent_workers=workers>0)
    n = len(ds); glob = np.empty((n,256),np.float32); pred = np.empty((n,256),np.float32)
    for x, idx in dl:
        x=x.to(device, non_blocking=True)
        with autocast(enabled=device.type=="cuda"):
            fmap, logits = model.encode_and_decode(x)
            g = global_pool(fmap)
            p, _ = roi_pool(fmap, torch.sigmoid(logits), "bilinear")
        ii=np.asarray(idx); glob[ii]=g.float().cpu().numpy(); pred[ii]=p.float().cpu().numpy()
    return glob, pred


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output-root",type=Path,default=DEFAULT_OUTPUT_ROOT); ap.add_argument("--device",default="cuda:0"); ap.add_argument("--batch-size",type=int,default=4); ap.add_argument("--workers",type=int,default=4); ap.add_argument("--limit",type=int,default=0)
    args=ap.parse_args(); device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    cfg=load_config(V6_CONFIG); frame=read_all2d_manifest(); full=len(frame)==2233
    if args.limit: frame=frame.iloc[:args.limit].copy(); full=False
    outcome_folds=outcome_phase_fold_map(); seen, legal_hashes=checkpoint_seen_patients()
    n=len(frame); by_global=np.empty((n,5,256),np.float32); by_pred=np.empty((n,5,256),np.float32); checkpoint_hashes={}
    for fold in range(1,6):
        ck=CHECKPOINT_ROOT/f"fold_{fold}/model.pt"; checkpoint=torch.load(ck,map_location="cpu")
        model=build_model("segresnet",cfg,load_pretrained=False).to(device).eval(); model.load_state_dict(checkpoint["state_dict"],strict=True)
        g,p=infer_fold(model,frame,cfg,device,args.batch_size,args.workers); by_global[:,fold-1]=g;by_pred[:,fold-1]=p;checkpoint_hashes[str(fold)]=sha256(ck);del model
    spatial=np.concatenate([by_global,by_pred],axis=-1).astype(np.float32)
    unseen=np.asarray([[str(pid) not in seen[k] for k in range(1,6)] for pid in frame.patient_id],dtype=bool)
    known_fold=np.asarray([outcome_folds.get(str(key),0) for key in frame.segmentation_key],dtype=np.int64)
    recommended=np.where((known_fold>0)&unseen[np.arange(n),np.maximum(known_fold-1,0)],known_fold,0).astype(np.int64)
    strict=recommended>0
    frame=frame.copy();frame["all2d_phase_row_index"]=np.arange(n);frame["outcome_outer_fold"]=known_fold;frame["strict_oof_available"]=strict;frame["recommended_oof_source_fold"]=recommended
    for fold in range(1,6): frame[f"checkpoint_{fold}_unseen"]=unseen[:,fold-1]
    out=args.output_root/"segresnet";out.mkdir(parents=True,exist_ok=True)
    # Pandas ``to_numpy`` can yield object arrays. Public NPZ identifiers must
    # be native Unicode so downstream readers can retain ``allow_pickle=False``.
    u = lambda x: np.asarray(x, dtype=str)
    write_npz(out/"all2d_phase_features_by_fold.npz", segmentation_key=u(frame.segmentation_key), series_uid=u(frame.series_uid), patient_id=u(frame.patient_id), phase=u(frame.phase), checkpoint_unseen_by_fold=unseen, strict_oof_available=strict, recommended_oof_source_fold=recommended, global_by_fold=by_global, pred_roi_by_fold=by_pred, phase_spatial_by_fold=spatial, source_model_folds=np.arange(1,6,dtype=np.int64), feature_version=np.asarray("segresnet_v6_strict_soft_predroi_all2d_phase_by_fold"))
    public_cols=["all2d_phase_row_index","segmentation_key","series_uid","patient_id","phase","image_path","image_shape","mask_shape","mask_nonzero_pixels","outcome_outer_fold","strict_oof_available","recommended_oof_source_fold",*[f"checkpoint_{i}_unseen" for i in range(1,6)]]
    frame[public_cols].to_csv(out/"all2d_phase_manifest.csv",index=False,encoding="utf-8")
    groups=frame.groupby("patient_id",sort=True)
    complete=[]; incomplete=[]
    for pid,g in groups:
        phases=dict(zip(g.phase,g.all2d_phase_row_index))
        if set(phases)=={"pre","post"}:
            complete.append({"patient_id":pid,"pre_phase_row_index":int(phases['pre']),"post_phase_row_index":int(phases['post']),"pre_segmentation_key":str(g.loc[g.phase.eq('pre'),'segmentation_key'].iloc[0]),"post_segmentation_key":str(g.loc[g.phase.eq('post'),'segmentation_key'].iloc[0])})
        else: incomplete.append({"patient_id":pid,"available_phases":"|".join(sorted(phases)),"missing_phase":"post" if 'post' not in phases else "pre","reason":"all2d_inventory_single_phase; no 1024-D feature fabricated"})
    pairs=pd.DataFrame(complete); inc=pd.DataFrame(incomplete)
    if full and (len(pairs)!=1009 or len(inc)!=207): raise AssertionError(f"unexpected all2d prepost counts {len(pairs)}/{len(inc)}")
    if len(pairs):
        pre=pairs.pre_phase_row_index.to_numpy();post=pairs.post_phase_row_index.to_numpy()
        z=np.concatenate([by_global[pre],by_pred[pre],by_global[post],by_pred[post]],axis=-1).astype(np.float32)
        if not np.array_equal(z[:,:,:256],by_global[pre]) or not np.array_equal(z[:,:,256:512],by_pred[pre]) or not np.array_equal(z[:,:,512:768],by_global[post]) or not np.array_equal(z[:,:,768:],by_pred[post]): raise AssertionError("pair construction mismatch")
        write_npz(out/"all2d_prepost_features_by_fold.npz", patient_id=u(pairs.patient_id), pre_segmentation_key=u(pairs.pre_segmentation_key), post_segmentation_key=u(pairs.post_segmentation_key), pre_phase_row_index=pre.astype(np.int64), post_phase_row_index=post.astype(np.int64), z_2d_raw_by_fold=z, source_model_folds=np.arange(1,6,dtype=np.int64), feature_version=np.asarray("segresnet_v6_strict_soft_predroi_all2d_prepost_by_fold"))
    pairs.to_csv(out/"all2d_prepost_manifest.csv",index=False,encoding="utf-8"); inc.to_csv(out/"all2d_incomplete_prepost.csv",index=False,encoding="utf-8")
    audit={"status":"PASS" if full else "SMOKE_PASS_PARTIAL","phase_rows":n,"complete_prepost_rows":len(pairs),"incomplete_patients":len(inc),"phase_shape":[n,5,512],"prepost_shape":[len(pairs),5,1024],"dtype":"float32","all_finite":bool(np.isfinite(spatial).all()),"predroi":"sigmoid(logits) -> bilinear resize -> continuous weighted pool; GT mask never opened","latent_averaging_applied":False,"checkpoint_sha256":checkpoint_hashes,"legal_training_manifest_sha256":legal_hashes,"strict_oof_rows":int(strict.sum()),"full_inventory":full}
    write_json(out/"AUDIT.json",audit);write_json(out/"SUCCESS.json",{"status":"success" if full else "smoke_partial","phase_rows":n,"checkpoint_count":5,"frozen_inference_only":True})
    print(json.dumps(audit,ensure_ascii=False,indent=2))
if __name__=="__main__":main()

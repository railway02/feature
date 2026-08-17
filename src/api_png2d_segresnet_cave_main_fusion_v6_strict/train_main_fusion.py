#!/usr/bin/env python3
"""Strict five-fold nested selection plus fresh refit for 2D--CAVE fusion."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from loader import load_and_audit, sha256, write_json
from model import MainFusionModel

ROOT = Path("/root/autodl-tmp/aneurysm")
DEFAULT_SP = ROOT / "outputs/api_png2d_spatial_backbones_v6_strict/expanded_strict/featurebanks/segresnet"
DEFAULT_CAVE = ROOT / "outputs/api_png2d_gtmask_roi_cave_v2_complete_mapping_fullseq/adverse_prepost_series_task_v3"
DEFAULT_OUT = ROOT / "outputs/api_png2d_segresnet_cave_main_fusion_v6_strict"


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def metrics(y, p) -> dict[str, float]:
    return {"AUROC": float(roc_auc_score(y, p)), "AUPRC": float(average_precision_score(y, p)), "Brier": float(brier_score_loss(y, p))}


def patient_inner_split(y: np.ndarray, patients: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd
    tab = pd.DataFrame({"patient_id": patients.astype(str), "target": y.astype(int)}).groupby("patient_id", as_index=False).agg(target=("target", "max"))
    split = StratifiedShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    i_tr, i_va = next(split.split(tab["patient_id"], tab["target"]))
    p_tr, p_va = set(tab.iloc[i_tr]["patient_id"]), set(tab.iloc[i_va]["patient_id"])
    tr, va = np.flatnonzero(np.isin(patients, list(p_tr))), np.flatnonzero(np.isin(patients, list(p_va)))
    if set(patients[tr]) & set(patients[va]): raise AssertionError("inner patient leakage")
    if len(np.unique(y[va])) < 2: raise AssertionError("inner valid lacks a class")
    return tr, va


def loader(spatial, temporal, y, idx, batch: int, shuffle: bool, seed: int) -> DataLoader:
    gen = torch.Generator().manual_seed(seed)
    data = TensorDataset(torch.from_numpy(spatial[idx]).float(), torch.from_numpy(temporal[idx]).float(), torch.from_numpy(y[idx]).float().view(-1, 1))
    return DataLoader(data, batch_size=batch, shuffle=shuffle, generator=gen, drop_last=False)


def build(args) -> MainFusionModel:
    return MainFusionModel(dropout=args.dropout).to(args.device)


def train_epoch(model, dl, opt, scaler, device, pos_weight, amp) -> float:
    model.train(); losses=[]; pw=torch.tensor([pos_weight],device=device)
    for s, t, y in dl:
        s,t,y=s.to(device),t.to(device),y.to(device); opt.zero_grad(set_to_none=True)
        with autocast(enabled=amp): loss=F.binary_cross_entropy_with_logits(model(s,t)["main_logit"],y,pos_weight=pw)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def infer(model, spatial, temporal, idx, batch, device) -> dict[str, np.ndarray]:
    model.eval(); collect={k:[] for k in ("z_main","main_logit","main_prob","spatial_gate","temporal_gate")}
    fake=np.zeros(len(spatial),np.float32)
    for s,t,_ in loader(spatial,temporal,fake,idx,batch,False,0):
        out=model(s.to(device),t.to(device))
        for k in collect: collect[k].append(out[k].detach().cpu().numpy().astype(np.float32))
    return {k:np.concatenate(v) for k,v in collect.items()}


def search(args, spatial, temporal, y, inner_train, inner_valid, fold_dir, seed) -> tuple[int, float]:
    set_seed(seed); model=build(args); opt=AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay); amp=args.amp and args.device.type=="cuda"; scaler=GradScaler(enabled=amp)
    pw=max(1,(len(inner_train)-int(y[inner_train].sum())))/max(1,int(y[inner_train].sum()))
    dl=loader(spatial,temporal,y,inner_train,args.batch_size,True,seed); best_epoch=0; best_ap=-1.; bad=0; rows=[]
    for epoch in range(1,args.max_epochs+1):
        loss=train_epoch(model,dl,opt,scaler,args.device,pw,amp); pred=infer(model,spatial,temporal,inner_valid,args.batch_size,args.device)["main_prob"].ravel(); ap=float(average_precision_score(y[inner_valid],pred)); auc=float(roc_auc_score(y[inner_valid],pred))
        rows.append({"epoch":epoch,"train_loss":loss,"inner_valid_AUPRC":ap,"inner_valid_AUROC":auc})
        # ``min_epochs`` is a true lower bound for both selection and refit,
        # not merely a lower bound on when patience may stop the search.
        if epoch >= args.min_epochs:
            if ap>best_ap+1e-6: best_epoch,best_ap,bad=epoch,ap,0
            else: bad+=1
            if bad>=args.patience: break
    with (fold_dir/"epoch_search.csv").open("w",newline="") as f: csv.DictWriter(f,fieldnames=rows[0]).writeheader(); csv.DictWriter(f,fieldnames=rows[0]).writerows(rows)
    return best_epoch,best_ap


def refit(args, spatial, temporal, y, development, epochs, fold_dir, seed) -> MainFusionModel:
    if epochs < 1: raise AssertionError("selected epoch must be positive")
    set_seed(seed); model=build(args); opt=AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay); amp=args.amp and args.device.type=="cuda"; scaler=GradScaler(enabled=amp)
    pw=max(1,(len(development)-int(y[development].sum())))/max(1,int(y[development].sum())); dl=loader(spatial,temporal,y,development,args.batch_size,True,seed)
    for _ in range(epochs): train_epoch(model,dl,opt,scaler,args.device,pw,amp)
    torch.save({"state_dict":model.state_dict(),"model_config":model.config(),"selected_epoch":epochs,"protocol":"fresh_refit_after_inner_patient_level_AUPRC_selection"},fold_dir/"model.pt")
    return model


def smoke_model(args) -> dict:
    set_seed(args.seed); model=build(args).eval(); s=torch.randn(3,1024,device=args.device); t=torch.randn(3,10240,device=args.device)
    with torch.no_grad(): a=model(s,t)
    expected={"z_2d":(3,256),"z_time":(3,256),"spatial_gate":(3,256),"temporal_gate":(3,256),"z_2d_interacted":(3,256),"z_time_interacted":(3,256),"h_main":(3,1024),"z_main":(3,256),"main_logit":(3,1),"main_prob":(3,1)}
    for k,shape in expected.items():
        if tuple(a[k].shape)!=shape or a[k].dtype!=torch.float32 or not torch.isfinite(a[k]).all(): raise AssertionError(f"smoke failed: {k}")
    if not (torch.all((a["main_prob"]>=0)&(a["main_prob"]<=1)) and torch.all((a["spatial_gate"]>=0)&(a["spatial_gate"]<=1)) and torch.all((a["temporal_gate"]>=0)&(a["temporal_gate"]<=1))): raise AssertionError("probability/gate range")
    path=args.output_root/"main_fusion"/"smoke_reload.pt"; path.parent.mkdir(parents=True,exist_ok=True); torch.save({"state_dict":model.state_dict(),"model_config":model.config()},path)
    clone=MainFusionModel(**torch.load(path,map_location=args.device)["model_config"]).to(args.device).eval(); clone.load_state_dict(torch.load(path,map_location=args.device)["state_dict"])
    with torch.no_grad(): b=clone(s,t)
    same=all(torch.equal(a[k],b[k]) for k in expected)
    path.unlink()
    return {"status":"PASS","shapes":{k:list(v) for k,v in expected.items()},"float32_finite":True,"probability_and_gate_range":True,"checkpoint_reload_exact":same}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--spatial-root",type=Path,default=DEFAULT_SP); ap.add_argument("--cave-root",type=Path,default=DEFAULT_CAVE); ap.add_argument("--output-root",type=Path,default=DEFAULT_OUT); ap.add_argument("--device",default="cuda:0"); ap.add_argument("--seed",type=int,default=20260813); ap.add_argument("--batch-size",type=int,default=32); ap.add_argument("--learning-rate",type=float,default=1e-4); ap.add_argument("--weight-decay",type=float,default=1e-3); ap.add_argument("--dropout",type=float,default=.2); ap.add_argument("--max-epochs",type=int,default=160); ap.add_argument("--min-epochs",type=int,default=15); ap.add_argument("--patience",type=int,default=20); ap.add_argument("--inner-val-fraction",type=float,default=.18); ap.add_argument("--amp",action="store_true"); ap.add_argument("--smoke-only",action="store_true"); args=ap.parse_args(); args.device=torch.device(args.device)
    if args.device.type=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    args.output_root.mkdir(parents=True,exist_ok=True)
    smoke=smoke_model(args); write_json(args.output_root/"main_fusion"/"SMOKE_TEST.json",smoke)
    if args.smoke_only: print(json.dumps(smoke,indent=2)); return
    st,sv=args.spatial_root/"train_spatial_features.npz",args.spatial_root/"valid_spatial_features.npz"; ct,cv=args.cave_root/"train_features.npz",args.cave_root/"valid_features.npz"
    train,valid,audit,_,_=load_and_audit(st,sv,ct,cv); write_json(args.output_root/"alignment"/"alignment_audit.json",audit)
    y,folds,groups=train["target"],train["fold"],train["patient_id"]; oof={k:np.full((781,d),np.nan,np.float32) for k,d in (("z_main",256),("main_logit",1),("main_prob",1),("spatial_gate",256),("temporal_gate",256))}; valid_by={k:[] for k in oof}; rows=[]
    for fold in range(1,6):
        fd=args.output_root/"main_fusion"/f"fold_{fold}"; fd.mkdir(parents=True,exist_ok=True); dev=np.flatnonzero(folds!=fold); hold=np.flatnonzero(folds==fold)
        if set(groups[dev])&set(groups[hold]): raise AssertionError("outer patient leakage")
        spatial=train["spatial_by_fold"][:,fold-1,:]; spatial_valid=valid["spatial_by_fold"][:,fold-1,:]
        inner_rel_tr,inner_rel_va=patient_inner_split(y[dev],groups[dev],args.inner_val_fraction,args.seed+fold); inner_tr,inner_va=dev[inner_rel_tr],dev[inner_rel_va]
        epoch,apv=search(args,spatial,train["temporal"],y,inner_tr,inner_va,fd,args.seed+fold*100); model=refit(args,spatial,train["temporal"],y,dev,epoch,fd,args.seed+fold*10000)
        h=infer(model,spatial,train["temporal"],hold,args.batch_size,args.device); v=infer(model,spatial_valid,valid["temporal"],np.arange(207),args.batch_size,args.device)
        for k in oof: oof[k][hold]=h[k]; valid_by[k].append(v[k])
        rows.append({"fold":fold,"selected_epoch":epoch,"inner_valid_AUPRC":apv,**{f"holdout_{k}":v for k,v in metrics(y[hold],h["main_prob"].ravel()).items()},"development_rows":int(len(dev)),"holdout_rows":int(len(hold)),"valid_used_for_selection":False,"spatial_source":"pred_combined_by_fold[:, fold-1, :]"})
    if not all(np.isfinite(v).all() for v in oof.values()): raise AssertionError("incomplete OOF")
    out=args.output_root; np.savez_compressed(out/"train_oof_main_outputs.npz",series_uid=train["series_uid"],patient_id=train["patient_id"],outer_fold=folds,source_fusion_fold=folds,**oof)
    np.savez_compressed(out/"valid_main_outputs_by_fold.npz",series_uid=valid["series_uid"],patient_id=valid["patient_id"],source_fusion_folds=np.arange(1,6,dtype=np.int64),**{f"{k}_by_fold":np.stack(v,axis=1) for k,v in valid_by.items()})
    overall={"protocol":"strict_outer_fold + patient_level_inner_AUPRC_selection + fresh_refit","train_oof":metrics(y,oof["main_prob"].ravel()),"valid_probability_mean":metrics(valid["target"],np.mean(np.stack(valid_by["main_prob"],axis=1),axis=1).ravel()),"valid_used_for_selection":False,"latent_averaging_applied":False,"input_sha256":{str(p):sha256(p) for p in (st,sv,ct,cv)},"folds":rows}
    write_json(out/"main_fusion"/"metrics.json",overall); write_json(out/"main_fusion"/"SUCCESS.json",{"status":"success","train_oof_rows":781,"valid_rows":207,"five_fold_valid_representations":True,"latent_averaging_applied":False});
    with (out/"main_fusion"/"fold_metrics.csv").open("w",newline="") as f: csv.DictWriter(f,fieldnames=rows[0]).writeheader(); csv.DictWriter(f,fieldnames=rows[0]).writerows(rows)
    print(json.dumps(overall,ensure_ascii=False,indent=2))

if __name__=="__main__": main()

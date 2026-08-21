#!/usr/bin/env python3
"""Formal Local Reference core781/207 outcome experiments.

Reuses the historical OutcomeModel and MainFusionModel implementations.  All
epoch selection is Train-only; each experiment creates exactly one full-Train
final model and one Valid207 prediction vector.
"""
from __future__ import annotations

import argparse, csv, json, os, random, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

V5 = Path('/root/autodl-tmp/aneurysm/code/api_png2d_segresnet_cave_fusion_v5_series_mapped_reference_ready')
FUSION = Path('/root/autodl-tmp/aneurysm/code/api_png2d_segresnet_cave_main_fusion_v6_strict')
sys.path.insert(0, str(V5)); sys.path.insert(0, str(FUSION))
from fusion_models import OutcomeModel
from model import MainFusionModel

CFG = dict(batch_size=32, learning_rate=1e-4, weight_decay=1e-3,
           hidden_dim=256, dropout=.2, max_epochs=160, min_epochs=15,
           patience=20, inner_val_fraction=.18, scaler_epsilon=1e-6)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def atomic_json(value, path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name('.'+path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n'); os.replace(tmp,path)

def patient_inner(y, patient, indices, fraction, seed):
    tab=pd.DataFrame({'patient_id':patient[indices].astype(str),'target':y[indices]}).groupby('patient_id',as_index=False).agg(target=('target','max'))
    a,b=next(StratifiedShuffleSplit(n_splits=1,test_size=fraction,random_state=seed).split(tab.patient_id,tab.target))
    pa=set(tab.iloc[a].patient_id); pb=set(tab.iloc[b].patient_id)
    tr=indices[np.isin(patient[indices].astype(str),list(pa))]; va=indices[np.isin(patient[indices].astype(str),list(pb))]
    assert not(set(patient[tr])&set(patient[va])) and len(np.unique(y[va]))==2
    return tr,va

def fit_scale(x, idx):
    mean=x[idx].mean(0,dtype=np.float64).astype('float32'); std=x[idx].std(0,dtype=np.float64).astype('float32'); std[std<CFG['scaler_epsilon']]=1
    return mean,std

def scale(x, ms): return ((x-ms[0])/ms[1]).astype('float32')

def build(kind, spatial_dim, external_dim, device):
    if kind=='single': return OutcomeModel('spatial_only',spatial_dim,1,hidden_dim=256,dropout=.2).to(device)
    return MainFusionModel(spatial_dim=spatial_dim,temporal_dim=external_dim,hidden_dim=256,fusion_mid_dim=512,dropout=.2).to(device)

def forward(model,kind,x,e=None):
    return model(spatial=x)['logit'] if kind=='single' else model(x,e)['main_logit']

def loader(x,e,y,idx,shuffle,seed):
    arrays=[torch.from_numpy(x[idx]).float()]
    if e is not None: arrays.append(torch.from_numpy(e[idx]).float())
    arrays.append(torch.from_numpy(y[idx].astype('float32')).view(-1,1))
    return DataLoader(TensorDataset(*arrays),batch_size=32,shuffle=shuffle,generator=torch.Generator().manual_seed(seed))

def train_epoch(model,kind,x,e,y,idx,opt,scaler,device,seed):
    model.train(); pos=max(1,int(y[idx].sum())); pw=torch.tensor([(len(idx)-pos)/pos],device=device); losses=[]
    for batch in loader(x,e,y,idx,True,seed):
        xb=batch[0].to(device); eb=batch[1].to(device) if e is not None else None; yb=batch[-1].to(device); opt.zero_grad(set_to_none=True)
        with autocast(enabled=device.type=='cuda'): loss=F.binary_cross_entropy_with_logits(forward(model,kind,xb,eb),yb,pos_weight=pw)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))

@torch.no_grad()
def predict(model,kind,x,e,idx,device):
    model.eval(); vals=[]; dummy=np.zeros(len(x),dtype='float32')
    for batch in loader(x,e,dummy,idx,False,0):
        xb=batch[0].to(device); eb=batch[1].to(device) if e is not None else None
        vals.append(torch.sigmoid(forward(model,kind,xb,eb)).cpu().numpy().ravel())
    return np.concatenate(vals)

def search(kind,x,e,y,tr,va,seed,device,history_path):
    set_seed(seed); model=build(kind,x.shape[1],0 if e is None else e.shape[1],device); opt=AdamW(model.parameters(),lr=1e-4,weight_decay=1e-3); scaler=GradScaler(enabled=device.type=='cuda')
    rows=[]; best_epoch=0; best_ap=-1.; bad=0
    for epoch in range(1,161):
        loss=train_epoch(model,kind,x,e,y,tr,opt,scaler,device,seed+epoch); p=predict(model,kind,x,e,va,device); ap=float(average_precision_score(y[va],p)); auc=float(roc_auc_score(y[va],p)); rows.append({'epoch':epoch,'train_loss':loss,'inner_valid_AUPRC':ap,'inner_valid_AUROC':auc})
        if epoch>=15:
            if ap>best_ap+1e-6: best_epoch,best_ap,bad=epoch,ap,0
            else: bad+=1
            if bad>=20: break
    pd.DataFrame(rows).to_csv(history_path,index=False)
    del model
    return best_epoch,best_ap

def refit(kind,x,e,y,idx,epochs,seed,device):
    set_seed(seed); model=build(kind,x.shape[1],0 if e is None else e.shape[1],device); opt=AdamW(model.parameters(),lr=1e-4,weight_decay=1e-3); scaler=GradScaler(enabled=device.type=='cuda')
    for epoch in range(1,epochs+1): train_epoch(model,kind,x,e,y,idx,opt,scaler,device,seed+epoch)
    return model

def metrics(y,p,threshold=.5):
    q=(p>=threshold).astype(int); tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel()
    return {'ROC_AUC':float(roc_auc_score(y,p)),'AUPRC':float(average_precision_score(y,p)),
            'sensitivity':float(tp/max(1,tp+fn)),'specificity':float(tn/max(1,tn+fp)),'accuracy':float((q==y).mean())}

def run_one(name,kind,x,xv,e,ev,y,yv,fold,patient,uids,vuids,out,base_seed,device):
    d=out/name; d.mkdir(parents=True,exist_ok=True); oof=np.full(len(y),np.nan,'float32'); fold_rows=[]; selected=[]
    for k in range(1,6):
        dev=np.flatnonzero(fold!=k); hold=np.flatnonzero(fold==k); itr,iva=patient_inner(y,patient,dev,.18,base_seed+k)
        xs=fit_scale(x,itr) if kind=='single' else (np.zeros(x.shape[1],dtype='float32'),np.ones(x.shape[1],dtype='float32')); sx=scale(x,xs) if kind=='single' else x
        es=None if e is None else fit_scale(e,itr); se=None if e is None else scale(e,es)
        epoch,ap=search(kind,sx,se,y,itr,iva,base_seed+k*100,device,d/f'fold_{k}_epoch_search.csv')
        xr=fit_scale(x,dev) if kind=='single' else (np.zeros(x.shape[1],dtype='float32'),np.ones(x.shape[1],dtype='float32')); rx=scale(x,xr) if kind=='single' else x
        er=None if e is None else fit_scale(e,dev); re=None if e is None else scale(e,er)
        model=refit(kind,rx,re,y,dev,epoch,base_seed+k*10000,device); oof[hold]=predict(model,kind,rx,re,hold,device)
        fm=metrics(y[hold],oof[hold]); fold_rows.append({'fold':k,'selected_epoch':epoch,'best_inner_AUPRC':ap,'development_n':len(dev),'holdout_n':len(hold),**fm}); selected.append(epoch)
        torch.save({'state_dict':model.state_dict(),'kind':kind,'selected_epoch':epoch,'x_mean':xr[0],'x_std':xr[1],'e_mean':None if er is None else er[0],'e_std':None if er is None else er[1]},d/f'fold_{k}_model.pt'); del model
    assert np.isfinite(oof).all(); pd.DataFrame({'series_uid':uids,'target':y,'fold':fold,'probability':oof}).to_csv(d/'TRAIN_OOF_PREDICTIONS.csv',index=False); pd.DataFrame(fold_rows).to_csv(d/'TRAIN_FOLD_METRICS.csv',index=False)
    # Train-only final epoch selection, then discard and fresh-refit on all 781.
    allidx=np.arange(len(y)); itr,iva=patient_inner(y,patient,allidx,.18,base_seed+999); xs=fit_scale(x,itr) if kind=='single' else (np.zeros(x.shape[1],dtype='float32'),np.ones(x.shape[1],dtype='float32')); sx=scale(x,xs) if kind=='single' else x; es=None if e is None else fit_scale(e,itr); se=None if e is None else scale(e,es)
    final_epoch,final_ap=search(kind,sx,se,y,itr,iva,base_seed+99900,device,d/'FINAL_TRAIN_ONLY_EPOCH_SEARCH.csv')
    xf=fit_scale(x,allidx) if kind=='single' else (np.zeros(x.shape[1],dtype='float32'),np.ones(x.shape[1],dtype='float32')); fx=scale(x,xf) if kind=='single' else x; fvx=scale(xv,xf) if kind=='single' else xv; ef=None if e is None else fit_scale(e,allidx); fe=None if e is None else scale(e,ef); fve=None if e is None else scale(ev,ef)
    model=refit(kind,fx,fe,y,allidx,final_epoch,base_seed+999000,device); vp=predict(model,kind,fvx,fve,np.arange(len(yv)),device)
    torch.save({'state_dict':model.state_dict(),'kind':kind,'selected_epoch':final_epoch,'x_mean':xf[0],'x_std':xf[1],'e_mean':None if ef is None else ef[0],'e_std':None if ef is None else ef[1],'valid_inference_count':1},d/'FINAL_MODEL.pt')
    pd.DataFrame({'series_uid':vuids,'target':yv,'probability':vp}).to_csv(d/'FINAL_VALID_PREDICTIONS.csv',index=False)
    report={'model':name,'architecture':'OutcomeModel(spatial_only)' if kind=='single' else 'historical MainFusionModel gated','config':CFG,'base_seed':base_seed,'folds':fold_rows,'train_oof':metrics(y,oof),'final_selected_epoch':final_epoch,'final_inner_AUPRC':final_ap,'final_valid207':metrics(yv,vp),'valid_inference_count':1,'outcome_fold_models_applied_to_valid':False,'formal_main_prob_by_fold_created':False}
    atomic_json(report,d/'FINAL_VALID_METRICS.json'); del model
    return report

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--only',nargs='*'); a=ap.parse_args(); device=torch.device(a.device)
    tr=np.load(a.out/'feature_master/core781_train.npz,.npz',allow_pickle=True); va=np.load(a.out/'feature_master/core207_valid.npz',allow_pickle=True)
    z=tr['z2d'].astype('float32'); zv=va['z2d'].astype('float32'); rl=tr['reg_linear'].astype('float32'); rlv=va['reg_linear'].astype('float32'); rf=np.c_[rl,tr['reg_nonlinear']].astype('float32'); rfv=np.c_[rlv,va['reg_nonlinear']].astype('float32')
    y=tr['target'].astype(int); yv=va['target'].astype(int); fold=tr['fold'].astype(int); patient=tr['patient_id'].astype(str); u=tr['series_uid'].astype(str); vu=va['series_uid'].astype(str)
    specs=[('3167_formal','single',z,zv,None,None,20260811),('reg_only_linear_formal','single',rl,rlv,None,None,20260811),('reg_only_full_formal','single',rf,rfv,None,None,20260811),('3168_formal_gated','fusion',z,zv,rl,rlv,20260813),('3169_formal_gated','fusion',z,zv,rf,rfv,20260813)]
    if a.only: specs=[s for s in specs if s[0] in set(a.only)]
    reports=[]
    for spec in specs: reports.append(run_one(*spec,y,yv,fold,patient,u,vu,a.out,spec[-1],device) if False else run_one(spec[0],spec[1],spec[2],spec[3],spec[4],spec[5],y,yv,fold,patient,u,vu,a.out,spec[6],device))
    rows=[]
    for name in ['3167_formal','reg_only_linear_formal','reg_only_full_formal','3168_formal_gated','3169_formal_gated']:
        p=a.out/name/'FINAL_VALID_METRICS.json'
        if p.is_file():
            r=json.loads(p.read_text()); rows.append({'Model':r['model'],'Train OOF AUC':r['train_oof']['ROC_AUC'],'Train OOF AUPRC':r['train_oof']['AUPRC'],'Final Valid207 AUC':r['final_valid207']['ROC_AUC'],'Final Valid207 AUPRC':r['final_valid207']['AUPRC'],'Final epoch':r['final_selected_epoch']})
    pd.DataFrame(rows).to_csv(a.out/'reports/FORMAL_CORE781_207_COMPARISON.csv',index=False); atomic_json({'models':rows},a.out/'reports/FORMAL_CORE781_207_COMPARISON.json')

if __name__=='__main__': main()

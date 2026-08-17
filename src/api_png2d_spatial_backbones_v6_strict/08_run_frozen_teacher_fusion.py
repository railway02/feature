#!/usr/bin/env python3
"""Strict outcome cross-fit with the frozen v5 fusion recipe.

Every outer fold selects its epoch on a patient-level inner split of the
development patients, then constructs a freshly initialized refit model on
all development patients.  Neither outer holdout nor independent Valid is
used for selection.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from common import atomic_csv, atomic_json, atomic_torch_save, canonical_hash, load_config, set_seed

def npz(path):
    with np.load(path, allow_pickle=False) as z: return {k: np.asarray(z[k]) for k in z.files}

def metric(y, p):
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    return {'auroc': float(roc_auc_score(y,p)) if len(np.unique(y)) > 1 else float('nan'), 'auprc': float(average_precision_score(y,p)) if len(np.unique(y)) > 1 else float('nan'), 'brier': float(brier_score_loss(y,p))}

def patient_inner_split(y, patient_ids, fraction, seed):
    patient = pd.DataFrame({'patient_id': np.asarray(patient_ids).astype(str), 'y': np.asarray(y).astype(int)}).groupby('patient_id', as_index=False).agg(y=('y','max'))
    tr, va = next(StratifiedShuffleSplit(n_splits=1, test_size=float(fraction), random_state=int(seed)).split(patient['patient_id'], patient['y']))
    tr_patient, va_patient = set(patient.iloc[tr].patient_id), set(patient.iloc[va].patient_id)
    if tr_patient & va_patient: raise RuntimeError('patient leakage in fusion inner split')
    ids = np.asarray(patient_ids).astype(str)
    return np.flatnonzero(np.isin(ids, list(tr_patient))), np.flatnonzero(np.isin(ids, list(va_patient)))

def loader(spatial, temporal, y, index, batch, shuffle, seed):
    index = np.asarray(index, dtype=int)
    ds = TensorDataset(torch.from_numpy(spatial[index]).float(), torch.from_numpy(temporal[index]).float(), torch.from_numpy(y[index].astype(np.float32)).view(-1,1))
    return DataLoader(ds, batch_size=int(batch), shuffle=bool(shuffle), generator=torch.Generator().manual_seed(int(seed)), drop_last=False)

def train_epoch(model, dl, optimizer, scaler, device, pos_weight, amp):
    model.train(); losses=[]; weight=torch.tensor([pos_weight], device=device)
    for s,t,y in dl:
        s,t,y=s.to(device),t.to(device),y.to(device); optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp): loss=F.binary_cross_entropy_with_logits(model(s,t)['logit'],y,pos_weight=weight)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))

@torch.no_grad()
def predict(model, spatial, temporal, y, index, batch, device):
    model.eval(); values=[]
    for s,t,_ in loader(spatial,temporal,y,index,batch,False,0): values.append(torch.sigmoid(model(s.to(device),t.to(device))['logit']).cpu().numpy().ravel())
    return np.concatenate(values)

def make_model(OutcomeModel, spatial, temporal, fusion, device):
    return OutcomeModel(mode='gated_interaction', spatial_dim=spatial.shape[1], temporal_dim=temporal.shape[1], hidden_dim=int(fusion['hidden_dim']), fusion_mid_dim=int(fusion['fusion_mid_dim']), dropout=float(fusion['dropout'])).to(device)

def select_epoch(OutcomeModel, spatial, temporal, y, inner_train, inner_valid, fusion, device, seed, fold_dir):
    set_seed(seed); model=make_model(OutcomeModel,spatial,temporal,fusion,device); opt=AdamW(model.parameters(),lr=float(fusion['learning_rate']),weight_decay=float(fusion['weight_decay'])); amp=bool(fusion['amp'] and device.type=='cuda'); scaler=GradScaler(enabled=amp)
    pos=max(1,int(y[inner_train].sum())); weight=max(1,len(inner_train)-pos)/pos; best_epoch=0; best_ap=-1.; bad=0; history=[]
    dl=loader(spatial,temporal,y,inner_train,fusion['batch_size'],True,seed)
    for epoch in range(1,int(fusion['max_epochs'])+1):
        loss=train_epoch(model,dl,opt,scaler,device,weight,amp); probability=predict(model,spatial,temporal,y,inner_valid,fusion['batch_size'],device); scores=metric(y[inner_valid],probability)
        history.append({'epoch':epoch,'train_loss':loss,'valid_AUPRC':scores['auprc'],'valid_AUROC':scores['auroc']}); print(json.dumps({'stage':'inner_epoch_selection','epoch':epoch,**history[-1]}),flush=True)
        if scores['auprc'] > best_ap + 1e-6: best_epoch,best_ap,bad=epoch,scores['auprc'],0
        else: bad += 1
        if epoch >= int(fusion['min_epochs']) and bad >= int(fusion['patience']): break
    atomic_csv(pd.DataFrame(history),fold_dir/'epoch_search.csv'); return best_epoch,best_ap

def refit(OutcomeModel, spatial, temporal, y, development, epochs, fusion, device, seed, fold_dir):
    set_seed(seed); model=make_model(OutcomeModel,spatial,temporal,fusion,device); opt=AdamW(model.parameters(),lr=float(fusion['learning_rate']),weight_decay=float(fusion['weight_decay'])); amp=bool(fusion['amp'] and device.type=='cuda'); scaler=GradScaler(enabled=amp)
    pos=max(1,int(y[development].sum())); weight=max(1,len(development)-pos)/pos; dl=loader(spatial,temporal,y,development,fusion['batch_size'],True,seed); history=[]
    for epoch in range(1,epochs+1): history.append({'epoch':epoch,'train_loss':train_epoch(model,dl,opt,scaler,device,weight,amp)})
    atomic_csv(pd.DataFrame(history),fold_dir/'fresh_refit_history.csv'); return model

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--family',choices=['segresnet','deeplabv3plus_resnet50_imagenet'],required=True); p.add_argument('--device',default='cuda:0'); a=p.parse_args(); cfg=load_config(a.config); device=torch.device(a.device)
    v5_cfg=json.loads(Path(cfg['sources']['v5_config']).read_text(encoding='utf-8')); fusion=v5_cfg['fusion']
    sys.path.insert(0,cfg['sources']['v5_code_root']); from fusion_models import OutcomeModel
    root=Path(cfg['output_root'])/'expanded_strict'/'featurebanks'/a.family; sptr=npz(root/'train_spatial_features.npz'); spva=npz(root/'valid_spatial_features.npz'); temtr=npz(cfg['sources']['temporal_train_npz']); temva=npz(cfg['sources']['temporal_valid_npz'])
    if not np.array_equal(sptr['series_uid'].astype(str),temtr['series_uid'].astype(str)) or not np.array_equal(spva['series_uid'].astype(str),temva['series_uid'].astype(str)): raise RuntimeError('spatial/CAVE UID order mismatch')
    x,xv,z,zv=sptr['pred_combined_by_fold'],spva['pred_combined_by_fold'],temtr['deep'],temva['deep']; y,folds=sptr['target'].astype(int),sptr['outer_fold'].astype(int)
    out=Path(cfg['output_root'])/'expanded_strict'/'fusion'/a.family; rep=Path(cfg['report_root'])/'expanded_strict_fusion'/a.family; out.mkdir(parents=True,exist_ok=True); rep.mkdir(parents=True,exist_ok=True); oof=np.full(len(y),np.nan,np.float32); valid_by=[]; fold_rows=[]
    for k in range(1,6):
        fold_dir=out/f'fold_{k}'; fold_dir.mkdir(parents=True,exist_ok=True); development=np.flatnonzero(folds!=k); holdout=np.flatnonzero(folds==k)
        if set(sptr['patient_id'][development].astype(str)) & set(sptr['patient_id'][holdout].astype(str)): raise RuntimeError(f'fold {k}: outcome patient leakage')
        inner_tr_rel,inner_va_rel=patient_inner_split(y[development],sptr['patient_id'][development],fusion['inner_val_fraction'],int(fusion['seed'])+k); inner_tr,inner_va=development[inner_tr_rel],development[inner_va_rel]
        best_epoch,best_ap=select_epoch(OutcomeModel,x[:,k-1],z,y,inner_tr,inner_va,fusion,device,int(fusion['seed'])+k*100,fold_dir)
        model=refit(OutcomeModel,x[:,k-1],z,y,development,best_epoch,fusion,device,int(fusion['seed'])+k*10000,fold_dir)
        hold_probability=predict(model,x[:,k-1],z,y,holdout,fusion['batch_size'],device); valid_probability=predict(model,xv[:,k-1],zv,spva['target'].astype(int),np.arange(len(xv)),fusion['batch_size'],device); oof[holdout]=hold_probability; valid_by.append(valid_probability)
        atomic_torch_save({'state_dict':model.state_dict(),'outer_fold':k,'selected_epoch':best_epoch,'teacher_architecture':'v5.fusion_models.OutcomeModel gated_interaction','fusion_config':fusion,'fresh_refit_on_all_outer_development':True},fold_dir/'model.pt')
        fold_rows.append({'fold':k,'selected_epoch':best_epoch,'best_inner_AUPRC':best_ap,**{f'OOF_{name}':value for name,value in metric(y[holdout],hold_probability).items()}}); del model
    if not np.isfinite(oof).all(): raise RuntimeError('incomplete OOF predictions')
    valid=np.mean(np.stack(valid_by),axis=0); atomic_csv(pd.DataFrame({'series_uid':sptr['series_uid'],'patient_id':sptr['patient_id'],'target':y,'outer_fold':folds,'probability':oof}),rep/'train_oof_predictions.csv'); atomic_csv(pd.DataFrame({'series_uid':spva['series_uid'],'patient_id':spva['patient_id'],'target':spva['target'],'probability':valid}),rep/'valid_predictions.csv'); atomic_csv(pd.DataFrame(fold_rows),rep/'fold_metrics.csv')
    atomic_json({'status':'success','family':a.family,'mode':'gated_interaction','teacher_architecture_unmodified':True,'fusion_config_source':cfg['sources']['v5_config'],'fusion_config_sha256':canonical_hash(fusion),'epoch_selection':'v5 patient-level stratified inner split, AUPRC, early stopping','fresh_refit':True,'valid_used_for_model_selection':False,'train_oof':metric(y,oof),'valid':metric(spva['target'].astype(int),valid),'valid_prediction_aggregation':'mean of five fusion probabilities, no latent spatial averaging'},rep/'SUCCESS.json')
if __name__=='__main__': main()

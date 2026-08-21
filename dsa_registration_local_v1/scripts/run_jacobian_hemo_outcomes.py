#!/usr/bin/env python3
"""Locked Train781/Valid207 y_abs outcome protocol for JAC42 and HEMO36.

This runner only consumes frozen feature banks.  It never imports or executes
registration, temporal motion correction, CAVE, or SegResNet inference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, roc_auc_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
MODEL_ROOT = Path('/root/autodl-tmp/aneurysm/code/api_png2d_segresnet_cave_main_fusion_v6_strict')
sys.path.insert(0, str(MODEL_ROOT))
from model import MainFusionModel  # noqa: E402
from dsa_local_reg.common import atomic_json, sha256_file  # noqa: E402
from dsa_local_reg.hemodynamics_v1 import compact36_columns  # noqa: E402
from dsa_local_reg.jacobian_derived import existing42_columns  # noqa: E402

CORE_ROOT = Path('/root/autodl-tmp/aneurysm/outputs/local_reference_core781_207_20260820T004500Z')
TECH_ROOT = PROJECT / 'outputs/local_reference_jacobian_hemo_20260820T110151Z'
MOTION_QC = PROJECT / 'outputs/local_reference_temporal_nonpeak_qc_20260820T125221Z/selected_temporal_nonpeak_qc.csv'
OUT_ROOT = Path('/root/autodl-tmp/aneurysm/outputs')

JAC = existing42_columns()
HEMO = compact36_columns()
VALIDITY = ['pre_hemo_valid', 'post_hemo_valid', 'hemo_valid', 'temporal_motion_invalid']


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pred = (p >= .5).astype(int)
    return {
        'ROC_AUC': float(roc_auc_score(y, p)), 'AUPRC': float(average_precision_score(y, p)),
        'Brier': float(brier_score_loss(y, p)), 'sensitivity': float(recall_score(y, pred, zero_division=0)),
        'specificity': float(recall_score(1-y, 1-pred, zero_division=0)), 'accuracy': float(accuracy_score(y, pred)),
    }


def patient_inner_split(y: np.ndarray, patients: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    table = pd.DataFrame({'patient_id': patients.astype(str), 'target': y.astype(int)}).groupby('patient_id', as_index=False).agg(target=('target', 'max'))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    a, b = next(splitter.split(table.patient_id, table.target))
    pa, pb = set(table.iloc[a].patient_id), set(table.iloc[b].patient_id)
    ia, ib = np.flatnonzero(np.isin(patients, list(pa))), np.flatnonzero(np.isin(patients, list(pb)))
    if set(patients[ia]) & set(patients[ib]) or len(np.unique(y[ib])) != 2: raise AssertionError('invalid patient-level inner split')
    return ia, ib


def fit_preprocessor(raw: np.ndarray, idx: np.ndarray) -> dict[str, np.ndarray]:
    source = raw[idx]
    all_nan = np.all(~np.isfinite(source), axis=0)
    if all_nan.any(): raise RuntimeError(f'external feature all NaN in training subset: {np.flatnonzero(all_nan).tolist()}')
    median = np.nanmedian(source, axis=0).astype(np.float32)
    filled = np.where(np.isfinite(source), source, median[None]).astype(np.float32)
    mean = filled.mean(axis=0).astype(np.float32); std = filled.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    return {'median': median, 'mean': mean, 'std': std}


def transform(raw: np.ndarray, prep: dict[str, np.ndarray]) -> np.ndarray:
    filled = np.where(np.isfinite(raw), raw, prep['median'][None]).astype(np.float32)
    out = (filled - prep['mean'][None]) / prep['std'][None]
    if not np.isfinite(out).all(): raise AssertionError('nonfinite external after Train-only preprocessing')
    return out.astype(np.float32)


def make_loader(z2d: np.ndarray, ext: np.ndarray, y: np.ndarray, idx: np.ndarray, batch: int, shuffle: bool, seed: int) -> DataLoader:
    gen = torch.Generator().manual_seed(seed)
    data = TensorDataset(torch.from_numpy(z2d[idx]).float(), torch.from_numpy(ext[idx]).float(), torch.from_numpy(y[idx]).float().reshape(-1, 1))
    return DataLoader(data, batch_size=batch, shuffle=shuffle, generator=gen, drop_last=False)


def build(device: torch.device, external_dim: int) -> MainFusionModel:
    return MainFusionModel(spatial_dim=1024, temporal_dim=external_dim, hidden_dim=256, fusion_mid_dim=512, dropout=.2).to(device)


def train_epoch(model: MainFusionModel, loader: DataLoader, opt: AdamW, scaler: GradScaler, device: torch.device, pos_weight: float, amp: bool) -> float:
    model.train(); losses=[]; pw=torch.tensor([pos_weight], device=device)
    for z, x, y in loader:
        z,x,y=z.to(device),x.to(device),y.to(device); opt.zero_grad(set_to_none=True)
        with autocast(enabled=amp): loss=F.binary_cross_entropy_with_logits(model(z,x)['main_logit'],y,pos_weight=pw)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def predict(model: MainFusionModel, z2d: np.ndarray, ext: np.ndarray, idx: np.ndarray, batch: int, device: torch.device) -> np.ndarray:
    model.eval(); parts=[]; fake=np.zeros(len(z2d),np.float32)
    for z,x,_ in make_loader(z2d,ext,fake,idx,batch,False,0): parts.append(model(z.to(device),x.to(device))['main_prob'].detach().cpu().numpy().ravel())
    return np.concatenate(parts).astype(np.float32)


def search_epoch(z2d, ext, y, train_idx, valid_idx, cfg, directory: Path, seed: int) -> tuple[int,float]:
    seed_all(seed); model=build(cfg['device'], ext.shape[1]); opt=AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['wd']); amp=cfg['amp']; scaler=GradScaler(enabled=amp)
    pw=float((len(train_idx)-int(y[train_idx].sum()))/max(1,int(y[train_idx].sum())))
    loader=make_loader(z2d,ext,y,train_idx,cfg['batch'],True,seed); rows=[]; best_epoch=0; best_ap=-np.inf; bad=0
    for epoch in range(1,cfg['max_epochs']+1):
        loss=train_epoch(model,loader,opt,scaler,cfg['device'],pw,amp); prob=predict(model,z2d,ext,valid_idx,cfg['batch'],cfg['device']); ap=float(average_precision_score(y[valid_idx],prob)); auc=float(roc_auc_score(y[valid_idx],prob))
        rows.append({'epoch':epoch,'train_loss':loss,'inner_valid_AUPRC':ap,'inner_valid_AUROC':auc})
        if epoch>=cfg['min_epochs']:
            if ap>best_ap+1e-6: best_epoch,best_ap,bad=epoch,ap,0
            else: bad+=1
            if bad>=cfg['patience']: break
    pd.DataFrame(rows).to_csv(directory,index=False)
    return int(best_epoch),float(best_ap)


def fresh_refit(z2d, ext, y, train_idx, epochs: int, cfg, seed: int, model_path: Path) -> MainFusionModel:
    seed_all(seed); model=build(cfg['device'], ext.shape[1]); opt=AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['wd']); scaler=GradScaler(enabled=cfg['amp'])
    pw=float((len(train_idx)-int(y[train_idx].sum()))/max(1,int(y[train_idx].sum())))
    loader=make_loader(z2d,ext,y,train_idx,cfg['batch'],True,seed)
    for _ in range(epochs): train_epoch(model,loader,opt,scaler,cfg['device'],pw,cfg['amp'])
    torch.save({'state_dict':model.state_dict(),'model_config':model.config(),'selected_epoch':epochs,'protocol':'fresh_refit_after_train_only_inner_AUPRC_selection'},model_path)
    return model


def core_npz() -> tuple[dict[str,np.ndarray],dict[str,np.ndarray]]:
    root=CORE_ROOT/'feature_master'
    # Historical strict feature NPZ stores frozen identity strings as object arrays.
    # It is a local, SHA-locked input; loading them is required for identity joins.
    train=np.load(root/'core781_train.npz,.npz',allow_pickle=True); valid=np.load(root/'core207_valid.npz',allow_pickle=True)
    return ({k:train[k] for k in train.files},{k:valid[k] for k in valid.files})


def load_bank(path: Path, columns: list[str], core: dict[str,np.ndarray], split: str) -> pd.DataFrame:
    bank=pd.read_csv(path,dtype={'series_uid':str,'patient_id':str})
    required={'series_uid','patient_id'}|set(columns)
    if missing:=sorted(required-set(bank.columns)): raise KeyError(f'{path} missing {missing}')
    if bank.duplicated(['series_uid','patient_id']).any(): raise AssertionError(f'{path}: duplicate composite identity')
    left=pd.DataFrame({'series_uid':core['series_uid'].astype(str),'patient_id':core['patient_id'].astype(str)})
    merged=left.merge(bank[['series_uid','patient_id']+columns],on=['series_uid','patient_id'],how='left',validate='one_to_one',indicator=True)
    if (merged['_merge']!='both').any(): raise AssertionError(f'{split}: bank identity join incomplete')
    return merged.drop(columns='_merge')


def build_master(run: Path) -> tuple[dict[str,np.ndarray],dict[str,np.ndarray],dict[str,Any]]:
    train,valid=core_npz()
    if train['z2d'].shape!=(781,1024) or valid['z2d'].shape!=(207,1024): raise AssertionError('frozen z2D shapes changed')
    if len(set(train['patient_id'].astype(str))&set(valid['patient_id'].astype(str))): raise AssertionError('Train/Valid patient overlap')
    train_j=load_bank(TECH_ROOT/'featurebanks/train800_jacobian_existing42.csv',JAC,train,'Train')
    valid_j=load_bank(TECH_ROOT/'featurebanks/valid211_jacobian_existing42.csv',JAC,valid,'Valid')
    hemo_cols=['pre_hemo_valid','post_hemo_valid','hemo_valid','hemo_invalid_reasons']+HEMO
    train_h=load_bank(TECH_ROOT/'featurebanks/train800_hemo_compact36.csv',hemo_cols,train,'Train')
    valid_h=load_bank(TECH_ROOT/'featurebanks/valid211_hemo_compact36.csv',hemo_cols,valid,'Valid')
    motion=pd.read_csv(MOTION_QC,dtype=str); motion=set(motion.loc[motion.group=='largest_rotation','series_uid'])
    if len(motion)!=10: raise AssertionError('fixed extreme rotation list must contain 10 UIDs')
    def one(core, jac, hemo, split):
        base=pd.DataFrame({'series_uid':core['series_uid'].astype(str),'patient_id':core['patient_id'].astype(str),'split':split,'target_y_abs':core['target'].astype(int)})
        if split=='Train': base['historical_outcome_fold']=core['fold'].astype(int)
        base=base.merge(jac,on=['series_uid','patient_id'],how='left',validate='one_to_one').merge(hemo,on=['series_uid','patient_id'],how='left',validate='one_to_one')
        base[['pre_hemo_valid','post_hemo_valid','hemo_valid']] = base[['pre_hemo_valid','post_hemo_valid','hemo_valid']].astype(np.int8)
        hit=base.series_uid.isin(motion); base['temporal_motion_invalid']=hit.astype(np.int8)
        base.loc[hit,HEMO]=np.nan; base.loc[hit,['pre_hemo_valid','post_hemo_valid','hemo_valid']]=0
        base.loc[hit,'hemo_invalid_reasons']=base.loc[hit,'hemo_invalid_reasons'].fillna('').astype(str).map(lambda x:(x+';' if x else '')+'temporal_transform_untrusted_extreme_rotation')
        return base
    a,b=one(train,train_j,train_h,'Train'),one(valid,valid_j,valid_h,'Valid')
    for name,frame,n in [('Train',a,781),('Valid',b,207)]:
        if len(frame)!=n or frame.duplicated(['series_uid','patient_id']).any() or not np.isfinite(frame[JAC].to_numpy(float)).all(): raise AssertionError(f'{name} master contract')
        if np.isinf(frame[HEMO].to_numpy(float)).any(): raise AssertionError(f'{name} HEMO inf')
    if int(a.temporal_motion_invalid.sum())!=8 or int(b.temporal_motion_invalid.sum())!=2: raise AssertionError('motion invalid split hit mismatch')
    for frame,name in [(a,'CORE781_TRAIN_JAC_HEMO_MASTER.csv'),(b,'CORE207_VALID_JAC_HEMO_MASTER.csv')]: frame.to_csv(run/'feature_master'/name,index=False)
    ext_a=np.column_stack([a[JAC+HEMO].to_numpy(np.float32),a[VALIDITY].to_numpy(np.float32)]); ext_b=np.column_stack([b[JAC+HEMO].to_numpy(np.float32),b[VALIDITY].to_numpy(np.float32)])
    np.savez_compressed(run/'feature_master/core781_train_outcome.npz',series_uid=train['series_uid'],patient_id=train['patient_id'],target=train['target'],fold=train['fold'],z2d=train['z2d'].astype(np.float32),external=ext_a)
    np.savez_compressed(run/'feature_master/core207_valid_outcome.npz',series_uid=valid['series_uid'],patient_id=valid['patient_id'],target=valid['target'],z2d=valid['z2d'].astype(np.float32),external=ext_b)
    audit={'status':'PASS','train_rows':len(a),'valid_rows':len(b),'motion_invalid_train':int(a.temporal_motion_invalid.sum()),'motion_invalid_valid':int(b.temporal_motion_invalid.sum()),'hemo_fully_trusted_train':int(a.hemo_valid.sum()),'hemo_fully_trusted_valid':int(b.hemo_valid.sum()),'jac42_finite':True,'z2d_shapes':[list(train['z2d'].shape),list(valid['z2d'].shape)],'outcome_used_only_as_frozen_target':True}
    atomic_json(audit,run/'feature_master/FEATURE_MASTER_AUDIT.json')
    atomic_json({'jac42':JAC,'hemo36':HEMO,'validity_indicators':VALIDITY,'external_feature_order':JAC+HEMO+VALIDITY},run/'feature_master/FEATURE_SCHEMA.json')
    return train,valid,audit


def run_spec(run: Path, name: str, ztr, zva, rawtr, rawva, y, folds, patients, valid_y, columns: list[str], cfg: dict) -> dict:
    out=run/name; out.mkdir(); rawtr,rawva=rawtr[:,columns],rawva[:,columns]; oof=np.full(len(y),np.nan,np.float32); fold_rows=[]
    for fold in range(1,6):
        dev,hold=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
        rel_a,rel_b=patient_inner_split(y[dev],patients[dev],cfg['inner_fraction'],cfg['seed']+fold); ia,ib=dev[rel_a],dev[rel_b]
        prep_search=fit_preprocessor(rawtr,ia); ext_search=transform(rawtr,prep_search)
        epoch,ap=search_epoch(ztr,ext_search,y,ia,ib,cfg,out/f'fold_{fold}_epoch_search.csv',cfg['seed']+fold*100)
        prep=fit_preprocessor(rawtr,dev); ext=transform(rawtr,prep); model=fresh_refit(ztr,ext,y,dev,epoch,cfg,cfg['seed']+fold*10000,out/f'fold_{fold}_model.pt')
        prob=predict(model,ztr,ext,hold,cfg['batch'],cfg['device']); oof[hold]=prob
        np.savez_compressed(out/f'fold_{fold}_preprocessor.npz',**prep,feature_indices=np.asarray(columns,dtype=np.int32))
        fold_rows.append({'fold':fold,'selected_epoch':epoch,'best_inner_AUPRC':ap,'development_n':len(dev),'holdout_n':len(hold),**metrics(y[hold],prob),'valid_used_for_selection':False})
    if not np.isfinite(oof).all(): raise AssertionError('incomplete OOF')
    pd.DataFrame(fold_rows).to_csv(out/'TRAIN_FOLD_METRICS.csv',index=False)
    pd.DataFrame({'series_uid':train_ids,'patient_id':patients,'target_y_abs':y,'historical_outcome_fold':folds,'probability':oof}).to_csv(out/'TRAIN_OOF_PREDICTIONS.csv',index=False)
    ia,ib=patient_inner_split(y,patients,cfg['inner_fraction'],cfg['seed']+9000); prep_search=fit_preprocessor(rawtr,ia); ext_search=transform(rawtr,prep_search)
    final_epoch,final_ap=search_epoch(ztr,ext_search,y,ia,ib,cfg,out/'FINAL_TRAIN_ONLY_EPOCH_SEARCH.csv',cfg['seed']+900000)
    prep_final=fit_preprocessor(rawtr,np.arange(len(y))); ext_train=transform(rawtr,prep_final); ext_valid=transform(rawva,prep_final)
    model=fresh_refit(ztr,ext_train,y,np.arange(len(y)),final_epoch,cfg,cfg['seed']+990000,out/'FINAL_MODEL.pt')
    checkpoint=torch.load(out/'FINAL_MODEL.pt',map_location='cpu'); checkpoint['preprocessor']=prep_final; checkpoint['external_feature_indices']=columns; checkpoint['external_feature_names']=[EXTERNAL_ORDER[i] for i in columns]; torch.save(checkpoint,out/'FINAL_MODEL.pt')
    np.savez_compressed(out/'FINAL_PREPROCESSOR.npz',**prep_final,feature_indices=np.asarray(columns,dtype=np.int32))
    valid_prob=predict(model,zva,ext_valid,np.arange(len(zva)),cfg['batch'],cfg['device'])
    pd.DataFrame({'series_uid':valid_ids,'patient_id':valid_patients,'target_y_abs':valid_y,'probability':valid_prob}).to_csv(out/'FINAL_VALID_PREDICTIONS.csv',index=False)
    final_metrics={'model':name,'external_dim':len(columns),'external_feature_names':[EXTERNAL_ORDER[i] for i in columns],'final_selected_epoch':final_epoch,'final_inner_AUPRC':final_ap,'train_oof':metrics(y,oof),'final_valid207':metrics(valid_y,valid_prob),'valid_inference_count':1,'outcome_fold_models_applied_to_valid':False,'preprocessor_fit_scope':'Train781_only'}
    atomic_json(final_metrics,out/'FINAL_VALID_METRICS.json')
    return final_metrics


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--run-id',default=f'local_reference_jacobian_hemo_outcome_core781_207_{time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())}'); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--amp',action='store_true'); ap.add_argument('--preflight-only',action='store_true'); args=ap.parse_args()
    if not args.run_id.startswith('local_reference_jacobian_hemo_outcome_core781_207_'): raise ValueError('invalid outcome RUN_ID')
    run=OUT_ROOT/args.run_id
    if run.exists(): raise FileExistsError(f'refusing overwrite: {run}')
    for x in ['cohort','feature_master','configs','reports','logs']: (run/x).mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    global train_ids,valid_ids,valid_patients,EXTERNAL_ORDER
    cfg={'seed':20260813,'batch':32,'lr':1e-4,'wd':1e-3,'max_epochs':160,'min_epochs':15,'patience':20,'inner_fraction':.18,'device':device,'amp':bool(args.amp and device.type=='cuda')}
    train,valid,audit=build_master(run); train_ids=train['series_uid'].astype(str); valid_ids=valid['series_uid'].astype(str); valid_patients=valid['patient_id'].astype(str); EXTERNAL_ORDER=JAC+HEMO+VALIDITY
    y,folds,patients=train['target'].astype(int),train['fold'].astype(int),train['patient_id'].astype(str); ztr,zva=train['z2d'].astype(np.float32),valid['z2d'].astype(np.float32); rawtr=np.load(run/'feature_master/core781_train_outcome.npz')['external']; rawva=np.load(run/'feature_master/core207_valid_outcome.npz')['external']
    atomic_json({'config':{**{k:v for k,v in cfg.items() if k!='device'},'device':str(device)},'contract':'NEXT_OUTCOME_TRAINING_EXECUTION_PLAN_ZH.md','outcome_target':'y_abs','upstream_rerun':False},run/'configs/LOCKED_OUTCOME_CONFIG.json')
    pd.DataFrame({'series_uid':sorted(pd.read_csv(MOTION_QC).query("group == 'largest_rotation'")['series_uid'].astype(str))}).to_csv(run/'configs/MOTION_INVALID_UIDS.csv',index=False)
    input_paths=[CORE_ROOT/'feature_master/core781_train.npz,.npz',CORE_ROOT/'feature_master/core207_valid.npz',TECH_ROOT/'featurebanks/train800_jacobian_existing42.csv',TECH_ROOT/'featurebanks/valid211_jacobian_existing42.csv',TECH_ROOT/'featurebanks/train800_hemo_compact36.csv',TECH_ROOT/'featurebanks/valid211_hemo_compact36.csv',MOTION_QC,MODEL_ROOT/'model.py']
    atomic_json({'sha256':{str(p):sha256_file(p) for p in input_paths}},run/'feature_master/INPUT_SHA256.json')
    # Fixed preflight is completed before a target-dependent training loop begins.
    atomic_json({'status':'PASS','cuda_available':torch.cuda.is_available(),'feature_master_audit':audit,'external_dims':{'jac42':42,'hemo36_validity':40,'jac42_hemo36_validity':82}},run/'configs/PREFLIGHT.json')
    if args.preflight_only:
        atomic_json({'pid':os.getpid(),'finished_utc':time.strftime('%FT%TZ',time.gmtime()),'status':'PREFLIGHT_PASS'},run/'logs/run.pid')
        print(json.dumps({'run':str(run),'status':'PREFLIGHT_PASS'},ensure_ascii=False)); return 0
    jac_idx=list(range(42)); hemo_idx=list(range(42,82)); both=list(range(82))
    results=[]
    for name,idx in [('jac42_formal_gated',jac_idx),('hemo36_formal_gated',hemo_idx),('jac42_hemo36_formal_gated',both)]: results.append(run_spec(run,name,ztr,zva,rawtr,rawva,y,folds,patients,valid['target'].astype(int),idx,cfg))
    base=json.loads((CORE_ROOT/'3167_formal/FINAL_VALID_METRICS.json').read_text())
    rows=[{'Model':'3167_frozen_baseline','Input':'z2D only','External dim':0,'Train OOF AUC':base['train_oof']['ROC_AUC'],'Train OOF AUPRC':base['train_oof']['AUPRC'],'Final Valid207 AUC':base['final_valid207']['ROC_AUC'],'Final Valid207 AUPRC':base['final_valid207']['AUPRC'],'Selected epoch':base['final_selected_epoch'],'Valid inference count':base['valid_inference_count']}]
    for r in results: rows.append({'Model':r['model'],'Input':'z2D + external','External dim':r['external_dim'],'Train OOF AUC':r['train_oof']['ROC_AUC'],'Train OOF AUPRC':r['train_oof']['AUPRC'],'Final Valid207 AUC':r['final_valid207']['ROC_AUC'],'Final Valid207 AUPRC':r['final_valid207']['AUPRC'],'Selected epoch':r['final_selected_epoch'],'Valid inference count':1})
    table=pd.DataFrame(rows); table.to_csv(run/'reports/FORMAL_JAC_HEMO_COMPARISON.csv',index=False)
    by={r['model']:r for r in results}; deltas={name:{'vs_3167_valid_auc':r['final_valid207']['ROC_AUC']-base['final_valid207']['ROC_AUC'],'vs_3167_valid_auprc':r['final_valid207']['AUPRC']-base['final_valid207']['AUPRC']} for name,r in by.items()}
    deltas['jac42_hemo36_vs_jac42']={'valid_auc':by['jac42_hemo36_formal_gated']['final_valid207']['ROC_AUC']-by['jac42_formal_gated']['final_valid207']['ROC_AUC'],'valid_auprc':by['jac42_hemo36_formal_gated']['final_valid207']['AUPRC']-by['jac42_formal_gated']['final_valid207']['AUPRC']}
    deltas['jac42_hemo36_vs_hemo36']={'valid_auc':by['jac42_hemo36_formal_gated']['final_valid207']['ROC_AUC']-by['hemo36_formal_gated']['final_valid207']['ROC_AUC'],'valid_auprc':by['jac42_hemo36_formal_gated']['final_valid207']['AUPRC']-by['hemo36_formal_gated']['final_valid207']['AUPRC']}
    report={'status':'PASS','run_id':args.run_id,'feature_master_audit':audit,'comparison':rows,'deltas':deltas,'forbidden_actions':{'outcome_valid_used_for_selection':False,'g0_or_temporal_or_jacobian_rerun':False,'segresnet_rerun':False,'3167_rerun_or_reinference':False}}
    atomic_json(report,run/'reports/FORMAL_JAC_HEMO_COMPARISON.json'); atomic_json(report,run/'reports/LOCAL_REFERENCE_JACOBIAN_HEMO_OUTCOME_FINAL_REPORT.json')
    (run/'reports/LOCAL_REFERENCE_JACOBIAN_HEMO_OUTCOME_FINAL_REPORT.md').write_text('# Local Reference JAC/HEMO outcome report\n\n```json\n'+json.dumps(report,ensure_ascii=False,indent=2)+'\n```\n',encoding='utf-8')
    atomic_json({'pid':os.getpid(),'finished_utc':time.strftime('%FT%TZ',time.gmtime()),'status':'COMPLETE'},run/'logs/run.pid')
    print(json.dumps({'run':str(run),'status':'PASS','models':[x['model'] for x in results]},ensure_ascii=False))
    return 0
if __name__=='__main__': raise SystemExit(main())

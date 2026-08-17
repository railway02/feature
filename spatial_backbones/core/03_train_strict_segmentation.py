#!/usr/bin/env python3
"""Expanded strict five-fold segmentation search and fresh refit."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import pandas as pd
import torch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from common import assert_signature, atomic_csv, atomic_json, atomic_torch_save, load_config, run_signature, set_seed, sha256_file
from data import estimate_pos_weight, expanded_strict_fold_split, split_hash
from model_interface import build_model, model_parameter_count
from trainer import evaluate, is_better, make_loader, runtime_batch, train_epoch

FAMILIES=['segresnet','deeplabv3plus_resnet50_imagenet']

def paths(cfg,family,fold):
    out=Path(cfg['output_root'])/'expanded_strict'/'segmentation'/family/f'fold_{fold}'
    rep=Path(cfg['report_root'])/'expanded_strict_segmentation'/family/f'fold_{fold}'
    out.mkdir(parents=True,exist_ok=True); rep.mkdir(parents=True,exist_ok=True); return out,rep

def signature(cfg,family,legal):
    pre=cfg['pretrained']['sha256'] if cfg['models'][family]['pretrained'] else 'random_initialization'
    return run_signature(cfg,split_hash(legal),family,pre)

def run_fold(cfg,family,fold,device):
    legal,inner_train,inner_valid,_,audit=expanded_strict_fold_split(cfg,fold); out,rep=paths(cfg,family,fold); sig=signature(cfg,family,legal)
    success=out/'SUCCESS.json'
    if success.is_file():
        prior=json.loads(success.read_text(encoding='utf-8')); assert_signature(prior['run_signature'],sig); return prior
    seed=int(cfg['development']['base_seed'])+fold*100; batch,accum=runtime_batch(cfg,family); settings=cfg['models'][family]
    atomic_csv(legal,rep/'segmentation_legal_split_manifest.csv'); atomic_json(audit,rep/'legality_audit.json')
    set_seed(seed); model=build_model(family,cfg,load_pretrained=True).to(device); opt=AdamW(model.parameters(),lr=float(settings['learning_rate']),weight_decay=float(settings['weight_decay'])); scaler=GradScaler(enabled=bool(cfg['development']['amp'] and device.type=='cuda'))
    pw=estimate_pos_weight(inner_train,cfg); train_loader=make_loader(inner_train,cfg,True,seed,batch); valid_loader=make_loader(inner_valid,cfg,False,seed+1,batch)
    snapshot={'protocol':'expanded_strict_segmentation_trainxlsx_patient_outer_crossfit','family':family,'outer_fold':fold,'run_signature':sig,'legal_rows':int(len(legal)),'legal_patients':int(legal.patient_id.nunique()),'inner_train_rows':int(len(inner_train)),'inner_valid_rows':int(len(inner_valid)),'inner_valid_adverse_only':True,'physical_batch_size':batch,'gradient_accumulation':accum,'pos_weight_search':pw,'threshold':cfg['development']['threshold'],'augmentation':cfg['augmentation'],'loss':cfg['loss'],'settings':settings,'model_parameters':model_parameter_count(model)}
    atomic_json(snapshot,rep/'config_snapshot.json')
    history=[]; start=1; best=None; best_epoch=0; bad=0; last=out/'search_last.pt'
    if cfg['development']['resume'] and last.is_file():
        state=torch.load(last,map_location='cpu'); assert_signature(state['run_signature'],sig); model.load_state_dict(state['state_dict']); opt.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler']); history=list(state['history']); start=int(state['epoch'])+1; best=state['best_metrics']; best_epoch=int(state['best_epoch']); bad=int(state['bad_epochs'])
    for epoch in range(start,int(cfg['development']['max_epochs'])+1):
        train=train_epoch(model,train_loader,opt,scaler,device,pw,cfg,accum); valid,_=evaluate(model,valid_loader,device,cfg); record={'epoch':epoch,**train,**{f'valid_{k}':v for k,v in valid.items()}}; history.append(record); atomic_csv(pd.DataFrame(history),rep/'search_history.csv'); print(json.dumps({'family':family,'fold':fold,'stage':'search',**record},ensure_ascii=False),flush=True)
        if is_better(valid,best):
            best,best_epoch,bad=valid,epoch,0; atomic_torch_save({'state_dict':model.state_dict(),'epoch':epoch,'metrics':valid,'run_signature':sig,'config_snapshot':snapshot},out/'search_best.pt')
        else: bad+=1
        atomic_torch_save({'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'history':history,'best_metrics':best,'best_epoch':best_epoch,'bad_epochs':bad,'run_signature':sig},last)
        if epoch>=int(cfg['development']['min_epochs']) and bad>=int(cfg['development']['patience']): break
    best_state=torch.load(out/'search_best.pt',map_location='cpu'); model.load_state_dict(best_state['state_dict']); final_valid,rows=evaluate(model,valid_loader,device,cfg); rows['gt_size_quartile']=pd.qcut(rows.gt_pixels,4,labels=['Q1-smallest','Q2','Q3','Q4-largest']).astype(str); atomic_csv(rows,rep/'search_best_inner_valid_predictions.csv')
    del model
    if device.type=='cuda':torch.cuda.empty_cache()
    # Fresh refit on every legal segmentation row; no holdout/valid.xlsx rows.
    set_seed(seed+10000); model=build_model(family,cfg,load_pretrained=True).to(device); opt=AdamW(model.parameters(),lr=float(settings['learning_rate']),weight_decay=float(settings['weight_decay'])); scaler=GradScaler(enabled=bool(cfg['development']['amp'] and device.type=='cuda')); refit_pw=estimate_pos_weight(legal,cfg); loader=make_loader(legal,cfg,True,seed+10000,batch); refit=[]
    for epoch in range(1,best_epoch+1):
        metric=train_epoch(model,loader,opt,scaler,device,refit_pw,cfg,accum); refit.append({'epoch':epoch,**metric}); atomic_csv(pd.DataFrame(refit),rep/'refit_history.csv'); print(json.dumps({'family':family,'fold':fold,'stage':'fresh_refit',**refit[-1]},ensure_ascii=False),flush=True)
    atomic_torch_save({'state_dict':model.state_dict(),'selected_epochs':best_epoch,'run_signature':sig,'config_snapshot':snapshot,'refit_rows':int(len(legal)),'refit_patients':int(legal.patient_id.nunique()),'pos_weight_refit':refit_pw},out/'model.pt')
    result={'status':'success','family':family,'outer_fold':fold,'protocol':'expanded_strict_segmentation_trainxlsx_patient_outer_crossfit','outer_holdout_evaluated_during_training':False,'valid_xlsx_used_during_training_or_selection':False,'best_epoch':best_epoch,'best_inner_valid_metrics':final_valid,'run_signature':sig,'search_best_checkpoint':str(out/'search_best.pt'),'model_checkpoint':str(out/'model.pt'),'model_sha256':sha256_file(out/'model.pt')}
    atomic_json(result,rep/'metrics.json'); atomic_json(result,success); return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--family',choices=FAMILIES+['all'],default='all'); p.add_argument('--fold',choices=['1','2','3','4','5','all'],default='all'); p.add_argument('--device',default='cuda:0'); a=p.parse_args(); cfg=load_config(a.config); os.environ['TORCH_HOME']=cfg['torch_home']
    if not (Path(cfg['report_root'])/'expanded_strict_preflight'/'SUCCESS.json').is_file() or not (Path(cfg['report_root'])/'expanded_strict_smoke'/'SUCCESS.json').is_file(): raise RuntimeError('expanded strict preflight and smoke must PASS before training')
    device=torch.device(a.device); families=FAMILIES if a.family=='all' else [a.family]; folds=range(1,6) if a.fold=='all' else [int(a.fold)]; results=[run_fold(cfg,f,k,device) for f in families for k in folds]; print(json.dumps(results,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

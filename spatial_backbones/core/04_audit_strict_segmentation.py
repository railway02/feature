#!/usr/bin/env python3
"""Evaluate each expanded-strict refit checkpoint only on its excluded patients."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import pandas as pd
import torch
from common import atomic_csv, atomic_json, load_config
from data import build_expanded_strict_population
from model_interface import build_model
from trainer import evaluate, make_loader

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--family',choices=['segresnet','deeplabv3plus_resnet50_imagenet'],required=True); p.add_argument('--device',default='cuda:0'); a=p.parse_args(); cfg=load_config(a.config); os.environ['TORCH_HOME']=cfg['torch_home']; device=torch.device(a.device)
 pool,valid_pool,_,_=build_expanded_strict_population(cfg); rows=[]; summaries=[]; valid_rows=[]; valid_summaries=[]
 for fold in range(1,6):
  holdout=pool[pool.fold.eq(fold)].copy(); checkpoint=Path(cfg['output_root'])/'expanded_strict'/'segmentation'/a.family/f'fold_{fold}'/'model.pt'
  if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
  model=build_model(a.family,cfg,load_pretrained=False).to(device).eval(); raw=torch.load(checkpoint,map_location='cpu'); model.load_state_dict(raw['state_dict'],strict=True); batch=int(cfg['models']['segresnet']['physical_batch_size']) if a.family=='segresnet' else 8; loader=make_loader(holdout,cfg,False,20260810+fold,batch); metrics,pred=evaluate(model,loader,device,cfg); pred['outer_fold']=fold; pred['segmentation_key']=pred['series_uid']; pred['is_adverse_extra_series']=pred['segmentation_key'].isin(cfg['expanded_strict_segmentation']['adverse_extra_png_keys']); rows.append(pred); summaries.append({'fold':fold,**metrics,'n_extra_adverse_series_rows':int(pred['is_adverse_extra_series'].sum())})
  # This is an independent, post-freeze evaluation of every refit checkpoint.
  # valid.xlsx was absent from training, epoch selection, and refit.
  valid_loader=make_loader(valid_pool,cfg,False,20260900+fold,batch); valid_metrics,valid_pred=evaluate(model,valid_loader,device,cfg); valid_pred['outer_fold']=fold; valid_pred['evaluation_population']='frozen_valid_xlsx'; valid_rows.append(valid_pred); valid_summaries.append({'fold':fold,**valid_metrics,'n_valid_rows':int(len(valid_pred))}); del model
  if device.type=='cuda': torch.cuda.empty_cache()
 detail=pd.concat(rows,ignore_index=True); valid_detail=pd.concat(valid_rows,ignore_index=True); out=Path(cfg['report_root'])/'expanded_strict_audit'/a.family; atomic_csv(detail,out/'strict_outer_holdout_predictions.csv'); atomic_csv(pd.DataFrame(summaries),out/'strict_outer_holdout_metrics.csv'); atomic_csv(valid_detail,out/'frozen_valid_predictions_by_fold.csv'); atomic_csv(pd.DataFrame(valid_summaries),out/'frozen_valid_metrics_by_fold.csv'); atomic_json({'status':'success','family':a.family,'protocol':'expanded_strict','n_outer_holdout_segmentation_rows':int(len(detail)),'outer_holdout_only_for_strict_oof':True,'valid_xlsx_evaluated_after_freeze':True,'valid_xlsx_rows_per_frozen_checkpoint':int(len(valid_pool)),'valid_xlsx_used_for_training_or_selection':False,'fold_metrics':summaries,'frozen_valid_fold_metrics':valid_summaries},out/'SUCCESS.json')
if __name__=='__main__':main()

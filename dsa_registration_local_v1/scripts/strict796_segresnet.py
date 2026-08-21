#!/usr/bin/env python3
"""Local Reference 796 strict SegResNet cross-fit and PredROI extraction.

It imports the frozen V6 implementation (data transforms, model, loss,
optimizer and checkpoint policy) rather than reimplementing a segmentation
network.  Only cohort plumbing is local to this formal experiment.
"""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.cuda.amp import GradScaler,autocast
ROOT=Path('/root/autodl-tmp'); CODE=ROOT/'aneurysm/code/api_png2d_spatial_backbones_v6_strict';sys.path.insert(0,str(CODE))
from common import load_config,atomic_csv,atomic_json,atomic_torch_save,set_seed
from data import unroll_phase_rows,SegmentationDataset,estimate_pos_weight
from model_interface import build_model,global_pool,roi_pool
from trainer import make_loader,train_epoch,evaluate,is_better

def cases(out,split):
 p=out/('train796_strict_folds.csv' if split=='Train' else 'valid211_case_manifest.csv');d=pd.read_csv(p,dtype={'series_uid':str,'patient_id':str})
 d=d.rename(columns={'pre_image_path':'pre_image','post_image_path':'post_image','pre_mask_path':'pre_mask','post_mask_path':'post_mask'})
 for c in ['pre_image','post_image','pre_mask','post_mask']:
  if c not in d: raise KeyError(f'{p}: {c}')
 return d
def inner(dev,fold,seed):
 ps=np.unique(dev.patient_id.astype(str));r=np.random.default_rng(seed+fold*100);r.shuffle(ps);n=max(1,round(len(ps)*.18));v=set(ps[:n]);return dev[~dev.patient_id.isin(v)],dev[dev.patient_id.isin(v)]
def fold_train(cfg,out,fold,device):
 ck=out/'segresnet' / f'fold_{fold}'/'model.pt';rep=out/'segresnet_reports'/f'fold_{fold}';ck.parent.mkdir(parents=True,exist_ok=True);rep.mkdir(parents=True,exist_ok=True)
 if ck.is_file(): return
 allc=cases(out,'Train');dev=allc[allc.fold.ne(fold)].copy();hold=allc[allc.fold.eq(fold)].copy();assert not(set(dev.patient_id)&set(hold.patient_id))
 tr,va=inner(dev,fold,20260819);trp=unroll_phase_rows(tr,cfg['data']['phases']);vap=unroll_phase_rows(va,cfg['data']['phases']);legal=unroll_phase_rows(dev,cfg['data']['phases'])
 seed=20260819+fold*100;set_seed(seed);model=build_model('segresnet',cfg,load_pretrained=False).to(device);opt=torch.optim.AdamW(model.parameters(),lr=cfg['models']['segresnet']['learning_rate'],weight_decay=cfg['models']['segresnet']['weight_decay']);sc=GradScaler(enabled=device.type=='cuda');pw=estimate_pos_weight(trp,cfg);tl=make_loader(trp,cfg,True,seed,4);vl=make_loader(vap,cfg,False,seed+1,4)
 hist=[];best=None;be=0;bad=0
 for ep in range(1,int(cfg['development']['max_epochs'])+1):
  a=train_epoch(model,tl,opt,sc,device,pw,cfg,1);b,_=evaluate(model,vl,device,cfg);hist.append({'epoch':ep,**a,**{'valid_'+k:v for k,v in b.items()}})
  if is_better(b,best):best=b;be=ep;bad=0
  else:bad+=1
  atomic_csv(pd.DataFrame(hist),rep/'epoch_search.csv')
  if ep>=int(cfg['development']['min_epochs']) and bad>=int(cfg['development']['patience']):break
 # Fresh development refit is the frozen strict policy; outer holdout never enters.
 del model;torch.cuda.empty_cache() if device.type=='cuda' else None;set_seed(seed+10000);model=build_model('segresnet',cfg,load_pretrained=False).to(device);opt=torch.optim.AdamW(model.parameters(),lr=cfg['models']['segresnet']['learning_rate'],weight_decay=cfg['models']['segresnet']['weight_decay']);sc=GradScaler(enabled=device.type=='cuda');pw=estimate_pos_weight(legal,cfg);dl=make_loader(legal,cfg,True,seed+10000,4)
 ref=[]
 for ep in range(1,be+1):ref.append({'epoch':ep,**train_epoch(model,dl,opt,sc,device,pw,cfg,1)})
 atomic_csv(pd.DataFrame(ref),rep/'fresh_refit_history.csv');atomic_csv(pd.concat([tr.assign(partition='inner_train'),va.assign(partition='inner_valid'),hold.assign(partition='outer_holdout')]),rep/'split_manifest.csv')
 atomic_torch_save({'state_dict':model.state_dict(),'selected_epoch':be,'outer_fold':fold,'protocol':'strict796_outer_crossfit_fresh_development_refit','outer_holdout_used_for_training':False},ck);atomic_json({'status':'success','fold':fold,'selected_epoch':be,'development_series':len(dev),'holdout_series':len(hold),'gtroi_used_for_feature':False},rep/'SUCCESS.json')
def phase_frame(c,cfg):return unroll_phase_rows(c,cfg['data']['phases'])
@torch.no_grad()
def features(model,frame,cfg,device):
 from torch.utils.data import DataLoader
 ds=SegmentationDataset(frame,cfg,augment=False);dl=DataLoader(ds,batch_size=4,shuffle=False,num_workers=4,pin_memory=device.type=='cuda',persistent_workers=True);g=p=None
 for x,_,idx,_,_ in dl:
  with autocast(enabled=device.type=='cuda'):
   f,l=model.encode_and_decode(x.to(device));a=global_pool(f);b,m=roi_pool(f,torch.sigmoid(l),'bilinear')
  if g is None:g=np.empty((len(ds),a.shape[1]),np.float32);p=np.empty_like(g)
  i=np.asarray(idx);g[i]=a.float().cpu();p[i]=b.float().cpu()
 return ds.rows,g,p
def pack(c,rows,g,p):
 look={(str(r.series_uid),str(r.phase)):i for i,r in enumerate(rows.itertuples(index=False))};z=[]
 for r in c.itertuples(index=False):
  a,b=look[(str(r.series_uid),'Pre')],look[(str(r.series_uid),'Post')];z.append(np.r_[g[a],p[a],g[b],p[b]])
 return np.stack(z).astype(np.float32)
def extract(cfg,out,device):
 tr,va=cases(out,'Train'),cases(out,'Valid');bytr=[];byva=[]
 for fold in range(1,6):
  ck=torch.load(out/'segresnet'/f'fold_{fold}'/'model.pt',map_location='cpu');m=build_model('segresnet',cfg,load_pretrained=False).to(device).eval();m.load_state_dict(ck['state_dict'],strict=True)
  r,g,p=features(m,phase_frame(tr,cfg),cfg,device);bytr.append(pack(tr,r,g,p));r,g,p=features(m,phase_frame(va,cfg),cfg,device);byva.append(pack(va,r,g,p));del m
 bytr=np.stack(bytr,1);byva=np.stack(byva,1);oof=bytr[np.arange(len(tr)),tr.fold.to_numpy()-1]
 assert oof.shape[0]==796 and byva.shape[:2]==(211,5) and np.isfinite(oof).all() and np.isfinite(byva).all()
 np.savez_compressed(out/'train796_strict_predroi_z2d.npz',z2d=oof,series_uid=tr.series_uid.to_numpy(),patient_id=tr.patient_id.to_numpy(),target=tr.target.to_numpy(),fold=tr.fold.to_numpy(),source_fold=tr.fold.to_numpy())
 np.savez_compressed(out/'valid211_predroi_z2d_by_fold.npz',z2d_by_fold=byva,series_uid=va.series_uid.to_numpy(),patient_id=va.patient_id.to_numpy(),target=va.target.to_numpy(),source_folds=np.arange(1,6))
 rows=[]
 for r in tr.itertuples(index=False):rows.append({'series_uid':r.series_uid,'patient_id':r.patient_id,'fold':r.fold,'checkpoint':str(out/'segresnet'/f'fold_{r.fold}'/'model.pt'),'feature_source':'automatic_PredROI','OOF':True})
 for r in va.itertuples(index=False):
  for f in range(1,6):rows.append({'series_uid':r.series_uid,'patient_id':r.patient_id,'fold':f,'checkpoint':str(out/'segresnet'/f'fold_{f}'/'model.pt'),'feature_source':'automatic_PredROI','OOF':False})
 atomic_csv(pd.DataFrame(rows),out/'segresnet_feature_manifest.csv')
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','extract']);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();cfg=load_config(ROOT/'aneurysm/configs/api_png2d_spatial_backbones_v6_strict.json');cfg['development']['base_seed']=20260819;cfg['development']['num_workers']=4;device=torch.device(a.device)
 if a.stage=='train':
  for f in range(1,6):fold_train(cfg,a.out,f,device)
 else:extract(cfg,a.out,device)
if __name__=='__main__':main()

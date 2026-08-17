#!/usr/bin/env python3
"""Extract per-fold Global/PredROI/GTROI features from frozen strict encoders."""
from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from common import atomic_json, load_config
from data import SegmentationDataset
from model_interface import build_model, global_pool, roi_pool

def phase_frame(cases,cfg):
 rows=[]
 for r in cases.itertuples(index=False):
  for phase in cfg['data']['phases']:
   k=phase.lower(); rows.append({'patient_id':str(r.patient_id),'series_uid':str(r.series_uid),'fold':int(r.fold),'phase':phase,'image_path':getattr(r,f'{k}_image'),'mask_path':getattr(r,f'{k}_mask')})
 return pd.DataFrame(rows)

@torch.no_grad()
def extract(model,frame,cfg,device,batch):
 from torch.utils.data import DataLoader
 ds=SegmentationDataset(frame,cfg,augment=False); dl=DataLoader(ds,batch_size=batch,shuffle=False,num_workers=int(cfg['development']['num_workers']),pin_memory=device.type=='cuda',persistent_workers=int(cfg['development']['num_workers'])>0)
 g=pr=gt=None
 for x,y,idx,_,_ in dl:
  x=x.to(device);y=y.to(device)
  with autocast(enabled=device.type=='cuda'):
   fmap,logits=model.encode_and_decode(x); zg=global_pool(fmap); zp,pm=roi_pool(fmap,torch.sigmoid(logits),'bilinear'); zt,gm=roi_pool(fmap,y,'area')
  if torch.any(gm<=1e-6): raise RuntimeError('GT ROI vanished at feature scale')
  if g is None:
   n=len(ds); g=np.empty((n,zg.shape[1]),np.float32); pr=np.empty((n,zp.shape[1]),np.float32); gt=np.empty((n,zt.shape[1]),np.float32)
  ii=np.asarray(idx); g[ii]=zg.float().cpu().numpy(); pr[ii]=zp.float().cpu().numpy(); gt[ii]=zt.float().cpu().numpy()
 return ds.rows,g,pr,gt

def pack(cases,rows,g,pr,gt):
 lookup={(str(r.series_uid),str(r.phase)):i for i,r in enumerate(rows.itertuples(index=False))}; glob=[]; pred=[]; oracle=[]
 for r in cases.itertuples(index=False):
  pre=lookup[(str(r.series_uid),'Pre')]; post=lookup[(str(r.series_uid),'Post')]
  glob.append(np.concatenate([g[pre],g[post]])); pred.append(np.concatenate([g[pre],pr[pre],g[post],pr[post]])); oracle.append(np.concatenate([g[pre],gt[pre],g[post],gt[post]]))
 return {'global':np.stack(glob).astype(np.float32),'pred_combined':np.stack(pred).astype(np.float32),'gt_combined':np.stack(oracle).astype(np.float32)}

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--family',choices=['segresnet','deeplabv3plus_resnet50_imagenet'],required=True); p.add_argument('--fold',choices=['1','2','3','4','5','all'],default='all');p.add_argument('--device',default='cuda:0');a=p.parse_args();cfg=load_config(a.config);os.environ['TORCH_HOME']=cfg['torch_home'];device=torch.device(a.device)
 case=pd.read_csv(cfg['sources']['v5_case_manifest'],dtype={'patient_id':str,'series_uid':str}); folds=range(1,6) if a.fold=='all' else [int(a.fold)]; outroot=Path(cfg['output_root'])/'expanded_strict'/'seg_features'/a.family
 for fold in folds:
  ck=Path(cfg['output_root'])/'expanded_strict'/'segmentation'/a.family/f'fold_{fold}'/'model.pt'; model=build_model(a.family,cfg,load_pretrained=False).to(device).eval(); model.load_state_dict(torch.load(ck,map_location='cpu')['state_dict'],strict=True); batch=4 if a.family=='segresnet' else 8
  summary={}
  for split in ['Train','Valid']:
   cases=case[case.split.eq(split)].copy().reset_index(drop=True); rows,g,pr,gt=extract(model,phase_frame(cases,cfg),cfg,device,batch); packed=pack(cases,rows,g,pr,gt); target=outroot/f'fold_{fold}';target.mkdir(parents=True,exist_ok=True); tmp=target/f'.{split.lower()}.tmp.npz';np.savez_compressed(tmp,**packed,series_uid=cases.series_uid.astype(str).to_numpy(),patient_id=cases.patient_id.astype(str).to_numpy(),target=cases.target.astype(int).to_numpy(),outer_fold=cases.fold.astype(int).to_numpy() if split=='Train' else np.zeros(len(cases),int));os.replace(tmp,target/f'{split.lower()}.npz');summary[split]={k:list(v.shape) for k,v in packed.items()}
  atomic_json({'status':'success','family':a.family,'fold':fold,'summary':summary,'valid_used_only_after_strict_training':True},outroot/f'fold_{fold}'/'SUCCESS.json');del model
if __name__=='__main__':main()

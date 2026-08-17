#!/usr/bin/env python3
"""Build unified by-fold and OOF views without averaging latent encoders."""
from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np
from common import atomic_json, load_config

def load(path):
 # Feature extraction serializes identifiers as NumPy object arrays.  These are
 # locally generated, trusted inputs; normalize them to fixed-width Unicode so
 # the unified featurebanks remain pickle-free for downstream consumers.
 with np.load(path, allow_pickle=True) as z:
  arrays = {k: np.asarray(z[k]) for k in z.files}
 for key in ('patient_id', 'series_uid'):
  if arrays[key].dtype == object:
   arrays[key] = arrays[key].astype(str)
 return arrays
def save(path,**arrays):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(f'.{path.stem}.tmp.npz');np.savez_compressed(tmp,**arrays);os.replace(tmp,path)
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--family',choices=['segresnet','deeplabv3plus_resnet50_imagenet'],required=True);a=p.parse_args();cfg=load_config(a.config);root=Path(cfg['output_root'])/'expanded_strict'/'seg_features'/a.family;out=Path(cfg['output_root'])/'expanded_strict'/'featurebanks'/a.family
 trains=[load(root/f'fold_{k}'/'train.npz') for k in range(1,6)];valids=[load(root/f'fold_{k}'/'valid.npz') for k in range(1,6)]
 for group,name in [(trains,'Train'),(valids,'Valid')]:
  ids=group[0]['series_uid'];
  if any(not np.array_equal(ids,x['series_uid']) for x in group[1:]):raise RuntimeError(f'{name} UID order mismatch')
 train=trains[0];valid=valids[0];folds=train['outer_fold'].astype(int)
 data_train={'patient_id':train['patient_id'],'series_uid':train['series_uid'],'target':train['target'],'outer_fold':folds,'global_by_fold':np.stack([x['global'] for x in trains],axis=1),'pred_combined_by_fold':np.stack([x['pred_combined'] for x in trains],axis=1),'gt_combined_by_fold':np.stack([x['gt_combined'] for x in trains],axis=1)}
 idx=np.arange(len(folds));data_train['global_oof']=data_train['global_by_fold'][idx,folds-1];data_train['pred_combined_oof']=data_train['pred_combined_by_fold'][idx,folds-1];data_train['gt_combined_oof']=data_train['gt_combined_by_fold'][idx,folds-1];data_train['oof_source_fold']=folds
 data_valid={'patient_id':valid['patient_id'],'series_uid':valid['series_uid'],'target':valid['target'],'global_by_fold':np.stack([x['global'] for x in valids],axis=1),'pred_combined_by_fold':np.stack([x['pred_combined'] for x in valids],axis=1),'gt_combined_by_fold':np.stack([x['gt_combined'] for x in valids],axis=1)}
 save(out/'train_spatial_features.npz',**data_train);save(out/'valid_spatial_features.npz',**data_valid)
 atomic_json({'status':'success','family':a.family,'train_rows':len(folds),'valid_rows':len(valid['series_uid']),'latent_averaging_forbidden':True,'train_keys':sorted(data_train),'valid_keys':sorted(data_valid)},out/'SUCCESS.json')
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Verify unified featurebank identifiers, finite values, and OOF fold routing."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from common import atomic_json,load_config
def read(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.asarray(z[k]) for k in z.files}
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--family',choices=['segresnet','deeplabv3plus_resnet50_imagenet'],required=True);a=p.parse_args();cfg=load_config(a.config);root=Path(cfg['output_root'])/'expanded_strict'/'featurebanks'/a.family;t=read(root/'train_spatial_features.npz');v=read(root/'valid_spatial_features.npz')
 checks={'train_rows':len(t['series_uid'])==781,'valid_rows':len(v['series_uid'])==207,'train_uid_unique':len(np.unique(t['series_uid']))==781,'valid_uid_unique':len(np.unique(v['series_uid']))==207,'oof_source_equals_outer_fold':np.array_equal(t['oof_source_fold'],t['outer_fold']),'finite_train':all(np.isfinite(t[k]).all() for k in ['global_by_fold','pred_combined_by_fold','gt_combined_by_fold','global_oof','pred_combined_oof']),'finite_valid':all(np.isfinite(v[k]).all() for k in ['global_by_fold','pred_combined_by_fold','gt_combined_by_fold'])}
 idx=np.arange(len(t['outer_fold']));checks['oof_view_matches_by_fold']=np.array_equal(t['pred_combined_oof'],t['pred_combined_by_fold'][idx,t['outer_fold'].astype(int)-1])
 if not all(checks.values()):raise RuntimeError(json.dumps(checks))
 atomic_json({'status':'PASS','family':a.family,'checks':checks,'latent_averaging_applied':False},root/'verification.json')
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Create the immutable 796/211 contracts for Local Reference formal work.

This is intentionally a data-contract step only: it never invokes registration,
SegResNet, outcome fitting, or reads an outcome during FOV selection.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path('/root/autodl-tmp'); REG=ROOT/'dsa_registration_local_reference_v1/outputs/local_reference_v1_20260819_overnight'
MASTER=ROOT/'aneurysm/manifests/api_temporal_data_master_v1_20260818/ready_mask_temporal_prepost_series.csv'
FROZEN=ROOT/'dsa_registration_local_reference_v1/outputs/formal_stage_a_c_20260819/stage_a/local_reference_train800_grouped_folds.csv'
DROP={'Train__481684__L__d160e0986a','Train__481684__R__06576556d1','Train__558832__R__06576556d1','Train__615855__VA__1b15013c42'}

def atomic_csv(x,p):
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name('.'+p.name+'.tmp');x.to_csv(t,index=False);os.replace(t,p)
def atomic_json(x,p):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name('.'+p.name+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)
def check(frame,n,train=True):
 assert len(frame)==n and frame.series_uid.nunique()==n, (len(frame),frame.series_uid.nunique())
 assert not frame.series_uid.duplicated().any()
 if train:
  assert frame.patient_id.astype(str).groupby(frame.patient_id.astype(str)).size().ge(1).all()
  assert frame.groupby('patient_id').fold.nunique().max()==1
  assert frame.target.notna().all() and frame.fold.notna().all()

def fov_select(train, out):
 # Uses only frozen registration geometry/QC and source metadata.  No target is
 # loaded into this table or used below.
 qc=pd.read_csv(REG/'train_registration_qc.csv',dtype={'series_uid':str,'patient_id':str})
 rows=[]
 for r in qc.itertuples(index=False):
  mp=REG/'train'/'cases'/str(r.series_uid)/'rigid_maps.npz'
  if mp.is_file():
   with np.load(mp,allow_pickle=False) as z: side=max(z['logj'].shape)
  else: side=np.nan
  rows.append((str(r.series_uid),side))
 geom=pd.DataFrame(rows,columns=['series_uid','g0_post_side'])
 # Source-resolution and crop-size difference are explicit geometry covariates.
 # Phase boxes/canvases come from the frozen preprocessing contract, not masks
 # or outcomes.  They give the requested source-resolution, border and
 # Pre/Post crop-difference coverage.
 import sys
 sys.path.insert(0,str(ROOT/'dsa_registration_local_reference_v1'))
 from dsa_local_reg.common import load_config
 from dsa_local_reg.preprocessing_adapter import load_local_reference_pairs
 pairs={p.series_uid:p for p in load_local_reference_pairs(load_config(ROOT/'dsa_registration_local_reference_v1/config/default.yaml'),split='Train')}
 gr=[]
 for uid,p in pairs.items():
  h,w=p.post.canvas_shape_yx; b=p.post.expanded_bbox
  gr.append((uid,max(h,w),min(b.x0,w-b.x1,b.y0,h-b.y1),abs(p.pre.expanded_bbox.height-p.post.expanded_bbox.height)+abs(p.pre.expanded_bbox.width-p.post.expanded_bbox.width)))
 g=pd.DataFrame(gr,columns=['series_uid','source_resolution','lesion_to_border_distance','prepost_crop_size_difference'])
 x=train[['series_uid','patient_id','series_path']].merge(geom,on='series_uid',how='left',validate='one_to_one').merge(g,on='series_uid',how='left',validate='one_to_one')
 x['stratum']=pd.cut(x.g0_post_side,[95,127,159,191,255,np.inf],labels=['96-127','128-159','160-191','192-255','>=256']).astype(str)
 chosen=[]
 for i,s in enumerate(['96-127','128-159','160-191','192-255','>=256']):
  z=x[x.stratum.eq(s)].sort_values(['source_resolution','lesion_to_border_distance','prepost_crop_size_difference','series_uid'],kind='mergesort')
  if len(z)<10: raise RuntimeError(f'FOV stratum {s}: only {len(z)} candidates')
  # systematic spread across deterministic order, frozen before any expanded run
  ix=np.linspace(0,len(z)-1,10,dtype=int);chosen.append(z.iloc[ix].assign(selection_rank=np.arange(1,11)))
 y=pd.concat(chosen,ignore_index=True); assert len(y)==50 and y.series_uid.nunique()==50
 atomic_csv(y,out/'fov50_series.csv')
 atomic_json({'selection':'frozen_geometry_only_systematic_stratified','outcome_used':False,'target_column_absent':True,'counts':y.stratum.value_counts().to_dict()},out/'fov50_selection.json')

def main():
 a=argparse.ArgumentParser();a.add_argument('--outcome-dir',type=Path,required=True);a.add_argument('--fov-dir',type=Path,required=True);args=a.parse_args()
 frozen=pd.read_csv(FROZEN,dtype={'series_uid':str,'patient_id':str})
 train=frozen[~frozen.series_uid.isin(DROP)].copy();check(train,796)
 # Canonical master is used to assert 800/211 membership and to supply images.
 m=pd.read_csv(MASTER,dtype={'series_uid':str,'patient_id':str}); mt=m[m.split.eq('Train')].copy();mv=m[m.split.eq('Valid')].copy();assert len(mt)==800 and len(mv)==211
 train=train.merge(mt.drop(columns=['patient_id'],errors='ignore'),on='series_uid',how='left',validate='one_to_one',suffixes=('','_master'))
 valid_case=pd.read_csv(ROOT/'dsa_registration_local_reference_v1/outputs/formal_stage_a_c_20260819/stage_a/local_reference_task_rows_valid.csv',dtype={'series_uid':str,'patient_id':str})
 # Valid labels/mapping and image paths are frozen in the Stage-A case contract.
 # The canonical master supplies membership validation only; it must not
 # overwrite this explicit record-to-series mapping.
 valid=valid_case.copy()
 assert set(valid.series_uid)==set(mv.series_uid)
 check(valid,211,False);assert valid.target.notna().all()
 atomic_csv(train,args.outcome_dir/'train796_strict_folds.csv');atomic_csv(valid,args.outcome_dir/'valid211_case_manifest.csv')
 for split,frame in [('train',train),('valid',valid)]:
  r=pd.read_csv(REG/f'{split}_registration_features.csv',dtype={'series_uid':str,'patient_id':str})
  r=r[r.series_uid.isin(frame.series_uid)].copy(); assert len(r)==len(frame) and r.series_uid.nunique()==len(frame)
  atomic_csv(r,args.outcome_dir/f'{split}{len(frame)}_registration_features.csv')
 audit={'train_rows':796,'valid_rows':211,'dropped_series_uid':sorted(DROP),'fold_counts':train.fold.value_counts().sort_index().to_dict(),'patient_cross_fold':False,'target_unique':True,'registration_subset_only':True}
 atomic_json(audit,args.outcome_dir/'COHORT_CONTRACT.json')
 fov_select(mt,args.fov_dir)
 print(json.dumps(audit,ensure_ascii=False))
if __name__=='__main__':main()

#!/usr/bin/env python
"""Explicitly aggregate series-level outputs into one patient row when the baseline does so.

No default policy is provided: the choice MUST match PredROI/CAVE's sample unit.
"""
import argparse
import pandas as pd
p=argparse.ArgumentParser(); p.add_argument('--features',required=True); p.add_argument('--manifest',required=True); p.add_argument('--out',required=True); p.add_argument('--policy',choices=['best_mapping','mean'],required=True)
a=p.parse_args(); f=pd.read_csv(a.features); m=pd.read_csv(a.manifest)
key=['patient_id','series_uid']
meta=m[['patient_id','series_uid','pre_mapping_score','post_mapping_score']].copy(); meta['pair_mapping_score']=meta[['pre_mapping_score','post_mapping_score']].min(axis=1)
x=f.merge(meta,on=key,how='left',validate='one_to_one')
if a.policy=='best_mapping':
    x=x.sort_values(['patient_id','pair_mapping_score','series_uid'],ascending=[True,False,True]).drop_duplicates('patient_id')
else:
    numeric=x.select_dtypes('number').columns.difference(['patient_id'])
    first=[c for c in ['split'] if c in x.columns]
    agg={c:'mean' for c in numeric}; agg.update({c:'first' for c in first}); x=x.groupby('patient_id',as_index=False).agg(agg)
if x.patient_id.duplicated().any(): raise RuntimeError('Aggregation did not produce unique patient_id')
x.to_csv(a.out,index=False); print('saved',a.out,'patients',x.patient_id.nunique(),'policy',a.policy)

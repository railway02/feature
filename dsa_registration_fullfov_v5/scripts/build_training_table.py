#!/usr/bin/env python
"""Merge patient-level registration features with authoritative labels without changing split/fold."""
import argparse
import pandas as pd
p=argparse.ArgumentParser(); p.add_argument('--features',required=True); p.add_argument('--labels',required=True); p.add_argument('--out',required=True); p.add_argument('--id-col',default='patient_id'); a=p.parse_args()
f=pd.read_csv(a.features); y=pd.read_csv(a.labels)
if a.id_col not in f or a.id_col not in y: raise ValueError('patient_id column missing')
if y[a.id_col].duplicated().any(): raise ValueError('labels table must have unique patient_id')
if f[a.id_col].duplicated().any(): raise ValueError('features table is not patient-level; aggregate explicitly first or train at series level consistently')
x=y.merge(f,on=a.id_col,how='left',validate='one_to_one',suffixes=('','_reg'))
if 'split' in y.columns and 'split_reg' in x.columns:
    bad=x['split_reg'].notna() & (x['split'].astype(str)!=x['split_reg'].astype(str))
    if bad.any(): raise ValueError(f'split mismatch after merge for {int(bad.sum())} patients')
x.to_csv(a.out,index=False); print('saved',a.out,'rows',len(x),'missing_reg',int(x['q_reg'].isna().sum()) if 'q_reg' in x else 'NA')

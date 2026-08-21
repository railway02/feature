#!/usr/bin/env python
import argparse
import pandas as pd

p=argparse.ArgumentParser()
p.add_argument('--reference',required=True,help='Authoritative PredROI fold/split table')
p.add_argument('--candidate',action='append',required=True,help='Registration/CAVE table; repeatable')
p.add_argument('--id-col',default='patient_id'); p.add_argument('--fold-col',default='fold'); p.add_argument('--split-col',default='split')
a=p.parse_args()
ref=pd.read_csv(a.reference)
keys=[a.id_col]
for col in [a.fold_col,a.split_col]:
    if col in ref.columns: keys.append(col)
if ref[a.id_col].duplicated().any(): raise ValueError('reference is not patient-level unique')
refkey=ref[keys].sort_values(a.id_col).reset_index(drop=True)
for path in a.candidate:
    x=pd.read_csv(path)
    if x[a.id_col].duplicated().any(): raise ValueError(f'{path}: duplicate patient_id')
    missing=[c for c in keys if c not in x.columns]
    if missing: raise ValueError(f'{path}: missing alignment columns {missing}')
    xkey=x[keys].sort_values(a.id_col).reset_index(drop=True)
    if not refkey.equals(xkey):
        z=refkey.merge(xkey,on=a.id_col,how='outer',suffixes=('_ref','_candidate'),indicator=True)
        raise ValueError(f'{path}: patient/fold/split mismatch; examples={z[z._merge!="both"].head(10).to_dict("records")}')
    print('ALIGNMENT PASS',path,'patients',len(xkey),'keys',keys)


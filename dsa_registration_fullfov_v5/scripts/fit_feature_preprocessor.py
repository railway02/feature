#!/usr/bin/env python
"""Fit median-imputation + z-score parameters on Train only and export finite z_reg matrices."""
import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from dsa_reg.feature_preprocess import TrainFeaturePreprocessor


def resolve_columns(groups, name, stack=None):
    stack=stack or []
    if name in stack: raise ValueError('Cyclic feature-group inheritance: '+str(stack+[name]))
    spec=groups[name]
    cols=[]
    parent=spec.get('inherit') if isinstance(spec,dict) else None
    if parent:
        cols.extend(resolve_columns(groups,parent,stack+[name]))
    if isinstance(spec,dict):
        cols.extend(spec.get('columns',[]))
    else:
        cols.extend(spec)
    # ordered unique
    return list(dict.fromkeys(cols))

p=argparse.ArgumentParser()
p.add_argument('--table',required=True)
p.add_argument('--groups',required=True)
p.add_argument('--experiment',required=True)
p.add_argument('--out-dir',required=True)
p.add_argument('--split-col',default='split')
p.add_argument('--train-value',default='Train')
p.add_argument('--add-missing-indicators',action='store_true')
a=p.parse_args()

df=pd.read_csv(a.table); groups=yaml.safe_load(Path(a.groups).read_text())
if a.experiment not in groups: raise KeyError(f'Unknown experiment {a.experiment}; choices={list(groups)}')
cols=resolve_columns(groups,a.experiment)
missing=[c for c in cols if c not in df.columns]
if missing: raise KeyError(f'Requested registration features missing from table: {missing}')
if not (30 <= len(cols) <= 80) and a.experiment != 'exp3168':
    raise ValueError(f'{a.experiment} raw feature count={len(cols)} outside requested 30-80 range')
train=df[df[a.split_col].astype(str)==str(a.train_value)]
if len(train)==0: raise RuntimeError('No Train rows found')
proc=TrainFeaturePreprocessor.fit(train,cols,add_missing_indicators=a.add_missing_indicators)
out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); proc.save(out/'feature_preprocessor.json')
X,names=proc.transform(df)
if a.experiment != 'exp3168' and not (30 <= X.shape[1] <= 80):
    raise ValueError(f'{a.experiment} final z_reg dim={X.shape[1]} outside requested 30-80 range')
np.save(out/'zreg.npy',X)
idx={'row_index':np.arange(len(df)), 'patient_id':df['patient_id'].values}
if 'series_uid' in df: idx['series_uid']=df['series_uid'].astype(str).values
pd.DataFrame(idx).to_csv(out/'zreg_index.csv',index=False)
Path(out/'zreg_feature_names.json').write_text(json.dumps(names,indent=2))
print('selected raw features',len(cols),'output dim',X.shape[1],'rows',X.shape[0])

#!/usr/bin/env python
"""Recompute only tau-dependent canonical log-Jacobian features from saved change_maps."""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.ndimage import label
import pandas as pd

p=argparse.ArgumentParser(); p.add_argument('--output-root',required=True); p.add_argument('--tau-json',required=True); p.add_argument('--out-csv',required=True); a=p.parse_args()
tau=float(json.loads(Path(a.tau_json).read_text())['tau']); rows=[]
for feat_path in Path(a.output_root).glob('*/*/*/features.json'):
    case=feat_path.parent; maps=case/'change_maps.npz'
    if not maps.exists(): continue
    feat=json.loads(feat_path.read_text()); z=np.load(maps); lj=z['canonical_logjac']
    valid=z['canonical_valid'].astype(bool) if 'canonical_valid' in z else np.isfinite(lj)
    roi=z['roi'].astype(bool) if 'roi' in z else np.ones_like(lj,dtype=bool)
    for name in ['lesion','boundary','peri','stable','roi']:
        if name not in z and name!='roi': continue
        m=roi if name=='roi' else z[name].astype(bool)
        vals=lj[m & valid & np.isfinite(lj)]
        feat[f'logjac_{name}_expansion_ratio'] = float(np.mean(vals>tau)) if vals.size else np.nan
        feat[f'logjac_{name}_contraction_ratio'] = float(np.mean(vals<-tau)) if vals.size else np.nan
    for key,mask in [('expansion',roi&valid&np.isfinite(lj)&(lj>tau)),('contraction',roi&valid&np.isfinite(lj)&(lj<-tau))]:
        lab,n=label(mask); largest=int(np.bincount(lab.ravel())[1:].max()) if n else 0; denom=int(np.sum(roi&valid))
        feat[f'logjac_largest_{key}_area']=float(largest)
        feat[f'logjac_largest_{key}_ratio']=float(largest/denom) if denom else np.nan
    feat['expansion_tau_locked']=tau; rows.append(feat)
pd.DataFrame(rows).to_csv(a.out_csv,index=False); print('saved',a.out_csv,'rows',len(rows),'tau',tau)

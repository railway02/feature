#!/usr/bin/env python
"""Estimate tau from successful Train stable-vessel |canonical logJ| only.

The clinical threshold is a registration-noise quantity.  Invalid registrations must
never contribute: their deformation fields are deliberately retained for audit but are
not evidence of stable-vessel noise.  This script therefore requires the per-series
``features.json`` to declare ``registration_valid == 1`` by default.
"""
import argparse, json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--output-root',required=True)
p.add_argument('--split',default='Train')
p.add_argument('--quantile',type=float,default=0.95)
p.add_argument('--max-per-series',type=int,default=20000)
p.add_argument('--seed',type=int,default=3407)
p.add_argument('--out',required=True)
p.add_argument('--include-invalid',action='store_true',
               help='Exploratory only; violates the locked clinical Train-success-only tau policy.')
a=p.parse_args()
if str(a.split) != 'Train' and not a.include_invalid:
    raise ValueError('Clinical tau calibration must use split=Train')
rng=np.random.default_rng(a.seed); samples=[]; n_series=0; skipped_invalid=0; skipped_missing_features=0
for f in Path(a.output_root,a.split).glob('*/*/change_maps.npz'):
    feature_path=f.parent/'features.json'
    if not feature_path.exists():
        skipped_missing_features += 1
        continue
    try:
        feature=json.loads(feature_path.read_text())
    except Exception as e:
        raise RuntimeError(f'Cannot read {feature_path}: {e}') from e
    if not a.include_invalid and int(feature.get('registration_valid', 0)) != 1:
        skipped_invalid += 1
        continue
    z=np.load(f)
    if 'canonical_logjac' not in z or 'stable' not in z:
        continue
    lj=z['canonical_logjac']; stable=z['stable'].astype(bool)
    valid=z['canonical_valid'].astype(bool) if 'canonical_valid' in z else np.isfinite(lj)
    vals=np.abs(lj[stable & valid & np.isfinite(lj)])
    if vals.size==0: continue
    if vals.size>a.max_per_series:
        vals=rng.choice(vals,size=a.max_per_series,replace=False)
    samples.append(vals.astype(np.float32)); n_series+=1
if not samples: raise RuntimeError('No Train stable-vessel canonical log-Jacobian samples found')
allv=np.concatenate(samples); tau=float(np.quantile(allv,a.quantile))
out={'tau':tau,'quantile':a.quantile,'n_series':n_series,'n_sampled_pixels':int(allv.size),
     'split':a.split,'seed':a.seed,'registration_valid_only':not a.include_invalid,
     'skipped_invalid_series':int(skipped_invalid),'skipped_missing_features':int(skipped_missing_features)}
Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))

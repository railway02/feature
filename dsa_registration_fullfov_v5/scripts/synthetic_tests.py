#!/usr/bin/env python
import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))
"""Fast mathematical tests plus optional backend direction tests."""
import tempfile
from pathlib import Path
import numpy as np
from dsa_reg.features import pure_numpy_jacobian

h=w=96
u=np.zeros((h,w,2),np.float32); u[...,0]=3.25; u[...,1]=-4.5
j=pure_numpy_jacobian(u)
assert np.max(np.abs(j-1)) < 1e-5
print('PASS pure translation: J≈1')

s=1.25
y,x=np.mgrid[0:h,0:w].astype(np.float32); cy=(h-1)/2; cx=(w-1)/2
u=np.zeros((h,w,2),np.float32); u[...,0]=(s-1)*(y-cy); u[...,1]=(s-1)*(x-cx)
j=pure_numpy_jacobian(u)
assert np.allclose(np.median(j[3:-3,3:-3]),s*s,rtol=1e-4,atol=1e-4)
assert np.median(np.log(j[3:-3,3:-3])) > 0
print('PASS canonical expansion: J=s²>1 and logJ>0')

s=0.80
u=np.zeros((h,w,2),np.float32); u[...,0]=(s-1)*(y-cy); u[...,1]=(s-1)*(x-cx)
j=pure_numpy_jacobian(u)
assert np.allclose(np.median(j[3:-3,3:-3]),s*s,rtol=1e-4,atol=1e-4)
assert np.median(np.log(j[3:-3,3:-3])) < 0
print('PASS canonical contraction: 0<J=s²<1 and logJ<0')

try:
    import ants
    from dsa_reg.registration_ants import _canonical_inverse_warp_jacobian
    for s in (1.20,0.82):
        q=1/s
        inv=np.zeros((h,w,2),np.float32); inv[...,0]=(s-1)*(y-cy); inv[...,1]=(s-1)*(x-cx)
        fwd=np.zeros((h,w,2),np.float32); fwd[...,0]=(q-1)*(y-cy); fwd[...,1]=(q-1)*(x-cx)
        mi=ants.from_numpy(np.zeros((h,w),np.float32)); fi=ants.from_numpy(np.zeros((h,w),np.float32))
        with tempfile.TemporaryDirectory() as td:
            ip=str(Path(td)/'InverseWarp.nii.gz'); fp=str(Path(td)/'Warp.nii.gz')
            ants.image_write(ants.from_numpy(inv,has_components=True),ip)
            ants.image_write(ants.from_numpy(fwd,has_components=True),fp)
            o=_canonical_inverse_warp_jacobian(mi,fi,ip,fp,[fp],[ip],geom=True)
        med=float(np.nanmedian(o['canonical_logjac'][8:-8,8:-8]))
        assert np.sign(med)==np.sign(np.log(s*s)) and np.isclose(med,np.log(s*s),rtol=.08,atol=.03)
    print('PASS ANTs backend: canonical expansion positive, contraction negative')
except ImportError:
    print('SKIP ANTs backend test (antspyx not installed here)')

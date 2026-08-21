import numpy as np
from dsa_reg.features import pure_numpy_jacobian

def test_translation_jacobian_one():
    u=np.zeros((32,32,2),np.float32); u[...,0]=3; u[...,1]=-2
    j=pure_numpy_jacobian(u)
    assert np.allclose(j,1)

def test_uniform_expansion():
    s=1.1; y,x=np.mgrid[0:32,0:32].astype(np.float32); c=15.5
    u=np.stack([(s-1)*(y-c),(s-1)*(x-c)],axis=-1)
    j=pure_numpy_jacobian(u)
    assert np.allclose(np.median(j),s*s,rtol=1e-5)

def test_uniform_contraction():
    s=0.8; y,x=np.mgrid[0:32,0:32].astype(np.float32); c=15.5
    u=np.stack([(s-1)*(y-c),(s-1)*(x-c)],axis=-1)
    j=pure_numpy_jacobian(u)
    assert 0 < np.median(j) < 1
    assert np.allclose(np.median(j),s*s,rtol=1e-5)

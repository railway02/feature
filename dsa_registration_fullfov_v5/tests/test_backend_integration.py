import tempfile
from pathlib import Path
import numpy as np
import pytest
import cv2
from scipy.ndimage import gaussian_filter


def test_simpleitk_canonical_effective_offset_if_available():
    sitk=pytest.importorskip('SimpleITK')
    from dsa_reg.registration_sitk import canonical_parameters
    t=sitk.Euler2DTransform(); t.SetCenter((50.0,40.0)); t.SetAngle(0.2); t.SetTranslation((3.0,-4.0))
    p=canonical_parameters(t,'rigid')
    inv=t.GetInverse(); expected=inv.TransformPoint((0.0,0.0))
    assert np.allclose([p['tx'],p['ty']],expected,atol=1e-8)


@pytest.mark.parametrize("s", [1.20, 0.82])
def test_ants_canonical_inverse_warp_direction_if_available(s):
    ants=pytest.importorskip('ants')
    from dsa_reg.registration_ants import _canonical_inverse_warp_jacobian
    h=w=96
    y,x=np.mgrid[0:h,0:w].astype(np.float32); cy=(h-1)/2; cx=(w-1)/2
    # ANTs vector components follow NumPy spatial-axis order for from_numpy fields:
    # component 0 displaces axis 0/y, component 1 displaces axis 1/x.
    inv=np.zeros((h,w,2),np.float32)   # moving -> fixed, uniform scale by s
    inv[...,0]=(s-1)*(y-cy); inv[...,1]=(s-1)*(x-cx)
    fwd=np.zeros((h,w,2),np.float32)   # fixed -> moving, inverse scale
    q=1.0/s
    fwd[...,0]=(q-1)*(y-cy); fwd[...,1]=(q-1)*(x-cx)
    mi=ants.from_numpy(np.zeros((h,w),np.float32)); fi=ants.from_numpy(np.zeros((h,w),np.float32))
    inv_img=ants.from_numpy(inv,has_components=True); fwd_img=ants.from_numpy(fwd,has_components=True)
    with tempfile.TemporaryDirectory() as td:
        invp=str(Path(td)/'InverseWarp.nii.gz'); fwdp=str(Path(td)/'Warp.nii.gz')
        ants.image_write(inv_img,invp); ants.image_write(fwd_img,fwdp)
        out=_canonical_inverse_warp_jacobian(mi,fi,invp,fwdp,[fwdp],[invp],geom=True)
    med=float(np.nanmedian(out['canonical_logjac'][8:-8,8:-8]))
    assert np.sign(med) == np.sign(np.log(s*s))
    assert np.isclose(med,np.log(s*s),rtol=0.08,atol=0.03)
    assert out['inverse_consistency_logjac_mae'] < 0.05


def test_ants_translation_field_has_unit_jacobian_if_available():
    ants=pytest.importorskip('ants')
    from dsa_reg.registration_ants import _canonical_inverse_warp_jacobian
    h=w=64
    inv=np.zeros((h,w,2),np.float32); inv[...,0]=3.0; inv[...,1]=-4.0
    fwd=-inv
    mi=ants.from_numpy(np.zeros((h,w),np.float32)); fi=ants.from_numpy(np.zeros((h,w),np.float32))
    with tempfile.TemporaryDirectory() as td:
        invp=str(Path(td)/'InverseWarp.nii.gz'); fwdp=str(Path(td)/'Warp.nii.gz')
        ants.image_write(ants.from_numpy(inv,has_components=True),invp)
        ants.image_write(ants.from_numpy(fwd,has_components=True),fwdp)
        out=_canonical_inverse_warp_jacobian(mi,fi,invp,fwdp,[fwdp],[invp],geom=True)
    assert np.allclose(np.nanmedian(out['canonical_jac_moving'][6:-6,6:-6]),1.0,atol=1e-5)


def test_actual_ants_registration_small_to_large_is_positive_in_canonical_direction():
    pytest.importorskip('ants')
    from dsa_reg.registration_ants import run_syn_residual
    h=w=80
    def image(radius):
        x=np.zeros((h,w),np.float32)
        cv2.line(x,(8,48),(72,48),0.6,2)
        cv2.line(x,(40,48),(58,20),0.5,2)
        cv2.circle(x,(40,39),radius,1.0,-1)
        return gaussian_filter(x,1.0)
    moving=image(6); fixed=image(10)
    moving_lesion=np.zeros((h,w),np.uint8); cv2.circle(moving_lesion,(40,39),6,1,-1)
    fixed_lesion=np.zeros((h,w),np.uint8); cv2.circle(fixed_lesion,(40,39),10,1,-1)
    cfg={
        'transform':'SyNOnly','metric':'CC','syn_sampling':2,
        'reg_iterations':[40,20,10],'grad_step':0.1,'flow_sigma':3,'total_sigma':1,
        'singleprecision':True,'use_geometric_jacobian':True,
        'lesion_metric':'MeanSquares','lesion_metric_weight':0.15,
    }
    with tempfile.TemporaryDirectory() as td:
        out=run_syn_residual(
            fixed,moving,np.ones((h,w),bool),np.ones((h,w),bool),str(Path(td)/'syn_'),cfg,
            fixed_lesion.astype(bool),moving_lesion.astype(bool),
        )
    med=float(np.nanmedian(out['canonical_logjac_moving'][moving_lesion.astype(bool)]))
    assert med > 0.10
    assert out['inverse_consistency_logjac_mae'] < 0.05
    assert out['residual_identity_linear_transforms']

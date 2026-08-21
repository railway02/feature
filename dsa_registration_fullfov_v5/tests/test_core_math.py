import numpy as np
import pandas as pd

from dsa_reg.features import pure_numpy_jacobian, morphology_features
from dsa_reg.hemodynamics import curve_features, normalized_phase_features
from dsa_reg.regions import measurement_regions, build_anchor
from dsa_reg.feature_preprocess import TrainFeaturePreprocessor
from dsa_reg.outcomes import derive_abs_rel_outcomes
from dsa_reg.pipeline import resolve_expansion_tau
from dsa_reg.qc import vascular_ncc
from dsa_reg.preprocessing import suppress_linear_border_artifacts


def test_jacobian_translation_and_expansion():
    h=w=96
    u=np.zeros((h,w,2),np.float32); u[...,0]=3.25; u[...,1]=-4.5
    j=pure_numpy_jacobian(u)
    assert np.max(np.abs(j-1)) < 1e-5
    s=1.25
    y,x=np.mgrid[0:h,0:w].astype(np.float32); cy=(h-1)/2; cx=(w-1)/2
    u[...,0]=(s-1)*(y-cy); u[...,1]=(s-1)*(x-cx)
    j=pure_numpy_jacobian(u)
    assert np.allclose(np.median(j[3:-3,3:-3]),s*s,rtol=1e-4,atol=1e-4)


def test_hemodynamic_features_have_peak_normalized_outputs():
    c=np.array([0,0,1,3,5,3,1,0],dtype=np.float32)
    f=curve_features(c,frame_interval_seconds=0.5,arrival_fraction=0.1)
    assert f['peak']==5
    assert f['ttp']==2.0
    assert f['ttp_from_arrival'] > 0
    assert np.isfinite(f['auc_peaknorm'])
    assert f['washin_peaknorm'] > 0
    assert f['washout_peaknorm'] < 0


def test_normalized_phase_features_are_dimensionless_and_resampled():
    # Same contrast shape sampled at different frame densities must retain comparable
    # unit-phase descriptors; raw TTP in frame units intentionally differs.
    coarse = np.array([0, 0, 1, 4, 5, 3, 1, 0], np.float32)
    dense = np.interp(np.linspace(0, 7, 29), np.arange(8), coarse).astype(np.float32)
    a, ac, am = normalized_phase_features(coarse, n_samples=32)
    b, bc, bm = normalized_phase_features(dense, n_samples=32)
    assert am['valid'] and bm['valid'] and ac.shape == bc.shape == (32,)
    assert 0 <= a['norm_ttp'] <= 1 and 0 <= b['norm_ttp'] <= 1
    assert np.isclose(a['norm_ttp'], b['norm_ttp'], atol=.08)
    assert np.isfinite(a['norm_auc']) and np.isfinite(b['norm_auc'])


def test_measurement_roi_not_entire_padded_crop():
    m=np.zeros((128,128),bool); m[60:68,60:68]=1
    v=np.zeros_like(m); v[62,20:108]=1
    valid=np.ones_like(m)
    r=measurement_regions(m,m,v,v,valid,roi_margin=20,vessel_roi_dilate=2)
    assert r['roi'].sum() < m.size
    assert r['lesion'].sum()==m.sum()


def test_local_vascular_anchor_rejects_remote_fragment_without_lesion_center_alignment():
    lesion = np.zeros((128, 128), bool); lesion[60:68, 60:68] = 1
    vessel = np.zeros_like(lesion); vessel[64, 35:92] = 1; vessel[5, 5:25] = 1
    anchor = build_anchor(vessel, lesion, exclusion_px=4, max_distance_px=35)
    assert not anchor[5, 10]
    assert anchor[64, 40] and anchor[64, 88]


def test_morphology_area_direction():
    pre=np.zeros((64,64),bool); pre[20:30,20:30]=1
    post=np.zeros((64,64),bool); post[18:32,18:32]=1
    f=morphology_features(pre,post)
    assert f['morph_area_delta'] > 0
    assert f['morph_area_logratio'] > 0
    assert f['morph_boundary_p90'] > 0


def test_train_feature_preprocessor_finite_and_train_only_stats():
    df=pd.DataFrame({'x':[1.0,np.nan,100.0,200.0],'y':[2.0,3.0,np.inf,5.0]})
    p=TrainFeaturePreprocessor.fit(df.iloc[:2],['x','y'],add_missing_indicators=True)
    X,names=p.transform(df)
    assert X.shape==(4,3)
    assert np.isfinite(X).all()
    assert 'x__missing' in names
    assert 'y__missing' not in names


def test_absolute_and_relative_outcome_definitions():
    df=pd.DataFrame({'post':[3,2,1,2],'follow':[2,2,2,2],'enlarged':[False,False,False,True]})
    out=derive_abs_rel_outcomes(df,'post','follow','enlarged')
    assert out.y_abs.tolist()==[1.0,1.0,1.0,1.0]
    assert out.y_rel.tolist()==[0.0,0.0,1.0,1.0]


def test_tau_artifact_must_be_train_calibrated(tmp_path):
    p=tmp_path/'tau.json'; p.write_text('{"tau": 0.073, "split": "Train"}')
    tau,source=resolve_expansion_tau({'features':{'expansion_tau_artifact':str(p),'expansion_tau_fallback':0.05}})
    assert tau==0.073 and source==str(p)
    p.write_text('{"tau": 0.073, "split": "Valid"}')
    import pytest
    with pytest.raises(ValueError):
        resolve_expansion_tau({'features':{'expansion_tau_artifact':str(p),'expansion_tau_fallback':0.05}})


def test_smoothed_vascular_ncc_rewards_subpixel_like_vessel_overlap():
    a = np.zeros((64, 64), bool); b = np.zeros_like(a)
    a[30, 8:56] = 1; b[31, 8:56] = 1
    assert vascular_ncc(a, b, np.ones_like(a), sigma_px=2) > .5


def test_border_artifact_suppression_keeps_central_vessel_tree():
    m = np.zeros((128, 128), bool)
    m[10:115, 8:10] = 1       # export border bar
    m[30:105, 64:67] = 1      # central parent vessel
    m[35:38, 45:85] = 1       # central branch
    clean, removed = suppress_linear_border_artifacts(m, .15, .25, .03)
    assert not np.any(clean[:, 8:10])
    assert np.any(clean[:, 64:67]) and np.any(clean[35:38, 45:85])
    assert len(removed) == 1 and removed[0]["orientation"] == "vertical"

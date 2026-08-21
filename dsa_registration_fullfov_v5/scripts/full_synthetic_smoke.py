#!/usr/bin/env python
import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))
"""End-to-end smoke test with generated 2-D DSA-like data.

Requires SimpleITK + antspyx. It is intended to be run on the user's actual server
before touching the real cohort. It verifies that manifest parsing, ROI crop, intra-rigid,
Similarity, SyNOnly, canonical Jacobian extraction, feature JSON and change_maps all execute.
"""
import json, tempfile
from pathlib import Path
import cv2, numpy as np, pandas as pd, yaml
from scipy.ndimage import gaussian_filter
from dsa_reg.pipeline import process_series

try:
    import SimpleITK, ants  # noqa
except ImportError as e:
    raise SystemExit('Install requirements.txt first: '+repr(e))


def draw_base(h=128,w=128,lesion_r=10):
    img=np.zeros((h,w),np.float32)
    # asymmetric parent-vessel tree
    cv2.line(img,(16,72),(112,72),1.0,3)
    cv2.line(img,(64,72),(90,35),.9,2)
    cv2.line(img,(78,72),(105,98),.8,2)
    cv2.circle(img,(64,62),lesion_r,1.0,-1)
    return gaussian_filter(img,1.0)


def warp_affine(img,angle_deg=0,scale=1,tx=0,ty=0):
    h,w=img.shape; M=cv2.getRotationMatrix2D((w/2,h/2),angle_deg,scale); M[:,2]+=np.array([tx,ty])
    return cv2.warpAffine(img,M,(w,h),flags=cv2.INTER_LINEAR,borderValue=0)


def seq_from(base, n=11, motion_sign=1):
    amps=np.array([0,.05,.15,.4,.75,1,.8,.55,.3,.12,.03],np.float32)[:n]
    out=[]
    for i,a in enumerate(amps):
        dx=motion_sign*(1 if i>6 else 0); dy=motion_sign*(-1 if i>8 else 0)
        f=warp_affine(base*a,tx=dx,ty=dy)
        noise=np.random.default_rng(100+i).normal(0,.01,f.shape).astype(np.float32)
        out.append(np.clip(f+noise,0,1))
    return np.stack(out)

with tempfile.TemporaryDirectory() as td:
    root=Path(td); (root/'frames/pre').mkdir(parents=True); (root/'frames/post').mkdir(parents=True); (root/'2d').mkdir()
    pre_base=draw_base(lesion_r=9)
    # Post differs in global pose/scale and has a modest local lesion expansion.
    post_base=warp_affine(draw_base(lesion_r=12),angle_deg=2.0,scale=1.03,tx=2,ty=-2)
    pre_seq=seq_from(pre_base); post_seq=seq_from(post_base,motion_sign=-1)
    pre_paths=[]; post_paths=[]
    for i,f in enumerate(pre_seq,3):
        p=root/f'frames/pre/IMG-0001-{i:05d}.jpg'; cv2.imwrite(str(p),(f*255).astype(np.uint8)); pre_paths.append(str(p))
    for i,f in enumerate(post_seq,4):
        p=root/f'frames/post/IMG-0002-{i:05d}.jpg'; cv2.imwrite(str(p),(f*255).astype(np.uint8)); post_paths.append(str(p))
    # Reference PNGs need not be exactly temporal frames; use a small peak-window mean.
    pre_ref=np.mean(pre_seq[4:7],axis=0); post_ref=np.mean(post_seq[4:7],axis=0)
    pre_ref_p=root/'2d/pre.png'; post_ref_p=root/'2d/post.png'
    cv2.imwrite(str(pre_ref_p),(pre_ref*255).astype(np.uint8)); cv2.imwrite(str(post_ref_p),(post_ref*255).astype(np.uint8))
    pre_m=np.zeros((128,128),np.uint8); cv2.circle(pre_m,(64,62),9,255,-1)
    post_m=np.zeros((128,128),np.uint8); cv2.circle(post_m,(66,60),12,255,-1)
    pre_mp=root/'2d/pre_mask.png'; post_mp=root/'2d/post_mask.png'; cv2.imwrite(str(pre_mp),pre_m); cv2.imwrite(str(post_mp),post_m)
    inds_pre=list(range(3,3+len(pre_paths))); inds_post=list(range(4,4+len(post_paths)))
    row=pd.Series({
        'split':'Train','patient_id':999001,'series_uid':'synthetic','series_id':'main',
        'pre_reference_image_path':str(pre_ref_p),'post_reference_image_path':str(post_ref_p),
        'pre_mask_path':str(pre_mp),'post_mask_path':str(post_mp),
        'pre_frame_paths':'|'.join(pre_paths),'post_frame_paths':'|'.join(post_paths),
        'pre_n_frames':len(pre_paths),'post_n_frames':len(post_paths),
        'pre_frozen_temporal_blocks_json':json.dumps([{'indices':inds_pre,'view_indices':{'contrast_core20':inds_pre}}]),
        'post_frozen_temporal_blocks_json':json.dumps([{'indices':inds_post,'view_indices':{'contrast_core20':inds_post}}]),
        # This synthetic fixture explicitly establishes its reference/mask identity
        # geometry.  Production code must not assume this for an unlabelled fixture.
        'pre_mapping_method':'synthetic_identity_verified','post_mapping_method':'synthetic_identity_verified',
        'pre_mapping_score':1.0,'post_mapping_score':1.0,
    })
    cfg=yaml.safe_load((Path(__file__).parents[1]/'config/default.yaml').read_text())
    cfg['paths']['output_root']=str(root/'out'); cfg['paths']['remap']={}
    cfg['geometry']['canvas_size']=[128,128]; cfg['roi']['size']=[128,128]
    cfg['intra_registration']['iterations']=60; cfg['global_registration']['iterations']=100
    cfg['global_registration']['run_methods']=['rigid','similarity']; cfg['nonrigid']['reg_iterations']=[30,20,10]
    feat=process_series(row,cfg)
    assert Path(root/'out/Train/999001/synthetic/features.json').exists()
    assert Path(root/'out/Train/999001/synthetic/change_maps.npz').exists()
    assert np.isfinite(feat['global_similarity_ncc'])
    assert 'q_reg' in feat and 0 <= feat['q_reg'] <= 1
    print('FULL SYNTHETIC SMOKE PASS')
    print('global similarity NCC:',feat['global_similarity_ncc'])
    print('nonrigid NCC:',feat.get('nonrigid_anchor_ncc'))
    print('q_reg:',feat['q_reg'],'valid:',feat['registration_valid'],feat.get('registration_invalid_reasons',''))

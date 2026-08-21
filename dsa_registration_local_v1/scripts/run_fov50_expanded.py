#!/usr/bin/env python3
"""Outcome-blind expanded-native-FOV technical sensitivity run (50 Train cases)."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path('/root/autodl-tmp/dsa_registration_local_reference_v1');sys.path.insert(0,str(ROOT))
import scripts.overnight_supervisor as g0
from dsa_local_reg.common import load_config
from dsa_local_reg.local_geometry import BBox,crop_with_border_median_padding,resize_whole_canvas
from dsa_local_reg.visualization import read_gray
from dsa_local_reg.preprocessing_adapter import load_local_reference_pairs

def roundup32(x): return int(np.ceil(x/32)*32)
def expanded_box(p):
 b=p.expanded_bbox;side=max(b.height,b.width);side=roundup32(max(256,1.5*side));cx=(b.x0+b.x1)/2;cy=(b.y0+b.y1)/2
 return BBox(int(round(cx-side/2)),int(round(cy-side/2)),int(round(cx-side/2))+side,int(round(cy-side/2))+side)
def phase_expanded(p):
 box=expanded_box(p);frames=[crop_with_border_median_padding(read_gray(x),box) for x in p.frame_paths]
 mask=resize_whole_canvas(read_gray(p.mask_path),p.canvas_shape_yx,is_mask=True)>0;cm=crop_with_border_median_padding(mask.astype(np.uint8),box).image>0
 scores=[float(np.mean(g0.norm(f.image)[cm])) if cm.any() else float(np.mean(g0.norm(f.image))) for f in frames];k=int(np.argmax(scores))
 return frames[k].image.astype(np.float32),cm,frames[k].valid_support,k,scores[k]
def corr_stats(a,b):
 from scipy.stats import pearsonr,spearmanr
 keep=np.isfinite(a)&np.isfinite(b)
 if keep.sum()<3:return dict(n=int(keep.sum()),median_abs_difference=np.nan,pearson=np.nan,spearman=np.nan)
 return dict(n=int(keep.sum()),median_abs_difference=float(np.median(np.abs(a[keep]-b[keep]))),pearson=float(pearsonr(a[keep],b[keep]).statistic),spearman=float(spearmanr(a[keep],b[keep]).statistic))
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 # Explicitly prohibit accidental outcome access: this process imports only the
 # selection list, local registration code and G0 QC/features.
 sel=pd.read_csv(a.out/'fov50_series.csv',dtype={'series_uid':str});assert 'target' not in sel
 cfg=load_config(ROOT/'config/default.yaml');pairs={x.series_uid:x for x in load_local_reference_pairs(cfg,split='Train')};items=[pairs[x] for x in sel.series_uid]
 if len(items)!=50:raise RuntimeError('fov50 UID reconciliation failed')
 old=g0.phase_data;g0.phase_data=phase_expanded
 try: rows=g0.run_many(items,a.out,'rigid',True,'expanded',a.workers)
 finally:g0.phase_data=old
 ex=pd.DataFrame(rows);ex.to_csv(a.out/'expanded_registration_features.csv',index=False)
 base=pd.read_csv(ROOT/'outputs/local_reference_v1_20260819_overnight/train_registration_features.csv',dtype={'series_uid':str});base=base[base.series_uid.isin(sel.series_uid)]
 q=base.merge(ex,on='series_uid',suffixes=('_g0','_expanded'),validate='one_to_one');
 qc=['registration_valid','folding_rate','inverse_consistency_logjac_mae','displacement_P95','abs_logJ_P99','metric_gain']
 paired=pd.DataFrame({'series_uid':q.series_uid})
 for c in qc:
  paired[c+'_g0']=q[c+'_g0'];paired[c+'_expanded']=q[c+'_expanded'];paired[c+'_difference']=q[c+'_expanded']-q[c+'_g0']
 paired=paired.merge(sel,on='series_uid',how='left',validate='one_to_one');paired.to_csv(a.out/'paired_metrics.csv',index=False)
 names=g0.DEFORM_NAMES;st=[]
 for name in names:
  z=corr_stats(q[name+'_g0'].to_numpy(float),q[name+'_expanded'].to_numpy(float));st.append({'feature':name,'region':name.split('_')[0] if not name.startswith('peri') else 'peri',**z})
 pd.DataFrame(st).to_csv(a.out/'jacobian_stability.csv',index=False)
 summary={'outcome_used':False,'n':50,'expanded_rule':'max(G0_side, round_up_to_32(max(256,1.5*G0_side)))','registration_valid_g0':int(q.registration_valid_g0.sum()),'registration_valid_expanded':int(q.registration_valid_expanded.sum()),'inverse_consistency_median_difference':float(np.nanmedian(q.inverse_consistency_logjac_mae_expanded-q.inverse_consistency_logjac_mae_g0)),'conclusion':'INCONCLUSIVE','note':'Contact sheets and qualitative review are required before KEEP_G0/RECOMMEND_EXPANDED.'}
 (a.out/'FOV50_TECHNICAL_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');(a.out/'FOV50_TECHNICAL_SUMMARY.md').write_text('# FOV50 technical summary\n\n```json\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n```\n')
if __name__=='__main__':main()

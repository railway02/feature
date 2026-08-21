#!/usr/bin/env python3
"""UID/key/provenance reconciliation for frozen automatic-PredROI features."""
from __future__ import annotations
import argparse,os,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path('/root/autodl-tmp/aneurysm');FB=ROOT/'outputs/api_complete_2d_cave_featurebanks_v1/segresnet';SEG=ROOT/'outputs/api_png2d_spatial_backbones_v6_strict/expanded_strict/segmentation/segresnet'
def key(path):return Path(str(path)).stem
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();tr=pd.read_csv(a.out/'train796_strict_folds.csv',dtype=str);va=pd.read_csv(a.out/'valid211_case_manifest.csv',dtype=str);m=pd.read_csv(FB/'all2d_prepost_manifest.csv',dtype=str)
 z=np.load(FB/'all2d_prepost_features_by_fold.npz',allow_pickle=False)['z_2d_raw_by_fold']
 assert z.shape==(1009,5,1024)
 # Train uses identity-mapped PNG keys; Valid keys are recovered from explicit
 # Stage-A image paths, never from an array row number.
 tr['pre_segmentation_key']=tr.png2d_png_key_pre;tr['post_segmentation_key']=tr.png2d_png_key_post
 va['pre_segmentation_key']=va.pre_image_path.map(key);va['post_segmentation_key']=va.post_image_path.map(key)
 allc=pd.concat([tr.assign(split='Train'),va.assign(split='Valid')],ignore_index=True)
 x=allc.merge(m,left_on=['patient_id','pre_segmentation_key','post_segmentation_key'],right_on=['patient_id','pre_segmentation_key','post_segmentation_key'],how='left',validate='one_to_one',indicator=True)
 x['feature_present']=x['_merge'].eq('both')
 legal={f:pd.read_csv(SEG.parent.parent/f'fold_{f}'/'segmentation_legal_split_manifest.csv',dtype=str) for f in range(1,6)}
 # The odd-looking resolved source is intentional: legal provenance is the
 # frozen segmentation legal manifest, while checkpoints live under SEG.
 rows=[];selected=[]
 for r in x.itertuples(index=False):
  candidates=[]
  for f,d in legal.items():
   patient_held=str(r.patient_id) not in set(d.patient_id.astype(str));pre_held=str(r.pre_segmentation_key) not in set(d.png_key.astype(str));post_held=str(r.post_segmentation_key) not in set(d.png_key.astype(str))
   if patient_held and pre_held and post_held:candidates.append(f)
  f=candidates[0] if candidates else None;has=bool(r.feature_present)
  rows.append({'series_uid':r.series_uid,'patient_id':r.patient_id,'feature_key':f'{r.pre_segmentation_key}|{r.post_segmentation_key}','pre_segmentation_key':r.pre_segmentation_key,'post_segmentation_key':r.post_segmentation_key,'selected_seg_fold':f,'selected_checkpoint':str(SEG/f'fold_{f}'/'model.pt') if f else '', 'pre_was_heldout':bool(f is not None),'post_was_heldout':bool(f is not None),'patient_was_heldout':bool(f is not None),'feature_source':'existing_featurebank' if has and f else ('unresolved_no_heldout_frozen_model' if not f else 'reextract_required'),'strict_oof':bool(has and f),'featurebank_row_index':getattr(r,'pre_phase_row_index',np.nan)})
  if has and f:selected.append((r.series_uid,f))
 out=pd.DataFrame(rows);out.to_csv(a.out/'segresnet_feature_reconciliation.csv',index=False)
 strict=out[out.strict_oof].copy(); unresolved=out[(out.split if 'split' in out else out.series_uid.str.startswith('Train'))] if False else out[~out.strict_oof]
 # Feature array row is taken from the manifest's pair order after the verified
 # key join, not assumed to match cohort order.
 lookup={(str(r.patient_id),str(r.pre_segmentation_key),str(r.post_segmentation_key)):i for i,r in enumerate(m.itertuples(index=False))}
 trainz=[];validz=[]
 for r in out.itertuples(index=False):
  i=lookup.get((str(r.patient_id),str(r.pre_segmentation_key),str(r.post_segmentation_key)))
  if str(r.series_uid).startswith('Train__') and r.strict_oof: trainz.append((r.series_uid,z[i,int(r.selected_seg_fold)-1]))
  if str(r.series_uid).startswith('Valid__') and i is not None:validz.append((r.series_uid,z[i]))
 if len(trainz):np.savez_compressed(a.out/'train_strict_oof_available_only.npz',series_uid=np.array([v[0] for v in trainz]),z2d=np.stack([v[1] for v in trainz]))
 if len(validz)==211:np.savez_compressed(a.out/'valid211_predroi_z2d_by_fold.npz',series_uid=np.array([v[0] for v in validz]),z2d_by_fold=np.stack([v[1] for v in validz]))
 summary={'Train796':{'strict_oof':int(sum(out.series_uid.str.startswith('Train__')&out.strict_oof)),'existing_feature':int(sum(out.series_uid.str.startswith('Train__')&out.strict_oof)),'reextracted_feature':0,'unresolved':int(sum(out.series_uid.str.startswith('Train__')&~out.strict_oof))},'Valid211':{'existing_feature':int(sum(out.series_uid.str.startswith('Valid__')&out.feature_source.ne('reextract_required'))),'unresolved':int(sum(out.series_uid.str.startswith('Valid__')&out.feature_source.eq('reextract_required')))}}
 (a.out/'SEGRESNET_RECONCILIATION_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()

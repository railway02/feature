#!/usr/bin/env python3
"""Join complete phase banks only through CAVE png_key -> all2D segmentation_key."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from loader import DEFAULT_OUTPUT_ROOT, no_label_fields, write_json, write_npz

def read(path):
    with np.load(path,allow_pickle=False) as z:
        no_label_fields(set(z.files),str(path));return {k:np.asarray(z[k]) for k in z.files}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT_ROOT);args=ap.parse_args();root=args.output_root
    s=read(root/'segresnet/all2d_phase_features_by_fold.npz');c=read(root/'cave/cave_phase_features.npz');cp=read(root/'cave/cave_prepost_features.npz')
    sm=pd.DataFrame({'all2d_phase_row_index':np.arange(len(s['segmentation_key'])),'segmentation_key':s['segmentation_key'].astype(str),'all2d_patient_id':s['patient_id'].astype(str),'all2d_phase':s['phase'].astype(str)})
    cm=pd.DataFrame({'cave_phase_row_index':np.arange(len(c['phase_uid'])),'phase_uid':c['phase_uid'].astype(str),'split':c['split'].astype(str),'patient_id':c['patient_id'].astype(str),'cave_series_uid':c['series_uid'].astype(str),'phase':c['phase'].astype(str),'png_key':c['png_key'].astype(str)})
    mapped=cm.merge(sm,left_on='png_key',right_on='segmentation_key',how='left',validate='one_to_one')
    mapped['cave_feature_available']=True;mapped['segresnet_feature_available']=mapped.all2d_phase_row_index.notna();mapped['mapping_status']=np.where(~mapped.segresnet_feature_available,'missing_all2d_png_key',np.where((mapped.patient_id==mapped.all2d_patient_id)&(mapped.phase==mapped.all2d_phase),'mapped_exact_png_key_patient_phase','mapped_key_but_patient_or_phase_mismatch'))
    if not mapped.segresnet_feature_available.all() or not (mapped.mapping_status=='mapped_exact_png_key_patient_phase').all():raise AssertionError('CAVE successful phase does not exactly map to all2d')
    out=root/'cross_modal';out.mkdir(parents=True,exist_ok=True);mapped.to_csv(out/'cave_to_all2d_phase_mapping.csv',index=False,encoding='utf-8')
    cpframe=pd.DataFrame({'split':cp['split'].astype(str),'series_uid':cp['series_uid'].astype(str),'patient_id':cp['patient_id'].astype(str),'pre_cave_phase_row_index':cp['pre_phase_row_index'].astype(int),'post_cave_phase_row_index':cp['post_phase_row_index'].astype(int)})
    lookup=mapped.set_index('cave_phase_row_index').all2d_phase_row_index.astype(int).to_dict();cpframe['pre_all2d_phase_row_index']=cpframe.pre_cave_phase_row_index.map(lookup);cpframe['post_all2d_phase_row_index']=cpframe.post_cave_phase_row_index.map(lookup)
    complete=cpframe.dropna().copy();incomplete=cpframe[cpframe.isna().any(axis=1)].copy();complete[['pre_all2d_phase_row_index','post_all2d_phase_row_index']]=complete[['pre_all2d_phase_row_index','post_all2d_phase_row_index']].astype(int)
    pre=complete.pre_all2d_phase_row_index.to_numpy();post=complete.post_all2d_phase_row_index.to_numpy(); z2=np.concatenate([s['global_by_fold'][pre],s['pred_roi_by_fold'][pre],s['global_by_fold'][post],s['pred_roi_by_fold'][post]],axis=-1).astype(np.float32);zt=cp['z_time_raw'][complete.index].astype(np.float32)
    if len(complete)!=992:raise AssertionError(f'expected 992 complete mapped series, got {len(complete)}')
    u=lambda x:np.asarray(x,dtype=str)
    write_npz(out/'complete_2d_cave_prepost_inputs.npz',series_uid=u(complete.series_uid),patient_id=u(complete.patient_id),split=u(complete.split),pre_all2d_phase_row_index=pre.astype(np.int64),post_all2d_phase_row_index=post.astype(np.int64),z_2d_raw_by_fold=z2,z_time_raw=zt,source_model_folds=np.arange(1,6,dtype=np.int64),feature_version=np.asarray('complete_2d_cave_gtmaskroi_v1'))
    missing_mapping=incomplete.assign(reason='CAVE_complete_prepost_but_missing_exact_all2d_mapping')
    cave_incomplete=pd.read_csv(root/'cave/cave_incomplete_prepost.csv',dtype=str,keep_default_na=False)
    cave_incomplete['reason']='CAVE_successful_phase_bank_lacks_one_phase; no 10240-D temporal pair or cross-modal pair fabricated'
    incomplete_out=pd.concat([missing_mapping,cave_incomplete],ignore_index=True,sort=False)
    incomplete_out.to_csv(out/'incomplete_cross_modal_series.csv',index=False,encoding='utf-8')
    audit={'status':'PASS','successful_cave_phase_rows':len(cm),'exact_png_key_phase_mappings':int((mapped.mapping_status=='mapped_exact_png_key_patient_phase').sum()),'complete_prepost_rows':len(complete),'incomplete_cave_prepost_mapping_rows':len(missing_mapping),'incomplete_due_to_only_one_successful_cave_phase':len(cave_incomplete),'incomplete_cross_modal_series_rows':len(incomplete_out),'z_2d_raw_by_fold_shape':list(z2.shape),'z_time_raw_shape':list(zt.shape),'all_finite':bool(np.isfinite(z2).all() and np.isfinite(zt).all()),'latent_averaging_applied':False,'outcome_labels_exported':False,'not_strict_outcome_oof':True}
    write_json(out/'AUDIT.json',audit);write_json(out/'SUCCESS.json',{'status':'success','complete_prepost_rows':len(complete),'mapping_key':'png_key -> segmentation_key'})
    print(json.dumps(audit,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

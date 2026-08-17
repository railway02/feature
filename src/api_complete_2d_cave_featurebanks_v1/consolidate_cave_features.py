#!/usr/bin/env python3
"""Package existing GTMask-ROI CAVE embeddings; never runs the CAVE encoder."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from loader import CAVE_ROOT, DEFAULT_OUTPUT_ROOT, read_availability, sha256, write_json, write_npz

def embedding_path(row) -> Path:
    return CAVE_ROOT / "cave_local_eligible_featurebank" / str(row.split).lower() / str(row.patient_id) / str(row.series_uid) / str(row.phase).lower() / "embedding_5120.npy"

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT_ROOT);ap.add_argument('--limit',type=int,default=0);args=ap.parse_args()
    avail=read_availability(); successful=avail.loc[avail.local_feature_available.eq('1')].copy().reset_index(drop=True);full=not args.limit
    if args.limit:successful=successful.iloc[:args.limit].copy();full=False
    values=[];paths=[];success_rows=[]
    for row in successful.itertuples(index=False):
        path=embedding_path(row)
        marker=path.with_name('.SUCCESS.json')
        if not path.is_file() or not marker.is_file():raise FileNotFoundError(f'missing existing CAVE embedding/success marker: {path}')
        x=np.load(path,allow_pickle=False)
        if x.shape!=(5120,) or not np.isfinite(x).all():raise AssertionError(f'invalid CAVE embedding: {path} {x.shape}')
        values.append(np.asarray(x,dtype=np.float32));paths.append(str(path));success_rows.append(row)
    successful=pd.DataFrame(success_rows); z=np.stack(values).astype(np.float32)
    out=args.output_root/'cave';out.mkdir(parents=True,exist_ok=True)
    u=lambda x:np.asarray(x,dtype=str)
    write_npz(out/'cave_phase_features.npz', phase_uid=u(successful.phase_uid), split=u(successful.split), patient_id=u(successful.patient_id), series_uid=u(successful.series_uid), phase=u(successful.phase), png_key=u(successful.png_key), source_embedding_path=u(paths), feature_version=np.repeat('cave_gtmask_roi_local_eligible_embedding5120_v2',len(successful)), z_time_phase_raw=z)
    successful.assign(source_embedding_path=paths,feature_version='cave_gtmask_roi_local_eligible_embedding5120_v2').to_csv(out/'cave_feature_manifest.csv',index=False,encoding='utf-8')
    # Keep every unavailable source phase as an exclusion record; no implicit "corruption" language.
    excluded=avail.loc[avail.local_feature_available.ne('1')].copy()
    excluded['exclusion_category']=np.where(excluded.local_eligible.ne('1'),'not_eligible_for_current_gtmask_roi_feature_version',np.where(excluded.runtime_feature_excluded.eq('1'),'runtime_exclusion_after_eligibility','other_unavailable'))
    excluded['interpretation']=np.where(excluded.png_key.eq(''),'no_2d_gt_mask_mapping_is_feature_version_ineligibility_not_source_dsa_corruption','see_feature_exclusion_reason')
    excluded.to_csv(out/'cave_exclusion_manifest.csv',index=False,encoding='utf-8')
    pairs=[]; incomplete=[]
    for (split,series),g in successful.groupby(['split','series_uid'],sort=True):
        phase_index=dict(zip(g.phase,g.index))
        if set(phase_index)=={'pre','post'}:
            pre,post=int(phase_index['pre']),int(phase_index['post']);pairs.append({'split':split,'series_uid':series,'patient_id':str(g.patient_id.iloc[0]),'pre_phase_row_index':pre,'post_phase_row_index':post})
        else:incomplete.append({'split':split,'series_uid':series,'patient_id':str(g.patient_id.iloc[0]),'available_phases':'|'.join(sorted(phase_index)),'missing_phase':'post' if 'post' not in phase_index else 'pre','reason':'only_one_successful_existing_gtmask_roi_cave_phase'})
    pairs=pd.DataFrame(pairs);inc=pd.DataFrame(incomplete)
    if full and (len(successful)!=2209 or len(pairs)!=992 or len(inc)!=225 or len(excluded)!=413):raise AssertionError(f'unexpected CAVE counts: {len(successful)}/{len(pairs)}/{len(inc)}/{len(excluded)}')
    if len(pairs):
        pre=pairs.pre_phase_row_index.to_numpy();post=pairs.post_phase_row_index.to_numpy();zt=np.concatenate([z[pre],z[post]],axis=1).astype(np.float32)
        if not np.array_equal(zt[:,:5120],z[pre]) or not np.array_equal(zt[:,5120:],z[post]):raise AssertionError('CAVE pre/post pairing mismatch')
        write_npz(out/'cave_prepost_features.npz', split=u(pairs.split),series_uid=u(pairs.series_uid),patient_id=u(pairs.patient_id),pre_phase_row_index=pre.astype(np.int64),post_phase_row_index=post.astype(np.int64),z_time_raw=zt,feature_version=np.asarray('cave_gtmask_roi_local_eligible_prepost_v2'))
    inc.to_csv(out/'cave_incomplete_prepost.csv',index=False,encoding='utf-8')
    audit={'status':'PASS' if full else 'SMOKE_PASS_PARTIAL','source_phase_rows':len(avail),'successful_phase_rows':len(successful),'complete_prepost_series':len(pairs),'incomplete_successful_series':len(inc),'unavailable_source_phase_rows':len(excluded),'successful_split_counts':successful.split.value_counts().to_dict(),'extraction_status_counts':avail.extraction_status.value_counts(dropna=False).to_dict(),'exclusion_category_counts':excluded.exclusion_category.value_counts().to_dict(),'feature_exclusion_reason_counts':excluded.feature_exclusion_reason.value_counts(dropna=False).to_dict(),'embedding_shape':[len(successful),5120],'prepost_shape':[len(pairs),10240],'dtype':'float32','all_finite':bool(np.isfinite(z).all()),'encoder_rerun':False,'feature_version':'GTMask-ROI local eligible existing embeddings only','availability_sha256':sha256(CAVE_ROOT/'tables/local_eligible/local_phase_availability.csv'),'full_inventory':full}
    write_json(out/'AUDIT.json',audit);write_json(out/'SUCCESS.json',{'status':'success' if full else 'smoke_partial','existing_embedding_repack_only':True,'successful_phase_rows':len(successful)})
    print(json.dumps(audit,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

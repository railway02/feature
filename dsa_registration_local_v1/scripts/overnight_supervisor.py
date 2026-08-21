#!/usr/bin/env python3
"""Crash-resumable Local Reference V1 registration supervisor.

The deliberately small state surface is per-series JSON.  An invalid case is a
completed case: it owns a neutral feature row and never silently disappears.
"""
from __future__ import annotations
import argparse, json, os, shutil, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np, pandas as pd, yaml, cv2

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from dsa_local_reg.common import load_config, atomic_json, sha256_file
from dsa_local_reg.preprocessing_adapter import load_local_reference_pairs
from dsa_local_reg.local_geometry import crop_with_border_median_padding, resize_whole_canvas
from dsa_local_reg.visualization import read_gray
from dsa_local_reg.v5_adapter import load_v5_module, validate_v5_registration_core

SYN={"transform":"SyNOnly","metric":"CC","syn_sampling":2,"reg_iterations":[60,40,20],"grad_step":0.10,"flow_sigma":3,"total_sigma":1,"singleprecision":True,"verbose":False,"lesion_metric_weight":0.0,"use_geometric_jacobian":True}
LIN={"metric":"correlation","shrink_factors":[4,2,1],"smoothing_sigmas":[2,1,0],"learning_rate":1.0,"min_step":1e-3,"iterations":160,"gradient_tolerance":1e-6}
DEFORM_NAMES=[f"{r}_{n}" for r in ("lesion","peri_lesion","whole_valid_local_roi") for n in ("logJ_mean","logJ_median","logJ_std","logJ_P10","logJ_P25","logJ_P75","logJ_P90","logJ_P95","abs_logJ_median","abs_logJ_P90","abs_logJ_P95","disp_median","disp_P90","disp_P95")]

def norm(a):
 a=np.asarray(a,np.float32); lo,hi=np.percentile(a[np.isfinite(a)],[1,99]) if np.isfinite(a).any() else (0,1); return np.clip((a-lo)/max(hi-lo,1e-6),0,1)
def corr(a,b,m=None):
 if m is None:m=np.ones_like(a,bool)
 x=a[m].ravel(); y=b[m].ravel()
 return float(np.corrcoef(x,y)[0,1]) if len(x)>8 and np.std(x)>1e-6 and np.std(y)>1e-6 else np.nan
def phase_data(p):
 frames=[crop_with_border_median_padding(read_gray(x),p.expanded_bbox) for x in p.frame_paths]
 mask=resize_whole_canvas(read_gray(p.mask_path),p.canvas_shape_yx,is_mask=True)>0
 cm=crop_with_border_median_padding(mask.astype(np.uint8),p.expanded_bbox).image>0
 scores=[float(np.mean(norm(f.image)[cm])) if cm.any() else float(np.mean(norm(f.image))) for f in frames]
 k=int(np.argmax(scores)); return frames[k].image.astype(np.float32),cm,frames[k].valid_support,k,scores[k]
def q(a,p): return float(np.nanpercentile(a,p)) if np.isfinite(a).any() else np.nan
def stats(logj,disp,mask):
 x=logj[mask & np.isfinite(logj)]; d=disp[mask & np.isfinite(disp)]
 if not len(x): return [0.]*14
 return [float(np.mean(x)),float(np.median(x)),float(np.std(x)),q(x,10),q(x,25),q(x,75),q(x,90),q(x,95),q(abs(x),50),q(abs(x),90),q(abs(x),95),q(d,50),q(d,90),q(d,95)]
def neutral(pair,reason,method):
 r={"series_uid":pair.series_uid,"patient_id":pair.patient_id,"split":pair.split,"primary_linear":method,"registration_valid":0,"linear_valid":0,"nonrigid_valid":0,"failure_reason":reason,"folding_rate":np.nan,"displacement_P95":np.nan,"abs_logJ_P99":np.nan,"inverse_consistency_logjac_mae":np.nan,"syn_seconds":np.nan}
 r.update({x:0. for x in DEFORM_NAMES}); return r
def save_sheet(path,pre,post,linear,syn,disp,logj,mask,title):
 import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
 # ``pre`` can have a different native G0 shape.  Residual panels are always
 # shown on the fixed/Post grid, never by silently resizing the registration input.
 fig,ax=plt.subplots(2,4,figsize=(16,8)); ims=[pre,post,linear,post-linear,syn,disp,logj,np.where(mask,logj,np.nan)]; names=["Pre","Post","linear aligned Pre","linear residual","SyN aligned Pre","displacement","signed logJ","mask + logJ"]
 for a,x,n in zip(ax.ravel(),ims,names): a.imshow(x,cmap="coolwarm" if "logJ" in n or "residual" in n else "gray");a.set_title(n);a.axis("off")
 fig.suptitle(title);fig.tight_layout();fig.savefig(path,dpi=130);plt.close(fig)
def run_case(pair, outroot, method, syn, stage):
 out=Path(outroot)/stage/"cases"/pair.series_uid; done=out/(method+".json")
 if done.is_file():
  # Resume only from a self-consistent terminal artifact.  Neutral/invalid rows
  # are valid terminal states; successful SyN rows must retain their maps + QC.
  try:
   cached=json.loads(done.read_text())
   valid=(cached.get("series_uid")==pair.series_uid and cached.get("primary_linear")==method)
   if syn and int(cached.get("registration_valid",0))==1:
    mp=out/(method+"_maps.npz"); sheet=out/(method+"_sheet.png")
    if not (mp.is_file() and sheet.is_file()): valid=False
    else:
     with np.load(mp,allow_pickle=False) as z: valid=valid and {"logj","disp","valid","folding"}.issubset(z.files)
   if valid: return cached
  except Exception: pass
 out.mkdir(parents=True,exist_ok=True); t0=time.time()
 try:
  sitk=load_v5_module(load_config(ROOT/"config/default.yaml"),"registration_sitk.py"); ants=load_v5_module(load_config(ROOT/"config/default.yaml"),"registration_ants.py")
  pre,pm,ps,pk,pscore=phase_data(pair.pre); post,fm,fs,fk,fscore=phase_data(pair.post)
  before=corr(post,pre) if pre.shape==post.shape else np.nan
  tx,meta=sitk.register_pair(post,pre,kind=method,fixed_mask=fm,moving_mask=pm,**LIN)
  linear=sitk.resample(pre,post,tx); lmask=sitk.resample(pm.astype(np.uint8),post,tx,is_mask=True)
  par=sitk.canonical_parameters(tx,method); after=corr(post,linear,fm & fs); support=float(np.mean(lmask & fs))
  r={"series_uid":pair.series_uid,"patient_id":pair.patient_id,"split":pair.split,"primary_linear":method,"pre_peak_index":pk,"post_peak_index":fk,"pre_peak_score":pscore,"post_peak_score":fscore,"metric_before":before,"metric_after":after,"metric_gain":after-before if np.isfinite(before) and np.isfinite(after) else np.nan,"structural_similarity_after":after,"support_loss":1-support,"linear_valid":1,"rotation":par["rotation_deg"],"tx":par["tx"],"ty":par["ty"],"scale":par["scale"],"failure_reason":""}
  if not syn:
   r.update({"nonrigid_valid":0,"registration_valid":0,"folding_rate":np.nan,"displacement_P95":np.nan,"abs_logJ_P99":np.nan,"inverse_consistency_logjac_mae":np.nan,"syn_seconds":0.}); r.update({x:0. for x in DEFORM_NAMES}); atomic_json(r,done); return r
  st=time.time(); z=ants.run_syn_residual(post,linear,fm,lmask,str(out/(method+"_syn_")),SYN,fixed_lesion=fm,moving_lesion=lmask); synsec=time.time()-st
  lj=z["canonical_logjac"]; dp=z["displacement"]; valid=z["canonical_valid"] & fs; fold=z["canonical_folding"] & valid
  dil=cv2.dilate(fm.astype(np.uint8),np.ones((15,15),np.uint8))>0; peri=dil & ~fm & valid
  r.update({"nonrigid_valid":int(not np.any(fold) and np.isfinite(lj[valid]).any()),"registration_valid":int(not np.any(fold) and np.isfinite(lj[valid]).any()),"folding_rate":float(np.mean(fold[valid])) if valid.any() else 1.,"displacement_P95":q(dp[valid],95),"abs_logJ_P99":q(abs(lj[valid]),99),"inverse_consistency_logjac_mae":z["inverse_consistency_logjac_mae"],"syn_seconds":synsec})
  vals=stats(lj,dp,fm&valid)+stats(lj,dp,peri)+stats(lj,dp,valid);r.update(dict(zip(DEFORM_NAMES,vals)))
  np.savez_compressed(out/(method+"_maps.npz"),logj=lj,disp=dp,valid=valid,folding=fold)
  save_sheet(out/(method+"_sheet.png"),pre,post,linear,z["warped_moving"],dp,lj,fm,f"{pair.series_uid} {method}")
  atomic_json(r,done);return r
 except Exception as e:
  r=neutral(pair,f"{type(e).__name__}:{e}",method);r["syn_seconds"]=time.time()-t0;atomic_json(r,done);return r
def pairs(split): return sorted(load_local_reference_pairs(load_config(ROOT/"config/default.yaml"),split=split),key=lambda x:x.series_uid)
def run_many(items,outroot,method,syn,stage,workers):
 rows=[]
 with ProcessPoolExecutor(max_workers=workers) as ex:
  fs=[ex.submit(run_case,x,str(outroot),method,syn,stage) for x in items]
  for n,f in enumerate(as_completed(fs),1):
   try: rows.append(f.result())
   except Exception as e: print("WORKER_UNCAUGHT",e,flush=True)
   print(f"[{stage}] {method} {n}/{len(items)}",flush=True)
 return rows
def write_rows(rows,path,expected):
 df=pd.DataFrame(rows); df=df.drop_duplicates("series_uid",keep="last").set_index("series_uid").reindex([x.series_uid for x in expected]).reset_index()
 # Defensive UID contract: synthesize neutral only if a crashed future failed before output.
 for i,x in enumerate(expected):
  if pd.isna(df.loc[i,"patient_id"]):
   df.loc[i]=pd.Series(neutral(x,"worker_crash_reconciled",str(df.get("primary_linear",pd.Series(["rigid"])).iloc[0])))
 df.to_csv(path,index=False); return df
def selection(rigid,sim,out):
 a=pd.DataFrame(rigid);b=pd.DataFrame(sim); ar=a[a.registration_valid==1];bs=b[b.registration_valid==1]
 # technical only; avoid threshold gaming: valid count then median metric gain, otherwise rigid.
 if len(bs)>len(ar): chosen,why="similarity","higher_technical_valid_count"
 elif len(ar)>len(bs): chosen,why="rigid","higher_technical_valid_count"
 elif len(bs) and np.nanmedian(bs.metric_gain)>np.nanmedian(ar.metric_gain)+0.02: chosen,why="similarity","material_metric_gain"
 else: chosen,why="rigid","INCONCLUSIVE_STAGE_D"
 p={"primary_linear":chosen,"selection_basis":"train_technical_only","outcome_used_for_registration_selection":False,"primary_selection_status":"SELECTED" if why!="INCONCLUSIVE_STAGE_D" else "FALLBACK_TECHNICAL","primary_selection_reason":why,"outcome_used":False,"rigid_valid":len(ar),"similarity_valid":len(bs)};atomic_json(p,out/"OVERNIGHT_PROVISIONAL_LINEAR_SELECTION.json");return p
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--run-id",default=time.strftime("overnight_%Y%m%dT%H%M%SZ",time.gmtime()));ap.add_argument("--workers",type=int,default=8);ap.add_argument("--stage",choices=["all","pilot","train","valid"],default="all");a=ap.parse_args();out=ROOT/"outputs"/a.run_id;out.mkdir(parents=True,exist_ok=True)
 cfg=load_config(ROOT/"config/default.yaml"); hashes=validate_v5_registration_core(cfg); atomic_json({"run_id":a.run_id,"started_utc":time.strftime("%FT%TZ",time.gmtime()),"workers":a.workers,"v5_hashes":hashes,"config_hash":cfg["_config_sha256"]},out/"BACKGROUND_RUN_INFO.json")
 tr=pairs("Train");va=pairs("Valid"); pilot=tr[::max(1,len(tr)//20)][:20]
 if a.stage in ("all","pilot"):
  rr=run_many(pilot,out,"rigid",True,"stage_d",a.workers); ss=run_many(pilot,out,"similarity",True,"stage_d",a.workers);pd.DataFrame(rr+ss).to_csv(out/"stage_d"/"technical_qc.csv",index=False);sel=selection(rr,ss,out)
 else: sel=json.loads((out/"OVERNIGHT_PROVISIONAL_LINEAR_SELECTION.json").read_text())
 locked={"primary_linear":sel["primary_linear"],"selection_basis":"train_technical_only","outcome_used_for_registration_selection":False,"geometry":"g0_native_independent_local_crop","fixed":"Post","moving":"Pre","syn":SYN,"neutral_invalid_policy":"zero_42D_and_registration_valid_0","v5_hashes":hashes};(out/"LOCKED_LOCAL_REFERENCE_REG_V1.yaml").write_text(yaml.safe_dump(locked,sort_keys=False))
 primary=sel["primary_linear"]
 if a.stage in ("all","train"):
  sec="similarity" if primary=="rigid" else "rigid";run_many(tr,out,sec,False,"train",a.workers); rows=run_many(tr,out,primary,True,"train",a.workers); d=write_rows(rows,out/"train_registration_features.csv",tr);d.to_csv(out/"train_registration_qc.csv",index=False)
 if a.stage in ("all","valid"):
  sec="similarity" if primary=="rigid" else "rigid";run_many(va,out,sec,False,"valid",a.workers); rows=run_many(va,out,primary,True,"valid",a.workers); d=write_rows(rows,out/"valid_registration_features.csv",va);d.to_csv(out/"valid_registration_qc.csv",index=False)
 summary={"run_id":a.run_id,"status":"COMPLETE" if a.stage=="all" else "PARTIAL","primary_linear":primary,"train_expected":800,"valid_expected":211}
 atomic_json(summary,out/"OVERNIGHT_RUN_SUMMARY.json")
 (out/"OVERNIGHT_RUN_SUMMARY.md").write_text("# Overnight run summary\n\n"+"\n".join(f"- {k}: {v}" for k,v in summary.items())+"\n",encoding="utf-8")
if __name__=="__main__": main()

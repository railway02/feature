#!/usr/bin/env python3
"""v8: frozen DeepLab decoder-fused spatial feature versus frozen v7 SegResNet OOF."""
from __future__ import annotations
import argparse, importlib.util, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

FAMILY = "deeplabv3plus_resnet50_imagenet"

def load_v7(cfg):
    path = Path(cfg["sources"]["v7_code_root"]) / "run.py"
    spec = importlib.util.spec_from_file_location("v7_masked", path)
    mod = importlib.util.module_from_spec(spec); sys.modules["v7_masked"] = mod; spec.loader.exec_module(mod)
    return mod

def cfg_load(path):
    raw = json.loads(Path(path).read_text())
    sys.path.insert(0, raw["sources"]["v6_code_root"])
    from common import atomic_csv, atomic_json, canonical_hash, load_config, sha256_file
    from model_interface import build_model
    return load_config(path), atomic_csv, atomic_json, canonical_hash, sha256_file, build_model

def fused_and_logits(model, x):
    decoder = model.model.decoder
    features = model.model.encoder(model.normalize_input(x))
    raw = decoder.aspp(features[-1])
    up = decoder.up(raw)
    high = decoder.block1(features[-4])
    fused = decoder.block2(torch.cat([up, high], dim=1))
    return fused, model.model.segmentation_head(fused)

def pool(feature, logits):
    feature, prob = feature.float(), torch.sigmoid(logits.float())
    # area is mass-preserving for downsampling 768→192; valid mask is all one
    # for this fully square corpus and therefore deliberately omitted.
    weight = F.interpolate(prob, size=feature.shape[-2:], mode="area")
    mass = weight.sum(dim=(-2,-1))
    global_feature = feature.mean(dim=(-2,-1))
    roi_feature = (feature * weight).sum(dim=(-2,-1)) / mass.clamp_min(1e-6)
    return global_feature, roi_feature, mass

def atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays); os.replace(tmp, path)

def preflight(cfg, atomic_json, sha256_file, build_model, device):
    v7 = load_v7(cfg); cases = v7.case_manifest(cfg)
    checks = {}
    for fold in range(1,6):
        path = Path(cfg["sources"]["v6_segmentation_root"])/FAMILY/f"fold_{fold}"/"model.pt"
        if not path.is_file() or not path.with_name("SUCCESS.json").is_file(): raise FileNotFoundError(path)
        checks[str(fold)] = sha256_file(path)
    model = build_model(FAMILY, cfg, load_pretrained=False).to(device).eval()
    ck = Path(cfg["sources"]["v6_segmentation_root"])/FAMILY/"fold_1"/"model.pt"
    model.load_state_dict(torch.load(ck,map_location="cpu")["state_dict"],strict=True)
    x=torch.zeros(1,1,768,768,device=device)
    with torch.no_grad():
        fused, manual=fused_and_logits(model,x); _, native=model.encode_and_decode(x)
    delta=float((manual-native).abs().max().cpu())
    if tuple(fused.shape[1:]) != tuple(cfg["feature_protocol"]["deeplab_expected_feature_shape"]) or delta>1e-6: raise RuntimeError((fused.shape,delta))
    atomic_json({"status":"success","family":FAMILY,"train_rows":len(cases),"checkpoint_sha256":checks,"fused_shape":list(fused.shape[1:]),"manual_vs_native_logits_max_abs":delta,"cave_used":False,"gt_masks_used":False,"v7_segresnet_reference":cfg["sources"]["v7_segresnet_oof"]},Path(cfg["report_root"])/"preflight"/"SUCCESS.json")

def extract(cfg, atomic_json, build_model, device):
    v7=load_v7(cfg); cases=v7.case_manifest(cfg); ds=v7.PhaseImageDataset(cases,cfg)
    dl=DataLoader(ds,batch_size=4,shuffle=False,num_workers=int(cfg["data"]["num_workers"]),pin_memory=True,persistent_workers=True)
    lookup={(u,p):i for i,(u,p,_) in enumerate(ds.rows)}
    for fold in range(1,6):
        model=build_model(FAMILY,cfg,load_pretrained=False).to(device).eval(); ck=Path(cfg["sources"]["v6_segmentation_root"])/FAMILY/f"fold_{fold}"/"model.pt"; model.load_state_dict(torch.load(ck,map_location="cpu")["state_dict"],strict=True)
        g=r=m=None
        for x,_,idx,_ in dl:
            x=x.to(device,non_blocking=True)
            with torch.no_grad(),autocast(enabled=True): fmap,logits=fused_and_logits(model,x)
            gg,rr,mm=pool(fmap,logits)
            if float(mm.min())<=cfg["feature_protocol"]["min_predroi_mass"]: raise RuntimeError(f"near-zero ROI mass fold {fold}")
            if g is None: g=np.empty((len(ds),256),np.float32);r=np.empty_like(g);m=np.empty(len(ds),np.float32)
            ii=idx.numpy();g[ii]=gg.cpu().numpy();r[ii]=rr.cpu().numpy();m[ii]=mm.squeeze(1).cpu().numpy()
        x=np.stack([np.concatenate([g[lookup[(str(z.series_uid),'Pre')]],r[lookup[(str(z.series_uid),'Pre')]],g[lookup[(str(z.series_uid),'Post')]],r[lookup[(str(z.series_uid),'Post')]]]) for z in cases.itertuples(index=False)]).astype(np.float32)
        out=Path(cfg["output_root"])/"features"/FAMILY/f"fold_{fold}"; atomic_npz(out/"train.npz",spatial_feature=x,series_uid=cases.series_uid.to_numpy(dtype=str),patient_id=cases.patient_id.to_numpy(dtype=str),target=cases.target.to_numpy(np.int64),outer_fold=cases.fold.to_numpy(np.int64),phase_predroi_mass=m.reshape(len(cases),2))
        atomic_json({"status":"success","family":FAMILY,"fold":fold,"rows":len(cases),"feature_shape":list(x.shape),"feature_tap":"decoder_fused","probability_resize":"area","min_predroi_mass":float(m.min()),"cave_used":False,"gt_masks_used":False},out/"SUCCESS.json"); del model

def verify(cfg, atomic_json):
    v7=load_v7(cfg); banks=[v7.load_npz(Path(cfg["output_root"])/"features"/FAMILY/f"fold_{k}"/"train.npz") for k in range(1,6)]; b=banks[0]
    c={"five_folds":len(banks)==5,"shape":all(x["spatial_feature"].shape==(781,1024) for x in banks),"uid_order":all(np.array_equal(b["series_uid"],x["series_uid"]) for x in banks[1:]),"finite":all(np.isfinite(x["spatial_feature"]).all() for x in banks),"roi_mass":all((x["phase_predroi_mass"]>cfg["feature_protocol"]["min_predroi_mass"]).all() for x in banks),"no_gt_or_temporal":all(not any(q.startswith("gt") or "cave" in q or "temporal" in q for q in x) for x in banks)}
    if not all(c.values()):raise RuntimeError(c)
    atomic_json({"status":"PASS","family":FAMILY,"checks":c},Path(cfg["output_root"])/"featurebanks"/FAMILY/"verification.json")

def oof(cfg, atomic_csv, atomic_json, canonical_hash, device):
    v7=load_v7(cfg); v7.FAMILIES=(FAMILY,); v7.outcome_oof(cfg,atomic_csv,atomic_json,lambda *a,**k: __import__('common').atomic_torch_save(*a,**k),canonical_hash,__import__('common').set_seed,device)

def compare(cfg,atomic_csv,atomic_json):
    deep=pd.read_csv(Path(cfg["report_root"])/"outcome_oof"/FAMILY/"train_oof_predictions.csv",dtype={"series_uid":str,"patient_id":str}); seg=pd.read_csv(cfg["sources"]["v7_segresnet_oof"],dtype={"series_uid":str,"patient_id":str})
    for c in ["series_uid","patient_id","target","outer_fold"]:
        if not np.array_equal(deep[c].to_numpy(),seg[c].to_numpy()):raise RuntimeError(c)
    v7=load_v7(cfg); y=deep.target.to_numpy(); p=deep.probability.to_numpy(); s=seg.probability.to_numpy(); rng=np.random.default_rng(cfg["bootstrap"]["seed"]); d={k:[] for k in ["auroc","auprc","brier"]}
    for _ in range(cfg["bootstrap"]["draws"]):
        ix=rng.integers(0,len(y),len(y)); a,b=v7.metrics(y[ix],p[ix]),v7.metrics(y[ix],s[ix])
        for k in d:d[k].append(a[k]-b[k])
    result={"status":"success","comparison":"v8_deeplab_decoder_fused_area_pooling_minus_v7_segresnet_bottleneck_bilinear_pooling","deeplab_v8":v7.metrics(y,p),"segresnet_v7":v7.metrics(y,s),"paired_bootstrap_delta":{k:{"point":v7.metrics(y,p)[k]-v7.metrics(y,s)[k],"ci95":[float(np.quantile(d[k],.025)),float(np.quantile(d[k],.975))]} for k in d},"draws":cfg["bootstrap"]["draws"],"cave_used":False,"comparison_caveat":"SegResNet is reused from v7 (96x96 bottleneck; bilinear ROI resize); it was not retrained or re-extracted in v8 by user decision."}
    root=Path(cfg["report_root"])/"comparison";atomic_json(result,root/"DEEPLAB_V8_VS_SEGRESNET_V7_OOF.json");atomic_csv(pd.DataFrame({"series_uid":deep.series_uid,"patient_id":deep.patient_id,"target":y,"outer_fold":deep.outer_fold,"deeplab_v8_probability":p,"segresnet_v7_probability":s}),root/"paired_oof_predictions.csv")

def main():
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["preflight","extract-train","verify","outcome-oof","compare"]);p.add_argument("--config",required=True);p.add_argument("--device",default="cuda:0");a=p.parse_args();cfg,acsv,ajson,ch,sh,build=cfg_load(a.config);os.environ["TORCH_HOME"]=cfg["torch_home"];dev=torch.device(a.device)
    if a.stage=="preflight":preflight(cfg,ajson,sh,build,dev)
    elif a.stage=="extract-train":extract(cfg,ajson,build,dev)
    elif a.stage=="verify":verify(cfg,ajson)
    elif a.stage=="outcome-oof":oof(cfg,acsv,ajson,ch,dev)
    else:compare(cfg,acsv,ajson)
if __name__=="__main__":main()

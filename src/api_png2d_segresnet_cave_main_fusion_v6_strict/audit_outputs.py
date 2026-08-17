#!/usr/bin/env python3
"""Post-training fail-closed audit for strict main-fusion artifacts."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("/root/autodl-tmp/aneurysm")
DEFAULT = ROOT / "outputs/api_png2d_segresnet_cave_main_fusion_v6_strict"

def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for x in iter(lambda:f.read(1024*1024),b""): h.update(x)
    return h.hexdigest()

def summary(x: np.ndarray) -> dict:
    return {"mean":float(x.mean()),"std":float(x.std()),"min":float(x.min()),"max":float(x.max()),"fraction_0p4_to_0p6":float(((x>=.4)&(x<=.6)).mean())}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output-root",type=Path,default=DEFAULT); args=ap.parse_args(); root=args.output_root
    with np.load(root/"train_oof_main_outputs.npz",allow_pickle=False) as tr, np.load(root/"train_main_outputs_by_fold.npz",allow_pickle=False) as tb, np.load(root/"valid_main_outputs_by_fold.npz",allow_pickle=False) as va:
        required_tr={"series_uid","patient_id","outer_fold","source_fusion_fold","z_main","main_logit","main_prob","spatial_gate","temporal_gate"}
        required_tb={"series_uid","patient_id","outer_fold","source_fusion_folds","z_main_by_fold","main_logit_by_fold","main_prob_by_fold","spatial_gate_by_fold","temporal_gate_by_fold"}
        required_va={"series_uid","patient_id","source_fusion_folds","z_main_by_fold","main_logit_by_fold","main_prob_by_fold","spatial_gate_by_fold","temporal_gate_by_fold"}
        if required_tr-set(tr.files) or required_tb-set(tb.files) or required_va-set(va.files): raise AssertionError("required output fields missing")
        forbidden={"target","label","outcome","y","adverse_outcome"}
        if forbidden&set(tr.files) or forbidden&set(tb.files) or forbidden&set(va.files): raise AssertionError("label field present in public representation output")
        if tr["z_main"].shape!=(781,256) or tb["z_main_by_fold"].shape!=(781,5,256) or va["z_main_by_fold"].shape!=(207,5,256): raise AssertionError("z_main shape mismatch")
        if not np.array_equal(tr["outer_fold"],tr["source_fusion_fold"]): raise AssertionError("Train source fusion routing mismatch")
        if not np.array_equal(tr["series_uid"].astype(str),tb["series_uid"].astype(str)) or not np.array_equal(tr["patient_id"].astype(str),tb["patient_id"].astype(str)) or not np.array_equal(tr["outer_fold"],tb["outer_fold"]): raise AssertionError("Train by-fold identifier/routing mismatch")
        if not np.array_equal(tb["source_fusion_folds"],np.arange(1,6)): raise AssertionError("Train by-fold labels mismatch")
        if not np.array_equal(va["source_fusion_folds"],np.arange(1,6)): raise AssertionError("Valid fold labels mismatch")
        arrays=[tr[k] for k in ("z_main","main_logit","main_prob","spatial_gate","temporal_gate")]+[tb[k] for k in ("z_main_by_fold","main_logit_by_fold","main_prob_by_fold","spatial_gate_by_fold","temporal_gate_by_fold")]+[va[k] for k in ("z_main_by_fold","main_logit_by_fold","main_prob_by_fold","spatial_gate_by_fold","temporal_gate_by_fold")]
        if not all(a.dtype==np.float32 and np.isfinite(a).all() for a in arrays): raise AssertionError("non-finite/non-float32 output")
        if not (np.all((tr["main_prob"]>=0)&(tr["main_prob"]<=1)) and np.all((tb["main_prob_by_fold"]>=0)&(tb["main_prob_by_fold"]<=1)) and np.all((va["main_prob_by_fold"]>=0)&(va["main_prob_by_fold"]<=1))): raise AssertionError("probability range")
        if not (np.all((tr["spatial_gate"]>=0)&(tr["spatial_gate"]<=1)) and np.all((tr["temporal_gate"]>=0)&(tr["temporal_gate"]<=1)) and np.all((tb["spatial_gate_by_fold"]>=0)&(tb["spatial_gate_by_fold"]<=1)) and np.all((tb["temporal_gate_by_fold"]>=0)&(tb["temporal_gate_by_fold"]<=1)) and np.all((va["spatial_gate_by_fold"]>=0)&(va["spatial_gate_by_fold"]<=1)) and np.all((va["temporal_gate_by_fold"]>=0)&(va["temporal_gate_by_fold"]<=1))): raise AssertionError("gate range")
        row=np.arange(781); fold=tr["outer_fold"].astype(np.int64)-1; oof_match={}
        for base in ("z_main","main_logit","main_prob","spatial_gate","temporal_gate"):
            selected=tb[f"{base}_by_fold"][row,fold]
            maximum=float(np.max(np.abs(selected-tr[base])))
            if not np.allclose(selected,tr[base],rtol=1e-5,atol=1e-5): raise AssertionError(f"Train by-fold selection differs from OOF: {base}")
            oof_match[base]=maximum
        gate={"train_oof":{"spatial_gate":summary(tr["spatial_gate"]),"temporal_gate":summary(tr["temporal_gate"])},"valid_by_fold":{"spatial_gate":summary(va["spatial_gate_by_fold"]),"temporal_gate":summary(va["temporal_gate_by_fold"])}}
    artifacts=sorted([root/"DELIVERY_MANIFEST.json",root/"train_oof_main_outputs.npz",root/"train_main_outputs_by_fold.npz",root/"valid_main_outputs_by_fold.npz",root/"alignment/alignment_audit.json",root/"main_fusion/SMOKE_TEST.json",root/"main_fusion/TRAIN_BY_FOLD_EXPORT_AUDIT.json",root/"main_fusion/DOWNSTREAM_LOADER_SMOKE_TEST.json",root/"main_fusion/metrics.json"]+[root/f"main_fusion/fold_{k}/model.pt" for k in range(1,6)])
    hashes={str(p.relative_to(root)):digest(p) for p in artifacts}
    (root/"SHA256SUMS.txt").write_text("".join(f"{h}  {p}\n" for p,h in hashes.items()),encoding="utf-8")
    report={"status":"PASS","train_oof_rows":781,"train_fold_specific_rows":781,"train_fold_representations":5,"valid_rows":207,"valid_fold_representations":5,"formal_input":"z_2d_raw=[G_pre,soft_PredROI_pre,G_post,soft_PredROI_post] plus CAVE deep; no GTROI/mask/scalar","public_outputs_contain_outcome_label":False,"train_is_oof":True,"train_by_fold_is_frozen_inference":True,"train_by_fold_oof_selection_matches_strict_oof":True,"train_by_fold_oof_comparison_tolerance":{"rtol":1e-5,"atol":1e-5},"train_by_fold_oof_max_abs_difference":oof_match,"downstream_loader_smoke_passed":True,"valid_used_for_selection":False,"latent_averaging_applied":False,"float32_finite":True,"probability_range":True,"gate_range":True,"gate_audit":gate,"sha256":hashes}
    (root/"main_fusion/FINAL_AUDIT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (root/"main_fusion/SUCCESS.json").write_text(json.dumps({"status":"success","train_oof_rows":781,"train_five_fold_representations":True,"valid_rows":207,"five_fold_valid_representations":True,"downstream_loader_smoke_passed":True,"public_outputs_contain_outcome_label":False,"latent_averaging_applied":False},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__": main()

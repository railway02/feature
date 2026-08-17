#!/usr/bin/env python3
"""Short actual-data smoke for both frozen confirmatory backbones."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from common import atomic_json, atomic_torch_save, load_config, sha256_file, set_seed
from data import expanded_strict_fold_split, SegmentationDataset
from losses import segmentation_loss
from model_interface import build_model, roi_pool

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True); parser.add_argument('--device',default='cuda:0'); args=parser.parse_args()
    cfg=load_config(args.config); os.environ['TORCH_HOME']=cfg['torch_home']; device=torch.device(args.device)
    if not (Path(cfg['report_root'])/'expanded_strict_preflight/SUCCESS.json').is_file(): raise RuntimeError('expanded strict preflight required')
    _, inner_train, _, _, _=expanded_strict_fold_split(cfg,1)
    ds=SegmentationDataset(inner_train.iloc[:2].copy(),cfg,augment=True)
    batch=[ds[0],ds[1]]; x=torch.stack([v[0] for v in batch]).to(device); y=torch.stack([v[1] for v in batch]).to(device)
    out=Path(cfg['output_root'])/'expanded_strict'/'smoke'; rep=Path(cfg['report_root'])/'expanded_strict_smoke'; out.mkdir(parents=True,exist_ok=True); rep.mkdir(parents=True,exist_ok=True)
    result={'status':'PASS','short_smoke_only':True,'data_rows':[str(inner_train.iloc[i].png_key) for i in range(2)],'models':{}}
    for family in ['segresnet','deeplabv3plus_resnet50_imagenet']:
        set_seed(20260810); model=build_model(family,cfg,load_pretrained=True).to(device); optimizer=AdamW(model.parameters(),lr=float(cfg['models'][family]['learning_rate']),weight_decay=float(cfg['models'][family]['weight_decay'])); scaler=GradScaler(enabled=device.type=='cuda')
        model.train(); optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=device.type=='cuda'):
            fmap, logits=model.encode_and_decode(x); loss=segmentation_loss(logits,y,14.0,cfg); pooled,mass=roi_pool(fmap,torch.sigmoid(logits),'bilinear')
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); model.eval()
        with torch.no_grad(), autocast(enabled=device.type=='cuda'): reference=model(x).float().cpu()
        checkpoint=out/f'{family}_smoke.pt'; atomic_torch_save({'state_dict':model.state_dict()},checkpoint)
        reloaded=build_model(family,cfg,load_pretrained=False).to(device).eval(); reloaded.load_state_dict(torch.load(checkpoint,map_location='cpu')['state_dict'],strict=True)
        with torch.no_grad(), autocast(enabled=device.type=='cuda'): restored=reloaded(x).float().cpu()
        delta=float((reference-restored).abs().max())
        if delta != 0.0: raise RuntimeError(f'{family} checkpoint smoke mismatch {delta}')
        result['models'][family]={'fmap_shape':list(fmap.shape),'logits_shape':list(logits.shape),'roi_shape':list(pooled.shape),'roi_mass_finite':bool(torch.isfinite(mass).all()),'loss':float(loss.detach().cpu()),'checkpoint':str(checkpoint),'checkpoint_sha256':sha256_file(checkpoint),'reload_max_abs_logit_difference':delta}
        del model,reloaded
        if device.type=='cuda': torch.cuda.empty_cache()
    atomic_json(result,rep/'SMOKE_EXPANDED_STRICT.json'); atomic_json(result,rep/'SUCCESS.json'); print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__': main()

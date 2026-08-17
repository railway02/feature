from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from assets import largest_component, resize_transform_from_json, restore_probability


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, value): return self.block(value)


class UNetSmall(nn.Module):
    def __init__(self, in_channels: int = 3, base: int = 16):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.bottleneck = DoubleConv(base * 8, base * 16)
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2); self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2); self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2); self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2); self.dec1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    @staticmethod
    def _cat(up, skip):
        if up.shape[-2:] != skip.shape[-2:]: up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([skip, up], dim=1)

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1)); e3 = self.enc3(self.pool(e2)); e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(self._cat(self.up4(b), e4)); d3 = self.dec3(self._cat(self.up3(d4), e3))
        d2 = self.dec2(self._cat(self.up2(d3), e2)); d1 = self.dec1(self._cat(self.up1(d2), e1))
        return self.out(d1)


def random_augment(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    channels, height, width = image.shape
    angle = float(rng.uniform(-8, 8)); scale = float(rng.uniform(0.92, 1.08))
    dx = float(rng.uniform(-0.04, 0.04) * width); dy = float(rng.uniform(-0.04, 0.04) * height)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale); matrix[:, 2] += [dx, dy]
    augmented = np.stack([cv2.warpAffine(channel, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101) for channel in image])
    target = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    alpha = float(rng.uniform(0.85, 1.15)); beta = float(rng.uniform(-0.08, 0.08))
    augmented = np.clip(augmented * alpha + beta + rng.normal(0, 0.015, augmented.shape), 0, 1)
    return augmented.astype(np.float32), target.astype(np.float32)


class SegmentationDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, augment: bool, seed: int):
        self.frame = frame.reset_index(drop=True); self.augment = augment; self.seed = seed

    def __len__(self): return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]; raw = np.load(row.sample_path, allow_pickle=False)
        image = raw["image"].astype(np.float32) / 255.0; mask = raw["mask"].astype(np.float32)
        if self.augment:
            image, mask = random_augment(image, mask, np.random.default_rng(self.seed + index + random.randint(0, 1_000_000)))
        return torch.from_numpy(image), torch.from_numpy(mask[None]), str(row.sample_uid)


class DiceBCELoss(nn.Module):
    def __init__(self, pos_weight: float, dice_weight: float, bce_weight: float):
        super().__init__(); self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32)); self.dw=dice_weight; self.bw=bce_weight

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)
        probability = torch.sigmoid(logits); dims=(1,2,3)
        intersection=(probability*target).sum(dims); denominator=probability.sum(dims)+target.sum(dims)
        dice = 1.0 - ((2*intersection+1.0)/(denominator+1.0)).mean()
        return self.bw*bce+self.dw*dice


def dice_score(probability: torch.Tensor, target: torch.Tensor, threshold: float=0.5) -> float:
    prediction=(probability>=threshold).float(); dims=(1,2,3)
    value=((2*(prediction*target).sum(dims)+1.0)/(prediction.sum(dims)+target.sum(dims)+1.0)).mean()
    return float(value.item())


def compute_pos_weight(frame: pd.DataFrame, cap: float) -> float:
    positive=0; total=0
    for path in frame.sample_path:
        raw=np.load(path,allow_pickle=False); mask=raw["mask"]; positive += int(mask.sum()); total += mask.size
    return float(min(max((total-positive)/max(positive,1),1.0),cap))


def make_loader(frame: pd.DataFrame, batch_size: int, augment: bool, seed: int, shuffle: bool) -> DataLoader:
    return DataLoader(SegmentationDataset(frame,augment,seed),batch_size=batch_size,shuffle=shuffle,num_workers=2,pin_memory=True,persistent_workers=False)


def train_model(train_frame: pd.DataFrame, valid_frame: pd.DataFrame | None, config: dict[str,Any], output_path: Path, seed: int, epochs: int | None=None) -> dict[str,Any]:
    set_seed(seed); device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=UNetSmall(int(config["input_channels_count"]),int(config["base_channels"])).to(device)
    pos_weight=compute_pos_weight(train_frame,float(config["foreground_pos_weight_cap"]))
    criterion=DiceBCELoss(pos_weight,float(config["dice_weight"]),float(config["bce_weight"])).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(config["learning_rate"]),weight_decay=float(config["weight_decay"]))
    scaler=torch.cuda.amp.GradScaler(enabled=bool(config["amp"]) and device.type=="cuda")
    batch_size=int(config["batch_size"]); effective=int(config["effective_batch_size"]); accumulation=max(1,math.ceil(effective/batch_size))
    train_loader=make_loader(train_frame,batch_size,True,seed,True)
    valid_loader=make_loader(valid_frame,batch_size,False,seed,False) if valid_frame is not None and len(valid_frame) else None
    maximum=int(epochs or config["max_epochs"]); patience=int(config["early_stop"]); best=-math.inf; best_epoch=0; wait=0; history=[]
    output_path.parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,maximum+1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses=[]
        for step,(image,target,_) in enumerate(train_loader,1):
            image=image.to(device,non_blocking=True); target=target.to(device,non_blocking=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=scaler.is_enabled()):
                loss=criterion(model(image),target)/accumulation
            scaler.scale(loss).backward(); losses.append(float(loss.item()*accumulation))
            if step%accumulation==0 or step==len(train_loader):
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        valid_dice=float("nan")
        if valid_loader is not None:
            model.eval(); values=[]
            with torch.inference_mode():
                for image,target,_ in valid_loader:
                    image=image.to(device); target=target.to(device); values.append(dice_score(torch.sigmoid(model(image)),target))
            valid_dice=float(np.mean(values)) if values else float("nan")
            score=valid_dice
        else:
            score=-float(np.mean(losses))
        history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"valid_dice":valid_dice})
        if score>best+1e-5:
            best=score; best_epoch=epoch; wait=0
            torch.save({"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"in_channels":int(config["input_channels_count"]),"base_channels":int(config["base_channels"]),"epoch":epoch,"score":score,"pos_weight":pos_weight,"seed":seed},output_path)
        else:
            wait+=1
        print(json.dumps({"epoch":epoch,"train_loss":history[-1]["train_loss"],"valid_dice":valid_dice,"best_epoch":best_epoch},sort_keys=True),flush=True)
        if valid_loader is not None and wait>=patience: break
    return {"best_epoch":best_epoch,"best_score":best,"epochs_ran":len(history),"pos_weight":pos_weight,"history":history,"device":str(device),"effective_batch_size":effective}


def load_model(checkpoint: Path, device: torch.device) -> nn.Module:
    payload=torch.load(checkpoint,map_location="cpu"); model=UNetSmall(int(payload["in_channels"]),int(payload["base_channels"])); model.load_state_dict(payload["state_dict"]); return model.to(device).eval()


def predict_frame(model: nn.Module, frame: pd.DataFrame, batch_size: int=2) -> dict[str,np.ndarray]:
    device=next(model.parameters()).device; loader=make_loader(frame,batch_size,False,42,False); outputs={}
    with torch.inference_mode():
        for image,_,uids in loader:
            probability=torch.sigmoid(model(image.to(device))).float().cpu().numpy()[:,0]
            for uid,value in zip(uids,probability): outputs[str(uid)]=value
    return outputs


def component_from_probability(probability: np.ndarray, threshold: float) -> np.ndarray:
    return largest_component((probability>=threshold).astype(np.uint8))


def bbox(mask: np.ndarray) -> tuple[int,int,int,int] | None:
    points=np.argwhere(mask>0)
    if not len(points): return None
    y0,x0=points.min(0); y1,x1=points.max(0)+1
    return int(x0),int(y0),int(x1),int(y1)


def expanded_square_box(box: tuple[int,int,int,int] | None, shape: tuple[int,int], factor: float, min_fraction: float, max_fraction: float, center: tuple[float,float] | None=None, allow_outside: bool=False) -> tuple[int,int,int,int]:
    height,width=shape; short=min(height,width)
    if box is None:
        if center is None: center=(width/2,height/2)
        cx,cy=center; side=short*min_fraction
    else:
        x0,y0,x1,y1=box; cx=(x0+x1)/2; cy=(y0+y1)/2; side=max(x1-x0,y1-y0)*factor
    side=max(short*min_fraction,min(short*max_fraction,side)); side=int(math.ceil(side)); half=side/2
    x0=int(math.floor(cx-half)); y0=int(math.floor(cy-half)); x1=x0+side; y1=y0+side
    if allow_outside:
        return x0,y0,x1,y1
    if x0<0: x1-=x0; x0=0
    if y0<0: y1-=y0; y0=0
    if x1>width: x0-=x1-width; x1=width
    if y1>height: y0-=y1-height; y1=height
    return max(0,x0),max(0,y0),min(width,x1),min(height,y1)


def threshold_metrics(probability: np.ndarray, gt: np.ndarray, threshold: float, roi_cfg: dict[str,Any]) -> dict[str,float]:
    pred=component_from_probability(probability,threshold); pred_box=bbox(pred); empty=float(pred_box is None)
    overlap=float(((pred>0)&(gt>0)).any()); center=None
    if pred_box is None:
        y,x=np.unravel_index(int(np.argmax(probability)),probability.shape); center=(float(x),float(y))
    expanded=expanded_square_box(pred_box,gt.shape,float(roi_cfg["padding_factor"]),float(roi_cfg["minimum_side_fraction"]),float(roi_cfg["maximum_side_fraction"]),center)
    x0,y0,x1,y1=expanded; coverage=float(gt[y0:y1,x0:x1].sum()/max(gt.sum(),1)); area=float((x1-x0)*(y1-y0)/gt.size)
    intersection=float(((pred>0)&(gt>0)).sum()); union=float(((pred>0)|(gt>0)).sum())
    dice=float(2*intersection/max(pred.sum()+gt.sum(),1)); iou=float(intersection/max(union,1))
    sensitivity=float(intersection/max(gt.sum(),1))
    gt_points=np.argwhere(gt>0); pred_points=np.argwhere(pred>0)
    if len(gt_points) and len(pred_points):
        centroid_distance=float(np.linalg.norm(gt_points.mean(0)-pred_points.mean(0)))
    else:
        centroid_distance=float("nan")
    gt_box=bbox(gt)
    if pred_box is not None and gt_box is not None:
        px0,py0,px1,py1=pred_box; gx0,gy0,gx1,gy1=gt_box
        bbox_intersection=max(0,min(px1,gx1)-max(px0,gx0))*max(0,min(py1,gy1)-max(py0,gy0))
        bbox_union=(px1-px0)*(py1-py0)+(gx1-gx0)*(gy1-gy0)-bbox_intersection
        bbox_iou=float(bbox_intersection/max(bbox_union,1))
    else: bbox_iou=0.0
    score=coverage-0.25*area-0.50*empty-0.25*(1.0-overlap)
    return {"coverage":coverage,"roi_area_ratio":area,"empty":empty,"on_target":overlap,"dice":dice,"iou":iou,"sensitivity":sensitivity,"centroid_distance_pixels":centroid_distance,"bbox_iou":bbox_iou,"score":score}


def read_sample_metadata(path: str|Path) -> dict[str,Any]:
    raw=np.load(path,allow_pickle=False); return json.loads(str(raw["metadata"].item()))


def restore_model_probability(sample_path: str|Path, probability: np.ndarray) -> np.ndarray:
    metadata=read_sample_metadata(sample_path); transform=resize_transform_from_json(metadata["resize_transform"]); return restore_probability(probability,transform)

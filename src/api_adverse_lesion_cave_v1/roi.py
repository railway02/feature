from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from assets import normalize01


def bbox_to_text(box: tuple[int,int,int,int]) -> str:
    return "|".join(map(str,box))


def bbox_from_text(value: str) -> tuple[int,int,int,int]:
    items=tuple(int(part) for part in str(value).split("|"))
    if len(items)!=4: raise ValueError(value)
    return items


def box_padding(box: tuple[int,int,int,int], shape: tuple[int,int]) -> tuple[int,int,int,int]:
    x0,y0,x1,y1=box; height,width=shape
    return max(0,-x0),max(0,-y0),max(0,x1-width),max(0,y1-height)


def crop_frames(frames: np.ndarray, box: tuple[int,int,int,int]) -> np.ndarray:
    x0,y0,x1,y1=box
    if not (x0<x1 and y0<y1):
        raise AssertionError(f"Invalid ROI box {box} for frames {frames.shape}")
    requested_h,requested_w=y1-y0,x1-x0
    if requested_h!=requested_w:
        raise AssertionError(f"ROI not square: {box}")
    height,width=frames.shape[1:]
    left,top,right,bottom=box_padding(box,(height,width))
    sx0,sy0=max(0,x0),max(0,y0); sx1,sy1=min(width,x1),min(height,y1)
    if not (sx0<sx1 and sy0<sy1):
        raise AssertionError(f"ROI does not intersect frames: {box} for {frames.shape}")
    cropped=frames[:,sy0:sy1,sx0:sx1]
    if any((left,top,right,bottom)):
        padded=np.empty((len(frames),requested_h,requested_w),dtype=frames.dtype)
        for index,frame in enumerate(frames):
            border=np.concatenate((frame[0],frame[-1],frame[:,0],frame[:,-1]))
            padded[index].fill(np.asarray(np.median(border),dtype=frames.dtype))
        padded[:,top:top+cropped.shape[1],left:left+cropped.shape[2]]=cropped
        cropped=padded
    if cropped.shape[1]!=cropped.shape[2]: raise AssertionError(f"ROI not square: {cropped.shape}")
    return cropped


def probability_mass_component(probability: np.ndarray, threshold: float) -> np.ndarray:
    binary=(probability>=threshold).astype(np.uint8); count,labels,_,_=cv2.connectedComponentsWithStats(binary,8)
    if count<=1: return binary
    scores=[float(probability[labels==index].sum()) for index in range(1,count)]
    selected=1+int(np.argmax(scores)); return (labels==selected).astype(np.uint8)


def save_bbox_overlay(background: np.ndarray, mask: np.ndarray, box: tuple[int,int,int,int], path: Path, title: str="") -> None:
    gray=np.rint(normalize01(background)*255).astype(np.uint8); canvas=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    selected=mask>0; canvas[selected]=np.rint(0.45*canvas[selected]+0.55*np.array([0,255,0],dtype=np.float32)).astype(np.uint8)
    x0,y0,x1,y1=box; cv2.rectangle(canvas,(x0,y0),(x1-1,y1-1),(0,0,255),2)
    if title: cv2.putText(canvas,title[:100],(8,26),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2,cv2.LINE_AA)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),canvas): raise RuntimeError(path)


def save_temporal_montage(frames: np.ndarray, box: tuple[int,int,int,int], path: Path, maximum: int=8) -> None:
    positions=np.rint(np.linspace(0,len(frames)-1,min(maximum,len(frames)))).astype(int); tiles=[]
    for position in positions:
        image=cv2.cvtColor(np.rint(normalize01(frames[position])*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
        x0,y0,x1,y1=box; cv2.rectangle(image,(x0,y0),(x1-1,y1-1),(0,0,255),2); tiles.append(cv2.resize(image,(256,256),interpolation=cv2.INTER_AREA))
    grid=np.hstack(tiles); path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),grid): raise RuntimeError(path)


def mask_features(mask: np.ndarray, prefix: str="") -> dict[str,float]:
    binary=(mask>0).astype(np.uint8); height,width=binary.shape; area=float(binary.sum()); result={}
    key=lambda name:f"{prefix}{name}"
    result[key("area_pixels")]=area; result[key("area_ratio")]=area/(height*width)
    count,labels,stats,centroids=cv2.connectedComponentsWithStats(binary,8); result[key("component_count")]=float(max(count-1,0))
    if count<=1:
        for name in ("bbox_width_ratio","bbox_height_ratio","bbox_area_ratio","centroid_x_ratio","centroid_y_ratio","perimeter","circularity","eccentricity"):
            result[key(name)]=float("nan")
        return result
    selected=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA])); component=(labels==selected).astype(np.uint8)
    x,y,w,h,component_area=stats[selected]; cx,cy=centroids[selected]
    contours,_=cv2.findContours(component,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); perimeter=float(sum(cv2.arcLength(item,True) for item in contours))
    circularity=float(4*np.pi*component_area/max(perimeter*perimeter,1e-8)); points=np.argwhere(component>0)
    eccentricity=float("nan")
    if len(points)>=3:
        covariance=np.cov(points.astype(np.float64).T); values=np.sort(np.linalg.eigvalsh(covariance)); eccentricity=float(np.sqrt(max(0,1-values[0]/max(values[-1],1e-8))))
    result.update({key("bbox_width_ratio"):w/width,key("bbox_height_ratio"):h/height,key("bbox_area_ratio"):(w*h)/(height*width),key("centroid_x_ratio"):cx/width,key("centroid_y_ratio"):cy/height,key("perimeter"):perimeter,key("circularity"):circularity,key("eccentricity"):eccentricity})
    return result

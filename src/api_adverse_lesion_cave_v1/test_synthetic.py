#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import torch

from assets import apply_orientation, make_resize_transform, resize_mask_to_model, resize_stack_to_model, restore_probability
from roi import bbox_from_text, bbox_to_text, crop_frames, probability_mass_component
from segmentation import UNetSmall, bbox, expanded_square_box


def main() -> int:
    array=np.arange(20).reshape(4,5); assert apply_orientation(array,"transpose").shape==(5,4)
    transform=make_resize_transform(40,60,64); stack=np.zeros((3,40,60),np.uint8); stack[:,10:20,20:30]=255; resized=resize_stack_to_model(stack,transform); assert resized.shape==(3,64,64)
    mask=np.zeros((40,60),np.uint8); mask[10:20,20:30]=1; model_mask=resize_mask_to_model(mask,transform); restored=restore_probability(model_mask.astype(np.float32),transform); assert restored.shape==mask.shape
    model=UNetSmall(3,4); output=model(torch.zeros(2,3,64,64)); assert output.shape==(2,1,64,64)
    probability=np.zeros((40,60),np.float32); probability[10:20,20:30]=0.9; component=probability_mass_component(probability,0.5); box=bbox(component); expanded=expanded_square_box(box,component.shape,1.5,0.12,0.45); assert bbox_from_text(bbox_to_text(expanded))==expanded
    frames=np.zeros((5,40,60),np.uint8); crop=crop_frames(frames,expanded); assert crop.shape[1]==crop.shape[2]
    frames[:,0,:]=7; outside=expanded_square_box((0,0,4,4),frames.shape[1:],3.0,0.12,0.45,allow_outside=True)
    padded=crop_frames(frames,outside); assert padded.shape[1:]==(outside[2]-outside[0],outside[2]-outside[0])
    assert outside[0]<0 or outside[1]<0
    print("[PASS] synthetic tests")
    return 0


if __name__=="__main__": raise SystemExit(main())

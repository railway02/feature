from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSegmentationModel(nn.Module):
    family: str

    def encode_and_decode(self, x: torch.Tensor):
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        return self.encode_and_decode(x)[1]


def global_pool(feature_map: torch.Tensor) -> torch.Tensor:
    return F.adaptive_avg_pool2d(feature_map, 1).flatten(1)


def roi_pool(feature_map: torch.Tensor, mask: torch.Tensor, mode: str = "bilinear", eps: float = 1e-6):
    if mode == "bilinear":
        weight = F.interpolate(mask, size=feature_map.shape[-2:], mode=mode, align_corners=False)
    elif mode in {"nearest", "area"}:
        weight = F.interpolate(mask, size=feature_map.shape[-2:], mode=mode)
    else:
        raise ValueError(mode)
    mass = weight.sum(dim=(-2, -1))
    pooled = (feature_map * weight).sum(dim=(-2, -1)) / mass.clamp_min(eps)
    return pooled, mass


def build_model(family: str, cfg: dict[str, Any], load_pretrained: bool = True):
    if family == "segresnet":
        from segresnet_model import CorrectedSegResNet
        return CorrectedSegResNet(cfg)
    if family == "deeplabv3plus_resnet50_imagenet":
        from deeplabv3plus_model import ImageNetDeepLabV3Plus
        return ImageNetDeepLabV3Plus(cfg, load_pretrained=load_pretrained)
    raise ValueError(f"Unknown model family: {family}")


def model_parameter_count(model: nn.Module) -> dict[str, int]:
    return {
        "total": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
    }

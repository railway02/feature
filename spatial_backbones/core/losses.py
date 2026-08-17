from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    probability = torch.sigmoid(logits)
    dimensions = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dimensions)
    denominator = probability.sum(dimensions) + target.sum(dimensions)
    return 1.0 - ((2.0 * intersection + eps) / (denominator + eps)).mean()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor, pos_weight: float, cfg):
    dice = soft_dice_loss(logits, target)
    weight = torch.tensor([float(pos_weight)], device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight)
    return float(cfg["loss"]["dice_weight"]) * dice + float(cfg["loss"]["bce_weight"]) * bce

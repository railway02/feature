from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def resize_weight(weight: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if weight.ndim == 3:
        weight = weight.unsqueeze(1)
    return F.interpolate(weight.float(), size=size, mode="bilinear", align_corners=False).clamp(0, 1)


def global_mean(features: torch.Tensor) -> torch.Tensor:
    return features.float().mean(dim=(-2, -1))


def weighted_mean(features: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    w = resize_weight(weight, features.shape[-2:])
    numerator = (features.float() * w).sum(dim=(-2, -1))
    denominator = w.sum(dim=(-2, -1)).clamp_min(eps)
    return numerator / denominator


def topk_abs_and_signed(
    features: torch.Tensor,
    vessel_probability: torch.Tensor,
    fraction: float = 0.10,
    threshold: float = 0.15,
    minimum_pixels: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight = resize_weight(vessel_probability, features.shape[-2:])
    abs_outputs, signed_outputs, fallback_outputs = [], [], []
    for batch_index in range(features.shape[0]):
        valid = weight[batch_index, 0] >= threshold
        fallback = int(valid.sum().item()) < minimum_pixels
        if fallback:
            valid = torch.ones_like(valid, dtype=torch.bool)
        selected = features[batch_index, :, valid].float()
        count = selected.shape[1]
        k = min(count, max(minimum_pixels, int(round(count * fraction))))
        positions = torch.topk(selected.abs(), k=k, dim=1, sorted=False).indices
        signed = torch.gather(selected, 1, positions)
        abs_outputs.append(signed.abs().mean(dim=1))
        signed_outputs.append(signed.mean(dim=1))
        fallback_outputs.append(float(fallback))
    return (
        torch.stack(abs_outputs),
        torch.stack(signed_outputs),
        torch.tensor(fallback_outputs, device=features.device),
    )


def pool_trajectory(sequence: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    batch, time, channels, height, width = sequence.shape
    flat = sequence.reshape(batch * time, channels, height, width)
    if weight is None:
        pooled = global_mean(flat)
    else:
        if weight.ndim == 3:
            weight = weight.unsqueeze(1)
        repeated = weight[:, None].expand(batch, time, *weight.shape[1:]).reshape(batch * time, *weight.shape[1:])
        pooled = weighted_mean(flat, repeated)
    return pooled.reshape(batch, time, channels)


def resample_trajectory(values: np.ndarray, length: int = 16) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError(f"Expected T,C trajectory, got {values.shape}")
    if len(values) == length:
        return values.astype(np.float32)
    old = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    new = np.linspace(0.0, 1.0, length, dtype=np.float32)
    output = np.empty((length, values.shape[1]), dtype=np.float32)
    for channel in range(values.shape[1]):
        output[:, channel] = np.interp(new, old, values[:, channel])
    return output


def build_embedding_bank(
    f4: torch.Tensor,
    f5: torch.Tensor,
    artery: torch.Tensor,
    vein: torch.Tensor,
    activity: torch.Tensor,
    fov: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
    artery = artery * fov
    vein = vein * fov
    vessel_max = torch.maximum(artery, vein)
    vessel_or = (1.0 - (1.0 - artery) * (1.0 - vein)).clamp(0, 1)
    active_vessel = (vessel_max * activity).clamp(0, 1)

    f5_abs, f5_signed, f5_fallback = topk_abs_and_signed(f5, vessel_max)
    f4_abs, f4_signed, f4_fallback = topk_abs_and_signed(f4, vessel_max)

    primary_blocks = {
        "f5_global_mean": global_mean(f5),
        "f5_vessel_mean": weighted_mean(f5, vessel_max),
        "f5_artery_mean": weighted_mean(f5, artery),
        "f5_vein_mean": weighted_mean(f5, vein),
        "f5_active_vessel_mean": weighted_mean(f5, active_vessel),
        "f5_vessel_top10_abs_magnitude": f5_abs,
        "f4_vessel_mean": weighted_mean(f4, vessel_max),
        "f4_artery_mean": weighted_mean(f4, artery),
        "f4_active_vessel_mean": weighted_mean(f4, active_vessel),
        "f4_vessel_top10_abs_magnitude": f4_abs,
    }
    auxiliary_blocks = {
        "f5_vessel_top10_signed_mean": f5_signed,
        "f4_vessel_top10_signed_mean": f4_signed,
        "f4_vein_mean": weighted_mean(f4, vein),
        "f5_vessel_or_mean": weighted_mean(f5, vessel_or),
        "f4_vessel_or_mean": weighted_mean(f4, vessel_or),
    }
    primary = torch.cat(list(primary_blocks.values()), dim=1)
    if primary.shape[1] != 5120:
        raise RuntimeError(f"Primary embedding is {primary.shape}, expected Bx5120")
    qc = {
        "f4_topk_fallback": float(f4_fallback.max().item()),
        "f5_topk_fallback": float(f5_fallback.max().item()),
    }
    return primary, auxiliary_blocks, qc

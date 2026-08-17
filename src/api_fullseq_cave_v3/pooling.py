from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

PRIMARY_BLOCKS = (
    "f5_global_mean", "f5_vessel_mean", "f5_artery_mean", "f5_vein_mean",
    "f5_active_vessel_mean", "f5_vessel_top10_abs_magnitude",
    "f4_vessel_mean", "f4_artery_mean", "f4_active_vessel_mean",
    "f4_vessel_top10_abs_magnitude",
)
AUXILIARY_BLOCKS = (
    "f5_vessel_top10_signed_mean", "f4_vessel_top10_signed_mean", "f4_vein_mean",
    "f5_vessel_union_mean", "f4_vessel_union_mean",
)
TRAJECTORY_REGIONS = ("global", "vessel", "artery", "vein", "active_vessel")
TRAJECTORY_SCALES = ("f4", "f5")


def embedding_feature_names() -> list[str]:
    return [f"{block}_{index:03d}" for block in PRIMARY_BLOCKS for index in range(512)]


def resize_weight(weight: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if weight.ndim == 3:
        weight = weight.unsqueeze(1)
    if weight.ndim != 4:
        raise ValueError(f"Weight must be BCHW, got {tuple(weight.shape)}")
    return F.interpolate(weight.float(), size=size, mode="bilinear", align_corners=False).clamp(0, 1)


def global_mean(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 4:
        raise ValueError(f"Features must be BCHW, got {tuple(features.shape)}")
    return features.float().mean(dim=(-2, -1))


def weighted_mean(features: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    w = resize_weight(weight, features.shape[-2:])
    denominator = w.sum(dim=(-2, -1))
    numerator = (features.float() * w).sum(dim=(-2, -1))
    fallback = denominator < eps
    result = numerator / denominator.clamp_min(eps)
    if fallback.any():
        result = torch.where(fallback, global_mean(features), result)
    return result


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
        count = int(selected.shape[1])
        k = min(count, max(minimum_pixels, int(round(count * fraction))))
        positions = torch.topk(selected.abs(), k=k, dim=1, sorted=False).indices
        signed = torch.gather(selected, 1, positions)
        abs_outputs.append(signed.abs().mean(dim=1))
        signed_outputs.append(signed.mean(dim=1))
        fallback_outputs.append(float(fallback))
    return (
        torch.stack(abs_outputs), torch.stack(signed_outputs),
        torch.tensor(fallback_outputs, device=features.device),
    )


def pool_trajectory(sequence: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    if sequence.ndim != 5:
        raise ValueError(f"Sequence must be BTCHW, got {tuple(sequence.shape)}")
    batch, time, channels, height, width = sequence.shape
    flat = sequence.reshape(batch * time, channels, height, width)
    if weight is None:
        pooled = global_mean(flat)
    else:
        if weight.ndim == 3:
            weight = weight.unsqueeze(1)
        repeated = weight[:, None].expand(batch, time, *weight.shape[1:]).reshape(
            batch * time, *weight.shape[1:]
        )
        pooled = weighted_mean(flat, repeated)
    return pooled.reshape(batch, time, channels)


def resample_trajectory(values: np.ndarray, length: int = 16) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or len(values) < 1:
        raise ValueError(f"Expected nonempty T,C trajectory, got {values.shape}")
    if len(values) == length:
        return values.copy()
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
    artery = artery.float().clamp(0, 1) * fov
    vein = vein.float().clamp(0, 1) * fov
    vessel = torch.maximum(artery, vein)
    vessel_union = (1.0 - (1.0 - artery) * (1.0 - vein)).clamp(0, 1)
    active_vessel = (vessel * activity.float().clamp(0, 1)).clamp(0, 1)

    f5_abs, f5_signed, f5_fallback = topk_abs_and_signed(f5, vessel)
    f4_abs, f4_signed, f4_fallback = topk_abs_and_signed(f4, vessel)
    primary_blocks = {
        "f5_global_mean": global_mean(f5),
        "f5_vessel_mean": weighted_mean(f5, vessel),
        "f5_artery_mean": weighted_mean(f5, artery),
        "f5_vein_mean": weighted_mean(f5, vein),
        "f5_active_vessel_mean": weighted_mean(f5, active_vessel),
        "f5_vessel_top10_abs_magnitude": f5_abs,
        "f4_vessel_mean": weighted_mean(f4, vessel),
        "f4_artery_mean": weighted_mean(f4, artery),
        "f4_active_vessel_mean": weighted_mean(f4, active_vessel),
        "f4_vessel_top10_abs_magnitude": f4_abs,
    }
    if tuple(primary_blocks) != PRIMARY_BLOCKS:
        raise AssertionError("Primary block order changed")
    auxiliary = {
        "f5_vessel_top10_signed_mean": f5_signed,
        "f4_vessel_top10_signed_mean": f4_signed,
        "f4_vein_mean": weighted_mean(f4, vein),
        "f5_vessel_union_mean": weighted_mean(f5, vessel_union),
        "f4_vessel_union_mean": weighted_mean(f4, vessel_union),
    }
    primary = torch.cat([primary_blocks[name] for name in PRIMARY_BLOCKS], dim=1)
    if primary.shape[1] != 5120 or not torch.isfinite(primary).all():
        raise FloatingPointError(f"Invalid primary embedding shape/values: {tuple(primary.shape)}")
    qc = {
        "f4_topk_fallback": float(f4_fallback.max().item()),
        "f5_topk_fallback": float(f5_fallback.max().item()),
        "artery_weight_sum": float(artery.sum().item()),
        "vein_weight_sum": float(vein.sum().item()),
        "vessel_weight_sum": float(vessel.sum().item()),
    }
    return primary, auxiliary, qc

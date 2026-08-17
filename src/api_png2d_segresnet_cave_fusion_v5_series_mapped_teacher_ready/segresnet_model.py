from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_segresnet(cfg: dict[str, Any]) -> nn.Module:
    from monai.networks.nets import SegResNet

    c = cfg["segresnet"]
    return SegResNet(
        spatial_dims=int(c["spatial_dims"]),
        in_channels=int(c["in_channels"]),
        out_channels=int(c["out_channels"]),
        init_filters=int(c["init_filters"]),
        blocks_down=tuple(int(x) for x in c["blocks_down"]),
        blocks_up=tuple(int(x) for x in c["blocks_up"]),
        dropout_prob=c.get("dropout_prob", None),
    )


def _resolve_state_dict(raw, state_key: str):
    obj = raw
    if state_key:
        for part in state_key.split("."):
            obj = obj[part]
        return obj

    if isinstance(raw, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            value = raw.get(key)
            if isinstance(value, dict) and value:
                return value
    return raw


def maybe_load_external_checkpoint(model, cfg):
    path = str(cfg["spatial"].get("external_checkpoint", "") or "").strip()
    if not path:
        return {"used": False}

    raw = torch.load(path, map_location="cpu")
    state = _resolve_state_dict(
        raw,
        str(cfg["spatial"].get("external_checkpoint_state_key", "") or ""),
    )
    if not isinstance(state, dict) or not state:
        raise TypeError("External checkpoint does not resolve to a non-empty state_dict")
    if not all(torch.is_tensor(v) for v in state.values()):
        raise TypeError("External checkpoint state_dict contains non-tensor values")

    for prefix in ("module.", "model.", "net.", "segresnet."):
        if state and all(str(k).startswith(prefix) for k in state):
            state = {str(k)[len(prefix):]: v for k, v in state.items()}

    result = model.load_state_dict(
        state,
        strict=bool(cfg["spatial"].get("external_checkpoint_strict", False)),
    )
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    if not bool(cfg["spatial"].get("external_checkpoint_allow_partial", False)) and (missing or unexpected):
        raise RuntimeError(f"External checkpoint is not architecture-compatible; missing={missing}, unexpected={unexpected}")
    return {
        "used": True,
        "path": path,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


def dice_loss(logits, target, eps=1e-6):
    p = torch.sigmoid(logits)
    dims = tuple(range(1, p.ndim))
    inter = (p * target).sum(dims)
    denom = p.sum(dims) + target.sum(dims)
    return 1.0 - ((2 * inter + eps) / (denom + eps)).mean()


def segmentation_loss(logits, target, pos_weight, cfg):
    d = dice_loss(logits, target)
    pw = torch.tensor([float(pos_weight)], device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)

    return (
        float(cfg["segresnet"]["dice_weight"]) * d
        + float(cfg["segresnet"]["bce_weight"]) * bce
    )


@torch.no_grad()
def dice_metric(logits, target, eps=1e-6):
    pred = (torch.sigmoid(logits) >= 0.5).float()
    dims = tuple(range(1, pred.ndim))
    inter = (pred * target).sum(dims)
    denom = pred.sum(dims) + target.sum(dims)
    return ((2 * inter + eps) / (denom + eps)).mean()


def encode(model, x):
    if hasattr(model, "encode") and callable(model.encode):
        result = model.encode(x)
        if isinstance(result, tuple):
            return result[0], result[1]
        return result, []

    # Older MONAI fallback.
    x = model.convInit(x)
    down = []
    for layer in model.down_layers:
        x = layer(x)
        down.append(x)
    return x, down


def encode_and_decode(model, x):
    fmap, down = encode(model, x)

    if hasattr(model, "decode") and callable(model.decode):
        # MONAI SegResNet.forward reverses down_x before decode.
        logits = model.decode(fmap, list(reversed(down)))
    else:
        # Robust but more expensive fallback.
        logits = model(x)

    return fmap, logits


def global_pool(fmap):
    return F.adaptive_avg_pool2d(fmap, 1).flatten(1)


def mask_pool(fmap, mask, resize_mode: str, eps: float, require_nonzero: bool = False):
    if resize_mode == "bilinear":
        weight = F.interpolate(mask, size=fmap.shape[-2:], mode="bilinear", align_corners=False)
    elif resize_mode in {"nearest", "area"}:
        weight = F.interpolate(mask, size=fmap.shape[-2:], mode=resize_mode)
    else:
        raise ValueError(f"Unknown ROI resize_mode={resize_mode}")

    numerator = (fmap * weight).sum(dim=(-2, -1))
    mass = weight.sum(dim=(-2, -1))
    if require_nonzero and torch.any(mass <= eps):
        bad = torch.nonzero(mass.squeeze(1) <= eps, as_tuple=False).flatten().tolist()
        raise RuntimeError(f"ROI vanished at feature-map scale for batch entries {bad}")
    return numerator / mass.clamp_min(eps), mass

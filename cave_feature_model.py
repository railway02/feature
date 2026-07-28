from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn


@dataclass
class CAVEOutputs:
    logits: torch.Tensor
    f4_sequence: torch.Tensor
    f5_sequence: torch.Tensor
    f4_last: torch.Tensor
    f5_last: torch.Tensor


class CAVEFeatureWrapper(nn.Module):
    """Wrap the official CAVE TemporalUNet without changing checkpoint keys.

    The wrapped model stays untouched. This class explicitly reproduces the
    official forward pass and additionally returns the ConvGRU f4/f5 sequence
    tensors and their final hidden maps.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> CAVEOutputs:
        # x: B,T,C,H,W
        if x.ndim != 5:
            raise ValueError(f"Expected B,T,C,H,W, got {tuple(x.shape)}")
        b, t, c, h, w = x.shape

        flat = x.reshape(b * t, c, h, w)
        x1 = self.model.inc(flat)
        _, c1, h1, w1 = x1.shape
        x1t = x1.reshape(b, t, c1, h1, w1)
        f1_seq, _ = self.model.rnn_inc(x1t)
        f1 = f1_seq[:, -1]

        x2 = self.model.down1(x1)
        _, c2, h2, w2 = x2.shape
        x2t = x2.reshape(b, t, c2, h2, w2)
        f2_seq, _ = self.model.rnn1(x2t)
        f2 = f2_seq[:, -1]

        x3 = self.model.down2(x2)
        _, c3, h3, w3 = x3.shape
        x3t = x3.reshape(b, t, c3, h3, w3)
        f3_seq, _ = self.model.rnn2(x3t)
        f3 = f3_seq[:, -1]

        x4 = self.model.down3(x3)
        _, c4, h4, w4 = x4.shape
        x4t = x4.reshape(b, t, c4, h4, w4)
        f4_seq, _ = self.model.rnn3(x4t)
        f4 = f4_seq[:, -1]

        x5 = self.model.down4(x4)
        _, c5, h5, w5 = x5.shape
        x5t = x5.reshape(b, t, c5, h5, w5)
        f5_seq, _ = self.model.rnn4(x5t)
        f5 = f5_seq[:, -1]

        decoded = self.model.up1(f5, f4)
        decoded = self.model.up2(decoded, f3)
        decoded = self.model.up3(decoded, f2)
        decoded = self.model.up4(decoded, f1)
        logits = self.model.outc(decoded)

        return CAVEOutputs(
            logits=logits,
            f4_sequence=f4_seq,
            f5_sequence=f5_seq,
            f4_last=f4,
            f5_last=f5,
        )


def strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if state and all(key.startswith("module.") for key in state):
        return {key[len("module."):]: value for key, value in state.items()}
    return state


def load_cave_av_convgru(
    repo_root: Path,
    checkpoint: Path,
    device: torch.device,
) -> tuple[nn.Module, CAVEFeatureWrapper]:
    import sys

    sys.path.insert(0, str(repo_root))
    from unet import TemporalUNet, ConvGRU  # type: ignore

    model = TemporalUNet(
        ConvGRU,
        n_channels=1,
        n_classes=2,
        bilinear=True,
        kernel_size=(3, 3),
        num_layers=2,
    )
    raw = torch.load(checkpoint, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    if not isinstance(raw, dict):
        raise TypeError(f"Unexpected checkpoint type: {type(raw)!r}")
    state = strip_module_prefix(raw)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch. missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    wrapper = CAVEFeatureWrapper(model).to(device).eval()
    return model, wrapper


@torch.inference_mode()
def verify_wrapper_equivalence(
    official_model: nn.Module,
    wrapper: CAVEFeatureWrapper,
    sample: torch.Tensor,
    atol: float = 1e-5,
) -> float:
    official = official_model(sample)
    wrapped = wrapper(sample).logits
    max_abs = float((official - wrapped).abs().max().item())
    if max_abs > atol:
        raise AssertionError(
            f"Wrapper changed official logits: max_abs={max_abs:.3e} > {atol:.3e}"
        )
    return max_abs

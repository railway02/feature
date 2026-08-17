from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn


@dataclass
class CAVEForward:
    logits: torch.Tensor
    f4_sequence: torch.Tensor
    f5_sequence: torch.Tensor

    @property
    def f4_last(self) -> torch.Tensor:
        return self.f4_sequence[:, -1]

    @property
    def f5_last(self) -> torch.Tensor:
        return self.f5_sequence[:, -1]


def _unwrap_sequence(output: Any) -> torch.Tensor:
    sequence = output[0] if isinstance(output, tuple) else output
    if isinstance(sequence, list):
        sequence = sequence[-1]
    if not isinstance(sequence, torch.Tensor) or sequence.ndim != 5:
        raise TypeError(f"Unexpected ConvGRU output: {type(sequence)!r}")
    return sequence


class CAVEHookExtractor:
    """Run the untouched official forward and capture rnn3/rnn4 outputs."""

    def __init__(self, model: nn.Module):
        self.model = model
        self._captured: dict[str, torch.Tensor] = {}
        self._handles = [
            model.rnn3.register_forward_hook(self._hook("f4")),
            model.rnn4.register_forward_hook(self._hook("f5")),
        ]

    def _hook(self, name: str):
        def callback(_module, _inputs, output):
            self._captured[name] = _unwrap_sequence(output)
        return callback

    @torch.inference_mode()
    def __call__(self, x: torch.Tensor) -> CAVEForward:
        self._captured.clear()
        logits = self.model(x)
        if set(self._captured) != {"f4", "f5"}:
            raise RuntimeError(f"Failed to capture CAVE features: {self._captured.keys()}")
        return CAVEForward(logits, self._captured["f4"], self._captured["f5"])

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key.removeprefix("module."): value for key, value in state.items()}


def load_checkpoint_state(path: Path) -> Dict[str, torch.Tensor]:
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(path, map_location="cpu")
    if isinstance(raw, nn.Module):
        state = raw.state_dict()
    elif isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
    elif isinstance(raw, dict):
        state = raw
    else:
        raise TypeError(f"Unsupported checkpoint object: {type(raw)!r}")
    return strip_module_prefix(state)


def load_cave_model(repo_root: Path, checkpoint: Path, device: torch.device) -> tuple[nn.Module, CAVEHookExtractor]:
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
    model.load_state_dict(load_checkpoint_state(checkpoint), strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, CAVEHookExtractor(model)


def git_commit(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()

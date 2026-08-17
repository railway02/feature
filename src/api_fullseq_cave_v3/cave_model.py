from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from common import sha256_tree


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
    """Execute the untouched official forward and capture rnn3/rnn4 outputs."""

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
    def __call__(self, tensor: torch.Tensor) -> CAVEForward:
        self._captured.clear()
        logits = self.model(tensor)
        if set(self._captured) != {"f4", "f5"}:
            raise RuntimeError(f"Failed to capture CAVE features: {sorted(self._captured)}")
        f4 = self._captured["f4"]
        f5 = self._captured["f5"]
        if f4.shape[:2] != tensor.shape[:2] or f5.shape[:2] != tensor.shape[:2]:
            raise AssertionError(
                f"Temporal feature length mismatch: input={tuple(tensor.shape)} "
                f"f4={tuple(f4.shape)} f5={tuple(f5.shape)}"
            )
        return CAVEForward(logits=logits, f4_sequence=f4, f5_sequence=f5)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.removeprefix("module."): value for key, value in state.items()}


def load_checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(path, map_location="cpu")
    if isinstance(raw, nn.Module):
        state = raw.state_dict()
    elif isinstance(raw, dict) and isinstance(raw.get("state_dict"), dict):
        state = raw["state_dict"]
    elif isinstance(raw, dict):
        state = raw
    else:
        raise TypeError(f"Unsupported checkpoint object: {type(raw)!r}")
    if not state:
        raise AssertionError("Checkpoint state is empty")
    return strip_module_prefix(state)


def git_commit(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()


def git_is_dirty(repo_root: Path) -> bool:
    output = subprocess.check_output(
        ["git", "-C", str(repo_root), "status", "--porcelain"], text=True
    )
    return bool(output.strip())


def cave_code_tree_hash(repo_root: Path) -> str:
    return sha256_tree(repo_root / "unet", suffixes=(".py",))


def load_cave_model(
    repo_root: Path,
    checkpoint: Path,
    device: torch.device,
) -> tuple[nn.Module, CAVEHookExtractor]:
    if not repo_root.is_dir() or not (repo_root / "unet").is_dir():
        raise FileNotFoundError(f"Invalid CAVE repository: {repo_root}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    sys.path.insert(0, str(repo_root))
    try:
        from unet import TemporalUNet, ConvGRU  # type: ignore
    finally:
        # Keep imported modules alive, but avoid changing future import resolution.
        if sys.path and sys.path[0] == str(repo_root):
            sys.path.pop(0)

    model = TemporalUNet(
        ConvGRU,
        n_channels=1,
        n_classes=2,
        bilinear=True,
        kernel_size=(3, 3),
        num_layers=2,
    )
    state = load_checkpoint_state(checkpoint)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise AssertionError(f"Checkpoint incompatibility: {incompatible}")
    model.to(device=device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, CAVEHookExtractor(model)

"""Quality-gated residual fusion for replacing the old CAVE branch."""
from __future__ import annotations


def _torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as e:
        raise ImportError("PyTorch is required; reuse the existing PredROI environment.") from e
    return torch, nn


def build_registration_residual_fusion(z2d_dim: int, zreg_dim: int, hidden: int = 128,
                                       init_gate_logit: float = -2.5, dropout: float = 0.1):
    torch, nn = _torch()

    class RegistrationResidualFusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.psi = nn.Sequential(
                nn.Linear(zreg_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, z2d_dim),
                nn.LayerNorm(z2d_dim),
            )
            self.gate_logit = nn.Parameter(torch.tensor(float(init_gate_logit)))

        def forward(self, z2d, zreg, qreg):
            if z2d.ndim != 2 or zreg.ndim != 2:
                raise ValueError("z2d and zreg must be [B,D]")
            if z2d.shape[0] != zreg.shape[0] or z2d.shape[1] != z2d_dim or zreg.shape[1] != zreg_dim:
                raise ValueError(
                    f"shape mismatch: z2d={tuple(z2d.shape)} expected [B,{z2d_dim}], "
                    f"zreg={tuple(zreg.shape)} expected [B,{zreg_dim}]"
                )
            # A proper Train-fitted preprocessor should already have removed NaN/Inf.
            # nan_to_num is a final safety net because 0 * NaN is still NaN.
            zreg = torch.nan_to_num(zreg, nan=0.0, posinf=0.0, neginf=0.0)
            if qreg.ndim == 1:
                qreg = qreg[:, None]
            if qreg.ndim != 2 or qreg.shape != (z2d.shape[0], 1):
                raise ValueError(f"qreg must be [B] or [B,1], got {tuple(qreg.shape)}")
            qreg = torch.nan_to_num(qreg, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
            g = torch.sigmoid(self.gate_logit)
            delta = self.psi(zreg)
            zfinal = z2d + qreg * g * delta
            return zfinal, {"gate": g.detach(), "mean_qreg": qreg.detach().mean()}

    return RegistrationResidualFusion()

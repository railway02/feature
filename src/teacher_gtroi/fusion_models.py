from __future__ import annotations

import torch
import torch.nn as nn


class FeatureProjection(nn.Module):
    """
    Teacher-aligned:
      Linear(input_dim, 256)
      LayerNorm(256)
      GELU()
      Dropout(0.2)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SpatialTemporalFusion(nn.Module):
    """
    Teacher-aligned bidirectional conditional gating:
      a2D = sigmoid(f2D([z2D,zT]))
      aT  = sigmoid(fT([zT,z2D]))

      z2D_hat = (1-a2D) z2D + a2D phi_T_to_2D(zT)
      zT_hat  = (1-aT) zT  + aT  phi_2D_to_T(z2D)

      hmain = [z2D_hat, zT_hat, product, abs-difference]
      1024 -> 256
    """
    def __init__(self, hidden_dim=256, fusion_mid_dim=512, dropout=0.2):
        super().__init__()
        pair = hidden_dim * 2

        self.gate_2d = nn.Sequential(
            nn.Linear(pair, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.gate_t = nn.Sequential(
            nn.Linear(pair, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.phi_t_to_2d = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.phi_2d_to_t = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, fusion_mid_dim),
            nn.LayerNorm(fusion_mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_mid_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, z2d, zt):
        a2d = self.gate_2d(torch.cat([z2d, zt], dim=-1))
        at = self.gate_t(torch.cat([zt, z2d], dim=-1))

        z2d_hat = (1.0 - a2d) * z2d + a2d * self.phi_t_to_2d(zt)
        zt_hat = (1.0 - at) * zt + at * self.phi_2d_to_t(z2d)

        hmain = torch.cat(
            [
                z2d_hat,
                zt_hat,
                z2d_hat * zt_hat,
                torch.abs(z2d_hat - zt_hat),
            ],
            dim=-1,
        )
        zmain = self.fusion(hmain)

        return {
            "zmain": zmain,
            "hmain": hmain,
            "gate_2d": a2d,
            "gate_t": at,
        }


class OutcomeModel(nn.Module):
    def __init__(
        self,
        mode: str,
        spatial_dim: int,
        temporal_dim: int,
        hidden_dim=256,
        fusion_mid_dim=512,
        dropout=0.2,
    ):
        super().__init__()
        allowed = {"cave_only", "spatial_only", "concat", "interaction", "gated_interaction"}
        if mode not in allowed:
            raise ValueError(f"Unknown mode={mode}")
        self.mode = mode

        self.proj_2d = (
            FeatureProjection(spatial_dim, hidden_dim, dropout)
            if mode != "cave_only"
            else None
        )
        self.proj_t = (
            FeatureProjection(temporal_dim, hidden_dim, dropout)
            if mode != "spatial_only"
            else None
        )

        self.gated = (
            SpatialTemporalFusion(
                hidden_dim=hidden_dim,
                fusion_mid_dim=fusion_mid_dim,
                dropout=dropout,
            )
            if mode == "gated_interaction"
            else None
        )

        if mode in {"cave_only", "spatial_only", "gated_interaction"}:
            head_in = hidden_dim
        elif mode == "concat":
            head_in = hidden_dim * 2
        else:
            head_in = hidden_dim * 4

        if mode == "interaction":
            self.interaction_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 4, fusion_mid_dim),
                nn.LayerNorm(fusion_mid_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_mid_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            head_in = hidden_dim
        else:
            self.interaction_fusion = None

        if mode == "concat":
            self.concat_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            head_in = hidden_dim
        else:
            self.concat_fusion = None

        # Teacher diagram only specifies lmain=fmain(zmain).
        # Keep the outcome head deliberately small.
        self.main_head = nn.Linear(head_in, 1)

    def forward(self, spatial=None, temporal=None):
        extra = {}

        if self.mode == "cave_only":
            zmain = self.proj_t(temporal)

        elif self.mode == "spatial_only":
            zmain = self.proj_2d(spatial)

        else:
            z2d = self.proj_2d(spatial)
            zt = self.proj_t(temporal)

            if self.mode == "concat":
                zmain = self.concat_fusion(torch.cat([z2d, zt], dim=-1))

            elif self.mode == "interaction":
                h = torch.cat(
                    [z2d, zt, z2d * zt, torch.abs(z2d - zt)],
                    dim=-1,
                )
                zmain = self.interaction_fusion(h)

            else:
                fused = self.gated(z2d, zt)
                zmain = fused["zmain"]
                extra = {
                    "gate_2d": fused["gate_2d"],
                    "gate_t": fused["gate_t"],
                    "hmain": fused["hmain"],
                }

        logit = self.main_head(zmain)
        return {"logit": logit, "zmain": zmain, **extra}

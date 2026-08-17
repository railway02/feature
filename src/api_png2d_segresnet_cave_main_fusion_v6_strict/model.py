"""Strict 2D--CAVE main-modality fusion model.

The public forward contract deliberately exposes the teacher-specified
intermediate tensors; this makes a saved fusion representation auditable and
prevents callers from silently substituting a different gating formulation.
"""
from __future__ import annotations

import torch
from torch import nn


class FeatureProjection(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MainFusionModel(nn.Module):
    """Teacher scheme part IV.2: bidirectional gated 2D--temporal fusion."""
    def __init__(
        self,
        spatial_dim: int = 1024,
        temporal_dim: int = 10240,
        hidden_dim: int = 256,
        fusion_mid_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.hidden_dim = hidden_dim
        self.fusion_mid_dim = fusion_mid_dim
        self.dropout = dropout
        self.proj_2d = FeatureProjection(spatial_dim, hidden_dim, dropout)
        self.proj_time = FeatureProjection(temporal_dim, hidden_dim, dropout)
        pair_dim = hidden_dim * 2
        self.spatial_gate_net = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.temporal_gate_net = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.phi_time_to_2d = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.phi_2d_to_time = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
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
        self.main_head = nn.Linear(hidden_dim, 1)

    def config(self) -> dict[str, int | float]:
        return {
            "spatial_dim": self.spatial_dim,
            "temporal_dim": self.temporal_dim,
            "hidden_dim": self.hidden_dim,
            "fusion_mid_dim": self.fusion_mid_dim,
            "dropout": self.dropout,
        }

    def forward(self, z_2d_raw: torch.Tensor, z_time_raw: torch.Tensor) -> dict[str, torch.Tensor]:
        z_2d = self.proj_2d(z_2d_raw)
        z_time = self.proj_time(z_time_raw)
        spatial_gate = torch.sigmoid(self.spatial_gate_net(torch.cat([z_2d, z_time], dim=-1)))
        temporal_gate = torch.sigmoid(self.temporal_gate_net(torch.cat([z_time, z_2d], dim=-1)))
        z_2d_interacted = (1.0 - spatial_gate) * z_2d + spatial_gate * self.phi_time_to_2d(z_time)
        z_time_interacted = (1.0 - temporal_gate) * z_time + temporal_gate * self.phi_2d_to_time(z_2d)
        h_main = torch.cat(
            [
                z_2d_interacted,
                z_time_interacted,
                z_2d_interacted * z_time_interacted,
                torch.abs(z_2d_interacted - z_time_interacted),
            ],
            dim=-1,
        )
        z_main = self.fusion(h_main)
        main_logit = self.main_head(z_main)
        main_prob = torch.sigmoid(main_logit)
        return {
            "z_2d": z_2d,
            "z_time": z_time,
            "spatial_gate": spatial_gate,
            "temporal_gate": temporal_gate,
            "z_2d_interacted": z_2d_interacted,
            "z_time_interacted": z_time_interacted,
            "h_main": h_main,
            "z_main": z_main,
            "main_logit": main_logit,
            "main_prob": main_prob,
        }

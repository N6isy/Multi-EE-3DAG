"""Lightweight point-wise MLP backbone used by the initial PointNet baseline."""

from __future__ import annotations

import torch
from torch import nn


class PointMLPBackbone(nn.Module):
    """Per-point MLP encoder returning `[B, N, C]` point features."""

    def __init__(self, *, input_channels: int = 3, hidden_dim: int = 128) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.out_channels = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.net(points)

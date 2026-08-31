"""Residual MLP (ResMLP) that estimates the diffusion distance of a cube state
to the solved state (scalar regression)."""
from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, n1: int, n2: int):
        super().__init__()
        self.fc1 = nn.Linear(n1, n2)
        self.bn1 = nn.BatchNorm1d(n2)
        self.fc2 = nn.Linear(n2, n1)
        self.bn2 = nn.BatchNorm1d(n1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        h = self.act(self.bn1(self.fc1(x)))
        h = self.bn2(self.fc2(h))
        return self.act(h + identity)


class ResMLP(nn.Module):
    """ResMLP: input linear -> N_res residual blocks -> scalar head.

    Matches the user-specified architecture:
      input_dim=324, n1=1024, n2=768, n_res=4, output_dim=1.
    """

    def __init__(self, input_dim: int = 324, n1: int = 1024, n2: int = 768,
                 n_res: int = 4, output_dim: int = 1):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, n1),
            nn.BatchNorm1d(n1),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([ResBlock(n1, n2) for _ in range(n_res)])
        self.head = nn.Linear(n1, output_dim)

    def forward(self, x):
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(cfg):
    model = ResMLP(input_dim=cfg.input_dim, n1=cfg.n1, n2=cfg.n2,
                   n_res=cfg.n_res, output_dim=cfg.output_dim)
    return model
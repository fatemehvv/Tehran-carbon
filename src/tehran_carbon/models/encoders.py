"""Per-modality encoders (small MLPs). Inputs are standardised features of shape (..., F)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as Fn


class MLPEncoder(nn.Module):
    """Linear + SiLU, then (layers - 1) residual blocks (Linear, SiLU, dropout)."""

    def __init__(self, f_in: int, hidden: int = 32, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(f_in, hidden)
        self.blocks = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(max(layers - 1, 0))])
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = Fn.silu(self.inp(x))
        for b in self.blocks:
            h = h + self.drop(Fn.silu(b(h)))
        return h

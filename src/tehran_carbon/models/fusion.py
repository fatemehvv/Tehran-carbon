"""Fusion of the input branches (traffic, energy, pollution, context). Each branch is a dict entry (..., F_m).

early  concatenate all features, one encoder                               (data-level fusion)
late   one encoder + prediction per branch, learned weight per branch       (decision-level; learned counterpart of
                                                                             the Dempster-Shafer reliabilities of RQ1 v2)
gated  one encoder per branch, softmax gate over branches for every sample  (feature-level, attention over domains;
                                                                             the gate weights show which domain
                                                                             matters where and when)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..registry import Registry
from .encoders import MLPEncoder

FUSIONS = Registry("fusion")


class Fusion(nn.Module):
    level = "feature"            # "feature": forward -> (..., d); "decision": forward -> (M, ..., d), then combine()

    def __init__(self, dims: dict[str, int]):
        super().__init__()
        self.names = list(dims)

    def weights(self) -> torch.Tensor | None:
        """Branch weights of the last forward pass ((..., M) for gated, (M,) for late), or None."""
        return None


@FUSIONS.register("early")
class EarlyFusion(Fusion):
    def __init__(self, dims, hidden=32, layers=2, dropout=0.1):
        super().__init__(dims)
        self.enc = MLPEncoder(sum(dims.values()), hidden, layers, dropout)

    def forward(self, xs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.enc(torch.cat([xs[m] for m in self.names], -1))


@FUSIONS.register("late")
class LateFusion(Fusion):
    level = "decision"

    def __init__(self, dims, hidden=32, layers=2, dropout=0.1):
        super().__init__(dims)
        self.enc = nn.ModuleDict({m: MLPEncoder(f, hidden, layers, dropout) for m, f in dims.items()})
        self.logit = nn.Parameter(torch.zeros(len(dims)))

    def forward(self, xs):
        return torch.stack([self.enc[m](xs[m]) for m in self.names])

    def combine(self, preds: torch.Tensor) -> torch.Tensor:
        """preds (M, ...) -> weighted mean over branches."""
        w = torch.softmax(self.logit, 0)
        return (w.view(-1, *[1] * (preds.dim() - 1)) * preds).sum(0)

    def weights(self):
        return torch.softmax(self.logit, 0).detach()


@FUSIONS.register("gated")
class GatedFusion(Fusion):
    def __init__(self, dims, hidden=32, layers=2, dropout=0.1):
        super().__init__(dims)
        self.enc = nn.ModuleDict({m: MLPEncoder(f, hidden, layers, dropout) for m, f in dims.items()})
        self.gate = nn.Linear(hidden * len(dims), len(dims))
        self._w = None

    def forward(self, xs):
        hs = [self.enc[m](xs[m]) for m in self.names]
        a = torch.softmax(self.gate(torch.cat(hs, -1)), -1)
        self._w = a.detach()
        return sum(a[..., i:i + 1] * h for i, h in enumerate(hs))

    def weights(self):
        return self._w

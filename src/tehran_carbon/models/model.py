"""CarbonModel = fusion(branch encoders) -> backbone -> linear head; predicts the target z of every node and month.

The parts are chosen by name in the config (model.fusion.type, model.backbone.type): RQ1 varies the fusion with no
backbone, RQ2 varies the backbone on top of the RQ1 fusion.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .backbones import BACKBONES
from .fusion import FUSIONS


class CarbonModel(nn.Module):
    def __init__(self, dims: dict[str, int], mcfg: dict):
        super().__init__()
        e, b = mcfg["encoder"], mcfg["backbone"]
        hidden = int(e["hidden"])
        self.fusion = FUSIONS.get(mcfg["fusion"]["type"])(dims, hidden, int(e["layers"]), float(e["dropout"]))
        self.backbone = BACKBONES.get(b["type"])(hidden, **{k: v for k, v in b.items() if k != "type"})
        n_heads = len(dims) if self.fusion.level == "decision" else 1
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(n_heads)])

    def forward(self, xs: dict[str, torch.Tensor], graph=None) -> torch.Tensor:
        """xs: branch -> (T, N, F_m) standardised features; returns (T, N)."""
        h = self.fusion(xs)
        if self.fusion.level == "decision":
            preds = torch.stack([head(self.backbone(hm, graph)).squeeze(-1) for head, hm in zip(self.heads, h)])
            return self.fusion.combine(preds)
        return self.heads[0](self.backbone(h, graph)).squeeze(-1)

    def load_from(self, state: dict, modules: list[str], freeze: bool) -> list[str]:
        """Copy the weights of the named sub-modules (e.g. ['fusion']) from a saved state dict (transfer between
        experiments: RQ1 encoders -> RQ2 GNN). Returns the copied keys."""
        own = self.state_dict()
        take = {k: v for k, v in state.items()
                if any(k == m or k.startswith(m + ".") for m in modules) and k in own and own[k].shape == v.shape}
        if not take:
            raise ValueError(f"no matching weights for modules {modules} (different features or sizes?)")
        self.load_state_dict(take, strict=False)
        if freeze:
            for name, p in self.named_parameters():
                if name in take:
                    p.requires_grad_(False)
        return sorted(take)

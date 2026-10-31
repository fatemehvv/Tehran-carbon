"""Spatial backbones on the fused node embeddings h (T, N, d), with PyTorch Geometric layers.

none  no message passing (control: the same network without a graph)
sage  multi-scale GraphSAGE: h <- h + act(sum_s SAGE_s(h, edges at dilation s) [+ W_od OD(h)])
gcn   multi-scale GCN (Kipf & Welling normalisation per scale), same optional OD channel
gat   GATv2 over the union of all scales; edge features = scale (one-hot) and, if edge_direction, the unit
      direction of the edge, so attention can weigh near/far and upwind/downwind neighbours differently
The OD channel (od.enabled) adds travel-demand links between districts (graph.od, data/graphs.py: ODOperator).
Months are processed as T disjoint copies of the graph (GridGraph.batched).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from ..registry import Registry

BACKBONES = Registry("backbone")


@BACKBONES.register("none")
class NoGraph(nn.Module):
    needs_graph = False

    def __init__(self, hidden: int, **_):
        super().__init__()

    def forward(self, h: torch.Tensor, graph=None) -> torch.Tensor:
        return h


class _MultiScale(nn.Module):
    needs_graph = True

    def __init__(self, hidden: int, layers: int, scales, dropout: float, od: dict | None = None, **_):
        super().__init__()
        self.convs = nn.ModuleList([nn.ModuleList([self.make_conv(hidden) for _ in scales]) for _ in range(layers)])
        self.od = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(layers)]) \
            if (od or {}).get("enabled") else None
        self.drop = nn.Dropout(dropout)

    def make_conv(self, hidden: int) -> nn.Module:
        raise NotImplementedError

    def forward(self, h, graph):
        T, N, d = h.shape
        per_scale = graph.batched(T, h.device)[0]
        if self.od is not None and graph.od is None:
            raise ValueError("od.enabled but the graph has no OD operator")
        x = h.reshape(T * N, d)
        for i, convs in enumerate(self.convs):
            m = sum(conv(x, e) for conv, e in zip(convs, per_scale))
            if self.od is not None:
                m = m + self.od[i](graph.od.to(h.device)(x.view(T, N, d))).reshape(T * N, d)
            x = x + self.drop(Fn.silu(m))
        return x.view(T, N, d)


@BACKBONES.register("sage")
class MultiScaleSAGE(_MultiScale):
    def make_conv(self, hidden):
        from torch_geometric.nn import SAGEConv
        return SAGEConv(hidden, hidden, aggr="mean")


@BACKBONES.register("gcn")
class MultiScaleGCN(_MultiScale):
    def make_conv(self, hidden):
        from torch_geometric.nn import GCNConv
        return GCNConv(hidden, hidden, add_self_loops=True, normalize=True, cached=False)


@BACKBONES.register("gat")
class MultiScaleGAT(nn.Module):
    needs_graph = True

    def __init__(self, hidden: int, layers: int, scales, heads: int, dropout: float, edge_direction: bool = False, **_):
        super().__init__()
        from torch_geometric.nn import GATv2Conv
        if hidden % heads:
            raise ValueError(f"hidden ({hidden}) must be divisible by heads ({heads})")
        self.n_scales, self.direction = len(scales), bool(edge_direction)
        edge_dim = self.n_scales + (2 if self.direction else 0)
        self.convs = nn.ModuleList([GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=edge_dim,
                                              add_self_loops=True, fill_value="mean") for _ in range(layers)])
        self.drop = nn.Dropout(dropout)

    def forward(self, h, graph):
        T, N, d = h.shape
        _, union, kind, direction = graph.batched(T, h.device)
        attr = Fn.one_hot(kind, self.n_scales).to(h.dtype)
        if self.direction:
            attr = torch.cat([attr, direction.to(h.dtype)], 1)
        x = h.reshape(T * N, d)
        for conv in self.convs:
            x = x + self.drop(Fn.silu(conv(x, union, edge_attr=attr)))
        return x.view(T, N, d)

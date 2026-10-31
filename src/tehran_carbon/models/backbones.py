"""Spatial backbones on the fused node embeddings h (T, N, d), with PyTorch Geometric layers.

none  no message passing (control: the same network without a graph)
sage  multi-scale GraphSAGE: h <- h + act(sum_s SAGE_s(h, edges at dilation s))
gcn   multi-scale GCN (Kipf & Welling normalisation per scale)
gat   GATv2 over the union of all scales; the scale is an edge feature, so attention can weigh near and far
      neighbours differently (the "attention between edges" of the proposal)
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

    def __init__(self, hidden: int, layers: int = 3, scales=(1, 3, 9), dropout: float = 0.1, **_):
        super().__init__()
        self.convs = nn.ModuleList([nn.ModuleList([self.make_conv(hidden) for _ in scales]) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)

    def make_conv(self, hidden: int) -> nn.Module:
        raise NotImplementedError

    def forward(self, h, graph):
        T, N, d = h.shape
        per_scale, _, _ = graph.batched(T, h.device)
        x = h.reshape(T * N, d)
        for convs in self.convs:
            m = sum(conv(x, e) for conv, e in zip(convs, per_scale))
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

    def __init__(self, hidden: int, layers: int = 3, scales=(1, 3, 9), heads: int = 2, dropout: float = 0.1, **_):
        super().__init__()
        from torch_geometric.nn import GATv2Conv
        if hidden % heads:
            raise ValueError(f"hidden ({hidden}) must be divisible by heads ({heads})")
        self.n_scales = len(scales)
        self.convs = nn.ModuleList([GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=self.n_scales,
                                              add_self_loops=True, fill_value="mean") for _ in range(layers)])
        self.drop = nn.Dropout(dropout)

    def forward(self, h, graph):
        T, N, d = h.shape
        _, union, kind = graph.batched(T, h.device)
        attr = Fn.one_hot(kind, self.n_scales).to(h.dtype)
        x = h.reshape(T * N, d)
        for conv in self.convs:
            x = x + self.drop(Fn.silu(conv(x, union, edge_attr=attr)))
        return x.view(T, N, d)

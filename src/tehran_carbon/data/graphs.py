"""Urban graphs (edge_index in PyTorch Geometric convention: row 0 = source, row 1 = target).

GridGraph      nodes = L0 cells; at scale d every cell receives from its 8 neighbours d cells away
               (dilations 1, 3, 9: three message-passing layers see ~23 km, the range of the target).
               Every edge also carries its unit direction (east, north) in km, used as an edge feature by GAT.
ODOperator     travel-demand links between districts (origin-destination flows, as the OD edges of HENCE):
               a cell receives the flow-weighted mean embedding of the OTHER districts its district exchanges
               trips with, through a district bottleneck (cells -> districts -> OD mix -> cells).
DistrictGraph  nodes = municipal districts, edges = shared boundary (district-level decisions in RQ3)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import Config


class GridGraph:
    def __init__(self, grid, scales):
        H, W = grid.H, grid.W
        self.H, self.W, self.N = H, W, H * W
        self.scales = [int(s) for s in scales]
        self.od = None                                         # optional ODOperator (model.backbone.od)
        idx = np.arange(self.N).reshape(H, W)
        self.edges, self.directions = [], []
        for d in self.scales:
            src, dst, vec = [], [], []
            for dy in (-d, 0, d):
                for dx in (-d, 0, d):
                    if dy == 0 and dx == 0:
                        continue
                    ys = np.arange(max(0, -dy), H - max(0, dy))
                    xs = np.arange(max(0, -dx), W - max(0, dx))
                    Yg, Xg = np.meshgrid(ys, xs, indexing="ij")
                    src.append(idx[Yg + dy, Xg + dx].ravel())
                    dst.append(idx[Yg, Xg].ravel())
                    v = np.array([dx * grid.km_x, -dy * grid.km_y])     # from target to source: (east, north) km
                    vec.append(np.tile(v / np.linalg.norm(v), (Yg.size, 1)))
            self.edges.append(torch.as_tensor(np.vstack([np.concatenate(src), np.concatenate(dst)]), dtype=torch.long))
            self.directions.append(torch.as_tensor(np.concatenate(vec) if vec else np.zeros((0, 2)), dtype=torch.float32))
        self._batched = {}

    def batched(self, T: int, device):
        """T disjoint copies (one per time step) for node features flattened to (T * N, d). Returns the per-scale
        edge_index list, the union of all scales, and for every union edge its scale id, unit direction (from the
        cell to the neighbour) and time step."""
        key = (T, str(device))
        if key not in self._batched:
            off = (torch.arange(T) * self.N).view(T, 1, 1)
            per = [(e.unsqueeze(0) + off).permute(1, 0, 2).reshape(2, -1).to(device) for e in self.edges]
            kind = torch.cat([torch.full((p.shape[1],), i, dtype=torch.long) for i, p in enumerate(per)]).to(device)
            direction = torch.cat([v.repeat(T, 1) for v in self.directions]).to(device)
            step = torch.cat([torch.arange(T).repeat_interleave(e.shape[1]) for e in self.edges]).to(device)
            self._batched = {key: (per, torch.cat(per, 1), kind, direction, step)}   # only the latest T is kept
        return self._batched[key]


class ODOperator:
    def __init__(self, fractions: np.ndarray, flows: np.ndarray, device="cpu"):
        """fractions (N cells x D districts): share of each cell in each district; flows (D x D): trips o -> d."""
        F = np.asarray(fractions, float)
        Wd = np.asarray(flows, float)
        Wd = Wd + Wd.T                                         # exchange in both directions
        np.fill_diagonal(Wd, 0.0)                              # trips inside a district: covered by spatial edges
        Wd = Wd / np.maximum(Wd.sum(1, keepdims=True), 1e-12)
        P = F.T / np.maximum(F.sum(0)[:, None], 1e-12)         # district <- cells (area-weighted mean)
        R = F / np.maximum(F.sum(1, keepdims=True), 1e-12)     # cell <- districts (cells outside get 0)
        self.P, self.W, self.R = (torch.as_tensor(a, dtype=torch.float32, device=device) for a in (P, Wd, R))

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        """h (T, N, d) -> (T, N, d)"""
        return self.R @ (self.W @ (self.P @ h))

    def to(self, device) -> "ODOperator":
        self.P, self.W, self.R = self.P.to(device), self.W.to(device), self.R.to(device)
        return self

    @classmethod
    def from_config(cls, cfg: dict, n_cells: int) -> "ODOperator":
        """OD flows of the study years (averaged) and the cell-district shares, as set in the config."""
        raw = Config.path(cfg, "raw")
        b, od = cfg["data"]["boundaries"], cfg["model"]["backbone"]["od"]
        fc = b["fractions_columns"]
        fr = pd.read_csv(raw / b["district_fractions_file"], encoding="utf-8-sig")
        ids = sorted(fr[fc["district"]].unique())
        col = {d: i for i, d in enumerate(ids)}
        F = np.zeros((n_cells, len(ids)))
        F[fr[fc["cell"]].values, fr[fc["district"]].map(col).values] = fr[fc["frac"]].values
        T = np.zeros((len(ids), len(ids)))
        for y in cfg["years"]:
            t = pd.read_csv(raw / od["file"].format(year=y), encoding="utf-8-sig")
            t = t[t[od["origin_column"]].isin(col) & t[od["destination_column"]].isin(col)]
            T[t[od["origin_column"]].map(col).values, t[od["destination_column"]].map(col).values] += t[od["flow_column"]].values
        return cls(F, T / len(cfg["years"]))


class DistrictGraph:
    def __init__(self, districts_file: Path, id_column: str, metric_crs: str, touch_tolerance_m: float):
        import geopandas as gpd
        d = gpd.read_file(districts_file).to_crs(metric_crs).sort_values(id_column).reset_index(drop=True)
        self.ids = d[id_column].tolist()
        geom = d.geometry.buffer(touch_tolerance_m)
        pairs = [(j, i) for i in range(len(d)) for j in range(len(d)) if i != j and geom.iloc[i].intersects(geom.iloc[j])]
        self.edge_index = torch.tensor(pairs, dtype=torch.long).T

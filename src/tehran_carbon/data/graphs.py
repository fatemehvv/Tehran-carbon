"""Urban graphs (edge_index in PyTorch Geometric convention: row 0 = source, row 1 = target).

GridGraph      nodes = L0 cells; at scale d every cell receives from its 8 neighbours d cells away
               (dilations 1, 3, 9: three message-passing layers see ~23 km, the range of the target)
DistrictGraph  nodes = municipal districts, edges = shared boundary (district-level decisions in RQ3)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class GridGraph:
    def __init__(self, H: int, W: int, scales):
        self.H, self.W, self.N = H, W, H * W
        self.scales = [int(s) for s in scales]
        idx = np.arange(self.N).reshape(H, W)
        self.edges = []
        for d in self.scales:
            src, dst = [], []
            for dy in (-d, 0, d):
                for dx in (-d, 0, d):
                    if dy == 0 and dx == 0:
                        continue
                    ys = np.arange(max(0, -dy), H - max(0, dy))
                    xs = np.arange(max(0, -dx), W - max(0, dx))
                    Yg, Xg = np.meshgrid(ys, xs, indexing="ij")
                    src.append(idx[Yg + dy, Xg + dx].ravel())
                    dst.append(idx[Yg, Xg].ravel())
            self.edges.append(torch.as_tensor(np.vstack([np.concatenate(src), np.concatenate(dst)]), dtype=torch.long))
        self._batched = {}

    def batched(self, T: int, device) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """T disjoint copies (one per month) for node features flattened to (T * N, d): per-scale edge_index list,
        the union of all scales, and the scale id of every union edge."""
        key = (T, str(device))
        if key not in self._batched:
            off = (torch.arange(T) * self.N).view(T, 1, 1)
            per = [(e.unsqueeze(0) + off).permute(1, 0, 2).reshape(2, -1).to(device) for e in self.edges]
            kind = torch.cat([torch.full((p.shape[1],), i, dtype=torch.long) for i, p in enumerate(per)]).to(device)
            self._batched = {key: (per, torch.cat(per, 1), kind)}      # only the latest T is kept (memory)
        return self._batched[key]


class DistrictGraph:
    def __init__(self, districts_file: Path, id_column: str, metric_crs: str, touch_tolerance_m: float):
        import geopandas as gpd
        d = gpd.read_file(districts_file).to_crs(metric_crs).sort_values(id_column).reset_index(drop=True)
        self.ids = d[id_column].tolist()
        geom = d.geometry.buffer(touch_tolerance_m)
        pairs = [(j, i) for i in range(len(d)) for j in range(len(d)) if i != j and geom.iloc[i].intersects(geom.iloc[j])]
        self.edge_index = torch.tensor(pairs, dtype=torch.long).T

"""Spatial validation design.

Monthly NO2 is autocorrelated over ~20 km, so a random split mostly measures how well a model copies its neighbour.
The main design holds out whole blocks of the autocorrelation range (leave one block out) and drops training cells
within a buffer around the test block (Roberts et al. 2017, Ecography; Ploton et al. 2020, Nat. Commun.).

fold_map: (H, W) int, fold id of every label-domain cell, -1 elsewhere. All months share the split.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from .grid import Grid


class SpatialSplit:
    def __init__(self, split: dict, grid: Grid, domain: np.ndarray, z: np.ndarray | None = None, seed: int = 0):
        """split: cfg['split']; z: (T, H, W) target, only needed when block_km is 'auto'."""
        self.grid, self.domain = grid, domain
        self.scheme = split["scheme"]
        self.buffer_km = float(split["buffer_km"]) if self.scheme != "random_kfold" else 0.0
        self.info = {"scheme": self.scheme, "buffer_km": self.buffer_km}
        side = split["block_km"]
        if side == "auto":
            self.info["range_km"] = self.correlogram_range(z, split["correlogram"], seed)
            side = float(np.ceil(self.info["range_km"] / 2) * 2)
        self.block_km = float(side)
        self.info["block_km"] = self.block_km

        rng = np.random.default_rng(seed + 1)
        blocks = self.blocks(self.block_km)
        if self.scheme == "block_lobo":
            f = blocks
        elif self.scheme == "block_kfold":
            f = rng.permutation(blocks.max() + 1)[blocks] % int(split["n_folds"])
        elif self.scheme == "random_kfold":
            f = rng.permutation(len(blocks)) % int(split["n_folds"])
        else:
            raise ValueError(f"unknown split scheme '{self.scheme}'")
        self.fold_map = np.full(domain.shape, -1, dtype=int)
        self.fold_map[domain] = f
        self.n_folds = int(f.max()) + 1
        self.info.update(n_folds=self.n_folds, test_cells_per_fold=np.bincount(f).tolist())

    def blocks(self, side_km: float) -> np.ndarray:
        """Block id (0..K-1) of every domain cell (row-major order): square blocks of side_km km."""
        rr, cc = np.where(self.domain)
        code = (rr * self.grid.km_y // side_km).astype(int) * 1000 + (cc * self.grid.km_x // side_km).astype(int)
        return np.searchsorted(np.unique(code), code)

    def masks(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Test = fold k; train = other domain cells farther than the buffer from every test cell."""
        te = self.fold_map == k
        if self.buffer_km <= 0:
            return self.domain & ~te, te
        dkm = distance_transform_edt(~te, sampling=(self.grid.km_y, self.grid.km_x))
        return self.domain & ~te & (dkm > self.buffer_km), te

    def inner_val(self, train: np.ndarray, k: int) -> np.ndarray:
        """One training fold held out for early stopping: the first fold id after k that has training cells."""
        folds = sorted(f for f in np.unique(self.fold_map[train]) if f >= 0)
        if len(folds) < 2:
            return np.zeros_like(train)
        v = next((f for f in folds if f > k), folds[0])
        return train & (self.fold_map == v)

    def correlogram_range(self, z: np.ndarray, c: dict, seed: int) -> float:
        """Distance (km) where the mean monthly correlation of the target first falls below c['threshold']."""
        if z is None:
            raise ValueError("block_km: auto needs the target")
        rr, cc = np.where(self.domain)
        xy = np.c_[cc * self.grid.km_x, rr * self.grid.km_y]
        rng = np.random.default_rng(seed)
        i, j = rng.integers(0, len(rr), int(c["pairs"])), rng.integers(0, len(rr), int(c["pairs"]))
        dist = np.hypot(*(xy[i] - xy[j]).T)
        edges = np.arange(0, float(c["max_km"]), float(c["bin_km"]))
        r = np.zeros(len(edges) - 1)
        for t in range(z.shape[0]):
            v = z[t][rr, cc]
            v = (v - v.mean()) / v.std()
            r += [(v[i] * v[j])[(dist >= a) & (dist < b) & (i != j)].mean() for a, b in zip(edges[:-1], edges[1:])]
        r /= z.shape[0]
        below = np.flatnonzero(r < float(c["threshold"]))
        return float((edges[below[0]] + edges[below[0] + 1]) / 2) if len(below) else float(c["max_km"])

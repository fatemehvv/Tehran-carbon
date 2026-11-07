"""Alternative hotspot definitions and scales (paper, branch paper_TC).

hot_labels   top (1 - q) of the valid domain cells of every step: q = 0.90 is the thesis label; 0.80 / 0.95 are the
             sensitivity labels of the paper
superpixels  groups of size x size cells (about one TROPOMI pixel for size 5: the target is binned from 3.5 x 5.5 km
             pixels, so the 1 km cells are not independent; Sun et al. 2018, AMT)
aggregate    mean of a (T, n_domain) array inside every group (finite values only)
"""
from __future__ import annotations

import numpy as np


def hot_labels(z: np.ndarray, q: float) -> np.ndarray:
    """(T, n) bool: z at or above the q-quantile of the finite values of its step."""
    hot = np.zeros(z.shape, bool)
    for t in range(z.shape[0]):
        ok = np.isfinite(z[t])
        if ok.sum() >= 2:
            hot[t] = ok & (z[t] >= np.quantile(z[t][ok], q))
    return hot


def superpixels(cells: np.ndarray, W: int, size: int) -> np.ndarray:
    """Group id (0..G-1) of every domain cell (cells = flat indices row * W + col)."""
    r, c = np.divmod(np.asarray(cells), W)
    code = (r // size) * 100000 + (c // size)
    return np.searchsorted(np.unique(code), code)


def aggregate(a: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """(T, n) -> (T, G): mean of the finite values of every group (NaN if none)."""
    G = int(groups.max()) + 1
    out = np.full((a.shape[0], G), np.nan)
    for t in range(a.shape[0]):
        ok = np.isfinite(a[t])
        s = np.bincount(groups[ok], weights=a[t][ok], minlength=G)
        n = np.bincount(groups[ok], minlength=G)
        out[t] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return out


def group_blocks(groups: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    """Block of every group = the block that holds most of its cells."""
    G = int(groups.max()) + 1
    out = np.zeros(G, int)
    for g in range(G):
        b = blocks[groups == g]
        out[g] = np.bincount(b).argmax()
    return out

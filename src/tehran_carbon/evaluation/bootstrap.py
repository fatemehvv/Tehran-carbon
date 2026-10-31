"""Block bootstrap: resample the spatial blocks with replacement (cells inside a block are near-copies of each other),
re-score the SAME out-of-fold predictions (no retraining) and take percentile intervals. Paired differences between
models use identical resamples, so their sampling noise cancels."""
from __future__ import annotations

import numpy as np

from .metrics import monthly_scores


class BlockBootstrap:
    def __init__(self, blocks: np.ndarray, n: int, seed: int):
        """blocks: (n_domain_cells,) block id of every cell in out-of-fold order."""
        rng = np.random.default_rng(seed)
        members = [np.flatnonzero(blocks == b) for b in range(int(blocks.max()) + 1)]
        self.samples = [np.concatenate([members[b] for b in rng.integers(0, len(members), len(members))])
                        for _ in range(n)]

    def scores(self, z: np.ndarray, hot: np.ndarray, pred: np.ndarray, metric: str) -> np.ndarray:
        """(n,) month-averaged metric on every resample (months without a hotspot in a resample are skipped)."""
        return np.array([np.nanmean(monthly_scores(z, hot, pred, [metric], idx)[metric]) for idx in self.samples])

    @staticmethod
    def interval(samples: np.ndarray, level: float) -> tuple[float, float]:
        s = samples[np.isfinite(samples)]
        a = (1 - level) / 2
        return float(np.quantile(s, a)), float(np.quantile(s, 1 - a))

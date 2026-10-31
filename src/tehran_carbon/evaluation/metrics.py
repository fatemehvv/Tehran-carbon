"""Hotspot and regression metrics, computed per month over the label domain and then averaged over months.

pr_auc    average precision of the hotspot label (primary; chance = share of hotspots, 0.10)
roc_auc   area under the ROC curve
f1_top    F1 when the predicted top-k cells (k = number of true hotspots) are called hotspots
spearman  rank correlation of predicted and observed z
r2        coefficient of determination of z
The vectorised implementations accept duplicated rows (needed by the block bootstrap).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def pr_auc(y: np.ndarray, s: np.ndarray, z=None) -> float:
    if y.sum() == 0:
        return np.nan
    yo = y[np.argsort(-s, kind="mergesort")]
    tp = np.cumsum(yo)
    return float((tp / np.arange(1, len(yo) + 1) * yo).sum() / yo.sum())


def roc_auc(y, s, z=None) -> float:
    npos = y.sum()
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def f1_top(y, s, z=None) -> float:
    k = int(y.sum())
    if k == 0:
        return np.nan
    top = np.zeros(len(s), bool)
    top[np.argsort(-s, kind="mergesort")[:k]] = True
    tp = (top & y).sum()
    return float(2 * tp / (top.sum() + k))


def spearman(y, s, z) -> float:
    a, b = rankdata(s), rankdata(z)
    return float(np.corrcoef(a, b)[0, 1])


def r2(y, s, z) -> float:
    return float(1 - ((z - s) ** 2).sum() / ((z - z.mean()) ** 2).sum())


METRICS = {"pr_auc": pr_auc, "roc_auc": roc_auc, "f1_top": f1_top, "spearman": spearman, "r2": r2}


def monthly_scores(z: np.ndarray, hot: np.ndarray, pred: np.ndarray, names, idx=None) -> dict[str, np.ndarray]:
    """z, hot, pred: (T, n) over the domain cells; idx: optional row index (bootstrap resample).
    Returns name -> (T,) score of every month (NaN for a month without hotspots in the sample)."""
    if idx is not None:
        z, hot, pred = z[:, idx], hot[:, idx], pred[:, idx]
    return {name: np.array([METRICS[name](hot[t].astype(bool), pred[t], z[t]) for t in range(z.shape[0])])
            for name in names}

"""Hotspot and regression metrics, computed per month over the label domain and then averaged over months.

pr_auc    average precision of the hotspot label (primary; chance = share of hotspots, 0.10)
roc_auc   area under the ROC curve
f1_top    F1 when the predicted top-k cells (k = number of true hotspots) are called hotspots
spearman  rank correlation of predicted and observed z
r2        coefficient of determination of z
mae/rmse  absolute / root-mean-square error of z (reported by the comparable studies)
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


def mae(y, s, z) -> float:
    return float(np.abs(z - s).mean())


def rmse(y, s, z) -> float:
    return float(np.sqrt(((z - s) ** 2).mean()))


METRICS = {"pr_auc": pr_auc, "roc_auc": roc_auc, "f1_top": f1_top, "spearman": spearman, "r2": r2, "mae": mae,
           "rmse": rmse}


def monthly_scores(z: np.ndarray, hot: np.ndarray, pred: np.ndarray, names, idx=None) -> dict[str, np.ndarray]:
    """z, hot, pred: (T, n) over the domain cells (T months or days); idx: optional row index (bootstrap resample).
    Each step is scored on its valid cells only (finite target and prediction; daily data has cloud gaps).
    Returns name -> (T,) score of every step (NaN for a step without valid cells or hotspots in the sample)."""
    if idx is not None:
        z, hot, pred = z[:, idx], hot[:, idx], pred[:, idx]
    out = {name: np.full(z.shape[0], np.nan) for name in names}
    for t in range(z.shape[0]):
        ok = np.isfinite(z[t]) & np.isfinite(pred[t])
        if ok.sum() < 2:
            continue
        zt, ht, pt = z[t][ok], hot[t][ok].astype(bool), pred[t][ok]
        for name in names:
            out[name][t] = METRICS[name](ht, pt, zt)
    return out

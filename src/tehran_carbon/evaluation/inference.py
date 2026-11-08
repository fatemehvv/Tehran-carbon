"""Inference for comparing two spatial models with few, unequal spatial blocks (paper, branch paper_TC).

The thesis block bootstrap resamples 12 blocks, and two of them hold about 95 % of the hotspot cell-months, so its
percentile intervals are lopsided and unreliable (few, unbalanced clusters: Cameron, Gelbach & Miller 2008; MacKinnon,
Nielsen & Webb 2023). These tools add:

  signflip_pooled   exact randomisation test: under H0 the two models are exchangeable inside every block, so the labels
                    "candidate" / "reference" are swapped block by block (2^K assignments, K = 12 -> 4096) and the
                    city-wide metric difference is recomputed. Rank metrics are computed on predictions rank-normalised
                    within each step first, so that mixing two models' maps does not depend on their calibration
  signflip_mean     the same on per-block differences (mean statistic)
  jackknife_se      delete-one-block jackknife of the difference; t intervals with K - 1 degrees of freedom
  verdict           four-way reading with an equivalence margin (Lakens 2017; non-inferiority / equivalence = TOST):
                    superior, equivalent, non-inferior, inferior or inconclusive
  holm              Holm (1979) step-down adjustment of a family of p-values
  correlogram, pair_counts, effective_n
                    effective number of independent cells of an autocorrelated map, n^2 / sum_ij rho(d_ij)
                    (Griffith 2005; Dutilleul 1993)
  dutilleul         correlation of two autocorrelated maps with Dutilleul's (1993) effective sample size
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.stats import rankdata

from .robustness import mean_score

RANK_METRICS = {"pr_auc", "roc_auc", "f1_top", "spearman"}


# ---------------------------------------------------------------------------------------------- randomisation tests
def rank_normalize(pred: np.ndarray) -> np.ndarray:
    """(T, n): ranks / (number of finite values) within every step; NaN stays NaN."""
    out = np.full(pred.shape, np.nan)
    for t in range(pred.shape[0]):
        ok = np.isfinite(pred[t])
        if ok.any():
            out[t][ok] = rankdata(pred[t][ok]) / ok.sum()
    return out


def assignments(K: int, max_exact: int = 16, n_random: int = 20000, seed: int = 0) -> tuple[np.ndarray, bool]:
    """(n, K) bool flip patterns; row 0 is the identity (nothing flipped). Exact enumeration up to max_exact blocks."""
    if K <= max_exact:
        return ((np.arange(2 ** K)[:, None] >> np.arange(K)) & 1).astype(bool), True
    f = np.random.default_rng(seed).random((n_random, K)) < 0.5
    f[0] = False
    return f, False


def _pvalues(st: np.ndarray) -> dict:
    obs, tol = st[0], 1e-12
    return dict(stat=float(obs), p_greater=float(np.mean(st >= obs - tol)),
                p_two=float(np.mean(np.abs(st) >= abs(obs) - tol)), n_perm=int(len(st)))


def signflip_pooled(z, hot, pc, pr, blocks: np.ndarray, metric: str, max_exact: int = 16, n_random: int = 20000,
                    seed: int = 0, rank: bool = True) -> dict:
    """Randomisation test of the city-wide metric difference candidate - reference (both (T, n), n = domain cells
    in the order of blocks). p_greater: one-sided (candidate better); p_two: two-sided."""
    if rank and metric in RANK_METRICS:
        pc, pr = rank_normalize(pc), rank_normalize(pr)
    K = int(blocks.max()) + 1
    flips, exact = assignments(K, max_exact, n_random, seed)
    if exact:                                          # complement of an assignment gives minus the statistic
        flips = flips[~flips[:, -1]]
    st = []
    for f in flips:
        cell = f[blocks][None]
        st.append(mean_score(z, hot, np.where(cell, pr, pc), metric) - mean_score(z, hot, np.where(cell, pc, pr), metric))
    st = np.asarray(st)
    if exact:
        st = np.concatenate([st, -st])
    out = _pvalues(st)
    out.update(exact=exact, blocks=K)
    return out


def signflip_mean(d: np.ndarray, max_exact: int = 16, n_random: int = 20000, seed: int = 0,
                  weights: np.ndarray | None = None) -> dict:
    """Randomisation test of the (weighted) mean of per-block differences d (NaN = block without a defined value,
    dropped). weights: e.g. the number of cells of every block, so that edge blocks of a few cells count little."""
    d = np.asarray(d, float)
    w = np.ones_like(d) if weights is None else np.asarray(weights, float)
    ok = np.isfinite(d) & np.isfinite(w) & (w > 0)
    d, w = d[ok], w[ok] / w[ok].sum() if ok.any() else w[ok]
    if len(d) == 0:
        return dict(stat=np.nan, p_greater=np.nan, p_two=np.nan, n_perm=0, exact=True, blocks=0)
    flips, exact = assignments(len(d), max_exact, n_random, seed)
    st = (np.where(flips, -1.0, 1.0) * d * w).sum(1)
    out = _pvalues(st)
    out.update(exact=exact, blocks=len(d))
    return out


def min_p(n_blocks: int) -> float:
    """Smallest one-sided p-value an exact sign-flip test over n_blocks can give."""
    return 1.0 / 2 ** n_blocks if n_blocks > 0 else np.nan


# ---------------------------------------------------------------------------------------------- intervals and verdicts
def jackknife_se(z, hot, pc, pr, blocks: np.ndarray, metric: str) -> tuple[float, float, int, np.ndarray]:
    """Full-sample difference, delete-one-block jackknife SE, number of blocks and the leave-one-out differences."""
    K = int(blocks.max()) + 1
    full = mean_score(z, hot, pc, metric) - mean_score(z, hot, pr, metric)
    d = np.array([mean_score(z, hot, pc, metric, np.flatnonzero(blocks != b))
                  - mean_score(z, hot, pr, metric, np.flatnonzero(blocks != b)) for b in range(K)])
    ok = np.isfinite(d)
    k = int(ok.sum())
    se = float(np.sqrt((k - 1) / k * ((d[ok] - d[ok].mean()) ** 2).sum())) if k > 1 else np.nan
    return float(full), se, k, d


def t_interval(est: float, se: float, df: int, level: float) -> tuple[float, float]:
    if not np.isfinite(se) or df < 1:
        return np.nan, np.nan
    h = float(stats.t.ppf(1 - (1 - level) / 2, df)) * se
    return est - h, est + h


def verdict(lo95: float, hi95: float, lo90: float, hi90: float, margin: float) -> str:
    """Four-way reading of a difference (candidate - reference) with an equivalence margin.
    superior: 95 % interval above 0 ('beyond margin' if also above the margin); equivalent: 90 % interval inside
    (-margin, margin) (TOST at 5 %); non-inferior: 95 % lower bound above -margin; inferior: 95 % interval below 0."""
    if not all(np.isfinite(v) for v in (lo95, hi95, lo90, hi90)):
        return "undefined"
    if lo95 > 0:
        return "superior beyond margin" if lo95 >= margin else "superior"
    if lo90 > -margin and hi90 < margin:
        return "equivalent"
    if lo95 > -margin:
        return "non-inferior"
    if hi95 < 0:
        return "inferior"
    return "inconclusive"


def holm(pvalues: dict) -> dict:
    """Holm step-down adjusted p-values (monotone, capped at 1). NaN entries are left out of the family."""
    items = sorted(((k, v) for k, v in pvalues.items() if np.isfinite(v)), key=lambda kv: kv[1])
    m, out, run = len(items), {k: np.nan for k in pvalues}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


# ---------------------------------------------------------------------------------------------- spatial dependence
def coords_km(cells: np.ndarray, grid) -> np.ndarray:
    """(n, 2) east / south coordinates in km of flat cell indices."""
    r, c = np.divmod(np.asarray(cells), grid.W)
    return np.c_[c * grid.km_x, r * grid.km_y]


def pair_counts(xy: np.ndarray, edges: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Number of unordered pairs (i < j) in every distance bin [edges[k], edges[k + 1])."""
    n = len(xy)
    counts = np.zeros(len(edges) - 1)
    j = np.arange(n)
    for a in range(0, n, chunk):
        i = np.arange(a, min(a + chunk, n))
        d = np.hypot(xy[i, None, 0] - xy[None, :, 0], xy[i, None, 1] - xy[None, :, 1])
        counts += np.histogram(d[j[None, :] > i[:, None]], bins=edges)[0]
    return counts


def correlogram(v: np.ndarray, xy: np.ndarray, edges: np.ndarray, n_pairs: int = 200000, seed: int = 0,
                return_se: bool = False):
    """Mean correlation of the standardised field per distance bin, averaged over steps. v: (T, n), NaN allowed.
    With return_se also the approximate standard error of every bin under no correlation, 1 / sqrt(pairs x steps)."""
    rng = np.random.default_rng(seed)
    n = v.shape[1]
    i, j = rng.integers(0, n, n_pairs), rng.integers(0, n, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    b = np.digitize(np.hypot(*(xy[i] - xy[j]).T), edges) - 1
    B = len(edges) - 1
    inb = (b >= 0) & (b < B)
    i, j, b = i[inb], j[inb], b[inb]
    acc, steps, used = np.zeros(B), np.zeros(B), np.zeros(B)
    for t in range(v.shape[0]):
        vt = v[t]
        fin = np.isfinite(vt)
        if fin.sum() < 3 or np.nanstd(vt) == 0:
            continue
        s = (vt - np.nanmean(vt)) / np.nanstd(vt)
        ok = fin[i] & fin[j]
        cnt = np.bincount(b[ok], minlength=B)
        tot = np.bincount(b[ok], weights=s[i[ok]] * s[j[ok]], minlength=B)
        has = cnt > 0
        acc[has] += tot[has] / cnt[has]
        steps[has] += 1
        used += cnt
    rho = np.where(steps > 0, acc / np.maximum(steps, 1), np.nan)
    if not return_se:
        return rho
    return rho, np.where(used > 0, 1.0 / np.sqrt(np.maximum(used, 1)), np.nan)


def truncate(rho: np.ndarray, se: np.ndarray | None = None) -> np.ndarray:
    """Correlogram used in sums: clipped at 0, NaN as 0, and (with se) 0 from the first bin that is not above
    2 standard errors on: sampling noise of an uncorrelated field would otherwise add up over many pairs."""
    r = np.nan_to_num(np.clip(rho, 0, None))
    if se is not None:
        bad = ~(np.asarray(rho) > 2 * np.nan_to_num(se, nan=np.inf))
        if bad.any():
            r[int(np.argmax(bad)):] = 0.0
    return r


def effective_n(rho: np.ndarray, se: np.ndarray | None = None, *, counts: np.ndarray, n: int) -> float:
    """n^2 / sum_ij rho(d_ij), with rho = 1 on the diagonal and the correlogram truncated as in truncate()
    (0 beyond the last bin)."""
    return float(n * n / (n + 2 * (counts * truncate(rho, se)).sum()))


def dutilleul(x: np.ndarray, y: np.ndarray, xy: np.ndarray, edges: np.ndarray, rank: bool = True,
              n_pairs: int = 200000, seed: int = 0) -> dict:
    """Correlation of two maps (Spearman if rank) and its test with Dutilleul's effective sample size
    n' = 1 + n^2 / tr(Rx Ry), the correlation matrices taken from the two correlograms."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, xy = np.asarray(x, float)[ok], np.asarray(y, float)[ok], xy[ok]
    if rank:
        x, y = rankdata(x), rankdata(y)
    n = len(x)
    r = float(np.corrcoef(x, y)[0, 1])
    rx = truncate(*correlogram(x[None], xy, edges, n_pairs, seed, return_se=True))
    ry = truncate(*correlogram(y[None], xy, edges, n_pairs, seed + 1, return_se=True))
    tr = n + 2 * (pair_counts(xy, edges) * rx * ry).sum()
    m = float(1 + n * n / max(tr, 1e-12))
    df = max(m - 2, 1.0)
    t = r * np.sqrt(df / max(1 - r * r, 1e-12))
    return dict(r=r, n=n, n_eff=m, t=float(t), p_two=float(2 * stats.t.sf(abs(t), df)))

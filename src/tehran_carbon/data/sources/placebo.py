"""Placebo branches (fusion paper, branch paper_fusion): the layers of a real domain at a random location.

A placebo layer keeps what a fusion branch can exploit without information about the city: the values (identical
marginal distribution), the spatial texture (power spectrum) and the correlation between the layers of its own
domain. It loses where the values are. Feeding it through the same branch as the real domain separates the gain of
the extra branch (architecture) from the gain of the information.

Construction (settings: cfg["data"]["inputs"]["placebo"]):
  - every layer of a domain is reduced to its map over the year (mean of the 12 months) and mirror-padded to
    (2H, 2W), so the field is periodic without edge jumps;
  - multivariate IAAFT (Schreiber & Schmitz 1996, PRL 77:635): one random phase field per domain is added to the
    Fourier phases of every layer of that domain (keeps their cross-spectra), then `iterations` rounds of
    "keep the amplitude spectrum, then restore the exact values by rank";
  - the (H, W) window is cut out and its values are set by rank to the exact values of the real map;
  - monthly layers: placebo map x the city-wide ratio of the month (month mean / year mean), so a layer that is an
    annual allocation times a monthly share (the inventory layers) keeps its seasonal cycle exactly;
  - `copy` layers (city-wide values such as heating degree-days) are copied unchanged, so a placebo branch has as
    many inputs as its real counterpart.
The random phases depend on the configured seed and `realization` only, never on the year: every year of a run has
its placebo at the same random location, as the real static layers stay in place.

`method: rotate_shift` (the registered fallback after check M4 failed for the IAAFT placebo): every layer of a domain
is rotated by 180 degrees and shifted cyclically by the same random offset (rows and columns each between a quarter and
three quarters of the grid), month by month. The texture, the values, the monthly cycle and the correlation between
the layers of the domain are kept exactly; only the wrap-around seams are new.

`method: conditional` (amendment 1 of the fusion pre-registration, after the label-domain check showed that both
placebos above are mostly zero inside the city and anti-correlated with the target): only the part of a layer that
the context layers do not explain is relocated. For every layer, on the annual map (log1p where the dataset uses it):
  - f = block-cross-fitted gradient boosting of the layer on the context layers and their box means (the same context
    inputs and lags as the model; folds = `cross_fit_folds` random groups of square tiles of `split.block_km` over the
    whole grid; the fit never sees the tile it predicts, so f carries no location beyond what the context carries);
  - r = layer - f, relocated by the rotation and cyclic shift of `rotate_shift` (one offset per domain and
    realisation, as there);
  - placebo = f + relocated r, back-transformed (expm1 for log1p layers), clipped at 0, times the city-wide monthly
    ratio for monthly layers (as for IAAFT).
The placebo keeps the dependence of the layer on the context (f is kept in place) and its values inside the city
approximately; it loses the location of what the context does not explain.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

from .base import SOURCES, Source

CONTEXT_STATIC = ("pop", "vol_total", "vol_nonres", "industrial_frac")      # from source "context"; viirs: "satellite"


def remap(values: np.ndarray, sorted_target: np.ndarray) -> np.ndarray:
    """Array of the shape of `values` whose entries are `sorted_target`, assigned in the rank order of `values`."""
    flat = values.ravel()
    out = np.empty_like(sorted_target)
    out[np.argsort(flat, kind="stable")] = sorted_target
    return out.reshape(values.shape)


def mirror(a: np.ndarray) -> np.ndarray:
    """(H, W) -> (2H, 2W) mirror padding (periodic without discontinuities)."""
    top = np.concatenate([a, a[:, ::-1]], axis=1)
    return np.concatenate([top, top[::-1, :]], axis=0)


def iaaft_multi(maps: list[np.ndarray], rng: np.random.Generator, iterations: int = 100) -> list[np.ndarray]:
    """Multivariate IAAFT surrogates of 2-D maps of the same shape, with one shared random phase field."""
    H, W = maps[0].shape
    shape = (2 * H, 2 * W)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(shape[0], shape[1] // 2 + 1))
    phases[0, 0] = 0.0
    out = []
    for m in maps:
        m = np.asarray(m, dtype=np.float64)
        if np.ptp(m) == 0:                                   # constant map: nothing to move
            out.append(m.copy())
            continue
        x = mirror(m)
        F = np.fft.rfft2(x)
        amp = np.abs(F)
        target = np.sort(x.ravel())
        s = np.fft.irfft2(amp * np.exp(1j * (np.angle(F) + phases)), s=shape)
        for _ in range(int(iterations)):
            s = remap(s, target)
            s = np.fft.irfft2(amp * np.exp(1j * np.angle(np.fft.rfft2(s))), s=shape)
        out.append(remap(s[:H, :W], np.sort(m.ravel())))
    return out


def context_design(cfg: dict, grid, year: int) -> np.ndarray:
    """(H*W, F) static context of a year as the model sees it: the context layers (log1p as configured) and their box
    means at the configured spatial lags. City-wide weather layers carry no spatial information and are left out."""
    d = cfg["data"]
    log1p = set(d.get("log1p") or [])
    ctx = SOURCES.get("context")(cfg, grid).load(year)
    sat = SOURCES.get("satellite")(cfg, grid).load(year)
    maps = {n: np.asarray(ctx[n], np.float64) for n in CONTEXT_STATIC}
    maps["viirs"] = np.asarray(sat["viirs"], np.float64)
    cols = []
    for n, a in maps.items():
        a = a.mean(0) if a.ndim == 3 else a
        a = np.log1p(a) if n in log1p else a
        cols.append(a)
        for k in d.get("spatial_lags") or []:
            cols.append(uniform_filter(a, size=2 * int(k) + 1, mode="nearest"))
    return np.stack([c.ravel() for c in cols], 1)


def tile_folds(grid, side_km: float, n_folds: int, seed: int) -> np.ndarray:
    """(H*W,) fold id: square tiles of side_km over the whole grid, assigned to n_folds folds at random."""
    rr, cc = np.indices((grid.H, grid.W))
    code = (rr * grid.km_y // side_km).astype(int) * 1000 + (cc * grid.km_x // side_km).astype(int)
    tiles = np.searchsorted(np.unique(code), code).ravel()
    perm = np.random.default_rng(seed).permutation(tiles.max() + 1)
    return perm[tiles] % int(n_folds)


def cross_fit(X: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Out-of-fold gradient-boosting prediction of y from X (the settings of the lineage audit)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    out = np.empty_like(y)
    for k in np.unique(folds):
        te = folds == k
        mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
        out[te] = mdl.fit(X[~te], y[~te]).predict(X[te])
    return out


@SOURCES.register("placebo")
class PlaceboSource(Source):
    name = "placebo"

    def load(self, year):
        p = self.p
        prefix = p.get("prefix", "plc_")
        out = {}
        X = folds = None                                             # context design, built once (conditional)
        for i, (domain, layers) in enumerate(p["domains"].items()):
            real = SOURCES.get(domain)(self.cfg, self.grid).load(year)
            missing = [n for n in list(layers) + list((p.get("copy") or {}).get(domain, [])) if n not in real]
            if missing:
                raise KeyError(f"placebo: layers {missing} are not produced by source '{domain}'")
            rng = np.random.default_rng([int(self.cfg["seed"]), int(p["realization"]), i])   # not the year
            arrays = [np.asarray(real[n], dtype=np.float64) for n in layers]
            if p.get("method", "iaaft") == "rotate_shift":
                H, W = arrays[0].shape[-2:]
                dy = int(rng.integers(H // 4, 3 * H // 4 + 1))
                dx = int(rng.integers(W // 4, 3 * W // 4 + 1))
                for n, a in zip(layers, arrays):
                    out[prefix + n] = np.roll(a[..., ::-1, ::-1], (dy, dx), axis=(-2, -1))
                for n in (p.get("copy") or {}).get(domain, []):
                    out[prefix + n] = np.asarray(real[n], dtype=np.float64).copy()
                continue
            if p.get("method", "iaaft") == "conditional":
                H, W = arrays[0].shape[-2:]
                dy = int(rng.integers(H // 4, 3 * H // 4 + 1))          # the offsets of rotate_shift
                dx = int(rng.integers(W // 4, 3 * W // 4 + 1))
                if X is None:
                    X = context_design(self.cfg, self.grid, year)
                    folds = tile_folds(self.grid, float(self.cfg["split"]["block_km"]),
                                       int(p.get("cross_fit_folds", 5)), int(self.cfg["seed"]))
                log1p = set(self.cfg["data"].get("log1p") or [])
                for n, a in zip(layers, arrays):
                    mp = a.mean(0) if a.ndim == 3 else a
                    if np.ptp(mp) == 0:
                        out[prefix + n] = a.copy()
                        continue
                    lg = n in log1p
                    v = np.log1p(np.clip(mp, 0, None)) if lg else mp
                    f = cross_fit(X, v.ravel(), folds).reshape(H, W)
                    r = np.roll((v - f)[::-1, ::-1], (dy, dx), axis=(0, 1))
                    q = f + r
                    q = np.clip(np.expm1(q) if lg else q, 0, None)
                    if a.ndim == 3:
                        ratio = a.reshape(a.shape[0], -1).mean(1) / mp.mean() if mp.mean() != 0 else np.ones(a.shape[0])
                        out[prefix + n] = ratio[:, None, None] * q[None]
                    else:
                        out[prefix + n] = q
                for n in (p.get("copy") or {}).get(domain, []):
                    out[prefix + n] = np.asarray(real[n], dtype=np.float64).copy()
                continue
            if p.get("method", "iaaft") != "iaaft":
                raise ValueError(f"placebo method {p.get('method')!r}: use iaaft, rotate_shift or conditional")
            maps = [a.mean(0) if a.ndim == 3 else a for a in arrays]
            surrogates = iaaft_multi(maps, rng, int(p.get("iterations", 100)))
            for n, a, mp, s in zip(layers, arrays, maps, surrogates):
                if a.ndim == 3:
                    ratio = a.reshape(a.shape[0], -1).mean(1) / mp.mean() if mp.mean() != 0 else np.ones(a.shape[0])
                    out[prefix + n] = ratio[:, None, None] * s[None]
                else:
                    out[prefix + n] = s
            for n in (p.get("copy") or {}).get(domain, []):
                out[prefix + n] = np.asarray(real[n], dtype=np.float64).copy()
        return out

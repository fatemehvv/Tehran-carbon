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
"""
from __future__ import annotations

import numpy as np

from .base import SOURCES, Source


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


@SOURCES.register("placebo")
class PlaceboSource(Source):
    name = "placebo"

    def load(self, year):
        p = self.p
        prefix = p.get("prefix", "plc_")
        out = {}
        for i, (domain, layers) in enumerate(p["domains"].items()):
            real = SOURCES.get(domain)(self.cfg, self.grid).load(year)
            missing = [n for n in list(layers) + list((p.get("copy") or {}).get(domain, [])) if n not in real]
            if missing:
                raise KeyError(f"placebo: layers {missing} are not produced by source '{domain}'")
            rng = np.random.default_rng([int(self.cfg["seed"]), int(p["realization"]), i])   # not the year
            arrays = [np.asarray(real[n], dtype=np.float64) for n in layers]
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

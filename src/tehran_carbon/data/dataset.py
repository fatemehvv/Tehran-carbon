"""The cell-month dataset shared by every experiment.

Unit of analysis: one L0 cell in one Jalali month, for every study year.
  X[year]    (12, H, W, F) features of every cell (features exist everywhere; labels only in the label domain)
  z[year]    (12, H, W)    target: log NO2, z-scored within each month over the label domain (only the spatial
                           pattern is kept; the seasonal level is removed)
  hot[year]  (12, H, W)    hotspot label = top (1 - hotspot_quantile) of the label domain in that month
Each feature belongs to one modality (traffic / energy / pollution / context); its spatial lags keep the modality.
The dataset holds ALL configured modalities; a model picks its branches with columns().

Caches: data/interim/<source>_<year>_<key>.npz (layers of one source) and data/processed/cellmonth_<hash>.npz.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

from ..config import Config
from ..utils.calendar import JalaliYear
from .grid import Grid
from .sources import SOURCES


class CellMonthDataset:
    CACHE_VERSION = 1            # bump when a loader changes so old interim caches are not reused

    def __init__(self, years, X, z, hot, label_domain, in_core, features, modality_of, base_of, lag_of, hash=""):
        self.years = [int(y) for y in years]
        self.X, self.z, self.hot = X, z, hot
        self.label_domain, self.in_core = label_domain, in_core
        self.features, self.modality_of, self.base_of, self.lag_of = features, modality_of, base_of, lag_of
        self.hash = hash
        self.shape = label_domain.shape
        self.cells = np.flatnonzero(label_domain.ravel())          # order of every out-of-fold array

    def columns(self, modalities: list[str], use_context: bool = True) -> dict[str, np.ndarray]:
        """Feature indices of each input branch: the listed domains, then context."""
        out = {}
        for m in list(modalities) + (["context"] if use_context else []):
            idx = np.array([i for i, mo in enumerate(self.modality_of) if mo == m], dtype=int)
            if len(idx) == 0:
                raise ValueError(f"modality '{m}' has no features in the dataset")
            out[m] = idx
        if not out:
            raise ValueError("no input features: give at least one modality or use_context")
        return out

    # ------------------------------------------------------------------ io
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"label_domain": self.label_domain, "in_core": self.in_core}
        for y in self.years:
            arrays.update({f"X_{y}": self.X[y], f"z_{y}": self.z[y], f"hot_{y}": self.hot[y]})
        np.savez_compressed(path, **arrays)
        meta = dict(years=self.years, features=self.features, modality_of=self.modality_of, base_of=self.base_of,
                    lag_of=self.lag_of, hash=self.hash, n_label_cells=int(self.label_domain.sum()),
                    n_core_cells=int(self.in_core.sum()))
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CellMonthDataset":
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        a = np.load(path)
        ys = meta["years"]
        return cls(ys, {y: a[f"X_{y}"] for y in ys}, {y: a[f"z_{y}"] for y in ys}, {y: a[f"hot_{y}"] for y in ys},
                   a["label_domain"], a["in_core"], meta["features"], meta["modality_of"], meta["base_of"],
                   meta["lag_of"], meta["hash"])

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, cfg: dict, force: bool = False, log=print) -> "CellMonthDataset":
        h = Config.data_hash(cfg)
        out = Config.path(cfg, "processed") / f"cellmonth_{h}.npz"
        if out.exists() and not force:
            log(f"dataset {h}: from cache")
            return cls.load(out)
        log(f"dataset {h}: building")
        d = cfg["data"]
        grid = Grid(cfg)
        raw = Config.path(cfg, "raw")
        years = [int(y) for y in cfg["years"]]

        # CORE = cells whose centre lies in the named counties
        import geopandas as gpd
        b = d["boundaries"]
        cnt = gpd.read_file(raw / b["counties_file"]).to_crs(4326)
        names = cnt[b["county_name_column"]].astype(str).str.strip()
        missing = set(d["core_counties"]) - set(names)
        if missing:
            raise ValueError(f"counties {missing} not in {b['counties_file']}; available: {sorted(names)}")
        poly = cnt[names.isin(d["core_counties"])].union_all()
        pts = gpd.GeoSeries(gpd.points_from_xy(grid.lon.ravel(), grid.lat.ravel()), crs=4326)
        in_core = pts.within(poly).values.reshape(grid.H, grid.W)

        # label domain = CORE cells that the inventory gives any CO2, in every year
        ld = d["label_domain"]
        domain = in_core.copy()
        for y in years:
            inv = pd.read_csv(raw / ld["file"].format(year=y), encoding="utf-8-sig").sort_values(ld["cell_column"])
            domain &= (inv[ld["columns"]].sum(axis=1).values > 0).reshape(grid.H, grid.W)

        # layers of every enabled source and year (interim cache per source and year)
        manifest = Config.digest((raw / "MANIFEST.csv").read_text(encoding="utf-8")) if (raw / "MANIFEST.csv").exists() else ""
        enabled = [n for n, p in d["inputs"].items() if (p or {}).get("enabled", True)]
        layers = {y: {} for y in years}
        for name in enabled:
            source = SOURCES.get(name)(cfg, grid)
            key = Config.digest({"v": cls.CACHE_VERSION, "source": name, "params": d["inputs"][name],
                                 "grid": cfg["grid"], "years": years, "seed": cfg["seed"], "manifest": manifest})
            for y in years:
                f = Config.path(cfg, "interim") / f"{name}_{y}_{key}.npz"
                if f.exists():
                    got = dict(np.load(f))
                else:
                    log(f"  source '{name}' {y}")
                    got = source.load(y)
                    f.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(f, **got)
                clash = set(got) & set(layers[y])
                if clash:
                    raise ValueError(f"layers {clash} are produced by two sources")
                layers[y].update(got)

        # feature list: every configured layer of every modality, plus its spatial lags
        modalities = {m: list(v) for m, v in d["modalities"].items()}
        if "embeddings" in enabled:
            modalities.setdefault("context", []).extend(sorted(k for k in layers[years[0]] if k.startswith("emb_")))
        lags = [int(k) for k in d.get("spatial_lags") or []]
        no_lag, log1p = set(d.get("no_lag") or []), set(d.get("log1p") or [])
        features, modality_of, base_of, lag_of = [], [], [], []
        for m, names_m in modalities.items():
            for n in names_m:
                for k in [0] + ([] if n in no_lag else lags):
                    features.append(n if k == 0 else f"{n}@lag{k}")
                    modality_of.append(m)
                    base_of.append(n)
                    lag_of.append(k)

        X, Z, HOT = {}, {}, {}
        t = d["target"]
        for y in years:
            L = layers[y]
            unknown = sorted(set(base_of) - set(L))
            if unknown:
                raise KeyError(f"features {unknown} are not produced by any source; available: {sorted(L)}")
            cols = []
            for n, k in zip(base_of, lag_of):
                a = L[n].astype(np.float64)
                a = np.log1p(a) if n in log1p else a
                a = np.broadcast_to(a, (12, grid.H, grid.W)) if a.ndim == 2 else a
                cols.append(a if k == 0 else uniform_filter(a, size=(1, 2 * k + 1, 2 * k + 1), mode="nearest"))
            X[y] = np.stack(cols, -1).astype(np.float32)
            bad = [features[j] for j in range(X[y].shape[-1]) if not np.isfinite(X[y][..., j]).all()]
            if bad:
                raise ValueError(f"non-finite values in {y}: {bad}")

            # target: per month, z-score of log NO2 over the label domain; hotspot = top share of the domain
            no2 = L[t["layer"]]
            z = np.full(no2.shape, np.nan)
            hot = np.zeros(no2.shape, bool)
            for m in range(12):
                v = np.log(np.clip(no2[m], float(t["log_floor"]), None))
                z[m] = (v - v[domain].mean()) / v[domain].std()
                hot[m] = domain & (z[m] >= np.quantile(z[m][domain], float(t["hotspot_quantile"])))
            if not np.isfinite(z[:, domain]).all():
                raise ValueError(f"target of {y} has missing values inside the label domain")
            Z[y], HOT[y] = z.astype(np.float32), hot

        data = cls(years, X, Z, HOT, domain, in_core, features, modality_of, base_of, lag_of, h)
        data.save(out)
        month_keys = {y: JalaliYear(y).gregorian_months for y in years}
        log(f"dataset {h}: {len(features)} features, {int(domain.sum())} label cells (CORE {int(in_core.sum())}), "
            f"target months {month_keys}")
        return data

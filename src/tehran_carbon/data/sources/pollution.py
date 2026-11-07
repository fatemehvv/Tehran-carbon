"""AQCC monitoring stations: monthly mean of the daily station values, interpolated to the grid by inverse distance.

StationHoldout (paper, branch paper_TC): in spatial CV the layer is built once per year from ALL stations, stations
inside the held-out block included, so the test block's own measurements enter its inputs (spatial pre-processing
leakage; Kapoor & Narayanan 2023, Patterns). With split.fold_aware_stations the layer is rebuilt for every fold from
the stations farther than the buffer from the test block. Transfer and final models keep all stations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, uniform_filter

from .base import SOURCES, Source


def idw(values: np.ndarray, lon: np.ndarray, lat: np.ndarray, grid, power: float, min_km: float) -> np.ndarray:
    """(H, W) inverse-distance-weighted map of station values (distances below min_km count as min_km)."""
    d = grid.distances_km(np.asarray(lon, float), np.asarray(lat, float))
    w = 1.0 / np.maximum(d, float(min_km)) ** float(power)
    return (w * np.asarray(values, float)[:, None, None]).sum(0) / w.sum(0)


def stations_near(rows, cols, inside, test: np.ndarray, buffer_km: float, km_y: float, km_x: float) -> np.ndarray:
    """Boolean per station: True if its cell is a test cell or lies within buffer_km of one (stations outside the
    grid are never near). test: (H, W) mask of the test block."""
    if not test.any():
        return np.zeros(len(rows), bool)
    dkm = distance_transform_edt(~test, sampling=(km_y, km_x))
    near = np.zeros(len(rows), bool)
    r, c, ok = np.asarray(rows), np.asarray(cols), np.asarray(inside, bool)
    near[ok] = dkm[r[ok], c[ok]] <= float(buffer_km)
    return near


@SOURCES.register("pollution")
class PollutionSource(Source):
    name = "pollution"

    def stations(self) -> pd.DataFrame:
        """AQCC stations with coordinates, indexed by station name."""
        p, c = self.p, self.p["columns"]
        st = pd.read_csv(self.file("stations_file"), encoding="utf-8-sig")
        st = st[st[c["station_id"]].astype(str).str.startswith(p["station_id_prefix"])
                & st[c["lat"]].notna() & st[c["lon"]].notna()]
        return st.drop_duplicates(c["station_name"]).set_index(c["station_name"])

    def station_months(self, year, code: str, st: pd.DataFrame | None = None) -> pd.DataFrame:
        """Valid station-months of one pollutant code: columns station, month, mean, size."""
        p, c = self.p, self.p["columns"]
        st = self.stations() if st is None else st
        air = pd.read_csv(self.raw / p["air_files"][year], encoding="utf-8-sig")
        air = air[(air[c["year"]] == year) & air[c["station"]].isin(st.index)].dropna(subset=[c["value"]])
        sm = (air[air[c["pollutant"]] == code].groupby([c["station"], c["month"]])[c["value"]]
              .agg(["mean", "size"]).reset_index())
        sm = sm[sm["size"] >= int(p["min_valid_days"])]
        return sm.rename(columns={c["station"]: "station", c["month"]: "month"})

    def load(self, year, keep=None):
        """keep: optional collection of station names to use (the others are left out); None = every station."""
        p, c = self.p, self.p["columns"]
        st = self.stations()
        out = {}
        for layer, code in p["pollutants"].items():
            sm = self.station_months(year, code, st)
            if keep is not None:
                sm = sm[sm["station"].isin(set(keep))]
            if sm.empty:
                raise ValueError(f"no station data for {code} in {year}" + (" after the hold-out" if keep is not None else ""))
            year_mean = sm.groupby("station")["mean"].mean()
            months = []
            for m in range(1, 13):
                cur = sm[sm["month"] == m].set_index("station")["mean"]
                if cur.empty:                       # no valid station this month: annual station means instead
                    cur = year_mean
                loc = st.loc[cur.index]
                months.append(idw(cur.values, loc[c["lon"]].values, loc[c["lat"]].values, self.grid,
                                  p["idw_power"], p["idw_min_km"]))
            out[layer] = np.stack(months)
        return out


class StationHoldout:
    """Rebuilds the station layers (and their spatial lags) of a dataset without the stations near a test block."""

    def __init__(self, cfg: dict, grid, data):
        d = cfg["data"]
        if d.get("temporal", "monthly") != "monthly":
            raise ValueError("fold_aware_stations is implemented for the monthly pollution source only")
        self.source = PollutionSource(cfg, grid)
        self.grid = grid
        c = self.source.p["columns"]
        st = self.source.stations()
        self.names = np.array(st.index.astype(str))
        self.rows, self.cols, self.inside = grid.rowcol(st[c["lon"]].values, st[c["lat"]].values)
        layers = set(self.source.p["pollutants"])
        self.features = [j for j, b in enumerate(data.base_of) if b in layers]
        self.base_of, self.lag_of = data.base_of, data.lag_of
        self.log1p = set(d.get("log1p") or [])
        self._cache = {}

    def keep(self, test: np.ndarray, buffer_km: float) -> list[str]:
        """Names of the stations farther than buffer_km from every test cell."""
        near = stations_near(self.rows, self.cols, self.inside, test, buffer_km, self.grid.km_y, self.grid.km_x)
        return sorted(self.names[~near])

    def patch(self, X: np.ndarray, year: int, keep) -> np.ndarray:
        """Copy of X (T, H, W, F) with every station feature rebuilt from the kept stations only."""
        key = (int(year), tuple(sorted(keep)))
        if key not in self._cache:
            self._cache = {key: self.source.load(int(year), keep)}         # one fold at a time is enough
        layers = self._cache[key]
        Xp = X.copy()
        for j in self.features:
            a = layers[self.base_of[j]].astype(np.float64)
            a = np.log1p(a) if self.base_of[j] in self.log1p else a
            if a.shape[0] != X.shape[0]:
                raise ValueError(f"station layer has {a.shape[0]} steps, X has {X.shape[0]}")
            k = int(self.lag_of[j])
            Xp[..., j] = (a if k == 0 else uniform_filter(a, size=(1, 2 * k + 1, 2 * k + 1), mode="nearest")).astype(np.float32)
        return Xp

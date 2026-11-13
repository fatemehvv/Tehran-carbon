"""Composite TROPOMI targets by wind (fusion paper, stage 2; docs/paper/fusion_paper/preregistration_fusion_v1.md).

Within each period of the Jalali year (`period: season` = months 1-3, 4-6, 7-9, 10-12; `half` = 1-6, 7-12), the
clear days (at least `min_clear_fraction` of the grid cells valid in the daily file) are ranked by the city-mean wind
speed at the satellite overpass. Strata:
  all    every clear day of the period
  calm   the `fraction` x n calmest days
  windy  the same number of windiest days
A composite is the mean of the selected days in every cell with at least `min_days` valid values (NaN otherwise).
It is placed on the middle month of its period (season: 2, 5, 8, 11; half: 3, 9); the other months are NaN, so they
have no target (data.target.min_valid_fraction > 0 leaves them out). Layers: no2_all, no2_calm, no2_windy.
Daily files: gee/GEE_daily_NO2_wind_L0.js (cloud fraction < 0.3; missing = nodata).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ...utils.calendar import JalaliYear
from .base import SOURCES, Source

PERIODS = {"season": [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)], "half": [(1, 2, 3, 4, 5, 6), (7, 8, 9, 10, 11, 12)]}
PLACE = {"season": [2, 5, 8, 11], "half": [3, 9]}
STRATA = ("all", "calm", "windy")


def select_days(month: np.ndarray, wind: np.ndarray, clear: np.ndarray, period: str, fraction: float) -> list[dict]:
    """For every period: {"place": month the composite is placed on, "all" / "calm" / "windy": day indices}.
    month: Jalali month of every day; wind: overpass wind speed; clear: bool per day."""
    out = []
    for months, place in zip(PERIODS[period], PLACE[period]):
        days = np.flatnonzero(np.isin(month, months) & clear & np.isfinite(wind))
        order = days[np.argsort(wind[days], kind="stable")]
        k = int(np.floor(len(order) * float(fraction)))
        out.append(dict(place=place, all=order, calm=order[:k], windy=order[len(order) - k:] if k else order[:0]))
    return out


def composite(daily: np.ndarray, days: np.ndarray, min_days: int) -> np.ndarray:
    """(H, W) mean of the selected days; NaN where fewer than min_days valid values."""
    if len(days) == 0:
        return np.full(daily.shape[1:], np.nan)
    sub = daily[days]
    n = np.isfinite(sub).sum(0)
    total = np.where(np.isfinite(sub), sub, 0.0).sum(0)
    return np.where(n >= int(min_days), total / np.maximum(n, 1), np.nan)


@SOURCES.register("satellite_composite")
class SatelliteCompositeSource(Source):
    name = "satellite_composite"

    def daily(self, year) -> np.ndarray:
        """(days of the Jalali year, H, W) daily NO2, NaN where missing."""
        import rasterio
        g, dates = self.grid, JalaliYear(year).dates
        with rasterio.open(self.file("file", year)) as r:
            g.check_raster(r.transform, r.shape)
            a = r.read().astype(np.float64)
            names = list(r.descriptions)
        a[np.isclose(a, float(self.p["nodata"]))] = np.nan
        band_of = {}
        for i, n in enumerate(names):                        # GEE toBands names: "<index>_YYYY_MM_dd"
            m = re.search(r"(\d{4})_(\d{2})_(\d{2})$", n or "")
            if m:
                band_of[pd.Timestamp(f"{m[1]}-{m[2]}-{m[3]}")] = i
        if not band_of and a.shape[0] == len(dates):
            return a
        out = np.full((len(dates), g.H, g.W), np.nan)
        for t, d in enumerate(dates):
            if d in band_of:
                out[t] = a[band_of[d]]
        return out

    def wind(self, year) -> np.ndarray:
        """City-mean wind speed at the overpass for every day of the Jalali year (NaN where missing)."""
        c = self.p["wind_columns"]
        w = pd.read_csv(self.file("wind_file", year), encoding="utf-8-sig")
        w = w.assign(**{c["date"]: pd.to_datetime(w[c["date"]])}).set_index(c["date"]).reindex(JalaliYear(year).dates)
        return np.hypot(w[c["u"]].values.astype(float), w[c["v"]].values.astype(float))

    def selection(self, year):
        """(daily array, list of period selections) of one year."""
        daily = self.daily(year)
        jy = JalaliYear(year)
        month = jy.month_of(jy.dates).values
        clear = np.isfinite(daily).reshape(daily.shape[0], -1).mean(1) >= float(self.p.get("min_clear_fraction", 0.5))
        return daily, select_days(month, self.wind(year), clear, self.p.get("period", "season"),
                                  float(self.p.get("fraction", 1 / 3)))

    def load(self, year):
        daily, sel = self.selection(year)
        g = self.grid
        out = {f"no2_{s}": np.full((12, g.H, g.W), np.nan) for s in STRATA}
        for per in sel:
            for s in STRATA:
                out[f"no2_{s}"][per["place"] - 1] = composite(daily, per[s], int(self.p.get("min_days", 3)))
        return out

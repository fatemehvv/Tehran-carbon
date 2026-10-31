"""Daily sources (data.temporal: daily). Every layer is (days of the Jalali year, H, W).

satellite_daily  TROPOMI tropospheric NO2 of each day (cloud fraction < 0.3), exported by gee/GEE_daily_NO2_wind_L0.js;
                 cloudy pixels / days without overpass = NaN
weather_daily    city-wide daily weather: bias-corrected ERA5 temperature (t2m_d) and heating degree-days (hdd_d),
                 wind at the satellite overpass (wind_u east, wind_v north, wind_d speed) and calendar flags
                 (weekend, nowruz) that drive the weekly / holiday traffic cycle
pollution_daily  AQCC station daily means, inverse-distance interpolated per day (missing day: annual station mean)
Monthly layers of the other sources are repeated over the days of their month by the dataset builder.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ...utils.calendar import JalaliYear
from .base import SOURCES, Source


@SOURCES.register("satellite_daily")
class SatelliteDailySource(Source):
    name = "satellite_daily"

    def load(self, year):
        import rasterio
        g, dates = self.grid, JalaliYear(year).dates
        with rasterio.open(self.file("file", year)) as r:
            g.check_raster(r.transform, r.shape)
            a = r.read().astype(np.float64)
            names = list(r.descriptions)
        a[np.isclose(a, float(self.p["nodata"]))] = np.nan
        band_of = {}
        for i, n in enumerate(names):                       # GEE toBands names: "<index>_YYYY_MM_dd"
            m = re.search(r"(\d{4})_(\d{2})_(\d{2})$", n or "")
            if m:
                band_of[pd.Timestamp(f"{m[1]}-{m[2]}-{m[3]}")] = i
        if not band_of and a.shape[0] == len(dates):        # no band names: bands are the days in order
            return {self.p["layer"]: a}
        out = np.full((len(dates), g.H, g.W), np.nan)
        for t, d in enumerate(dates):
            if d in band_of:
                out[t] = a[band_of[d]]
        return {self.p["layer"]: out}


@SOURCES.register("weather_daily")
class WeatherDailySource(Source):
    name = "weather_daily"

    def load(self, year):
        p, g = self.p, self.grid
        jy = JalaliYear(year)
        dates, T = jy.dates, len(jy.dates)
        wc, ec = p["wind_columns"], p["era5_columns"]
        w = pd.read_csv(self.file("wind_file", year), encoding="utf-8-sig")
        w = w.assign(**{wc["date"]: pd.to_datetime(w[wc["date"]])}).set_index(wc["date"]).reindex(dates)
        u = w[wc["u"]].interpolate(limit_direction="both").values
        v = w[wc["v"]].interpolate(limit_direction="both").values
        e = pd.read_csv(self.file("era5_file"), parse_dates=[ec["date"]]).set_index(ec["date"]).reindex(dates)
        t2m = (e[ec["t2m"]] + (e[ec["t2m_bias"]] if ec.get("t2m_bias") else 0.0)).interpolate(limit_direction="both").values
        series = {"t2m_d": t2m, "hdd_d": np.clip(float(p["hdd_base_C"]) - t2m, 0, None), "wind_u": u, "wind_v": v,
                  "wind_d": np.hypot(u, v), "weekend": np.isin(dates.dayofweek, p["weekend_weekdays"]).astype(float),
                  "nowruz": (np.arange(T) < int(p["nowruz_days"])).astype(float)}
        return {k: np.broadcast_to(np.asarray(s, float)[:, None, None], (T, g.H, g.W)).copy() for k, s in series.items()}


@SOURCES.register("pollution_daily")
class PollutionDailySource(Source):
    name = "pollution_daily"

    def load(self, year):
        p, c, g = self.p, self.p["columns"], self.grid
        jy = JalaliYear(year)
        T = len(jy.dates)
        st = pd.read_csv(self.file("stations_file"), encoding="utf-8-sig")
        st = st[st[c["station_id"]].astype(str).str.startswith(p["station_id_prefix"])
                & st[c["lat"]].notna() & st[c["lon"]].notna()]
        st = st.drop_duplicates(c["station_name"]).set_index(c["station_name"])
        air = pd.read_csv(self.raw / p["air_files"][year], encoding="utf-8-sig")
        air = air[(air[c["year"]] == year) & air[c["station"]].isin(st.index) & (air[c["valid_hours"]] >= int(p["min_valid_hours"]))]
        air = air.dropna(subset=[c["value"]])
        first_day = np.concatenate([[0], np.cumsum(jy.days)])[:-1]      # day index of the 1st of each Jalali month
        air = air.assign(t=first_day[air[c["month"]].values - 1] + air[c["day"]].values - 1)

        out = {}
        for layer, code in p["pollutants"].items():
            sd = air[air[c["pollutant"]] == code]
            if sd.empty:
                raise ValueError(f"no station data for {code} in {year}")
            year_mean = sd.groupby(c["station"])[c["value"]].mean()
            by_day = {t: grp.groupby(c["station"])[c["value"]].mean() for t, grp in sd.groupby("t")}
            days = []
            for t in range(T):
                cur = by_day.get(t, year_mean)
                loc = st.loc[cur.index]
                d = g.distances_km(loc[c["lon"]].values, loc[c["lat"]].values)
                w = 1.0 / np.maximum(d, float(p["idw_min_km"])) ** float(p["idw_power"])
                days.append((w * cur.values[:, None, None]).sum(0) / w.sum(0))
            out[layer] = np.stack(days)
        return out

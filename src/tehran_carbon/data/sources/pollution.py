"""AQCC monitoring stations: monthly mean of the daily station values, interpolated to the grid by inverse distance."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import SOURCES, Source


@SOURCES.register("pollution")
class PollutionSource(Source):
    name = "pollution"

    def load(self, year):
        p, c = self.p, self.p["columns"]
        st = pd.read_csv(self.file("stations_file"), encoding="utf-8-sig")
        st = st[st[c["station_id"]].astype(str).str.startswith(p["station_id_prefix"])
                & st[c["lat"]].notna() & st[c["lon"]].notna()]
        st = st.drop_duplicates(c["station_name"]).set_index(c["station_name"])
        air = pd.read_csv(self.raw / p["air_files"][year], encoding="utf-8-sig")
        air = air[(air[c["year"]] == year) & air[c["station"]].isin(st.index)].dropna(subset=[c["value"]])

        out = {}
        for layer, code in p["pollutants"].items():
            sm = (air[air[c["pollutant"]] == code].groupby([c["station"], c["month"]])[c["value"]]
                  .agg(["mean", "size"]).reset_index())
            sm = sm[sm["size"] >= int(p["min_valid_days"])]
            if sm.empty:
                raise ValueError(f"no station data for {code} in {year}")
            year_mean = sm.groupby(c["station"])["mean"].mean()
            months = []
            for m in range(1, 13):
                cur = sm[sm[c["month"]] == m].set_index(c["station"])["mean"]
                if cur.empty:                       # no valid station this month: annual station means instead
                    cur = year_mean
                loc = st.loc[cur.index]
                d = self.grid.distances_km(loc[c["lon"]].values, loc[c["lat"]].values)
                w = 1.0 / np.maximum(d, float(p["idw_min_km"])) ** float(p["idw_power"])
                months.append((w * cur.values[:, None, None]).sum(0) / w.sum(0))
            out[layer] = np.stack(months)
        return out

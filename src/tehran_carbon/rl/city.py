"""Baseline transport and emissions of one study year on the D municipal districts (inputs of the RQ3 simulator).

All quantities come from the RQ1 frozen outputs and the transport tables (settings: rl in the rq3 experiments; see
docs/preregistration_rq3.md):
  C (D, D)        restricted car vehicle trips o -> d of a typical day (gravity OD of RQ1)
  L (D, D)        trip distance in km
  m (12,)         typical days per month of car activity: annual x gasoline_share_t (sums to the days of the year)
  ef              t CO2 per car vehicle-km, calibrated so that the baseline reproduces the car share (L1) of the
                  road CO2 inside the districts
  bus_*           annual CO2 / energy of the municipal buses, their service share and riders per day by district
  road_co2 (12, D) road CO2 of each district and month (L2 grid x monthly profile), road_mj (12,) road fuel energy
  h_free, h0 (D,) hours per km at free flow (VKT-weighted over the district's links) and at the reported speed
Travel time per km of a district whose car volume is scaled by x: h_free + (h0 - h_free) x^eta, i.e. the delay
changes with elasticity eta (congestion-charge evidence: London, Stockholm).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..utils.calendar import JalaliYear


class CityTransport:
    def __init__(self, ids, C, L, m, days, hdd, ef, bus_co2, bus_mj, bus_share, bus_riders, road_co2, road_mj, speed,
                 h_free, h0):
        self.ids, self.C, self.L = list(ids), np.asarray(C, float), np.asarray(L, float)
        self.m, self.days, self.hdd = np.asarray(m, float), np.asarray(days, float), np.asarray(hdd, float)
        self.ef, self.bus_co2, self.bus_mj = float(ef), float(bus_co2), float(bus_mj)
        self.bus_share, self.bus_riders = np.asarray(bus_share, float), np.asarray(bus_riders, float)
        self.road_co2, self.road_mj, self.speed = np.asarray(road_co2, float), np.asarray(road_mj, float), float(speed)
        self.h0 = np.asarray(h0, float)
        self.h_free = np.minimum(np.asarray(h_free, float), self.h0)        # no negative delay
        self.D = len(self.ids)

    def time_ratio(self, x: np.ndarray, eta: float) -> np.ndarray:
        """(D,) travel time per km relative to the baseline, for a car-volume ratio x of each district."""
        return (self.h_free + (self.h0 - self.h_free) * np.asarray(x, float) ** eta) / self.h0

    def vkt_by_district(self, trips: np.ndarray) -> np.ndarray:
        """(D,) vehicle-km of an (o, d) trip matrix, half to the origin and half to the destination district."""
        v = trips * self.L
        return 0.5 * (v.sum(1) + v.sum(0))

    @classmethod
    def from_config(cls, cfg: dict, year: int, F: np.ndarray, ids: list[str]) -> "CityTransport":
        """F (N cells x D districts): cell-district shares of the analysis grid; ids: district ids in F's order."""
        r, raw, y = cfg["rl"], Config.path(cfg, "raw"), int(year)
        col = {d: i for i, d in enumerate(ids)}
        jy = JalaliYear(y)
        days = np.array(jy.days, float)

        o = r["od"]
        od = pd.read_csv(raw / o["file"].format(year=y), encoding="utf-8-sig")
        od = od[od[o["origin"]].isin(col) & od[o["destination"]].isin(col)]
        i, j = od[o["origin"]].map(col).values, od[o["destination"]].map(col).values
        C, L = np.zeros((len(ids),) * 2), np.zeros((len(ids),) * 2)
        C[i, j] = od[o["restricted"]].sum(axis=1).values
        L[i, j] = od[o["distance"]].values

        p = r["profile"]
        prof = pd.read_csv(raw / p["file"].format(year=y), encoding="utf-8-sig").sort_values(p["month"])
        share = prof[p["share"]].values.astype(float)
        m = share * days.sum()

        s = r["l1_lines"]
        lines = pd.read_csv(raw / s["file"].format(year=y), encoding="utf-8-sig")
        lines = lines[lines[s["sector_column"]] == s["sector"]]
        by = lines.groupby(s["subsector_column"])[[s["activity_column"], s["co2_column"]]].sum()
        factors = cfg["inventory"]["l1"]["factors"]
        mj = {k: by.loc[k, s["activity_column"]] * float(s["activity_scale"]) * float(np.prod([factors[f][0] for f in fs]))
              for k, fs in s["energy_factors"].items()}
        car_share = by.loc[s["car"], s["co2_column"]].sum() / by[s["co2_column"]].sum()

        c = r["city"]
        city = pd.read_csv(raw / c["file"], encoding="utf-8-sig").set_index(c["year"]).loc[y]
        t = r["district_trips"]
        dt = pd.read_csv(raw / t["file"], encoding="utf-8-sig")
        dt = dt[dt[t["year"]] == y].set_index(t["district"])[t["attraction"]].reindex(ids)
        if dt.isna().any():
            raise ValueError(f"district trips of {y} miss {list(dt.index[dt.isna()])}")
        bus_share = (dt / dt.sum()).values
        bus_riders = float(city[c["trips_M"]]) * 1e6 * float(city[c["bus_share"]]) * bus_share

        g = r["road_grid"]
        grid = pd.read_csv(raw / g["file"].format(year=y), encoding="utf-8-sig").sort_values(g["cell"])
        road = np.zeros(F.shape[0])
        road[grid[g["cell"]].values] = grid[g["column"]].values
        road_co2 = share[:, None] * (road @ F)[None]
        road_mj = share * sum(mj.values())
        # the baseline OD reproduces the car share of the road CO2 inside the districts
        ef = car_share * road_co2.sum() / (days.sum() * (C * L).sum())

        # speeds: free flow from the RQ1 links (VKT-weighted), congested = the REPORTED district speed (the RQ1
        # assignment is far too congested in the centre: modelled 6-10 km/h vs reported 13-27 km/h)
        k = r["links"]
        lk = pd.read_csv(raw / k["file"].format(year=y), encoding="utf-8-sig")
        lk = lk[lk[k["district"]].isin(col) & (lk[k["free_speed"]] > 0)]
        w, d = (lk[k["volume"]] * lk[k["length"]]).values, lk[k["district"]].map(col).values
        h_free = np.bincount(d, w / lk[k["free_speed"]].values, len(ids)) / np.maximum(np.bincount(d, w, len(ids)), 1e-12)
        v = r["district_speeds"]
        rep = pd.read_csv(raw / v["file"].format(year=y), encoding="utf-8-sig", index_col=0)[v["speed"]].reindex(ids)
        if rep.isna().any():
            raise ValueError(f"reported speeds of {y} miss {list(rep.index[rep.isna()])}")
        return cls(ids, C, L, m, days, prof[p["hdd"]].values, ef, by.loc[s["bus"], s["co2_column"]], mj[s["bus"]],
                   bus_share, bus_riders, road_co2, road_mj, city[c["speed"]], h_free, 1 / rep.values)

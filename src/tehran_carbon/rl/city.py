"""Baseline transport and emissions of one study year on the D municipal districts (inputs of the RQ3 simulator).

All quantities come from the RQ1 frozen outputs and the transport tables (settings: rl in the rq3 experiments; see
docs/preregistration_rq3.md):
  C (D, D)        restricted car vehicle trips o -> d of a typical day (gravity OD of RQ1)
  L (D, D)        trip distance in km
  m (12,)         typical days per month of car activity: annual x gasoline_share_t (sums to the days of the year)
  ef              t CO2 per car vehicle-km, calibrated so that the baseline reproduces the L1 car CO2
  bus_*           annual CO2 / energy of the municipal buses, their service share and riders per day by district
  road_co2 (12, D) road CO2 of each district and month (L2 grid x monthly profile), road_mj (12,) road fuel energy
  links           peak v/c, free-flow speed and VKT weight of every road link, with its district (BPR speeds)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..utils.calendar import JalaliYear


class CityTransport:
    def __init__(self, ids, C, L, m, days, hdd, ef, bus_co2, bus_mj, bus_share, bus_riders, road_co2, road_mj, speed,
                 link_district, link_w, link_vff, link_vc, bpr):
        self.ids, self.C, self.L = list(ids), np.asarray(C, float), np.asarray(L, float)
        self.m, self.days, self.hdd = np.asarray(m, float), np.asarray(days, float), np.asarray(hdd, float)
        self.ef, self.bus_co2, self.bus_mj = float(ef), float(bus_co2), float(bus_mj)
        self.bus_share, self.bus_riders = np.asarray(bus_share, float), np.asarray(bus_riders, float)
        self.road_co2, self.road_mj, self.speed = np.asarray(road_co2, float), np.asarray(road_mj, float), float(speed)
        self.link_district, self.link_w = np.asarray(link_district, int), np.asarray(link_w, float)
        self.link_vff, self.link_vc = np.asarray(link_vff, float), np.asarray(link_vc, float)
        self.alpha, self.beta = float(bpr["alpha"]), float(bpr["beta"])
        self.D = len(self.ids)
        self.base_hours = self._hours_per_km(np.ones(self.D))

    def _hours_per_km(self, x: np.ndarray) -> np.ndarray:
        """(D,) VKT-weighted hours per km on the links of each district when its car volume is scaled by x (BPR)."""
        v = self.link_vff / (1 + self.alpha * (self.link_vc * x[self.link_district]) ** self.beta)
        return np.bincount(self.link_district, self.link_w / v, minlength=self.D)

    def time_ratio(self, x: np.ndarray) -> np.ndarray:
        """(D,) travel time per km relative to the baseline, for a car-volume ratio x of each district."""
        return np.where(self.base_hours > 0, self._hours_per_km(x) / np.maximum(self.base_hours, 1e-12), 1.0)

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
        car_co2 = by.loc[s["car"], s["co2_column"]].sum()
        ef = car_co2 / (days.sum() * (C * L).sum())

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

        k = r["links"]
        lk = pd.read_csv(raw / k["file"].format(year=y), encoding="utf-8-sig")
        lk = lk[lk[k["district"]].isin(col) & (lk[k["free_speed"]] > 0)]
        return cls(ids, C, L, m, days, prof[p["hdd"]].values, ef, by.loc[s["bus"], s["co2_column"]], mj[s["bus"]],
                   bus_share, bus_riders, road_co2, road_mj, city[c["speed"]], lk[k["district"]].map(col).values,
                   (lk[k["volume"]] * lk[k["length"]]).values, lk[k["free_speed"]].values, lk[k["vc"]].values, r["bpr"])

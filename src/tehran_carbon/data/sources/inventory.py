"""Traffic and energy domains from the RQ1 emission inventory (frozen L1/L2 outputs for now).

Monthly CO2 of a sector in a cell = annual L2 cell value (sum of the configured columns) x the L1 monthly share
of that sector (config: sectors: {layer: {columns: [...], share: <profile column>}}).
traffic adds link statistics of the L2 traffic assignment (each link is assigned to the cell of its midpoint):
    road_km_<class>   road length by class      vkt_peak   peak-hour vehicle-km      vc_mean   length-weighted v/c
energy adds the city-wide heating degree-days of each month.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import SOURCES, Source


class InventorySource(Source):
    def load(self, year):
        g, cell, month = self.grid, self.p["cell_column"], self.p["month_column"]
        grid = pd.read_csv(self.file("l2_grid_file", year), encoding="utf-8-sig").sort_values(cell)
        if len(grid) != g.n or not (grid[cell].values == np.arange(g.n)).all():
            raise ValueError(f"L2 grid of {year} does not cover the {g.H}x{g.W} grid")
        self.profiles = pd.read_csv(self.file("l1_profiles_file", year), encoding="utf-8-sig").sort_values(month)
        if len(self.profiles) != 12:
            raise ValueError(f"L1 monthly profiles of {year} must have 12 rows")
        out = {}
        for layer, s in self.p["sectors"].items():
            annual = grid[s["columns"]].sum(axis=1).values.reshape(g.H, g.W)
            out[layer] = self.profiles[s["share"]].values[:, None, None] * annual[None]
        return out


@SOURCES.register("traffic")
class TrafficSource(InventorySource):
    name = "traffic"

    def load(self, year):
        out = super().load(year)
        g, c = self.grid, self.p["link_columns"]
        L = pd.read_csv(self.file("links_file", year), encoding="utf-8-sig")
        row, col, inside = g.rowcol(L[c["lon"]].values, L[c["lat"]].values)
        L = L.assign(cell=row * g.W + col)[inside]
        # OSM 'highway' may hold several tags ("trunk;primary") and *_link variants: first tag, link -> its class
        cls = L[c["road_class"]].astype(str).str.split(";").str[0].str.replace("_link", "", regex=False)
        cls = cls.map(self.p["road_classes"])
        length = L[c["length_km"]]
        for name in sorted(set(self.p["road_classes"].values())):
            km = length.where(cls == name, 0.0).groupby(L.cell).sum()
            out[name] = km.reindex(range(g.n), fill_value=0.0).values.reshape(g.H, g.W)
        vkt = (L[c["volume"]] * length).groupby(L.cell).sum().reindex(range(g.n), fill_value=0.0)
        out["vkt_peak"] = vkt.values.reshape(g.H, g.W)
        vc = ((L[c["vc"]] * length).groupby(L.cell).sum() / length.groupby(L.cell).sum())
        out["vc_mean"] = vc.reindex(range(g.n), fill_value=0.0).values.reshape(g.H, g.W)
        return out


@SOURCES.register("energy")
class EnergySource(InventorySource):
    name = "energy"

    def load(self, year):
        out = super().load(year)
        hdd = self.profiles[self.p["hdd_column"]].values.astype(float)
        out["hdd"] = np.broadcast_to(hdd[:, None, None], (12, self.grid.H, self.grid.W)).copy()
        return out

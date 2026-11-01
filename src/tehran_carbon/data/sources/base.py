"""Base class of the data-source loaders.

A source turns raw files into named layers on the L0 grid for one Jalali year:
    load(year) -> {layer name: (H, W) static array or (12, H, W) monthly array}, in physical units
Its settings are cfg["data"]["inputs"][<name>]. A new data source = one subclass + one config block.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ...config import Config
from ...registry import Registry
from ...utils.calendar import JalaliYear
from ..grid import Grid

SOURCES = Registry("data source")


class Source:
    name = ""

    def __init__(self, cfg: dict, grid: Grid):
        self.cfg = cfg
        self.grid = grid
        self.raw = Config.path(cfg, "raw")
        self.p = cfg["data"]["inputs"][self.name]

    def file(self, key: str, year: int | None = None) -> Path:
        """Raw file of setting `key`: a path, or {year: path} for files that differ by year. The path may contain {year}
        and {gyear} (Gregorian year of Nowruz). `year_proxy: {1404: 1403}` reads another year's file where a year has
        no data of this source (stated as a limitation wherever it is used)."""
        spec = self.p[key]
        if isinstance(spec, dict):
            spec = spec[year]
        year = (self.p.get("year_proxy") or {}).get(year, year) if year else year
        name = spec.format(year=year, gyear=JalaliYear(year).gregorian_year if year else None)
        path = self.raw / name
        if not path.exists():
            raise FileNotFoundError(f"raw file missing: {path} (see data/raw/MANIFEST.csv and README)")
        return path

    def load(self, year: int) -> dict[str, np.ndarray]:
        raise NotImplementedError

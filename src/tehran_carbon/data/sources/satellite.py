"""TROPOMI tropospheric NO2 (monthly composites; the target) and VIIRS night lights (annual), exported on L0 by GEE."""
from __future__ import annotations

import numpy as np
import rasterio

from ...utils.calendar import JalaliYear
from .base import SOURCES, Source


@SOURCES.register("satellite")
class SatelliteSource(Source):
    name = "satellite"

    def load(self, year):
        jy = JalaliYear(year)
        with rasterio.open(self.file("file", year)) as r:
            self.grid.check_raster(r.transform, r.shape)
            bands = {d: r.read(i + 1).astype(np.float64) for i, d in enumerate(r.descriptions)}
        # Jalali month -> Gregorian month of largest overlap (Farvardin -> April ... Esfand -> March)
        no2 = np.stack([bands[self.p["no2_band"].format(gy=gy, gm=gm)] for gy, gm in jy.gregorian_months])
        viirs = np.clip(np.nan_to_num(bands[self.p["viirs_band"].format(gyear=jy.gregorian_year)]), 0, None)
        return {"no2": no2, "viirs": viirs}

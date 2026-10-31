"""City context: GHSL population and building volume, OSM industrial land share, ERA5 monthly temperature and wind."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...utils.calendar import JalaliYear
from .base import SOURCES, Source


@SOURCES.register("context")
class ContextSource(Source):
    name = "context"

    def load(self, year):
        import geopandas as gpd
        import rasterio
        from rasterio.warp import Resampling, reproject
        g, p = self.grid, self.p
        out = {}

        # GHSL 100 m -> L0 cells (sum)
        with rasterio.open(self.file("ghsl_file")) as src:
            for name, band in p["ghsl_bands"].items():
                dst = np.zeros((g.H, g.W))
                reproject(src.read(band).astype(float), dst, src_transform=src.transform, src_crs=src.crs,
                          dst_transform=g.transform, dst_crs="EPSG:4326", resampling=Resampling.sum)
                out[name] = np.clip(dst, 0, None)

        # share of each cell covered by OSM industrial / power land use, areas in a metric CRS
        cells = g.cell_polygons().to_crs(p["area_crs"])
        ind = gpd.read_file(self.file("industrial_file"))[["geometry"]].to_crs(p["area_crs"])
        ind["geometry"] = ind.buffer(0)
        ov = gpd.overlay(ind.dissolve().explode(index_parts=False), cells, how="intersection")
        frac = (ov.area.groupby(ov.cell).sum() / cells.set_index("cell").area).reindex(range(g.n)).fillna(0.0)
        out["industrial_frac"] = frac.values.reshape(g.H, g.W).clip(0, 1)

        # city-wide monthly weather (same value in every cell)
        c = p["weather_columns"]
        met = pd.read_csv(self.file("weather_file"), parse_dates=[c["date"]])
        met["t2m"] = met[c["t2m"]] + (met[c["t2m_bias"]] if c.get("t2m_bias") else 0.0)
        month = JalaliYear(year).month_of(met[c["date"]])
        means = met.assign(m=month.values)[month.values > 0].groupby("m")[["t2m", c["wind"]]].mean()
        if len(means) != 12:
            raise ValueError(f"weather file does not cover all 12 months of {year}")
        out["t2m"] = np.broadcast_to(means["t2m"].values[:, None, None], (12, g.H, g.W)).copy()
        out["wind"] = np.broadcast_to(means[c["wind"]].values[:, None, None], (12, g.H, g.W)).copy()
        return out

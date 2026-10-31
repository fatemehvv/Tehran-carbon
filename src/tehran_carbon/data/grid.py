"""The L0 analysis grid: regular lon/lat cells, cell = row * W + col, north-west corner at (lon0, lat0)."""
from __future__ import annotations

import numpy as np


class Grid:
    def __init__(self, cfg: dict):
        g = cfg["grid"]
        self.H, self.W = int(g["H"]), int(g["W"])
        self.lon0, self.lat0 = float(g["lon0"]), float(g["lat0"])
        self.cells_per_degree = int(g["cells_per_degree"])
        self.res = 1.0 / self.cells_per_degree
        self.n = self.H * self.W
        lat_c = self.lat0 - self.H * self.res / 2
        self.km_y = float(g["km_per_degree"]) / self.cells_per_degree                     # cell height in km
        self.km_x = self.km_y * np.cos(np.radians(lat_c))                                 # cell width in km
        cols, rows = np.meshgrid(np.arange(self.W), np.arange(self.H))
        self.lon = self.lon0 + (cols + 0.5) * self.res                                    # cell centres (H, W)
        self.lat = self.lat0 - (rows + 0.5) * self.res

    @property
    def transform(self):
        from affine import Affine
        return Affine(self.res, 0.0, self.lon0, 0.0, -self.res, self.lat0)

    def rowcol(self, lon, lat) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cell indices of points and a mask of the points inside the grid."""
        col = np.floor((np.asarray(lon, float) - self.lon0) / self.res).astype(int)
        row = np.floor((self.lat0 - np.asarray(lat, float)) / self.res).astype(int)
        return row, col, (row >= 0) & (row < self.H) & (col >= 0) & (col < self.W)

    def check_raster(self, transform, shape) -> None:
        """Fail loudly if a raster is not exactly on this grid."""
        ok = (abs(transform.a - self.res) < 1e-9 and abs(transform.e + self.res) < 1e-9
              and abs(transform.c - self.lon0) < 1e-6 and abs(transform.f - self.lat0) < 1e-6
              and tuple(shape) == (self.H, self.W))
        if not ok:
            raise ValueError(f"raster {shape} {transform} is not on the {self.H}x{self.W} L0 grid")

    def cell_polygons(self):
        """GeoDataFrame of all cells (column `cell`), EPSG:4326."""
        import geopandas as gpd
        from shapely.geometry import box
        r = self.res
        geoms = [box(self.lon0 + c * r, self.lat0 - (rw + 1) * r, self.lon0 + (c + 1) * r, self.lat0 - rw * r)
                 for rw in range(self.H) for c in range(self.W)]
        return gpd.GeoDataFrame({"cell": np.arange(self.n)}, geometry=geoms, crs=4326)

    def distances_km(self, lon, lat) -> np.ndarray:
        """(n_points, H, W) distance in km from each point to every cell centre (local flat approximation)."""
        dx = (self.lon[None] - np.asarray(lon, float)[:, None, None]) * self.cells_per_degree * self.km_x
        dy = (self.lat[None] - np.asarray(lat, float)[:, None, None]) * self.cells_per_degree * self.km_y
        return np.hypot(dx, dy)

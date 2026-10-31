"""Pretrained satellite embeddings as frozen features (the foundation model itself is not trained).

Default: Google DeepMind AlphaEarth Foundations, GEE `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` (64-d, 10 m, annual),
averaged per L0 cell by gee/export_alphaearth_L0.js. Any other per-cell embedding GeoTIFF on the L0 grid works the
same way (e.g. a TorchGeo Landsat encoder run offline): set `file` to it.

The bands are reduced by PCA fitted on all cells of all study years together (no target information, one shared
basis for both years) and returned as emb_00 ... emb_<k-1>.
"""
from __future__ import annotations

import numpy as np

from .base import SOURCES, Source


@SOURCES.register("embeddings")
class EmbeddingSource(Source):
    name = "embeddings"

    def load(self, year):
        import rasterio
        from sklearn.decomposition import PCA
        stack = {}
        for y in self.cfg["years"]:
            with rasterio.open(self.file("file", y)) as r:
                self.grid.check_raster(r.transform, r.shape)
                stack[y] = np.nan_to_num(r.read().astype(np.float64))          # (bands, H, W)
        B = stack[year].shape[0]
        pca = PCA(n_components=int(self.p["pca_dims"]), random_state=int(self.cfg["seed"]))
        pca.fit(np.concatenate([s.reshape(B, -1).T for s in stack.values()]))
        z = pca.transform(stack[year].reshape(B, -1).T).T.reshape(-1, self.grid.H, self.grid.W)
        return {f"emb_{i:02d}": z[i] for i in range(z.shape[0])}

"""Figures of a finished run: observed vs out-of-fold predicted target (mean over the time steps) and, for gated fusion,
the mean weight of every input branch (which domain the model relies on where)."""
from __future__ import annotations

import numpy as np


class RunMaps:
    DPI = 160

    def __init__(self, run, data, grid):
        self.run, self.data, self.grid = run, data, grid
        g = grid
        self.extent = [g.lon0, g.lon0 + g.W * g.res, g.lat0 - g.H * g.res, g.lat0]

    def _field(self, values: np.ndarray) -> np.ndarray:
        out = np.full(self.grid.n, np.nan)
        out[self.data.cells] = values
        return out.reshape(self.grid.H, self.grid.W)

    def draw(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        data, run = self.data, self.run
        for y in data.years:
            f = run.dir / "preds" / f"cv_{y}.npz"
            if not f.exists():
                continue
            pred = np.load(f)["pred"]
            obs = data.z[y].reshape(data.z[y].shape[0], -1)[:, data.cells]
            wf = run.dir / "preds" / f"weights_{y}.npz"
            wts = np.load(wf) if wf.exists() else None
            n_w = 0 if wts is None else len(wts["branches"])
            fig, ax = plt.subplots(1, 2 + n_w, figsize=(4.2 * (2 + n_w), 3.2), squeeze=False)
            ax = ax[0]
            lim = float(np.nanpercentile(np.abs(np.nanmean(obs, 0)), 98))
            for a, (title, v) in zip(ax, [(f"{y} observed z", np.nanmean(obs, 0)), (f"{y} out-of-fold predicted z", np.nanmean(pred, 0))]):
                im = a.imshow(self._field(v), extent=self.extent, cmap="RdBu_r", vmin=-lim, vmax=lim)
                a.set_title(title, fontsize=9)
                fig.colorbar(im, ax=a, shrink=0.75)
            for i in range(n_w):
                im = ax[2 + i].imshow(self._field(np.nanmean(wts["weights"][..., i], 0)), extent=self.extent,
                                      cmap="viridis", vmin=0, vmax=1)
                ax[2 + i].set_title(f"weight: {wts['branches'][i]}", fontsize=9)
                fig.colorbar(im, ax=ax[2 + i], shrink=0.75)
            for a in ax:
                a.tick_params(labelsize=7)
            fig.tight_layout()
            fig.savefig(run.path("figures", f"maps_{y}.png"), dpi=self.DPI)
            plt.close(fig)

"""Remaining registered checks of the fusion paper (python main.py fusion check). Nothing is trained.

M4 placebo quality (preregistration_fusion_v1.md). For every realisation of fus2_placebo_v1, on the full grid where
   the surrogates are made, every placebo layer is compared with its real layer (annual map, with the log1p of the
   dataset where the dataset uses it):
     |Spearman(placebo, real)|                          location kept? (should be near 0)
     max |correlogram(placebo) - correlogram(real)|      over the bins up to 20 km (texture kept?)
   Operational rule (fixed in this code before it was first run): a realisation passes if the median over its layers
   of |Spearman| is < 0.10 and the median of the correlogram difference is <= 0.05. If more than two realisations
   fail, the placebo verdicts are labelled "placebo check failed".
Lineage audit (descriptive). How much of each energy, traffic and station layer the context layers predict: block-CV
   R2 (the 12 blocks of 22 km, 2-km buffer) of the annual map of the layer (lag 0, log1p as in the dataset) from the
   context layers and their box means, with OLS and gradient boosting, per year.
Figure 1 of the paper: a real layer and three of its placebo realisations.
Outputs (run folder): fusion_placebo_check.csv, fusion_lineage.csv, fusion_checks.json; figure in
docs/paper/fusion_paper/figures/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..config import ROOT, Config
from ..data.dataset import CellMonthDataset
from ..data.grid import Grid
from ..data.sources.base import SOURCES
from ..data.sources.placebo import PlaceboSource
from ..data.splits import SpatialSplit
from ..evaluation.inference import coords_km, correlogram

FIGURES = ROOT / "docs" / "paper" / "fusion_paper" / "figures"
CONTEXT = ["pop", "vol_total", "vol_nonres", "viirs", "industrial_frac"]


def annual(a: np.ndarray) -> np.ndarray:
    return a.mean(0) if a.ndim == 3 else a


class FusionChecks:
    def __init__(self, cfg: dict, year: int = 1402):
        self.cfg, self.year = cfg, int(year)
        cfgs = Config().resolve("fus2_placebo_v1")
        self.plc = {}
        for c in cfgs:                                        # one config per realisation (the layers do not depend
            r = int(c["data"]["inputs"]["placebo"]["realization"])   # on the branches of the model)
            self.plc.setdefault(r, c)
        self.base = Config().resolve("paper2_rq1_ablation_v1")[0]

    # ------------------------------------------------------------------ M4
    def placebo(self, run):
        c0 = next(iter(self.plc.values()))
        grid = Grid(c0)
        p = c0["data"]["inputs"]["placebo"]
        log1p = set(c0["data"].get("log1p") or [])
        real = {d: SOURCES.get(d)(c0, grid).load(self.year) for d in p["domains"]}
        edges = np.arange(0.0, 20.0 + 1e-9, 2.0)
        xy = coords_km(np.arange(grid.n), grid)
        rows, maps = [], {}
        for r, c in sorted(self.plc.items()):
            run.log(f"M4: realisation {r}")
            out = PlaceboSource(c, grid).load(self.year)
            for d, layers in p["domains"].items():
                for n in layers:
                    a, s = annual(np.asarray(real[d][n], float)), annual(np.asarray(out["plc_" + n], float))
                    if n in log1p:
                        a, s = np.log1p(a), np.log1p(s)
                    if np.ptp(a) == 0:
                        continue
                    rho = float(spearmanr(a.ravel(), s.ravel())[0])
                    ca = correlogram(a.ravel()[None], xy, edges, seed=0)
                    cs = correlogram(s.ravel()[None], xy, edges, seed=0)
                    rows.append(dict(realisation=r, domain=d, layer=n, abs_spearman=abs(rho),
                                     max_correlogram_diff=float(np.nanmax(np.abs(ca - cs)))))
                    if n == "inv_residential":
                        maps["real"], maps[f"plc{r}"] = a, s
        t = pd.DataFrame(rows)
        per = t.groupby("realisation").agg(median_abs_spearman=("abs_spearman", "median"),
                                           median_correlogram_diff=("max_correlogram_diff", "median")).reset_index()
        per["passed"] = (per.median_abs_spearman < 0.10) & (per.median_correlogram_diff <= 0.05)
        failed = int((~per.passed).sum())
        verdict = "passed" if failed <= 2 else "placebo check failed"
        run.save_table("fusion_placebo_check.csv", t)
        run.save_table("fusion_placebo_check_summary.csv", per)
        run.log("M4 per realisation:\n" + per.round(3).to_string(index=False) + f"\nM4: {verdict} ({failed} failed)")
        self.figure_maps(maps, grid, c0)
        return dict(verdict=verdict, failed=failed, per_realisation=per.to_dict("records"))

    def figure_maps(self, maps, grid, cfg):
        from ..viz.paper_figures import save, style
        from ..viz.figures import INK2, WIDTH
        if "real" not in maps:
            return
        plt = style()
        data = CellMonthDataset.build(self.base)
        dom = data.label_domain
        keys = ["real"] + [k for k in ("plc0", "plc1", "plc2") if k in maps]
        fig, axes = plt.subplots(1, len(keys), figsize=(WIDTH, WIDTH / len(keys) * 0.62))
        vmin, vmax = np.percentile(maps["real"], [1, 99.5])
        for ax, k in zip(axes, keys):
            ax.imshow(maps[k], cmap="Greys", vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.contour(dom.astype(float), levels=[0.5], colors=INK2, linewidths=0.5)
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title("a  Real layer" if k == "real" else f"{'bcd'[int(k[-1])]}  Placebo, realisation {k[-1]}",
                         fontsize=8)
        fig.text(0.5, -0.02, "Residential CO₂ of the inventory (log scale, annual mean); outline: label domain",
                 ha="center", color=INK2, fontsize=7.5)
        save(fig, FIGURES / "fig1_placebo_maps")

    # ------------------------------------------------------------------ lineage
    def lineage(self, run):
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.linear_model import LinearRegression
        data = CellMonthDataset.build(self.base, log=run.log)
        split = SpatialSplit(self.base["split"], Grid(self.base), data.label_domain, seed=int(self.base["seed"]))
        feats = data.features
        xcols = [j for j, b in enumerate(data.base_of) if b in CONTEXT]
        targets = [(m, b) for m in ("energy", "traffic", "pollution") for b in self.base["data"]["modalities"][m]
                   if b not in ("hdd",)]
        rows = []
        for y in data.years:
            A = data.X[y].mean(0)                                  # (H, W, F) annual mean
            Xc = A[..., xcols].reshape(-1, len(xcols))
            for m, b in targets:
                j = feats.index(b)
                yv = A[..., j].ravel()
                if np.ptp(yv[data.cells]) == 0:
                    continue
                for name, make in (("OLS", LinearRegression),
                                   ("GBM", lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                                                random_state=0))):
                    pred = np.full(yv.shape, np.nan)
                    for k in range(split.n_folds):
                        tr, te = split.masks(k)
                        tri, tei = np.flatnonzero(tr.ravel()), np.flatnonzero(te.ravel())
                        mdl = make().fit(Xc[tri], yv[tri])
                        pred[tei] = mdl.predict(Xc[tei])
                    c = data.cells
                    ok = np.isfinite(pred[c])
                    sse = ((yv[c][ok] - pred[c][ok]) ** 2).sum()
                    sst = ((yv[c][ok] - yv[c][ok].mean()) ** 2).sum()
                    rows.append(dict(year=y, domain=m, layer=b, model=name, r2_from_context=float(1 - sse / sst)))
        t = pd.DataFrame(rows)
        run.save_table("fusion_lineage.csv", t)
        run.log("lineage audit (block-CV R2 of each layer from the context layers):\n"
                + t.pivot_table(index=["domain", "layer"], columns=["model", "year"], values="r2_from_context")
                .round(3).to_string())
        return t

    def run(self, run) -> dict:
        m4 = self.placebo(run)
        lin = self.lineage(run)
        out = dict(M4=m4, lineage=lin.to_dict("records"))
        run.save_json("fusion_checks.json", out)
        return {"M4": m4["verdict"], "lineage_rows": len(lin)}

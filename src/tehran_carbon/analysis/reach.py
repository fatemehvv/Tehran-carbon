"""Skill against reach (python main.py paper reach --exp paper3_reach_slx_v1 paper3_reach_sage_v1 paper3_reach_mlp_v1).

The main figure of the paper: for every finished run of the listed experiments, the cross-validated metrics against
the reach of the model, i.e. how far from a cell its prediction can look:
  SLX (baseline)            the largest box half-width of its windows (model.baseline.lags; 0 = own cell)
  neural, no graph          the largest spatial lag of its features (data.spatial_lags; 0 = own cell)
  neural, graph backbone    layers x the largest dilation (model.backbone.scales): 9 cells per SAGE layer
Cells are converted to km with the mean cell size ((km_x + km_y) / 2). For runs that stored every seed, the metric of
every seed is added (mean, SD, min, max), so that the ensemble is not mistaken for a single model.
The zero crossing of the target's correlogram on the same years is drawn as the physical reference (the rule of
docs/paper/preregistration_paper_v2.md).
Outputs: reach_curve.csv, reach_curve.png, reach_correlogram.csv
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml

from ..config import ROOT
from ..data.dataset import CellMonthDataset
from ..data.grid import Grid
from ..evaluation.inference import coords_km, correlogram
from ..evaluation.metrics import monthly_scores
from ..tracking import RunRegistry


def reach_cells(cfg: dict) -> float:
    """How far (box half-width / graph hops x dilation, in cells) a model's prediction of a cell can look."""
    m = cfg["model"]
    if m["type"] == "baseline":
        if m["baseline"]["name"] not in ("ols", "gbm", "rf"):
            return np.nan                                             # physical operators: no fixed window
        return float(max(m["baseline"].get("lags") or [0]))
    b = m["backbone"]
    if b["type"] == "none":
        return float(max(cfg["data"].get("spatial_lags") or [0]))
    return float(int(b["layers"]) * max(int(s) for s in b["scales"]))


def family(cfg: dict) -> str:
    m = cfg["model"]
    if m["type"] == "baseline":
        return f"SLX-{m['baseline']['name'].upper()}"
    return "MLP" if m["backbone"]["type"] == "none" else m["backbone"]["type"].upper()


def zero_crossing(rho: np.ndarray, edges: np.ndarray) -> float:
    """Distance (km) where the correlogram first falls to 0, linear between bin centres."""
    mid = (edges[:-1] + edges[1:]) / 2
    for i in range(1, len(rho)):
        if np.isfinite(rho[i - 1]) and np.isfinite(rho[i]) and rho[i - 1] > 0 >= rho[i]:
            return float(mid[i - 1] + (mid[i] - mid[i - 1]) * rho[i - 1] / (rho[i - 1] - rho[i]))
    return np.nan


class ReachCurve:
    def __init__(self, base_cfg: dict, experiments: list[str]):
        self.cfg, self.experiments = base_cfg, experiments
        self.metrics = list(base_cfg["analysis"]["paper"]["metrics"])

    def run(self, run) -> dict:
        grid = Grid(self.cfg)
        cbar = (grid.km_x + grid.km_y) / 2
        rows, data = [], None
        for exp in self.experiments:
            found = RunRegistry(self.cfg).find(exp).sort_values("finished")
            latest = {r.variant: r for _, r in found.iterrows()}
            if not latest:
                run.log(f"{exp}: no finished runs - skipped")
            for variant, r in latest.items():
                folder = ROOT / r.path
                rcfg = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
                met = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
                if data is None:
                    data = CellMonthDataset.build(rcfg, log=run.log)
                k = reach_cells(rcfg)
                for y, vals in met["cv"].items():
                    row = dict(experiment=exp, variant=variant, family=family(rcfg), year=int(y), reach_cells=k,
                               reach_km=k * cbar, **{m: vals[m] for m in self.metrics})
                    mf = folder / "preds" / f"cv_{y}_members.npz"
                    if mf.exists() and int(y) in data.z:
                        a = np.load(mf)
                        if np.array_equal(a["cells"], data.cells):
                            z = data.z[int(y)].reshape(data.z[int(y)].shape[0], -1)[:, data.cells]
                            hot = data.hot[int(y)].reshape(z.shape[0], -1)[:, data.cells]
                            for m in self.metrics:
                                s = np.array([np.nanmean(monthly_scores(z, hot, p, [m])[m]) for p in a["pred"]])
                                row.update({f"{m}_seed_mean": s.mean(), f"{m}_seed_sd": s.std(ddof=1) if len(s) > 1 else np.nan,
                                            f"{m}_seed_min": s.min(), f"{m}_seed_max": s.max()})
                    rows.append(row)
        if not rows:
            raise ValueError("no finished runs of the listed experiments")
        table = pd.DataFrame(rows).sort_values(["family", "year", "reach_cells"])
        run.save_table("reach_curve.csv", table)

        c = self.cfg["analysis"]["paper"]["correlogram"]
        edges = np.arange(0.0, float(c["max_km"]) + 1e-9, float(c["bin_km"]))
        xy = coords_km(data.cells, grid)
        cg = []
        for y in sorted(table.year.unique()):
            v = data.z[int(y)].reshape(data.z[int(y)].shape[0], -1)[:, data.cells]
            rho = correlogram(v, xy, edges, seed=int(self.cfg["seed"]))
            cg.append(dict(year=int(y), zero_crossing_km=zero_crossing(rho, edges)))
        cg = pd.DataFrame(cg)
        run.save_table("reach_correlogram.csv", cg)
        self.figure(table, float(cg.zero_crossing_km.mean()), run)
        run.log("reach curve:\n" + table[["family", "year", "reach_km"] + self.metrics].round(3).to_string(index=False)
                + f"\ncorrelogram zero crossing: {cg.zero_crossing_km.round(1).tolist()} km")
        return {"rows": len(table), "zero_crossing_km": round(float(cg.zero_crossing_km.mean()), 1)}

    def figure(self, table: pd.DataFrame, zc: float, run) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # one colour per model family, one line style per year; single-reach families (e.g. one GNN depth) are drawn as
        # points with the range over single seeds as an error bar, so the ensemble is not mistaken for one model
        fams = sorted(table.family.unique())
        colour = {f: plt.get_cmap("tab10")(i % 10) for i, f in enumerate(fams)}
        years = sorted(table.year.unique())
        style = {y: ["-", "--", ":", "-."][i % 4] for i, y in enumerate(years)}
        fig, axes = plt.subplots(1, len(self.metrics), figsize=(5.2 * len(self.metrics), 4.2), squeeze=False)
        for ax, m in zip(axes[0], self.metrics):
            for (fam, y), g in table.groupby(["family", "year"]):
                g = g.dropna(subset=["reach_km"]).sort_values("reach_km")
                lo, hi = f"{m}_seed_min", f"{m}_seed_max"
                seeds = lo in g and g[lo].notna().any()
                if len(g) > 1:
                    ax.plot(g.reach_km, g[m], color=colour[fam], ls=style[y], marker="o", ms=3.5, lw=1.4,
                            label=f"{fam} {y}")
                    if seeds:
                        ax.fill_between(g.reach_km, g[lo], g[hi], color=colour[fam], alpha=0.15, lw=0)
                else:
                    # the seed range is drawn on its own: the ensemble can lie outside it (it can beat every seed)
                    x = (g.reach_km + (0.4 if years.index(y) else -0.4)).to_numpy()
                    if seeds:
                        ax.vlines(x, g[lo], g[hi], color=colour[fam], lw=1.2)
                        ax.scatter(np.r_[x, x], np.r_[g[lo], g[hi]], color=colour[fam], marker="_", s=40)
                    ax.plot(x, g[m], color=colour[fam], ls="none", marker="s" if years.index(y) else "o", ms=5,
                            label=f"{fam} {y}" + (" (ensemble; bar = single seeds)" if seeds else ""))
            if np.isfinite(zc):
                ax.axvline(zc, color="0.4", ls="--", lw=1)
                ax.text(zc, ax.get_ylim()[0], " correlation range", rotation=90, va="bottom", ha="right",
                        fontsize=8, color="0.4")
            ax.set_xlabel("reach (km)")
            ax.set_ylabel(m)
            ax.grid(alpha=0.3)
        axes[0][-1].legend(fontsize=7, frameon=False, loc="lower right")
        fig.tight_layout()
        fig.savefig(run.path("reach_curve.png"), dpi=200)
        plt.close(fig)

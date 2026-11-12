"""Checks asked for by the reviewers (python main.py paper review). Nothing is trained; GNN and SLX runs are read.

  years        how alike are the targets of the years? Spearman between the annual-mean z maps of every pair of years
               and the mean over months of the same-month Spearman (is the held-out year new information?)
  correlogram  zero crossing of the target correlogram (the reach rule R*) for the raw target, after removing a planar
               and a quadratic trend in the coordinates, and on central crops of the domain (80 %, 60 % of its extent):
               how much of R* is the domain and the large-scale gradient? Also the effective sample size of each.
  interpolation  baselines that use no predictor at all, only the target of the training cells of the same month,
               with the folds and buffer of the paper: inverse-distance weighting and a Gaussian process on the
               coordinates (kernel fitted once per year on the annual mean of a subsample). Scored like the models,
               next to the GNN and the reach-matched SLX-OLS read from their runs.
  seeds        calibration of the exact block sign-flip test: every pair of single-seed GNNs of the same run is a
               comparison with no true difference; report how often p_greater falls below 0.05 and 0.025.
Outputs: review_years.csv, review_correlogram.csv, review_interpolation.csv, review_seed_calibration.csv,
         review_checks.json
"""
from __future__ import annotations

from itertools import combinations
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

from ..config import ROOT, Config
from ..data.dataset import CellMonthDataset
from ..data.grid import Grid
from ..data.splits import SpatialSplit
from ..evaluation.inference import coords_km, correlogram, effective_n, pair_counts, signflip_pooled
from ..evaluation.robustness import mean_score
from ..tracking import RunRegistry
from .reach import zero_crossing

DEV = "paper2_rq2_sage_v1"                 # data, folds and the GNN of the development years
HELD_OUT = "paper2_c1404_sage5_v1"         # data of the held-out year
SLX_MATCHED = ("paper2_slx_reach_v1", {"model.baseline.name": "ols", "model.baseline.lags": [1, 3, 9, 15, 27]})


def latest(cfg: dict, experiment: str, where: dict | None = None):
    """(run_id, folder) of the latest finished run of experiment whose config matches where, or None."""
    rows = RunRegistry(cfg).find(experiment).sort_values("finished")
    for _, r in rows.iloc[::-1].iterrows():
        folder = ROOT / r.path
        rcfg = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
        if all(Config.get(rcfg, k, None) == v for k, v in (where or {}).items()):
            return r.run_id, folder
    return None


def detrend(v: np.ndarray, xy: np.ndarray, order: int) -> np.ndarray:
    """v (T, n) minus a least-squares polynomial trend in the coordinates (order 1: plane; 2: quadratic), per step."""
    x, y = (xy - xy.mean(0)).T / np.abs(xy - xy.mean(0)).max()
    cols = [np.ones_like(x), x, y] + ([x * x, x * y, y * y] if order >= 2 else [])
    A = np.stack(cols, 1)
    out = np.full(v.shape, np.nan)
    for t in range(v.shape[0]):
        ok = np.isfinite(v[t])
        if ok.sum() > A.shape[1] + 2:
            coef = np.linalg.lstsq(A[ok], v[t][ok], rcond=None)[0]
            out[t][ok] = v[t][ok] - A[ok] @ coef
    return out


def idw(xy_tr: np.ndarray, v_tr: np.ndarray, xy_te: np.ndarray, power: float = 2.0) -> np.ndarray:
    d = np.maximum(cdist(xy_te, xy_tr), 1e-6)
    w = d ** -power
    return (w @ v_tr) / w.sum(1)


class ReviewChecks:
    def __init__(self, base_cfg: dict):
        self.cfg = base_cfg
        self.pc = base_cfg["analysis"]["paper"]
        c = self.pc["correlogram"]
        self.edges = np.arange(0.0, float(c["max_km"]) + 1e-9, float(c["bin_km"]))
        sf = self.pc["signflip"]
        self.sf = dict(max_exact=int(sf["max_exact"]), n_random=int(sf["n_random"]), seed=int(sf["seed"]))
        self.dev_cfg = Config().resolve(DEV)[0]
        self.held_cfg = Config().resolve(HELD_OUT)[0]

    # ------------------------------------------------------------------ data
    def load(self, run):
        dev = CellMonthDataset.build(self.dev_cfg, log=run.log)
        held = CellMonthDataset.build(self.held_cfg, log=run.log)
        grid = Grid(self.dev_cfg)
        cells = dev.cells
        z = {y: dev.z[y].reshape(dev.z[y].shape[0], -1)[:, cells] for y in dev.years}
        hot = {y: dev.hot[y].reshape(dev.z[y].shape[0], -1)[:, cells] for y in dev.years}
        if np.array_equal(held.cells, cells):
            for y in held.years:
                z.setdefault(y, held.z[y].reshape(held.z[y].shape[0], -1)[:, cells])
        else:
            run.log("held-out dataset has another label domain: years check on the development years only")
        return dev, grid, cells, z, hot

    # ------------------------------------------------------------------ checks
    def years(self, z: dict, run) -> pd.DataFrame:
        rows = []
        for a, b in combinations(sorted(z), 2):
            ma, mb = np.nanmean(z[a], 0), np.nanmean(z[b], 0)
            ok = np.isfinite(ma) & np.isfinite(mb)
            same = [spearmanr(z[a][t], z[b][t], nan_policy="omit")[0] for t in range(min(len(z[a]), len(z[b])))]
            rows.append(dict(year_a=a, year_b=b, spearman_annual_mean=float(spearmanr(ma[ok], mb[ok])[0]),
                             spearman_same_month_mean=float(np.nanmean(same)),
                             spearman_same_month_min=float(np.nanmin(same))))
        t = pd.DataFrame(rows)
        run.log("target similarity between years:\n" + t.round(3).to_string(index=False))
        return t

    def correlogram_checks(self, z: dict, xy: np.ndarray, dev_years, run) -> pd.DataFrame:
        rows = []
        centre, half = xy.mean(0), np.abs(xy - xy.mean(0)).max(0)
        for y in dev_years:
            for crop in (1.0, 0.8, 0.6):
                keep = np.all(np.abs(xy - centre) <= crop * half + 1e-9, 1)
                for name, order in (("raw", 0), ("planar trend removed", 1), ("quadratic trend removed", 2)):
                    v = z[y][:, keep]
                    if order:
                        v = detrend(v, xy[keep], order)
                    rho, se = correlogram(v, xy[keep], self.edges, seed=int(self.cfg["seed"]), return_se=True)
                    counts = pair_counts(xy[keep], self.edges)
                    rows.append(dict(year=y, crop=crop, cells=int(keep.sum()), target=name,
                                     extent_km=float(2 * crop * half.max()),
                                     zero_crossing_km=zero_crossing(rho, self.edges),
                                     n_eff=effective_n(rho, se, counts=counts, n=int(keep.sum())),
                                     rho_8_10km=float(rho[np.searchsorted(self.edges, 8.0)])))
        t = pd.DataFrame(rows)
        run.log("correlogram zero crossing (km) by target and crop:\n"
                + t.pivot_table(index=["year", "target"], columns="crop", values="zero_crossing_km").round(1).to_string())
        return t

    def interpolation(self, dev, grid, cells, z, hot, run) -> pd.DataFrame:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        split = SpatialSplit(self.dev_cfg["split"], grid, dev.label_domain, seed=int(self.dev_cfg["seed"]))
        xy = coords_km(cells, grid)
        pos = {c: i for i, c in enumerate(cells)}
        rng = np.random.default_rng(int(self.cfg["seed"]))
        rows = []
        for y in dev.years:
            T = z[y].shape[0]
            pred = {"IDW": np.full(z[y].shape, np.nan), "GP": np.full(z[y].shape, np.nan)}
            # kernel of the year: fitted once on the annual mean of a subsample (hyper-parameters only)
            m = np.nanmean(z[y], 0)
            sub = rng.choice(np.flatnonzero(np.isfinite(m)), size=min(600, int(np.isfinite(m).sum())), replace=False)
            gp0 = GaussianProcessRegressor(ConstantKernel(1.0) * RBF(10.0, (0.5, 200.0)) + WhiteKernel(0.05),
                                           normalize_y=True, random_state=0).fit(xy[sub], m[sub])
            kernel = gp0.kernel_
            run.log(f"  {y}: GP kernel {kernel}")
            for k in range(split.n_folds):
                tr, te = split.masks(k)
                itr = np.array([pos[c] for c in np.flatnonzero(tr.ravel()) if c in pos])
                ite = np.array([pos[c] for c in np.flatnonzero(te.ravel()) if c in pos])
                if len(itr) < 10 or len(ite) == 0:
                    continue
                for t in range(T):
                    ok = itr[np.isfinite(z[y][t, itr])]
                    if len(ok) < 10:
                        continue
                    pred["IDW"][t, ite] = idw(xy[ok], z[y][t, ok], xy[ite])
                    gp = GaussianProcessRegressor(kernel, optimizer=None, normalize_y=True).fit(xy[ok], z[y][t, ok])
                    pred["GP"][t, ite] = gp.predict(xy[ite])
            others = {"IDW (target only)": pred["IDW"], "GP on coordinates (target only)": pred["GP"]}
            for label, (exp, where) in {"GNN, 3 layers (ensemble)": (DEV, None),
                                        "SLX-OLS, 22.7 km": SLX_MATCHED}.items():
                found = latest(self.dev_cfg, exp, where)
                if found is not None:
                    others[label] = np.load(found[1] / "preds" / f"cv_{y}.npz")["pred"]
            for label, p in others.items():
                rows.append(dict(year=y, model=label, **{m: mean_score(z[y], hot[y], p, m) for m in self.pc["metrics"]}))
        t = pd.DataFrame(rows)
        run.log("interpolation baselines vs models (same folds and buffer):\n" + t.round(3).to_string(index=False))
        return t

    def seeds(self, dev, z, hot, run) -> pd.DataFrame:
        found = latest(self.dev_cfg, DEV)
        if found is None:
            run.log("seed calibration skipped: no finished run of " + DEV)
            return pd.DataFrame()
        split = SpatialSplit(self.dev_cfg["split"], Grid(self.dev_cfg), dev.label_domain, seed=int(self.dev_cfg["seed"]))
        blocks = split.blocks(split.block_km)
        rows = []
        for y in dev.years:
            f = found[1] / "preds" / f"cv_{y}_members.npz"
            if not f.exists():
                run.log(f"seed calibration: no stored seeds for {y}")
                continue
            mem = np.load(f)["pred"]
            for i, j in combinations(range(mem.shape[0]), 2):
                for m in self.pc["metrics"]:
                    s = signflip_pooled(z[y], hot[y], mem[i], mem[j], blocks, m, **self.sf)
                    rows.append(dict(year=y, seed_a=i, seed_b=j, metric=m, delta=s["stat"],
                                     p_greater=s["p_greater"], p_two=s["p_two"]))
        t = pd.DataFrame(rows)
        if len(t):
            run.log("seed-vs-seed calibration (true null): share of p below 0.05 / 0.025:\n"
                    + t.groupby("metric").agg(tests=("p_greater", "size"),
                                              below_005=("p_greater", lambda p: float((p < 0.05).mean())),
                                              below_0025=("p_greater", lambda p: float((p < 0.025).mean())),
                                              two_sided_below_005=("p_two", lambda p: float((p < 0.05).mean())))
                    .round(3).to_string())
        return t

    # ------------------------------------------------------------------ all
    def run(self, run, which=("years", "correlogram", "interpolation", "seeds")) -> dict:
        dev, grid, cells, z, hot = self.load(run)
        xy = coords_km(cells, grid)
        out = {}
        if "years" in which:
            t = self.years(z, run)
            run.save_table("review_years.csv", t)
            out["years"] = t.round(3).to_dict("records")
        if "correlogram" in which:
            t = self.correlogram_checks(z, xy, dev.years, run)
            run.save_table("review_correlogram.csv", t)
            g = t.groupby(["target", "crop"]).zero_crossing_km.mean().round(1)
            out["zero_crossing_km"] = {f"{a} / crop {c}": float(v) for (a, c), v in g.items()}
        if "interpolation" in which:
            t = self.interpolation(dev, grid, cells, z, hot, run)
            run.save_table("review_interpolation.csv", t)
            out["interpolation"] = t.round(3).to_dict("records")
        if "seeds" in which:
            t = self.seeds(dev, z, hot, run)
            run.save_table("review_seed_calibration.csv", t)
            if len(t):
                out["seed_calibration_below_005"] = t.groupby("metric").p_greater.apply(lambda p: float((p < 0.05).mean())).to_dict()
        run.save_json("review_checks.json", out)
        return {k: v for k, v in out.items() if k != "interpolation"}

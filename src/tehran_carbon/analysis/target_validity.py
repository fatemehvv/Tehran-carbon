"""What does the target measure? (python main.py paper target [--exp <experiment> ...]; settings: analysis.paper)

  stations    per month, Spearman across the AQCC stations between the station NO2 (monthly mean of daily values) and
              the TROPOMI target z at the station's cell (and the mean of its 3 x 3 neighbourhood): is the satellite
              pattern the ground pattern? (cf. Kafi et al. 2026, Sci. Total Environ., for Tehran)
  elevation   needs analysis.paper.dem_file (gee/GEE_DEM_L0.js). The north of the city is ~700 m higher than the south;
              a column over high ground holds less air, and Rey-Pommier et al. (2025, ESSD) report artefacts near
              Tochal. Per month: Spearman of z with -elevation and the PR-AUC of -elevation as a hotspot score; for
              the runs of --exp, Spearman of their out-of-fold prediction with z, raw and with elevation partialled out
  daily       the monthly mean of the daily cloud-screened NO2 (cloud fraction < 0.3; config of
              analysis.paper.daily_experiment) as an alternative target: Spearman with z and the overlap of the hotspots
  n_eff       effective number of independent cells of z (correlogram; inference.effective_n) and the number of
              TROPOMI pixels the domain holds (analysis.paper.footprint_km)
Outputs: target_stations.csv, target_elevation.csv, target_daily.csv, target_neff.csv, target_validity.json
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from ..config import ROOT, Config
from ..data.dataset import CellMonthDataset
from ..data.grid import Grid
from ..data.sources.pollution import PollutionSource
from ..evaluation.inference import coords_km, correlogram, effective_n, pair_counts
from ..evaluation.metrics import pr_auc
from ..tracking import RunRegistry
from ..utils.calendar import JalaliYear


def partial_spearman(x, y, c) -> float:
    """Spearman of x and y with c partialled out (Pearson of rank residuals after a linear fit on rank(c))."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    rx, ry, rc = rankdata(x[ok]), rankdata(y[ok]), rankdata(c[ok])
    A = np.c_[np.ones(ok.sum()), rc]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


class TargetValidity:
    def __init__(self, base_cfg: dict, experiments: list[str] | None = None):
        self.cfg = base_cfg
        self.pc = base_cfg["analysis"]["paper"]
        self.experiments = experiments or []

    def stations(self, data, grid, run) -> pd.DataFrame:
        src = PollutionSource(self.cfg, grid)
        c = src.p["columns"]
        st = src.stations()
        code = self.pc["station_pollutant"]
        rows = []
        for y in data.years:
            sm = src.station_months(y, code, st)
            loc = st.loc[sm["station"]]
            r, cc, inside = grid.rowcol(loc[c["lon"]].values, loc[c["lat"]].values)
            sm = sm.assign(row=r, col=cc, inside=inside)[inside]
            z = data.z[y]
            for m, g in sm.groupby("month"):
                zt = z[int(m) - 1]
                own = zt[g.row.values, g.col.values]
                nb = np.array([np.nanmean(zt[max(a - 1, 0):a + 2, max(b - 1, 0):b + 2])
                               if np.isfinite(zt[max(a - 1, 0):a + 2, max(b - 1, 0):b + 2]).any() else np.nan
                               for a, b in zip(g.row.values, g.col.values)])
                for name, sat in [("cell", own), ("3x3", nb)]:
                    ok = np.isfinite(sat)
                    rho, pv = spearmanr(g["mean"].values[ok], sat[ok]) if ok.sum() >= 5 else (np.nan, np.nan)
                    rows.append(dict(year=y, month=int(m), satellite=name, stations=int(ok.sum()), spearman=rho, p=pv))
        t = pd.DataFrame(rows)
        run.log("stations vs TROPOMI (mean Spearman over months):\n"
                + t.groupby(["year", "satellite"]).agg(months=("month", "size"), stations=("stations", "median"),
                                                       spearman=("spearman", "mean"),
                                                       p_below_005=("p", lambda s: int((s < 0.05).sum())))
                .round(3).to_string())
        return t

    def latest_runs(self, experiment: str):
        rows = RunRegistry(self.cfg).find(experiment).sort_values("finished")
        out = {}
        for _, r in rows.iterrows():
            out[r.variant] = (r.run_id, ROOT / r.path)
        return out

    def elevation(self, data, grid, run) -> pd.DataFrame | None:
        f = Config.path(self.cfg, "raw") / self.pc["dem_file"]
        if not f.exists():
            run.log(f"elevation: {f} missing (run gee/GEE_DEM_L0.js) - skipped")
            return None
        import rasterio
        with rasterio.open(f) as r:
            grid.check_raster(r.transform, r.shape)
            dem = r.read(1).astype(float)
        elev = dem.ravel()[data.cells]
        rows = []
        for y in data.years:
            z = data.z[y].reshape(data.z[y].shape[0], -1)[:, data.cells]
            hot = data.hot[y].reshape(z.shape[0], -1)[:, data.cells]
            for t in range(z.shape[0]):
                ok = np.isfinite(z[t])
                if ok.sum() < 3:
                    continue
                rows.append(dict(year=y, month=t + 1, model="elevation only (-elevation)",
                                 spearman=float(spearmanr(-elev[ok], z[t][ok])[0]),
                                 pr_auc=pr_auc(hot[t][ok], -elev[ok]), partial_spearman=np.nan))
            for exp in self.experiments:
                for variant, (rid, folder) in self.latest_runs(exp).items():
                    pf = folder / "preds" / f"cv_{y}.npz"
                    if not pf.exists():
                        continue
                    a = np.load(pf)
                    if not np.array_equal(a["cells"], data.cells):
                        run.log(f"elevation: {rid} has another label domain - skipped")
                        continue
                    pred = a["pred"]
                    for t in range(z.shape[0]):
                        ok = np.isfinite(z[t]) & np.isfinite(pred[t])
                        if ok.sum() < 3:
                            continue
                        rows.append(dict(year=y, month=t + 1, model=f"{exp} [{variant or '-'}]",
                                         spearman=float(spearmanr(pred[t][ok], z[t][ok])[0]),
                                         pr_auc=pr_auc(hot[t][ok], pred[t][ok]),
                                         partial_spearman=partial_spearman(pred[t], z[t], elev)))
        t = pd.DataFrame(rows)
        run.log("elevation (means over months):\n"
                + t.groupby(["year", "model"])[["spearman", "pr_auc", "partial_spearman"]].mean().round(3).to_string())
        return t

    def daily(self, data, grid, run) -> pd.DataFrame | None:
        from ..data.sources.daily import SatelliteDailySource
        dcfg = Config().resolve(self.pc["daily_experiment"])[0]
        sp = dcfg["data"]["inputs"]["satellite_daily"]
        floor = float(self.cfg["data"]["target"]["log_floor"])
        q = float(self.cfg["data"]["target"]["hotspot_quantile"])
        rows = []
        for y in data.years:
            try:
                arr = SatelliteDailySource(dcfg, grid).load(y)[sp["layer"]]
            except FileNotFoundError as e:
                run.log(f"daily: {e} - skipped")
                return None
            month = JalaliYear(y).month_of(JalaliYear(y).dates).values
            z = data.z[y]
            for m in range(1, 13):
                sub = arr[month == m]
                n = np.isfinite(sub).sum(0)
                total = np.where(np.isfinite(sub), sub, 0.0).sum(0)
                mean = np.where(n >= int(self.pc["daily_min_days"]), total / np.maximum(n, 1), np.nan)
                a = mean.ravel()[data.cells]
                b = z[m - 1].ravel()[data.cells]
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() < 10:
                    continue
                v = np.log(np.clip(a[ok], floor, None))
                zd = (v - v.mean()) / v.std()
                hd, hm = zd >= np.quantile(zd, q), b[ok] >= np.quantile(b[ok], q)
                rows.append(dict(year=y, month=m, cells=int(ok.sum()), median_days=float(np.median(n.ravel()[data.cells])),
                                 spearman=float(spearmanr(zd, b[ok])[0]),
                                 hotspot_jaccard=float((hd & hm).sum() / max((hd | hm).sum(), 1))))
        t = pd.DataFrame(rows)
        run.log("daily cloud-screened monthly target vs the target:\n"
                + t.groupby("year")[["cells", "median_days", "spearman", "hotspot_jaccard"]].mean().round(3).to_string())
        return t

    def neff(self, data, grid, run) -> pd.DataFrame:
        c = self.pc["correlogram"]
        edges = np.arange(0.0, float(c["max_km"]) + 1e-9, float(c["bin_km"]))
        xy = coords_km(data.cells, grid)
        counts = pair_counts(xy, edges)
        n = len(data.cells)
        fx, fy = (float(v) for v in self.pc["footprint_km"])
        rows = []
        for y in data.years:
            v = data.z[y].reshape(data.z[y].shape[0], -1)[:, data.cells]
            rho, se = correlogram(v, xy, edges, seed=int(self.cfg["seed"]), return_se=True)
            rows.append(dict(year=y, cells=n, n_eff_correlogram=effective_n(rho, se, counts=counts, n=n),
                             n_tropomi_pixels=n * grid.km_x * grid.km_y / (fx * fy),
                             **{f"rho_{a:g}_{b:g}km": r for a, b, r in zip(edges[:-1], edges[1:], rho)}))
        t = pd.DataFrame(rows)
        run.log("effective sample size:\n" + t[["year", "cells", "n_eff_correlogram", "n_tropomi_pixels"]].round(1).to_string(index=False))
        return t

    def run(self, run) -> dict:
        data = CellMonthDataset.build(self.cfg, log=run.log)
        grid = Grid(self.cfg)
        out, tables = {}, {}
        for name, fn in [("stations", self.stations), ("elevation", self.elevation), ("daily", self.daily),
                         ("neff", self.neff)]:
            t = fn(data, grid, run)
            if t is not None:
                run.save_table(f"target_{name}.csv", t)
                out[name], tables[name] = len(t), t
        summary = {f"n_eff_{int(r.year)}": round(float(r.n_eff_correlogram), 1) for r in tables["neff"].itertuples()}
        run.save_json("target_validity.json", dict(rows=out, **summary))
        return summary

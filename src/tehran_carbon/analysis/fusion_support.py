"""Checks of the composite targets before any stage 2 model (python main.py fusion support --exp fus1_all_v1).

Rules: docs/paper/fusion_paper/preregistration_fusion_v1.md, section "Checks before any model is trained".
  M1  pipeline       Spearman(T-all, T0) over the domain on the placed months, mean over months, >= 0.95 in both years
  M2  manipulation   T-calm sharper than T-windy in both years: correlogram zero crossing >= 2 km shorter AND
                     rho(8-10 km) >= 0.05 lower
  M3  reliability    split-half Spearman (odd / even days, in date order, of each stratum) >= 0.90, mean over periods,
                     both years, for T-calm and T-windy
No model is trained. Outputs (run folder): fusion_support.csv (one row per year, stratum and measure),
fusion_support_correlogram.csv, and the verdicts in the run summary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..config import Config
from ..data.dataset import CellMonthDataset
from ..data.grid import Grid
from ..data.sources.composite import STRATA, SatelliteCompositeSource, composite
from ..evaluation.inference import coords_km, correlogram


def zmap(c: np.ndarray, domain: np.ndarray, floor: float) -> np.ndarray:
    """(H, W) composite -> z over its valid domain cells (log, standardised), NaN elsewhere."""
    ok = domain & np.isfinite(c)
    out = np.full(c.shape, np.nan)
    if ok.sum() < 3:
        return out
    v = np.log(np.clip(c[ok], floor, None))
    out[ok] = (v - v.mean()) / v.std()
    return out


def zero_crossing(rho: np.ndarray, centers: np.ndarray) -> float:
    """Distance where the correlogram first falls to 0 (linear interpolation); NaN if it never does."""
    below = np.flatnonzero(np.nan_to_num(rho, nan=1.0) <= 0)
    if not len(below):
        return float("nan")
    k = int(below[0])
    if k == 0:
        return float(centers[0])
    a, b = rho[k - 1], rho[k]
    return float(centers[k - 1] + a / (a - b) * (centers[k] - centers[k - 1]))


def rho_at(rho: np.ndarray, edges: np.ndarray, lo: float) -> float:
    i = int(np.argmin(np.abs(edges[:-1] - lo)))
    return float(rho[i])


class FusionSupport:
    def __init__(self, cfg: dict, experiments: list[str]):
        self.cfg = cfg
        if not experiments:
            raise SystemExit("fusion support needs --exp <an experiment with data.inputs.satellite_composite>")
        self.ccfg = Config().resolve(experiments[0])[0]
        if "satellite_composite" not in self.ccfg["data"]["inputs"]:
            raise SystemExit(f"{experiments[0]} has no data.inputs.satellite_composite")
        self.base = Config().resolve("paper2_rq1_ablation_v1")[0]       # T0: the monthly target of the thesis

    def run(self, run) -> dict:
        data = CellMonthDataset.build(self.base, log=run.log)
        grid = Grid(self.base)
        src = SatelliteCompositeSource(self.ccfg, grid)
        p = src.p
        floor = float(self.base["data"]["target"]["log_floor"])
        dom = data.label_domain
        cells = data.cells
        c = self.cfg["analysis"]["paper"]["correlogram"]
        edges = np.arange(0.0, float(c["max_km"]) + 1e-9, float(c["bin_km"]))
        centers = (edges[:-1] + edges[1:]) / 2
        xy = coords_km(cells, grid)
        rows, crow, verdict = [], [], {}
        for y in data.years:
            daily, sel = src.selection(y)
            z0 = data.z[y]
            maps = {s: [] for s in STRATA}
            for per in sel:
                n_days = {s: int(len(per[s])) for s in STRATA}
                rows += [dict(year=y, stratum=s, measure="days", month=per["place"], value=n_days[s]) for s in STRATA]
                for s in STRATA:
                    zm = zmap(composite(daily, per[s], int(p.get("min_days", 3))), dom, floor)
                    maps[s].append(zm)
                    if s == "all":                                       # M1
                        a, b = zm.ravel()[cells], z0[per["place"] - 1].ravel()[cells]
                        ok = np.isfinite(a) & np.isfinite(b)
                        rows.append(dict(year=y, stratum=s, measure="spearman_vs_T0", month=per["place"],
                                         value=float(spearmanr(a[ok], b[ok])[0]) if ok.sum() > 10 else np.nan))
                    else:                                                # M3: odd / even days in date order
                        d = np.sort(per[s])
                        h1 = zmap(composite(daily, d[0::2], 1), dom, floor).ravel()[cells]
                        h2 = zmap(composite(daily, d[1::2], 1), dom, floor).ravel()[cells]
                        ok = np.isfinite(h1) & np.isfinite(h2)
                        rows.append(dict(year=y, stratum=s, measure="split_half", month=per["place"],
                                         value=float(spearmanr(h1[ok], h2[ok])[0]) if ok.sum() > 10 else np.nan))
            for s in STRATA:                                             # M2 and the description of every version
                v = np.stack([m.ravel()[cells] for m in maps[s]])
                rho = correlogram(v, xy, edges, seed=int(self.base["seed"]))
                crow += [dict(year=y, stratum=s, km=float(k), rho=float(r)) for k, r in zip(centers, rho)]
                rows.append(dict(year=y, stratum=s, measure="zero_crossing_km", month=0, value=zero_crossing(rho, centers)))
                rows.append(dict(year=y, stratum=s, measure="rho_8_10km", month=0, value=rho_at(rho, edges, 8.0)))
            v0 = np.stack([z0[per["place"] - 1].ravel()[cells] for per in sel])
            rho0 = correlogram(v0, xy, edges, seed=int(self.base["seed"]))
            crow += [dict(year=y, stratum="T0", km=float(k), rho=float(r)) for k, r in zip(centers, rho0)]
            rows.append(dict(year=y, stratum="T0", measure="zero_crossing_km", month=0, value=zero_crossing(rho0, centers)))
            rows.append(dict(year=y, stratum="T0", measure="rho_8_10km", month=0, value=rho_at(rho0, edges, 8.0)))
        t = pd.DataFrame(rows)
        t.to_csv(run.path("fusion_support.csv"), index=False)
        pd.DataFrame(crow).to_csv(run.path("fusion_support_correlogram.csv"), index=False)

        def val(y, s, m):
            x = t[(t.year == y) & (t.stratum == s) & (t.measure == m)].value
            return float(np.nanmean(x)) if len(x) else float("nan")

        m1 = {y: val(y, "all", "spearman_vs_T0") for y in data.years}
        m2 = {y: dict(zc_calm=val(y, "calm", "zero_crossing_km"), zc_windy=val(y, "windy", "zero_crossing_km"),
                      r_calm=val(y, "calm", "rho_8_10km"), r_windy=val(y, "windy", "rho_8_10km")) for y in data.years}
        m3 = {y: dict(calm=val(y, "calm", "split_half"), windy=val(y, "windy", "split_half")) for y in data.years}
        verdict = dict(
            M1=dict(values=m1, passed=bool(all(v >= 0.95 for v in m1.values()))),
            M2=dict(values=m2, passed=bool(all(np.isfinite([d["zc_calm"], d["zc_windy"]]).all()
                                              and d["zc_calm"] <= d["zc_windy"] - 2.0
                                              and d["r_calm"] <= d["r_windy"] - 0.05 for d in m2.values()))),
            M3=dict(values=m3, passed=bool(all(d["calm"] >= 0.90 and d["windy"] >= 0.90 for d in m3.values()))),
            period=p.get("period", "season"))
        run.save_json("fusion_support.json", verdict)
        run.log("days per stratum and placed month:\n" + t[t.measure == "days"].pivot_table(
            index=["year", "month"], columns="stratum", values="value").to_string())
        for k in ("M1", "M2", "M3"):
            run.log(f"{k}: {'PASSED' if verdict[k]['passed'] else 'FAILED'}  {verdict[k]['values']}")
        return {k: verdict[k]["passed"] for k in ("M1", "M2", "M3")}

"""Amendment 1 of the fusion paper: the conditional placebo (python main.py fusion amend). Nothing is trained.

Registered in docs/paper/fusion_paper/preregistration_fusion_v1.md ("Amendment 1") before the runs of
fus3_cond_placebo_v1, fus3_ref_v1 and fus3_ctx_v1 (configs/config.yaml; data/sources/placebo.py, method conditional).

Check M5 (before the verdict of G1 is read). On the label domain, annual maps (log1p as in the dataset), every energy
layer that varies on the domain, for every realisation and both development years:
  M5a  dependence on the context kept:  | max_c |rho(placebo, c)| - max_c |rho(real, c)| |  <= 0.10
  M5b  values kept:                     Kolmogorov-Smirnov distance(placebo, real)            <= 0.10
  M5c  residual relocated:              |rho(placebo - f, real - f)|                           <  0.10
(c = the five context layers; f = the cross-fitted context prediction of the layer, as in the placebo.) A realisation
passes if the medians over its layers and years meet all three; the check fails if more than two of five realisations
fail, and G1 is then labelled "conditional placebo check failed". Also reported: share of zero cells and Spearman
with the target of real and placebo layers (descriptive).

Family G (1402, 1403; Holm over the 2 tests, alpha 0.05), primary metric Spearman, margin 0.02, the inference of the
other families (jackknife verdict, block sign-flip, seed s against realisation s):
  G1      context + energy (gated, fus3_ref_v1)  vs  context + conditional placebo energy (gated), mean of 5 realisations
Secondary estimates (no verdict):
  K1      conditional placebo energy - context only            (cost of an added input that agrees with the context)
  K2      real energy - context only, same machine            (the ablation)
  K3      conditional placebo - rotation-and-shift placebo     (fus2_placebo_v2; other machine; skipped if not found)
Outputs (run folder): fusion_amend_m5.csv, fusion_amend.csv, fusion_amend_holm.csv, fusion_amend.json.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr

from ..config import Config
from ..data.dataset import CellMonthDataset
from ..data.grid import Grid
from ..data.sources.base import SOURCES
from ..data.sources.placebo import PlaceboSource, context_design, cross_fit, tile_folds
from ..evaluation.inference import holm
from .fusion_eval import ALPHA, E, E_PLC, FusionEvaluation, plc, sel

COND, REF, CTX3 = "fus3_cond_placebo_v1", "fus3_ref_v1", "fus3_ctx_v1"
CONTEXT = ["pop", "vol_total", "vol_nonres", "viirs", "industrial_frac"]
YEARS = [1402, 1403]
PAIRS_G = [
    ("G1", "G", YEARS, sel(REF, E, "gated"), plc(COND, E_PLC)),
    ("K1", "secondary", YEARS, plc(COND, E_PLC), sel(CTX3)),
    ("K2", "secondary", YEARS, sel(REF, E, "gated"), sel(CTX3)),
    ("K3", "secondary", YEARS, plc(COND, E_PLC), plc("fus2_placebo_v2", E_PLC)),
]


def rho(a, b) -> float:
    return float(spearmanr(a, b)[0]) if np.ptp(a) > 0 and np.ptp(b) > 0 else np.nan


class FusionAmendment:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.ev = FusionEvaluation(cfg)

    # ------------------------------------------------------------------ M5
    def m5(self, run) -> dict:
        cfgs = {}
        for c in Config().resolve(COND):
            cfgs.setdefault(int(c["data"]["inputs"]["placebo"]["realization"]), c)
        c0 = cfgs[0]
        p = c0["data"]["inputs"]["placebo"]
        grid = Grid(c0)
        log1p = set(c0["data"].get("log1p") or [])
        data = CellMonthDataset.build(Config().resolve(REF)[0], log=run.log)
        cells = data.cells
        folds = tile_folds(grid, float(c0["split"]["block_km"]), int(p.get("cross_fit_folds", 5)), int(c0["seed"]))
        layers = p["domains"]["energy"]
        rows = []
        for y in YEARS:
            zt = np.nanmean(data.z[y], 0).ravel()[cells]
            ctx = [data.X[y][..., data.features.index(n)].mean(0).ravel()[cells] for n in CONTEXT]
            real = SOURCES.get("energy")(c0, grid).load(y)
            X = context_design(c0, grid, y)
            for r, c in sorted(cfgs.items()):
                run.log(f"M5: {y}, realisation {r}")
                out = PlaceboSource(c, grid).load(y)
                for n in layers:
                    a, s = np.asarray(real[n], float), np.asarray(out["plc_" + n], float)
                    a, s = (a.mean(0) if a.ndim == 3 else a), (s.mean(0) if s.ndim == 3 else s)
                    if n in log1p:
                        a, s = np.log1p(np.clip(a, 0, None)), np.log1p(np.clip(s, 0, None))
                    if np.ptp(a.ravel()[cells]) == 0:
                        continue
                    f = cross_fit(X, a.ravel(), folds)[cells]
                    ad, sd = a.ravel()[cells], s.ravel()[cells]
                    rows.append(dict(
                        year=y, realisation=r, layer=n,
                        dependence_gap=abs(max(abs(rho(sd, v)) for v in ctx) - max(abs(rho(ad, v)) for v in ctx)),
                        ks=float(ks_2samp(ad, sd).statistic),
                        residual_abs_spearman=abs(rho(sd - f, ad - f)),
                        zero_share_real=float((ad == ad.min()).mean()), zero_share_placebo=float((sd == ad.min()).mean()),
                        spearman_target_real=rho(ad, zt), spearman_target_placebo=rho(sd, zt)))
        t = pd.DataFrame(rows)
        per = t.groupby("realisation").median(numeric_only=True).drop(columns="year").reset_index()
        per["passed"] = (per.dependence_gap <= 0.10) & (per.ks <= 0.10) & (per.residual_abs_spearman < 0.10)
        failed = int((~per.passed).sum())
        verdict = "passed" if failed <= 2 else "conditional placebo check failed"
        run.save_table("fusion_amend_m5.csv", t)
        run.log("M5 per realisation (medians over layers and years):\n" + per.round(3).to_string(index=False)
                + f"\nM5: {verdict} ({failed} of {len(per)} failed)")
        return dict(verdict=verdict, failed=failed, per_realisation=per.to_dict("records"))

    # ------------------------------------------------------------------ family G
    def tests(self, run, m5_verdict: str) -> pd.DataFrame:
        rows = []
        for name, fam, years, cs, rs in PAIRS_G:
            run.log(f"{name} ...")
            try:
                rows += self.ev.compare(name, fam, years, cs, rs, run)
            except ValueError as e:                        # K3: the placebo runs of the server may be absent
                if fam == "G":
                    raise
                run.log(f"{name}: skipped ({e})")
        t = pd.DataFrame(rows)
        t["in_holm"] = (t.family == "G") & (t.part == "primary")
        sub = t[t.in_holm]
        keys = [f"{r.test}|{r.year}" for r in sub.itertuples()]
        adj = holm(dict(zip(keys, sub.p_signflip)))
        hl = []
        for r, k in zip(sub.itertuples(), keys):
            v = r.verdict if not str(r.verdict).startswith("superior") or adj[k] < ALPHA \
                else "superior, not significant after Holm"
            if m5_verdict != "passed":
                v = f"{v} (conditional placebo check failed)"
            hl.append(dict(family="G", test=r.test, year=r.year, metric=r.metric, delta=r.delta,
                           p_signflip=r.p_signflip, p_holm=adj[k], verdict=r.verdict, verdict_holm=v))
        holm_t = pd.DataFrame(hl)
        run.save_table("fusion_amend.csv", t)
        run.save_table("fusion_amend_holm.csv", holm_t)
        cols = ["test", "year", "metric", "candidate", "reference", "delta", "jack_low95", "jack_high95", "verdict"]
        run.log("family G and secondary estimates:\n" + t[cols].round(4).to_string(index=False))
        run.log("Holm (family G):\n" + holm_t.round(4).to_string(index=False))
        return holm_t

    def run(self, run) -> dict:
        m5 = self.m5(run)
        holm_t = self.tests(run, m5["verdict"])
        run.save_json("fusion_amend.json", dict(M5=m5, holm=holm_t.to_dict("records")))
        return {"M5": m5["verdict"], "G": holm_t.verdict_holm.tolist()}

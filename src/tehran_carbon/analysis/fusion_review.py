"""Analyses asked for by the reviewers of the fusion paper (python main.py fusion review). Nothing is trained.

Reviews: docs/paper/fusion_paper/review/. All three read stored predictions and the dataset; they run on a laptop.

  null       placebo against placebo. For every placebo experiment (constructions A = _v1, B = _v2), set of branches
             and year: the score of each of the 5 realisations, the spread among them (the null of "no information
             difference", which includes the realisation variance), and where the real-source model ranks among them
             (rank 1 of 6 = above every realisation)
  domain     placebo check M4 on the label domain (the 2,436 cells the model is trained and tested on): share of
             zero cells, Kolmogorov-Smirnov distance and |Spearman| between placebo and real layer, the placebo's
             largest |Spearman| with the target and with any context layer, and the largest correlogram difference
             up to 20 km, per layer and realisation (annual maps, log1p as in the dataset)
  absolute   absolute Spearman and PR-AUC of every model (real, placebo, early, gated, context only, 1404) and of
             trivial spatial baselines: each context layer and its box means alone, and minus the distance to the
             population-weighted centre of the city
Runs are found in runs/ and in docs/paper/fusion_paper/results_*/runs/ (the folders pushed from the server), from their
meta.json and config.yaml. Outputs (run folder): fusion_review_null.csv, fusion_review_domain.csv,
fusion_review_absolute.csv, fusion_review.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import ks_2samp, spearmanr

from ..config import ROOT, Config
from ..data.dataset import CellMonthDataset
from ..data.grid import Grid
from ..data.sources.base import SOURCES
from ..data.sources.placebo import PlaceboSource
from ..evaluation.inference import coords_km, correlogram
from ..evaluation.robustness import mean_score

R = [0, 1, 2, 3, 4]
E, TEP = ["energy"], ["traffic", "energy", "pollution"]
CONTEXT = ["pop", "vol_total", "vol_nonres", "viirs", "industrial_frac"]


class RunIndex:
    """Every finished, non-ad-hoc run folder in runs/ and in the pushed result folders, with its config."""

    def __init__(self):
        roots = [ROOT / "runs"] + sorted((ROOT / "docs" / "paper" / "fusion_paper").glob("results_*/runs"))
        self.runs = []
        seen = set()
        for root in roots:
            if not root.exists():
                continue
            for f in sorted(root.glob("*/meta.json")):
                m = json.loads(f.read_text(encoding="utf-8"))
                if m.get("status") != "done" or m.get("overrides") or f.parent.name in seen:
                    continue
                c = f.parent / "config.yaml"
                if not c.exists():
                    continue
                seen.add(f.parent.name)
                self.runs.append((m.get("experiment"), f.parent, yaml.safe_load(c.read_text(encoding="utf-8"))))

    def find(self, experiment: str, **where) -> list[tuple[Path, dict]]:
        out = []
        for e, folder, cfg in self.runs:
            if e != experiment:
                continue
            if all(Config.get(cfg, k, None) == v for k, v in where.items()):
                out.append((folder, cfg))
        return out

    def one(self, experiment: str, **where) -> tuple[Path, dict]:
        f = self.find(experiment, **where)
        if not f:
            raise FileNotFoundError(f"no finished run of {experiment} {where} in runs/ or docs/paper/fusion_paper/results_*/")
        return sorted(f, key=lambda x: x[0].name)[-1]


def preds(folder: Path, y: int, cells) -> np.ndarray:
    a = np.load(folder / "preds" / f"cv_{y}.npz")
    if not np.array_equal(a["cells"], cells):
        raise ValueError(f"{folder.name}: other label domain")
    return a["pred"]


class FusionReview:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.idx = RunIndex()
        self._data = {}

    def data(self, cfg):
        h = Config.data_hash(cfg)
        if h not in self._data:
            self._data[h] = CellMonthDataset.build(cfg)
        return self._data[h]

    def truth(self, cfg, y):
        d = self.data(cfg)
        T = d.z[y].shape[0]
        return d, d.z[y].reshape(T, -1)[:, d.cells], d.hot[y].reshape(T, -1)[:, d.cells]

    # ------------------------------------------------------------------ null
    def null(self, run) -> pd.DataFrame:
        sets = [("energy, gated", "fus2_placebo_{v}", ["energy_plc"], "gated", "paper2_rq1_ablation_v1", E, [1402, 1403]),
                ("energy, early", "fus2_placebo_early_{v}", ["energy_plc"], "early", "paper2_rq1_ablation_v1", E, [1402, 1403]),
                ("traffic + stations, gated", "fus2_placebo_{v}", ["energy", "traffic_plc", "pollution_plc"], "gated",
                 "paper2_rq1_ablation_v1", TEP, [1402, 1403]),
                ("energy, gated", "fus2_c1404_placebo_{v}", ["energy_plc"], "gated", "fus2_c1404_real_v1", E, [1404]),
                ("traffic + stations, gated", "fus2_c1404_placebo_{v}", ["energy", "traffic_plc", "pollution_plc"],
                 "gated", "fus2_c1404_real_v1", TEP, [1404])]
        rows = []
        for ver, label in (("v1", "A"), ("v2", "B")):
            for name, pexp, pm, fus, rexp, rm, years in sets:
                try:
                    pl = [self.idx.one(pexp.format(v=ver), **{"data.inputs.placebo.realization": r,
                                                            "model.modalities": pm, "model.fusion.type": fus}) for r in R]
                    real = self.idx.one(rexp, **{"model.modalities": rm, "model.fusion.type": fus})
                except FileNotFoundError as e:
                    run.log(f"null: skipped ({e})")
                    continue
                for y in years:
                    d, z, hot = self.truth(real[1], y)
                    for metric in ("spearman", "pr_auc"):
                        s = np.array([mean_score(z, hot, preds(f, y, d.cells), metric) for f, _ in pl])
                        sr = mean_score(z, hot, preds(real[0], y, d.cells), metric)
                        pair = np.array([s[i] - s[j] for i in range(5) for j in range(5) if i < j])
                        rows.append(dict(placebo=label, comparison=name, year=y, metric=metric, real=sr,
                                         placebo_mean=s.mean(), placebo_min=s.min(), placebo_max=s.max(),
                                         placebo_sd=s.std(ddof=1), null_pair_abs_max=np.abs(pair).max(),
                                         delta=sr - s.mean(), delta_over_sd=(sr - s.mean()) / s.std(ddof=1),
                                         real_rank_of_6=int(1 + (s > sr).sum())))
        t = pd.DataFrame(rows)
        if not t.empty:
            run.log("placebo vs placebo (null) and the real model's rank among the realisations:\n"
                    + t.round(3).to_string(index=False))
        return t

    # ------------------------------------------------------------------ M4 on the label domain
    def domain(self, run) -> pd.DataFrame:
        base = Config().resolve("paper2_rq1_ablation_v1")[0]
        d = self.data(base)
        y = 1402
        dom = d.label_domain
        cells = d.cells
        zt = np.nanmean(d.z[y], 0).ravel()[cells]                           # annual mean of the monthly target
        ctx = {}
        for n in CONTEXT:
            j = d.features.index(n)
            ctx[n] = d.X[y][..., j].mean(0).ravel()[cells]
        xy = coords_km(cells, Grid(base))
        edges = np.arange(0.0, 20.0 + 1e-9, 2.0)
        rows = []
        for ver, label in (("v1", "A"), ("v2", "B")):
            cfgs = {}
            for c in Config().resolve(f"fus2_placebo_{ver}"):
                cfgs.setdefault(int(c["data"]["inputs"]["placebo"]["realization"]), c)
            c0 = cfgs[0]
            p = c0["data"]["inputs"]["placebo"]
            log1p = set(c0["data"].get("log1p") or [])
            grid = Grid(c0)
            real = {dm: SOURCES.get(dm)(c0, grid).load(y) for dm in p["domains"]}
            for r, c in sorted(cfgs.items()):
                out = PlaceboSource(c, grid).load(y)
                for dm, layers in p["domains"].items():
                    for n in layers:
                        a = np.asarray(real[dm][n], float)
                        s = np.asarray(out["plc_" + n], float)
                        a, s = (a.mean(0) if a.ndim == 3 else a), (s.mean(0) if s.ndim == 3 else s)
                        if n in log1p:
                            a, s = np.log1p(a), np.log1p(s)
                        a, s = a.ravel()[cells], s.ravel()[cells]
                        if np.ptp(a) == 0:
                            continue
                        ca = correlogram(a[None], xy, edges, seed=0)
                        cs = correlogram(s[None], xy, edges, seed=0) if np.ptp(s) > 0 else np.full_like(ca, np.nan)
                        rows.append(dict(placebo=label, realisation=r, domain=dm, layer=n,
                                         zero_share_real=float((a == a.min()).mean()),
                                         zero_share_placebo=float((s == a.min()).mean()),
                                         ks=float(ks_2samp(a, s).statistic),
                                         abs_spearman_real=abs(float(spearmanr(a, s)[0])) if np.ptp(s) > 0 else np.nan,
                                         spearman_target_real=float(spearmanr(a, zt)[0]),
                                         spearman_target_placebo=float(spearmanr(s, zt)[0]) if np.ptp(s) > 0 else np.nan,
                                         max_abs_spearman_context_real=max(abs(float(spearmanr(a, v)[0])) for v in ctx.values()),
                                         max_abs_spearman_context_placebo=max(abs(float(spearmanr(s, v)[0])) for v in ctx.values())
                                         if np.ptp(s) > 0 else np.nan,
                                         max_correlogram_diff=float(np.nanmax(np.abs(ca - cs)))))
        t = pd.DataFrame(rows)
        summ = t.groupby(["placebo", "realisation"]).median(numeric_only=True).reset_index()
        run.log("placebo on the label domain (medians over the layers):\n" + summ.round(3).to_string(index=False))
        return t

    # ------------------------------------------------------------------ absolute skill
    def absolute(self, run) -> pd.DataFrame:
        items = [("context only", "paper2_rq1_context_only_v1", {}, [1402, 1403]),
                 ("energy, early", "paper2_rq1_ablation_v1", {"model.modalities": E, "model.fusion.type": "early"}, [1402, 1403]),
                 ("energy, gated", "paper2_rq1_ablation_v1", {"model.modalities": E, "model.fusion.type": "gated"}, [1402, 1403]),
                 ("all three, gated", "paper2_rq1_ablation_v1", {"model.modalities": TEP, "model.fusion.type": "gated"}, [1402, 1403]),
                 ("context only", "fus2_c1404_context_v1", {}, [1404]),
                 ("energy, gated", "fus2_c1404_real_v1", {"model.modalities": E, "model.fusion.type": "gated"}, [1404]),
                 ("all three, gated", "fus2_c1404_real_v1", {"model.modalities": TEP, "model.fusion.type": "gated"}, [1404])]
        for ver, lab in (("v1", "A"), ("v2", "B")):
            items += [(f"placebo energy {lab}, gated", f"fus2_placebo_{ver}", {"model.modalities": ["energy_plc"]}, [1402, 1403]),
                      (f"placebo energy {lab}, early", f"fus2_placebo_early_{ver}", {"model.modalities": ["energy_plc"]}, [1402, 1403]),
                      (f"energy + placebo traffic/stations {lab}, gated", f"fus2_placebo_{ver}",
                       {"model.modalities": ["energy", "traffic_plc", "pollution_plc"]}, [1402, 1403]),
                      (f"placebo energy {lab}, gated", f"fus2_c1404_placebo_{ver}", {"model.modalities": ["energy_plc"]}, [1404]),
                      (f"energy + placebo traffic/stations {lab}, gated", f"fus2_c1404_placebo_{ver}",
                       {"model.modalities": ["energy", "traffic_plc", "pollution_plc"]}, [1404])]
        rows = []
        for label, exp, where, years in items:
            found = self.idx.find(exp, **where)
            if not found:
                run.log(f"absolute: no runs for {label} ({exp})")
                continue
            for y in years:
                d, z, hot = self.truth(found[0][1], y)
                for metric in ("spearman", "pr_auc"):
                    s = [mean_score(z, hot, preds(f, y, d.cells), metric) for f, _ in found]
                    rows.append(dict(model=label, year=y, metric=metric, value=float(np.mean(s)),
                                     min=float(np.min(s)), max=float(np.max(s)), runs=len(s)))
        # trivial spatial baselines: one context feature (or box mean) alone; minus the distance to the centre
        base = Config().resolve("paper2_rq1_ablation_v1")[0]
        d = self.data(base)
        grid = Grid(base)
        xy = coords_km(np.arange(grid.n), grid)
        for y in d.years:
            T = d.z[y].shape[0]
            z, hot = d.z[y].reshape(T, -1)[:, d.cells], d.hot[y].reshape(T, -1)[:, d.cells]
            cands = {}
            for j, f in enumerate(d.features):
                if d.base_of[j] in CONTEXT:
                    cands[f] = d.X[y][..., j].reshape(T, -1)[:, d.cells]
            pop = d.X[y][..., d.features.index("pop")].mean(0).ravel()
            w = np.expm1(pop) if "pop" in set(base["data"].get("log1p") or []) else pop
            centre = (xy * w[:, None]).sum(0) / w.sum()
            dist = np.hypot(*(xy[d.cells] - centre).T)
            cands["minus distance to the population-weighted centre"] = np.broadcast_to(-dist, (T, len(d.cells)))
            sc = {f: mean_score(z, hot, v, "spearman") for f, v in cands.items()}
            best = max(sc, key=sc.get)
            for f in sorted(set([best, "minus distance to the population-weighted centre"])):
                for metric in ("spearman", "pr_auc"):
                    rows.append(dict(model=f"trivial: {f}", year=y, metric=metric,
                                     value=mean_score(z, hot, np.asarray(cands[f], float), metric), min=np.nan,
                                     max=np.nan, runs=0))
        t = pd.DataFrame(rows)
        run.log("absolute skill:\n" + t.pivot_table(index="model", columns=["metric", "year"], values="value")
                .round(3).to_string())
        return t

    def run(self, run) -> dict:
        out = {}
        for name, fn in (("null", self.null), ("domain", self.domain), ("absolute", self.absolute)):
            run.log(f"--- {name}")
            t = fn(run)
            run.save_table(f"fusion_review_{name}.csv", t)
            out[name] = len(t)
        run.save_json("fusion_review.json", out)
        return out

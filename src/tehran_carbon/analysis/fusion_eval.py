"""Confirmatory tests of the fusion paper (python main.py fusion evaluate).

Every comparison is the one written in docs/paper/fusion_paper/preregistration_fusion_v1.md; candidate and reference
runs are fixed here, not chosen after the results. Primary metric Spearman (margin 0.02); PR-AUC rows are descriptive.

  family F (1402, 1403; Holm)
    F1a/F1b  context + energy (gated)            vs  context + placebo energy (gated), mean over 5 realisations
    F2a/F2b  context + energy + traffic + stations (gated)  vs  context + energy + placebo traffic + placebo stations
    F3       D = delta(calm) - delta(windy), delta = Spearman(all three, gated) - Spearman(energy, gated), pooled
             over the two years; leaves the family if check M2 failed (python main.py fusion support)
    F4       delta_rel(T0) - delta_rel(T-8 km), delta_rel = delta / (1 - Spearman(energy, gated)), pooled
  family C (1404; Holm)
    C1, C2   F1 and F2 on the held-out year (graph-free network)
  secondary (estimates with jackknife intervals, no verdict)
    architecture effect: placebo energy (gated / early) - context only; the fusion gain delta on every target version

Placebo references: every score is averaged over the 5 realisations, also inside the jackknife and inside the sign-flip
test (the same block flips for every realisation). Seeds: candidate seed s against placebo realisation s.
Outputs (run folder): fusion_eval.csv, fusion_holm.csv, fusion_eval.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..data.dataset import CellMonthDataset
from ..data.grid import Grid
from ..data.splits import SpatialSplit
from ..evaluation.hypotheses import HypothesisTest
from ..evaluation.inference import (assignments, holm, rank_normalize, signflip_mean, t_interval, verdict,
                                    RANK_METRICS)
from ..evaluation.metrics import monthly_scores
from ..evaluation.robustness import mean_score
from ..tracking import RunRegistry
from ..config import ROOT

REAL = "paper2_rq1_ablation_v1"
CTX = "paper2_rq1_context_only_v1"
E, TEP = ["energy"], ["traffic", "energy", "pollution"]
E_PLC, TEP_PLC = ["energy_plc"], ["energy", "traffic_plc", "pollution_plc"]
R = [0, 1, 2, 3, 4]
ALPHA = 0.05                                   # family-wise level of each family (pre-registration)


def sel(exp, mods=None, fusion=None, **where):
    w = dict(where)
    if mods is not None:
        w["model.modalities"] = mods
    if fusion is not None:
        w["model.fusion.type"] = fusion
    return {"experiment": exp, "where": w}


def plc(exp, mods, fusion="gated"):
    return [sel(exp, mods, fusion, **{"data.inputs.placebo.realization": r}) for r in R]


# name, family, years, candidate selector(s), reference selector(s) (a list = placebo realisations, averaged)
PAIRS = [
    ("F1", "F", [1402, 1403], sel(REAL, E, "gated"), plc("fus2_placebo_v1", E_PLC)),
    ("F2", "F", [1402, 1403], sel(REAL, TEP, "gated"), plc("fus2_placebo_v1", TEP_PLC)),
    ("C1", "C", [1404], sel("fus2_c1404_real_v1", E, "gated"), plc("fus2_c1404_placebo_v1", E_PLC)),
    ("C2", "C", [1404], sel("fus2_c1404_real_v1", TEP, "gated"), plc("fus2_c1404_placebo_v1", TEP_PLC)),
    # secondary: architecture effect (placebo branch - context only)
    ("A_gated", "secondary", [1402, 1403], plc("fus2_placebo_v1", E_PLC), sel(CTX)),
    ("A_early", "secondary", [1402, 1403], plc("fus2_placebo_early_v1", E_PLC, "early"), sel(CTX)),
    ("A_gated_1404", "secondary", [1404], plc("fus2_c1404_placebo_v1", E_PLC), sel("fus2_c1404_context_v1")),
    ("I_early", "secondary", [1402, 1403], sel(REAL, E, "early"), plc("fus2_placebo_early_v1", E_PLC, "early")),
]
# fusion gain delta = (all three, gated) - (energy, gated) on every target version: (name, years, cand, ref)
GAINS = [
    ("T0", [1402, 1403], sel(REAL, TEP, "gated"), sel(REAL, E, "gated")),
    ("T-all", [1402, 1403], sel("fus1_all_v1", TEP, "gated"), sel("fus1_all_v1", E, "gated")),
    ("T-calm", [1402, 1403], sel("fus1_calm_v1", TEP, "gated"), sel("fus1_calm_v1", E, "gated")),
    ("T-windy", [1402, 1403], sel("fus1_windy_v1", TEP, "gated"), sel("fus1_windy_v1", E, "gated")),
] + [(f"T-{s:g}km", [1402, 1403], sel("fus1_coarse_v1", TEP, "gated", **{"data.target.smooth_km": s}),
      sel("fus1_coarse_v1", E, "gated", **{"data.target.smooth_km": s})) for s in (2.0, 4.0, 8.0, 12.0)] + [
    ("1404-T0", [1404], sel("fus2_c1404_real_v1", TEP, "gated"), sel("fus2_c1404_real_v1", E, "gated")),
    ("1404-T-8km", [1404], sel("fus1_c1404_coarse_v1", TEP, "gated"), sel("fus1_c1404_coarse_v1", E, "gated")),
]


class Arm:
    """The runs of one side of a comparison: one run, or the placebo realisations (scores averaged)."""

    def __init__(self, finder, spec):
        specs = spec if isinstance(spec, list) else [spec]
        self.runs = []
        for s in specs:
            found = finder.select(s)
            if len(found) != 1:
                raise ValueError(f"{s} matches {len(found)} finished runs (need exactly 1)")
            self.runs.append(found[0])
        self.cfg = self.runs[0][2]

    def preds(self, y, cells):
        out = []
        for rid, folder, _ in self.runs:
            a = np.load(folder / "preds" / f"cv_{y}.npz")
            if not np.array_equal(a["cells"], cells):
                raise ValueError(f"{rid}: other label domain")
            out.append(a["pred"])
        return out

    def members(self, y):
        out = []
        for _, folder, _ in self.runs:
            f = folder / "preds" / f"cv_{y}_members.npz"
            out.append(np.load(f)["pred"] if f.exists() else None)
        return out


def score(z, hot, preds, metric, idx=None):
    return float(np.mean([mean_score(z, hot, p, metric, idx) for p in preds]))


def truth_and_blocks(cfg, years, log):
    data = CellMonthDataset.build(cfg, log=log)
    split = SpatialSplit(cfg["split"], Grid(cfg), data.label_domain, seed=int(cfg["seed"]))
    t = {y: (data.z[y].reshape(data.z[y].shape[0], -1)[:, data.cells],
             data.hot[y].reshape(data.z[y].shape[0], -1)[:, data.cells]) for y in years}
    return data, t, split.blocks(split.block_km)


def jack(fun, K):
    """Full estimate, delete-one-block jackknife SE and number of usable blocks; fun(idx or None) -> estimate."""
    d = np.array([fun(b) for b in range(K)])
    ok = np.isfinite(d)
    k = int(ok.sum())
    se = float(np.sqrt((k - 1) / k * ((d[ok] - d[ok].mean()) ** 2).sum())) if k > 1 else np.nan
    return se, k


def pooled_signflip(z, hot, pcs, prs, blocks, metric, sf):
    """Exact block sign-flip of S(candidate) - S(reference). With several runs on a side (placebo realisations), the
    k-th candidate run is paired with the k-th reference run (k modulo the number of runs), the statistic is the mean
    over the pairs, and the same block flip is applied to every pair. Returns (p one-sided greater, p two-sided)."""
    if metric in RANK_METRICS:
        pcs, prs = [rank_normalize(p) for p in pcs], [rank_normalize(p) for p in prs]
    n = max(len(pcs), len(prs))
    pairs = [(pcs[k % len(pcs)], prs[k % len(prs)]) for k in range(n)]
    K = int(blocks.max()) + 1
    flips, exact = assignments(K, sf["max_exact"], sf["n_random"], sf["seed"])
    if exact:                                          # the complement of a flip gives minus the statistic
        flips = flips[~flips[:, -1]]
    st = []
    for f in flips:
        cell = f[blocks][None]
        st.append(np.mean([mean_score(z, hot, np.where(cell, b, a), metric) - mean_score(z, hot, np.where(cell, a, b), metric)
                           for a, b in pairs]))
    st = np.asarray(st)
    if exact:
        st = np.concatenate([st, -st])
    obs = st[0]
    return float(np.mean(st >= obs - 1e-12)), float(np.mean(np.abs(st) >= abs(obs) - 1e-12))


def with_placebo(pairs, version: str):
    """The comparisons with the placebo experiments of another version (fus2_*placebo*_v1 -> _<version>)."""
    def ren(spec):
        if isinstance(spec, list):
            return [ren(x) for x in spec]
        e = spec["experiment"]
        return dict(spec, experiment=e[:-3] + "_" + version) if "placebo" in e and e.endswith("_v1") else spec
    return [(n, f, y, ren(c), ren(r)) for n, f, y, c, r in pairs]


class FusionEvaluation:
    def __init__(self, cfg: dict, placebo: str = "v1"):
        self.cfg = cfg
        self.pairs = PAIRS if placebo == "v1" else with_placebo(PAIRS, placebo)
        self.placebo = placebo
        self.pc = cfg["analysis"]["paper"]
        self.margin = float(self.pc["margin"])
        s = self.pc["signflip"]
        self.sf = dict(max_exact=int(s["max_exact"]), n_random=int(s["n_random"]), seed=int(s["seed"]))
        self.finder = HypothesisTest(cfg, "fusion", {"metric": "spearman"})

    # ---------------------------------------------------------------------------------- plain comparisons
    def compare(self, name, family, years, cspec, rspec, run):
        c, r = Arm(self.finder, cspec), Arm(self.finder, rspec)
        data, truth, blocks = truth_and_blocks(c.cfg, years, run.log)
        K = int(blocks.max()) + 1
        rows = []
        for y in years:
            z, hot = truth[y]
            pcs, prs = c.preds(y, data.cells), r.preds(y, data.cells)
            for metric in ("spearman", "pr_auc"):
                est = score(z, hot, pcs, metric) - score(z, hot, prs, metric)

                def drop(b):
                    idx = np.flatnonzero(blocks != b)
                    return score(z, hot, pcs, metric, idx) - score(z, hot, prs, metric, idx)
                se, k = jack(drop, K)
                lo95, hi95 = t_interval(est, se, k - 1, 0.95)
                lo90, hi90 = t_interval(est, se, k - 1, 0.90)
                row = dict(test=name, family=family, year=y, metric=metric,
                           part="primary" if (metric == "spearman" and family in "FC") else "descriptive",
                           candidate=score(z, hot, pcs, metric), reference=score(z, hot, prs, metric), delta=est,
                           jack_se=se, jack_blocks=k, jack_low95=lo95, jack_high95=hi95, jack_low90=lo90,
                           jack_high90=hi90, margin=self.margin)
                if family in "FC":
                    row["verdict"] = verdict(lo95, hi95, lo90, hi90, self.margin)
                    if metric == "spearman":
                        row["p_signflip"], row["p_signflip_two"] = pooled_signflip(z, hot, pcs, prs, blocks, metric, self.sf)
                # seeds: candidate seed s against the matching placebo realisation s (or the reference run)
                mc, mr = c.members(y), r.members(y)
                if mc[0] is not None and len(c.runs) == 1:
                    ref_s = [mean_score(z, hot, p, metric) for p in prs]
                    ds = [mean_score(z, hot, m, metric) - ref_s[i % len(ref_s)] for i, m in enumerate(mc[0])]
                    row.update(seed_delta_mean=float(np.mean(ds)), seed_delta_min=float(np.min(ds)),
                               seed_delta_max=float(np.max(ds)), seed_delta_positive=int(np.sum(np.array(ds) > 0)))
                if len(prs) > 1 or len(pcs) > 1:            # spread over the placebo realisations
                    many = pcs if len(pcs) > 1 else prs
                    sc = [mean_score(z, hot, p, metric) for p in many]
                    row.update(realisation_min=float(np.min(sc)), realisation_max=float(np.max(sc)))
                rows.append(row)
        return rows

    # ---------------------------------------------------------------------------------- fusion gains and contrasts
    def gain_parts(self, years, cspec, rspec, run):
        c, r = Arm(self.finder, cspec), Arm(self.finder, rspec)
        data, truth, blocks = truth_and_blocks(c.cfg, years, run.log)
        parts = {}
        for y in years:
            z, hot = truth[y]
            parts[y] = (z, hot, c.preds(y, data.cells)[0], r.preds(y, data.cells)[0])
        return parts, blocks

    @staticmethod
    def gain(parts, rel=False, idx=None):
        """Mean over years of delta (or delta_rel) = S(cand) - S(ref) over the cells idx."""
        v = []
        for z, hot, pc, pr in parts.values():
            sc, sr = mean_score(z, hot, pc, "spearman", idx), mean_score(z, hot, pr, "spearman", idx)
            v.append((sc - sr) / (1 - sr) if rel else sc - sr)
        return float(np.mean(v))

    @staticmethod
    def block_gain(parts, blocks, rel=False):
        """Per-block gain (mean over years and months), and the cells of every block."""
        K = int(blocks.max()) + 1
        g = np.full(K, np.nan)
        cells = np.array([(blocks == b).sum() for b in range(K)])
        for b in range(K):
            idx = np.flatnonzero(blocks == b)
            vals = []
            for z, hot, pc, pr in parts.values():
                sc = np.nanmean(monthly_scores(z, hot, pc, ["spearman"], idx)["spearman"])
                sr = np.nanmean(monthly_scores(z, hot, pr, ["spearman"], idx)["spearman"])
                vals.append((sc - sr) / (1 - sr) if rel else sc - sr)
            g[b] = np.nanmean(vals) if np.isfinite(vals).any() else np.nan
        return g, cells

    def contrast(self, name, a, b, rel, run):
        """Contrast gain(a) - gain(b) pooled over the years, a and b = (years, cand, ref) of two target versions."""
        pa, blocks = self.gain_parts(*a, run)
        pb, blocks_b = self.gain_parts(*b, run)
        if not np.array_equal(blocks, blocks_b):
            raise ValueError(f"{name}: the two target versions have different blocks")
        K = int(blocks.max()) + 1
        est = self.gain(pa, rel) - self.gain(pb, rel)

        def drop(k):
            idx = np.flatnonzero(blocks != k)
            return self.gain(pa, rel, idx) - self.gain(pb, rel, idx)
        se, k = jack(drop, K)
        lo95, hi95 = t_interval(est, se, k - 1, 0.95)
        lo90, hi90 = t_interval(est, se, k - 1, 0.90)
        ga, cells = self.block_gain(pa, blocks, rel)
        gb, _ = self.block_gain(pb, blocks, rel)
        s = signflip_mean(ga - gb, weights=cells, **self.sf)
        return dict(test=name, family="F", year="pooled", metric="spearman_rel" if rel else "spearman", part="primary",
                    candidate=self.gain(pa, rel), reference=self.gain(pb, rel), delta=est, jack_se=se, jack_blocks=k,
                    jack_low95=lo95, jack_high95=hi95, jack_low90=lo90, jack_high90=hi90, margin=self.margin,
                    verdict=verdict(lo95, hi95, lo90, hi90, self.margin), p_signflip=s["p_greater"],
                    p_signflip_two=s["p_two"])

    def m2_passed(self, run) -> bool | None:
        t = RunRegistry(self.cfg).find("fusion_support")
        if t.empty:
            run.log("no fusion_support run found: F3 is reported but kept out of Holm")
            return None
        f = ROOT / t.sort_values("finished").iloc[-1].path / "fusion_support.json"
        return bool(json.loads(f.read_text(encoding="utf-8"))["M2"]["passed"])

    # ---------------------------------------------------------------------------------- all
    def run(self, run) -> dict:
        rows = []
        run.log(f"placebo version: {self.placebo}")
        for name, fam, years, cs, rs in self.pairs:
            run.log(f"{name} ...")
            rows += self.compare(name, fam, years, cs, rs, run)
        g = {n: (yrs, c, r) for n, yrs, c, r in GAINS}
        rows.append(self.contrast("F3", g["T-calm"], g["T-windy"], False, run))
        rows.append(self.contrast("F4", g["T0"], g["T-8km"], True, run))
        for n, (yrs, c, r) in g.items():                       # secondary: the gain on every target version
            run.log(f"gain {n} ...")
            rows += [dict(x, test=f"gain_{n}", family="secondary", part="descriptive", verdict=None)
                     for x in self.compare(f"gain_{n}", "secondary", yrs, c, r, run)]
        t = pd.DataFrame(rows)
        m2 = self.m2_passed(run)
        t["in_holm"] = False
        prim = (t.part == "primary") & (t.family.isin(["F", "C"]))
        t.loc[prim, "in_holm"] = True
        if m2 is not True:
            t.loc[t.test == "F3", "in_holm"] = False
            t.loc[t.test == "F3", "verdict"] = "uninformative (manipulation check M2 failed)" if m2 is False else t.loc[t.test == "F3", "verdict"]
        hl = []
        for fam in ("F", "C"):
            sub = t[t.in_holm & (t.family == fam)]
            keys = [f"{r.test}|{r.year}" for r in sub.itertuples()]
            adj = holm(dict(zip(keys, sub.p_signflip)))
            for r, k in zip(sub.itertuples(), keys):
                ph = adj[k]
                v = r.verdict if not str(r.verdict).startswith("superior") or ph < ALPHA \
                    else "superior, not significant after Holm"
                hl.append(dict(family=fam, test=r.test, year=r.year, metric=r.metric, delta=r.delta,
                               p_signflip=r.p_signflip, p_holm=ph, verdict=r.verdict, verdict_holm=v))
        holm_t = pd.DataFrame(hl)
        run.save_table("fusion_eval.csv", t)
        run.save_table("fusion_holm.csv", holm_t)
        run.save_json("fusion_eval.json", dict(m2_passed=m2, holm=holm_t.to_dict("records")))
        cols = ["test", "year", "metric", "candidate", "reference", "delta", "jack_low95", "jack_high95", "verdict"]
        run.log("primary and secondary rows (Spearman):\n"
                + t[t.metric.isin(["spearman", "spearman_rel"])][cols].round(4).to_string(index=False))
        run.log("Holm:\n" + holm_t.round(4).to_string(index=False))
        return {"tests": len(t), "m2_passed": m2}

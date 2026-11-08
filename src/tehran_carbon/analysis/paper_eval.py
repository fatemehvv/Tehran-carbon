"""Paper reading of one model comparison (python main.py paper evaluate --hypothesis <name> [<name> ...]).

The same two runs as the thesis decision rule (HypothesisTest.prepare: candidate and best reference chosen after the
results) are read with the inference of the paper (evaluation/inference.py; settings: analysis.paper):

  primary      the metrics of analysis.paper.metrics (PR-AUC of the city-wide hotspots, q = 0.90, and Spearman over all
               cells), or only the metrics a hypothesis names (`metrics:` in its spec; the others become `descriptive`
               rows): difference, delete-one-block jackknife intervals (95 % and 90 %), the thesis block-bootstrap
               interval, exact block sign-flip tests (pooled metric, and the cell-weighted mean of the per-block
               differences) and the four-way verdict with the margin (`margin:` in the spec, else analysis.paper.margin)
  thresholds   PR-AUC with the other hotspot labels (q in analysis.paper.thresholds)
  superpixel   the same metrics on groups of size x size cells (about one TROPOMI pixel)
  seeds        if the runs stored every seed (train.save_members): the difference for every candidate seed and the
               seed-to-seed standard deviation of each model (input of the margin rule of preregistration_paper_v2.md)
With several hypotheses, Holm's adjustment is applied to the one-sided pooled sign-flip p-values of the primary rows
(one per hypothesis, metric and test year).
Outputs (run folder): paper_<name>.csv for every hypothesis, paper_holm.csv, paper_evaluate.json
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluation.bootstrap import BlockBootstrap
from ..evaluation.hotspots import aggregate, group_blocks, hot_labels, superpixels
from ..evaluation.hypotheses import HypothesisTest
from ..evaluation.inference import holm, jackknife_se, signflip_mean, signflip_pooled, t_interval, verdict
from ..evaluation.robustness import block_table, mean_score


def members(folder, y: int, cells: np.ndarray) -> np.ndarray | None:
    """(S, T, n) predictions of every seed of a run, or None if the run did not store them."""
    f = folder / "preds" / f"cv_{y}_members.npz"
    if not f.exists():
        return None
    a = np.load(f)
    if not np.array_equal(a["cells"], cells):
        raise ValueError(f"{folder.name}: members have another label domain")
    return a["pred"]


class PaperEvaluation(HypothesisTest):
    def __init__(self, base_cfg: dict, name: str, spec: dict):
        super().__init__(base_cfg, name, spec)
        self.pc = base_cfg["analysis"]["paper"]
        self.margin = float(spec.get("margin", self.pc["margin"]))
        sf = self.pc["signflip"]
        self.sf = dict(max_exact=int(sf["max_exact"]), n_random=int(sf["n_random"]), seed=int(sf["seed"]))

    def row(self, z, hot, pc, pr, blocks, metric, label, boot=None, tests=True) -> dict:
        est, se, k, _ = jackknife_se(z, hot, pc, pr, blocks, metric)
        lo95, hi95 = t_interval(est, se, k - 1, 0.95)
        lo90, hi90 = t_interval(est, se, k - 1, 0.90)
        r = dict(label=label, metric=metric, candidate=mean_score(z, hot, pc, metric),
                 reference=mean_score(z, hot, pr, metric), delta=est, jack_se=se, jack_blocks=k, jack_low95=lo95,
                 jack_high95=hi95, jack_low90=lo90, jack_high90=hi90, margin=self.margin,
                 verdict=verdict(lo95, hi95, lo90, hi90, self.margin))
        if boot is not None:
            r["boot_low95"], r["boot_high95"] = BlockBootstrap.interval(
                boot.scores(z, hot, pc, metric) - boot.scores(z, hot, pr, metric), 0.95)
        if tests:
            s = signflip_pooled(z, hot, pc, pr, blocks, metric, **self.sf)
            bt = block_table(z, hot, pc, pr, blocks, metric)
            b = signflip_mean(bt.delta.to_numpy(), weights=bt.cells.to_numpy(), **self.sf)   # cell-weighted
            r.update(p_signflip_pooled=s["p_greater"], p_signflip_pooled_two=s["p_two"], signflip_exact=s["exact"],
                     blocks_valid=b["blocks"], p_signflip_blocks=b["p_greater"])
        return r

    def run(self, run) -> dict:
        p = self.prepare(run)
        metrics = list(self.pc["metrics"])
        primary = list(self.spec.get("metrics") or metrics)              # a hypothesis may name its primary metric(s)
        size, W = int(self.pc["superpixel"]), int(p.data.shape[1])
        rows = []
        for y in p.years:
            z, hot = p.truth[y]
            pc, pr = p.oof(p.cdir, y), p.oof(p.rdir, y)
            for m in primary + [m for m in metrics if m not in primary]:  # primary rows enter Holm; others are descriptive
                rows.append(dict(year=y, part="primary" if m in primary else "descriptive",
                                 **self.row(z, hot, pc, pr, p.blocks, m, "q=0.90", p.boot)))
            for q in self.pc["thresholds"]:                             # other hotspot labels
                if abs(float(q) - 0.90) < 1e-9:
                    continue
                rows.append(dict(year=y, part="threshold",
                                 **self.row(z, hot_labels(z, float(q)), pc, pr, p.blocks, "pr_auc", f"q={q}", tests=False)))
            g = superpixels(p.data.cells, W, size)                      # about one TROPOMI pixel
            zg, pcg, prg = aggregate(z, g), aggregate(pc, g), aggregate(pr, g)
            hg, bg = hot_labels(zg, 0.90), group_blocks(g, p.blocks)
            for m in metrics:
                rows.append(dict(year=y, part="superpixel",
                                 **self.row(zg, hg, pcg, prg, bg, m, f"{size}x{size} cells", tests=False)))
            mc, mr = members(p.cdir, y, p.data.cells), members(p.rdir, y, p.data.cells)
            if mc is not None:                                          # seed-to-seed spread
                for m in metrics:
                    sc = np.array([mean_score(z, hot, a, m) for a in mc])
                    sr = np.array([mean_score(z, hot, a, m) for a in mr]) if mr is not None else np.array([mean_score(z, hot, pr, m)])
                    rows.append(dict(year=y, part="seeds", label=f"{len(sc)} candidate seeds", metric=m,
                                     candidate=float(sc.mean()), reference=float(sr.mean()),
                                     delta=float(sc.mean() - sr.mean()), delta_min=float(sc.min() - sr.mean()),
                                     delta_max=float(sc.max() - sr.mean()), seed_sd_candidate=float(sc.std(ddof=1)) if len(sc) > 1 else np.nan,
                                     seed_sd_reference=float(sr.std(ddof=1)) if len(sr) > 1 else np.nan))
        table = pd.DataFrame(rows)
        table.insert(0, "hypothesis", self.name)
        table.insert(1, "candidate_run", p.cid)
        table.insert(2, "reference_run", p.best.run_id)
        run.save_table(f"paper_{self.name}.csv", table)
        prim = table[table.part == "primary"]
        run.log(f"{self.name}: {p.cid} vs {p.best.run_id}\n"
                + prim[["year", "metric", "delta", "jack_low95", "jack_high95", "p_signflip_pooled", "blocks_valid",
                        "verdict"]].round(4).to_string(index=False))
        return {"table": table, "candidate": p.cid, "reference": p.best.run_id}


def evaluate_family(base_cfg: dict, names: list[str], specs: dict, run) -> dict:
    """PaperEvaluation of every named hypothesis, then Holm over the primary rows of all of them."""
    tables, out = [], {}
    for n in names:
        res = PaperEvaluation(base_cfg, n, specs[n]).run(run)
        tables.append(res["table"])
        out[n] = {"candidate": res["candidate"], "reference": res["reference"]}
    allt = pd.concat(tables, ignore_index=True)
    prim = allt[allt.part == "primary"].copy()
    keys = [f"{r.hypothesis}|{r.metric}|{r.year}" for r in prim.itertuples()]
    adj = holm(dict(zip(keys, prim.p_signflip_pooled)))
    prim["p_holm"] = [adj[k] for k in keys]
    alpha = float(base_cfg["analysis"]["paper"]["alpha"])
    prim["verdict_holm"] = [v if not str(v).startswith("superior") or (np.isfinite(ph) and ph < alpha)
                            else "superior, not significant after Holm" for v, ph in zip(prim.verdict, prim.p_holm)]
    cols = ["hypothesis", "year", "metric", "delta", "p_signflip_pooled", "p_holm", "verdict", "verdict_holm"]
    run.save_table("paper_holm.csv", prim[cols])
    run.save_json("paper_evaluate.json", dict(hypotheses=out, primary=prim.to_dict("records")))
    run.log("Holm over the family:\n" + prim[cols[1:]].round(4).to_string(index=False))
    return {n: v["candidate"] for n, v in out.items()}

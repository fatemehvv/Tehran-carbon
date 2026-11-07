"""Robustness checks of one hypothesis comparison (paper, branch paper_TC; NOT part of the thesis decisions).

The 95 % interval of HypothesisTest resamples only 12 spatial blocks, and the hotspots sit in a few of them, so its
percentile intervals can be lopsided. These checks look at the SAME two runs (the hypothesis' candidate and its best
reference, chosen by HypothesisTest.prepare) in three other ways (rules: docs/paper/preregistration_paper_checks.md):

  blocks       the metric inside every spatial block (months without a hotspot in the block are skipped): in how many
               blocks the candidate is ahead, a Wilcoxon signed-rank test and a sign test over the blocks
  jackknife    the delta with one block left out at a time: smallest and largest delta, jackknife standard error and a
               t interval with K - 1 degrees of freedom
  persistence  the OBSERVED map of every other study year (and their mean), month by month, scored on the test year:
               what re-using an old satellite map achieves. A reference ceiling, not a competitor (it cannot answer a
               scenario). Paired delta candidate - persistence with the same block bootstrap as the hypothesis.
Reading of one test year
  robust          bootstrap CI and jackknife CI above 0, and the candidate ahead in more than half of the valid blocks
  fragile         bootstrap CI above 0, but the jackknife CI includes 0 or the candidate leads in at most half the blocks
  no effect shown the bootstrap CI includes 0 (or the delta is not positive)
Outputs (run folder): <name>_blocks.csv, <name>_jackknife.csv, <name>_persistence.csv, <name>_robustness.csv/.json
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .bootstrap import BlockBootstrap
from .hypotheses import HypothesisTest
from .metrics import monthly_scores


def mean_score(z, hot, pred, metric: str, idx=None) -> float:
    """Month-averaged metric over the cells idx (all domain cells when idx is None)."""
    return float(np.nanmean(monthly_scores(z, hot, pred, [metric], idx)[metric]))


def block_table(z, hot, pc, pr, blocks: np.ndarray, metric: str) -> pd.DataFrame:
    """One row per spatial block: the metric of candidate and reference inside the block, over the months in which
    the block has at least one hotspot (and both scores are defined)."""
    rows = []
    total = max(int(hot.sum()), 1)
    for b in range(int(blocks.max()) + 1):
        idx = np.flatnonzero(blocks == b)
        sc = monthly_scores(z, hot, pc, [metric], idx)[metric]
        sr = monthly_scores(z, hot, pr, [metric], idx)[metric]
        both = np.isfinite(sc) & np.isfinite(sr)
        rows.append(dict(block=b, cells=len(idx), hotspot_share=float(hot[:, idx].sum()) / total,
                         months=int(both.sum()),
                         candidate=float(sc[both].mean()) if both.any() else np.nan,
                         reference=float(sr[both].mean()) if both.any() else np.nan))
    t = pd.DataFrame(rows)
    t["delta"] = t.candidate - t.reference
    return t


def block_tests(table: pd.DataFrame) -> dict:
    """Over the blocks with a defined delta: blocks won, Wilcoxon signed-rank and sign-test p-values (two-sided)."""
    d = table.delta.dropna().to_numpy()
    won, nonzero = int((d > 0).sum()), int((d != 0).sum())
    try:
        p_wil = float(stats.wilcoxon(d).pvalue) if nonzero else np.nan
    except ValueError:
        p_wil = np.nan
    p_sign = float(stats.binomtest(won, nonzero, 0.5).pvalue) if nonzero else np.nan
    return dict(blocks_valid=len(d), blocks_won=won, wilcoxon_p=p_wil, sign_p=p_sign)


def jackknife(z, hot, pc, pr, blocks: np.ndarray, metric: str, level: float) -> tuple[pd.DataFrame, dict]:
    """Delta with each block left out in turn; jackknife SE and t interval around the full-sample delta."""
    K = int(blocks.max()) + 1
    full = mean_score(z, hot, pc, metric) - mean_score(z, hot, pr, metric)
    d = np.array([mean_score(z, hot, pc, metric, np.flatnonzero(blocks != b))
                  - mean_score(z, hot, pr, metric, np.flatnonzero(blocks != b)) for b in range(K)])
    se = float(np.sqrt((K - 1) / K * ((d - d.mean()) ** 2).sum()))
    t = float(stats.t.ppf(1 - (1 - level) / 2, K - 1))
    table = pd.DataFrame({"left_out_block": np.arange(K), "delta": d})
    return table, dict(jack_min=float(d.min()), jack_max=float(d.max()), jack_positive=int((d > 0).sum()),
                       jack_se=se, jack_low=full - t * se, jack_high=full + t * se)


def persistence_maps(data, test_year: int) -> list[tuple[str, np.ndarray]]:
    """Observed maps (T, n_domain) of the other study years with the same number of steps, and their mean."""
    T = data.z[test_year].shape[0]
    maps = [(str(a), data.z[a].reshape(T, -1)[:, data.cells]) for a in data.years
            if a != test_year and data.z[a].shape[0] == T]
    if len(maps) > 1:
        maps.append(("mean_" + "_".join(name for name, _ in maps), np.nanmean([m for _, m in maps], axis=0)))
    return maps


def reading(boot_low: float, delta: float, jack_low: float, won: int, valid: int) -> str:
    if not (delta > 0 and boot_low > 0):
        return "no effect shown"
    return "robust" if jack_low > 0 and won > valid / 2 else "fragile"


class RobustnessChecks(HypothesisTest):
    def run(self, run) -> dict:
        p = self.prepare(run)
        m = self.metric
        blocks_all, jack_all, pers_rows, summary = [], [], [], []
        for y in p.years:
            z, hot = p.truth[y]
            pc, pr = p.oof(p.cdir, y), p.oof(p.rdir, y)
            sc, sr = p.score(pc, y), p.score(pr, y)
            bc = p.boot.scores(z, hot, pc, m)                     # candidate on every resample (reused below)
            lo, hi = BlockBootstrap.interval(bc - p.boot.scores(z, hot, pr, m), p.level)

            bt = block_table(z, hot, pc, pr, p.blocks, m)
            bt.insert(0, "year", y)
            blocks_all.append(bt)
            tests = block_tests(bt)
            jt, js = jackknife(z, hot, pc, pr, p.blocks, m, p.level)
            jt.insert(0, "year", y)
            jack_all.append(jt)
            summary.append(dict(year=y, candidate=sc, reference=sr, delta=sc - sr, boot_low=lo, boot_high=hi, **js,
                                **tests, reading=reading(lo, sc - sr, js["jack_low"], tests["blocks_won"],
                                                         tests["blocks_valid"])))

            for name, pmap in persistence_maps(p.data, y):
                plo, phi = BlockBootstrap.interval(bc - p.boot.scores(z, hot, pmap, m), p.level)
                sp = p.score(pmap, y)
                pers_rows.append(dict(year=y, observed_map=name, persistence=sp, candidate=sc, delta=sc - sp,
                                      ci_low=plo, ci_high=phi))

        summary = pd.DataFrame(summary)
        run.save_table(f"{self.name}_robustness.csv", summary)
        run.save_table(f"{self.name}_blocks.csv", pd.concat(blocks_all, ignore_index=True))
        run.save_table(f"{self.name}_jackknife.csv", pd.concat(jack_all, ignore_index=True))
        run.save_table(f"{self.name}_persistence.csv", pd.DataFrame(pers_rows))
        result = dict(hypothesis=self.name, metric=m, candidate=p.cid, reference=p.best.run_id,
                      years=summary.to_dict("records"), persistence=pers_rows)
        run.save_json(f"{self.name}_robustness.json", result)
        run.log(f"{self.name} robustness: candidate {p.cid} vs reference {p.best.run_id}\n"
                f"{summary.round(4).to_string(index=False)}")
        if pers_rows:
            run.log(f"persistence:\n{pd.DataFrame(pers_rows).round(4).to_string(index=False)}")
        out = {"candidate": p.cid, "reference": p.best.run_id}
        out.update({f"reading_{r.year}": r.reading for r in summary.itertuples()})
        return out

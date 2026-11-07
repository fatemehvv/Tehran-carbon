"""Power of the paper's block-level tests, estimated from comparisons that already exist
(python main.py paper power --hypothesis H2 H2c ...). Written to fix the confirmatory design BEFORE the new runs.

For one hypothesis (its candidate and best reference) and one test year, the per-block differences d_b are taken from
the existing out-of-fold predictions (evaluation/robustness.py: block_table; blocks without a defined value, e.g. no
hotspot for PR-AUC, are dropped). Their spread around their mean is the block-to-block noise. For a true effect E,
simulated per-block differences are E + noise, with the noise drawn
  normal      from N(0, sd(d_b)), and
  empirical   by resampling the centred d_b,
over the same number of valid blocks, and tested with the exact one-sided sign-flip test of the block mean at alpha
(analysis.paper.alpha; Holm makes the effective level smaller still). Power = share of simulations with p < alpha.
Also reported: the smallest p any exact sign-flip test over that many blocks can reach (1 / 2^K) and the smallest
effect with at least 80 % power. This is an approximation: blocks are taken as independent.
Outputs: paper_power.csv, paper_power.json
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluation.hypotheses import HypothesisTest
from ..evaluation.inference import assignments, min_p
from ..evaluation.robustness import block_table


def simulate_power(resid: np.ndarray, effects, alpha: float, n_sim: int, seed: int, kind: str) -> list[float]:
    """Power of the exact one-sided sign-flip test of the block mean for every effect."""
    K = len(resid)
    if K == 0:
        return [np.nan] * len(effects)
    signs = np.where(assignments(K, max_exact=20)[0], -1.0, 1.0)          # (2^K, K); row 0 = identity
    rng = np.random.default_rng(seed)
    sd = resid.std(ddof=1) if K > 1 else 0.0
    out = []
    for e in effects:
        noise = (rng.normal(0, sd, (n_sim, K)) if kind == "normal"
                 else rng.choice(resid, size=(n_sim, K), replace=True))
        d = e + noise                                                     # (n_sim, K)
        st = d @ signs.T / K                                              # (n_sim, 2^K)
        p = (st >= st[:, :1] - 1e-12).mean(1)
        out.append(float((p < alpha).mean()))
    return out


class PowerAnalysis(HypothesisTest):
    def run(self, run) -> dict:
        pc_ = self.cfg["analysis"]["paper"]
        pw = pc_["power"]
        alpha, n_sim, seed = float(pc_["alpha"]), int(pw["n_sim"]), int(pw["seed"])
        effects = [float(e) for e in pw["effects"]]
        fine = np.round(np.arange(0.005, 0.2001, 0.005), 3)
        p = self.prepare(run)
        rows = []
        for y in p.years:
            z, hot = p.truth[y]
            pc, pr = p.oof(p.cdir, y), p.oof(p.rdir, y)
            for m in pc_["metrics"]:
                d = block_table(z, hot, pc, pr, p.blocks, m).delta.to_numpy()
                d = d[np.isfinite(d)]
                resid = d - d.mean() if len(d) else d
                base = dict(hypothesis=self.name, year=y, metric=m, blocks_valid=len(d),
                            block_mean_delta=float(d.mean()) if len(d) else np.nan,
                            block_sd=float(d.std(ddof=1)) if len(d) > 1 else np.nan, min_p=min_p(len(d)),
                            reachable=bool(len(d) and min_p(len(d)) < alpha))
                for kind in ["normal", "empirical"]:
                    pw_e = simulate_power(resid, effects, alpha, n_sim, seed, kind)
                    pw_f = simulate_power(resid, list(fine), alpha, max(n_sim // 4, 200), seed, kind)
                    ok = [e for e, v in zip(fine, pw_f) if np.isfinite(v) and v >= 0.8]
                    rows.append(dict(base, noise=kind, mde80=float(ok[0]) if ok else np.nan,
                                     **{f"power_{e:g}": v for e, v in zip(effects, pw_e)}))
        t = pd.DataFrame(rows)
        run.save_table(f"paper_power_{self.name}.csv", t)
        run.log(f"{self.name} power (alpha {alpha}):\n{t.round(3).to_string(index=False)}")
        return {"table": t}


def power_family(base_cfg: dict, names: list[str], specs: dict, run) -> dict:
    tables = [PowerAnalysis(base_cfg, n, specs[n]).run(run)["table"] for n in names]
    t = pd.concat(tables, ignore_index=True)
    run.save_table("paper_power.csv", t)
    run.save_json("paper_power.json", t.to_dict("records"))
    return {"rows": len(t)}

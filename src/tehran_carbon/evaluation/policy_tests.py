"""Pre-registered decision rules of RQ3 (config: hypotheses.H3 / H4 with kind: policy; docs/preregistration_rq3.md).

Both use the transfers of the latest run of the experiment (agent and static policies tuned on year a, evaluated on
year b != a). Episode returns are paired across policies (common random numbers), so the bootstrap resamples episodes.
H3 effectiveness  agent (returns averaged over seeds) - best tuned static policy of the test year
H4 risk           CVaR (mean of the worst share of returns) of the policy screened by simulation on the training
                  year - mean CVaR over the whole candidate set (a choice made without simulation)
Verdict per rule: supported (delta > 0, CI above 0 in every transfer), partial (delta > 0 everywhere), not supported.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import ROOT
from ..tracking import RunRegistry


class PolicyHypothesis:
    def __init__(self, base_cfg: dict, name: str, spec: dict):
        self.name, self.spec = name, spec
        rows = RunRegistry(base_cfg).find(spec["experiment"]).sort_values("finished")
        if rows.empty:
            raise ValueError(f"{name}: no finished run of {spec['experiment']}")
        self.run_id, self.folder = rows.iloc[-1].run_id, ROOT / rows.iloc[-1].path
        self.tuned = json.loads((self.folder / "tuned.json").read_text(encoding="utf-8"))

    def load(self, a: str, b: str) -> tuple[list[str], np.ndarray]:
        """Policy names and returns (P, K); the agent seeds are merged into one row 'rl' (mean over seeds)."""
        d = np.load(self.folder / "preds" / f"returns_{a}to{b}.npz")
        names, fam, R = list(d["names"]), d["families"], d["returns"]
        keep = fam != "rl"
        return [n for n, k in zip(names, keep) if k] + ["rl"], np.vstack([R[keep], R[~keep].mean(0)])

    def cvar(self, R: np.ndarray) -> np.ndarray:
        """Mean of the worst `cvar` share of the returns along the last axis."""
        k = max(1, int(np.ceil(float(self.spec["cvar"]) * R.shape[-1])))
        return np.sort(R, -1)[..., :k].mean(-1)

    def statistic(self, a: str, b: str):
        """(delta function of the episode indices, description) of one transfer a -> b."""
        names, Rb = self.load(a, b)
        if self.spec["test"] == "effectiveness":
            # families named in exclude_families (e.g. the reactive weather rule) are reported, not references
            skip = set(self.spec.get("exclude_families") or [])
            refs = [names.index(n) for f, n in self.tuned[a].items() if f not in skip]
            best = refs[int(np.argmax(Rb[refs].mean(1)))]
            extra = {f"rl_minus_{f}": float(Rb[-1].mean() - Rb[names.index(n)].mean()) for f, n in self.tuned[a].items() if f in skip}
            return (lambda idx: Rb[-1, idx].mean() - Rb[best, idx].mean()), dict(reference=names[best], **extra)
        names_a, Ra = self.load(a, a)
        if names_a != names:
            raise ValueError(f"{self.name}: policy sets of {a}->{a} and {a}->{b} differ")
        screened = int(np.argmax(Ra.mean(1)))
        loss_share = (Rb < 0).mean(1)
        return (lambda idx: self.cvar(Rb[screened, idx]) - self.cvar(Rb[:, idx]).mean()), dict(
            screened=names[screened], loss_share_screened=float(loss_share[screened]),
            loss_share_set=float(loss_share.mean()), harmful_policies=int((loss_share > 0.5).sum()), n_policies=len(names))

    def run(self, run) -> dict:
        b = self.spec["bootstrap"]
        rng = np.random.default_rng(int(b["seed"]))
        level = float(b["level"])
        years = sorted(self.tuned)
        rows = []
        for a in years:
            for t in years:
                if a == t:
                    continue
                stat, info = self.statistic(a, t)
                K = len(np.load(self.folder / "preds" / f"returns_{a}to{t}.npz")["seeds"])
                boot = np.array([stat(rng.integers(0, K, K)) for _ in range(int(b["n"]))])
                lo, hi = np.quantile(boot, [(1 - level) / 2, 1 - (1 - level) / 2])
                rows.append(dict(test=f"transfer_{a}to{t}", delta=float(stat(np.arange(K))), ci_low=float(lo),
                                 ci_high=float(hi), **info))
        d = pd.DataFrame(rows)
        ok = bool((d.delta > 0).all())
        verdict = "supported" if ok and (d.ci_low > 0).all() else ("partial" if ok else "not supported")
        run.save_table(f"{self.name}_decision.csv", d)
        result = dict(hypothesis=self.name, test=self.spec["test"], run=self.run_id, verdict=verdict, tests=rows)
        run.save_json(f"{self.name}_decision.json", result)
        run.log(f"{self.name}: {verdict}\n{d.round(4).to_string(index=False)}")
        return {"verdict": verdict, "run": self.run_id}

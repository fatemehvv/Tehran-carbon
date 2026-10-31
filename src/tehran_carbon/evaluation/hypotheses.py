"""Pre-registered decision rule for a hypothesis (config: hypotheses.<name>).

candidate   one run (selector: experiment + optional `where` on its resolved config)
references  the best of the selected runs by the primary metric (mean over years), chosen AFTER the results, which is
            conservative for the candidate
Verdict
  supported      in every year delta > 0 with the 95 % block-bootstrap CI above 0, and delta > 0 in every transfer
  partial        delta > 0 in every year and every transfer, but at least one CI includes 0
  not supported  otherwise
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..config import ROOT, Config
from ..data.dataset import CellMonthDataset
from ..data.grid import Grid
from ..data.splits import SpatialSplit
from ..tracking import RunRegistry
from .bootstrap import BlockBootstrap
from .metrics import monthly_scores


class HypothesisTest:
    def __init__(self, base_cfg: dict, name: str, spec: dict):
        self.cfg, self.name, self.spec = base_cfg, name, spec
        self.metric = spec["metric"]
        self.registry = RunRegistry(base_cfg)

    def select(self, selector: dict) -> list[tuple[str, Path, dict]]:
        """(run_id, folder, config) of the latest finished run of every matching variant."""
        found = {}
        rows = self.registry.find(selector["experiment"]).sort_values("finished")
        for _, r in rows.iterrows():
            folder = ROOT / r.path
            rcfg = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
            ok = True
            for key, cond in (selector.get("where") or {}).items():
                val = Config.get(rcfg, key, None)
                ok &= (val in cond["in"]) if isinstance(cond, dict) and "in" in cond else (val == cond)
            if ok:
                found[r.variant] = (r.run_id, folder, rcfg)          # later rows overwrite earlier ones
        return list(found.values())

    def run(self, run) -> dict:
        cands = self.select(self.spec["candidate"])
        if len(cands) != 1:
            raise ValueError(f"{self.name}: candidate selector matches {len(cands)} runs (need exactly 1)")
        refs = [x for sel in self.spec["references"] for x in self.select(sel)]
        if not refs:
            raise ValueError(f"{self.name}: no finished reference runs")
        cid, cdir, ccfg = cands[0]
        data = CellMonthDataset.build(ccfg, log=run.log)
        years = data.years
        truth = {y: (data.z[y].reshape(data.z[y].shape[0], -1)[:, data.cells], data.hot[y].reshape(data.z[y].shape[0], -1)[:, data.cells]) for y in years}

        def oof(folder, y):
            a = np.load(folder / "preds" / f"cv_{y}.npz")
            if not np.array_equal(a["cells"], data.cells):
                raise ValueError(f"{folder.name} has a different label domain than the candidate")
            return a["pred"]

        def score(pred, y):
            return float(np.nanmean(monthly_scores(*truth[y], pred, [self.metric])[self.metric]))

        table = []
        for rid, folder, _ in [cands[0]] + refs:
            table.append(dict(run_id=rid, role="candidate" if rid == cid else "reference",
                              **{f"{self.metric}_{y}": score(oof(folder, y), y) for y in years}))
        table = pd.DataFrame(table)
        table["mean"] = table[[f"{self.metric}_{y}" for y in years]].mean(1)
        best = table[table.role == "reference"].sort_values("mean", ascending=False).iloc[0]
        rdir = next(f for rid, f, _ in refs if rid == best.run_id)
        run.log(f"{self.name}: candidate {cid} vs best reference {best.run_id}")

        split = SpatialSplit(ccfg["split"], Grid(ccfg), data.label_domain, seed=int(ccfg["seed"]))
        bcfg = ccfg["eval"]["bootstrap"]
        boot = BlockBootstrap(split.blocks(split.block_km), int(bcfg["n"]), int(bcfg["seed"]))
        level = float(bcfg["level"])
        rows, ok_all, ci_all = [], True, True
        for y in years:
            pc, pr = oof(cdir, y), oof(rdir, y)
            diff = boot.scores(*truth[y], pc, self.metric) - boot.scores(*truth[y], pr, self.metric)
            lo, hi = BlockBootstrap.interval(diff, level)
            sc, sr = score(pc, y), score(pr, y)
            rows.append(dict(test=f"cv_{y}", candidate=sc, reference=sr, delta=sc - sr, ci_low=lo, ci_high=hi,
                             persistence=np.nan))
            ok_all &= sc > sr
            ci_all &= lo > 0
        for a in years:
            for b in years:
                if a == b or not (cdir / "preds" / f"transfer_{a}to{b}.npz").exists():
                    continue
                sc = score(np.load(cdir / "preds" / f"transfer_{a}to{b}.npz")["pred"], b)
                sr = score(np.load(rdir / "preds" / f"transfer_{a}to{b}.npz")["pred"], b)
                # persistence = last year's OBSERVED map: a reference ceiling, not a competitor (it cannot respond
                # to scenarios, so it is useless for RQ3)
                rows.append(dict(test=f"transfer_{a}to{b}", candidate=sc, reference=sr, delta=sc - sr, ci_low=np.nan,
                                 ci_high=np.nan, persistence=score(truth[a][0], b)))
                ok_all &= sc > sr
        verdict = "supported" if ok_all and ci_all else ("partial" if ok_all else "not supported")
        decision = pd.DataFrame(rows)
        run.save_table(f"{self.name}_scores.csv", table)
        run.save_table(f"{self.name}_decision.csv", decision)
        result = dict(hypothesis=self.name, metric=self.metric, candidate=cid, reference=best.run_id,
                      verdict=verdict, tests=rows)
        run.save_json(f"{self.name}_decision.json", result)
        run.log(f"{self.name}: {verdict}\n{decision.round(4).to_string(index=False)}")
        return {"verdict": verdict, "candidate": cid, "reference": best.run_id}

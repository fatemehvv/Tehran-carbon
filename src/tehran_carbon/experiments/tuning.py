"""Random search of hyper-parameters on the development years (python main.py tune --exp <experiment>; kind: tune).

cfg["tune"]:
  trials       number of trials (all drawn before any is scored: random search, Bergstra & Bengio 2012, JMLR)
  seed         seed of the draws and of the inner folds
  inner_folds  the domain blocks (split.block_km) are grouped into this many folds (block_kfold, with the buffer)
  metric       score of a trial: the metric of the stitched out-of-fold maps, mean over the tuning years
  years        tuning years (default: every year except split.test_year)
  space        {dotted.config.key: spec}; spec = {choice: [...]} | {loguniform: [lo, hi]} | {uniform: [lo, hi]} |
               {intuniform: [lo, hi]}
Every trial trains one seed (the first of train.seeds) on the training cells of all tuning years of an inner fold and
predicts that fold's test cells in every tuning year. The test year is never read.
Outputs: tune_trials.csv (every trial), tune_best.json (the best values; they are written into a new config version
by hand, so that the final runs keep a plain, versioned config).
"""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd

from ..config import Config
from ..data.dataset import CellMonthDataset
from ..data.graphs import GridGraph
from ..data.grid import Grid
from ..data.sources.pollution import StationHoldout
from ..data.splits import SpatialSplit
from ..evaluation.metrics import monthly_scores
from ..models.backbones import BACKBONES
from ..models.baselines import BASELINES
from ..training.trainer import NeuralPredictor


def sample(space: dict, rng: np.random.Generator) -> dict:
    """One draw of every key of the search space."""
    out = {}
    for key, spec in space.items():
        (kind, val), = spec.items()
        if kind == "choice":
            v = val[int(rng.integers(len(val)))]
        elif kind == "loguniform":
            v = float(np.exp(rng.uniform(np.log(val[0]), np.log(val[1]))))
        elif kind == "uniform":
            v = float(rng.uniform(val[0], val[1]))
        elif kind == "intuniform":
            v = int(rng.integers(int(val[0]), int(val[1]) + 1))
        else:
            raise ValueError(f"unknown search-space kind '{kind}' for {key}")
        out[key] = copy.deepcopy(v)
    return out


class RandomSearch:
    def __init__(self, cfg: dict, run):
        self.cfg, self.run = cfg, run
        self.t = cfg["tune"]

    def run_all(self) -> dict:
        cfg, run, t = self.cfg, self.run, self.t
        base = copy.deepcopy(cfg)
        base["train"].update(seeds=[int(base["train"]["seeds"][0])], save_final=False)
        data = CellMonthDataset.build(base, log=run.log)
        grid = Grid(base)
        test = base["split"].get("test_year")
        years = [int(y) for y in (t.get("years") or [y for y in data.years if test is None or y != int(test)])]
        scfg = dict(base["split"], scheme="block_kfold", n_folds=int(t["inner_folds"]))
        split = SpatialSplit(scfg, grid, data.label_domain, seed=int(t["seed"]))
        m = base["model"]
        neural = m["type"] == "neural"
        graph = GridGraph(grid, m["backbone"]["scales"]) if neural and BACKBONES.get(m["backbone"]["type"]).needs_graph else None
        branches = list(data.columns(m["modalities"], m["use_context"]))
        holdout = (StationHoldout(base, grid, data)
                   if base["split"].get("fold_aware_stations") and "pollution" in branches else None)
        cells, metric = data.cells, t["metric"]
        rng = np.random.default_rng(int(t["seed"]))
        trials = [sample(t["space"], rng) for _ in range(int(t["trials"]))]
        run.log(f"random search: {len(trials)} trials x {split.n_folds} inner folds, years {years}, metric {metric}")
        rows = []
        for i, params in enumerate(trials):
            c = copy.deepcopy(base)
            for k, v in params.items():
                Config.set(c, k, v)
            t0 = time.time()
            oof = {y: np.full((data.X[y].shape[0], grid.n), np.nan, np.float32) for y in years}
            for k in range(split.n_folds):
                tr, te = split.masks(k)
                Xs = [data.X[y] for y in years]
                if holdout is not None:
                    keep = holdout.keep(te, split.buffer_km)
                    Xs = [holdout.patch(X, y, keep) for X, y in zip(Xs, years)]
                p = (NeuralPredictor(c, data, graph, run.log) if neural
                     else BASELINES.get(c["model"]["baseline"]["name"])(c, data, graph, run.log))
                p.fit(Xs, [data.z[y] for y in years], [tr] * len(years))
                te_cells = np.flatnonzero(te.ravel())
                for X, y in zip(Xs, years):
                    oof[y][:, te_cells] = p.predict(X, te).reshape(X.shape[0], -1)[:, te_cells]
            per_year = {}
            for y in years:
                z = data.z[y].reshape(data.z[y].shape[0], -1)[:, cells]
                hot = data.hot[y].reshape(z.shape[0], -1)[:, cells]
                per_year[y] = float(np.nanmean(monthly_scores(z, hot, oof[y][:, cells], [metric])[metric]))
            score = float(np.mean(list(per_year.values())))
            rows.append(dict(trial=i, score=score, seconds=round(time.time() - t0, 1),
                             **{f"{metric}_{y}": v for y, v in per_year.items()},
                             **{k: (str(v) if isinstance(v, list) else v) for k, v in params.items()}))
            run.save_table("tune_trials.csv", pd.DataFrame(rows))          # rewritten after every trial
            run.log(f"  trial {i:02d}: {metric}={score:.4f} ({rows[-1]['seconds']} s) {params}")
        table = pd.DataFrame(rows)
        best = int(table.score.idxmax())
        out = dict(experiment=cfg["experiment"]["name"], metric=metric, years=years, inner_folds=split.n_folds,
                   trials=len(trials), best_trial=best, best_score=float(table.score[best]), best=trials[best],
                   score_sd_over_trials=float(table.score.std()))
        run.save_json("tune_best.json", out)
        run.log(f"best trial {best}: {metric}={out['best_score']:.4f} {trials[best]}")
        return {"best_score": round(out["best_score"], 4), "best_trial": best}

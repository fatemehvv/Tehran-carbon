"""Timing pilot (python main.py paper timing --exp <experiment> [--variant i] [--epochs 20]).

Fits one predictor on the training cells of fold 0 of the first scored year, with one seed and a few epochs, and
extrapolates the cost of the full experiment and of the hyper-parameter search of the paper (50 trials). It is the
input of the compute rule of preregistration_paper_v2.md (Hyperband for the neural families if the full search would
exceed the budget). Nothing is stored in the run history except this analysis run.
Output: timing_<experiment>.json
"""
from __future__ import annotations

import copy
import time

import numpy as np

from ..experiments.cross_validation import CrossValidation


def time_experiment(cfg: dict, run, epochs: int, trials: int = 50, inner_folds: int = 4) -> dict:
    c = copy.deepcopy(cfg)
    full_epochs, seeds = int(c["train"]["epochs"]), len(c["train"]["seeds"])
    c["train"].update(epochs=int(epochs), seeds=[int(c["train"]["seeds"][0])], save_final=False)
    cv = CrossValidation(c, run)
    data, split = cv.data, cv.split
    test = c["split"].get("test_year")
    years = [y for y in data.years if y != int(test)] if test else data.years[:1]
    tr, te = split.masks(0)
    Xs, zs = [data.X[y] for y in years], [data.z[y] for y in years]
    if cv.holdout is not None:
        keep = cv.holdout.keep(te, split.buffer_km)
        Xs = [cv.holdout.patch(X, y, keep) for X, y in zip(Xs, years)]
    t0 = time.time()
    cv.predictor().fit(Xs, zs, [tr] * len(Xs))
    fit_s = time.time() - t0
    per_fit = fit_s * (full_epochs / epochs if cv.neural else 1.0)            # one seed, full epochs
    n_scored = 1 if test else len(data.years)
    n_transfer = 0 if test or not c["split"]["transfer"] else len(data.years) * (len(data.years) - 1)
    n_final = 1 if c["train"]["save_final"] and cv.neural and not test else 0
    fits = (split.n_folds * n_scored + n_transfer + n_final) * (seeds if cv.neural else 1)
    out = dict(experiment=cfg["experiment"]["name"], variant=cfg["experiment"]["variant_label"], neural=cv.neural,
               pilot_epochs=int(epochs) if cv.neural else None, pilot_fit_s=round(fit_s, 2),
               per_fit_full_s=round(per_fit, 2), fits_per_run=int(fits),
               run_hours=round(fits * per_fit / 3600, 2),
               search_hours=round(trials * inner_folds * per_fit / 3600, 2),
               device=str(getattr(cv.predictor(), "device", "cpu")), folds=int(split.n_folds),
               train_cells_fold0=int(tr.sum()), years=[int(y) for y in years])
    run.save_json(f"timing_{out['experiment']}.json", out)
    run.log(" ".join(f"{k}={v}" for k, v in out.items()))
    return out


def summarize(rows: list[dict]) -> dict:
    return {"run_hours_total": float(np.sum([r["run_hours"] for r in rows])),
            "search_hours_total": float(np.sum([r["search_hours"] for r in rows]))}

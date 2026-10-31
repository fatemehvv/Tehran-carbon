"""Train one resolved config under the validation design and store everything in its run folder.

  spatial CV   for every year and fold: train on the other folds (minus the buffer; one inner block held out for
               early stopping), predict the test fold for all 12 months -> preds/folds/<year>/fold<k>.npz
               (a fold that already exists is skipped, so an interrupted run can be resumed)
               folds are stitched into one out-of-fold map per month -> preds/cv_<year>.npz (+ weights_<year>.npz)
  transfer     train on all label cells of one year, predict the other -> preds/transfer_<a>to<b>.npz
  final        train on all label cells of all years -> models/final.pt (reused by init_from: RQ1 -> RQ2 -> RQ3)
Out-of-fold arrays are (12, n_domain) in the order of dataset.cells.
"""
from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd
import torch

from ..data.dataset import CellMonthDataset
from ..data.graphs import GridGraph
from ..data.grid import Grid
from ..data.splits import SpatialSplit
from ..evaluation.metrics import monthly_scores
from ..models.backbones import BACKBONES
from ..models.baselines import BASELINES
from ..training.trainer import NeuralPredictor
from ..viz.maps import RunMaps


class CrossValidation:
    def __init__(self, cfg: dict, run):
        self.cfg, self.run = cfg, run
        self.data = CellMonthDataset.build(cfg, log=run.log)
        self.grid = Grid(cfg)
        zall = np.concatenate([self.data.z[y] for y in self.data.years])
        self.split = SpatialSplit(cfg["split"], self.grid, self.data.label_domain, z=zall, seed=int(cfg["seed"]))
        m = cfg["model"]
        self.neural = m["type"] == "neural"
        if not self.neural and m["type"] != "baseline":
            raise ValueError(f"unknown model.type '{m['type']}'")
        self.graph = (GridGraph(self.grid.H, self.grid.W, m["backbone"]["scales"])
                      if self.neural and BACKBONES.get(m["backbone"]["type"]).needs_graph else None)
        self.use_val = self.neural and bool(cfg["split"]["inner_val"])
        self.branches = list(self.data.columns(m["modalities"], m["use_context"]))

    def predictor(self):
        if self.neural:
            return NeuralPredictor(self.cfg, self.data, self.graph, self.run.log)
        return BASELINES.get(self.cfg["model"]["baseline"]["name"])(self.cfg, self.data, self.graph, self.run.log)

    def run_all(self, smoke: bool = False) -> dict:
        cfg, run, data, split = self.cfg, self.run, self.data, self.split
        dom, cells, names = data.label_domain, data.cells, cfg["eval"]["metrics"]
        run.save_json("split.json", split.info)
        np.save(run.path("preds", "fold_map.npy"), split.fold_map)
        e = cfg["experiment"]
        run.log(f"{e['name']} [{e['variant_label'] or '-'}]: {len(data.features)} features, {len(cells)} label cells, "
                f"{split.n_folds} folds ({split.scheme}, {split.block_km:.0f} km, buffer {split.buffer_km} km)")
        metrics = {"cv": {}, "transfer": {}}

        # ------------------------------------------------------------ spatial CV
        for y in data.years:
            X, z = data.X[y], data.z[y]
            for k in range(split.n_folds):
                f = run.path("preds", "folds", str(y), f"fold{k:02d}.npz")
                if f.exists():
                    continue
                tr, te = split.masks(k)
                va = split.inner_val(tr, k) if self.use_val else None
                p = self.predictor().fit([X], [z], [tr & ~va if va is not None else tr], [va] if va is not None else None)
                pred, w = p.predict(X, te, return_weights=True)
                te_cells = np.flatnonzero(te.ravel())
                extra = {}
                if w is not None:
                    extra["weights"] = (w.reshape(12, -1, w.shape[-1])[:, te_cells] if w.ndim == 4 else w).astype(np.float32)
                np.savez_compressed(f, pred=pred.reshape(12, -1)[:, te_cells].astype(np.float32), cells=te_cells, **extra)
                run.log(f"  {y} fold {k:02d}: train {int(tr.sum())} cells, test {int(te.sum())}")
                if smoke:
                    run.log("smoke run: one fold only, no scoring")
                    return {"smoke": True}

            full = np.full((12, self.grid.n), np.nan, np.float32)
            wfull = None
            for k in range(split.n_folds):
                a = np.load(run.path("preds", "folds", str(y), f"fold{k:02d}.npz"))
                full[:, a["cells"]] = a["pred"]
                if "weights" in a and a["weights"].ndim == 3:            # gated: weights per cell-month
                    if wfull is None:
                        wfull = np.full((12, self.grid.n, a["weights"].shape[-1]), np.nan, np.float32)
                    wfull[:, a["cells"]] = a["weights"]
            oof = full[:, cells]
            np.savez_compressed(run.path("preds", f"cv_{y}.npz"), pred=oof, cells=cells)
            if wfull is not None:
                np.savez_compressed(run.path("preds", f"weights_{y}.npz"), weights=wfull[:, cells], cells=cells,
                                    branches=np.array(self.branches))
            s = monthly_scores(z.reshape(12, -1)[:, cells], data.hot[y].reshape(12, -1)[:, cells], oof, names)
            metrics["cv"][str(y)] = {n: float(np.nanmean(v)) for n, v in s.items()}
            run.log(f"  {y} CV: " + ", ".join(f"{n}={v:.3f}" for n, v in metrics["cv"][str(y)].items()))

        # ------------------------------------------------------------ temporal transfer
        if cfg["split"]["transfer"] and len(data.years) > 1:
            for a, b in permutations(data.years, 2):
                va = split.inner_val(dom, -1) if self.use_val else None
                p = self.predictor().fit([data.X[a]], [data.z[a]], [dom & ~va if va is not None else dom],
                                         [va] if va is not None else None)
                pred = p.predict(data.X[b], dom).reshape(12, -1)[:, cells]
                np.savez_compressed(run.path("preds", f"transfer_{a}to{b}.npz"), pred=pred, cells=cells)
                s = monthly_scores(data.z[b].reshape(12, -1)[:, cells], data.hot[b].reshape(12, -1)[:, cells], pred, names)
                metrics["transfer"][f"{a}to{b}"] = {n: float(np.nanmean(v)) for n, v in s.items()}
                run.log(f"  transfer {a}->{b}: " + ", ".join(f"{n}={v:.3f}" for n, v in metrics["transfer"][f"{a}to{b}"].items()))

        # ------------------------------------------------------------ final model on all label cells
        if cfg["train"]["save_final"] and self.neural:
            va = split.inner_val(dom, -1) if self.use_val else None
            n = len(data.years)
            p = self.predictor().fit([data.X[y] for y in data.years], [data.z[y] for y in data.years],
                                     [dom & ~va if va is not None else dom] * n, [va] * n if va is not None else None)
            ck = p.checkpoint()
            ck.update(features=data.features, data_hash=data.hash, experiment=cfg["experiment"])
            torch.save(ck, run.path("models", "final.pt"))
            run.log("  final model saved: models/final.pt")

        run.save_json("metrics.json", metrics)
        RunMaps(run, data, self.grid).draw()
        primary = cfg["eval"]["primary"]
        summary = {f"{primary}_{y}": round(v[primary], 4) for y, v in metrics["cv"].items()}
        summary.update({f"{primary}_{k}": round(v[primary], 4) for k, v in metrics["transfer"].items()})
        return summary

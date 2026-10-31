"""Is the trained GNN usable for decisions? (output of RQ2, input of RQ3)

The final model of a run (models/final.pt, trained on all label cells of all years) maps an emission field plus the
fixed city context to the expected spatial pattern of the combustion signal (z of satellite NO2). Scenarios scale the
inventory CO2 features (inventory_prefix) cell by cell; a cell inside several districts takes the area-weighted mix.
z is a within-month relative pattern, so only RELATIVE indices are used:
  HCI  hotspot contrast index = mean predicted z over the baseline hotspot cells - mean over the domain
  PEX  population exposure    = share of the domain population in cells above the baseline hotspot threshold
Checks (settings: analysis.response):
  1. sign      cut CO2 in ONE district at a time: z inside must fall relative to the rest (contrast < 0)
  2. locality  how fast the effect decays with distance from the district (km at half of the inside effect)
  3. targeted  the same total CO2 cut, concentrated in the districts with the highest baseline z vs uniform -> HCI
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import distance_transform_edt

from ..config import ROOT, Config
from ..data.dataset import CellMonthDataset
from ..data.graphs import GridGraph, ODOperator
from ..data.grid import Grid
from ..models.backbones import BACKBONES
from ..tracking import RunRegistry
from ..training.trainer import NeuralPredictor


class ResponseChecks:
    def __init__(self, base_cfg: dict, experiment: str):
        self.a = base_cfg["analysis"]["response"]
        rows = RunRegistry(base_cfg).find(experiment).sort_values("finished")
        if rows.empty:
            raise ValueError(f"no finished run of {experiment}")
        self.folder = ROOT / rows.iloc[-1].path
        self.run_id = rows.iloc[-1].run_id
        self.cfg = yaml.safe_load((self.folder / "config.yaml").read_text(encoding="utf-8"))
        if self.cfg["data"].get("spatial_lags"):
            raise ValueError("response checks need a dataset without spatial lags (data.spatial_lags: [])")
        if not (self.folder / "models" / "final.pt").exists():
            raise FileNotFoundError(f"{self.folder.name} has no models/final.pt (train.save_final)")

    def _scenario(self, X: np.ndarray, mult: np.ndarray) -> np.ndarray:
        """Scale the inventory features by a (H, W) multiplier (in physical units, then back to the feature scale)."""
        Xs = X.copy()
        for j, logged in zip(self.inv, self.logged):
            Xs[..., j] = np.log1p(np.expm1(X[..., j]) * mult) if logged else X[..., j] * mult
        return Xs

    def _indices(self, Z: np.ndarray) -> dict:
        hci = [Z[t][self.hot[t]].mean() - Z[t][self.dom].mean() for t in range(Z.shape[0])]
        pex = [self.pop[self.dom & (Z[t] >= self.thr[t])].sum() / self.pop[self.dom].sum() for t in range(Z.shape[0])]
        return {"HCI": float(np.mean(hci)), "PEX": float(np.mean(pex))}

    def run(self, run) -> dict:
        a, cfg = self.a, self.cfg
        data = CellMonthDataset.build(cfg, log=run.log)
        grid = Grid(cfg)
        b = cfg["model"]["backbone"]
        graph = GridGraph(grid, b["scales"]) if BACKBONES.get(b["type"]).needs_graph else None
        if graph is not None and b["od"]["enabled"]:
            graph.od = ODOperator.from_config(cfg, grid.n)
        model = NeuralPredictor.load(cfg, data, graph, self.folder / "models" / "final.pt", run.log)
        y = int(a["year"])
        X, self.dom = data.X[y], data.label_domain
        logged = set(cfg["data"]["log1p"])
        self.inv = [j for j, n in enumerate(data.base_of) if n.startswith(a["inventory_prefix"])]
        self.logged = [data.base_of[j] in logged for j in self.inv]
        pj = data.features.index(a["population_feature"])
        self.pop = np.expm1(X[0][..., pj]) if a["population_feature"] in logged else X[0][..., pj]

        base = model.predict(X)
        q = float(cfg["data"]["target"]["hotspot_quantile"])
        self.thr = [np.quantile(base[t][self.dom], q) for t in range(12)]
        self.hot = [self.dom & (base[t] >= self.thr[t]) for t in range(12)]
        B = self._indices(base)

        # cell x district shares; multiplier of a district-level factor vector f: 1 + F (f - 1)
        bd = cfg["data"]["boundaries"]
        fc = bd["fractions_columns"]
        fr = pd.read_csv(Config.path(cfg, "raw") / bd["district_fractions_file"], encoding="utf-8-sig")
        ids = sorted(fr[fc["district"]].unique())
        F = np.zeros((grid.n, len(ids)))
        F[fr[fc["cell"]].values, fr[fc["district"]].map({d: i for i, d in enumerate(ids)}).values] = fr[fc["frac"]].values
        inside = {d: (F[:, i] > float(a["inside_min_fraction"])).reshape(grid.H, grid.W) & self.dom for i, d in enumerate(ids)}
        edges = np.arange(0, float(a["max_km"]), float(a["ring_km"]))

        # 1-2. one district at a time
        rows, decay = [], []
        for i, d in enumerate(ids):
            if not inside[d].any():
                continue
            f = np.ones(len(ids))
            f[i] = float(a["district_factor"])
            Z = model.predict(self._scenario(X, (1 + F @ (f - 1)).reshape(grid.H, grid.W)))
            dz = (Z - base).mean(0)
            dkm = distance_transform_edt(~inside[d], sampling=(grid.km_y, grid.km_x))
            prof = np.array([dz[self.dom & (dkm >= e) & (dkm < e + float(a["ring_km"]))].mean() for e in edges])
            half = next((float(edges[k]) for k, v in enumerate(prof) if v > prof[0] / 2), np.nan) if prof[0] < 0 else np.nan
            I = self._indices(Z)
            contrast = float(dz[inside[d]].mean() - dz[self.dom & ~inside[d]].mean())
            rows.append(dict(district=d, dz_inside=float(dz[inside[d]].mean()), contrast=contrast, right_sign=contrast < 0,
                             half_effect_km=half, dHCI=I["HCI"] - B["HCI"], dPEX=I["PEX"] - B["PEX"]))
            decay.append(prof)
        C = pd.DataFrame(rows)

        # 3. uniform vs targeted cut of the same total CO2
        co2 = sum(np.expm1(X[..., j]) if lg else X[..., j] for j, lg in zip(self.inv, self.logged)).sum(0)
        cut = float(a["total_cut"])
        Iu = self._indices(model.predict(self._scenario(X, np.full((grid.H, grid.W), 1 - cut))))
        zd = pd.Series({d: base.mean(0)[inside[d]].mean() for d in ids if inside[d].any()})
        top = list(zd.sort_values().index[-int(a["targeted_districts"]):])
        mask = np.any([inside[d] for d in top], 0)
        local_cut = float(min(cut * co2[self.dom].sum() / co2[mask].sum(), float(a["max_local_cut"])))
        It = self._indices(model.predict(self._scenario(X, np.where(mask, 1 - local_cut, 1.0))))

        summary = dict(model_run=self.run_id, year=y, baseline_HCI=B["HCI"], baseline_PEX=B["PEX"],
                       sign_test_share_right=float(C.right_sign.mean()),
                       half_effect_km_median=float(np.nanmedian(C.half_effect_km)),
                       uniform_dHCI=Iu["HCI"] - B["HCI"], uniform_dPEX=Iu["PEX"] - B["PEX"],
                       targeted_districts=top, targeted_local_cut=local_cut,
                       targeted_dHCI=It["HCI"] - B["HCI"], targeted_dPEX=It["PEX"] - B["PEX"])
        run.save_table("response_checks.csv", C)
        run.save_json("response_summary.json", summary)
        dec = pd.DataFrame(np.array(decay), columns=[f"{e:g}km" for e in edges])
        dec.insert(0, "district", C.district.values)
        run.save_table("response_decay.csv", dec)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        for p in decay:
            ax[0].plot(edges, p, color="#1f5f99", alpha=0.3)
        ax[0].plot(edges, np.nanmean(decay, 0), color="k", lw=2, label="mean of districts")
        ax[0].axhline(0, color="grey", lw=0.5)
        ax[0].set_xlabel("distance from the district (km)")
        ax[0].set_ylabel("change of predicted z")
        ax[0].legend(frameon=False)
        ax[1].bar([f"uniform -{cut:.0%}", f"targeted ({len(top)} districts)"], [summary["uniform_dHCI"], summary["targeted_dHCI"]],
                  color=["#999999", "#1f5f99"])
        ax[1].set_ylabel("change of hotspot contrast index")
        fig.tight_layout()
        fig.savefig(run.path("figures", "response.png"), dpi=160)
        plt.close(fig)
        run.log("response checks: " + ", ".join(f"{k}={v}" for k, v in summary.items()))
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()}

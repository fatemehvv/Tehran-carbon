"""District-level link between RQ2 and RQ1.

The out-of-fold maps (every cell predicted by a model that never saw its block; mean of 12 months) are averaged over
the 22 districts with the cell-area shares, ranked into equal-count classes (L, M, H), and compared with the RQ1 v2
Dempster-Shafer classes and the inventory-only classes (linear-weighted Cohen's kappa). The observed NO2 map is
treated the same way. Note: satellite NO2 is one of the six DS evidence sources, so agreement with DS is not fully
independent of the target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import cohen_kappa_score

from ..config import ROOT, Config
from ..data.dataset import CellMonthDataset
from ..data.grid import Grid
from ..tracking import RunRegistry


class DistrictComparison:
    def __init__(self, base_cfg: dict, experiments: list[str]):
        self.cfg, self.experiments = base_cfg, list(experiments)
        self.registry = RunRegistry(base_cfg)

    def run(self, run) -> dict:
        a, b = self.cfg["analysis"]["districts"], self.cfg["data"]["boundaries"]
        fc, raw = b["fractions_columns"], Config.path(self.cfg, "raw")
        sources = {}
        for exp in self.experiments:
            rows = self.registry.find(exp).sort_values("finished")
            if rows.empty:
                raise ValueError(f"no finished run of {exp}")
            folder = ROOT / rows.iloc[-1].path
            sources[exp] = (folder, yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8")))
        first_cfg = next(iter(sources.values()))[1]
        data = CellMonthDataset.build(first_cfg, log=run.log)
        n = Grid(first_cfg).n
        fr = pd.read_csv(raw / b["district_fractions_file"], encoding="utf-8-sig")
        fr = fr[fr[fc["cell"]].isin(data.cells)]
        labels = list(a["labels"])
        code = {lab: i for i, lab in enumerate(labels)}

        tables, kappas = [], []
        for y in data.years:
            ds = pd.read_csv(raw / a["ds_file"].format(year=y), encoding="utf-8-sig").set_index(a["id_column"])
            maps = {"observed": data.z[y].reshape(12, -1)[:, data.cells].mean(0)}
            for exp, (folder, _) in sources.items():
                p = np.load(folder / "preds" / f"cv_{y}.npz")
                if not np.array_equal(p["cells"], data.cells):
                    raise ValueError(f"{folder.name} has a different label domain")
                maps[exp] = p["pred"].mean(0)
            out = pd.DataFrame(index=ds.index)
            for name, v in maps.items():
                g = np.full(n, np.nan)
                g[data.cells] = v
                w = fr.assign(v=g[fr[fc["cell"]].values] * fr[fc["frac"]], w=fr[fc["frac"]]).groupby(fc["district"])[["v", "w"]].sum()
                out[name] = w.v / w.w
                out[f"{name}_class"] = pd.qcut(out[name].rank(method="first"), len(labels), labels=labels).astype(str)
            for ref, column in a["references"].items():
                out[f"{ref}_class"] = ds[column]
            out["year"] = y
            tables.append(out.reset_index())
            for name in maps:
                for ref in list(a["references"]) + (["observed"] if name != "observed" else []):
                    x, r = out[f"{name}_class"].map(code), out[f"{ref}_class"].map(code)
                    ok = x.notna() & r.notna()
                    kappas.append(dict(year=y, map=name, reference=ref, n=int(ok.sum()),
                                       kappa_linear=cohen_kappa_score(x[ok], r[ok], weights="linear"),
                                       exact_agreement=float((x[ok] == r[ok]).mean())))
        K = pd.DataFrame(kappas)
        run.save_table("district_classes.csv", pd.concat(tables))
        run.save_table("district_kappa.csv", K)
        run.log("district agreement (linear kappa):\n" + K.round(2).to_string(index=False))
        return {f"kappa_{r.map}_vs_{r.reference}_{r.year}": round(float(r.kappa_linear), 3) for r in K.itertuples()}

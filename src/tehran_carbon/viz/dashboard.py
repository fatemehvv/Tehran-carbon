"""Data of the decision-support dashboard (python main.py dashboard -> docs/dashboard/data.js).

The page docs/dashboard/index.html is static and opens offline (double-click); it reads window.DASHBOARD from data.js.
Every number comes from finished runs; nothing is trained:
  policies   every static policy of the daily grid (rq3_daily_robust_v1) and the three RL agents, on both transfers:
             return, CVaR10, CO2, travel time, exposure, mean restriction per district, budget used per day
  districts  the 22 districts on the analysis grid (cropped), for the restriction map
  severity   the day's NO2 severity of each test year
  hotspots   observed and predicted mean z of the held-out year 1404 (c1404_sage5_v1)
  inventory  L1 sector totals
"""
from __future__ import annotations

import json
import re

import numpy as np
import yaml

from ..config import ROOT
from .figures import ThesisFigures

TRANSFERS = [("1402", "1403"), ("1403", "1402")]
RL_RUNS = [("rq3_daily_robust_v1", "rl_g099", 0.99), ("rq3_daily_g0999_v1", "rl_g0999", 0.999),
           ("rq3_daily_g1_v1", "rl_g1", 1.0)]


def _params(name: str) -> dict:
    """'hotkm r=0.2 b=0 k=3 rank=trip_km' -> {'r': 0.2, 'k': 3}; the family is the first word."""
    p = dict(re.findall(r"(\w+)=([-\w.]+)", name))
    out = {}
    if "r" in p:
        out["r"] = float(p["r"])
    if "k" in p:
        out["k"] = int(p["k"])
    if "below" in p:
        out["below"] = float(p["below"])
    return out


def _r(x, n=3):
    return [None if not np.isfinite(v) else round(float(v), n) for v in np.asarray(x, float).ravel()]


class DashboardData:
    def __init__(self, cfg: dict, log=print):
        self.f, self.log = ThesisFigures(cfg, log), log

    def policies(self, rmax: float) -> list[dict]:
        out = []
        base = self.f.folder("rq3_daily_robust_v1")
        tuned = json.loads((base / "tuned.json").read_text())
        for a, b in TRANSFERS:
            z = np.load(base / "preds" / f"returns_{a}to{b}.npz")
            entries = [(str(n), str(fam), z["returns"][i], z["components"][i], z["actions"][i])
                       for i, (n, fam) in enumerate(zip(z["names"], z["families"])) if fam != "rl"]
            for exp, fam, gamma in RL_RUNS:
                if self.f.runs(exp).empty:
                    continue
                zz = np.load(self.f.folder(exp) / "preds" / f"returns_{a}to{b}.npz")
                rl = zz["families"] == "rl"
                entries.append((f"RL gamma={gamma}", fam, zz["returns"][rl].mean(0), zz["components"][rl].mean(0),
                                zz["actions"][rl].mean(0)))
            T = 366 if b == "1403" else 365
            for name, fam, R, C, A in entries:
                k = max(1, int(np.ceil(0.1 * len(R))))
                c = C.mean(0)
                r = A[:T, :A.shape[1] // 2]
                out.append(dict(transfer=f"{a}-{b}", family=fam, name=name, params=_params(name),
                                tuned=name in tuned[a].values() or fam.startswith("rl"),
                                ret=round(float(R.mean()), 3), cvar=round(float(np.sort(R)[:k].mean()), 3),
                                co2_kt=round(float(c[0] / 1e3), 1), hours_m=round(float(c[1] / 1e6), 2),
                                exposure_pct=round(float(c[3] / T * 100), 2),
                                district=_r(r.mean(0), 4), budget_day=_r(r.mean(1) / rmax, 4)))
        return out

    def run(self) -> dict:
        from ..data.dataset import CellMonthDataset
        from ..data.graphs import ODOperator
        from ..data.grid import Grid
        from ..experiments.policy_search import PolicySearch
        from ..experiments.response import FrozenResponseModel
        from ..rl.city import CityTransport

        dcfg = yaml.safe_load((self.f.folder("rq3_daily_v1") / "config.yaml").read_text(encoding="utf-8"))
        rmax = float(dcfg["rl"]["levers"]["restriction_max"])
        ps = PolicySearch(dcfg, None)
        ps.frozen = FrozenResponseModel(dcfg, dcfg["rl"]["response_experiment"], lambda *a: None)
        severity = {y: _r(ps.daily_inputs(int(y))[1], 3) for y in ("1402", "1403")}
        months = {y: CityTransport.daily_months(int(y)).tolist() for y in ("1402", "1403")}

        rcfg = ps.frozen.cfg
        grid = Grid(rcfg)
        F, ids = ODOperator.district_fractions(rcfg, grid.n)
        owner = np.where(F.sum(1) > 0, F.argmax(1), -1).reshape(grid.H, grid.W)
        rr, cc = np.nonzero(owner >= 0)
        owner = owner[rr.min():rr.max() + 1, cc.min():cc.max() + 1]

        hf = self.f.folder("c1404_sage5_v1")
        hcfg = yaml.safe_load((hf / "config.yaml").read_text(encoding="utf-8"))
        data, hgrid = CellMonthDataset.build(hcfg, log=lambda *a: None), Grid(hcfg)
        p = np.load(hf / "preds" / "cv_1404.npz")
        obs, pred = np.full(hgrid.n, np.nan), np.full(hgrid.n, np.nan)
        obs[p["cells"]] = np.nanmean(data.z[1404].reshape(data.z[1404].shape[0], -1)[:, p["cells"]], 0)
        pred[p["cells"]] = np.nanmean(p["pred"], 0)
        obs, pred = obs.reshape(hgrid.H, hgrid.W), pred.reshape(hgrid.H, hgrid.W)
        hr, hc = np.nonzero(np.isfinite(obs))
        sl = (slice(hr.min(), hr.max() + 1), slice(hc.min(), hc.max() + 1))

        inv = []
        f = self.f.folder("inv_l1_v1")
        import pandas as pd
        for y in ("1402", "1403"):
            s = pd.read_csv(f / f"L1_summary_{y}.csv", encoding="utf-8-sig").set_index("group")
            for g, key in [("Buildings (CORE)", "buildings"), ("Road transport (CORE)", "road"),
                           ("Industry gas incl. refinery fuel (CORE)", "industry"), ("Power plants inside CORE", "power")]:
                inv.append(dict(year=y, sector=key, mt=round(float(s.MtCO2_central[g]), 2),
                                p05=round(float(s.MtCO2_p05[g]), 2), p95=round(float(s.MtCO2_p95[g]), 2)))

        out = dict(
            generated=pd.Timestamp.now().strftime("%Y-%m-%d"),
            budget_days=float(dcfg["rl"]["restriction_budget_days"]), rmax=rmax,
            districts=dict(ids=ids, H=int(owner.shape[0]), W=int(owner.shape[1]), owner=owner.ravel().tolist()),
            policies=self.policies(rmax), severity=severity, months=months,
            hotspots=dict(H=int(obs[sl].shape[0]), W=int(obs[sl].shape[1]), obs=_r(obs[sl], 2), pred=_r(pred[sl], 2)),
            inventory=inv)
        path = ROOT / "docs" / "dashboard" / "data.js"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("window.DASHBOARD = " + json.dumps(out, ensure_ascii=False, separators=(",", ":")) + ";\n",
                        encoding="utf-8")
        self.log(f"  {path.relative_to(ROOT)}: {len(out['policies'])} policies, {path.stat().st_size / 1e3:.0f} kB")
        return {"policies": len(out["policies"])}

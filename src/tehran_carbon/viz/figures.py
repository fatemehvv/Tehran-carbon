"""Thesis figures and tables from the finished runs (python main.py figures). Nothing is trained.

Output: docs/figures/figNN_*.png (300 dpi, A4 text width) and docs/tables/tNN_*.csv. Every figure reads the latest
finished run of the experiments it names; a figure whose runs are missing is skipped with a message.
Colours: the validated categorical order (blue, orange, aqua, yellow, ...), one sequential blue ramp, a blue-grey-red
diverging map; text stays in grey ink, never in a series colour.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..config import ROOT
from ..tracking import RunRegistry

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV = ["#104281", "#2a78d6", "#f0efec", "#e34948", "#a32b2a"]
WIDTH = 6.3                                         # inches, A4 text width
YEARS = ["1402", "1403"]
POLICY_LABEL = {"none": "No action", "uniform": "Uniform", "winter": "Winter only", "hotspot": "Hotspot (by NO2)",
                "calm": "Calm days", "hotkm": "Hotspot (by car-km)"}
POLICY_COLOR = {"RL (gamma 0.99)": SERIES[0], "RL (gamma 0.999)": SERIES[2], "RL (gamma 1)": SERIES[6],
                "Hotspot (by car-km)": SERIES[1], "Hotspot (by NO2)": SERIES[3], "Winter only": MUTED}


def _emphasis(policy: str) -> str:
    """Bars and dots: the agents in blue, the car-km hotspot rule in orange, every other static rule in grey."""
    return SERIES[0] if policy.startswith("RL") else SERIES[1] if "car-km" in policy else MUTED


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "legend.frameon": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 2, "figure.dpi": 100,
        "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white"})
    return plt


class ThesisFigures:
    def __init__(self, cfg: dict, log=print):
        self.cfg, self.log, self.reg = cfg, log, RunRegistry(cfg)
        self.fig_dir, self.tab_dir = ROOT / "docs" / "figures", ROOT / "docs" / "tables"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.tab_dir.mkdir(parents=True, exist_ok=True)
        self.plt = _style()

    # ------------------------------------------------------------------ helpers
    def runs(self, experiment: str) -> pd.DataFrame:
        return self.reg.find(experiment).sort_values("finished")

    def folder(self, experiment: str, variant: str | None = None) -> Path:
        rows = self.runs(experiment)
        if variant is not None:
            rows = rows[rows.variant == variant]
        if rows.empty:
            raise FileNotFoundError(f"no finished run of {experiment} {variant or ''}")
        return ROOT / rows.iloc[-1].path

    def metrics(self, experiment: str, variant: str | None = None) -> dict:
        return json.loads((self.folder(experiment, variant) / "metrics.json").read_text(encoding="utf-8"))

    def pr_auc(self, experiment: str, part: str, key: str, variant: str | None = None) -> float:
        try:
            return float(self.metrics(experiment, variant)[part][key]["pr_auc"])
        except (FileNotFoundError, KeyError):
            return np.nan

    def decision(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.folder(f"hypothesis_{name}") / f"{name}_decision.csv", encoding="utf-8-sig")

    def save(self, fig, name: str) -> None:
        fig.savefig(self.fig_dir / f"{name}.png")
        self.plt.close(fig)
        self.log(f"  {name}.png")

    def table(self, df: pd.DataFrame, name: str) -> None:
        df.to_csv(self.tab_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
        self.log(f"  {name}.csv")

    def run_all(self) -> dict:
        done, skipped = [], []
        for name in sorted(n for n in dir(self) if n[:3] == "fig" and n[3:5].isdigit()):
            try:
                getattr(self, name)()
                done.append(name)
            except FileNotFoundError as e:
                self.log(f"  skipped {name}: {e}")
                skipped.append(name)
        return {"done": done, "skipped": skipped}

    # ------------------------------------------------------------------ inventory
    def fig01_inventory(self):
        f = self.folder("inv_l1_v1")
        groups = {"Buildings (CORE)": "Buildings", "Road transport (CORE)": "Road transport",
                  "Industry gas incl. refinery fuel (CORE)": "Industry", "Power plants inside CORE": "Power plants"}
        rows = []
        for y in YEARS:
            s = pd.read_csv(f / f"L1_summary_{y}.csv", encoding="utf-8-sig").set_index("group")
            for g, label in list(groups.items()) + [("Scope 1 total CORE", "Scope 1 total")]:
                rows.append(dict(year=y, sector=label, MtCO2=s.MtCO2_central[g], p05=s.MtCO2_p05[g], p95=s.MtCO2_p95[g]))
        t = pd.DataFrame(rows)
        self.table(t.round(2), "t01_inventory")
        fig, ax = self.plt.subplots(figsize=(WIDTH, 2.4))
        labels = list(groups.values())[::-1]
        y0 = np.arange(len(labels))
        for i, y in enumerate(YEARS):
            d = t[t.year == y].set_index("sector").loc[labels]
            pos = y0 + (0.2 if i == 0 else -0.2)
            ax.barh(pos, d.MtCO2, height=0.36, color=SERIES[i], label=y, edgecolor="white", linewidth=1)
            ax.errorbar(d.MtCO2, pos, xerr=[d.MtCO2 - d.p05, d.p95 - d.MtCO2], fmt="none", ecolor=INK2, lw=0.8, capsize=2)
            for p, v, hi in zip(pos, d.MtCO2, d.p95):
                ax.text(hi + 0.3, p, f"{v:.1f}", va="center", fontsize=7.5, color=INK2)
        ax.set_yticks(y0, labels)
        ax.set_xlabel("Mt CO2 per year (central; bars: 5-95 % Monte Carlo)")
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right")
        ax.set_title("Scope 1 CO2 of Tehran (CORE) by sector, L1 inventory", loc="left")
        self.save(fig, "fig01_inventory")

    # ------------------------------------------------------------------ RQ1
    def fig02_rq1_ablation(self):
        combos = ["traffic", "energy", "pollution", "traffic+energy", "traffic+pollution", "energy+pollution",
                  "traffic+energy+pollution"]
        fusions = ["early", "late", "gated"]
        M = np.full((2, len(combos), len(fusions)), np.nan)
        rows = []
        for i, c in enumerate(combos):
            for j, fu in enumerate(fusions):
                for k, y in enumerate(YEARS):
                    M[k, i, j] = self.pr_auc("rq1_ablation_v2", "cv", y, f"modalities={c}|fusion={fu}")
                rows.append(dict(domains=c, fusion=fu, **{f"pr_auc_{y}": round(M[k, i, j], 3) for k, y in enumerate(YEARS)}))
        self.table(pd.DataFrame(rows), "t02_rq1_ablation")
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("seq", SEQ)
        fig, axes = self.plt.subplots(1, 2, figsize=(WIDTH, 3.0), sharey=True)
        names = [c.replace("traffic", "T").replace("energy", "E").replace("pollution", "P").replace("+", " + ")
                 for c in combos]
        for k, ax in enumerate(axes):
            im = ax.imshow(M[k], cmap=cmap, vmin=0.5, vmax=0.8, aspect="auto")
            for i in range(len(combos)):
                for j in range(len(fusions)):
                    v = M[k, i, j]
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7.5,
                            color="white" if v > 0.7 else INK)
            ax.set_xticks(range(3), fusions)
            ax.set_yticks(range(len(combos)), names)
            ax.grid(False)
            ax.set_title(f"{YEARS[k]} (spatial-block CV)", loc="left")
        cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
        cb.set_label("PR-AUC (random = 0.10)")
        fig.suptitle("RQ1: PR-AUC by input domains and fusion  (T traffic, E energy, P pollution)", x=0.02, ha="left",
                     fontsize=9, y=1.04)
        self.save(fig, "fig02_rq1_ablation")

    # ------------------------------------------------------------------ RQ2
    def fig03_rq2_models(self):
        models = [  # label, development experiment (1402 / 1403), confirmatory experiment (1404)
            ("GNN (SAGE; 3 layers, 1404: 5)", "rq2_sage_v2", "c1404_sage5_v1"),
            ("SLX-RF", "rq2_base_slx_rf_v1", "c1404_base_slx_rf_v1"),
            ("SLX-GBM", "rq2_base_slx_gbm_v1", "c1404_base_slx_gbm_v1"),
            ("SLX-OLS", "rq2_base_slx_ols_v1", "c1404_base_slx_ols_v1"),
            ("MLP (no graph)", "rq2_mlp_v2", None),
            ("GBM (own cell)", "rq2_base_gbm_v1", "c1404_base_gbm_v1"),
            ("OLS (own cell)", "rq2_base_ols_v1", "c1404_base_ols_v1"),
            ("Physical operator", "rq2_base_forward_v1", "c1404_base_forward_v1")]
        rows = []
        for label, dev, conf in models:
            r = dict(model=label, **{y: self.pr_auc(dev, "cv", y) for y in YEARS})
            r["1404"] = self.pr_auc(conf, "cv", "1404") if conf else np.nan
            rows.append(r)
        t = pd.DataFrame(rows)
        self.table(t.round(3), "t03_rq2_models")
        fig, ax = self.plt.subplots(figsize=(WIDTH, 3.0))
        y0 = np.arange(len(t))[::-1]
        ax.axvline(0.10, color=MUTED, lw=1, ls="--")
        ax.text(0.105, y0[0] + 0.2, "random", fontsize=7, color=MUTED)
        for i, (col, name) in enumerate([("1402", "1402 (development)"), ("1403", "1403 (development)"),
                                         ("1404", "1404 (confirmatory)")]):
            ax.plot(t[col], y0, "o", ms=6, color=SERIES[i], label=name, mec="white", mew=1)
        ax.set_yticks(y0, t.model)
        ax.set_xlim(0, 0.9)
        ax.set_xlabel("PR-AUC of the hotspot map (top 10 % NO2 cells; spatial-block CV)")
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right")
        ax.set_title("RQ2: GNN vs traditional models", loc="left")
        self.save(fig, "fig03_rq2_models")

    def fig04_receptive_field(self):
        layers = {1: None, 2: None, 3: "main", 4: None}
        rows = []
        for n in layers:
            if n == 3:
                v = {y: self.pr_auc("rq2_sage_v2", "cv", y) for y in YEARS}
            else:
                v = {y: self.pr_auc("rq2_receptive_field_v2", "cv", y, f"backbone.layers={n}") for y in YEARS}
            rows.append(dict(layers=n, **v))
        t = pd.DataFrame(rows)
        if t[YEARS].isna().all().all():
            raise FileNotFoundError("no receptive-field runs")
        self.table(t.round(3), "t04_receptive_field")
        slx = {y: self.pr_auc("rq2_base_slx_gbm_v1", "cv", y) for y in YEARS}
        fig, ax = self.plt.subplots(figsize=(WIDTH * 0.62, 2.6))
        for i, y in enumerate(YEARS):
            ax.plot(t.layers, t[y], "-o", color=SERIES[i], ms=6, mec="white", mew=1, label=f"GNN {y}")
            ax.axhline(slx[y], color=SERIES[i], lw=1, ls=":")
        ax.text(4.05, np.nanmean(list(slx.values())), "SLX-GBM\n(dotted)", fontsize=7, color=INK2, va="center")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("GNN layers (receptive field)")
        ax.set_ylabel("PR-AUC")
        ax.legend(loc="lower right")
        ax.set_title("RQ2: a wider receptive field helps", loc="left")
        self.save(fig, "fig04_receptive_field")

    def fig05_hypotheses(self):
        """Forest plot of the PR-AUC hypotheses (RQ1, RQ2) and the table of every verdict."""
        spec = [("H1", "RQ1 fusion vs best single domain"), ("H1c", "RQ1 fusion, confirmatory 1404"),
                ("H2", "RQ2 GNN vs best traditional"), ("H2c", "RQ2 GNN, confirmatory 1404"),
                ("H2c_robust", "RQ2 GNN vs best of all, 1404 (not pre-reg.)"), ("H2b", "RQ2b daily GNN with wind")]
        rows = []
        for h, label in spec:
            try:
                d = self.decision(h)
            except FileNotFoundError:
                continue
            verdict = json.loads((self.folder(f"hypothesis_{h}") / f"{h}_decision.json").read_text())["verdict"]
            for r in d.itertuples():
                rows.append(dict(hypothesis=h, label=label, test=r.test, delta=r.delta, ci_low=r.ci_low,
                                 ci_high=r.ci_high, verdict=verdict))
        t = pd.DataFrame(rows)
        for h in ["H3", "H4", "H3d", "H3d_robust", "H3d_g0999", "H3d_g1"]:
            try:
                d = self.decision(h)
            except FileNotFoundError:
                continue
            verdict = json.loads((self.folder(f"hypothesis_{h}") / f"{h}_decision.json").read_text())["verdict"]
            for r in d.itertuples():
                rows.append(dict(hypothesis=h, label="RQ3 policy", test=r.test, delta=r.delta, ci_low=r.ci_low,
                                 ci_high=r.ci_high, verdict=verdict, reference=getattr(r, "reference", "")))
        self.table(pd.DataFrame(rows).round(4), "t05_hypotheses")
        t = t[t.ci_low.notna()]                       # the tests with a bootstrap interval (CV years, 1404)
        fig, ax = self.plt.subplots(figsize=(WIDTH, 3.2))
        y0 = np.arange(len(t))[::-1]
        colors = {"supported": SERIES[2], "partial": SERIES[3], "not supported": MUTED}
        for yy, r in zip(y0, t.itertuples()):
            c = colors[r.verdict]
            ax.plot([r.ci_low, r.ci_high], [yy, yy], color=c, lw=2, solid_capstyle="round")
            ax.plot(r.delta, yy, "o", color=c, ms=7, mec="white", mew=1)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_yticks(y0, [f"{r.hypothesis}  {r.test.replace('cv_', '')}" for r in t.itertuples()])
        ax.set_xlabel("Difference in PR-AUC, candidate - reference (95 % block-bootstrap interval)")
        ax.grid(axis="y", visible=False)
        from matplotlib.lines import Line2D
        ax.legend([Line2D([], [], color=c, marker="o", lw=2) for c in colors.values()], list(colors),
                  loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, title="verdict", title_fontsize=8)
        ax.set_title("RQ1 / RQ2 decision rules (positive = hypothesis direction)", loc="left")
        self.save(fig, "fig05_hypotheses")

    def fig06_maps_1404(self):
        """Observed vs predicted mean z on the held-out year 1404 and the agreement of the two hotspot maps."""
        from ..data.dataset import CellMonthDataset
        from ..data.grid import Grid
        f = self.folder("c1404_sage5_v1")
        cfg = yaml.safe_load((f / "config.yaml").read_text(encoding="utf-8"))
        data, grid = CellMonthDataset.build(cfg, log=lambda *a: None), Grid(cfg)
        p = np.load(f / "preds" / "cv_1404.npz")
        obs = np.nanmean(data.z[1404].reshape(data.z[1404].shape[0], -1)[:, p["cells"]], 0)
        pred = np.nanmean(p["pred"], 0)

        def field(v):
            out = np.full(grid.n, np.nan)
            out[p["cells"]] = v
            return out.reshape(grid.H, grid.W)

        ext = [grid.lon0, grid.lon0 + grid.W * grid.res, grid.lat0 - grid.H * grid.res, grid.lat0]
        from matplotlib.colors import LinearSegmentedColormap, ListedColormap
        div = LinearSegmentedColormap.from_list("div", DIV)
        lim = float(np.nanpercentile(np.abs(obs), 98))
        q = 0.9
        ho, hp = obs >= np.nanquantile(obs, q), pred >= np.nanquantile(pred, q)
        agree = np.where(ho & hp, 3, np.where(ho, 2, np.where(hp, 1, 0))).astype(float)
        fig, axes = self.plt.subplots(1, 3, figsize=(WIDTH, 2.6), layout="constrained")
        for ax, v, title in [(axes[0], obs, "Observed NO2 (z, mean 1404)"), (axes[1], pred, "GNN out-of-fold prediction")]:
            im = ax.imshow(field(v), extent=ext, cmap=div, vmin=-lim, vmax=lim)
            ax.set_title(title, loc="left", fontsize=8)
        cb = fig.colorbar(im, ax=axes[:2], shrink=0.75, pad=0.02, location="bottom", aspect=40)
        cb.set_label("z (standardised log NO2)")
        cmap = ListedColormap(["#f0efec", SERIES[1], SERIES[0], SERIES[2]])
        axes[2].imshow(field(agree), extent=ext, cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest")
        axes[2].set_title("Hotspots (top 10 %)", loc="left", fontsize=8)
        from matplotlib.patches import Patch
        axes[2].legend([Patch(color=SERIES[2]), Patch(color=SERIES[0]), Patch(color=SERIES[1])],
                       ["both", "observed only", "predicted only"], loc="upper center", bbox_to_anchor=(0.5, -0.08),
                       fontsize=7, ncol=1)
        for ax in axes:
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        share = (ho & hp).sum() / max(ho.sum(), 1)
        self.log(f"  1404 hotspot cells found by the GNN: {share:.0%}")
        self.save(fig, "fig06_maps_1404")

    # ------------------------------------------------------------------ RQ3 monthly
    def fig07_rq3_lambda(self):
        rows = []
        main = self.metrics("rq3_policy_v1")["mean_return"]
        variants = [(1.0, main)]
        for _, r in self.runs("rq3_lambda_v1").iterrows():
            lam = float(r.variant.split("=")[1])
            variants.append((lam, json.loads((ROOT / r.path / "metrics.json").read_text())["mean_return"]))
        for lam, m in variants:
            for k in ["1402to1403", "1403to1402"]:
                best = max(v for p, v in m[k].items() if p != "rl")
                rows.append(dict(lam=lam, transfer=k, rl=m[k]["rl"], best_static=best, diff=m[k]["rl"] - best))
        t = pd.DataFrame(rows).sort_values(["transfer", "lam"])
        self.table(t.round(4), "t06_rq3_lambda")
        fig, ax = self.plt.subplots(figsize=(WIDTH * 0.62, 2.6))
        for i, (k, g) in enumerate(t.groupby("transfer")):
            ax.plot(g.lam, g["diff"], "-o", color=SERIES[i], ms=6, mec="white", mew=1,
                    label=k.replace("to", " -> "))
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.25, 0.5, 1, 2, 4], ["0.25", "0.5", "1 (main)", "2", "4"])
        ax.set_xlabel("weight of travel time in the reward (lambda)")
        ax.set_ylabel("RL - best static (return)")
        ax.legend(loc="upper left", title="transfer", title_fontsize=8)
        ax.set_title("RQ3 monthly: RL gains when time costs more", loc="left")
        self.save(fig, "fig07_rq3_lambda")

    # ------------------------------------------------------------------ RQ3 daily
    def daily_policies(self, a: str, b: str) -> list[dict]:
        """Tuned static policies (of the robustness run when it exists) and every RL variant on transfer a -> b."""
        base = "rq3_daily_robust_v1" if not self.runs("rq3_daily_robust_v1").empty else "rq3_daily_v1"
        out = []
        f = self.folder(base)
        tuned = json.loads((f / "tuned.json").read_text())[a]
        z = np.load(f / "preds" / f"returns_{a}to{b}.npz")
        names = list(z["names"])
        for fam, name in tuned.items():
            i = names.index(name)
            out.append(dict(policy=POLICY_LABEL.get(fam, fam), detail=name, returns=z["returns"][i],
                            comp=z["components"][i], act=z["actions"][i]))
        rl_runs = [(base, "RL (gamma 0.99)"), ("rq3_daily_g0999_v1", "RL (gamma 0.999)"), ("rq3_daily_g1_v1", "RL (gamma 1)")]
        for exp, label in rl_runs:
            if self.runs(exp).empty:
                continue
            g = self.folder(exp)
            zz = np.load(g / "preds" / f"returns_{a}to{b}.npz")
            rl = zz["families"] == "rl"
            out.append(dict(policy=label, detail=f"PPO, {rl.sum()} seeds", returns=zz["returns"][rl].mean(0),
                            comp=zz["components"][rl].mean(0), act=zz["actions"][rl].mean(0)))
        return out

    def fig08_daily_returns(self):
        rng = np.random.default_rng(0)
        rows = []
        for a, b in [("1402", "1403"), ("1403", "1402")]:
            for p in self.daily_policies(a, b):
                R = p["returns"]
                boot = np.array([R[rng.integers(0, len(R), len(R))].mean() for _ in range(2000)])
                rows.append(dict(transfer=f"{a}to{b}", policy=p["policy"], detail=p["detail"], mean=R.mean(),
                                 ci_low=np.quantile(boot, 0.025), ci_high=np.quantile(boot, 0.975)))
        t = pd.DataFrame(rows)
        t = t[t.policy != "No action"]
        fig, axes = self.plt.subplots(1, 2, figsize=(WIDTH, 2.9), sharey=True)
        order = t[t.transfer == "1402to1403"].sort_values("mean").policy.tolist()
        for ax, (k, g) in zip(axes, t.groupby("transfer", sort=False)):
            g = g.set_index("policy").loc[order]
            y0 = np.arange(len(g))
            ax.barh(y0, g["mean"], color=[_emphasis(p) for p in g.index], height=0.6, edgecolor="white")
            ax.errorbar(g["mean"], y0, xerr=[g["mean"] - g.ci_low, g.ci_high - g["mean"]], fmt="none", ecolor=INK2, lw=0.8)
            for yy, v, hi in zip(y0, g["mean"], g.ci_high):
                ax.text(hi + 0.4, yy, f"{v:.1f}", va="center", fontsize=7.5, color=INK2)
            ax.set_yticks(y0, order)
            ax.grid(axis="y", visible=False)
            ax.set_title(f"train {k[:4]} -> test {k[-4:]}", loc="left")
            ax.set_xlabel("mean episode return (30 episodes)")
        fig.suptitle("RQ3 daily: annual return of each policy under the 30-day restriction budget", x=0.02,
                     ha="left", fontsize=9, y=1.04)
        self.save(fig, "fig08_daily_returns")

    def fig09_daily_timing(self):
        """When in the year each policy spends its restriction budget (test year 1403), with the day's severity."""
        from ..experiments.policy_search import PolicySearch
        from ..experiments.response import FrozenResponseModel
        from ..rl.city import CityTransport
        f = self.folder("rq3_daily_v1")
        cfg = yaml.safe_load((f / "config.yaml").read_text(encoding="utf-8"))
        ps = PolicySearch(cfg, None)
        ps.frozen = FrozenResponseModel(cfg, cfg["rl"]["response_experiment"], lambda *a: None)
        _, sev = ps.daily_inputs(1403)
        rmax = float(cfg["rl"]["levers"]["restriction_max"])
        pols = self.daily_policies("1402", "1403")
        month = CityTransport.daily_months(1403)
        days = np.arange(len(sev))
        fig, (a1, a2) = self.plt.subplots(2, 1, figsize=(WIDTH, 3.6), sharex=True, height_ratios=[1, 2.2])
        a1.fill_between(days, sev, 1, where=sev > 1, color=SERIES[7], alpha=0.35, lw=0)
        a1.plot(days, sev, color=INK2, lw=0.8)
        a1.axhline(1, color=MUTED, lw=0.8)
        a1.set_ylabel("NO2 severity\n(day / year mean)")
        a1.set_title("RQ3 daily, test year 1403: pollution severity and cumulative use of the restriction budget",
                     loc="left")
        show = ["RL (gamma 0.99)", "RL (gamma 0.999)", "RL (gamma 1)", "Hotspot (by car-km)", "Hotspot (by NO2)",
                "Winter only"]
        rows = []
        for name in show:
            p = next((q for q in pols if q["policy"] == name), None)
            if p is None:
                continue
            used = np.cumsum(p["act"][:len(sev), :p["act"].shape[1] // 2].mean(1) / rmax)
            a2.plot(days, used, color=POLICY_COLOR[name], lw=2 if name.startswith("RL") else 1.4,
                    ls="-" if name.startswith("RL") else "--", label=name)
            day = p["act"][:len(sev), :p["act"].shape[1] // 2].mean(1)
            top = sev >= np.quantile(sev, 0.9)
            rows.append(dict(policy=name, budget_days=used[-1],
                             **{f"used_by_month_{m}": used[month == m][-1] for m in (3, 6, 9, 12)},
                             share_on_top10pct_severity_days=day[top].sum() / max(day.sum(), 1e-12),
                             corr_with_severity=np.corrcoef(day, sev)[0, 1] if day.std() > 0 else np.nan))
        self.table(pd.DataFrame(rows).round(3), "t07_daily_timing")
        starts = [np.flatnonzero(month == m)[0] for m in range(1, 13)]
        a2.set_xticks(starts, ["Far", "Ord", "Kho", "Tir", "Mor", "Sha", "Meh", "Aba", "Aza", "Dey", "Bah", "Esf"])
        a2.set_ylabel("budget used (days)")
        a2.set_ylim(0, 32)
        a2.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=7.5)
        for ax in (a1, a2):
            ax.grid(axis="x", visible=False)
        self.save(fig, "fig09_daily_timing")

    def fig10_daily_districts(self):
        """Mean restriction of each district: the RL agent vs the two hotspot rules (test year 1403)."""
        from ..data.graphs import ODOperator
        from ..data.grid import Grid
        f = self.folder("rq3_daily_v1")
        cfg = yaml.safe_load((f / "config.yaml").read_text(encoding="utf-8"))
        rcfg = yaml.safe_load((self.folder(cfg["rl"]["response_experiment"]) / "config.yaml").read_text(encoding="utf-8"))
        grid = Grid(rcfg)
        F, ids = ODOperator.district_fractions(rcfg, grid.n)
        pols = {p["policy"]: p for p in self.daily_policies("1402", "1403")}
        show = [n for n in ["RL (gamma 0.99)", "Hotspot (by car-km)", "Hotspot (by NO2)"] if n in pols]
        owner = np.where(F.sum(1) > 0, F.argmax(1), -1)
        rr, cc = np.nonzero(owner.reshape(grid.H, grid.W) >= 0)
        r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1          # crop to the 22 districts
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("seq", ["#f0efec"] + SEQ[1:])
        fig, axes = self.plt.subplots(1, len(show), figsize=(WIDTH, 2.4))
        rows = []
        for ax, name in zip(np.atleast_1d(axes), show):
            act = pols[name]["act"]
            r = act[:, :act.shape[1] // 2].mean(0)
            rows.append(dict(policy=name, **{d: round(float(v), 4) for d, v in zip(ids, r)}))
            field = np.where(owner >= 0, r[np.maximum(owner, 0)], np.nan).reshape(grid.H, grid.W)[r0:r1, c0:c1]
            im = ax.imshow(field, cmap=cmap, vmin=0, vmax=0.1, interpolation="nearest")
            ax.set_title(name, loc="left", fontsize=8)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
        cb.set_label("mean restriction over the year")
        self.table(pd.DataFrame(rows), "t08_daily_districts")
        self.save(fig, "fig10_daily_districts")

    # ------------------------------------------------------------------ RQ4
    def fig11_scorecard(self):
        """Policy scorecard (RQ4): CO2, travel time, exposure, risk and stability of every tuned policy (daily design)."""
        rows = []
        for a, b in [("1402", "1403"), ("1403", "1402")]:
            for p in self.daily_policies(a, b):
                R, C = p["returns"], p["comp"].mean(0)
                k = max(1, int(np.ceil(0.1 * len(R))))
                rows.append(dict(transfer=f"{a}to{b}", policy=p["policy"], detail=p["detail"], return_mean=R.mean(),
                                 cvar10=np.sort(R)[:k].mean(), co2_saved_kt=C[0] / 1e3, hours_lost_M=C[1] / 1e6,
                                 exposure_change_pct=C[3] / len(p["act"]) * 100))
        t = pd.DataFrame(rows)
        t = t[t.policy != "No action"]
        s = t.groupby("policy").agg(return_mean=("return_mean", "mean"), return_min_transfer=("return_mean", "min"),
                                    cvar10=("cvar10", "mean"), co2_saved_kt=("co2_saved_kt", "mean"),
                                    hours_lost_M=("hours_lost_M", "mean"),
                                    exposure_change_pct=("exposure_change_pct", "mean")).reset_index()
        s["co2_t_per_1000_hours_lost"] = np.where(s.hours_lost_M > 0, s.co2_saved_kt / s.hours_lost_M, np.nan)
        s = s.sort_values("return_mean", ascending=False)
        self.table(t.round(3), "t09_scorecard_by_transfer")
        self.table(s.round(3), "t10_scorecard")
        LABEL_AT = {"RL (gamma 1)": (-6, 5, "right"), "RL (gamma 0.999)": (6, -11, "left"),
                    "Winter only": (-6, 5, "right"), "Calm days": (6, 5, "left"),
                    "Uniform": (5, -12, "left")}                                              # clear of the neighbours
        fig, ax = self.plt.subplots(figsize=(WIDTH * 0.75, 3.0))
        for r in s.itertuples():
            ax.plot(r.hours_lost_M, r.co2_saved_kt, "o", color=_emphasis(r.policy), ms=8, mec="white", mew=1)
            dx, dy, ha = LABEL_AT.get(r.policy, (5, 4, "left"))
            ax.annotate(r.policy, (r.hours_lost_M, r.co2_saved_kt), xytext=(dx, dy), textcoords="offset points", ha=ha,
                        fontsize=7, color=INK2)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.margins(y=0.12)
        ax.set_xlabel("travel time lost (million person-hours per year; < 0 = saved)")
        ax.set_ylabel("CO2 saved (kt per year)")
        ax.set_title("RQ4: CO2 saved vs travel time, same 30-day budget", loc="left")
        self.save(fig, "fig11_scorecard")

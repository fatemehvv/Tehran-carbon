"""Publication figures of the paper (python main.py paper figures). Nothing is trained or re-scored.

Reads the committed result tables of stage A (docs/paper/results_stageA/) and writes, to docs/paper/figures/:
  fig3_correlogram     correlogram of the monthly target, its zero crossing and the effective sample size
  fig4_reach           skill against reach: SLX lines, the GNN and the no-graph network as points with seed ranges
  fig5_differences     GNN minus reference with jackknife intervals and the equivalence band (forest plot)
  fig6_seeds           single seeds, the five-seed ensemble and the deterministic SLX references
each as .png (300 dpi) and .pdf (vector), plus figure_data_comparisons.csv (the numbers behind figures 5 and 6).
Figure 1 (the literature funnel) is a drawn vector file in the same folder; figure 2 (maps) is a thesis figure.

One style for every figure:
  - the validated categorical order of the thesis figures (viz/figures.py) with fixed roles in every figure:
    GNN blue, SLX-OLS orange, SLX-GBM aqua, SLX-RF yellow, the no-graph network grey; reference lines in grey ink;
  - recessive axes and grid, text in grey ink (never in a series colour), bold panel titles "a  ..." at the top left;
  - A4 text width (6.3 in), sans-serif type at 8-9 pt, legends below the panels.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT
from .figures import GRID, INK, INK2, MUTED, SERIES, WIDTH

RESULTS = ROOT / "docs" / "paper" / "results_stageA"
OUT = ROOT / "docs" / "paper" / "figures"
BAND = "#f0efec"                                    # neutral fill (equivalence band, pixel size)
ROLE = {"SLX-OLS": SERIES[1], "SLX-GBM": SERIES[2], "SLX-RF": SERIES[3], "SAGE": SERIES[0], "MLP": MUTED}
LABEL = {"SLX-OLS": "SLX-OLS", "SLX-GBM": "SLX-GBM", "SLX-RF": "SLX-RF",
         "SAGE": "GNN, 5-seed ensemble", "MLP": "No-graph network, 5-seed ensemble"}
DODGE = {"SAGE": -1.8, "MLP": -1.6}                 # km; networks drawn left of the SLX point at the same reach,
                                                    # clear of it and of the dashed correlation-range line
METRIC = {"spearman": "Spearman ρ", "pr_auc": "PR-AUC (top 10 %)"}
LETTERS = "abcdefgh"

# the comparisons of figures 5 and 6: (group, table of evaluate_paper2, year, row label)
COMPARISONS = [("shorter", "paper_PA_H2.csv", 1402, "CV 1402"),
               ("shorter", "paper_PA_H2.csv", 1403, "CV 1403"),
               ("shorter", "paper_PA_H2c.csv", 1404, "Held-out 1404"),
               ("matched", "paper_PA_H2_reach.csv", 1402, "CV 1402"),
               ("matched", "paper_PA_H2_reach.csv", 1403, "CV 1403"),
               ("matched", "paper_PA_H2c_all.csv", 1404, "Held-out 1404")]
GROUP = {"shorter": "Pre-specified reference (SLX, 12.6 km)", "matched": "Reference with extended reach"}


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7.5, "legend.frameon": False, "legend.handlelength": 1.8,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 1.8, "figure.dpi": 100, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42, "ps.fonttype": 42})
    return plt


def save(fig, stem: Path) -> list[Path]:
    """stem.png (300 dpi) and stem.pdf (vector, fonts embedded as TrueType)."""
    import matplotlib.pyplot as plt
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    for p in paths:
        fig.savefig(p)
    plt.close(fig)
    return paths


def reach_figure(table: pd.DataFrame, zc: float, stem: Path, metrics=("spearman", "pr_auc")) -> list[Path]:
    """Skill against reach, one panel per metric (rows) and year (columns). SLX families are lines; networks that were
    run at one reach are points (the five-seed ensemble) with a bar over the single seeds, because the ensemble can lie
    outside the seed range. The dashed line is the zero crossing of the target correlogram."""
    plt = style()
    table = table.copy()
    table["year"] = table.year.astype(int)
    years = sorted(table.year.unique())
    metrics = [m for m in metrics if m in table]
    fams = [f for f in ROLE if f in set(table.family)] + sorted(set(table.family) - set(ROLE))
    fig, axes = plt.subplots(len(metrics), len(years), figsize=(WIDTH, 2.3 * len(metrics) + 0.6),
                             sharex=True, sharey="row", squeeze=False)
    for i, m in enumerate(metrics):
        for j, y in enumerate(years):
            ax = axes[i][j]
            for fam in fams:
                g = table[(table.family == fam) & (table.year == y)].dropna(subset=["reach_km", m])
                g = g.sort_values("reach_km")
                if g.empty:
                    continue
                c, lab = ROLE.get(fam, INK2), LABEL.get(fam, fam)
                lo, hi = f"{m}_seed_min", f"{m}_seed_max"
                if len(g) > 1:
                    gappy = fam == "MLP"                          # run at a few reaches only: points, no line
                    ax.plot(g.reach_km, g[m], color=c, marker="o", ms=5 if gappy else 4.5, mec="white", mew=0.8,
                            ls="none" if gappy else "-", label=lab, zorder=3)
                    continue
                x, v = float(g.reach_km.iloc[0]) + DODGE.get(fam, 0.0), float(g[m].iloc[0])
                if lo in g and pd.notna(g[lo].iloc[0]):
                    ax.vlines(x, float(g[lo].iloc[0]), float(g[hi].iloc[0]), color=c, lw=3, alpha=0.4, zorder=2)
                big = fam == "SAGE"
                ax.plot(x, v, ls="none", marker="D" if big else "o", ms=6.5 if big else 5.5, color=c,
                        mec="white", mew=1, label=lab, zorder=4)
            if np.isfinite(zc):
                ax.axvline(zc, color=INK2, ls=(0, (4, 3)), lw=0.9, zorder=1)
            ax.set_title(f"{LETTERS[i * len(years) + j]}   {y} · {METRIC.get(m, m)}")
            ax.set_xlim(-2.5, 41)
            if i == len(metrics) - 1:
                ax.set_xlabel("reach (km)")
            if j == 0:
                ax.set_ylabel(METRIC.get(m, m))
    if np.isfinite(zc):                             # name the dashed line once, low in panel a where the plot is empty
        ax = axes[0][0]
        ax.text(zc + 0.7, 0.04, f"target correlation\nrange ({zc:.0f} km)", transform=ax.get_xaxis_transform(),
                ha="left", va="bottom", fontsize=7, color=INK2)
    handles = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    order = [LABEL.get(f, f) for f in fams if LABEL.get(f, f) in handles]
    fig.legend([handles[k] for k in order], order, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.09, 1, 1), h_pad=1.2, w_pad=1.0)
    return save(fig, stem)


class PaperFigures:
    """Figures 3-6 of the manuscript from the committed stage A tables."""

    def __init__(self, results: Path = RESULTS, out: Path = OUT):
        self.res, self.out = Path(results), Path(out)

    def table(self, *parts: str) -> pd.DataFrame:
        return pd.read_csv(self.res.joinpath(*parts), encoding="utf-8-sig")       # the tables carry a BOM

    def all(self) -> list[Path]:
        return self.correlogram() + self.reach() + self.differences() + self.seeds()

    # ------------------------------------------------------------------ data
    def comparisons(self) -> pd.DataFrame:
        """Primary rows (difference and jackknife intervals) and seed rows (range of single seeds) per comparison."""
        rows = []
        for group, name, year, label in COMPARISONS:
            t = self.table("evaluate_paper2", name)
            t = t[t.year.astype(int) == year]
            for m in METRIC:
                p = t[(t.part == "primary") & (t.metric == m)].iloc[0]
                s = t[(t.part == "seeds") & (t.metric == m)]
                s = s.iloc[0] if len(s) else None
                rows.append(dict(
                    group=group, year=year, label=label, metric=m, delta=p["delta"], margin=p["margin"],
                    lo95=p["jack_low95"], hi95=p["jack_high95"], lo90=p["jack_low90"], hi90=p["jack_high90"],
                    verdict=p["verdict"], ensemble=p["candidate"], reference=p["reference"],
                    seed_mean=np.nan if s is None else s["candidate"],
                    seed_min=np.nan if s is None else s["reference"] + s["delta_min"],
                    seed_max=np.nan if s is None else s["reference"] + s["delta_max"]))
        d = pd.DataFrame(rows)
        self.out.mkdir(parents=True, exist_ok=True)
        d.to_csv(self.out / "figure_data_comparisons.csv", index=False, encoding="utf-8-sig")
        return d

    # ------------------------------------------------------------------ figures
    def correlogram(self) -> list[Path]:
        plt = style()
        t = self.table("target", "target_neff.csv").sort_values("year")
        zc = float(self.table("reach", "reach_correlogram.csv").zero_crossing_km.mean())
        cols = [c for c in t.columns if c.startswith("rho_")]
        mid = np.array([np.mean([float(v) for v in c[4:-2].split("_")]) for c in cols])     # rho_0_2km -> 1 km
        fig, ax = plt.subplots(figsize=(WIDTH * 0.66, 2.7))
        ax.axvspan(0, 4.4, color=BAND, lw=0, zorder=0)                     # one TROPOMI pixel (3.5 x 5.5 km)
        ax.text(4.9, 0.03, "grey band: one TROPOMI pixel", transform=ax.get_xaxis_transform(),
                ha="left", va="bottom", fontsize=6.5, color=INK2)          # bottom left: the curve starts near 1
        ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
        looks = [dict(color=INK2, ls="-", marker="o"), dict(color=MUTED, ls=(0, (4, 2)), marker="s")]
        for k, (_, r) in enumerate(t.iterrows()):
            ax.plot(mid, r[cols].to_numpy(float), ms=3.8, mec="white", mew=0.6, zorder=3,
                    label=f"{int(r['year'])}  (effective n = {r['n_eff_correlogram']:.1f})", **looks[k % len(looks)])
        ax.axvline(zc, color=INK2, ls=(0, (4, 3)), lw=0.9, zorder=2)
        # the curve is positive left of the crossing and negative right of it: text goes bottom left, legend top right
        ax.annotate(f"zero crossing\n{zc:.1f} km", xy=(zc, 0), xytext=(zc - 2.5, -0.35), fontsize=7.5, color=INK2,
                    ha="right", va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
        ax.set_xlim(0, mid.max() + 1)
        ax.set_ylim(-0.6, 1.05)
        ax.set_xlabel("distance between cells (km)")
        ax.set_ylabel("correlation of the monthly target")
        ax.legend(loc="upper right")
        fig.tight_layout()
        return save(fig, self.out / "fig3_correlogram")

    def reach(self) -> list[Path]:
        review = self.res.parent / "results_review" / "reach" / "reach_curve.csv"    # with GNN depths and short SLX
        t = pd.read_csv(review, encoding="utf-8-sig") if review.exists() else self.table("reach", "reach_curve.csv")
        zc = float(self.table("reach", "reach_correlogram.csv").zero_crossing_km.mean())
        return reach_figure(t, zc, self.out / "fig4_reach")

    def differences(self) -> list[Path]:
        """Forest plot: GNN ensemble minus reference, 95 % (thin) and 90 % (thick) jackknife intervals, ±margin band."""
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        plt = style()
        d = self.comparisons()
        ypos = {("shorter", 1402): 1, ("shorter", 1403): 2, ("shorter", 1404): 3,
                ("matched", 1402): 5.5, ("matched", 1403): 6.5, ("matched", 1404): 7.5}
        heads = {"shorter": 0, "matched": 4.5}
        fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.3), sharey=True)
        for k, (ax, m) in enumerate(zip(axes, METRIC)):
            dm = d[d.metric == m]
            mg = float(dm.margin.iloc[0])
            lim = max(abs(dm.lo95.min()), abs(dm.hi95.max()), mg) * 1.12      # symmetric: the sign reads at a glance
            if m == "spearman":                                         # the registered margin is for Spearman only
                ax.axvspan(-mg, mg, color=BAND, lw=0, zorder=0)
            ax.axvline(0, color=INK2, lw=0.8, zorder=1)
            for _, r in dm.iterrows():
                y = ypos[(r["group"], int(r["year"]))]
                ax.hlines(y, r["lo95"], r["hi95"], color=SERIES[0], lw=1.1, zorder=2)
                ax.hlines(y, r["lo90"], r["hi90"], color=SERIES[0], lw=3.4, zorder=3)
                ax.plot(r["delta"], y, ls="none", marker="o", ms=6, color=SERIES[0], mec="white", mew=1, zorder=4)
            ax.set_xlim(-lim, lim)
            ax.set_title(f"{LETTERS[k]}   {METRIC[m]}")
            ax.set_xlabel("GNN minus reference")
            ax.grid(axis="y", visible=False)
        for g, y in heads.items():
            axes[0].text(0.01, y, GROUP[g], transform=axes[0].get_yaxis_transform(), ha="left", va="center",
                         fontsize=7.5, fontweight="bold", color=INK2, zorder=5,
                         bbox=dict(facecolor="white", edgecolor="none", pad=1.5))   # over the band and zero line
        labels = {ypos[(g, y)]: lab for g, _, y, lab in COMPARISONS}
        axes[0].set_yticks(list(labels), list(labels.values()))
        axes[0].set_ylim(8.2, -0.7)
        h = [Line2D([], [], ls="none", marker="o", ms=6, color=SERIES[0], mec="white", label="difference"),
             Line2D([], [], color=SERIES[0], lw=3.4, label="90 % interval (jackknife)"),
             Line2D([], [], color=SERIES[0], lw=1.1, label="95 % interval (jackknife)"),
             Patch(facecolor=BAND, label=f"equivalence margin, Spearman (±{float(d.margin.iloc[0]):.2f})")]
        fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.5)
        return save(fig, self.out / "fig5_differences")

    def seeds(self) -> list[Path]:
        """Per year: range of the five single seeds (bar, | = their mean), the ensemble (dot), the deterministic SLX
        references with equal or longer reach (black diamond) and with 12.6 km (open diamond)."""
        from matplotlib.lines import Line2D
        plt = style()
        d = self.comparisons()
        years = [1402, 1403, 1404]
        lane = {"gnn": -0.2, "longer": 0.08, "shorter": 0.3}     # three lanes per row: close values stay visible
        fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.8), sharey=True)
        for k, (ax, m) in enumerate(zip(axes, METRIC)):
            for i, y in enumerate(years):
                r = d[(d.group == "matched") & (d.metric == m) & (d.year == y)].iloc[0]
                s = d[(d.group == "shorter") & (d.metric == m) & (d.year == y)].iloc[0]
                g = i + lane["gnn"]
                ax.hlines(g, r["seed_min"], r["seed_max"], color=SERIES[0], lw=7, alpha=0.3, zorder=2)
                ax.plot(r["seed_mean"], g, ls="none", marker="|", ms=11, mew=1.6, color=SERIES[0], zorder=3)
                ax.plot(r["ensemble"], g, ls="none", marker="o", ms=6.5, color=SERIES[0], mec="white", mew=1,
                        zorder=5)
                ax.plot(r["reference"], i + lane["longer"], ls="none", marker="D", ms=5.5, color=INK, mec="white",
                        mew=0.8, zorder=4)
                ax.plot(s["reference"], i + lane["shorter"], ls="none", marker="D", ms=5.5, mfc="white", mec=MUTED,
                        mew=1.2, zorder=4)
            ax.set_title(f"{LETTERS[k]}   {METRIC[m]}")
            ax.set_xlabel(METRIC[m])
            ax.grid(axis="y", visible=False)
        axes[0].set_yticks(range(len(years)), ["CV 1402", "CV 1403", "Held-out 1404"])
        axes[0].set_ylim(len(years) - 0.45, -0.55)
        for ax in axes:                                          # faint separators between the years
            for b in range(1, len(years)):
                ax.axhline(b - 0.5, color=GRID, lw=0.8, zorder=0)
        h = [Line2D([], [], color=SERIES[0], lw=7, alpha=0.3, label="GNN, five single seeds (range)"),
             Line2D([], [], ls="none", marker="|", ms=11, mew=1.6, color=SERIES[0], label="GNN, mean of single seeds"),
             Line2D([], [], ls="none", marker="o", ms=6.5, color=SERIES[0], mec="white", label="GNN, 5-seed ensemble"),
             Line2D([], [], ls="none", marker="D", ms=5.5, color=INK, mec="white", label="SLX, extended reach (1404: 22.7 km)"),
             Line2D([], [], ls="none", marker="D", ms=5.5, mfc="white", mec=MUTED, mew=1.2, label="SLX, 12.6 km (pre-specified)")]
        fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.16, 1, 1), w_pad=1.5)
        return save(fig, self.out / "fig6_seeds")

"""Figures 2 and 3 of the fusion paper (python main.py fusion figures). Nothing is trained or re-scored.

Reads the committed tables (docs/paper/fusion_paper/results_tests/*/fusion_eval.csv and
results_stage2/runs/*fusion_support*/fusion_support_correlogram.csv) and writes to docs/paper/fusion_paper/figures/:
  fig2_information_architecture  real source minus placebo (information) and placebo minus context (architecture),
                                 Spearman and PR-AUC, with jackknife 95 % intervals and the equivalence band
  fig3_target_support            correlograms of the native, all-day, calm and windy targets; the fusion gain against
                                 the smoothing scale of the target
Same style and validated palette as the figures of the companion paper (viz/paper_figures.py). Fixed roles: energy
information blue, traffic + stations information orange, architecture (placebo branch) grey.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT
from .figures import INK2, MUTED, SERIES, WIDTH
from .paper_figures import BAND, save, style

BASE = ROOT / "docs" / "paper" / "fusion_paper"
OUT = BASE / "figures"
ROWS = [("F1", "Energy vs its placebo", SERIES[0], 1402), ("F1", "", SERIES[0], 1403), ("C1", "", SERIES[0], 1404),
        ("F2", "Traffic + stations vs their placebos", SERIES[1], 1402), ("F2", "", SERIES[1], 1403),
        ("C2", "", SERIES[1], 1404),
        ("A_gated", "Placebo branch vs context only", MUTED, 1402), ("A_gated", "", MUTED, 1403),
        ("A_gated_1404", "", MUTED, 1404)]


def latest(pattern: str) -> Path:
    found = sorted(BASE.glob(pattern))
    if not found:
        raise FileNotFoundError(f"no file matches {BASE / pattern}")
    return found[-1]


class FusionFigures:
    def __init__(self):
        self.eval = pd.read_csv(latest("results_tests/*/fusion_eval.csv"))
        self.corr = pd.read_csv(latest("results_stage2/runs/*fusion_support*/fusion_support_correlogram.csv"))

    def all(self) -> list[Path]:
        return self.information() + self.support()

    def information(self) -> list[Path]:
        plt = style()
        t = self.eval
        fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.1), sharey=True)
        ypos = np.arange(len(ROWS))[::-1]
        for ax, metric, title in zip(axes, ("spearman", "pr_auc"), ("a  Spearman ρ (primary)", "b  PR-AUC (descriptive)")):
            if metric == "spearman":
                ax.axvspan(-0.02, 0.02, color=BAND, zorder=0, lw=0)
            ax.axvline(0, color=INK2, lw=0.8, zorder=1)
            for yp, (test, _, col, year) in zip(ypos, ROWS):
                r = t[(t.test == test) & (t.year.astype(str) == str(year)) & (t.metric == metric)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ax.plot([r.jack_low95, r.jack_high95], [yp, yp], color=col, lw=1.6, solid_capstyle="round", zorder=2)
                ax.plot(r.delta, yp, "o", color=col, ms=5.5, mec="white", mew=1.0, zorder=3)
            ax.set_title(title)
            ax.set_xlabel("Difference")
            ax.grid(axis="y", visible=False)
        axes[0].set_yticks(ypos)
        axes[0].set_yticklabels([str(y) for *_, y in ROWS])
        for yp, (_, label, col, _) in zip(ypos, ROWS):
            if label:
                axes[0].annotate(label, xy=(0, yp + 0.45), xycoords=("axes fraction", "data"), fontsize=7.5,
                                 color=INK2, ha="left", va="bottom", annotation_clip=False)
        fig.text(0.5, -0.04, "Points: five-seed ensembles (placebos averaged over five realisations); lines: jackknife "
                 "95 % intervals; band: equivalence margin ±0.02", ha="center", color=INK2, fontsize=7)
        fig.subplots_adjust(wspace=0.08)
        return save(fig, OUT / "fig2_information_architecture")

    def support(self) -> list[Path]:
        plt = style()
        fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.7))
        ax = axes[0]
        roles = [("T0", "Monthly target", INK2, "-"), ("all", "Seasonal, all clear days", SERIES[2], "-"),
                 ("calm", "Seasonal, calmest third", SERIES[0], "-"), ("windy", "Seasonal, windiest third", SERIES[1], "-")]
        c = self.corr[self.corr.year == 1402]
        for key, label, col, ls in roles:
            s = c[c.stratum == key]
            ax.plot(s.km, s.rho, ls, color=col, lw=1.6, label=label)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Correlation")
        ax.set_title("a  Correlogram of the target, 1402")
        ax.legend(loc="upper right", fontsize=7)
        ax = axes[1]
        t = self.eval[(self.eval.metric == "spearman") & self.eval.test.str.startswith("gain_")]
        sig = {"gain_T0": 0.0, "gain_T-2km": 2.0, "gain_T-4km": 4.0, "gain_T-8km": 8.0, "gain_T-12km": 12.0}
        ax.axhspan(-0.02, 0.02, color=BAND, zorder=0, lw=0)
        ax.axhline(0, color=INK2, lw=0.8)
        for year, col, dx in ((1402, SERIES[0], -0.25), (1403, SERIES[1], 0.25)):
            s = t[(t.test.isin(sig)) & (t.year.astype(str) == str(year))].copy()
            s["sigma"] = s.test.map(sig)
            s = s.sort_values("sigma")
            ax.vlines(s.sigma + dx, s.jack_low95, s.jack_high95, color=col, lw=1.4)
            ax.plot(s.sigma + dx, s.delta, "o-", color=col, ms=5, mec="white", mew=1.0, lw=1.2, label=str(year))
        ax.set_xticks([0, 2, 4, 8, 12])
        ax.set_xlabel("Smoothing of the target, σ (km); 0 = monthly target")
        ax.set_ylabel("Fusion gain (Spearman)")
        ax.set_title("b  All three sources minus energy")
        ax.legend(loc="lower right", fontsize=7)
        fig.subplots_adjust(wspace=0.32)
        return save(fig, OUT / "fig3_target_support")

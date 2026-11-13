"""Figures of the fusion paper (python main.py fusion figures). Nothing is trained or re-scored.

Reads the committed tables and writes to docs/paper/fusion_paper/figures/ (.png at 300 dpi and .pdf):
  fig1_placebo_maps              a real layer and placebo realisations of both constructions (A: IAAFT surrogates,
                                 B: rotation and shift); needs figures/fig1_maps_fus2_placebo_v1.npz and
                                 ..._v2.npz from python main.py fusion check --exp fus2_placebo_v1 / v2
  fig2_information_architecture  real source minus placebo (information) and placebo minus context (architecture),
                                 Spearman and PR-AUC, both placebo constructions, jackknife 95 % intervals
  fig3_target_support            correlograms of the target versions; the fusion gain against the smoothing scale
Tables: results_tests/*/fusion_eval.csv (placebo A), results_v2/runs/*fusion_evaluate*/fusion_eval.csv (placebo B),
results_stage2/runs/*fusion_support*/fusion_support_correlogram.csv.
Same style and validated palette as the figures of the companion paper (viz/paper_figures.py). Fixed roles: energy
information blue, traffic + stations information orange, architecture (placebo branch) grey; placebo A = open
marker, placebo B = filled marker.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT
from .figures import INK, INK2, MUTED, SERIES, WIDTH
from .paper_figures import BAND, save, style

BASE = ROOT / "docs" / "paper" / "fusion_paper"
OUT = BASE / "figures"
GROUPS = [("Energy vs its placebo", SERIES[0], [("F1", 1402), ("F1", 1403), ("C1", 1404)]),
          ("Traffic + stations vs their placebos", SERIES[1], [("F2", 1402), ("F2", 1403), ("C2", 1404)]),
          ("Placebo branch vs context only", MUTED, [("A_gated", 1402), ("A_gated", 1403), ("A_gated_1404", 1404)])]
VERSIONS = [("A", "Placebo A: IAAFT surrogates", +0.18, "white"), ("B", "Placebo B: rotation and shift", -0.18, None)]


def latest(pattern: str) -> Path | None:
    found = sorted(BASE.glob(pattern))
    return found[-1] if found else None


class FusionFigures:
    def __init__(self):
        a = latest("results_tests/*/fusion_eval.csv")
        b = latest("results_v2/runs/*fusion_evaluate*/fusion_eval.csv")
        self.eval = {k: pd.read_csv(f) for k, f in (("A", a), ("B", b)) if f is not None}
        if not self.eval:
            raise FileNotFoundError("no fusion_eval.csv under results_tests/ or results_v2/")
        self.corr = pd.read_csv(latest("results_stage2/runs/*fusion_support*/fusion_support_correlogram.csv"))

    def all(self) -> list[Path]:
        return self.maps() + self.information() + self.support()

    # ------------------------------------------------------------------ figure 1
    def maps(self) -> list[Path]:
        fa, fb = OUT / "fig1_maps_fus2_placebo_v1.npz", OUT / "fig1_maps_fus2_placebo_v2.npz"
        if not (fa.exists() and fb.exists()):
            print("figure 1 skipped: run python main.py fusion check --exp fus2_placebo_v1 and --exp fus2_placebo_v2")
            return []
        A, B = np.load(fa), np.load(fb)
        panels = [("a  Real", A["real"]), ("b  A, r = 0", A["plc0"]), ("c  A, r = 1", A["plc1"]),
                  ("d  B, r = 0", B["plc0"]), ("e  B, r = 1", B["plc1"])]
        plt = style()
        fig, axes = plt.subplots(1, len(panels), figsize=(WIDTH, 1.45))
        vmin, vmax = np.percentile(A["real"], [1, 99.5])
        for ax, (title, m) in zip(axes, panels):
            ax.imshow(m, cmap="Greys", vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.contour(A["domain"].astype(float), levels=[0.5], colors=SERIES[0], linewidths=0.6)
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(title, fontsize=8, pad=3)
        fig.text(0.5, -0.06, "Residential CO₂ of the inventory (log scale, annual mean). A: IAAFT surrogate; B: rotation by "
                 "180° and cyclic shift. Blue outline: label domain", ha="center", color=INK2, fontsize=7)
        fig.subplots_adjust(wspace=0.05)
        return save(fig, OUT / "fig1_placebo_maps")

    # ------------------------------------------------------------------ figure 2
    def information(self) -> list[Path]:
        plt = style()
        rows, y, ylab = [], 0.0, []
        group_y = []
        for g, (label, col, items) in enumerate(GROUPS):
            group_y.append((label, y + 0.75))
            for test, year in items:
                rows.append((test, year, col, y))
                ylab.append((y, str(year)))
                y -= 1.0
            y -= 0.9                                              # gap between groups (room for the group label)
        fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.6), sharey=True)
        for ax, metric, title in zip(axes, ("spearman", "pr_auc"), ("a  Spearman ρ (primary)", "b  PR-AUC (descriptive)")):
            if metric == "spearman":
                ax.axvspan(-0.02, 0.02, color=BAND, zorder=0, lw=0)
            ax.axvline(0, color=INK2, lw=0.8, zorder=1)
            for test, year, col, yp in rows:
                for ver, _, dy, face in VERSIONS:
                    t = self.eval.get(ver)
                    if t is None:
                        continue
                    r = t[(t.test == test) & (t.year.astype(str) == str(year)) & (t.metric == metric)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    ax.plot([r.jack_low95, r.jack_high95], [yp + dy, yp + dy], color=col, lw=1.3,
                            solid_capstyle="round", zorder=2)
                    ax.plot(r.delta, yp + dy, "o", color=col, ms=4.8, mfc=face or col, mec=col if face else "white",
                            mew=1.1, zorder=3)
            ax.set_title(title, pad=8)
            ax.set_xlabel("Difference")
            ax.grid(axis="y", visible=False)
            ax.set_ylim(y + 0.5, 1.2)
        axes[0].set_yticks([p for p, _ in ylab])
        axes[0].set_yticklabels([s for _, s in ylab])
        for label, yp in group_y:
            axes[0].text(0.02, yp, label, transform=axes[0].get_yaxis_transform(), fontsize=7.5, color=INK,
                         fontweight="bold", va="center", ha="left")
        handles = [plt.Line2D([], [], marker="o", ls="none", ms=5, color=MUTED, mfc=face or MUTED,
                              mec=MUTED if face else "white", label=lab) for _, lab, _, face in VERSIONS]
        fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.07), fontsize=7.5)
        fig.text(0.5, -0.11, "Five-seed ensembles; placebos averaged over five realisations; lines: jackknife 95 % "
                 "intervals; band: equivalence margin ±0.02. Both constructions failed the registered placebo check M4.",
                 ha="center", color=INK2, fontsize=7)
        fig.subplots_adjust(wspace=0.06)
        return save(fig, OUT / "fig2_information_architecture")

    # ------------------------------------------------------------------ figure 3
    def support(self) -> list[Path]:
        plt = style()
        fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.7))
        ax = axes[0]
        roles = [("T0", "Monthly target", INK2), ("all", "Seasonal, all clear days", SERIES[2]),
                 ("calm", "Seasonal, calmest third", SERIES[0]), ("windy", "Seasonal, windiest third", SERIES[1])]
        c = self.corr[self.corr.year == 1402]
        for key, label, col in roles:
            s = c[c.stratum == key]
            ax.plot(s.km, s.rho, color=col, lw=1.6, label=label)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Correlation")
        ax.set_title("a  Correlogram of the target, 1402")
        ax.legend(loc="upper right", fontsize=7)
        ax = axes[1]
        t = next(iter(self.eval.values()))
        t = t[(t.metric == "spearman") & t.test.str.startswith("gain_")]
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

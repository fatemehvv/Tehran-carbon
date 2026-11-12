# -*- coding: utf-8 -*-
"""Thesis figure (Persian labels): skill against reach, from docs/paper/results_stageA/reach/reach_curve.csv (branch
paper_TC). Usage: python fig_reach.py <reach_curve.csv> <font dir with Vazirmatn-*.ttf> <out.png>"""
import sys
from pathlib import Path

import arabic_reshaper
import matplotlib
import pandas as pd
from bidi.algorithm import get_display

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

CSV, FONT, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
for f in FONT.glob("Vazirmatn-*.ttf"):
    font_manager.fontManager.addfont(str(f))
INK, INK2, MUTED, GRID = "#1f2328", "#424a53", "#8c959f", "#e6e6e6"
plt.rcParams.update({"font.family": "Vazirmatn", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "legend.frameon": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white"})
FA = str.maketrans("0123456789.", "۰۱۲۳۴۵۶۷۸۹٫")


def fa(s: str) -> str:
    # Matplotlib >= 3.11 shapes and orders right-to-left text itself (libraqm); older versions need reshaping
    s = s.translate(FA)
    if tuple(int(v) for v in matplotlib.__version__.split(".")[:2]) >= (3, 11):
        return s
    return get_display(arabic_reshaper.reshape(s.replace("ۀ", "هٔ")))


def tick(fmt):
    return FuncFormatter(lambda v, _: fmt.format(v).translate(FA))


d = pd.read_csv(CSV, encoding="utf-8-sig")
COL = {"SLX-OLS": "#2a78d6", "SLX-GBM": "#e8663a", "SLX-RF": "#2a9d8f"}
NAME = {"SLX-OLS": "وقفۀ مکانی، حداقل مربعات", "SLX-GBM": "وقفۀ مکانی، تقویت گرادیانی", "SLX-RF": "وقفۀ مکانی، جنگل تصادفی"}
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
for ax, metric, ylab in ((axes[0], "spearman", "همبستگی رتبه‌ای اسپیرمن"), (axes[1], "pr_auc", "سطح زیر منحنی دقت–بازیابی")):
    for fam, c in COL.items():
        for year, ls, mk in ((1402, "-", "o"), (1403, "--", "s")):
            t = d[(d.family == fam) & (d.year == year)].sort_values("reach_km")
            ax.plot(t.reach_km, t[metric], ls=ls, marker=mk, ms=3.5, lw=1.4, color=c, mec="white", mew=0.6)
    for fam, c, dx in (("MLP", MUTED, 0.0), ("SAGE", "#7b3fb5", 0.0)):
        for year, mk, off in ((1402, "o", -0.6), (1403, "s", 0.6)):
            t = d[(d.family == fam) & (d.year == year)]
            if t.empty:
                continue
            r = t.iloc[0]
            x = r.reach_km + off
            lo, hi = r[f"{metric}_seed_min"], r[f"{metric}_seed_max"]
            ax.plot([x, x], [lo, hi], color=c, lw=1.2)
            ax.plot(x, r[metric], marker=mk, ms=6, color=c, mec="white", mew=0.8, ls="none")
    ax.axvline(24.0, color=MUTED, lw=1, ls=":")
    ax.text(24.6, ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]), fa("برد همبستگی هدف"), fontsize=7.5,
            color=INK2, rotation=90, va="bottom")
    ax.set_xlabel(fa("شعاع دید مدل (کیلومتر)"))
    ax.set_ylabel(fa(ylab))
    ax.xaxis.set_major_formatter(tick("{:.0f}"))
    ax.yaxis.set_major_formatter(tick("{:.2f}"))
    ax.set_xticks([0, 10, 20, 30, 40])
from matplotlib.lines import Line2D  # noqa: E402
handles = [Line2D([], [], color=c, lw=1.6) for c in COL.values()] + [
    Line2D([], [], color="#7b3fb5", marker="o", ls="none", ms=6), Line2D([], [], color=MUTED, marker="o", ls="none", ms=6),
    Line2D([], [], color=INK2, lw=1.4, ls="-", marker="o", ms=3.5), Line2D([], [], color=INK2, lw=1.4, ls="--", marker="s", ms=3.5)]
labels = [fa(n) for n in NAME.values()] + [fa("شبکۀ گراف سه‌لایه (میانگین پنج بذر؛ خط: بازۀ تک‌بذرها)"),
                                          fa("پرسپترون چندلایه، بدون همسایگی"), fa("۱۴۰۲"), fa("۱۴۰۳")]
fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.17), fontsize=7.8, columnspacing=1.2,
           handlelength=2.2)
fig.tight_layout()
fig.savefig(OUT)
print(OUT)

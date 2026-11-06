"""Persian thesis figures (every label, legend and number in Persian).

Eight figures are drawn again from the saved tables (docs/tables/*.csv). Three figures need run outputs that are not in
the repository (the 1404 maps, the daily budget curves and the district maps): their plot areas are cut from the
English images (docs/figures) and redrawn with Persian titles, axes, ticks, colour bars and legends.
Matplotlib >= 3.10 lays out Persian text itself (shaping and right-to-left order); the font is Vazirmatn.

    python fa_figures.py <tables dir> <english figures dir> <output dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from PIL import Image  # noqa: E402

TAB, ENG, OUT = (Path(a) for a in sys.argv[1:4])
OUT.mkdir(parents=True, exist_ok=True)

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV = ["#104281", "#2a78d6", "#f0efec", "#e34948", "#a32b2a"]
WIDTH = 6.3

plt.rcParams.update({
    "font.family": ["Vazirmatn", "DejaVu Sans"], "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 2, "figure.dpi": 100,
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white", "axes.unicode_minus": True})

FA = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def fa(x, nd: int | None = None) -> str:
    """Number (or text with digits) in Persian digits; '.' -> Persian decimal sign, '-' -> minus sign."""
    if isinstance(x, str):
        return x.translate(FA)
    s = f"{x:.{nd}f}" if nd is not None else f"{x:g}"
    return s.replace("-", "−").replace(".", "٫").translate(FA)


def fa_ticks(ax, axis="both", nd=None):
    f = FuncFormatter(lambda v, p: fa(v, nd))
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(f)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(f)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("written", name)


def read(name):
    return pd.read_csv(TAB / f"{name}.csv", encoding="utf-8-sig")


YEARS = ["1402", "1403"]
POL = {"Hotspot (by car-km)": "کانون بر اساس کیلومتر سفر", "RL (gamma 1)": "عامل، ضریب تنزیل ۱",
       "RL (gamma 0.999)": "عامل، ضریب تنزیل ۰٫۹۹۹", "RL (gamma 0.99)": "عامل، ضریب تنزیل ۰٫۹۹",
       "Hotspot (by NO2)": "کانون بر اساس آلودگی", "Winter only": "فقط فصل سرد", "Calm days": "روزهای کم‌باد",
       "Uniform": "یکنواخت"}


def emphasis(p):
    return SERIES[0] if p.startswith("RL") else SERIES[1] if "car-km" in p else MUTED


# ---------------------------------------------------------------------------------------------- inventory
def fig_inventory():
    t = read("t01_inventory")
    t["year"] = t.year.astype(str)
    sec = {"Buildings": "ساختمان", "Road transport": "حمل‌ونقل جاده‌ای", "Industry": "صنعت", "Power plants": "نیروگاه"}
    labels = list(sec)[::-1]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.4))
    y0 = np.arange(len(labels))
    for i, y in enumerate(YEARS):
        d = t[t.year == y].set_index("sector").loc[labels]
        pos = y0 + (0.2 if i == 0 else -0.2)
        ax.barh(pos, d.MtCO2, height=0.36, color=SERIES[i], label=fa(y), edgecolor="white", linewidth=1)
        ax.errorbar(d.MtCO2, pos, xerr=[d.MtCO2 - d.p05, d.p95 - d.MtCO2], fmt="none", ecolor=INK2, lw=0.8, capsize=2)
        for p, v, hi in zip(pos, d.MtCO2, d.p95):
            ax.text(hi + 0.3, p, fa(v, 1), va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y0, [sec[s] for s in labels])
    fa_ticks(ax, "x")
    ax.set_xlabel("میلیون تن CO₂ در سال (مقدار مرکزی؛ میله‌ها: بازۀ ۵ تا ۹۵ درصد مونت‌کارلو)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    save(fig, "fa_inventory")


# ---------------------------------------------------------------------------------------------- RQ1
def fig_ablation():
    t = read("t02_rq1_ablation")
    combos = ["traffic", "energy", "pollution", "traffic+energy", "traffic+pollution", "energy+pollution",
              "traffic+energy+pollution"]
    fus = ["early", "late", "gated"]
    name = {"traffic": "ترافیک", "energy": "انرژی", "pollution": "آلودگی"}
    M = np.array([[[t[(t.domains == c) & (t.fusion == f)][f"pr_auc_{y}"].item() for f in fus] for c in combos]
                  for y in YEARS])
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0), sharey=True)
    for k, ax in enumerate(axes):
        im = ax.imshow(M[k], cmap=cmap, vmin=0.5, vmax=0.8, aspect="auto")
        for i in range(len(combos)):
            for j in range(3):
                v = M[k, i, j]
                ax.text(j, i, fa(v, 3), ha="center", va="center", fontsize=7.5, color="white" if v > 0.7 else INK)
        ax.set_xticks(range(3), ["زودهنگام", "دیرهنگام", "دروازه‌ای"])
        ax.set_yticks(range(len(combos)), [" + ".join(name[p] for p in c.split("+")) for c in combos])
        ax.grid(False)
        ax.set_title(f"{fa(YEARS[k])} (اعتبارسنجی بلوکی)", loc="center")
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label("سطح زیر منحنی دقت–بازیابی (حدس تصادفی = ۰٫۱۰)")
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: fa(v, 2)))
    save(fig, "fa_ablation")


# ---------------------------------------------------------------------------------------------- RQ2
def fig_models():
    t = read("t03_rq2_models")
    lab = ["شبکۀ گراف (سه لایه؛ ۱۴۰۴: پنج لایه)", "جنگل تصادفی با وقفۀ مکانی", "تقویت گرادیانی با وقفۀ مکانی",
           "حداقل مربعات با وقفۀ مکانی", "پرسپترون چندلایه (بدون گراف)", "تقویت گرادیانی (خود سلول)",
           "حداقل مربعات (خود سلول)", "عملگر فیزیکی پیشرو"]
    fig, ax = plt.subplots(figsize=(WIDTH, 3.0))
    y0 = np.arange(len(t))[::-1]
    ax.axvline(0.10, color=MUTED, lw=1, ls="--")
    ax.text(0.108, y0[0] + 0.25, "حدس تصادفی", fontsize=7, color=MUTED)
    for i, (col, nm) in enumerate([("1402", "۱۴۰۲ (توسعه)"), ("1403", "۱۴۰۳ (توسعه)"), ("1404", "۱۴۰۴ (سال کنارگذاشته)")]):
        ax.plot(t[col], y0, "o", ms=6, color=SERIES[i], label=nm, mec="white", mew=1)
    ax.set_yticks(y0, lab)
    ax.set_xlim(0, 0.9)
    fa_ticks(ax, "x")
    ax.set_xlabel("سطح زیر منحنی دقت–بازیابی نقشۀ کانون (۱۰ درصد بالای NO₂؛ اعتبارسنجی بلوکی)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    save(fig, "fa_models")


def fig_receptive():
    t = read("t04_receptive_field")
    slx = {"1402": 0.757, "1403": 0.791}                      # SLX-GBM, t03 / Table "H2" of chapter 4
    fig, ax = plt.subplots(figsize=(WIDTH * 0.62, 2.6))
    for i, y in enumerate(YEARS):
        ax.plot(t.layers, t[y], "-o", color=SERIES[i], ms=6, mec="white", mew=1, label=f"شبکۀ گراف {fa(y)}")
        ax.axhline(slx[y], color=SERIES[i], lw=1, ls=":")
    ax.text(4.2, np.mean(list(slx.values())), "تقویت گرادیانی\nبا وقفۀ مکانی\n(نقطه‌چین)", fontsize=7, color=INK2,
            va="center")
    ax.set_xlim(0.8, 5.0)
    ax.set_xticks([1, 2, 3, 4])
    fa_ticks(ax, "x")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: fa(v, 3)))
    ax.set_xlabel("تعداد لایه‌های شبکۀ گراف (میدان دید)")
    ax.set_ylabel("سطح زیر منحنی دقت–بازیابی")
    ax.legend(loc="lower right")
    save(fig, "fa_receptive")


def fig_comparisons():
    """Every PR-AUC comparison of questions 1 and 2 with its 95 % interval (replaces the 'decision rule' forest)."""
    t = read("t05_hypotheses")
    t = t[t.ci_low.notna() & t.hypothesis.isin(["H1", "H1c", "H2", "H2c", "H2c_robust", "H2b"])]
    lab = {("H1", "cv_1402"): "همجوشی سه حوزه در برابر بهترین تک‌حوزه، ۱۴۰۲",
           ("H1", "cv_1403"): "همجوشی سه حوزه در برابر بهترین تک‌حوزه، ۱۴۰۳",
           ("H1c", "cv_1404"): "همجوشی سه حوزه در برابر فقط انرژی، ۱۴۰۴",
           ("H2", "cv_1402"): "شبکۀ گراف در برابر بهترین مدل سنتی، ۱۴۰۲",
           ("H2", "cv_1403"): "شبکۀ گراف در برابر بهترین مدل سنتی، ۱۴۰۳",
           ("H2c", "cv_1404"): "شبکۀ گراف در برابر جنگل تصادفی با وقفۀ مکانی، ۱۴۰۴",
           ("H2c_robust", "cv_1404"): "شبکۀ گراف در برابر بهترین مدل سنتی ۱۴۰۴ (افزوده پس از نتایج)",
           ("H2b", "cv_1402"): "شبکۀ گراف روزانه با باد در برابر بهترین مدل سنتی، ۱۴۰۲",
           ("H2b", "cv_1403"): "شبکۀ گراف روزانه با باد در برابر بهترین مدل سنتی، ۱۴۰۳"}
    fig, ax = plt.subplots(figsize=(WIDTH, 3.4))
    y0 = np.arange(len(t))[::-1]
    above, incl = SERIES[2], MUTED
    for yy, r in zip(y0, t.itertuples()):
        c = above if r.ci_low > 0 else incl
        ax.plot([r.ci_low, r.ci_high], [yy, yy], color=c, lw=2, solid_capstyle="round")
        ax.plot(r.delta, yy, "o", color=c, ms=7, mec="white", mew=1)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y0, [lab[(r.hypothesis, r.test)] for r in t.itertuples()])
    fa_ticks(ax, "x", 2)
    ax.set_xlabel("اختلاف سطح زیر منحنی دقت–بازیابی: مدل پیشنهادی منهای گزینۀ مقایسه (بازۀ ۹۵ درصد)")
    ax.grid(axis="y", visible=False)
    ax.legend([Line2D([], [], color=c, marker="o", lw=2) for c in (above, incl)],
              ["بازه کاملاً بالای صفر", "بازه شامل صفر"], loc="upper center", bbox_to_anchor=(0.35, -0.17), ncol=2)
    save(fig, "fa_comparisons")


def crop_box(a, region, white=245):
    """Bounding box (x0, y0, x1, y1) of the non-white pixels of a (H, W, 3) array inside region (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = region
    sub = a[y0:y1, x0:x1, :3]
    m = (sub < white).any(2)
    ys, xs = np.nonzero(m)
    return x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1


def fig_maps_1404():
    im = Image.open(ENG / "image13.png").convert("RGB")
    a = np.asarray(im)
    regions = [(60, 175, 650, 545), (690, 175, 1270, 545), (1330, 70, 1900, 440)]
    boxes = [crop_box(a, r) for r in regions]
    crops = [a[y0:y1, x0:x1] for x0, y0, x1, y1 in boxes]
    fig = plt.figure(figsize=(WIDTH, 2.75))
    titles = ["میانگین NO₂ مشاهده‌شده (z، ۱۴۰۴)", "پیش‌بینی بیرون از بلوک شبکۀ گراف", "کانون‌ها (۱۰ درصد بالا)"]
    axes = [fig.add_axes([0.01 + 0.335 * i, 0.30, 0.31, 0.60]) for i in range(3)]
    for ax, c, tt in zip(axes, crops, titles):
        ax.imshow(c, interpolation="none")
        ax.set_title(tt, fontsize=8)
        ax.axis("off")
    div = LinearSegmentedColormap.from_list("div", DIV)
    lim = 1.91                                   # read from the colour bar of the English figure
    cax = fig.add_axes([0.06, 0.17, 0.56, 0.035])
    sm = matplotlib.cm.ScalarMappable(cmap=div, norm=matplotlib.colors.Normalize(-lim, lim))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
    cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: fa(v, 1)))
    cb.set_label("لگاریتم استانداردشدۀ NO₂ (z)")
    axes[2].legend([Patch(color=SERIES[2]), Patch(color=SERIES[0]), Patch(color=SERIES[1])],
                   ["هر دو", "فقط مشاهده", "فقط پیش‌بینی"], loc="upper center", bbox_to_anchor=(0.5, -0.02),
                   fontsize=7.5, ncol=3, handlelength=1.2, columnspacing=1.0)
    save(fig, "fa_maps_1404")


# ---------------------------------------------------------------------------------------------- RQ3
def fig_lambda():
    t = read("t06_rq3_lambda")
    fig, ax = plt.subplots(figsize=(WIDTH * 0.62, 2.6))
    for i, (k, g) in enumerate(t.groupby("transfer")):
        a, b = k.split("to")
        ax.plot(g.lam, g["diff"], "-o", color=SERIES[i], ms=6, mec="white", mew=1,
                label=f"آموزش {fa(a)}، آزمون {fa(b)}")
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.25, 0.5, 1, 2, 4], ["۰٫۲۵", "۰٫۵", "۱ (اصلی)", "۲", "۴"])
    ax.minorticks_off()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: fa(v, 2)))
    ax.set_xlabel("وزن زمان سفر در پاداش (λ)")
    ax.set_ylabel("بازدۀ عامل منهای بهترین سیاست ایستا")
    ax.legend(loc="upper left")
    save(fig, "fa_lambda")


def fig_daily_returns():
    t = read("t09_scorecard_by_transfer")
    order = t[t.transfer == "1402to1403"].sort_values("return_mean").policy.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.9), sharey=True)
    for ax, (k, g) in zip(axes, t.groupby("transfer", sort=False)):
        g = g.set_index("policy").loc[order]
        y0 = np.arange(len(g))
        ax.barh(y0, g.return_mean, color=[emphasis(p) for p in g.index], height=0.6, edgecolor="white")
        for yy, v in zip(y0, g.return_mean):
            ax.text(v + 0.4, yy, fa(v, 1), va="center", fontsize=7.5, color=INK2)
        ax.set_yticks(y0, [POL[p] for p in order])
        ax.grid(axis="y", visible=False)
        a, b = k.split("to")
        ax.set_title(f"آموزش {fa(a)}، آزمون {fa(b)}", loc="center")
        ax.set_xlabel("میانگین بازدۀ سالانه (۳۰ دوره)")
        ax.set_xlim(0, 33)
        fa_ticks(ax, "x")
    save(fig, "fa_daily_returns")


def fig_daily_timing():
    """Budget curves cut from the English figure; axes, ticks and legend in Persian."""
    a = np.asarray(Image.open(ENG / "image17.png").convert("RGB"))
    # calibration of the English figure (pixels): x = day 0 at 249, Esfand 1st (day 336) at 1474
    px_day = (1474 - 249) / 336.0
    x_of = lambda px: (px - 249) / px_day                                     # noqa: E731
    # top panel: y = 0 at 311, 4 at 141; bottom: y = 0 at 914, 30 at 428
    top_y = lambda py: (311 - py) / ((311 - 141) / 4.0)                       # noqa: E731
    bot_y = lambda py: (914 - py) / ((914 - 428) / 30.0)                      # noqa: E731
    X0, X1 = 184, 1646
    T0, T1 = 85, 318
    B0, B1 = 397, 913
    top, bot = a[T0:T1, X0:X1], a[B0:B1, X0:X1]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(WIDTH, 3.7), sharex=True, height_ratios=[1, 2.2])
    a1.imshow(top, extent=[x_of(X0), x_of(X1), top_y(T1), top_y(T0)], aspect="auto", interpolation="lanczos")
    a2.imshow(bot, extent=[x_of(X0), x_of(X1), bot_y(B1), bot_y(B0)], aspect="auto", interpolation="lanczos")
    for ax in (a1, a2):
        ax.grid(False)
        ax.set_xlim(x_of(X0), x_of(X1))
    a1.set_ylim(top_y(T1), top_y(T0))
    a2.set_ylim(bot_y(B1), bot_y(B0))
    a1.set_yticks([0, 2, 4])
    a2.set_yticks([0, 5, 10, 15, 20, 25, 30])
    fa_ticks(a1, "y")
    fa_ticks(a2, "y")
    a1.set_ylabel("شدت NO₂\n(روز به میانگین سال)")
    a2.set_ylabel("بودجۀ مصرف‌شده (روز)")
    days = [31] * 6 + [30] * 6                                                # Jalali 1403 (Esfand 30 days)
    starts = np.concatenate([[0], np.cumsum(days)[:-1]])
    months = ["فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور", "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند"]
    a2.set_xticks(starts, months, fontsize=6.3)
    handles = [Line2D([], [], color=SERIES[0], lw=2), Line2D([], [], color=SERIES[2], lw=2),
               Line2D([], [], color=SERIES[6], lw=2), Line2D([], [], color=SERIES[1], lw=1.4, ls="--"),
               Line2D([], [], color=SERIES[3], lw=1.4, ls="--"), Line2D([], [], color=MUTED, lw=1.4, ls="--")]
    names = ["RL (gamma 0.99)", "RL (gamma 0.999)", "RL (gamma 1)", "Hotspot (by car-km)", "Hotspot (by NO2)",
             "Winter only"]
    a2.legend(handles, [POL[n] for n in names], ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=7.5)
    save(fig, "fa_daily_timing")


def fig_daily_districts():
    a = np.asarray(Image.open(ENG / "image16.png").convert("RGB"))
    regions = [(20, 195, 400, 400), (450, 195, 830, 400), (880, 195, 1260, 400)]
    crops = []
    for r in regions:
        x0, y0, x1, y1 = crop_box(a, r)
        crops.append(a[y0:y1, x0:x1])
    fig = plt.figure(figsize=(WIDTH, 2.0))
    titles = ["عامل، ضریب تنزیل ۰٫۹۹", "کانون بر اساس کیلومتر سفر", "کانون بر اساس آلودگی"]
    for i, (c, tt) in enumerate(zip(crops, titles)):
        ax = fig.add_axes([0.005 + 0.29 * i, 0.05, 0.27, 0.72])
        ax.imshow(c, interpolation="none")
        ax.set_title(tt, fontsize=8)
        ax.axis("off")
    cmap = LinearSegmentedColormap.from_list("seq", ["#f0efec"] + SEQ[1:])
    cax = fig.add_axes([0.895, 0.08, 0.02, 0.78])
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(0, 0.1))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_ticks([0, 0.02, 0.04, 0.06, 0.08, 0.1])
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: fa(v, 2)))
    cb.set_label("میانگین شدت محدودیت در طول سال")
    save(fig, "fa_daily_districts")


# ---------------------------------------------------------------------------------------------- RQ4
def fig_scorecard():
    s = read("t10_scorecard")
    at = {"RL (gamma 1)": (-6, 5, "right"), "RL (gamma 0.999)": (6, 6, "left"), "Winter only": (-6, 6, "right"),
          "Calm days": (6, 6, "left"), "Uniform": (5, -12, "left"), "RL (gamma 0.99)": (-6, -13, "right"),
          "Hotspot (by car-km)": (6, -3, "left"), "Hotspot (by NO2)": (6, 2, "left")}
    fig, ax = plt.subplots(figsize=(WIDTH * 0.78, 3.0))
    for r in s.itertuples():
        ax.plot(r.hours_lost_M, r.co2_saved_kt, "o", color=emphasis(r.policy), ms=8, mec="white", mew=1)
        dx, dy, ha = at.get(r.policy, (5, 4, "left"))
        ax.annotate(POL[r.policy], (r.hours_lost_M, r.co2_saved_kt), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, fontsize=7, color=INK2)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.margins(x=0.12, y=0.12)
    fa_ticks(ax)
    ax.set_xlabel("تغییر زمان سفر (میلیون نفر–ساعت در سال؛ منفی = صرفه‌جویی)")
    ax.set_ylabel("کاهش CO₂ (هزار تن در سال)")
    save(fig, "fa_scorecard")


# ---------------------------------------------------------------------------------------------- method chapter
def fig_framework():
    fig, ax = plt.subplots(figsize=(WIDTH, 4.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 80)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec, size=7.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK, linespacing=1.25)

    def arrow(x0, y0, x1, y1, **kw):
        p = dict(arrowstyle="-|>", color=INK2, lw=1.1, shrinkA=0, shrinkB=0, mutation_scale=9)
        p.update(kw)
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=p)

    blue, blue_e = "#e7f0fb", SERIES[0]
    orange, orange_e = "#fdeee6", SERIES[1]
    ax.text(99.5, 76, "لایۀ پیش‌بینی (پرسش‌های اول و دوم)", ha="right", va="center", fontsize=8.5, color=SERIES[0],
            fontweight="bold")
    ax.text(99.5, 33, "لایۀ تصمیم (پرسش‌های سوم و چهارم)", ha="right", va="center", fontsize=8.5, color=SERIES[1],
            fontweight="bold")
    w, h, y = 17.6, 20, 49
    xs = [81.4, 61.4, 41.4, 21.4, 1.4]
    texts = ["داده‌های ثانویه:\nماهواره، هواشناسی،\nانرژی، حمل‌ونقل\nو ایستگاه‌های پایش",
             "سیاهۀ انتشار\nسطح اول\nو\nمدل ترافیک",
             "سطح دوم سیاهه:\nتخصیص مکانی\nو زمانی",
             "ویژگی‌های\nچهار شاخه\nو متغیر هدف NO₂",
             "مدل‌های همجوشی\nو گراف در برابر\nمدل‌های سنتی؛\nبلوکی و سال ۱۴۰۴"]
    for x, tt in zip(xs, texts):
        box(x, y, w, h, tt, blue, blue_e)
    for x0, x1 in zip(xs[:-1], xs[1:]):
        arrow(x0 - 0.5, y + h / 2, x1 + w + 0.5, y + h / 2)
    y2 = 4
    xs2 = [1.4, 21.4, 41.4, 61.4, 81.4]
    texts2 = ["مدل گراف\nفریزشده به‌عنوان\nمدل مواجهه",
              "شبیه‌ساز\n۲۲ منطقه؛\nماهانه و روزانه",
              "سیاست‌های ایستا\nو عامل یادگیری\nتقویتی",
              "سنجش سیاست‌ها\nبا اعداد تصادفی\nمشترک",
              "کارنامۀ سیاست‌ها\nو داشبورد\nتصمیم‌یار"]
    for x, tt in zip(xs2, texts2):
        box(x, y2, w, h, tt, orange, orange_e)
    for x0, x1 in zip(xs2[:-1], xs2[1:]):
        arrow(x0 + w + 0.5, y2 + h / 2, x1 - 0.5, y2 + h / 2)
    arrow(1.4 + w / 2, y - 0.6, 1.4 + w / 2, y2 + h + 0.6)
    arrow(61.4 + w / 2, y - 0.6, 21.4 + w * 0.7, y2 + h + 0.6, color=MUTED, lw=1, ls="--", mutation_scale=8)
    ax.text(53.5, 36.5, "سفرها، سرعت‌ها و CO₂ جاده‌ای", fontsize=6.8, color=INK2, ha="left")
    save(fig, "fa_framework")


if __name__ == "__main__":
    for f in [fig_framework, fig_inventory, fig_ablation, fig_models, fig_receptive, fig_comparisons, fig_maps_1404,
              fig_lambda, fig_daily_returns, fig_daily_timing, fig_daily_districts, fig_scorecard]:
        f()

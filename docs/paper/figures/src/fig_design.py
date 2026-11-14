"""Figure 4: study design (shared network and the reach-matched test), in English and Persian.

    python fig_design.py            -> ../fig_design_en.svg and ../fig_design_fa.svg

Pure Python, no dependencies; render to PDF and PNG with render_svg.js (Playwright/Chromium). The Persian version is
laid out right to left (shapes mirrored, text not), except the reach axis, which keeps kilometres increasing to the
right; it uses Vazirmatn (pass the font folder to render_svg.js). Sized for a full-width figure: the smallest text is
17 px of a 1400-px canvas, about 6 pt at 190 mm. What is drawn comes from configs/config.yaml (input layers, encoders,
gate, graph dilations 1/3/9, 9 cells of reach per GNN layer, SLX windows, no-graph box means) and Sections 2-3 of the
manuscript. The maps are schematic.
"""
from __future__ import annotations

import math
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
W, H = 1400, 1300
INK, INK2, MUTED, GRID, PAPER = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#f4f3ef"
GNN, SLX, NOG, NAVY = "#2a78d6", "#eb6834", "#8a8984", "#104281"
GNN_T, SLX_T, NOG_T, NAVY_T, ZONE = "#dce9f8", "#fbe3d8", "#ecebe7", "#e3eaf3", "#f3f6fa"
S1, S2, S3, S4 = 17, 18.5, 20, 23

TXT = {
    "en": dict(
        a="a", b="b", ta="Inputs, fusion network and target (all neural models)", inputs="Inputs: 2,436 cells × 36 months",
        cards=[("Traffic", "road CO₂, roads, traffic", "7 layers"), ("Energy", "building, industry, power CO₂", "5 layers"),
               ("Pollution", "CO at stations", "1 layer"), ("Context", "people, buildings, lights, weather", "7 layers")],
        enc="encoders", gate=["Gate", "softmax", "weight per", "cell and month"], fusion="fusion (RQ1)",
        gnn=["GNN: message passing", "grid graph, 1–5 layers", "neighbours at 1, 3, 9 cells", "reach = 7.6 km × layers",
             "no-graph network: none"],
        head=["linear", "head"], pred="Predicted", target="Target", mse="MSE",
        mapcap=["NO₂ (z), TROPOMI target", "outlined: top 10 % (hotspots)"],
        box=["No-graph network: the neighbourhood enters here, as box means of each input.",
             "SLX: the same inputs and box means go to OLS, GBM or RF, without this network."],
        tb="From the thesis comparison to the test in this study", study="THIS STUDY",
        thesis="Thesis: GNN against the best SLX", leg=["GNN, 3 layers: 22.7 km", "best SLX window: 12.6 km"],
        ahead=["GNN ahead in all three", "pre-registered tests"],
        warn=["Two things differed at once:", "how neighbours are combined", "and how far the model sees."],
        same="Same reach, one difference at a time",
        same2="shown at 22.7 km; also compared at 7.6, 15.1, 30.3 and 37.8 km",
        cols=[("GNN", "learned messages", "neural network"), ("Same network, no graph", "fixed box means", "same neural network"),
              ("SLX: OLS, GBM, RF", "fixed box means", "linear model or trees")],
        only="only the graph differs", both="graph and learner differ", varied="Reach varied for every model",
        corr="target correlation length (24 km)", rows=["SLX", "GNN", "No graph"],
        axis="reach (km): how far from a cell a prediction can look", pair="thesis pair",
        ev=["Same evaluation for all: 12 blocks of 22 km, 2-km buffer · held-out year 1404 · 5 seeds",
            "Spearman (primary), PR-AUC · permutation test, jackknife, ±0.02 equivalence, Holm"],
        digits=None, font='"Liberation Sans", Arial, Helvetica, sans-serif', mirror=False,
        title="Study design: shared network and the reach-matched test"),
    "fa": dict(
        a="الف", b="ب", ta="ورودی‌ها، شبکهٔ همجوشی و هدف (همهٔ مدل‌های عصبی)", inputs="ورودی‌ها: ۲٬۴۳۶ سلول × ۳۶ ماه",
        cards=[("ترافیک", "CO₂ جاده، راه‌ها، ترافیک", "۷ لایه"), ("انرژی", "CO₂ ساختمان، صنعت، نیروگاه", "۵ لایه"),
               ("آلودگی", "CO ایستگاه‌ها", "۱ لایه"), ("زمینه", "جمعیت، ساختمان، نور شب، هوا", "۷ لایه")],
        enc="رمزگذارها", gate=["دروازه", "softmax", "وزن هر حوزه", "در هر سلول و ماه"], fusion="همجوشی (RQ1)",
        gnn=["GNN: پیام‌رسانی", "گراف شبکه‌ای، ۱ تا ۵ لایه", "همسایه‌ها در ۱، ۳ و ۹ سلول", "شعاع = ۷٫۶ km × لایه‌ها",
             "شبکهٔ بدون گراف: ندارد"],
        head=["سر", "خطی"], pred="پیش‌بینی", target="هدف", mse="MSE",
        mapcap=["NO₂ (z)، هدف TROPOMI", "مرزدار: ۱۰٪ بالا (کانون‌ها)"],
        box=["شبکهٔ بدون گراف: همسایگی از اینجا وارد می‌شود، به شکل میانگین مربعی هر ورودی.",
             "SLX: همین ورودی‌ها و میانگین‌ها را به OLS، GBM یا RF می‌دهد، بدون این شبکه."],
        tb="از مقایسهٔ پایان‌نامه تا آزمون این مقاله", study="این مقاله",
        thesis="پایان‌نامه: GNN در برابر بهترین SLX", leg=["GNN سه‌لایه: ۲۲٫۷ km", "بهترین پنجرهٔ SLX:\u200f ۱۲٫۶ km"],
        ahead=["GNN در هر سه آزمون", "پیش‌ثبت‌شده جلو بود"],
        warn=["دو چیز هم‌زمان فرق داشت:", "نحوهٔ ترکیب همسایه‌ها", "و اینکه مدل تا کجا را می‌بیند."],
        same="شعاع برابر، هر بار فقط یک تفاوت",
        same2="اینجا ۲۲٫۷ km؛ در ۷٫۶، ۱۵٫۱، ۳۰٫۳ و ۳۷٫۸ km هم مقایسه شد",
        cols=[("GNN", "پیام‌های آموخته‌شده", "شبکهٔ عصبی"), ("همان شبکه، بدون گراف", "میانگین مربعی ثابت", "همان شبکهٔ عصبی"),
              ("SLX: OLS، GBM، RF", "میانگین مربعی ثابت", "مدل خطی یا درخت‌ها")],
        only="فقط گراف فرق دارد", both="گراف و یادگیرنده فرق دارند", varied="شعاع برای همهٔ مدل‌ها تغییر داده شد",
        corr="طول همبستگی هدف (۲۴ km)", rows=["SLX", "GNN", "بدون گراف"],
        axis="شعاع (km): پیش‌بینی هر سلول تا کجا را می‌بیند", pair="جفت پایان‌نامه",
        ev=["ارزیابی یکسان برای همه: ۱۲ بلوک ۲۲ کیلومتری، حاشیهٔ ۲ km · سال کنار گذاشتهٔ ۱۴۰۴ · ۵ seed",
            "Spearman (اصلی)، PR-AUC · آزمون جایگشت، jackknife، هم‌ارزی ±۰٫۰۲، Holm"],
        digits="۰۱۲۳۴۵۶۷۸۹", font='Vazirmatn, "DejaVu Sans", sans-serif', mirror=True,
        title="طراحی مطالعه: شبکهٔ مشترک و آزمون هم‌شعاع"),
}


class Canvas:
    """Shapes go to a group that is mirrored for right-to-left layouts; text and the reach axis are never mirrored."""

    def __init__(self, lang):
        self.t = TXT[lang]
        self.rtl = self.t["mirror"]
        self.shapes, self.texts, self.fixed = [], [], []

    def tx(self, x):
        return W - x if self.rtl else x

    def num(self, s):
        d = self.t["digits"]
        return s if not d else "".join(d[int(c)] if c.isdigit() else c for c in str(s))

    def text(self, x, y, s, size=S2, fill=INK, weight=400, anchor="start", italic=False, fixed=False, ltr=False):
        st = ' font-style="italic"' if italic and not self.rtl else ""
        if fixed:                                                   # literal position (reach axis)
            xx, dr = x, (' direction="rtl"' if self.rtl and not ltr else "")
        else:
            xx, dr = self.tx(x), (' direction="rtl"' if self.rtl else "")
        (self.fixed if fixed else self.texts).append(
            f'<text x="{xx:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" font-weight="{weight}" '
            f'text-anchor="{anchor}"{dr}{st}>{s}</text>')

    def add(self, s, fixed=False):
        (self.fixed if fixed else self.shapes).append(s)

    def rect(self, x, y, w, h, fill="none", stroke="none", sw=1.5, r=6, dash=None, fixed=False):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}"{d}/>', fixed)

    def line(self, x1, y1, x2, y2, stroke=INK2, sw=1.6, dash=None, arrow=False, both=False, fixed=False):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = (' marker-start="url(#arr)"' if both else "") + (' marker-end="url(#arr)"' if arrow or both else "")
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
                 f'stroke-width="{sw}"{d}{m}/>', fixed)

    def path(self, d, stroke=INK2, sw=1.6, arrow=False):
        m = ' marker-end="url(#arr)"' if arrow else ""
        self.add(f'<path d="{d}" stroke="{stroke}" stroke-width="{sw}" fill="none"{m}/>')

    def circle(self, x, y, r, fill="none", stroke="none", sw=1.6, fixed=False):
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>', fixed)

    def badge(self, x, y, n, fill):
        self.circle(x, y, 15, fill=fill, sw=0)
        self.text(x, y + 6.5, self.num(n), size=18, fill="#ffffff", weight=700, anchor="middle")

    def svg(self):
        g = (f'<g transform="translate({W},0) scale(-1,1)">' if self.rtl else "<g>") + "\n".join(self.shapes) + "</g>"
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
                f'aria-labelledby="t d">\n<title id="t">{self.t["title"]}</title>\n'
                '<desc id="d">Panel a: four input domains, per-domain encoders, a softmax gate, graph message passing '
                'for the GNN, a linear head and the TROPOMI NO2 target; the no-graph network and SLX models take box '
                'means of the inputs instead. Panel b: the thesis compared a GNN that sees 22.7 km with SLX models that '
                'see 12.6 km, so the graph and the reach differed at once. This study gives the GNN, the same network '
                'without a graph and SLX models the same reach, varies it from 0 to 37.8 km, and evaluates all of them '
                'in the same way.</desc>\n<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
                f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker></defs>\n'
                f'<style>text {{ font-family: {self.t["font"]}; }}</style>\n<rect width="{W}" height="{H}" fill="#ffffff"/>\n'
                + g + "\n" + "\n".join(self.fixed) + "\n" + "\n".join(self.texts) + "\n</svg>\n")


def stencil(layers):
    """Cells whose information reaches the centre cell through `layers` rounds of the dilated grid graph."""
    step = [(0, 0)] + [(a * d, b * d) for d in (1, 3, 9) for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b]
    s = {(0, 0)}
    for _ in range(layers):
        s = {(x + a, y + b) for (x, y) in s for (a, b) in step}
    return s


def draw_reach(c_, cx, cy, size, half, gnn_layers=0, box_half=0, box_col=SLX, box_tint=SLX_T):
    c = size / (2 * half + 1)
    c_.rect(cx - size / 2, cy - size / 2, size, size, fill="#ffffff", stroke=GRID, sw=1, r=2)
    if box_half:
        b = (2 * box_half + 1) * c
        c_.rect(cx - b / 2, cy - b / 2, b, b, fill=box_tint, stroke=box_col, sw=1.8, r=1)
    if gnn_layers:
        d = []
        for (i, j) in stencil(gnn_layers):
            if abs(i) <= half and abs(j) <= half:
                d.append(f"M{cx + i * c - c * .36:.2f},{cy + j * c - c * .36:.2f}h{c * .72:.2f}v{c * .72:.2f}h{-c * .72:.2f}z")
        c_.add(f'<path d="{"".join(d)}" fill="{GNN}"/>')
        g = (2 * 9 * gnn_layers + 1) * c
        c_.rect(cx - g / 2, cy - g / 2, g, g, stroke=GNN, sw=1.5, r=1, dash="5 3")
    c_.rect(cx - c * .9, cy - c * .9, c * 1.8, c * 1.8, fill=INK, r=1)


def minimap(c_, x, y, w, h, seed=0.0):
    """Schematic smooth NO2 field, darker = higher; the top 10 % of cells outlined."""
    nx, ny = 14, 10
    cw, ch = w / nx, h / ny
    vals = []
    for j in range(ny):
        for i in range(nx):
            u, v = (i - 7.5) / 7.0, (j - 5.0) / 5.0
            z = math.exp(-(u * u * 1.4 + v * v * 1.1)) + 0.35 * math.exp(-((u + .55) ** 2 * 6 + (v - .4) ** 2 * 5))
            vals.append(z + 0.04 * math.sin(3.1 * i + 1.7 * j + seed))
    srt = sorted(vals)
    thr, lo, hi = srt[int(len(srt) * 0.9)], srt[0], srt[-1]
    ramp = ["#f4f3ef", "#dedcd5", "#c6c3ba", "#a9a69c", "#8a8984", "#6b6a66", "#52514e"]
    for j in range(ny):
        for i in range(nx):
            z = vals[j * nx + i]
            k = min(len(ramp) - 1, int((z - lo) / (hi - lo + 1e-9) * len(ramp)))
            c_.add(f'<rect x="{x + i * cw:.1f}" y="{y + j * ch:.1f}" width="{cw - .8:.1f}" height="{ch - .8:.1f}" fill="{ramp[k]}"/>')
            if z >= thr:
                c_.add(f'<rect x="{x + i * cw:.1f}" y="{y + j * ch:.1f}" width="{cw - .8:.1f}" height="{ch - .8:.1f}" '
                       f'fill="none" stroke="{INK}" stroke-width="1.7"/>')


def build(lang):
    c = Canvas(lang)
    t = c.t
    # ===================================================================== panel a: shared architecture
    c.text(18, 42, t["a"], size=30, weight=700)
    c.text(50 if not c.rtl else 84, 42, t["ta"], size=S4, weight=700)
    c.badge(39, 92, 1, INK2)
    c.text(62, 99, t["inputs"], size=S3, weight=700)
    ys, CW, EX = [118, 180, 242, 304], 316, 376
    for (name, sub, k), y in zip(t["cards"], ys):
        c.rect(24, y, CW, 54, fill=PAPER, stroke="#d9d7d0", sw=1)
        c.text(38, y + 23, name, size=S2, weight=700)
        c.text(24 + CW - 12, y + 23, k, size=S1, fill=INK2, anchor="end")
        c.text(38, y + 45, sub, size=S1, fill=INK2)
    c.text(EX + 46, 108, t["enc"], size=S1, fill=INK2, anchor="middle")
    for y in ys:
        c.line(24 + CW, y + 27, EX - 4, y + 27, arrow=True)
        c.rect(EX, y + 7, 92, 40, fill="#ffffff", stroke=INK2, sw=1.3)
        c.text(EX + 46, y + 33, "MLP", size=S2, anchor="middle")
    gx, gy, gw, gh = 528, 172, 140, 136
    for i, y in enumerate(ys):
        c.path(f"M{EX + 92},{y + 27} C{EX + 124},{y + 27} {EX + 124},{gy + gh / 2} {gx - 3},{gy + gh / 2}", arrow=(i == 0))
    c.rect(gx, gy, gw, gh, fill=NAVY_T, stroke=NAVY, sw=1.8, r=10)
    c.badge(gx + gw / 2, gy - 26, 2, INK2)
    c.text(gx + gw / 2, gy + 34, t["gate"][0], size=S3, weight=700, anchor="middle")
    for k, yy in ((1, 60), (2, 84), (3, 106)):
        c.text(gx + gw / 2, gy + yy, t["gate"][k], size=S1, fill=INK2, anchor="middle")
    c.text(gx + gw / 2, gy + gh + 26, t["fusion"], size=S1, fill=INK2, anchor="middle")
    NX, NW = 718, 266
    c.line(gx + gw, gy + gh / 2, NX - 4, gy + gh / 2, arrow=True)
    c.rect(NX, 160, NW, 160, fill=GNN_T, stroke=GNN, sw=1.9, r=10)
    c.text(NX + 16, 192, t["gnn"][0], size=S2, weight=700)
    c.text(NX + 16, 220, t["gnn"][1], size=S1, fill=INK2)
    c.text(NX + 16, 244, t["gnn"][2], size=S1, fill=INK2)
    c.text(NX + 16, 276, t["gnn"][3], size=S2, weight=700)
    c.text(NX + 16, 304, t["gnn"][4], size=S1, fill=INK2, italic=True)
    HX = 1022
    c.line(NX + NW, 240, HX - 4, 240, arrow=True)
    c.rect(HX, 214, 92, 52, fill="#ffffff", stroke=INK2, sw=1.3)
    c.text(HX + 46, 236, t["head"][0], size=S1, anchor="middle")
    c.text(HX + 46, 257, t["head"][1], size=S1, anchor="middle")
    c.line(HX + 92, 240, 1142, 240, arrow=True)
    c.text(1196, 176, t["pred"], size=S2, weight=700, anchor="middle")
    minimap(c, 1146, 190, 100, 72, seed=0.7)
    c.text(1340, 176, t["target"], size=S2, weight=700, anchor="middle")
    minimap(c, 1290, 190, 100, 72, seed=2.1)
    c.line(1252, 226, 1284, 226, both=True)
    c.text(1268, 214, t["mse"], size=15, fill=INK2, anchor="middle")
    c.text(1268, 290, t["mapcap"][0], size=S1, fill=INK2, anchor="middle")
    c.text(1268, 312, t["mapcap"][1], size=S1, fill=INK2, anchor="middle")
    c.rect(24, 384, 720, 70, fill=NOG_T, stroke=NOG, sw=1.5, r=8, dash="6 4")
    c.text(40, 412, t["box"][0], size=S1)
    c.text(40, 438, t["box"][1], size=S1, fill=INK2)
    c.path(f"M{EX + 46},384 L{EX + 46},362", stroke=NOG, sw=1.7, arrow=True)

    # ===================================================================== panel b: the comparison
    TOP, DX = 488, 474
    c.text(18, TOP + 12, t["b"], size=30, weight=700)
    c.text(50, TOP + 12, t["tb"], size=S4, weight=700)
    c.rect(DX + 12, TOP + 34, W - DX - 24, H - TOP - 46, fill=ZONE, r=12)
    c.line(DX, TOP + 34, DX, H - 12, stroke=NAVY, sw=2.8, dash="10 7")
    c.text(DX + 30, TOP + 64, t["study"], size=S2, fill=NAVY, weight=700)
    # thesis
    c.badge(39, TOP + 78, 3, INK2)
    c.text(62, TOP + 85, t["thesis"], size=S3, weight=700)
    draw_reach(c, 240, TOP + 226, 210, 27, gnn_layers=3, box_half=15)
    y0 = TOP + 364
    c.add(f'<rect x="56" y="{y0 - 13}" width="16" height="16" fill="{GNN}"/>')
    c.text(82, y0 + 1, t["leg"][0], size=S2)
    c.rect(56, y0 + 17, 16, 16, fill=SLX_T, stroke=SLX, sw=1.8, r=1)
    c.text(82, y0 + 31, t["leg"][1], size=S2)
    c.text(56, y0 + 76, t["ahead"][0], size=S2, weight=700)
    c.text(56, y0 + 100, t["ahead"][1], size=S2, weight=700)
    c.rect(40, y0 + 124, 412, 112, fill="#ffffff", stroke=INK2, sw=1.5, r=10)
    c.circle(70, y0 + 156, 14, stroke=INK, sw=2)
    c.text(70, y0 + 163, "!", size=20, weight=700, anchor="middle")
    c.text(96, y0 + 156, t["warn"][0], size=S2, weight=700)
    c.text(96, y0 + 182, t["warn"][1], size=S2)
    c.text(96, y0 + 206, t["warn"][2], size=S2)
    c.line(452, y0 + 180, DX - 3, y0 + 180, stroke=NAVY, sw=2.4, arrow=True)
    # this study: equal reach
    c.badge(DX + 44, TOP + 102, 4, NAVY)
    c.text(DX + 68, TOP + 109, t["same"], size=S3, weight=700)
    c.text(DX + 68, TOP + 134, t["same2"], size=S1, fill=INK2)
    xs = [660, 940, 1220]
    SY, SS = TOP + 258, 140
    for cx, (title, l1, l2), kind in zip(xs, t["cols"], ("gnn", "nog", "slx")):
        c.text(cx, TOP + 174, title, size=S2, weight=700, anchor="middle")
        if kind == "gnn":
            draw_reach(c, cx, SY, SS, 27, gnn_layers=3)
        elif kind == "nog":
            draw_reach(c, cx, SY, SS, 27, box_half=27, box_col=NOG, box_tint=NOG_T)
        else:
            draw_reach(c, cx, SY, SS, 27, box_half=27)
        c.text(cx, SY + SS / 2 + 28, l1, size=S2, anchor="middle")
        c.text(cx, SY + SS / 2 + 52, l2, size=S1, fill=INK2, anchor="middle")
    BY = SY + SS / 2 + 66
    c.path(f"M660,{BY} v10 H940 v-10", stroke=INK2, sw=1.5)
    c.text(800, BY + 32, t["only"], size=S2, weight=700, anchor="middle")
    c.path(f"M660,{BY + 44} v10 H1220 v-10", stroke=MUTED, sw=1.3)
    c.text(1080, BY + 76, t["both"], size=S1, fill=INK2, anchor="middle")
    # reach axis: kilometres always increase to the right
    RT = BY + 140
    c.text(DX + 30, RT - 40, t["varied"], size=S2, weight=700)
    if c.rtl:
        RX0, RX1 = W - 1344, W - 660
        lab_x, lab_anchor = RX1 + 18, "end"                 # row labels to the right of the axis
        pair_x = RX1 + 40
    else:
        RX0, RX1 = 660, 1344
        lab_x, lab_anchor = RX0 - 18, "end"
        pair_x = DX + 42

    def rx(km):
        return RX0 + km / 40.0 * (RX1 - RX0)

    for g in [7.6, 15.1, 22.7, 30.3, 37.8]:
        c.line(rx(g), RT - 16, rx(g), RT + 76, stroke="#d8dee6", sw=1.2, fixed=True)
    c.line(rx(24), RT - 22, rx(24), RT + 80, stroke=NAVY, sw=1.7, dash="5 4", fixed=True)
    c.text(rx(24) + (-7 if c.rtl else 7), RT - 26, t["corr"], size=S1, fill=NAVY, anchor="start", fixed=True)
    rows = [(SLX, [0, 0.8, 2.5, 7.6, 12.6, 15.1, 22.7, 30.3, 37.8]), (GNN, [7.6, 15.1, 22.7, 30.3, 37.8]),
            (NOG, [0, 22.7, 37.8])]
    for k, ((col, kms), name) in enumerate(zip(rows, t["rows"])):
        yy = RT + 4 + k * 30
        c.text(lab_x, yy + 6, name, size=S1, anchor=lab_anchor, fixed=True)
        for km in kms:
            c.circle(rx(km), yy, 7.5, fill=col, stroke=ZONE, sw=2, fixed=True)
    for km, k in ((12.6, 0), (22.7, 1)):
        c.circle(rx(km), RT + 4 + k * 30, 13, stroke=INK, sw=1.7, fixed=True)
    AY = RT + 94
    c.line(RX0, AY, RX1, AY, stroke=INK2, sw=1.3, fixed=True)
    for tk in range(0, 41, 10):
        c.line(rx(tk), AY, rx(tk), AY + 6, stroke=INK2, sw=1.3, fixed=True)
        c.text(rx(tk), AY + 26, c.num(tk), size=S1, fill=INK2, anchor="middle", fixed=True, ltr=True)
    c.text((RX0 + RX1) / 2, AY + 52, t["axis"], size=S1, fill=INK2, anchor="middle", fixed=True)
    c.circle(pair_x, AY + 20, 10, stroke=INK, sw=1.7, fixed=True)
    c.text(pair_x + 18, AY + 26, t["pair"], size=S1, fill=INK2, anchor="end" if c.rtl else "start", fixed=True)
    # evaluation
    EY = H - 90
    c.rect(DX + 26, EY, W - DX - 50, 64, fill="#ffffff", stroke="#cfd8e3", sw=1.2, r=8)
    c.text(DX + 42, EY + 26, t["ev"][0], size=S1)
    c.text(DX + 42, EY + 51, t["ev"][1], size=S1, fill=INK2)
    out = HERE / f"fig_design_{lang}.svg"
    out.write_text(c.svg(), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    for lang in TXT:
        build(lang)

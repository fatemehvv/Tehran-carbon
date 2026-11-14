"""Figure: study design (architecture and the reach-matched test). Writes fig_design_en.svg next to the other figures.

Pure Python, no dependencies; render to PDF and PNG with render_svg.js (Playwright/Chromium).
Sized for a full-width figure (180-190 mm): the smallest text is 17 px of a 1400-px canvas, about 6.2-6.5 pt.
What is drawn comes from configs/config.yaml (input layers, encoders, gate, graph dilations 1/3/9, 9 cells of reach
per GNN layer, SLX windows, no-graph box means) and Sections 2-3 of the manuscript. The maps are schematic.
"""
from __future__ import annotations

import math
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "fig_design_en.svg"
W, H = 1400, 1300
INK, INK2, MUTED, GRID, PAPER = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#f4f3ef"
GNN, SLX, NOG, NAVY = "#2a78d6", "#eb6834", "#8a8984", "#104281"
GNN_T, SLX_T, NOG_T, NAVY_T, ZONE = "#dce9f8", "#fbe3d8", "#ecebe7", "#e3eaf3", "#f3f6fa"
S1, S2, S3, S4 = 17, 18.5, 20, 23          # small, body, emphasis, heading (px)

el: list[str] = []


def add(s):
    el.append(s)


def text(x, y, s, size=S2, fill=INK, weight=400, anchor="start", italic=False):
    st = ' font-style="italic"' if italic else ""
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" font-weight="{weight}" '
        f'text-anchor="{anchor}"{st}>{s}</text>')


def rect(x, y, w, h, fill="none", stroke="none", sw=1.5, r=6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"{d}/>')


def line(x1, y1, x2, y2, stroke=INK2, sw=1.6, dash=None, arrow=False, both=False):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = (' marker-start="url(#arr)"' if both else "") + (' marker-end="url(#arr)"' if arrow or both else "")
    add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}"{d}{m}/>')


def path(d, stroke=INK2, sw=1.6, arrow=False, marker="arr"):
    m = f' marker-end="url(#{marker})"' if arrow else ""
    add(f'<path d="{d}" stroke="{stroke}" stroke-width="{sw}" fill="none"{m}/>')


def badge(x, y, n, fill):
    add(f'<circle cx="{x}" cy="{y}" r="15" fill="{fill}"/>')
    text(x, y + 6.5, str(n), size=18, fill="#ffffff", weight=700, anchor="middle")


def stencil(layers):
    """Cells whose information reaches the centre cell through `layers` rounds of the dilated grid graph."""
    step = [(0, 0)] + [(a * d, b * d) for d in (1, 3, 9) for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b]
    s = {(0, 0)}
    for _ in range(layers):
        s = {(x + a, y + b) for (x, y) in s for (a, b) in step}
    return s


def draw_reach(cx, cy, size, half, gnn_layers=0, box_half=0, box_col=SLX, box_tint=SLX_T):
    c = size / (2 * half + 1)
    rect(cx - size / 2, cy - size / 2, size, size, fill="#ffffff", stroke=GRID, sw=1, r=2)
    if box_half:
        b = (2 * box_half + 1) * c
        rect(cx - b / 2, cy - b / 2, b, b, fill=box_tint, stroke=box_col, sw=1.8, r=1)
    if gnn_layers:
        d = []
        for (i, j) in stencil(gnn_layers):
            if abs(i) <= half and abs(j) <= half:
                d.append(f"M{cx + i * c - c * 0.36:.2f},{cy + j * c - c * 0.36:.2f}h{c * .72:.2f}v{c * .72:.2f}h{-c * .72:.2f}z")
        add(f'<path d="{"".join(d)}" fill="{GNN}"/>')
        g = (2 * 9 * gnn_layers + 1) * c
        rect(cx - g / 2, cy - g / 2, g, g, stroke=GNN, sw=1.5, r=1, dash="5 3")
    rect(cx - c * 0.9, cy - c * 0.9, c * 1.8, c * 1.8, fill=INK, r=1)


def minimap(x, y, w, h, seed=0.0):
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
            add(f'<rect x="{x + i * cw:.1f}" y="{y + j * ch:.1f}" width="{cw - .8:.1f}" height="{ch - .8:.1f}" fill="{ramp[k]}"/>')
            if z >= thr:
                add(f'<rect x="{x + i * cw:.1f}" y="{y + j * ch:.1f}" width="{cw - .8:.1f}" height="{ch - .8:.1f}" '
                    f'fill="none" stroke="{INK}" stroke-width="1.7"/>')


add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

# ================================================================================ panel a: shared architecture
text(18, 42, "a", size=30, weight=700)
text(50, 42, "Inputs, fusion network and target (all neural models)", size=S4, weight=700)
badge(39, 92, 1, INK2)
text(62, 99, "Inputs: 2,436 cells × 36 months", size=S3, weight=700)
cards = [("Traffic", "road CO₂, roads, traffic", 7), ("Energy", "building, industry, power CO₂", 5),
         ("Pollution", "CO at stations", 1), ("Context", "people, buildings, lights, weather", 7)]
ys = [118, 180, 242, 304]
CW = 316
for (name, sub, k), y in zip(cards, ys):
    rect(24, y, CW, 54, fill=PAPER, stroke="#d9d7d0", sw=1)
    text(38, y + 23, name, size=S2, weight=700)
    text(24 + CW - 12, y + 23, f"{k} layer{'s' if k > 1 else ''}", size=S1, fill=INK2, anchor="end")
    text(38, y + 45, sub, size=S1, fill=INK2)
EX = 376
text(EX + 46, 108, "encoders", size=S1, fill=INK2, anchor="middle")
for y in ys:
    line(24 + CW, y + 27, EX - 4, y + 27, arrow=True)
    rect(EX, y + 7, 92, 40, fill="#ffffff", stroke=INK2, sw=1.3)
    text(EX + 46, y + 33, "MLP", size=S2, anchor="middle")
gx, gy, gw, gh = 528, 172, 140, 136
for i, y in enumerate(ys):
    path(f"M{EX + 92},{y + 27} C{EX + 124},{y + 27} {EX + 124},{gy + gh / 2} {gx - 3},{gy + gh / 2}", arrow=(i == 0))
rect(gx, gy, gw, gh, fill=NAVY_T, stroke=NAVY, sw=1.8, r=10)
badge(gx + gw / 2, gy - 26, 2, INK2)
text(gx + gw / 2, gy + 34, "Gate", size=S3, weight=700, anchor="middle")
text(gx + gw / 2, gy + 60, "softmax", size=S1, fill=INK2, anchor="middle")
text(gx + gw / 2, gy + 84, "weight per", size=S1, fill=INK2, anchor="middle")
text(gx + gw / 2, gy + 106, "cell and month", size=S1, fill=INK2, anchor="middle")
text(gx + gw / 2, gy + gh + 26, "fusion (RQ1)", size=S1, fill=INK2, anchor="middle")
NX, NW = 718, 266
line(gx + gw, gy + gh / 2, NX - 4, gy + gh / 2, arrow=True)
rect(NX, 160, NW, 160, fill=GNN_T, stroke=GNN, sw=1.9, r=10)
text(NX + 16, 192, "GNN: message passing", size=S2, weight=700)
text(NX + 16, 220, "grid graph, 1–5 layers", size=S1, fill=INK2)
text(NX + 16, 244, "neighbours at 1, 3, 9 cells", size=S1, fill=INK2)
text(NX + 16, 276, "reach = 7.6 km × layers", size=S2, weight=700)
text(NX + 16, 304, "no-graph network: none", size=S1, fill=INK2, italic=True)
HX = 1022
line(NX + NW, 240, HX - 4, 240, arrow=True)
rect(HX, 214, 92, 52, fill="#ffffff", stroke=INK2, sw=1.3)
text(HX + 46, 236, "linear", size=S1, anchor="middle")
text(HX + 46, 257, "head", size=S1, anchor="middle")
line(HX + 92, 240, 1142, 240, arrow=True)
text(1196, 176, "Predicted", size=S2, weight=700, anchor="middle")
minimap(1146, 190, 100, 72, seed=0.7)
text(1340, 176, "Target", size=S2, weight=700, anchor="middle")
minimap(1290, 190, 100, 72, seed=2.1)
line(1252, 226, 1284, 226, both=True)
text(1268, 214, "MSE", size=15, fill=INK2, anchor="middle")
text(1268, 290, "NO₂ (z), TROPOMI target", size=S1, fill=INK2, anchor="middle")
text(1268, 312, "outlined: top 10 % (hotspots)", size=S1, fill=INK2, anchor="middle")
# where the box means enter
rect(24, 384, 720, 70, fill=NOG_T, stroke=NOG, sw=1.5, r=8, dash="6 4")
text(40, 412, "No-graph network: the neighbourhood enters here, as box means of each input.", size=S1)
text(40, 438, "SLX: the same inputs and box means go to OLS, GBM or RF, without this network.", size=S1,
     fill=INK2)
path(f"M{EX + 46},384 L{EX + 46},362", stroke=NOG, sw=1.7, arrow=True)

# ================================================================================ panel b: the comparison
TOP = 488
text(18, TOP + 12, "b", size=30, weight=700)
text(50, TOP + 12, "From the thesis comparison to the test in this study", size=S4, weight=700)
DX = 474
rect(DX + 12, TOP + 34, W - DX - 24, H - TOP - 46, fill=ZONE, r=12)
line(DX, TOP + 34, DX, H - 12, stroke=NAVY, sw=2.8, dash="10 7")
text(DX + 30, TOP + 64, "THIS STUDY", size=S2, fill=NAVY, weight=700)

# thesis
badge(39, TOP + 78, 3, INK2)
text(62, TOP + 85, "Thesis: GNN against the best SLX", size=S3, weight=700)
draw_reach(240, TOP + 226, 210, 27, gnn_layers=3, box_half=15)
y0 = TOP + 364
add(f'<rect x="56" y="{y0 - 13}" width="16" height="16" fill="{GNN}"/>')
text(82, y0 + 1, "GNN, 3 layers: 22.7 km", size=S2)
rect(56, y0 + 17, 16, 16, fill=SLX_T, stroke=SLX, sw=1.8, r=1)
text(82, y0 + 31, "best SLX window: 12.6 km", size=S2)
text(56, y0 + 76, "GNN ahead in all three", size=S2, weight=700)
text(56, y0 + 100, "pre-registered tests", size=S2, weight=700)
rect(40, y0 + 124, 412, 112, fill="#ffffff", stroke=INK2, sw=1.5, r=10)
add(f'<circle cx="70" cy="{y0 + 156}" r="14" fill="none" stroke="{INK}" stroke-width="2"/>')
text(70, y0 + 163, "!", size=20, weight=700, anchor="middle")
text(96, y0 + 156, "Two things differed at once:", size=S2, weight=700)
text(96, y0 + 182, "how neighbours are combined", size=S2)
text(96, y0 + 206, "and how far the model sees.", size=S2)
line(452, y0 + 180, DX - 3, y0 + 180, stroke=NAVY, sw=2.4, arrow=True)

# this study: equal reach
badge(DX + 44, TOP + 102, 4, NAVY)
text(DX + 68, TOP + 109, "Same reach, one difference at a time", size=S3, weight=700)
text(DX + 68, TOP + 134, "shown at 22.7 km; also compared at 7.6, 15.1, 30.3 and 37.8 km", size=S1, fill=INK2)
cols = [(660, "GNN", "learned messages", "neural network", "gnn"),
        (940, "Same network, no graph", "fixed box means", "same neural network", "nog"),
        (1220, "SLX: OLS, GBM, RF", "fixed box means", "linear model or trees", "slx")]
SY, SS = TOP + 258, 140
for cx, title, l1, l2, kind in cols:
    text(cx, TOP + 174, title, size=S2, weight=700, anchor="middle")
    if kind == "gnn":
        draw_reach(cx, SY, SS, 27, gnn_layers=3)
    elif kind == "nog":
        draw_reach(cx, SY, SS, 27, box_half=27, box_col=NOG, box_tint=NOG_T)
    else:
        draw_reach(cx, SY, SS, 27, box_half=27)
    text(cx, SY + SS / 2 + 28, l1, size=S2, anchor="middle")
    text(cx, SY + SS / 2 + 52, l2, size=S1, fill=INK2, anchor="middle")
BY = SY + SS / 2 + 66
path(f"M660,{BY} v10 H940 v-10", stroke=INK2, sw=1.5)
text(800, BY + 32, "only the graph differs", size=S2, weight=700, anchor="middle")
path(f"M660,{BY + 44} v10 H1220 v-10", stroke=MUTED, sw=1.3)
text(1080, BY + 76, "graph and learner differ", size=S1, fill=INK2, anchor="middle")

# reach varied
RX0, RX1 = 660, 1344


def rx(km):
    return RX0 + km / 40.0 * (RX1 - RX0)


RT = BY + 140
text(DX + 30, RT - 40, "Reach varied for every model", size=S2, weight=700)
for g in [7.6, 15.1, 22.7, 30.3, 37.8]:
    line(rx(g), RT - 16, rx(g), RT + 76, stroke="#d8dee6", sw=1.2)
line(rx(24), RT - 22, rx(24), RT + 80, stroke=NAVY, sw=1.7, dash="5 4")
text(rx(24) + 7, RT - 26, "target correlation length (24 km)", size=S1, fill=NAVY)
rows = [("SLX", SLX, [0, 0.8, 2.5, 7.6, 12.6, 15.1, 22.7, 30.3, 37.8]),
        ("GNN", GNN, [7.6, 15.1, 22.7, 30.3, 37.8]),
        ("No graph", NOG, [0, 22.7, 37.8])]
for k, (name, col, kms) in enumerate(rows):
    yy = RT + 4 + k * 30
    text(RX0 - 18, yy + 6, name, size=S1, anchor="end")
    for km in kms:
        add(f'<circle cx="{rx(km):.1f}" cy="{yy}" r="7.5" fill="{col}" stroke="{ZONE}" stroke-width="2"/>')
for km, k in ((12.6, 0), (22.7, 1)):
    add(f'<circle cx="{rx(km):.1f}" cy="{RT + 4 + k * 30}" r="13" fill="none" stroke="{INK}" stroke-width="1.7"/>')
AY = RT + 94
line(RX0, AY, RX1, AY, stroke=INK2, sw=1.3)
for t in range(0, 41, 10):
    line(rx(t), AY, rx(t), AY + 6, stroke=INK2, sw=1.3)
    text(rx(t), AY + 26, str(t), size=S1, fill=INK2, anchor="middle")
text((RX0 + RX1) / 2, AY + 52, "reach (km): how far from a cell a prediction can look", size=S1, fill=INK2,
     anchor="middle")
add(f'<circle cx="{DX + 42}" cy="{AY + 20}" r="10" fill="none" stroke="{INK}" stroke-width="1.7"/>')
text(DX + 60, AY + 26, "thesis pair", size=S1, fill=INK2)

# evaluation
EY = H - 90
rect(DX + 26, EY, W - DX - 50, 64, fill="#ffffff", stroke="#cfd8e3", sw=1.2, r=8)
text(DX + 42, EY + 26, "Same evaluation for all: 12 blocks of 22 km, 2-km buffer · held-out year 1404 · 5 seeds",
     size=S1)
text(DX + 42, EY + 51, "Spearman (primary), PR-AUC · permutation test, jackknife, ±0.02 equivalence, Holm",
     size=S1, fill=INK2)

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
       'aria-labelledby="t d">\n<title id="t">Study design: architecture and the reach-matched test</title>\n'
       '<desc id="d">Panel a: four input domains, per-domain encoders, a softmax gate, graph message passing for '
       'the GNN, a linear head and the TROPOMI NO2 target; the no-graph network and SLX models take box means of '
       'the inputs instead. Panel b: the thesis compared a GNN that sees 22.7 km with SLX models that see 12.6 km, '
       'so the graph and the reach differed at once. This study gives the GNN, the same network without a graph '
       'and SLX models the same reach, varies it from 0 to 37.8 km, and evaluates all of them in the same way.'
       '</desc>\n<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
       f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker></defs>\n'
       '<style>text { font-family: "Liberation Sans", Arial, Helvetica, sans-serif; }</style>\n'
       + "\n".join(el) + "\n</svg>\n")
OUT.write_text(svg, encoding="utf-8")
print(OUT)

"""Graphical abstract (Elsevier: at least 1328 x 531 px, landscape). Writes ../graphical_abstract.svg.

The chart uses the committed reach curve (docs/paper/results_review/reach/reach_curve.csv): Spearman, mean of 1402
and 1403, five-seed ensembles for the networks and the best of OLS, GBM and RF for SLX at each reach.
Render with: node render_svg.js ../graphical_abstract.svg
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from fig_design import GNN, SLX, NOG, NAVY, INK, INK2, MUTED, GRID, SLX_T, NOG_T, ZONE, stencil

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1] / "graphical_abstract.svg"
W, H = 1600, 690
el = []


def add(s):
    el.append(s)


def text(x, y, s, size=24, fill=INK, weight=400, anchor="start"):
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}">{s}</text>')


def rect(x, y, w, h, fill="none", stroke="none", sw=1.5, r=4, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def reach(cx, cy, size, half, layers=0, box=0, col=SLX, tint=SLX_T):
    c = size / (2 * half + 1)
    rect(cx - size / 2, cy - size / 2, size, size, fill="#ffffff", stroke=GRID, sw=1, r=2)
    if box:
        b = (2 * box + 1) * c
        rect(cx - b / 2, cy - b / 2, b, b, fill=tint, stroke=col, sw=2.2, r=1)
    if layers:
        d = "".join(f"M{cx + i * c - c * .36:.2f},{cy + j * c - c * .36:.2f}h{c * .72:.2f}v{c * .72:.2f}h{-c * .72:.2f}z"
                    for (i, j) in stencil(layers) if abs(i) <= half and abs(j) <= half)
        add(f'<path d="{d}" fill="{GNN}"/>')
        g = (2 * 9 * layers + 1) * c
        rect(cx - g / 2, cy - g / 2, g, g, stroke=GNN, sw=1.8, r=1, dash="6 4")
    rect(cx - c, cy - c, 2 * c, 2 * c, fill=INK, r=1)


def arrow(x1, x2, y):
    add(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{NAVY}" stroke-width="5" marker-end="url(#arr)"/>')


# data
vals = defaultdict(lambda: defaultdict(float))
for row in csv.DictReader(open(ROOT / "docs/paper/results_review/reach/reach_curve.csv", encoding="utf-8-sig")):
    fam = "SLX" if row["family"].startswith("SLX") else row["family"]
    k = round(float(row["reach_km"]), 1)
    v = float(row["spearman"])
    vals[(fam, k)][row["year"]] = max(vals[(fam, k)][row["year"]], v) if fam == "SLX" else v
series = {}
for (fam, k), d in vals.items():
    series.setdefault(fam, []).append((k, (d["1402"] + d["1403"]) / 2))
for fam in series:
    series[fam].sort()

add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
text(W / 2, 54, "Why do graph neural networks beat simple spatial models in mapping urban NO₂?", size=32, weight=700,
     anchor="middle")
text(W / 2, 92, "Intra-urban NO₂ hotspots from TROPOMI, Tehran, 2,436 cells of 1 km", size=23, fill=INK2, anchor="middle")

# 1. the usual comparison
X1 = 255
text(X1, 150, "The usual comparison", size=27, weight=700, anchor="middle")
reach(X1, 318, 250, 27, layers=3, box=15)
add(f'<rect x="{X1 - 150}" y="{466}" width="18" height="18" fill="{GNN}"/>')
text(X1 - 122, 483, "GNN sees 22.7 km", size=23)
rect(X1 - 150, 500, 18, 18, fill=SLX_T, stroke=SLX, sw=2.2, r=1)
text(X1 - 122, 517, "SLX baseline sees 12.6 km", size=23)
text(X1, 572, "The GNN wins, but sees farther:", size=23, weight=700, anchor="middle")
text(X1, 602, "graph and reach differ at once", size=23, fill=INK2, anchor="middle")
arrow(498, 560, 330)

# 2. this study
rect(575, 122, 470, 520, fill=ZONE, r=14)
X2 = 810
text(X2, 160, "This study: same reach", size=27, weight=700, fill=NAVY, anchor="middle")
for cx, lab, kind in ((655, "GNN", "g"), (810, "no graph", "n"), (965, "SLX", "s")):
    if kind == "g":
        reach(cx, 268, 128, 27, layers=3)
    elif kind == "n":
        reach(cx, 268, 128, 27, box=27, col=NOG, tint=NOG_T)
    else:
        reach(cx, 268, 128, 27, box=27)
    text(cx, 362, lab, size=24, weight=700, anchor="middle")
text(X2, 412, "Same network with and without", size=23, anchor="middle")
text(X2, 442, "the graph, and SLX models,", size=23, anchor="middle")
text(X2, 472, "all given the same reach", size=23, anchor="middle")
text(X2, 530, "reach varied from 0 to 38 km", size=22, fill=INK2, anchor="middle")
text(X2, 562, "12 spatial blocks + a held-out year", size=22, fill=INK2, anchor="middle")
arrow(1052, 1110, 330)

# 3. result chart
X0, X1c, Y0, Y1 = 1190, 1560, 520, 190            # plot box
y_lo, y_hi = 0.65, 1.0


def px(km):
    return X0 + km / 40 * (X1c - X0)


def py(v):
    return Y0 - (v - y_lo) / (y_hi - y_lo) * (Y0 - Y1)


text(1370, 150, "Result: reach, not the graph", size=27, weight=700, fill=NAVY, anchor="middle")
for g in (0.7, 0.8, 0.9):
    add(f'<line x1="{X0}" y1="{py(g):.1f}" x2="{X1c}" y2="{py(g):.1f}" stroke="{GRID}" stroke-width="1.5"/>')
    text(X0 - 10, py(g) + 8, f"{g:.1f}", size=21, fill=INK2, anchor="end")
add(f'<line x1="{px(24):.1f}" y1="{Y1}" x2="{px(24):.1f}" y2="{Y0}" stroke="{NAVY}" stroke-width="2" stroke-dasharray="6 5"/>')
text(px(24) + 8, Y0 - 44, "target correlation", size=20, fill=NAVY)
text(px(24) + 8, Y0 - 20, "length, 24 km", size=20, fill=NAVY)
add(f'<line x1="{X0}" y1="{Y0}" x2="{X1c}" y2="{Y0}" stroke="{INK2}" stroke-width="1.5"/>')
for tk in (0, 10, 20, 30, 40):
    text(px(tk), Y0 + 30, str(tk), size=21, fill=INK2, anchor="middle")
text((X0 + X1c) / 2, Y0 + 62, "reach (km)", size=22, fill=INK2, anchor="middle")
add(f'<text x="{X0 - 62}" y="{(Y0 + Y1) / 2:.1f}" font-size="22" fill="{INK2}" text-anchor="middle" '
    f'transform="rotate(-90 {X0 - 62} {(Y0 + Y1) / 2:.1f})">skill (Spearman)</text>')
for fam, col, dash in (("SLX", SLX, None), ("SAGE", GNN, None), ("MLP", NOG, "8 6")):
    pts = series[fam]
    if not dash:                                  # the no-graph network has three reaches: points only
        d = " ".join(f"{px(k):.1f},{py(v):.1f}" for k, v in pts)
        add(f'<polyline points="{d}" fill="none" stroke="{col}" stroke-width="3.5"/>')
    for k, v in pts:
        add(f'<circle cx="{px(k):.1f}" cy="{py(v):.1f}" r="{8 if dash else 6.5}" fill="{col}" stroke="#ffffff" stroke-width="2"/>')
text(px(3), py(0.80) - 4, "SLX", size=22, weight=700)
text(px(8.6), py(0.905) + 30, "GNN", size=22, weight=700)
text(px(1.5), py(0.681) + 8, "no graph", size=22, weight=700)
text(px(22.7) - 12, py(0.954) - 16, "no graph", size=20, fill=INK2, anchor="end")
text(1375, 638, "All models follow one curve;", size=22, weight=700, anchor="middle")
text(1375, 666, "at equal reach the graph adds nothing", size=22, anchor="middle")

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
       'aria-labelledby="t d">\n<title id="t">Reach, not the graph: graphical abstract</title>\n'
       '<desc id="d">Left: in the usual comparison the GNN sees 22.7 km and the SLX baseline 12.6 km. Middle: this '
       'study gives the GNN, the same network without a graph and SLX models the same reach. Right: skill against '
       'reach; all models rise and level off on one curve near the target correlation length, so at equal reach '
       'the graph adds nothing.</desc>\n<defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="4" '
       f'markerHeight="4" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{NAVY}"/></marker></defs>\n'
       '<style>text { font-family: "Liberation Sans", Arial, Helvetica, sans-serif; }</style>\n'
       + "\n".join(el) + "\n</svg>\n")
OUT.write_text(svg, encoding="utf-8")
print(OUT)

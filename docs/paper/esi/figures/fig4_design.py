"""Fig. 4 of the Earth Science Informatics version: the study design, laid out for a full-page figure.

The content is that of docs/paper/figures/src/fig_design.py (English), re-arranged on a 1050 x 1410 canvas so that the
smallest text (17.5 px) prints at 8 pt when the figure is 174 mm wide, as Springer asks (8-12 pt at final size).
Panel a wraps the network onto two rows; panel b puts the thesis comparison above the test of this study. Two labels
follow the article: "message passing vs box means" (the two networks also differ in which cells they use and where
they aggregate) and "correlogram zero crossing (24 km)". The drawing helpers are imported from the original script.

    python docs/paper/esi/figures/fig4_design.py && node docs/paper/figures/src/render_svg.js docs/paper/esi/figures/Fig4.svg
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "figures" / "src"))
import fig_design as fd  # noqa: E402

W, H = 1050, 1410
fd.W, fd.H = W, H                                     # Canvas.svg() and the mirroring read the module globals
S1, S2, S3, S4 = 17.5, 19, 20.5, 23
INK, INK2, MUTED, GRID, PAPER = fd.INK, fd.INK2, fd.MUTED, fd.GRID, fd.PAPER
GNN, SLX, NOG, NAVY = fd.GNN, fd.SLX, fd.NOG, fd.NAVY
GNN_T, SLX_T, NOG_T, NAVY_T, ZONE = fd.GNN_T, fd.SLX_T, fd.NOG_T, fd.NAVY_T, fd.ZONE


def build() -> Path:
    c = fd.Canvas("en")
    t = c.t
    # ================================================================ panel a: shared network, two rows
    c.text(14, 40, "a", size=30, weight=700)
    c.text(44, 40, t["ta"], size=S4, weight=700)
    c.badge(36, 88, 1, INK2)
    c.text(58, 95, t["inputs"], size=S3, weight=700)
    ys, CW, EX = [114, 176, 238, 300], 320, 372
    for (name, sub, k), y in zip(t["cards"], ys):
        c.rect(20, y, CW, 54, fill=PAPER, stroke="#d9d7d0", sw=1)
        c.text(32, y + 23, name, size=S2, weight=700)
        c.text(20 + CW - 10, y + 23, k, size=S1, fill=INK2, anchor="end")
        c.text(32, y + 45, sub, size=S1, fill=INK2)
    c.text(EX + 46, 104, t["enc"], size=S1, fill=INK2, anchor="middle")
    for y in ys:
        c.line(20 + CW, y + 27, EX - 4, y + 27, arrow=True)
        c.rect(EX, y + 7, 92, 40, fill="#ffffff", stroke=INK2, sw=1.3)
        c.text(EX + 46, y + 33, "MLP", size=S2, anchor="middle")
    gx, gy, gw, gh = 500, 166, 140, 140
    for i, y in enumerate(ys):
        c.path(f"M{EX + 92},{y + 27} C{EX + 118},{y + 27} {EX + 118},{gy + gh / 2} {gx - 3},{gy + gh / 2}", arrow=(i == 0))
    c.rect(gx, gy, gw, gh, fill=NAVY_T, stroke=NAVY, sw=1.8, r=10)
    c.badge(gx + gw / 2, gy - 26, 2, INK2)
    c.text(gx + gw / 2, gy + 34, t["gate"][0], size=S3, weight=700, anchor="middle")
    for k, yy in ((1, 62), (2, 87), (3, 111)):
        c.text(gx + gw / 2, gy + yy, t["gate"][k], size=S1, fill=INK2, anchor="middle")
    c.text(gx + gw / 2, gy + gh + 26, t["fusion"], size=S1, fill=INK2, anchor="middle")
    NX, NW, NY, NH = 676, 354, 150, 172
    c.line(gx + gw, gy + gh / 2, NX - 4, gy + gh / 2, arrow=True)
    c.rect(NX, NY, NW, NH, fill=GNN_T, stroke=GNN, sw=1.9, r=10)
    c.text(NX + 16, NY + 34, t["gnn"][0], size=S2, weight=700)
    c.text(NX + 16, NY + 62, t["gnn"][1], size=S1, fill=INK2)
    c.text(NX + 16, NY + 87, t["gnn"][2], size=S1, fill=INK2)
    c.text(NX + 16, NY + 120, t["gnn"][3], size=S2, weight=700)
    c.text(NX + 16, NY + 150, t["gnn"][4], size=S1, fill=INK2, italic=True)
    # second row: linear head under the GNN box, then prediction and target
    HX, HY = 690, 380
    c.line(HX + 46, NY + NH, HX + 46, HY - 4, arrow=True)
    c.rect(HX, HY, 92, 52, fill="#ffffff", stroke=INK2, sw=1.3)
    c.text(HX + 46, HY + 22, t["head"][0], size=S1, anchor="middle")
    c.text(HX + 46, HY + 43, t["head"][1], size=S1, anchor="middle")
    MX1, MX2, MY, MW, MH = 822, 942, 388, 84, 60
    c.line(HX + 92, HY + 26, MX1 - 4, HY + 26, arrow=True)
    c.text(MX1 + MW / 2, MY - 12, t["pred"], size=S2, weight=700, anchor="middle")
    fd.minimap(c, MX1, MY, MW, MH, seed=0.7)
    c.text(MX2 + MW / 2, MY - 12, t["target"], size=S2, weight=700, anchor="middle")
    fd.minimap(c, MX2, MY, MW, MH, seed=2.1)
    c.line(MX1 + MW + 4, MY + 30, MX2 - 4, MY + 30, both=True)
    c.text((MX1 + MW + MX2) / 2, MY + 20, t["mse"], size=15, fill=INK2, anchor="middle")
    c.text(900, MY + MH + 30, t["mapcap"][0], size=S1, fill=INK2, anchor="middle")
    c.text(900, MY + MH + 53, t["mapcap"][1], size=S1, fill=INK2, anchor="middle")
    # where the neighbourhood enters without a graph
    c.rect(20, 380, 610, 100, fill=NOG_T, stroke=NOG, sw=1.5, r=8, dash="6 4")
    c.text(34, 408, "No-graph network: the neighbourhood enters here,", size=S1)
    c.text(34, 431, "as box means of each input.", size=S1)
    c.text(34, 461, "SLX: the same inputs and box means go to OLS, GBM or RF.", size=S1, fill=INK2)
    c.path(f"M{EX + 46},380 L{EX + 46},358", stroke=NOG, sw=1.7, arrow=True)

    # ================================================================ panel b, top: the thesis comparison
    TOP = 520
    c.text(14, TOP + 12, "b", size=30, weight=700)
    c.text(44, TOP + 12, t["tb"], size=S4, weight=700)
    c.badge(36, TOP + 58, 3, INK2)
    c.text(58, TOP + 65, t["thesis"], size=S3, weight=700)
    fd.draw_reach(c, 112, TOP + 158, 150, 27, gnn_layers=3, box_half=15)
    LX = 230
    c.add(f'<rect x="{LX}" y="{TOP + 101}" width="16" height="16" fill="{GNN}"/>')
    c.text(LX + 26, TOP + 115, t["leg"][0], size=S2)
    c.rect(LX, TOP + 131, 16, 16, fill=SLX_T, stroke=SLX, sw=1.8, r=1)
    c.text(LX + 26, TOP + 145, t["leg"][1], size=S2)
    c.text(LX, TOP + 192, t["ahead"][0], size=S2, weight=700)
    c.text(LX, TOP + 216, t["ahead"][1], size=S2, weight=700)
    WX, WY, WW, WH = 600, TOP + 92, 430, 112
    c.rect(WX, WY, WW, WH, fill="#ffffff", stroke=INK2, sw=1.5, r=10)
    c.circle(WX + 30, WY + 32, 14, stroke=INK, sw=2)
    c.text(WX + 30, WY + 39, "!", size=20, weight=700, anchor="middle")
    c.text(WX + 56, WY + 32, t["warn"][0], size=S2, weight=700)
    c.text(WX + 56, WY + 58, t["warn"][1], size=S2)
    c.text(WX + 56, WY + 82, t["warn"][2], size=S2)

    # ================================================================ panel b, bottom: this study
    ZY = TOP + 242
    c.rect(12, ZY, W - 24, H - ZY - 8, fill=ZONE, r=12)
    c.line(WX + WW / 2, WY + WH, WX + WW / 2, ZY + 18, stroke=NAVY, sw=2.4, arrow=True)
    c.text(30, ZY + 32, t["study"], size=S2, fill=NAVY, weight=700)
    c.badge(48, ZY + 66, 4, NAVY)
    c.text(72, ZY + 73, t["same"], size=S3, weight=700)
    c.text(72, ZY + 98, t["same2"], size=S1, fill=INK2)
    xs = [215, 525, 835]
    SY, SS = ZY + 186, 110
    for cx, (title, l1, l2), kind in zip(xs, t["cols"], ("gnn", "nog", "slx")):
        c.text(cx, ZY + 128, title, size=S2, weight=700, anchor="middle")
        if kind == "gnn":
            fd.draw_reach(c, cx, SY, SS, 27, gnn_layers=3)
        elif kind == "nog":
            fd.draw_reach(c, cx, SY, SS, 27, box_half=27, box_col=NOG, box_tint=NOG_T)
        else:
            fd.draw_reach(c, cx, SY, SS, 27, box_half=27)
        c.text(cx, SY + SS / 2 + 26, l1, size=S2, anchor="middle")
        c.text(cx, SY + SS / 2 + 49, l2, size=S1, fill=INK2, anchor="middle")
    BY = SY + SS / 2 + 58
    c.path(f"M{xs[0]},{BY} v10 H{xs[1]} v-10", stroke=INK2, sw=1.5)
    c.text((xs[0] + xs[1]) / 2, BY + 30, "message passing vs box means", size=S2, weight=700, anchor="middle")
    c.path(f"M{xs[0]},{BY + 40} v10 H{xs[2]} v-10", stroke=MUTED, sw=1.3)
    c.text((xs[0] + xs[2]) / 2, BY + 70, t["both"], size=S1, fill=INK2, anchor="middle")
    # reach axis
    RT = BY + 112
    c.text(30, RT - 36, t["varied"], size=S2, weight=700)
    RX0, RX1 = 150, 1010

    def rx(km):
        return RX0 + km / 40.0 * (RX1 - RX0)

    for g in [7.6, 15.1, 22.7, 30.3, 37.8]:
        c.line(rx(g), RT - 14, rx(g), RT + 70, stroke="#d8dee6", sw=1.2)
    c.line(rx(24), RT - 20, rx(24), RT + 74, stroke=NAVY, sw=1.7, dash="5 4")
    c.text(rx(24) + 7, RT - 26, "correlogram zero crossing (24 km)", size=S1, fill=NAVY)
    rows = [(SLX, [0, 0.8, 2.5, 7.6, 12.6, 15.1, 22.7, 30.3, 37.8]), (GNN, [7.6, 15.1, 22.7, 30.3, 37.8]),
            (NOG, [0, 22.7, 37.8])]
    for k, ((col, kms), name) in enumerate(zip(rows, t["rows"])):
        yy = RT + 4 + k * 28
        c.text(RX0 - 18, yy + 6, name, size=S1, anchor="end")
        for km in kms:
            c.circle(rx(km), yy, 7.5, fill=col, stroke=ZONE, sw=2)
    for km, k in ((12.6, 0), (22.7, 1)):
        c.circle(rx(km), RT + 4 + k * 28, 13, stroke=INK, sw=1.7)
    AY = RT + 90
    c.line(RX0, AY, RX1, AY, stroke=INK2, sw=1.3)
    for tk in range(0, 41, 10):
        c.line(rx(tk), AY, rx(tk), AY + 6, stroke=INK2, sw=1.3)
        c.text(rx(tk), AY + 26, str(tk), size=S1, fill=INK2, anchor="middle")
    c.text((RX0 + RX1) / 2, AY + 48, t["axis"], size=S1, fill=INK2, anchor="middle")
    c.circle(42, AY + 42, 10, stroke=INK, sw=1.7)
    c.text(60, AY + 48, t["pair"], size=S1, fill=INK2)
    # evaluation
    EY = H - 70
    c.rect(26, EY, W - 52, 56, fill="#ffffff", stroke="#cfd8e3", sw=1.2, r=8)
    c.text(40, EY + 23, "Same evaluation for all: 12 blocks of 22 km, 2-km buffer · held-out year 1404 · five seeds",
           size=S1)
    c.text(40, EY + 46, "Spearman (primary), PR-AUC · sign-flip test, jackknife, ±0.02 equivalence, Holm",
           size=S1, fill=INK2)
    out = HERE / "Fig4.svg"
    out.write_text(c.svg(), encoding="utf-8")
    print(out, f"{W} x {H} px; smallest text {S1} px = {S1 * 174 / W * 72 / 25.4:.1f} pt at 174 mm")
    return out


if __name__ == "__main__":
    build()

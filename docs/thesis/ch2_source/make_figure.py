# -*- coding: utf-8 -*-
"""Figure 1 of chapter 2 (gap matrix, two panels) as HTML. Usage: python make_figure.py <font dir> <out.html>
Then: node shot.js <out.html> <out.png>  (Playwright screenshot of .wrap at device scale 2)."""
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import content as C  # noqa: E402

FONT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
MARK = {"yes": '<span class="y">●</span>', "no": '<span class="n">○</span>', "nr": '<span class="r">–</span>'}
faces = "".join(f"@font-face {{ font-family:'Vazirmatn'; font-weight:{w}; src:url('file://{FONT}/Vazirmatn-{n}.ttf'); }}\n"
                for w, n in ((400, "Regular"), (500, "Medium"), (600, "SemiBold"), (700, "Bold")))
css = faces + """
body { margin:0; background:#fff; font-family:'Vazirmatn'; color:#1f2328; }
.wrap { width:1200px; padding:24px 30px 20px; box-sizing:border-box; }
h2 { font-size:24px; font-weight:700; margin:6px 0 6px; }
h2.second { margin-top:30px; }
table { border-collapse:collapse; width:100%; table-layout:fixed; }
th { font-weight:600; font-size:19px; line-height:1.4; padding:10px 8px; border-bottom:2px solid #1f2328; vertical-align:bottom; }
th:first-child { width:310px; }
td { text-align:center; font-size:26px; padding:6px 4px; border-bottom:1px solid #d0d7de; }
td.name { text-align:right; font-size:20px; font-weight:500; padding-right:6px; }
tr.me td { background:#FFD6C1; font-weight:700; border-bottom:2px solid #1f2328; }
.y { color:#1f2328; } .n { color:#57606a; } .r { color:#8c959f; }
.legend { margin-top:16px; font-size:19px; color:#424a53; text-align:center; word-spacing:2px; }
"""
parts = []
for i, (title, cols, rows) in enumerate(C.FIGURE_PANELS):
    head = "".join(f"<th>{html.escape(c)}</th>" for c in ["پژوهش"] + cols)
    body = "".join(f'<tr class="{"me" if name == "این پژوهش" else ""}"><td class="name">{html.escape(name)}</td>'
                   + "".join(f"<td>{MARK[v]}</td>" for v in vals) + "</tr>" for name, vals in rows)
    parts.append(f'<h2 class="{"second" if i else ""}">({html.escape(title)}</h2><table><thead><tr>{head}</tr></thead>'
                 f"<tbody>{body}</tbody></table>")
legend = "".join(f'<span style="margin:0 16px">{html.escape(s.strip())}</span>' for s in C.FIGURE_LEGEND.split("      "))
OUT.write_text(f'<!doctype html><html lang="fa" dir="rtl"><head><meta charset="utf-8"><style>{css}</style></head><body>'
               f'<div class="wrap">{"".join(parts)}<div class="legend">{legend}</div></div></body></html>', encoding="utf-8")
print(OUT)

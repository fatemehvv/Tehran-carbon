# -*- coding: utf-8 -*-
"""Insert chapter 2 (all four research questions) into the full chapters 3-5 thesis file, renumber every citation in
order of first appearance (IEEE) and rebuild the reference list.

Usage: python build.py <unpacked full file (review fixes applied)> <output dir> <figure.jpg> <out.ris> <lit cache dir>
The lit cache dir holds meta.json (Crossref/OpenAlex metadata by DOI) and arxiv.json (arXiv metadata by id), used only
for the RIS file."""
import html
import json
import re
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import content as C  # noqa: E402

SRC, OUT, FIG, RIS, LIT = map(Path, sys.argv[1:6])
from docxutil import EN, FA, body_elems, esc, fa, ptext  # noqa: E402

if OUT.exists():
    shutil.rmtree(OUT)
shutil.copytree(SRC, OUT)
x = (OUT / "word/document.xml").read_text(encoding="utf-8")
fx = (OUT / "word/footnotes.xml").read_text(encoding="utf-8")
els = body_elems(x)
P = [x[s:e] for s, e, t in els]


def find(pred, start=0):
    return next(i for i in range(start, len(P)) if pred(P[i]))


# ------------------------------------------------------------------ locate the template chapter 2 by its text
h_ch2 = find(lambda p: 'w:val="Heading1"' in p and ptext(p) == "مروری بر منابع")
h_ch3 = find(lambda p: 'w:val="Heading1"' in p and ptext(p) == "روش پژوهش", h_ch2)
k_sect = find(lambda p: "<w:sectPr" in p, h_ch2)
assert k_sect < h_ch3
k_fig = find(lambda p: "w:drawing" in p, h_ch2)
k_capf, k_capt = k_fig + 1, find(lambda p: 'w:val="CaptionforTables"' in p, h_ch2)
k_tbl = k_capt + 1
assert "SEQ" in P[k_capf] and "SEQ" in P[k_capt] and els[k_tbl][2] == "tbl" and k_tbl < k_sect
fig_para, cap_fig, cap_tab, tbl = P[k_fig], P[k_capf], P[k_capt], P[k_tbl]
R0, R1 = els[h_ch2 + 1][0], els[k_sect][1]
region = x[R0:R1]
sect = re.search(r"<w:pPr>.*?</w:pPr>", P[k_sect], flags=re.S).group(0)

h_ref = find(lambda p: 'w:val="Heading1"' in p and ptext(p) == "مراجع")
ref_k = [k for k in range(h_ref + 1, len(P)) if re.match(r"\[\d+\]", ptext(P[k]))]
old_text = {}
for k in ref_k:
    m = re.match(r"\[(\d+)\](.*)", ptext(P[k]), flags=re.S)
    old_text[int(m.group(1))] = esc(html.unescape(m.group(2).strip()))
N_OLD = len(old_text)
assert sorted(old_text) == list(range(1, N_OLD + 1)), sorted(old_text)
L0, L1 = els[ref_k[0]][0], els[ref_k[-1]][1]
for n, key in C.OLD_KEYS.items():                                      # the map must name the right paper
    assert re.match(r"[a-z]+", key).group(0)[:5] in html.unescape(old_text[n]).lower().replace(" ", "") or key == "tfl2004", (n, key)

# ------------------------------------------------------------------ citation order
CITE = re.compile(r"\[\[([^\]]+)\]\]")
order = []


def see(key: str) -> None:
    if key not in order:
        order.append(key)


for kind, text in C.BLOCKS:
    for m in CITE.finditer(text or ""):
        for k in m.group(1).split(","):
            see(k.strip())
for row in C.TABLE_ROWS:
    for cell in row:
        for m in CITE.finditer(cell):
            for k in m.group(1).split(","):
                see(k.strip())
key_of_old = {n: C.OLD_KEYS.get(n, f"old{n}") for n in range(1, N_OLD + 1)}
PCITE = re.compile(r"\[([۰-۹]+)\]")
assert not PCITE.findall(ptext(x[:R0])), "citations before chapter 2"
rest = x[R1:L0]
for t in re.findall(r"<w:t(?: [^>]*)?>([^<]*)</w:t>", rest):
    for m in PCITE.finditer(t):
        see(key_of_old[int(m.group(1).translate(EN))])
num = {k: i + 1 for i, k in enumerate(order)}
missing = [k for k in order if not k.startswith("old") and k not in C.NEW_REFS and k not in C.OLD_KEYS.values()]
assert not missing, f"no reference text for {missing}"
lost = {n for n in range(1, N_OLD + 1) if key_of_old[n] not in num}
assert lost == C.DROPPED, f"unexpected change in uncited references: {sorted(lost ^ C.DROPPED)}"

# ------------------------------------------------------------------ footnotes
fid0 = max(int(i) for i in re.findall(r'<w:footnote [^>]*w:id="(\d+)"', fx)) + 1
new_fn = []


def footnote(text: str) -> int:
    fid = fid0 + len(new_fn)
    new_fn.append(f'<w:footnote w:id="{fid}"><w:p><w:pPr><w:pStyle w:val="FootnoteText"/><w:bidi w:val="0"/></w:pPr>'
                  f'<w:r><w:rPr><w:rStyle w:val="FootnoteReference"/></w:rPr><w:footnoteRef/></w:r>'
                  f'<w:r><w:t xml:space="preserve"> {esc(text)}</w:t></w:r></w:p></w:footnote>')
    return fid


# ------------------------------------------------------------------ paragraphs
def run(text: str, bold: bool = False) -> str:
    b = "<w:b/><w:bCs/>" if bold else ""
    return (f'<w:r><w:rPr><w:rFonts w:hint="cs"/>{b}<w:rtl/><w:lang w:bidi="fa-IR"/></w:rPr>'
            f'<w:t xml:space="preserve">{esc(text)}</w:t></w:r>')


def cites(keys: str) -> str:
    return "، ".join(f"[{fa(num[k.strip()])}]" for k in keys.split(","))


def runs_of(text: str, notes: bool = True) -> str:
    out = []
    text = text.replace(" [[", " [[")
    m = re.match(r"\*\*(.+?)\*\*\s*", text)
    if m:
        out += [run(m.group(1), bold=True), run(" ")]
        text = text[m.end():]
    for tok in re.split(r"(\[\[[^\]]+\]\]|\{\{[^}]+\}\})", text):
        if tok.startswith("[["):
            out.append(run(cites(tok[2:-2])))
        elif tok.startswith("{{"):
            assert notes, tok
            out.append(f'<w:r><w:rPr><w:rStyle w:val="FootnoteReference"/><w:rtl/></w:rPr>'
                       f'<w:footnoteReference w:id="{footnote(tok[2:-2])}"/></w:r>')
        elif tok:
            out.append(run(tok))
    return "".join(out)


def body(text: str) -> str:
    return f'<w:p><w:pPr><w:rPr><w:rtl/><w:lang w:bidi="fa-IR"/></w:rPr></w:pPr>{runs_of(text)}</w:p>'


def heading(level: int, text: str) -> str:
    return (f'<w:p><w:pPr><w:pStyle w:val="Heading{level}"/><w:rPr><w:rtl/><w:lang w:bidi="fa-IR"/></w:rPr></w:pPr>'
            f'{runs_of(text)}</w:p>')


def figure() -> str:
    w, h = Image.open(FIG).size
    cx = 5760000                                          # 16 cm: the text width of the page
    cy = round(cx * h / w)
    p = re.sub(r'<wp:extent cx="\d+" cy="\d+"/>', f'<wp:extent cx="{cx}" cy="{cy}"/>', fig_para)
    p = re.sub(r'<a:ext cx="\d+" cy="\d+"/>', f'<a:ext cx="{cx}" cy="{cy}"/>', p)
    old = "- حوزه‌های پژوهشی در توسعۀ پایدار"
    assert old in cap_fig
    return p + cap_fig.replace(old, C.FIGURE_CAPTION)


def table() -> str:
    rows = re.findall(r"<w:tr[ >].*?</w:tr>", tbl, flags=re.S)
    head, tmpl = rows[0], rows[1]
    pct = ["330", "750", "870", "650", "750", "850", "800"]  # of 5000: wider journal and aim columns
    grid = [598, 1359, 1576, 1178, 1359, 1540, 1450]

    def widths(r):
        it = iter(pct)
        return re.sub(r'<w:tcW w:w="\d+" w:type="pct"/>', lambda m: f'<w:tcW w:w="{next(it)}" w:type="pct"/>', r)
    head, tmpl = widths(head), widths(tmpl)
    head = re.sub(r"(<w:tr[^>]*>)(<w:trPr>)?", lambda m: m.group(1) + "<w:trPr><w:tblHeader/>"
                  + ("" if m.group(2) else "</w:trPr>"), head, count=1)    # repeat the header on every page
    cell_p = re.compile(r'(<w:p [^>]*><w:pPr><w:pStyle w:val="a1"/>(?:<w:rPr>.*?</w:rPr>)?</w:pPr>)(</w:p>)', flags=re.S)
    assert len(cell_p.findall(tmpl)) == 7
    body_rows = []
    for i, row in enumerate(C.TABLE_ROWS, 1):
        it = iter([fa(i)] + list(row))
        body_rows.append(cell_p.sub(lambda m: m.group(1) + runs_of(next(it), notes=False) + m.group(2), tmpl))
    pre = tbl[:tbl.index(rows[0])]
    pre = re.sub(r"<w:tblGrid>.*?</w:tblGrid>", "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{g}"/>' for g in grid)
                 + "</w:tblGrid>", pre, flags=re.S)
    return pre + head + "".join(body_rows) + "</w:tbl>"


parts = []
for kind, text in C.BLOCKS:
    if kind == "h2":
        parts.append(heading(2, text))
    elif kind == "h3":
        parts.append(heading(3, text))
    elif kind == "p":
        parts.append(body(text))
    elif kind == "figure":
        parts.append(figure())
    elif kind == "table":
        parts.append(cap_tab + table() + body(""))
parts.append(f"<w:p>{sect}</w:p>")                         # the section break that ended chapter 2
new_ch2 = "".join(parts)


# ------------------------------------------------------------------ renumber chapters 3-5 and rebuild the list
def renumber(t: re.Match) -> str:
    return t.group(1) + PCITE.sub(lambda m: f"[{fa(num[key_of_old[int(m.group(1).translate(EN))]])}]", t.group(2)) + t.group(3)


rest_new = re.sub(r"(<w:t(?: [^>]*)?>)([^<]*)(</w:t>)", renumber, rest)
ref_ppr = ('<w:pPr><w:tabs><w:tab w:val="left" w:pos="680"/></w:tabs><w:bidi w:val="0"/><w:spacing w:after="80"/>'
           '<w:ind w:left="680" w:hanging="680"/></w:pPr>')
old_of_key = {v: n for n, v in C.OLD_KEYS.items()}
ref_list = []
for k in order:
    text = old_text[int(k[3:])] if k.startswith("old") else (
        old_text[old_of_key[k]] if k in old_of_key else esc(C.NEW_REFS[k]))
    ref_list.append(f'<w:p>{ref_ppr}<w:r><w:t>[{num[k]}]</w:t></w:r><w:r><w:tab/></w:r>'
                    f'<w:r><w:t xml:space="preserve">{text}</w:t></w:r></w:p>')

x_new = x[:R0] + new_ch2 + rest_new + "".join(ref_list) + x[L1:]
x_new = x_new.replace("حوزه‌های پژوهشی در توسعۀ پایدار", C.FIGURE_CAPTION[2:])   # cached list of figures

# footnotes: add the new ones, then keep only referenced notes, stored and numbered in document order as Word does
# (LibreOffice pairs them by order, not by id); the template's chapter-2 notes fall out here
fx = fx.replace("</w:footnotes>", "".join(new_fn) + "</w:footnotes>")
seq = []
for fid in re.findall(r'footnoteReference w:id="(\d+)"', x_new):
    if fid not in seq:
        seq.append(fid)
new_id = {old: str(i + 1) for i, old in enumerate(seq)}
x_new = re.sub(r'footnoteReference w:id="(\d+)"', lambda m: f'footnoteReference w:id="{new_id[m.group(1)]}"', x_new)
(OUT / "word/document.xml").write_text(x_new, encoding="utf-8")
notes = {m.group(1): m.group(0) for m in re.finditer(r'<w:footnote [^>]*w:id="(-?\d+)"[^>]*>.*?</w:footnote>', fx, flags=re.S)}
head_fx = fx[:fx.index("<w:footnote ")]
special = "".join(notes[k] for k in ("-1", "0"))
body_fx = "".join(re.sub(r'w:id="\d+"', f'w:id="{new_id[k]}"', notes[k], count=1) for k in seq)
(OUT / "word/footnotes.xml").write_text(head_fx + special + body_fx + "</w:footnotes>", encoding="utf-8")

# figure image (the JPEG that the template's Figure 1 used)
rid = re.search(r'r:embed="([^"]+)"', fig_para).group(1)
rels = (OUT / "word/_rels/document.xml.rels").read_text(encoding="utf-8")
target = re.search(rf'Id="{rid}"[^>]*Target="([^"]+)"', rels).group(1)
Image.open(FIG).convert("RGB").save(OUT / "word" / target, "JPEG", quality=93)

# ------------------------------------------------------------------ RIS for Mendeley / EndNote (chapter-2 references)
meta = json.loads((LIT / "meta.json").read_text(encoding="utf-8"))
arx = json.loads((LIT / "arxiv.json").read_text(encoding="utf-8"))
meta = {k.lower(): v for k, v in meta.items()}
ris = []
for k in [k for k in order if not k.startswith("old")]:
    doi, aid = C.DOIS.get(k, ""), C.ARXIV.get(k, "")
    if doi.lower() in meta:
        m = meta[doi.lower()]
        typ = {"journal-article": "JOUR", "proceedings-article": "CPAPER", "book": "BOOK",
               "edited-book": "BOOK", "report": "RPRT"}.get(m["type"], "GEN")
        title = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", m["title"]))
        rec = [f"TY  - {typ}", f"TI  - {html.unescape(title)}"] + [f"AU  - {a}" for a in m["authors"]] + [
            f"PY  - {m['year']}", f"T2  - {html.unescape(m['venue'])}", f"VL  - {m['volume']}", f"IS  - {m['issue']}",
            f"SP  - {m['page'] or m['article']}", f"DO  - {doi}"]
    elif aid in arx:
        a = arx[aid]
        rec = ["TY  - GEN", f"TI  - {a['title']}"] + [f"AU  - {n}" for n in a["authors"]] + [
            f"PY  - {a['published'][:4]}", f"UR  - https://arxiv.org/abs/{aid}"]
    else:
        ref = html.unescape(old_text[old_of_key[k]]) if k in old_of_key else C.NEW_REFS[k]
        rec = ["TY  - GEN", f"N1  - {ref}"] + ([f"DO  - {doi}"] if doi else [])
    ris.append("\n".join(line for line in rec if not re.match(r"^[A-Z0-9]{2}  - $", line)) + "\nER  - \n")
RIS.write_text("\n".join(ris), encoding="utf-8")

print(json.dumps({"references": len(order), "chapter2_references": len([k for k in order if not k.startswith('old')]),
                  "footnotes_added": len(new_fn), "table_rows": len(C.TABLE_ROWS),
                  "old_to_new": {n: num[key_of_old[n]] for n in range(1, N_OLD + 1) if key_of_old[n] in num}},
                 ensure_ascii=False))

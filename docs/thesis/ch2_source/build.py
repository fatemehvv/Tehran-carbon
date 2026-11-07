# -*- coding: utf-8 -*-
"""Insert chapter 2 into the RQ1-RQ2 thesis file, renumber every citation in order of first appearance (IEEE) and
rebuild the reference list. Usage: python build.py <unpacked original dir> <output dir> <figure.jpg> <out.ris>"""
import html
import json
import re
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import content as C  # noqa: E402

SRC, OUT, FIG, RIS = map(Path, sys.argv[1:5])
LIT = Path(__file__).parent.parent / "lit"
FA = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
EN = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")


def fa(n) -> str:
    return str(n).translate(FA)


def esc(s: str) -> str:
    return html.escape(s, quote=False)


if OUT.exists():
    shutil.rmtree(OUT)
shutil.copytree(SRC, OUT)
x = (OUT / "word/document.xml").read_text(encoding="utf-8")
fx = (OUT / "word/footnotes.xml").read_text(encoding="utf-8")
paras = list(re.finditer(r"<w:p[ >].*?</w:p>", x, flags=re.S))


def ptext(p: str) -> str:
    return "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", p))


# ------------------------------------------------------------------ locate the pieces (indices checked by hand)
assert ptext(paras[241].group(0)) == "مروری بر منابع" and ptext(paras[320].group(0)) == "فصل سوم"
assert "<w:sectPr" in paras[315].group(0)
fig_para, cap_fig, cap_tab = paras[275].group(0), paras[276].group(0), paras[285].group(0)
assert "w:drawing" in fig_para and "SEQ" in cap_fig and "SEQ" in cap_tab
tbl = re.search(r"<w:tbl>.*</w:tbl>", x[paras[285].end():paras[314].start()], flags=re.S).group(0)
R0, R1 = paras[242].start(), paras[315].end()
region = x[R0:R1]
sect = re.search(r"<w:pPr>.*?</w:pPr>", paras[315].group(0), flags=re.S).group(0)

ref_head = next(i for i, p in enumerate(paras) if 'w:val="Heading1"' in p.group(0) and ptext(p.group(0)) == "مراجع")
ref_paras = [p for p in paras[ref_head + 1:] if re.match(r"\[\d+\]", ptext(p.group(0)))]
old_text = {}
for p in ref_paras:
    m = re.search(r"<w:t>\[(\d+)\]</w:t><w:tab/><w:t[^>]*>(.*?)</w:t>", p.group(0), flags=re.S)
    old_text[int(m.group(1))] = m.group(2)                   # already XML-escaped
assert sorted(old_text) == list(range(1, 37)), sorted(old_text)
L0, L1 = ref_paras[0].start(), ref_paras[-1].end()

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
key_of_old = {n: C.OLD_KEYS.get(n, f"old{n}") for n in range(1, 37)}
PCITE = re.compile(r"\[([۰-۹]+)\]")
assert not PCITE.findall("".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", x[:R0]))), "citations before chapter 2"
rest = x[R1:L0]
for t in re.findall(r"<w:t[^>]*>([^<]*)</w:t>", rest):
    for m in PCITE.finditer(t):
        see(key_of_old[int(m.group(1).translate(EN))])
num = {k: i + 1 for i, k in enumerate(order)}
missing = [k for k in order if not k.startswith("old") and k not in C.NEW_REFS and k not in C.OLD_KEYS.values()]
assert not missing, f"no reference text for {missing}"
lost = [n for n in range(1, 37) if key_of_old[n] not in num]
assert not lost, f"old references no longer cited: {lost}"

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
def run(text: str, bold: bool = False, lang: bool = True) -> str:
    b = "<w:b/><w:bCs/>" if bold else ""
    lg = '<w:lang w:bidi="fa-IR"/>' if lang else ""
    return f'<w:r><w:rPr><w:rFonts w:hint="cs"/>{b}<w:rtl/>{lg}</w:rPr><w:t xml:space="preserve">{esc(text)}</w:t></w:r>'


def cites(keys: str) -> str:
    return "، ".join(f"[{fa(num[k.strip()])}]" for k in keys.split(","))


def runs_of(text: str, notes: bool = True) -> str:
    out = []
    text = text.replace(" [[", "\u00a0[[")
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
    cx = 5760000                                         # 16 cm: the text width of the page
    cy = round(cx * h / w)
    p = re.sub(r'<wp:extent cx="\d+" cy="\d+"/>', f'<wp:extent cx="{cx}" cy="{cy}"/>', fig_para)
    p = re.sub(r'<a:ext cx="\d+" cy="\d+"/>', f'<a:ext cx="{cx}" cy="{cy}"/>', p)
    old = "- حوزه‌های پژوهشی در توسعۀ پایدار"
    assert old in cap_fig
    return p + cap_fig.replace(old, C.FIGURE_CAPTION)


def table() -> str:
    rows = re.findall(r"<w:tr[ >].*?</w:tr>", tbl, flags=re.S)
    head, tmpl = rows[0], rows[1]
    pct = ["330", "750", "870", "650", "750", "850", "800"]            # of 5000: wider journal and aim columns
    grid = [598, 1359, 1576, 1178, 1359, 1540, 1450]

    def widths(r):
        it = iter(pct)
        return re.sub(r'<w:tcW w:w="\d+" w:type="pct"/>', lambda m: f'<w:tcW w:w="{next(it)}" w:type="pct"/>', r)
    head, tmpl = widths(head), widths(tmpl)
    head = head.replace("<w:trPr>", "<w:trPr><w:tblHeader/>", 1)          # repeat the header on every page
    cell_p = re.compile(r"(<w:p [^>]*><w:pPr><w:pStyle w:val=\"a1\"/><w:rPr><w:rtl/></w:rPr></w:pPr>)(</w:p>)")
    body_rows = []
    for i, row in enumerate(C.TABLE_ROWS, 1):
        cells = [fa(i)] + list(row)
        it = iter(cells)
        r = cell_p.sub(lambda m: m.group(1) + runs_of(next(it), notes=False) + m.group(2), tmpl)
        body_rows.append(r)
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
parts.append(f"<w:p>{sect}</w:p>")                      # the section break that ended chapter 2
new_ch2 = "".join(parts)

# ------------------------------------------------------------------ renumber chapters 3-5 and rebuild the list
def renumber(t: re.Match) -> str:
    return t.group(1) + PCITE.sub(lambda m: f"[{fa(num[key_of_old[int(m.group(1).translate(EN))]])}]", t.group(2)) + t.group(3)


rest_new = re.sub(r"(<w:t[^>]*>)([^<]*)(</w:t>)", renumber, rest)
ref_ppr = ('<w:pPr><w:tabs><w:tab w:val="left" w:pos="680"/></w:tabs><w:bidi w:val="0"/><w:spacing w:after="80"/>'
           '<w:ind w:left="680" w:hanging="680"/></w:pPr>')
ref_list = []
for k in order:
    text = old_text[int(k[3:])] if k.startswith("old") else (
        old_text[next(n for n, v in C.OLD_KEYS.items() if v == k)] if k in C.OLD_KEYS.values() else esc(C.NEW_REFS[k]))
    ref_list.append(f'<w:p>{ref_ppr}<w:r><w:t>[{num[k]}]</w:t><w:tab/><w:t xml:space="preserve">{text}</w:t></w:r></w:p>')

x_new = x[:R0] + new_ch2 + rest_new + "".join(ref_list) + x[L1:]
x_new = x_new.replace("حوزه‌های پژوهشی در توسعۀ پایدار", C.FIGURE_CAPTION[2:])   # cached list of figures
(OUT / "word/document.xml").write_text(x_new, encoding="utf-8")

# footnotes: drop the template's chapter-2 notes, add the new ones
for fid in set(re.findall(r'footnoteReference w:id="(\d+)"', region)):
    fx = re.sub(rf'<w:footnote [^>]*w:id="{fid}"[^>]*>.*?</w:footnote>', "", fx, flags=re.S)
fx = fx.replace("</w:footnotes>", "".join(new_fn) + "</w:footnotes>")
# store and number the footnotes in document order, as Word does (LibreOffice pairs them by order, not by id)
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
fx = head_fx + special + body_fx + "</w:footnotes>"
(OUT / "word/footnotes.xml").write_text(fx, encoding="utf-8")

# figure image (the JPEG that Figure 1 of the template used)
rid = re.search(r'r:embed="([^"]+)"', fig_para).group(1)
rels = (OUT / "word/_rels/document.xml.rels").read_text(encoding="utf-8")
target = re.search(rf'Id="{rid}"[^>]*Target="([^"]+)"', rels).group(1)
Image.open(FIG).convert("RGB").save(OUT / "word" / target, "JPEG", quality=93)

# ------------------------------------------------------------------ RIS for Mendeley / EndNote (chapter-2 references)
meta, arx = json.load(open(LIT / "refs.json")), json.load(open(LIT / "arxiv.json"))
arx = {k.split("v")[0]: v for k, v in arx.items()}
ris = []
ch2_keys = [k for k in order if not k.startswith("old")]
for k in ch2_keys:
    if k in C.DOIS and C.DOIS[k] in meta:
        m = meta[C.DOIS[k]]
        typ = {"journal-article": "JOUR", "proceedings-article": "CPAPER"}.get(m["type"], "GEN")
        title = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", m["title"]))
        title = re.sub(r"\b(CO|NO|N)\s+(2|x|X)\b", r"\1\2", title)          # subscripts lost with the tags
        rec = [f"TY  - {typ}", f"TI  - {title}"] + [f"AU  - {a}" for a in m["authors"]] + [
            f"PY  - {m['year']}", f"T2  - {html.unescape(m['venue'])}", f"VL  - {m['volume']}", f"IS  - {m['issue']}",
            f"SP  - {m['page'] or m['article']}", f"DO  - {C.DOIS[k]}"]
    elif k in C.ARXIV and C.ARXIV[k] in arx:
        a = arx[C.ARXIV[k]]
        rec = ["TY  - GEN", f"TI  - {a['title']}"] + [f"AU  - {n}" for n in a["authors"]] + [
            f"PY  - {a['year']}", f"UR  - https://arxiv.org/abs/{C.ARXIV[k]}", f"N1  - {a['comment']}"]
    else:
        rec = ["TY  - GEN", f"N1  - {html.unescape(C.NEW_REFS.get(k, ''))}"]
    ris.append("\n".join(line for line in rec if not line.endswith("-  ") and not line.endswith(" - ")) + "\nER  - \n")
RIS.write_text("\n".join(ris), encoding="utf-8")

print(json.dumps({"references": len(order), "chapter2_references": len(ch2_keys), "footnotes_added": len(new_fn),
                  "first_new_footnote": fid0, "old_to_new": {n: num[key_of_old[n]] for n in range(1, 37)}},
                 ensure_ascii=False))

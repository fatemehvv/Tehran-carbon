# -*- coding: utf-8 -*-
"""Build thesis version 2 (docs/thesis/thesis_v2_chapters_1-5.docx) from the full chapters 3-5 file with the review
fixes applied (port_review_fixes.py of docs/thesis/ch2_source).

  python build_v2.py <unpacked full file, fixes applied> <out dir> <figure1.jpg> <figure dir> <out.ris> <lit dir>

What it does, in order:
  1. cover page, Persian and English abstracts, running header: real title and names
  2. chapter 1 written (content_v2.CH1); chapter 2 written (content_ch2)
  3. chapters 3-5: every citation [n] becomes a key token; table numbers >= 20 move up by 2 and figure numbers >= 9 by 1
     (two new tables and one new figure); section references 3-4-14.. and 4-2-9.. move up by 1 (two new sections)
  4. new sections 3-4-14 and 4-2-9 inserted, paragraphs of 4-3, chapter 5 and two tables rewritten (content_v2)
  5. all citations numbered by first appearance in the whole thesis (IEEE), reference list rebuilt, footnotes
     renumbered in document order
Table of contents and lists of figures and tables: tocfix_v2.py afterwards.
"""
import html
import json
import re
import shutil
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import content_ch2 as C2  # noqa: E402
import content_v2 as V  # noqa: E402
from docxutil import EN, FA, body_elems, esc, fa, ptext  # noqa: E402

SRC, OUT, FIG1, FIGDIR, RIS, LIT = map(Path, sys.argv[1:7])
if OUT.exists():
    shutil.rmtree(OUT)
shutil.copytree(SRC, OUT)
x = (OUT / "word/document.xml").read_text(encoding="utf-8")
fx = (OUT / "word/footnotes.xml").read_text(encoding="utf-8")
rels = (OUT / "word/_rels/document.xml.rels").read_text(encoding="utf-8")
E = body_elems(x)
P = [x[s:e] for s, e, t in E]
TAG = [t for s, e, t in E]
N_FIG_NEW_AT, N_TAB_NEW_AT, N_TAB_NEW = 9, 20, 2        # new figure 9, new tables 20 and 21


def find(pred, start=0, stop=None):
    return next(i for i in range(start, len(P) if stop is None else stop) if pred(P[i]))


def is_h(p, level, text=None):
    return f'w:val="Heading{level}"' in p and (text is None or ptext(p) == text)


# ------------------------------------------------------------------------------------------------ locate
i_abs = find(lambda p: is_h(p, 2, "چکيده"))
i_toc = find(lambda p: is_h(p, 2, "فهرست مطالب"), i_abs)
i_ch1 = find(lambda p: is_h(p, 1, "مقدمه"), i_toc)
i_ch1_end = find(lambda p: "<w:sectPr" in p, i_ch1)
i_ch2 = find(lambda p: is_h(p, 1, "مروری بر منابع"), i_ch1_end)
i_ch2_end = find(lambda p: "<w:sectPr" in p, i_ch2)
i_ch3 = find(lambda p: is_h(p, 1, "روش پژوهش"), i_ch2_end)
i_ins3 = find(lambda p: is_h(p, 3, "شبیه‌ساز سیاست (پرسش سوم)"), i_ch3)
i_ins4 = find(lambda p: is_h(p, 3, "پرسش سوم: یادگیری تقویتی در طرح ماهانه"), i_ins3)
i_ref = find(lambda p: is_h(p, 1, "مراجع"), i_ins4)
i_eabs = find(lambda p: ptext(p) == "Abstract" and "Heading2" in p, i_ref)
ref_k = [k for k in range(i_ref + 1, i_eabs) if re.match(r"\[\d+\]", ptext(P[k]))]

# chapter 2 template pieces (as in docs/thesis/ch2_source/build.py)
k_fig = find(lambda p: "w:drawing" in p, i_ch2, i_ch2_end)
fig2_para, fig2_cap = P[k_fig], P[k_fig + 1]
k_capt = find(lambda p: 'w:val="CaptionforTables"' in p, i_ch2, i_ch2_end)
tab2_cap, tab2 = P[k_capt], P[k_capt + 1]
sect2 = re.search(r"<w:pPr>.*?</w:pPr>", P[i_ch2_end], flags=re.S).group(0)

# templates for the new chapter-4 tables, figure and list items
k_t15 = find(lambda p: ptext(p).startswith("جدول 15-") and "CaptionforTables" in p, i_ch3)
TCAP, TTBL = P[k_t15], P[k_t15 + 1]
k_f8 = find(lambda p: ptext(p).startswith("شکل 8-") and 'w:val="Caption"' in p, i_ch3)
FCAP, FDRAW = P[k_f8], P[k_f8 - 1]
LI_PPR = re.search(r"<w:pPr>.*?</w:pPr>", P[find(lambda p: ptext(p).startswith("طراحی عامل یادگیری تقویتی با فضای عمل"), i_ins4)],
                   flags=re.S).group(0)

old_text = {}
for k in ref_k:
    m = re.match(r"\[(\d+)\](.*)", ptext(P[k]), flags=re.S)
    old_text[int(m.group(1))] = esc(html.unescape(m.group(2).strip()))
N_OLD = len(old_text)
key_of_old = {n: C2.OLD_KEYS.get(n, f"old{n}") for n in range(1, N_OLD + 1)}
old_of_key = {v: n for n, v in C2.OLD_KEYS.items()}
for n, key in C2.OLD_KEYS.items():
    assert re.match(r"[a-z]+", key).group(0)[:5] in html.unescape(old_text[n]).lower().replace(" ", "") or key in (
        "tfl2004", "ipcc2006e"), (n, key)

# ------------------------------------------------------------------------------------------------ footnotes and runs
fid0 = max(int(i) for i in re.findall(r'<w:footnote [^>]*w:id="(\d+)"', fx)) + 1
new_fn = []


def footnote(text: str) -> int:
    fid = fid0 + len(new_fn)
    new_fn.append(f'<w:footnote w:id="{fid}"><w:p><w:pPr><w:pStyle w:val="FootnoteText"/><w:bidi w:val="0"/></w:pPr>'
                  f'<w:r><w:rPr><w:rStyle w:val="FootnoteReference"/></w:rPr><w:footnoteRef/></w:r>'
                  f'<w:r><w:t xml:space="preserve"> {esc(text)}</w:t></w:r></w:p></w:footnote>')
    return fid


def run(text: str, bold: bool = False, rtl: bool = True) -> str:
    b = "<w:b/><w:bCs/>" if bold else ""
    r = '<w:rtl/><w:lang w:bidi="fa-IR"/>' if rtl else ""
    return f'<w:r><w:rPr><w:rFonts w:hint="cs"/>{b}{r}</w:rPr><w:t xml:space="preserve">{esc(text)}</w:t></w:r>'


def tok(keys: str) -> str:
    return "، ".join(f"⟦c:{k.strip()}⟧" for k in keys.split(","))


def runs_of(text: str, notes: bool = True) -> str:
    out = []
    text = text.replace(" [[", " [[")
    m = re.match(r"\*\*(.+?)\*\*\s*", text)
    if m:
        out += [run(m.group(1), bold=True), run(" ")]
        text = text[m.end():]
    for t in re.split(r"(\[\[[^\]]+\]\]|\{\{[^}]+\}\})", text):
        if t.startswith("[["):
            out.append(run(tok(t[2:-2])))
        elif t.startswith("{{"):
            assert notes, t
            out.append(f'<w:r><w:rPr><w:rStyle w:val="FootnoteReference"/><w:rtl/></w:rPr>'
                       f'<w:footnoteReference w:id="{footnote(t[2:-2])}"/></w:r>')
        elif t:
            out.append(run(t))
    return "".join(out)


def body(text: str) -> str:
    return f'<w:p><w:pPr><w:rPr><w:rtl/><w:lang w:bidi="fa-IR"/></w:rPr></w:pPr>{runs_of(text)}</w:p>'


def item(text: str) -> str:
    return f"<w:p>{LI_PPR}{runs_of(text)}</w:p>"


def heading(level: int, text: str) -> str:
    return (f'<w:p><w:pPr><w:pStyle w:val="Heading{level}"/><w:rPr><w:rtl/><w:lang w:bidi="fa-IR"/></w:rPr></w:pPr>'
            f'{runs_of(text)}</w:p>')


def caption(template: str, number: int, text: str) -> str:
    """A copy of a caption paragraph with another SEQ result and text (bookmarks dropped: tocfix adds its own)."""
    c = re.sub(r"<w:bookmark(Start|End) [^>]*/>", "", template)
    c = re.sub(r'(<w:fldChar w:fldCharType="separate"/></w:r><w:r(?: [^>]*)?>(?:<w:rPr>.*?</w:rPr>)?<w:t>)\d+(</w:t>)',
               rf"\g<1>{number}\2", c, count=1, flags=re.S)
    a = c.index("</w:t>", c.index('w:fldCharType="end"')) if False else None
    end = c.index('<w:fldChar w:fldCharType="end"/>')
    end = c.index("</w:r>", end) + len("</w:r>")
    return c[:end] + f'<w:r><w:rPr><w:rtl/></w:rPr><w:t xml:space="preserve">- {esc(text)}</w:t></w:r></w:p>'


def table_xml(header, rows, pcts) -> str:
    """A table in the style of the chapter-4 tables (header shading, centred cells, repeated header row)."""
    pre = TTBL[:TTBL.index("<w:tr")]
    width = 9060
    grid = "".join(f'<w:gridCol w:w="{round(int(p) / 5000 * width)}"/>' for p in pcts)
    pre = re.sub(r"<w:tblGrid>.*?</w:tblGrid>", f"<w:tblGrid>{grid}</w:tblGrid>", pre, flags=re.S)

    def cell(text, pct, head):
        shd = '<w:shd w:val="clear" w:color="auto" w:fill="FFD6C1"/>' if head else ""
        return (f'<w:tc><w:tcPr><w:tcW w:w="{pct}" w:type="pct"/>{shd}<w:vAlign w:val="center"/></w:tcPr>'
                f'<w:p><w:pPr><w:pStyle w:val="a1"/><w:rPr><w:rtl/></w:rPr></w:pPr>{run(text, bold=head)}</w:p></w:tc>')
    trh = '<w:tr><w:trPr><w:cantSplit/><w:tblHeader/><w:jc w:val="center"/></w:trPr>'
    trb = '<w:tr><w:trPr><w:cantSplit/><w:jc w:val="center"/></w:trPr>'
    out = pre + trh + "".join(cell(h, p, True) for h, p in zip(header, pcts)) + "</w:tr>"
    for r in rows:
        out += trb + "".join(cell(v, p, False) for v, p in zip(r, pcts)) + "</w:tr>"
    return out + "</w:tbl>"


media_n = [0]


def figure_xml(path: Path, cap: str, number: int) -> str:
    media_n[0] += 1
    name = f"v2_fig{media_n[0]}{path.suffix}"
    shutil.copy(path, OUT / "word/media" / name)
    rid = f"rIdV2{media_n[0]}"
    global rels
    rels = rels.replace("</Relationships>", f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/'
                                            f'officeDocument/2006/relationships/image" Target="media/{name}"/></Relationships>')
    w, h = Image.open(path).size
    cx = 5760000
    cy = round(cx * h / w)
    pid = 9000 + media_n[0]
    d = re.sub(r'r:embed="[^"]+"', f'r:embed="{rid}"', FDRAW)
    d = re.sub(r'<wp:extent cx="\d+" cy="\d+"/>', f'<wp:extent cx="{cx}" cy="{cy}"/>', d)
    d = re.sub(r'<a:ext cx="\d+" cy="\d+"/>', f'<a:ext cx="{cx}" cy="{cy}"/>', d)
    d = re.sub(r'<wp:docPr id="\d+" name="[^"]*"/>', f'<wp:docPr id="{pid}" name="Figure {pid}"/>', d)
    d = re.sub(r'<pic:cNvPr id="\d+" name="[^"]*"/>', f'<pic:cNvPr id="{pid}" name="{name}"/>', d)
    d = d.replace("<w:lastRenderedPageBreak/>", "")
    return d + caption(FCAP, number, cap)


def blocks_xml(blocks) -> str:
    out = []
    for b in blocks:
        kind = b[0]
        if kind in ("h2", "h3"):
            out.append(heading(int(kind[1]), b[1]))
        elif kind == "p":
            out.append(body(b[1]))
        elif kind == "li":
            out.append(item(b[1]))
        elif kind == "table":
            _, n, cap, head, rows, pcts = b
            out.append(caption(TCAP, n, cap) + table_xml(head, rows, pcts) + body(""))
        elif kind == "figure":
            _, f, cap, n = b
            out.append(figure_xml(FIGDIR / f, cap, n))
        else:
            raise ValueError(kind)
    return "".join(out)


# ------------------------------------------------------------------------------------------------ old text of chapters 3-5
PCITE = re.compile(r"\[([۰-۹]+)\]")


def remap_t(text: str) -> str:
    text = PCITE.sub(lambda m: f"⟦c:{key_of_old[int(m.group(1).translate(EN))]}⟧", text)

    def sec(m):
        a, b, c = (int(g.translate(EN)) for g in m.groups())
        if (a, b) == (3, 4) and c >= 14 or (a, b) == (4, 2) and c >= 9:
            c += 1
        return f"{fa(a)}-{fa(b)}-{fa(c)}"
    return re.sub(r"([۰-۹]+)-([۰-۹]+)-([۰-۹]+)", sec, text)


def remap_numbers(par: str) -> str:
    """table / figure numbers in one paragraph, also when 'جدول ' and the number sit in different runs"""
    nodes = list(re.finditer(r"(<w:t(?: [^>]*)?>)([^<]*)(</w:t>)", par))
    full = "".join(n.group(2) for n in nodes)
    owner = [(i, j) for i, n in enumerate(nodes) for j in range(len(n.group(2)))]
    edits = {}
    for m in re.finditer(r"(جدول|شکل)([\s ]*)(\d+)", full):
        n = int(m.group(3))
        if n > 60:
            continue
        new = n + N_TAB_NEW if (m.group(1) == "جدول" and n >= N_TAB_NEW_AT) else (
            n + 1 if (m.group(1) == "شکل" and n >= N_FIG_NEW_AT) else n)
        if new == n:
            continue
        s0, e0 = m.start(3), m.end(3)
        first = owner[s0][0]
        assert all(owner[k][0] == first for k in range(s0, e0)), "number split across runs"
        edits.setdefault(first, []).append((owner[s0][1], owner[e0 - 1][1] + 1, str(new)))
    if not edits:
        return par
    out, pos = [], 0
    for i, n in enumerate(nodes):
        if i not in edits:
            continue
        t = n.group(2)
        for a, b, new in sorted(edits[i], reverse=True):
            t = t[:a] + new + t[b:]
        out += [par[pos:n.start(2)], t]
        pos = n.end(2)
    return "".join(out) + par[pos:]


def transform_old(el: str) -> str:
    el = re.sub(r"<w:p[ >].*?</w:p>", lambda m: remap_numbers(m.group(0)), el, flags=re.S)
    return re.sub(r"(<w:t(?: [^>]*)?>)([^<]*)(</w:t>)", lambda m: m.group(1) + remap_t(m.group(2)) + m.group(3), el)


def replace_text(seg: str, old: str, new: str) -> str:
    nodes = list(re.finditer(r"(<w:t(?: [^>]*)?>)([^<]*)(</w:t>)", seg))
    full = "".join(n.group(2) for n in nodes)
    i = full.find(old)
    assert i >= 0 and full.find(old, i + 1) < 0, (old[:60], full[:120])
    j, acc, out = i + len(old), 0, []
    for n in nodes:
        a, b = acc, acc + len(n.group(2))
        acc = b
        if b <= i or a >= j:
            continue
        lo, hi = max(i, a) - a, min(j, b) - a
        out.append((n.start(2), n.end(2), n.group(2)[:lo] + (new if a <= i < b else "") + n.group(2)[hi:]))
    for s0, e0, t in reversed(out):
        seg = seg[:s0] + esc(html.unescape(t)) + seg[e0:]
    return re.sub(r"<w:t>([^<]*)</w:t>", lambda m: f'<w:t xml:space="preserve">{m.group(1)}</w:t>'
                  if m.group(1)[:1] == " " or m.group(1)[-1:] == " " else m.group(0), seg)


new = {}                       # element index -> replacement xml
before = {}                    # element index -> xml inserted before it

# 1. cover, abstracts, header
def set_text(p: str, text: str) -> str:
    nodes = list(re.finditer(r"(<w:t(?: [^>]*)?>)([^<]*)(</w:t>)", p))
    out, pos = [], 0
    for i, n in enumerate(nodes):
        out += [p[pos:n.start(2)], esc(text) if i == 0 else ""]
        pos = n.end(2)
    return "".join(out) + p[pos:]


for i in range(0, i_abs):
    t = ptext(P[i])
    if t == "عنوان پايان‌نامه":
        new[i] = set_text(P[i], V.TITLE)
    elif t == "نام دانشجو":
        new[i] = set_text(P[i], V.STUDENT)
    elif t.startswith("دكتر ......"):
        new[i] = set_text(P[i], V.SUPERVISOR)
abs_pars = [k for k in range(i_abs + 1, i_toc) if ptext(P[k]).strip()]
assert ptext(P[abs_pars[-1]]).startswith("واژگان كليدي")
new[abs_pars[0]] = "".join(body(t) for t in V.ABSTRACT_FA)
for k in abs_pars[1:-1]:
    new[k] = ""
new[abs_pars[-1]] = ("<w:p>" + run("واژگان کلیدی:", bold=True) + run(" " + V.KEYWORDS_FA) + "</w:p>")
eabs = [k for k in range(i_eabs + 1, len(P)) if ptext(P[k]).strip() and "<w:sectPr" not in P[k]]
k_kw = next(k for k in eabs if ptext(P[k]).startswith("Keywords"))
eb = [k for k in eabs if k < k_kw]
ppr_en = re.search(r"<w:pPr>.*?</w:pPr>", P[eb[0]], flags=re.S)
ppr_en = ppr_en.group(0) if ppr_en else ""
new[eb[0]] = "".join(f'<w:p>{ppr_en}{run(t, rtl=False)}</w:p>' for t in V.ABSTRACT_EN)
for k in eb[1:]:
    new[k] = ""
ppr_kw = re.search(r"<w:pPr>.*?</w:pPr>", P[k_kw], flags=re.S)
new[k_kw] = (f'<w:p>{ppr_kw.group(0) if ppr_kw else ""}' + run("Keywords:", bold=True, rtl=False)
             + run(" " + V.KEYWORDS_EN, rtl=False) + "</w:p>")
for k in range(k_kw, len(P)):
    t = ptext(P[k])
    if t == "Thesis Title":
        new[k] = set_text(P[k], "Multimodal Data Fusion for Predictive Analysis of Carbon Footprint Reduction in Tehran")
    elif t == "Student Name":
        new[k] = set_text(P[k], "Fatemeh Vafaeian")
    elif t.startswith("Dr. ----") and "Supervisor" in ptext(P[k - 1]):
        new[k] = set_text(P[k], "Dr. Vahid Khatibi")

# 2. chapter 1 and chapter 2
new[i_ch1 + 1] = blocks_xml(V.CH1)
for k in range(i_ch1 + 2, i_ch1_end):
    new[k] = ""


def fig1() -> str:
    w, h = Image.open(FIG1).size
    cx = 5760000
    cy = round(cx * h / w)
    p = re.sub(r'<wp:extent cx="\d+" cy="\d+"/>', f'<wp:extent cx="{cx}" cy="{cy}"/>', fig2_para)
    p = re.sub(r'<a:ext cx="\d+" cy="\d+"/>', f'<a:ext cx="{cx}" cy="{cy}"/>', p)
    old = "- حوزه‌های پژوهشی در توسعۀ پایدار"
    assert old in fig2_cap
    return p + fig2_cap.replace(old, C2.FIGURE_CAPTION)


def table1() -> str:
    rows = re.findall(r"<w:tr[ >].*?</w:tr>", tab2, flags=re.S)
    head, tmpl = rows[0], rows[1]
    pct = ["330", "750", "870", "650", "750", "850", "800"]
    grid = [598, 1359, 1576, 1178, 1359, 1540, 1450]

    def widths(r):
        it = iter(pct)
        return re.sub(r'<w:tcW w:w="\d+" w:type="pct"/>', lambda m: f'<w:tcW w:w="{next(it)}" w:type="pct"/>', r)
    head, tmpl = widths(head), widths(tmpl)
    head = re.sub(r"(<w:tr[^>]*>)(<w:trPr>)?", lambda m: m.group(1) + "<w:trPr><w:tblHeader/>"
                  + ("" if m.group(2) else "</w:trPr>"), head, count=1)
    cell_p = re.compile(r'(<w:p [^>]*><w:pPr><w:pStyle w:val="a1"/>(?:<w:rPr>.*?</w:rPr>)?</w:pPr>)(</w:p>)', flags=re.S)
    out = []
    for i, row in enumerate(C2.TABLE_ROWS, 1):
        it = iter([fa(i)] + list(row))
        out.append(cell_p.sub(lambda m: m.group(1) + runs_of(next(it), notes=False) + m.group(2), tmpl))
    pre = tab2[:tab2.index(rows[0])]
    pre = re.sub(r"<w:tblGrid>.*?</w:tblGrid>", "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{g}"/>' for g in grid)
                 + "</w:tblGrid>", pre, flags=re.S)
    return pre + head + "".join(out) + "</w:tbl>"


ch2 = []
for kind, text in C2.BLOCKS:
    if kind in ("h2", "h3"):
        ch2.append(heading(int(kind[1]), text))
    elif kind == "p":
        ch2.append(body(text))
    elif kind == "figure":
        ch2.append(fig1())
    elif kind == "table":
        ch2.append(tab2_cap + table1() + body(""))
ch2.append(f"<w:p>{sect2}</w:p>")
new[i_ch2 + 1] = "".join(ch2)
for k in range(i_ch2 + 2, i_ch2_end + 1):
    new[k] = ""

# 3. chapters 3-5: transform, then the edits of content_v2
for k in range(i_ch3, i_ref):
    P[k] = transform_old(P[k])
done = set()
for k in range(i_ch3, i_ref):
    t = ptext(P[k])
    for key, blocks in V.REPLACE.items():
        if blocks and t.startswith(key):
            new[k] = blocks_xml(blocks)
            done.add(key)
    for start, old, rep in V.EDITS:
        if t.startswith(start):
            P[k] = replace_text(P[k], old, rep)
            done.add(start)
    if TAG[k] == "tbl" and V.POLICY_CELL[0] in t:
        P[k] = replace_text(P[k], V.POLICY_CELL[0], V.POLICY_CELL[1])
        done.add("policy")
missing = [k for k, b in V.REPLACE.items() if b and k not in done] + [s for s, _, _ in V.EDITS if s not in done]
assert not missing and "policy" in done, missing
# summary table (now Table 27): add rows after the last row of the second question
k_sum = find(lambda p: ptext(p).startswith("جدول 27-") and "CaptionforTables" in p, i_ins4)
tbl = P[k_sum + 1]
rows = re.findall(r"<w:tr[ >].*?</w:tr>", tbl, flags=re.S)
last2 = max(i for i, r in enumerate(rows) if ptext(re.findall(r"<w:tc>.*?</w:tc>", r, flags=re.S)[0]).strip() == "دوم")
tmpl = rows[last2]


def row_with(cells):
    tcs = re.findall(r"<w:tc>.*?</w:tc>", tmpl, flags=re.S)
    out = tmpl
    for tc, val in zip(tcs, cells):
        out = out.replace(tc, set_text(tc, val), 1)
    return out


add = "".join(row_with(c) for c in V.SUMMARY_ADD)
P[k_sum + 1] = tbl.replace(rows[last2], rows[last2] + add, 1)
# future work: two items after the last one
k_fu = find(lambda p: ptext(p).startswith("طراحی عامل یادگیری تقویتی با فضای عمل"), i_ins4)
new[k_fu] = new.get(k_fu, P[k_fu]) + blocks_xml(V.FUTURE_ADD)
# new sections
before[i_ins3] = blocks_xml(V.CH3_NEW)
before[i_ins4] = blocks_xml(V.CH4_NEW)

# ------------------------------------------------------------------------------------------------ assemble
parts, pos = [], 0
for k in range(len(E)):
    s, e, _ = E[k]
    parts.append(x[pos:s])
    if k in before:
        parts.append(before[k])
    parts.append(new.get(k, P[k]) if k not in range(ref_k[0], ref_k[-1] + 1) else "")
    if k == ref_k[0]:
        parts.append("⟦REFLIST⟧")
    pos = e
parts.append(x[pos:])
x_new = "".join(parts)

# ------------------------------------------------------------------------------------------------ citations
order = []
for t in re.findall(r"⟦c:([^⟧]+)⟧", x_new):
    if t not in order:
        order.append(t)
num = {k: i + 1 for i, k in enumerate(order)}
unknown = [k for k in order if not k.startswith("old") and k not in C2.NEW_REFS and k not in old_of_key]
assert not unknown, unknown
lost = {n for n in range(1, N_OLD + 1) if key_of_old[n] not in num}
assert lost == C2.DROPPED, sorted(lost ^ C2.DROPPED)
x_new = re.sub(r"⟦c:([^⟧]+)⟧", lambda m: f"[{fa(num[m.group(1)])}]", x_new)
ref_ppr = ('<w:pPr><w:tabs><w:tab w:val="left" w:pos="680"/></w:tabs><w:bidi w:val="0"/><w:spacing w:after="80"/>'
           '<w:ind w:left="680" w:hanging="680"/></w:pPr>')
reflist = []
for k in order:
    text = old_text[int(k[3:])] if k.startswith("old") else (
        old_text[old_of_key[k]] if k in old_of_key else esc(C2.NEW_REFS[k]))
    reflist.append(f'<w:p>{ref_ppr}<w:r><w:t>[{num[k]}]</w:t></w:r><w:r><w:tab/></w:r>'
                   f'<w:r><w:t xml:space="preserve">{text}</w:t></w:r></w:p>')
x_new = x_new.replace("⟦REFLIST⟧", "".join(reflist))
x_new = x_new.replace("حوزه‌های پژوهشی در توسعۀ پایدار", C2.FIGURE_CAPTION[2:])
assert "⟦" not in x_new

# ------------------------------------------------------------------------------------------------ template instructions
# the two "update this list, then delete this red text" notes of the template, and the picture of the Word button that
# follows the second one (its run only: the same paragraph opens the list-of-figures field)
for _ in range(2):
    i = x_new.index("این متن قرمزرنگ را حذف کنید")
    a = max(x_new.rfind("<w:p ", 0, i), x_new.rfind("<w:p>", 0, i))
    b = x_new.index("</w:p>", i) + len("</w:p>")
    assert "fldChar" not in x_new[a:b]
    x_new = x_new[:a] + x_new[b:]
    if _ == 1:
        d = x_new.index("<w:drawing>", a)
        assert x_new.find("</w:p>", a) > d                      # the picture is in the very next paragraph
        r0 = max(x_new.rfind("<w:r>", a, d), x_new.rfind("<w:r ", a, d))
        r1 = x_new.index("</w:r>", x_new.index("</w:drawing>", d)) + len("</w:r>")
        x_new = x_new[:r0] + x_new[r1:]
assert "قرمزرنگ" not in x_new

# ------------------------------------------------------------------------------------------------ footnotes
fx = fx.replace("</w:footnotes>", "".join(new_fn) + "</w:footnotes>")
seq = []
for fid in re.findall(r'footnoteReference w:id="(\d+)"', x_new):
    if fid not in seq:
        seq.append(fid)
new_id = {o: str(i + 1) for i, o in enumerate(seq)}
x_new = re.sub(r'footnoteReference w:id="(\d+)"', lambda m: f'footnoteReference w:id="{new_id[m.group(1)]}"', x_new)
notes = {m.group(1): m.group(0) for m in re.finditer(r'<w:footnote [^>]*w:id="(-?\d+)"[^>]*>.*?</w:footnote>', fx, flags=re.S)}
head_fx = fx[:fx.index("<w:footnote ")]
fx = head_fx + "".join(notes[k] for k in ("-1", "0")) + "".join(
    re.sub(r'w:id="\d+"', f'w:id="{new_id[k]}"', notes[k], count=1) for k in seq) + "</w:footnotes>"

(OUT / "word/document.xml").write_text(x_new, encoding="utf-8")
(OUT / "word/footnotes.xml").write_text(fx, encoding="utf-8")
(OUT / "word/_rels/document.xml.rels").write_text(rels, encoding="utf-8")
for h in (OUT / "word").glob("header*.xml"):
    t = h.read_text(encoding="utf-8")
    # the header shows the title through STYLEREF "عنوان پايان‌نامه" (the style of the cover title): only the cached
    # result changes, the field code keeps the style name
    t2 = re.sub(r"(<w:t(?: [^>]*)?>)عنوان پايان‌نامه(</w:t>)", lambda m: m.group(1) + V.TITLE + m.group(2), t)
    if t2 != t:
        assert re.search(r'<w:instrText[^>]*>\s*"عنوان پايان‌نامه"', t2)
        h.write_text(t2, encoding="utf-8")
# Figure 1 image (the JPEG of the template's Figure 1)
rid = re.search(r'r:embed="([^"]+)"', fig2_para).group(1)
target = re.search(rf'Id="{rid}"[^>]*Target="([^"]+)"', rels).group(1)
Image.open(FIG1).convert("RGB").save(OUT / "word" / target, "JPEG", quality=93)

# ------------------------------------------------------------------------------------------------ RIS (all references that are not old thesis references)
meta = {k.lower(): v for k, v in json.loads((LIT / "meta.json").read_text(encoding="utf-8")).items()}
arx = json.loads((LIT / "arxiv.json").read_text(encoding="utf-8"))
ris = []
for k in [k for k in order if not k.startswith("old")]:
    doi, aid = C2.DOIS.get(k, ""), C2.ARXIV.get(k, "")
    if doi.lower() in meta:
        m = meta[doi.lower()]
        typ = {"journal-article": "JOUR", "proceedings-article": "CPAPER", "book": "BOOK", "edited-book": "BOOK",
               "report": "RPRT"}.get(m["type"], "GEN")
        title = html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", m["title"])))
        rec = [f"TY  - {typ}", f"TI  - {title}"] + [f"AU  - {a}" for a in m["authors"]] + [
            f"PY  - {m['year']}", f"T2  - {html.unescape(m['venue'])}", f"VL  - {m['volume']}", f"IS  - {m['issue']}",
            f"SP  - {m['page'] or m['article']}", f"DO  - {doi}"]
    elif aid in arx:
        a = arx[aid]
        rec = ["TY  - GEN", f"TI  - {a['title']}"] + [f"AU  - {n}" for n in a["authors"]] + [
            f"PY  - {a['published'][:4]}", f"UR  - https://arxiv.org/abs/{aid}"]
    else:
        ref = html.unescape(old_text[old_of_key[k]]) if k in old_of_key else C2.NEW_REFS[k]
        rec = ["TY  - GEN", f"N1  - {ref}"] + ([f"DO  - {doi}"] if doi else [])
    ris.append("\n".join(line for line in rec if not re.match(r"^[A-Z0-9]{2}  - $", line)) + "\nER  - \n")
RIS.write_text("\n".join(ris), encoding="utf-8")

old_to_new = {n: num[key_of_old[n]] for n in range(1, N_OLD + 1) if key_of_old[n] in num}
print(json.dumps({"references": len(order), "new_references": len([k for k in order if not k.startswith("old")]),
                  "footnotes_added": len(new_fn), "old_to_new": old_to_new}, ensure_ascii=False))

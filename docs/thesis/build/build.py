# -*- coding: utf-8 -*-
"""Rebuild chapters 3-5 of the thesis docx from content.py, keeping the university template.

    python build.py <template_v1.docx or its unpacked dir> <persian figures dir> <output unpacked dir> [pages.json]

template_v1.docx is the version before this rewrite, with Word's split runs already merged.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image

import content as C

SRC, FIGS, OUT = (Path(a) for a in sys.argv[1:4])
if SRC.suffix == ".docx":                                   # unpack the template next to the output
    import zipfile
    unpacked = OUT.parent / (SRC.stem + "_unpacked")
    if unpacked.exists():
        shutil.rmtree(unpacked)
    zipfile.ZipFile(SRC).extractall(unpacked)
    SRC = unpacked
PAGES = json.loads(Path(sys.argv[4]).read_text()) if len(sys.argv) > 4 and Path(sys.argv[4]).exists() else {}
if OUT.exists():
    shutil.rmtree(OUT)
shutil.copytree(SRC, OUT)
W = OUT / "word"
doc = (W / "document.xml").read_text(encoding="utf-8")
fnx = (W / "footnotes.xml").read_text(encoding="utf-8")
rels = (W / "_rels" / "document.xml.rels").read_text(encoding="utf-8")

FA = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
CH = {3: C.CH3, 4: C.CH4, 5: C.CH5}
NUM_FA = {3: "۳", 4: "۴", 5: "۵"}

# ------------------------------------------------------------------------------------------ pre-pass: numbering
heads, tabs, figs, eqs = {}, {}, {}, {}
toc = []                                   # (level, number text, title, bookmark, chapter)
tno, fno, bm_id = 1, 1, 5000               # table 1 and figure 1 are in chapter 2
for ch, blocks in CH.items():
    h2 = h3 = e = 0
    for b in blocks:
        if b[0] == "h2":
            h2, h3 = h2 + 1, 0
            heads[b[1]] = dict(num=f"{ch} ـ {h2}", fa=f"{ch}-{h2}".translate(FA), bm=f"_TocC{ch}{h2:02d}00")
            toc.append((2, f"{ch} ـ {h2})", b[2], heads[b[1]]["bm"], ch))
        elif b[0] == "h3":
            h3 += 1
            heads[b[1]] = dict(num=f"{ch} ـ {h2} ـ {h3}", fa=f"{ch}-{h2}-{h3}".translate(FA),
                               bm=f"_TocC{ch}{h2:02d}{h3:02d}")
            toc.append((3, f"{ch} ـ {h2} ـ {h3})", b[2], heads[b[1]]["bm"], ch))
        elif b[0] == "table":
            tno += 1
            tabs[b[1]] = dict(n=tno, ref=f"_Reft_{b[1]}", toc=f"_TocT{tno:02d}", cap=b[2])
        elif b[0] == "fig":
            fno += 1
            figs[b[1]] = dict(n=fno, ref=f"_Reff_{b[1]}", toc=f"_TocF{fno:02d}", cap=b[2])
        elif b[0] == "eq":
            e += 1
            eqs[b[1]] = f"{ch}-{e}".translate(FA)

# ------------------------------------------------------------------------------------------ inline renderer
cites: dict[str, int] = {}
footnotes: list[tuple[int, str]] = []
fn_next = 20
TOKEN = re.compile(r"(\*\*|==|\^\[[^\]]*\]|\[\[c:[^\]]+\]\]|\{\{[tfes]:[^}]+\}\})")
SIGN = re.compile(r"([۰-۹][۰-۹٫]*)([‑+])(?=[\s،؛)٪]|$)")


def nbid():
    global bm_id
    bm_id += 1
    return bm_id


def fix_signs(s: str) -> str:
    """'۰٫۰۲۱‑' / '۰٫۰۲۱+' (sign typed after the digits) -> LRM + sign + digits, as in the template."""
    return SIGN.sub(lambda m: "‎" + ("-" if m.group(2) == "‑" else "+") + m.group(1), s)


def rpr(bold=False, hl=False, extra=""):
    return ("<w:rPr><w:rFonts w:hint=\"cs\"/>" + ("<w:b/><w:bCs/>" if bold else "") + extra +
            ("<w:highlight w:val=\"yellow\"/>" if hl else "") + "<w:rtl/><w:lang w:bidi=\"fa-IR\"/></w:rPr>")


def run(text, bold=False, hl=False):
    if not text:
        return ""
    return f'<w:r>{rpr(bold, hl)}<w:t xml:space="preserve">{escape(fix_signs(text))}</w:t></w:r>'


def ref_field(bm, label, n, bold=False, hl=False):
    return (f'<w:r><w:rPr><w:rtl/></w:rPr><w:fldChar w:fldCharType="begin"/><w:instrText xml:space="preserve"> REF {bm}'
            f' \\h </w:instrText><w:fldChar w:fldCharType="separate"/></w:r>' + run(label + " ", bold, hl) +
            f'<w:r><w:rPr><w:noProof/><w:rtl/></w:rPr><w:t>{n}</w:t></w:r>'
            '<w:r><w:rPr><w:rtl/></w:rPr><w:fldChar w:fldCharType="end"/></w:r>')


def rich(s: str, allow_fn=True) -> str:
    out, bold, hl = [], False, False
    for tok in TOKEN.split(s):
        if not tok:
            continue
        if tok == "**":
            bold = not bold
        elif tok == "==":
            hl = not hl
        elif tok.startswith("^["):
            if not allow_fn:
                continue
            global fn_next
            footnotes.append((fn_next, tok[2:-1]))
            out.append(f'<w:r><w:rPr><w:rStyle w:val="FootnoteReference"/><w:rtl/></w:rPr>'
                       f'<w:footnoteReference w:id="{fn_next}"/></w:r>')
            fn_next += 1
        elif tok.startswith("[[c:"):
            nums = []
            for k in tok[4:-2].split(","):
                k = k.strip()
                if k not in C.REFS:
                    raise KeyError(f"unknown reference {k}")
                cites.setdefault(k, len(cites) + 1)
                nums.append(f"[{str(cites[k]).translate(FA)}]")
            out.append(run("، ".join(nums), bold, hl))
        elif tok.startswith("{{"):
            kind, key = tok[2], tok[4:-2]
            if kind == "t":
                out.append(ref_field(tabs[key]["ref"], "جدول", tabs[key]["n"], bold, hl))
            elif kind == "f":
                out.append(ref_field(figs[key]["ref"], "شکل", figs[key]["n"], bold, hl))
            elif kind == "e":
                out.append(run(eqs[key], bold, hl))
            elif kind == "s":
                out.append(run("بخش " + heads[key]["fa"], bold, hl))
        else:
            out.append(run(tok, bold, hl))
    if bold or hl:
        raise ValueError(f"unbalanced markup in: {s[:80]}")
    return "".join(out)


BODY_PPR = '<w:pPr><w:rPr><w:rtl/><w:lang w:bidi="fa-IR"/></w:rPr></w:pPr>'


def para(text):
    return f"<w:p>{BODY_PPR}{rich(text)}</w:p>"


def heading(level, title, bm, extra_end=""):
    i = nbid()
    return (f'<w:p><w:pPr><w:pStyle w:val="Heading{level}"/><w:rPr><w:rtl/><w:lang w:bidi="fa-IR"/></w:rPr></w:pPr>'
            f'<w:bookmarkStart w:id="{i}" w:name="{bm}"/>'
            f'<w:r><w:rPr><w:rFonts w:hint="cs"/><w:rtl/><w:lang w:bidi="fa-IR"/></w:rPr><w:t xml:space="preserve">'
            f'{escape(title)}</w:t></w:r><w:bookmarkEnd w:id="{i}"/>{extra_end}</w:p>')


def bullets(items):
    return "".join(f'<w:p><w:pPr><w:pStyle w:val="ListParagraph"/><w:numPr><w:ilvl w:val="0"/><w:numId w:val="23"/>'
                   f'</w:numPr><w:rPr><w:rtl/><w:lang w:bidi="fa-IR"/></w:rPr></w:pPr>{rich(x)}</w:p>' for x in items)


def caption(style, label, seq, n, ref_bm, toc_bm, text, centre=True):
    a, b = nbid(), nbid()
    jc = '<w:jc w:val="center"/>' if centre else ""
    return (f'<w:p><w:pPr><w:pStyle w:val="{style}"/>{jc}<w:rPr><w:rtl/><w:lang w:bidi="fa-IR"/></w:rPr></w:pPr>'
            f'<w:bookmarkStart w:id="{a}" w:name="{ref_bm}"/><w:bookmarkStart w:id="{b}" w:name="{toc_bm}"/>'
            + run(label + " ") +
            f'<w:r><w:rPr><w:rtl/></w:rPr><w:fldChar w:fldCharType="begin"/><w:instrText xml:space="preserve"> SEQ {seq}'
            f' \\* ARABIC </w:instrText><w:fldChar w:fldCharType="separate"/></w:r>'
            f'<w:r><w:rPr><w:noProof/><w:rtl/></w:rPr><w:t>{n}</w:t></w:r>'
            f'<w:r><w:rPr><w:rtl/></w:rPr><w:fldChar w:fldCharType="end"/></w:r><w:bookmarkEnd w:id="{a}"/>'
            + rich("- " + text, allow_fn=False) + f'<w:bookmarkEnd w:id="{b}"/></w:p>')


TEXT_W = 9070


def table(key, widths, header, rows):
    t = tabs[key]
    assert abs(sum(widths) - 100) < 0.01, (key, sum(widths))
    grid = "".join(f'<w:gridCol w:w="{round(w * TEXT_W / 100)}"/>' for w in widths)

    def cell(text, w, head):
        shd = '<w:shd w:val="clear" w:color="auto" w:fill="FFD6C1"/>' if head else ""
        body = rich(f"**{text}**" if head and text else text)
        return (f'<w:tc><w:tcPr><w:tcW w:w="{round(w * 50)}" w:type="pct"/>{shd}<w:vAlign w:val="center"/></w:tcPr>'
                f'<w:p><w:pPr><w:pStyle w:val="a1"/><w:rPr><w:rtl/></w:rPr></w:pPr>{body}</w:p></w:tc>')

    def row(cells, head):
        trpr = '<w:trPr><w:cantSplit/>' + ('<w:tblHeader/>' if head else '') + '<w:jc w:val="center"/></w:trPr>'
        return f"<w:tr>{trpr}" + "".join(cell(c, w, head) for c, w in zip(cells, widths)) + "</w:tr>"

    cap = caption("CaptionforTables", "جدول", "جدول", t["n"], t["ref"], t["toc"], t["cap"], centre=False)
    return (cap + '<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:bidiVisual/><w:tblW w:w="5000" w:type="pct"/>'
            '<w:jc w:val="center"/><w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" w:firstColumn="1" '
            'w:lastColumn="0" w:noHBand="0" w:noVBand="1"/></w:tblPr>'
            f"<w:tblGrid>{grid}</w:tblGrid>" + row(header, True) + "".join(row(r, False) for r in rows) + "</w:tbl>"
            + f"<w:p>{BODY_PPR}</w:p>")


new_rels: dict[str, str] = {}
docpr = 3000


def figure(key, file, width_in):
    global docpr
    f = figs[key]
    rid = new_rels.setdefault(file, f"rIdFa{len(new_rels) + 1:02d}")
    w_px, h_px = Image.open(FIGS / file).size
    cx = int(width_in * 914400)
    cy = int(cx * h_px / w_px)
    max_cy = int(8.3 * 914400)
    if cy > max_cy:
        cx, cy = int(cx * max_cy / cy), max_cy
    docpr += 1
    A = "http://schemas.openxmlformats.org/drawingml/2006/main"
    PIC = "http://schemas.openxmlformats.org/drawingml/2006/picture"
    pic = (f'<w:p><w:pPr><w:keepNext/><w:jc w:val="center"/></w:pPr><w:r><w:rPr><w:noProof/></w:rPr><w:drawing>'
           f'<wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
           f'<wp:effectExtent l="0" t="0" r="0" b="0"/><wp:docPr id="{docpr}" name="Figure {docpr}"/>'
           f'<wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="{A}" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
           f'<a:graphic xmlns:a="{A}"><a:graphicData uri="{PIC}"><pic:pic xmlns:pic="{PIC}"><pic:nvPicPr>'
           f'<pic:cNvPr id="{docpr}" name="{file}"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="{rid}"/>'
           f'<a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/>'
           f'<a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>'
           f'</a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>')
    return pic + caption("Caption", "شکل", "شکل", f["n"], f["ref"], f["toc"], f["cap"])


def equation(key, math):
    return ('<w:p><w:pPr><w:keepLines/><w:tabs><w:tab w:val="center" w:pos="4535"/></w:tabs><w:bidi w:val="0"/>'
            '<w:spacing w:before="120" w:after="120"/><w:ind w:firstLine="0"/><w:jc w:val="left"/></w:pPr>'
            f'<w:r><w:rPr><w:rFonts w:hint="cs"/><w:lang w:bidi="fa-IR"/></w:rPr><w:t xml:space="preserve">({eqs[key]})</w:t></w:r>'
            f'<w:r><w:tab/></w:r><m:oMath>{math}</m:oMath></w:p>')


def render(blocks, first_extra_end=""):
    out = []
    for i, b in enumerate(blocks):
        k = b[0]
        if k in ("h2", "h3"):
            out.append(heading(2 if k == "h2" else 3, b[2], heads[b[1]]["bm"], first_extra_end if i == 0 else ""))
        elif k == "p":
            out.append(para(b[1]))
        elif k == "bullets":
            out.append(bullets(b[1]))
        elif k == "table":
            out.append(table(b[1], b[3], b[4], b[5]))
        elif k == "fig":
            out.append(figure(b[1], b[3], b[4]))
        elif k == "eq":
            out.append(equation(b[1], b[2]))
        else:
            raise ValueError(k)
    return "".join(out)


# ------------------------------------------------------------------------------------------ chapter regions
def para_end_after(s, i):
    return s.index("</w:p>", i) + len("</w:p>")


def sect_para(rid):
    """(start, end) of the paragraph that holds the section break with footer rId."""
    i = doc.index(f'<w:footerReference w:type="default" r:id="{rid}"/>')
    start = doc.rfind("<w:p ", 0, i)
    return start, para_end_after(doc, i)


def sect_xml(rid):
    i = doc.index(f'<w:footerReference w:type="default" r:id="{rid}"/>')
    a = doc.rfind("<w:sectPr", 0, i)
    return doc[a:doc.index("</w:sectPr>", i) + len("</w:sectPr>")]


bounds = {3: ("rId21", "rId22"), 4: ("rId22", "rId34"), 5: ("rId34", "rId35")}
h1_bm = {}
pieces = []
cursor = 0
for ch in (3, 4, 5):
    a, b = bounds[ch]
    _, region_start = sect_para(a)
    region_end = sect_para(b)[1]
    region = doc[region_start:region_end]
    h2 = region.index('<w:pStyle w:val="Heading2"/>')
    prefix = region[:region.rfind("<w:p ", 0, h2)]
    h1 = re.search(r'<w:pStyle w:val="Heading1"/>.*?<w:bookmarkStart w:id="\d+" w:name="(_Toc\d+)"/>', prefix, re.S)
    h1_bm[ch] = h1.group(1)
    opened = re.findall(r'<w:bookmarkStart w:id="(\d+)"', prefix)
    closed = set(re.findall(r'<w:bookmarkEnd w:id="(\d+)"', prefix))
    extra = "".join(f'<w:bookmarkEnd w:id="{x}"/>' for x in opened if x not in closed)
    body = render(CH[ch], extra)
    tail = f'<w:p><w:pPr><w:rPr><w:rtl/><w:lang w:bidi="fa-IR"/></w:rPr>{sect_xml(b)}</w:pPr></w:p>'
    pieces.append(doc[cursor:region_start])
    pieces.append(prefix + body + tail)
    cursor = region_end
pieces.append(doc[cursor:])
doc = "".join(pieces)

# ------------------------------------------------------------------------------------------ references
i0 = doc.index("<w:t>[1]</w:t>")
s0 = doc.rfind("<w:p ", 0, i0)
last = [m.start() for m in re.finditer(r"<w:t>\[\d+\]</w:t><w:tab/>", doc)][-1]
e0 = para_end_after(doc, last)
missing = [k for k in C.REFS if k not in cites]
if missing:
    print("references never cited (left out):", missing)
ref_ps = []
for k, n in sorted(cites.items(), key=lambda kv: kv[1]):
    text = C.REFS[k]
    hl = text.startswith("==")
    text = text.strip("=")
    hlr = '<w:highlight w:val="yellow"/>' if hl else ""
    rtl = '<w:rFonts w:hint="cs"/>' + hlr + "<w:rtl/>" if hl else ""
    ref_ps.append('<w:p><w:pPr><w:tabs><w:tab w:val="left" w:pos="680"/></w:tabs><w:bidi w:val="0"/>'
                  '<w:spacing w:after="80"/><w:ind w:left="680" w:hanging="680"/></w:pPr>'
                  f'<w:r><w:t>[{n}]</w:t><w:tab/></w:r><w:r><w:rPr>{rtl}</w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>')
doc = doc[:s0] + "".join(ref_ps) + doc[e0:]


# ------------------------------------------------------------------------------------------ TOC, list of figures, list of tables
def para_with_anchor(bm):
    i = doc.index(f'w:anchor="{bm}"')
    return doc.rfind("<w:p ", 0, i), para_end_after(doc, i)


TOC_RPR = {1: '<w:rPr><w:rFonts w:asciiTheme="minorHAnsi" w:eastAsiaTheme="minorEastAsia" w:hAnsiTheme="minorHAnsi" '
              'w:cstheme="minorBidi"/><w:b w:val="0"/><w:color w:val="auto"/><w:sz w:val="22"/><w:szCs w:val="22"/>'
              '<w:rtl/><w:lang w:bidi="ar-SA"/></w:rPr>',
           2: '<w:rPr><w:rFonts w:asciiTheme="minorHAnsi" w:eastAsiaTheme="minorEastAsia" w:hAnsiTheme="minorHAnsi" '
              'w:cstheme="minorBidi"/><w:sz w:val="22"/><w:szCs w:val="22"/><w:rtl/><w:lang w:bidi="ar-SA"/></w:rPr>'}
TOC_RPR[3] = TOC_RPR[2]


def pageref(bm, page):
    return ('<w:r><w:rPr><w:webHidden/><w:rtl/></w:rPr><w:tab/></w:r>'
            '<w:r><w:rPr><w:webHidden/><w:rtl/></w:rPr><w:fldChar w:fldCharType="begin"/></w:r>'
            f'<w:r><w:rPr><w:webHidden/><w:rtl/></w:rPr><w:instrText xml:space="preserve"> PAGEREF {bm} \\h </w:instrText></w:r>'
            '<w:r><w:rPr><w:webHidden/><w:rtl/></w:rPr><w:fldChar w:fldCharType="separate"/></w:r>'
            f'<w:r><w:rPr><w:webHidden/><w:rtl/></w:rPr><w:t>{page}</w:t></w:r>'
            '<w:r><w:rPr><w:webHidden/><w:rtl/></w:rPr><w:fldChar w:fldCharType="end"/></w:r>')


def toc_entry(level, text, bm):
    return (f'<w:p><w:pPr><w:pStyle w:val="TOC{level}"/>{TOC_RPR[level]}</w:pPr><w:hyperlink w:anchor="{bm}" w:history="1">'
            f'<w:r><w:rPr><w:rStyle w:val="Hyperlink"/><w:rtl/></w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'
            f'{pageref(bm, PAGES.get(bm, ""))}</w:hyperlink></w:p>')


titles = {3: "روش پژوهش", 4: "یافته‌ها و تفسیر آن‌ها", 5: "جمع‌بندی"}
entries = []
for ch in (3, 4, 5):
    entries.append(toc_entry(1, f"{ch}) {titles[ch]}", h1_bm[ch]))
    entries += [toc_entry(lv, f"{num} {title}", bm) for lv, num, title, bm, c in toc if c == ch]
a, _ = para_with_anchor("_Toc244546950")
_, b = para_with_anchor("_Toc244546993")
doc = doc[:a] + "".join(entries) + doc[b:]

LOF_PPR = ('<w:pPr><w:pStyle w:val="TableofFigures"/><w:tabs><w:tab w:val="right" w:leader="dot" w:pos="9060"/></w:tabs>'
           '<w:rPr><w:rFonts w:asciiTheme="minorHAnsi" w:eastAsiaTheme="minorEastAsia" w:hAnsiTheme="minorHAnsi" '
           'w:cstheme="minorBidi"/><w:noProof/><w:sz w:val="22"/><w:szCs w:val="22"/><w:rtl/></w:rPr></w:pPr>')


def list_entry(label, n, cap, bm):
    text = TOKEN.sub(lambda m: "", cap)
    return (f'<w:p>{LOF_PPR}<w:hyperlink w:anchor="{bm}" w:history="1"><w:r><w:rPr><w:rStyle w:val="Hyperlink"/><w:noProof/>'
            f'<w:rtl/><w:lang w:bidi="fa-IR"/></w:rPr><w:t xml:space="preserve">{label} {n}- {escape(text)}</w:t></w:r>'
            f'{pageref(bm, PAGES.get(bm, ""))}</w:hyperlink></w:p>')


a, _ = para_with_anchor("_Toc244546999")
_, b = para_with_anchor("_Toc244547009")
doc = doc[:a] + "".join(list_entry("شکل", f["n"], f["cap"], f["toc"]) for f in figs.values()) + doc[b:]
a, _ = para_with_anchor("_Toc244547011")
_, b = para_with_anchor("_Toc244547029")
doc = doc[:a] + "".join(list_entry("جدول", t["n"], t["cap"], t["toc"]) for t in tabs.values()) + doc[b:]

# page numbers of the TOC entries that are kept (references, appendices)
for bm in ("_Toc244546994", "_Toc244546995", "_Toc244546996"):
    if bm in PAGES and f'w:anchor="{bm}"' in doc:
        a, b = para_with_anchor(bm)
        p = doc[a:b]
        k = p.rfind('w:fldCharType="separate"')
        m = re.compile(r"(<w:t[^>]*>)([^<]*)(</w:t>)").search(p, k)
        if m:
            p = p[:m.start()] + m.group(1) + PAGES[bm] + m.group(3) + p[m.end():]
            doc = doc[:a] + p + doc[b:]

# ------------------------------------------------------------------------------------------ footnotes, images, settings
fnx = re.sub(r'<w:footnote w:id="(\d+)">.*?</w:footnote>',
             lambda m: "" if int(m.group(1)) >= 20 else m.group(0), fnx, flags=re.S)
new_fn = "".join(f'<w:footnote w:id="{i}"><w:p><w:pPr><w:pStyle w:val="FootnoteText"/><w:bidi w:val="0"/></w:pPr>'
                 f'<w:r><w:rPr><w:rStyle w:val="FootnoteReference"/></w:rPr><w:footnoteRef/></w:r>'
                 f'<w:r><w:t xml:space="preserve"> {escape(t)}</w:t></w:r></w:p></w:footnote>' for i, t in footnotes)
fnx = fnx.replace("</w:footnotes>", new_fn + "</w:footnotes>")

used = set(re.findall(r'r:embed="([^"]+)"', doc)) | set(re.findall(r'r:id="([^"]+)"', doc))
for m in re.finditer(r'<Relationship Id="([^"]+)" Type="[^"]+/image" Target="(media/[^"]+)"/>', rels):
    if m.group(1) not in used:
        rels = rels.replace(m.group(0), "")
        (W / m.group(2)).unlink(missing_ok=True)
add = "".join(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
              f'Target="media/{f}"/>' for f, rid in new_rels.items())
rels = rels.replace("</Relationships>", add + "</Relationships>")
for f in new_rels:
    shutil.copy(FIGS / f, W / "media" / f)

st = (W / "settings.xml").read_text(encoding="utf-8")
if "<w:updateFields" not in st:
    for tag in ("<w:hdrShapeDefaults", "<w:footnotePr", "<w:endnotePr", "<w:compat"):
        if tag in st:
            st = st.replace(tag, '<w:updateFields w:val="true"/>' + tag, 1)
            break
(W / "settings.xml").write_text(st, encoding="utf-8")

(W / "document.xml").write_text(doc, encoding="utf-8")
(W / "footnotes.xml").write_text(fnx, encoding="utf-8")
(W / "_rels" / "document.xml.rels").write_text(rels, encoding="utf-8")
json.dump(dict(heads={k: v["bm"] for k, v in heads.items()}, toc=[(lv, num, t, bm) for lv, num, t, bm, _ in toc],
               h1=h1_bm, figs={k: v["toc"] for k, v in figs.items()}, tabs={k: v["toc"] for k, v in tabs.items()},
               captions={**{v["toc"]: f"شکل {v['n']}" for v in figs.values()},
                         **{v["toc"]: f"جدول {v['n']}" for v in tabs.values()}}),
          open(OUT.parent / "structure.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"tables {tno - 1}, figures {fno - 1}, equations {len(eqs)}, footnotes {len(footnotes)}, references {len(cites)}")

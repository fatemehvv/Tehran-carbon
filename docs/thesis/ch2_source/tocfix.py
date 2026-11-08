# -*- coding: utf-8 -*-
"""Table of contents and lists of figures/tables for the built thesis.

  python tocfix.py structure <unpacked dir>          add bookmarks to the new chapter-2 headings, replace the template's
                                                     chapter-2 entries of the table of contents, fix the Figure 1 entry,
                                                     and ask Word to refresh all fields on opening
  python tocfix.py pages <unpacked dir> <thesis.pdf> write page numbers into every entry, taken from a rendering
Page numbers from a LibreOffice rendering can differ slightly from Word's; Word refreshes them on opening (updateFields).
"""
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import content as C  # noqa: E402
from docxutil import body_elems, esc, ptext  # noqa: E402

mode, D = sys.argv[1], Path(sys.argv[2])
DOC = D / "word/document.xml"
x = DOC.read_text(encoding="utf-8")


def elems():
    return [(s, e, t, x[s:e]) for s, e, t in body_elems(x)]


def is_entry(p):
    return re.search(r'w:pStyle w:val="(TOC\d|TableofFigures)"', p) and "w:anchor=" in p


def set_entry(p, anchor=None, text=None, page=None):
    if anchor:
        p = re.sub(r'w:anchor="[^"]+"', f'w:anchor="{anchor}"', p)
        p = re.sub(r"(<w:instrText[^>]*>)(\d{6,})( </w:instrText>)", r"\g<1>" + anchor.replace("_Toc", "") + r"\3", p)
    if text is not None:                  # all text runs before the tab -> one run with the new text
        a = p.index("<w:hyperlink")
        a = p.index(">", a) + 1
        b = p.index("<w:tab/>", a)
        b = max(p.rfind("<w:r>", a, b), p.rfind("<w:r ", a, b))      # start of the run that holds the tab
        rpr = re.search(r"<w:rPr>.*?</w:rPr>", p[a:b], flags=re.S)
        p = p[:a] + f'<w:r>{rpr.group(0) if rpr else ""}<w:t xml:space="preserve">{esc(text)}</w:t></w:r>' + p[b:]
    if page is not None:                  # cached result of the PAGEREF field
        p = re.sub(r'(<w:fldChar w:fldCharType="separate"/></w:r>\s*<w:r(?: [^>]*)?>(?:<w:rPr>.*?</w:rPr>)?<w:t(?: [^>]*)?>)[^<]*(</w:t>)',
                   lambda m: m.group(1) + page + m.group(2), p, count=1, flags=re.S) if p.count("fldCharType=\"separate\"") == 1 else \
            re.sub(r'(PAGEREF.*?<w:fldChar w:fldCharType="separate"/></w:r>\s*<w:r(?: [^>]*)?>(?:<w:rPr>.*?</w:rPr>)?<w:t(?: [^>]*)?>)[^<]*(</w:t>)',
                   lambda m: m.group(1) + page + m.group(2), p, count=1, flags=re.S)
    return p


if mode == "structure":
    E = elems()
    h2 = next(i for i, (s, e, t, p) in enumerate(E) if 'w:val="Heading1"' in p and ptext(p) == "مروری بر منابع")
    h3 = next(i for i in range(h2 + 1, len(E)) if 'w:val="Heading1"' in E[i][3])
    bid = max(int(i) for i in re.findall(r'<w:bookmarkStart w:id="(\d+)"', x)) + 1
    new_heads, edits = [], {}
    n2 = n3 = 0
    for i in range(h2 + 1, h3):
        p = E[i][3]
        lv = re.search(r'w:val="Heading([23])"', p)
        if not lv:
            continue
        if lv.group(1) == "2":
            n2, n3 = n2 + 1, 0
            label = f"2 ـ {n2}) {ptext(p)}"
        else:
            n3 += 1
            label = f"2 ـ {n2} ـ {n3}) {ptext(p)}"
        name = f"_Toc9{bid:08d}"
        a = p.index("</w:pPr>") + len("</w:pPr>")
        edits[i] = p[:a] + f'<w:bookmarkStart w:id="{bid}" w:name="{name}"/>' + p[a:].replace("</w:p>", f'<w:bookmarkEnd w:id="{bid}"/></w:p>', 1)
        new_heads.append((int(lv.group(1)), name, label))
        bid += 1
    # table-of-contents entries of chapter 2 (between "2) مروری بر منابع" and "3) روش پژوهش")
    T = [i for i, (s, e, t, p) in enumerate(E) if re.search(r'w:pStyle w:val="TOC\d"', p) and "w:anchor=" in p]
    t2 = next(i for i in T if ptext(E[i][3]).startswith("2) مروری بر منابع"))
    t3 = next(i for i in T if ptext(E[i][3]).startswith("3) روش پژوهش"))
    old = list(range(t2 + 1, t3))
    tmpl = {2: next(E[i][3] for i in old if 'w:val="TOC2"' in E[i][3]), 3: next(E[i][3] for i in old if 'w:val="TOC3"' in E[i][3])}
    edits[old[0]] = "".join(set_entry(tmpl[lv], anchor=name, text=label, page="0") for lv, name, label in new_heads)
    for i in old[1:]:
        edits[i] = ""
    # Figure 1 entry in the list of figures
    f1 = next(i for i, (s, e, t, p) in enumerate(E) if 'w:val="TableofFigures"' in p and ptext(p).startswith("شکل 1-"))
    edits[f1] = set_entry(E[f1][3], text="شکل 1" + C.FIGURE_CAPTION)
    out, pos = [], 0
    for i in sorted(edits):
        out += [x[pos:E[i][0]], edits[i]]
        pos = E[i][1]
    x = "".join(out) + x[pos:]
    DOC.write_text(x, encoding="utf-8")
    st = D / "word/settings.xml"
    s = st.read_text(encoding="utf-8")
    if "w:updateFields" not in s:
        # schema order: updateFields comes right before hdrShapeDefaults / footnotePr / ... / compat / rsids
        nxt = min(i for i in (s.find(f"<{t}") for t in ("w:hdrShapeDefaults", "w:footnotePr", "w:endnotePr", "w:compat",
                                                          "w:docVars", "w:rsids", "m:mathPr", "w:themeFontLang",
                                                          "w:clrSchemeMapping", "w:shapeDefaults", "w:decimalSymbol")) if i >= 0)
        s = s[:nxt] + '<w:updateFields w:val="true"/>' + s[nxt:]
        st.write_text(s, encoding="utf-8")
    print("chapter-2 headings:", len(new_heads), "entries replaced:", len(old))

elif mode == "pages":
    PDF = Path(sys.argv[3])
    n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(PDF)], capture_output=True, text=True).stdout).group(1))
    BIDI = dict.fromkeys(map(ord, "‪‫‬‭‮‎‏"), None)

    def norm(t):
        t = t.translate(BIDI).replace("ي", "ی").replace("ك", "ک").replace("ۀ", "ه").replace("ة", "ه")
        t = t.replace("لا", "ال").replace("ﻻ", "ال").replace("ؤ", "و").replace("أ", "ا")      # PDF text splits the lam-alef
        return re.sub(r"[^\w]|_|\d|[۰-۹]", "", t)
    pages, labels, heads, is_list = [], [], [], []
    for k in range(1, n + 1):
        t = subprocess.run(["pdftotext", "-f", str(k), "-l", str(k), str(PDF), "-"], capture_output=True, text=True).stdout
        lines = [l.translate(BIDI).strip() for l in t.splitlines() if l.translate(BIDI).strip()]
        bb = subprocess.run(["pdftotext", "-f", str(k), "-l", str(k), "-bbox", str(PDF), "-"], capture_output=True, text=True).stdout
        foot = [w.translate(BIDI).strip() for y, w in re.findall(r'<word xMin="[^"]+" yMin="([^"]+)"[^>]*>([^<]*)</word>', bb)
                if float(y) > 740]                           # the page number sits in the footer
        lab = next((w for w in foot if re.fullmatch(r"\d+|[A-Za-z]|[\u0600-\u06FF]{1,2}", w)), "")
        labels.append(lab)
        pages.append(norm(t))
        heads.append(norm(" ".join(lines[:4])))              # a heading that opens a page is in its first lines
        is_list.append(t.count("......") >= 5)                # table of contents and the lists of figures and tables
    E = elems()
    entries = [(i, p) for i, (s, e, t, p) in enumerate(E) if is_entry(p)]
    body0 = x.index('w:val="Heading1"', x.index("<w:body>"))
    body0 = next(m.start() for m in re.finditer(r'<w:pStyle w:val="Heading1"/>', x) if "مقدمه" in ptext(x[m.start():x.index("</w:p>", m.start())]))
    targets = {}
    for i, p in entries:
        a = re.search(r'w:anchor="([^"]+)"', p).group(1)
        m = re.search(rf'<w:bookmarkStart w:id="\d+" w:name="{a}"/>', x)
        if not m:
            continue
        ps = max(x.rfind("<w:p ", 0, m.start()), x.rfind("<w:p>", 0, m.start()))
        if x.rfind("</w:p>", ps, m.start()) >= 0:            # bookmark between paragraphs: it marks the next one
            ps = min(i for i in (x.find("<w:p ", m.start()), x.find("<w:p>", m.start())) if i >= 0)
        pe = x.index("</w:p>", ps)
        targets[i] = (m.start(), norm(ptext(x[ps:pe]))[:24], m.start() < body0)
    first_body = next(k for k, l in enumerate(labels) if l == "1")
    order = sorted(targets.items(), key=lambda kv: kv[1][0])        # by position of the target in the document
    found = {}
    cur = first_body
    for i, (pos, key, front) in order:
        if front:                                                    # front matter: the page that opens with it
            k = next((k for k in range(0, first_body) if key and key[:12] in heads[k]), None)
        else:
            k = next((k for k in range(cur, n) if key and key in pages[k] and not is_list[k]), None)
            if k is not None:
                cur = k
        if k is None:
            print("not found:", ptext(E[i][3])[:60]); continue
        found[i] = k
    edits = {i: set_entry(E[i][3], page=labels[k]) for i, k in found.items() if labels[k]}
    out, pos = [], 0
    for i in sorted(edits):
        out += [x[pos:E[i][0]], edits[i]]
        pos = E[i][1]
    x = "".join(out) + x[pos:]
    DOC.write_text(x, encoding="utf-8")
    print("entries numbered:", len(edits), "of", len(entries))

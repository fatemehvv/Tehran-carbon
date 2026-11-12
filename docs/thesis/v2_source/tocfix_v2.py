# -*- coding: utf-8 -*-
"""Table of contents and lists of figures/tables for version 2 of the thesis.

  python tocfix_v2.py structure <unpacked dir>          rebuild every chapter entry of the table of contents (chapters 1-5)
                                                        from the headings, rebuild both lists from the captions, add
                                                        bookmarks where a heading or caption has none, and ask Word to
                                                        refresh all fields on opening
  python tocfix_v2.py pages <unpacked dir> <thesis.pdf> write page numbers into every entry, taken from a rendering
Page numbers from a LibreOffice rendering can differ slightly from Word's; Word refreshes them on opening (updateFields).
"""
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
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
        p, k = re.subn(r"(<w:instrText[^>]*>)(\d{6,})( </w:instrText>)", r"\g<1>" + anchor.replace("_Toc", "") + r"\3", p)
        assert k == 1, "PAGEREF target not found"
    if text is not None:                  # all text runs before the tab -> one run with the new text
        a = p.index("<w:hyperlink")
        a = p.index(">", a) + 1
        b = p.index("<w:tab/>", a)
        b = max(p.rfind("<w:r>", a, b), p.rfind("<w:r ", a, b))      # start of the run that holds the tab
        rpr = re.search(r"<w:rPr>.*?</w:rPr>", p[a:b], flags=re.S)
        p = p[:a] + f'<w:r>{rpr.group(0) if rpr else ""}<w:t xml:space="preserve">{esc(text)}</w:t></w:r>' + p[b:]
    if page is not None:                  # cached result of the PAGEREF field (the last field result in the entry)
        h = p.index("<w:hyperlink")
        m = list(re.finditer(r'(<w:fldChar w:fldCharType="separate"/></w:r>\s*<w:r(?: [^>]*)?>(?:<w:rPr>.*?</w:rPr>)?<w:t(?: [^>]*)?>)[^<]*(</w:t>)',
                             p[h:], flags=re.S))[-1]
        p = p[:h + m.start()] + m.group(1) + page + m.group(2) + p[h + m.end():]
    return p


def splice(x, E, edits):
    out, pos = [], 0
    for i in sorted(edits):
        out += [x[pos:E[i][0]], edits[i]]
        pos = E[i][1]
    return "".join(out) + x[pos:]


if mode == "structure":
    E = elems()
    bid = max(int(i) for i in re.findall(r'<w:bookmarkStart w:id="(\d+)"', x)) + 1
    edits = {}

    def anchor_of(i):
        """Bookmark of heading/caption i; one is added if it has none."""
        global bid
        p = edits.get(i, E[i][3])
        m = re.search(r'<w:bookmarkStart w:id="\d+" w:name="(_Toc\d+)"/>', p)
        if m:
            return m.group(1)
        name = f"_Toc9{bid:08d}"
        a = p.index("</w:pPr>") + len("</w:pPr>")
        edits[i] = p[:a] + f'<w:bookmarkStart w:id="{bid}" w:name="{name}"/>' + p[a:].replace("</w:p>", f'<w:bookmarkEnd w:id="{bid}"/></w:p>', 1)
        bid += 1
        return name

    # ---- table of contents: chapters 1-5 (numbered headings from "1) مقدمه" up to "مراجع")
    h1 = next(i for i, (s, e, t, p) in enumerate(E) if 'w:val="Heading1"' in p and ptext(p) == "مقدمه")
    ref = next(i for i in range(h1, len(E)) if 'w:val="Heading1"' in E[i][3] and ptext(E[i][3]) == "مراجع")
    heads, c = [], [0, 0, 0]
    for i in range(h1, ref):
        p = E[i][3]
        lv = re.search(r'w:pStyle w:val="Heading([123])"', p)
        if not lv:
            continue
        lv = int(lv.group(1))
        assert '<w:numId w:val="0"/>' not in p, ptext(p)      # every heading of chapters 1-5 is numbered
        c[lv - 1] += 1
        c[lv:] = [0] * (3 - lv)
        label = " ـ ".join(str(n) for n in c[:lv]) + ") " + ptext(p).strip()
        heads.append((lv, anchor_of(i), label))
    T = [i for i, (s, e, t, p) in enumerate(E) if re.search(r'w:pStyle w:val="TOC\d"', p) and "w:anchor=" in p]
    t1 = next(i for i in T if ptext(E[i][3]).startswith("1) مقدمه"))
    tr = next(i for i in T if ptext(E[i][3]).startswith("مراجع"))
    old = [i for i in T if t1 <= i < tr]
    assert old == list(range(t1, tr)) and not any("instrText>TOC<" in E[i][3] for i in old)
    tmpl = {lv: next(E[i][3] for i in old if f'w:val="TOC{lv}"' in E[i][3]) for lv in (1, 2, 3)}
    edits[old[0]] = "".join(set_entry(tmpl[lv], anchor=a, text=label, page="0") for lv, a, label in heads)
    for i in old[1:]:
        edits[i] = ""

    # ---- lists of figures and tables, from the captions in document order
    def rebuild(prefix, style):
        caps = [i for i, (s, e, t, p) in enumerate(E) if f'w:pStyle w:val="{style}"' in p and ptext(p).startswith(prefix)]
        items = [(anchor_of(i), ptext(E[i][3]).strip()) for i in caps]
        nums = [int(re.match(prefix + r" (\d+)-", t).group(1)) for a, t in items]
        assert nums == list(range(1, len(nums) + 1)), (prefix, nums)
        L = [i for i, (s, e, t, p) in enumerate(E) if 'w:val="TableofFigures"' in p and "w:anchor=" in p and ptext(p).startswith(prefix)]
        assert L == list(range(L[0], L[-1] + 1)) and len(items) >= len(L) >= 3
        plain = E[L[1]][3]
        assert "fldCharType=\"begin\"" not in plain.split("<w:hyperlink")[0] and plain.rstrip().endswith("</w:hyperlink></w:p>")
        src = [E[i][3] for i in L[:-1]] + [plain] * (len(items) - len(L)) + [E[L[-1]][3]]
        new = [set_entry(sp, anchor=a, text=t, page="0") for sp, (a, t) in zip(src, items)]
        edits[L[0]] = "".join(new)
        for i in L[1:]:
            edits[i] = ""
        return len(items), len(L)

    nf = rebuild("شکل", "Caption")
    nt = rebuild("جدول", "CaptionforTables")
    x = splice(x, E, edits)
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
    print("headings:", len(heads), "toc entries replaced:", len(old), "figures (new, old):", nf, "tables (new, old):", nt)

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
        lab = next((w for w in foot if re.fullmatch(r"\d+|[A-Za-z]|[؀-ۿ]{1,2}", w)), "")
        labels.append(lab)
        pages.append(norm(t))
        heads.append(norm(lines[0]) if lines else "")         # a front-matter heading opens its page: its first line
        is_list.append(t.count("......") >= 5)                # table of contents and the lists of figures and tables
    E = elems()
    entries = [(i, p) for i, (s, e, t, p) in enumerate(E) if is_entry(p)]
    body0 = next(m.start() for m in re.finditer(r'<w:pStyle w:val="Heading1"/>', x) if ptext(x[m.start():x.index("</w:p>", m.start())]) == "مقدمه")
    targets = {}
    for i, p in entries:
        a = re.search(r'w:anchor="([^"]+)"', p).group(1)
        m = re.search(rf'<w:bookmarkStart w:id="\d+" w:name="{a}"/>', x)
        if not m:
            print("no bookmark:", ptext(p)[:60]); continue
        ps = max(x.rfind("<w:p ", 0, m.start()), x.rfind("<w:p>", 0, m.start()))
        if x.rfind("</w:p>", ps, m.start()) >= 0:            # bookmark between paragraphs: it marks the next one
            ps = min(i for i in (x.find("<w:p ", m.start()), x.find("<w:p>", m.start())) if i >= 0)
        pe = x.index("</w:p>", ps)
        targets[i] = (m.start(), norm(ptext(x[ps:pe]))[:24], m.start() < body0)
    first_body = next(k for k, l in enumerate(labels) if l == "1")
    order = sorted(targets.items(), key=lambda kv: kv[1][0])        # by position of the target in the document
    found, missing = {}, []
    cur = first_body
    for i, (pos, key, front) in order:
        if front:                                                    # front matter: the page that opens with it
            k = next((k for k in range(0, first_body) if key and key[:12] in heads[k]), None)
        else:
            k = next((k for k in range(cur, n) if key and key in pages[k] and not is_list[k]), None)
            if k is not None:
                cur = k
        if k is None:
            missing.append(ptext(E[i][3])[:60]); continue
        found[i] = k
    for m in missing:
        print("not found:", m)
    edits = {i: set_entry(E[i][3], page=labels[k]) for i, k in found.items() if labels[k]}
    x = splice(x, E, edits)
    DOC.write_text(x, encoding="utf-8")
    print("entries numbered:", len(edits), "of", len(entries))

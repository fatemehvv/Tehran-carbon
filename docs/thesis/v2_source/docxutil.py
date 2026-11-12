# -*- coding: utf-8 -*-
"""Small helpers shared by build.py and tocfix.py."""
import html
import re

FA = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
EN = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")


def fa(n) -> str:
    return str(n).translate(FA)


def esc(s: str) -> str:
    return html.escape(s, quote=False)


def ptext(p: str) -> str:
    return "".join(re.findall(r"<w:t(?: [^>]*)?>([^<]*)</w:t>", p))


def body_elems(x: str):
    """top-level elements of w:body (paragraphs and tables) as (start, end, tag)"""
    b0, b1 = x.index("<w:body>") + len("<w:body>"), x.rindex("</w:body>")
    out, i = [], b0
    pat = re.compile(r"<w:(p|tbl|sectPr|bookmarkStart|bookmarkEnd)[ >/]")
    while True:
        m = pat.search(x, i)
        if not m or m.start() >= b1:
            return out
        tag, s = m.group(1), m.start()
        if tag in ("bookmarkStart", "bookmarkEnd"):
            i = x.index(">", s) + 1
            continue
        depth, e = 0, None
        for mm in re.compile(rf"<w:{tag}(?=[ >/])|</w:{tag}>").finditer(x, s):
            if mm.group(0).startswith("</"):
                depth -= 1
            elif x[x.index(">", mm.start()) - 1] == "/":
                if depth == 0:
                    e = x.index(">", mm.start()) + 1
                    break
                continue
            else:
                depth += 1
            if depth == 0:
                e = mm.end()
                break
        out.append((s, e, tag))
        i = e

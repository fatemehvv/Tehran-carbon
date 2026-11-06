"""Tiny Office Math (OMML) builder for the thesis equations."""
from xml.sax.saxutils import escape

RPR = '<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/></w:rPr>'


def t(s, plain=False):
    sty = '<m:rPr><m:sty m:val="p"/></m:rPr>' if plain else ""
    return f'<m:r>{sty}{RPR}<m:t xml:space="preserve">{escape(s)}</m:t></m:r>'


def p(s):
    """Upright text (names such as NCV, EF, softmax)."""
    return t(s, plain=True)


def sub(b, s):
    return f"<m:sSub><m:e>{b}</m:e><m:sub>{s}</m:sub></m:sSub>"


def sup(b, s):
    return f"<m:sSup><m:e>{b}</m:e><m:sup>{s}</m:sup></m:sSup>"


def subsup(b, s, u):
    return f"<m:sSubSup><m:e>{b}</m:e><m:sub>{s}</m:sub><m:sup>{u}</m:sup></m:sSubSup>"


def frac(n, d):
    return f"<m:f><m:num>{n}</m:num><m:den>{d}</m:den></m:f>"


def par(e, beg="(", end=")"):
    return (f'<m:d><m:dPr><m:begChr m:val="{beg}"/><m:endChr m:val="{end}"/></m:dPr>'
            f"<m:e>{e}</m:e></m:d>")


def nsum(lo, e, hi=None):
    hide = "" if hi is not None else '<m:supHide m:val="1"/>'
    return (f'<m:nary><m:naryPr><m:chr m:val="∑"/><m:limLoc m:val="undOvr"/>{hide}</m:naryPr>'
            f"<m:sub>{lo}</m:sub><m:sup>{hi or ''}</m:sup><m:e>{e}</m:e></m:nary>")


def J(*xs):
    return "".join(xs)

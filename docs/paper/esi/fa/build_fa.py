"""Persian study version of the Earth Science Informatics paper: paper_esi_fa.md -> paper_esi_fa.pdf.

The Markdown is converted by pandoc (MathML for the equations) and printed to an A4, right-to-left PDF by the
preinstalled Chromium (Playwright), with the Vazirmatn font. The figures are those of the English version
(../figures), and the reference list is generated from ../references.bib, so it stays the same as in English.

    python docs/paper/esi/fa/build_fa.py [font-folder]

font-folder holds Vazirmatn-Regular.ttf and Vazirmatn-Bold.ttf; without it the font is fetched with `npm pack vazirmatn`
(OFL licence). Needs pandoc, pdftoppm (poppler) and node with Playwright.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ESI = HERE.parent
SRC, OUT = HERE / "paper_esi_fa.md", HERE / "paper_esi_fa.pdf"

CSS = """
@font-face { font-family: Vazirmatn; font-weight: 400; src: url('fonts/Vazirmatn-Regular.ttf'); }
@font-face { font-family: Vazirmatn; font-weight: 700; src: url('fonts/Vazirmatn-Bold.ttf'); }
@page { size: A4; margin: 17mm 16mm 18mm 16mm; }
html { direction: rtl; }
body { font-family: Vazirmatn, sans-serif; font-size: 10.5pt; line-height: 1.85; color: #111; text-align: justify;
       margin: 0; max-width: none; }
header { text-align: center; margin-bottom: 4mm; }
h1.title { font-size: 17pt; line-height: 1.6; margin: 0 0 3mm; }
.en-title { direction: ltr; text-align: center; font-size: 10pt; color: #555; font-style: italic; margin-bottom: 3mm; }
.authors { text-align: center; font-size: 9.5pt; color: #333; margin-bottom: 4mm; }
.note { border: 1px solid #c9d6e8; background: #f3f7fc; padding: 3mm 4mm; font-size: 9pt; border-radius: 2mm;
        margin: 0 0 6mm; }
h2 { font-size: 13.5pt; margin: 7mm 0 2mm; border-bottom: 1px solid #ddd; padding-bottom: 1mm; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 5mm 0 1.5mm; break-after: avoid; }
p { margin: 0 0 2.5mm; }
code { font-family: monospace; direction: ltr; unicode-bidi: embed; font-size: 9pt; }
table { border-collapse: collapse; width: 100%; font-size: 8.3pt; line-height: 1.55; margin: 1mm 0 5mm;
        break-inside: auto; }
th, td { border-bottom: 1px solid #ddd; padding: 1.2mm 1.5mm; vertical-align: top; text-align: right; }
th { background: #f1f1ef; font-weight: 700; border-bottom: 1.5px solid #888; }
tr { break-inside: avoid; }
.table-caption, .figure-caption { font-size: 8.8pt; line-height: 1.7; color: #222; margin: 1mm 0 2mm; }
.figure-caption { margin-bottom: 6mm; }
img { display: block; max-width: 100%; margin: 4mm auto 1mm; }
img.small { max-width: 68%; }
img.tall { max-height: 215mm; width: auto; }
p:has(> img) { break-inside: avoid; break-after: avoid; }
math { direction: ltr; }
math[display="block"] { margin: 3mm 0; font-size: 10.5pt; }
.refs { direction: ltr; text-align: left; font-size: 8.5pt; line-height: 1.5; }
.refs p { margin: 0 0 1.6mm; padding-left: 5mm; text-indent: -5mm; }
ol { padding-right: 6mm; }
"""

PERSIAN = "؀-ۿ"
PAREN = re.compile(r"(?<![\]\w])\(([^()\n" + PERSIAN + r"$]*[A-Za-z][^()\n" + PERSIAN + r"$]*)\)")


def ltr_parentheses(text: str) -> str:
    """Wrap Latin parentheses (citations, model names) in an LTR span so they read correctly inside Persian lines.
    Math ($...$), display equations and image lines are left alone."""
    out = []
    for line in text.splitlines():
        if line.startswith("$$") or line.startswith("!["):
            out.append(line)
            continue
        parts = line.split("$")
        for i in range(0, len(parts), 2):                     # even parts are outside inline math
            parts[i] = PAREN.sub(lambda m: f'<span dir="ltr">({m.group(1)})</span>', parts[i])
        out.append("$".join(parts))
    return "\n".join(out)


def bib_entries(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    entries = []
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.S):
        fields = {k.lower(): v for k, v in re.findall(r"\n\s*(\w+) = \{(.*?)\},?(?=\n)", m.group(3), re.S)}
        fields["key"] = m.group(2)
        entries.append(fields)
    return entries


def clean(s: str) -> str:
    s = s.replace("{\\O}", "Ø").replace("--", "–").replace("\\&", "&")
    return re.sub(r"[{}]", "", s).strip()


def short_name(name: str) -> str:
    fam, _, given = name.partition(", ")
    initials = "".join(p[0] for p in re.split(r"[\s\-]+", given) if p and p[0].isalpha())
    return f"{clean(fam)} {initials}".strip()


def reference(e: dict) -> str:
    authors = ", ".join(short_name(a.strip()) for a in e.get("author", "").split(" and "))
    s = f"{authors} ({e.get('year', '')}) {clean(e.get('title', ''))}."
    venue = e.get("journal") or e.get("booktitle") or e.get("howpublished") or e.get("school") or ""
    if venue:
        s += f" {clean(venue)}"
        if e.get("volume"):
            s += f" {e['volume']}" + (f"({e['number']})" if e.get("number") else "")
        if e.get("pages"):
            s += f":{clean(e['pages'])}"
        s += "."
    if e.get("doi"):
        s += f" https://doi.org/{e['doi']}"
    return s


def references_md() -> str:
    entries = sorted(bib_entries(ESI / "references.bib"),
                     key=lambda e: (clean(e.get("author", "")).split(",")[0].lower(), e.get("year", "")))
    body = "\n\n".join(reference(e) for e in entries)
    return "\n\n## منابع\n\nفهرست منابع همان فهرست نسخهٔ انگلیسی است.\n\n::: {.refs dir=\"ltr\"}\n" + body + "\n:::\n"


def font_dir(arg: str | None, tmp: Path) -> Path:
    if arg:
        return Path(arg)
    subprocess.run(["npm", "pack", "vazirmatn", "--silent"], cwd=tmp, check=True, capture_output=True)
    tgz = next(tmp.glob("vazirmatn-*.tgz"))
    subprocess.run(["tar", "-xzf", tgz.name], cwd=tmp, check=True)
    return tmp / "package" / "fonts" / "ttf"


def main() -> None:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "figures").mkdir()
        (tmp / "fonts").mkdir()
        fdir = font_dir(sys.argv[1] if len(sys.argv) > 1 else None, tmp)
        for name in ("Vazirmatn-Regular.ttf", "Vazirmatn-Bold.ttf"):
            shutil.copy(fdir / name, tmp / "fonts" / name)
        for n in range(1, 8):                                  # figures of the English version, as PNG
            pdf, png = ESI / "figures" / f"Fig{n}.pdf", ESI / "figures" / f"Fig{n}.png"
            if pdf.exists():
                subprocess.run(["pdftoppm", "-r", "170", "-png", "-singlefile", str(pdf), str(tmp / "figures" / f"Fig{n}")],
                               check=True)
            else:
                shutil.copy(png, tmp / "figures" / f"Fig{n}.png")
        (tmp / "style.css").write_text(CSS, encoding="utf-8")
        md = ltr_parentheses(SRC.read_text(encoding="utf-8")) + references_md()
        (tmp / "paper.md").write_text(md, encoding="utf-8")
        subprocess.run(["pandoc", "paper.md", "-f", "markdown+raw_html+fenced_divs+pipe_tables", "-t", "html5", "-s",
                        "--mathml", "--css", "style.css", "-o", "paper.html"], cwd=tmp, check=True)
        js = tmp / "print.js"
        js.write_text("""
const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage();
  await p.goto('file://' + process.argv[2], { waitUntil: 'load' });
  await p.evaluate(() => document.fonts.ready);
  await p.pdf({ path: process.argv[3], format: 'A4', printBackground: true, preferCSSPageSize: true,
    displayHeaderFooter: true, headerTemplate: '<div></div>',
    footerTemplate: '<div style="font-size:8px;width:100%;text-align:center;color:#666"><span class="pageNumber"></span></div>' });
  await b.close();
})();
""", encoding="utf-8")
        subprocess.run(["node", str(js), str(tmp / "paper.html"), str(OUT)], check=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()

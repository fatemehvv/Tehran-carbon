# -*- coding: utf-8 -*-
"""Port the supervisor-review fixes of the RQ1-RQ2 chapters file into the full four-question chapters file.

Usage: python port_review_fixes.py <unpacked thesis_chapters_3-5.docx> <unpacked thesis_chapters_3-5_RQ1-RQ2.docx> <out dir>

The fixes (commits 95b8124 and fd456ca on the RQ1-RQ2 file): section 3-2 keeps only the development / held-out split;
tools and data sources are named in footnotes instead of cited; OCO-2/OCO-3 text and Table 2 rewritten; NO2-as-proxy
paragraph; grid and area-share paragraphs; hash and automated-test lines removed; VIIRS sentence corrected.
Paragraphs that exist in both files are copied from the fixed file (its citations mapped to the full file's numbers,
its footnotes copied); paragraphs that only the full file has (simulator, RL) get the same edits in place."""
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from docxutil import EN, FA, body_elems, ptext as text  # noqa: E402

FULL, FIXED, OUT = map(Path, sys.argv[1:4])
if OUT.exists():
    shutil.rmtree(OUT)
shutil.copytree(FULL, OUT)
x = (FULL / "word/document.xml").read_text(encoding="utf-8")
xf = (FIXED / "word/document.xml").read_text(encoding="utf-8")
els = body_elems(x); efs = body_elems(xf)
fn_full = (FULL / "word/footnotes.xml").read_text(encoding="utf-8")
fn_fix = (FIXED / "word/footnotes.xml").read_text(encoding="utf-8")

def ref_list(xx, ee):
    head = next(i for i, (s, e, t) in enumerate(ee) if t == "p" and 'w:val="Heading1"' in xx[s:e] and text(xx[s:e]) == "مراجع")
    out = {}
    for s, e, t in ee[head + 1:]:
        m = re.match(r"\[(\d+)\](.*)", text(xx[s:e]))
        if m: out[int(m.group(1))] = re.sub(r"\s+", " ", m.group(2)).strip()
    return out
rf, rx = ref_list(x, els), ref_list(xf, efs)
by_text = {v: k for k, v in rf.items()}
fix2full = {n: by_text[t] for n, t in rx.items()}          # every fixed reference exists in the full list
assert len(fix2full) == len(rx), set(rx) - set(fix2full)

fid_next = [max(int(i) for i in re.findall(r'<w:footnote [^>]*w:id="(\d+)"', fn_full)) + 1]
new_notes = []
def copy_note(fid):
    m = re.search(rf'<w:footnote [^>]*w:id="{fid}"[^>]*>.*?</w:footnote>', fn_fix, flags=re.S)
    nid = fid_next[0]; fid_next[0] += 1
    new_notes.append(re.sub(r'w:id="\d+"', f'w:id="{nid}"', m.group(0), count=1))
    return nid

def transplant(k):
    s, e, t = efs[k]; seg = xf[s:e]
    seg = re.sub(r' w14:(paraId|textId)="[^"]*"', "", seg)
    seg = re.sub(r'<w:footnoteReference w:id="(\d+)"/>', lambda m: f'<w:footnoteReference w:id="{copy_note(m.group(1))}"/>', seg)
    def cite(m):
        return m.group(1) + re.sub(r"\[([۰-۹]+)\]", lambda c: "[" + str(fix2full[int(c.group(1).translate(EN))]).translate(FA) + "]", m.group(2)) + m.group(3)
    return re.sub(r"(<w:t(?: [^>]*)?>)([^<]*)(</w:t>)", cite, seg)

def replace_text(seg, old, new):
    """replace `old` (may span runs) by `new` inside one paragraph/table; new text goes into the first touched w:t"""
    nodes = list(re.finditer(r"(<w:t(?: [^>]*)?>)([^<]*)(</w:t>)", seg))
    full = "".join(n.group(2) for n in nodes)
    i = full.find(old)
    assert i >= 0 and full.find(old, i + 1) < 0, (old, full[:200])
    j = i + len(old)
    out, pos, acc = [], 0, 0
    for n in nodes:
        a, b = acc, acc + len(n.group(2)); acc = b
        txt = n.group(2)
        if b <= i or a >= j:
            continue
        lo, hi = max(i, a) - a, min(j, b) - a
        rep = new if a <= i < b else ""
        txt2 = txt[:lo] + rep + txt[hi:]
        out.append((n.start(2), n.end(2), txt2, n.group(1)))
    res = seg
    for s0, e0, t2, open_tag in reversed(out):
        res = res[:s0] + t2 + res[e0:]
    # keep leading/trailing spaces visible
    res = re.sub(r"<w:t>([^<]*)</w:t>", lambda m: f'<w:t xml:space="preserve">{m.group(1)}</w:t>' if m.group(1)[:1] == " " or m.group(1)[-1:] == " " else m.group(0), res)
    return res

def P(k): s, e, t = els[k]; return x[s:e]
new = {}            # full element index -> replacement xml ("" deletes)
def check(k, start):
    assert text(P(k)).startswith(start), (k, text(P(k))[:60])

check(333, "سه اصل روش‌شناختی"); new[333] = transplant(309)
check(334, "ابزارها."); seg = P(334)
for c in "۲۳۴۵۶۷":
    seg = replace_text(seg, f" [{c}]", "")
seg = replace_text(seg, " ۴۹ آزمون خودکار درستی پیکربندی، تقسیم داده، معیارها، مدل‌ها و شبیه‌ساز را بررسی می‌کنند.", "")
new[334] = seg
check(336, "انتخاب متغیر هدف"); new[336] = transplant(312)
check(337, "گزینه‌های مستقیم‌تر"); new[337] = transplant(313)
check(338, "ماهوارۀ سوم"); new[338] = transplant(314) + transplant(315)
assert els[340][2] == "tbl" and els[317 - 0][2]; new[340] = transplant(317)
check(343, "انتخاب مدل گراف"); new[343] = transplant(320)
check(344, "انتخاب روش ارزیابی"); new[344] = transplant(321)
for k in range(348, 356):
    seg = P(k); t = text(seg)
    seg = replace_text(seg, t[t.rindex(" ("):], "") if t.rstrip().endswith((")؛", ").")) else seg
    if " (پرسش‌های اول و دوم)" in text(seg):
        seg = replace_text(seg, " (پرسش‌های اول و دوم)", "")
    new[k] = seg
check(357, "واحد تحلیل"); new[357] = transplant(329) + transplant(330)
check(358, "محدودۀ مرکزی"); new[358] = transplant(331) + transplant(332)
check(361, "داده‌ها در هفت گروه"); new[361] = transplant(335)
assert els[363][2] == "tbl"
t3 = transplant(337)
rows = re.findall(r"<w:tr[ >].*?</w:tr>", t3, flags=re.S)
for r in rows:
    cells = re.findall(r"<w:tc>.*?</w:tc>", r, flags=re.S)
    ct = [text(c) for c in cells]
    if ct[2].startswith("آمار سوخت ناوگان اتوبوسرانی") or ct[2].startswith("آمار شهرداری تهران و سازمان حمل‌ونقل"):
        last = cells[-1]; t3 = t3.replace(last, replace_text(last, ct[-1], ct[-1] + "؛ شبیه‌ساز"), 1)
new[363] = t3
check(367, "روشنایی شبانه، جمعیت و ساختمان"); new[367] = transplant(341)
check(450, "مدل اصلی."); seg = P(450)
seg = replace_text(seg, "کتابخانۀ استاندارد یادگیری گراف پیاده شدند [۳].", "کتابخانۀ استاندارد یادگیری گراف پیاده شدند."); new[450] = seg
check(491, "عامل یادگیری تقویتی."); new[491] = replace_text(P(491), "پیاده‌سازی استاندارد آن [۶]", "پیاده‌سازی استاندارد آن")
check(509, "صحت‌سنجی در پنج سطح"); new[509] = replace_text(P(509), "پنج سطح", "شش سطح")
check(514, "آزمون‌های خودکار و تکرارپذیری"); seg = replace_text(P(514), "آزمون‌های خودکار و تکرارپذیری:", "تکرارپذیری:")
seg = replace_text(seg, "۴۹ آزمون خودکار درستی کد را بررسی می‌کنند و عامل", "عامل"); new[514] = seg
check(535, "همۀ ۴۹ آزمون خودکار"); new[535] = ""

parts, pos = [], 0
for k in sorted(new):
    s, e, t = els[k]
    parts.append(x[pos:s]); parts.append(new[k]); pos = e
parts.append(x[pos:])
x2 = "".join(parts)
(OUT / "word/document.xml").write_text(x2, encoding="utf-8")
fn2 = fn_full.replace("</w:footnotes>", "".join(new_notes) + "</w:footnotes>")
(OUT / "word/footnotes.xml").write_text(fn2, encoding="utf-8")
print("edited elements:", len(new), "new footnotes:", len(new_notes))
print("fixed->full refs:", fix2full)

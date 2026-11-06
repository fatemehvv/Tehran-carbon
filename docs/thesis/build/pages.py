import json, re, subprocess, sys
pdf, structure, out = sys.argv[1:4]
st = json.load(open(structure, encoding="utf-8"))
n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout).group(1))
txt = [subprocess.run(["pdftotext", "-f", str(i), "-l", str(i), pdf, "-"], capture_output=True, text=True).stdout for i in range(1, n + 1)]
norm = lambda s: re.sub(r"[^ء-ۓ۰-۹]", "", s.replace("‌", "").replace("ـ", ""))
T = [norm(t) for t in txt]
def find(s, start):
    s = norm(s)
    for i in range(start, n):
        if s in T[i]:
            return i
    return None
def find_label(s, start):
    s = norm(s)
    for i in range(start, n):
        if s in T[i] and len(T[i]) < 80:
            return i
    return None
ch1 = find_label("فصل اول", 0)
pages, miss = {}, []
labels = {3: "فصل سوم", 4: "فصل چهارم", 5: "فصل پنجم"}
cur = ch1
chap_start = {}
for ch in (3, 4, 5):
    chap_start[ch] = find_label(labels[ch], cur)
    pages[st["h1"][str(ch)]] = chap_start[ch] - ch1 + 1
    cur = chap_start[ch]
cur = chap_start[3]
for lv, num, title, bm in st["toc"]:
    i = find(title, cur)
    if i is None:
        i = find(title.split()[-1], cur)
    if i is None:
        miss.append(title); continue
    pages[bm] = i - ch1 + 1
    cur = i
for bm, label in (("_Toc244546994", "مراجع"), ("_Toc244546995", "پیوست‌ها")):
    i = find_label(label, chap_start[5] + 1)
    if i is not None:
        pages[bm] = i - ch1 + 1
i = find("پیوست الف", chap_start[5] + 1)
if i is not None and "_Toc244546995" in pages and i - ch1 + 1 >= pages["_Toc244546995"]:
    pages["_Toc244546996"] = i - ch1 + 1
elif "_Toc244546995" in pages:
    pages["_Toc244546996"] = pages["_Toc244546995"] + 1
caps = json.load(open(sys.argv[4], encoding="utf-8")) if len(sys.argv) > 4 else {}
prev = chap_start[3]
for bm, (label, cap) in caps.items():
    i = None
    for k in (40, 25, 15, 10):
        i = find(cap[:k], prev)
        if i is not None:
            break
    if i is None:
        i = prev
    prev = i
    if i is None:
        miss.append(cap); continue
    pages[bm] = i - ch1 + 1
pages = {k: str(v) for k, v in pages.items()}
json.dump(pages, open(out, "w"), indent=1)
print("found", len(pages), "missing", miss)

"""Build the offline single-file explainer from reach_explainer.src.html.

The source loads three.js r128 from cdnjs (used for the hosted artifact).
This script inlines three.min.js so the page also opens without internet:

    npm pack three@0.128.0 && tar -xzf three-0.128.0.tgz package/build/three.min.js
    python docs/paper/explainer/build.py package/build/three.min.js
"""
import sys
from pathlib import Path

HERE = Path(__file__).parent
CDN = '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>'

src = (HERE / "reach_explainer.src.html").read_text(encoding="utf-8")
three = Path(sys.argv[1]).read_text(encoding="utf-8")
head, body = src.split("<!--BODY-->", 1)
assert CDN in body
body = body.replace(CDN, "<script>\n" + three + "\n</script>")
out = (
    '<!doctype html>\n<html lang="fa" dir="rtl">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
    + head + "</head>\n<body>\n" + body + "</body>\n</html>\n"
)
(HERE / "reach_explainer.html").write_text(out, encoding="utf-8")
print("wrote", HERE / "reach_explainer.html", len(out) // 1024, "KB")

"""Fig2.png: panel letters a-c added to the thesis map figure (docs/figures/fig06_maps_1404.png), as Springer asks.

The map figure itself is drawn by `python main.py figures` from run outputs that are not in the repository, so the
letters are added to the 300-dpi image: a 40-px white margin on the left, and bold lowercase letters at the size of
the panel titles (8 pt), left of each title.

    python docs/paper/esi/figures/add_panel_letters.py
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[2] / "figures" / "fig06_maps_1404.png"
PAD = 40
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 33)
im = Image.open(SRC).convert("RGB")
out = Image.new("RGB", (im.width + PAD, im.height), "white")
out.paste(im, (PAD, 0))
d = ImageDraw.Draw(out)
# (letter, x of its left edge, baseline): left of the titles at rows 137-165 (a, b) and 30-61 (c)
for letter, x, base in [("a", 6, 161), ("b", 662 + PAD - 36, 161), ("c", 1293 + PAD - 36, 55)]:
    d.text((x, base), letter, font=font, fill="#0b0b0b", anchor="ls")
out.save(HERE / "Fig2.png", dpi=(300, 300))
print("wrote", HERE / "Fig2.png", out.size)

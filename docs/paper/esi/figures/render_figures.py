"""Journal copies of the plotted figures (Fig3, Fig5-Fig7) for the Earth Science Informatics version.

Draws the same figures as `python main.py paper figures` (src/tehran_carbon/viz/paper_figures.py) from the committed
result tables, with two changes for the journal: no lettering below 8 pt at the figures' own width (Springer asks for
8-12 pt at final size), and "five-seed" spelled out as in the text. The originals in docs/paper/figures/ are not touched.

    python docs/paper/esi/figures/render_figures.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "src"))

from tehran_carbon.viz import paper_figures as pf  # noqa: E402

MIN_PT = 8.0
pf.LABEL.update({"SAGE": "GNN, five-seed ensemble", "MLP": "No-graph network, five-seed ensemble"})
_save = pf.save


def save(fig, stem):
    """Raise every text to at least MIN_PT and spell out 'five-seed' before saving."""
    from matplotlib.text import Text
    for t in fig.findobj(Text):
        if t.get_text():
            if t.get_fontsize() < MIN_PT:
                t.set_fontsize(MIN_PT)
            t.set_text(t.get_text().replace("5-seed", "five-seed"))
    return _save(fig, stem)


pf.save = save

NAMES = {"fig3_correlogram": "Fig3", "fig4_reach": "Fig5", "fig5_differences": "Fig6", "fig6_seeds": "Fig7"}

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        figs = pf.PaperFigures(out=Path(tmp))
        figs.correlogram(); figs.reach(); figs.differences(); figs.seeds()
        for src, dst in NAMES.items():
            shutil.copy(Path(tmp) / f"{src}.pdf", HERE / f"{dst}.pdf")
            print("wrote", HERE / f"{dst}.pdf")

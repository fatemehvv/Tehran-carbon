"""Paper figures (viz/paper_figures.py) from the committed stage A tables: every figure is written, and the numbers
behind figures 5 and 6 match the evaluation tables."""
import numpy as np
import pytest

pytest.importorskip("matplotlib")

from tehran_carbon.viz.paper_figures import RESULTS, PaperFigures  # noqa: E402

pytestmark = pytest.mark.skipif(not (RESULTS / "reach" / "reach_curve.csv").exists(),
                                reason="stage A result tables not present")


def test_all_figures_written(tmp_path):
    paths = PaperFigures(out=tmp_path).all()
    names = sorted({p.stem for p in paths})
    assert names == ["fig3_correlogram", "fig4_reach", "fig5_differences", "fig6_seeds"]
    assert all(p.exists() and p.stat().st_size > 5_000 for p in paths)
    assert (tmp_path / "figure_data_comparisons.csv").exists()


def test_comparison_numbers(tmp_path):
    d = PaperFigures(out=tmp_path).comparisons()
    assert len(d) == 12                                       # 6 comparisons x 2 metrics
    held_out = d[(d.group == "matched") & (d.year == 1404) & (d.metric == "spearman")].iloc[0]
    assert np.isclose(held_out.delta, 0.0042, atol=5e-4)      # manuscript 4.3: +0.004 [-0.026, +0.035]
    assert held_out.lo95 < 0 < held_out.hi95
    seeds = d[(d.group == "matched") & (d.year == 1402) & (d.metric == "pr_auc")].iloc[0]
    assert np.isclose(seeds.seed_min, 0.7486, atol=1e-3)      # equals reach_curve.csv pr_auc_seed_min of SAGE 1402
    assert seeds.seed_min <= seeds.seed_mean <= seeds.seed_max

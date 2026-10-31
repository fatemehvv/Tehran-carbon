import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, roc_auc_score

from tehran_carbon.config import Config
from tehran_carbon.data.grid import Grid
from tehran_carbon.data.splits import SpatialSplit
from tehran_carbon.evaluation.metrics import monthly_scores, pr_auc, roc_auc
from tehran_carbon.utils.calendar import JalaliYear


def test_jalali_years():
    a, b = JalaliYear(1402), JalaliYear(1403)
    assert a.spans[0][0] == pd.Timestamp("2023-03-21") and sum(a.days) == 365 and a.days[-1] == 29
    assert b.spans[0][0] == pd.Timestamp("2024-03-20") and sum(b.days) == 366 and b.days[-1] == 30
    assert a.gregorian_months[0] == (2023, 4) and a.gregorian_months[9] == (2024, 1) and a.gregorian_months[11] == (2024, 3)
    assert a.gregorian_year == 2023
    assert list(b.month_of(["2024-03-20", "2025-03-19", "2025-03-21"])) == [1, 12, 0]


def small_grid(H=30, W=40):
    cfg = Config().defaults
    return Grid({"grid": {**cfg["grid"], "H": H, "W": W}})


@pytest.mark.parametrize("scheme", ["block_lobo", "block_kfold", "random_kfold"])
def test_split_covers_domain_and_respects_buffer(scheme):
    g = small_grid()
    dom = np.ones((g.H, g.W), bool)
    dom[:3] = False
    s = SpatialSplit({"scheme": scheme, "block_km": 8, "buffer_km": 2.0, "n_folds": 5}, g, dom)
    assert ((s.fold_map >= 0) == dom).all()
    for k in range(s.n_folds):
        tr, te = s.masks(k)
        assert not (tr & te).any() and te.any()
        if s.buffer_km > 0:
            ty, tx = np.where(te)
            ry, rx = np.where(tr)
            d = np.hypot((ry[:, None] - ty[None]) * g.km_y, (rx[:, None] - tx[None]) * g.km_x).min()
            assert d > s.buffer_km
        va = s.inner_val(tr, k)
        assert not (va & ~tr).any()


def test_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    for _ in range(5):
        s = rng.normal(size=500)
        y = (s + rng.normal(size=500)) > 1.2
        assert pr_auc(y, s) == pytest.approx(average_precision_score(y, s), abs=1e-9)
        assert roc_auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-9)
    z = rng.normal(size=(3, 200))
    p = z + 0.1 * rng.normal(size=z.shape)
    sc = monthly_scores(z, z > 1.0, p, ["pr_auc", "spearman", "r2", "mae", "rmse"])
    assert all(v.shape == (3,) for v in sc.values()) and (sc["spearman"] > 0.9).all()
    assert sc["mae"][0] == pytest.approx(mean_absolute_error(z[0], p[0]))
    assert sc["rmse"][0] == pytest.approx(np.sqrt(mean_squared_error(z[0], p[0])))
    # missing targets (clouds) are ignored per step; an empty step gives NaN
    zg = z.copy()
    zg[0, :50] = np.nan
    zg[1] = np.nan
    sg = monthly_scores(zg, zg > 1.0, p, ["mae"])["mae"]
    assert sg[0] == pytest.approx(mean_absolute_error(z[0, 50:], p[0, 50:])) and np.isnan(sg[1])


def test_jalali_days():
    assert len(JalaliYear(1402).dates) == 365 and len(JalaliYear(1403).dates) == 366
    assert JalaliYear(1403).dates[0] == pd.Timestamp("2024-03-20")

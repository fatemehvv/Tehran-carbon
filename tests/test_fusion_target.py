"""Stage 2 of the fusion paper: wind composites and smoothed targets without leakage (synthetic data only)."""
import numpy as np

from tehran_carbon.data.sources.composite import composite, select_days
from tehran_carbon.data.target_support import smooth_field


def test_calm_and_windy_thirds_have_equal_days_and_the_right_winds():
    rng = np.random.default_rng(0)
    month = np.repeat(np.arange(1, 13), 30)
    wind = rng.random(len(month)) * 6
    clear = rng.random(len(month)) > 0.2
    for per in select_days(month, wind, clear, "season", 1 / 3):
        assert per["place"] in (2, 5, 8, 11)
        assert len(per["calm"]) == len(per["windy"]) > 0
        assert wind[per["calm"]].max() <= wind[per["windy"]].min()          # calm days are the calmest
        assert set(per["calm"]) | set(per["windy"]) <= set(per["all"])
        assert clear[per["all"]].all()                                      # only clear days
        assert np.isin(month[per["all"]], {2: [1, 2, 3], 5: [4, 5, 6], 8: [7, 8, 9], 11: [10, 11, 12]}[per["place"]]).all()


def test_composite_needs_min_days():
    d = np.full((4, 2, 2), np.nan)
    d[:, 0, 0] = [1.0, 2.0, 3.0, 4.0]
    d[0, 1, 1] = 5.0
    c = composite(d, np.arange(4), min_days=2)
    assert c[0, 0] == 2.5 and np.isnan(c[1, 1]) and np.isnan(c[0, 1])


def test_smoothed_labels_do_not_see_the_test_block():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(30, 40))
    train = np.ones_like(z, bool)
    train[10:20, 15:25] = False                                             # test block (+ buffer)
    a = smooth_field(z, train, (3.0, 3.0))
    z2 = z.copy()
    z2[10:20, 15:25] += 100.0                                               # change the test block only
    b = smooth_field(z2, train, (3.0, 3.0))
    assert np.allclose(a[train], b[train])                                  # training labels unchanged: no leak
    assert np.isnan(a[~train]).all()
    assert abs(np.nanmean(a)) < 1e-9 and abs(np.nanstd(a) - 1) < 1e-9       # standardised again


def test_smoothing_makes_the_field_smoother():
    rng = np.random.default_rng(2)
    z = rng.normal(size=(30, 40))
    mask = np.ones_like(z, bool)
    s = smooth_field(z, mask, (2.0, 2.0))
    lag = lambda m: np.corrcoef(m[:, :-1].ravel(), m[:, 1:].ravel())[0, 1]  # noqa: E731
    assert lag(s) > lag(z) + 0.5

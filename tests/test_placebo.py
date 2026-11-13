"""Placebo branches of the fusion paper (data/sources/placebo.py). Synthetic maps only: no data files needed."""
import numpy as np

from tehran_carbon.data.sources.base import SOURCES, Source
from tehran_carbon.data.sources.placebo import PlaceboSource, iaaft_multi, remap

H, W = 24, 40


def city(seed=0):
    """A smooth 'city': two Gaussian blobs and a little noise, values >= 0."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    m = 5 * np.exp(-((yy - 10) ** 2 + (xx - 15) ** 2) / 30.0) + 2 * np.exp(-((yy - 16) ** 2 + (xx - 30) ** 2) / 15.0)
    return m + 0.05 * rng.random((H, W))


def lag1(m):
    a, b = m[:, :-1].ravel(), m[:, 1:].ravel()
    return np.corrcoef(a, b)[0, 1]


def test_remap_gives_exact_values_in_rank_order():
    v = np.array([[3.0, 1.0], [2.0, 4.0]])
    out = remap(v, np.array([10.0, 20.0, 30.0, 40.0]))
    assert np.array_equal(out, np.array([[30.0, 10.0], [20.0, 40.0]]))


def test_surrogate_keeps_values_and_texture_but_moves_the_city():
    a, b = city(0), city(1) ** 2
    sa, sb = iaaft_multi([a, b], np.random.default_rng(5), iterations=50)
    assert np.array_equal(np.sort(sa.ravel()), np.sort(a.ravel()))          # identical marginal distribution
    assert np.array_equal(np.sort(sb.ravel()), np.sort(b.ravel()))
    assert abs(lag1(sa) - lag1(a)) < 0.1                                     # similar spatial smoothness
    assert abs(np.corrcoef(sa.ravel(), a.ravel())[0, 1]) < 0.9               # not the same map
    r_real = np.corrcoef(a.ravel(), b.ravel())[0, 1]                         # the two layers stay related
    r_plc = np.corrcoef(sa.ravel(), sb.ravel())[0, 1]
    assert r_plc > 0.3 * r_real


def test_surrogate_is_deterministic_and_depends_on_the_realisation():
    a = city(0)
    s1 = iaaft_multi([a], np.random.default_rng(3), iterations=20)[0]
    s2 = iaaft_multi([a], np.random.default_rng(3), iterations=20)[0]
    s3 = iaaft_multi([a], np.random.default_rng(4), iterations=20)[0]
    assert np.array_equal(s1, s2)
    assert not np.array_equal(s1, s3)


def test_constant_map_is_left_alone():
    c = np.full((H, W), 2.5)
    assert np.array_equal(iaaft_multi([c], np.random.default_rng(0))[0], c)


if "fake_domain" not in SOURCES:
    @SOURCES.register("fake_domain")
    class FakeDomain(Source):
        name = "fake_domain"

        def load(self, year):
            share = np.linspace(0.5, 1.5, 12) / 12.0
            annual = city(0) * (1.0 + 0.01 * (int(year) - 1402))
            return {"layer_m": share[:, None, None] * annual[None], "layer_s": city(2), "cw": np.broadcast_to(
                np.arange(12.0)[:, None, None], (12, H, W)).copy()}


def placebo(realization=0):
    cfg = {"seed": 0, "paths": {"raw": "data/raw"}, "data": {"inputs": {"fake_domain": {}, "placebo": {
        "realization": realization, "iterations": 20, "prefix": "plc_",
        "domains": {"fake_domain": ["layer_m", "layer_s"]}, "copy": {"fake_domain": ["cw"]}}}}}
    src = PlaceboSource.__new__(PlaceboSource)
    src.cfg, src.grid, src.p = cfg, None, cfg["data"]["inputs"]["placebo"]
    return src


def test_placebo_source_layers_shapes_seasonal_cycle_and_copies():
    out = placebo().load(1402)
    assert set(out) == {"plc_layer_m", "plc_layer_s", "plc_cw"}
    real = SOURCES.get("fake_domain").load(None, 1402)
    assert out["plc_layer_m"].shape == (12, H, W) and out["plc_layer_s"].shape == (H, W)
    # the city-wide monthly cycle of a monthly layer is kept exactly
    assert np.allclose(out["plc_layer_m"].reshape(12, -1).mean(1), real["layer_m"].reshape(12, -1).mean(1))
    assert np.array_equal(out["plc_cw"], real["cw"])                         # copies are unchanged
    assert np.array_equal(np.sort(out["plc_layer_s"].ravel()), np.sort(real["layer_s"].ravel()))


def test_placebo_location_is_the_same_in_every_year_and_differs_by_realisation():
    a, b = placebo(0).load(1402), placebo(0).load(1403)
    m_a, m_b = a["plc_layer_m"].mean(0).ravel(), b["plc_layer_m"].mean(0).ravel()
    assert np.corrcoef(m_a, m_b)[0, 1] > 0.99                                # same random location across years
    c = placebo(1).load(1402)
    assert np.corrcoef(m_a, c["plc_layer_m"].mean(0).ravel())[0, 1] < 0.95


def test_rotate_shift_keeps_values_cycle_and_moves_the_city():
    src = placebo(0)
    src.p = dict(src.p, method="rotate_shift")
    out = src.load(1402)
    real = SOURCES.get("fake_domain").load(None, 1402)
    m, r = out["plc_layer_m"], real["layer_m"]
    assert np.array_equal(np.sort(m.ravel()), np.sort(r.ravel()))            # same values
    assert np.allclose(m.reshape(12, -1).mean(1), r.reshape(12, -1).mean(1))  # same monthly cycle
    assert abs(lag1(m[0]) - lag1(r[0])) < 0.15                               # same texture (up to the seams)
    assert np.corrcoef(m.mean(0).ravel(), r.mean(0).ravel())[0, 1] < 0.9     # the city has moved
    assert np.array_equal(out["plc_cw"], real["cw"])

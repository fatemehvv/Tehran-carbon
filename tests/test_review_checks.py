"""Review round: helpers of analysis/review_checks.py and the new experiments / hypotheses of configs/config.yaml."""
import numpy as np

from tehran_carbon.analysis.review_checks import detrend, idw
from tehran_carbon.config import Config


def test_detrend_removes_plane():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 50, (400, 2))
    v = (2.0 + 0.3 * xy[:, 0] - 0.1 * xy[:, 1])[None] + 0.0 * rng.normal(size=(1, 400))
    r = detrend(v, xy, 1)
    assert np.nanmax(np.abs(r)) < 1e-8


def test_idw_exact_at_sites_and_bounded():
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    v = np.array([1.0, 2.0, 3.0])
    p = idw(xy, v, np.array([[0.0, 0.0], [5.0, 5.0]]))
    assert abs(p[0] - 1.0) < 1e-6 and 1.0 <= p[1] <= 3.0


def test_review_experiments_resolve():
    c = Config()
    assert len(c.resolve("paper2_rq2_depth_v1")) == 4
    assert len(c.resolve("paper2_rq2_mlp_lag_v1")) == 2
    assert len(c.resolve("paper2_slx_short_v1")) == 12
    assert len(c.resolve("paper2_slx_buffer_v2")) == 9
    depth = {cfg["model"]["backbone"]["layers"] for cfg in c.resolve("paper2_rq2_depth_v1")}
    assert depth == {1, 2, 4, 5}
    for cfg in c.resolve("paper2_rq2_mlp_lag_v1"):
        assert cfg["model"]["backbone"]["type"] == "none" and cfg["train"]["save_members"]
    for cfg in c.resolve("paper2_slx_buffer_v2"):
        assert cfg["split"]["buffer_km"] in (2.0, 10.0, 20.0) and cfg["model"]["baseline"]["lags"] == [1, 3, 9, 15, 27]
        assert cfg["model"]["modalities"] == ["traffic", "energy"]


def test_review_hypotheses_name_existing_experiments():
    c = Config()
    known = set(c.raw["experiments"]) if hasattr(c, "raw") else None
    for name, spec in c.hypotheses.items():
        if not name.startswith("PR_"):
            continue
        for sel in [spec["candidate"]] + spec["references"]:
            assert c.resolve(sel["experiment"]), (name, sel["experiment"])
            if known is not None:
                assert sel["experiment"] in known

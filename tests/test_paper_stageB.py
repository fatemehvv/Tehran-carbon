"""Paper v2, stage B (branch paper_TC): weighted block test, random search, reach curve helpers."""
import copy
import json

import numpy as np
import pandas as pd

from tehran_carbon.analysis.reach import family, reach_cells, zero_crossing
from tehran_carbon.config import Config
from tehran_carbon.evaluation.inference import signflip_mean
from tehran_carbon.experiments.tuning import sample


def test_weighted_signflip_mean():
    d = np.array([0.1, 0.2, 0.05, 0.3, 0.15, 0.12])
    assert np.isclose(signflip_mean(d, weights=np.ones(6))["p_greater"], signflip_mean(d)["p_greater"])
    s = signflip_mean(np.array([0.1, 0.2, -5.0]), weights=np.array([10, 10, 0]))     # a zero-weight block is dropped
    assert s["blocks"] == 2 and np.isclose(s["p_greater"], 0.25)
    big = signflip_mean(np.array([0.1, -0.05]), weights=np.array([100, 1]))
    assert big["stat"] > 0                            # the large block dominates the weighted mean


def test_random_search_draws():
    space = {"a": {"choice": [1, 2, 3]}, "b": {"loguniform": [1e-4, 1e-2]}, "c": {"uniform": [0.0, 0.5]},
             "d": {"intuniform": [2, 4]}, "e": {"choice": [[1, 3], [9]]}}
    draws = [sample(space, np.random.default_rng(0)) for _ in range(2)]
    assert draws[0] == draws[1]                       # same seed, same draw
    many = [sample(space, r) for r in [np.random.default_rng(1)] for _ in range(200)]
    assert {x["a"] for x in many} == {1, 2, 3} and {x["d"] for x in many} == {2, 3, 4}
    assert all(1e-4 <= x["b"] <= 1e-2 and 0 <= x["c"] <= 0.5 for x in many)
    assert {tuple(x["e"]) for x in many} == {(1, 3), (9,)}


def test_reach_of_the_stage_b_configs():
    conf = Config()
    sage = conf.resolve("paper3_reach_sage_v1")
    assert [reach_cells(c) for c in sage] == [9.0 * L for L in range(1, 7)] and family(sage[0]) == "SAGE"
    slx = conf.resolve("paper3_reach_slx_v1")
    assert sorted({reach_cells(c) for c in slx}) == [0, 1, 3, 9, 15, 18, 27, 36, 45, 54]
    assert {family(c) for c in slx} == {"SLX-OLS", "SLX-GBM"}
    (mlp,) = conf.resolve("paper3_reach_mlp_v1")
    assert reach_cells(mlp) == 0 and family(mlp) == "MLP"
    (tune,) = conf.resolve("paper3_tune_sage_v1")
    assert tune["experiment"]["kind"] == "tune" and reach_cells(tune) == 36   # the reach rule: 4 layers


def test_zero_crossing():
    assert np.isclose(zero_crossing(np.array([0.5, 0.2, -0.2]), np.array([0.0, 2.0, 4.0, 6.0])), 4.0)
    assert np.isnan(zero_crossing(np.array([0.5, 0.4]), np.array([0.0, 2.0, 4.0])))


def test_random_search_end_to_end(tmp_path, monkeypatch):
    from test_robustness import GH, GW, fake_dataset
    from tehran_carbon.experiments import tuning
    from tehran_carbon.tracking import Run
    rng = np.random.default_rng(0)
    data = fake_dataset(rng)
    for y in data.years:                              # an informative feature: the target plus noise
        data.X[y][..., 0] = (np.nan_to_num(data.z[y]) + 0.5 * rng.normal(size=data.z[y].shape)).astype(np.float32)
    monkeypatch.setattr(tuning.CellMonthDataset, "build", classmethod(lambda cls, cfg, **kw: data))
    cfg = copy.deepcopy(Config().defaults)
    cfg["paths"]["runs"] = str(tmp_path / "runs")
    cfg["grid"].update(H=GH, W=GW)
    cfg["split"]["block_km"] = 6
    cfg["model"].update(type="baseline", modalities=[])
    cfg["model"]["baseline"].update(name="ols", lags=[])
    cfg["tune"] = {"trials": 3, "seed": 0, "inner_folds": 3, "metric": "spearman", "years": [1402, 1403],
                   "space": {"model.baseline.lags": {"choice": [[], [1]]}}}
    cfg["experiment"] = dict(name="tune_v1", rq="RQ2", kind="tune", description="", version=1, variant={},
                             variant_label="", overrides={})
    with Run(cfg) as run:
        out = tuning.RandomSearch(cfg, run).run_all()
    t = pd.read_csv(run.dir / "tune_trials.csv")
    best = json.loads((run.dir / "tune_best.json").read_text(encoding="utf-8"))
    assert len(t) == 3 and np.isfinite(t.score).all() and best["trials"] == 3
    assert np.isclose(best["best_score"], t.score.max()) and out["best_trial"] == best["best_trial"]

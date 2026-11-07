"""Paper checks (branch paper_TC): robustness of a comparison and the sector-yield physical baseline."""
import copy

import numpy as np
import pytest
from scipy.stats import spearmanr

from tehran_carbon.config import Config
from tehran_carbon.evaluation.robustness import (block_table, block_tests, jackknife, mean_score, persistence_maps,
                                                 reading)
from tehran_carbon.models.baselines import BASELINES

T, N, K = 4, 60, 6                                    # months, domain cells, blocks of 10 cells


def toy(seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(T, N))
    hot = z >= np.quantile(z, 0.8, axis=1, keepdims=True)
    blocks = np.repeat(np.arange(K), N // K)
    return z, hot, blocks, rng


def test_block_table_perfect_candidate_wins_every_block():
    z, hot, blocks, rng = toy()
    pc, pr = z.copy(), rng.normal(size=z.shape)
    t = block_table(z, hot, pc, pr, blocks, "pr_auc")
    assert len(t) == K and t.cells.sum() == N
    assert np.isclose(t.hotspot_share.sum(), 1.0)
    valid = t.dropna(subset=["delta"])
    assert (valid.candidate == 1.0).all() and (valid.delta >= 0).all()
    s = block_tests(t)
    assert s["blocks_won"] == int((valid.delta > 0).sum()) and s["blocks_valid"] == len(valid)


def test_block_without_hotspots_is_skipped():
    z, hot, blocks, rng = toy(1)
    hot[:, blocks == 0] = False                        # block 0 never holds a hotspot
    t = block_table(z, hot, z, rng.normal(size=z.shape), blocks, "pr_auc")
    assert t.loc[0, "months"] == 0 and np.isnan(t.loc[0, "delta"])


def test_jackknife_matches_direct_leave_one_out():
    z, hot, blocks, rng = toy(2)
    pc, pr = z + 0.5 * rng.normal(size=z.shape), rng.normal(size=z.shape)
    table, s = jackknife(z, hot, pc, pr, blocks, "pr_auc", 0.95)
    idx = np.flatnonzero(blocks != 3)
    direct = mean_score(z, hot, pc, "pr_auc", idx) - mean_score(z, hot, pr, "pr_auc", idx)
    assert np.isclose(table.delta[3], direct)
    full = mean_score(z, hot, pc, "pr_auc") - mean_score(z, hot, pr, "pr_auc")
    assert s["jack_low"] <= full <= s["jack_high"] and s["jack_se"] >= 0
    assert s["jack_positive"] == K                     # a good candidate stays ahead whatever block is left out


def test_identical_models_give_no_effect():
    z, hot, blocks, rng = toy(3)
    p = rng.normal(size=z.shape)
    table, s = jackknife(z, hot, p, p, blocks, "pr_auc", 0.95)
    assert np.allclose(table.delta, 0) and s["jack_se"] == 0
    assert block_tests(block_table(z, hot, p, p, blocks, "pr_auc"))["blocks_won"] == 0


def test_persistence_maps():
    class D:
        years = [1402, 1403, 1404]
        cells = np.array([0, 2, 3])
        z = {y: np.arange(12 * 4, dtype=float).reshape(12, 2, 2) + y for y in years}
    maps = dict(persistence_maps(D, 1404))
    assert list(maps) == ["1402", "1403", "mean_1402_1403"]
    assert maps["1403"].shape == (12, 3) and np.allclose(maps["1403"][:, 0], D.z[1403][:, 0, 0])
    assert np.allclose(maps["mean_1402_1403"], (maps["1402"] + maps["1403"]) / 2)
    D.z[1402] = np.zeros((365, 2, 2))                  # a daily year has other steps: left out
    assert list(dict(persistence_maps(D, 1404))) == ["1403"]


@pytest.mark.parametrize("boot_low, delta, jack_low, won, valid, expected", [
    (0.01, 0.05, 0.02, 9, 12, "robust"),
    (0.01, 0.05, -0.01, 9, 12, "fragile"),             # jackknife interval includes 0
    (0.01, 0.05, 0.02, 6, 12, "fragile"),              # ahead in only half of the blocks
    (-0.01, 0.05, 0.02, 9, 12, "no effect shown"),
    (0.01, -0.02, 0.02, 9, 12, "no effect shown"),
])
def test_reading(boot_low, delta, jack_low, won, valid, expected):
    assert reading(boot_low, delta, jack_low, won, valid) == expected


# ------------------------------------------------------------------ sector-yield physical baseline
H, W = 8, 9


class SectorData:
    """Two inventory sectors: inv_road drives the target, inv_power is a huge point source with no NO2 signal."""
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        road = np.abs(rng.normal(size=(12, H, W))) * 10
        power = np.zeros((12, H, W))
        power[:, 1, 1] = 5000.0
        self.base_of = ["inv_road", "inv_power", "pop"]
        self.lag_of = [0, 0, 0]
        self.modality_of = ["traffic", "energy", "context"]
        self.X = np.stack([np.log1p(road), np.log1p(power), rng.normal(size=(12, H, W))], -1).astype(np.float32)
        self.z = (np.log1p(road) + 0.05 * rng.normal(size=(12, H, W))).astype(np.float32)


def sector_cfg(name):
    cfg = copy.deepcopy(Config().defaults)
    cfg["grid"].update(H=H, W=W)
    cfg["model"].update(type="baseline")
    cfg["model"]["baseline"].update(name=name, lags=[])
    cfg["model"]["baseline"]["params"]["forward_sectors"] = {"inventory_prefix": "inv_", "sigmas_km": [0, 2]}
    return cfg


def test_forward_sectors_ignores_a_sector_without_signal():
    d = SectorData()
    tr = np.ones((H, W), bool)
    sectors = BASELINES.get("forward_sectors")(sector_cfg("forward_sectors"), d).fit([d.X], [d.z], [tr])
    total = BASELINES.get("forward_operator")(sector_cfg("forward_operator"), d).fit([d.X], [d.z], [tr])
    road, power = sectors.lr.coef_
    assert road > 0 and power < 0.1 * road and (sectors.lr.coef_ >= 0).all()
    rho = lambda p: spearmanr(p.predict(d.X).ravel(), d.z.ravel()).correlation
    assert rho(sectors) > rho(total)


def test_forward_sectors_mask_and_cloud_gaps():
    d = SectorData(1)
    z = d.z.copy()
    z[np.random.default_rng(2).random(z.shape) < 0.3] = np.nan
    m = np.zeros((H, W), bool)
    m[:4] = True
    p = BASELINES.get("forward_sectors")(sector_cfg("forward_sectors"), d).fit([d.X], [z], [np.ones((H, W), bool)])
    pred = p.predict(d.X, m)
    assert pred.shape == (12, H, W) and np.isfinite(pred[:, m]).all() and np.isnan(pred[:, ~m]).all()


# ------------------------------------------------------------------ end to end on fake finished runs
GH, GW = 30, 40                                       # ~28 x 30 km; 6 km blocks -> about 25 blocks


def fake_dataset(rng):
    from tehran_carbon.data.dataset import CellMonthDataset
    dom = np.zeros((GH, GW), bool)
    dom[3:27, 4:36] = True
    years = [1402, 1403]
    z, hot = {}, {}
    for y in years:
        zz = np.full((12, GH, GW), np.nan, np.float32)
        hh = np.zeros((12, GH, GW), bool)
        for t in range(12):
            v = rng.normal(size=dom.sum())
            zz[t][dom] = v
            hh[t][dom] = v >= np.quantile(v, 0.9)
        z[y], hot[y] = zz, hh
    X = {y: np.zeros((12, GH, GW, 1), np.float32) for y in years}
    return CellMonthDataset(years, X, z, hot, dom, dom, ["f"], ["context"], ["f"], [0])


def finished_run(cfg, name, data, noise, rng):
    from tehran_carbon.tracking import Run
    c = copy.deepcopy(cfg)
    c["experiment"] = dict(name=name, rq="RQ2", kind="train", description="", version=1, variant={},
                           variant_label="", overrides={})
    with Run(c) as run:
        for y in data.years:
            zz = data.z[y].reshape(12, -1)[:, data.cells]
            pred = zz + noise * rng.normal(size=zz.shape)
            np.savez_compressed(run.path("preds", f"cv_{y}.npz"), pred=pred.astype(np.float32), cells=data.cells)
        for a, b in [(1402, 1403), (1403, 1402)]:
            zz = data.z[b].reshape(12, -1)[:, data.cells]
            np.savez_compressed(run.path("preds", f"transfer_{a}to{b}.npz"),
                                pred=(zz + noise * rng.normal(size=zz.shape)).astype(np.float32), cells=data.cells)
        run.finish("done", {})


def test_hypothesis_and_robustness_end_to_end(tmp_path, monkeypatch):
    from tehran_carbon.evaluation import hypotheses
    from tehran_carbon.evaluation.robustness import RobustnessChecks
    from tehran_carbon.tracking import Run
    rng = np.random.default_rng(0)
    data = fake_dataset(rng)
    monkeypatch.setattr(hypotheses.CellMonthDataset, "build", classmethod(lambda cls, cfg, **kw: data))
    cfg = copy.deepcopy(Config().defaults)
    cfg["paths"]["runs"] = str(tmp_path / "runs")
    cfg["grid"].update(H=GH, W=GW)
    cfg["split"]["block_km"] = 6
    cfg["eval"]["bootstrap"]["n"] = 200
    finished_run(cfg, "cand_v1", data, 0.3, rng)
    finished_run(cfg, "ref_a_v1", data, 1.5, rng)
    finished_run(cfg, "ref_b_v1", data, 3.0, rng)
    spec = {"metric": "pr_auc", "candidate": {"experiment": "cand_v1"},
            "references": [{"experiment": "ref_a_v1"}, {"experiment": "ref_b_v1"}]}
    acfg = dict(cfg, experiment=dict(name="analysis", rq=None, kind="analysis", description="", version=None,
                                     variant={}, variant_label="", overrides={}))

    with Run(acfg) as run:
        decision = hypotheses.HypothesisTest(acfg, "Hx", spec).run(run)
        robust = RobustnessChecks(acfg, "Hx", spec).run(run)
    assert decision["verdict"] == "supported"
    assert robust["candidate"] == decision["candidate"] and robust["reference"] == decision["reference"]
    assert "ref_a_v1" in decision["reference"]                 # the best reference, chosen after the results
    import pandas as pd
    s = pd.read_csv(run.dir / "Hx_robustness.csv")
    assert list(s.year) == [1402, 1403] and set(s.reading) == {"robust"}
    b = pd.read_csv(run.dir / "Hx_blocks.csv")
    n_blocks = int(hypotheses.SpatialSplit(cfg["split"], hypotheses.Grid(cfg), data.label_domain).blocks(6).max()) + 1
    assert len(b) == 2 * n_blocks and np.isclose(b[b.year == 1402].hotspot_share.sum(), 1.0)
    p = pd.read_csv(run.dir / "Hx_persistence.csv")
    assert list(p.observed_map.astype(str)) == ["1403", "1402"]   # the other year's observed map
    d = pd.read_csv(run.dir / "Hx_decision.csv")
    assert np.allclose(d.delta[:2], s.delta)                     # both checks compare the same two runs

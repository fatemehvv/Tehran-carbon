"""Paper v2 (branch paper_TC): inference with few spatial blocks, hotspot variants, power, and the paper commands."""
import copy

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from tehran_carbon.config import Config
from tehran_carbon.evaluation.hotspots import aggregate, group_blocks, hot_labels, superpixels
from tehran_carbon.evaluation.inference import (assignments, correlogram, dutilleul, effective_n, holm, jackknife_se,
                                                min_p, pair_counts, rank_normalize, signflip_mean, signflip_pooled,
                                                t_interval, verdict)

T, N, K = 4, 60, 6                                    # months, domain cells, blocks of 10 cells


def toy(seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(T, N))
    hot = z >= np.quantile(z, 0.8, axis=1, keepdims=True)
    blocks = np.repeat(np.arange(K), N // K)
    return z, hot, blocks, rng


def test_assignments_exact_and_identity_first():
    f, exact = assignments(4)
    assert exact and f.shape == (16, 4) and not f[0].any() and len({tuple(r) for r in f}) == 16
    g, exact = assignments(20, max_exact=16, n_random=100)
    assert not exact and g.shape == (100, 20) and not g[0].any()


def test_rank_normalize_keeps_order_and_nan():
    a = np.array([[3.0, 1.0, np.nan, 2.0]])
    r = rank_normalize(a)
    assert np.isnan(r[0, 2]) and np.allclose(r[0, [1, 3, 0]], [1 / 3, 2 / 3, 1.0])


def test_signflip_identical_models_give_p_one():
    z, hot, blocks, rng = toy()
    p = rng.normal(size=z.shape)
    for m in ["pr_auc", "spearman"]:
        s = signflip_pooled(z, hot, p, p, blocks, m)
        assert s["stat"] == 0 and s["p_two"] == 1.0 and s["n_perm"] == 2 ** K and s["exact"]


def test_signflip_perfect_candidate_reaches_the_smallest_p():
    z, hot, blocks, rng = toy(1)
    s = signflip_pooled(z, hot, z, rng.normal(size=z.shape), blocks, "spearman")
    assert s["stat"] > 0 and np.isclose(s["p_greater"], 1 / 2 ** K)


def test_signflip_mean_of_block_differences():
    s = signflip_mean(np.array([0.1, 0.2, 0.05, 0.3, 0.15, 0.12]))
    assert np.isclose(s["p_greater"], 1 / 64) and np.isclose(s["p_two"], 2 / 64) and s["blocks"] == 6
    s = signflip_mean(np.array([0.1, np.nan, -0.1]))
    assert s["blocks"] == 2 and s["p_two"] == 1.0
    assert min_p(2) == 0.25 and np.isnan(signflip_mean(np.array([np.nan]))["p_greater"])


def test_verdict_cases():
    assert verdict(0.05, 0.1, 0.06, 0.09, 0.02) == "superior beyond margin"
    assert verdict(0.005, 0.03, 0.008, 0.025, 0.02) == "superior"
    assert verdict(-0.01, 0.012, -0.008, 0.01, 0.02) == "equivalent"
    assert verdict(-0.015, 0.05, -0.01, 0.04, 0.02) == "non-inferior"
    assert verdict(-0.1, -0.01, -0.09, -0.02, 0.02) == "inferior"
    assert verdict(-0.1, 0.1, -0.08, 0.08, 0.02) == "inconclusive"
    assert verdict(np.nan, 1, 0, 1, 0.02) == "undefined"


def test_holm():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": np.nan})
    assert np.isclose(adj["a"], 0.03) and np.isclose(adj["c"], 0.06) and np.isclose(adj["b"], 0.06)
    assert np.isnan(adj["d"])


def test_jackknife_and_t_interval():
    z, hot, blocks, rng = toy(2)
    est, se, k, d = jackknife_se(z, hot, z + 0.3 * rng.normal(size=z.shape), rng.normal(size=z.shape), blocks,
                                 "spearman")
    assert k == K and se > 0 and est > 0 and len(d) == K
    lo, hi = t_interval(est, se, k - 1, 0.95)
    lo9, hi9 = t_interval(est, se, k - 1, 0.90)
    assert lo < lo9 < est < hi9 < hi


def lattice(side=30):
    rr, cc = np.divmod(np.arange(side * side), side)
    return np.c_[cc, rr].astype(float)                 # 1 km cells


def test_effective_n_iid_vs_smooth():
    rng = np.random.default_rng(3)
    side = 30
    xy, edges = lattice(side), np.arange(0, 21, 2.0)
    counts = pair_counts(xy, edges)
    assert 0 < counts.sum() <= side * side * (side * side - 1) / 2
    iid = rng.normal(size=(3, side * side))
    smooth = np.stack([gaussian_filter(rng.normal(size=(side, side)), 4).ravel() for _ in range(3)])
    n_iid = effective_n(*correlogram(iid, xy, edges, return_se=True), counts=counts, n=side * side)
    n_smooth = effective_n(*correlogram(smooth, xy, edges, return_se=True), counts=counts, n=side * side)
    assert n_iid > 0.5 * side * side and n_smooth < 0.2 * n_iid


def test_dutilleul_shrinks_the_sample_of_smooth_maps():
    rng = np.random.default_rng(4)
    side = 30
    xy, edges = lattice(side), np.arange(0, 21, 2.0)
    a, b = (gaussian_filter(rng.normal(size=(side, side)), 4).ravel() for _ in range(2))
    d = dutilleul(a, b, xy, edges)
    assert d["n"] == side * side and d["n_eff"] < 0.2 * d["n"] and 0 <= d["p_two"] <= 1
    assert dutilleul(a, a, xy, edges)["r"] > 0.999


def test_hot_labels_share_and_nan():
    z = np.arange(20, dtype=float)[None].repeat(2, 0)
    z[1, :5] = np.nan
    h = hot_labels(z, 0.9)
    assert h[0].sum() == 2 and not h[1, :5].any() and h[1].sum() >= 1


def test_superpixels_aggregate_and_blocks():
    cells = np.array([0, 1, 5, 10, 11, 55])            # W = 10: (0,0) (0,1) (0,5) (1,0) (1,1) (5,5)
    g = superpixels(cells, 10, 5)
    assert g[0] == g[1] == g[3] == g[4] and len(set(g)) == 3
    out = aggregate(np.array([[1, 2, 3, 4, np.nan, 6.0]]), g)
    assert np.isclose(out[0, g[0]], 7 / 3) and np.isclose(out[0, g[2]], 3) and np.isclose(out[0, g[5]], 6)
    assert list(group_blocks(g, np.array([0, 0, 1, 1, 1, 2]))) == [0, 1, 2]


def test_power_grows_with_the_effect_and_respects_the_smallest_p():
    from tehran_carbon.analysis.power import simulate_power
    resid = np.array([-0.02, 0.01, 0.015, -0.005, 0.0, 0.0])
    pw = simulate_power(resid, [0.0, 0.05], 0.05, 500, 0, "normal")
    assert pw[0] < 0.2 and pw[1] > 0.8
    assert simulate_power(np.array([0.01, -0.01]), [1.0], 0.025, 100, 0, "empirical") == [0.0]   # min p = 0.25


def test_paper_evaluate_and_power_end_to_end(tmp_path, monkeypatch):
    from test_robustness import GH, GW, fake_dataset, finished_run
    from tehran_carbon.analysis.paper_eval import evaluate_family
    from tehran_carbon.analysis.power import power_family
    from tehran_carbon.evaluation import hypotheses
    from tehran_carbon.tracking import Run
    rng = np.random.default_rng(0)
    data = fake_dataset(rng)
    monkeypatch.setattr(hypotheses.CellMonthDataset, "build", classmethod(lambda cls, cfg, **kw: data))
    cfg = copy.deepcopy(Config().defaults)
    cfg["paths"]["runs"] = str(tmp_path / "runs")
    cfg["grid"].update(H=GH, W=GW)
    cfg["split"]["block_km"] = 6
    cfg["eval"]["bootstrap"]["n"] = 100
    cfg["analysis"]["paper"]["signflip"].update(max_exact=8, n_random=200)
    cfg["analysis"]["paper"]["power"].update(n_sim=100)
    finished_run(cfg, "cand_v1", data, 0.3, rng)
    finished_run(cfg, "ref_v1", data, 1.5, rng)
    specs = {"Hx": {"metric": "pr_auc", "candidate": {"experiment": "cand_v1"},
                    "references": [{"experiment": "ref_v1"}]}}
    acfg = dict(cfg, experiment=dict(name="analysis", rq=None, kind="analysis", description="", version=None,
                                     variant={}, variant_label="", overrides={}))
    with Run(acfg) as run:
        evaluate_family(acfg, ["Hx"], specs, run)
        power_family(acfg, ["Hx"], specs, run)
    t = pd.read_csv(run.dir / "paper_Hx.csv")
    prim = t[t.part == "primary"]
    assert set(prim.metric) == {"pr_auc", "spearman"} and set(prim.year) == {1402, 1403}
    assert (prim.delta > 0).all() and prim.verdict.str.startswith("superior").all()
    assert set(t.part) == {"primary", "threshold", "superpixel"}             # no seeds stored by these runs
    h = pd.read_csv(run.dir / "paper_holm.csv")
    assert len(h) == 4 and (h.p_holm >= h.p_signflip_pooled - 1e-12).all()
    p = pd.read_csv(run.dir / "paper_power.csv")
    assert set(p.noise) == {"normal", "empirical"} and (p.blocks_valid > 0).all()


def test_paper_configs_resolve_and_keep_the_thesis_hashes():
    conf = Config()
    for name in [n for n in conf.experiments if n.startswith("paper2_")]:
        for c in conf.resolve(name):
            if "pollution" in c["model"]["modalities"]:
                assert c["split"]["fold_aware_stations"] is True
            if c["model"]["type"] == "neural":
                assert c["train"]["seeds"] == [0, 1, 2, 3, 4] and c["train"]["save_members"] is True
    for name, spec in conf.hypotheses.items():
        if name.startswith("PA_"):
            for sel in [spec["candidate"]] + spec["references"]:
                assert sel["experiment"] in conf.experiments
    # the paper settings live in `analysis`, which the config hash leaves out
    c = conf.resolve("rq2_sage_v2")[0]
    d = copy.deepcopy(c)
    d["analysis"]["paper"]["margin"] = 0.5
    assert Config.config_hash(c) == Config.config_hash(d)

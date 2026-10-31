"""Regression checks of the new pipeline against the earlier RQ1/RQ2 results (need data/raw; a few minutes)."""
import numpy as np
import pandas as pd

from conftest import RAW, needs_data
from tehran_carbon.config import Config


@needs_data
def test_dataset_matches_legacy_rq2():
    from tehran_carbon.data.dataset import CellMonthDataset
    (cfg,) = Config().resolve("rq2_mlp_v1")
    data = CellMonthDataset.build(cfg)
    old = np.load(RAW / "legacy" / "RQ2_dataset.npz")
    assert int(data.label_domain.sum()) == int(old["label_domain"].sum()) == 2436
    assert (data.label_domain == old["label_domain"]).all()
    assert (data.in_core == old["in_core"]).all()
    dom = data.label_domain
    for y in data.years:
        assert np.abs(data.z[y][:, dom] - old[f"z_{y}"][:, dom]).max() < 1e-4
        assert (data.hot[y] == old[f"hot_{y}"]).mean() > 0.999


@needs_data
def test_main_split_matches_legacy_rq2():
    from tehran_carbon.data.dataset import CellMonthDataset
    from tehran_carbon.data.grid import Grid
    from tehran_carbon.data.splits import SpatialSplit
    (cfg,) = Config().resolve("rq2_mlp_v1")
    data = CellMonthDataset.build(cfg)
    s = SpatialSplit(cfg["split"], Grid(cfg), data.label_domain)
    old = np.load(RAW / "legacy" / "RQ2_cv_schemes.npz")["fold_main_block22_LOBO"]
    assert (s.fold_map == old).all()


@needs_data
def test_l1_matches_frozen_rq1():
    from tehran_carbon.inventory.l1 import L1Inventory
    (cfg,) = Config().resolve("inv_l1_v1")
    for y in cfg["years"]:
        chk = L1Inventory(cfg, y).run()["check"].set_index("group")
        assert abs(chk.loc["Scope 1 total CORE", "diff_pct"]) < 0.5
        assert (chk.diff_pct.abs() < 2.0).all(), chk

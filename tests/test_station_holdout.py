"""Paper v2 (branch paper_TC): station hold-out per fold and the prediction of every seed."""
import copy

import numpy as np

from conftest import needs_data
from tehran_carbon.config import Config
from tehran_carbon.data.graphs import GridGraph
from tehran_carbon.data.grid import Grid
from tehran_carbon.data.sources.pollution import idw, stations_near
from tehran_carbon.training.trainer import NeuralPredictor

GRID20 = Grid({"grid": {**Config().defaults["grid"], "H": 20, "W": 20}})


def test_stations_near_the_test_block_are_left_out():
    test = np.zeros((20, 20), bool)
    test[5:10, 5:10] = True
    rows, cols = np.array([7, 7, 7, 0, 30]), np.array([7, 11, 15, 0, 0])
    inside = np.array([True, True, True, True, False])
    near = stations_near(rows, cols, inside, test, 2.0, GRID20.km_y, GRID20.km_x)
    # inside the block; 2 cells (~1.5 km) east of it; 6 cells (~4.5 km) east; far away; outside the grid
    assert list(near) == [True, True, False, False, False]
    assert not stations_near(rows, cols, inside, np.zeros_like(test), 2.0, GRID20.km_y, GRID20.km_x).any()


def test_idw_reproduces_a_station_and_ignores_a_removed_one():
    lon, lat = GRID20.lon[[2, 15], [2, 15]], GRID20.lat[[2, 15], [2, 15]]
    m = idw(np.array([1.0, 3.0]), lon, lat, GRID20, 2.0, 0.1)
    assert m.shape == (20, 20) and abs(m[2, 2] - 1.0) < 0.01 and abs(m[15, 15] - 3.0) < 0.01
    only = idw(np.array([1.0]), lon[:1], lat[:1], GRID20, 2.0, 0.1)
    assert np.allclose(only, 1.0)                     # one station left: the same value everywhere


def test_predict_members_average_is_the_prediction():
    from test_models import GRID, H, W, TinyData
    cfg = copy.deepcopy(Config().defaults)
    cfg["model"]["modalities"] = ["traffic", "energy"]
    cfg["model"]["backbone"]["type"] = "sage"
    cfg["train"].update(epochs=2, seeds=[0, 1, 2], device="cpu")
    d = TinyData()
    graph = GridGraph(GRID, cfg["model"]["backbone"]["scales"])
    p = NeuralPredictor(cfg, d, graph).fit([d.X], [d.z], [np.ones((H, W), bool)])
    mem = p.predict_members(d.X)
    assert mem.shape == (3, 12, H, W) and np.allclose(mem.mean(0), p.predict(d.X), atol=1e-6)
    assert not np.allclose(mem[0], mem[1])            # the seeds differ
    cfg["model"]["backbone"]["type"] = "none"         # own-cell model: only the masked cells are computed
    mask = np.zeros((H, W), bool)
    mask[:2] = True
    q = NeuralPredictor(cfg, d).fit([d.X], [d.z], [np.ones((H, W), bool)])
    m2 = q.predict_members(d.X, mask)
    assert np.isnan(m2[:, :, ~mask]).all() and np.isfinite(m2[:, :, mask]).all()


@needs_data
def test_station_holdout_matches_the_dataset_and_drops_the_test_stations():
    from tehran_carbon.data.dataset import CellMonthDataset
    from tehran_carbon.data.sources.pollution import StationHoldout
    from tehran_carbon.data.splits import SpatialSplit
    cfg = Config().resolve("paper2_rq1_ablation_v1")[0]          # station layer with its spatial lags
    data = CellMonthDataset.build(cfg)
    grid = Grid(cfg)
    h = StationHoldout(cfg, grid, data)
    assert h.features and any(data.lag_of[j] > 0 for j in h.features)
    y = data.years[0]
    full = h.patch(data.X[y], y, list(h.names))                  # every station kept: the dataset itself
    assert np.allclose(full[..., h.features], data.X[y][..., h.features], rtol=1e-5, atol=1e-5)
    other = [j for j in range(data.X[y].shape[-1]) if j not in h.features]
    assert np.array_equal(full[..., other], data.X[y][..., other])
    split = SpatialSplit(cfg["split"], grid, data.label_domain)
    changed = 0
    for k in range(split.n_folds):
        _, te = split.masks(k)
        keep = set(h.keep(te, split.buffer_km))
        for name, r, c, ok in zip(h.names, h.rows, h.cols, h.inside):
            if ok and te[r, c]:
                assert name not in keep                          # no station of the test block survives
        if len(keep) < len(h.names):
            Xk = h.patch(data.X[y], y, sorted(keep))
            changed += not np.allclose(Xk[..., h.features], data.X[y][..., h.features])
            assert np.array_equal(Xk[..., other], data.X[y][..., other])
    assert changed > 0                                           # at least one fold had a station to leave out

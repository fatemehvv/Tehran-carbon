import copy

import numpy as np
import pytest
import torch

from tehran_carbon.config import Config
from tehran_carbon.data.graphs import GridGraph, ODOperator
from tehran_carbon.data.grid import Grid
from tehran_carbon.models.baselines import BASELINES
from tehran_carbon.models.model import CarbonModel
from tehran_carbon.training.trainer import NeuralPredictor

H, W, T = 6, 7, 2
DIMS = {"traffic": 4, "energy": 3, "context": 2}
GRID = Grid({"grid": {**Config().defaults["grid"], "H": H, "W": W}})


def tiny_od() -> ODOperator:
    """3 districts: cells of rows 0-1, 2-3, 4-5."""
    F = np.zeros((H * W, 3))
    for c in range(H * W):
        F[c, (c // W) // 2] = 1.0
    return ODOperator(F, np.array([[5.0, 2.0, 1.0], [2.0, 5.0, 0.0], [1.0, 0.0, 5.0]]))


@pytest.mark.parametrize("fusion", ["early", "late", "gated"])
@pytest.mark.parametrize("backbone", ["none", "sage", "gcn", "gat"])
def test_model_shapes(fusion, backbone):
    m = copy.deepcopy(Config().defaults["model"])
    m["fusion"]["type"], m["backbone"]["type"] = fusion, backbone
    model = CarbonModel(DIMS, m)
    xs = {k: torch.randn(T, H * W, f) for k, f in DIMS.items()}
    graph = GridGraph(GRID, m["backbone"]["scales"]) if backbone != "none" else None
    out = model(xs, graph)
    assert out.shape == (T, H * W) and torch.isfinite(out).all()
    out.sum().backward()


def test_od_channel():
    od = tiny_od()
    h = torch.randn(T, H * W, 5)
    out = od(h)
    assert out.shape == h.shape
    # district 1 exchanges trips only with district 0 -> its cells receive district 0's mean embedding
    assert torch.allclose(out[:, 2 * W], h[:, :2 * W].mean(1), atol=1e-5)
    m = copy.deepcopy(Config().defaults["model"])
    m["backbone"]["type"], m["backbone"]["od"]["enabled"] = "sage", True
    graph = GridGraph(GRID, m["backbone"]["scales"])
    graph.od = od
    model = CarbonModel(DIMS, m)
    y = model({k: torch.randn(T, H * W, f) for k, f in DIMS.items()}, graph)
    assert y.shape == (T, H * W)
    y.sum().backward()
    assert all(p.grad is not None for p in model.backbone.od.parameters())


def test_edge_directions_are_unit_vectors():
    g = GridGraph(GRID, [1, 3])
    _, union, kind, direction = g.batched(T, "cpu")
    assert direction.shape == (union.shape[1], 2) and kind.shape == (union.shape[1],)
    assert torch.allclose(direction.norm(dim=1), torch.ones(len(direction)), atol=1e-5)


class TinyData:
    """Stand-in for CellMonthDataset with random features."""
    def __init__(self):
        rng = np.random.default_rng(0)
        self.modality_of = ["traffic"] * 3 + ["energy"] * 2 + ["context"] * 2
        self.base_of = ["inv_road", "road_km_primary", "vkt_peak", "inv_residential", "hdd", "pop", "t2m"]
        self.lag_of = [0] * 7
        self.shape = (H, W)
        self.hash = "tiny"
        self.X = rng.normal(size=(12, H, W, 7)).astype(np.float32)
        self.X[..., [0, 3]] = np.abs(self.X[..., [0, 3]])          # inventory features are log1p(t) >= 0
        self.z = (self.X[..., 0] + 0.1 * rng.normal(size=(12, H, W))).astype(np.float32)

    def columns(self, modalities, use_context=True):
        names = list(modalities) + (["context"] if use_context else [])
        return {m: np.array([i for i, mo in enumerate(self.modality_of) if mo == m]) for m in names}


@pytest.mark.parametrize("backbone", ["none", "sage"])
def test_predictor_fit_predict(backbone):
    cfg = copy.deepcopy(Config().defaults)
    cfg["model"]["modalities"] = ["traffic", "energy"]
    cfg["model"]["backbone"]["type"] = backbone
    cfg["train"].update(epochs=5, seeds=[0, 1], device="cpu")
    d = TinyData()
    tr = np.zeros((H, W), bool)
    tr[:4] = True
    va, te = np.zeros_like(tr), ~tr
    va[3] = True
    graph = GridGraph(GRID, cfg["model"]["backbone"]["scales"])
    p = NeuralPredictor(cfg, d, graph).fit([d.X], [d.z], [tr & ~va], [va])
    pred, w = p.predict(d.X, te, return_weights=True)
    assert pred.shape == (12, H, W) and np.isfinite(pred[:, te]).all()
    assert w.shape == (12, H, W, 3)                             # gated: traffic, energy, context


def test_checkpoint_roundtrip(tmp_path):
    cfg = copy.deepcopy(Config().defaults)
    cfg["model"]["modalities"] = ["traffic", "energy"]
    cfg["model"]["backbone"]["type"] = "sage"
    cfg["train"].update(epochs=3, seeds=[0], device="cpu")
    d = TinyData()
    graph = GridGraph(GRID, cfg["model"]["backbone"]["scales"])
    p = NeuralPredictor(cfg, d, graph).fit([d.X], [d.z], [np.ones((H, W), bool)])
    torch.save(dict(p.checkpoint(), data_hash=d.hash), tmp_path / "final.pt")
    q = NeuralPredictor.load(cfg, d, graph, tmp_path / "final.pt")
    assert np.allclose(p.predict(d.X), q.predict(d.X), atol=1e-6)


@pytest.mark.parametrize("name", ["ols", "gbm", "forward_operator"])
def test_baselines(name):
    cfg = copy.deepcopy(Config().defaults)
    cfg["model"].update(type="baseline", modalities=["traffic", "energy"])
    cfg["model"]["baseline"].update(name=name, lags=[1])
    d = TinyData()
    tr = np.ones((H, W), bool)
    p = BASELINES.get(name)(cfg, d).fit([d.X], [d.z], [tr])
    assert p.predict(d.X).shape == (12, H, W)

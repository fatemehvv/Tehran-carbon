"""NeuralPredictor: fit/predict wrapper around CarbonModel (same interface as the baselines).

    fit(Xs, ys, train_masks, val_masks=None)   lists over years: X (12, H, W, F), y (12, H, W), masks (H, W)
    predict(X, mask=None)                      -> (12, H, W) mean over seeds (NaN outside `mask` for own-cell models)

- Features are standardised with mean/sd over all cells and months of the training years (features are known
  everywhere; no target information is used).
- Loss: MSE of z on the training cells only; AdamW; one model per seed, predictions averaged over seeds.
- Early stopping on the validation cells (an inner spatial block), keeping the best epoch.
- Own-cell models (backbone none) only compute the rows they need; graph models see all nodes.
"""
from __future__ import annotations

import copy
import random

import numpy as np
import torch

from ..config import Config
from ..models.backbones import BACKBONES
from ..models.model import CarbonModel


class NeuralPredictor:
    kind = "neural"

    def __init__(self, cfg: dict, data, graph=None, log=print):
        self.cfg, self.log = cfg, log
        m, t = cfg["model"], cfg["train"]
        self.cols = data.columns(m["modalities"], m["use_context"])
        self.dims = {k: len(v) for k, v in self.cols.items()}
        self.H, self.W = data.shape
        dev = t["device"]
        self.device = torch.device("cuda" if dev == "auto" and torch.cuda.is_available() else
                                   ("cpu" if dev == "auto" else dev))
        if t.get("threads"):
            torch.set_num_threads(int(t["threads"]))
        self.needs_graph = BACKBONES.get(m["backbone"]["type"]).needs_graph
        if self.needs_graph and graph is None:
            raise ValueError(f"backbone '{m['backbone']['type']}' needs a graph")
        self.graph = graph if self.needs_graph else None
        self.mpb = int(t["months_per_batch"])
        self.models: list[CarbonModel] = []
        self.history: list[list] = []

    def _inputs(self, X: np.ndarray, rows: np.ndarray | None) -> dict[str, torch.Tensor]:
        """(T, H, W, F) -> branch -> standardised (T, n, F_m) tensor (n = rows, or all nodes)."""
        T, H, W, F = X.shape
        x = X.reshape(T, H * W, F)
        x = ((x[:, rows] if rows is not None else x) - self.mu) / self.sd
        return {k: torch.as_tensor(x[..., idx].astype(np.float32), device=self.device) for k, idx in self.cols.items()}

    def fit(self, Xs: list, ys: list, train_masks: list, val_masks: list | None = None) -> "NeuralPredictor":
        t = self.cfg["train"]
        F = Xs[0].shape[-1]
        flat = np.concatenate([X.reshape(-1, F) for X in Xs])
        self.mu, self.sd = flat.mean(0), flat.std(0) + 1e-6

        # batches: one per (year, block of months_per_batch months); masks are per node and shared by the months
        batches = []
        for i, X in enumerate(Xs):
            tr = train_masks[i].ravel()
            va = val_masks[i].ravel() if val_masks is not None and val_masks[i].any() else None
            rows = None if self.needs_graph else np.flatnonzero(tr | va if va is not None else tr)
            trt = torch.as_tensor(tr if rows is None else tr[rows], device=self.device)
            vat = None if va is None else torch.as_tensor(va if rows is None else va[rows], device=self.device)
            for a in range(0, X.shape[0], self.mpb):
                y = ys[i][a:a + self.mpb]
                y = y.reshape(y.shape[0], -1)
                y = np.nan_to_num(y if rows is None else y[:, rows]).astype(np.float32)
                batches.append((self._inputs(X[a:a + self.mpb], rows), torch.as_tensor(y, device=self.device), trt, vat))
        has_val = any(b[3] is not None for b in batches)

        init = self.cfg["model"].get("init_from")
        for s in t["seeds"]:
            random.seed(int(s))
            np.random.seed(int(s))
            torch.manual_seed(int(s))
            model = CarbonModel(self.dims, self.cfg["model"]).to(self.device)
            if init:
                ck = torch.load(Config.path(self.cfg, "runs") / init["run"] / "models" / "final.pt",
                                map_location=self.device, weights_only=False)
                keys = model.load_from(ck["state_dicts"][0], list(init["modules"]), bool(init["freeze"]))
                self.log(f"    init_from {init['run']}: {len(keys)} tensors{' (frozen)' if init['freeze'] else ''}")
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(t["lr"]),
                                    weight_decay=float(t["weight_decay"]))
            best, best_state, wait, curve = np.inf, None, 0, []
            for ep in range(int(t["epochs"])):
                model.train()
                tl = 0.0
                for xs, y, tr, _ in batches:
                    opt.zero_grad()
                    loss = ((model(xs, self.graph) - y)[:, tr] ** 2).mean()
                    loss.backward()
                    opt.step()
                    tl += float(loss)
                vl = None
                if has_val:
                    model.eval()
                    se, n = 0.0, 0
                    with torch.no_grad():
                        for xs, y, _, va in batches:
                            if va is not None:
                                d = (model(xs, self.graph) - y)[:, va]
                                se, n = se + float((d ** 2).sum()), n + d.numel()
                    vl = se / max(n, 1)
                    if vl < best - 1e-6:
                        best, best_state, wait = vl, copy.deepcopy(model.state_dict()), 0
                    else:
                        wait += 1
                curve.append((ep, tl / len(batches), vl))
                if has_val and wait >= int(t["early_stopping"]["patience"]):
                    break
            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()
            self.models.append(model)
            self.history.append(curve)
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray, mask: np.ndarray | None = None, return_weights: bool = False):
        """Mean prediction over seeds; with return_weights also the branch weights of the fusion
        ((12, H, W, M) for gated, (M,) for late, None for early)."""
        T, N = X.shape[0], self.H * self.W
        rows = None if (self.needs_graph or mask is None) else np.flatnonzero(mask.ravel())
        P = np.full((len(self.models), T, N), np.nan, np.float32)
        weights = []
        for i, model in enumerate(self.models):
            wm = []
            for a in range(0, T, self.mpb):
                p = model(self._inputs(X[a:a + self.mpb], rows), self.graph).cpu().numpy()
                if rows is None:
                    P[i, a:a + self.mpb] = p
                else:
                    P[i, a:a + self.mpb][:, rows] = p
                w = model.fusion.weights()
                if w is not None:
                    wm.append(w.cpu().numpy())
            if wm:
                weights.append(np.concatenate(wm, 0) if wm[0].ndim == 3 else wm[0])
        pred = P.mean(0).reshape(T, self.H, self.W)          # cells not computed stay NaN (same in every seed)
        if not return_weights:
            return pred
        if not weights:
            return pred, None
        w = np.mean(weights, 0)
        if w.ndim == 1:                                       # late fusion: one weight per branch
            return pred, w
        full = np.full((T, N, w.shape[-1]), np.nan, np.float32)
        full[:, rows if rows is not None else slice(None)] = w
        return pred, full.reshape(T, self.H, self.W, -1)

    def checkpoint(self) -> dict:
        return dict(mu=torch.as_tensor(self.mu), sd=torch.as_tensor(self.sd),
                    cols={k: v.tolist() for k, v in self.cols.items()}, dims=self.dims, model_cfg=self.cfg["model"],
                    state_dicts=[m.state_dict() for m in self.models], history=self.history)

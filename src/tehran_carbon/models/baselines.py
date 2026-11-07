"""Traditional models with the same fit/predict interface as NeuralPredictor (scikit-learn).

ols / gbm / rf      own-cell features of the selected branches, plus optional SLX neighbourhood means (box half-widths
                    `lags`, on the spatially varying own-cell features). Hyper-parameters: model.baseline.params.<name>
forward_operator    physics baseline: what a km-scale satellite column would see of the inventory:
                    score = log1p(Gaussian-smoothed total inventory CO2); sigma (from model.baseline.params) and a
                    linear map to z are fitted on the training cells only
forward_sectors     the same, with one non-negative NO2 yield per inventory sector (paper check): sectors emit very
                    different NOx per tonne of CO2, so the plain total lets large point sources (power plants) dominate
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from ..data.grid import Grid
from ..registry import Registry

BASELINES = Registry("baseline")


class TabularBaseline:
    kind = "baseline"
    estimator = None                         # scikit-learn class, set by the subclasses

    def __init__(self, cfg: dict, data, graph=None, log=print):
        m, b = cfg["model"], cfg["model"]["baseline"]
        self.idx = np.concatenate(list(data.columns(m["modalities"], m["use_context"]).values()))
        self.lags = [int(k) for k in b.get("lags") or []]
        no_lag = set(cfg["data"].get("no_lag") or [])
        self.lag_cols = [j for j in self.idx if data.lag_of[j] == 0 and data.base_of[j] not in no_lag]
        self.params = dict(b["params"].get(b["name"]) or {})

    def features(self, X: np.ndarray) -> np.ndarray:
        out = [X[..., self.idx]]
        for k in self.lags:
            out.append(np.stack([uniform_filter(X[..., j], size=(1, 2 * k + 1, 2 * k + 1), mode="nearest")
                                 for j in self.lag_cols], -1))
        return np.concatenate(out, -1)

    def fit(self, Xs, ys, train_masks, val_masks=None):
        Zs = [self.features(X) for X in Xs]
        F = Zs[0].shape[-1]
        A = np.concatenate([Z[:, m].reshape(-1, F) for Z, m in zip(Zs, train_masks)])
        y = np.concatenate([Y[:, m].ravel() for Y, m in zip(ys, train_masks)])
        allz = np.concatenate([Z.reshape(-1, Z.shape[-1]) for Z in Zs])
        self.mu, self.sd = allz.mean(0), allz.std(0) + 1e-9
        ok = np.isfinite(y)
        self.model = self.estimator(**self.params).fit((A[ok] - self.mu) / self.sd, y[ok])
        return self

    def predict(self, X, mask=None, return_weights=False):
        Z = (self.features(X) - self.mu) / self.sd
        T, H, W, F = Z.shape
        m = np.ones((H, W), bool) if mask is None else mask
        pred = np.full((T, H, W), np.nan, np.float32)
        pred[:, m] = self.model.predict(Z[:, m].reshape(-1, F)).reshape(T, -1)
        return (pred, None) if return_weights else pred


@BASELINES.register("ols")
class OLS(TabularBaseline):
    estimator = LinearRegression


@BASELINES.register("gbm")
class GBM(TabularBaseline):
    estimator = HistGradientBoostingRegressor


@BASELINES.register("rf")
class RF(TabularBaseline):
    estimator = RandomForestRegressor


@BASELINES.register("forward_operator")
class ForwardOperator:
    kind = "baseline"
    params_key = "forward_operator"          # block of model.baseline.params read by this class

    def __init__(self, cfg: dict, data, graph=None, log=print):
        b = cfg["model"]["baseline"]
        p = b["params"][self.params_key]
        prefix = p["inventory_prefix"]
        self.inv = [j for j, (n, k) in enumerate(zip(data.base_of, data.lag_of)) if n.startswith(prefix) and k == 0]
        if not self.inv:
            raise ValueError(f"{self.params_key} needs the {prefix}* features in the dataset")
        logged = set(cfg["data"].get("log1p") or [])
        self.logged = [data.base_of[j] in logged for j in self.inv]
        self.sigmas = [float(s) for s in p["sigmas_km"]]
        self.grid = Grid(cfg)

    def score(self, X: np.ndarray, sigma: float) -> np.ndarray:
        inv = X[..., self.inv].astype(np.float64)
        tot = sum(np.expm1(inv[..., i]) if lg else inv[..., i] for i, lg in enumerate(self.logged))
        if sigma > 0:
            tot = gaussian_filter(tot, sigma=(0, sigma / self.grid.km_y, sigma / self.grid.km_x), mode="constant")
        return np.log1p(tot)

    def fit(self, Xs, ys, train_masks, val_masks=None):
        y = np.concatenate([Y[:, m].ravel() for Y, m in zip(ys, train_masks)])
        ok = np.isfinite(y)                  # missing targets (cloudy cells of the daily data)
        y = y[ok]
        best, self.sigma, rows = -np.inf, None, None
        for s in self.sigmas:
            a = np.concatenate([self.score(X, s)[:, m].ravel() for X, m in zip(Xs, train_masks)])[ok]
            rho = spearmanr(a, y).correlation
            if rho > best:
                best, self.sigma, rows = rho, s, a
        if self.sigma is None:
            raise ValueError("forward_operator: no smoothing scale gives a valid correlation on the training cells")
        self.lr = LinearRegression().fit(rows[:, None], y)
        return self

    def predict(self, X, mask=None, return_weights=False):
        sc = self.score(X, self.sigma)
        pred = self.lr.predict(sc.reshape(-1, 1)).reshape(sc.shape).astype(np.float32)
        if mask is not None:
            pred = np.where(mask[None], pred, np.nan)
        return (pred, None) if return_weights else pred


@BASELINES.register("forward_sectors")
class ForwardSectors(ForwardOperator):
    """z ~ a + sum_s b_s * log1p(K_sigma * E_s), b_s >= 0: every inventory sector s gets its own NO2 yield.

    One Gaussian width sigma (the column footprint) for all sectors, chosen on the training cells by the Spearman
    correlation of the fitted score, as forward_operator. The yields are non-negative, so a sector can only add NO2;
    a sector with no NO2 signal (e.g. gas-fired power plants) gets a yield near 0 instead of dominating the total."""
    params_key = "forward_sectors"

    def layers(self, X: np.ndarray, sigma: float) -> np.ndarray:
        """(T, H, W, S): log1p of every smoothed sector layer."""
        out = []
        for i, j in enumerate(self.inv):
            e = X[..., j].astype(np.float64)
            e = np.expm1(e) if self.logged[i] else e
            if sigma > 0:
                e = gaussian_filter(e, sigma=(0, sigma / self.grid.km_y, sigma / self.grid.km_x), mode="constant")
            out.append(np.log1p(np.clip(e, 0, None)))
        return np.stack(out, -1)

    def fit(self, Xs, ys, train_masks, val_masks=None):
        y = np.concatenate([Y[:, m].ravel() for Y, m in zip(ys, train_masks)])
        ok = np.isfinite(y)
        y = y[ok]
        S = len(self.inv)
        best, self.sigma, self.lr = -np.inf, None, None
        for s in self.sigmas:
            A = np.concatenate([self.layers(X, s)[:, m].reshape(-1, S) for X, m in zip(Xs, train_masks)])[ok]
            lr = LinearRegression(positive=True).fit(A, y)
            rho = spearmanr(lr.predict(A), y).correlation
            if np.isfinite(rho) and rho > best:
                best, self.sigma, self.lr = rho, s, lr
        if self.sigma is None:
            raise ValueError("forward_sectors: no smoothing scale gives a valid correlation on the training cells")
        return self

    def predict(self, X, mask=None, return_weights=False):
        L = self.layers(X, self.sigma)
        pred = self.lr.predict(L.reshape(-1, L.shape[-1])).reshape(L.shape[:-1]).astype(np.float32)
        if mask is not None:
            pred = np.where(mask[None], pred, np.nan)
        return (pred, None) if return_weights else pred

"""The RQ3 simulator as a gymnasium environment: one episode = one study year, one step = one Jalali month.

Action (2 D values in [-1, 1]): restriction r_i in [0, r_max] (share of restricted car trips INTO district i that are
removed) and bus boost b_i in [0, b_max] (extra bus service relative to the district's base). Extra bus service is
limited by an annual budget (share of the annual base service); a boost beyond what is left is scaled down.
Behaviour parameters (shift share, its gain with extra service, ridership elasticity, share of new riders who drove)
are drawn per episode from triangular ranges; the agent only sees the last month's outcome.
Reward of a month (every term relative to the road sector of that month; docs/preregistration_rq3.md):
    dC / R  -  lambda dT / T0  -  mu dE / E  -  kappa dPEX / PEX0
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from .city import CityTransport


class Exposure:
    """Population exposure (PEX) of a month when the road features of each district are scaled, from the frozen RQ2
    model (FrozenResponseModel). The hotspot threshold of each month is that of the baseline prediction."""

    def __init__(self, frozen, year: int, e: dict, quantile: float):
        self.fr, self.e = frozen, e
        self.X = frozen.data.X[int(year)]
        self.dom = frozen.data.label_domain
        self.pop = frozen.physical(self.X[0], e["population_feature"])
        self.base = frozen.model.predict(self.X)
        self.thr = np.array([np.quantile(self.base[t][self.dom], quantile) for t in range(12)])
        self.pex0 = np.array([self.pex(t, self.base[t]) for t in range(12)])
        # mean baseline z of the domain cells of each district (observation; hotspot ranking of static policies)
        Fd = frozen.F * self.dom.ravel()[:, None]
        self.zbar = np.stack([np.nan_to_num(self.base[t].ravel()) @ Fd for t in range(12)]) / np.maximum(Fd.sum(0), 1e-12)

    def pex(self, t: int, z: np.ndarray) -> float:
        return float(self.pop[self.dom & (z >= self.thr[t])].sum() / self.pop[self.dom].sum())

    def __call__(self, t: int, road: np.ndarray, activity: np.ndarray) -> float:
        """road / activity (D,): district factors of the road CO2 feature and of the traffic activity features."""
        mult = {self.e["road_feature"]: self.fr.cell_multiplier(road)}
        mult.update({n: self.fr.cell_multiplier(activity) for n in self.e["activity_features"]})
        return self.pex(t, self.fr.model.predict(self.fr.scenario(self.X[t:t + 1], mult))[0])


class CarbonPolicyEnv(gym.Env):
    def __init__(self, city: CityTransport, rl: dict, exposure: Exposure | None = None):
        self.city, self.r, self.exposure = city, rl, exposure
        D = city.D
        self.r_max, self.b_max = float(rl["levers"]["restriction_max"]), float(rl["levers"]["bus_boost_max"])
        w = rl["reward"]
        self.lam, self.mu, self.kappa = float(w["lambda_mobility"]), float(w["mu_energy"]), float(w["kappa_exposure"])
        self.zbar = exposure.zbar if exposure is not None else np.zeros((12, D))
        self.hot_rank = np.argsort(-self.zbar.mean(0))                 # districts by baseline predicted z, highest first
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2 * D,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (8 + 2 * D,), np.float32)

    # ------------------------------------------------------------------ gymnasium interface
    def reset(self, seed: int | None = None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.theta = {k: float(rng.triangular(v[1], v[0], v[2])) for k, v in self.r["behaviour"].items()}
        self.noise = np.exp(rng.normal(0.0, float(self.r["demand_noise_sd"]), 12))
        self.t, self.budget = 0, float(self.r["bus_budget"])
        self.last = np.zeros(4)
        return self._obs(), {}

    def step(self, action):
        r, b = self.levers(action)
        out = self.outcome(self.t, r, b)
        self.budget -= out["bus_used"]
        self.last = np.array([out["shift_share"], out["voluntary_share"], out["dC_rel"], out["dT_rel"]])
        self.t += 1
        done = self.t == 12
        return (np.zeros(self.observation_space.shape, np.float32) if done else self._obs()), out["reward"], done, False, out

    @property
    def month(self) -> int:
        """Jalali month (1..12) of the next step."""
        return self.t + 1

    # ------------------------------------------------------------------ levers
    def levers(self, action) -> tuple[np.ndarray, np.ndarray]:
        """Physical levers of a normalised action; the bus boost is scaled down to the remaining budget."""
        a = np.clip(np.asarray(action, float), -1, 1)
        D = self.city.D
        r = (a[:D] + 1) / 2 * self.r_max
        b = (a[D:] + 1) / 2 * self.b_max
        need = self._bus_used(self.t, b)
        if need > self.budget:
            b = b * max(self.budget, 0.0) / need
        return r, b

    def action_of(self, r: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Normalised action of physical levers (static policies)."""
        return np.concatenate([2 * np.asarray(r) / self.r_max - 1, 2 * np.asarray(b) / self.b_max - 1]).astype(np.float32)

    def _bus_used(self, t: int, b: np.ndarray) -> float:
        """Extra bus service of a month, as a share of the annual base service."""
        c = self.city
        return float((b * c.bus_share).sum() * c.days[t] / c.days.sum())

    # ------------------------------------------------------------------ one month
    def outcome(self, t: int, r: np.ndarray, b: np.ndarray) -> dict:
        c, th, occ = self.city, self.theta, float(self.r["occupancy"])
        C0 = c.C * self.noise[t]                                           # car trips (o, d) of a typical day
        into = np.maximum(C0.sum(0), 1e-12)
        phi = np.minimum(1.0, th["shift"] + th["shift_boost"] * b / self.b_max)
        vol = np.minimum(1.0, th["elasticity"] * b * c.bus_riders * th["from_car"] / occ / into)
        removed = C0 * r[None]
        stay = (C0 - removed) * (1 - vol[None])
        shifted = removed * phi[None] + (C0 - removed) * vol[None]
        suppressed = removed * (1 - phi[None])

        vkt0, vkt1 = c.vkt_by_district(C0), c.vkt_by_district(stay)
        car_saved = c.ef * c.m[t] * (vkt0 - vkt1)                         # (D,) t CO2 of the month
        bus_used = self._bus_used(t, b)
        bus_d = c.bus_co2 * b * c.bus_share * c.days[t] / c.days.sum()   # (D,) extra bus CO2
        dC = float(car_saved.sum() - bus_d.sum())

        tau = c.time_ratio(vkt1 / np.maximum(vkt0, 1e-12))
        tau_od = 0.5 * (tau[:, None] + tau[None])
        T0 = c.m[t] * occ * (C0 * c.L).sum() / c.speed
        T1 = c.m[t] * occ * (((stay * tau_od + suppressed) * c.L).sum() / c.speed
                             + (shifted * c.L).sum() / float(self.r["transit_speed_kmh"]))
        dE = c.bus_mj * bus_used

        R = c.road_co2[t].sum()
        if self.exposure is not None and self.kappa != 0:
            road = 1 - (car_saved - bus_d) / np.maximum(c.road_co2[t], 1e-12)
            pex = self.exposure(t, np.maximum(road, 0.0), vkt1 / np.maximum(vkt0, 1e-12))
            dpex = (pex - self.exposure.pex0[t]) / self.exposure.pex0[t]
        else:
            pex, dpex = np.nan, 0.0
        reward = dC / R - self.lam * (T1 - T0) / T0 - self.mu * dE / c.road_mj[t] - self.kappa * dpex
        moved = removed.sum()
        return dict(reward=float(reward), co2_saved_t=dC, dC_rel=dC / R, hours_lost=float(T1 - T0),
                    dT_rel=float((T1 - T0) / T0), energy_MJ=float(dE), pex=pex, dPEX_rel=float(dpex),
                    bus_used=bus_used, shift_share=float((removed * phi[None]).sum() / moved) if moved > 0 else 0.0,
                    voluntary_share=float((vol * (C0 - removed).sum(0)).sum() / into.sum()),
                    restriction=r, bus_boost=b)

    def _obs(self) -> np.ndarray:
        c, t = self.city, self.t
        ang = 2 * np.pi * t / 12
        head = [np.sin(ang), np.cos(ang), c.hdd[t] / max(c.hdd.max(), 1e-12),
                self.budget / max(float(self.r["bus_budget"]), 1e-12)]
        share = c.road_co2[t] / max(c.road_co2[t].sum(), 1e-12)
        return np.concatenate([head, self.last, self.zbar[t], share]).astype(np.float32)

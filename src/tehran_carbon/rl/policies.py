"""Policies of the RQ3 comparison. Every policy maps (observation, env) to a normalised action.

none      no action
uniform   the same restriction r and bus boost b in every district and month
winter    (r, b) everywhere, only in the configured months (the pollution season)
hotspot   (r, b) in the k districts with the highest baseline predicted z, every month (rank: trip_km = the k districts
          with the most car-km of the trips into them, the CO2 a restriction can remove)
calm      (daily design) (r, b) everywhere on the days whose standardised weather feature is below a threshold (a simple
          weather rule: restrict on calm days; reported as a secondary comparison, not one of the static references)
rl        a trained stable-baselines3 agent (deterministic action)
The static grids (rl.static in the config) are tuned on the training year; see experiments/policy_search.py.
"""
from __future__ import annotations

import itertools

import numpy as np


class StaticPolicy:
    def __init__(self, family: str, restriction: float = 0.0, bus_boost: float = 0.0, months=None, districts=None,
                 feature: int | None = None, below: float | None = None, rank: str = "z"):
        self.family, self.restriction, self.bus_boost = family, float(restriction), float(bus_boost)
        self.months = set(months) if months else None
        self.districts = int(districts) if districts else None
        self.feature, self.below, self.rank = feature, below, rank
        parts = ([f"r={self.restriction:g}", f"b={self.bus_boost:g}"] + ([f"k={self.districts}"] if self.districts else [])
                 + ([f"below={self.below:g}"] if self.below is not None else []) + ([f"rank={rank}"] if rank != "z" else []))
        self.name = family if family == "none" else f"{family} " + " ".join(parts)

    def act(self, obs, env) -> np.ndarray:
        D = env.city.D
        on = np.zeros(D, bool)
        calm = self.below is None or env.weather[env.t, self.feature] < self.below
        if self.family != "none" and calm and (self.months is None or env.month in self.months):
            order = env.trip_km_rank if self.rank == "trip_km" else env.hot_rank
            on[order[:self.districts] if self.districts else slice(None)] = True
        return env.action_of(np.where(on, self.restriction, 0.0), np.where(on, self.bus_boost, 0.0))

    @staticmethod
    def grid(static: dict, weather_features=()) -> list["StaticPolicy"]:
        """`none` plus every combination of each family's grid (rl.static); `feature` names one of weather_features."""
        out = [StaticPolicy("none")]
        for family, g in static.items():
            ks, below = g.get("districts") or [None], g.get("below") or [None]
            f = list(weather_features).index(g["feature"]) if "feature" in g else None
            for r, b, k, w in itertools.product(g["restriction"], g["bus_boost"], ks, below):
                out.append(StaticPolicy(family, r, b, g.get("months"), k, f, w, g.get("rank", "z")))
        return out


class AgentPolicy:
    def __init__(self, name: str, model):
        self.name, self.family, self.model = name, "rl", model

    def act(self, obs, env) -> np.ndarray:
        return self.model.predict(obs, deterministic=True)[0]

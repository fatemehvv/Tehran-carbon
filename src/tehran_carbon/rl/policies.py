"""Policies of the RQ3 comparison. Every policy maps (observation, env) to a normalised action.

none      no action
uniform   the same restriction r and bus boost b in every district and month
winter    (r, b) everywhere, only in the configured months (the pollution season)
hotspot   (r, b) in the k districts with the highest baseline predicted z, every month
rl        a trained stable-baselines3 agent (deterministic action)
The static grids (rl.static in the config) are tuned on the training year; see experiments/policy_search.py.
"""
from __future__ import annotations

import itertools

import numpy as np


class StaticPolicy:
    def __init__(self, family: str, restriction: float = 0.0, bus_boost: float = 0.0, months=None, districts=None):
        self.family, self.restriction, self.bus_boost = family, float(restriction), float(bus_boost)
        self.months = set(months) if months else None
        self.districts = int(districts) if districts else None
        parts = [f"r={self.restriction:g}", f"b={self.bus_boost:g}"] + ([f"k={self.districts}"] if self.districts else [])
        self.name = family if family == "none" else f"{family} " + " ".join(parts)

    def act(self, obs, env) -> np.ndarray:
        D = env.city.D
        on = np.zeros(D, bool)
        if self.family != "none" and (self.months is None or env.month in self.months):
            on[env.hot_rank[:self.districts] if self.districts else slice(None)] = True
        return env.action_of(np.where(on, self.restriction, 0.0), np.where(on, self.bus_boost, 0.0))

    @staticmethod
    def grid(static: dict) -> list["StaticPolicy"]:
        """`none` plus every combination of each family's grid (rl.static)."""
        out = [StaticPolicy("none")]
        for family, g in static.items():
            ks = g.get("districts") or [None]
            for r, b, k in itertools.product(g["restriction"], g["bus_boost"], ks):
                out.append(StaticPolicy(family, r, b, g.get("months"), k))
        return out


class AgentPolicy:
    def __init__(self, name: str, model):
        self.name, self.family, self.model = name, "rl", model

    def act(self, obs, env) -> np.ndarray:
        return self.model.predict(obs, deterministic=True)[0]

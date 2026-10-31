"""RQ3 experiment: RL agent vs static policies in the district simulator (docs/preregistration_rq3.md).

For every study year a (training year):
  1. tune each static family on a (mean return over rl.evaluation.tune_episodes, seeds apart from the evaluation)
  2. train the agent on a, one model per seed (models/ppo_<a>_s<seed>.zip)
Then every policy (all static grid members, every agent) is evaluated on every year b with common random numbers:
episode k of year b draws the same behaviour parameters and noise for every policy, so comparisons are paired.
Outputs: preds/returns_<a>to<b>.npz (names, returns (P, K), components (P, K, 4), mean agent actions by month),
tuned.json (tuned static policy per family and training year), metrics.json (mean returns).
"""
from __future__ import annotations

import time

import numpy as np

from ..rl.city import CityTransport
from ..rl.env import CarbonPolicyEnv, Exposure
from ..rl.policies import AgentPolicy, StaticPolicy
from .response import FrozenResponseModel

COMPONENTS = ["co2_saved_t", "hours_lost", "energy_MJ", "dPEX_rel"]


class PolicySearch:
    def __init__(self, cfg: dict, run):
        self.cfg, self.run, self.r = cfg, run, cfg["rl"]

    def env(self, year: int) -> CarbonPolicyEnv:
        if year not in self.envs:
            ex = Exposure(self.frozen, year, self.r["exposure"], float(self.frozen.cfg["data"]["target"]["hotspot_quantile"]))
            city = CityTransport.from_config(self.cfg, year, self.frozen.F, self.frozen.ids)
            self.envs[year] = CarbonPolicyEnv(city, self.r, ex)
            self.run.log(f"  {year}: car EF {city.ef * 1e6:.1f} g/vkm, car trips {city.C.sum():,.0f}/day, "
                         f"road CO2 {city.road_co2.sum() / 1e6:.2f} Mt (22 districts), baseline PEX {ex.pex0.mean():.3f}")
        return self.envs[year]

    @staticmethod
    def rollout(env: CarbonPolicyEnv, policy, seed: int) -> tuple[float, np.ndarray, np.ndarray]:
        """Return, summed components (4,) and the levers of every month (12, 2 D) of one episode."""
        obs, _ = env.reset(seed=seed)
        total, comp, acts = 0.0, np.zeros(len(COMPONENTS)), []
        for _ in range(12):
            obs, reward, done, _, out = env.step(policy.act(obs, env))
            total += reward
            comp += [out[k] for k in COMPONENTS]
            acts.append(np.concatenate([out["restriction"], out["bus_boost"]]))
        return total, comp, np.array(acts)

    def evaluate(self, env, policies, seeds) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        R = np.zeros((len(policies), len(seeds)))
        C = np.zeros((len(policies), len(seeds), len(COMPONENTS)))
        A = np.zeros((len(policies), 12, 2 * env.city.D))
        for i, p in enumerate(policies):
            for k, s in enumerate(seeds):
                R[i, k], C[i, k], a = self.rollout(env, p, int(s))
                A[i] += a / len(seeds)
        return R, C, A

    def train_agent(self, year: int, seed: int, timesteps: int):
        from stable_baselines3 import PPO
        a = self.r["agent"]
        env = CarbonPolicyEnv(self.env(year).city, self.r, self.env(year).exposure)
        model = PPO("MlpPolicy", env, seed=seed, device=a["device"], verbose=0, **a["params"])
        model.learn(total_timesteps=timesteps)
        model.save(self.run.path("models", f"ppo_{year}_s{seed}.zip"))
        return model

    def run_all(self, smoke: bool = False) -> dict:
        r, years = self.r, [int(y) for y in self.cfg["years"]]
        ev, ag = dict(r["evaluation"]), dict(r["agent"])
        if smoke:
            ev.update(r["smoke"]["evaluation"])
            ag.update(r["smoke"]["agent"])
        self.frozen = FrozenResponseModel(self.cfg, r["response_experiment"], self.run.log)
        self.run.log(f"frozen response model: {self.frozen.run_id}")
        self.envs = {}
        statics = StaticPolicy.grid(r["static"])
        eval_seeds = int(ev["seed"]) + np.arange(int(ev["episodes"]))
        tune_seeds = int(ev["tune_seed"]) + np.arange(int(ev["tune_episodes"]))

        static_eval = {}
        for y in years:
            env, t0 = self.env(y), time.time()
            static_eval[y] = self.evaluate(env, statics, eval_seeds)
            self.run.log(f"static policies on {y}: {len(statics)} x {len(eval_seeds)} episodes, "
                         f"{(time.time() - t0) / (12 * len(statics) * len(eval_seeds)) * 1e3:.1f} ms per step")

        tuned, agents, metrics = {}, {}, {}
        for a in years:
            R, _, _ = self.evaluate(self.env(a), statics, tune_seeds)
            best = {}
            for i, p in enumerate(statics):
                if p.family not in best or R[i].mean() > best[p.family][1]:
                    best[p.family] = (p.name, float(R[i].mean()))
            tuned[a] = {f: n for f, (n, _) in best.items()}
            self.run.log(f"tuned on {a}: " + "; ".join(f"{n} ({v:.4f})" for n, v in best.values()))
            agents[a] = []
            for s in ag["seeds"]:
                self.run.log(f"training agent on {a}, seed {s} ({int(ag['timesteps'])} steps)")
                agents[a].append(AgentPolicy(f"rl_s{s}", self.train_agent(a, int(s), int(ag["timesteps"]))))

        for a in years:
            for b in years:
                Ra, Ca, Aa = self.evaluate(self.env(b), agents[a], eval_seeds)
                Rs, Cs, As = static_eval[b]
                names = [p.name for p in statics] + [p.name for p in agents[a]]
                np.savez_compressed(self.run.path("preds", f"returns_{a}to{b}.npz"), names=np.array(names),
                                    families=np.array([p.family for p in statics] + ["rl"] * len(agents[a])),
                                    returns=np.concatenate([Rs, Ra]), components=np.concatenate([Cs, Ca]),
                                    actions=np.concatenate([As, Aa]), seeds=eval_seeds, districts=np.array(self.frozen.ids))
                means = dict(zip(names, np.concatenate([Rs, Ra]).mean(1)))
                metrics[f"{a}to{b}"] = {"rl": float(Ra.mean()), **{f: means[n] for f, n in tuned[a].items()}}
                self.run.log(f"{a} -> {b}: " + ", ".join(f"{k} {v:.4f}" for k, v in metrics[f"{a}to{b}"].items()))
        self.run.save_json("tuned.json", {str(a): t for a, t in tuned.items()})
        self.run.save_json("metrics.json", {"mean_return": metrics})
        return {k: {p: round(v, 4) for p, v in m.items()} for k, m in metrics.items()}

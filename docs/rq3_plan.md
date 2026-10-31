# RQ3 (reinforcement learning): how it plugs into this repository

Status: **design only**; nothing is implemented yet. The design follows the proposal (sections 4.13 and 17).

## Proposal

- **RQ3.** How can reinforcement learning design adaptive interventions that reduce the carbon footprint?
- **H3.** RL gives more effective adaptive interventions than static policies.
- **H4.** Simulating intervention scenarios before implementation makes it possible to evaluate mitigation
  strategies and lower policy risk.
- **Decision space.** Operational traffic-management variables.
- **Reward.** A balance between carbon reduction and transport efficiency, including the energy use involved.

## Building blocks that already exist

| need | where |
|---|---|
| emissions by sector, district and month | `inventory/` (L1 now; L2 allocation to be ported from the frozen RQ1 v2 outputs) |
| where an intervention changes the hotspot | `models/final.pt` of the chosen RQ2 run (fusion + GNN), loaded frozen |
| district graph (22 nodes, shared boundaries) | `data/graphs.py: DistrictGraph` |
| cell → district shares | `data/raw/rq1_frozen/grid_district_fractions.csv` |
| run tracking, config versions | `tracking.py`, `configs/config.yaml` |

## Planned additions (when RQ3 starts)

- **Dependencies.** Add `gymnasium` and `stable-baselines3` to `requirements.txt`.
- **`src/tehran_carbon/rl/env.py`: `CityCarbonEnv(gymnasium.Env)`.**
  - State: sector emissions per district, predicted hotspot indices (HCI, PEX) from the frozen RQ2 model, month,
    weather.
  - Action: per-district traffic levers (e.g. demand restriction level, speed management, bus/metro shift).
  - Reward: −CO₂ − λ · mobility loss − μ · energy. The weights come from the config.
- **`src/tehran_carbon/rl/policies.py`.** Static baselines for H3 (no action, uniform cut, fixed schedule).
- **Agents.** PPO / DQN from stable-baselines3, behind a registry like the models.
- **Config.** An `rl:` block in `defaults` (levers, bounds, reward weights, episodes) and `rq3_*_v1` experiments.
- **H3 rule.** A decision rule in `hypotheses` (RL vs best static policy on held-out years / scenarios).
- **Later.** An optional SUMO (microscopic traffic simulator) backend behind the same environment interface.

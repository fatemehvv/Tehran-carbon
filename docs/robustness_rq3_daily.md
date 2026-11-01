# RQ3 daily design: robustness checks after H3d (NOT pre-registered)

Written 2026-11-01, **after** the H3d result (supported) and **before** these runs. The H3d verdict stays as it is;
these checks are reported as robustness checks only.

## What the H3d agent did

From `preds/returns_*.npz` of `20261101-160831_rq3_daily_v1_77ef33`:
- **When.** It spent its whole 30-day budget in months 1-4 (Farvardin-Tir) and none in the pollution season. None of
  its restriction fell on the 10 % most polluted days. Its daily restriction correlates −0.38 with the day's severity.
- **Where.** About r_max in the 5 districts with the most car traffic (2, 3, 6, 12, 4). The static hotspot reference
  ranks districts by predicted z (3, 6, 7, 8); only 37 % of the agent's restriction fell there.

Two explanations follow, and each check tests one.

## Check 1: a fairer static reference (`rq3_daily_robust_v1`, `H3d_robust`)

The CO₂ term dominates the reward, and the hotspot reference does not rank districts by CO₂. New static family
`hotkm`: r every day in the k districts with the most car-km of the trips into them (`rank: trip_km`),
k ∈ {1, 2, 3, 5, 8}, r ∈ {0.1, 0.2, 0.3}. It is tuned on the training year like the other families. Everything else is as in
`rq3_daily_v1`, and the agent (γ = 0.99, same seeds) is retrained. The H3d rule is applied with `hotkm` among the
references.

## Check 2: discounting (`rq3_daily_g0999_v1`, `rq3_daily_g1_v1`, `H3d_g0999`, `H3d_g1`)

With γ = 0.99 over 365 steps, a reward 100 days ahead counts 37 % (γ = 0.999: 90 %). Spending the budget early is then
optimal for the agent, whatever the season. The evaluation sums undiscounted returns. As check 1, with γ = 0.999 and
γ = 1.0. Reported: the decision rule as in check 1, and when in the year the agent spends its budget.

## Change before the full runs

The smoke run (1 tuning episode) chose `hotkm` k = 3, the edge of the first grid k ∈ {3, 5, 8}. The grid was widened to
k ∈ {1, 2, 3, 5, 8} so that the reference is not cut off by its grid. A stronger reference only makes the check harder
for the agent.

## Results (2026-11-01)

Runs: `rq3_daily_robust_v1`, `rq3_daily_g0999_v1`, `rq3_daily_g1_v1` (3 agent seeds each, 30 evaluation episodes,
common random numbers). Tables: `docs/tables/t07_daily_timing.csv`, `t10_scorecard.csv`; figures `fig08`-`fig11`.

| policy | return 1402→1403 | return 1403→1402 | budget used by end of month 3 (days) | share on the 10 % most severe days |
|---|---|---|---|---|
| hotkm r=0.2, k=3 (districts 2, 3, 6) | **28.9** | **28.4** | 8.5 | 11 % |
| RL γ = 1 | 22.6 | 22.2 | 16.5 | 0 % |
| RL γ = 0.999 | 21.7 | 22.2 | 17.8 | 0 % |
| RL γ = 0.99 | 19.0 | 19.2 | 25.6 | 0 % |
| hotspot by z (H3d reference) | 17.7 | 17.1 | 11.3 | 1 % |

| check | Δ RL − best static (1402→1403 / 1403→1402) | verdict |
|---|---|---|
| H3d_robust (γ 0.99) | −9.9 / −9.2 | not supported |
| H3d_g0999 | −7.2 / −6.2 | not supported |
| H3d_g1 | −6.3 / −6.2 | not supported |

- **Check 1.** A static rule ranked by car-km beats every agent by a wide margin. It saves the most CO₂ (590 kt/yr),
  saves travel time (−9.1 M person-hours) and lowers exposure the most. The H3d advantage came from targeting districts
  better than the z-ranked reference, not from adapting to the day.
- **Check 2.** A larger γ spreads the budget over more of the year and raises the return (19 → 22.5), but no agent
  spends any budget on the most polluted days. Discounting was not the cause: in this simulator the CO₂ and travel-time
  terms dominate the reward and do not depend on the day.
- **Reproducibility.** The retrained γ = 0.99 agent matches the H3d run (19.02 vs 19.00).
- The H3d verdict stays "supported" under its pre-registered rule; these checks are reported next to it.

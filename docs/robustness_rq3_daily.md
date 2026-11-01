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

# RQ3 / H3, H4: pre-registration of the policy experiments

Written 2026-11-01, **before any agent is trained or any policy is evaluated**. Settings: `defaults.rl`, the
`rq3_*` experiments and `hypotheses.H3` / `hypotheses.H4` in `configs/config.yaml`. After results exist, nothing here
is changed; a deviation becomes a new config version and is listed in the results chapter.

Decisions taken with the thesis author on 2026-11-01:
- **Levers.** Two levers per district: a private-car restriction and extra bus service.
- **Simulator.** A macroscopic simulator in Python, built from the RQ1 data. SUMO is left as future work behind the
  same interface.
- **Reward weights.** One pre-registered main set decides H3/H4. A sweep over λ is a secondary analysis (trade-off
  curve).

## Questions and hypotheses (proposal)

- **RQ3.** How can reinforcement learning design adaptive interventions that reduce the carbon footprint?
- **H3.** RL gives more effective adaptive interventions than static policies.
- **H4.** Simulating intervention scenarios before implementation makes it possible to evaluate mitigation
  strategies and lowers policy risk.

## Simulator (`src/tehran_carbon/rl/`)

**Unit.** 22 districts × 12 Jalali months; one episode = one year (12 steps). Built separately for 1402 and 1403.

**Inputs.** All inputs come from the RQ1 frozen outputs and the transport tables already in `data/raw`.

| quantity | source |
|---|---|
| car trips C_od, distance L_od (typical day) | RQ1 gravity OD (`OD_gravity_typical_day.csv`): private car + ride-hailing (taxi app) vehicle trips |
| car CO₂ (annual) | L1 lines `gasoline` + `CNG` (CORE) |
| bus CO₂, diesel litres (annual) | L1 line `diesel_municipal_bus` |
| monthly profile s_t | L1 `gasoline_share` (car activity of month t = annual × s_t) |
| road CO₂ of district i, month t | L2 grid `road_transport` × s_t, cells → districts by area share |
| speed response | L2 road links: free-flow speed and all-day mean speed v_day (BPR α = 0.15, β = 4, anchored at v_day: the effective v/c is the one that gives v_day); peak volume × length as the VKT weight |
| base speed, mean trip length, bus share, total trips | `traffic/city_transport_by_year.csv` |
| bus riders by district | city bus person-trips × district share of attracted trips (`traffic/district_trips_by_year.csv`) |

**Calibration.**
- The car emission factor per vehicle-km is chosen so that the baseline reproduces the car part of the road CO₂
  inside the 22 districts: EF = s_car · R_year / (365 · Σ C_od L_od).
- s_car is the share of the L1 road CO₂ that comes from gasoline and CNG.
- This keeps RQ3 consistent with RQ1, and the CO₂ term on the same base as R_t.

**Actions (per district i, per month).**
- Restriction r_i ∈ [0, 0.3]: the share of restricted car trips **into** district i that are removed. This works
  like Tehran's entry-based traffic zone.
- Bus boost b_i ∈ [0, 1]: extra bus service in district i, relative to its base (1 = double).
- **Annual bus budget.** The extra bus service over the year may not exceed 25 % of the annual base service. A boost
  beyond the remaining budget is scaled down.

**Behaviour.**
- Of the removed trips, a share φ_i = min(1, φ + φ_b · b_i / b_max) switches to transit. The rest are not made
  (suppressed).
- Extra service also attracts car drivers voluntarily: ε · b_i · bus riders_i · g / occupancy car trips into i, where
  ε is the service elasticity of ridership and g is the share of new riders who drove before.
- Monthly demand noise: lognormal, sd 0.05.
- **Uncertain parameters.** φ, φ_b, ε and g are drawn per episode from triangular distributions (config). The agent
  does not see them; it only sees the last month's outcome.

**Outcomes per month.** Car VKT is allocated half to the origin and half to the destination district.
- **Net CO₂ reduction** ΔC = car CO₂ saved − extra bus CO₂.
- **Travel time** (person-hours):
  - remaining car trips at the BPR-adjusted speed of their two districts;
  - shifted trips at the door-to-door transit speed;
  - each suppressed trip costs its own car travel time.
  - ΔT = T − T₀.
- **Energy** ΔE: diesel energy of the extra bus service (MJ).
- **Exposure** (the bridge from RQ2):
  - The road CO₂, `vkt_peak` and `vc_mean` features of each district are scaled to the simulated values.
  - The frozen RQ2 model (latest `rq2_sage_v2` run, `models/final.pt`) predicts the month's NO₂ pattern.
  - PEX = share of the domain population in cells above the month's baseline hotspot threshold (as in the RQ2
    response checks).

**Reward (per month).**

    r_t = ΔC_t / R_t  −  λ · ΔT_t / T₀_t  −  μ · ΔE_t / E_t  −  κ · ΔPEX_t / PEX₀_t

- R_t: baseline road CO₂ of the month. T₀_t: baseline car person-hours. E_t: baseline road fuel energy of the month.
- Every term is a relative change of the road sector, so λ = μ = κ = 1 means "1 % more travel time (or energy, or
  exposure) is as bad as 1 % more road CO₂".
- **Main weights:** λ = μ = κ = 1. This is a neutral equal weighting, chosen in advance; no monetary values are
  assumed.

**Observation.**
- month (sin, cos) and the month's heating degree-days
- remaining bus budget
- the last month's outcomes: realised shift share, ΔC/R, ΔT/T₀
- per district: baseline predicted z (mean) and share of the month's road CO₂

## Policies

**Static policies.** These are fixed rules, tuned by grid search on the **training** year (mean return over 16
episodes), so they are strong baselines.

| family | rule | grid |
|---|---|---|
| `none` | no action | — |
| `uniform` | the same (r, b) in every district and month | r ∈ {0.1, 0.2, 0.3}, b ∈ {0, 0.5, 1} |
| `winter` | (r, b) everywhere, only in Aban–Bahman (months 8–11, the pollution season) | as uniform |
| `hotspot` | (r, b) in the k districts with the highest baseline predicted z, every month | k ∈ {4, 8}, r, b as uniform |

**RL agent.**
- PPO (stable-baselines3), MLP policy.
- 150,000 steps per seed, 3 seeds (0, 1, 2), trained on one year.
- The returns of the 3 seeds are averaged per evaluation episode.

**Evaluation.**
- Every policy is evaluated on 100 episodes of each year.
- **Common random numbers:** episode k draws the same parameters and noise for every policy, so comparisons are
  paired.
- Training on 1402 and testing on 1403 (and the reverse) are the **transfers** that decide H3/H4. The same-year
  results are reported but do not decide.

## Decision rules

**H3 (effectiveness).**
- The candidate is the RL agent.
- The reference is chosen in two stages:
  - Each family is tuned on the training year, which gives one policy per family: `none`, `uniform`, `winter`,
    `hotspot`.
  - Of these four, the reference is the one that is best on the test year, chosen after the results. This is
    conservative.
- Δ = mean episode return of RL − reference. CI: paired bootstrap over episodes, 2,000 resamples, 95 %.
- **Supported:** Δ > 0 with the CI above 0 in both transfers.
- **Partial:** Δ > 0 in both transfers, at least one CI includes 0.
- **Not supported:** otherwise.

**H4 (simulation lowers policy risk).**
- **Screened choice:** the policy with the highest mean return on the training year, among all static grid policies
  and RL.
- **Unscreened choice:** a policy picked from the same candidate set without simulation, i.e. the average over the
  set.
- **Risk** = CVaR₁₀ (mean of the worst 10 % of episode returns on the test year). The share of episodes with
  return < 0 (worse than doing nothing) is also reported.
- Δ = CVaR₁₀(screened) − mean CVaR₁₀ over the set. CI: bootstrap over episodes.
- **Supported:** Δ > 0 with the CI above 0 in both transfers.
- **Partial:** Δ > 0 in both, at least one CI includes 0.
- **Not supported:** otherwise.

## Secondary analyses (do not change the verdicts)

1. **Trade-off curve.** λ ∈ {0.25, 0.5, 2, 4} (`rq3_lambda_v1`, 1 seed): CO₂ reduction vs travel-time loss of RL and
   of the tuned static policies.
2. **What the agent does.** The mean action by month and district, and the use of the bus budget over the year.
3. **Components.** CO₂, travel time, energy and exposure of every policy.

## Changes before the first full run

The smoke run (2,400 agent steps, 2 episodes; an installation check) showed three errors in the simulator. They
were fixed on 2026-11-01, **before any full run**:
1. **BPR on the peak v/c.**
   - Travel-time changes used the peak-hour v/c (up to 8 on some links). This overstated the congestion relief of
     all-day trips (a 10 % restriction "saved" 312 M person-hours a year).
   - BPR is now anchored at each link's all-day mean speed v_day of the RQ1 assignment.
2. **Inconsistent CO₂ base.**
   - The emission factor reproduced the car CO₂ of the whole CORE, while the reward divides by the road CO₂ of the
     22 districts. A 30 % restriction then "cut" 34 % of road CO₂.
   - The factor now reproduces the car share of the 22-district road CO₂.
3. **Suppressed trips were free.**
   - The code cancelled their time against the baseline. The design above (a trip that is not made costs its own
     travel time) was not implemented.
   - It is now implemented, with a unit test.

The design, the levers, the reward weights, the policies and the decision rules are unchanged.

## Known limitations

- **No network assignment.** Car VKT is split between the origin and destination districts. Route choice and
  through-traffic are not modelled; BPR speeds react to the district's own VKT change only.
- **Modelled demand.** The OD matrix is the RQ1 gravity model, not an observed survey.
- **Assumed behaviour.** The behavioural parameters (shift share, elasticity, share of new riders who drove before)
  come from the literature as triangular ranges. They are not estimated for Tehran; the ranges are sampled so the
  policies must be robust to them.
- **Short-run model.** Monthly steps, no long-run effects (car ownership, land use, induced demand).
- **Exposure is relative.** It comes from the RQ2 model and is a relative index (z pattern), not a concentration.
  A uniform cut barely changes it (as in the RQ2 response checks).
- **Fixed pollution feature.** Station CO (a model input) is kept at its observed value in the scenarios.

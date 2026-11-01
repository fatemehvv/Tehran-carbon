# RQ3 daily design (H3d): pre-registration

Written 2026-11-01, **before any daily agent is trained or any daily policy is evaluated**. Settings: `rq3_daily_v1`
and `hypotheses.H3d` in `configs/config.yaml`. The monthly design and its verdicts (`docs/preregistration_rq3.md`:
H3 "not supported", H4 "supported") stay as they are and are reported.

## Why a daily design

In the monthly design the best policy was static: "the most restriction everywhere". The months are predictable, so
an adaptive policy had nothing to adapt to, and RL matched the best static policy (99 %).

Tehran's real problem is daily:
- Pollution peaks on **calm, cold days** (inversion), which are not known in advance.
- The city's own instruments are daily: odd-even plates, emergency closures on high-pollution days.
- Restriction days are limited, because every restriction day costs mobility and political capital.

With a limited number of restriction days, the question is **which days**. This is a sequential decision under
uncertainty, the setting RL is built for (proposal: "adaptive interventions").

## Simulator

The monthly simulator (`docs/preregistration_rq3.md`) is used unchanged, with these differences:
- **Steps.** One step = one day; one episode = one Jalali year (365 / 366 days). Every monthly quantity is spread
  evenly over the days of its month (`CityTransport.daily`).
- **Exposure model.** The frozen daily GNN with wind on its edges (`rq2d_gat_wind_v1`, RQ2b). Its inputs are scaled in
  the scenario, as in the monthly design.
- **Weather.** The agent sees the day's weather: wind speed and direction (u, v) at the satellite overpass, and
  temperature. These are city-wide, observed (ERA5), and standardised over the year.
- **Severity.** The exposure term of a day is weighted by s_t = the day's observed city-mean NO₂ (TROPOMI, label
  domain) divided by the year's mean. Days without a retrieval take the mean of their month. The agent does **not**
  see s_t.
- **Restriction budget.** Over the year, the restriction may sum to at most **30 days** of full restriction (r_max)
  everywhere, the scale of Tehran's emergency plans. A restriction beyond what is left is scaled down. The agent sees
  the remaining budget.
- **Bus boost.** Off (bus service cannot change from day to day).
- **Unchanged.** Behaviour parameters and their ranges, demand noise and the reward weights (λ = μ = κ = 1).
  Reward of a day:

      dC / R  −  λ dT / T₀  −  μ dE / E  −  κ · s_t · dPEX / PEX₀

## Policies

**Static references.** These are fixed rules that do not react to the day, tuned on the training year (8 episodes).

| family | rule | grid |
|---|---|---|
| `none` | no action | — |
| `uniform` | r every day, everywhere (the budget lasts the year at r = 0.025) | r ∈ {0.025, 0.05, 0.1} |
| `winter` | r every day of Aban–Bahman (months 8–11), everywhere | r ∈ {0.05, 0.075, 0.1} |
| `hotspot` | r every day, in the k districts with the highest baseline predicted z | k ∈ {4, 8}, r ∈ {0.1, 0.2, 0.3} |

**Secondary comparison (not a reference): `calm`.** A simple weather rule: r everywhere on the days whose standardised
wind speed is below a threshold (−1, −0.5 or 0 sd), with r ∈ {0.1, 0.2, 0.3}. It reacts to the day like the agent, but
with one hand-made rule. RL − tuned `calm` is reported.

**Agent.**
- PPO, MLP policy.
- 146,000 steps per seed (400 years of days), 3 seeds, trained on one year.
- The returns of the 3 seeds are averaged per evaluation episode.

**Evaluation.**
- 30 episodes of each year, common random numbers.
- Train on 1402 and test on 1403, and the reverse. These are the transfers that decide; same-year results are
  reported only.

## Decision rule (H3d)

- Δ = mean episode return of RL − the best tuned static reference (`none`, `uniform`, `winter`, `hotspot`) on the test
  year, chosen after the results.
- CI: paired bootstrap over episodes, 2,000 resamples, 95 %.
- **Supported:** Δ > 0 with the CI above 0 in both transfers.
- **Partial:** Δ > 0 in both, at least one CI includes 0.
- **Not supported:** otherwise.

## Known limitations

- **Perfect weather knowledge.** The agent sees the day's actual weather. A real policy would use a forecast; this is
  an upper bound on what weather information is worth.
- **Severity in the reward.** The daily severity comes from TROPOMI and includes retrieval noise. Days without a
  retrieval take the mean of their month.
- **Relative exposure.** The exposure model predicts the relative pattern (z) of each day. The absolute level enters
  only through s_t.
- **Monthly inputs.** Within a month, emissions and traffic are spread evenly; the weekly cycle is not modelled.
- **Monthly limitations.** All limitations of the monthly simulator apply.

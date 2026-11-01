# Confirmatory test on the held-out year 1404 (H1c, H2c): pre-registration

Written 2026-11-01, **before any model is trained on or scored against 1404**. The 1404 files were only checked for
integrity (bands, grid, coverage; weather compared with the old file on the 1402/1403 overlap). Settings: the
`c1404_*` experiments and `hypotheses.H1c` / `H2c` in `configs/config.yaml`. After results exist, nothing here is
changed.

## Why a confirmatory test

The pre-registered tests on 1402/1403 (H1, H2) gave "not supported". In both, the effect pointed the right way in most
tests, but the intervals were wide: there are only about 4 independent areas per month. These verdicts stay as they are
(`docs/results_summary_fa.md`).

From here on, 1402 and 1403 are **development data**. They were used for two things:
- learning what makes the models better: a larger receptive field;
- rehearsing the design below with the two years we have.

The candidates and references below were chosen on the rehearsal only. **1404 is new data that no model and no
decision has seen.**

## Design

- **Test year.** 1404 (2025-03-21 .. 2026-03-20), monthly, the same 2,436-cell label domain and the same target recipe
  (TROPOMI NO₂, log, z-scored per month, hotspot = top 10 %).
- **Validation.**
  - The same 12 spatial blocks of 22 km, with a 2 km buffer.
  - For every block k, a model is trained on the training cells of block k in **1402 and 1403 together**, and predicts
    block k in **1404**. This is a new place and a new year at the same time (`split.test_year`).
- **Inputs of 1404.**
  - NO₂ and VIIRS: `gee/GEE_grid_NO2_weather_1404.js`.
  - Weather: ERA5-Land, mapped to the scale of the old weather file by a linear fit on the 1402/1403 overlap
    (correlation 0.999 for t2m, 0.85 for wind; `gee/harmonise_weather_1404.py`).
  - Station CO: AQCC 1404.
  - **The inventory and traffic of 1404 do not exist yet: the 1403 values stand in** (`year_proxy`). Emissions change
    slowly from year to year; this is a stated limitation.
- **Training.** As the v2 runs: 300 epochs, 3 seeds, no early stopping.

## Rehearsal (development data only)

The design above, run with the two years we have: train on one year, test the held-out blocks of the other. PR-AUC:

| model | test 1402 | test 1403 | mean |
|---|---|---|---|
| SAGE 3 layers (3 domains, gated) | 0.751 | 0.752 | 0.752 |
| SAGE 4 layers | 0.783 | 0.793 | 0.788 |
| **SAGE 5 layers** | **0.823** | 0.776 | **0.799** |
| SAGE 6 layers | 0.774 | 0.811 | 0.793 |
| SAGE 4 layers, energy only | 0.761 | 0.796 | 0.778 |
| SAGE 4 layers, pollution only | 0.724 | 0.785 | 0.755 |
| SAGE 4 layers, traffic only | 0.758 | 0.686 | 0.722 |
| **SLX-RF** (best traditional) | 0.761 | 0.720 | **0.741** |
| SLX-GBM | 0.727 | 0.737 | 0.732 |
| SLX-OLS | 0.707 | 0.749 | 0.728 |
| GBM / OLS (own cell) | 0.523 / 0.513 | 0.520 / 0.498 | — |
| physical forward operator | 0.398 | 0.325 | 0.361 |

**Selection rule:** in each comparison, the candidate and the reference are the best model of their family by the
rehearsal mean. Both sides are selected the same way, on development data.

## Hypotheses and decision rules

**H2c (RQ2).** The GNN locates combustion hotspots in a new year and new places better than the best traditional
model.
- Candidate: `c1404_sage5_v1`, SAGE with 5 layers, 3 domains, gated fusion.
- Reference, fixed in advance: `c1404_base_slx_rf_v1`, SLX-RF.

**H1c (RQ1).** Fusing traffic, energy and pollution locates hotspots better than the best single domain.
- Candidate: `c1404_sage4_v1`, SAGE 4 layers, 3 domains, gated.
- Reference, fixed in advance: `c1404_sage4_single_v1` [energy], the same network with energy only.
- 4 layers, because the single domains were rehearsed with 4 layers.

**Rule (both).**
- Δ = PR-AUC(candidate) − PR-AUC(reference) on 1404, the mean over the 12 months.
- 95 % CI: block bootstrap over the 12 spatial blocks, 2,000 resamples (the same procedure as H1/H2).
- **Supported:** Δ > 0 and the CI lies above 0.
- **Partial:** Δ > 0, the CI includes 0.
- **Not supported:** Δ ≤ 0.

The other models are trained and reported descriptively: the remaining traditional models, and the traffic-only and
pollution-only networks. They do not change the verdicts.

## Relation to the earlier rules

- The earlier rules required the effect in **every** year **and** every transfer. Here there is one held-out test.
  This is the standard development / test design, so a single test is the natural rule.
- The earlier verdicts (H1, H2 "not supported") are **not replaced**. Both are reported:
  - the pre-registered development tests;
  - this confirmatory test.

## Known limitations

- **Stand-in inputs.** The 1403 inventory and traffic stand in for 1404, so a model cannot use real changes of
  emissions in 1404. This works against all models equally.
- **Weather mapping.** The 1404 weather is a mapped ERA5-Land export, not the identical product of 1402/1403. The
  mapping residual is 0.4 °C for t2m and 0.25 m/s for wind.
- **One test year.** The test year has the same small number of independent areas per month (about 4). The interval
  can still be wide.

# Does the NO₂ target favour some sectors? A check (not a thesis hypothesis)

Written 2026-11-02, **before the analysis was run**, after the question "why NO₂ and not CO₂?" was raised again. The
NOx : CO₂ ratio of combustion differs between fuels: natural gas in buildings is expected to give less NOx per tonne of
CO₂ than road diesel and gasoline. Buildings are the largest sector of the inventory, so the NO₂ hotspots may
under-represent them.

This check changes no model and no verdict. It says how far "NO₂ hotspot" may be read as "CO₂ hotspot".

## Data

- Target recipe as in the models: TROPOMI monthly NO₂ (`data/raw/satellite/Tehran_NO2_grid_2023_2025.tif`), log, z-score
  per month over the 2,436-cell label domain, hotspot = top 10 %.
- Label domain and the 12 blocks: `preds/fold_map.npy` of `rq2_sage_v2`.
- Sectors: annual L2 inventory × the L1 monthly share of the sector, as in `configs/config.yaml`: road (`road_transport`),
  buildings (`buildings_residential` + `buildings_commercial`), industry, power (`power_plants_CORE` + `power_plants_METRO`).
- Years: 1402 and 1403. Year 1404 has no inventory of its own, so it is left out.

## Measures

1. **Hotspot capture.** The share of a sector's CO₂ of the month emitted inside that month's NO₂ hotspots.
   Δ₁ = capture(road) − capture(buildings), mean over the 12 months.
   Also reported: capture of total CO₂, against the most any 10 % of the cells could hold according to the inventory.
2. **Pattern correlation.** Spearman between z and the Gaussian-smoothed sector layer, mean over the months. NO₂ is
   transported and the column is smooth, so each sector is smoothed with its own best σ from the list of the forward
   operator (0 to 15 km). This gives each sector its best chance.
3. **NO₂ per tonne of CO₂.** Each month, OLS of the raw NO₂ column on the four smoothed sector layers (tonnes per cell) over
   the domain, plus an intercept. The coefficient is the NO₂ column per tonne of the sector's CO₂.
   Δ₃ = coefficient(road) − coefficient(buildings), mean over the months. One σ for all sectors: the one with the best
   mean R² (the others are reported as sensitivity). Reported for all months, the cold months 8–11 and the others.
   The correlation between the road and buildings layers is reported: if it is high, this measure cannot separate them well.

**Uncertainty.** Block bootstrap over the 12 blocks, 2,000 resamples, 95 % (the same design as the hypotheses).

## Reading rule, fixed now

The expected direction of a bias against buildings is Δ₁ > 0 and Δ₃ > 0.

- **Bias shown:** Δ₁ and Δ₃ are both positive with the interval above 0, in both years.
- **Partial evidence:** Δ₁ or Δ₃ is positive with the interval above 0 in both years, but not both measures.
- **Not shown:** otherwise. This does not prove there is no bias; the intervals are wide, as in the hypotheses.

## Limits known in advance

- Association is not attribution. Road and buildings are concentrated in the same dense parts of the city.
- The layers are the inventory's own spatial allocation, not measured emissions.
- NO₂ also depends on chemistry, terrain and wind, which this check does not model.
- `docs/analysis/sector_bias_check.py` is a standalone script, run once on the laptop because the server was off. It does
  not import the project code and it is not a tracked run in `runs/`. To make it a tracked run, it needs an `analyze`
  command on a server.

## Results (2026-11-02)

Tables: `docs/tables/t11_sector_bias.csv`, `t12_sector_bias_sigma.csv`. The first version of the script failed on a year
conversion (`year - 621`) before it produced any number; that line was fixed, nothing else changed.

| sector | share of the domain's CO₂ (1402 / 1403) | CO₂ inside the NO₂ hotspots, 95 % interval (1402) | same (1403) | Spearman with z (1402 / 1403) |
|---|---|---|---|---|
| road | 34.4 % / 33.5 % | 34.3 % [2.1, 46.9] | 34.5 % [1.9, 52.2] | 0.89 / 0.90 |
| buildings | 48.7 % / 50.5 % | 32.9 % [2.6, 45.7] | 33.7 % [2.5, 53.4] | 0.90 / 0.91 |
| industry | 9.5 % / 9.3 % | 1.1 % [0.0, 3.0] | 0.8 % [0.0, 2.4] | −0.07 / −0.10 |
| power | 7.4 % / 6.7 % | 3.8 % [0.0, 9.8] | 3.6 % [0.0, 12.6] | 0.26 / 0.26 |

| test | 1402 | 1403 |
|---|---|---|
| Δ₁ capture, road − buildings | +0.014 [−0.042, +0.049] | +0.008 [−0.063, +0.042] |
| Δ₃ NO₂ per Mt CO₂, road − buildings (σ = 12 km) | +0.32 [−8.8, +10.2] | −3.67 [−12.6, +5.5] |
| correlation of the road and buildings layers (σ = 12 km) | 0.999 | 0.999 |
| total CO₂ in the NO₂ hotspots / in the best 10 % of the cells | 27.6 % / 45.5 % | 28.2 % / 44.7 % |

**Verdict under the rule: a bias against buildings is not shown.** This does not prove there is no bias:
- After smoothing, the road and buildings layers are almost the same map (r = 0.999; r = 0.58 for unsmoothed cells), so a
  spatial test cannot separate them. Δ₃ also changes sign with σ in 1402, and its intervals are very wide.
- The direction does not favour the guess either: Spearman is slightly higher for buildings at every σ.

**Not expected in advance (descriptive, outside the rule).** The NO₂ hotspots hold little of the CO₂ of industry (about 1 %, on
10 % of the cells, interval clearly below 10 %). Power is low too, but its interval reaches 10 %. Industry and power are
16 to 17 % of the domain's CO₂. A NO₂ hotspot therefore marks the dense urban core: it holds about a third of the road and
buildings CO₂, and not the industrial sites. The right wording is "hotspot of urban combustion", not "hotspot of CO₂".

The intervals of the capture are wide because two of the 12 blocks hold about 95 % of the hotspot cell-months (blocks 6 and 2).

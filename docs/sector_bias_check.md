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

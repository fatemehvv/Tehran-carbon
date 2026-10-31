# RQ1 / H1: pre-registration of the analysis

Written 2026-10-31, **before any RQ1 model is trained**. The settings referred to here are `defaults` and the
`rq1_*` experiments in `configs/config.yaml` (git commit of this file). After results exist, nothing here is changed.
A deviation becomes a new config version, and it is listed in the results chapter with its reason.

## Question and hypothesis

- **RQ1.** What is the best architecture for fusing heterogeneous urban data (traffic, air pollution, energy)?
- **H1.** Fusing traffic, pollution and energy data in one framework predicts the urban carbon-combustion pattern
  more accurately than single-domain analysis.

**What changed from RQ1 v2.** RQ1 v2 (inventory + Dempster–Shafer over 22 districts) tested one fixed design on
n = 22 districts. No alternative design was compared, and pollution data was evidence only, not an input. Here the
fusion design is chosen empirically, and all three domains are model inputs. The test uses 2,436 cells × 24 months
under honest spatial validation. The inventory (L1/L2) is still used, but only to build features (energy and
activity → CO₂).

## Data

- **Unit.** One cell of the 1/120° L0 grid (≈ 0.75 × 0.93 km) in one Jalali month. Years: 1402 and 1403.
- **Label domain.** The 2,436 cells inside CORE (Tehran, Rey and Shemiranat counties) that the inventory gives
  CO₂. This is the same domain as RQ2.
- **Target.**
  - The TROPOMI tropospheric NO₂ column, monthly composite.
  - It is log-transformed and z-scored within each month over the domain.
  - **Hotspot** = top 10 % of the domain in that month.
  - NO₂ is co-emitted with CO₂ by combustion and is short-lived. It is observed independently of our inventory, so
    there is no circularity.

**Input branches** (identical lists for every model; the spatial lags are box means with half-widths of 3 and
9 cells):

| branch | features |
|---|---|
| traffic | road CO₂ of the month (L2 traffic model), road km by class, peak vehicle-km, mean v/c |
| energy | residential, commercial, industry and power-plant CO₂ of the month; heating degree-days |
| pollution | AQCC station CO, monthly, inverse-distance interpolated |
| context (always given) | GHSL population and building volume, VIIRS, OSM industrial share, ERA5 temperature and wind |

Station NO₂/NOₓ are left out of the primary design because they are the target species. They are tested in
`rq1_pollution_no2_v1`. The 1403 air files contain only CO, NO₂ and NOₓ, so CO is the only other pollutant
available in both years.

## Models (fixed now, no tuning on test data)

Each branch has its own MLP encoder: hidden size 32, 2 layers, dropout 0.1. There is no graph; the graph is RQ2's
question. Three fusion types are compared:

| fusion | description |
|---|---|
| early | concatenate all features, one encoder |
| late | one prediction per branch; global learned weights. This is the learned counterpart of the DS reliabilities of RQ1 v2 |
| gated (primary) | softmax gate over the branch embeddings for every cell-month. The weights are interpretable |

**Ablation `rq1_ablation_v1`.** All 7 non-empty subsets of {traffic, energy, pollution} × the 3 fusion types,
with context always included.

**References.**
- `rq1_context_only_v1`: context only.
- `rq1_base_forward_v1`: physical forward operator on the inventory.
- `rq1_base_gbm_v1`: gradient boosting on the same features.

**Training.**
- AdamW, learning rate 3e-3, weight decay 1e-4, up to 300 full-batch epochs.
- Early stopping (patience 30) on one inner training block.
- MSE of z on training cells.
- 3 seeds, with predictions averaged.

## Validation and metrics

- **Main design.**
  - Spatial leave-one-block-out with 22 km blocks (the autocorrelation range), which gives 12 blocks.
  - Training cells within 2 km of the test block are dropped.
  - Each year is analysed separately: 1402 and 1403 are two replicates.
- **Temporal transfer.** Train on one year, predict the other.
- **Primary metric.** PR-AUC of the hotspot label (chance = 0.10), per month on the pooled out-of-fold map, then
  averaged over the 12 months.
- **Secondary metrics.** ROC-AUC, F1 of the predicted top 10 %, Spearman, R², MAE, RMSE. MAE and RMSE make the
  results comparable with OpenCarbon and HENCE.
- **Uncertainty.** Block bootstrap over the 22 km blocks, 2,000 resamples. Paired differences use identical
  resamples.

## Decision rule for H1 (`hypotheses.H1` in the config)

The candidate is gated fusion with all three domains. The reference is the best single-domain variant (any fusion
type), chosen after the results, which is conservative for the candidate.

- **Supported:**
  - in both years, ΔPR-AUC > 0 with the 95 % CI above 0, **and**
  - Δ > 0 in both temporal transfers.
- **Partially supported:** Δ > 0 in both years and both transfers, but at least one CI includes 0.
- **Not supported:** otherwise.

**Answer to RQ1.** The fusion type × domain subset with the best primary score. It is reported with its CI against
the others and with the gate-weight maps, which show which domain matters where and when.

## Secondary analyses (pre-specified)

1. Fusion types compared at equal inputs (early vs late vs gated, all three domains).
2. Gate-weight maps per month (`preds/weights_<year>.npz`, `figures/maps_<year>.png`).
3. Pretrained AlphaEarth satellite embeddings added to context (`rq1_gated_emb_v1`). They are kept only if they
   improve the score under the same validation.
4. Station NO₂/NOₓ sensitivity (`rq1_pollution_no2_v1`).

## Known limitations

- **Few independent observations.** Monthly TROPOMI has a ~3.5 × 5.5 km native pixel and ~20 km
  autocorrelation, so there are only ≈ 4 independent patches per month. The CIs will be wide.
- **NO₂ is not CO₂.** NO₂ marks combustion, not CO₂ itself; diesel and power plants are NOₓ-rich per unit CO₂.
- **Station coverage.** 18–23 stations, so the interpolated pollution field is smooth.
- **Shared inventory.** The traffic and energy inventory features share the L1 fuel totals; the road CO₂ includes
  the fuel closure.

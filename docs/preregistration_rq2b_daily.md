# RQ2b / H2b: pre-registration of the daily analysis

Written 2026-10-31, **before the daily data are downloaded and before any daily model is trained**. Settings: the
`rq2d_*` experiments and `hypotheses.H2b` in `configs/config.yaml`. After results exist, nothing here is changed; a
deviation becomes a new config version and is listed in the results chapter.

## Why a daily analysis

The monthly analysis (H2) found that seeing the neighbourhood is what locates hotspots (MLP 0.47 → GNN 0.80 PR-AUC).
However, a traditional model with fixed neighbourhood windows (SLX-GBM) came close to the GNN, and the verdict was
"not supported".

The likely reasons are stated in advance:
1. **Smooth target.** A monthly mean NO₂ map is smooth (~4 independent patches per month), so fixed isotropic
   windows already capture most of it.
2. **No wind direction.** The monthly mean averages the wind direction away, so a graph model cannot use what it is
   built for: learning from which neighbour the signal arrives.

Daily TROPOMI fields together with the wind at the satellite overpass remove both limits. This is the setting where
message passing should matter most (PM2.5-GNN, Wang et al. 2020; AirPhyNet, Hettige et al. 2024).

## Question and hypothesis

- **RQ2b.** With daily data and the day's wind, does a wind-aware GNN locate combustion hotspots better than
  traditional models?
- **H2b.** A GNN whose edges carry the day's wind beats the best traditional model.

## Data

- **Unit.** L0 cell (1/120°) × day, Jalali 1402 (365 days) and 1403 (366 days). The same 2,436-cell label domain.
- **Target.**
  - Daily TROPOMI tropospheric NO₂ (OFFL L3, cloud fraction < 0.3), from `gee/GEE_daily_NO2_wind_L0.js`.
  - Log-transformed (floor 1e-6 mol/m²) and z-scored per day over the clear domain cells.
  - A day is used only if at least 50 % of the domain is clear; cloudy cells are missing values, excluded from the
    loss and the metrics.
  - Hotspot = top 10 % of the clear domain cells of that day.
- **Features (identical for every model).**
  - Inventory and context: the monthly layers repeated over the days of their month.
  - Daily layers:
    - temperature and heating degree-days (ERA5, bias-corrected)
    - wind at 10 UTC (east and north components, speed; ERA5-Land, city mean)
    - weekend (Friday) and Nowruz flags
    - station CO interpolated per day
- **Spatial lags.** None. The SLX baselines add their own neighbourhood means at 1, 3, 9 and 15 cells.

## Models (fixed now)

| id | model |
|---|---|
| `rq2d_mlp_v1` | control: early fusion of the branches, no graph |
| `rq2d_sage_v1` | multi-scale GraphSAGE (isotropic neighbourhood) |
| `rq2d_gat_v1` | multi-scale GATv2, edge features = scale + edge direction (no wind) |
| **`rq2d_gat_wind_v1`** | **primary**: as `rq2d_gat_v1` plus, on every edge, the day's wind component carried from the neighbour towards the cell (m/s; positive = the neighbour is upwind) |
| `rq2d_base_ols_v1`, `rq2d_base_slx_ols_v1`, `rq2d_base_gbm_v1`, `rq2d_base_slx_gbm_v1`, `rq2d_base_forward_v1` | traditional models on the same features (the SLX models also see the wind as a feature, but with fixed, direction-blind windows) |

**Training.**
- Hidden size 32, dropout 0.1.
- AdamW, learning rate 3e-3, weight decay 1e-4.
- 30 epochs of 8-day batches (≈1,400 gradient steps per year).
- No early stopping (see the RQ1/RQ2 deviation).
- 1 seed, for cost. The random forest is left out: it costs too much on about 0.9 M rows per year.

**Fusion.** Early fusion. It was the simplest and among the best in the first RQ1 runs, and it is closest to the
input layer of the earlier MS-SAGE.

## Validation, metrics, decision rule

These are the same as H2.
- **Validation.** 22 km leave-one-block-out with a 2 km buffer, years separately, plus the two temporal transfers.
- **Metrics.** PR-AUC per day (primary), averaged over the usable days. Also ROC-AUC, F1 top 10 %, Spearman, R²,
  MAE and RMSE.
- **Uncertainty.** Block bootstrap with 500 resamples (fewer than the monthly 2,000, because daily scoring costs
  ~30× more).

**H2b rule.** The candidate is `rq2d_gat_wind_v1`. The reference is the best traditional model, chosen after the
results.
- **Supported:** in both years, ΔPR-AUC > 0 with the 95 % CI above 0, and Δ > 0 in both transfers.
- **Partial:** Δ > 0 everywhere, at least one CI includes 0.
- **Not supported:** otherwise.

## Secondary analyses (do not change the verdict)

1. **Value of the wind on the edges.** `rq2d_gat_wind_v1` vs `rq2d_gat_v1`.
2. **Value of the graph.** `rq2d_sage_v1` / `rq2d_gat_v1` vs `rq2d_mlp_v1`.
3. **Daily vs monthly.** Daily vs monthly scores of the same model families (descriptive; different targets).

## Known limitations

- **Clouds.** Many winter days have few clear cells and are dropped. Winter (highest heating emissions) is
  therefore under-represented.
- **Wind.** One city-wide wind vector per day; there is no spatial wind field over the 1 km grid.
- **Noisier target.** The daily target is much noisier than the monthly one (single overpass, retrieval noise), so
  absolute scores are expected to be lower than monthly.
- **Monthly inputs.** Inventory and traffic are monthly / annual and do not vary within a month. Only weather, wind,
  station CO and the calendar flags vary by day.

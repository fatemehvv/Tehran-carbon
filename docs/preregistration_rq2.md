# RQ2 / H2: pre-registration of the analysis (new repository)

Written 2026-10-31, **before any model of this repository is trained**. The settings are `defaults` and the
`rq2_*` experiments in `configs/config.yaml`. This replaces the 2026-10-12 pre-registration of the earlier package
for the re-run on the new code. Its question, target, domain, validation and decision rule are kept. The changes
are listed at the end. After results exist, nothing here is changed; a deviation becomes a new config version.

## Question and hypothesis

- **RQ2.** Which GNN-based prediction models can locate Tehran's carbon-combustion hotspots with high accuracy?
- **H2.** By modelling spatial dependencies between the nodes of the urban graph, a GNN reveals the relations in the
  fused multi-source data better than traditional models.

## Data, target and validation

These are the same as RQ1 (`docs/preregistration_rq1.md`).

- **Unit and target.** 1 km L0 cell × Jalali month, 1402 and 1403. The 2,436-cell label domain. Target = log
  TROPOMI NO₂, z-scored per month. Hotspot = top 10 %.
- **Features.** The same branches (traffic, energy, pollution, context), **without fixed spatial lags**
  (`data.spatial_lags: []`): the GNN must learn the neighbourhood itself. The SLX baselines get fixed neighbourhood
  means at 1, 3, 9 and 15 cells.
- **Main design.** 22 km leave-one-block-out, 2 km buffer, years separately; temporal transfer 1402 ↔ 1403.
- **Metrics.** Primary PR-AUC. Secondary ROC-AUC, F1 of the top 10 %, Spearman, R², MAE, RMSE.
- **Uncertainty.** Block bootstrap (2,000).

## Models (fixed now)

| id | model |
|---|---|
| `rq2_mlp_v1` | control: the same network (gated fusion of the branches) without a graph |
| **`rq2_sage_v1`** | **primary GNN**: gated fusion, then 3 layers of multi-scale GraphSAGE (neighbours 1, 3 and 9 cells away; ≈ 23 km reach), residual |
| `rq2_gcn_v1` | multi-scale GCN, same graph (secondary) |
| `rq2_gat_v1` | GATv2 over all scales, with edge features = scale + unit direction of the edge; **1 seed** for cost (earlier run: ~2.7 h per fold with 3 seeds) (secondary) |
| `rq2_base_ols_v1`, `rq2_base_slx_ols_v1`, `rq2_base_gbm_v1`, `rq2_base_slx_gbm_v1`, `rq2_base_slx_rf_v1`, `rq2_base_forward_v1` | traditional models: own-cell and SLX OLS / gradient boosting / random forest; physical forward operator |

**Training.**
- Hidden size 32, dropout 0.1.
- AdamW, learning rate 3e-3, weight decay 1e-4, up to 300 epochs.
- Early stopping (patience 30) on one inner training block.
- 3 seeds averaged (GAT: 1).

## Decision rule for H2 (`hypotheses.H2`)

The candidate is `rq2_sage_v1`. The reference is the best traditional model by mean PR-AUC, chosen after the
results.

- **Supported:** in both years, ΔPR-AUC > 0 with the 95 % CI above 0, and Δ > 0 in both transfers.
- **Partial:** Δ > 0 everywhere, at least one CI includes 0.
- **Not supported:** otherwise.

**Persistence.** For the transfers, the observed map of the other year ("persistence") is reported as a reference
ceiling. It is not a competitor: it cannot respond to scenarios.

## Secondary analyses (pre-specified; none changes the H2 verdict)

1. **Other GNNs.** GCN; GAT with edge direction.
2. **OD links (`rq2_sage_od_v1`).** The primary GNN plus travel-demand links between districts, as the OD edges of
   HENCE (Zeng et al. 2024). A cell receives the flow-weighted mean embedding of the other districts its district
   exchanges trips with. The OD matrix is the gravity-model OD of RQ1 v2 (modelled, not observed), averaged over the
   two years.
3. **Receptive field.**
   - 1, 2 and 4 layers (`rq2_receptive_field_v1`).
   - 0 layers = `rq2_mlp_v1`; 3 layers = `rq2_sage_v1`.
   - 3 layers with the adjacent cells only (`rq2_sage_single_scale_v1`).
4. **Design optimism.** MLP, SAGE and SLX-GBM under naive random 10-fold CV (`*_random_v1`), compared with the
   22 km blocks.
5. **District level (`main.py analyze districts`).** Out-of-fold maps → 22 districts → L/M/H terciles.
   Linear-weighted κ with the RQ1 v2 DS classes, the inventory-only classes and the observed NO₂ classes. NO₂ is one
   of the DS sources, so this is not fully independent.
6. **Response model (`main.py analyze response`, bridge to RQ3).** The final `rq2_sage_v1` model, trained on all
   cells of both years:
   - −30 % CO₂ in each district: share of districts where the hotspot falls (sign test).
   - The distance at half effect.
   - A uniform vs targeted (4 hottest districts) cut of the same total 10 % CO₂: change of the hotspot contrast
     index (HCI) and population exposure (PEX).

## Changes from the earlier pre-registration (2026-10-12)

- **GNN layers.** From PyTorch Geometric (SAGEConv / GCNConv / GATv2Conv) instead of hand-written layers.
- **Fusion.** A gated fusion of the branches feeds the graph (the earlier input was one MLP on all features).
- **Early stopping.** On an inner block, instead of fixed 300 epochs.
- **GAT.** Pre-registered with 1 seed (the earlier run had to reduce to 1 seed during the run). It also gets the edge
  direction.
- **APPNP and the 11 km block scheme for neural models.** Not run (they were not run in the earlier re-run either).
- **Feature-group permutation importance.** Dropped: permuting over the whole box breaks the spatial structure (as
  concluded in the earlier v2 re-run). The importance of the domains comes from the RQ1 ablation and the gate
  weights.
- **New.** OD links (secondary 2) and MAE/RMSE (comparability with OpenCarbon and HENCE).

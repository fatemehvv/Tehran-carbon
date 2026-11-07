# Paper checks (branch `paper_TC`): what is run and how it is read

Written 2026-11-07, **before any run of the `paper_*` experiments and before any robustness check**. Settings: the
`paper_*` experiments and the hypotheses `P2_reach`, `P2_reach4`, `P2c_reach` in `configs/config.yaml`. After
results exist, nothing here is changed; a deviation becomes a new config version and is listed at the end.

## Status

- These are **not thesis hypotheses.** The thesis verdicts (H1, H2, H2b, H1c, H2c, ...) stay as they are.
- They were added **after** the thesis results, to answer the questions a reviewer is expected to ask
  (`docs/literature_table_ch2.md`, and the review of RQ1–RQ2).
- The **candidates' results are already known** (the thesis GNN runs). Only the new references and the robustness
  statistics are unknown when this is written. The rules below are fixed before those exist.
- `defaults` is not touched, so the config hash of every thesis experiment is unchanged (checked: all 161 resolved
  thesis configs give the same hash before and after the change).

## Check 1: reach-matched SLX models (`paper_slx_reach_v1`, `paper_c1404_slx_reach_v1`)

**Why.** The largest SLX window of the thesis has a box half-width of 15 cells (about 12 km). The GNN sees
L layers × 9 cells: 27 cells (about 23 km) with 3 layers, 45 cells (about 38 km) with 5. The thesis says that if the GNN
beats SLX, the gain comes from *learning how to combine* the neighbourhood, not from *seeing* it. That only holds if
both see equally far.

**Design.** The SLX OLS, GBM and RF of the thesis, with box half-widths {1, 3, 9, 15} plus 27, plus 27 and 36, plus
27, 36 and 45 (= the reach of 3, 4 and 5 SAGE layers). Everything else as the thesis SLX models: the same features,
hyper-parameters, 22 km blocks with a 2 km buffer, transfers; and, for 1404, the confirmatory design of H2c.

**Rule.** The thesis rule (`HypothesisTest`: supported / partial / not supported), with the best reference chosen after
the results among the thesis SLX models **and** the reach-matched ones:

| name | candidate | data |
|---|---|---|
| `P2_reach` | `rq2_sage_v2` (3 layers) | 1402, 1403 and the two transfers |
| `P2_reach4` | `rq2_receptive_field_v2`, 4 layers | 1402, 1403 (no transfers) |
| `P2c_reach` | `c1404_sage5_v1` (5 layers, the H2c candidate) | 1404 |

**Reading for the paper.**
- **"Learned aggregation"**: `P2c_reach` is supported or partial, **and** `P2_reach4` has Δ > 0 in both years. The
  paper may then say the GNN's advantage is not only its reach.
- **"Reach, not architecture"**: `P2c_reach` has Δ ≤ 0. The paper's main claim becomes that a reach matched to
  the autocorrelation range of the target matters more than the model family. Title and abstract change accordingly.
- **Mixed** (anything else): both results are reported, and the advantage is called "partly reach".

## Check 2: persistence on the held-out year

**Why.** In the transfers, the observed map of the other year scores 0.78; the GNN scores 0.782 on 1404. A reviewer will
ask whether the model beats re-using an old satellite map.

**Design.** `python main.py robustness --hypothesis H2c` (and `P2c_reach`) scores the observed 1402 map, the observed
1403 map and their mean, month by month, on 1404. Paired Δ (candidate − persistence) with the same block bootstrap.

**Reading.** Descriptive, no verdict. Persistence is not a competitor: it uses the target of the held-out blocks, and it
cannot answer a scenario. If the candidate scores below persistence, the paper says so plainly, and justifies the model
by its use as a scenario (response) model only, not by its accuracy.

## Check 3: per-block results and leave-one-block-out

**Why.** The bootstrap resamples only 12 blocks, the hotspots sit in a few of them, and the intervals are lopsided
(for example +0.002 [−0.010, +0.161] for H1c).

**Design.** `python main.py robustness --hypothesis <name>` for H2, H2c, H2c_robust, H1c, P2_reach, P2_reach4 and
P2c_reach (`src/tehran_carbon/evaluation/robustness.py`):
- the metric inside every block, over the months with a hotspot in that block: blocks won, Wilcoxon signed-rank test
  and sign test over the blocks;
- the jackknife: Δ with one block left out at a time, its range, and a t interval with K − 1 degrees of freedom;
- the share of all hotspot cell-months in every block (how concentrated the hotspots are).

**Reading** (one per test year):
- **robust**: bootstrap CI above 0, jackknife CI above 0, and the candidate ahead in more than half of the valid blocks;
- **fragile**: bootstrap CI above 0, but the jackknife CI includes 0 or the candidate leads in at most half the blocks;
- **no effect shown**: the bootstrap CI includes 0 (or Δ ≤ 0).

The paper reports this reading next to every bootstrap verdict. If H2c is **fragile**, the abstract does not call the
advantage significant; it gives the effect size and both intervals.

## Check 4: a fairer physical baseline (`paper_forward_sectors_v1`, `paper_c1404_forward_sectors_v1`)

**Why.** The thesis physical operator smooths the **sum** of all inventory layers, power plants included. Power plants
hold about 4 % of their CO₂ in the NO₂ hotspots, so the sum peaks in the wrong cells. That makes the physical
baseline (0.35) look weaker than it needs to be.

**Design.** `forward_sectors`: z ~ a + Σ_s b_s · log1p(K_σ ∗ E_s), with one non-negative yield b_s per inventory
sector and one Gaussian width σ, both fitted on the training cells (σ by Spearman correlation, as the thesis
operator). Same validation as the thesis models.

**Reading.** Descriptive. If `forward_sectors` reaches the SLX models (within 0.02 PR-AUC of the best SLX model), the
discussion sentence "the relation between emissions and the NO₂ pattern is more complex than an isotropic dispersion"
is withdrawn, and the paper says a sector-weighted dispersion is nearly as good. Otherwise the sentence stays, with the
fairer baseline as its evidence.

## Run order

In a separate folder on the server (`~/Tehran-carbon-paper`, with copies of `data/raw` and `runs/` of the thesis
folder), on branch `paper_TC`:

1. `train` the four `paper_*` experiments.
2. `evaluate` P2_reach, P2_reach4, P2c_reach.
3. `robustness` for H2, H2c, H2c_robust, H1c, P2_reach, P2_reach4, P2c_reach.

## Known limitations

- **Known candidates.** The references are new, the candidates are not. A stronger reference only makes the checks
  harder for the candidate, but the choice of checks was made knowing the thesis results.
- **Box vs graph.** An SLX box mean and a dilated SAGE neighbourhood with the same reach are not the same operator.
  The check equalises how far each model sees, not how.
- **Edges of the grid.** A box half-width of 45 cells is larger than half the grid height (72 rows); the box is padded
  with the nearest values at the border (`mode="nearest"`, as the thesis SLX models).
- **Stand-in 1404 inputs.** As in H2c: the 1403 inventory and traffic stand in for 1404.

## Deviations

None yet.

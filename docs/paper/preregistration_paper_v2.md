# Paper v2 (branch `paper_TC`): pre-registration of the stages and the confirmatory tests

**Status: DRAFT, written 2026-11-07, before any `paper2_*` run.** Plan: `docs/paper/plan_q1_fa.md` (version 4).
This draft is finalised once, after stage A, with three numbers that stage A measures (the margin δ, the power of every
confirmatory test, the compute budget) and the names of the frozen candidate configs. Finalisation happens **before any
stage B, C or D run is scored and before the 1405 data are opened**. The commit that finalises it is pushed to GitHub;
its push time is the registration time (an OSF copy is optional). After that, nothing here changes; a deviation becomes
a new config version and is listed under *Deviations*.

These are **not thesis hypotheses**. The thesis verdicts (H1, H2, H1c, H2c, ...) stay as they are and are reported.

## What is known before writing

- All thesis results (`docs/results_summary_fa.md`, `docs/tables/`), in particular:
  - one Gaussian-smoothed sector layer of the inventory (buildings, σ = 8 km) has a mean monthly Spearman of 0.90–0.91
    with z, and four sector layers reach R² = 0.89 at σ = 12 km (`t12`, in-sample);
  - with the same inputs, the RQ1 fusion types differ by up to 0.18 PR-AUC (`t02`);
  - the road and building layers correlate 0.999 once smoothed by 8 km or more (`t12`).
- **Not known:** every `paper2_*` result, the stage A readings of the existing thesis comparisons with the new inference,
  the paper checks of `preregistration_paper_checks.md` (not run yet), and everything about 1405.

## Stage A (descriptive; no verdict of the paper depends on it except through the three numbers below)

1. **Leakage fix.** In the thesis the station layers (pollution domain) are interpolated once per year from all
   stations, the stations inside the held-out block included. `split.fold_aware_stations` rebuilds them in every fold
   from the stations farther than the buffer (2 km) from the test block, for every year the fold uses (the 1404 design
   included). The RQ1 ablation, the primary GNN, the no-graph control, the traditional models, the reach-matched SLX
   and the 1404 designs are re-run that way (`paper2_*`), neural models with 5 seeds, every seed stored.
2. **Readings** (`python main.py paper evaluate`): `PA_H1`, `PA_H1_drivers`, `PA_H2`, `PA_H2_reach`, `PA_H2c`,
   `PA_H2c_all`, `PA_H1c`, and the thesis hypotheses with the same inference for comparison. Reported as sensitivity
   of the thesis results to the leakage fix. They are not confirmatory: 1402–1404 have been seen.
3. **Target validity** (`paper target`): stations vs TROPOMI, elevation, the daily cloud-screened target, n_eff.
4. **Power** (`paper power`, on the existing comparisons H1, H2, H2c, H1c) and **timing** (`paper timing`).

## The three numbers fixed after stage A

- **Margin δ** (equivalence and non-inferiority), per co-primary metric:
  δ = max(0.02, 2 × SD_seed), where SD_seed is the standard deviation over the 5 stored seeds of the city-wide metric,
  averaged over the test years and over the two neural models `paper2_rq2_sage_v1` and the three-domain gated variant
  of `paper2_rq1_ablation_v1`. Sensitivity readings with δ = 0.01 and 0.03 are reported next to the main one.
- **Power** of each confirmatory test at δ and 2δ (block-level approximation, `paper power`). A test with power below
  0.5 at δ is labelled *low power* in the paper. It is **not** dropped.
- **Compute rule.** If `paper timing` estimates more than 168 GPU-hours for the 50-trial searches of all neural families
  together, the neural searches use Hyperband / successive halving (Li et al. 2018, JMLR) with the same 50 trials; the
  tree models always use full 50-trial random search.

## Inference (fixed now)

- **Co-primary metrics:** PR-AUC of the city-wide hotspots (top 10 % of every month, the thesis label) and Spearman of
  the predicted and observed z over all domain cells. Each is tested at α = 0.025.
- **Test of superiority:** exact block sign-flip randomisation test of the city-wide difference (one-sided; candidate
  and reference swapped block by block; rank metrics on predictions rank-normalised within each month). Holm's
  adjustment over the whole confirmatory family (hypotheses × metrics).
- **Intervals:** delete-one-block jackknife, t with K − 1 degrees of freedom (95 % and 90 %). The thesis block bootstrap
  is reported next to them.
- **Four-way verdict** (per hypothesis and metric): *superior* (95 % interval above 0; *beyond margin* if above δ),
  *equivalent* (90 % interval inside (−δ, δ)), *non-inferior* (95 % lower bound above −δ), *inferior* (95 % interval
  below 0), otherwise *inconclusive*. A Holm-adjusted p-value above α overrides *superior* to "superior, not significant
  after Holm".
- **Comparisons with many references:** the reference is the best of its set on the test year, chosen after the results
  (conservative for the candidate), as in the thesis.

## Confirmatory tests (structure fixed now; candidate configs frozen at the end of stage C)

**Test year: 1405, months 1–7** (Farvardin–Mehr 1405), new places and a new year (training on the training blocks of
1402–1404). Only **driver** inputs (traffic, energy, context): there are no 1405 station data, and a station layer is an
outcome, not a cause, so it does not belong in a scenario model. Inventory and traffic of 1403 stand in for 1405
(as in the 1404 design). 1404 and the cross-validation on 1402–1403 are reported as secondary, descriptive results.

| name | comparison | reading |
|---|---|---|
| T1 | the graph model with the physical reach R vs the best SLX model with the same reach | four-way verdict |
| T2 | physics + GNN residual (P2) vs physics alone (P0): does learning add more than δ? | four-way verdict; "learning adds" only if *superior beyond margin* |
| T3 | fusion of the driver domains (traffic + energy) vs the better single driver domain | four-way verdict |
| T4 | the response half-distance of P2 and of the pure GNN lies inside [σ̂/2, 2σ̂] | inside / outside, with a bootstrap interval over the probed cells |

- **Physical reach R = 3σ̂.** σ̂ is the Gaussian width that the sector-yield operator (`forward_sectors`) selects on the
  training cells, over the grid 0–15 km. It is cross-checked, not tuned, against x₀ = τ·u with τ = 2.44 ± 0.68 h
  (Beirle & Wagner 2024, AMT) and the ERA5 wind.
- **P0** = a + b·log(B(h) + Σ_s β_s·(G_σ̂ ∗ E_s)), β_s ≥ 0, with road and buildings merged (they are collinear at σ̂).
  B(h) is a background that depends on elevation (Fioletov et al. 2025). **P2** = P0 + a GNN fitted on the residual of
  P0, with the context inputs orthogonalised on the emissions.
- **Hyper-parameters:** 50-trial random search per family on 1402–1403 (inner block CV, 4 inner folds). Search spaces:
  trees from Probst, Boulesteix & Bischl (2019, JMLR), GNN from You, Ying & Leskovec (2020, NeurIPS). They are written
  in the config before the search starts.

**Expected directions** (written now so that the opposite result is visible): T1 non-inferior or equivalent;
T2 not beyond margin; T3 equivalent; T4 outside for the pure GNN.

## Exploratory (no verdict)

CNN vs SAGE at the same reach; modality Shapley values and interactions; redundancy as a function of smoothing scale;
satellite-based diagnosis of the inventory (residual maps, Rey-Pommier et al. 2025); the other hotspot labels
(q = 0.80, 0.95); superpixels; elevation; the daily cloud-screened target; ModDrop.

## Known limitations

- One city; two development years and 1404 already seen.
- Hotspots sit in about two of the twelve blocks, so PR-AUC tests have little power; this is why Spearman is co-primary.
- 1405 has seven warm months only (weaker contrast than winter), stand-in inventory, no station data.
- Blocks are treated as exchangeable units; months are not independent and are not counted as units.

## Deviations

None yet.

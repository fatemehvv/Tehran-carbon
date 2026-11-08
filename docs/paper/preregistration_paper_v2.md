# Paper v2 (branch `paper_TC`): pre-registration of the confirmatory test

**Status: FINAL, 2026-11-08.** Draft written 2026-11-07 before any `paper2_*` run. Finalised after stage A with the three
numbers stage A measured, **before any stage B run and before the 1405 data are opened**. The push time of this commit on
GitHub is the registration time. After it, nothing here changes; a deviation becomes a new config version and is listed
under *Deviations*. Plan: `docs/paper/plan_q1_fa.md`; stage A results and assessment: `docs/paper/results_stageA/`,
`docs/paper/assessment_stageA_fa.md`.

These are **not thesis hypotheses**. The thesis verdicts stay as they are and are reported.

## What is known when this is finalised (stage A, 1402–1404)

- Against SLX models with the thesis windows (≤ 15 cells), the GNN is better in Spearman (+0.025 / +0.035).
- Against reach-matched SLX models, the GNN is not better: Spearman −0.001 / +0.010 (dev years), +0.004 on 1404. On
  1404, SLX-OLS with windows up to 27 cells ties the 5-layer GNN.
- The physical operator (sector yields, smoothed inventory) scores clearly lower (Spearman ≈ 0.85, PR-AUC ≈ 0.55).
- The target correlogram on 1402–1403 falls to zero at **24 km**; its effective sample size is about 5.
- Neural PR-AUC varies strongly between seeds (SD 0.03–0.075); Spearman does not (SD ≈ 0.005).
- PR-AUC has 5 valid blocks, so no sign-flip test of it can reach p < 0.031.
- The station leak was real; fixing it changed nothing.
- **Unknown:** everything about 1405, every stage B result (reach curve, tuned GNN).

## The three numbers of the draft, now fixed

| number | rule (draft) | value |
|---|---|---|
| margin δ | max(0.02, 2 × SD_seed) | **Spearman: δ = 0.02** (SD_seed 0.0077). PR-AUC: 0.073 (SD_seed 0.037): PR-AUC is descriptive only |
| power | block-level simulation | PR-AUC: 0 at every effect (5 valid blocks, smallest p 0.031). Spearman (block-mean approximation): below 0.5 at δ, labelled **low power**; the pooled sign-flip test used for the decision gave p ≈ 0.001 for effects of +0.025 in stage A |
| compute | Hyperband if the searches exceed 168 GPU-hours | timing pilot: about 1.2 GPU-hours per search → full random search, no Hyperband |

## Confirmatory test (one family, Holm over its tests)

**Test year: 1405, months 1–7** (Farvardin–Mehr 1405), new places and a new year: training on the training blocks of
1402–1404 (`split.test_year`, the design of H2c). Inventory and traffic of 1403 stand in for 1405. **Driver inputs only**
(traffic, energy, context), with no station layer.

**Primary metric: Spearman** of predicted and observed z over all domain cells, mean over months. PR-AUC and the other
metrics are reported as descriptive results.

| name | comparison | reading |
|---|---|---|
| **T1** | **GNN vs SLX at the same reach.** Candidate: SAGE, 4 layers (36 cells), gated fusion of traffic + energy + context, the hyper-parameters of the stage B search, 5 seeds averaged. Reference, fixed now: SLX-OLS with windows [1, 3, 9, 15, 18, 27, 36], the same inputs | four-way verdict with δ = 0.02 |
| **T3** | **Fusion of the drivers vs the better single driver.** Candidate: the T1 GNN. References: the same GNN with traffic only and with energy only; the better one on 1405 is chosen after the results | four-way verdict with δ = 0.02 |

- **Reach rule, fixed now.** R* = the zero crossing of the target correlogram on 1402–1403 = 24 km. The GNN depth is the
  smallest whose reach (9 cells per layer, mean cell size 0.84 km) covers R*: 4 layers = 36 cells ≈ 30 km. The reference
  sees exactly as far. Cross-check, not tuned: x₀ = τ·u, with τ = 2.44 ± 0.68 h (Beirle & Wagner 2024) and the ERA5 wind.
- **Tests.**
  - Superiority: exact block sign-flip test of the pooled difference, one-sided; ranks rank-normalised within each month.
  - Equivalence and non-inferiority: delete-one-block jackknife, t with K − 1 df (90 % and 95 % intervals).
  - Family α = 0.05, Holm over T1 and T3.
  - A *superior* reading whose Holm-adjusted p is ≥ 0.05 is reported as "superior, not significant after Holm".
- **Seeds.** The candidate is the 5-seed average, as in the thesis. The spread of single seeds is reported next to it.
  A verdict that holds for the average but not for most single seeds is reported as such.
- **Expected directions** (written now, so that the opposite result is visible):
  - T1 **equivalent** (or non-inferior);
  - T3 equivalent or inconclusive.

## Stage B (before 1405; nothing in it is confirmatory)

- **Skill against reach on 1402–1403** (`paper3_reach_*`, `python main.py paper reach`): SLX-OLS and SLX-GBM with nested
  windows up to 0–54 cells, SAGE with 1–6 layers, the no-graph network; driver inputs.
- **GNN hyper-parameters** (`paper3_tune_sage_v1`, `python main.py tune`):
  - 50-trial random search on 1402–1403, 4 inner block folds, Spearman.
  - The reach is not searched (rule above).
  - Search space from the GraphGym design space (You, Ying & Leskovec 2020): hidden size, encoder depth, dropouts,
    learning rate, weight decay, epochs.
  - The best values are written into the stage C configs as a new version.
  - The reference (OLS) has no hyper-parameters.
- **Descriptive reading of the T1 pair on 1402–1403**, untuned (`PB_T1_dev`).

## Exploratory (no verdict)

- the hotspot labels q = 0.80 / 0.95, superpixels, the daily cloud-screened target, elevation, stations;
- modality Shapley values and redundancy as a function of smoothing scale;
- impulse response of the final models;
- the physical operator;
- persistence (the observed maps of 1402–1404 scored on 1405).

## Known limitations

- One city (by choice: no second city is added). Two development years; 1404 has been seen.
- The effective sample size of the target is about 5, so tests have little power. "Equivalent" needs the jackknife interval
  inside ±0.02, which may not be reached ("inconclusive").
- 1405 has seven warm months only, stand-in inventory and traffic, and no station data.
- The authors saw, in stage A, that a reach-matched SLX ties the GNN. The reach rule, the reference and the candidate's
  architecture are fixed here before 1405 is opened. The reference's windows are fixed by the rule, not by the stage A
  results (the best stage A window on 1404 was 27 cells; the rule gives 36).

## Deviations from the draft of 2026-11-07 (made before any stage B run and before 1405)

1. **Primary metric:** Spearman alone, instead of PR-AUC and Spearman co-primary. Reason: PR-AUC cannot reach significance
   with 5 valid blocks, and its seed-to-seed SD makes its margin (0.073) uninformative. Family α = 0.05 instead of
   0.025 per metric.
2. **T2 (physics + GNN vs physics) and T4 (response half-distance) are exploratory, not confirmatory.** Reason: the
   physical operator alone is far behind (Spearman 0.85 vs ≈ 0.93), and the stage A result moved the paper's question to
   reach.
3. **Hyper-parameter search for the GNN only.** The reference (OLS) has none. The GBM of the reach curve keeps its defaults.
4. **The reach rule is now concrete** (24 km → 4 layers / 36 cells), and the T1 reference is a single model fixed in
   advance instead of "the best SLX with the same reach".
5. **Block-level sign-flip test (secondary): cell-weighted mean.** Two edge blocks hold 3 and 5 cells, where per-block
   Spearman means nothing.
6. **No second city** (decision of the author, 2026-11-08).

## Deviations after finalisation

7. **Stage B and the confirmatory test on 1405 were not carried out** (decision of the author, 2026-11-08: no new
   training runs). This covers the `paper3_*` runs, the 50-trial search and T1 / T3 on 1405. Their configs and code stay
   on the branch.
   - The paper reports stage A, which includes the held-out-year design on 1404 registered for the thesis.
   - It also reports one analysis-only command run on 2026-11-08: `python main.py paper reach` on the stage A runs
     (`paper2_base_v1`, `paper2_slx_reach_v1`, `paper2_rq2_mlp_v1`, `paper2_rq2_sage_v1`). It trains nothing, and its
     skill-against-reach curve is **exploratory**.
   - No verdict of the paper is called confirmatory beyond what stage A and the thesis registrations support. The paper
     states that the registered 1405 test and the hyper-parameter search were not run.

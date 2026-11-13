# Fusion paper (branch `paper_fusion`): pre-registration

**Status: FINAL, 2026-11-13.** Written before any `fus*` run and before any composite target, placebo layer or smoothed
target exists. It is frozen by the commit that adds the stage 1 code and configs, pushed before the first `fus*` run;
the push time on GitHub is the registration time. After that, nothing here changes: a deviation becomes a new config
version and is listed under *Deviations*. Plan and literature: `docs/paper/fusion_paper/novelty_plan_fa.md`,
`lit_search_fa.md`.

**Stages.** The server time available on 2026-11-13 is about 4 hours, so the runs come in two stages under this one
registration:
- **Stage 1** (code and configs in the registration commit): the placebo tests **F1, F2, C1, C2**.
  - Configs: `fus2_placebo_v1`, `fus2_placebo_early_v1`, `fus2_c1404_real_v1`, `fus2_c1404_context_v1`,
    `fus2_c1404_placebo_v1`.
  - Real-branch references on 1402 / 1403: the existing runs `paper2_rq1_ablation_v1` and
    `paper2_rq1_context_only_v1`.
- **Stage 2** (code and configs added later, without changing this text): the target-support tests **F3, F4** and the
  target versions T-all, T-calm, T-windy, T-σ.
  - M1–M3 are run before any stage 2 model.
  - No stage 2 result exists when stage 1 is read, and no stage 1 result changes any stage 2 rule.
- **Holm.** Holm over family F is applied once, after stage 2, to all six tests. Stage 1 verdicts are reported first
  without the adjustment and are labelled as such.
- **Analysis code.** The analysis code (placebo averaging, the contrasts of F3 and F4) reads stored predictions only. It
  can be added after the runs, as long as it implements the rules written here.

> **خلاصهٔ فارسی.** این سند پیش از هر اجرا می‌گوید چه چیزی آزموده می‌شود، با چه آماری، و هر نتیجه چطور خوانده
> می‌شود. دو پرسش اصلی دارد:
> - **(۱) دارونما:** آیا سود افزودن حوزهٔ انرژی (و ترافیک و ایستگاه‌ها) از اطلاعات خودش است یا فقط از شاخهٔ اضافه؟
>   مقایسه با یک «منبع ساختگی» هم‌شکل در همان معماری.
> - **(۲) تفکیک هدف:** آیا سود همجوشی وقتی هدف تیزتر است (روزهای آرام) بیشتر و وقتی درشت‌تر است کمتر است؟
>
> سال ۱۴۰۴ آزمون تأییدی است. معیار اصلی Spearman است با حاشیهٔ ۰٫۰۲. Holm روی هر خانوادهٔ آزمون اعمال می‌شود.
> هیچ مدل گرافی در کار نیست.

These are **not thesis hypotheses** and they do not overlap with the first paper (branch `paper_TC`): no model in this
study has a graph, a reach comparison or an SLX reference. The neighbourhood inputs are fixed at the thesis setting
(box means with half-widths 3 and 9 cells) and are never a factor of the design.

## What is known when this is written

From stage A of the first paper (`docs/paper/results_stageA/`; leak-free station layer, 5 seeds, monthly TROPOMI target,
graph-free gated network unless stated):

- **Fusion of all three domains vs energy (both with context), gated:**
  - PR-AUC +0.022 / +0.040 (1402 / 1403);
  - Spearman −0.021 / −0.022, negative for every one of the five seeds.
- **Traffic + energy vs energy:** Spearman −0.028 / −0.026.
- **Context only** (one branch, early fusion): PR-AUC 0.576 / 0.592, Spearman ≈ 0.91.
- **Energy with context:**
  - early: PR-AUC 0.621 / 0.676;
  - gated: PR-AUC 0.770 / 0.757, Spearman 0.918 / 0.927.
- **With equal inputs, the spread over early / late / gated** (mean 0.124 / 0.094 PR-AUC) is at least as large as the
  spread over single domains (0.106 / 0.103).
- **Redundancy:**
  - road and building CO₂ layers correlate at 0.58 per cell and at 0.998 after Gaussian smoothing with σ = 8 km;
  - the target correlogram crosses zero at 24 km; n_eff ≈ 4.9.
- **Lineage:** the residential, commercial and industrial CO₂ layers are allocated by GHSL residential volume, GHSL
  non-residential volume and OSM industrial area (thesis, section 3-4-5). These are layers that the context branch
  also contains.
- **Seed noise:**
  - seed SD of PR-AUC 0.03–0.075; of Spearman ≈ 0.005–0.008.
  - PR-AUC has 5 valid blocks (smallest possible p = 0.031).
  - The block sign-flip test was liberal for Spearman in a null calibration (5 of 20 below 0.05).
- **Unknown:** every result with a placebo branch, a composite target or a smoothed target; every graph-free result
  on 1404.

## Fixed design (all experiments)

| item | setting |
|---|---|
| network | the thesis RQ1 network (`paper2_rq1_ablation_v1`): one MLP encoder per branch (hidden 32, 2 layers, dropout 0.1), `backbone: none`, AdamW 3e-3, weight decay 1e-4, 300 epochs, MSE on z |
| inputs | branches as in the thesis; context is always a branch; box means at half-widths 3 and 9 cells for every spatial layer |
| validation | 12 blocks of 22 km, leave one block out, 2-km buffer; station layer rebuilt per fold without the stations near the test block (`split.fold_aware_stations`) |
| seeds | 5 (0–4); the prediction is the 5-seed mean; every seed is stored (`train.save_members`) and reported |
| years | 1402 and 1403 for development; **1404 held out** (train on the training blocks of 1402 + 1403, predict the held-out block of 1404; inventory and traffic of 1403 stand in for 1404) |
| primary metric | Spearman between predicted and observed z over the domain cells, mean over the time steps that have a target |
| descriptive metrics | PR-AUC of the top 10 % (and q = 0.80 / 0.95), the other thesis metrics |
| margin | δ = 0.02 (Spearman), the value of the first paper (max(0.02, 2 × SD_seed)) |

### Target versions (the second factor)

| name | definition |
|---|---|
| **T0** | the monthly TROPOMI target of the thesis (runs exist: `paper2_rq1_ablation_v1`, `paper2_rq1_context_only_v1`) |
| **T-all** | seasonal composite of all clear days (daily files `satellite/daily/Tehran_NO2_daily_{year}.tif`), cells with fewer than 3 valid days left out |
| **T-calm / T-windy** | within each Jalali season (months 1–3, 4–6, 7–9, 10–12), the clear days are ranked by the city-mean wind speed at 10 UTC (`weather/Tehran_wind10UTC_{year}.csv`); calm = lowest third, windy = highest third, the same number of days |
| **T-σ** | T0 smoothed with a Gaussian of σ = 2, 4, 8, 12 km. Test truth: smoothed over all domain cells. Training labels of each fold: normalised convolution over that fold's training cells only (`split.fold_aware_target`), so no test-block value enters a training label |

- **Placement.** A seasonal composite is placed on the middle month of its season (months 2, 5, 8, 11). The other
  months have no target (`data.target.min_valid_fraction`), so each year has 4 target maps.
- **Why seasonal.** Calm-day coverage of TROPOMI needs about 3–6 months (Liu et al., 2024, ACP 24:3717).

### Input sets and the placebo (the first factor)

- **Real branches:**
  - context only (one branch, early fusion);
  - energy;
  - traffic;
  - stations;
  - traffic + energy;
  - all three.
- **Placebo branch Ẽ, T̃, P̃.**
  - **Same as the real branch:** the number of layers, the marginal distributions, the spatial power spectrum, the
    cross-correlation of the domain's own layers, and the box means.
  - **Different:** the location is random.
  - **Construction:**
    - multivariate IAAFT surrogates (Schreiber & Schmitz, 1996) of each layer's annual map, on the full grid after
      mirror padding, with the same random phases for all layers of a domain;
    - monthly layers = surrogate × the city-wide monthly share of the real layer;
    - the station placebo is built from the all-station interpolated layer, so location information is destroyed by
      the randomisation.
  - **Realisations:** 5, numbered 0–4.

## Checks before any model is trained (`python main.py fusion support`, `python main.py fusion lineage`)

| check | rule | if it fails |
|---|---|---|
| **M1 composite pipeline** | Spearman(T-all, T0) over the domain ≥ 0.95 on the middle months, mean over months, both years | stop; fix the code; no model is run |
| **M2 manipulation** | in both years, T-calm is sharper than T-windy: zero crossing of the correlogram ≥ 2 km shorter **and** ρ(8–10 km) ≥ 0.05 lower | F3 is reported as **uninformative** (not as "no effect") and leaves the family; T-calm / T-windy runs are still made and reported descriptively |
| **M3 reliability** | split-half (odd / even days of each third) Spearman ≥ 0.90, mean over seasons, both years | switch to half-year composites (months 1–6, 7–12; placed on months 3 and 9) and repeat M2. This is the only allowed fallback |
| **M4 placebo** | over the 5 realisations, on the full grid where the surrogates are made: median \|Spearman(placebo layer, real layer)\| < 0.10; correlogram within ± 0.05 of the real layer at 0–20 km. Read from the stored datasets after the stage 1 runs, before any stage 1 verdict is read | reported; a realisation that fails is listed. If more than two fail, the placebo verdicts are labelled "placebo check failed" and the rotation-plus-shift placebo (180° rotation, random cyclic shift) is run as a new config version |
| **lineage audit** (descriptive) | block-CV R² of each energy and traffic layer from the context layers | none; reported as a table |

## Confirmatory family F (development years; Holm over the 6 tests, α = 0.05)

Candidate and reference are fixed now. Δ = Spearman(candidate) − Spearman(reference).

| test | candidate | reference | target | years | expected (written now) |
|---|---|---|---|---|---|
| **F1a, F1b** information of energy | context + energy, gated | context + Ẽ, gated (mean over the 5 realisations) | T0 | 1402, 1403 | Δ ≥ 0 but inside ±δ (equivalent or inconclusive) |
| **F2a, F2b** information of traffic + stations given energy | context + energy + traffic + stations, gated | context + energy + T̃ + P̃, gated | T0 | 1402, 1403 | equivalent or inferior |
| **F3** target support | D = Δ_calm − Δ_windy, with Δ = Spearman(all three, gated) − Spearman(energy, gated) | | T-calm, T-windy | pooled (mean of the two years) | D > 0, small; probably inconclusive (low power) |
| **F4** coarsening | Δ_rel(T0) − Δ_rel(T-8 km), with Δ_rel = Δ / (1 − Spearman(energy, gated)) and Δ as in F3 | | T0, T-8 | pooled | > 0 |

- **F1 and F2, inference** (the paper-evaluation code of the first paper, `python main.py paper evaluate`):
  - exact block sign-flip test of the pooled difference, one-sided, with predictions rank-normalised within each time
    step;
  - delete-one-block jackknife, t with K − 1 df;
  - four-way verdict with δ = 0.02.
  - For the placebo reference, every per-block quantity is averaged over the 5 realisations before the test.
- **F3 and F4, inference** (`python main.py fusion did`):
  - D and the Δ_rel contrast are computed per block (cell-weighted);
  - delete-one-block jackknife recomputes all terms;
  - exact block sign-flip of the per-block contrast, one-sided;
  - the same four-way verdict on the contrast with δ = 0.02.
- **Holm** over the six one-sided sign-flip p-values. A *superior* reading whose Holm-adjusted p is ≥ 0.05 is reported
  as "superior, not significant after Holm".
- **If M2 fails,** F3 leaves the family and Holm is over 5 tests.

## Confirmatory family C (held-out year 1404; Holm over 2 tests)

| test | comparison | expected |
|---|---|---|
| **C1** | F1 on 1404: context + energy vs context + Ẽ, gated, T0 | equivalent or inconclusive |
| **C2** | F2 on 1404 | equivalent or inferior |

- **T-calm and T-windy on 1404 are descriptive only,** and only if the 1404 daily NO₂ and wind files are added
  (`gee/GEE_daily_NO2_wind_L0.js`) before the first 1404 run.

## Secondary estimates (reported with intervals, no verdict)

- **Architecture effect:** Spearman(context + Ẽ, gated) − Spearman(context only); likewise for T̃ and P̃; and the same
  with early and late fusion.
- PR-AUC versions of F1–F4 and C1–C2.
- The fusion gain Δ for every target version: T0, T-all, T-calm, T-windy, T-2 … T-12 km.

## Exploratory (no verdict)

- early and late fusion for every comparison;
- traffic-only and stations-only branches with their placebos;
- context + a copy of context (pure capacity);
- energy residualised on context;
- the pre-training scale-resolved redundancy index (unique R² of each domain at the target's correlation scale; RV
  coefficient) against Δ over all target versions. Exploratory because such indices have been unreliable elsewhere
  (Zhang et al., 2025);
- seed stability of the gate weights (prior art: Vulli et al., 2026);
- box means up to 27 cells as a sensitivity of the neighbourhood input (not a design factor);
- placebo-adjusted modality Shapley values; the "fusion as insurance" regret table.

## Seeds

- Every confirmatory candidate and reference is a 5-seed mean.
- **Single seeds:**
  - The difference of every candidate seed against the reference is reported next to the verdict.
  - For placebo references, the difference is taken against the matching realisation.
- A verdict that holds for the mean but not for most single seeds is reported as such.

## Known limitations

- One city; one target product; the target is standardised over the domain in each time step.
- n_eff ≈ 5: tests have low power. The sign-flip test is liberal for Spearman, so a significant Spearman p is read
  with caution.
- Seasonal composites give 4 target maps per year: F3 rests on fewer time steps than F1, F2 and F4.
- Wind at 10 m (city mean, ERA5-Land) is a proxy for the plume transport wind.
- Calm and windy days may also differ in chemistry and boundary-layer depth. The within-season split controls the
  season only.
- The placebo keeps the spectrum and the distribution of a layer but not every higher-order spatial feature. A smooth
  random field can itself act as a spatial basis (Behrens & Viscarra Rossel, 2020); this is reported if seen.
- 1404 uses the inventory and traffic of 1403.

## Deviations

None yet.

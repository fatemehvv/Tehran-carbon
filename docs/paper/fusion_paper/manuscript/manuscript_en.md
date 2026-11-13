# Do urban data sources add information to fusion models? A pre-registered placebo test for satellite NO₂ mapping in Tehran

**Fatemeh Vafaeian¹, Vahid Khatibi¹\***

¹ [Affiliation to be completed]  
\* Corresponding author: [e-mail to be completed]

*Draft manuscript, 13 November 2026. All numbers come from the runs on branch `paper_fusion` of the project repository
(`docs/paper/fusion_paper/results_stage1`, `results_stage2`, `results_tests`, `results_check`, `results_v2`). Items in
[brackets] are to be completed before submission. Revised after two internal reviews (`docs/paper/fusion_paper/review/`).*

---

## Highlights

- Adding a data source to a fusion network also adds inputs and parameters.
- Placebo branches move a source's layers to a random location in the same network.
- A relocated, information-free input lowered map skill by 0.01–0.03.
- Energy layers added almost nothing over the urban-form data they derive from.
- Both placebo constructions failed their registered texture check.

---

## Abstract

Fusing urban data sources is a common route to better maps of emissions and air pollution, and the value of a source is
usually judged by ablation: the source is added to or removed from a multi-branch network. Adding a source, however,
also adds inputs and parameters. We test this confound with placebo branches, the layers of a source moved to a random
location by IAAFT surrogates or by rotation with a cyclic shift, trained in the same network. The task is the
intra-urban pattern of monthly TROPOMI NO₂ in Tehran on 2,436 cells of about 0.75 × 0.93 km, fused from traffic,
energy-inventory and station data with urban-form context, under a pre-registered protocol with 22-km spatial blocks, a
held-out year, five seeds and five placebo realisations. An information-free placebo input lowered the Spearman
correlation of the map by 0.01–0.03. Real energy layers scored 0.018–0.038 above their placebos but only −0.001 to
+0.013 above the urban-form context alone. Gradient boosting on the context predicts three of the four energy layers
with cross-validated R² of 0.89–0.98. Most of the real-minus-placebo contrast is therefore the cost of the placebo
input, not information in the energy layers. Traffic and station layers scored below their placebos in all 30 seed
comparisons. Neither placebo construction passed the registered texture check, because the single urban core of a
bounded map cannot be moved without being reshaped or cut. Fusion did not improve the map on targets smoothed over 2–12
km, and calm-day composites were not sharper than windy-day ones. Ablation gains of the size reported for urban data
fusion need a placebo control, and the placebo itself needs checking.

**Keywords:** data fusion; multimodal learning; ablation; placebo control; surrogate data; TROPOMI; NO₂; urban
emissions; spatial cross-validation; Tehran

---

## 1. Introduction

Maps of where a city burns fuel guide emission inventories and mitigation, and satellite NO₂ has become a practical
tracer of urban combustion (Beirle et al., 2019; Goldberg et al., 2024). Models that predict the NO₂ pattern from urban
data can screen cities for combustion hotspots and test inventories. Many such models fuse several data sources —
traffic models, emission inventories, monitoring stations, satellite imagery, points of interest — in networks with one
branch per source. Examples come from carbon-emission mapping (Zeng et al., 2025) and crop-yield prediction (Mena et
al., 2025), following the broader practice of multimodal and cross-domain fusion (Zheng, 2015; Baltrušaitis et al.,
2019).

Whether fusion helps is usually judged by **ablation**. A source is removed from, or added to, the network, and the
change in skill is credited to that source. Adding a source, however, also adds input columns, parameters and, in
gated or attention-based fusion, a new competitor for the weights. A change in skill can therefore come from the
information in the source or from the added input itself. The machine-learning literature has shown that this is not
neutral:
- multimodal networks can be worse than their best unimodal part because they overfit (Wang et al., 2020);
- modalities compete during joint training (Huang et al., 2022);
- the cross-modal interactions credited to a model are often absent (Hessel & Lee, 2020).

Modality importance is usually measured by permuting a modality at test time (Gat et al., 2021) or by Shapley values
over subsets (Hu et al., 2022). Both measure reliance on a modality, not the value of its information over an input of
the same shape.

**Placebo inputs have a long history.**
- *Feature selection:* comparing a feature with permuted copies of itself is the idea of shadow features (Kursa &
  Rudnicki, 2010); model-X knockoffs formalise copies that keep the dependence on the other features (Candès et al.,
  2018).
- *Spatial statistics:* surrogate maps that keep autocorrelation but not location are standard null models:
  - torus translations (Harms et al., 2001) and random shifts (Mrkvička et al., 2021);
  - spin tests (Alexander-Bloch et al., 2018);
  - spectral surrogates (Schreiber & Schmitz, 1996; Prichard & Theiler, 1994; Wagner & Dray, 2015; Burt et al., 2020).

  These null models are known to struggle with non-stationary maps and boundaries (Markello & Misic, 2021).
- *Species distribution modelling:*
  - pseudo-predictors made from paintings were rated as good as real environmental predictors (Fourcade et al., 2018);
  - shifted and rotated environmental layers have been used to refit models and estimate false positives (Ridder et
    al., 2024);
  - spatially structured but meaningless predictors can carry skill (Meyer et al., 2019; Behrens & Viscarra Rossel,
    2020).

What we did not find is such a surrogate used as a **whole placebo branch** in a multi-source fusion network: an input
of the same shape as a real source, trained in the same network, whose comparison with the real source separates the
information in the source from the effect of adding the input.

**Does fusion pay when the target is sharper or coarser?** If two sources differ only at scales finer than the target
resolves, the target cannot reward their difference. Correlations rise as data are aggregated (Gehlke & Biehl, 1934),
and the choice of support changes model skill and predictor importance (Gotway & Young, 2002; Friedrichs-Manthey et
al., 2020; Moudrý et al., 2023), including for air pollution in Tehran (Zhalehdoost & Taleai, 2025).
- In Tehran, the road and building emission layers of the inventory correlate at 0.58 in single cells but at 0.998
  once smoothed over 8 km, while the monthly TROPOMI pattern varies over tens of kilometres (Vafaeian & Khatibi, in
  review).
- That companion study found that fusing these sources did not beat the best single source.
- Low-wind TROPOMI observations have been selected to reduce the smearing of NO₂ by wind (Goldberg et al., 2024), and
  calm-wind coverage of a city needs several months of data (Liu et al., 2024).

We ask three questions, in a pre-registered design:
1. **Information.** Does each urban data source score better than a placebo input of the same shape?
2. **Added input.** How large is the effect of adding an information-free input, compared with the effect of a real
   source?
3. **Target support.** Does the value of fusion change when the satellite target is sharper (calm days) or coarser
   (smoothed)?

## 2. Study area and data

### 2.1 Grid, years and target

The grid covers 50.85–51.90°E and 35.35–35.95°N with 72 × 126 cells of 1/120° (about 0.75 × 0.93 km). The label domain
holds the 2,436 cells of the three central counties (Tehran, Rey, Shemiranat) that the emission inventory assigns any
CO₂.

We use three Iranian (Solar Hijri) years: 1402 (March 2023 – March 2024) and 1403 for development, and 1404 (March 2025 –
March 2026) as a held-out year.

The target is the monthly mean tropospheric NO₂ column of the TROPOMI offline level-3 product in Google Earth Engine
(`COPERNICUS/S5P/OFFL/L3_NO2`, band `tropospheric_NO2_column_number_density`), averaged on the grid. The TROPOMI
footprint (3.5 × 5.5 km at nadir) is coarser than the grid, so the effective resolution of the target is several
kilometres. The target is:
- log-transformed, with non-positive values set to a floor of 10⁻⁷ mol m⁻² before the logarithm;
- standardised over the domain in each month (z), so that only the spatial pattern is kept. The standardisation uses all
  domain cells, as in the companion study; it changes no rank metric.

The target is smooth. Its correlogram crosses zero at about 24 km, and its effective number of independent cells is
about 5 per month, computed as n² / Σᵢⱼ ρ(dᵢⱼ) from the truncated correlogram (Griffith, 2005; Vafaeian & Khatibi,
in review).

### 2.2 Data sources

| Source | Layers | Inputs per cell |
|---|---|---|
| **Traffic** | road-transport CO₂ of a bottom-up inventory; road length in four classes; peak-hour vehicle-kilometres and volume-to-capacity ratio of a traffic assignment | 21 |
| **Energy** | residential, commercial, industrial and power-sector CO₂ of the same inventory; monthly heating degree-days | 13 |
| **Stations** | CO of the municipal monitoring network, interpolated by inverse distance | 3 |
| **Context** (always present) | GHSL population, total and non-residential building volume, VIIRS night lights, industrial land share, ERA5 temperature and wind | 17 |

Every spatial layer enters with its value in the cell and its means over square windows of half-width 3 and 9 cells (up
to about 7.6 km).
- **Station layer.** CO is the only pollutant measured in both years that is not the target species. The layer is
  rebuilt for each fold without the stations within 2 km of the test block, which avoids pre-processing leakage (Kapoor
  & Narayanan, 2023).
- **1404.** For 1404, the inventory and traffic layers of 1403 stand in. The held-out year therefore tests transfer to a
  new target year with the previous year's layers, not with new source data.

**Shared lineage.** The sources are not independent.
- The residential and commercial CO₂ of the inventory is allocated in proportion to GHSL residential and non-residential
  building volume. Industrial CO₂ is allocated by OSM industrial area. The context source contains these same layers.
- The trips of the traffic model are distributed by GHSL population and non-residential volume.
- The station layer is a smooth interpolation of a few points.

Emission inventories inherit the spatial pattern of their allocation proxies (Geng et al., 2017; Gately & Hutyra,
2017; Mu et al., 2022); here the proxies are also given to the model as a separate source.

## 3. Methods

### 3.1 Fusion network

Every model is the fusion network of the companion study, without a graph:
- **Encoders:** one encoder per source, a two-layer MLP with 32 hidden units and dropout 0.1.
- **Fusion:** either a softmax gate over the encoders for every cell and month (**gated fusion**; Arevalo et al.,
  2017), or concatenation of all inputs into one encoder (**early fusion**).
- **Head:** a linear head predicts z.
- **Training:**
  - AdamW, learning rate 3 × 10⁻³, weight decay 10⁻⁴, 300 epochs, mean squared error;
  - all hyper-parameters fixed before the study;
  - five seeds, with the prediction the mean of the five, as in the companion study.
- **Context:** context is always a branch, so a model with "energy" has two branches (energy, context), and the
  context-only model has one.

### 3.2 Placebo branches

A placebo of a source keeps what a branch can exploit without information on where the city is: the values, the
spatial texture and the correlation between the layers of the source. It loses the location. We used two
constructions.

**Placebo A (registered first): IAAFT surrogates.**
- Each layer's annual map is mirror-padded to twice its size, so the field is periodic without edge jumps.
- Multivariate IAAFT surrogates are generated (Schreiber & Schmitz, 1996):
  - the same random phase field is added to the Fourier phases of every layer of a source, which keeps their
    cross-spectra;
  - 100 iterations alternate between restoring the amplitude spectrum and restoring the exact values by rank.
- The original window is cut out, and its values are set by rank to the real values.
- Monthly layers are the surrogate times the city-wide monthly factor of the real layer.

**Placebo B (the registered fallback): rotation and cyclic shift.**
- Every layer of a source is rotated by 180° and shifted cyclically, with one random offset for all months, years and
  layers of a realisation: between one quarter and three quarters of the grid in each direction.
- It keeps the values, the monthly cycle and the correlation between the layers of the source exactly, and the texture
  everywhere except at the wrap-around seams.

**For both constructions:**
- City-wide layers (degree-days) are copied unchanged, so a placebo branch has as many inputs as the real one.
- A placebo depends on the realisation, not on the year: it stays in place across years, as real static layers do.
- We use five realisations. Every placebo comparison averages scores over the five, also inside the jackknife and inside
  the randomisation test.
- Placebos are built on the full grid. The station placebo is built from the all-station layer, and the relocation
  removes any information on the test block. Both constructions break the dependence of a source on the context
  layers as well as on the target; Section 5.1 discusses what this implies.

Fig. 1 shows a real layer and two realisations of each construction.

![Figure 1](../figures/fig1_placebo_maps.png)

**Fig. 1.** The residential CO₂ layer of the inventory (log scale, annual mean; one grey scale for all panels; white =
zero) and two realisations of each placebo construction. (a) Real layer; (b, c) placebo A, IAAFT surrogates,
realisations 0 and 1; (d, e) placebo B, rotation by 180° and cyclic shift, realisations 0 and 1. Blue outline: label
domain.

### 3.3 Target versions

**Composites by wind.**
- The daily TROPOMI fields (cloud fraction < 0.3) of each Jalali season are ranked by the city-mean wind speed at
  10 UTC (ERA5-Land).
- Composites:
  - **calm** = mean of the calmest third of the clear days;
  - **windy** = the windiest third, the same number of days;
  - **all** = every clear day.
- A composite is placed on the middle month of its season, which gives 4 target maps per year.
- Seasons rather than months were registered because calm-day coverage needs several months of data (Liu et al., 2024).
- Each third held 15–29 days.

**Smoothed targets.**
- The monthly target is smoothed with Gaussian kernels of σ = 2, 4, 8 and 12 km by normalised convolution over the
  domain, then standardised again.
- To avoid leakage, the training labels of every fold are smoothed over that fold's training cells only. The test
  truth is the field smoothed over all cells. No value of a test block or its buffer enters a training label.

### 3.4 Validation and metrics

**Validation.**
- The domain is divided into 22-km blocks (12 blocks). Each block is held out once, and training cells within 2 km of
  it are removed (Roberts et al., 2017).
- For 1404, every fold trains on the training blocks of 1402 and 1403 and predicts the held-out block of 1404: held-out
  places in a held-out year, with the 1403 source layers (Meyer et al., 2018).
- The 2-km buffer is short compared with the correlation length of the target, so absolute skill reflects
  interpolation within the city; the comparisons share the folds and the buffer.

**Metrics.**
- **Primary:** Spearman correlation between predicted and observed z over the domain, averaged over the time steps that
  have a target.
- **Descriptive:** PR-AUC of the monthly top 10 % (hotspots), computed as average precision per month and averaged
  over months (chance level 0.10). It rests on very few independent units and varies strongly between seeds.

### 3.5 Inference

A difference between two models is computed on identical out-of-fold predictions.

**Each comparison gets:**
- **Interval:** a delete-one-block jackknife with K − 1 degrees of freedom. The models are not refitted for the
  jackknife; the score is recomputed without each block.
- **Randomisation test:** an exact block sign-flip test (2¹² assignments). The predictions of the two models are
  rank-normalised within each month and swapped block by block, and the domain-wide difference is recomputed.
- **Verdict** with a margin of δ = 0.02 (Lakens, 2017), taken from the jackknife intervals and unadjusted for
  multiplicity:
  - superior: the 95 % interval lies above 0;
  - equivalent: the 90 % interval lies inside ±δ;
  - non-inferior: the 95 % lower bound lies above −δ;
  - inferior: the 95 % interval lies below 0;
  - inconclusive: otherwise.
- **Holm's adjustment** (Holm, 1979) of the one-sided sign-flip p-values within each family.

The verdict and the p-value answer different questions and can disagree. The margin is the one registered in the
companion study: the larger of 0.02 and twice the seed-to-seed standard deviation of Spearman (0.008).

**Caveats.**
- The blocks are 22 km wide while the target decorrelates at about 24 km, so neighbouring blocks are dependent. The
  intervals and tests are therefore approximate.
- A null calibration in the companion study found the sign-flip test liberal for Spearman (5 of 20 null comparisons
  below 0.05). Significant Spearman p-values are read with caution.

**Placebo against placebo.** As an empirical null that includes the variance between realisations, the scores of the
five realisations of each placebo are compared with each other. The real model's rank among them is reported.

**Contrasts between target versions** (F3, F4) are computed per block and pooled over the two development years:
- jackknife over blocks;
- sign-flip of the cell-weighted per-block contrasts.

Per-block Spearman values are a different estimand from the domain-wide Spearman of the other tests.

**Seeds.** Seed i of the candidate is compared with realisation i of the placebo (or with the reference ensemble).
These counts describe consistency over seeds and years. They are not independent replications.

### 3.6 Pre-registration

The protocol was registered before any run of this study (`docs/paper/fusion_paper/preregistration_fusion_v1.md`,
commit `9e28ddc`, pushed to the public repository on 13 November 2026) [to be deposited on OSF with a time stamp]. It
was written after the companion study on the same data, network and target, so its expected directions were informed
by that study's finding that fusion did not beat the best single source.

**Family F** (1402, 1403; Holm over F1a, F1b, F2a, F2b, F3, F4, separately for each placebo construction):
- **F1:** context + energy vs context + placebo energy.
- **F2:** context + energy + traffic + stations vs context + energy + placebo traffic + placebo stations. F2 tests
  traffic and stations jointly.
- **F3:** the fusion gain on calm minus windy composites.
- **F4:** the relative fusion gain on the native target minus that on the target smoothed with 8 km.
- The fusion gain is Spearman(all three sources, gated) − Spearman(energy, gated). The relative gain divides it by
  1 − Spearman(energy).

**Family C** (1404; Holm over C1 and C2, separately for each construction): C1 and C2 repeat F1 and F2 on the
held-out year.

**Checks:**
- **Before the stage-2 models:**
  - **M1:** the all-day composite matches the monthly target (Spearman ≥ 0.95).
  - **M2:** calm composites are sharper than windy ones: zero crossing of the correlogram ≥ 2 km shorter **and**
    correlation at 8–10 km ≥ 0.05 lower, in both years.
  - **M3:** split-half reliability ≥ 0.90.
  - If M2 failed, F3 was to be reported as uninformative and removed from its family.
- **Before the placebo verdicts are read — M4:**
  - Over the five realisations, on the full grid, the median |Spearman| between placebo and real layer must be < 0.10
    (location removed).
  - The correlogram must stay within ±0.05 of the real one at 0–20 km (texture kept).
  - If more than two realisations failed, the placebo verdicts were to be labelled "placebo check failed" and placebo B
    run as a new version.
- **Lineage audit** (descriptive): block-CV R² of each energy, traffic and station layer from the context layers, with
  OLS and gradient boosting.

**Expected directions, written before the runs:**
- F1: equivalent or inconclusive.
- F2: equivalent or inferior.
- F3: positive but probably inconclusive.
- F4: positive.

**Contingencies and deviations.**
- *Registered contingencies:*
  - M2 failed, so F3 was removed from family F;
  - M4 failed for placebo A, so placebo B was run.
- *Not registered:*
  - Placebo B also failed M4, and the protocol did not say what to do then. Reporting both constructions side by side
    is a post-hoc decision.
  - The operational form of M4 (medians over the layers of a realisation) was fixed in code before the check was run.
  - In F4, edge blocks with a reference Spearman of 1 were dropped from the sign-flip test, which therefore used fewer
    than 2¹² assignments.
- *Requested by reviewers (post hoc):*
  - the placebo-versus-placebo null;
  - M4 computed on the label domain;
  - the absolute-skill table;
  - the trivial spatial baselines.
- *Secondary estimates:* the added-input estimates (Section 4.3) and the early-fusion comparisons were registered as
  secondary estimates without a verdict.

## 4. Results

### 4.1 Absolute skill

**Table 1.** Absolute skill of every model, out of fold (Spearman ρ over the label domain, averaged over months; PR-AUC
of the top-decile hotspots). Five-seed ensembles; placebo models averaged over five realisations. Entries marked [R]
come from `python main.py fusion review` and are to be filled in.

| Model | ρ 1402 | ρ 1403 | ρ 1404 | PR-AUC 1402 | PR-AUC 1403 | PR-AUC 1404 |
|---|---|---|---|---|---|---|
| *Baselines without training* | | | | | | |
| Best single context layer | [R] | [R] | [R] | [R] | [R] | [R] |
| Minus distance to the population-weighted centre | [R] | [R] | [R] | [R] | [R] | [R] |
| IDW of the target (companion study) | 0.678 | 0.689 | – | 0.341 | 0.374 | – |
| GP on coordinates (companion study) | 0.861 | 0.808 | – | 0.718 | 0.610 | – |
| *Fusion networks* | | | | | | |
| Context only | 0.914 | 0.914 | 0.912 | 0.576 | 0.592 | 0.637 |
| Context + placebo energy, gated, A / B | 0.893 / 0.901 | 0.889 / 0.901 | 0.886 / 0.887 | 0.684 / 0.688 | 0.675 / 0.666 | 0.644 / 0.642 |
| Context + energy, gated | 0.918 | 0.927 | 0.911 | 0.770 | 0.757 | 0.658 |
| Context + placebo energy, early, A / B | 0.895 / 0.902 | 0.891 / 0.899 | – | 0.625 / 0.672 | 0.681 / 0.686 | – |
| Context + energy, early | 0.923 | 0.928 | – | 0.621 | 0.676 | – |
| Context + energy + placebo traffic and stations, gated, A / B | 0.912 / 0.912 | 0.918 / 0.918 | 0.901 / 0.904 | 0.714 / 0.696 | 0.772 / 0.755 | 0.686 / 0.675 |
| Context + energy + traffic + stations, gated | 0.898 | 0.906 | 0.883 | 0.792 | 0.797 | 0.769 |

All fusion networks ranked the cells with Spearman 0.88–0.93. The context-only model already reached 0.91, above a
Gaussian process on the coordinates (0.81–0.86, same blocks; Vafaeian & Khatibi, in review). The differences that the
tests below are about are therefore a few hundredths on top of a high common level. PR-AUC varied far more between
models (0.58–0.80) than Spearman did. With few independent hotspot areas, it is the less stable metric. [Text on the
trivial baselines to be added after the review run.]

### 4.2 Information beyond the placebo

Table 2 and Fig. 2 give the tests of information; Fig. 2 also shows the effect of the added input itself (Section 4.3).

**Table 2.** Real source minus placebo branch (gated fusion; Spearman; five-seed ensembles, placebos averaged over five
realisations), for placebo A (IAAFT) and placebo B (rotation and shift). Intervals are delete-one-block jackknife 95 %
intervals. p is the one-sided block sign-flip p-value after Holm adjustment within the family. Both constructions
failed check M4 (Section 4.5).

| Test | Comparison | Year | Placebo | Δ | 95 % interval | Seeds > 0 | Verdict | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| F1 | energy vs placebo energy | 1402 | A | +0.025 | −0.010, +0.061 | 4 / 5 | non-inferior | 0.10 |
| | | | B | +0.018 | −0.011, +0.047 | 4 / 5 | non-inferior | 0.20 |
| F1 | | 1403 | A | +0.038 | +0.002, +0.075 | 5 / 5 | superior, not significant after Holm | 0.056 |
| | | | B | +0.026 | −0.011, +0.063 | 5 / 5 | non-inferior | 0.037 |
| C1 | | 1404 | A | +0.025 | −0.004, +0.055 | 4 / 5 | non-inferior | 0.082 |
| | | | B | +0.024 | −0.009, +0.056 | 4 / 5 | non-inferior | 0.030 |
| F2 | traffic + stations vs their placebos, energy present | 1402 | A | −0.014 | −0.078, +0.049 | 0 / 5 | inconclusive | 1.0 |
| | | | B | −0.015 | −0.086, +0.057 | 0 / 5 | inconclusive | 1.0 |
| F2 | | 1403 | A | −0.012 | −0.072, +0.047 | 0 / 5 | inconclusive | 1.0 |
| | | | B | −0.013 | −0.080, +0.054 | 0 / 5 | inconclusive | 1.0 |
| C2 | | 1404 | A | −0.018 | −0.111, +0.075 | 0 / 5 | inconclusive | 0.68 |
| | | | B | −0.021 | −0.125, +0.084 | 0 / 5 | inconclusive | 0.71 |

**Energy.** The model with the real energy layers scored above the model with their placebo in every year and with both
constructions, by 0.018–0.038, and in 26 of 30 seed pairs.
- The jackknife intervals include zero in five of the six comparisons, and the verdicts are non-inferior. The 1403
  comparison with placebo A is superior by its interval but not significant after Holm adjustment.
- With placebo B, the randomisation tests of 1403 and 1404 remain below 0.05 after Holm adjustment; the test is liberal
  for Spearman (Section 3.5).

**What the energy contrast measures.** The contrast splits into two parts:

real − placebo = (real − context only) + (context only − placebo).

| Year | Real − context only | Context only − placebo, A / B | Real − placebo, A / B |
|---|---|---|---|
| 1402 | +0.004 | +0.021 / +0.014 | +0.025 / +0.018 |
| 1403 | +0.013 | +0.025 / +0.013 | +0.038 / +0.026 |
| 1404 | −0.001 | +0.027 / +0.025 | +0.025 / +0.024 |

Most of the contrast is the cost of the placebo input, not a gain from the energy layers. The real energy layers improved
the ranking over context alone by at most 0.013.

Two readings fit these numbers:
1. The energy layers carry little information beyond the context, and any added input without information costs some
   skill (Section 4.3); the real layers avoid that cost because they agree with the context.
2. The placebo is not a neutral input. Both constructions break the dependence of the energy layers on the context
   (Table 5), so the placebo branch sends a signal that conflicts with the context branch. The cost would then be
   specific to the placebo construction.

These data cannot separate the two. A conditional placebo that relocates only the part of each layer not explained by
the context would (Section 5.4). The result that holds under both readings is the small gain over context alone.

**Spread over realisations.** [Placeholder for the placebo-against-placebo null: the spread of the five realisations and
the rank of the real-source model among them, from `fusion_review_null.csv`.]

**Traffic and stations.** Given energy, the real traffic and station layers scored below their placebos in every
year, with both constructions and in all 30 seed pairs. The intervals are wide and the comparisons inconclusive. Still, no
year, construction or seed points to information that a relocated copy of the same layers does not also provide.

**PR-AUC** (descriptive). The real sources scored higher than their placebos:
- energy +0.08 to +0.09 in 1402–1403 and +0.01 to +0.02 in 1404;
- traffic and stations +0.03 to +0.10.

These numbers are smaller than the spread over placebo realisations: the placebo-energy PR-AUC ranged from 0.52 to
0.85 (A) and 0.61 to 0.74 (B) in 1403.

![Figure 2](../figures/fig2_information_architecture.png)

**Fig. 2.** Information and added input. Blue: energy minus placebo energy. Orange: traffic + stations minus their
placebos, energy present. Grey: placebo-energy model minus context only (the effect of an added input without
information; Section 4.3). Spearman (a, primary) and PR-AUC (b, descriptive) for 1402, 1403 and the held-out year 1404.
Open markers: placebo A; filled markers: placebo B. Lines: jackknife 95 % intervals; band: equivalence margin ±0.02.
Both placebo constructions failed check M4.

### 4.3 The effect of an added input without information

The grey rows of Fig. 2 show these estimates.

**Table 3.** Placebo-energy model minus the context-only model (secondary estimates, no verdict), for placebo A and B.
Spearman with 95 % jackknife intervals; PR-AUC point estimates.

| Fusion | Metric | Placebo | 1402 | 1403 | 1404 |
|---|---|---|---|---|---|
| gated | Spearman | A | −0.021 (−0.049, +0.007) | −0.025 (−0.048, −0.002) | −0.027 (−0.053, −0.001) |
| | | B | −0.014 (−0.035, +0.008) | −0.013 (−0.035, +0.009) | −0.025 (−0.051, +0.001) |
| early | Spearman | A | −0.019 (−0.045, +0.006) | −0.023 (−0.062, +0.016) | – |
| | | B | −0.012 (−0.038, +0.014) | −0.015 (−0.045, +0.015) | – |
| gated | PR-AUC | A | +0.108 | +0.084 | +0.007 |
| | | B | +0.112 | +0.074 | +0.005 |
| early | PR-AUC | A | +0.049 | +0.089 | – |
| | | B | +0.096 | +0.094 | – |

**Ranking.** An added input without information lowered the Spearman correlation of the map by 0.012–0.027.
- It did so with both constructions and both fusion types, so the effect is not specific to the gate.
- Its size is similar to the gain of the real energy layers over context alone (+0.004, +0.013, −0.001).

**Hotspots.** The same input raised hotspot PR-AUC in the development years by 0.05–0.11.
- The real energy layers raised it, relative to context alone, by +0.19, +0.17 and +0.02.
- Part of the PR-AUC gain credited to a real source in an ablation table is therefore also obtained with relocated
  layers.
- The cause is not identified here. PR-AUC rests on a few hotspot areas, and the spread over realisations is large
  (Section 4.2).

**Early fusion.** With early fusion, real energy did not beat placebo energy in PR-AUC (A: −0.004 and −0.005; B: −0.052
and −0.010), while it kept a Spearman advantage of +0.021 to +0.037.

### 4.4 Target support

**Checks.**
- **M1 passed:** the composite of all clear days matched the monthly target (Spearman 0.98 in both years).
- **M3 passed:** split-half reliability was 0.92–0.96.
- **M2 failed:** calm-day composites were not sharper than windy-day composites (Fig. 3a).
  - The correlogram crossed zero at 23.6 km (calm) and 24.9 km (windy) in 1402, and at 23.9 and 24.7 km in 1403.
  - The correlation at 8–10 km was 0.753 against 0.756, and 0.754 against 0.776.
- By the registered rule, **F3 is uninformative** (D = −0.004, 95 % interval −0.034 to +0.025) and was removed from the
  Holm family.

**Table 4.** Fusion gain, Spearman(all three sources) − Spearman(energy), gated fusion, on every target version.
Blank cells were not run: for 1404 only the native and the 8-km targets were registered.

| Target | 1402 | 1403 | 1404 |
|---|---|---|---|
| monthly (native) | −0.021 | −0.022 | −0.028 |
| seasonal, all clear days | −0.017 | −0.016 | |
| seasonal, calm third | −0.026 | −0.015 | |
| seasonal, windy third | −0.009 | −0.023 | |
| smoothed, σ = 2 km | −0.010 | −0.015 | |
| smoothed, σ = 4 km | −0.010 | −0.019 | |
| smoothed, σ = 8 km | −0.006 | −0.011 | −0.008 |
| smoothed, σ = 12 km | +0.001 | +0.001 | |

- **Fusion never improved the ranking.** Its small loss on the native target shrank as the target was smoothed and
  vanished at 12 km (Fig. 3b).
- **F4.** The registered contrast was −0.180 (95 % interval −0.549 to +0.189; inconclusive). Its direction is opposite to
  the prediction: the relative fusion gain was not larger on the sharper target, it was more negative.
- **Skill falls with smoothing.** Skill fell for every model as the target was smoothed (the energy model from 0.92–0.93
  to 0.86 at 12 km), probably because the models' neighbourhood inputs reach only about 7.6 km.
- **Technical note.** In the per-block contrast of F4, a few edge blocks of 3–5 cells had a reference Spearman of 1, so
  their relative gain was undefined and they left the block-level sign-flip test. The jackknife is not affected.

![Figure 3](../figures/fig3_target_support.png)

**Fig. 3.** Target support. (a) Correlogram of the monthly target and of the seasonal composites of all, the calmest and
the windiest clear days, 1402. (b) Fusion gain (all three sources minus energy, Spearman) against the smoothing scale of
the target, with jackknife 95 % intervals.

### 4.5 Placebo check and lineage audit

**Check M4.** Both constructions removed the location, and neither kept the texture within the registered tolerance.

| Construction | Median \|Spearman\| with the real layer | Median of the largest correlogram difference, 0–20 km | Realisations passing |
|---|---|---|---|
| A (IAAFT) | 0.05–0.09 | 0.21–0.36 | 0 of 5 |
| B (rotation and shift) | 0.04–0.10 | 0.10–0.20 | 0 of 5 |

- With placebo B, the texture differs only at the wrap-around seams. The single urban core of Tehran, which fills much
  of the grid, is cut by them (Fig. 1d, e).
- In a bounded grid with one urban core, a relocation that removes the location either reshapes the core (A) or splits
  it (B).
- By the registered rule, all placebo verdicts carry the label "placebo check failed".

**Lineage audit.**

**Table 5.** Block-cross-validated R² (12 blocks of 22 km, 2-km buffer) of the annual map of each layer predicted from
the context layers (population, building volume, night lights, industrial share, with their box means). Values for
1402; the values for 1403 are within ±0.02, except for the station layer, where both years are given as 1402 / 1403.

| Source | Layer | Gradient boosting | OLS |
|---|---|---|---|
| energy | residential CO₂ | 0.98 | 0.89 |
| energy | industrial CO₂ | 0.95 | 0.60 |
| energy | commercial CO₂ | 0.89 | 0.77 |
| energy | power-plant CO₂ | < 0 | 0.00 |
| traffic | peak vehicle-kilometres | 0.71 | 0.69 |
| traffic | road CO₂ | 0.47 | 0.50 |
| traffic | volume-to-capacity ratio | 0.36 | 0.26 |
| traffic | road length by class | < 0.03 | 0.06–0.15 |
| stations | CO, interpolated (1402 / 1403) | −0.77 / −0.05 | −0.43 / −0.68 |

Three of the four energy layers are almost fully predictable from the urban-form layers that the model already
receives. The power-plant layer, the road network and the station layer are not, but they did not beat their
placebos either.

### 4.6 Additional checks requested in review

These checks were not registered (Section 3.6).
- **Check M4 on the label domain.** [Placeholder, from `fusion_review_domain.csv`. Reports, on the 2,436 cells of the
  label domain rather than the full grid:
  - share of zero cells;
  - Kolmogorov–Smirnov distance and |Spearman| between placebo and real layer;
  - the largest |Spearman| of each placebo with the target and with the context layers;
  - the largest correlogram difference.]
- **Trivial baselines.** [Placeholder, from `fusion_review_absolute.csv`; Table 1.]

## 5. Discussion

### 5.1 What the placebo control showed

Ablation compares a model with a source to a model without it. The placebo compares it with a model that has an input of
the same shape and texture but at the wrong place. In this study, the two comparisons told different stories.

- **Ranking.**
  - Ablation credits the energy layers with almost nothing (+0.004, +0.013 and −0.001 over context alone).
  - The placebo comparison credits them with 0.018–0.038, but most of that is the cost of the placebo input
    (Section 4.2).
- **Hotspots.**
  - Ablation credits the energy layers with +0.17 to +0.19 in PR-AUC in the development years.
  - A relocated copy of the same layers gave +0.07 to +0.11.

The control was therefore useful mainly as a warning. Part of the hotspot gain of a source is obtained without its
information, and the ranking effect of any added input is as large as the information effect being measured.

**What the placebo is not.** The placebo removes the location of a source and, with it, the source's dependence on the
context layers. It is therefore a test of "this source, at its true place, against the same values elsewhere", not a
test of conditional information given the context. Where a source is largely derived from the context, as the energy
inventory is here, the two questions differ, and the placebo contrast overstates the information of the source. A
conditional placebo (Section 5.4) addresses the second question.

**Relation to the literature.** These results agree with machine-learning evidence that multimodal networks overfit and
that their modalities compete (Wang et al., 2020; Huang et al., 2022). They also agree with ecological work showing that
null predictors with realistic spatial structure can look skilful (Fourcade et al., 2018; Ridder et al., 2024).

**A placebo with the right texture is hard to build.** Neither construction met the registered texture tolerance,
because the non-stationary map of a single city cannot be moved inside a bounded grid without being reshaped or cut. The
same difficulty is known for null models of brain maps (Markello & Misic, 2021).
- The two constructions err in different ways: A reshapes the texture, B keeps it but adds seams.
- They gave the same signs and similar sizes for every comparison. Their agreement is a robustness check, not a passed
  test.
- A placebo drawn from the same layers of another city (a donor-city placebo) would keep a whole urban core.

### 5.2 Why the urban sources add little

Two reasons are visible in the data.
- **The sources share their lineage.** The energy inventory is allocated by the same building volumes that the context
  source holds: the context layers predict the residential, industrial and commercial emission layers with R² of
  0.98, 0.95 and 0.89 (Table 5). The traffic model distributes trips by the same population. The station layer is a smooth
  interpolation of a few points. What remains in the energy layers beyond context — the power plants, the split
  between residential and total volume, the county totals — is small, and so is the information measured against the
  placebo.
- **The target is too smooth to reward differences between sources.** The target has about five independent units per
  month, and the sources become almost identical at the scale over which it varies.

Smoothing the target further removed the small cost of fusion rather than revealing a benefit. In this setting, the
native target is already coarse enough that the sources are redundant, and coarser targets only make them more so.

### 5.3 Calm days did not sharpen the target

Low-wind selection of TROPOMI NO₂ has been used to reduce the smearing of urban plumes (Goldberg et al., 2024), and
low-wind data need several months to cover a city (Liu et al., 2024). In our implementation, the composites were not
sharper on calm days: both calm and windy composites decorrelated at about 24 km. The implementation was:
- seasonal composites;
- the calmest and windiest thirds of the clear days;
- ERA5-Land wind at 10 m at 10 UTC.

At this scale the pattern of the city may be set more by the instrument footprint (3.5 × 5.5 km), by averaging over many
days and by the urban form than by daily transport. Other choices might give a sharper target. Examples are
oversampling, wind at plume height, or stricter wind thresholds with longer periods. We did not test them, and the
result applies to the implementation above.

### 5.4 Recommendations and next steps

1. **Report a placebo next to every ablation of a data source.**
   - Use several realisations and, if possible, more than one construction.
   - Run it with every fusion type compared.
   - Report the decomposition into gain over the reference and cost of the placebo (Section 4.2).
2. **Check the placebo.** Check texture, value distribution and dependence on the other inputs, on the cells where the
   model is trained and tested (Section 4.6). A single-city map is the hard case.
3. **Use a rank metric over the whole map as the primary metric.** Treat top-decile metrics as descriptive when the
   target has few independent units.
4. **Report the lineage of every source.** Measure how much of it the other sources predict before training.
5. **Report single seeds, placebo realisations and absolute skill, not only ensembles and differences.**

**Next steps** (not done here):
- a *conditional placebo* that keeps the part of each layer predicted by the context and relocates only the rest;
- a *duplicate-context branch* that adds a copy of the context as a branch, to measure the cost of an added input that
  carries no new information but does not conflict with the context;
- a *placebo-against-placebo null* with more realisations;
- a *donor-city placebo*.

### 5.5 Limitations

- **Scope.** One city, one target product and one family of fusion networks without a graph.
- **Placebo check.** Both constructions failed the registered texture check M4. Their agreement is the basis of the
  placebo results, not a passed check. A smooth surrogate can itself act as a spatial basis (Behrens & Viscarra Rossel,
  2020).
- **What the contrast measures.** The placebo breaks the dependence of a source on the context, so the energy contrast
  mostly reflects the cost of the placebo input (Section 4.2). A conditional placebo was not run.
- **Low power.**
  - The effective sample size of the target is about five per month, so most comparisons have low power.
  - Neighbouring 22-km blocks are not independent at a correlation length of 24 km.
  - The randomisation test was liberal for Spearman in a null calibration of the companion study.
- **Validation design.** The 2-km buffer is short compared with the correlation length, so absolute skill includes
  interpolation within the city. Models with a longer neighbourhood reach scored higher in the companion study.
- **Target support.**
  - The calm–windy contrast rests on four composites per year.
  - Wind at 10 m is a proxy for the transport wind.
  - The smoothed targets were evaluated with the neighbourhood inputs fixed at about 7.6 km, and skill fell for every
    model as the target was smoothed.
- **Held-out year.** 1404 used the 1403 inventory and traffic layers.
- **Prior knowledge.** The design was written after the companion study on the same data, which may have shaped the
  expected directions.

## 6. Conclusions

Ablations of data sources in multi-branch fusion networks confound the information in a source with the effect of the
added input. We tested a placebo control for this — the same layers at a random location, trained in the same network —
in mapping the intra-urban NO₂ pattern of Tehran from TROPOMI.

- **Added input.** An input without information lowered the ranking skill of the map by 0.01–0.03 and raised hotspot
  PR-AUC by 0.05–0.11. These effects are as large as those usually credited to data sources.
- **Energy.** The energy-inventory layers scored above their placebos (0.018–0.038). Most of that contrast was the cost
  of the placebo: over the urban-form context alone they gained at most 0.013, and the context predicted three of the
  four energy layers almost fully.
- **Traffic and stations.** The traffic and station layers did not score above their placebos.
- **Target support.** Fusion did not help at any target resolution tested, and calm-day composites were not sharper.
- **Placebo check failed.** Neither placebo construction passed the registered texture check. The single urban core of
  a bounded map cannot be relocated without being reshaped or cut.

A placebo control is cheap and changes how an ablation table reads. To measure conditional information, however, it
must keep the dependence of a source on the other inputs, and its quality must be checked.

---

## Data and code availability

*[Draft: confirm the bracketed items before submission.]*

- **Code and results.** The code, configurations, pre-registration, run registry and every result table are openly
  available at https://github.com/fatemehvv/Tehran-carbon (branch `paper_fusion`).
  - Placebo branches: `src/tehran_carbon/data/sources/placebo.py`.
  - Composites: `src/tehran_carbon/data/sources/composite.py`.
  - Smoothed targets: `src/tehran_carbon/data/target_support.py`.
  - Tests: `src/tehran_carbon/analysis/fusion_eval.py`.
  - Checks and lineage audit: `src/tehran_carbon/analysis/fusion_checks.py`.
  - Analyses requested in review: `src/tehran_carbon/analysis/fusion_review.py`; the reviews and responses are in
    `docs/paper/fusion_paper/review/`.
  - Figures: `src/tehran_carbon/viz/fusion_figures.py`.
  - A versioned copy will be archived on Zenodo on acceptance [DOI to be added].
- **Data.** Public data (TROPOMI, ERA5, VIIRS, GHSL) and the Earth Engine export scripts are listed as in the companion
  study. The energy, traffic and station data are [as in the companion study].

## CRediT authorship contribution statement

*[Draft: to be confirmed by both authors.]*

- **Fatemeh Vafaeian:** Conceptualization, Methodology, Software, Data curation, Formal analysis, Investigation,
  Visualization, Writing – original draft.
- **Vahid Khatibi:** Supervision, Validation, Writing – review & editing.

## Funding

[This research did not receive any specific grant from funding agencies in the public, commercial or not-for-profit
sectors.]

## Acknowledgements

This publication contains modified Copernicus Sentinel data (2023–2026) and modified Copernicus Climate Change Service
information (2023–2026).

## Declaration of generative AI and AI-assisted technologies in the writing process

*[Draft: adapt it so that it describes exactly how the tool was used.]*

During the preparation of this work the authors used Claude (Anthropic) to help write and review analysis code and to
draft and edit the text. After using this tool, the authors reviewed and edited the content as needed and take full
responsibility for the content of the published article.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have
appeared to influence the work reported in this paper.

---

## References

*DOIs, authors, volumes and pages checked against Crossref on 13 November 2026.*

- Alexander-Bloch, A. F., Shou, H., Liu, S., Satterthwaite, T. D., Glahn, D. C., Shinohara, R. T., Vandekar, S. N., & Raznahan, A. (2018). On testing for spatial correspondence between maps of human brain structure and function. *NeuroImage*, 178, 540–551. https://doi.org/10.1016/j.neuroimage.2018.05.070
- Arevalo, J., Solorio, T., Montes-y-Gómez, M., & González, F. A. (2017). Gated multimodal units for information fusion. ICLR Workshop Track. arXiv:1702.01992.
- Baltrušaitis, T., Ahuja, C., & Morency, L.-P. (2019). Multimodal machine learning: a survey and taxonomy. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 41, 423–443. https://doi.org/10.1109/TPAMI.2018.2798607
- Behrens, T., & Viscarra Rossel, R. A. (2020). On the interpretability of predictors in spatial data science: the information horizon. *Scientific Reports*, 10, 16737. https://doi.org/10.1038/s41598-020-73773-y
- Beirle, S., Borger, C., Dörner, S., Li, A., Hu, Z., Liu, F., Wang, Y., & Wagner, T. (2019). Pinpointing nitrogen oxide emissions from space. *Science Advances*, 5, eaax9800. https://doi.org/10.1126/sciadv.aax9800
- Burt, J. B., Helmer, M., Shinn, M., Anticevic, A., & Murray, J. D. (2020). Generative modeling of brain maps with spatial autocorrelation. *NeuroImage*, 220, 117038. https://doi.org/10.1016/j.neuroimage.2020.117038
- Candès, E., Fan, Y., Janson, L., & Lv, J. (2018). Panning for gold: 'model-X' knockoffs for high dimensional controlled variable selection. *Journal of the Royal Statistical Society Series B: Statistical Methodology*, 80, 551–577. https://doi.org/10.1111/rssb.12265
- Fourcade, Y., Besnard, A. G., & Secondi, J. (2018). Paintings predict the distribution of species, or the challenge of selecting environmental predictors and evaluation statistics. *Global Ecology and Biogeography*, 27, 245–256. https://doi.org/10.1111/geb.12684
- Friedrichs-Manthey, M., Langhans, S. D., Hein, T., Borgwardt, F., Kling, H., Jähnig, S. C., & Domisch, S. (2020). From topography to hydrology — the modifiable area unit problem impacts freshwater species distribution models. *Ecology and Evolution*, 10, 2956–2968. https://doi.org/10.1002/ece3.6110
- Gat, I., Schwartz, I., & Schwing, A. (2021). Perceptual score: what data modalities does your model perceive? *Advances in Neural Information Processing Systems*, 34. arXiv:2110.14375.
- Gately, C. K., & Hutyra, L. R. (2017). Large uncertainties in urban-scale carbon emissions. *Journal of Geophysical Research: Atmospheres*, 122, 11242–11260. https://doi.org/10.1002/2017JD027359
- Gehlke, C. E., & Biehl, K. (1934). Certain effects of grouping upon the size of the correlation coefficient in census tract material. *Journal of the American Statistical Association*, 29, 169–170. https://doi.org/10.1080/01621459.1934.10506247
- Geng, G., Zhang, Q., Martin, R. V., Lin, J., Huo, H., Zheng, B., Wang, S., & He, K. (2017). Impact of spatial proxies on the representation of bottom-up emission inventories: a satellite-based analysis. *Atmospheric Chemistry and Physics*, 17, 4131–4145. https://doi.org/10.5194/acp-17-4131-2017
- Goldberg, D. L., Tao, M., Kerr, G. H., Ma, S., Tong, D. Q., Fiore, A. M., Dickens, A. F., Adelman, Z. E., & Anenberg, S. C. (2024). Evaluating the spatial patterns of U.S. urban NOx emissions using TROPOMI NO₂. *Remote Sensing of Environment*, 300, 113917. https://doi.org/10.1016/j.rse.2023.113917
- Gotway, C. A., & Young, L. J. (2002). Combining incompatible spatial data. *Journal of the American Statistical Association*, 97, 632–648. https://doi.org/10.1198/016214502760047140
- Griffith, D. A. (2005). Effective geographic sample size in the presence of spatial autocorrelation. *Annals of the Association of American Geographers*, 95, 740–760. https://doi.org/10.1111/j.1467-8306.2005.00484.x
- Harms, K. E., Condit, R., Hubbell, S. P., & Foster, R. B. (2001). Habitat associations of trees and shrubs in a 50-ha neotropical forest plot. *Journal of Ecology*, 89, 947–959. https://doi.org/10.1111/j.1365-2745.2001.00615.x
- Hessel, J., & Lee, L. (2020). Does my multimodal model learn cross-modal interactions? It's harder to tell than you might think! *Proceedings of EMNLP 2020*, 861–877. https://doi.org/10.18653/v1/2020.emnlp-main.62
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6, 65–70.
- Hu, P., Li, X., & Zhou, Y. (2022). SHAPE: an unified approach to evaluate the contribution and cooperation of individual modalities. *Proceedings of IJCAI-22*, 3064–3070. https://doi.org/10.24963/ijcai.2022/425
- Huang, Y., Lin, J., Zhou, C., Yang, H., & Huang, L. (2022). Modality competition: what makes joint training of multi-modal network fail in deep learning? (Provably). *Proceedings of ICML*, PMLR 162, 9226–9259.
- Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*, 4, 100804. https://doi.org/10.1016/j.patter.2023.100804
- Kursa, M. B., & Rudnicki, W. R. (2010). Feature selection with the Boruta package. *Journal of Statistical Software*, 36(11), 1–13. https://doi.org/10.18637/jss.v036.i11
- Lakens, D. (2017). Equivalence tests: a practical primer for t tests, correlations, and meta-analyses. *Social Psychological and Personality Science*, 8, 355–362. https://doi.org/10.1177/1948550617697177
- Liu, F., Beirle, S., Joiner, J., Choi, S., Tao, Z., Knowland, K. E., Smith, S. J., Tong, D. Q., Ma, S., Fasnacht, Z. T., & Wagner, T. (2024). High-resolution mapping of nitrogen oxide emissions in large US cities from TROPOMI retrievals of tropospheric nitrogen dioxide columns. *Atmospheric Chemistry and Physics*, 24, 3717–3728. https://doi.org/10.5194/acp-24-3717-2024
- Markello, R. D., & Misic, B. (2021). Comparing spatial null models for brain maps. *NeuroImage*, 236, 118052. https://doi.org/10.1016/j.neuroimage.2021.118052
- Mena, F., Pathak, D., Najjar, H., Sanchez, C., Helber, P., Bischke, B., et al. (2025). Adaptive fusion of multi-modal remote sensing data for optimal sub-field crop yield prediction. *Remote Sensing of Environment*, 318, 114547. https://doi.org/10.1016/j.rse.2024.114547
- Meyer, H., Reudenbach, C., Hengl, T., Katurji, M., & Nauss, T. (2018). Improving performance of spatio-temporal machine learning models using forward feature selection and target-oriented validation. *Environmental Modelling & Software*, 101, 1–9. https://doi.org/10.1016/j.envsoft.2017.12.001
- Meyer, H., Reudenbach, C., Wöllauer, S., & Nauss, T. (2019). Importance of spatial predictor variable selection in machine learning applications — moving from data reproduction to spatial prediction. *Ecological Modelling*, 411, 108815. https://doi.org/10.1016/j.ecolmodel.2019.108815
- Moudrý, V., Keil, P., Gábor, L., Lecours, V., Zarzo-Arias, A., Barták, V., Malavasi, M., Rocchini, D., Torresani, M., Gdulová, K., Grattarola, F., Leroy, F., Marchetto, E., Thouverai, E., Prošek, J., Wild, J., & Šímová, P. (2023). Scale mismatches between predictor and response variables in species distribution modelling: a review of practices for appropriate grain selection. *Progress in Physical Geography*, 47, 467–482. https://doi.org/10.1177/03091333231156362
- Mrkvička, T., Dvořák, J., González, J. A., & Mateu, J. (2021). Revisiting the random shift approach for testing in spatial statistics. *Spatial Statistics*, 42, 100430. https://doi.org/10.1016/j.spasta.2020.100430
- Mu, Q., Denby, B. R., Wærsted, E. G., & Fagerli, H. (2022). Downscaling of air pollutants in Europe using uEMEP_v6. *Geoscientific Model Development*, 15, 449–465. https://doi.org/10.5194/gmd-15-449-2022
- Prichard, D., & Theiler, J. (1994). Generating surrogate data for time series with several simultaneously measured variables. *Physical Review Letters*, 73, 951–954. https://doi.org/10.1103/PhysRevLett.73.951
- Ridder, G., Hardy, O. J., & Ovaskainen, O. (2024). Generating spatially realistic environmental null models with the shift-&-rotate approach helps evaluate false positives in species distribution modelling. *Methods in Ecology and Evolution*, 15, 2331–2342. https://doi.org/10.1111/2041-210X.14443
- Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40, 913–929. https://doi.org/10.1111/ecog.02881
- Schreiber, T., & Schmitz, A. (1996). Improved surrogate data for nonlinearity tests. *Physical Review Letters*, 77, 635–638. https://doi.org/10.1103/PhysRevLett.77.635
- Vafaeian, F., & Khatibi, V. (in review). Reach, not graphs: neighbourhood reach explains the skill of distance-based graph neural networks in mapping intra-urban NO₂ hotspots from TROPOMI. [Journal to be added]
- Wagner, H. H., & Dray, S. (2015). Generating spatially constrained null models for irregularly spaced data using Moran spectral randomization methods. *Methods in Ecology and Evolution*, 6, 1169–1178. https://doi.org/10.1111/2041-210X.12407
- Wang, W., Tran, D., & Feiszli, M. (2020). What makes training multi-modal classification networks hard? *Proceedings of CVPR 2020*, 12695–12705. https://doi.org/10.1109/CVPR42600.2020.01271
- Zeng, J., Liu, Y., Zhang, G., Ding, J., Lin, Y., Yuan, J., & Li, Y. (2025). OpenCarbon: a contrastive learning-based cross-modality neural approach for high-resolution carbon emission prediction using open data. *Proceedings of IJCAI-25*, 9999–10007. https://doi.org/10.24963/ijcai.2025/1111
- Zhalehdoost, A., & Taleai, M. (2025). Unravelling the importance of spatial and temporal resolutions in modeling urban air pollution using a machine learning approach. *Scientific Reports*, 15, 27708. https://doi.org/10.1038/s41598-025-13639-3
- Zheng, Y. (2015). Methodologies for cross-domain data fusion: an overview. *IEEE Transactions on Big Data*, 1, 16–34. https://doi.org/10.1109/TBDATA.2015.2465959

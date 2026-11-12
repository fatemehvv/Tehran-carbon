# Reach, not graphs: neighbourhood reach explains the skill of graph neural networks in mapping urban combustion hotspots from TROPOMI NO₂

**Fatemeh Vafaeian¹, Vahid Khatibi¹\***

¹ [Affiliation to be completed]  
\* Corresponding author: [e-mail to be completed]

*Draft manuscript, 12 November 2026. All numbers come from the runs on branch `paper_TC` of the project repository (see
Data and code availability). Items in [brackets] are to be completed before submission.*

---

## Highlights

- Graph neural network skill in mapping urban NO₂ hotspots is explained by neighbourhood reach, not by the graph.
- Skill rises steeply until the model sees about 12 km and saturates within the target's 24-km correlation length.
- At equal reach, a linear spatial-lag model matches a five-layer graph neural network on an unseen year.
- Averaging training seeds and a block bootstrap overstated the graph model's advantage.
- A correlation-length rule gives graph and baseline models the same reach.

---

## Abstract

Graph neural networks (GNNs) are increasingly used to map urban emissions and air pollution, and their gains are
usually attributed to learning on a graph. We test this attribution for intra-urban combustion hotspots observed from
space. In Tehran, a megacity with some of the highest tropospheric NO₂ columns among large cities worldwide, we predict
the monthly TROPOMI NO₂ pattern on a 1-km grid (2,436 cells) from bottom-up traffic and energy emission layers, ground
monitoring and urban-form data. We compare a multi-scale GraphSAGE model with spatially lagged regression (SLX) models
whose neighbourhood reach is varied systematically from 0 to 38 km and, for the main comparison, set for all models
by one rule: the correlation length of the target. All models are validated with 22-km spatial blocks
and on a held-out year, with leakage-free pre-processing, five training seeds and inference designed for few spatial
blocks.

Skill is governed by reach. Models restricted to their own cell reach a Spearman correlation of 0.66–0.77, every SLX model
reaches 0.91–0.93 once it sees about 12 km, and further gains are small, inside the target's correlation length (24 km).
At equal reach (22.7 km), a single GNN exceeds the best SLX model by only 0.006–0.008 in Spearman, below a pre-specified
equivalence margin of 0.02, and not at all in hotspot precision (PR-AUC); the GNN's apparent advantage in PR-AUC comes
from averaging five seeds. On the held-out year, a linear SLX model with a 23-km window matches a five-layer GNN
(Spearman 0.929 vs 0.933; PR-AUC 0.785 vs 0.782), at a small fraction of the computing cost. The target's effective
sample size is about five, so evaluation choices decide verdicts: a block bootstrap declared the GNN significantly better
on the held-out year, whereas exact block randomisation tests did not.

For satellite-observed urban combustion hotspots, it is neighbourhood reach, not graph structure, that delivers
predictive skill. Spatial deep-learning models should be compared with reach-matched spatial baselines.

**Keywords:** graph neural networks; receptive field; spatial lag of X; TROPOMI; NO₂; urban emissions; spatial
cross-validation; Tehran

---

## 1. Introduction

Cities emit most of the world's fossil CO₂, and policies to cut these emissions need to know *where* in a city
combustion is concentrated. Satellite NO₂ is increasingly used as a tracer of urban fossil combustion. NO₂ is co-emitted
with CO₂, it is short-lived, and since 2018 the TROPOMI instrument on Sentinel-5P has mapped it daily at a few
kilometres resolution (Beirle et al., 2019; Li et al., 2025). Recent work uses TROPOMI NO₂ to locate urban CO₂ plumes
and to estimate city emissions (Reuter et al., 2019; Ahn et al., 2025), to derive gridded NOx emissions
(Rey-Pommier et al., 2025), and to evaluate kilometre-scale bottom-up inventories (Goldberg et al., 2024). Models that
link emission and urban-form data to the observed NO₂ pattern can screen cities for combustion hotspots, test
inventories and support scenario analysis.

Deep learning, and graph neural networks in particular, has become the method of choice for such urban mapping tasks.
Recent models learn urban carbon emissions from satellite images and points of interest (Zeng et al., 2025), from road
networks and origin–destination flows (Zeng et al., 2024), at the level of individual buildings (Yap et al., 2025), or
across 329 Chinese cities with a graph-based multi-branch framework (Shao et al., 2026). GNNs with physically
motivated edges are used for air-quality prediction (Wang et al., 2020; Hettige et al., 2024). Graph networks have been
fused with high-resolution satellite data to predict PM2.5 and NO₂ (Wang et al., 2025), and used to map PM2.5 from
satellite and station data at scales from city regions to the globe (Bai et al., 2025; Teng et al., 2026; Yuan et al.,
2026). GraphSAGE has been applied to urban NO₂ (Gueterbock et al., 2025). The gains of these models are usually
credited to the graph: to learning how information should flow between neighbouring places.

A graph model, however, differs from a conventional model in two ways at once. It *learns* how to combine its
neighbourhood, and it *sees* a neighbourhood whose size grows with every message-passing layer: its receptive field, or
reach (Luo et al., 2016). Within graph learning, the receptive field is treated as a design choice to be tuned or
learned (Gao et al., 2025), and convolutional models have been found to fail where their effective receptive field is
too small for the process (Chen et al., 2026). Yet conventional spatial models used as baselines are often given
no neighbourhood at all (own-cell regression, random forests on local features), or a much smaller one than the GNN.
Under such comparisons, any advantage of the GNN may simply reflect that it looks further.

The spatial econometric answer to "the neighbours matter" is the spatial lag of X (SLX) model, which adds neighbourhood
means of the predictors (Halleck Vega & Elhorst, 2015). In air-quality science, land-use regression does the same with
buffers (Hoek et al., 2008). Both can be given any reach at negligible cost. Explicit neighbourhood information of this
kind helps other learners too:

- spatial-lag inputs improve neural networks (Xiao et al., 2023);
- spatial context in the modelling step improves convolutional classifiers of urban form (Fleischmann & Arribas-Bel,
  2024);
- kriging-based spatial-lag features improve tabular learners in soil mapping (Schmidinger et al., 2026).

Evaluation adds a second difficulty. Satellite NO₂ is strongly autocorrelated, so random cross-validation overstates
skill and blocks of the autocorrelation range are needed (Roberts et al., 2017; Ploton et al., 2020; Meyer & Pebesma,
2022). A recent review of machine learning for satellite NO₂ finds that reported skill varies widely between studies,
partly because their validation strategies differ (Zhang et al., 2026). The appropriate design depends on the intended
prediction task (Wadoux et al., 2021; Milà et al., 2022;
Linnenbrink et al., 2026). With blocks of tens of kilometres, a single city yields only a dozen blocks, and hotspots
concentrate in a few of them. Standard bootstrap inference is unreliable with so few, so unequal clusters (Cameron et
al., 2008; MacKinnon et al., 2023). The randomness of training also matters, because deep models vary from seed to seed
(Bouthillier et al., 2021). Pre-processing that uses information from the test area can inflate skill (Kapoor &
Narayanan, 2023).

The question has begun to surface in GIScience and environmental modelling:

- **GNNs against spatial regression.**
  - Graph convolutional networks have been compared with spatial and geographically weighted regression (Zhu et al.,
    2021; Liu et al., 2025).
  - Graph-based deep spatial regression fitted house prices better than spatial-regression baselines (Zhu et al.,
    2026).
  - GNNs have beaten spatial-lag, geographically weighted and gradient-boosting baselines for neighbourhood health
    outcomes under spatial cross-validation (Zhao et al., 2026).
  - They have also beaten linear, spatial-regression and XGBoost baselines for bike-share ridership (Mohseni et al.,
    2026).
- **The link to spatial regression and geostatistics.**
  - An attention-based GNN can be read as a generalised spatial regression model (Liang et al., 2023).
  - GNNs entered spatial statistics because message passing mimics interaction between geographical neighbours (Ma et
    al., 2025).
  - Neural networks for geospatial data admit a graph representation whose aggregation weights resemble kriging (Zhan &
    Datta, 2024).
  - Graph convolutions have been used for the spatial interpolation of correlated data (Abémgnigni Njifon &
    Schuhmacher, 2024).
- **Simple baselines remain strong.**
  - For urban NO₂ in Berlin, a random forest and a spatial GNN reached the same accuracy (R² ≈ 0.6; Schmidt & Ong,
    2026).
  - In urban PM2.5 imputation, simple spatial interpolation outperformed standard data-driven GNNs unless physics was
    built in (Du et al., 2026).
- **Graph models are sensitive to evaluation.**
  - GNN residuals can remain spatially autocorrelated (Bao et al., 2024).
  - A GNN that generalised within a fixed graph struggled under leave-one-cluster-out validation (Flores et al., 2026).

Table 1 audits how far the GNN and its baselines see in comparable studies whose full text describes both.

- **What the baselines see.** Their own location only, a neighbourhood of a different size, or one that is not
  reported.
- **The closest case** is Liang et al. (2023).
  - Their linear SLX baseline averaged over the same nearest stations as their GNN.
  - Their nonlinear baseline saw no neighbours.
  - The GNN also drew on a second, non-geographic neighbourhood.
- **No study in Table 1** gives a nonlinear spatial-lag baseline the GNN's reach, or varies the reach of both models.

**Table 1.** How far the GNN and its baselines see in comparable studies (full texts read; "n.r.": not reported).

| Study | Task and units | GNN and its reach | Baselines and what they see | Reach matched? |
|---|---|---|---|---|
| Wang et al. (2020) | PM2.5 forecasts; 184 cities, China | Cities linked if < 300 km apart with no ridge > 1,200 m between them; one graph step per 3-h step | MLP, LSTM, GRU: own city only. GC-LSTM: the same graph. nodesFC-GRU: all cities | Only for GC-LSTM, itself a graph model |
| Liang et al. (2023) | Trip generation; about 1,000 bike-share stations, New York | One attention layer over the 5 nearest and the 5 most similar stations | Linear SLX: nearby stations (apparently the same 5). Linear regression, XGBoost: own-station features only | Only for the linear SLX |
| Klemmer et al. (2023) | Four geographic regression datasets | GCN, GAT, GraphSAGE: 2 layers on a 5-nearest-neighbour graph | Exact Gaussian process: all training points | No: the baseline sees more |
| Hettige et al. (2024) | PM2.5 forecasts; 35 and 11 stations, Beijing and Shenzhen | Distance- and wind-weighted station graph; reach n.r. | Historical average, ODE models: none. DCRNN, STGCN: 3-step graph diffusion. AirFormer: 50 and 200 km. GTS: learned 30-nearest graph | No |
| Gueterbock et al. (2025) | Hourly NO₂ with TROPOMI; 8 sites, Bristol | GraphSAGE, 2 hops (3 and 5 sampled neighbours) | XGBoost, MLP, CNN: per-site inputs; neighbours n.r. | n.r. |
| Du et al. (2026) | PM2.5 imputation; 41 sensors, Fresno | 5 layers on complete graphs (all sensors) | Inverse-distance weighting, mean fill: all sensors. Kriging, Gaussian process: n.r. Geo-LSTM: K nearest, K n.r. | Partly; simple interpolation beat the data-driven GNNs |
| Zhao et al. (2026) | Health outcome; 4,835 areas of 1–4 km², London | GATv2, 2 layers on contiguity (2 steps) | Spatial lag model, GWR: weights and bandwidth n.r. RF, LightGBM: none | n.r.; 2- and 3-step graphs lowered skill |
| **This study** | NO₂ hotspots; 2,436 cells of 1 km, Tehran | GraphSAGE, 3 layers × 9 cells: 22.7 km (5 layers, 37.8 km, on the held-out year) | SLX-OLS, SLX-GBM, SLX-RF: 0–37.8 km. No-graph network: own cell | **Yes**: reach set by one rule for all models, and varied |

[Rows to be added once the full texts of Schmidt & Ong (2026), Mohseni et al. (2026), Zhu et al. (2026) and Shao et al.
(2026) have been read.]

We also searched for such a study in two ways:

- a structured ScienceDirect search: five Boolean queries combining graph neural networks with spatial lag, receptive
  field, air pollution, satellite NO₂ and urban emissions, giving 191 unique records, most from 2025–2026;
- a semantic search of the wider literature.

Neither found a study that compares a GNN with a nonlinear spatial-lag or buffer baseline constrained to the same reach,
nor one that varies the reach of both. The search record is in the repository (`docs/paper/lit/`).

Choosing a neighbourhood from the range of spatial dependence is itself established practice:

- window sizes for image texture (Franklin et al., 1996);
- the scale range of contextual covariates in machine learning (Behrens et al., 2019);
- the size of spatial cross-validation blocks (Valavi et al., 2019);
- the depth of a GNN relative to the radius a task needs (Alon & Yahav, 2021);
- the receptive field of a GNN for turbulent transport (Freitas et al., 2026).

It has not been used to give a GNN and its baselines the same reach. To our knowledge, the reach of a GNN has therefore
never been separated from its graph architecture, in urban emission and air-pollution mapping or elsewhere. Fig. 1
traces how the literature narrows to this gap. Here we close it for intra-urban combustion hotspots observed by TROPOMI
in Tehran. Tehran is a megacity of about nine million people at the foot of the Alborz mountains, with the highest mean
NO₂ column among large cities worldwide (Huber et al., 2026). Our contributions are:

1. **A reach-matched evaluation protocol.** One rule sets the reach of the GNN and of its spatial-lag baselines
   before any model is fitted: the correlation length of the target. We also vary the reach of the SLX models from 0
   to 38 km. Models are compared under 22-km block cross-validation and on a held-out year.
2. **A physical interpretation of the needed reach.** We relate the reach at which skill saturates to the correlation
   length of the satellite target and to the transport scale of NOx.
3. **A quantified account of evaluation pitfalls on a real case:** the target's effective sample size, the hidden
   advantage of seed averaging, the gap between bootstrap and exact randomisation inference, and a test for spatial
   pre-processing leakage.
4. **A test of multi-source fusion** of traffic, energy and pollution data under the same design.

Decision rules for the main comparisons were registered before the corresponding runs (Section 3.6).

![Figure 1](../figures/fig1_literature_funnel_en.png)

**Fig. 1.** From the literature to the gap. Satellite NO₂ is used as a tracer of urban combustion; deep learning, and
graph neural networks in particular, map urban emissions and air pollution; their gains are credited to the graph.
A GNN, however, also sees further than its baselines, and no study has controlled this reach. This study does.

---

## 2. Study area and data

### 2.1 Grid, years and label domain

The analysis grid covers 50.85–51.90°E and 35.35–35.95°N with 72 × 126 cells of 1/120° (about 0.75 × 0.93 km). The
label domain holds the 2,436 cells in the three central counties (Tehran, Rey, Shemiranat) to which the emission
inventory assigns any CO₂. This excludes mountains and desert, so a model cannot score by separating city from non-city.
Fig. 2 shows the domain, the observed target of the held-out year and its hotspots.

![Figure 2](../../figures/fig06_maps_1404.png)

**Fig. 2.** The label domain and the target on the held-out year 1404: observed NO₂ pattern (z, annual mean of the
monthly maps), the out-of-fold prediction of the GNN, and the hotspots (top 10 %) found by both, by the observation only
and by the prediction only.

We use three Iranian (Solar Hijri) years: 1402 (21 March 2023 – 19 March 2024) and 1403 (20 March 2024 – 20 March 2025)
for development, and 1404 (21 March 2025 – 20 March 2026) as a held-out year.

### 2.2 Target: the monthly pattern of TROPOMI NO₂

The target is the monthly mean tropospheric NO₂ column of the TROPOMI offline level-3 product (Google Earth Engine,
`COPERNICUS/S5P/OFFL/L3_NO2`), exported on the grid. For each month, the log column is standardised over the domain
(z), so only the spatial pattern is kept. A hotspot is a cell in the top 10 % of z in that month.

The target is not the inventory. Using a modelled CO₂ map as the target would let a model reproduce the inventory's own
allocation.

### 2.3 Predictors

| Group | Layers |
|---|---|
| **Traffic** | Road-transport CO₂ of a bottom-up inventory (IPCC Tier 1 with Monte Carlo uncertainty, allocated to 1 km); road length by class; peak-hour vehicle-kilometres and volume-to-capacity ratio of a traffic assignment |
| **Energy** | Residential, commercial, industrial and power-sector CO₂ of the same inventory; monthly heating degree-days |
| **Pollution** | CO from the municipal monitoring stations, interpolated by inverse distance |
| **Context** | GHSL population and building volume, VIIRS night lights, industrial land share, monthly ERA5 temperature and wind speed |

Monthly sector layers are the annual allocation multiplied by monthly sector shares. For 1404 no inventory exists yet, so
the 1403 inventory and traffic layers stand in for all models.

### 2.4 What the target measures

| Check | Result |
|---|---|
| **Autocorrelation** | The monthly z pattern is extremely smooth. Its correlogram is 0.95 at 0–2 km and 0.76 at 8–10 km, and it crosses zero at **24 km** (Fig. 3). The effective number of independent cells (Griffith, 2005) is **4.9** in both development years, against 2,436 cells and about 88 TROPOMI nadir pixels (3.5 × 5.5 km). |
| **Sensitivity to cloud screening** | A target built from daily scenes with cloud fraction < 0.3 agrees with the standard target: Spearman 0.98–0.99, overlap of hotspot sets (Jaccard) 0.73–0.76. |
| **Terrain** | The north of the city is about 700 m higher than the south, and a TROPOMI emission inversion reported terrain artefacts near Mount Tochal (Rey-Pommier et al., 2025). Elevation correlates with z (\|ρ\| = 0.49), but partialling it out barely changes model skill (Spearman 0.949 → 0.940). |
| **Ground stations** | The satellite pattern agrees only weakly with the ranking of the 16–17 monitoring stations (mean monthly Spearman 0.11 in 1402 and 0.30 in 1403). The target therefore describes column NO₂, not street-level concentrations (cf. Kafi et al., 2026). |

In short, the target is a smooth column pattern with a correlation length of about 24 km and very few independent
degrees of freedom.

![Figure 3](../figures/fig3_correlogram.png)

**Fig. 3.** Correlogram of the monthly target in 1402 and 1403 (2-km distance bins). The curve
crosses zero at 24.0 km; the effective number of independent cells is 4.9 of 2,436. Grey band: one TROPOMI pixel
(3.5 × 5.5 km).

---

## 3. Methods

### 3.1 Models

All models use the same predictors (Section 2.3) and the same target. The model families are:

| Model | Description |
|---|---|
| **Own-cell models** | Ordinary least squares (OLS), histogram gradient boosting (GBM) and random forests (RF) on the predictors of the cell itself |
| **SLX models** | The same estimators plus box means of every spatially varying predictor over nested windows of half-width 1, 3, 9 and 15 cells (the windows of the original study design). Extended to 27, 36 and 45 cells to match the reach of three to five GraphSAGE layers. This is the SLX specification of spatial econometrics and the buffer design of land-use regression. |
| **No-graph network** | One encoder per input group, a softmax gate over the groups (gated fusion) and a linear head. No neighbourhood information. |
| **Multi-scale GraphSAGE (GNN)** | The same gated encoder followed by L residual GraphSAGE layers (Hamilton et al., 2017) with mean aggregation over three edge sets: the 8 neighbours at distances of 1, 3 and 9 cells. One layer reaches 9 cells; L layers reach 9L cells. |
| **Physical operator** | Gaussian-smoothed inventory layers with one non-negative NO₂ yield per sector; width and yields fitted on the training cells |

**Hyper-parameters** of the neural models (hidden size 32, two encoder layers, dropout 0.1, AdamW, learning rate
3 × 10⁻³, 300 epochs) and of GBM and RF were fixed before the original study's runs and are not tuned here. Every neural
model is trained with five seeds. As in the original study, its prediction is the mean of the five (the ensemble). We
also score every seed on its own.

### 3.2 Reach

The reach of a model is how far, in cells, a predictor value can be from a cell and still influence that cell's
prediction:

- SLX: the largest window half-width.
- No-graph network and own-cell models: zero.
- GNN: L × 9 cells.

Cells are converted to kilometres with the mean cell size (0.84 km). Reaches compared here are 0, 12.6, 22.7, 30.3 and
37.8 km for SLX, and 0 and 22.7 km for the development GNN (three layers). On the held-out year the GNN has 37.8 km
(five layers, the depth fixed in the original confirmatory design).

**A correlation-length rule for reach (R\*).** To fix reach without tuning, we set it to the correlation length of the
target: the distance R\* at which the correlogram of the monthly target first crosses zero (Section 2.4; 24.0 km in
both development years).

- **The principle is borrowed.** We adapt a principle used for texture windows, contextual covariates, cross-validation
  blocks and GNN depth (Section 1).
- **The use is new.** We use it to give a GNN and its spatial-lag baselines the same reach, chosen before either is
  fitted.
- **On our grid.** R\* corresponds to SLX windows of up to 27 cells (22.7 km), which is also the reach of the
  three-layer GNN.

We call a comparison reach-matched when every model sees the same distance, here R\*. R\* was computed after the
stage A runs (Section 3.6). Results at R\* are therefore consistent with the rule, not a blind test of it.

### 3.3 Validation designs

- **Spatial cross-validation.** The domain is divided into 22-km square blocks (the autocorrelation range estimated
  for the original design), giving 12 blocks. Each block is held out once, and training cells within 2 km of the test
  block are removed.
- **Held-out year (1404).** Each fold trains on the training blocks of 1402 and 1403 and predicts the held-out block in
  1404: new places in a new year (Meyer et al., 2018).
- **Leakage control.** The station layer was originally interpolated once from all stations, including those inside the
  test block. Here it is rebuilt for every fold from the stations farther than 2 km from the test block. In one fold,
  this removes 19 of the 28 stations.

### 3.4 Metrics

- **Primary metric: Spearman correlation** between predicted and observed z over all domain cells, averaged over
  months.
- **PR-AUC** of the hotspot label (chance level 0.10), as a measure of hotspot precision.
- **Sensitivity analyses:** hotspot labels at the top 20 % and top 5 %, and the same metrics on superpixels of 5 × 5
  cells, about one TROPOMI pixel.

### 3.5 Inference with few spatial blocks

Differences between two models are computed on identical out-of-fold predictions, and are assessed with four tools:

1. **An exact block sign-flip randomisation test.** Under the null hypothesis the two models are exchangeable within
   each block. We swap their predictions block by block over all 2¹² assignments and recompute the city-wide difference.
   Predictions are rank-normalised within each month before mixing.
2. **Delete-one-block jackknife intervals** with K − 1 degrees of freedom.
3. **A four-way verdict** with an equivalence margin of δ = 0.02 in Spearman: superior, equivalent (90 % interval inside
   ±δ, i.e. two one-sided tests; Lakens, 2017), non-inferior, inferior, or inconclusive. The margin equals the larger of
   0.02 and twice the seed-to-seed standard deviation of Spearman (0.0077).
4. **Holm's adjustment** across comparisons (Holm, 1979).

The block bootstrap of the original study is reported for comparison.

PR-AUC is defined in only 5 of the 12 blocks, because the others never contain a hotspot. The smallest p-value any exact
test on it can reach is therefore 1/32 = 0.031, and it cannot be significant at 0.025. This is why Spearman is the
primary metric.

### 3.6 Pre-registration and transparency

The decision rules of the original study, including the held-out-year test on 1404, were written before any model was
evaluated. The checks reported here were written before the corresponding runs:

- reach-matched SLX models;
- the leakage control;
- the physical operator;
- the inference rules.

The registrations are timestamped by the project's version control. Deviations are listed in the registration
documents. In particular, a planned second held-out year (1405) and a hyper-parameter search were registered but not
carried out (Section 5.4). All runs were executed on one RTX 4080 GPU, and every run is logged with its configuration
hash, data hash and code version.

---

## 4. Results

### 4.1 Skill is governed by reach

Table 3 and Fig. 4 show skill against reach. The pattern is the same for every model family and both development years.

- **Own cell (reach 0).** Models reach a Spearman of 0.66–0.77 and a PR-AUC of 0.45–0.53.
- **About 12 km.** Every SLX estimator reaches 0.91–0.93 in Spearman and 0.71–0.80 in PR-AUC.
- **Beyond 12 km.** The curves flatten: SLX models at 22.7–37.8 km reach Spearman 0.924–0.950.
- **Most of the gain arrives before the target's correlation length** (24 km). Seeing further than that adds little.

**Table 3.** Cross-validated skill against reach (22-km blocks; 1402 / 1403). Ensemble = mean of five seeds; single =
mean score of the individual seeds.

| Reach (km) | Model | Spearman | PR-AUC |
|---|---|---|---|
| 0 | OLS | 0.662 / 0.668 | 0.504 / 0.533 |
| 0 | GBM | 0.768 / 0.745 | 0.520 / 0.521 |
| 0 | RF | 0.759 / 0.743 | 0.520 / 0.501 |
| 0 | No-graph network (ensemble) | 0.685 / 0.676 | 0.490 / 0.449 |
| 12.6 | SLX-OLS | 0.922 / 0.934 | 0.763 / 0.753 |
| 12.6 | SLX-GBM | 0.924 / 0.912 | 0.763 / 0.791 |
| 12.6 | SLX-RF | 0.917 / 0.915 | 0.706 / 0.798 |
| 22.7 | SLX-OLS | 0.938 / 0.931 | 0.803 / 0.737 |
| 22.7 | SLX-GBM | 0.938 / 0.934 | 0.733 / 0.769 |
| 22.7 | SLX-RF | 0.934 / 0.930 | 0.696 / 0.797 |
| 22.7 | **GNN, 3 layers (ensemble)** | **0.949 / 0.947** | **0.800 / 0.841** |
| 22.7 | **GNN, 3 layers (single seed)** | **0.944 / 0.942** | **0.771 / 0.804** |
| 30.3 | SLX-OLS | 0.942 / 0.929 | 0.735 / 0.733 |
| 30.3 | SLX-GBM | 0.938 / 0.933 | 0.764 / 0.750 |
| 30.3 | SLX-RF | 0.939 / 0.932 | 0.710 / 0.784 |
| 37.8 | SLX-OLS | 0.935 / 0.924 | 0.804 / 0.749 |
| 37.8 | SLX-GBM | 0.950 / 0.936 | 0.781 / 0.788 |
| 37.8 | SLX-RF | 0.943 / 0.937 | 0.716 / 0.792 |

![Figure 4](../figures/fig4_reach.png)

**Fig. 4.** Skill against reach under 22-km block cross-validation, Spearman (a, b) and PR-AUC (c, d), 1402 (left) and
1403 (right). Lines: SLX models with windows of growing reach. Points: the networks, each run at one reach (no-graph
network at 0 km, GNN at 22.7 km), drawn slightly to the side of that reach; the point is the five-seed ensemble and the
bar the range of the single seeds. Dashed line: zero crossing of the target correlogram (24 km).

### 4.2 At equal reach, the graph adds little

**At exactly the same reach (22.7 km):**

- A single GNN exceeds the best SLX model by **0.006 (1402) and 0.008 (1403)** in Spearman. This is below the
  equivalence margin of 0.02.
- In PR-AUC, a single GNN is not better than the best SLX model at that reach (0.771 vs 0.803 in 1402; 0.804 vs 0.797 in
  1403).

**Against the best SLX model at any reach** (Table 4, Fig. 5), the GNN ensemble differs by −0.001 and +0.010 in Spearman. The
jackknife intervals ([−0.034, +0.032] and [−0.028, +0.048]) and the randomisation tests (p = 0.58 and 0.13) show no
advantage. The PR-AUC differences (+0.019 and +0.053) are inconclusive, and they shrink to −0.010 and +0.016 for single
seeds.

**Only against the shorter-reach SLX models of the original design** (12.6 km) does the GNN look clearly better:
+0.025 and +0.035 in Spearman (randomisation p = 0.001 and 0.0007; Holm-adjusted p = 0.021 and 0.016), and +0.037 and
+0.051 in PR-AUC. The comparison that made the GNN look better was a comparison of reaches.

**Table 4.** GNN (ensemble) minus reference, cross-validation on 1402 / 1403. Intervals: delete-one-block jackknife,
95 %. p: one-sided exact block sign-flip test.

| Reference | Metric | Δ 1402 [95 % CI] | p | Δ 1403 [95 % CI] | p |
|---|---|---|---|---|---|
| Best SLX, reach 12.6 km (SLX-GBM) | Spearman | +0.025 [−0.003, +0.052] | 0.001 | +0.035 [−0.013, +0.082] | 0.0007 |
| | PR-AUC | +0.037 [+0.021, +0.053] | 0.031 | +0.051 [+0.025, +0.077] | 0.10 |
| Best SLX, any reach (SLX-GBM, 37.8 km) | Spearman | −0.001 [−0.034, +0.032] | 0.58 | +0.010 [−0.028, +0.048] | 0.13 |
| | PR-AUC | +0.019 [−0.058, +0.096] | 0.094 | +0.053 [−0.023, +0.130] | 0.11 |

### 4.3 On a held-out year, a linear model with the right reach matches the GNN

On 1404, new places in a new year, the five-layer GNN reaches Spearman 0.933 and PR-AUC 0.782.

- **Against the pre-specified reference of the original design** (SLX-RF, reach 12.6 km), the GNN is ahead: +0.029 in
  Spearman, [+0.001, +0.056], p = 0.0095.
- **Against SLX models with the GNN's reach, the advantage disappears** (Fig. 5). The best of them is plain OLS with windows up
  to 27 cells (22.7 km): Spearman 0.929, PR-AUC 0.785. The difference is +0.004 [−0.026, +0.035] in Spearman (p = 0.30)
  and −0.003 [−0.190, +0.183] in PR-AUC.
- **Cost.** This linear model's cross-validation ran in 8 s on a CPU, against 39 min on a GPU for the GNN.
- **Caveat.** The best reach-matched model was identified after the results among fifteen variants, which favours it.
  Its margin over the GNN is nonetheless nil, not large.
- **At the rule's reach.** R\* is computed from the target alone, not from model skill, and selects the same 22.7-km
  windows.
  - Spearman: all three SLX estimators are within 0.02 of the GNN (OLS 0.929, RF 0.919, GBM 0.916, against 0.933).
  - PR-AUC: only OLS matches it (0.785; RF 0.727, GBM 0.642, against 0.782).

![Figure 5](../figures/fig5_differences.png)

**Fig. 5.** GNN (five-seed ensemble) minus reference in Spearman (a) and PR-AUC (b). Upper rows: references that see
less (SLX, 12.6 km; the design of the original study). Lower rows: references that see as far or further (best SLX at
any reach in cross-validation; SLX-OLS at the GNN's reach of 22.7 km on the held-out year). Thick and thin bars:
90 % and 95 % delete-one-block jackknife intervals. Grey band: equivalence margin (±0.02).

### 4.4 Seed averaging hides instability

| Metric | Seed-to-seed SD of the GNN | Effect of averaging five seeds |
|---|---|---|
| Spearman | ≈ 0.005 | ≈ 0.005 |
| PR-AUC | 0.028 (1402), 0.038 (1403), 0.075 (1404) | 0.03–0.04 (e.g. 0.800 vs 0.771 in 1402) |

The GNN's rank skill is stable across seeds; its hotspot precision is not (Fig. 6). Single seeds in 1403 range from 0.756 to 0.858
in PR-AUC. Deterministic SLX models get no such benefit from averaging. A comparison of a five-seed ensemble with
single deterministic baselines therefore favours the network by about the size of the effects usually reported.

![Figure 6](../figures/fig6_seeds.png)

**Fig. 6.** Seeds against ensemble, Spearman (a) and PR-AUC (b). Each row has three lanes so that close values stay
visible.

- **Upper lane:** the GNN. The bar is the range of the five single-seed GNNs (tick: their mean); the dot is the
  five-seed ensemble.
- **Middle lane:** SLX with equal or longer reach (black diamond).
- **Lower lane:** SLX at 12.6 km (open diamond).

### 4.5 How the evaluation decides the verdict

The original study's confirmatory test on 1404 compared the GNN with the pre-specified SLX-RF.

- **With the block bootstrap**, the PR-AUC difference was +0.065 with a 95 % interval of [+0.013, +0.163], and the
  hypothesis was declared supported.
- **With the same predictions and inference for few blocks**, the jackknife interval is [−0.019, +0.149] and the exact
  randomisation p is 0.10. The verdict becomes *non-inferior*.

With an effective sample size of about five (Section 2.4), the choice of inference is not a technicality: it is the
difference between "significantly better" and "no evidence of a difference".

### 4.6 Fusion of data sources does not help

**Fusing traffic, energy and pollution data**, compared with the best single source (energy) under the same gated
network:

- PR-AUC: +0.022 and +0.040, within seed noise (non-inferior).
- Spearman: −0.021 and −0.022, negative for all five seeds.
- On 1404 the fused and the energy-only GNN are equivalent (Spearman −0.007 [−0.022, +0.007]).

**Two observations explain this:**

1. **Urban form already gives the ranking.** A model with urban-form context alone (population, building volume, night
   lights) already reaches Spearman 0.91, though only 0.58–0.59 in PR-AUC. Urban form gives the overall ranking, and the
   emission layers sharpen the top of it.
2. **The emission sources are redundant at the target's scale.** The road and building emission layers correlate 0.58
   at the scale of single cells, but 0.998–0.999 once smoothed over 8 km or more, the scale at which the target varies.

### 4.7 Physics alone is not enough; leakage was real but harmless

- **Physical operator.** The operator with one NO₂ yield per sector reaches Spearman 0.85 and PR-AUC 0.51–0.55 (0.85 and
  0.55 on 1404). This is much better than the original operator, which smoothed all sectors together (PR-AUC 0.35), but
  clearly below every SLX model with reach ≥ 12 km. Learning from urban-form and activity data is necessary; the graph
  is not.
- **Leakage.** Rebuilding the station layer without the test-block stations did not reduce skill. The pollution-only
  network scored 0.715 and 0.726 in PR-AUC, against 0.648 and 0.691 with the original pre-processing (differences within
  seed noise).

---

## 5. Discussion

### 5.1 Why reach, and not the graph, delivers skill

The monthly NO₂ pattern of Tehran is a smooth field: a correlation length of 24 km and about five effective degrees of
freedom over the city. Two physical processes make it so:

- **The instrument.** TROPOMI averages over pixels of 3.5 × 5.5 km.
- **Transport.** NOx is carried downwind during its daytime lifetime of a few hours. Beirle & Wagner (2024) report a mean
  of 2.4 h over 100 cities, of the same order as earlier megacity estimates (Beirle et al., 2011). At typical boundary-layer wind speeds of a few m s⁻¹, that is a transport scale of the order
  of 10–30 km.

The value of a cell is therefore determined by the emissions and urban form within a radius of the same order. Any model
that sees this neighbourhood can predict it. How the neighbourhood is combined matters little when the field is this
smooth: a box mean, a learned mean (GraphSAGE) and a learned non-linear combination give nearly the same answer.

Our skill-against-reach curves are the empirical counterpart. They rise until the model sees about half the correlation
length and flatten before it.

### 5.2 Implications for spatial deep learning

The graph-learning literature for urban emissions and air quality has grown rapidly, but its baselines are rarely given
the same reach as the proposed models. Our results suggest that part of the reported gains may be gains of reach.

Our results agree with scattered evidence and make it sharper:

- In Berlin, a random forest matched a spatial GNN for urban NO₂ (Schmidt & Ong, 2026).
- GNNs can be written as generalised spatial regressions (Liang et al., 2023; Zhu et al., 2026).
- Explicit spatial-lag inputs recover much of what spatial context offers (Xiao et al., 2023; Schmidinger et al., 2026).

What these studies leave open is whether the remaining gains of graph models come from the graph or from the size of
the neighbourhood. Holding reach equal answers this for a smooth target: the extra flexibility of the graph adds little.
The graph should matter where its edges carry information that a distance-based neighbourhood does not, such as climate
teleconnections (Chen et al., 2026) or flows on a transport network (Mohseni et al., 2026).

We recommend five practices:

1. **Reach-matched baselines.** Report a spatial-lag (SLX or buffer-based) baseline whose reach equals the model's
   receptive field. It costs seconds. Where the task gives no reach, set it for all models from the correlation length
   of the target.
2. **A skill-against-reach curve.** Report the reach at which skill saturates, and relate it to the correlation length of
   the target.
3. **Single seeds as well as ensembles.** Report single-seed results next to ensembles, or ensemble the baselines too.
4. **Inference built for few spatial blocks.** Use exact block randomisation tests, jackknife intervals and equivalence
   margins, and report the effective sample size of the target.
5. **No test information in pre-processing.** Build interpolated covariates inside the cross-validation, so that the test
   area never informs them.

### 5.3 Practical implications

For screening urban combustion hotspots from satellite data, the simplest adequate model is a linear regression on
neighbourhood means of emission and urban-form layers over about 10–25 km. It is transparent, fast and easy to transfer
to other cities with similar data. Its residuals can be read directly against the inventory. A GNN does not add
predictive value for this task in Tehran, although it may still be useful where non-local links matter (for example,
traffic flows across the network).

### 5.4 Limitations

- **One city.** The correlation length, and hence the needed reach, will differ between cities. The rule we propose
  ("match the reach to the correlation length") is general; the number is not.
- **No hyper-parameter search.** The models used the hyper-parameters fixed before the original study. A tuned GNN might
  gain a little, but the plateau of all model families at the same level makes a large gain unlikely.
- **A single held-out year, with stand-in inputs.** Its inventory and traffic inputs are those of the previous year. A
  second held-out year (1405) and a 50-trial search of the GNN were registered but not carried out.
- **NO₂, not CO₂.** NO₂ hotspots mark dense urban combustion. In Tehran they hold about a third of road and building CO₂
  but only about 1 % of industrial CO₂.
- **Column, not street level.** The satellite pattern agrees only weakly with the station network.
- **Low power.** The effective sample size limits every test. Some comparisons remain inconclusive rather than
  equivalent.

---

## 6. Conclusions

In mapping intra-urban combustion hotspots from TROPOMI NO₂ in Tehran, the skill of a graph neural network is explained
by how far it sees, not by the graph.

- Skill rises steeply until a model sees about 12 km and saturates within the target's 24-km correlation length.
- At equal reach, a linear spatial-lag model is as good as a multi-layer GraphSAGE model, also on a held-out year, at a
  fraction of the cost.
- The apparent advantages of the graph model came from a shorter-reach baseline, from averaging training seeds and from
  inference unsuited to a target with about five effective degrees of freedom.

We recommend reach-matched baselines and skill-against-reach curves as standard practice for spatial deep learning in
urban environmental mapping.

---

## Data and code availability

*[Draft: confirm the bracketed items before submission.]*

- **Code and results.** The code, configurations, run registry, pre-registrations, every result table and the figure
  scripts are openly available at https://github.com/fatemehvv/Tehran-carbon (branch `paper_TC`). A versioned copy
  will be archived on Zenodo on acceptance [DOI to be added].
- **Public data.** These are freely available. The Google Earth Engine scripts that export them on the study grid are
  in `gee/`.
  - TROPOMI NO₂ (Copernicus Sentinel-5P, offline level 3);
  - ERA5 reanalysis;
  - VIIRS night lights;
  - GHSL population and building volume;
  - SRTM elevation.
- **Energy and traffic statistics.** The annual and monthly statistics behind the emission inventory and the traffic
  layers come from public reports of Iranian agencies [names to be confirmed: Tehran Municipality Transportation and
  Traffic Organization; Ministry of Energy; National Iranian Oil Products Distribution Company].
  - They are listed with checksums in `data/raw/MANIFEST.csv`.
  - The derived 1-km inventory and traffic layers are [included in the repository / available from the corresponding
    author on request].
- **Station data.** Data from the municipal air-quality monitoring network [Tehran Air Quality Control Company] are
  [available from the data owner on request / redistributed with permission].

## CRediT authorship contribution statement

*[Draft: to be confirmed by both authors.]*

- **Fatemeh Vafaeian:** Conceptualization, Methodology, Software, Data curation, Formal analysis, Investigation,
  Visualization, Writing – original draft.
- **Vahid Khatibi:** Conceptualization, Methodology, Supervision, Validation, Writing – review & editing.

## Funding

[This research did not receive any specific grant from funding agencies in the public, commercial or not-for-profit
sectors.] *(Replace if there was funding.)*

## Acknowledgements

This publication contains modified Copernicus Sentinel data (2023–2026) and modified Copernicus Climate Change Service
information (2023–2026). Neither the European Commission nor ECMWF is responsible for any use that may be made of the
Copernicus information or data it contains. [We thank the Tehran Air Quality Control Company for the monitoring data.]

## Declaration of generative AI and AI-assisted technologies in the writing process

*[Draft: adapt it so that it describes exactly how the tool was used.]*

During the preparation of this work the authors used Claude (Anthropic) to help write and review analysis code and to
draft and edit parts of the text. After using this tool, the authors reviewed and edited the content as needed and take
full responsibility for the content of the published article.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have
appeared to influence the work reported in this paper.

---

## References

All DOIs were checked against Crossref (arXiv DOIs against arXiv) on 12 November 2026. Five entries have no DOI:
four conference papers (NeurIPS, ICLR, MLSys) and Holm (1979).

- Abémgnigni Njifon, M., & Schuhmacher, D. (2024). Graph convolutional networks for spatial interpolation of correlated data. *Spatial Statistics*, 60, 100822. https://doi.org/10.1016/j.spasta.2024.100822
- Ahn, D., Goldberg, D., Liu, F., Anderson, D., Coombes, T., Loughner, C., et al. (2025). Satellite-based analysis of CO₂ emissions from global cities: regional, economic, and demographic attributes. *AGU Advances*, 6, e2025AV001747. https://doi.org/10.1029/2025AV001747
- Alon, U., & Yahav, E. (2021). On the bottleneck of graph neural networks and its practical implications. *International Conference on Learning Representations (ICLR)*. arXiv:2006.05205. https://doi.org/10.48550/arXiv.2006.05205
- Bai, K., Li, K., Qiu, S., Zheng, Z., Jiao, P., Sun, Y., et al. (2025). SCAGAT: a scene-aware ensemble graph attention network for global PM2.5 pollution mapping via land–atmosphere interactions. *ISPRS Journal of Photogrammetry and Remote Sensing*, 225, 19–35. https://doi.org/10.1016/j.isprsjprs.2025.04.019
- Bao, L.-L., Zhang, C.-X., Zhang, J.-S., & Guo, R. (2024). A two-stage spatial prediction modeling approach based on graph neural networks and neural processes. *Expert Systems with Applications*, 258, 125173. https://doi.org/10.1016/j.eswa.2024.125173
- Behrens, T., Viscarra Rossel, R. A., Kerry, R., MacMillan, R., Schmidt, K., Lee, J., Scholten, T., & Zhu, A.-X. (2019). The relevant range of scales for multi-scale contextual spatial modelling. *Scientific Reports*, 9, 14800. https://doi.org/10.1038/s41598-019-51395-3
- Beirle, S., Boersma, K. F., Platt, U., Lawrence, M. G., & Wagner, T. (2011). Megacity emissions and lifetimes of nitrogen oxides probed from space. *Science*, 333, 1737–1739. https://doi.org/10.1126/science.1207824
- Beirle, S., Borger, C., Dörner, S., Li, A., Hu, Z., Liu, F., Wang, Y., & Wagner, T. (2019). Pinpointing nitrogen oxide emissions from space. *Science Advances*, 5, eaax9800. https://doi.org/10.1126/sciadv.aax9800
- Beirle, S., & Wagner, T. (2024). A new method for estimating megacity NOx emissions and lifetimes from satellite observations. *Atmospheric Measurement Techniques*, 17, 3439–3453. https://doi.org/10.5194/amt-17-3439-2024
- Bouthillier, X., et al. (2021). Accounting for variance in machine learning benchmarks. *Proceedings of Machine Learning and Systems*, 3.
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors. *Review of Economics and Statistics*, 90, 414–427. https://doi.org/10.1162/rest.90.3.414
- Chen, D., Yang, Y., Hu, J., Kasoar, M., Arcucci, R., & Cheng, S. (2026). Explainable global wildfire prediction model using graph neural networks. *Geodata and AI*, 7, 100052. https://doi.org/10.1016/j.geoai.2026.100052
- Du, S., Wei, H., Lee, D., Hu, Z., & Pan, S. (2026). Graph-based physics-guided urban PM2.5 air quality imputation with constrained monitoring data. *ACM Transactions on Sensor Networks*, 22, 1–23. https://doi.org/10.1145/3734869
- Fleischmann, M., & Arribas-Bel, D. (2024). Decoding (urban) form and function using spatially explicit deep learning. *Computers, Environment and Urban Systems*, 112, 102147. https://doi.org/10.1016/j.compenvurbsys.2024.102147
- Flores, M., Soliz, M., & Mollineda, R. A. (2026). Graph neural networks for soil organic carbon estimation: data-efficient modeling of multimodal spatial dependencies. *Machine Learning with Applications*, 24, 100878. https://doi.org/10.1016/j.mlwa.2026.100878
- Franklin, S. E., Wulder, M. A., & Lavigne, M. B. (1996). Automated derivation of geographic window sizes for use in remote sensing digital image texture analysis. *Computers & Geosciences*, 22, 665–673. https://doi.org/10.1016/0098-3004(96)00009-X
- Freitas, A., de Wit, X. M., Gabbana, A., Woodward, M., Toschi, F., Lin, Y. T., & Livescu, D. (2026). Learning turbulent transport via Mori–Zwanzig graph neural networks. arXiv:2606.14918. https://doi.org/10.48550/arXiv.2606.14918
- Gao, H., Yang, F., Yang, Y., Xu, Y., & Su, Y. (2025). Adaptive receptive field graph neural networks. *Neural Networks*, 190, 107658. https://doi.org/10.1016/j.neunet.2025.107658
- Goldberg, D. L., Tao, M., Kerr, G. H., Ma, S., Tong, D. Q., Fiore, A. M., et al. (2024). Evaluating the spatial patterns of U.S. urban NOx emissions using TROPOMI NO₂. *Remote Sensing of Environment*, 300, 113917. https://doi.org/10.1016/j.rse.2023.113917
- Griffith, D. A. (2005). Effective geographic sample size in the presence of spatial autocorrelation. *Annals of the Association of American Geographers*, 95, 740–760. https://doi.org/10.1111/j.1467-8306.2005.00484.x
- Gueterbock, F., Santos-Rodriguez, R., & Clark, J. N. (2025). Improving local air quality predictions using transfer learning on satellite data and graph neural networks. arXiv:2505.05479. https://doi.org/10.48550/arXiv.2505.05479
- Halleck Vega, S., & Elhorst, J. P. (2015). The SLX model. *Journal of Regional Science*, 55, 339–363. https://doi.org/10.1111/jors.12188
- Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. *Advances in Neural Information Processing Systems*, 30.
- Hettige, K. H., et al. (2024). AirPhyNet: harnessing physics-guided neural networks for air quality prediction. *International Conference on Learning Representations (ICLR)*.
- Hoek, G., Beelen, R., de Hoogh, K., Vienneau, D., Gulliver, J., Fischer, P., & Briggs, D. (2008). A review of land-use regression models to assess spatial variation of outdoor air pollution. *Atmospheric Environment*, 42, 7561–7578. https://doi.org/10.1016/j.atmosenv.2008.05.057
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6, 65–70.
- Huber, D., Kerr, G., Nawaz, M., Runkel, S., Anenberg, S., Goldberg, D., et al. (2026). Global NO₂ changes between 2019 and 2024 as observed by TROPOMI in urban areas and emerging hotspots. *Atmospheric Chemistry and Physics*, 26, 3783–3803. https://doi.org/10.5194/acp-26-3783-2026
- Kafi, A., Hosseinipoor, M., & Shahne, M. (2026). Evaluating the suitability of Sentinel-5P satellite data for air quality monitoring in Tehran. *Science of the Total Environment*, 1049, 182069. https://doi.org/10.1016/j.scitotenv.2026.182069
- Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*, 4, 100804. https://doi.org/10.1016/j.patter.2023.100804
- Klemmer, K., Safir, N., & Neill, D. B. (2023). Positional encoder graph neural networks for geographic data. *Proceedings of the 26th International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR 206. arXiv:2111.10144. https://doi.org/10.48550/arXiv.2111.10144
- Lakens, D. (2017). Equivalence tests: a practical primer for t tests, correlations, and meta-analyses. *Social Psychological and Personality Science*, 8, 355–362. https://doi.org/10.1177/1948550617697177
- Li, H., Qiu, J., Zhang, K., & Zheng, B. (2025). Monitoring fossil fuel CO₂ emissions from co-emitted NO₂ observed from space: progress, challenges, and future perspectives. *Frontiers of Environmental Science & Engineering*, 19, 2. https://doi.org/10.1007/s11783-025-1922-x
- Liang, Y., Ding, F., Huang, G., & Zhao, Z. (2023). Deep trip generation with graph neural networks for bike sharing system expansion. *Transportation Research Part C: Emerging Technologies*, 154, 104241. https://doi.org/10.1016/j.trc.2023.104241
- Linnenbrink, J., Nowosad, J., & Meyer, H. (2026). Moving beyond spatial and random cross-validation in environmental modelling: a call for prediction-domain adaptive evaluation. arXiv:2605.13689. https://doi.org/10.48550/arXiv.2605.13689
- Liu, P., Chen, Y., Liang, X., Li, H., Biljecki, F., & Stouffs, R. (2025). A graph neural network for small-area estimation: integrating spatial regularisation, heterogeneous spatial units, and Bayesian inference. *International Journal of Geographical Information Science*, 40, 2358–2396. https://doi.org/10.1080/13658816.2025.2597971
- Luo, W., Li, Y., Urtasun, R., & Zemel, R. (2016). Understanding the effective receptive field in deep convolutional neural networks. *Advances in Neural Information Processing Systems*, 29.
- Ma, X., Zhang, Z., & Song, Y. (2025). Geographically informed graph neural networks. *Spatial Statistics*, 69, 100920. https://doi.org/10.1016/j.spasta.2025.100920
- MacKinnon, J. G., Nielsen, M. Ø., & Webb, M. D. (2023). Cluster-robust inference: a guide to empirical practice. *Journal of Econometrics*, 232, 272–299. https://doi.org/10.1016/j.jeconom.2022.04.001
- Meyer, H., Reudenbach, C., Hengl, T., Katurji, M., & Nauss, T. (2018). Improving performance of spatio-temporal machine learning models using forward feature selection and target-oriented validation. *Environmental Modelling & Software*, 101, 1–9. https://doi.org/10.1016/j.envsoft.2017.12.001
- Meyer, H., & Pebesma, E. (2022). Machine learning-based global maps of ecological variables and the challenge of assessing them. *Nature Communications*, 13, 2208. https://doi.org/10.1038/s41467-022-29838-9
- Milà, C., Mateu, J., Pebesma, E., & Meyer, H. (2022). Nearest neighbour distance matching leave-one-out cross-validation for map validation. *Methods in Ecology and Evolution*, 13, 1304–1316. https://doi.org/10.1111/2041-210X.13851
- Mohseni, G., Nourinejad, M., & Park, P. Y. (2026). Bike-sharing ridership prediction for network expansion using graph neural networks. *Transportation Research Part A: Policy and Practice*, 203, 104701. https://doi.org/10.1016/j.tra.2025.104701
- Ploton, P., Mortier, F., Réjou-Méchain, M., Barbier, N., Picard, N., Rossi, V., et al. (2020). Spatial validation reveals poor predictive performance of large-scale ecological mapping models. *Nature Communications*, 11, 4540. https://doi.org/10.1038/s41467-020-18321-y
- Reuter, M., Buchwitz, M., Schneising, O., Krautwurst, S., O'Dell, C. W., Richter, A., et al. (2019). Towards monitoring localized CO₂ emissions from space: co-located regional CO₂ and NO₂ enhancements observed by the OCO-2 and S5P satellites. *Atmospheric Chemistry and Physics*, 19, 9371–9383. https://doi.org/10.5194/acp-19-9371-2019
- Rey-Pommier, A., Héraud, A., Chevallier, F., Ciais, P., Christoudias, T., Kushta, J., & Sciare, J. (2025). Global gridded NOx emissions using TROPOMI observations. *Earth System Science Data*, 17, 3329–3351. https://doi.org/10.5194/essd-17-3329-2025
- Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40, 913–929. https://doi.org/10.1111/ecog.02881
- Schmidinger, J., Barkov, V., Vogel, S., Atzmueller, M., & Heuvelink, G. B. M. (2026). Kriging prior regression: a case for kriging-based spatial features with TabPFN in soil mapping. *Computers and Electronics in Agriculture*, 243, 111352. https://doi.org/10.1016/j.compag.2025.111352
- Schmidt, R., & Ong, L. L. S. (2026). From leaves to breezes: machine learning based prediction of nitrogen dioxide concentration from surrounding urban greenery and meteorological, spatial, and traffic characteristics in Berlin, Germany. *Ecological Informatics*, 93, 103568. https://doi.org/10.1016/j.ecoinf.2025.103568
- Shao, W., Tu, Y., Yue, T., Zhao, N., Zhou, H., Zhang, L., et al. (2026). Mapping Chinese urban carbon emissions with a graph-based multi-branch deep learning framework. *GIScience & Remote Sensing*, 63, 2631837. https://doi.org/10.1080/15481603.2026.2631837
- Teng, M., Liang, M., Wang, S., & Ding, Y. (2026). Seamless hourly PM2.5 mapping across China with a graph spatiotemporal deep neural network. *Environmental Modelling & Software*, 198, 106915. https://doi.org/10.1016/j.envsoft.2026.106915
- Valavi, R., Elith, J., Lahoz-Monfort, J. J., & Guillera-Arroita, G. (2019). blockCV: an R package for generating spatially or environmentally separated folds for k-fold cross-validation of species distribution models. *Methods in Ecology and Evolution*, 10, 225–232. https://doi.org/10.1111/2041-210X.13107
- Wadoux, A. M. J.-C., Heuvelink, G. B. M., de Bruin, S., & Brus, D. J. (2021). Spatial cross-validation is not the right way to evaluate map accuracy. *Ecological Modelling*, 457, 109692. https://doi.org/10.1016/j.ecolmodel.2021.109692
- Wang, S., Li, Y., Zhang, J., Meng, Q., Meng, L., & Gao, F. (2020). PM2.5-GNN: a domain knowledge enhanced graph neural network for PM2.5 forecasting. *Proceedings of the 28th International Conference on Advances in Geographic Information Systems (SIGSPATIAL)*, 163–166. https://doi.org/10.1145/3397536.3422208
- Wang, S., Zhang, Y., Lin, X., Piao, X., Hu, Y., & Yin, B. (2025). CMDNet: a cross-modality spatiotemporal graph network for enhanced air pollution prediction with high-resolution satellite data. *IEEE Transactions on Geoscience and Remote Sensing*, 63, 1–14. https://doi.org/10.1109/TGRS.2025.3569594
- Xiao, S., Song, Y., & Wang, Z. (2023). Nonparametric spatial autoregressive model using deep neural networks. *Spatial Statistics*, 57, 100766. https://doi.org/10.1016/j.spasta.2023.100766
- Yap, W., Wu, A. N., Miller, C., & Biljecki, F. (2025). Revealing building operating carbon dynamics for multiple cities. *Nature Sustainability*, 8, 1199–1210. https://doi.org/10.1038/s41893-025-01615-8
- Yuan, D., Wang, Z., Huang, H., Fu, Z., Wu, Q., Jhang, S., & Pedrycz, W. (2026). Spatially directional dual-attention graph networks for fine-grained PM2.5 mapping. *Applied Soft Computing*, 203, 116231. https://doi.org/10.1016/j.asoc.2026.116231
- Zeng, J., Liu, Y., Ding, J., Yuan, J., & Li, Y. (2024). Estimating on-road transportation carbon emissions from open data of road network and origin-destination flow data. *Proceedings of the AAAI Conference on Artificial Intelligence*, 38, 22493–22501. https://doi.org/10.1609/aaai.v38i20.30257
- Zeng, J., Liu, Y., Zhang, G., Ding, J., Lin, Y., Yuan, J., & Li, Y. (2025). OpenCarbon: a contrastive learning-based cross-modality neural approach for high-resolution carbon emission prediction using open data. *Proceedings of the 34th International Joint Conference on Artificial Intelligence (IJCAI)*, 9999–10007. https://doi.org/10.24963/ijcai.2025/1111
- Zhan, W., & Datta, A. (2024). Neural networks for geospatial data. *Journal of the American Statistical Association*, 120, 535–547. https://doi.org/10.1080/01621459.2024.2356293
- Zhang, R., Bao, Z., Bai, S., Sumiya, E., Dalantai, S., & Bao, Y. (2026). Advances and challenges of machine learning in satellite-based atmospheric NO₂ monitoring. *Atmospheric Pollution Research*, 17, 103066. https://doi.org/10.1016/j.apr.2026.103066
- Zhao, M., Šćepanović, S., Law, S., Obadić, I., Wu, C., & Quercia, D. (2026). UST-GNN: a unified spatial–topological graph neural network framework for urban analytics demonstrated through a case study on urban health prediction. *Computers, Environment and Urban Systems*, 129, 102466. https://doi.org/10.1016/j.compenvurbsys.2026.102466
- Zhu, D., Liu, Y., Yao, X., & Fischer, M. M. (2021). Spatial regression graph convolutional neural networks: a deep learning paradigm for spatial multivariate distributions. *GeoInformatica*, 26, 645–676. https://doi.org/10.1007/s10707-021-00454-x
- Zhu, D., Wang, S., & Luo, P. (2026). Uncovering spatial process heterogeneity from graph-based deep spatial regression. *ISPRS Journal of Photogrammetry and Remote Sensing*, 232, 509–523. https://doi.org/10.1016/j.isprsjprs.2025.12.008

---

## Tables and figures

**Table 1.** Reach of GNNs and their baselines in comparable studies (Section 1).

**Table 2.** Data sources: the predictor table of Section 2.3 with sources and years (to be laid out from
`configs/config.yaml`, `data/raw/MANIFEST.csv`).

**Table 3.** Cross-validated skill against reach (Section 4.1).

**Table 4.** GNN minus reference (Section 4.2).

**Table S1 (supplementary).** All comparisons with jackknife and bootstrap intervals, randomisation p-values and verdicts:
`docs/paper/results_stageA/evaluate_paper2/` and `evaluate_thesis/`.

**Figures.** All figures share one style: the same colours for the same models in every figure (GNN blue, SLX-OLS
orange, SLX-GBM aqua, SLX-RF yellow, no-graph network grey), grey text and recessive axes. Each is saved as PNG (300
dpi) and PDF (vector) in `docs/paper/figures/`. Figures 3–6 are drawn from the committed result tables by
`python main.py paper figures`; nothing is re-run.

| Figure | Content | File | Source |
|---|---|---|---|
| 1 | From the literature to the gap (funnel) | `fig1_literature_funnel_en.svg` / `.png` | drawn; Persian version `fig1_literature_funnel_fa` for the thesis |
| 2 | Domain, observed and predicted target, hotspots of 1404 | `docs/figures/fig06_maps_1404.png` | thesis figures (`python main.py figures`) |
| 3 | Correlogram of the target | `fig3_correlogram` | `results_stageA/target/target_neff.csv`, `reach/reach_correlogram.csv` |
| 4 | Skill against reach | `fig4_reach` | `results_stageA/reach/reach_curve.csv` |
| 5 | GNN minus reference, jackknife intervals, equivalence band | `fig5_differences` | `results_stageA/evaluate_paper2/` (primary rows) |
| 6 | Single seeds, ensemble and SLX references | `fig6_seeds` | the same tables (`part = seeds` rows) |

The numbers behind Figures 5 and 6 are written to `docs/paper/figures/figure_data_comparisons.csv`. Fig. 1 can also
serve as the graphical abstract.

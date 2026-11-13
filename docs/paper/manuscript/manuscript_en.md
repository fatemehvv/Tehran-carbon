# Reach, not graphs: neighbourhood reach explains the skill of distance-based graph neural networks in mapping intra-urban NO₂ hotspots from TROPOMI

**Fatemeh Vafaeian¹, Vahid Khatibi¹**\*

¹ [Affiliation to be completed]  
\* Corresponding author: [e-mail to be completed]

*Draft manuscript, 13 November 2026. All numbers come from the runs on branch `paper_TC` of the project repository (see
Data and code availability). Items in [brackets] are to be completed before submission.*

---

## Highlights

- On a distance-based graph, GNN skill for NO₂ hotspots tracks reach.
- At 22.7 km, the same network without a graph matched the GNN.
- Spatial-lag skill rose steeply up to about 12 km of reach, then gained little.
- Fusion raised hotspot PR-AUC within seed noise but lowered rank agreement.
- Seed averaging and the choice of inference changed several verdicts.

---

## Abstract

The gains of graph neural networks (GNNs) in mapping urban emissions and air pollution are usually credited to
learning on a graph, yet a GNN also sees further than its baselines. We separate the two in mapping intra-urban NO₂ hotspots in Tehran from the monthly TROPOMI NO₂ pattern on a 1-km grid
(2,436 cells), asking whether fusing traffic, energy and station data improves the maps (RQ1) and whether a GNN
improves on conventional spatial models (RQ2). A multi-scale GraphSAGE model on a distance-based graph is compared
with spatial-lag-of-X (SLX) models, and with the same network fed box means of its neighbourhood instead of a graph, at
matched maximum reach from 0 to 38 km. Validation uses 22-km spatial blocks, a held-out year,
fold-specific station interpolation, five seeds and block randomisation tests. Skill depends mainly on reach: the
best SLX models rise from a Spearman correlation of about 0.75 for the cell alone to 0.91–0.93 at 12.6 km and gain at
most 0.03 beyond. The GNN's lead over SLX at equal reach shrinks from about 0.04 at 7.6 km to −0.004 and +0.009 at
37.8 km, all comparisons inconclusive, and the network fed box means is statistically equivalent to the GNN at 22.7 km
(Spearman differences −0.005 and −0.008). On the held-out year the GNN and SLX are equivalent in Spearman at 30.3 km and
differ by 0.002 at 37.8 km. The GNN's PR-AUC leads shrink for individual fits. Fusion raised PR-AUC within seed noise and lowered Spearman; the main emission layers become
nearly identical at the scale over which the target varies. With about five effective degrees of
freedom, the choice of inference changed several verdicts. For this smooth target and distance-based
graph, reach explains most of the GNN's skill; graph models need reach-matched baselines.

**Keywords:** graph neural networks; receptive field; spatial lag of X; TROPOMI; NO₂; data fusion; spatial
cross-validation; Tehran

---

## 1. Introduction

Policies to cut urban fossil CO₂ emissions depend on knowing *where* in a city combustion is concentrated. Satellite NO₂ has
become a practical tracer of urban fossil combustion: it is co-emitted with CO₂, it is short-lived, and since 2018 the
TROPOMI instrument on Sentinel-5P has mapped it daily at a resolution of a few kilometres (Beirle et al., 2019; Li et
al., 2025). It has been used to locate urban CO₂ plumes and estimate city emissions (Reuter et al., 2019; Ahn et al.,
2025), to derive gridded NOx emissions (Rey-Pommier et al., 2025) and to evaluate kilometre-scale inventories (Goldberg
et al., 2024). Models that link emission and urban-form data to the observed NO₂ pattern can therefore screen cities
for combustion hotspots, test inventories and support scenario analysis.

Two choices shape such models. The first is which data to use. Bottom-up inventories, traffic models and ground
stations each describe part of urban combustion, and fusing them is an obvious way to try to improve a map. The second
is which model to use, and here graph neural networks have become the method of choice. Recent models learn urban
carbon emissions from satellite images and points of interest (Zeng et al., 2025), from road networks and
origin–destination flows (Zeng et al., 2024), at the level of individual buildings (Yap et al., 2025) and across 329
Chinese cities (Shao et al., 2026). GNNs with physically motivated edges are used for air-quality prediction (Wang et
al., 2020; Hettige et al., 2024), graph networks have been fused with high-resolution satellite data to predict PM2.5
and NO₂ (Wang et al., 2025) and to map PM2.5 from regional to global scales (Bai et al., 2025; Teng et al., 2026; Yuan et
al., 2026), and GraphSAGE has been applied to urban NO₂ (Gueterbock et al., 2025). The gains of these models are
usually credited to the graph, that is, to learning how information should flow between neighbouring places.

A graph model, however, differs from a conventional model in two ways at once. It *learns* how to combine its
neighbourhood, and it *sees* a neighbourhood whose size grows with every message-passing layer: its receptive field,
which we call its reach (Luo et al., 2016). Within graph learning the receptive field is treated as a design choice to
be tuned or learned (Gao et al., 2025), and convolutional models fail where it is too small for the process (Chen et
al., 2026). The baselines against which GNNs are judged, by contrast, often see no neighbourhood at all, or a much
smaller one. Under such comparisons an advantage of the GNN may simply mean that it looks further. The conventional
answer to "the neighbours matter" is the spatial lag of X (SLX) model of spatial econometrics, which adds neighbourhood
means of the predictors (Halleck Vega & Elhorst, 2015), and the buffer variables of land-use regression do the same in
air-quality science (Hoek et al., 2008). Both can be given any reach at negligible cost, and explicit neighbourhood
information of this kind improves neural networks (Xiao et al., 2023), convolutional classifiers of urban form
(Fleischmann & Arribas-Bel, 2024) and tabular learners (Schmidinger et al., 2026).

Research on graph learning has addressed a related question. On citation and social graphs, linear models applied to
features averaged over the same number of hops matched graph convolutional networks (Wu et al., 2019; Frasca et al.,
2020); the receptive field can be set apart from the number of layers (Gasteiger et al., 2019; Zeng et al., 2021); and
simple, fairly tuned baselines often match GNNs, with evaluation choices changing the ranking of models (Shchur et al.,
2018; Errica et al., 2020; Huang et al., 2021). A spatial-regression reading of graph convolution relates the number of
layers to the order of the spatial lag (Zhu et al., 2021). These results concern non-spatial graphs, were obtained
without spatial validation and did not involve geographic spatial-lag baselines.

In geospatial and environmental work, comparisons of GNNs with spatial regression are common, but their baselines
differ in reach from the GNN. Graph convolutional networks have been compared with spatial and geographically weighted
regression (Zhu et al., 2021; Liu et al., 2025), deep spatial regression on a graph fitted house prices better than
spatial-regression baselines in sample (Zhu et al., 2026), and GNNs have beaten spatial-lag, geographically weighted,
gradient-boosting and linear baselines for health outcomes and bike-share ridership (Zhao et al., 2026; Mohseni et al.,
2026). GNNs can be read as generalised spatial regressions (Liang et al., 2023; Ma et al., 2025), and neural networks
for geospatial data relate closely to kriging (Zhan & Datta, 2024; Abémgnigni Njifon & Schuhmacher, 2024). Simple
baselines remain strong: in Berlin a random forest matched a spatial GNN for urban NO₂ (Schmidt & Ong, 2026), a graph
model of city CO₂ did not beat an MLP or XGBoost until extra constraints were added (Shao et al., 2026), and simple
interpolation beat data-driven GNNs for urban PM2.5 (Du et al., 2026). Evaluation matters as much: GNN residuals can
remain spatially autocorrelated (Bao et al., 2024), a GNN can fail under leave-one-cluster-out validation (Flores et al.,
2026), reported skill for satellite NO₂ varies with the validation strategy (Zhang et al., 2026), random
cross-validation overstates skill for autocorrelated targets (Roberts et al., 2017; Ploton et al., 2020; Meyer & Pebesma,
2022; Wadoux et al., 2021; Milà et al., 2022; Linnenbrink et al., 2026), bootstrap inference is unreliable with few,
unequal clusters (Cameron et al., 2008; MacKinnon et al., 2023), deep models vary from seed to seed (Bouthillier et al.,
2021), and pre-processing that sees the test area inflates skill (Kapoor & Narayanan, 2023).

Table 1 compares the reach of GNNs and their baselines in eleven studies whose full texts describe both. In most of them, the baselines use only their own location, a neighbourhood of a different size, or one whose size is not reported. Two studies come close. In Liang et al. (2023) a linear SLX baseline averaged over the same nearest stations
as the GNN, but the nonlinear baseline saw no neighbours. In Schmidt & Ong (2026) the random forest received the
measured NO₂ of the other stations, weighted as in the graph, and then matched the GNN. Neither study made reach the
object of the comparison, varied it, or tied it to the scale of the target. A structured ScienceDirect search (five
Boolean queries, 191 unique records) and a semantic search of the wider literature found no study that does
(`docs/paper/lit/`). Choosing a neighbourhood from the range of spatial dependence is itself established practice, for
texture windows (Franklin et al., 1996), contextual covariates (Behrens et al., 2019), cross-validation blocks (Valavi
et al., 2019), GNN depth (Alon & Yahav, 2021) and the receptive field of a GNN for turbulent transport (Freitas et al.,
2026), but it has not been used to give a GNN and its baselines the same reach. Fig. 1 summarises how the literature
narrows to this gap.

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
| Schmidt & Ong (2026) | Hourly NO₂; 16 stations, Berlin (3 held out) | Graph convolution, contrastive embedding and attention over all stations; edges weighted by similarity of distance to the city centre | RF: buffers of 25–500 m, plus the NO₂ of all other stations with the same weights (its most important feature) | Roughly (same weights, lag of the measured NO₂); R² 0.58 vs 0.61 |
| Mohseni et al. (2026) | Station-to-station ridership; bike share, Toronto | GraphSAGE, 6 layers (6 hops) | Linear regression, XGBoost, ANN: own-station features (points of interest within 2.5 km). Spatial regression: weights n.r. | No |
| Zhu et al. (2026) | House prices; census block groups, Boston | 3 graph-convolution modules on a 20-nearest-neighbour graph (3 hops) | OLS: none. SAR: predefined adjacency (linear). GWR: 64 nearest neighbours | Linear baselines only; in-sample fit, no held-out test |
| Shao et al. (2026) | Annual CO₂; 329 cities, China | Graph convolution over all cities, weights 1/distance; depth n.r. | Linear regression on night lights, MLP, XGBoost: own-city features only | No; the graph model without added constraints did not beat them |
| **This study** | NO₂ hotspots; 2,436 cells of 1 km, Tehran | GraphSAGE on a distance-based graph, 3 layers × 9 cells: 22.7 km (4 and 5 layers, 30.3 and 37.8 km, on the held-out year) | SLX-OLS, SLX-GBM, SLX-RF: 0–37.8 km. Same network without a graph: box means at 0, 22.7 and 37.8 km | **Yes**: SLX reach matched to the GNN's, and varied |

[Rows to be added once the full texts of Ma et al. (2025), Teng et al. (2026), Yuan et al. (2026) and Bai et al. (2025)
have been read.]

We address two research questions for intra-urban NO₂ hotspots observed by TROPOMI in Tehran, a megacity of about nine
million people at the foot of the Alborz mountains with one of the highest NO₂ columns among large cities worldwide
(Huber et al., 2026). RQ1 asks whether fusing traffic, energy and station data improves the maps over the best single
source. RQ2 asks whether a GNN improves on conventional spatial models, and whether any improvement comes from the graph
or from reach. Our contribution is a reach-matched evaluation: SLX baselines, and the same network without a graph, are
given the GNN's reach, and reach is varied for all of them, under spatial block cross-validation and on a held-out
year. Matching is by maximum nominal reach: the models see up to the same distance, though not always the same cells. We
relate the reach at which skill levels off to the correlation length of the target, quantify how seed
averaging and the choice of inference change verdicts in a target with few effective degrees of freedom, and explain
the RQ1 result by the same scale argument. Decision rules for the main comparisons were registered before the
corresponding runs (Section 3.6).

![Figure 1](../figures/fig1_literature_funnel_en.png)

**Fig. 1.** From the literature to the gap. Satellite NO₂ is used as a tracer of urban combustion; deep learning, and
graph neural networks in particular, map urban emissions and air pollution, and their gains are credited to the graph.
A GNN, however, also sees further than its baselines, and none of the audited studies varied the reach of a GNN and its
baselines together under spatial validation.

---

## 2. Study area and data

### 2.1 Grid, years and label domain

The analysis grid covers 50.85–51.90°E and 35.35–35.95°N with 72 × 126 cells of 1/120° (about 0.75 × 0.93 km). The label domain includes the 2,436 cells of the three central counties (Tehran, Rey, Shemiranat) to which the emission inventory assigns any CO₂. This excludes mountains and desert, so that a model cannot gain skill simply by separating urban from non-urban cells. We use three Iranian (Solar Hijri) years: 1402 (21 March 2023 – 19 March 2024) and 1403 (20 March 2024 – 20
March 2025) for development, and 1404 (21 March 2025 – 20 March 2026) as a held-out year. Fig. 2 shows the domain, the
observed target of the held-out year and its hotspots.

![Figure 2](../../figures/fig06_maps_1404.png)

**Fig. 2.** The label domain and the target on the held-out year 1404: observed NO₂ pattern (z, annual mean of the
monthly maps), the out-of-fold prediction of the GNN, and the hotspots (top 10 %) found by both, by the observation only
and by the prediction only.

### 2.2 Target

The target is the monthly mean tropospheric NO₂ column of the TROPOMI offline level-3 product (Google Earth Engine,
`COPERNICUS/S5P/OFFL/L3_NO2`) on the grid. For each month the log column is standardised over the domain (z), so that
only the spatial pattern is kept, and a hotspot is a cell in the top 10 % of z in that month. The standardisation and
the threshold use all domain cells of the month and thus define the target as a city-wide ranking; they leave the rank
metrics of Section 3.4 unchanged but also set the scale of the training targets in every fold. The target is
deliberately not the inventory: using a modelled CO₂ map would let a model reproduce the inventory's own allocation.

The target is an extremely smooth field. The monthly correlogram is 0.95 at 0–2 km and 0.76 at 8–10 km and crosses zero at 24 km (Fig. 3). The effective number of independent cells (Griffith, 2005) is 4.9 in both development years,
against 2,436 cells and about 88 TROPOMI nadir pixels (3.5 × 5.5 km). The pattern is also stable between years: the Spearman correlation is 0.993–0.997 between the annual mean maps of any two years and 0.955–0.963 on average between the same months. A target built from daily scenes with cloud fraction below 0.3 agrees with the standard one
(Spearman 0.98–0.99; overlap of hotspot sets 0.73–0.76). The north of the city lies about 700 m higher than the south,
and a TROPOMI emission inversion reported terrain artefacts near Mount Tochal (Rey-Pommier et al., 2025); elevation
correlates with z (|ρ| = 0.49), but partialling it out barely changes model skill (Spearman 0.949 → 0.940). The
satellite pattern agrees only weakly with the ranking of the 16–17 stations with NO₂ records inside the grid (mean
monthly Spearman 0.11 in 1402 and 0.30 in 1403), so it describes column NO₂ rather than street-level concentrations
(cf. Kafi et al., 2026).

![Figure 3](../figures/fig3_correlogram.png)

**Fig. 3.** Correlogram of the monthly target in 1402 and 1403 (2-km distance bins). The curve crosses zero at 24.0 km;
the effective number of independent cells is 4.9 of 2,436. Grey band: one TROPOMI pixel (3.5 × 5.5 km).

### 2.3 Predictors

**Table 2.** Predictor groups. [Sources, years and processing of every layer to be listed in Supplementary Table S2,
from `configs/config.yaml` and `data/raw/MANIFEST.csv`.]

| Group | Layers |
|---|---|
| **Traffic** | Road-transport CO₂ of a bottom-up inventory (IPCC Tier 1 with Monte Carlo uncertainty, allocated to 1 km); road length by class; peak-hour vehicle-kilometres and volume-to-capacity ratio of a traffic assignment |
| **Energy** | Residential, commercial, industrial and power-sector CO₂ of the same inventory; monthly heating degree-days |
| **Pollution** | CO from the 28 stations of the municipal monitoring network, interpolated by inverse distance |
| **Context** | GHSL population and building volume, VIIRS night lights, industrial land share, monthly ERA5 temperature and wind speed |

Monthly sector layers are the annual allocation multiplied by monthly sector shares. No inventory is yet available for 1404, so all models use the 1403 inventory and traffic layers.

---

## 3. Methods

### 3.1 Models

All models use the same target and, unless stated, the same predictors (Table 2).

| Model | Description |
|---|---|
| **Own-cell models** | Ordinary least squares (OLS), histogram gradient boosting (GBM) and random forests (RF) on the predictors of the cell itself |
| **SLX models** | The same estimators plus box means of every spatially varying predictor over nested square windows of half-width up to 1, 3, 9, 15, 18, 27, 36 or 45 cells. This is the SLX specification of spatial econometrics and the buffer design of land-use regression. The windows of the original study went up to 15 cells. |
| **No-graph network** | One encoder per input group, a softmax gate over the groups (gated fusion) and a linear head. Given either no neighbourhood information or the same box means as the SLX models. |
| **Multi-scale GraphSAGE (GNN)** | The same gated encoder followed by L residual GraphSAGE layers (Hamilton et al., 2017) with mean aggregation over three edge sets: the 8 neighbours at distances of 1, 3 and 9 cells. One layer reaches 9 cells; L layers reach 9L cells. |
| **Physical operator** | Gaussian-smoothed inventory layers with one non-negative NO₂ yield per sector, width and yields fitted on the training cells |

The edges of the GNN join cells at fixed grid distances and carry no information beyond distance, so the network is
close to a dilated convolution with learned, nonlinear aggregation. The comparison therefore tests learned against
fixed aggregation up to the same distance; it does not test graphs whose edges carry other information, such as road
links, observed travel flows or wind. The no-graph network fed box means is the decisive control: it shares the GNN's encoder, training, seeds and
ensembling, and supplies spatial context as fixed box means instead of learned message passing. The two also differ in
which cells contribute (all cells of a square against the sparse graph stencils), in how they are weighted, and in
whether aggregation comes before or after the nonlinear encoder, so the comparison tests two ways of supplying the
same maximum reach rather than one component of the GNN in isolation.

Message passing and SLX windows both use the predictors of every grid cell, including those of the held-out block; only
the target of held-out cells is unseen. This is the mapping task, in which predictors are known everywhere. All
hyper-parameters were fixed before the runs of the original study and none was tuned here. The neural models use hidden
size 32, two encoder layers, dropout 0.1, AdamW, a learning rate of 3 × 10⁻³ and 300 epochs without early stopping. Every neural model is trained with five seeds. As in the original study, the five predictions are averaged to form the five-seed ensemble, and each individual fit is also scored on its own. OLS is deterministic, and GBM and RF are run with one seed. The
fusion networks of RQ1 (Section 4.6) are no-graph networks with box means at half-widths of 3 and 9 cells (reach
7.6 km).

### 3.2 Reach and reach-matched comparisons

The reach of a model is how far, in cells, a predictor value can be from a cell and still influence its prediction: the
largest window half-width for SLX models and the no-graph network, 9L cells for a GNN with L layers, and zero for
own-cell models. Reach is counted in cells on the same grid for every model, and both box windows and graph stencils
are square, so reach is a Chebyshev distance. A box window covers every cell within its reach; the graph reaches the
same distance through sparse stencils composed over layers, so it does not draw on every cell inside the square.
Distances in kilometres are nominal and use the mean cell size (0.84 km). Reach is the largest distance a model can use; its effective receptive field may be shorter (Luo et al., 2016). We call a comparison reach-matched when the models
have the same reach in cells. The GNN was run with one to five layers (7.6–37.8 km) in cross-validation and with four
and five layers on the held-out year; SLX models were run at 0–37.8 km, and the no-graph network at 0, 22.7 and
37.8 km.

As a heuristic for the reach a task needs, we use the distance R\* at which the correlogram of the target first crosses
zero (24.0 km; Section 2.2), close to the 27-cell (22.7 km) reach of a three-layer GNN. R\* is not a
physical constant. The target is standardised over a bounded domain and has a large-scale gradient, so its zero
crossing depends on both: it falls to 19.3–19.4 km after removing a planar trend, to 13.9–14.1 km after removing a
quadratic trend, and to 21.5–21.7 km and 18.0–18.3 km on domains cropped to 80 % and 60 % of their extent. R\* was
computed after the first runs, so results near it are consistent with the heuristic but do not test it blindly.

### 3.3 Validation

The domain is divided into 22-km square blocks (the autocorrelation range estimated for the original design), giving
12 blocks. Each block is held out once, and training cells within 2 km of it are removed. For the held-out year, each fold trains on the training blocks of 1402 and 1403 and predicts the held-out block in 1404, which tests prediction at new locations in a new year (Meyer et al., 2018). Because the target barely changes between years (Section 2.2), the held-out year tests stability
more than new information.

The station layer was originally interpolated once from all stations, including those inside the test block. Here it is
rebuilt for every fold from the stations farther than 2 km from the test block; in one fold this removes 19 of the 28
stations. The 2-km buffer is short compared with the correlation length, so cross-validation measures interpolation
within the city more than extrapolation. We therefore also ran the reach-matched comparison with buffers of 10 and 20 km.
With a 20-km buffer, no station remains in 4 of the 12 folds. These runs therefore use only the driver inputs (traffic, energy and context), and the 2-km runs were repeated with the same inputs for comparison.

### 3.4 Metrics

The primary metric is the Spearman correlation between predicted and observed z over the domain cells, averaged over
months. PR-AUC, the area under the precision–recall curve of the hotspot label (chance level 0.10), is reported as a descriptive measure, because the hotspots of a month form a few contiguous areas and PR-AUC therefore rests on very few independent units. Sensitivity analyses use hotspot labels at the top 20 % and top 5 % and the same metrics on superpixels of 5 × 5
cells, about one TROPOMI pixel. To measure how much of the skill is due to the smoothness of the target alone, we also
interpolate the target of the training cells of the same month, with the same folds and buffer, by inverse-distance
weighting and by a Gaussian process on the coordinates.

### 3.5 Inference with few spatial blocks

Differences between two models are computed from their out-of-fold predictions for the same cells and months. The main test is a block sign-flip
randomisation test that enumerates all 2¹² = 4,096 assignments: after rank-normalising the predictions within each month,
the predictions of the two models are swapped block by block, with the same swap in every month, and the city-wide
difference of the metric is recomputed. Enumeration makes the test exhaustive, but its validity rests on the blocks
being exchangeable under the null hypothesis, which dependence between neighbouring blocks only approximates (see
below). All 12 blocks enter this test, also for PR-AUC, because blocks without hotspots still contribute negative cells;
as a check, the test is also applied to the cell-weighted mean of per-block differences. Intervals are delete-one-block
jackknife intervals with K − 1 degrees of freedom: each block is removed in turn and the city-wide difference is
recomputed from the other eleven.

Each comparison receives one of five verdicts, applied in this order: superior (95 % interval above zero), equivalent
(90 % interval inside ±δ, i.e. two one-sided tests; Lakens, 2017), non-inferior (95 % lower bound above −δ), inferior
(95 % interval below zero) or inconclusive. Equivalence is checked before inferiority, so a difference whose 95 %
interval excludes zero but whose 90 % interval lies inside ±δ is called equivalent: it is detectable but smaller than
the margin. The registration set δ = 0.02 for Spearman, the larger of 0.02 and twice the seed-to-seed standard
deviation measured then (0.0077); 0.02 is about a tenth of the gain from no neighbourhood to a reach of 12.6 km. PR-AUC
verdicts use the same margin but are descriptive. The verdicts come from the intervals; Holm's adjustment (Holm, 1979)
is applied to the randomisation p-values within each family: the 22 primary comparisons of the registered family, and
separately the exploratory comparisons of this study. Table S1 lists every comparison with both interval levels,
unadjusted and adjusted p-values and family. The block bootstrap of the original study is reported for comparison.

The correlation length (24 km) exceeds the block size, so neighbouring blocks are dependent and the nominal level of
the randomisation test is only approximate. As a diagnostic, we applied the test to 20 comparisons between pairs of
individual fits of the same GNN (five seeds, two years), which have no true difference in expectation; the pairs share
data and seeds, so this checks calibration only roughly. For PR-AUC, 1 of the 20 comparisons fell below p = 0.05, as
expected; for Spearman, 5 of 20 did. The test is therefore liberal for Spearman, and we read significant Spearman
p-values with caution. A liberal test does not make non-significant results more precise; the precision of each
comparison is given by its interval. The effective number of independent cells of the target (Section 2.2) describes
the target field; it indicates, but does not equal, the information available to compare two models.

### 3.6 Pre-registration and transparency

This paper re-examines the model comparison of a master's thesis, the "original study" (Vafaeian, 2026), whose
models, windows, folds and hyper-parameters are reused unless stated. Its decision rules, including a held-out-year
test on 1404, were written before any model was evaluated. The checks of this paper (reach-matched SLX models, the
leakage control, the physical operator and the inference rules) were registered before the corresponding runs, and the
registrations are timestamped by the project's version control, with deviations listed in the registration documents.
A planned second held-out year (1405) and a hyper-parameter search were registered but not carried out. The GNN depth
curve, the no-graph network with box means, the short SLX windows, the larger buffers and the interpolation baselines
were added after the first results and are exploratory. All runs used one RTX 4080 GPU, and every run is logged with its
configuration hash, data hash and code version.

---

## 4. Results

### 4.1 Skill rises with reach and levels off

Every model family improves steeply as it is allowed to see further, and then gains much less (Fig. 4, Table 3). Models
restricted to their own cell reach a Spearman correlation of 0.66–0.77. SLX models gain little from the nearest cells
(0.73–0.79 at 0.8–2.5 km), reach 0.86–0.87 at 7.6 km and 0.91–0.93 at 12.6 km, and gain less beyond: at 22.7–37.8 km
they reach 0.92–0.95, a further gain of the best SLX model of 0.026 in 1402 and 0.003 in 1403 from 12.6 to 37.8 km. The GNN follows the same curve, from 0.90–0.91 with one layer (7.6 km) to 0.94 with two layers and
0.95 with three to five, with no gain beyond three layers. The no-graph network rises from 0.68 without neighbourhood
information to 0.954–0.958 when fed box means up to 22.7 or 37.8 km. The no-graph network was run at three reaches only, so
its curve between them is not known. Gains thus become small beyond 12–23 km, a range that overlaps the correlation
length after the large-scale trend is removed (14–19 km; Section 3.2) and lies below the raw zero crossing (24 km). PR-AUC follows the same pattern with more
scatter between estimators and years.

**Table 3.** Cross-validated Spearman correlation against reach, 1402 / 1403. SLX: the best of OLS, GBM and RF at each
reach (estimator in brackets; Fig. 4 shows every estimator). Networks: five-seed ensembles.

| Reach (km) | Best SLX | GNN | No-graph network |
|---|---|---|---|
| 0 | 0.768 (GBM) / 0.745 (GBM) | – | 0.685 / 0.676 |
| 0.8 | 0.767 (GBM) / 0.757 (GBM) | – | – |
| 2.5 | 0.780 (GBM) / 0.788 (GBM) | – | – |
| 7.6 | 0.866 (RF) / 0.868 (OLS) | 0.902 / 0.907 | – |
| 12.6 | 0.924 (GBM) / 0.934 (OLS) | – | – |
| 15.1 | 0.921 (GBM) / 0.924 (OLS) | 0.938 / 0.942 | – |
| 22.7 | 0.938 (OLS) / 0.934 (GBM) | 0.949 / 0.947 | 0.954 / 0.955 |
| 30.3 | 0.942 (OLS) / 0.933 (GBM) | 0.950 / 0.949 | – |
| 37.8 | 0.950 (GBM) / 0.937 (RF) | 0.946 / 0.946 | 0.958 / 0.956 |

![Figure 4](../figures/fig4_reach.png)

**Fig. 4.** Skill against reach under 22-km block cross-validation, Spearman (a, b) and PR-AUC (c, d), 1402 (left) and
1403 (right). Lines: SLX models and the GNN with growing reach (five-seed ensembles for the GNN). Points: the no-graph
network, run at 0, 22.7 and 37.8 km. Dashed line: zero crossing of the target correlogram (24 km).

### 4.2 At equal reach, the graph adds little over neighbourhood means

Table 4 compares the GNN at each depth with models of exactly the same reach. Against SLX models, the GNN's lead in
Spearman shrinks as reach grows: about 0.04 at 7.6 km, 0.02 at 15.1 km, 0.011–0.016 at 22.7–30.3 km, and −0.004 and
+0.009 at 37.8 km. None of these differences is significant after Holm adjustment, all jackknife intervals include
zero, and for individual fits the leads are smaller still (0.028–0.033 at 7.6 km, 0.007–0.008 at 22.7 km, −0.008 and
+0.005 at 37.8 km). Part of the GNN's lead over SLX at short reach therefore comes from its learner and ensemble, and the
lead disappears where skill levels off. In PR-AUC the GNN ensemble leads the SLX references by 0.02–0.11 at
22.7–37.8 km, and several jackknife intervals exclude zero, but no randomisation test is significant after Holm
adjustment, and the single-seed leads are about half as large (0.035–0.073 at 22.7 km, −0.007 to +0.055 at 37.8 km).

The no-graph network fed box means shares the GNN's encoder, training, seeds and ensemble, and differs in how spatial
context is supplied (Section 3.1). At 22.7 km it is statistically equivalent to the GNN in both years,
and slightly better (Spearman −0.005 and −0.008 for GNN minus no-graph network, 90 % intervals inside ±0.02); at 37.8 km
it is again slightly better (−0.012 and −0.010), equivalent in 1403 and inconclusive in 1402. In 1403 the 95 %
intervals exclude zero at both reaches, so the box-mean network is detectably, though not materially, better. In PR-AUC the difference changes sign between years at both reaches (+0.014 and −0.032 at 22.7 km; +0.030 and −0.010 at 37.8 km). For the ranking, replacing message passing by
fixed neighbourhood means thus cost nothing at these reaches; for the hotspots the evidence does not favour either.

**Table 4.** GNN (five-seed ensemble) minus a reference of equal reach, Spearman, cross-validation, 1402 / 1403.
SLX: at each reach, the estimator with the best mean Spearman over 1402 and 1403 (RF at 7.6 km, OLS at 15.1 km, GBM at
22.7–37.8 km), so the differences can differ from those implied by the per-year best in Table 3. Intervals: delete-one-block jackknife, 95 %. p: one-sided block sign-flip test over all 4,096 assignments, unadjusted (liberal for Spearman;
Section 3.5). None is significant after Holm adjustment within the exploratory family.

| Reach | Reference | Δ 1402 [95 % CI] | p | Δ 1403 [95 % CI] | p | Verdict |
|---|---|---|---|---|---|---|
| 7.6 km | SLX | +0.036 [−0.024, +0.096] | 0.033 | +0.040 [−0.020, +0.100] | 0.027 | inconclusive |
| 15.1 km | SLX | +0.022 [−0.041, +0.085] | 0.15 | +0.018 [−0.035, +0.072] | 0.17 | inconclusive |
| 22.7 km | SLX | +0.011 [−0.034, +0.056] | 0.11 | +0.013 [−0.028, +0.053] | 0.056 | inconclusive |
| 30.3 km | SLX | +0.011 [−0.023, +0.046] | 0.084 | +0.016 [−0.026, +0.059] | 0.042 | inconclusive |
| 37.8 km | SLX | −0.004 [−0.035, +0.028] | 0.79 | +0.009 [−0.031, +0.050] | 0.14 | inconclusive |
| 22.7 km | No-graph network, box means | −0.005 [−0.017, +0.007] | 0.93 | −0.008 [−0.016, −0.001] | 0.92 | equivalent |
| 37.8 km | No-graph network, box means | −0.012 [−0.030, +0.005] | 0.995 | −0.010 [−0.017, −0.003] | 0.98 | inconclusive / equivalent |

The comparison of the original study differed because its reference models had a shorter reach. Against SLX-GBM at 12.6 km, the
pre-specified reference, the three-layer GNN led by +0.025 and +0.035 in Spearman (randomisation p = 0.001 and 0.0007;
Holm-adjusted 0.021 and 0.016) and by +0.037 and +0.051 in PR-AUC. These leads are partly a comparison of reaches, and
given the liberal behaviour of the Spearman test, even the remaining part is uncertain: the jackknife intervals include
zero (−0.003 to +0.052 and −0.013 to +0.082), and the block-level version of the test is not significant (p = 0.058
and 0.055). Fig. 5 shows the differences against the pre-specified and against extended-reach references.

![Figure 5](../figures/fig5_differences.png)

**Fig. 5.** GNN (five-seed ensemble) minus reference in Spearman (a) and PR-AUC (b), with 90 % (thick) and 95 % (thin)
delete-one-block jackknife intervals. Upper rows: the pre-specified references of the original study (SLX-GBM at
12.6 km in cross-validation, SLX-RF at 12.6 km on the held-out year). Lower rows: references with extended reach (the
best SLX model by Spearman at any reach in cross-validation; SLX-OLS at 22.7 km on the held-out year, shorter than the
GNN's 37.8 km; the comparisons at exactly equal reach are in Table 5). Grey band in (a): equivalence margin (±0.02).

### 4.3 On the held-out year, GNN and SLX at equal reach rank alike

In 1404, the five-layer GNN (37.8 km) scores 0.933 in Spearman correlation and 0.782 in PR-AUC, and the four-layer GNN (30.3 km) scores 0.931 and 0.785 (Table 5). At 30.3 km the best SLX model (OLS, 0.934) is statistically equivalent to the GNN (−0.003, 95 % interval
−0.019 to +0.014); at 37.8 km the best SLX model (RF, 0.932) differs by +0.002 (95 % interval −0.026 to +0.029; inconclusive). The SLX
reference is the estimator with the best Spearman at that reach. In PR-AUC the five-seed ensembles lead it by
0.049–0.086 (by 0.043–0.058 against the SLX estimator with the best PR-AUC), but individual fits lead by less: the mean
single seed scores 0.758 at 30.3 km and 0.741 at 37.8 km, against 0.699 and 0.733 for the references. A linear SLX model with a shorter reach of 22.7 km (OLS) does as well
as either GNN (0.929 and 0.785), although it was chosen among fifteen variants after the results, which favours it. Its
full cross-validation took 8 s on one CPU, against 39 min on one RTX 4080 GPU for the GNN with all folds and five
seeds; both times cover training and prediction of the reported workflow.

Against the pre-specified reference of the original study (SLX-RF at 12.6 km), the five-layer GNN led in Spearman by
+0.029 (95 % interval +0.001 to +0.056, p = 0.0095), which is not significant after Holm adjustment over the registered
family (p = 0.19).

**Table 5.** Held-out year 1404: GNN and SLX at equal reach, and a shorter-reach SLX model. GNN: five-seed ensemble, and
the mean of its individual fits. Differences in the text are computed from unrounded values.

| Reach | Model | Spearman | PR-AUC |
|---|---|---|---|
| 30.3 km | GNN, 4 layers | 0.931 | 0.785 |
| | GNN, 4 layers, mean of individual fits | 0.926 | 0.758 |
| | SLX-OLS / SLX-RF / SLX-GBM | 0.934 / 0.923 / 0.914 | 0.699 / 0.727 / 0.641 |
| 37.8 km | GNN, 5 layers | 0.933 | 0.782 |
| | GNN, 5 layers, mean of individual fits | 0.928 | 0.741 |
| | SLX-OLS / SLX-RF / SLX-GBM | 0.930 / 0.932 / 0.927 | 0.739 / 0.733 / 0.664 |
| 22.7 km | SLX-OLS | 0.929 | 0.785 |

### 4.4 Seed averaging hides instability

Across seeds, the standard deviation of the GNN's Spearman correlation is about 0.005, but that of its PR-AUC is 0.028 in 1402, 0.038 in 1403 and 0.075 in 1404; in 1403 the individual fits range from 0.756 to 0.858 in PR-AUC (Fig. 6). Averaging five
seeds raises PR-AUC by 0.03–0.04 (for example 0.800 against 0.771 in 1402) and Spearman by about 0.005. The SLX models
were run once, so they get no such benefit, and a comparison of a five-seed ensemble with single baselines favours the
network by about the size of the PR-AUC leads in Table 4. This is why the GNN's PR-AUC leads at equal reach
(Sections 4.2–4.3) are smaller and more variable for individual fits. Whether ensembling the stochastic SLX baselines would
close the gap was not tested.

![Figure 6](../figures/fig6_seeds.png)

**Fig. 6.** Seeds against ensemble, Spearman (a) and PR-AUC (b), with three lanes per row so that close values stay
visible. Upper lane: the GNN (bar: range of the five individual fits; tick: their mean; dot: five-seed ensemble). Middle
lane: SLX with extended reach (best SLX model by Spearman at any reach in cross-validation; SLX-OLS at 22.7 km on 1404).
Lower lane: the pre-specified SLX reference at 12.6 km.

### 4.5 How the evaluation decides the verdict

The original study's confirmatory test on 1404 compared the GNN with the pre-specified SLX-RF at 12.6 km in PR-AUC. With
the block bootstrap the difference was +0.065 (95 % interval +0.013 to +0.163), and the hypothesis was declared
supported. With the same predictions and the inference of Section 3.5, the jackknife interval is −0.019 to +0.149 and
the randomisation p is 0.10, so the verdict becomes non-inferior (lower bound above −0.02). With the leakage-free predictions of this paper, the
difference is +0.046, the jackknife interval (+0.011 to +0.082) calls the GNN superior, but the randomisation p is 0.098
and 1.0 after Holm adjustment. With about five effective degrees of freedom, procedures that are each defensible
disagree, and the choice of inference decides between "significantly better" and "no evidence of a difference".

### 4.6 Fusion of data sources does not help (RQ1)

Under the same gated network, fusing traffic, energy and station data did not consistently beat the best single source
(energy, pre-specified in the registration). In PR-AUC the fused model scored +0.022 and +0.040 higher (95 % intervals
−0.013 to +0.057 and −0.010 to +0.091; non-inferior); in Spearman it scored 0.021 and 0.022 lower, for all five seeds,
but with wide intervals (inconclusive). On 1404, where the registered design compares four-layer GNNs, fused and
energy-only models were equivalent in Spearman (−0.007, 90 % interval −0.019 to +0.005) and differed by −0.011 in PR-AUC
(inconclusive). Fusion thus helps neither metric clearly, and its sign depends on the metric. Two observations are
consistent with this. First, urban form already gives the ranking: a
network with urban-form context alone (population, building volume, night lights) reaches Spearman 0.91, though only
0.58–0.59 in PR-AUC, and the emission layers sharpen the top of the ranking rather than the ranking itself. Second, the two
largest emission layers are redundant at the scale of the target: road and building CO₂ correlate at 0.58 for single
cells but at 0.998–0.999 once smoothed over 8 km or more, the scale over which the target varies. We checked this only
for these two layers; whether the station layer and the other inputs are equally redundant was not tested, so the
shared explanation for fusion and the graph (at the scale that matters, different inputs and different ways of
combining a neighbourhood carry the same information) remains a hypothesis.

Rebuilding the station layer without the test-block stations did not lower skill (station-only network: PR-AUC 0.715
and 0.726, against 0.648 and 0.691 with the original pre-processing), so for this model and metric the leak did not
inflate the original results.
A physical operator with one NO₂ yield per sector reached Spearman 0.85 and PR-AUC 0.51–0.55, far better than the
original operator that smoothed all sectors together (PR-AUC 0.35) but below every SLX model with a reach of 12.6 km or
more; as it uses the inventory layers only, this comparison does not isolate physics from learning.

### 4.7 What the target allows

Much of the skill comes from the smoothness of the target itself. Interpolating the target of the training cells of
the same month, without any environmental predictor, reaches Spearman 0.86 and 0.81 (1402 and 1403) with a Gaussian process on the coordinates and
0.68–0.69 with inverse-distance weighting (PR-AUC 0.72 and 0.61, and 0.34–0.37). The predictors and models add about
0.08–0.14 in Spearman to the Gaussian process, and the GNN and the reach-matched SLX models add the same amount.

Under other hotspot definitions and on superpixels the picture is similar (Table 6). Against the shorter-reach
references the GNN ensemble leads under every definition. Against references with extended reach, SLX models rank
better on superpixels in cross-validation (−0.027 and −0.017), while the GNN ensemble leads at the top 20 % and 5 % in
1403 (+0.040 and +0.116); on superpixels the target and predictions are averaged within
each superpixel (partial superpixels included) and the top-10 % label is recomputed, and the top-5 % label marks about 120 cells per month in a few contiguous areas, so these values
rest on very few independent units, and single-seed values are not available for them.

**Table 6.** GNN (five-seed ensemble) minus reference under other hotspot definitions and on 5 × 5-cell superpixels
(delete-one-block jackknife estimates; intervals in Table S1).

| Reference | Year | PR-AUC, top 20 % | PR-AUC, top 5 % | Spearman, superpixels | PR-AUC, superpixels |
|---|---|---|---|---|---|
| SLX-GBM, 12.6 km | 1402 | +0.025 | +0.044 | +0.026 | +0.026 |
| | 1403 | +0.027 | +0.144 | +0.041 | +0.024 |
| SLX-RF, 12.6 km | 1404 | +0.025 | +0.049 | +0.033 | +0.022 |
| Best SLX by Spearman, any reach | 1402 | +0.014 | +0.028 | −0.027 | +0.011 |
| | 1403 | +0.040 | +0.116 | −0.017 | +0.035 |
| SLX-OLS, 22.7 km | 1404 | −0.013 | +0.002 | +0.008 | −0.025 |

Larger buffers, which force the models to extrapolate from the periphery towards the centre, change the picture
(driver inputs, reach 22.7 km). With the paper's 2-km buffer the GNN and the best SLX model are level (Spearman 0.943
and 0.945 against 0.940 and 0.935). With a 10-km buffer the GNN keeps 0.913 and 0.911 while the best SLX model falls to
0.856 and 0.845; with a 20-km buffer the GNN keeps 0.791 and 0.774 while SLX models fail (best 0.35 and 0.24). With a 20-km buffer, the model in the central fold trains on only 270 peripheral cells. Tree ensembles cannot predict beyond the range of
their training targets and OLS extrapolates wildly, so the failure is one of the conventional learners. Whether the
GNN's robustness comes from the graph or from the neural learner was not tested, because the no-graph network was not
run with larger buffers.

---

## 5. Discussion

### 5.1 Why reach, and not the graph, delivers skill

The monthly NO₂ pattern of Tehran is a smooth field, with a correlation length of 14–24 km depending on how its
large-scale gradient is treated and about five effective degrees of freedom over the city. Two processes make it so.
TROPOMI averages over pixels of 3.5 × 5.5 km, and NOx is carried downwind during its daytime lifetime of a few hours:
Beirle & Wagner (2024) report a mean of 2.4 h over 100 cities, of the same order as earlier megacity estimates (Beirle
et al., 2011), which at boundary-layer wind speeds of a few metres per second gives a transport scale of 10–30 km. These
scales are consistent with the value of a cell depending mainly on emissions and urban form within a radius of the same
order, which would explain why models that see this neighbourhood predict it well; our experiments show the reach at
which skill levels off, not the physical source area. How the neighbourhood is combined matters little for the ranking
when the field is this smooth: box means with a linear or tree model, the same box means fed to a neural network, and
learned message passing reach similar skill once they see 12–23 km. For the top decile, scores vary more across estimators, and no SLX estimator is best in every year. The GNN ensemble keeps an edge that largely disappears in its individual fits.

We first used the 24-km zero crossing as a heuristic for the needed reach, but it is not a physical constant: it falls to 14–19 km after detrending and to 18–22 km on smaller domains. The reach beyond which skill gains little
(12–23 km) overlaps the detrended correlation length more than the raw one. The practical rule is therefore to vary
reach and look where skill levels off, using the correlation length as a first guess rather than as a fixed answer.

### 5.2 Implications for spatial deep learning

Graph-learning research on urban emissions and air quality has grown rapidly, but its baselines are rarely given the reach of the proposed models. Our results suggest that part of the reported gains are gains of reach. They
agree with scattered evidence: a random forest matched a spatial GNN for urban NO₂ in Berlin once it received the NO₂ of
the other stations weighted as in the graph (Schmidt & Ong, 2026), a graph model of city CO₂ did not beat an MLP or
XGBoost without extra constraints (Shao et al., 2026), GNNs can be written as generalised spatial regressions (Liang et
al., 2023; Zhu et al., 2026), and explicit spatial-lag inputs recover much of what spatial context offers (Xiao et al.,
2023; Schmidinger et al., 2026). They also agree with results on non-spatial graphs, where linear models on
pre-propagated features matched graph convolutions (Wu et al., 2019; Frasca et al., 2020), and with the wider experience
that complex models are often matched by simple, well-specified baselines (Ferrari Dacrema et al., 2019; Zeng et al.,
2023). What these studies left open is whether the remaining gains come from the graph or from the size of the
neighbourhood. The no-graph network fed box means answers this for a smooth target and a distance-based graph: the
graph adds nothing over neighbourhood means.

This does not make graphs useless. They should matter where edges carry information that distance does not, such as
climate teleconnections (Chen et al., 2026), flows on a transport network (Mohseni et al., 2026), wind-aligned
transport, or irregular units for which box windows are hard to define; and our results with larger buffers suggest
that neural models extrapolate more gracefully than tree ensembles, whatever the role of the graph. We recommend five
practices for spatial GNN studies: give baselines the model's reach, which costs seconds; report skill against reach
and relate the level-off point to the correlation length of the target; report individual fits next to ensembles, or
ensemble the baselines too; use inference built for few spatial blocks and report the effective sample size of the
target; and build interpolated covariates inside the cross-validation.

### 5.3 Practical implications

For screening the NO₂ hotspots of a city from satellite data, a linear regression on neighbourhood means of emission and
urban-form layers over about 10–25 km was an adequate model here. It is fast; whether it transfers to other cities is a
question for evaluation. Its coefficients are hard to interpret because neighbourhood means at different scales are highly collinear, but its residual map can still be compared directly with the inventory. For RQ1, the station
layer and the separate emission sectors added little to a map that urban form and one emission source already rank
well; the effort of data fusion is better spent where sources differ at the scale of the target. We found no evidence
that the graph adds predictive value for this task in Tehran; a neural network fed neighbourhood means ranked as well
and is simpler. When the model must extrapolate to unsampled parts of the city, the GNN degraded far less than the
tree ensembles and OLS (Section 4.7); whether the neural network without a graph does the same was not tested.

### 5.4 Limitations

The study covers one city, one target and a distance-based graph. The correlation length, and hence the needed reach,
will differ between cities and targets, and graphs whose edges carry other information may behave differently. No
hyper-parameters were tuned; the SLX references were chosen from several variants after the results, while the GNN was
not tuned at all, and a tuned GNN might gain a little. The held-out year adds little new information, because the
target barely changes between years and its inventory and traffic inputs are those of the previous year; a second
held-out year (1405) was registered but not run. The traffic and energy layers are the inventory's spatial allocation
of sector totals, not measured emissions, so the redundancy of road and building CO₂ at the scale of the target
(Section 4.6) may partly reflect how the inventory allocates them; whether measured emission maps would be equally
redundant was not tested. The reach-matched comparisons do not depend on this, because all models receive the same
inputs. The 2-km buffer makes cross-validation mostly an interpolation test,
and in the larger-buffer runs, where the GNN outperformed SLX, the role of the graph was not separated from that of the
learner. The target is standardised over all domain cells of each month, including the held-out block. It describes
column NO₂, which agrees only weakly with the station network, and NO₂ hotspots mark dense urban combustion: in Tehran
they hold about a third of road and building CO₂ but only about 1 % of industrial CO₂. Finally, the effective sample
size limits every test, and the randomisation test is liberal for Spearman; in cross-validation the reach-matched comparisons with SLX are
inconclusive rather than equivalent. Equivalence was established against the no-graph network at 22.7 km (both years)
and at 37.8 km (1403), and against SLX-OLS on the held-out year at 30.3 km. Target standardisation uses all domain cells
of a month; a version scaled on training cells only was not run.

---

## 6. Conclusions

In mapping intra-urban NO₂ hotspots from TROPOMI in Tehran, the skill of a distance-based graph neural network is
largely explained by how far it sees. Spatial-lag models improve steeply up to a reach of about 12 km and gain little
beyond; at equal reach the GNN's lead over them shrinks to within ±0.01 in Spearman, and the same network without a
graph, fed neighbourhood means, is equivalent to it at 22.7 km. The apparent advantages of the graph model in the
original study came from a shorter-reach baseline, from averaging training seeds and from inference unsuited to a
target with about five effective degrees of freedom. Fusing traffic, energy and station data did not clearly help
either, consistent with the main emission layers becoming almost identical at the scale over which the target varies. We recommend reach-matched
baselines and skill-against-reach curves as standard practice for graph models in spatial environmental mapping.

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

All DOIs were checked against Crossref (arXiv DOIs against arXiv) on 12 November 2026. Six entries have no DOI:
four conference papers (NeurIPS, ICLR, MLSys), Holm (1979) and the thesis (Vafaeian, 2026).

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
- Errica, F., Podda, M., Bacciu, D., & Micheli, A. (2020). A fair comparison of graph neural networks for graph classification. *International Conference on Learning Representations (ICLR)*. arXiv:1912.09893. https://doi.org/10.48550/arXiv.1912.09893
- Ferrari Dacrema, M., Cremonesi, P., & Jannach, D. (2019). Are we really making much progress? A worrying analysis of recent neural recommendation approaches. *Proceedings of the 13th ACM Conference on Recommender Systems*, 101–109. https://doi.org/10.1145/3298689.3347058
- Flores, M., Soliz, M., & Mollineda, R. A. (2026). Graph neural networks for soil organic carbon estimation: data-efficient modeling of multimodal spatial dependencies. *Machine Learning with Applications*, 24, 100878. https://doi.org/10.1016/j.mlwa.2026.100878
- Frasca, F., Rossi, E., Eynard, D., Chamberlain, B., Bronstein, M., & Monti, F. (2020). SIGN: scalable inception graph neural networks. arXiv:2004.11198. https://doi.org/10.48550/arXiv.2004.11198
- Franklin, S. E., Wulder, M. A., & Lavigne, M. B. (1996). Automated derivation of geographic window sizes for use in remote sensing digital image texture analysis. *Computers & Geosciences*, 22, 665–673. https://doi.org/10.1016/0098-3004(96)00009-X
- Freitas, A., de Wit, X. M., Gabbana, A., Woodward, M., Toschi, F., Lin, Y. T., & Livescu, D. (2026). Learning turbulent transport via Mori–Zwanzig graph neural networks. arXiv:2606.14918. https://doi.org/10.48550/arXiv.2606.14918
- Gao, H., Yang, F., Yang, Y., Xu, Y., & Su, Y. (2025). Adaptive receptive field graph neural networks. *Neural Networks*, 190, 107658. https://doi.org/10.1016/j.neunet.2025.107658
- Gasteiger, J., Bojchevski, A., & Günnemann, S. (2019). Predict then propagate: graph neural networks meet personalized PageRank. *International Conference on Learning Representations (ICLR)*. arXiv:1810.05997. https://doi.org/10.48550/arXiv.1810.05997
- Goldberg, D. L., Tao, M., Kerr, G. H., Ma, S., Tong, D. Q., Fiore, A. M., et al. (2024). Evaluating the spatial patterns of U.S. urban NOx emissions using TROPOMI NO₂. *Remote Sensing of Environment*, 300, 113917. https://doi.org/10.1016/j.rse.2023.113917
- Griffith, D. A. (2005). Effective geographic sample size in the presence of spatial autocorrelation. *Annals of the Association of American Geographers*, 95, 740–760. https://doi.org/10.1111/j.1467-8306.2005.00484.x
- Gueterbock, F., Santos-Rodriguez, R., & Clark, J. N. (2025). Improving local air quality predictions using transfer learning on satellite data and graph neural networks. arXiv:2505.05479. https://doi.org/10.48550/arXiv.2505.05479
- Halleck Vega, S., & Elhorst, J. P. (2015). The SLX model. *Journal of Regional Science*, 55, 339–363. https://doi.org/10.1111/jors.12188
- Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. *Advances in Neural Information Processing Systems*, 30.
- Hettige, K. H., et al. (2024). AirPhyNet: harnessing physics-guided neural networks for air quality prediction. *International Conference on Learning Representations (ICLR)*.
- Hoek, G., Beelen, R., de Hoogh, K., Vienneau, D., Gulliver, J., Fischer, P., & Briggs, D. (2008). A review of land-use regression models to assess spatial variation of outdoor air pollution. *Atmospheric Environment*, 42, 7561–7578. https://doi.org/10.1016/j.atmosenv.2008.05.057
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6, 65–70.
- Huang, Q., He, H., Singh, A., Lim, S.-N., & Benson, A. R. (2021). Combining label propagation and simple models out-performs graph neural networks. *International Conference on Learning Representations (ICLR)*. arXiv:2010.13993. https://doi.org/10.48550/arXiv.2010.13993
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
- Shchur, O., Mumme, M., Bojchevski, A., & Günnemann, S. (2018). Pitfalls of graph neural network evaluation. *Relational Representation Learning Workshop, NeurIPS 2018*. arXiv:1811.05868. https://doi.org/10.48550/arXiv.1811.05868
- Teng, M., Liang, M., Wang, S., & Ding, Y. (2026). Seamless hourly PM2.5 mapping across China with a graph spatiotemporal deep neural network. *Environmental Modelling & Software*, 198, 106915. https://doi.org/10.1016/j.envsoft.2026.106915
- Vafaeian, F. (2026). [Thesis title to be completed]. MSc thesis, [university to be completed]. Supervisor: V. Khatibi.
- Valavi, R., Elith, J., Lahoz-Monfort, J. J., & Guillera-Arroita, G. (2019). blockCV: an R package for generating spatially or environmentally separated folds for k-fold cross-validation of species distribution models. *Methods in Ecology and Evolution*, 10, 225–232. https://doi.org/10.1111/2041-210X.13107
- Wadoux, A. M. J.-C., Heuvelink, G. B. M., de Bruin, S., & Brus, D. J. (2021). Spatial cross-validation is not the right way to evaluate map accuracy. *Ecological Modelling*, 457, 109692. https://doi.org/10.1016/j.ecolmodel.2021.109692
- Wang, S., Li, Y., Zhang, J., Meng, Q., Meng, L., & Gao, F. (2020). PM2.5-GNN: a domain knowledge enhanced graph neural network for PM2.5 forecasting. *Proceedings of the 28th International Conference on Advances in Geographic Information Systems (SIGSPATIAL)*, 163–166. https://doi.org/10.1145/3397536.3422208
- Wang, S., Zhang, Y., Lin, X., Piao, X., Hu, Y., & Yin, B. (2025). CMDNet: a cross-modality spatiotemporal graph network for enhanced air pollution prediction with high-resolution satellite data. *IEEE Transactions on Geoscience and Remote Sensing*, 63, 1–14. https://doi.org/10.1109/TGRS.2025.3569594
- Wu, F., Zhang, T., Holanda de Souza, A., Fifty, C., Yu, T., & Weinberger, K. Q. (2019). Simplifying graph convolutional networks. *Proceedings of the 36th International Conference on Machine Learning (ICML)*, PMLR 97, 6861–6871. arXiv:1902.07153. https://doi.org/10.48550/arXiv.1902.07153
- Xiao, S., Song, Y., & Wang, Z. (2023). Nonparametric spatial autoregressive model using deep neural networks. *Spatial Statistics*, 57, 100766. https://doi.org/10.1016/j.spasta.2023.100766
- Yap, W., Wu, A. N., Miller, C., & Biljecki, F. (2025). Revealing building operating carbon dynamics for multiple cities. *Nature Sustainability*, 8, 1199–1210. https://doi.org/10.1038/s41893-025-01615-8
- Yuan, D., Wang, Z., Huang, H., Fu, Z., Wu, Q., Jhang, S., & Pedrycz, W. (2026). Spatially directional dual-attention graph networks for fine-grained PM2.5 mapping. *Applied Soft Computing*, 203, 116231. https://doi.org/10.1016/j.asoc.2026.116231
- Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are transformers effective for time series forecasting? *Proceedings of the AAAI Conference on Artificial Intelligence*, 37, 11121–11128. https://doi.org/10.1609/aaai.v37i9.26317
- Zeng, H., Zhang, M., Xia, Y., Srivastava, A., Malevich, A., Kannan, R., Prasanna, V., Jin, L., & Chen, R. (2021). Decoupling the depth and scope of graph neural networks. *Advances in Neural Information Processing Systems*, 34. arXiv:2201.07858. https://doi.org/10.48550/arXiv.2201.07858
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

**Table 2.** Predictor groups (Section 2.3).

**Table 3.** Cross-validated skill against reach (Section 4.1).

**Table 4.** GNN minus reference in cross-validation (Section 4.2).

**Table 5.** Held-out year at equal reach (Section 4.3).

**Table 6.** Other hotspot definitions and superpixels (Section 4.7).

**Table S1 (supplementary).** All comparisons with jackknife and bootstrap intervals, randomisation p-values (pooled
and block-level), Holm-adjusted p-values of the registered family of 22 tests, and verdicts:
`docs/paper/results_stageA/evaluate_paper2/` (including `paper_holm.csv`) and `evaluate_thesis/`.

**Table S2 (supplementary).** Sources, years and processing of every predictor layer [to be laid out from
`configs/config.yaml` and `data/raw/MANIFEST.csv`].

**Figures.** All figures use the same model colours (GNN blue, SLX-OLS orange, SLX-GBM aqua, SLX-RF yellow, no-graph network grey), grey text and unobtrusive axes. Each is saved as PNG (300
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

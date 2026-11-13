# Literature search record (novelty check)

This folder records the search behind the novelty statement in the manuscript (Section 1): that no published study
compares a graph neural network with a spatial-lag or buffer baseline **constrained to the same reach**, or varies the
reach of both.

## 1. ScienceDirect (8 November 2026)

- **Field:** "Title, abstract or author-specified keywords" (Advanced search). ScienceDirect allows at most 8 Boolean
  connectors per field and adds plurals and spelling variants itself.
- **Export:** the first results page of each query, ranked by relevance (25 records for Q1, Q3, Q4 and Q5; 100 for Q2),
  as RIS with abstracts.
- **Records:** 200 exported, 191 unique DOIs. Years: 2019–2026; 98 records from 2026 and 49 from 2025.

| Query | Search string | Records |
|---|---|---|
| Q1 | `("graph neural network" OR "graph convolutional network" OR GraphSAGE) AND ("spatial lag" OR "land use regression" OR "spatial regression") AND ("receptive field" OR neighborhood)` | 25 |
| Q2 | `("graph neural network" OR "graph convolutional network") AND ("nitrogen dioxide" OR NO2 OR "air pollution") AND (TROPOMI OR "Sentinel-5P" OR satellite)` | 100 |
| Q3 | `"receptive field" AND ("graph neural network" OR GNN) AND (spatial OR urban OR geographic) AND baseline` | 25 |
| Q4 | `("graph neural network" OR "graph convolutional network") AND ("carbon emission" OR "CO2 emission") AND (urban OR city) AND ("spatial lag" OR "random forest" OR baseline)` | 25 |
| Q5 | `("deep learning" OR "neural network") AND ("spatial lag" OR "spatially lagged") AND ("linear regression" OR "land use regression")` | 25 |

**Screening.** Every record was screened on its title and abstract.

- **Matches: none.** No record compares a GNN with a spatial-lag, buffer or land-use-regression baseline at equal reach,
  and none varies the reach of both models.
- **Closest records.** These are cited in the manuscript; `sciencedirect_screening.csv` lists all records, with the
  cited ones marked:
  - **GNN ≈ conventional baseline for urban NO₂.** In Berlin, a random forest and a spatial GNN reach the same accuracy
    (Schmidt & Ong, 2026).
  - **GNNs as generalised spatial regression.** Liang et al. (2023); Zhu, Wang & Luo (2026); Ma, Zhang & Song (2025).
  - **Spatial-lag inputs give much of the gain from spatial context.** Xiao et al. (2023); Schmidinger et al. (2026);
    Fleischmann & Arribas-Bel (2024).
  - **The receptive field as a design choice inside graph learning.** Gao et al. (2025); Chen et al. (2026, where
    convolutional baselines are limited by their effective receptive field).
  - **GNNs beating spatial regression without matched reach.** Mohseni et al. (2026); Zhao et al. (2026).
  - **Evaluation caveats.**
    - Residual autocorrelation after a GNN (Bao et al., 2024).
    - Failure of a GNN under leave-one-cluster-out (Flores et al., 2026).
    - Reported skill depends on the validation strategy (Zhang et al., 2026).
  - **Graph models for satellite-based air-pollution mapping.** Bai et al. (2025); Teng et al. (2026); Yuan et al. (2026).

The raw RIS exports are not committed because they contain publisher abstracts. The CSV keeps the bibliographic fields.

## 2. Semantic search (Consensus)

A Consensus literature review was run on the same two questions:

- whether any GNN study uses a reach-matched spatial-lag baseline;
- whether skill has been related to the correlation length of the target.

It found no matching study. The relevant references it returned were checked against Crossref and are cited in the
manuscript: Zhu et al. (2021), Zhan & Datta (2024), Abémgnigni Njifon & Schuhmacher (2024), Liu et al. (2025), Du et
al. (2026) and Zhao et al. (2026).

## 3. Reference checks

All DOIs cited in the manuscript were checked against the Crossref API on 8 November 2026.

## 4. Prior art for the reach rule (12 November 2026)

**Question.** Has the neighbourhood of a model been set from the range of spatial dependence before? And is the name
"reach-matched" taken?

**Answer.** The principle exists; the name and the use do not.

| Source | What it sets from the dependence range |
|---|---|
| Franklin, Wulder & Lavigne (1996), *Computers & Geosciences* 22:665 | Window size for image texture, from semivariograms |
| Behrens et al. (2019), *Scientific Reports* 9:14800 | Range of scales of contextual covariates for machine learning, from the variogram |
| Valavi et al. (2019), *Methods Ecol. Evol.* 10:225 (blockCV) | Size of spatial cross-validation blocks, from the autocorrelation range |
| Alon & Yahav (2021), ICLR | GNN depth should match the "problem radius" of the task |
| Freitas et al. (2026), arXiv:2606.14918 | Receptive field of a GNN for turbulent transport, set from the integral length scale; no non-graph baseline |

- **Name.** Web searches for "reach-matched" and "range-matched" with GNN or spatial baselines return nothing.
- **Use.** None of the sources uses the rule to give a GNN and its spatial-lag baselines the same reach.
- **In the manuscript.** The rule is presented as an adaptation of this principle (Section 3.2), not as a new method.

## 5. Reach audit of comparable studies (Table 1 of the manuscript)

Full texts were read for eleven studies that compare a GNN with baselines on spatial data.

- **Sources.**
  - Seven: arXiv versions or open proceedings.
  - Four (Schmidt & Ong; Mohseni et al.; Zhu, Wang & Luo; Shao et al.): PDFs supplied by the user on 12 November 2026.
    Publisher sites block automated reading.
- **For each study:** how far the GNN sees, and how far each baseline sees.
- **Main finding:** no study makes reach the object of the comparison. None varies the reach of both models or sets it
  by one rule. Where neighbourhoods coincide, it is incidental.
- **Closest cases.**
  - Liang et al. (2023, *TR-C* 154:104241): the linear SLX used the same 5 nearest stations as the GNN, but XGBoost saw
    no neighbours and the GNN also used 5 non-geographic neighbours.
  - Schmidt & Ong (2026, *Ecol. Inform.* 93:103568): the random forest received a "weighted mean pollution" feature,
    the NO₂ of all other stations weighted as in the graph. It was the forest's most important feature. RF R² 0.584
    vs GNN 0.613, on three test stations. This supports our result.
- **Also telling:** Shao et al. (2026). The graph model trained with city supervision only reached R² 0.65, against
  linear regression 0.66, MLP 0.68 and XGBoost 0.64. Its final 0.78 came from added weak-supervision losses
  (provincial totals, masking, smoothness).

| Study | Source read | GNN reach | Baselines' reach |
|---|---|---|---|
| Wang et al. 2020 (PM2.5-GNN) | arXiv:2002.12898 | cities < 300 km, no ridge > 1,200 m; 1 step per 3 h | MLP/LSTM/GRU none; GC-LSTM same graph; nodesFC-GRU all |
| Liang et al. 2023 | arXiv:2303.11977 | 1 attention layer, 5 nearest + 5 similar stations | linear SLX nearby (≈ same 5); LR, XGBoost none |
| Klemmer et al. 2023 (PE-GNN) | arXiv:2111.10144 | 2 layers on 5-NN graph | exact GP all training points |
| Hettige et al. 2024 (AirPhyNet) | arXiv:2402.03784 | distance/wind graph, reach not reported | none / 3-step diffusion / 50–200 km / 30-NN |
| Gueterbock et al. 2025 | arXiv:2505.05479 | GraphSAGE 2 hops | not reported |
| Du et al. 2026 (GraPhy) | arXiv:2506.06917 | 5 layers, complete graph | IDW and mean: all sensors; others not reported |
| Zhao et al. 2026 (UST-GNN) | arXiv:2504.04739 | GATv2 2 layers, contiguity | SLM, GWR not reported; RF, LightGBM none |
| Schmidt & Ong 2026 | PDF (user) | GCN + contrastive embedding + attention, all 16 stations, weights by similarity of distance to centre | RF: buffers 25–500 m + NO₂ of all other stations with the same weights |
| Mohseni et al. 2026 | PDF (user) | GraphSAGE 6 layers (6 hops) | LR, XGBoost, ANN own station (POIs ≤ 2.5 km); spatial regression weights not reported |
| Zhu, Wang & Luo 2026 | PDF (user) | 3 GCN modules on 20-NN graph | OLS none; SAR predefined adjacency; GWR 64 NN; in-sample only |
| Shao et al. 2026 | PDF (user) | graph convolution over 329 cities, 1/distance weights; depth not reported | LR (night lights), MLP, XGBoost own city |

**Still to read** (the user can download the PDFs):

- Ma et al. 2025 (*Spatial Statistics*)
- Teng et al. 2026, Yuan et al. 2026, Bai et al. 2025

## 6. LLM-assisted search (ChatGPT, 13 November 2026)

**What was run.** ChatGPT with web search was given the four questions of the novelty check: (1) a GNN compared with a
non-graph baseline given the same reach on purpose; (2) the reach of both varied; (3) the needed reach tied to the
correlation length of the target; (4) a non-graph model with neighbourhood-aggregated features matching a GNN. It was
also asked for the reach of the four studies still to read. It reported about 100 screened records from a general web
and scholarly index (arXiv, publisher and author pages; IEEE and ACM only through domain-restricted web queries). It had
no access to Google Scholar, Scopus or Web of Science, and no Elsevier full texts. This is not a reproducible database
search; it is recorded as a supplementary check.

**What was checked here.** DOIs against Crossref and arXiv; claims against the full text where it was open.

| Study | Exists | Claim | Checked against | Use |
|---|---|---|---|---|
| Chen, Chen & Bruna (2020), arXiv:2010.15116 | yes (arXiv) | For community detection on synthetic SBM graphs, an sGNN variant was chosen "for a fairer comparison with GA-MLPs" so that it "has the same receptive field as the chosen GA-MLP models" (Appendix L.1); at equal receptive field the GNN outperformed GA-MLP-A and only matched an oracle GA-MLP (Section 6.3, Fig. 3). Their Table 1 varies K = 1–5 for both, but counts equivalence classes, not predictive skill | full text (arXiv v2) | Cited (Introduction and Section 5.2): reach-matched comparison exists on non-spatial graphs, with the opposite outcome where the target is graph structure |
| Yang et al. (2022), *Neurocomputing* 505, 289–303, 10.1016/j.neucom.2022.07.030 | yes (Crossref) | A spatio-temporal semi-variogram that flattens beyond graph distance 2 sets a two-layer GCN for link prediction in dynamic networks | **not verified**: no open full text or abstract | Not cited until the PDF is read. If confirmed, it joins the prior art of Section 4 (choosing the neighbourhood from the dependence range) |
| Frasca et al. (2020), SIGN | yes (arXiv) | Pre-computed diffusions fed to an MLP; receptive fields not matched against the compared GNNs | — | Already cited |
| Maurya, Liu & Murata (2023), *CAAI Trans. Intell. Technol.* 8, 14–28, 10.1049/cit2.12166 | yes (Crossref) | A two-layer MLP on pre-computed hop features (K = 3) beats several GNNs; reach not matched | not checked | Not cited: same point as SGC and SIGN |
| Ma et al. 2025; Teng et al. 2026; Yuan et al. 2026; Bai et al. 2025 | yes | Only abstracts were available; no reach could be established | — | Still to read (Section 5) |

**Effect on the novelty statement.** A reach-matched comparison of a GNN with a non-graph model exists on non-spatial
graphs (Chen et al., 2020), so the manuscript now says that no *spatial or environmental* study was found that makes
reach the object of the comparison, varies it, or ties it to the scale of the target.

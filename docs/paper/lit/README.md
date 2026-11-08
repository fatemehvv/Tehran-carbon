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

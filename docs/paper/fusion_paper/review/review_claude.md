<!-- Internal review of manuscript_en.md (version of commit before the review round), pasted by the author on 13 November 2026. Kept verbatim as a record. Triage and actions: README_fa.md -->

# Review 1 (Claude)

I'll review it as a strict referee. One thing first: only the manuscript Markdown reached me. The three figures and the pre-registration weren't attached. So my comments on the figures are limited to their captions and cross-references, and I've checked pre-registration compliance only against what Section 3.6 says. Let me run a few checks before writing.

I've reviewed the manuscript; the full report is below. Only the Markdown manuscript reached me. The three figures and the pre-registration file were not attached, so:
- my figure comments rest on the captions and cross-references only;
- I checked the pre-registration only against what Section 3.6 says it contains.

Each comment is tagged **[Error]** (wrong or unsupported as written), **[Required]** (a missing analysis or information without which the claim cannot be assessed) or **[Taste]** (my preference; the authors may reasonably decline).

---

## A. Summary

The paper argues that ablating a data source in a multi-branch fusion network confounds the information in the source with the effect of adding a branch. It proposes "placebo branches" to separate the two: the same layers relocated by IAAFT surrogates (A) or by 180° rotation plus a cyclic shift (B). The test case is the intra-urban pattern of monthly TROPOMI NO₂ over Tehran, mapped with gated or early fusion of traffic, energy-inventory and station layers on top of an always-present urban-form context branch. Validation uses 12 spatial blocks, three years (one held out), five seeds and five placebo realisations.

The reported results are:
- Energy beats its placebo by 0.018–0.038 in Spearman. Traffic and stations do not.
- An information-free branch lowers Spearman by about 0.02 and raises top-decile PR-AUC by up to 0.11.
- Energy layers are almost fully predictable from the context layers.
- In a second experiment, calm-day composites were no sharper than windy-day ones, and fusion did not help at any smoothing of the target.
- Both placebo constructions failed the authors' own registered texture check (M4).

## B. Recommendation: **Major revision** (close to reject)

The question matters, the pre-registration and the reporting of single seeds and realisations are commendable, and the arithmetic is mostly internally consistent. However, the central inferential claim — that real-minus-placebo measures "information" and placebo-minus-context measures "architecture" — does not follow from the design:
- The placebo breaks the dependence between a source and the context branch, not only between the source and the target.
- The placebo almost certainly changes the input distribution inside the label domain.
- The "architecture" effect appears equally in early fusion, which has no extra branch.

Together with a failed validity check, ~5 effective units and an uninformative second experiment, the current claims are overstated. The study may be salvageable with a conditional placebo, a few cheap controls and reframed conclusions.

---

## C. Major comments

**C1. [Error] Real-minus-placebo does not isolate "information"; it confounds information with redundancy with the context branch.**
*Where:* Abstract ("Energy layers beat their placebos"), Highlights 2 and 4, Sections 4.1, 4.2 and 5.2, Table 4.

*Problem:* Relocating a source removes two things at once: its co-location with the target and its co-location with the context layers. Your own numbers show what this does.
- Real energy minus context-only (gated, Spearman; Section 4.2) is **+0.004 / +0.013 / −0.001**. In practice, energy adds nothing over context.
- Placebo minus context (Table 2, gated) is −0.021 / −0.025 / −0.027 (A) and −0.014 / −0.013 / −0.025 (B).
- The F1/C1 differences in Table 1 are exactly the sum of the two. For example, 1402 A: 0.004 + 0.021 = 0.025. For 1404 A and B the whole difference is the placebo penalty.

So "energy beats its placebo" means "the placebo hurts and energy does not." A simple explanation fits all your results without any target information in the energy layers:
- Energy is the most redundant source (gradient-boosting R² of 0.98, 0.95 and 0.89 from context, Table 4). Adding it adds almost no new input direction for the network to overfit.
- A relocated copy is by construction independent of context, so it adds a new, spatially structured, target-irrelevant direction.
- Traffic road length and stations are the *least* redundant (R² < 0.03 to < 0), and they behave like placebos.

This is the opposite of the reading in Section 5.2, where lineage explains why sources add *little*. Under this alternative, lineage explains why energy *looks* informative against the placebo.

*Why it matters:* The paper's single positive finding, and the interpretation of all placebo contrasts, depend on this.

*What to do:*
1. Build a **conditional placebo**, the spatial analogue of a knockoff. The relevant literature is model-X knockoffs, Candès, Fan, Janson and Lv (2018), JRSS B 80(3), 551–577, DOI 10.1111/rssb.12265. Concretely: regress each layer on the context layers (you already have the block-CV gradient-boosting models from the lineage audit), relocate only the residual with A or B, and add back the fitted part. This keeps the source's dependence on context and removes only its unique location information.
2. Add a **duplicate-context branch**: a second branch fed with a copy, or a lightly perturbed copy, of the context layers. It adds a branch with no new information and no new noise direction.
3. Report real-minus-context with the same intervals and tests as real-minus-placebo, and discuss F1 in that light.
4. Reword the Abstract, Highlights and Section 6 until (1) is done. At present, "carry information beyond their values and texture" (Section 4.1) is not supported.

**C2. [Error/Required] The placebo probably does not "keep the values" inside the label domain, and check M4 is computed on the wrong domain.**
*Where:* Sections 2.1, 3.2 and 3.6 (M4: "on the full grid"), Section 4.4.

*Problem:*
- Section 2.1 defines the label domain as the cells "that the emission inventory assigns any CO₂". The inventory layers are therefore zero on about 73 % of the 72 × 126 grid (9,072 cells, of which 2,436 are labelled).
- Placebo B shifts by ¼–¾ of the grid in each direction. Most label-domain cells then receive zeros from outside the city, plus fragments of the core cut at the seams.
- Placebo A rank-maps the values over the whole padded window, so it redistributes the same zero mass.

Inside the domain where the model is trained and tested, the placebo's marginal distribution, its fraction of zeros and its range differ from the real layer's. The "information-free branch" may be a mostly-constant branch with a few displaced blobs. Its penalty (C1) and its PR-AUC behaviour (C4) may come from this, not from "architecture". M4 is evaluated "on the full grid", so it cannot detect the problem.

*What to do:*
- For every layer and realisation, report within the label domain: the fraction of zero cells, a two-sample KS distance from the real layer, |Spearman| with the real layer, with the target and with each context layer.
- Re-run M4 on the label domain.
- If the marginals differ, rank-map placebo values within the label domain to the real label-domain values.
- State whether the station placebo is rebuilt per fold, as the real station layer is (Section 2.2).

**C3. [Required] Both constructions failed M4; the "the two agree" argument is too weak to carry the conclusions.**
*Where:* Abstract (last paragraph), Highlight 5, Sections 4.4, 5.1 ("the conclusions do not rest on one construction") and 5.5.

*Problem:* A and B are not independent checks. Both:
- relocate the same single core away from its real position;
- break dependence with context (C1);
- inherit the same domain problem (C2);
- share the seeds, folds and network.

Their agreement shows robustness to how texture is distorted, not robustness to the confounds that matter. Section 4.4 also explains why any relocation on a bounded one-core grid must fail the ±0.05 tolerance. If so, the registered tolerance was unattainable by design, and the paper should say plainly that check M4 was badly chosen rather than present the failure as a property of the city.

*What I would require:*
1. **A per-realisation dose-response.** Across the 10 realisations (A and B), regress each realisation's Δ (Spearman and PR-AUC) on its texture error, its label-domain |ρ| with the target and with context, and its fraction of zeros. If Δ does not depend on texture error, the M4 failure matters less. If it does, the verdicts are not interpretable.
2. **Bracketing controls:**
   - a deliberately bad placebo (i.i.d. permutation of label-domain cells, no texture);
   - an i.i.d. Gaussian-noise branch of the same width;
   - the conditional placebo of C1.
3. **A demonstration that M4 can be passed at all**, e.g. on a synthetic stationary field with the label-domain variogram. Otherwise drop it as a criterion and say why.
4. **The label.** Carry "placebo check failed" verbatim into every place a placebo verdict appears: Highlights 3–4, the Abstract bullets and the Section 6 bullets.

Related prior art your discussion should engage:
- The torus-translation test of Harms et al. (2001), Journal of Ecology 89, 947–959 (DOI 10.1111/j.1365-2745.2001.00615.x) is essentially your placebo B.
- The spin test of Alexander-Bloch et al. (2018), NeuroImage, DOI 10.1016/j.neuroimage.2018.05.070 is the rotation analogue.
- The comparison of spatial null models in Markello & Misic (2021), DOI 10.1016/j.neuroimage.2021.118052 found that naive null models yield elevated false positive rates, and even spatially-constrained null models suffer from inflated false positive rates and variable performance. Its lessons on non-stationary maps and boundaries apply directly.

**C4. [Error] The "architecture" effect is not an effect of adding a branch, and its PR-AUC component is within realisation noise.**
*Where:* Title ("separate architecture from information"), Research question 2, Highlight 3, Abstract "Architecture" bullet, Sections 4.2 and 5.1.

*Problems:*
- **Early fusion has the same effect.** Early fusion concatenates all inputs into one encoder (Section 3.1), so it has no extra branch and no gate. Yet placebo-minus-context is −0.019/−0.023 (A) and −0.012/−0.015 (B) under early fusion (Table 2), versus −0.013 to −0.027 under gated fusion. The effect is that of **adding uninformative input columns**. That is a well-known phenomenon, e.g. spatially structured predictors degrading spatial CV in Meyer et al. (2019), which you cite. It is not specific to multi-branch architecture.
- **The PR-AUC gain is within realisation noise.** Section 4.1 reports that placebo-energy PR-AUC alone ranged **0.52–0.85 (A)** across realisations in 1403. That spread is 3–5 times the +0.07 to +0.11 "architecture" gain. A plausible mechanism: in some realisations a relocated core happens to overlap the observed hotspots.
- **The mechanism is not explored.** You report no gate weights, no analysis of where the top-decile gains occur, and no regularisation sensitivity. All hyper-parameters are fixed: 300 epochs, weight decay 10⁻⁴, no early stopping.

*What to do:*
- Rename the quantity: it is an added-input effect, not an architecture effect.
- Add the duplicate-context and capacity-matched controls (a wider context encoder with the same parameter count).
- Report gate weights per branch.
- Report per-realisation PR-AUC.
- Rerun Table 2 with one stronger regularisation setting.
- Remove PR-AUC from the Abstract and Highlights, or present it only with the realisation spread next to it.

**C5. [Error] The inference is more confident than the design allows; several verdicts and phrases overclaim.**
*Where:* Sections 3.4–3.6 and 4.1, Table 1, Section 5.5.

1. **Effective sample size.** Section 2.1 gives ~5 independent cells per month. A jackknife over 12 blocks with K − 1 = 11 degrees of freedom treats the blocks as independent. The blocks are 22 km wide and the target decorrelates at ~24 km, with correlation 0.75 at 8–10 km. The intervals are therefore too narrow, and the block sign-flip test assumes independent, symmetric block contrasts that adjacent blocks do not provide.
2. **A test known to be liberal.** Section 5.5 says the randomisation test was liberal in the companion study's null calibration. Section 4.1 nevertheless presents "with placebo B, the randomisation tests of 1403 and 1404 remain below 0.05 after Holm" as evidence. Those p-values (0.037, 0.030) also come with jackknife intervals that include zero, while the larger 1403 A effect (+0.038, interval excluding 0) has p = 0.056. Either remove the p-value claim or calibrate the test (see 6 below).
3. **Mixed frameworks.** Verdicts come from unadjusted jackknife intervals, while p-values are Holm-adjusted. "Superior, not significant after Holm" (Table 1, F1 1403 A) mixes the two. Use multiplicity-adjusted intervals for the verdicts (e.g. Bonferroni-level intervals within each family), or state that the verdicts are unadjusted.
4. **The family is undefined.** Is Family F {F1, F2, F3, F4} × {1402, 1403} × {A, B}? Is B a separate family, as a "new version" (Section 3.6) suggests? List the members and the Holm order explicitly, before and after F3 was removed.
5. **The margin δ = 0.02 is not justified.** With baseline Spearman ≈ 0.92–0.93 (Section 4.3), 0.02 is about a quarter of the remaining 1 − ρ. It is also the same size as every effect in the paper, which almost guarantees "non-inferior/inconclusive". Justify it as a smallest effect size of interest, or show the verdicts under 0.01 and 0.03.
6. **An empirical null is available at no cost.** You already have 5 realisations per construction, so 10 placebo-vs-placebo pairs per construction. Their Δ distribution is the null for "no information difference" and includes realisation variance. Report real-minus-placebo against it, for Spearman and especially PR-AUC.
7. **Seed counts are not independent replications.** "26 of 30 seeds" and "all 30 seeds negative" (Abstract, Section 4.1) pool three years whose target maps are nearly identical, two constructions, and seeds evaluated on the same 12 blocks. Present these as descriptive, not as replication. "Consistently over three years" overstates the evidence.

**C6. [Required] The 2-km buffer is far smaller than both the input reach and the target's correlation length.**
*Where:* Sections 2.2 and 3.4.

*Problem:* Inputs include window means with a half-width of 9 cells (~7.6 km). The target still correlates at ~0.75 at 8–10 km (Section 4.3), yet training cells are removed only within 2 km of the test block. Training cells 2–8 km from a test block therefore share most of their input window and have strongly correlated labels. This inflates absolute skill (Spearman 0.92–0.93). It may also affect the comparisons, because models with more inputs may exploit near-block neighbours differently.

Per-month z-standardisation over the whole domain also uses test-block values. That is a minor preprocessing leak, but you are strict about leakage elsewhere (stations, smoothed labels), so be consistent.

*What to do:* Repeat at least F1, F2 and Table 2 with buffers of ~8 km (the input reach) and ~24 km (the correlogram range), and report how the deltas change.

**C7. [Error/Required] The target-support experiment is uninformative as designed, and two of its conclusions overreach.**
*Where:* Sections 3.3, 4.3, 5.2, 5.3 and 5.5, Table 3.

- **F3** was correctly removed after M2 failed. However, the M2 manipulation was weak:
  - 15–29 days per tercile;
  - L3 grid boxes without oversampling;
  - ERA5-Land 10-m city-mean wind in a basin bounded by the Alborz;
  - a relative tercile within each season rather than an absolute threshold.

  The study you cite for calm-day compositing did something different: Goldberg et al. (2024) oversample the satellite data over multiple years, filtering to include data when near-surface wind speeds are <3.2 m/s. The general conclusion in Section 5.3 ("A sharper target … would need another instrument or oversampling, not wind selection") is therefore not supported. Your design tested neither multi-year oversampling nor an absolute wind threshold. Restrict the conclusion to your implementation.
  
  Please also check that Liu et al. (2024) actually uses calm-day composites as Sections 1 and 3.3 imply. My understanding is that it is a flux-divergence method, but I am unsure.
- **F4** uses a relative gain, Δ / (1 − ρ_energy), computed per block. With ρ near 1 the statistic is ill-conditioned: edge blocks of 3–5 cells had ρ = 1 and dropped out of the sign-flip test (Section 4.3). The 95 % interval of −0.549 to +0.189 reflects this. Spearman over 3–5 cells is not a meaningful statistic. Also, the primary metric is a domain-wide Spearman (Section 3.4), but F3/F4 are per-block. These are different estimands; say so.
- **Smoothing is confounded in two ways.**
  - Inputs reach ~7.6 km while σ goes up to 12 km. You acknowledge this, but the control is easy: smooth the inputs with the same kernel, or add a window at the target scale.
  - Training labels are smoothed over training cells only, with normalised convolution, while the test truth is smoothed over all cells. Near the held-out block the two differ systematically, and the difference grows with σ. This alone can make "skill fall with smoothing". Quantify the train/test label mismatch per σ.
- *[Taste]* Given these problems and the null outcome, I would move the target-support experiment to the supplement (or a short section) and give the placebo control the paper's full weight. As it stands, the title, Highlights and Discussion are split across two messages.

**C8. [Required] Absolute skill and simple baselines are missing.**
*Where:* Section 4. Absolute skill appears only in passing in Section 4.3 ("0.92–0.93").

*Problem:* With ~5 effective units and one dominant urban core, a trivial predictor may already reach Spearman ≈ 0.9. Candidates include distance to the city centre, latitude/longitude or kriging of the target from training blocks. In that case all the effects in the paper (~0.02) sit on a ceiling.

*What to do:*
- Add a table of absolute Spearman and PR-AUC per model and year: context-only, energy, all three sources, each placebo, early and gated.
- Add a trivial spatial baseline and a non-neural model (e.g. gradient boosting) on the same inputs.
- State the PR-AUC no-skill baseline (0.10).

**C9. [Required] Novelty is overstated, and the overlap with the companion study cannot be assessed.**
*Where:* Section 1 ("We found no study …"), Sections 2.1, 3.1 and 5.5.

*Novelty of the placebo.*
- Comparing a real feature against a permuted copy of itself is established: Boruta finds relevant features by comparing original attributes' importance with importance achievable at random, estimated using their permuted copies (shadows) (Kursa & Rudnicki 2010, J. Stat. Softw. 36(11), doi:10.18637/jss.v036.i11).
- Knockoffs (C1) formalise the conditional version.
- Spatially constrained nulls (Harms et al. 2001; Alexander-Bloch et al. 2018; Burt et al. 2020) and meaningless spatial predictors (Fourcade et al. 2018; Behrens & Viscarra Rossel 2020) are cited or closely related.

What is new is narrower: using a spatial surrogate as a *whole-branch* control during training of a multimodal fusion network. That is a useful, modest increment. Position it against shadow features and knockoffs explicitly, and soften "We found no study".

*The target-resolution part* repeats well-known scale and modifiable-areal-unit (MAUP) effects (Gehlke & Biehl 1934; Gotway & Young 2002) and adds a null result. It is not new.

*Overlap with the companion study.* The papers share:
- the city, grid, target and years (presumably);
- the network ("the fusion network of the companion study, without a graph");
- the five-seed ensembling;
- the effective-n estimate and the null calibration of the test;
- the central finding that fusion does not beat the best single source.

The companion's title ("Reach, not graphs …") makes the same kind of argument: an apparent architectural gain is an artefact. The editor and reviewers need the companion manuscript as confidential material. They also need a table of what is reused (data, code, model runs, figures) and a statement of whether any runs in Tables 1–3 are identical to runs in the companion. The two papers may be better merged, or the shared parts cut back to citations.

**C10. [Required] The pre-registration cannot be verified, and several post-hoc elements are not labelled as such.**
*Where:* Section 3.6, "Deviation" paragraph.

- **Verifiability.** A commit hash (`9e28ddc`) in a project repository is not an independent timestamp: git dates can be set freely, and the repository is not public. Deposit the protocol on OSF or Zenodo (or another time-stamped registry) and link it, or provide a verifiable public commit date.
- **Prior knowledge.** The registration came after the companion study on the same data, network and target. The authors already knew that fusion did not beat the best single source, and the "expected directions" reflect that. Disclose this. The study is confirmatory only conditional on that knowledge.
- **Unregistered results presented as main results.** Research question 2 is presented as part of "a pre-registered design" (Section 1). Its estimates (Table 2) carry "no verdict" and are not listed in Families F or C. The same applies to the early-fusion real-vs-placebo comparisons (Section 4.2, text only) and the PR-AUC arguments of Section 5.1. State for each whether it was registered (secondary, descriptive) or post hoc.
- **Unlisted deviations.**
  - The F4 sign-flip test ran on fewer blocks than the registered 2¹² assignments. This is a deviation and belongs in the Deviation paragraph.
  - The pre-registration apparently did not say what to do if placebo B also failed. Reporting both and relying on their agreement is a post-hoc decision; say so.
- **Results against your own expectation.** F1 was expected to be "equivalent or inconclusive"; the observed verdicts are non-inferior or superior. Say explicitly that this went against the expected direction. It is the most interesting result and needs C1 to be interpreted.

**C11. [Required] Reproducibility and declarations are incomplete.**
*Where:* Front matter, Data and code availability, CRediT, Funding, AI declaration.

- **No repository URL.** The paths `docs/paper/fusion_paper/…` and branch `paper_fusion` are given without one.
- **The availability section is truncated.** The "Code" parent bullet is missing; lines 530–531 start with an orphaned sub-list.
- **Restricted data are undocumented.** The energy, traffic and station data are "[as in the companion study]". State the access conditions here; the section must stand on its own. If the municipal data are restricted, say so and describe how to request access.
- **Missing artefacts.** Release the placebo realisations (or their random seeds), the fold and block definitions, and the out-of-fold predictions. These allow every table to be recomputed without retraining.
- **Truncated or placeholder front matter.** The author list ("Fatemeh V") and CRediT entry are truncated, the CRediT section refers to "both authors" while one is listed, and the affiliation, e-mail, funding and AI declaration are placeholders. *[Taste]* Make the AI declaration specific: which code, which text, and how it was checked.

---

## D. Minor comments

1. **Title** — [Taste/Error]: "hotspot mapping", but the primary metric is a whole-map Spearman and PR-AUC is explicitly descriptive (Section 3.4). Retitle, or make hotspots primary.
2. **Highlights.** Highlight 5 is 104 characters; Elsevier's usual limit is 85, and Highlight 4 is exactly 85. Highlight 2 ("separate the two") overclaims given C1–C4. Highlight 3 is not supported by the numbers (E-2).
3. **Abstract.** It is 348 words and bulleted; most Elsevier journals expect a single paragraph of 250–300 words, so check the guide for authors. "Cells of 1 km" should read ~0.75 × 0.93 km. "R² = 0.98, 0.95 and 0.89" should say gradient boosting (OLS gives 0.89, 0.60 and 0.77).
4. **Section 1:** "correlate at 0.58 in single cells but at 0.998 once smoothed over 8 km" has no source. Is it from the companion study, or computed here? Give the table or reference.
5. **Section 1:** Mena et al. (2025) is crop-yield fusion. That is fine as an example of branch-per-source fusion, but say so, since the sentence is about urban emissions.
6. **Section 2.1:** say how the effective number of ~5 independent cells per month was computed (method and reference), not only "(Vafaeian & Khatibi, in review)".
7. **Section 2.1:** "that the emission inventory assigns any CO₂" → "to which the emission inventory assigns any CO₂".
8. **Section 2.1/3.3:**
   - Specify the TROPOMI L3 product (e.g. the Earth Engine OFFL L3 NO₂ collection), the qa_value filter, the regridding method and the processor versions across 2023–2026.
   - The 3.5 × 5.5 km pixels are mapped onto ~0.84-km cells, so the "native" target already has an effective resolution of ~5 km. Say so where you discuss resolution.
   - Discuss the coarse a-priori profiles used in the air-mass factor; they reduce urban contrast.
9. **Section 2.2:**
   - Why CO from the stations, rather than station NO₂, for an NO₂ target?
   - A CO₂ inventory is used as a proxy for NOₓ, but NOₓ/CO₂ ratios differ strongly by sector (power plants, diesel freight). Discuss.
10. **Section 2.2:** For 1404 the 1403 inventory and traffic layers stand in. C1/C2 therefore test transfer to a new target year with the *same* layers, not new data. Say so wherever "held-out year" is used as evidence.
11. **Section 3.1:** what does the softmax gate take as input? Why 300 epochs with no early stopping? Were these hyper-parameters tuned on Tehran in the companion study?
12. **Section 3.2 A:**
    - Mirror padding creates four mirrored copies of the core before phase randomisation; discuss the effect.
    - IAAFT on zero-inflated fields: how are ties handled in the rank step?
    - "Its values are set by rank to the real values": over which cells, the full grid or the label domain? See C2.
13. **Section 3.2 B:** "shifted cyclically, month by month, by the same random offset" reads as if the offset changed by month. Say "the same offset for all months and layers of a realisation".
14. **Section 3.3:** composites are placed on the middle month of their season. Which month's inputs (ERA5, HDD, VIIRS) are paired with them? Define the Jalali seasons; the text uses "Solar Hijri" and "Jalali" interchangeably, so pick one.
15. **Section 3.3:** "M3 split-half reliability ≥ 0.90": reliability of which target, monthly or composite?
16. **Section 3.4:** give the number of labelled cells in each of the 12 blocks. Section 4.3 implies some blocks hold only 3–5 cells, and such blocks have little meaning in a jackknife or sign-flip.
17. **Section 3.5:**
    - Define the verdict rules formally: superior = lower bound > 0? Equivalent = interval inside ±δ? Non-inferior = lower bound > −δ?
    - Explain how a domain-wide Spearman is decomposed into the per-block contrasts that the sign-flip test needs.
    - Define "matching placebo realisation" for the seed comparisons: is seed *i* of the real model paired with realisation *i*, trained with which seed?
18. **Section 3.6 F2** tests traffic and stations jointly, so it cannot attribute anything to either. Report them separately, at least descriptively; Table 4 suggests they behave differently.
19. **Table 1:** the column "Seeds > 0" shows "0 / 5" while the text says "all 30 seeds negative". Zero would also be possible; report the signs explicitly.
20. **Table 2 / Section 4.2:** the early-fusion real-vs-placebo results (PR-AUC −0.004 to −0.052; Spearman +0.021 to +0.037) appear only in the text. Put them in Table 1 or a new table. Why is early fusion missing for 1404?
21. **Table 3:**
    - Add jackknife intervals (Fig. 3b has them).
    - Explain the blank 1404 cells.
    - 1403 is non-monotone at σ = 4 km (−0.015 → −0.019).
22. **Section 4.3:** "probably because the models' neighbourhood inputs reach only about 7.6 km" is speculation. Test it (C7) or label it.
23. **Section 4.4:**
    - The M4 table has no number.
    - Define "median of the largest correlogram difference" (median over layers? over realisations?).
    - Give the values per layer and per realisation in the supplement.
    - B's |Spearman| range of 0.04–0.10 touches the < 0.10 threshold, yet the text says "Both constructions removed the location"; check that this is strictly true.
24. **Table 4:**
    - The stations entries "(−0.77 / −0.05)" and "(−0.43 / −0.68)" are unexplained (1402/1403?).
    - The caption says "1403 within ±0.02 except where stated", but nothing else is stated.
    - Gradient boosting is below OLS for road length (< 0.03 vs 0.06–0.15), which suggests overfitting; comment.
    - The 0.98 for residential CO₂ follows from how the layer is built (allocation proportional to residential volume, which is total minus non-residential volume, both in context). Say this; it confirms the allocation recipe rather than being a finding.
25. **Section 5.1:** "about +0.19 in PR-AUC" uses 1402 only (1403: +0.17; 1404: +0.02). Give the range. "With any data, cost ranking skill" is contradicted by early fusion (C4).
26. **Section 5.2:** this explanation conflicts with C1. Rewrite after the conditional placebo.
27. **Section 5.3:** "set by the footprint … rather than by daily transport" is not tested; mark it as a hypothesis.
28. **Section 5.4:** add to the recommendations a conditional placebo, placebo-vs-placebo spreads, and checks of the placebo's distribution within the training domain.
29. **Section 5.5:** add C1, C2 and C6 to the limitations.
30. **Fig. 1:** Section 4.4 cites "Fig. 1d, e", but the caption defines no panel letters. State whether the colour scale is shared between panels.
31. **Fig. 2:** Section 4.2 refers to "the grey rows of Fig. 2"; the caption mentions neither grey rows nor whether early fusion is shown. The caption lists 1404, but there are no early-fusion runs for 1404.
32. **Fig. 3:** panel (a) shows 1402 only; add 1403. Panel (b) has intervals that Table 3 lacks; make them consistent.
33. **References:**
    - Remove the remaining "[check]" tags after checking.
    - Behrens & Viscarra Rossel (2020) is correct as cited: DOI 10.1038/s41598-020-73773-y, authors Thorsten Behrens and Raphael A. Viscarra Rossel.
    - Complete the author list of Moudrý et al. (2023).
    - Add Kursa & Rudnicki (2010), Candès et al. (2018), Harms et al. (2001) and Alexander-Bloch et al. (2018) per C3 and C9.
34. **Front matter:** move the repository branch and folder notes from the italic note under the title into the Data availability section.

---

## E. Numbers that disagree

| # | Location 1 | Value 1 | Location 2 | Value 2 | Comment |
|---|---|---|---|---|---|
| 1 | Abstract, "Architecture" bullet | Spearman lowered by 0.013–0.027 | Section 4.2 text; Table 2 early, B, 1402 | 0.012–0.027; −0.012 | The Abstract uses gated fusion only, the text includes early fusion. |
| 2 | Abstract and Highlight 3 | Placebo branch "changed skill as much as the real sources did" | Section 4.2 (real energy − context) vs Table 2 (placebo − context), gated | Spearman: +0.004/+0.013/−0.001 vs −0.013 to −0.027; PR-AUC: +0.19/+0.17/+0.02 vs +0.005 to +0.112 | Larger for Spearman, smaller for PR-AUC; "as much as" is wrong for both. |
| 3 | Abstract | "2,436 cells of 1 km" | Section 2.1 | 1/120° ≈ 0.75 × 0.93 km | |
| 4 | Abstract and Section 5.2 | R² = 0.98, 0.95, 0.89 | Table 4, OLS column | 0.89, 0.60, 0.77 | The Abstract does not say the values are gradient boosting. |
| 5 | Section 4.1, PR-AUC energy real − placebo | "+0.08 to +0.09 in 1402–1403" | Section 4.2 (+0.17 in 1403) minus Table 2 gated PR-AUC, B, 1403 (+0.074) | ≈ +0.096 | Possibly rounding of "+0.17"; report the exact values. |
| 6 | Section 5.1 | Energy credited with "about +0.19" PR-AUC | Section 4.2 | +0.19 / +0.17 / +0.02 | Only 1402 is quoted. |
| 7 | Table 1, F1 1403 A | Δ +0.038, interval (+0.002, +0.075), p = 0.056 | Table 1, F1 1403 B | Δ +0.026, interval (−0.011, +0.063), p = 0.037 | Not an arithmetic error, but the interval and the test point in opposite directions. Explain (C5). |
| 8 | Section 3.6, M4 | Location removed if median \|ρ\| < 0.10 | Section 4.4 table, B | 0.04–0.10 | The upper end touches the threshold, yet the text says "both constructions removed the location". |
| 9 | Section 3.4 | Primary metric: domain-wide Spearman | Sections 3.5 and 4.3 (F3/F4) | Per-block Spearman, blocks of 3–5 cells with ρ = 1 | Different estimands. |
| 10 | Section 3.1 and Research question 2 | The effect is of "the branch itself" | Table 2, early fusion (no extra branch) | −0.012 to −0.023 | The effect appears without a branch. |
| 11 | Highlight 4 and Abstract "Energy" bullet | Energy "beat" its placebo | Table 1 verdicts | Non-inferior in 5 of 6; superior once, not significant after Holm | "Beat" holds only for point estimates. |
| 12 | Section 5.1 | "Conclusions do not rest on one construction" | Section 4.4 | 0 of 5 realisations pass M4 for either construction | Agreement between two failed checks. |
| 13 | Table 4 caption | "1403 within ±0.02 except where stated" | Table 4, stations row | Two values per cell, not explained | |
| 14 | Section 4.2 | "The grey rows of Fig. 2" | Fig. 2 caption | No grey rows mentioned | |
| 15 | Section 4.4 | "Fig. 1d, e" | Fig. 1 caption | No panel letters defined | |
| 16 | Fig. 2 caption | Covers 1402, 1403 and 1404 | Table 2 | Early fusion not run for 1404 | Check what Fig. 2 actually shows. |
| 17 | Fig. 3b | Fusion gain with jackknife intervals | Table 3 | No intervals; 1404 mostly blank | |
| 18 | Section 3.6 | Family F tests F1–F4 with Holm | Section 4.3 | F3 removed; F4 sign-flip on fewer than 12 blocks | The family and the test changed; list both under Deviations. |

These agree, which I checked:
- Energy +0.018 to +0.038 and 26/30 seeds (Abstract vs Table 1).
- Traffic and stations −0.012 to −0.021 and 0/30 seeds.
- PR-AUC +0.05 to +0.11 (Section 4.2 vs Table 2).
- The decomposition real − placebo = (real − context) + (context − placebo), for every gated Spearman cell.
- The early-fusion PR-AUC decomposition.
- F3 D ≈ −0.004 from Table 3.
- Grid dimensions (126 × 72 cells over 1.05° × 0.60°).
- Input counts per source (21 / 13 / 3 / 17).
- The M2 numbers against the M2 criteria.

## F. Questions to the authors

1. Are the inventory layers exactly zero outside the label domain? For each placebo realisation, what fraction of label-domain cells is non-zero?
2. What does the gate see, and what weights does it give placebo branches compared with real ones?
3. How is a domain-wide Spearman difference split into the per-block contrasts used in the sign-flip test?
4. Which seed and realisation are paired in "Seeds > 0"?
5. Is the station placebo rebuilt per fold?
6. Which members, years and constructions make up Family F and Family C, in what Holm order, before and after F3 was removed?
7. When, and where publicly, was the protocol registered relative to the companion study's results? What did it specify if placebo B also failed?
8. Are any model runs shared with the companion study?
9. How was the effective sample size of ~5 per month computed?
10. Why CO rather than NO₂ from the stations?
11. Which TROPOMI L3 product, qa filter and regridding did you use?
12. Why were early fusion, the composites and the σ ≠ 8 km targets not run for 1404?
13. Were the fixed hyper-parameters tuned on these data in the companion study?
14. Is M3 computed on the monthly target or on the composites?

## G. What would move my recommendation to minor revision

1. **The conditional (context-preserving) placebo and the duplicate-context branch (C1, C4).** Reinterpret F1 and Table 2 accordingly. If energy then equals its conditional placebo, the paper becomes a clean and useful negative-result paper. I would support that.
2. **Placebo diagnostics within the label domain**, M4 recomputed there, the per-realisation dose-response, and the bracketing controls (C2, C3).
3. **Placebo-vs-placebo null distributions**, explicit families, multiplicity-adjusted verdicts, a justified δ, and the p-value claims dropped (C5).
4. **Sensitivity to an 8-km and a 24-km buffer** (C6).
5. **An absolute-skill table and trivial spatial baselines** (C8).
6. **Reframed claims:**
   - "architecture" becomes "added-input effect";
   - PR-AUC leaves the Abstract and Highlights;
   - the "placebo check failed" label appears wherever verdicts appear;
   - the target-support part is shortened or moved to the supplement, with Section 5.3's general claim restricted (C7).
7. **The companion manuscript, a table of what is reused, a publicly timestamped pre-registration, a full deviation list, a public code repository and complete declarations** (C9–C11).

Items 1 and 2 matter most. Without them, the paper's main methodological claim stays unsupported.

---

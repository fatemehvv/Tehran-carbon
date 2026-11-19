# Review prompt: check the paper as the supervisor would

Attach `paper_esi.pdf` and `supplementary/ESM_1.pdf` from this folder, then paste the text below.

```text
I am a master's student. Before I send this manuscript to my supervisor, I want it to need as few corrections as possible in his eyes. Attached are the manuscript (paper_esi.pdf, prepared for Springer's Earth Science Informatics) and its supplementary file (ESM_1.pdf).

My supervisor's background is computational intelligence, not atmospheric science. He has reviewed for IEEE Transactions on Fuzzy Systems, for the IEEE International Conference on Fuzzy Systems (FUZZ-IEEE 2019) and for Artificial Intelligence in Medicine (Elsevier). Read the manuscript as he would: a senior AI/ML reviewer who checks the soundness of the experimental design, the fairness of model comparisons, the statistics, the clarity of the method and whether the claims follow from the results.

Please do the following.

1. Methodology through an AI/ML reviewer's eyes. Look for anything he is likely to question: fairness of baselines (tuning, ensembling, information each model sees), leakage, the validation design, the choice of metrics, statistical tests and multiple-comparison control, reproducibility, ablations that are missing, and limitations that are not stated clearly enough. Distinguish real weaknesses from points that are already answered in the text or in the supplementary file.

2. Claims against evidence. Check that the title, abstract, discussion and conclusions claim no more than the results show, and that the numbers in the text match the tables, figures and the supplementary file.

3. Presentation. Structure, flow between sections, clarity of the research questions and the contribution, notation and terminology used consistently, figure and table quality, and anything that would make an expert reader stop and re-read.

4. Writing. Mark sentences that still read as machine-written or stiff: repetitive sentence patterns, uniform rhythm, stock connectors, formulaic hedges, vague or inflated wording. For each, quote the sentence and give one rewrite that keeps every number, citation and claim exactly as it is. Do not rewrite sentences that already read naturally.

5. His likely first questions. List the five to ten questions he is most likely to ask in our first meeting about this paper, with a short note on whether the manuscript already answers each one and where.

Known and intentional; do not report these as problems:
- The title was chosen by the authors and stays.
- The traffic and energy layers are estimates, not measurements, and the text says so on purpose.
- Non-public data (estimated traffic and energy layers, station data) are deliberately not shared and their sources are not named; code is available on request.
- A master's thesis is mentioned as unpublished and is not in the reference list (Springer's rule for unpublished work).
- The GNN was not tuned; this is stated as a limitation.
- The single-column layout is the submission format.
- The use of an AI tool is disclosed in Section 3.6, as Springer requires.

Output (answer in Persian; keep quotations and suggested English text in English):
- A one-line verdict: ready to send to the supervisor / ready after minor fixes / needs more work.
- A table of at most 25 issues, most important first, with columns: number, section and page, problem, quotation, suggested fix, priority (must fix / should fix / optional).
- The list of likely questions from item 5.
- Do not invent facts or rules. If something cannot be checked from the two files, say so.
```

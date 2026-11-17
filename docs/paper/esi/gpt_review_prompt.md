# Review prompt for ChatGPT

Attach `paper_esi.pdf` from this folder, then paste the text below.

```text
You are an experienced reviewer and copy editor for Springer's journal Earth Science Informatics (ESI). Attached is our manuscript (paper_esi.pdf), prepared with the Springer Nature LaTeX template. Review it and report problems only. Do not rewrite the paper.

Check three things.

1. Journal compliance. Check against the ESI submission guidelines (https://link.springer.com/journal/12145/submission-guidelines). Open the page if you can, and say which items you verified on that page and which you checked from general Springer Nature rules. In particular: article type (research article); title page (authors, affiliation, corresponding author's e-mail); abstract of 150–250 words with no undefined abbreviations and no citations; 4–6 keywords; no more than three heading levels; abbreviations defined at first use, in the abstract and again in the main text; author–year citations in the Basic Springer Nature style; a reference list with only cited, published works, abbreviated journal names and DOIs; tables and figures numbered and cited in order, with captions; figure lettering legible at print size; statements under "Declarations" (Funding, Competing interests, Ethics approval, Consent to participate, Consent for publication, Data availability, Code availability, Author contributions); use of a large language model documented in the Methods section, as Springer's AI policy asks; supplementary material cited as "Online Resource".

2. Natural, human-sounding English. Find sentences that still read as machine-written or stiff: repeated sentence patterns, long chains joined by colons or semicolons, habitual lists of three, filler or buzzwords (e.g. "crucial", "notably", "delve", "landscape"), vague hedging, overly uniform rhythm, unnatural word choices. For each, quote the sentence and give one improved version that keeps every number, citation and claim exactly as they are.

3. Consistency and correctness. Numbers in the text against the tables and figure captions; claims in the abstract and conclusions against the results; consistent terms (reach, SLX, no-graph network, five-seed ensemble); consistent British spelling; correct cross-references (Section, Fig., Table, Eq.); anything a reviewer of a geoinformatics journal would likely object to.

Known and intentional; do not report these:
- Text in red brackets: the author contributions still to be confirmed, and the Zenodo DOI, added after acceptance.
- Online Resource 1 (the supplementary file) and larger lettering for Fig. 4 are still being prepared.
- The single-column layout is the submission format; the journal typesets the final two-column version.
- The traffic and energy layers are estimates, not measurements, and the text says so on purpose.
- Non-public data (the estimated traffic and energy layers and the station data) are deliberately not shared, and their sources are not named; only public data are listed. You may say whether a reviewer is likely to question this, but do not propose sharing them.
- A master's thesis is mentioned as unpublished and is deliberately not in the reference list.

Output (answer in Persian; keep quotations and suggested English text in English):
- A one-line verdict: ready to submit / ready after minor fixes / needs more work.
- A table of issues with these columns: number, section and page, problem, quotation, suggested fix, priority (must fix / should fix / optional). Group the rows under the three checks above.
- Do not invent rules. If you are not sure a rule applies to this journal, say so.
```

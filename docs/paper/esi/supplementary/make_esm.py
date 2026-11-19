"""Online Resource 1 of the Earth Science Informatics version: writes online_resource_1.tex.

Section S1 (literature search), Table S2 (predictor layers) and Table S3 (model settings) are written below from
docs/paper/lit/README.md, configs/config.yaml and the model code. Table S1 is built from the committed evaluation tables
and Table S4 (RQ1 runs) from the committed training logs of docs/paper/results_stageA/logs; nothing is re-run:

  registered family (22 tests)           docs/paper/results_stageA/evaluate_paper2/
  exploratory, equal reach (32 tests)    docs/paper/results_review/evaluate_step5/
  exploratory, larger buffers (12 tests) docs/paper/results_review/evaluate_buffer/
  original-study predictions (18 tests)  docs/paper/results_stageA/evaluate_thesis/

Model names come from the run registries (runs_stageA.csv, runs_review.csv); the eight runs of the original study
that are not in them are named from their experiment.

    python docs/paper/esi/supplementary/make_esm.py && (cd docs/paper/esi/supplementary && bash build.sh)
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
STAGE_A, REVIEW = PAPER / "results_stageA", PAPER / "results_review"

FAMILIES = [  # (key, title, folder, hypotheses in order)
    ("F1", "Registered comparisons (Holm family of 22 tests)", STAGE_A / "evaluate_paper2",
     ["PA_H1", "PA_H1_drivers", "PA_H2", "PA_H2_reach", "PA_H1c", "PA_H2c", "PA_H2c_all"]),
    ("F2", "Exploratory comparisons at equal reach (Holm family of 32 tests)", REVIEW / "evaluate_step5",
     ["PR_depth_L1", "PR_depth_L2", "PR_depth_L3", "PR_depth_L4", "PR_depth_L5", "PR_graph_r23", "PR_graph_r38",
      "PR_c1404_r30", "PR_c1404_r38"]),
    ("F3", "Exploratory comparisons with larger buffers, driver inputs (Holm family of 12 tests)",
     REVIEW / "evaluate_buffer", ["PR_buffer02", "PR_buffer10", "PR_buffer20"]),
    ("F4", "Predictions of the original study, with its pre-processing (Holm family of 18 tests)",
     STAGE_A / "evaluate_thesis", ["H1", "H2", "P2_reach", "H1c", "H2c", "P2c_reach"]),
]

PURPOSE = {
    "PA_H1": "RQ1: gated fusion of traffic, energy and station data against the best single source",
    "PA_H1_drivers": "RQ1: gated fusion of traffic and energy against the better of the two",
    "PA_H2": "RQ2: GNN against the pre-specified reference of the original study",
    "PA_H2_reach": "RQ2: GNN against the best SLX model at any reach",
    "PA_H1c": "RQ1, held-out year: fused GNN against the energy-only GNN",
    "PA_H2c": "RQ2, held-out year: GNN against the pre-specified reference",
    "PA_H2c_all": "RQ2, held-out year: GNN against the best conventional model",
    "PR_depth_L1": "GNN with one layer against SLX at the same reach",
    "PR_depth_L2": "GNN with two layers against SLX at the same reach",
    "PR_depth_L3": "GNN with three layers against SLX at the same reach",
    "PR_depth_L4": "GNN with four layers against SLX at the same reach",
    "PR_depth_L5": "GNN with five layers against SLX at the same reach",
    "PR_graph_r23": "GNN against the same network fed box means, 22.7 km",
    "PR_graph_r38": "GNN against the same network fed box means, 37.8 km",
    "PR_c1404_r30": "Held-out year: GNN against SLX at 30.3 km",
    "PR_c1404_r38": "Held-out year: GNN against SLX at 37.8 km",
    "PR_buffer02": "GNN against SLX at 22.7 km, driver inputs, 2-km buffer",
    "PR_buffer10": "GNN against SLX at 22.7 km, driver inputs, 10-km buffer",
    "PR_buffer20": "GNN against SLX at 22.7 km, driver inputs, 20-km buffer",
    "H1": "RQ1 of the original study: fusion against the best single source",
    "H2": "RQ2 of the original study: GNN against the pre-specified SLX reference",
    "P2_reach": "Original-study GNN against the best SLX model at any reach",
    "H1c": "Held-out year of the original study: fused against energy-only GNN",
    "H2c": "Held-out year of the original study: GNN against the pre-specified reference",
    "P2c_reach": "Held-out year of the original study: GNN against the best SLX at any reach",
}

ESTIMATOR = {"ols": "OLS", "gbm": "GBM", "rf": "RF"}
SOURCES = {"traffic": "T", "energy": "E", "pollution": "S"}
ORIGINAL = {  # original-study runs that are not in the committed registries
    "rq2_base_slx_gbm_v1": "SLX-GBM, 12.6 km",
    "rq2_sage_v2": "GNN 3L, 22.7 km",
    "c1404_base_slx_rf_v1": "SLX-RF, 12.6 km",
    "c1404_sage4_single_v1": "GNN 4L, E only",
    "c1404_sage4_v1": "GNN 4L, T+E+S",
    "c1404_sage5_v1": "GNN 5L, 37.8 km",
}


CELL_KM = 22.7 / 27            # nominal cell size used for reach in the article (27 cells = 22.7 km)


def lags_km(lags: str) -> str:
    return "0 km" if not lags else f"{CELL_KM * max(int(x) for x in lags.split('+')):.1f} km"


def name(run: str, reg: pd.DataFrame, role: str, hyp: str) -> str:
    """Short model label of a run."""
    if run in reg.index:
        exp, var = reg.at[run, "experiment"], reg.at[run, "variant"]
        var = "" if (isinstance(var, float) and math.isnan(var)) else str(var)
        kv = dict(p.split("=", 1) for p in var.split("|") if "=" in p)
        if "baseline.name" in kv:
            est = ESTIMATOR[kv["baseline.name"]]
            lab = f"SLX-{est}, {lags_km(kv.get('baseline.lags', ''))}" if kv.get("baseline.lags") else f"{est}, own cell"
            if exp.startswith("paper_") and not exp.startswith("paper2_"):
                lab += " (orig.)"
            if "buffer_km" in kv:
                lab = f"SLX-{est}, 22.7 km"
            return lab
        if "buffer_km" in kv and "baseline.name" not in kv:
            return "GNN 3L, 22.7 km" if exp == "paper2_rq2_buffer_v2" else exp
        if exp == "paper2_slx_buffer_v2":
            return f"SLX-{ESTIMATOR[kv['baseline.name']]}, 22.7 km"
        if exp == "paper2_rq2_sage_v1":
            return "GNN 3L, 22.7 km"
        if exp == "paper2_rq2_depth_v1":
            L = int(kv["backbone.layers"]); return f"GNN {L}L, {CELL_KM * 9 * L:.1f} km"
        if exp == "paper2_c1404_sage4_v1":
            return "GNN 4L, 30.3 km"
        if exp == "paper2_c1404_sage5_v1":
            return "GNN 5L, 37.8 km"
        if exp == "paper2_c1404_sage4_single_v1":
            return "GNN 4L, E only"
        if exp == "paper2_rq2_mlp_lag_v1":
            return f"No-graph, {lags_km(kv['spatial_lags'])}"
        if exp == "paper2_rq1_ablation_v1":
            mods = "+".join(SOURCES[m] for m in kv["modalities"].split("+"))
            return f"Fusion {mods}" if "+" in mods else f"{mods} only"
        return exp
    exp = re.sub(r"^\d{8}-\d{6}_|_[0-9a-f]{6}$", "", run)
    if exp == "rq1_ablation_v2":
        return "Fusion T+E+S" if role == "candidate" else "Best single source"
    return ORIGINAL.get(exp, exp)


def num(x, sign=True, nd=3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "−")


def pval(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    if x >= 0.995:
        return "1" if x >= 0.9995 else f"{x:.3f}"
    return f"{x:#.2g}"                      # two significant figures, trailing zeros kept (0.10)


def ci(lo, hi) -> str:
    return "--" if any(isinstance(v, float) and math.isnan(v) for v in (lo, hi)) else f"[{num(lo)}, {num(hi)}]"


METRIC = {"spearman": "Spearman", "pr_auc": "PR-AUC"}


def verdict(v) -> str:
    return "--" if not isinstance(v, str) else v.replace("superior beyond margin", "superior$^\\dagger$")
LABEL = {"q=0.8": "top 20\\,\\%", "q=0.95": "top 5\\,\\%", "5x5 cells": "superpixels"}


def table_s1() -> tuple[str, str, str]:
    reg = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in
                     [STAGE_A / "runs_stageA.csv", REVIEW / "runs_review.csv"]]).drop_duplicates("run_id")
    reg = reg.set_index("run_id")
    prim, sens, seeds, key = [], [], [], []
    for fam, title, folder, hyps in FAMILIES:
        holm = pd.read_csv(folder / "paper_holm.csv", encoding="utf-8-sig")
        heads = {}
        for part, ncol in ((prim, 13), (sens, 10), (seeds, 8)):  # the family title, added only above its first row
            heads[id(part)] = (len(part), f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textbf{{{fam}. {title}}}}}\\\\[2pt]")
        for h in hyps:
            t = pd.read_csv(folder / f"paper_{h}.csv", encoding="utf-8-sig")
            hid = h.replace("_", "\\_")
            key.append(f"{hid} & {fam} & {PURPOSE[h]} \\\\")
            for _, r in t.iterrows():
                cand, ref = name(r.candidate_run, reg, "candidate", h), name(r.reference_run, reg, "reference", h)
                y, m = int(r.year), METRIC[r.metric]
                if r.part == "primary":
                    hp = holm[(holm.hypothesis == h) & (holm.year == r.year) & (holm.metric == r.metric)]
                    ph = float(hp.p_holm.iloc[0]) if len(hp) else float("nan")
                    prim.append(" & ".join([hid, cand, ref, str(y), m, num(r.delta), ci(r.jack_low95, r.jack_high95),
                                            ci(r.jack_low90, r.jack_high90), verdict(r.verdict), pval(r.p_signflip_pooled),
                                            pval(r.get("p_signflip_blocks", float("nan"))), pval(ph),
                                            ci(r.get("boot_low95", float("nan")), r.get("boot_high95", float("nan")))])
                                + " \\\\")
                elif r.part in ("threshold", "superpixel"):
                    sens.append(" & ".join([hid, cand, ref, str(y), LABEL.get(r.label, r.label), m, num(r.delta),
                                            ci(r.jack_low95, r.jack_high95), ci(r.jack_low90, r.jack_high90),
                                            verdict(r.verdict)]) + " \\\\")
                elif r.part == "seeds":
                    seeds.append(" & ".join([hid, cand, ref, str(y), m, num(r.delta), num(r.delta_min),
                                             num(r.delta_max)]) + " \\\\")
        for part in (seeds, sens, prim):
            at, head = heads[id(part)]
            if len(part) > at:
                part.insert(at, head)
    return "\n".join(key), "\n".join(prim), "\n".join(sens), "\n".join(seeds)


LOG_RE = re.compile(r"\] (\S+) \[([^\]]*)\]: ")
CV_RE = re.compile(r"(\d{4}) CV: pr_auc=([\d.]+),.*spearman=([\d.]+)")
INPUTS = {"traffic": "Traffic", "energy": "Energy", "pollution": "Stations", "traffic+energy": "Traffic + energy",
          "traffic+pollution": "Traffic + stations", "energy+pollution": "Energy + stations",
          "traffic+energy+pollution": "Traffic + energy + stations"}


def log_scores(path: Path) -> dict:
    """{variant: {year: (PR-AUC, Spearman)}} of the five-seed ensembles, from a training log."""
    out, cur = {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = LOG_RE.search(line)
        if m:
            cur = m.group(2)
            continue
        m = CV_RE.search(line)
        if m and cur is not None:
            out.setdefault(cur, {})[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return out


def table_s4() -> str:
    """RQ1 runs: gated networks with each subset of input groups (context always included)."""
    logs = STAGE_A / "logs"
    cv = log_scores(logs / "train_paper2_rq1_ablation_v1.log")
    ctx = log_scores(logs / "train_paper2_rq1_context_only_v1.log")
    single = log_scores(logs / "train_paper2_c1404_sage4_single_v1.log")
    fused = log_scores(logs / "train_paper2_c1404_sage4_v1.log")
    rows = []
    for mods, lab in INPUTS.items():
        sc = cv[f"modalities={mods}|fusion=gated"]
        rows.append(f"{lab} & {sc[1402][0]:.3f} / {sc[1403][0]:.3f} & {sc[1402][1]:.3f} / {sc[1403][1]:.3f} \\\\")
    sc = next(iter(ctx.values()))
    rows.append(f"None (context only) & {sc[1402][0]:.3f} / {sc[1403][0]:.3f} & {sc[1402][1]:.3f} / {sc[1403][1]:.3f} \\\\")
    held = []
    for mods in ("traffic", "energy", "pollution"):
        sc = single[f"modalities={mods}"][1404]
        held.append(f"{INPUTS[mods]} & {sc[0]:.3f} & {sc[1]:.3f} \\\\")
    sc = next(iter(fused.values()))[1404]
    held.append(f"{INPUTS['traffic+energy+pollution']} & {sc[0]:.3f} & {sc[1]:.3f} \\\\")
    return "\n".join(rows), "\n".join(held)


HEAD = r"""%% Online Resource 1 of the Earth Science Informatics version. Written by make_esm.py; do not edit by hand.
\documentclass[pdflatex,sn-basic]{sn-jnl}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{array}
\usepackage{longtable}
\usepackage{pdflscape}
\usepackage{xcolor}
\usepackage{textcomp}
\usepackage{newunicodechar}
\usepackage{manyfoot}
\usepackage{geometry}
\newunicodechar{−}{\ensuremath{-}}
\newunicodechar{×}{\ensuremath{\times}}
\newunicodechar{≈}{\ensuremath{\approx}}
\newunicodechar{₂}{\textsubscript{2}}
\newunicodechar{°}{\textdegree}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\raggedbottom
\newcommand{\tabno}[1]{\renewcommand{\thetable}{#1}}  % tables are numbered by hand (L1, S1a, ...)
\begin{document}
\title[Online Resource 1]{Online Resource 1\\[4pt]\normalsize Reach, not the graph, explains the skill of distance-based graph neural networks in mapping Tehran's NO₂ hotspots from TROPOMI}
\author*[1]{\fnm{Vahid} \sur{Khatibi}}\email{khatibi@iust.ac.ir}
\author[1]{\fnm{Fatemeh} \sur{Vafaeian}}
\affil[1]{\orgdiv{Department of Intelligent Systems Engineering, School of Industrial Engineering}, \orgname{Iran University of Science and Technology}, \orgaddress{\city{Tehran}, \country{Iran}}}
\abstract{Supplementary material to the article in \emph{Earth Science Informatics}. Section S1 records the literature search behind the research gap stated in the Introduction. Table S1 lists every model comparison with its intervals, $p$-values, Holm family and verdict. Table S2 describes the years and processing of every predictor layer, Table S3 the settings of every model and Table S4 the runs of RQ1. References are those of the article.}
\keywords{Literature search, Model comparisons, Predictor layers, Model settings}
\maketitle
"""

S1 = r"""
\section*{S1 Literature search}

\subsection*{S1.1 Structured search in ScienceDirect}

The search was run on 8 November 2026 in the field ``Title, abstract or author-specified keywords'' of the ScienceDirect advanced search, which allows at most eight Boolean connectors per field and adds plurals and spelling variants itself. For each query the first results page, ranked by relevance, was exported with abstracts (25 records for Q1, Q3, Q4 and Q5; 100 for Q2). The 200 exported records hold 191 unique DOIs from 2019 to 2026, 98 of them from 2026 and 49 from 2025.

\tabno{L1}
\begin{table}[h]
\caption{Queries of the structured search}\label{tab:queries}
\footnotesize
\begin{tabular}{@{}lL{10.3cm}r@{}}
\toprule
Query & Search string & Records \\
\midrule
Q1 & (``graph neural network'' OR ``graph convolutional network'' OR GraphSAGE) AND (``spatial lag'' OR ``land use regression'' OR ``spatial regression'') AND (``receptive field'' OR neighborhood) & 25 \\
Q2 & (``graph neural network'' OR ``graph convolutional network'') AND (``nitrogen dioxide'' OR NO2 OR ``air pollution'') AND (TROPOMI OR ``Sentinel-5P'' OR satellite) & 100 \\
Q3 & ``receptive field'' AND (``graph neural network'' OR GNN) AND (spatial OR urban OR geographic) AND baseline & 25 \\
Q4 & (``graph neural network'' OR ``graph convolutional network'') AND (``carbon emission'' OR ``CO2 emission'') AND (urban OR city) AND (``spatial lag'' OR ``random forest'' OR baseline) & 25 \\
Q5 & (``deep learning'' OR ``neural network'') AND (``spatial lag'' OR ``spatially lagged'') AND (``linear regression'' OR ``land use regression'') & 25 \\
\botrule
\end{tabular}
\end{table}

Every record was screened on its title and abstract. No record compares a GNN with a spatial-lag, buffer or land-use-regression baseline at equal reach, and none varies the reach of both models. The closest records are cited in the article: a random forest that matched a spatial GNN for urban NO₂ in Berlin (Schmidt and Ong 2026); GNNs read as generalised spatial regressions (Liang et al. 2023; Zhu et al. 2026; Ma et al. 2025); spatial-lag inputs that recover much of the gain from spatial context (Xiao et al. 2023; Schmidinger et al. 2026; Fleischmann and Arribas-Bel 2024); the receptive field as a design choice in graph learning (Gao et al. 2025; Chen et al. 2026); GNNs that beat spatial regression without matched reach (Mohseni et al. 2026; Zhao et al. 2026); evaluation caveats (Bao et al. 2024; Flores et al. 2026; Zhang et al. 2026); and graph models for satellite-based air-pollution mapping (Bai et al. 2025; Teng et al. 2026; Yuan et al. 2026). The bibliographic fields of all 191 records, with the cited ones marked, are available from the corresponding author on request; the exported abstracts are not shared because they belong to the publishers.

\subsection*{S1.2 Semantic and assisted searches}

A literature review in Consensus asked the same two questions: whether any GNN study uses a reach-matched spatial-lag baseline, and whether skill has been related to the correlation length of the target. It found no matching study; the relevant references it returned were checked against Crossref and are cited in the article (Zhu et al. 2022; Zhan and Datta 2025; Abémgnigni Njifon and Schuhmacher 2024; Liu et al. 2026; Du et al. 2026; Zhao et al. 2026).

On 13 November 2026 ChatGPT with web search was given four questions: (1) a GNN compared with a non-graph baseline given the same reach on purpose; (2) the reach of both varied; (3) the needed reach tied to the correlation length of the target; (4) a non-graph model with neighbourhood-aggregated features matching a GNN. It reported about 100 screened records from a general web and scholarly index, without access to Google Scholar, Scopus, Web of Science or Elsevier full texts. This is not a reproducible database search and is reported only as a supplementary check. Every study it named was checked against Crossref or arXiv, and claims were checked against the full text where it was open. It led to one addition: on synthetic graphs whose community structure is the target, Chen et al. (2021) gave a GNN and a graph-augmented multilayer perceptron the same receptive field, and the GNN performed better. The research gap in the article is therefore stated for spatial and environmental studies only.

\subsection*{S1.3 Reach audit of comparable studies}

For Table 1 of the article, the full texts of fifteen studies that compare a GNN with baselines on spatial data were read: seven from arXiv or open proceedings, and eight from PDFs obtained by the authors because publisher sites block automated reading. For each study we recorded how far the GNN sees and how far each baseline sees. No study makes reach the object of the comparison, varies the reach of both models or sets it by one rule; where neighbourhoods coincide, it is incidental. The three closest cases (Liang et al. 2023; Schmidt and Ong 2026; Bai et al. 2025) are described in the Introduction of the article.

\subsection*{S1.4 Prior use of the dependence range to set a neighbourhood}

Setting the neighbourhood of a model from the range of spatial dependence is established practice (Table~\ref{tab:prior}), but none of these sources uses it to give a GNN and its spatial-lag baselines the same reach. Web searches for ``reach-matched'' and ``range-matched'' together with GNN or spatial baselines returned no results.

\tabno{L2}
\begin{table}[h]
\caption{Earlier uses of the dependence range to set a neighbourhood}\label{tab:prior}
\footnotesize
\begin{tabular}{@{}L{4.6cm}L{7.6cm}@{}}
\toprule
Source & What is set from the dependence range \\
\midrule
Franklin et al. (1996) & Window size for image texture, from semivariograms \\
Behrens et al. (2019) & Range of scales of contextual covariates for machine learning, from the variogram \\
Valavi et al. (2019) & Size of spatial cross-validation blocks, from the autocorrelation range \\
Alon and Yahav (2021) & GNN depth matched to the ``problem radius'' of the task \\
Freitas et al. (2026) & Receptive field of a GNN for turbulent transport, from the integral length scale; no non-graph baseline \\
\botrule
\end{tabular}
\end{table}
"""

S1_TABLE = r"""
\clearpage
\section*{S2 All model comparisons (Table S1)}

Each comparison is candidate minus reference, computed from out-of-fold predictions for the same cells and months, as in Section 3.5 of the article. Neural candidates are five-seed ensembles; SLX references were run once. $\Delta$: difference of the metric averaged over months. CI: delete-one-block jackknife intervals ($K = 12$ blocks, $t$ with 11 degrees of freedom). Verdict: Eq.~(6) of the article with margin $\delta = 0.02$ (descriptive for PR-AUC); superior$^\dagger$, superior with the whole 95\,\% interval beyond the margin. $p$: one-sided block sign-flip randomisation test over all 4,096 assignments, pooled over cells; $p_{\mathrm{blk}}$: the same test on the cell-weighted mean of per-block differences; $p_{\mathrm{Holm}}$: Holm-adjusted $p$ within the family. Boot.: block-bootstrap interval, the inference used in the original study, reported for comparison. Families F1--F3 use the station layer rebuilt in every fold; F4 uses the predictions and pre-processing of the original study.

The verdicts in Tables S1a and S1b are taken from the unadjusted jackknife intervals; Holm's adjustment applies only to the one-sided randomisation tests ($p_{\mathrm{Holm}}$), so equivalence and non-inferiority verdicts are nominal. Tables S1b and S1c are descriptive and carry no tests. The reference of each comparison is the best of its candidate references by the comparison's metric, averaged over the years of the comparison and chosen after the results: by PR-AUC in F1 and F4, by Spearman in F2 and F3. For PR\_c1404\_r30 and PR\_c1404\_r38 this choice is made on the held-out year 1404 itself and favours the SLX reference.

Model labels: GNN $L$L, multi-scale GraphSAGE with $L$ layers; No-graph, the same network fed box means; SLX-OLS, SLX-GBM, SLX-RF, spatial-lag-of-X models with ordinary least squares, gradient boosting or random forests; distances are reach. T, E and S: traffic, energy and station inputs. ``(orig.)'': reference run with the original station pre-processing. F3 models use driver inputs only.

\tabno{S1}
\begin{table}[h]
\caption{The comparisons and their families}\label{tab:key}
\footnotesize
\begin{tabular}{@{}llL{9.0cm}@{}}
\toprule
ID & Family & Comparison \\
\midrule
%KEY%
\botrule
\end{tabular}
\end{table}

\newgeometry{left=1.4cm,right=1.4cm,top=1.8cm,bottom=1.8cm}
\begin{landscape}
\scriptsize
\setlength{\tabcolsep}{3pt}
\tabno{S1a}
\begin{longtable}{@{}lllclrllllllll@{}}
\caption{Primary comparisons}\label{tab:s1a}\\
\toprule
ID & Candidate & Reference & Year & Metric & $\Delta$ & 95\,\% CI & 90\,\% CI & Verdict & $p$ & $p_{\mathrm{blk}}$ & $p_{\mathrm{Holm}}$ & Boot. 95\,\% CI \\
\midrule
\endfirsthead
\toprule
ID & Candidate & Reference & Year & Metric & $\Delta$ & 95\,\% CI & 90\,\% CI & Verdict & $p$ & $p_{\mathrm{blk}}$ & $p_{\mathrm{Holm}}$ & Boot. 95\,\% CI \\
\midrule
\endhead
\bottomrule
\endfoot
%PRIM%
\end{longtable}

\tabno{S1b}
\begin{longtable}{@{}lllcllrlll@{}}
\caption{Other hotspot definitions (top 20\,\% and top 5\,\% of cells) and 5 × 5-cell superpixels}\label{tab:s1b}\\
\toprule
ID & Candidate & Reference & Year & Label & Metric & $\Delta$ & 95\,\% CI & 90\,\% CI & Verdict \\
\midrule
\endfirsthead
\toprule
ID & Candidate & Reference & Year & Label & Metric & $\Delta$ & 95\,\% CI & 90\,\% CI & Verdict \\
\midrule
\endhead
\bottomrule
\endfoot
%SENS%
\end{longtable}

\tabno{S1c}
\begin{longtable}{@{}lllclrrr@{}}
\caption{Single training runs. Mean $\Delta$: mean of the five single-seed scores of the candidate minus the mean single-seed score of the reference (for a neural reference) or its single score (SLX references, run once); Min and Max: the lowest and highest single-seed score of the candidate minus the same reference value. Seeds are not paired. Only comparisons whose runs stored every seed appear}\label{tab:s1c}\\
\toprule
ID & Candidate & Reference & Year & Metric & Mean $\Delta$ & Min $\Delta$ & Max $\Delta$ \\
\midrule
\endfirsthead
\toprule
ID & Candidate & Reference & Year & Metric & Mean $\Delta$ & Min $\Delta$ & Max $\Delta$ \\
\midrule
\endhead
\bottomrule
\endfoot
%SEEDS%
\end{longtable}
\end{landscape}
\restoregeometry
"""

S2_TABLE = r"""
\clearpage
\section*{S3 Predictor layers (Table S2)}

All layers are on the 1/120° grid of the article (72 × 126 cells). The development years are 1402 and 1403. No inventory exists yet for 1404, so the held-out year uses the 1403 inventory and traffic layers, with its own station, weather and night-light data. The traffic and energy layers are estimates, not measurements. Every spatially varying layer also enters the SLX models and the no-graph network as box means (Section 3.1 of the article).

\tabno{S2}
{\footnotesize
\begin{longtable}{@{}L{1.5cm}L{3.1cm}L{1.7cm}L{5.9cm}@{}}
\caption{Years and processing of the predictor layers. $\log(1+x)$ marks layers that are log-transformed before use}\label{tab:s2}\\
\toprule
Group & Layer & Time step & Processing \\
\midrule
\endfirsthead
\toprule
Group & Layer & Time step & Processing \\
\midrule
\endhead
\bottomrule
\endfoot
Traffic & Road-transport CO₂ & Monthly & Bottom-up inventory (IPCC Tier 1 with Monte Carlo uncertainty) of each year, allocated to 1-km cells; monthly values are the annual cell value times the monthly share of road-transport fuel use; $\log(1+x)$ \\
 & Road length by class (motorway and trunk, primary, secondary, tertiary) & Static per year & Sum of link lengths in each cell from the road network of the traffic model (OpenStreetMap road classes) \\
 & Peak-hour vehicle-kilometres & Static per year & Sum over the links in each cell of peak-hour volume times link length, from a traffic assignment model; $\log(1+x)$ \\
 & Volume-to-capacity ratio & Static per year & Length-weighted mean volume-to-capacity ratio of the links in each cell, from the same assignment \\
Energy & Residential and commercial CO₂ & Monthly & Same inventory; monthly share a blend of a flat profile and the heating-degree-day profile of the year; $\log(1+x)$ \\
 & Industrial and power-sector CO₂ & Monthly & Same inventory; flat monthly share (1/12); $\log(1+x)$ \\
 & Heating degree-days (base 18 °C) & Monthly & City-wide monthly value, the same in every cell \\
Pollution & CO from the municipal monitoring network & Monthly & Daily means at 28 stations averaged to months (station-months with fewer than 15 valid days dropped), then interpolated by inverse distance (power 2, minimum distance 0.1 km); rebuilt in every fold without the stations within 2 km of the test block \\
Context & Population, total and non-residential building volume & Static & GHSL 2020 at 100 m, summed to 1-km cells; the same for every year; $\log(1+x)$ \\
 & Night lights & Annual & Mean VIIRS monthly radiance (\texttt{avg\_rad}) from March to February of the Gregorian year that overlaps the Solar Hijri year most; negative values set to 0; $\log(1+x)$ \\
 & Industrial land share & Static & Share of each cell covered by OpenStreetMap industrial and power land use, areas in UTM zone 39N \\
 & Temperature and wind speed & Monthly & ERA5-Land daily means over the city averaged to months, the same in every cell; temperature with a constant bias correction. The 1404 values come from a later ERA5-Land export over a different area, mapped onto the earlier one by a linear fit on the overlapping days (correlation 0.999 for temperature and 0.85 for wind) \\
\end{longtable}}
\noindent{\scriptsize IPCC, Intergovernmental Panel on Climate Change; GHSL, Global Human Settlement Layer; VIIRS, Visible Infrared Imaging Radiometer Suite; ERA5-Land, land component of the fifth-generation reanalysis of the European Centre for Medium-Range Weather Forecasts\par}

\clearpage
\section*{S4 Model settings (Table S3)}

All settings were fixed before the runs of the original study and none was tuned. They are those of the configuration file of the analysis code, which is available from the corresponding author on request.

\tabno{S3}
\begin{table}[h]
\caption{Settings of every model}\label{tab:s3}
\footnotesize
\begin{tabular}{@{}L{3.0cm}L{9.6cm}@{}}
\toprule
Item & Setting \\
\midrule
Predictor scaling & Every model: each feature (including box means) standardised with its mean and standard deviation over all grid cells and months of the training years; predictors are known everywhere, so no target information enters \\
Box means & Uniform square filter of half-width $k$ cells, edge values repeated beyond the grid; city-wide monthly variables (heating degree-days, temperature, wind) get no box means \\
OLS & Ordinary least squares (scikit-learn \texttt{LinearRegression}) \\
GBM & Histogram gradient boosting (scikit-learn \texttt{HistGradientBoostingRegressor}): 300 iterations, learning rate 0.05, at most 31 leaves, random state 0; one run \\
RF & Random forest (scikit-learn \texttt{RandomForestRegressor}): 300 trees, at least 5 samples per leaf, 33\,\% of the features per split, random state 0; one run \\
Encoders (networks) & One per input group (traffic, energy, stations, context): a linear layer with SiLU activation followed by one residual block (linear layer, SiLU, dropout 0.1); hidden size 32 \\
Fusion & Softmax gate over the encoded groups for every cell and month (Eq.~2 of the article) \\
GNN & $L$ residual GraphSAGE layers (Eq.~3 of the article) with mean aggregation over three edge sets (neighbours at 1, 3 and 9 cells), SiLU and dropout 0.1, on the full grid of 72 × 126 cells; linear head \\
Training (networks) & Squared error on the training label cells; AdamW, learning rate $3 \times 10^{-3}$, weight decay $10^{-4}$, 300 epochs without early stopping; one batch per year of 12 months; seeds 0--4, predictions averaged into a five-seed ensemble \\
RQ1 networks & No-graph networks with box means at half-widths of 3 and 9 cells (reach 7.6 km); the context group always included \\
Physical operator & $z = a + \sum_s b_s \log(1 + K_\sigma * E_s)$ over the inventory sectors $s$, with $K_\sigma$ a Gaussian of width $\sigma$ and $b_s \ge 0$ (non-negative least squares); $\sigma$ chosen from 0, 1, 2, 3, 4, 6, 8, 10, 12 and 15 km by the Spearman correlation on the training cells \\
Interpolation baselines & Target of the training cells of the same month and fold only. Inverse-distance weighting with power 2. Gaussian process on the coordinates (scikit-learn), kernel constant $\times$ squared exponential (length scale bounded to 0.5--200 km) plus white noise, fitted once per year on the annual mean map at 600 random cells and then applied with fixed hyper-parameters to every month and fold \\
Hardware & Networks on one NVIDIA RTX 4080 GPU; SLX models, the physical operator and the interpolation baselines on the CPU \\
\botrule
\end{tabular}
\end{table}

\section*{S5 Runs of RQ1 (Table S4)}

The registration of RQ1 fixed the rule for the reference (the single input group with the highest mean PR-AUC over 1402 and 1403), not the group; under this rule the reference was energy. On the held-out year the registered design compares four-layer GNNs, with energy as the pre-specified single source. Every network also receives the context group. The pair energy + stations scored highest in cross-validation but was not part of a registered comparison and was not tested.

\tabno{S4}
\begin{table}[h]
\caption{Five-seed ensembles of the RQ1 networks. Cross-validation (1402 / 1403): gated no-graph networks with box means up to 7.6 km. Held-out year 1404: gated four-layer GNNs (30.3 km). The traffic and energy layers are estimates}\label{tab:s4}
\footnotesize
\begin{tabular}{@{}L{5.2cm}ll@{}}
\toprule
Input groups besides context & PR-AUC & Spearman \\
\midrule
\multicolumn{3}{@{}l}{\textit{Cross-validation, 1402 / 1403}} \\
%S4CV%
\midrule
\multicolumn{3}{@{}l}{\textit{Held-out year 1404}} \\
%S4HELD%
\botrule
\end{tabular}
\end{table}

\end{document}
"""

if __name__ == "__main__":
    key, prim, sens, seeds = table_s1()
    body = (S1_TABLE.replace("%KEY%", key).replace("%PRIM%", prim).replace("%SENS%", sens)
            .replace("%SEEDS%", seeds))
    s4cv, s4held = table_s4()
    tex = HEAD + S1 + body + S2_TABLE.replace("%S4CV%", s4cv).replace("%S4HELD%", s4held)
    (HERE / "online_resource_1.tex").write_text(tex, encoding="utf-8")
    print("wrote", HERE / "online_resource_1.tex", f"({prim.count(chr(92) * 2)} primary rows)")

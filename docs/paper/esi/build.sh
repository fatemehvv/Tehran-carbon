#!/usr/bin/env bash
# Builds the Earth Science Informatics version.
#   paper_esi.pdf       single column, the layout to submit (Springer Nature LaTeX template, sn-basic)
#   paper_esi_2col.pdf  the same text in the class's two-column layout (option iicol), a preview of the journal look
# Needs a TeX Live with latexmk; run from this folder: bash build.sh
set -euo pipefail
cd "$(dirname "$0")"
latexmk -pdf -interaction=nonstopmode -halt-on-error paper_esi.tex
sed 's/\\documentclass\[pdflatex,sn-basic\]{sn-jnl}/\\documentclass[pdflatex,sn-basic,iicol]{sn-jnl}/' paper_esi.tex > paper_esi_2col.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error paper_esi_2col.tex
latexmk -c paper_esi.tex paper_esi_2col.tex
rm -f paper_esi_2col.tex paper_esi.bbl paper_esi_2col.bbl

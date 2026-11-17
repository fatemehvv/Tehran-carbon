#!/usr/bin/env bash
# Builds ESM_1.pdf (Online Resource 1) from the committed result tables. Run from anywhere: bash build.sh
set -euo pipefail
cd "$(dirname "$0")"
python3 make_esm.py
export TEXINPUTS=".:..:" BSTINPUTS=".:..:"
latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=ESM_1 online_resource_1.tex
latexmk -c -jobname=ESM_1 online_resource_1.tex

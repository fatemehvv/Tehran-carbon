#!/usr/bin/env bash
# Review round: runs and analyses asked for by the reviewers (docs/paper/review/review_triage_fa.md).
# Run from the repository root on branch paper_TC, on the server (GPU for the GNN steps).
# Every step is a tracked run (runs/registry.csv); a step that fails can be re-run on its own.
# Rough times from the timing pilot (docs/paper/results_stageA/timing): one GNN cross-validation ~0.8 h on a GPU.
set -euo pipefail

echo "== 0. tests"
python -m pytest tests/test_review_checks.py tests/test_paper_figures.py -q

echo "== 1. package 2: CPU only (SLX runs, ~1-1.5 h in total; RF is the slow one)"
python main.py train --exp paper2_slx_short_v1          # 12 runs: SLX at 0.8 / 2.5 / 7.6 / 15.1 km
python main.py train --exp paper2_slx_buffer_v2         # 9 runs: SLX at 22.7 km, driver inputs, 2 / 10 / 20 km buffers

echo "== 2. package 2: analyses on existing runs (minutes; the interpolation baseline ~10-30 min)"
python main.py paper evaluate --hypothesis PR_c1404_r30 PR_c1404_r38 PR_depth_L3
python main.py paper review

echo "== 3. package 3: GPU (~5-6 h)"
python main.py train --exp paper2_rq2_depth_v1          # 4 runs: SAGE with 1, 2, 4, 5 layers (~3.2 h)
python main.py train --exp paper2_rq2_mlp_lag_v1        # 2 runs: same network, no graph, box means (~1-1.5 h)
python main.py paper evaluate --hypothesis PR_depth_L1 PR_depth_L2 PR_depth_L3 PR_depth_L4 PR_depth_L5 \
    PR_graph_r23 PR_graph_r38 PR_c1404_r30 PR_c1404_r38
python main.py paper reach --exp paper2_base_v1 paper2_slx_short_v1 paper2_slx_reach_v1 \
    paper2_rq2_mlp_v1 paper2_rq2_mlp_lag_v1 paper2_rq2_sage_v1 paper2_rq2_depth_v1

echo "== 4. optional (GPU, ~2.4 h): the GNN with driver inputs at 2 / 10 / 20 km buffers"
if [ "${WITH_BUFFER_GNN:-0}" = "1" ]; then
  python main.py train --exp paper2_rq2_buffer_v2
  python main.py paper evaluate --hypothesis PR_buffer02 PR_buffer10 PR_buffer20
fi

echo "== done. Copy the outputs of the paper_* analysis runs (runs/<id>/*.csv, *.json, *.png, log.txt) to"
echo "   docs/paper/results_review/ (one folder per analysis) and push, as for results_stageA."

echo "== 5. follow-up (GPU, ~40 min): the same learner without a graph at every reach and under larger buffers"
if [ "${WITH_FOLLOWUP:-0}" = "1" ]; then
  python main.py train --exp paper2_rq2_mlp_lag_short_v1    # 6 runs, ~4 min each
  python main.py train --exp paper2_rq2_mlp_lag_buffer_v1   # 3 runs, driver inputs, buffers 2/10/20 km
  python main.py paper evaluate --hypothesis PR_graph_L1 PR_graph_L2 PR_graph_r23 PR_graph_L4 PR_graph_r38 \
      PR_graph_buf02 PR_graph_buf10 PR_graph_buf20
  python main.py paper reach --exp paper2_base_v1 paper2_slx_short_v1 paper2_slx_reach_v1 paper2_rq2_mlp_v1 \
      paper2_rq2_mlp_lag_short_v1 paper2_rq2_mlp_lag_v1 paper2_rq2_sage_v1 paper2_rq2_depth_v1
fi

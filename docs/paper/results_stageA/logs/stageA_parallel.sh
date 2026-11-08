#!/bin/bash
# Stage A, parallel: 2 CPU queues (sklearn baselines) + 2 GPU queues (neural); evaluate after all finish.
source /venv/main/bin/activate
cd ~/Tehran-carbon-paper
run() { name=$1; shift; echo "[$(date +%T)] START $name" >> logs/stageA_status.log
  python main.py "$@" > logs/$name.log 2>&1; echo "[$(date +%T)] END $name exit=$?" >> logs/stageA_status.log; }
queue() { for e in "$@"; do run train_$e train --exp $e; done; }
( while kill -0 2451
4637 2>/dev/null; do sleep 5; done
  echo "[$(date +%T)] END train_paper_slx_reach_v1 (process finished; exit code not captured)" >> logs/stageA_status.log
  queue paper_c1404_slx_reach_v1 paper2_slx_reach_v1 paper2_c1404_slx_reach_v1 ) &
( queue paper2_base_v1 paper2_c1404_base_v1 ) &
( queue paper2_rq2_sage_v1 paper2_c1404_sage4_v1 paper2_rq2_mlp_v1 ) &
( queue paper2_c1404_sage5_v1 paper2_rq1_ablation_v1 paper2_rq1_context_only_v1 paper2_c1404_sage4_single_v1 ) &
wait
run evaluate_paper2 paper evaluate --hypothesis PA_H1 PA_H1_drivers PA_H2 PA_H2_reach PA_H2c PA_H2c_all PA_H1c
run evaluate_thesis paper evaluate --hypothesis H1 H2 H2c H1c P2_reach P2c_reach
echo "[$(date +%T)] ALL DONE" >> logs/stageA_status.log

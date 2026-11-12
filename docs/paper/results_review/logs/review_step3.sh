#!/bin/bash
# Review round, step 3 only (step 1 stopped at paper2_slx_buffer_v1 buffer 20 km; steps 1-2 rest wait for a decision).
# The two GPU trainings run in parallel; evaluate and reach run after both finish.
source /venv/main/bin/activate
cd ~/Tehran-carbon-paper
st=logs/review_step3_status.log
run() { name=$1; shift; echo "[$(date +%T)] START $name" >> $st
  python main.py "$@" > logs/review3_$name.log 2>&1; rc=$?; echo "[$(date +%T)] END $name exit=$rc" >> $st; return $rc; }
run train_depth train --exp paper2_rq2_depth_v1 & a=$!
run train_mlp_lag train --exp paper2_rq2_mlp_lag_v1 & b=$!
wait $a; ra=$?; wait $b; rb=$?
if [ $ra -ne 0 ] || [ $rb -ne 0 ]; then echo "[$(date +%T)] STOPPED: a training failed" >> $st; exit 1; fi
run evaluate paper evaluate --hypothesis PR_depth_L1 PR_depth_L2 PR_depth_L3 PR_depth_L4 PR_depth_L5 PR_graph_r23 PR_graph_r38 PR_c1404_r30 PR_c1404_r38
run reach paper reach --exp paper2_base_v1 paper2_slx_short_v1 paper2_slx_reach_v1 paper2_rq2_mlp_v1 paper2_rq2_mlp_lag_v1 paper2_rq2_sage_v1 paper2_rq2_depth_v1
echo "[$(date +%T)] ALL DONE" >> $st

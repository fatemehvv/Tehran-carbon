#!/bin/bash
# Review round (user order, code 25c6175). GPU trainings paper2_rq2_depth_v1 (pid 7527) and paper2_rq2_mlp_lag_v1
# (pid 7528) were started at 18:07 on code 342d203 and keep running. paper2_slx_buffer_v1 is not used.
source /venv/main/bin/activate
cd ~/Tehran-carbon-paper
st=logs/review_round_status.log
note() { echo "[$(date +%T)] $*" >> $st; }
run() { name=$1; shift; note "START $name"; python main.py "$@" > logs/rr_$name.log 2>&1; rc=$?; note "END $name exit=$rc"; return $rc; }
note "code $(git log --oneline -1)"
# 2. tests
note "START pytest"; python -m pytest tests/test_review_checks.py -q > logs/rr_pytest.log 2>&1; rc=$?; note "END pytest exit=$rc"
if [ $rc -ne 0 ]; then note "STOPPED: tests failed (GPU trainings keep running)"; exit 1; fi
# 4. CPU queue, in parallel with the GPU trainings
( run train_slx_buffer_v2 train --exp paper2_slx_buffer_v2
  run evaluate_step4 paper evaluate --hypothesis PR_c1404_r30 PR_c1404_r38 PR_depth_L3
  run review paper review ) &
cpu=$!
# 3. wait for the GPU trainings
while kill -0 7527 2>/dev/null || kill -0 7528 2>/dev/null; do sleep 30; done
for f in logs/review3_train_depth.log logs/review3_train_mlp_lag.log; do
  if grep -q "FAILED" $f; then note "GPU training FAILED: $f"; gpu_fail=1; else note "GPU training done: $f"; fi
done
wait $cpu
if [ -n "$gpu_fail" ]; then note "STOPPED before step 5: a GPU training failed"; exit 1; fi
# 5. after step 3
run evaluate_step5 paper evaluate --hypothesis PR_depth_L1 PR_depth_L2 PR_depth_L3 PR_depth_L4 PR_depth_L5 PR_graph_r23 PR_graph_r38 PR_c1404_r30 PR_c1404_r38
run reach paper reach --exp paper2_base_v1 paper2_slx_short_v1 paper2_slx_reach_v1 paper2_rq2_mlp_v1 paper2_rq2_mlp_lag_v1 paper2_rq2_sage_v1 paper2_rq2_depth_v1
run figures paper figures
# 6. optional GPU step
run train_buffer_v2 train --exp paper2_rq2_buffer_v2 && \
run evaluate_buffer paper evaluate --hypothesis PR_buffer02 PR_buffer10 PR_buffer20
note "ALL DONE"

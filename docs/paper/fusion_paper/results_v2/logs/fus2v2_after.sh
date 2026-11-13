#!/bin/bash
# Placebo v2: waits for both training queues; if both succeed, runs evaluate --placebo v2 and figures, then saves and pushes.
cd ~/Tehran-carbon-fusion
export PATH=/venv/main/bin:$PATH
S=logs/fus2v2_status.log
st() { echo "$(date '+%F %T') $*" >> "$S"; }
st "watcher started"
until [ "$(cat fus2v2_dev.log fus2v2_1404.log | grep -c '^QUEUE_EXIT=')" -eq 2 ]; do sleep 60; done
st "queues ended: $(grep -h '^QUEUE_EXIT=' fus2v2_dev.log fus2v2_1404.log | tr '\n' ' ')"
if grep -q -E "Traceback|Error" fus2v2_dev.log fus2v2_1404.log || [ "$(grep -h '^QUEUE_EXIT=0' fus2v2_dev.log fus2v2_1404.log | wc -l)" -ne 2 ]; then
  st "STOP: a training queue failed; evaluate and figures not run"; exit 1; fi
python main.py fusion evaluate --placebo v2 > fusion_evaluate_v2.log 2>&1; rc=$?; st "evaluate exit=$rc"
[ $rc -ne 0 ] && { st "STOP: evaluate failed"; exit 1; }
python main.py fusion figures > fusion_figures.log 2>&1; rc=$?; st "figures exit=$rc"
[ $rc -ne 0 ] && { st "STOP: figures failed"; exit 1; }

OUT=docs/paper/fusion_paper/results_v2
mkdir -p "$OUT/runs" "$OUT/logs"
START=$(cat logs/fus2v2_start.txt)
for d in runs/*_fus2_placebo_v2_* runs/*_fus2_placebo_early_v2_* runs/*_fus2_c1404_placebo_v2_* runs/*_fusion_check_* runs/*_fusion_evaluate_*; do
  [ "$(basename "$d" | cut -c1-15)" \> "$START" ] && cp -r "$d" "$OUT/runs/"; done
find "$OUT/runs" -name "*.pt" -delete
cp fusion_check_v2.log fus2v2_dev.log fus2v2_1404.log fusion_evaluate_v2.log fusion_figures.log logs/fus2v2_*.log logs/fus2v2_after.sh "$OUT/logs/" 2>/dev/null
ids=$(ls "$OUT/runs" | tr '\n' '|' | sed 's/|$//')
{ head -1 runs/registry.csv; grep -E "^($ids)," runs/registry.csv; } > "$OUT/registry_v2.csv"
st "saved: $(du -sh "$OUT" | cut -f1), $(ls "$OUT/runs" | wc -l) run folders"
export GIT_SSH_COMMAND="ssh -i ~/.ssh/fusion_deploy -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
git add -f "$OUT" docs/paper/fusion_paper/figures
git commit -q -m "Fusion placebo v2: training, checks, confirmatory tests and figures (pushed from the server)" -m "$(grep '^M4:' fusion_check_v2.log)"
for i in $(seq 1 72); do
  if git pull -q --rebase --autostash origin paper_fusion && git push -q origin HEAD:paper_fusion; then
    st "PUSHED $(git log --oneline -1)"; exit 0; fi
  sleep 300
done
st "PUSH FAILED after 72 tries"

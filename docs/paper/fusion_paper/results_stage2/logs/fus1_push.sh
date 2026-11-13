#!/bin/bash
# Stage 2: waits for the three fus1 queues to end, copies the outputs into the repo and pushes them to paper_fusion.
# Runs on the server on its own, so the push does not depend on the laptop.
cd ~/Tehran-carbon-fusion
S=logs/fus1_status.log
st() { echo "$(date '+%F %T') $*" >> "$S"; }
st "push watcher started"
until [ "$(cat fus1_q1.log fus1_q2.log fus1_q3.log | grep -c '^QUEUE_EXIT=')" -eq 3 ]; do sleep 60; done
st "queues ended: $(grep -h '^QUEUE_EXIT=' fus1_q1.log fus1_q2.log fus1_q3.log | tr '\n' ' ')"

OUT=docs/paper/fusion_paper/results_stage2
mkdir -p "$OUT/runs" "$OUT/logs"
cp -r runs/*_fus1_* runs/*_fusion_support_* "$OUT/runs/" 2>/dev/null
find "$OUT/runs" -name "*.pt" -delete
cp fus1_q*.log logs/fus1_*.log logs/fus1_*.txt logs/fus1_push.sh "$OUT/logs/" 2>/dev/null
{ head -1 runs/registry.csv; grep ",fus1_\|,fusion_support," runs/registry.csv; } > "$OUT/registry_fus1.csv"
st "push: $(du -sh "$OUT" | cut -f1), $(ls "$OUT/runs" | wc -l) run folders (incl. 2 ad hoc smoke runs and fusion_support)"

git remote set-url --push origin git@github.com:fatemehvv/Tehran-carbon.git
export GIT_SSH_COMMAND="ssh -i ~/.ssh/fusion_deploy -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
git add -f "$OUT"
git commit -q -m "Fusion stage 2: outputs of the fus1 runs and the fusion support check (pushed from the server)" \
  -m "Queues: $(grep -h '^QUEUE_EXIT=' fus1_q1.log fus1_q2.log fus1_q3.log | tr '\n' ' ')"
for i in $(seq 1 72); do                      # every 5 min for 6 h
  if git pull -q --rebase --autostash origin paper_fusion && git push -q origin HEAD:paper_fusion; then
    st "PUSHED $(git log --oneline -1)"; exit 0
  fi
  sleep 300
done
st "PUSH FAILED after 72 tries"

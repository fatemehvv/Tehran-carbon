#!/bin/bash
# Waits for fus2_stage1.sh to end, copies the fus2 run folders and logs into the repo and pushes them to paper_fusion.
# Runs on the server on its own, so the push does not depend on the laptop.
cd ~/Tehran-carbon-fusion
S=logs/fus2_status.log
st() { echo "$(date '+%F %T') $*" >> "$S"; }
until grep -q "ALL DONE\|STOP" "$S"; do sleep 60; done

OUT=docs/paper/fusion_paper/results_stage1
mkdir -p "$OUT/runs" "$OUT/logs"
cp -r runs/*_fus2_* "$OUT/runs/" 2>/dev/null
find "$OUT/runs" -name "*.pt" -delete
cp logs/fus2_*.log logs/fus2_*.sh "$OUT/logs/"
{ head -1 runs/registry.csv; grep "fus2_" runs/registry.csv; } > "$OUT/registry_fus2.csv"
st "push: $(du -sh "$OUT" | cut -f1) in $OUT"

git config user.name "Fatemeh Vafaiean"
git config user.email "fa77vafa@gmail.com"
git remote set-url --push origin git@github.com:fatemehvv/Tehran-carbon.git
export GIT_SSH_COMMAND="ssh -i ~/.ssh/fusion_deploy -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
git add -f "$OUT"
git commit -q -m "Fusion stage 1: outputs of the five fus2 runs (pushed from the server)" -m "Status: $(tail -2 "$S" | head -1)"
for i in $(seq 1 72); do                      # every 5 min for 6 h, in case the deploy key is added late
  if git pull -q --rebase --autostash origin paper_fusion && git push -q origin HEAD:paper_fusion; then
    st "PUSHED $(git log --oneline -1)"; exit 0
  fi
  sleep 300
done
st "PUSH FAILED after 72 tries"

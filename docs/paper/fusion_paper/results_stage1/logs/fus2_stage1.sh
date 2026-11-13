#!/bin/bash
# Fusion paper stage 1 (branch paper_fusion): pytest -> serial dataset warm-up -> 3 parallel training queues.
cd ~/Tehran-carbon-fusion
PY=/venv/main/bin/python
S=logs/fus2_status.log
st() { echo "$(date '+%F %T') $*" >> "$S"; }
EXPS="fus2_placebo_v1 fus2_placebo_early_v1 fus2_c1404_real_v1 fus2_c1404_context_v1 fus2_c1404_placebo_v1"

st "START $(git log --oneline -1)"
$PY -m pytest -q > logs/fus2_pytest.log 2>&1; rc=$?; st "pytest exit=$rc"
[ $rc -ne 0 ] && { st "STOP: pytest failed"; exit 1; }

# every dataset (all sweep variants) is built once, serially, so the parallel queues only read the cache
$PY - $EXPS > logs/fus2_prepare.log 2>&1 <<'EOF'
import sys
sys.path.insert(0, "src")
from tehran_carbon.config import Config
from tehran_carbon.data.dataset import CellMonthDataset
done = set()
for exp in sys.argv[1:]:
    for cfg in Config().resolve(exp):
        h = Config.data_hash(cfg)
        if h not in done:
            print(f"== {exp} {cfg['experiment']['variant_label']} -> {h}", flush=True)
            CellMonthDataset.build(cfg)
            done.add(h)
print(f"{len(done)} datasets ready")
EOF
rc=$?; st "prepare exit=$rc ($(tail -1 logs/fus2_prepare.log))"
[ $rc -ne 0 ] && { st "STOP: prepare failed"; exit 1; }

q() { local name=$1; shift
  for e in "$@"; do
    st "$name BEGIN $e"
    $PY main.py train --exp "$e" > "logs/fus2_train_$e.log" 2>&1
    st "$name END $e exit=$?"
  done; }
q Q1 fus2_placebo_v1 &
q Q2 fus2_c1404_real_v1 fus2_c1404_context_v1 fus2_placebo_early_v1 &
q Q3 fus2_c1404_placebo_v1 &
wait
st "ALL DONE"

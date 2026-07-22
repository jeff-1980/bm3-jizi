#!/usr/bin/env bash
# launch_secondary.sh — secondary single-variable ablations
# (bm3_frozen/frozen_noconv/frozen_nogate/frozen_As4d/s4d), out-of-band in tmux.
# clean + awgn@{10,6,0,-2,-6,-10} × {bm3_frozen,frozen_noconv,frozen_nogate,
# frozen_As4d,s4d} × seed 0-4 × 50ep = 175 cells.
#
# WRITE-ONLY DELIVERABLE at authoring time: guardrail 0 (no >2 epoch / >2
# cell training in this session) — this script is authored but NOT executed
# here. Launch it manually:
#   bash launch_secondary.sh
#
# Launch contract inherited BYTE-FOR-BYTE from launch_frozensel.sh (same
# VENV, same cd, same tmux new-session/tee pattern, same --run-dir usage
# against xjtu_noisy_harness.py). Only the harness flag (--secondary instead
# of --frozensel), tmux session name, RUNDIR, and LOG changed. All
# parameters are hardcoded in this script and in xjtu_noisy_harness.py's
# --secondary branch (arms=["bm3_frozen","frozen_noconv","frozen_nogate",
# "frozen_As4d","s4d"], AWGN column, 5 seeds, 50 epochs) — no caller-supplied
# env vars.
#
# REUSE NOTE (bm3_frozen/s4d cells, per task step 5 — "s4d/frozen 可复用已有
#结果免重跑，脚本注明"): results/frozensel_20260702-1827/cells.jsonl already
# contains bm3_frozen and s4d cells for this EXACT config (same AWGN column,
# same FULL_SEEDS=[0..4], same FULL_EPOCHS=50, same TRAIN_COND/TEST_COND). To
# avoid retraining them here, before launching, an operator MAY pre-seed the
# new run's cells.jsonl with those rows so xjtu_noisy_harness.py's built-in
# resume-by-completed-key logic (run_grid(), keyed on (condition, arm, seed))
# skips them automatically — only frozen_noconv/frozen_nogate/frozen_As4d
# actually train. This is a documented MANUAL step (kept out of this script's
# own execution to preserve launch_frozensel.sh's byte-for-byte launch
# mechanics — no extra logic in the tmux command line itself):
#
#   mkdir -p results/secondary_<TS>
#   python3 -c "
#   import json
#   src = 'results/frozensel_20260702-1827/cells.jsonl'
#   dst = 'results/secondary_<TS>/cells.jsonl'
#   reuse_arms = {'bm3_frozen', 's4d'}
#   with open(src) as f, open(dst, 'a') as out:
#       for line in f:
#           cell = json.loads(line)
#           if cell['arm'] in reuse_arms:
#               out.write(line)
#   "
#
# Then run: bash launch_secondary.sh
set -euo pipefail
cd /home/jeffwork/exp/bm3-defense/xjtu_noisy_defense_20260627
VENV=/home/jeffwork/论文8/venv/bin/python
TS=$(date +%Y%m%d-%H%M)
RUNDIR=results/secondary_${TS}
LOG=secondary_${TS}.log
tmux kill-session -t xjtu_sec 2>/dev/null || true
tmux new-session -d -s xjtu_sec "$VENV xjtu_noisy_harness.py --secondary --run-dir $RUNDIR 2>&1 | tee $LOG"
echo "launched session=xjtu_sec run-dir=$RUNDIR log=$LOG"

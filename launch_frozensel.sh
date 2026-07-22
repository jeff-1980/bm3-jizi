#!/usr/bin/env bash
# launch_frozensel.sh — frozen-selectivity arm (bm3_kin/bm3_frozen/s4d), out-of-band in tmux.
# clean + awgn@{10,6,0,-2,-6,-10} × {bm3_kin,bm3_frozen,s4d} × seed 0-4 × 50ep = 105 cells.
#
# WRITE-ONLY DELIVERABLE at authoring time: guardrail 0b (no >2 epoch / >2
# cell training in that session) — this script is authored but NOT executed
# here. Launch it manually:
#   bash launch_frozensel.sh
#
# Launch contract inherited byte-for-byte from launch_ablation.sh (the script
# that actually produced results/ablation_nokin_20260630-0958): same VENV,
# same cd, same tmux new-session/tee pattern, same --run-dir usage against
# xjtu_noisy_harness.py. Only the harness flag (--frozensel instead of
# --ablation), tmux session name, RUNDIR, and LOG changed. All parameters are
# hardcoded in this script and in xjtu_noisy_harness.py's --frozensel branch
# (arms=["bm3_kin","bm3_frozen","s4d"], AWGN column, 5 seeds, 50 epochs) — no
# caller-supplied env vars.
set -euo pipefail
cd /home/jeffwork/exp/bm3-defense/xjtu_noisy_defense_20260627
VENV=/home/jeffwork/论文8/venv/bin/python
TS=$(date +%Y%m%d-%H%M)
RUNDIR=results/frozensel_${TS}
LOG=frozensel_${TS}.log
tmux kill-session -t xjtu_frz 2>/dev/null || true
tmux new-session -d -s xjtu_frz "$VENV xjtu_noisy_harness.py --frozensel --run-dir $RUNDIR 2>&1 | tee $LOG"
echo "launched session=xjtu_frz run-dir=$RUNDIR log=$LOG"

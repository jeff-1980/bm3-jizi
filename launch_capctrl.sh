#!/usr/bin/env bash
# launch_capctrl.sh — capacity control (s4d_wide ~178K), out-of-band in tmux.
# clean + awgn@{10,6,0,-2,-6,-10} × s4d_wide × seed 0-4 × 50ep = 35 cells.
set -euo pipefail
cd /home/jeffwork/exp/bm3-defense/xjtu_noisy_defense_20260627
VENV=/home/jeffwork/论文8/venv/bin/python
TS=$(date +%Y%m%d-%H%M)
RUNDIR=results/capctrl_s4dwide_${TS}
LOG=capctrl_${TS}.log
tmux kill-session -t xjtu_cap 2>/dev/null || true
tmux new-session -d -s xjtu_cap "$VENV xjtu_noisy_harness.py --capctrl --run-dir $RUNDIR 2>&1 | tee $LOG"
echo "launched session=xjtu_cap run-dir=$RUNDIR log=$LOG"

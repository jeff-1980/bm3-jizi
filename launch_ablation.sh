#!/usr/bin/env bash
# launch_ablation.sh — λ_kin=0 ablation arm (bm3_nokin), out-of-band in tmux.
# clean + awgn@{10,6,0,-2,-6,-10} × bm3_nokin × seed 0-4 × 50ep = 35 cells.
set -euo pipefail
cd /home/jeffwork/exp/bm3-defense/xjtu_noisy_defense_20260627
VENV=/home/jeffwork/论文8/venv/bin/python
TS=$(date +%Y%m%d-%H%M)
RUNDIR=results/ablation_nokin_${TS}
LOG=ablation_${TS}.log
tmux kill-session -t xjtu_abl 2>/dev/null || true
tmux new-session -d -s xjtu_abl "$VENV xjtu_noisy_harness.py --ablation --run-dir $RUNDIR 2>&1 | tee $LOG"
echo "launched session=xjtu_abl run-dir=$RUNDIR log=$LOG"

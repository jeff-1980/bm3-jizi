#!/usr/bin/env bash
# launch_fixed.sh — start the bug-fixed full grid in a detached tmux session.
# Bug fixes applied in xjtu_noisy_harness.py:
#   line 128: x = x + apply_noise_at_snr(...)   (was: x = apply_noise_at_snr(...) — replaced signal with pure noise)
#   line 217-219: SNR verification uses additive path + injected-noise power
#   line 72: FULL_SNRS now [10, 6, 0, -2, -6, -10] (added +6/+10 dB)
set -euo pipefail

VENV=/home/jeffwork/论文8/venv/bin/python
WORKDIR=/home/jeffwork/exp/bm3-defense/xjtu_noisy_defense_20260627
cd "$WORKDIR"

TS=$(date +%Y%m%d-%H%M)
RUNDIR=results/fullgrid_fixed_${TS}
LOG=full_run_fixed_${TS}.log

tmux kill-session -t xjtu_fix 2>/dev/null || true
tmux new-session -d -s xjtu_fix \
  "$VENV xjtu_noisy_harness.py --full --run-dir $RUNDIR 2>&1 | tee $LOG"

sleep 4
echo "LAUNCHED rundir=$RUNDIR log=$LOG"
tmux ls

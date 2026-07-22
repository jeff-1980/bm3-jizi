#!/usr/bin/env bash
# launch_graft.sh — constructive graft: s4d_plus_gate only (arch_map.md §6,
# bm3_models.py), out-of-band in tmux.
# clean + awgn@{10,6,0,-2,-6,-10} × {s4d_plus_gate} × seed 0-4 × 50ep = 35 cells.
#
# WRITE-ONLY DELIVERABLE at authoring time: guardrail 0 (no >2 epoch / >2
# cell training in this session) — this script is authored but NOT executed
# here. Launch it manually:
#   bash launch_graft.sh
#
# Launch contract inherited BYTE-FOR-BYTE from launch_secondary.sh (same
# VENV, same cd, same tmux new-session/tee pattern, same --run-dir usage
# against xjtu_noisy_harness.py). Only these differ (per task spec):
#   - harness flag: --graft instead of --secondary
#   - tmux session name: xjtu_graft instead of xjtu_sec
#   - RUNDIR timestamp resolution: seconds (%Y%m%d-%H%M%S), not minutes —
#     task step 5 explicitly specifies results/graft_$(date +%Y%m%d-%H%M%S)
#   - LOG basename: graft_<TS>.log instead of secondary_<TS>.log
# arms=["s4d_plus_gate"] is hardcoded in xjtu_noisy_harness.py's --graft
# branch (mirrors --secondary's own hardcoded arm list) — no caller-supplied
# env vars, no other parameter changed (same AWGN column, same FULL_SEEDS,
# same FULL_EPOCHS).
#
# R-statistic note (graft_prereg_provenance.md, UNCHANGED by this script):
# bm3_frozen/s4d baseline values for R = mean((s4d_plus_gate - s4d) /
# (bm3_frozen - s4d)) are taken from secondary_20260703-1034 (same batch,
# per graft_prereg_provenance.md) — this grid only needs to produce
# s4d_plus_gate cells, not retrain bm3_frozen/s4d.
set -euo pipefail
cd /home/jeffwork/exp/bm3-defense/xjtu_noisy_defense_20260627
VENV=/home/jeffwork/论文8/venv/bin/python
TS=$(date +%Y%m%d-%H%M%S)
RUNDIR=results/graft_${TS}
LOG=graft_${TS}.log
tmux kill-session -t xjtu_graft 2>/dev/null || true
tmux new-session -d -s xjtu_graft "$VENV xjtu_noisy_harness.py --graft --run-dir $RUNDIR 2>&1 | tee $LOG"
echo "launched session=xjtu_graft run-dir=$RUNDIR log=$LOG"

#!/usr/bin/env bash
# launch_perclass.sh — per-class confusion/precision-recall prep grid
# (bm3_kin/bm3_frozen/s4d/frozen_nogate/s4d_plus_gate), out-of-band in tmux.
# {clean, awgn@+0dB, awgn@-6dB} × 5 arms × seed 0-4 × 20ep = 75 cells, with
# per-cell predictions.jsonl dumped via --dump_predictions
# (xjtu_noisy_harness.py's --perclass branch; see also preddump_unitcheck.py
# for the single-cell smoke that validated --dump_predictions's contract).
#
# WRITE-ONLY DELIVERABLE at authoring time: guardrail 0 (no >2 epoch / >2
# cell training in this session) — this script is authored but NOT executed
# here. Launch it manually:
#   bash launch_perclass.sh
#
# Launch contract inherited BYTE-FOR-BYTE from launch_secondary.sh (same
# VENV, same cd, same TS precision (%Y%m%d-%H%M), same tmux new-session/tee
# pattern, same --run-dir usage against xjtu_noisy_harness.py). Only these
# differ:
#   - harness flags: --perclass --dump_predictions instead of --secondary
#   - tmux session name: xjtu_perclass instead of xjtu_sec
#   - RUNDIR: results/perclass_${TS} instead of results/secondary_${TS}
#   - LOG basename: perclass_${TS}.log instead of secondary_${TS}.log
# arms/conditions/seeds/epochs (bm3_kin/bm3_frozen/s4d/frozen_nogate/
# s4d_plus_gate × clean+awgn@{0,-6}dB × 5 seeds × 20 epochs) are hardcoded in
# xjtu_noisy_harness.py's --perclass branch — no caller-supplied env vars,
# matching --secondary/--graft's own hardcoded-arm-list convention.
set -euo pipefail
cd /home/jeffwork/exp/bm3-defense/xjtu_noisy_defense_20260627
VENV=/home/jeffwork/论文8/venv/bin/python
TS=$(date +%Y%m%d-%H%M)
RUNDIR=results/perclass_${TS}
LOG=perclass_${TS}.log
tmux kill-session -t xjtu_perclass 2>/dev/null || true
tmux new-session -d -s xjtu_perclass "$VENV xjtu_noisy_harness.py --perclass --dump_predictions --run-dir $RUNDIR 2>&1 | tee $LOG"
echo "launched session=xjtu_perclass run-dir=$RUNDIR log=$LOG"

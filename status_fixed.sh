#!/usr/bin/env bash
# status_fixed.sh — one-shot status of the bug-fixed full grid run.
WORKDIR=/home/jeffwork/exp/bm3-defense/xjtu_noisy_defense_20260627
RUNDIR=$WORKDIR/results/fullgrid_fixed_20260628-2248
LOG=$WORKDIR/full_run_fixed_20260628-2248.log
cd "$WORKDIR"

echo "===== PYTHON PROC ====="
pgrep -af "python xjtu_noisy_harness.py" | grep -v "bash -c" || echo "NO PYTHON PROC (run ended or died)"

echo "===== CELLS DONE ====="
if [ -f "$RUNDIR/cells.jsonl" ]; then
  wc -l < "$RUNDIR/cells.jsonl"
else
  echo "0 (cells.jsonl not yet created)"
fi

echo "===== NOISE VERIFICATION (Bug-2 fix proof) ====="
if [ -f "$RUNDIR/noise_verification.json" ]; then
  grep -E "actual_snr_db|target_snr_db|all_pass|snr_check" "$RUNDIR/noise_verification.json"
else
  echo "not written yet"
fi

echo "===== TMUX PANE (last 25 lines) ====="
tmux capture-pane -t xjtu_fix -p 2>/dev/null | tail -25 || echo "no tmux session"

echo "===== LOG TAIL ====="
tail -8 "$LOG" 2>/dev/null || echo "log empty"

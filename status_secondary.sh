#!/usr/bin/env bash
# status_secondary.sh — one-shot status of the launch_secondary.sh grid.
#
# Fixed contract, inherited byte-for-byte in shape from status_frozensel.sh
# (WORKDIR / RUNDIR / LOG hardcoded, cd, PYTHON PROC / CELLS DONE / NOISE
# VERIFICATION / TMUX PANE / LOG TAIL sections, tail -8 log line count).
# No positional args, no ${1:-}, no `ls -dt` auto-discovery, no
# basename/sed-derived TS — this is a template with the run's literal
# timestamp filled in, exactly as status_frozensel.sh hardcodes its own
# PENDING_TS placeholder rather than discovering it.
#
# launch_secondary.sh has not been executed as of authoring this file (see
# its header comment). RUNDIR/LOG below use PENDING_TS as a placeholder.
# Once `bash launch_secondary.sh` is actually run, it will print the real
# timestamp (`launched session=xjtu_sec run-dir=... log=...`); replace
# PENDING_TS below with that literal value (two occurrences) before using
# this script — do not add a variable or argument to do this automatically.
WORKDIR=/home/jeffwork/exp/bm3-defense/xjtu_noisy_defense_20260627
RUNDIR=$WORKDIR/results/secondary_PENDING_TS
LOG=$WORKDIR/secondary_PENDING_TS.log
cd "$WORKDIR"

echo "===== PYTHON PROC ====="
pgrep -af "python xjtu_noisy_harness.py" | grep -v "bash -c" || echo "NO PYTHON PROC (run ended or died)"

echo "===== CELLS DONE ====="
if [ -f "$RUNDIR/cells.jsonl" ]; then
  wc -l < "$RUNDIR/cells.jsonl"
else
  echo "0 (cells.jsonl not yet created)"
fi

echo "===== NOISE VERIFICATION ====="
if [ -f "$RUNDIR/noise_verification.json" ]; then
  grep -E "actual_snr_db|target_snr_db|all_pass|snr_check" "$RUNDIR/noise_verification.json"
else
  echo "not written yet"
fi

echo "===== TMUX PANE (last 25 lines) ====="
tmux capture-pane -t xjtu_sec -p 2>/dev/null | tail -25 || echo "no tmux session"

echo "===== LOG TAIL ====="
tail -8 "$LOG" 2>/dev/null || echo "log empty"

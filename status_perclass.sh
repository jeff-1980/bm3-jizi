#!/usr/bin/env bash
# status_perclass.sh — one-shot status of the launch_perclass.sh grid.
#
# Structure inherited from status_graft.sh (WORKDIR, cd, RUN DIR / PYTHON
# PROC / CELLS DONE / NOISE VERIFICATION / TMUX PANE / LOG TAIL sections,
# tail -8 log line count) — RUNDIR is auto-discovered via a glob restricted
# to results/perclass_2* (a literal "2" right after the "perclass_" prefix,
# i.e. only directories whose suffix is a numeric timestamp starting with a
# 2xxx year). This is deliberate: results/ already holds non-run directories
# that share the "perclass_" prefix with suffixes that are NOT run
# timestamps (e.g. results/perclass_needs_human_20260707-1150/) — restricting
# the glob to perclass_2* means this script can never accidentally pick up
# that (or any future perclass_<word>_*-style) directory as if it were a
# training run.
WORKDIR=/home/jeffwork/exp/bm3-defense/xjtu_noisy_defense_20260627
cd "$WORKDIR"

RUNDIR=$(ls -dt "$WORKDIR"/results/perclass_2*/ 2>/dev/null | head -1)
RUNDIR=${RUNDIR%/}
if [ -n "$RUNDIR" ]; then
  TS=$(basename "$RUNDIR" | sed 's/^perclass_//')
else
  TS="UNKNOWN"
fi
LOG=$WORKDIR/perclass_${TS}.log

echo "===== RUN DIR ====="
echo "${RUNDIR:-none found (no results/perclass_2* directory yet)}"

echo "===== PYTHON PROC ====="
pgrep -af "python xjtu_noisy_harness.py" | grep -v "bash -c" || echo "NO PYTHON PROC (run ended or died)"

echo "===== CELLS DONE ====="
if [ -n "$RUNDIR" ] && [ -f "$RUNDIR/cells.jsonl" ]; then
  wc -l < "$RUNDIR/cells.jsonl"
else
  echo "0 (cells.jsonl not yet created, or no results/perclass_2* run dir found)"
fi

echo "===== PREDICTIONS.JSONL DUMPED ====="
if [ -n "$RUNDIR" ]; then
  find "$RUNDIR" -mindepth 2 -name predictions.jsonl 2>/dev/null | wc -l
else
  echo "0 (no run dir found)"
fi

echo "===== NOISE VERIFICATION ====="
if [ -n "$RUNDIR" ] && [ -f "$RUNDIR/noise_verification.json" ]; then
  grep -E "actual_snr_db|target_snr_db|all_pass|snr_check" "$RUNDIR/noise_verification.json"
else
  echo "not written yet"
fi

echo "===== TMUX PANE (last 25 lines) ====="
tmux capture-pane -t xjtu_perclass -p 2>/dev/null | tail -25 || echo "no tmux session"

echo "===== LOG TAIL ====="
tail -8 "$LOG" 2>/dev/null || echo "log empty"

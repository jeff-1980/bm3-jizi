#!/usr/bin/env bash
# status_graft.sh — one-shot status of the launch_graft.sh grid.
#
# Structure inherited from status_secondary.sh (WORKDIR, cd, PYTHON PROC /
# CELLS DONE / NOISE VERIFICATION / TMUX PANE / LOG TAIL sections, tail -8
# log line count) — but unlike status_secondary.sh's hardcoded PENDING_TS
# placeholder, RUNDIR here is auto-discovered via a glob restricted to
# results/graft_2* (a literal "2" right after the "graft_" prefix, i.e. only
# directories whose suffix is a numeric timestamp starting with a 2xxx
# year). This is deliberate per task step 5: results/ already holds
# non-run directories that share arm-name prefixes with suffixes that are
# NOT run timestamps (e.g. this dir's own *_prereg_*/*_needs_human_*
# directories for other arms) — restricting the glob to graft_2* means this
# script can never accidentally pick up a future results/graft_prereg_*-style
# directory as if it were a training run.
WORKDIR=/home/jeffwork/exp/bm3-defense/xjtu_noisy_defense_20260627
cd "$WORKDIR"

RUNDIR=$(ls -dt "$WORKDIR"/results/graft_2*/ 2>/dev/null | head -1)
RUNDIR=${RUNDIR%/}
if [ -n "$RUNDIR" ]; then
  TS=$(basename "$RUNDIR" | sed 's/^graft_//')
else
  TS="UNKNOWN"
fi
LOG=$WORKDIR/graft_${TS}.log

echo "===== RUN DIR ====="
echo "${RUNDIR:-none found (no results/graft_2* directory yet)}"

echo "===== PYTHON PROC ====="
pgrep -af "python xjtu_noisy_harness.py" | grep -v "bash -c" || echo "NO PYTHON PROC (run ended or died)"

echo "===== CELLS DONE ====="
if [ -n "$RUNDIR" ] && [ -f "$RUNDIR/cells.jsonl" ]; then
  wc -l < "$RUNDIR/cells.jsonl"
else
  echo "0 (cells.jsonl not yet created, or no results/graft_2* run dir found)"
fi

echo "===== NOISE VERIFICATION ====="
if [ -n "$RUNDIR" ] && [ -f "$RUNDIR/noise_verification.json" ]; then
  grep -E "actual_snr_db|target_snr_db|all_pass|snr_check" "$RUNDIR/noise_verification.json"
else
  echo "not written yet"
fi

echo "===== TMUX PANE (last 25 lines) ====="
tmux capture-pane -t xjtu_graft -p 2>/dev/null | tail -25 || echo "no tmux session"

echo "===== LOG TAIL ====="
tail -8 "$LOG" 2>/dev/null || echo "log empty"

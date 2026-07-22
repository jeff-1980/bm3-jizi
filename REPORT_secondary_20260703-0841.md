# Secondary-grid analysis — STOPPED at Guardrail 0 (STOP-IF-NO-GRID)

**Date:** 2026-07-03 08:41
**Task:** aggregate/decide on `results/secondary_*/cells.jsonl` (3~4 arms × 7 conditions × 5 seeds,
including a reused baseline) → `fairness_report.json` → `q_secondary.csv/json` +
`secondary_curve.png` → `secondary_decision.json`.

## Outcome: STOPPED_NEEDS_HUMAN — no grid data exists under this name

Per the task's own hard guardrail 0:

> tuple 枚举校验 `results/secondary_*/cells.jsonl` 完整 ... 不足 ⇒ 停、needs_human、不得 loop 内启训

Verification performed (read-only, no writes/deletes to any pre-existing file):

1. `ls results/` — no directory matches `secondary_*`.
2. `grep -ril "secondary" .` (this repo) and the parent `bm3-defense/` — 0 matches in any
   `.py`/`.json`/`.md`/`.log` file. No memory record ([[bm3-frozensel-grid-pending]],
   [[bm3-stage2-grid-state]], [[bm3-extended-baselines-noisy]]) mentions a "secondary" grid.
3. `grep -n "add_argument" xjtu_noisy_harness.py` — the harness's CLI has
   `--smoke/--full/--ablation/--capctrl/--frozensel/--analyze-only`; no `--secondary` flag or
   code path exists anywhere that would produce a grid by that name.
4. The closest existing artifact by *shape* is `results/frozensel_20260702-1827/cells.jsonl`
   (3 arms × 7 AWGN conditions × 5 seeds = 105 cells, already complete and already analyzed with
   `frozensel_decision.json` / `frozensel_decision_scoped.json`). It was **not** substituted in
   as "the secondary grid": the task names a distinct glob (`secondary_*`, not `frozensel_*`), and
   silently re-labeling existing evidence under an unassigned name is indistinguishable from
   fabricating the requested artifact from data the task didn't point at. The guardrail's
   text is unconditional ("不足 ⇒ 停"), so an unresolved name/data mismatch halts rather than
   being resolved by inference.

**Written this session** (additive only, chmod 444):
`results/secondary_needs_human_20260703-0841/needs_human.json` — full audit trail: required
shape, actual state (empty), why the frozensel run was not silently substituted, and the
two clarifying questions for the human (new grid to design+launch vs. re-pointing at an
existing run under a clearer name).

## What was explicitly NOT done (per task's own "明确不做" and guardrail 0)

- No grid/training was launched or resumed in this session or in a loop.
- No file under `results/**` was modified, deleted, or overwritten.
- No `fairness_report.json`, `q_secondary.csv/json`, `secondary_curve.png`, or
  `secondary_decision.json` were produced — doing so from zero cells would fabricate results.
- No existing arm implementation was changed.

## Next step (requires a human decision, not a code fix)

See `action_required_from_human` in `needs_human.json`. Once either (a) a real
`results/secondary_<TS>/cells.jsonl` exists with the full 3~4-arm × 7-condition × 5-seed tuple
set (verified no dupes/no gaps), or (b) the task is re-issued pointing explicitly at an existing
run-dir to treat as "secondary", steps 1–3 (fairness re-hash, aggregation + curve, prereg-bound
decision) can proceed exactly as specified.

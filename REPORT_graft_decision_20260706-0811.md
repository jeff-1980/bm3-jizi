# Constructive graft — R-statistic adjudication (post-grid)

**Date:** 2026-07-06 08:11
**Task:** analyze the already-completed `s4d_plus_gate` 35-cell grid
(`results/graft_20260705-224137/`) against `graft_prereg_provenance.md`'s
predeclared R statistic. No training was launched in this session (the grid
already existed on disk, produced by a prior session — see §0). No existing
`results/**` file was touched; no existing implementation code was modified.

## Outcome: ALL GATES PASS — mechanical verdict = **GRAFT_INSUFFICIENT** (mean R = 0.1742)

All new artifacts live in `results/graft_analysis_20260706-0811/` (a new
timestamped directory), each `chmod 444` immediately after writing. The
producing script, `graft_aggregate.py`, is a brand-new file — it does not
modify `xjtu_noisy_harness.py`, `bm3_models.py`, `models_extended.py`,
`graft_prereg_provenance.md`, or any other pre-existing file (verified: their
mtimes are unchanged from before this session, all dated 2026-07-05 or
earlier). **No existing implementation code was changed in this session** —
there is nothing to declare per the "若必须修改既有实现代码" guardrail clause.

### 0. Hard guardrail — STOP-IF-NO-GRID

`graft_aggregate.py`'s `stop_if_no_grid()` globs `results/graft_*/cells.jsonl`
(matches exactly one file: `results/graft_20260705-224137/cells.jsonl`),
filters to `arm == "s4d_plus_gate"`, and requires the row set to equal the
cross product `{clean, awgn@+10dB, awgn@+6dB, awgn@+0dB, awgn@-2dB,
awgn@-6dB, awgn@-10dB} x {seed 0..4}` exactly (no missing/duplicate/extra).
Result: **35/35, 0 duplicates, 0 missing, 0 extra** — guardrail passes, so the
script proceeds. Had it failed, the script writes
`results/graft_needs_human_<TS>/needs_human.json` and exits (status 4)
**without ever launching training** — this path was exercised only as dead
code (not triggered), since the grid was already complete; see
`results/graft_analysis_20260706-0811/tuple_enum_report.json` for the passing
report.

Per the task's explicit "不得 loop 内启训": this session never ran
`launch_graft.sh`, never called `xjtu_noisy_harness.py --graft`, and never
invoked any training. It only read the two already-complete `cells.jsonl`
files (`results/graft_20260705-224137/` for `s4d_plus_gate`,
`results/secondary_20260703-1034/` for the `bm3_frozen`/`s4d`/`frozen_nogate`
baselines, per `graft_prereg_provenance.md`'s own baseline-sourcing rule).

### 1. `fairness_report.json` — byte-for-byte eval_sha256

`independent_eval_sha()` recomputes `eval_sha256` from scratch via
`xjtu_noisy_harness.H.make_cross_condition_split()` /
`H.XJTUDataset()`/`H.XJTUDatasetNoisy()`/`H.eval_fingerprint()` — the exact
same construction `run_grid()` used, but never reading a stored hash value.
Result: `6c20b367522ce5db59720120f8a7cd12792c7e070c6ae84bd1985511998727de`,
matching the task's literal prefix (`6c20b367522c`) and every one of the 35
graft-grid rows' `fairness.eval_sha256`, byte for byte
(`n_mismatches: 0`, `all_pass: true`). A non-gating supplementary check
extends the same recomputed value to the 105 baseline cells reused from
`results/secondary_20260703-1034/` (also 0 mismatches) — confirming the R
statistic's numerator and denominator arms share the same fairness
invariant. See `results/graft_analysis_20260706-0811/fairness_report.json`.

### 2. Aggregation, curve, per-level R, paired significance (report-only)

- `q_graft.json`/`q_graft.csv`: mean±std `macro_F1` per (condition, arm) for
  all 4 arms — `s4d_plus_gate` (from the graft grid) and
  `bm3_frozen`/`s4d`/`frozen_nogate` (reused from
  `results/secondary_20260703-1034/`, same batch as
  `graft_prereg_provenance.md` mandates).
- `graft_curve_table.csv` / `graft_curve.png`: macro-F1-vs-SNR curve,
  `s4d_plus_gate` overlaid on the three baselines, mean±std error bars
  (clean plotted at 15dB, matching `frozensel_aggregate.py`'s convention).
  Visually, `s4d_plus_gate` tracks `s4d` closely at clean/high-SNR and only
  partially closes the gap to `bm3_frozen` in the deep-noise region.
- **Per-level R** (all 7 conditions, `R = (s4d_plus_gate − s4d) / (bm3_frozen
  − s4d)`, evidence only — the prereg only defines the decision-relevant mean
  over the 3 deep-noise bins):

  | condition | Δ(gate−s4d) pp | Δ(frozen−s4d) pp | R |
  |---|---|---|---|
  | clean | −8.68 | −3.37 | 2.572 (frozen also below s4d here; not decision-relevant) |
  | awgn@+10dB | −4.40 | −1.51 | 2.920 (same caveat) |
  | awgn@+6dB | −1.84 | +1.90 | −0.968 (sign-flipped; not decision-relevant) |
  | **awgn@+0dB** | **+3.58** | **+14.40** | **0.2487** |
  | **awgn@-2dB** | **+4.22** | **+17.64** | **0.2392** |
  | **awgn@-6dB** | **+0.61** | **+17.43** | **0.0348** |
  | awgn@-10dB | +6.81 | +12.30 | 0.5537 (outside the prereg's 3-bin window) |

  (bolded rows = the 3 deep-noise bins the prereg's mean-R rule is defined
  over.)
- **Deep-noise 3-bin mean R = (0.2487 + 0.2392 + 0.0348) / 3 = 0.1742.**
- `graft_paired_significance.json`: paired (same-seed) t-test + Wilcoxon
  signed-rank for `Δ(gate−s4d)` and `Δ(frozen−s4d)` per condition —
  significant (p<0.05) for `Δ(gate−s4d)` only at `awgn@-2dB` and
  `awgn@-10dB`; not significant at either `awgn@+0dB` or `awgn@-6dB`. Per
  `graft_prereg_provenance.md`'s explicit "禁未预注册判据", **this
  significance analysis is reported for transparency only and does not gate
  the verdict below** — the R-magnitude rule is the sole decisive criterion.

### 3. `graft_decision.json` — mechanical verdict

Applies `graft_prereg_provenance.md`'s verbatim rule:

```
R = mean over {0,-2,-6dB} of (s4d_plus_gate - s4d) / (bm3_frozen - s4d)
R >= 0.7   => CONSTRUCTIVE_CONFIRMED
0.3<=R<0.7 => PARTIAL_RECOVERY
R < 0.3    => GRAFT_INSUFFICIENT
```

- `preregistered: true`
- `prereg_source: "graft_prereg_provenance.md (materialized pre-grid)"` —
  the file is 397 bytes, mtime 2026-07-05 21:20, predating the grid's start
  (2026-07-05 22:41:37 per `graft_20260705-224137.log`'s `[START]` line) and
  this analysis run.
- `reviewer_verdict_file: ".orchestrate/0705-213350/r1_review.json"`,
  `reviewer_verdict: "PASS"` — resolved dynamically (same closed-round-only
  rule as `frozensel_aggregate.py`'s `_resolve_latest_reviewer_verdict()`, so
  an in-progress review round can never cite itself). This is the latest
  **closed** review round in the repo that concerns the graft work: it
  scoped-reviewed the harness dispatch (`--graft`), `graft_unitcheck.json`,
  the smoke test, and `launch_graft.sh`/`status_graft.sh` — all PASS — before
  the full 35-cell grid was launched using that exact reviewed code path.
  (`.orchestrate/0706-080556/` exists but contains only `review_schema.json`
  with no `result.json` — an open round, correctly excluded.)
- **Verdict: `GRAFT_INSUFFICIENT`** (mean R = 0.1742 < 0.3): grafting the
  Mamba-3-style output gate onto the never-selective `BearS4D` baseline
  recovers only ~17% of the gap between `bm3_frozen` (selectivity removed,
  gate present) and `s4d` (neither) in the three deep-noise bins — well
  short of the 0.3 "partial recovery" floor. The gate branch alone is not a
  sufficient constructive substitute for the rest of BM3's non-selective
  architecture.

## Files (all new, all `chmod 444`, all in `results/graft_analysis_20260706-0811/`)

- `tuple_enum_report.json` — §0 guardrail evidence (35/35, 0 dup/missing/extra)
- `fairness_report.json` — §1 evidence
- `q_graft.json` / `q_graft.csv` — §2 aggregation
- `graft_curve_table.csv` / `graft_curve.png` — §2 curve
- `graft_paired_significance.json` — §2 report-only significance
- `graft_decision.json` — §3 mechanical verdict

## Guardrail self-certification

- **Additive-only**: every product above is new, in a new timestamped
  directory; no pre-existing `results/**` file's mtime changed (verified via
  `ls -ldt results/*/` and explicit `stat` on the graft/secondary run dirs
  used as read-only inputs).
- **chmod 444**: verified via `find results/graft_analysis_20260706-0811/
  -type f ! -perm 444` → empty.
- **No implementation-code edits**: `graft_aggregate.py` is a new file only;
  `xjtu_noisy_harness.py`/`bm3_models.py`/`models_extended.py`/
  `graft_prereg_provenance.md` mtimes unchanged from before this session.
- **No training launched**: confirmed by inspection of this session's own
  actions — only `graft_aggregate.py` (read-only over two existing
  `cells.jsonl` files) was executed.

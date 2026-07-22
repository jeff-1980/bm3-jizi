# Frozen-selectivity ablation: fairness → aggregation → decision (2026-07-03)

## 0. Hard guardrail (STOP-IF-NO-GRID) — PASSED

Tuple-enumeration check of `results/frozensel_20260702-1827/cells.jsonl` before doing
anything else:

- 105/105 lines present, 0 duplicate `(condition, arm, seed)` keys, 0 missing keys
  against the full cross product of 3 arms × {clean, awgn@+10dB, awgn@+6dB,
  awgn@+0dB, awgn@-2dB, awgn@-6dB, awgn@-10dB} × 5 seeds (0-4).
- Grid was already complete (launched 2026-07-02 18:27, per prior-session memory);
  did not launch or resume any training in this session.

Since the grid was already ≥ min (63) and in fact full (105), proceeded to steps 1-3.

## 1. Fairness report — PASSED, checked before aggregation

Ran `frozensel_aggregate.py` (pre-existing script from the prior session, per
`frozensel_prereg_20260702-1826/frozensel_prereg.json`), which:
- independently recomputes `eval_sha256` (full 64-hex SHA256) from the harness's
  own `make_cross_condition_split`/`XJTUDataset`/`XJTUDatasetNoisy("clean",...)`
  construction, and
- compares it byte-for-byte against every one of the 105 cells' stored
  `fairness.eval_sha256`.

Result written to `results/frozensel_20260702-1827/fairness_report.json`:
`all_pass=true`, `n_mismatches=0`, `hash_lengths_all_64=true`, expected hash
`6c20b367...998727de` (64 chars) matches all 105 cells exactly. Aggregation
proceeded only because this gate passed (the script itself refuses to write any
downstream product otherwise).

## 2. Aggregation — per-condition/per-arm mean±std, curves, deltas

Outputs (all in `results/frozensel_20260702-1827/`, chmod 444):
- `q_frozensel.json` / `.csv` — mean_macro_f1, std_macro_f1, n_seeds_used (=5 for
  every cell), per (condition, arm).
- `frozensel_curve_table.csv`, `frozensel_curve.png` — macro-F1 vs SNR, 3 arms,
  mean±std error bars, chance-floor line at 0.55.
- `frozensel_decision.json` — per-condition `delta_kin_minus_frozen_pp` and
  `delta_kin_minus_s4d_pp` (context only, per prereg).

**Bug found and fixed** in the pre-existing `frozensel_aggregate.py`
(`snr_for_plot_map()`, line ~160-163): `build_conditions(["awgn"], H.FULL_SNRS)`
always prepends its own `{"noise_type": "clean", "snr_db": None, "label": "clean"}`
entry. The original loop iterated over all of `build_conditions()`'s output and
overwrote the hardcoded `m["clean"] = 15.0` with `None`, silently dropping the
`clean` condition's point from `frozensel_curve_table.csv` and
`frozensel_curve.png` (verified: the first-run curve.png's x-axis only spanned
-10..10 dB, missing the clean point entirely).

**Fix declared** (`frozensel_aggregate.py`, in `snr_for_plot_map()`): added
`if c["label"] == "clean": continue` before the assignment, so the loop no
longer overwrites the hardcoded clean→15.0 mapping. This is the only line
changed in any pre-existing implementation file this session.

Because `results/frozensel_20260702-1827/frozensel_curve_table.csv` and
`frozensel_curve.png` were already written and chmod 444 in this same session
(before the bug was noticed), they were **not** overwritten (additive-only
guardrail + the files are literally read-only). Instead, corrected versions were
regenerated via a new one-off script, `frozensel_regen_curve_fixed.py` (imports
the now-fixed `frozensel_aggregate.aggregate/write_curve_table/write_curve_figure`
and re-runs only those two artifacts), written into a new directory:
`results/frozensel_curvefix_20260703-0748/frozensel_curve_table.csv` and
`frozensel_curve.png` (chmod 444). These are the authoritative curve
table/plot; the originals in `frozensel_20260702-1827/` are left in place
unmodified as the (buggy-on-the-clean-point-only) first-pass artifact, per
"only additive, never overwrite."

`q_frozensel.*` and `frozensel_decision.json` are unaffected by this bug (they
don't use `snr_for_plot_map()`).

### Δ(frozen − s4d) and paired significance (this session's addition)

The prereg's decision rule only reports `delta_kin_minus_frozen_pp` and
`delta_kin_minus_s4d_pp` (context only), and its significance test is a
non-overlapping mean±std band heuristic scoped to kin-vs-frozen. Task step 2
asked for Δ(frozen−s4d) per point plus paired significance for both deltas, so
a new script `frozensel_paired_significance.py` (read-only over `cells.jsonl`,
no existing file touched) computes, per condition, matched-by-seed (same seed
= same split/init/noise-instance across arms):
- `delta_kin_minus_frozen` and `delta_frozen_minus_s4d`, each seed's paired
  difference, plus a paired t-test (`scipy.stats.ttest_rel`) and Wilcoxon
  signed-rank test (`scipy.stats.wilcoxon`) as a non-parametric cross-check.

Written to `results/frozensel_pairedsig_20260703-0746/frozensel_paired_significance.json`
and `.csv` (chmod 444).

**Result:** kin-vs-frozen: 0/7 conditions significant at p<0.05 (paired
t-test) — consistent with the prereg's mean±std-band verdict of 7/7 ties.
frozen-vs-s4d: 4/7 conditions significant (awgn@+6dB, awgn@-2dB, awgn@-6dB,
awgn@-10dB), all in the direction of bm3_frozen > s4d.

## 3. Decision — `frozensel_decision.json`

**Verdict: `SELECTIVITY_CLAIM_FALSIFIED`**

Mechanical application of the prereg's rule (`frozensel_prereg_20260702-1826/frozensel_prereg.json`,
`decision_rule`), citing this run's fairness pass and the reviewer verdict in
`.orchestrate/0702-181858/codex_last_message.txt`:

- All 7 conditions are "meaningful" (both bm3_kin and bm3_frozen above the
  0.55 chance floor).
- `kin_sig_wins = 0`, `frozen_sig_wins = 0`, `ties = 7` — every condition's
  mean±std bands overlap between bm3_kin and bm3_frozen. bm3_kin does not
  significantly beat bm3_frozen in a strict majority of meaningful conditions
  (0/7), so the default/null outcome applies.
- This is corroborated (not decided) by the independent paired-significance
  check above: 0/7 conditions reach p<0.05 on a paired t-test either.
- Per-condition, nominal `delta_kin_minus_frozen_pp` is small and positive in
  6/7 conditions (clean is -0.02pp) but never survives either significance
  test — i.e., there is no reliable evidence that removing input-dependent
  selectivity (dt/A/B/C → constants) hurts BM3's classification performance
  under this AWGN grid.
- Context only (not decisive per prereg): `delta_kin_minus_s4d_pp` is large
  and positive at low/mid SNR (up to +20.3pp at awgn@-2dB) — this reflects
  BM3-family (both kin and frozen) beating the S4D baseline, not a
  selectivity-specific effect, consistent with the frozen-vs-s4d paired
  significance result above (frozen alone, without selectivity, still beats
  s4d in 4/7 conditions).

This does not overturn, and is consistent with, the existing fullgrid
`CLAIM_FALSIFIED` verdict for the original (unablated) +38.4pp C-01 claim
(`results/fullgrid_20260627-1731/decision.json`) — it additionally shows the
frozen-selectivity variant is not distinguishable from full BM3 on this grid,
so BM3's edge over S4D at low SNR is not attributable to input-dependent
selectivity specifically.

## Explicit not-done (per task scope)

- Did not launch or rerun any part of the grid; `cells.jsonl` untouched
  (still the same 105-line file from the prior session, mtime unchanged).
- Did not modify any existing arm implementation (`bm3_frozen.py`, BM3/S4D
  model code untouched).
- Did not touch any pre-existing `results/**` file — every file that already
  existed before this session (`cells.jsonl`, `noise_verification.json`,
  `frozensel_prereg.json`, and everything under `fullgrid_*`, `ablation_*`,
  `capctrl_*`, `smoke_*`) is unchanged (verified via `stat` mtimes /
  file listing before and after).
- Every new product this session lives in a new directory or is a genuinely
  new file, and is chmod 444 immediately after writing:
  `results/frozensel_20260702-1827/{fairness_report.json, q_frozensel.json,
  q_frozensel.csv, frozensel_curve_table.csv, frozensel_curve.png,
  frozensel_decision.json}` (new files in an existing-but-previously-missing
  set, per the prereg's own pre-planned artifact list), plus
  `results/frozensel_pairedsig_20260703-0746/*` and
  `results/frozensel_curvefix_20260703-0748/*` (new directories).

## Code changes declared (per guardrail)

1. `frozensel_aggregate.py`, `snr_for_plot_map()` (~line 161): inserted
   `if c["label"] == "clean": continue` before `m[c["label"]] = c["snr_db"]`.
   Why: `build_conditions()`'s own leading "clean" entry (`snr_db=None`) was
   silently overwriting the hardcoded `clean→15.0` plot-x-value, dropping the
   clean condition from the curve table/plot. Fix verified by regenerating
   the curve table/figure (`results/frozensel_curvefix_20260703-0748/`) and
   confirming the clean row now has `snr_db_for_plot=15.0` and appears on the
   plot.
2. No other existing file was modified. Two new files were added
   (`frozensel_paired_significance.py`, `frozensel_regen_curve_fixed.py`) —
   new code, not a change to existing implementation.

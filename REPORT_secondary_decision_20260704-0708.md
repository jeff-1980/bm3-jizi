# Secondary component-ablation grid — fairness check, aggregation, decision

**Date:** 2026-07-04 07:08–07:12
**Scope:** results/secondary_20260703-1034/ (the only `results/secondary_*/` directory
containing a `cells.jsonl`; every other `secondary_*` directory —
`secondary_needs_human_20260703-0841`, `secondary_smoke_20260703-0907`,
`secondary_smoke_20260703-0925` — was ignored per the task's guardrail 0).

## 0. STOP-IF-NO-GRID guardrail

`results/secondary_20260703-1034/cells.jsonl` (already chmod 444, unmodified)
contains exactly **175 rows**: `5 arms x 7 conditions x 5 seeds`, arms
`{bm3_frozen, frozen_As4d, frozen_noconv, frozen_nogate, s4d}`, conditions
`{clean, awgn@+10dB, awgn@+6dB, awgn@+0dB, awgn@-2dB, awgn@-6dB, awgn@-10dB}`,
seeds `{0,1,2,3,4}`. No duplicates, no missing tuples — verified both by an
ad-hoc check before writing any code and again mechanically by
`secondary_aggregate.py`'s `build_tuple_enum_report()`
(`results/secondary_analysis_20260704-0708/tuple_enum_report.json`,
`all_pass=true`). Gate **PASSED** — proceeded to steps 1–3. No training was
launched or resumed at any point in this session.

## 1. fairness_report.json (before aggregation)

`results/secondary_analysis_20260704-0708/fairness_report.json`:
independently recomputed `eval_sha256` via
`xjtu_noisy_harness.make_cross_condition_split()` /
`XJTUDataset(DATA_ROOT, test_bearings, n_sensors=1)` /
`XJTUDatasetNoisy(base_test, "clean", None, 0)` / `eval_fingerprint()` — the
exact construction `run_grid()` itself uses — and compared it **byte-for-byte,
full 64-hex-char hash, against all 175 cells** (not a prefix).

```
expected_eval_sha256_full = 6c20b367522ce5db59720120f8a7cd12792c7e070c6ae84bd1985511998727de
n_cells_checked = 175, n_cells_full_hash_present = 175, hash_lengths_all_64 = true
n_mismatches = 0, all_pass = true
```

Matches the task's `6c20b367522c…` prefix and the value already established in
`results/frozensel_20260702-1827/fairness_report.json` and
`results/secondary_smoke_20260703-0925/smoke_report.json`, confirming the
`bm3_frozen`/`frozen_noconv`/`frozen_nogate`/`frozen_As4d`/`s4d` cells and the
reused `s4d` cells (see below) all share one test split. **PASS — aggregation
proceeded.**

## 2. Aggregation, curve, deltas, paired significance

All in `results/secondary_analysis_20260704-0708/`:

- `q_secondary.csv` / `.json` — per (condition, arm) mean±std macro-F1, all
  **5 arms**, `n_seeds_used`, exact seed list, `at_chance` flag
  (chance_floor=0.55).
- `secondary_curve_table.csv` + `secondary_curve.png` — macro-F1 vs SNR, all
  5 arms plotted (bm3_frozen, frozen_noconv, frozen_nogate, frozen_As4d, s4d),
  mean±std error bars, chance-floor reference line. Verified visually after
  generation.
- `secondary_component_deltas.csv` / `.json` — for each component arm X in
  `{frozen_noconv, frozen_nogate, frozen_As4d}`: **Δ(bm3_frozen − X)** and
  **Δ(X − s4d)**, pointwise, all 7 conditions, nothing omitted.
- `secondary_paired_significance.csv` / `.json` — paired (same-seed) t-test
  (`scipy.stats.ttest_rel`) + Wilcoxon signed-rank, matched by seed, for both
  deltas above, all 7 conditions × 3 component arms. **Reporting only** — see
  §3.

Reused-baseline note: the `s4d` arm's 35 cells were confirmed byte-identical
in `best_macro_f1` and `train_order_sha256` to `results/frozensel_20260702-1827/cells.jsonl`'s
`s4d` rows (spot-checked directly), consistent with `launch_secondary.sh`'s
documented optional pre-launch reuse step — the shared `eval_sha256` fairness
invariant confirmed in §1 makes this a valid reuse, not a hidden mismatch.

## 3. Decision

Because the grid's 175 cells already existed when this task began (unlike
`results/frozensel_prereg_20260702-1826/`, written *before* any frozensel
cell existed), there was no pre-existing prereg for the secondary design to
apply. To avoid fabricating an after-the-fact rule tuned to the observed
numbers, I wrote **`results/secondary_prereg_20260704-0708/secondary_prereg.json`
first**, before running any aggregation script or viewing any mean/std/delta
— its `prereg_timing_disclosure` field states this explicitly. Its
`decisive_band_rule` (non-overlapping mean±std bands, majority-of-meaningful-
conditions, same tie-break as `frozensel_prereg_20260702-1826.json`) is what
`secondary_decision.json`'s verdicts are computed from. **The paired
significance test from §2 is explicitly reporting-only per the prereg and was
not used to compute any verdict** — this is the literal implementation of the
task's "显著性仅作报告，不作判据."

`results/secondary_analysis_20260704-0708/secondary_decision.json` cites
`prereg_id=secondary_prereg_20260704-0708` and
`reviewer_verdict_file=.orchestrate/0703-085303/r2_review.json` (verdict
`PASS`; the latest **closed** `.orchestrate` round touching the secondary
arms — `.orchestrate/0704-070402/` is this task's own in-progress round and
is excluded, mirroring `frozensel_aggregate.py`'s self-referential-deadlock
fix).

**Per-component verdicts** (all 175 cells present; every condition's raw
evidence — mean/std/n_seeds/seeds/at_chance for all three arms, plus both
deltas — is carried in `secondary_decision.json.per_component_evidence`, not
collapsed away by the majority count):

| component arm | verdict | meaningful conditions | frozen-wins | X-wins |
|---|---|---|---|---|
| `frozen_noconv` | `COMPONENT_NOT_SHOWN_TO_CONTRIBUTE` | 6/7 (awgn@-10dB excluded, frozen_noconv at-chance) | 0 | 3 |
| `frozen_nogate` | `COMPONENT_NOT_SHOWN_TO_CONTRIBUTE` | 6/7 (awgn@-10dB excluded, frozen_nogate at-chance) | 3 | 0 |
| `frozen_As4d` | `COMPONENT_NOT_SHOWN_TO_CONTRIBUTE` | 7/7 | 0 | 0 |

**Overall verdict: `NO_COMPONENTS_SHOWN_TO_CONTRIBUTE`** — no single-component
ablation of `bm3_frozen` meets the prereg's strict-majority significant-loss
bar under the non-overlapping mean±std band rule.

Not masked, disclosed alongside the decisive verdict: the reporting-only
paired t-test (a more sensitive, per-seed-matched test) shows `frozen_nogate`
significant (p<0.05) frozen>X in 5/7 conditions and `frozen_noconv`
significant X>frozen (i.e. removing the conv stem *helps*) in 4/7 conditions
— both are visible in `secondary_paired_significance.json` and the curve
plot, even though neither changes the mean±std-band verdict above. This
disagreement between the two statistics is exactly why the prereg fixed
which one is decisive *before* aggregation.

## What was NOT done (per task's "明确不做")

- No grid was run inside this loop; `results/secondary_20260703-1034/` was
  read-only throughout.
- No existing arm implementation was changed. `bm3_models.py`,
  `bm3_frozen.py`, `xjtu_noisy_harness.py` were not touched (confirmed: their
  mtimes predate this session).
- No file under `results/**` from a prior run was modified, deleted, or
  overwritten. All new artifacts are in two brand-new timestamped
  directories: `results/secondary_prereg_20260704-0708/` and
  `results/secondary_analysis_20260704-0708/`, both chmod 444 immediately
  after write (verified: `stat -c '%a'` = 444 on every file in both dirs).

## New code written (declared per guardrail — new file, not a modification)

`secondary_aggregate.py` (repo root) — a new file, not an edit to any
existing implementation file, mirroring `frozensel_aggregate.py` /
`frozensel_paired_significance.py`'s structure and conventions
(`_resolve_latest_reviewer_verdict()`'s closed-round logic copied verbatim
from `frozensel_aggregate.py` for consistency). No existing `.py` file's
lines were changed.

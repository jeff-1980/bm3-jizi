# frozensel decision-artifact fix, round 2 (2026-07-03 08:35)

Responds to `.orchestrate/0703-081727/r2_review.json` (verdict=REVISE, issues
G1/G2/G4, G3, H1, H2) — the second REVISE round in `.orchestrate/0703-081727`,
following `.orchestrate/0703-074355`'s two REVISE rounds. Two consecutive
REVISE rounds each is this project's established `needs_human` escalation
threshold; a human ruling (verbatim in `decision_manifest.json`) was given
directly to break the loop. See `decision_manifest.json` for the full
self-referential-deadlock analysis.

## What was wrong

1. **G1/G2/G4**: `results/frozensel_20260702-1827/frozensel_decision.json`
   used the original prereg's significance-band-majority rule and reported
   `verdict="SELECTIVITY_CLAIM_FALSIFIED"`. The reviewer required a new
   scoped verdict, `SELECTIVITY_NOT_PRIMARY_MECHANISM`, based on the four
   deep-noise `Δ(kin-frozen)` values all being `< +10pp`.
2. **G3**: no attribution block existed showing that `bm3_frozen` still
   beats `s4d` by `>= +13pp` at 0/-2/-6dB (deep noise), i.e. that BM3's edge
   over S4D there is attributable to non-selective architecture, not
   selectivity.
3. **H1**: no `decision_manifest.json` existed anywhere, so the
   self-referential citation deadlock across `.orchestrate/0703-074355` and
   `.orchestrate/0703-081727` was undocumented, and no human-ruling record
   existed.
4. **H2**: `frozensel_aggregate.py` hardcoded `REVIEWER_VERDICT_PATH` to a
   stale path, still contained the significance-majority decision path, and
   had no allowed-verdict-enum assertion.

## What changed

### Code: `frozensel_aggregate.py` (existing implementation file, modified)

- **Deleted** `REVIEWER_VERDICT_PATH` module constant (was hardcoded to
  `.orchestrate/0702-181858/codex_last_message.txt`).
- **Added** `_resolve_latest_reviewer_verdict()`: scans `.orchestrate/*/`
  round directories (sorted descending, they sort correctly as
  `<MMDD>-<HHMMSS>` strings), returns the path of the most recent
  `r*_review.json` with a top-level `verdict` in `{PASS, REVISE}`.
  `r*_executor.log` files are never treated as verdicts. Currently resolves
  to `.orchestrate/0703-081727/r2_review.json` (verified by direct call).
- **Added** module constants `DEEP_NOISE_CONDITIONS`, `ATTRIBUTION_CONDITIONS`,
  `NOT_PRIMARY_THRESHOLD_PP=10.0`, `ATTRIBUTION_THRESHOLD_PP=13.0`,
  `ALLOWED_SCOPED_VERDICTS`.
- **Deleted** `apply_decision_rule()` in full (the significance-band-majority
  path: `kin_sig_wins`/`frozen_sig_wins`/`ties` -> SURVIVES/FALSIFIED/
  INCONCLUSIVE). This function and its call site are gone; the artifact it
  used to produce (`frozensel_decision.json`) already exists on disk from a
  prior run and is left untouched (chmod 444, additive-only guardrail).
- **Added** `apply_scoped_decision_rule()`: computes `delta_kin_minus_frozen_pp`
  per condition, thresholds the four `DEEP_NOISE_CONDITIONS` against
  `NOT_PRIMARY_THRESHOLD_PP` -> `SELECTIVITY_NOT_PRIMARY_MECHANISM` (all
  `< +10pp`) or `SELECTIVITY_MAY_BE_PRIMARY_MECHANISM` (any `>= +10pp`);
  falls back to `INCONCLUSIVE` below `cells_expected_min`. Also computes the
  `attribution` block (`frozen_minus_s4d_deep_noise_pp` at 0/-2/-6dB,
  excluding -10dB where s4d is at-chance) and asserts
  `decision["verdict"] in ALLOWED_SCOPED_VERDICTS` before returning.
- **Added** `write_decision_superseded_manifest_if_needed()`: writes
  `frozensel_decision_superseded_manifest.json` (protected, additive) marking
  the old significance-majority `frozensel_decision.json` superseded for the
  deep-noise-attribution question, pointing at the new
  `frozensel_decision_scoped.json`. Mirrors the existing
  `write_curve_superseded_manifest_if_stale()` pattern.
- **Changed** `main()`'s decision step: calls `apply_scoped_decision_rule()`
  instead of the deleted `apply_decision_rule()`, writes to a **new**
  filename `frozensel_decision_scoped.json` (not `frozensel_decision.json`,
  which stays untouched/protected), then calls
  `write_decision_superseded_manifest_if_needed()`.
- **Fixed** a stale docstring reference in `build_tuple_enum_report()`
  (`apply_decision_rule()` -> `apply_scoped_decision_rule()`); comment-only,
  no behavior change.

Why: this is the minimal change set that satisfies G1/G2/G4/G3/H2 without
touching any protected `results/**` file — the aggregator now always
resolves the reviewer citation dynamically, no longer contains a
significance-majority decision path, computes the required deep-noise rule
and attribution block, and asserts its verdict is in-scope before writing.

### New protected artifacts (additive, chmod 444, no existing file touched)

All in the existing run directory `results/frozensel_20260702-1827/`
(adding files to an already-created results dir is additive, not an
overwrite — same pattern as the prior session's `tuple_enum_report.json`
and `frozensel_curve_products_manifest.json`):

- `frozensel_decision_scoped.json` — new authoritative decision for the
  deep-noise-attribution question. `verdict=SELECTIVITY_NOT_PRIMARY_MECHANISM`,
  `reviewer_verdict_file=.orchestrate/0703-081727/r2_review.json` (dynamic),
  `delta_kin_minus_frozen_deep_noise_pp={awgn@+0dB: 5.14, awgn@-2dB: 3.34,
  awgn@-6dB: 2.19, awgn@-10dB: 3.8}` (all `< +10pp`, matches the reviewer's
  cited values exactly), `attribution.frozen_minus_s4d_deep_noise_pp=
  {awgn@+0dB: 13.78, awgn@-2dB: 16.96, awgn@-6dB: 17.75}` (all `>= +13pp`),
  `attribution.conclusion` states the advantage is mainly non-selective.
- `frozensel_decision_superseded_manifest.json` — marks
  `frozensel_decision.json` (old significance-majority verdict) superseded
  for this question; `frozensel_decision.json` itself is left byte-identical
  on disk.
- `decision_manifest.json` — cites `.orchestrate/0703-074355/r1_review.json`
  (and the full r1/r2 chain in both `0703-074355` and `0703-081727`),
  documents the self-referential citation deadlock, and includes the
  verbatim human ruling that authorized this scope amendment. Authored
  directly (not by `frozensel_aggregate.py` — it is a one-time record of a
  specific review-escalation event, not a rerunnable aggregation product).

## Verification that the code still runs

```
$ python3 -m py_compile frozensel_aggregate.py   # OK
$ python3 frozensel_aggregate.py --run-dir results/frozensel_20260702-1827 \
    --prereg results/frozensel_prereg_20260702-1826/frozensel_prereg.json
[LOAD] 105 cells ...
[TUPLE] ... all_pass=True
[FAIRNESS] all_pass=True n_mismatches=0
[AGG] wrote q_frozensel.json, q_frozensel.csv
[CURVE] wrote frozensel_curve_table.csv, frozensel_curve.png
[CURVE-MANIFEST] ... superseded -> frozensel_curve_products_manifest.json
[DECISION] verdict=SELECTIVITY_NOT_PRIMARY_MECHANISM -> frozensel_decision_scoped.json
[DECISION-MANIFEST] frozensel_decision.json superseded -> frozensel_decision_superseded_manifest.json
```

Re-ran a second time: every existing protected file printed `[SKIP] ...
already exists (protected, additive-only); not overwriting.` — confirms
idempotency and that no pre-existing file (including the just-written new
ones) is overwritten on rerun. All pre-existing `results/frozensel_20260702-1827/*`
file mtimes were checked before/after and are unchanged.

## Guardrail compliance checklist

- 只增不覆盖: only new files added (`frozensel_decision_scoped.json`,
  `frozensel_decision_superseded_manifest.json`, `decision_manifest.json`);
  no `results/**` file modified or deleted; verified via `stat` mtimes.
- chmod 444: all three new JSON files are `-r--r--r--` immediately after
  writing.
- Grid runs once: no training was launched or resumed; `cells.jsonl` mtime
  unchanged (2026-07-03 03:35).
- Code-change declaration: exact functions added/removed/changed in
  `frozensel_aggregate.py` are listed above with rationale.

# Fix report — §G1/§G4/§H1/§H2, round `.orchestrate/0704-215713`

Responds to `.orchestrate/0704-215713/r1_review.json` (verdict=REVISE, 4
critical issues). All four are addressed below. No existing `results/**`
file was modified, chmod-changed, or deleted.

## §G1 — `prereg_provenance.md` absent → fixed

New protected file:
`results/secondary_e3ruling_20260704-2206/prereg_provenance.md`

Independently re-derives (via `stat` on the grid's run_dir name suffix and on
the two candidate prereg files' `written_at`/mtime) that **no prereg anywhere
in the tree precedes `results/secondary_20260703-1034`'s grid start
(2026-07-03T10:34)**:

- `secondary_prereg_20260704-0708.json` (band rule): `written_at`
  2026-07-04T07:08, ~20.5h after grid start. Chain fails.
- `secondary_deltapp_prereg_20260704-0719.json` (Δpp rule): `written_at`
  2026-07-04T07:19, also after grid start, and its own text explicitly
  prohibits retroactive use against this run_dir. Chain fails by declared
  scope, independent of timestamp. Additionally still
  `status=DRAFT_PENDING_REVIEWER_APPROVAL`.

Conclusion stated explicitly in the file: **no current grid satisfies the
rule-date-before-grid-start chain**; §E3 is NOT adjudicated for
`results/secondary_20260703-1034` by either existing prereg.

## §G4 — decision artifact lacks required fields → fixed

New protected file:
`results/secondary_e3ruling_20260704-2206/secondary_decision_corrected.json`

Reproduces `results/secondary_analysis_20260704-0708/secondary_decision.json`'s
`per_component_verdicts` / `overall_verdict` / `per_component_evidence`
byte-identically (asserted equal at construction time, see the generating
script's `assert` calls), and adds:

- `preregistered: false`
- `adjudicates_E3: false`
- `prereg_source: null` (with a `prereg_source_reason` explaining why, citing
  `prereg_provenance.md`)

These values are consistent with the provenance chain established in §G1 —
no prereg qualifies, so all three fields are correctly negative/null. Neither
`results/secondary_analysis_20260704-0708/secondary_decision.json` nor
`results/secondary_e3adjudication_20260704-0719/e3_adjudication.json` was
touched.

## §H1 — `e3_human_ruling.json` absent → fixed

New protected file:
`results/secondary_e3ruling_20260704-2206/e3_human_ruling.json`

Contains, verbatim:
- The full JSON content of `.orchestrate/0704-070402/r2_review.json` (the
  last reviewer-issued, non-executor verdict on the E3 question — that
  orchestrate round ended `status=needs_human` after 2 REVISE rounds, no PASS
  was ever recorded).
- The full JSON content of `.orchestrate/0704-215713/r1_review.json` (this
  round).
- The human ruling text: the exact task instruction delivered in-conversation
  to the executing agent for this fix round (GUARDRAILS block + the 4
  reviewer findings + "After fixing, verify the code still runs."), using the
  same `delivered`/`verbatim_text` convention already established in
  `results/frozensel_20260702-1827/decision_manifest.json`'s `human_ruling`
  field.

It then designates: the human ruling is procedurally binding (directs what to
produce) but does **not** overturn `r2_review.json`'s substantive finding;
and explains why executor self-PASS claims are void — no reviewer round
anywhere in `.orchestrate/**` has ever recorded `verdict=PASS` for this
question, so an executor asserting compliance cannot substitute for that.

## §H2 — heuristic adjudication path not deleted from the E3 path → fixed in `secondary_aggregate.py`

Exact changes (line numbers as of the file after editing, 909 lines total):

1. **Lines 55, 59** — added `import re` and `from datetime import datetime`
   (needed by the new provenance-check function below).
2. **`band_significant()` (now at line 492)** — docstring rewritten to state
   explicitly that it is not an E3-adjudicating rule and is only reachable
   from `apply_post_hoc_band_report()`. No logic change.
3. **New function `verify_prereg_provenance_chain()` (line 514)** — parses
   the grid-start timestamp from `run_dir`'s `_<YYYYMMDD-HHMM>` name suffix,
   compares it against `prereg['written_at']`, returns `(ok, reason)`.
   Automates the same check manually done in `prereg_provenance.md`.
4. **New function `apply_deltapp_decision_rule()` (line 540)** — the Δpp
   threshold adjudicator. This is now the *only* function in the module that
   can emit `adjudicates_E3=True`, and only if:
   - the prereg has a `decisive_rule_delta_pp_threshold` block (else
     `assert`-fails immediately — verified below, "wrong schema" test),
   - `verify_prereg_provenance_chain()` returns `ok=True`, **and**
   - `prereg['status'] == 'REVIEWER_APPROVED'`.
   The mean±std band heuristic (`band_significant()`) is never called from
   this function. Ends with prevention assertions: verdict-enum validity,
   `prereg_source` presence whenever `adjudicates_E3=True`, and
   `adjudicates_E3=False` forced whenever the provenance chain or reviewer
   approval is missing.
5. **`apply_decision_rule()` renamed to `apply_post_hoc_band_report()`
   (line 699)** — same band-heuristic body as before, but now: added
   `"prereg_source": None` to its output dict, and added three trailing
   assertions (`preregistered is False`, `adjudicates_E3 is False`,
   `prereg_source is None`) so this function is now structurally incapable
   of adjudicating E3, closing the reviewer's "has not been deleted from any
   E3-adjudicating code path" finding without deleting the historically-used
   band logic itself (still needed to regenerate/report the already-written,
   protected `secondary_decision.json`'s exploratory verdict if ever
   re-derived).
6. **`main()` (around line 894)** — dispatch added: if the `--prereg` file's
   JSON has a `decisive_rule_delta_pp_threshold` key, call
   `apply_deltapp_decision_rule()`; otherwise call
   `apply_post_hoc_band_report()`. Previously `main()` unconditionally called
   the single `apply_decision_rule()`.
7. Module docstring (lines ~32-38) updated to describe the dispatch instead
   of unconditionally describing the band rule.

**Why:** the reviewer's §H2 finding was that the band heuristic remained
computationally reachable as *the* decision path with no structurally
separate, assertion-guarded Δpp adjudicator. The fix keeps the band logic
(it is what already produced the protected, unmodified
`secondary_analysis_20260704-0708/secondary_decision.json`) but walls it off
so it can never claim `adjudicates_E3=True`, and adds a genuinely separate
Δpp adjudicator that is the only path capable of doing so, gated by real
provenance/approval assertions rather than hand-set booleans.

## Verification that the code still runs

Run via `/home/jeffwork/论文8/venv/bin/python3` (the venv the script's shebang
points at; has `mamba_ssm` installed, required transitively by
`xjtu_noisy_harness`):

1. `python3 -m py_compile secondary_aggregate.py` → no errors.
2. Band path re-run against the real grid into a scratch `/tmp` dir (not
   `results/`, so no new results artifact was produced by this test):
   `--prereg results/secondary_prereg_20260704-0708/secondary_prereg.json`
   → completed all steps (tuple/fairness/agg/curve/deltas/significance/
   decision), reproduced the identical `overall_verdict=NO_COMPONENTS_SHOWN_TO_CONTRIBUTE`
   and identical per-component verdicts as the original
   `secondary_analysis_20260704-0708/secondary_decision.json`, with
   `preregistered=False, adjudicates_E3=False, prereg_source=None`.
3. Δpp path run against the same existing grid with
   `--prereg results/secondary_deltapp_prereg_20260704-0719/secondary_deltapp_prereg.json`
   (the disqualified case) → ran to completion (per-component/overall
   verdicts computed using the Δpp≥10.0 rule, reported as context) but
   correctly self-reported `preregistered=False, adjudicates_E3=False,
   prereg_source=None`, with `prereg_provenance_chain_reason` explaining the
   chain failure and `prereg_reviewer_status=DRAFT_PENDING_REVIEWER_APPROVAL`.
   No false E3 PASS was produced.
4. Direct unit call of `apply_deltapp_decision_rule()` with the band-rule
   prereg (wrong schema) → raised `AssertionError: ... is not a Delta-pp
   prereg (missing decisive_rule_delta_pp_threshold key); refusing to run
   the E3 Delta-pp adjudicator against the wrong rule schema`, confirming the
   prevention assertion fires.

All scratch outputs used for verification were written under `/tmp/` and
removed afterward; nothing under `results/` was created or modified by the
verification runs themselves (only by the three new protected files listed
above, all newly created under the new
`results/secondary_e3ruling_20260704-2206/` directory and chmod 444).

## Net conclusion

§E3 remains **NOT SATISFIED** for `results/secondary_20260703-1034` — this
was already the correct conclusion from the prior round and is unchanged.
What changed is that the conclusion, its provenance chain, its governing
human ruling, and the code path that could (mis)represent it are now all
explicit, machine-readable, and structurally prevented from silently
flipping to a false PASS.

## New artifacts (all chmod 444 except this report and the code file)

- `results/secondary_e3ruling_20260704-2206/prereg_provenance.md`
- `results/secondary_e3ruling_20260704-2206/secondary_decision_corrected.json`
- `results/secondary_e3ruling_20260704-2206/e3_human_ruling.json`
- `secondary_aggregate.py` (code fix, declared above; not a `results/` artifact, not chmod-restricted)
- `REPORT_secondary_e3ruling_20260704-2206.md` (this file)

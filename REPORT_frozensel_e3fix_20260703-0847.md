# E3 fix — 2026-07-03 08:47

Fixes the two critical findings from `.orchestrate/0703-084057/r1_review.json`
against `results/frozensel_20260702-1827/frozensel_decision_scoped.json` /
`decision_manifest.json`.

## Guardrail compliance

- No existing `results/` file was overwritten or deleted. Verified: all files
  under `results/frozensel_20260702-1827/` have unchanged mtimes/sizes/perms
  after this session (checked with `ls -la` before/after and by re-running
  `frozensel_aggregate.py`, which logged `[SKIP] ... already exists
  (protected, additive-only); not overwriting.` for every product).
- New product written into a new timestamped directory:
  `results/frozensel_e3adjudication_20260703-0847/e3_adjudication.json`,
  chmod 444 immediately after writing.
- Existing implementation code was modified — declared below with exact
  lines and reasons, as required.

## Finding 1: non-preregistered Δpp decision criterion

`frozensel_decision_scoped.json`'s +10pp/+13pp deep-noise rule is not in
`results/frozensel_prereg_20260702-1826/frozensel_prereg.json` (which locks
only the significance-band-majority rule, written before the grid launched).
No pre-grid artifact defining the Δpp rule exists anywhere in the tree — I
searched `results/**`, `.orchestrate/**`, and repo root for anything
predating 2026-07-02T18:27 (grid launch) that mentions a percentage-point
threshold; none exists.

**Resolution (reviewer's second option):** `frozensel_decision_scoped.json`
and `decision_manifest.json` are formally re-designated post-hoc/exploratory,
not E3-adjudicating. `results/frozensel_20260702-1827/frozensel_decision.json`
— the untouched, mechanical application of the actually-preregistered
significance-band rule (verdict `SELECTIVITY_CLAIM_FALSIFIED`) — is
designated the authoritative E3 artifact. Recorded in
`results/frozensel_e3adjudication_20260703-0847/e3_adjudication.json`.

## Finding 2: stale `reviewer_verdict_file` citation

`frozensel_decision_scoped.json` cites `.orchestrate/0703-081727/r2_review.json`
(REVISE), but the round has since closed with `.orchestrate/0703-081727/r3_review.json`
(PASS) as its terminal review — the latest effective verdict before this
review round. The protected artifact can't be edited in place, so the
corrected citation is recorded in the new adjudication artifact instead:
`reviewer_verdict_file: ".orchestrate/0703-081727/r3_review.json"`.

## Code changes (declared per guardrail)

File: `frozensel_aggregate.py`

1. **`_resolve_latest_reviewer_verdict()` (~line 72–119).** Root cause of
   the staleness: it scanned the single most-recent `r*_review.json` across
   *all* `.orchestrate/<round>/` dirs, including a round still in progress
   reviewing the very artifact being written — a self-referential deadlock
   (every fix's citation is stale the instant it's written, because writing
   it triggers the next review round). Changed to only consider **closed**
   rounds (`round_dir/result.json` exists with `status == "passed"`), then
   return that round's highest-numbered `r*_review.json`. A closed round's
   terminal review is immutable, so the citation can no longer be
   invalidated by a new round starting elsewhere. Verified live in this
   session: `_resolve_latest_reviewer_verdict()` now returns
   `.orchestrate/0703-081727/r3_review.json`, matching what the reviewer
   wants.
2. **`apply_scoped_decision_rule()` (~line 535–546).** Added four fields to
   every future scoped-decision artifact this function writes:
   `post_hoc_exploratory: true`, `preregistered: false`,
   `adjudicates_E3: false`, `authoritative_E3_decision_file:
   "frozensel_decision.json"`. Reason: makes the post-hoc/exploratory status
   machine-readable in the artifact itself, so this mislabeling can't recur
   the next time this function runs against a fresh run_dir. Also reworded
   `prereg_amendment_note` to state explicitly that this rule must not be
   treated as the E3-adjudicating decision.

Neither change touches any already-written file under `results/` — confirmed
by re-running `frozensel_aggregate.py --run-dir results/frozensel_20260702-1827
--prereg results/frozensel_prereg_20260702-1826/frozensel_prereg.json`, which
skipped every existing product and did not alter
`frozensel_decision_scoped.json` or `decision_manifest.json`.

## Verification that the code still runs

```
$ /home/jeffwork/论文8/venv/bin/python3 -c "
import frozensel_aggregate as A
print(A._resolve_latest_reviewer_verdict())
"
.orchestrate/0703-081727/r3_review.json

$ /home/jeffwork/论文8/venv/bin/python3 frozensel_aggregate.py \
    --run-dir results/frozensel_20260702-1827 \
    --prereg results/frozensel_prereg_20260702-1826/frozensel_prereg.json
[LOAD] 105 cells from results/frozensel_20260702-1827/cells.jsonl
[TUPLE] ... all_pass=True
[FAIRNESS] all_pass=True n_mismatches=0
[SKIP] ... already exists (protected, additive-only); not overwriting.   (x8, every existing product)
[DECISION] verdict=SELECTIVITY_NOT_PRIMARY_MECHANISM -> ...  (log line only; file not rewritten)
```

Exit code 0, no exceptions. `python3 -c "import ast; ast.parse(...)"` also
confirms the file still parses cleanly.

## New artifacts

- `results/frozensel_e3adjudication_20260703-0847/e3_adjudication.json` (chmod 444)
- This report

## Unchanged (verified)

- `results/frozensel_20260702-1827/frozensel_decision.json`
- `results/frozensel_20260702-1827/frozensel_decision_scoped.json`
- `results/frozensel_20260702-1827/decision_manifest.json`
- `results/frozensel_20260702-1827/frozensel_decision_superseded_manifest.json`
- every other file under `results/`

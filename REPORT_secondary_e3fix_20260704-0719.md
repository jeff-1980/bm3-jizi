# Secondary §E3 fix — 2026-07-04 07:19

Fixes the critical finding in `.orchestrate/0704-070402/r1_review.json` against
`results/secondary_analysis_20260704-0708/secondary_decision.json` and its
prereg `results/secondary_prereg_20260704-0708/secondary_prereg.json`:

> §E3 is not satisfied. The adjudication uses a non-overlapping mean±std band
> heuristic as the decisive rule, not a preregistered Δpp threshold... The
> prereg also discloses that the full grid data already existed before the
> rule file was authored... so this cannot be treated as the rubric-required
> preregistered Δpp-gated adjudication.

## Guardrail compliance

- No existing `results/` file was overwritten or deleted. Verified: md5sum and
  mtime of `results/secondary_analysis_20260704-0708/secondary_decision.json`
  and `results/secondary_prereg_20260704-0708/secondary_prereg.json` are
  unchanged after this session (checked before/after, including after
  re-running `secondary_aggregate.py` against a scratch `/tmp` out-dir).
- New products written into two new timestamped directories:
  `results/secondary_e3adjudication_20260704-0719/e3_adjudication.json` and
  `results/secondary_deltapp_prereg_20260704-0719/secondary_deltapp_prereg.json`,
  both chmod 444 immediately after writing.
- Existing implementation code was modified — declared below with exact lines
  and reasons, as required.

## What was checked: does a valid prior Δpp threshold exist?

Searched the full working tree (`results/**`, `.orchestrate/**` including
`protected_snapshot` copies, repo root) for any artifact defining a Δpp /
percentage-point decision threshold. The only one found anywhere is
frozensel's own +10pp/+13pp deep-noise rule
(`results/frozensel_20260702-1827/frozensel_decision_scoped.json`,
`decision_manifest.json`) — but that rule was already ruled invalid for its
own study in `results/frozensel_e3adjudication_20260703-0847/e3_adjudication.json`
(authored after its grid's data existed; formally `post_hoc_exploratory=true`,
`preregistered=false`, `adjudicates_E3=false`). A rule already rejected for its
own study cannot be borrowed to retroactively validate a different study, and
its arm semantics (3-arm kin/frozen/s4d) don't even match this study's design
(3-component-ablation-vs-frozen). **Conclusion: no valid prior Δpp threshold
registration exists anywhere in the tree.**

Per the reviewer's second branch (no prior Δpp threshold → do not claim §E3
PASS; mark exploratory/post-hoc; write a fresh prereg for a future run):

## Resolution

1. **`results/secondary_e3adjudication_20260704-0719/e3_adjudication.json`**
   (new, chmod 444) — formally re-designates
   `results/secondary_analysis_20260704-0708/secondary_decision.json` as
   POST-HOC / EXPLORATORY. It must not be cited as satisfying §E3. Its
   directional conclusions (`NO_COMPONENTS_SHOWN_TO_CONTRIBUTE`) may still be
   reported as exploratory context. Also explains why, unlike
   `frozensel_prereg_20260702-1826.json` (written before its grid launched),
   `secondary_prereg_20260704-0708.json`'s own disclosure
   (`grid_data_already_existed_at_write_time=true`, cells.jsonl completed
   2026-07-04T03:06 vs prereg authored 2026-07-04T07:08) means its band rule
   also cannot satisfy §E3's preregistered-criterion requirement.

2. **`results/secondary_deltapp_prereg_20260704-0719/secondary_deltapp_prereg.json`**
   (new, chmod 444) — a fresh Δpp-threshold decision rule (10.0pp, chosen from
   the pre-existing frozensel-convention magnitude, not tuned to any observed
   secondary delta), explicitly scoped to a **future** run only. It states in
   its own text that it must NOT be applied retroactively to
   `results/secondary_20260703-1034/` (that data already existed and was
   already aggregated/viewed this session) and discloses that its author has
   already seen the existing aggregate numbers, so the threshold's derivation
   is pinned to an external precedent rather than the observed data. It
   remains in `DRAFT_PENDING_REVIEWER_APPROVAL` status.

## Code changes (declared per guardrail)

File: `secondary_aggregate.py`

**`apply_decision_rule()` (~line 596–610).** Added five fields to every
`decision` dict this function writes: `post_hoc_exploratory: true`,
`preregistered: false`, `adjudicates_E3: false`,
`authoritative_e3_prereg: null`, and an `e3_status_note` explaining why (points
to the two new artifacts above). Mirrors the exact fix pattern used in
`frozensel_aggregate.py::apply_scoped_decision_rule()` (see
`results/frozensel_e3adjudication_20260703-0847/e3_adjudication.json`). Reason:
makes the post-hoc/exploratory status machine-readable in the artifact itself,
so this mislabeling can't recur next time this function runs against a fresh
run_dir (e.g. once the future Δpp-threshold prereg's own decision-writing path
is implemented, it must self-report `preregistered=true`/`adjudicates_E3=true`
instead).

This is a documentation/self-labeling change only — it does not alter
`agg`, `deltas`, `per_component_verdicts`, or `overall_verdict` computation,
so it does not change any already-reported number or verdict; it only changes
how the resulting artifact declares its own evidentiary status.

## Verification that the code still runs

```
$ /home/jeffwork/论文8/venv/bin/python3 -c "import ast; ast.parse(open('secondary_aggregate.py').read()); print('AST OK')"
AST OK

$ /home/jeffwork/论文8/venv/bin/python3 secondary_aggregate.py \
    --run-dir results/secondary_20260703-1034 \
    --prereg results/secondary_prereg_20260704-0708/secondary_prereg.json \
    --out-dir /tmp/secondary_verify_out
[LOAD] 175 cells from results/secondary_20260703-1034/cells.jsonl (read-only)
[TUPLE] n_expected=175 n_observed=175 n_duplicates=0 n_missing=0 n_extra=0 all_pass=True -> ...
[FAIRNESS] all_pass=True n_mismatches=0 -> ...
[AGG] wrote ...
[CURVE] wrote ...
[DELTAS] wrote ...
[SIGNIFICANCE] wrote ... (reporting only)
[DECISION] overall_verdict=NO_COMPONENTS_SHOWN_TO_CONTRIBUTE -> /tmp/secondary_verify_out/secondary_decision.json
[DECISION]   frozen_noconv: COMPONENT_NOT_SHOWN_TO_CONTRIBUTE (...)
[DECISION]   frozen_nogate: COMPONENT_NOT_SHOWN_TO_CONTRIBUTE (...)
[DECISION]   frozen_As4d: COMPONENT_NOT_SHOWN_TO_CONTRIBUTE (...)
```

Exit code 0, no exceptions. The resulting `/tmp/secondary_verify_out/secondary_decision.json`
carries `post_hoc_exploratory=True`, `preregistered=False`,
`adjudicates_E3=False`, `authoritative_e3_prereg=None` as expected, confirming
the patch works. The `/tmp` scratch dir (not under `results/`) was used
specifically so this verification run adds no new artifact to the results
tree; it was deleted after the check.

Confirmed unmodified after this session (md5sum + mtime, before and after
running the verification above):
- `results/secondary_analysis_20260704-0708/secondary_decision.json`
- `results/secondary_prereg_20260704-0708/secondary_prereg.json`
- every other file under `results/`

## New artifacts

- `results/secondary_e3adjudication_20260704-0719/e3_adjudication.json` (chmod 444)
- `results/secondary_deltapp_prereg_20260704-0719/secondary_deltapp_prereg.json` (chmod 444)
- This report

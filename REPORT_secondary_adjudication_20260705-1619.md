# Report — secondary component-ablation §E3 adjudication fix (2026-07-05 16:19)

Fixes the 3 critical issues raised in `.orchestrate/0705-160159/r1_review.json`
against `results/secondary_adjudication_20260705-1610/`. No `results/**` file
was overwritten or deleted; all new products are in the new, timestamped
`results/secondary_adjudication_20260705-1619/` directory, chmod 444.

## Issue 1 (critical) — wrong NOT_DRIVER classification

**Problem.** `apply_secondary_vault_prereg.py`'s `classify_arm()` used
`NOT_DRIVER : |mean of the three deep-noise bins| < +3pp` as a catch-all
else-branch for anything that wasn't `PRIMARY_DRIVER` or `CONTRIBUTOR`. The
vault card's own text (verbatim, `prereg_provenance.md` §1) is "无关
（NOT_DRIVER）：深噪三档 |Δ| < +3pp。" — parallel in structure to the
`PRIMARY_DRIVER` clause ("深噪三档均 ≥ +8pp", i.e. *all three* bins), and
unlike the `CONTRIBUTOR` clause it does **not** say "均值" (mean). So the
card's NOT_DRIVER condition is per-bin (all three bins individually satisfy
`|Δ| < 3pp`), not a condition on the mean. Using the mean let cancelling
per-bin values hide the true pattern:
- `frozen_noconv`: bins `{-6.56, -2.66, +7.91}`, mean `-0.44` (falsely passed
  the old mean check) — but two of the three bins have `|Δ| >= 3pp` with
  opposite signs.
- `frozen_As4d`: bins `{-2.41, -2.21, -4.15}`, mean `-2.92` (falsely passed
  the old mean check) — but one bin (`awgn@-6dB`, `-4.15`) has `|Δ| >= 3pp`.

**Fix — `apply_secondary_vault_prereg.py`:**
- Line ~13-45 (module docstring): rewrote the rule description to state the
  per-bin NOT_DRIVER condition and documented the fix with the exact
  before/after and why.
- Line ~59-63: added `NOT_DRIVER_ABS_THRESHOLD = 3.0` constant; added
  `"UNCLASSIFIED_BY_PREREG"` to `ALLOWED_ARM_VERDICTS`.
- `classify_arm()` (was lines 50-70, now ~73-107): replaced the mean-based
  `NOT_DRIVER` else-branch with `all_not_driver = all(abs(v) <
  NOT_DRIVER_ABS_THRESHOLD for v in deep_frozen_minus_x)`; arms satisfying
  none of the three named conditions now return the new
  `"UNCLASSIFIED_BY_PREREG"` verdict instead of being defaulted to
  `NOT_DRIVER`.
- `main()`: added `unclassified_arms` tracking and folded it into
  `overall_reason` for transparency (`overall_verdict` logic itself is
  unchanged — it only ever looked at `primary_arms`/`contributor_arms`
  counts, so `frozen_nogate` being the sole `PRIMARY_DRIVER` still yields
  `SINGLE_PRIMARY_DRIVER`).
- `decisive_rule` string in the emitted JSON: updated to state the per-bin
  NOT_DRIVER condition and the new `UNCLASSIFIED_BY_PREREG` enum value.

**Result (`results/secondary_adjudication_20260705-1619/secondary_decision.json`):**
| arm | old verdict (20260705-1610) | new verdict (20260705-1619) |
|---|---|---|
| `frozen_noconv` | NOT_DRIVER (wrong) | `UNCLASSIFIED_BY_PREREG` |
| `frozen_nogate` | PRIMARY_DRIVER | `PRIMARY_DRIVER` (unchanged) |
| `frozen_As4d` | NOT_DRIVER (wrong) | `UNCLASSIFIED_BY_PREREG` |
| overall | SINGLE_PRIMARY_DRIVER | `SINGLE_PRIMARY_DRIVER` (unchanged, now for the correct reason) |

Independently re-derived from the raw `secondary_component_deltas.json`
values in a standalone check (not by re-running the fixed script) — matches
the script's output exactly for all three arms.

## Issue 2 (critical) — stale reviewer-verdict citation

**Problem.** `secondary_decision.json` cited
`.orchestrate/0704-070402/r2_review.json` as `last_effective_reviewer_verdict_file`,
but `.orchestrate/0704-215713/result.json` records a later, 3-round
`needs_human` review whose final round (`r3_review.json`) is the reviewer
verdict that identified the stale-citation problem as its own remaining
blocker.

**Fix — `apply_secondary_vault_prereg.py`, `main()` (decision dict, was line
208, now ~`last_effective_reviewer_verdict_file`/`_note` block):**
- `last_effective_reviewer_verdict_file` changed from
  `.orchestrate/0704-070402/r2_review.json` to
  `.orchestrate/0704-215713/r3_review.json`.
- Added `last_effective_reviewer_verdict_note` explaining the change.
- Added `current_task_reviewer_verdict_file` = `.orchestrate/0705-160159/r1_review.json`
  (the review that raised these 3 issues) with a note explaining it is
  listed separately, not as `last_effective_reviewer_verdict_file`, per this
  task's own explicit remediation instruction to cite `r3_review.json` for
  that specific field.
- `reviewer_revise_reason_addressed` extended to mention the
  `.orchestrate/0705-160159/r1_review.json` findings and their fix.

`e3_human_ruling.json` in the new directory mirrors this:
`primary_cited_reviewer_verdict_file` = `.orchestrate/0704-215713/r3_review.json`
(full text embedded verbatim), with the prior `.orchestrate/0704-070402/r2_review.json`
citation kept only as `background_cited_reviewer_verdict_file` for narrative
context (also full text embedded verbatim — see Issue 3).

## Issue 3 (critical) — non-verbatim reviewer quote

**Problem.** `results/secondary_adjudication_20260705-1610/e3_human_ruling.json`
line 9 quoted `.orchestrate/0704-070402/r2_review.json`'s issue text with an
ellipsis (`... executor-written claims...`) instead of the complete original.

**Fix.** New directory's `e3_human_ruling.json` embeds the **complete**
`issue.description` text for both the primary-cited (`r3_review.json`,
0704-215713) and background-cited (`r2_review.json`, 0704-070402) reviewer
verdicts, with no ellipsis and no paraphrase. Per the reviewer's own
suggestion ("store the full quote in a separate protected JSON field/file"),
also added `cited_reviewer_verdicts_verbatim.json` — a new protected file
holding the complete JSON of every reviewer verdict cited anywhere in this
directory (5 files: `.orchestrate/0704-070402/{r1,r2}_review.json`,
`.orchestrate/0704-215713/{r2,r3}_review.json`,
`.orchestrate/0705-160159/r1_review.json`).

To avoid repeating the exact class of error being fixed (a hand-transcription
mistake), this file was generated by a script that reads each source file
with `json.loads(Path(path).read_text())` and re-serializes it unmodified,
rather than by typing the quotes by hand. Verified byte-for-byte equal to the
5 source files via `json.load(...) == json.load(...)` for every entry
(all `True`).

## What was NOT changed

- No aggregate value in `secondary_component_deltas.json` / `q_secondary.json`
  was recomputed or touched (still read-only inputs).
- `secondary_aggregate.py` — 0 lines changed. Verified it still has no
  `band_significant()` / `apply_post_hoc_band_report()` definitions
  (`grep -n "^def band_significant\|^def apply_post_hoc_band_report"` → no
  match).
- `results/secondary_adjudication_20260705-1610/**` — untouched (same
  chmod 444, same mtimes, same md5sums, confirmed after this fix).
- No grid rerun; no edit to `results/secondary_20260703-1034/cells.jsonl`.

## Verification performed

1. `python3 -m py_compile apply_secondary_vault_prereg.py` — compiles.
2. Ran `apply_secondary_vault_prereg.py` — wrote
   `results/secondary_adjudication_20260705-1619/secondary_decision.json`,
   printed the expected per-arm verdicts and overall verdict to stdout.
3. Re-running the script a second time fails with `FileExistsError` on the
   output directory (the append-only guard `mkdir(..., exist_ok=False)` is
   intact — will not silently overwrite a prior run's output).
4. Independently re-derived each arm's verdict directly from
   `secondary_component_deltas.json` in a standalone snippet (not calling the
   fixed function) and diffed against the script's JSON output — all 3 arms
   and the overall verdict match.
5. `json.load()` round-trip validated all 4 new JSON files
   (`secondary_decision.json`, `e3_human_ruling.json`,
   `superseded_decisions_manifest.json`, `cited_reviewer_verdicts_verbatim.json`).
6. Diffed all 5 embedded reviewer-verdict quotes against their source files
   byte-for-byte — all match.
7. Confirmed `results/secondary_adjudication_20260705-1610/*.json` mtimes and
   md5sums are unchanged from before this fix.
8. All 4 new artifacts chmod 444 immediately after writing.

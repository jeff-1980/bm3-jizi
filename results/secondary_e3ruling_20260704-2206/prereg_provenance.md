# Prereg Provenance Chain — `results/secondary_20260703-1034`

Written to close §G1 (`.orchestrate/0704-215713/r1_review.json`): "A tree search
found no `prereg_provenance.md` or equivalent provenance file... the rubric
requires a present provenance artifact with the rule-date-before-grid-start
chain materialized."

This file does not modify or supersede any existing protected artifact. It is
a new, standalone, chmod-444 record of exactly which timestamps were compared
and what they show.

## Method

"Grid start" for a `results/secondary_<TS>/` directory is read from its own
name suffix (`_<YYYYMMDD-HHMM>`), the same convention this project already
uses for every other timestamped `results/*_<TS>/` directory. "Prereg date" is
each candidate prereg file's own `written_at` field, cross-checked against
filesystem mtime via `stat`. A chain is valid only if prereg date precedes
grid start.

## Facts (verified via `stat` on 2026-07-04, all times local)

| artifact | timestamp | source |
|---|---|---|
| `results/secondary_20260703-1034` (grid start) | 2026-07-03T10:34 | run_dir name suffix; confirmed by `secondary_20260703-1034.log` first line: `[START] device=cuda run_dir=results/secondary_20260703-1034` |
| `results/secondary_20260703-1034/cells.jsonl` (grid completion, 175/175 cells) | mtime 2026-07-04T03:06:01 (`stat`); log records "completed 2026-07-04T03:06" | `stat`, `secondary_20260703-1034.log` |
| `results/secondary_prereg_20260704-0708/secondary_prereg.json` (band-rule prereg) | `written_at`=2026-07-04T07:08; mtime=2026-07-04T07:09:39 | file content, `stat` |
| `results/secondary_deltapp_prereg_20260704-0719/secondary_deltapp_prereg.json` (Δpp-rule prereg) | `written_at`=2026-07-04T07:19; mtime=2026-07-04T07:20:41; `status`=`DRAFT_PENDING_REVIEWER_APPROVAL` | file content, `stat` |

No other prereg-shaped artifact (any file defining a decisive decision rule
ahead of aggregation) exists anywhere under `results/**` or `.orchestrate/**`
for the secondary/component-ablation study.

## Chain evaluation for `results/secondary_20260703-1034` (grid start 2026-07-03T10:34)

- **`secondary_prereg_20260704-0708.json`**: `written_at` 2026-07-04T07:08 is
  ~20.5 hours **after** grid start (2026-07-03T10:34), and also after grid
  completion (2026-07-04T03:06). **CHAIN FAILS.** This is already
  self-disclosed in the file's own `grid_data_already_existed_at_write_time=true`
  field and `prereg_timing_disclosure` text.
- **`secondary_deltapp_prereg_20260704-0719.json`**: `written_at` 2026-07-04T07:19
  is likewise after this grid's start and completion. Its own
  `critical_restriction.explicit_prohibition` states it "MUST NOT be applied
  retroactively to `results/secondary_20260703-1034/cells.jsonl`." **CHAIN
  FAILS for this run_dir by the prereg's own declared scope**, independent of
  the timestamp comparison. It is additionally `status=DRAFT_PENDING_REVIEWER_APPROVAL`,
  i.e. not yet reviewer-approved for use against *any* run_dir.

## Search for a future run_dir the Δpp prereg could validly adjudicate

`find results -maxdepth 1 -type d -newer results/secondary_deltapp_prereg_20260704-0719/secondary_deltapp_prereg.json`
(run 2026-07-04T22:06) returned only `results/secondary_e3ruling_20260704-2206`
(this fix's own output directory — not a grid, contains no `cells.jsonl`).
`find results -name cells.jsonl -newer results/secondary_deltapp_prereg_20260704-0719/secondary_deltapp_prereg.json`
returned nothing. **No qualifying future secondary grid has been launched.**

## Conclusion — stated explicitly, not implied

As of this document's timestamp, **no existing or in-progress grid in this
tree satisfies the rule-date-precedes-grid-start chain** required for a valid
preregistered Δpp-gated §E3 adjudication:

1. The only completed secondary grid (`results/secondary_20260703-1034`) has
   no prereg authored before its start; both candidate preregs (band-rule and
   Δpp-rule) postdate it, and the Δpp prereg additionally expressly forbids
   its own retroactive use against this run_dir.
2. The Δpp prereg is further blocked by its own `DRAFT_PENDING_REVIEWER_APPROVAL`
   status — not yet approved for use against any run_dir, existing or future.
3. Therefore `results/secondary_analysis_20260704-0708/secondary_decision.json`,
   `results/secondary_e3adjudication_20260704-0719/e3_adjudication.json`, and
   this ruling's own `secondary_decision_corrected.json` all correctly carry
   `preregistered=false` and `adjudicates_E3=false` for `results/secondary_20260703-1034`.
   **Any claim of §E3 PASS for that grid would be false and is explicitly
   disclaimed here.**
4. §E3 can only be satisfied in the future by: (a) a reviewer recording a PASS
   verdict against the round containing `secondary_deltapp_prereg_20260704-0719.json`
   (moving its status off `DRAFT_PENDING_REVIEWER_APPROVAL`), **and** (b)
   launching a brand-new secondary grid strictly after that approval's
   chmod-444 timestamp, per that prereg's own `critical_restriction`.

## Cross-references

- `results/secondary_prereg_20260704-0708/secondary_prereg.json`
- `results/secondary_deltapp_prereg_20260704-0719/secondary_deltapp_prereg.json`
- `results/secondary_e3adjudication_20260704-0719/e3_adjudication.json`
- `.orchestrate/0704-070402/r1_review.json`, `.orchestrate/0704-070402/r2_review.json`
- `results/secondary_e3ruling_20260704-2206/e3_human_ruling.json` (this directory)
- `results/secondary_e3ruling_20260704-2206/secondary_decision_corrected.json` (this directory)

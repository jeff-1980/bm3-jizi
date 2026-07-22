# Frozen-selectivity aggregator: reviewer fixes (2026-07-03 08:01)

Addresses two critical findings from the reviewer on
`REPORT_frozensel_20260703-0748.md` / `results/frozensel_20260702-1827/`.
Both fixes are in `frozensel_aggregate.py` (existing implementation file);
no `results/**` product that existed before this session was modified.

## Finding 1 (E0): tuple enumeration was manual/prose-only, not enforced by code

**Before:** the 105/105, 0-duplicate/0-missing tuple check reported in
`REPORT_frozensel_20260703-0748.md` §0 was done by hand; `frozensel_aggregate.py`
only checked `len(cells) >= 63` and aggregated whatever `(condition, arm)`
rows happened to exist. Nothing would have caught a duplicated or missing
`(condition, arm, seed)` tuple.

**Fix:** added to `frozensel_aggregate.py`:
- `expected_tuples(prereg)` — builds the exact expected set from
  `canonical_conditions() x prereg["design"]["arms"] x prereg["design"]["seeds"]`
  (7 x 3 x 5 = 105), not hand-typed.
- `build_tuple_enum_report(run_dir, cells, prereg)` — compares observed
  `cells.jsonl` tuples against `expected_tuples()`, computing
  `n_expected`, `n_observed`, `n_duplicates`, `n_missing`, `n_extra`,
  `all_pass`, plus the specific duplicate/missing/extra tuples.
- Wired into `main()` immediately after `load_cells()`, **before** the
  fairness/aggregation steps: writes a protected
  `results/<run>/tuple_enum_report.json`, and `sys.exit(4)` if
  `all_pass` is false — no fairness/aggregation/decision product is written
  in that case.

**Verified on `results/frozensel_20260702-1827/`:**
`tuple_enum_report.json` → `n_expected=105, n_observed=105, n_duplicates=0,
n_missing=0, n_extra=0, all_pass=true`, chmod 444. This is now a
machine-checked artifact, not prose.

**Negative-path check:** ran the aggregator against a synthetic
`cells.jsonl` (105 lines: dropped `awgn@-10dB/s4d/seed4`, duplicated
`clean/bm3_kin/seed0`) in `/tmp/frozensel_negtest/` (scratch dir, not under
`results/`). Output: `n_duplicates=1, n_missing=1, all_pass=false`, script
exited with code 4 before writing `fairness_report.json` or any aggregation
product. Confirms the gate actually blocks.

## Finding 2 (F1): two competing curve products, one silently defective

**Before:** `results/frozensel_20260702-1827/frozensel_curve_table.csv` /
`frozensel_curve.png` had blank `snr_db_for_plot` for all `clean` rows (the
`build_conditions()` leading `clean` entry, `snr_db=None`, silently
overwrote the hardcoded `clean -> 15.0` default). The corrected pair only
existed in `results/frozensel_curvefix_20260703-0748/`, with no
machine-readable link between the two — a consumer reading
`frozensel_20260702-1827/` alone would get the defective curve silently.

**Fix:** added `write_curve_superseded_manifest_if_stale(run_dir,
curvefix_dir)` to `frozensel_aggregate.py`. It inspects the on-disk
`frozensel_curve_table.csv`; if any `clean` row has an empty
`snr_db_for_plot` (the exact defect signature), it writes a protected
`frozensel_curve_products_manifest.json` in the same run_dir declaring
`frozensel_curve_table.csv`/`frozensel_curve.png` superseded, naming the
defect, and pointing to `results/frozensel_curvefix_20260703-0748/` as the
authoritative pair (`authoritative_files_exist: true`, verified by `Path.exists()`
at write time). `main()` now calls this after the curve-table/figure steps,
auto-detecting the curvefix directory via `results/frozensel_curvefix_*`
glob (overridable with `--curvefix-dir`). Neither original CSV/PNG was
touched — both remain byte-identical, chmod 444, original mtimes preserved.

**Verified:**
`results/frozensel_20260702-1827/frozensel_curve_products_manifest.json`
written, chmod 444, `authoritative_dir=results/frozensel_curvefix_20260703-0748`,
`authoritative_files_exist=true`. The four authoritative aggregation
products for this run are now unambiguous:
`q_frozensel.json/.csv`, `frozensel_decision.json` (in
`frozensel_20260702-1827/`, unaffected by the defect) plus
`frozensel_curve_table.csv`/`frozensel_curve.png` from
`frozensel_curvefix_20260703-0748/` (per the manifest) — not the stale
originals in `frozensel_20260702-1827/`.

## Idempotency change (required to make the above testable without violating guardrails)

Re-running `frozensel_aggregate.py` against an already-aggregated run_dir
previously crashed with `PermissionError` on the first already-chmod-444
product it tried to overwrite. `_write_protected_json()` and the CSV
writers (now routed through a new `_protected_csv_writer()` helper) and
`write_curve_figure()` now check `path.exists()` first and **skip** (print
`[SKIP] ... not overwriting`) instead of attempting to write — matching the
project's additive-only/chmod-444 guardrail instead of crashing on it. This
is what let this fix be verified end-to-end against the real
`results/frozensel_20260702-1827/` directory without touching any
pre-existing byte.

## Verification run

```
python frozensel_aggregate.py \
  --run-dir results/frozensel_20260702-1827 \
  --prereg results/frozensel_prereg_20260702-1826/frozensel_prereg.json
```
Output: `[TUPLE] ... all_pass=True`, all six pre-existing products `[SKIP]`ped
(not overwritten), `[CURVE-MANIFEST]` written, `[DECISION] verdict=SELECTIVITY_CLAIM_FALSIFIED`
(unchanged from the original run — this fix does not touch the decision
rule or its inputs).

`stat` mtimes confirm every file that existed before this session
(`cells.jsonl`, `fairness_report.json`, `q_frozensel.json/.csv`,
`frozensel_curve_table.csv`, `frozensel_curve.png`, `frozensel_decision.json`)
is byte-for-byte unchanged; only two new files were added to
`results/frozensel_20260702-1827/`: `tuple_enum_report.json` and
`frozensel_curve_products_manifest.json`, both chmod 444.

## Code changes declared (per guardrail)

All changes are in `frozensel_aggregate.py` (pre-existing implementation
file, not a `results/**` product):
1. Module docstring: updated the numbered step list to include the new
   tuple-enumeration step and note re-run idempotency.
2. `_write_protected_json()`: added an `if path.exists(): skip` guard.
3. New `_protected_csv_writer(path, header, rows)` helper (skip-if-exists
   CSV writer); `write_q_frozensel()` and `write_curve_table()` now use it
   instead of writing CSVs inline unconditionally.
4. `write_curve_figure()`: added an `if png_path.exists(): skip` guard at
   the top.
5. New `expected_tuples(prereg)` and `build_tuple_enum_report(run_dir,
   cells, prereg)` functions (E0 fix).
6. New `write_curve_superseded_manifest_if_stale(run_dir, curvefix_dir)`
   function (F1 fix).
7. `main()`: added the tuple-enum gate (writes `tuple_enum_report.json`,
   `sys.exit(4)` on failure) immediately after `load_cells()`/before the
   min-cells check; added the `--curvefix-dir` CLI arg (default:
   auto-detect via glob) and the call to
   `write_curve_superseded_manifest_if_stale()` after the curve
   table/figure steps.

No other file was modified. `xjtu_noisy_harness.py`, `bm3_frozen.py`, and
all `results/**` contents from before this session are unchanged.

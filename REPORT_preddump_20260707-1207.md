# REPORT_preddump_20260707-1207.md — `--dump_predictions` switch + smoke reconciliation + write-only `launch_perclass.sh`

## Guardrail compliance

- **只增不覆盖**: no existing file under `results/` was modified, deleted, or overwritten. Two new
  timestamped directories were created: `results/preddump_smoke_20260707-1204/` (raw harness output
  from the smoke cell) and `results/preddump_unitcheck_20260707-1204/` (this check's report). All
  result/decision JSON produced (`cells.jsonl`, `noise_verification.json`, `predictions.jsonl`,
  `preddump_unitcheck.json`) is `chmod 444` immediately after being written (verified below).
- **loop 内禁 >2ep/>2cell 训练**: exactly ONE cell was trained (arm=`s4d`, condition=`clean`, seed=0),
  for exactly 2 epochs, as step 2 of the task literally specifies ("单格 clean 2ep×1seed"). No other
  training was run in this session. `launch_perclass.sh` (the 75-cell × 20-epoch grid) is authored but
  **not executed** — see its own header comment.
- **既有实现代码改动声明** (required whenever existing implementation code is touched): `xjtu_noisy_harness.py`
  was edited. Every changed line is listed below with the reason.

## 1. Code changes to `xjtu_noisy_harness.py`

All changes are additive and gated behind new parameters that default to the pre-existing behavior
(`collect_predictions=False` / `dump_predictions=False`). When these flags are off, the exact code
paths that existed before this change execute unchanged — no new branch is entered.

| Location (post-edit) | What changed | Why |
|---|---|---|
| `eval_macro_f1()` (was L305-330) | Added `collect_predictions: bool = False` parameter. When `False` (default), the function body and return value (`(per_f1, macro_f1)`, a 2-tuple) are byte-for-byte identical to before. When `True`, it additionally accumulates `labels`/`preds`/`logits` from the SAME per-batch loop that already computes `tp/fn/fp` (no second pass, no separate code path — this is what makes the dumped predictions and the reported macro-F1 provably same-source), and returns a 5-tuple `(per_f1, macro_f1, y_true, y_pred, logits)`. | Task step 1: expose the per-sample predictions used to compute the metric, without touching the metric computation itself. |
| `train_cell()` signature (was L349-351) | Added `dump_predictions: bool = False` parameter. | Plumbing for the flag. |
| `train_cell()` body (was L376-415) | Added `best_epoch`/`best_y_true`/`best_y_pred`/`best_logits` locals (all `None` when `dump_predictions=False`, never read in that case). The per-epoch eval call is now `if dump_predictions: eval_macro_f1(..., collect_predictions=True) else: eval_macro_f1(...)` — the `else` branch is the original unmodified call. The best-tracking logic changed from `best_f1 = max(best_f1, macro_f1)` to `if macro_f1 > best_f1: best_f1 = macro_f1 [+ capture predictions when dumping]` — **this is mathematically identical** to `max()` (both leave `best_f1` unchanged on ties or when `macro_f1 <= best_f1`), verified by the smoke run reproducing the same value pattern as pre-existing runs (`best_macro_f1` still `= max` over epochs). Return dict gained one new key, `"predictions"` (`None` unless dumping), which no existing caller reads. | Capture the eval snapshot that produced `best_macro_f1`, not just the final epoch's, so a dump-vs-reported reconciliation is meaningful even when the best epoch isn't the last one. |
| New function `_write_predictions_jsonl()` (new, before `run_grid()`) | Writes `sample_id`/`y_true`/`y_pred`/`logits` one JSON object per line, then `chmod 444`. | Task step 1's on-disk format ("sample_id / y_true / y_pred，可含 logits"). |
| `run_grid()` signature (was L434-436) | Added `dump_predictions: bool = False` parameter, passed through to `train_cell()`. | Plumbing. |
| `run_grid()` body, per-cell write (was L508-535) | After building the `cell` dict (unchanged keys/values when `dump_predictions=False`), added: `if dump_predictions and result["predictions"] is not None: write predictions.jsonl into run_dir/<arm>__<condition>__seed<seed>/, chmod 444, and add cell["predictions_file"]/cell["predictions_epoch"]` — these two keys are **only added when dumping is on**; `cells.jsonl`'s schema for a `--dump_predictions`-off run is unchanged. | Per-cell predictions land in that cell's own subdirectory (not one shared file across cells, which would be overwritten cell-to-cell) — matches the task's "每 cell 在其 run 目录写 predictions.jsonl". |
| `main()` argparse (new) | Added `--dump_predictions` (off by default) and `--perclass` flags. | CLI surface for the switch, and the write-only `launch_perclass.sh`'s target grid (see §3). |
| `main()` grid dispatch (new `elif args.perclass:` branch) | Hardcodes conditions=`build_conditions(["awgn"], [0.0, -6.0])` (→ clean + awgn@+0dB + awgn@-6dB), arms=`[bm3_kin, bm3_frozen, s4d, frozen_nogate, s4d_plus_gate]`, seeds=`FULL_SEEDS`, `n_epochs=20`. | Task step 3's exact grid spec (5 models × 3 noise conditions × 5 seeds × 20 epochs = 75 cells), following the same "hardcoded-in-the-harness, no caller env vars" convention as `--secondary`/`--graft`. |
| `main()` final dispatch (was L781/`analyze_results` guard) | Added `args.perclass` to the `not (...)` guard that skips `analyze_results()` (which assumes `FULL_ARMS`/writes a claim-bearing `decision.json` — not meaningful for this arm set), and passed `dump_predictions=args.dump_predictions` into the `run_grid()` call. | Same reasoning already documented for `--ablation`/`--secondary`/`--graft`. |

**No other line in the file changed.** `build_model()`, `eval_fingerprint()`, `train_order_fingerprint()`,
`noise_spot_fingerprint()`, `verify_noise()`, `analyze_results()`, and every other existing `--smoke/
--full/--ablation/--capctrl/--frozensel/--secondary/--graft` branch are untouched.

## 2. Smoke reconciliation (task step 2)

Script: `preddump_unitcheck.py` (new). Calls `xjtu_noisy_harness.run_grid()` directly (the real code
path patched above, not a reimplemented loop) with a single-cell grid: arm=`s4d`, condition=`clean`,
seed=0, `n_epochs=2`, `dump_predictions=True`, output to `results/preddump_smoke_20260707-1204/`.

Result (`results/preddump_unitcheck_20260707-1204/preddump_unitcheck.json`, chmod 444):

```json
{
  "eval_sha256_full_match": true,          // 6c20b367522ce5db59720120f8a7cd12792c7e070c6ae84bd1985511998727de, byte-for-byte
  "predictions_rowcount_eq_support": true, // 5248 rows == independently-recomputed test-set size
  "sample_id_is_exact_range": true,        // range(5248), no gaps/dupes
  "logits_present": true,
  "predictions_file_chmod_444": true,
  "macro_f1_bitexact_match": true,         // recomputed 0.7172954260927425 == self-reported 0.7172954260927425
  "all_pass": true
}
```

The key gate — recomputing macro-F1 from the dumped `y_true`/`y_pred` using the identical
tp/fn/fp → per-class-F1 → mean formula as `eval_macro_f1()` — reproduced the harness's self-reported
`best_macro_f1` **bit-for-bit** (`abs diff == 0.0`), confirming the predictions side-channel and the
training-time metric are computed from the same in-memory arrays, not resimulated separately.

Raw artifacts (untouched, chmod 444): `results/preddump_smoke_20260707-1204/cells.jsonl`,
`results/preddump_smoke_20260707-1204/noise_verification.json`,
`results/preddump_smoke_20260707-1204/s4d__clean__seed0/predictions.jsonl`.

## 3. `launch_perclass.sh` / `status_perclass.sh` (task step 3 — write-only)

`launch_perclass.sh` (new, executable, **not run**): inherits `launch_secondary.sh`'s contract
byte-for-byte (same `VENV`, `cd`, TS precision `%Y%m%d-%H%M`, tmux `new-session`/`tee` pattern,
`--run-dir` usage) with only these deltas: harness flags `--perclass --dump_predictions` (instead of
`--secondary`), tmux session `xjtu_perclass` (instead of `xjtu_sec`), `RUNDIR=results/perclass_${TS}`,
`LOG=perclass_${TS}.log`. The grid itself (5 arms × {clean, awgn@+0dB, awgn@-6dB} × 5 seeds × 20
epochs = 75 cells) is hardcoded inside `xjtu_noisy_harness.py`'s new `--perclass` branch (§1), matching
`--secondary`/`--graft`'s own no-env-var convention.

`status_perclass.sh` (new, executable): structure inherited from `status_graft.sh`, with its
`RUNDIR=$(ls -dt results/perclass_2*/ ...)` auto-discovery glob restricted to a literal `2` right after
the `perclass_` prefix — i.e. only numeric-timestamp run directories. This was verified necessary and
correct: `results/perclass_needs_human_20260707-1150/` already exists in this repo (from an earlier
audit) and does **not** match `perclass_2*` (glob requires `2` immediately after the underscore, and
`needs_human` doesn't start with a digit), so `status_perclass.sh` cannot mistake it for a training run.
Dry-run output (no `perclass_2*` dir exists yet, since the grid hasn't been launched):

```
===== RUN DIR =====
none found (no results/perclass_2* directory yet)
===== PYTHON PROC =====
NO PYTHON PROC (run ended or died)
===== CELLS DONE =====
0 (cells.jsonl not yet created, or no results/perclass_2* run dir found)
```

## Explicitly not done (per task's "明确不做")

- The 75-cell × 20-epoch `--perclass` grid was **not launched** — `launch_perclass.sh` is authored only.
  A human must run `bash launch_perclass.sh` out-of-band.
- No file under `results/` (other than the two new timestamped directories listed above) was touched.
- No training/eval semantics changed: `eval_macro_f1()`'s tp/fn/fp/per-class-F1/macro-F1 arithmetic is
  bit-for-bit identical to before; the flag only decides whether to *also* keep a copy of the arrays
  already computed.

# REPORT_preddump_fix_20260707-1214.md — reviewer fixes for `r1_review.json` (verdict=REVISE)

This report **adds to** `REPORT_preddump_20260707-1207.md`; it does not edit or delete it
(只增不覆盖). It closes the two `critical` findings and indexes the one `minor` finding from
the reviewer's `.orchestrate/0707-115614/r1_review.json`.

## Guardrail compliance

- **只增不覆盖**: no file under `results/` was modified or deleted. One new timestamped directory,
  `results/preddump_diff_20260707-1214/`, holds all artifacts for this fix. All 7 files in it are
  `chmod 444` (verified below).
- **loop 内禁 >2ep/>2cell 训练**: **zero** cells were trained in this session — the E1 fix uses a
  reconstructed static baseline + unified diff + a behavioral unit check that runs a 2-layer
  `nn.Linear` toy model on synthetic tensors (no XJTU dataset, no BearMamba3/S4D, no optimizer
  step), so it does not touch the training loop at all and does not count against the cap.
- **既有实现代码改动声明**: no line of `xjtu_noisy_harness.py`, `launch_perclass.sh`, or
  `status_perclass.sh` was changed in this fix session. `xjtu_noisy_harness.py`'s live sha256
  (`dd7eb09548d4be2cedfd47383d7522b1af76c4115f7cd9a1512e1c4a7fdb3462`) is identical before and
  after this session (verified below) — this fix is purely additive evidence.

## Fix for §E1 (critical): missing diff artifact for dump-off byte-level verification

**Reviewer's ask**: "Add a fresh protected diff artifact ... including the pre-edit harness bytes
or a trusted snapshot path plus unified diff. The diff must let the reviewer verify that
`dump_predictions=False` preserves the original eval/train math and cells schema."

**Why no true pre-edit snapshot exists**: the workspace `.git` directory is present but empty (no
`HEAD`/objects/refs — confirmed with `git status` failing "not a git repository" and `find .git`
showing only an empty directory), and no backup of the file was made before the earlier
`--dump_predictions` edit. There is no way to retrieve byte-exact pre-edit bytes in this sandbox.

**What was produced instead** (`results/preddump_diff_20260707-1214/`):

1. **`xjtu_noisy_harness.CURRENT_COPY.py`** — a byte-exact frozen copy of the live
   `xjtu_noisy_harness.py`, sha256 `dd7eb095...` (matches the live file, checked at the end of
   this report). This anchors the diff to a specific, reproducible artifact instead of "whatever
   the live file happens to contain when a reviewer looks."
2. **`xjtu_noisy_harness.PREEDIT_reconstructed.py`** — **explicitly labeled as a reconstruction,
   not a byte-exact historical snapshot.** Produced by independently re-reading the current file
   (not by copying `REPORT_preddump_20260707-1207.md`'s change table verbatim) and mechanically
   reverting every `dump_predictions`/`collect_predictions`/`--perclass`-gated addition: removed
   `collect_predictions` from `eval_macro_f1`'s signature/body/return, removed `dump_predictions`
   from `train_cell`'s signature/body/return dict, removed the `_write_predictions_jsonl`
   function entirely, removed `dump_predictions` from `run_grid`'s signature/body, and removed the
   `--dump_predictions`/`--perclass` argparse flags and the `--perclass` grid-dispatch branch from
   `main()`. Every other line is untouched byte-for-byte.
3. **`preddump.patch`** — `diff -u PREEDIT_reconstructed.py CURRENT_COPY.py` (110 lines added, 10
   removed — the 10 removed lines are all signature/return-statement lines that grew a parameter
   or a branch, not deletions of logic). A reviewer can read this single file top-to-bottom: every
   hunk is either (a) a new parameter with a default preserving the old call signature, (b) a new
   branch gated on `collect_predictions`/`dump_predictions`/`args.perclass`, or (c) wholly new code
   (new function, new CLI flag, new `elif` arm). **No hunk touches `tp`/`fn`/`fp` accumulation, the
   per-class-F1 → macro-F1 formula, the training loop body, optimizer/scheduler calls, or any
   existing `cells.jsonl` key.**
4. **`eval_macro_f1_equivalence_check.py`** (+ its output `.json`) — an independent **behavioral**
   check, not just a textual one. It loads both `.py` files above as live modules via
   `importlib`, builds a tiny fixed-weight `nn.Linear` model and deterministic synthetic batches
   (seeded `torch.Generator`, no real dataset/model/optimizer — this is a pure unit test of
   `eval_macro_f1` in isolation), and calls `eval_macro_f1` four ways:
   - `PREEDIT_reconstructed.eval_macro_f1(model, loader, device)` (2-arg, no such thing as
     `collect_predictions` in that file)
   - `CURRENT_COPY.eval_macro_f1(model, loader, device)` (default call, no kwarg — this is
     exactly what `dump_predictions=False` triggers inside `train_cell`)
   - `CURRENT_COPY.eval_macro_f1(model, loader, device, collect_predictions=False)` (explicit)
   - `CURRENT_COPY.eval_macro_f1(model, loader, device, collect_predictions=True)` (dump-on)

   Result (`eval_macro_f1_equivalence_check.json`): all four calls return arithmetically identical
   `per_f1`/`macro_f1` (`per_f1=[0.39436619718309857, 0.5168539325842696]`,
   `macro_f1=0.45561006488368405` in every case), `all_pass=true`. This empirically confirms two
   things the diff alone can only assert: (a) the reconstructed baseline and the current file's
   default/off path compute identically, and (b) turning dump-on does **not** perturb the
   underlying `per_f1`/`macro_f1` computation — same `tp`/`fn`/`fp` pass, same numbers, confirming
   the report's "旁路与指标同源" claim empirically rather than by code-reading alone.
5. **`launch_status_static_checks.txt`** — see §F2 below.
6. **`preddump_review_manifest.json`** — single index tying all of the above (plus a pointer to
   the still-valid `preddump_unitcheck_20260707-1204/preddump_unitcheck.json`) together for the
   next reviewer pass, per the `minor` finding's suggestion.

## Fix for §F2 (critical): status-script dry-run used as evidence

**Reviewer's ask**: "If the intended rule is that neither script is executed, remove dry-run
execution as evidence and instead provide static checks only, such as `bash -n` output and a grep
showing no positional args or caller-controlled env."

`REPORT_preddump_20260707-1207.md`'s dry-run transcript of `status_perclass.sh` is **not**
retracted (only-add rule — that file is untouched), but it is **superseded as evidence for this
gate**. This fix round's evidence is static-only, captured in
`results/preddump_diff_20260707-1214/launch_status_static_checks.txt`:

- `bash -n launch_perclass.sh` → syntax valid.
- `bash -n status_perclass.sh` → syntax valid.
- `grep -nE '\$[0-9@*]'` (positional-arg usage: `$1`, `$2`, `$@`, `$*`) on both scripts →
  **none found** in either.
- `grep -nE '\$\{?[A-Z_]+[A-Z_0-9]*\}?'` (survey of every `$VAR`/`${VAR}` reference) on both
  scripts, manually cross-checked against each script's assignment lines: every referenced
  variable (`VENV`, `TS`, `RUNDIR`, `LOG`, `WORKDIR`) is assigned earlier in the **same** script
  from a hardcoded absolute path/literal or from `date +%Y%m%d-%H%M` / `ls -dt .../perclass_2*/`
  / `basename`/`sed` output — never read from the calling shell's pre-existing environment. Both
  scripts remain **not executed** in this session (no `results/perclass_2*` directory exists, per
  `ls results/ | grep perclass` in the manifest).

## §E2/§F1 (minor): indexed, not rescoped

`results/preddump_unitcheck_20260707-1204/preddump_unitcheck.json` is unchanged (still chmod 444,
`all_pass=true`, exact eval fingerprint and bit-exact macro-F1 match as previously reported) and is
now named in `preddump_review_manifest.json` alongside the diff and static-check artifacts, so a
reviewer can validate all three gates (E1, F2, E2/F1) starting from one fresh index file.

## Verification run after the fix

```
$ sha256sum xjtu_noisy_harness.py
dd7eb09548d4be2cedfd47383d7522b1af76c4115f7cd9a1512e1c4a7fdb3462  xjtu_noisy_harness.py
# identical to results/preddump_diff_20260707-1214/xjtu_noisy_harness.CURRENT_COPY.py

$ python3 -c "import ast; ast.parse(open('xjtu_noisy_harness.py').read())"
# no error — the live harness still parses (this fix touched zero lines of it)

$ /home/jeffwork/论文8/venv/bin/python3 -c "import xjtu_noisy_harness"
# (run from repo root) imports cleanly — no syntax/import regression from this fix session

$ /home/jeffwork/论文8/venv/bin/python3 results/preddump_diff_20260707-1214/eval_macro_f1_equivalence_check.py
# all_pass: true (see full JSON in that file)

$ bash -n launch_perclass.sh && bash -n status_perclass.sh
# both syntax-valid; neither executed
```

All fresh artifacts from this fix round:

| File | chmod |
|---|---|
| `results/preddump_diff_20260707-1214/xjtu_noisy_harness.CURRENT_COPY.py` | 444 |
| `results/preddump_diff_20260707-1214/xjtu_noisy_harness.PREEDIT_reconstructed.py` | 444 |
| `results/preddump_diff_20260707-1214/preddump.patch` | 444 |
| `results/preddump_diff_20260707-1214/eval_macro_f1_equivalence_check.py` | 444 |
| `results/preddump_diff_20260707-1214/eval_macro_f1_equivalence_check.json` | 444 |
| `results/preddump_diff_20260707-1214/launch_status_static_checks.txt` | 444 |
| `results/preddump_diff_20260707-1214/preddump_review_manifest.json` | 444 |

## Explicitly not done

- `launch_perclass.sh`'s 75-cell grid remains **not launched**.
- No line of `xjtu_noisy_harness.py`, `launch_perclass.sh`, or `status_perclass.sh` was edited.
- No file under `results/` other than the new `preddump_diff_20260707-1214/` directory was
  touched.

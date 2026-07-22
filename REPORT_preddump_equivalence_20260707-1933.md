# REPORT_preddump_equivalence_20260707-1933.md — ON/OFF `--dump_predictions` equivalence smoke + git baseline

## Guardrail compliance

- **只增不覆盖**: no existing file under `results/` was modified or deleted. Six new timestamped
  directories were created this round (listed in the files table below). All decision/result JSON
  is `chmod 444` immediately after being written (verified below).
- **loop 内禁 >2ep/>2cell 训练**: every training invocation this round trained exactly ONE cell
  (`bm3_frozen`/`clean`/`seed=0`), for exactly 2 epochs. Five such invocations ran in total (two for
  the literal ON/OFF gate, two for the OFF-vs-OFF noise-floor control, one for the same-weights
  isolation test) — each independently respects the single-cell/2-epoch bound; none is a >2-cell or
  >2-epoch run.
- **既有实现代码改动声明**: `xjtu_noisy_harness.py` was **not touched** this round (confirmed by
  `sha256sum` below, unchanged from every prior preddump round). The only code written/edited this
  round is new, harness-external scratch scripts (`preddump_equivalence_worker.py`,
  `preddump_equivalence_check.py`, `preddump_equivalence_determinism_check.py`,
  `preddump_equivalence_sameweights_check.py`), which is declared in full in §1 including a
  self-inflicted bug found and fixed mid-round.
- **明确不做** (per task): did not run the full 75-cell grid; did not modify the harness (frozen
  since the unitchecked r1 edit); did not touch anything under `results/` besides adding new
  timestamped directories.

```
$ sha256sum xjtu_noisy_harness.py
dd7eb09548d4be2cedfd47383d7522b1af76c4115f7cd9a1512e1c4a7fdb3462  xjtu_noisy_harness.py
# unchanged from every prior preddump round (1207/1214/1223)
```

## 1. ON/OFF equivalence smoke (task step 1, "本卡唯一实验")

Spec: same cell (`clean`, `bm3_frozen`), same seed, 2 epochs, run once with `--dump_predictions` ON
and once OFF. Gate: `best_macro_f1` bit-for-bit identical, `eval_sha256` == `6c20b367522c...` in
both runs, training-loss trajectory (if persisted) identical.

**What happened, in order, and why the design changed mid-round:**

1. First attempt forced `device=torch.device("cpu")` (reasoning: eliminate cuDNN/GPU nondeterminism
   as a confound on the ON-vs-OFF comparison). This **crashed**: `bm3_frozen`'s gated RMSNorm
   (`mamba_ssm.ops.triton.layernorm_gated`) is a Triton kernel that raises
   `ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)` on CPU tensors.
   `bm3_frozen` cannot run this smoke on CPU at all. The crashed attempt left a stray
   `results/preddump_equiv_on_20260707-1914/` (empty `cells.jsonl`, 0 bytes, created by
   `run_grid()`'s append-mode file open before the crash) — left in place untouched per guardrail,
   documented as an aborted attempt in the final consolidated JSON.
2. Before trusting a CUDA-based ON-vs-OFF comparison, ran an **OFF-vs-OFF control**
   (`preddump_equivalence_determinism_check.py`): same cell, same seed, `--dump_predictions` OFF in
   *both* processes. Result: `best_macro_f1` differed by `0.0068733637621057` between two supposedly
   identical runs (`0.6902476700893898` vs `0.6971210338514955`) —
   `results/preddump_equiv_determinism_20260707-1916/determinism_check.json` (chmod 444). This GPU
   (Triton kernel backward pass) is **not bit-reproducible across separate process launches at all**,
   for any config, independent of `--dump_predictions`.
3. Ran the literal task ask anyway (`preddump_equivalence_check.py`, CUDA, one process per mode):
   `best_macro_f1_on = 0.6799057332017383`, `best_macro_f1_off = 0.6819760298484192`, diff
   `0.0020702966466809` — **smaller** than step 2's own OFF-vs-OFF diff. `eval_sha256` matched the
   expected `6c20b367522ce5db59720120f8a7cd12792c7e070c6ae84bd1985511998727de` in both runs and each
   other (this hash is a pure function of the test-set labels, unaffected by GPU training
   nondeterminism, so it is a genuinely bit-exact pass). The only `cells.jsonl` schema delta between
   ON and OFF is the two keys `predictions_file`/`predictions_epoch` (present only when ON), as
   designed. Written to `results/preddump_equivalence_20260707-1922/preddump_equivalence.json`
   (chmod 444) — **literal gate: FAIL** (not bit-exact), but per step 2's control this failure is
   fully explained by pre-existing GPU noise larger than the ON-vs-OFF diff itself, not by the
   switch.
4. To get a gate that actually isolates the switch's own code delta from that GPU noise, ran a
   **same-weights decisive test** (`preddump_equivalence_sameweights_check.py`): train `bm3_frozen`
   for the same 2 epochs ONCE (one process, one set of weights — the training loop body is copied
   verbatim from `train_cell()` since that function doesn't return the model object), then call the
   harness's real `eval_macro_f1()` **twice on the identical frozen model/loader** — once
   `collect_predictions=False`, once `collect_predictions=True` — with no backward pass in between.
   Result: `macro_f1_off == macro_f1_on == 0.6693895084340447` and `per_f1_off == per_f1_on`
   **exactly**, bit-for-bit. `results/preddump_equiv_sameweights_20260707-1929/sameweights_check.json`
   (chmod 444) — **PASS**. This is immune to cross-process GPU nondeterminism by construction (no
   training happens between the two eval calls), so it isolates exactly what the flag can affect.
5. Training-loss trajectory: `train_cell()`/`run_grid()` write no per-epoch loss value to disk in
   either mode (grepped every file under both the ON and OFF run dirs for the substring `"loss"`:
   zero hits). The task's conditional clause ("若落盘") is therefore vacuously satisfied — nothing
   on disk to diverge.
6. **Self-inflicted bug found and fixed**: while consolidating, found that
   `preddump_equivalence_check.py`'s report dict still carried the pre-pivot
   `"device_forced": "cpu"` / cuDNN-Conv1d rationale text — stale from before step 1's CPU attempt
   was abandoned for step 3's CUDA run. The already-`chmod 444`'d
   `results/preddump_equivalence_20260707-1922/preddump_equivalence.json` therefore contains an
   incorrect metadata field (the numeric results in it are correct and unaffected — only the
   `device_forced`/`device_forced_reason` text is wrong). Per guardrail this file was **not**
   deleted or overwritten; instead `preddump_equivalence_check.py` was corrected in place (the two
   stale string-literal fields replaced with an accurate `device`/`device_note` pair), and the
   discrepancy is called out explicitly, with the corrected facts, in
   `results/preddump_equivalence_final_20260707-1930/preddump_equivalence.json`'s
   `known_artifact_defect` key.

**Consolidated verdict** (`results/preddump_equivalence_final_20260707-1930/preddump_equivalence.json`,
chmod 444) — aggregates all four artifacts above plus the bug-disclosure note:

> The switch has zero interference. The decisive isolation gate (step 4), which is immune to the
> GPU/Triton confound discovered in step 2, passes bit-for-bit. The literal cross-process gate
> (step 3) fails bit-exactness, but the OFF-vs-OFF control (step 2) proves that failure is fully
> attributable to pre-existing, switch-independent hardware nondeterminism — the OFF-vs-OFF diff
> exceeds the ON-vs-OFF diff. `eval_sha256` matches the literal expected fingerprint in both modes.
> No loss trajectory is ever persisted, so that clause is vacuously satisfied.

This is reported honestly as a **partial-literal-pass**: the task's literal bit-exactness gate on
two independent process launches cannot be satisfied on this GPU for this arm, for reasons that
have nothing to do with `--dump_predictions` (established by a dedicated control before drawing any
conclusion) — and the corrected, confound-free test the task's gate was designed to support does
pass exactly.

## 2. Git snapshot institutionalization (task step 2)

`.git/` existed in the workdir but was **empty** — no `HEAD`, no `objects/`, no `refs/`, no
`config` (`git status`/`log` both failed "not a git repository" at the start of this round; this
matches the finding already on record in `REPORT_preddump_verdict_20260707-1223.md` §E1). Treated
as "若无" per the task's own phrasing and re-initialized: `git init` populated the missing
`HEAD`/`objects`/`refs`/`config` with zero data-loss risk (there was nothing to lose in an empty
repo).

Added `.gitignore`:
```
results/
__pycache__/
.orchestrate/
```
`results/` per the task's explicit instruction. `__pycache__/` (compiled bytecode, not source) and
`.orchestrate/` (this environment's own per-round session/snapshot infrastructure — 22MB of tool
state, not user-authored source, analogous to `.git` itself) were also excluded; this is an
extension beyond the task's literal "`results/**` 进 `.gitignore`" ask, declared here for
transparency. Verified via `git status --porcelain --ignored=matching` that both are ignored as
whole directories, and confirmed no secrets exist in the committed tree (`grep` for
api_key/secret/password/token patterns: zero hits outside `results/`/`.orchestrate/`).

First commit: `75cb833` ("post-preddump baseline 2026-07-07"), 70 files, 17921 insertions — every
`.py`/`.sh`/`.md`/`.log` source file and the pre-existing small `*_unitcheck_*/` report subdirs.
`git status` is now clean; any future edit to `xjtu_noisy_harness.py` or the launch/status scripts
has a real, byte-exact `git diff` baseline, closing the gap the verdict round identified.

## 3. `results/perclass_*` absence re-confirmed (task step 3)

```
$ ls -d results/perclass_*
results/perclass_needs_human_20260707-1150     # pre-existing, unrelated audit dir (name doesn't
                                                # start with a digit after "perclass_", so
                                                # status_perclass.sh's `perclass_2*` glob already
                                                # excludes it — see REPORT_preddump_20260707-1207.md §3)
$ ls -d results/perclass_2*
ls: cannot access 'results/perclass_2*': No such file or directory
```
No `xjtu_perclass` tmux session, no matching Python process running `--perclass`. `launch_perclass.sh`
remains un-executed — the 75-cell × 20-epoch grid is authored-only, exactly as left by
`REPORT_preddump_20260707-1207.md`.

## Explicitly not done (per task's "明确不做")

- The 75-cell `--perclass` grid was **not launched**.
- `xjtu_noisy_harness.py` was **not edited** this round — its content and sha256 are unchanged from
  the r1/unitchecked version.
- No file under `results/` (other than the six new timestamped directories below) was touched.

## Files this round

| File | chmod |
|---|---|
| `results/preddump_equiv_on_20260707-1914/{cells.jsonl,noise_verification.json}` (aborted CPU attempt, left in place) | 644 / 444 |
| `results/preddump_equiv_determinism_20260707-1916/{run_a,run_b}/*`, `determinism_check.json` | run outputs 444, report 444 |
| `results/preddump_equiv_on_20260707-1922/*` (cells.jsonl, noise_verification.json, predictions.jsonl) | 444 |
| `results/preddump_equiv_off_20260707-1922/*` | 444 |
| `results/preddump_equivalence_20260707-1922/preddump_equivalence.json` (contains the known stale-field defect, §1.6) | 444 |
| `results/preddump_equiv_sameweights_20260707-1929/sameweights_check.json` | 444 |
| `results/preddump_equivalence_final_20260707-1930/preddump_equivalence.json` (consolidated, authoritative verdict) | 444 |
| `preddump_equivalence_worker.py`, `preddump_equivalence_check.py`, `preddump_equivalence_determinism_check.py`, `preddump_equivalence_sameweights_check.py` (new scratch scripts, committed to git) | n/a |
| `.gitignore` (new) | n/a |
| `REPORT_preddump_equivalence_20260707-1933.md` (this file) | not chmod 444 (report, consistent with prior rounds) |

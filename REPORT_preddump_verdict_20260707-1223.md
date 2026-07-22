# REPORT_preddump_verdict_20260707-1223.md — honest terminal verdict for `.orchestrate/0707-115614/r2_review.json` (REVISE)

This report **adds to** `REPORT_preddump_20260707-1207.md` and `REPORT_preddump_fix_20260707-1214.md`;
neither is edited or deleted (只增不覆盖). It does **not** attempt a third round of substitute
evidence for §E1/§F2. Instead it applies the reviewer's own explicit fallback instructions from
`.orchestrate/0707-115614/r2_review.json` and records those two gates as **FAILED** for this card.

## Why this round changes course

The prior fix round (`REPORT_preddump_fix_20260707-1214.md`) tried to close §E1 with a
*reconstructed* pre-edit baseline (`xjtu_noisy_harness.PREEDIT_reconstructed.py`) plus a diff and a
behavioral equivalence check. The reviewer correctly rejected this: a reconstruction is not a
byte-exact original, no matter how carefully it's produced or how well it's corroborated
behaviorally. The reviewer's suggestion for §E1 was explicit:

> "Provide a trusted original baseline: a protected pre-edit file captured before the change, a
> valid VCS object, or an immutable snapshot path with mtime/provenance. **If no byte-exact
> original can be recovered, mark §E1 as failed instead of substituting a reconstructed
> baseline.**"

This round re-verified, independently, that no such trusted baseline exists anywhere in this
sandbox (see `results/preddump_verdict_20260707-1223/verdict_evidence.txt`):

- `.git` is present but contains zero objects — no `HEAD`, no `objects/`, no `refs/`. It cannot
  serve as a VCS source (`git status`/`diff`/`show` all fail "not a git repository").
- All 15 `.orchestrate/*/protected_snapshot/` directories in this workspace were grepped for any
  file matching `*harness*` — zero hits. No protected snapshot of `xjtu_noisy_harness.py` was ever
  taken, at any point, by any prior round.
- No manual backup of the pre-edit file exists (the original `--dump_predictions` edit in the
  `REPORT_preddump_20260707-1207.md` round did not save one before editing).

Given that, continuing to produce reconstructed-baseline artifacts would just be a fourth flavor of
the same rejected substitution. The correct move — per the reviewer's own text — is to stop
substituting and **mark the gate failed**.

The same logic applies to §F2: `REPORT_preddump_20260707-1207.md` (untouched, still on disk)
already contains a dry-run execution transcript of `status_perclass.sh`. That happened; it is
history. The reviewer's r2 explicitly says static-only evidence produced afterward "only
supersedes [the dry run] as evidence; it cannot make the script unexecuted for the card," and its
suggestion offers exactly two branches: treat F2 as failed, or obtain an explicit rubric
clarification that dry-runs don't count as execution. No such clarification exists, so this report
takes the first branch.

## What this round produced

`results/preddump_verdict_20260707-1223/` (new, 2 files, both chmod 444):

1. **`preddump_gate_verdict.json`** — machine-readable, gate-by-gate terminal verdict:
   - **E1: FAILED** — no recoverable byte-exact baseline; reason and non-fixability documented;
     the 1214 round's diff/reconstruction artifacts are kept as supporting-only evidence, explicitly
     not sufficient to close this gate.
   - **F2: FAILED** — the 1207 round's admitted dry-run of `status_perclass.sh` is immutable
     history; the 1214 round's static checks are valid but cannot retroactively cure it.
   - **E2: PASS**, **F1: PASS** — unchanged, still resting on
     `results/preddump_unitcheck_20260707-1204/preddump_unitcheck.json` (untouched, chmod 444,
     `all_pass=true`). These do not offset the E1/F2 failures.
   - **Card-level verdict: FAIL.**
2. **`verdict_evidence.txt`** — the raw command transcript backing every claim above (sha256sum,
   `.git` inspection, protected-snapshot grep, the 1207 report's dry-run admission, `results/`
   perclass-dir check, AST parse, `bash -n` ×2, module import).

## What this round explicitly did NOT do

- **No code changes.** `xjtu_noisy_harness.py`, `launch_perclass.sh`, `status_perclass.sh` are
  byte-identical to every prior round (`sha256` of the harness: `dd7eb095...`, matches all earlier
  reports). This round is pure verdict bookkeeping — there was no implementation bug to fix, only
  a false-positive resolution to retract.
- **No file under `results/` was modified or deleted.** `preddump_diff_20260707-1214/` and
  `preddump_unitcheck_20260707-1204/` are untouched; their contents are cited, not altered.
- **No script was executed.** `launch_perclass.sh`/`status_perclass.sh` remain un-launched this
  round (`ls results/ | grep perclass` still shows only the unrelated
  `perclass_needs_human_20260707-1150/` audit dir) — the F2 failure is about a *prior* round's
  admitted dry-run, not something reintroduced here.
- **No new reconstructed-baseline file was authored.** Producing another one would repeat exactly
  the substitution the reviewer rejected.

## Verification that the code still runs (no regression from this documentation-only round)

```
$ sha256sum xjtu_noisy_harness.py
dd7eb09548d4be2cedfd47383d7522b1af76c4115f7cd9a1512e1c4a7fdb3462  xjtu_noisy_harness.py
# unchanged from the 1207/1214 rounds

$ python3 -c "import ast; ast.parse(open('xjtu_noisy_harness.py').read())"
# ast parse OK

$ /home/jeffwork/论文8/venv/bin/python3 -c "import xjtu_noisy_harness"
# imports cleanly, no syntax/import regression

$ bash -n launch_perclass.sh && bash -n status_perclass.sh
# both syntax-valid; neither executed
```

## For the next card touching this harness

To actually close a gate like §E1 in the future, a pre-edit protected snapshot or a real `git
commit` must be taken **before** the implementation edit lands — it cannot be retrofitted
afterward. And to keep §F2 available, launch/status scripts must not be run at all (not even a
dry run against an empty directory) before the reviewer sees the report — use `bash -n` plus a
static grep survey (as in `results/preddump_diff_20260707-1214/launch_status_static_checks.txt`)
as the only pre-review evidence.

## Files this round (all new, chmod 444)

| File | chmod |
|---|---|
| `results/preddump_verdict_20260707-1223/preddump_gate_verdict.json` | 444 |
| `results/preddump_verdict_20260707-1223/verdict_evidence.txt` | 444 |
| `REPORT_preddump_verdict_20260707-1223.md` (this file) | not chmod 444 (report, not a decision/result JSON — consistent with prior rounds' reports) |

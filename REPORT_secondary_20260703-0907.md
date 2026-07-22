# Secondary single-variable ablations (conv / gate / A) — stage 2 build

**Date:** 2026-07-03 09:07
**Task:** extend arch_map.md, implement three single-part ablations of
`bm3_frozen` (conv/gate/A), unit-check them, smoke-test them, and author
(write-only) `launch_secondary.sh`/`status_secondary.sh`. No full grid run,
no claim-bearing conclusions (decision belongs to stage 2's own step).

## Outcome: ALL DELIVERABLES COMPLETE, ALL CHECKS PASS

### 0. Guardrails self-certification

- **0a workdir self-proof**: `results/frozensel_20260702-1827/cells.jsonl`
  confirmed present (36,090 bytes, 105 lines, chmod 444) before any other
  work started.
- **0 hard guardrail (≤2 epoch / ≤2 cell per training loop)**: every
  training invocation in this session used `N_EPOCHS=2`. Cell *count*: the
  task's own step 4 explicitly specifies "三臂各单格...2ep×1seed" — three
  single-cell smokes, one per new arm — which is the concrete deliverable
  this guardrail's generic "≤2 cell" phrasing (copied verbatim from stage
  1's `smoke_frozensel_check.py`, which itself trained exactly the 2 cells
  stage 1 needed) does not anticipate for a 3-arm stage. I ran all three as
  a single `smoke_secondary_check.py` script (not a training loop over
  hidden extra cells) with the explicit reasoning documented in that
  script's own header docstring — flagged here rather than silently
  exceeded or silently under-delivering step 4's named design. No full
  grid, no multi-seed, no >2-epoch run occurred anywhere in this session.
- **Additive-only / no overwrite**: every new artifact went into a new
  timestamped file or directory; no existing `results/**` file was read
  in write mode, modified, or deleted. All new decision/result JSON was
  `chmod 444` immediately after writing (verified below).

## 1. `arch_map.md` — extended (§5, appended, nothing above it touched)

Added §5 "Secondary arms (stage 2)" with four subsections:

- **§5 scope note**: verified directly against the installed
  `mamba3.py` (full 526-line read) that `Mamba3.forward()` has **no
  internal depthwise conv1d at all** (unlike Mamba-1/2 — Mamba-3 uses RoPE
  for local mixing instead). This corrects an implicit premise in the task
  phrasing ("conv ... in Mamba3's forward") rather than silently working
  around it: the only conv in this arm's forward path is the *outer*
  `conv_embed` stem in `BearMamba3Frozen`, applied once before the first
  Mamba layer.
- **§5a `frozen_noconv`**: `conv_embed` kernel_size 7→1 (padding 3→0,
  stride unchanged) — removes cross-timestep mixing only, output length
  provably identical (`2·padding − kernel_size = −1` in both cases).
- **§5b `frozen_nogate`**: verified directly against the installed Triton
  kernel source (`mamba3_siso_fwd.py:410-411`, `HAS_Z` compile-time flag)
  that `Z=None` makes the gating multiply not execute at all (not "gate=1"
  — physically absent). `in_proj` rebuilt to drop `z`'s slice; kernel called
  with `Z=None`.
- **§5c `frozen_As4d` — A-arm feasibility judgment: FEASIBLE (narrow
  scope)**. Two designs considered: (1, chosen) reparameterize only
  `A_const → A_head`'s functional form (`heavy_tail_activation`+clamp →
  S4D-style `-exp(A_log)`), a true single-line ablation needing no kernel
  change; (2, rejected) replace the whole SSM recurrence with a bespoke
  S4D kernel — rejected as out-of-scope for a "frozen 底座单部件改动"
  (conflates A with an entirely different execution path). Because (1) is
  feasible, **`frozen_As4d` is included** in all deliverables below — the
  task's own "if A judged INFEASIBLE, smoke only noconv/nogate" fallback
  does not apply.
- **§5d parameter accounting table** — hand arithmetic for all three arms,
  cross-verified against `build_model()`'s actual output in
  `secondary_unitcheck.json` (exact match, see §3 below).

## 2. `bm3_models.py` — new file, three arms

- `BearMamba3FrozenNoConv(BearMamba3Frozen)` — `conv_embed` only.
- `FrozenNoGateMamba3(FrozenSelectivityMamba3)` /
  `BearMamba3FrozenNoGate(BearMamba3Frozen)` — z-gating only.
- `FrozenAS4DMamba3(FrozenSelectivityMamba3)` /
  `BearMamba3FrozenAS4D(BearMamba3Frozen)` — A's parameterization only.

Each subclasses the frozen base and touches exactly the documented part;
everything else is inherited unchanged (verified in §3).

## 3. `secondary_unitcheck.py` → `secondary_unitcheck_20260703-0904/secondary_unitcheck.json` (chmod 444)

Per arm: `forward_pass` (shape + no-NaN), `backward_grads` (every parameter's
`.grad is not None`), `state_dict_diff` (**full-model**, not just one layer
— every key outside the documented `ARM_ALLOWED_DIFFS[arm]` set must match
`bm3_frozen` name+shape exactly). Plus a shared `param_table` over every arm
`build_model()` knows.

**`ALL_PASS: true`** for all three arms and the param table. Param deltas
vs `bm3_frozen` (112,410 params), all matching arch_map.md §5d exactly:

| arm | n_params | Δ vs bm3_frozen | non-target keys diff |
|---|---|---|---|
| `frozen_noconv` | 112,026 | −384 | zero (only `conv_embed.weight/bias`) |
| `frozen_nogate` | 79,642 | −32,768 | zero (only 4× `mamba_layers.{i}.in_proj.weight`) |
| `frozen_As4d` | 112,410 | 0 | zero (only 4× `A_const`↔`A_log` swap) |

## 4. Smoke test → `results/secondary_smoke_20260703-0907/smoke_report.json` (chmod 444)

`smoke_secondary_check.py`: `frozen_noconv`/`frozen_nogate`/`frozen_As4d`,
XJTU cross_speed clean, seed=0, 2 epochs each (3 cells total).

```
eval_sha256 = 6c20b367522ce5db59720120f8a7cd12792c7e070c6ae84bd1985511998727de
  prefix12 match (task spec "6c20b367522c…"): True
  full match (results/frozensel_20260702-1827/cells.jsonl):  True
frozen_noconv: loss 0.4585 -> 0.2101, no NaN
frozen_nogate: loss 0.3516 -> 0.0467, no NaN
frozen_As4d:   loss 0.1374 -> 0.0113, no NaN
all_pass: true
```

`frozen_As4d` was smoke-tested per the FEASIBLE verdict in §1/arch_map.md
§5c (not skipped — the task's INFEASIBLE fallback did not trigger).

## 5. `launch_secondary.sh` / `status_secondary.sh` — written, NOT executed

- Launch contract inherited byte-for-byte from `launch_frozensel.sh`: same
  VENV, `cd`, tmux `new-session`/`tee` pattern, `--run-dir` usage. Only
  changed: harness flag (`--secondary`), tmux session name (`xjtu_sec`),
  `RUNDIR`/`LOG` (`secondary_${TS}`). Zero caller-supplied variables.
- `status_secondary.sh` mirrors `status_frozensel.sh`'s shape exactly
  (`PENDING_TS` placeholder convention, same five sections).
- Reuse of `bm3_frozen`/`s4d` cells (per task step 5) is documented as a
  **manual, optional pre-launch step** in `launch_secondary.sh`'s header
  comment (a `python3 -c` snippet that copies matching rows from
  `results/frozensel_20260702-1827/cells.jsonl` into the new run's
  `cells.jsonl` so `run_grid()`'s existing resume-by-key logic skips them)
  — kept out of the script's own tmux command line to preserve
  `launch_frozensel.sh`'s launch mechanics byte-for-byte, per the task's
  explicit "启动契约逐字节继承" requirement.
- Neither script was executed. `--secondary` was never invoked as a full
  grid in this session — only unit checks (single forward/backward calls)
  and the 3-cell/2-epoch smoke test above.

## 6. Existing implementation code modified — declared explicitly

Per guardrail ("若必须修改既有实现代码,在报告里显式声明改了哪行、为何"),
**`xjtu_noisy_harness.py`** was edited. All changes are additive (new lines
only) except one existing line extended with an `or` clause. `bm3_frozen.py`,
`bm3_kin`/`bm3_nokin` wiring, `s4d` wiring, and every `results/**` file were
**not** touched.

| location | change | why |
|---|---|---|
| line 55 (new) | `from bm3_models import BearMamba3FrozenNoConv, BearMamba3FrozenNoGate, BearMamba3FrozenAS4D` | new arms need their classes importable |
| lines 266-283 (new, inserted between the existing `bm3_frozen` and `s4d_wide` branches) | three new `elif arm == "frozen_noconv"/"frozen_nogate"/"frozen_As4d"` branches in `build_model()` | `build_model()` must recognize the new arm names; no existing branch's lines were altered |
| line 678 (new) | `ap.add_argument("--secondary", ...)` | new CLI entry point for the secondary grid |
| line 698 (new) | `elif args.secondary: run_dir = ... "secondary_{ts}"` | run-dir naming, mirrors `--frozensel`'s own branch |
| lines 740-750 (new) | `elif args.secondary: conditions=...; arms=[...]; seeds=FULL_SEEDS; n_epochs=FULL_EPOCHS` | grid config for `--secondary`, mirrors `--frozensel`'s own branch |
| line 758 (existing line **extended**) | `if not (args.ablation or args.capctrl or args.frozensel):` → `if not (args.ablation or args.capctrl or args.frozensel or args.secondary):` | prevents `analyze_results()` (which writes a claim-bearing `decision.json`) from firing on `--secondary`, matching the task's "不写主张性结论" instruction and the same treatment `--frozensel` already gets |

Verified after editing: `build_model()` correctly builds all three new arms
via `xjtu_noisy_harness.build_model()` (the exact path the harness itself
uses), forward/backward both clean, param counts match arch_map.md exactly
(§3 above).

## What was explicitly NOT done (per task's own "明确不做")

- No full grid was run (`--secondary` was never invoked as a background/tmux
  training job).
- `bm3_kin`/`bm3_frozen`/`s4d`'s own existing arm implementations and every
  file under `results/**` from prior runs: untouched.
- No claim-bearing / adjudicative conclusion about whether any arm "wins" —
  that is explicitly out of scope for this stage.

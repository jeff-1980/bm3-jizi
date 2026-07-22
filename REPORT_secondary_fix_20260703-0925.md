# REPORT_secondary_fix_20260703-0925.md — fixes for r1 review (`.orchestrate/0703-085303`)

Reviewer verdict was REVISE with 3 critical issues, all against existing
implementation files (`bm3_models.py`, `secondary_unitcheck.py`). Per
guardrails, every changed line in those existing files is declared below,
with the reason. No `results/` file, and no prior `secondary_unitcheck_*`/
`secondary_smoke_*` directory, was modified or deleted — every new artifact
went into a fresh timestamped directory.

## Issue 1 (critical) — `secondary_unitcheck.py` only compared shapes, not values

**Why it was wrong**: `check_state_dict_diff` compared `state_dict()` key
presence + `tensor.shape` only. Two independently-initialized `nn.Linear`s
can have identical name+shape while holding completely different numbers —
so "identical" in the old report was a false positive.

**Fix** (`secondary_unitcheck.py`):
- Added `_tensor_max_abs_diff(a, b)` — computes `max|a-b|` in float64
  (lossless upcast for bf16/fp32) for floating tensors, `torch.equal` for
  non-floating ones.
- Rewrote `check_state_dict_diff(arm, arm_model, base_model)` (was
  `check_state_dict_diff(arm, arm_model, base_shapes)`) to take the actual
  models (not precomputed shapes) and, for every key present with matching
  shape in both, additionally require `max_abs_diff == 0` to count as
  `"identical"`; a nonzero diff on a non-whitelisted key is now
  `UNEXPECTED_VALUE_DIFF` and fails the check. Added
  `unexpected_value_diff_keys` to the JSON report and to the pass condition.
- Removed the now-unused `state_shapes()` helper (superseded by direct
  `state_dict()` access) and updated `main()`'s call site to pass
  `bm3_frozen_model` instead of `base_shapes`.
- Updated `ARM_ALLOWED_DIFFS["frozen_noconv"]` for the new stem's key names
  (see Issue 3) and the module docstring's description of check #3.

**Result**: regenerated `secondary_unitcheck_20260703-0920/secondary_unitcheck.json`
(chmod 444) — `ALL_PASS: true`, and independently re-verified outside this
script (`torch.manual_seed(0)`, direct `torch.equal`/`max_abs_diff` over
every shared-name key) that all three arms have **0** non-target numeric
diffs (56/58/54 shared keys respectively).

## Issue 2 (critical) — nogate/As4d discarded-and-reinitialized mamba_layers, perturbing non-target params

**Why it was wrong**: `BearMamba3FrozenNoGate`/`BearMamba3FrozenAS4D`
called `super().__init__()` (which already builds `n_layers` correctly-
seeded `FrozenSelectivityMamba3` layers, in the same RNG position as a
standalone `bm3_frozen` model), then threw those away and built brand-new
`FrozenNoGateMamba3`/`FrozenAS4DMamba3` layers via their constructors. Each
constructor re-runs `Mamba3.__init__`'s random init, consuming a *different*
slice of the RNG stream (after `bm3_frozen`'s own layers + layer_norms +
norm + classifier) — so `dt_bias`, `A_const`/`B_const`/`C_const`, `B_bias`/
`C_bias`, `D`, `out_proj.weight`, etc. all silently diverged numerically
from `bm3_frozen`, even though every name/shape matched.

**Fix** (`bm3_models.py`):
- Added `_SHARED_SCALAR_PARAMS` and `_copy_nontarget_frozen_params(dst,
  src, exclude=())` — copies `dt_bias`/`A_const`/`B_const`/`C_const`/
  `B_bias`/`C_bias`/`D` (bare `nn.Parameter`s) via `.copy_()`, plus
  `B_norm`/`C_norm` (`load_state_dict`) and `out_proj.weight`, from an
  already-built source layer into a destination layer, `exclude`-ing
  whichever key is the arm's own intentional target.
- `BearMamba3FrozenNoGate.__init__`: now iterates `self.mamba_layers` (the
  already-correct layers `super().__init__()` built), constructs a
  `FrozenNoGateMamba3` per original layer, calls
  `_copy_nontarget_frozen_params(new_layer, layer)` (copies everything
  including `A_const`, since this arm doesn't touch A), then sets
  `new_layer.in_proj.weight` by **slicing** `layer.in_proj.weight[layer.d_inner:]`
  (dropping the `z` rows) instead of leaving the fresh-random reduced
  `Linear` — so even the surviving `[x, trap, angle]` rows of `in_proj` stay
  byte-identical to `bm3_frozen`, not merely same-shaped.
- `BearMamba3FrozenAS4D.__init__`: same pattern, `exclude=("A_const",)`
  (A_const doesn't exist on the new layer — it was `del`eted and replaced by
  `A_log`, which intentionally keeps its own deterministic init, independent
  of the source layer's `A_const` value), and `in_proj.weight` copied in
  full (this arm doesn't touch `in_proj` at all).

**Result**: independently verified 0 non-target diffs for both arms (see
Issue 1's verification), vs. the reviewer's reported 12 (`frozen_nogate`)
and 16 (`frozen_As4d`) non-target diffs before the fix.

## Issue 3 (critical) — `frozen_noconv` still called `nn.Conv1d`

**Why it was wrong**: `BearMamba3FrozenNoConv.__init__` set
`self.conv_embed = nn.Conv1d(n_sensors, d_model, kernel_size=1, ...)`. This
is mathematically equivalent to removing cross-timestep mixing (a
`kernel_size=1` conv has no receptive field beyond the current timestep),
but it is, literally, still an `nn.Conv1d` instantiation — not sufficient
if the rubric requires zero `conv1d`/`Conv1d` calls in this arm's forward
path.

**Fix** (`bm3_models.py`):
- Added `StridedLinearStem(n_sensors, d_model, stride, ...)` — a plain
  `nn.Module` with **no** `nn.Conv1d`/`F.conv1d` anywhere: does
  `x[:, :, ::stride]` (explicit stride slicing, no convolution) followed by
  a per-timestep `nn.Linear(n_sensors, d_model)` channel lift, then
  transposes back to Conv1d's `(batch, d_model, L_out)` output convention.
  Exposes `.weight`/`.bias` properties (delegating to the inner `Linear`) so
  `BearMamba3Frozen.forward`'s `self.conv_embed.weight.dtype` cast — which
  is inherited unmodified — keeps working without touching `bm3_frozen.py`.
  Provably the same map as the removed `kernel_size=1` Conv1d
  (`out[:,d,t] = sum_c weight[d,c]*x[:,c,t*stride] + bias[d]`), and the same
  output length as the original `kernel=7,pad=3` stem for every `L_in`
  (docstring works the algebra).
- `BearMamba3FrozenNoConv.__init__`: `self.conv_embed = StridedLinearStem(
  n_sensors, d_model, conv_stride, dtype=dtype)` (was `nn.Conv1d(...)`).
- `secondary_unitcheck.py`'s `ARM_ALLOWED_DIFFS["frozen_noconv"]`: updated
  from `{"conv_embed.weight": ..., "conv_embed.bias": ...}` to also list
  the new key names (`conv_embed.proj.weight`, `conv_embed.proj.bias`) as
  allowed-added, and the old Conv1d key names as allowed-missing (they no
  longer exist at all, rather than existing with a new shape).
- `arch_map.md` §5a/§5d: updated to describe `StridedLinearStem` and record
  why the prior `kernel_size=1` `Conv1d` revision was superseded.

**Verification**: `grep -n "Conv1d\|conv1d" bm3_models.py` — the only hits
are in comments/docstrings; no `nn.Conv1d(` call remains anywhere in
`bm3_models.py`'s forward path.

## Not changed

`bm3_frozen.py` was NOT modified. `BearMamba3Frozen.__init__` still builds
a transient `nn.Conv1d` (bm3_frozen.py:154) before
`BearMamba3FrozenNoConv.__init__` immediately replaces it with
`StridedLinearStem` — that transient object is discarded, never called
(not in the forward path), and not present in the final model's
`state_dict()` (verified: `frozen_noconv`'s state_dict has no
`conv_embed.weight`/`conv_embed.bias` Conv1d keys, only
`conv_embed.proj.*`). This mirrors the arm's own prior design (which always
replaced `conv_embed` post-`super().__init__()`) and required no change to
the shared base class.

## Verification performed after the fix

1. `secondary_unitcheck.py` run end-to-end:
   `ALL_PASS: true`, all three arms pass forward/backward/state_dict_diff
   (now value-checked)/param_delta. New artifact:
   `secondary_unitcheck_20260703-0920/secondary_unitcheck.json` (chmod 444).
2. Independent re-check (outside `secondary_unitcheck.py`, same
   `torch.manual_seed(0)` protocol as the reviewer used): 0 non-target
   numeric diffs for all three arms.
3. `grep` confirms no `nn.Conv1d(` call remains in `bm3_models.py`.
4. `smoke_secondary_check.py` re-run end-to-end (3 cells, 2 epochs each,
   seed 0, XJTU clean): loss decreases epoch-over-epoch for all three arms,
   no NaN, `eval_sha256` full-hash match. New artifact:
   `results/secondary_smoke_20260703-0925/smoke_report.json` (chmod 444).

No existing `results/` file, and no prior `secondary_unitcheck_*`/
`secondary_smoke_*` output, was overwritten or deleted.

# arch_map.md — Mamba-3 selectivity mechanism map (this directory)

Scope: `mamba_ssm.modules.mamba3.Mamba3` as used by `bearmamba3/model.py`
(`BearMamba3`, arm `bm3_kin`/`bm3_nokin` in `xjtu_noisy_harness.py`), and the
frozen ablation `FrozenSelectivityMamba3` / `BearMamba3Frozen` in
`bm3_frozen.py` (arm `bm3_frozen`).

Source of truth: `/home/jeffwork/论文8/venv/lib/python3.12/site-packages/mamba_ssm/modules/mamba3.py`.

## 0. Base-class identity check (CWRU dir vs this dir)

Both `xjtu_noisy_harness.py` (this dir) and `smoke_harness.py`
(`../extended_baselines_noisy_20260626-2235/`) shebang
`/home/jeffwork/论文8/venv/bin/python3`. Confirmed live:

```
$ /home/jeffwork/论文8/venv/bin/python3 -c \
  "import mamba_ssm.modules.mamba3 as m; print(m.__file__)"
/home/jeffwork/论文8/venv/lib/python3.12/site-packages/mamba_ssm/modules/mamba3.py
```

Both directories resolve `Mamba3` to the exact same file on disk (single
shared venv, not two separate installs). **There is no version diff to
reconcile.** `bm3_frozen.py` in this directory is therefore a literal,
unmodified copy of `FrozenSelectivityMamba3`/`BearMamba3Frozen` from
`../extended_baselines_noisy_20260626-2235/models_extended.py` (r1's
implementation, lines 233-389 as of 2026-07-02) — no adaptation was needed or
made.

Config used by this dir's `build_model()` for `bm3_kin`/`bm3_nokin`/`bm3_frozen`:
`d_model=64, d_state=128, n_layers=4, n_sensors=1, conv_stride=2, is_mimo=False,
use_batchnorm=False, dtype=bfloat16` — identical to `BearMamba3Frozen`'s own
defaults and to CWRU's `bm3_nokin` config (`is_mimo=False` there too, per
CWRU docstring: "matches BearMamba3's actual configuration in this repo").
With these values: `d_inner=128, headdim=64, nheads=2, num_bc_heads=1
(ngroups=1), mimo_rank=1 (forced, is_mimo=False), num_rope_angles=32`.

## 1. Selective mechanism → input dependence (selective `Mamba3`)

`Mamba3.forward()` (mamba3.py:160-278), SISO branch (`is_mimo=False`,
mamba3.py:248-278):

`in_proj` (mamba3.py:107-108) is a single `nn.Linear(d_model, d_in_proj)`
with `d_in_proj = 2*d_inner + 2*d_state*num_bc_heads*mimo_rank + 3*nheads +
num_rope_angles` (= 550 for the config above). One projection of the input
token `u` emits, in order (mamba3.py:176-186):

| slice | width | consumer | input-dependent? |
|---|---|---|---|
| `z` | `d_inner`=128 | gate | fixed shape, per-token value — yes, but not "selectivity" (gate, not dt/A/B/C) |
| `x` | `d_inner`=128 | value stream `V` | same as above |
| `B` | `d_state*num_bc_heads*mimo_rank`=128 | selective B | **yes** |
| `C` | `d_state*num_bc_heads*mimo_rank`=128 | selective C | **yes** |
| `dd_dt` | `nheads`=2 | additive term into `DT` | **yes** |
| `dd_A` | `nheads`=2 | `A` (mamba3.py:194, `_A = -heavy_tail_activation(dd_A)`) | **yes** |
| `trap` | `nheads`=2 | trapezoidal integration correction | yes, but positional/numerical, not S6 selectivity |
| `angle` | `num_rope_angles`=32 | RoPE angle | yes, but positional, not S6 selectivity |

- **dt**: `DT = F.softplus(dd_dt + self.dt_bias)` (mamba3.py:196). `dt_bias`
  is a static `nn.Parameter` (mamba3.py:110-118, shape `(nheads,)`); `dd_dt`
  is the input-dependent additive term sliced from `in_proj(u)`. Input
  dependence flows entirely through `dd_dt`.
- **A**: `_A = -heavy_tail_activation(dd_A.float())`, clamped to
  `max=-A_floor` (mamba3.py:194-195). Unlike classic Mamba's `A_log` (a
  static per-channel parameter), **Mamba-3's `A` has no static form at all**
  — it is always `heavy_tail_activation` applied to a live `in_proj` slice
  (`dd_A`). There is no "A_log-equivalent" parameter to freeze; freezing A
  requires introducing a new constant (see §3).
- **B, C**: sliced from `in_proj(u)`, reshaped to `(b, l, mimo_rank,
  num_bc_heads, d_state)` (mamba3.py:189-190), then passed through
  `B_norm`/`C_norm` (`RMSNormGated`, mamba3.py:125-127, 205-206). Both are
  full per-token projections — canonical S6 selectivity.
- **z (gate)**, **trap**, **angle**: also sliced from the same `in_proj(u)`
  call and hence technically "input-dependent," but none of the three is
  part of the classic S6 selectivity mechanism (which is specifically
  {dt, A, B, C} — the SSM recurrence parameters). `z` is the output gate,
  `trap` is a trapezoidal-integration numerical correction, `angle` drives
  RoPE (a positional encoding, not a per-token content-selection signal).
  Frozen semantics S1 targets {dt, A, B, C} only — see §3.

## 2. `FrozenSelectivityMamba3` — what was removed / replaced

`in_proj` is rebuilt (bm3_frozen.py `__init__`) to emit only
`[z, x, trap, angle]`, width `2*d_inner + nheads + num_rope_angles` = 290
(down from 550 — the removed 260 = `B`(128) + `C`(128) + `dd_dt`(2) +
`dd_A`(2), exactly the four selective slices in §1). No other in_proj slice
is touched.

| mechanism | selective version | frozen version |
|---|---|---|
| dt | `softplus(dd_dt + dt_bias)`, per-token | `softplus(dt_bias)` only — `dt_bias` **inherited unmodified** from `Mamba3.__init__`, `dd_dt` slice deleted |
| A | `-heavy_tail_activation(dd_A)`, per-token, no static analogue in `Mamba3` | new `A_const = nn.Parameter(zeros(nheads))`; `A_head = -heavy_tail_activation(A_const)`, same clamp `max=-A_floor` |
| B | `in_proj` slice → `B_norm` | new `B_const = nn.Parameter(ones(mimo_rank, num_bc_heads, d_state))`, broadcast over `(batch, seqlen)` → same `B_norm` module |
| C | `in_proj` slice → `C_norm` | new `C_const = nn.Parameter(randn(...)/sqrt(d_state))`, broadcast over `(batch, seqlen)` → same `C_norm` module |
| z, trap, angle, `B_bias`, `C_bias`, `D`, `out_proj`, SSD kernel (`mamba3_siso_combined`) | input-dependent (z/trap/angle) or static, kernel unchanged | **identical**, unchanged code path |

`A_const` init to zeros ⇒ `heavy_tail_activation(0) = 1` ⇒ `A_head` starts at
`-1`, a neutral decay rate in the same order of magnitude as `Mamba3`'s own
dt-range init (`dt_min=0.001, dt_max=0.1`). `B_const` inits to all-ones (an
uninformative constant token for `B_norm`'s RMSNorm, matching `Mamba3`'s own
`B_bias`/`C_bias` all-ones init at mamba3.py:121-122). `C_const` inits
`randn/sqrt(d_state)` — a small-variance identity-like init consistent with
`RMSNormGated`'s expected input scale.

## 3. S1-S5 mapping (see also inline `bm3_frozen.py` verification, §3 below)

- S1 (dt/B/C don't depend on input): satisfied — §2 table. `dt` collapses to
  `softplus(dt_bias)`, a pure function of the static parameter; `B`/`C`
  collapse to `B_norm(B_const)`/`C_norm(C_const)`, broadcast, no `u` term
  anywhere in their computation.
- S2 (three constants are `nn.Parameter`, broadcast-compatible): `A_const`
  `(nheads,)`, `B_const`/`C_const` `(mimo_rank, num_bc_heads, d_state)` =
  `(1, 1, 128)` for this config — all `nn.Parameter`, all `.expand()`-broadcast
  to `(batch, seqlen, ...)` in `forward()` without reshape errors (verified in
  `frozensel_unitcheck.json`).
- S3 (selective projection physically removed): `in_proj` is a freshly
  constructed `nn.Linear` with the smaller width — the old, larger `in_proj`
  (with B/C/dd_dt/dd_A slices) is not "masked" or "ignored," it no longer
  exists as an attribute after `__init__` reassigns `self.in_proj`.
- S4 (rest of block identical): RoPE, trapezoidal correction, biases, `D`
  skip, `out_proj`, and the `mamba3_siso_combined` kernel call are the exact
  same call shape as `Mamba3.forward`'s SISO branch (mamba3.py:248-278) —
  compare line-by-line with `bm3_frozen.py::FrozenSelectivityMamba3.forward`.
- S5 (A's static-vs-dynamic invariant preserved): `Mamba3`'s own A is
  *always* input-projected (§1) — there is no static `A_log` in the
  selective version to "not change." Freezing A therefore *requires*
  introducing `A_const` (§2); this is the correct behavior per S5, not a
  violation of it (S5 forbids frozen from *changing* an already-static A's
  parameterization — there is none here to change).

## 4. Parameter accounting (this dir, `d_model=64, d_state=128, n_layers=4`)

Per-layer `in_proj` width: selective 550 → frozen 290 (Δ = -260 × d_model =
-16640 weights/layer). Per-layer new constants: `A_const`(2) +
`B_const`(128) + `C_const`(128) = 258/layer.
Net per layer: `-16640 + 258 = -16382`; × 4 layers = `-65528`.

Total model params (conv stem + 4 Mamba layers + LayerNorms + classifier,
verified by `build_model()` on this dir's harness):

| arm | params |
|---|---|
| `bm3_kin` / `bm3_nokin` (selective) | 177,938 |
| `bm3_frozen` | 112,410 |

`177938 - 112410 = 65528` — matches the `in_proj`-shrink arithmetic above
exactly; see `frozensel_unitcheck.json` for the full state_dict-keyed diff.

---

## 5. Secondary arms (stage 2) — conv / gate / A, each a single-part change on top of `bm3_frozen`

Scope note before the per-arm write-up: **`Mamba3.forward()` has no internal
depthwise/causal `conv1d` module at all** (mamba3.py:160-278, full read above
— unlike classic Mamba-1/Mamba-2, which run a short causal `nn.Conv1d` over
`[x, B, C]` before the SSM scan, Mamba-3 replaces that local-mixing role with
RoPE on `B`/`C`, §1 above). So there is no in-block "conv" to remove from
`Mamba3`/`FrozenSelectivityMamba3` itself. The only convolution anywhere in
this arm's forward path is the **outer stem** `conv_embed` in
`BearMamba3Frozen.__init__`/`forward` (bm3_frozen.py:154-155, 171),
`nn.Conv1d(n_sensors, d_model, kernel_size=7, stride=conv_stride, padding=3)`
— applied once, before the first Mamba layer, to lift the raw 1-channel
signal to `d_model=64` channels and downsample by `conv_stride=2`. "conv" for
this ablation set means that stem. This is recorded here rather than silently
reinterpreted, because it is a correction of what the task phrasing implies
("conv ... in Mamba3's forward") — verified against the installed
`mamba3.py` source, not assumed.

New module: `bm3_models.py` (this dir). Every arm below subclasses
`BearMamba3Frozen`/`FrozenSelectivityMamba3` (bm3_frozen.py) and changes
**exactly one** part; every other line of the frozen base's forward path is
untouched (verified per-arm in `secondary_unitcheck.json`'s state_dict diff,
mirroring `frozensel_unitcheck.json`'s method in §3 above).

### 5a. `frozen_noconv` — stem conv removed

- **Location**: `BearMamba3Frozen.__init__`, `self.conv_embed =
  nn.Conv1d(n_sensors, d_model, kernel_size=7, stride=conv_stride,
  padding=3)` (bm3_frozen.py:154-155); consumed once in `forward` at
  `h = self.conv_embed(x...)` (bm3_frozen.py:171). Nowhere inside any
  `Mamba3`/`FrozenSelectivityMamba3` layer (see scope note above).
- **Removal method (revised 2026-07-03)**: `BearMamba3FrozenNoConv`
  (bm3_models.py) replaces `conv_embed` with `StridedLinearStem` — a
  **non-convolutional** module (`bm3_models.py`, no `nn.Conv1d`/`F.conv1d`
  call anywhere) built from explicit stride slicing (`x[:, :, ::stride]`)
  plus a plain `nn.Linear(n_sensors, d_model)` channel lift applied at each
  surviving timestep. A prior revision used `nn.Conv1d(..., kernel_size=1,
  ...)` instead — mathematically the same per-timestep map (a kernel-1 conv
  has zero cross-timestep receptive field) but still, literally, a `Conv1d`
  call; that revision is superseded because the rubric requires no
  conv1d/Conv1d call in this arm's forward path at all, not just a
  kernel-1 one. `StridedLinearStem` removes the kernel's cross-timestep
  mixing (the size-7 receptive field that lets one output position see 7
  neighboring input samples) while keeping the channel lift
  (`n_sensors→d_model`) and the downsample stride identical, computing the
  exact same numbers a `kernel_size=1` `Conv1d` would (`out[:,d,t] =
  sum_c weight[d,c]*x[:,c,t*stride] + bias[d]`) via slicing + `Linear`
  instead of `Conv1d`. Output sequence length is provably unchanged: a
  `stride`-step Python slice of a length-`L` sequence yields
  `ceil(L/stride) = floor((L-1)/stride)+1` elements, the same closed form
  `Conv1d(kernel_size=1, padding=0)` gives, which in turn equals the
  original `kernel_size=7, padding=3` stem's `L_out` for every `L_in`
  (`2*padding-kernel_size = -1` in both: `2*3-7` and `2*0-1`) — so
  downstream shapes need zero adjustment. Nothing else in
  `BearMamba3FrozenNoConv` differs from `BearMamba3Frozen` — `mamba_layers`
  stays `FrozenSelectivityMamba3`, unmodified, and (verified numerically,
  not just by shape, in `secondary_unitcheck.json`) numerically identical
  to `bm3_frozen`'s own layers.
- **Feasibility**: trivially FEASIBLE — a stem hyperparameter/module change,
  no kernel/shape implications downstream.

### 5b. `frozen_nogate` — z-gating removed

- **Location**: `z` is sliced from `in_proj(u)`'s output (bm3_frozen.py:92-94,
  inherited unchanged from `Mamba3.forward`'s `[z, x, B, C, dd_dt, dd_A,
  trap, angle]` layout, §1 above — `z`'s slot survives the S1-freeze
  untouched). It is consumed exactly once, as the `Z=` argument to
  `mamba3_siso_combined` (bm3_frozen.py:124-129). The **gating multiply
  itself is not visible in `mamba3.py` at all** — it lives inside the fused
  Triton kernel: `mamba3_siso_fwd.py`, docstring line 474
  (`"Z: Gating tensor ... Applies SiLU gating: out = out * silu(Z)"`) and
  the actual op at line 411, guarded by a `HAS_Z: tl.constexpr` compile-time
  flag (`HAS_Z=Z is not None`, mamba3_siso_combined.py:686) — read directly
  from the installed kernel source, not inferred from the docstring alone
  (`mamba3_siso_fwd.py:410-411`: `if HAS_Z: acc_o = acc_o * silu(z_block...)`).
  When `Z=None`, `HAS_Z=False` and that whole branch — including the load
  of `z_block` (line 372-373) — never executes; gating is not "set to 1",
  it is compiled out.
- **Removal method**: `FrozenNoGateMamba3(FrozenSelectivityMamba3)`
  (bm3_models.py) (1) rebuilds `in_proj` to width `d_inner + nheads +
  num_rope_angles` (162 for this config), dropping exactly the `d_inner`=128
  slice that was `z` — matching the "S3: selective projection physically
  removed" precedent §3 uses for B/C, so the gate is removed the same way
  selectivity was, not merely bypassed — and (2) calls
  `mamba3_siso_combined(..., Z=None, ...)`. `x`/`trap`/`angle` slices,
  `A_const`/`B_const`/`C_const`, `dt_bias`, `B_norm`/`C_norm`, `out_proj`
  are all unchanged from `FrozenSelectivityMamba3`.
- **Single-arm isolation (revised 2026-07-03)**: `BearMamba3FrozenNoGate`
  (bm3_models.py) builds `FrozenNoGateMamba3` layers, then overwrites every
  non-target parameter (`dt_bias`, `A_const`, `B_const`, `C_const`,
  `B_bias`, `C_bias`, `D`, `B_norm`, `C_norm`, `out_proj.weight`) in place
  with `.copy_()` from the corresponding `bm3_frozen`-style layer
  `BearMamba3Frozen.__init__` already built for that same model instance,
  and builds the surviving `in_proj` rows by **slicing** that same layer's
  original `in_proj.weight` (dropping the `z` rows) rather than a fresh
  random `Linear`. A prior revision constructed `FrozenNoGateMamba3` layers
  directly via `mamba_layers = nn.ModuleList([FrozenNoGateMamba3(...) for
  _ in range(n_layers)])`, which re-runs `Mamba3.__init__`'s random
  initialization for every "non-target" parameter — same name/shape as
  `bm3_frozen`, but a *different* draw from the RNG stream (that draw comes
  after `bm3_frozen`'s own layers, `layer_norms`, `norm`, and `classifier`
  in construction order), so the shape-only state_dict diff previously in
  `secondary_unitcheck.py` reported "identical" while the actual numbers
  differed in 12 keys. Fixed by copying values instead of re-initializing;
  verified independently (not just via `secondary_unitcheck.json`'s own
  check) — see `secondary_unitcheck.py`'s `check_state_dict_diff`, which
  now compares tensor values (`max_abs_diff == 0`), not just names/shapes.
- **Feasibility**: FEASIBLE — `Z` is already an `Optional[Tensor]` kernel
  argument (`mamba3_siso_combined.py:451`, default `None`), so `Z=None` is a
  supported, first-class code path, not a hack.

### 5c. `frozen_As4d` — A's parameterization only

- **Location**: in `FrozenSelectivityMamba3.forward` (bm3_frozen.py:105-106):
  `A_head = -heavy_tail_activation(self.A_const); A_head = torch.clamp(A_head,
  max=-self.A_floor)`. `A_const` itself (`nn.Parameter(zeros(nheads))`,
  bm3_frozen.py:74) was already introduced in stage 1 to replace Mamba-3's
  input-projected `dd_A` (§2 above — Mamba-3's A has no static analogue in
  the selective model, so `A_const` *is* the static analogue, and freezing
  it is what "S1" already means for A). There is therefore nothing further
  to make *input-independent* — A is already 100% static in `bm3_frozen`.
  What this arm ablates instead is **which static functional form** produces
  `A_head` from that static parameter: `bm3_frozen` uses Mamba-3's own
  `heavy_tail_activation` (a piecewise `1+x` / `1/(1-x)` map, mamba3.py:27-41,
  chosen by the Mamba-3 authors for WSD-training stability) plus an
  `A_floor` clamp; classic structured state-space models (S4D/S4/Mamba-1)
  instead parameterize a negative-real diagonal `A` directly as
  `A = -exp(A_log)`, which is unconditionally negative with no clamp needed.
- **Feasibility judgment: FEASIBLE, narrow scope.** Two designs were
  considered:
  1. *(chosen)* **Reparameterize only the `A_const → A_head` map** in
     `FrozenAS4DMamba3(FrozenSelectivityMamba3)` (bm3_models.py): drop
     `A_const`, add `A_log = nn.Parameter(log([1, ..., nheads]))`
     (S4D-real/HiPPO-style arithmetic-progression init — `nheads=2` here
     gives `A_head` init `[-1, -2]`, the same order of magnitude as
     `bm3_frozen`'s own `A_const=0 → A_head=-1` init, §2 above), and compute
     `A_head = -torch.exp(self.A_log)`. Every other line — `in_proj` shape,
     `dt`/`B`/`C`, the `mamba3_siso_combined` call signature, `out_proj` — is
     byte-identical to `FrozenSelectivityMamba3`. This is a true
     single-variable ablation (one line's functional form) and needs no
     kernel change, since the kernel only ever consumes the already-combined
     `ADT`/`DT` tensors (mamba3_siso_combined.py signature) — it has no
     opinion on how `A_head` was produced. **This is the design used for
     `frozen_As4d`.**
  2. *(rejected, judged INFEASIBLE as a "single-part change")* Replace the
     entire recurrence with a textbook S4D convolution/scan (bypassing
     `mamba3_siso_combined` altogether). Rejected because it is not a
     "frozen 底座单部件改动" (single-component change to the frozen base) by
     construction — it would replace the shared, fused, gradient-checked
     Triton kernel with a bespoke implementation, conflating an A-only
     ablation with an entirely different execution path (chunked-scan
     numerics, RoPE handling, `D`-skip, trapezoidal correction would all
     need reimplementing or dropping), making any resulting effect
     un-attributable to A specifically. Out of scope for this stage.
- Because design 1 is feasible, **`frozen_As4d` is INCLUDED** in the smoke
  and (write-only) full-grid arm list below — the task's own fallback
  ("若 A 臂判 INFEASIBLE 则只烟雾 noconv/nogate 两臂") does not apply.
- **Single-arm isolation (revised 2026-07-03)**: same fix and same prior bug
  as §5b's nogate arm — `BearMamba3FrozenAS4D` now copies every non-target
  parameter (`dt_bias`, `B_const`, `C_const`, `B_bias`, `C_bias`, `D`,
  `B_norm`, `C_norm`, `out_proj.weight`, and the full `in_proj.weight`,
  which this arm doesn't touch at all) verbatim from the `bm3_frozen`-style
  layer already built for that model instance, instead of re-initializing a
  fresh `FrozenAS4DMamba3(...)` (which previously left 16 non-target keys
  numerically diverged, including `C_const`, `dt_bias`, `in_proj.weight`,
  `out_proj.weight`). Only `A_const`(dropped)/`A_log`(new) are exempt from
  the copy, by design — `A_log` keeps its own deterministic
  arithmetic-progression init regardless of the source layer's `A_const`.

### 5d. Parameter accounting (three secondary arms, this dir's config)

| arm | change vs `bm3_frozen` (112,410 params) | Δ params | total params |
|---|---|---|---|
| `frozen_noconv` | `conv_embed`: `Conv1d(1,64,k=7,pad=3)` → `StridedLinearStem` (stride-slice + `Linear(1,64)`, no `Conv1d`); weight `64·1·7=448→64·1=64`, bias `64→64` unchanged | `(64+64)-(448+64) = -384` | 112,026 |
| `frozen_nogate` | `in_proj` per layer: out_features `290→162` (drops the `z` slice, width `d_inner`=128), `in_features=d_model=64` unchanged, no bias | `4 layers × (162-290) × 64 = -32,768` | 79,642 |
| `frozen_As4d` | `A_const(nheads=2,)` → `A_log(nheads=2,)`, same shape, per layer | `4 layers × (2-2) = 0` | 112,410 |

All three deltas AND every non-target key's tensor VALUE (not just its
name/shape) are verified against `build_model()`'s actual output in
`secondary_unitcheck.json` — `check_state_dict_diff` computes
`max_abs_diff` (float64) for every shared-name, shared-shape key and fails
the check if any non-whitelisted key is nonzero.

---

## 6. Constructive graft (stage 3) — `s4d_plus_gate`: grafting the gate onto S4D

Scope: §5 above is ablative (start from the selective base, remove one
part). This arm inverts the direction: start from the never-selective
baseline (`BearS4D`, `models_extended.py:203-226`, arm `s4d`, 100,162
params — verified §6.4 below) and graft ON one part borrowed from Mamba-3's
mechanism, to test whether that part alone recovers any of the
`bm3_frozen`-vs-`s4d` gap. Predeclared statistic/cutoffs:
`graft_prereg_provenance.md` (untouched by this section — no criteria are
restated or altered here). New module: `BearS4DPlusGate` (`bm3_models.py`,
arm `s4d_plus_gate`), registered in this dir's `build_model()`
(`xjtu_noisy_harness.py`).

### 6.1 Why "gate" is the part grafted, and what it means for S4D

§1 above establishes that Mamba-3's per-token `in_proj(u)` emits, among
others, `z` (gate) and the four canonical S6 selectivity slices `{B, C,
dd_dt, dd_A}`. `z` is deliberately NOT part of {dt, A, B, C} — S1's
selectivity-freeze (§2/§3) leaves `z` completely untouched in
`bm3_frozen`/`FrozenSelectivityMamba3`, and §5b's `frozen_nogate` arm
already isolates removing *only* `z` from the selective side. `s4d_plus_gate`
is that same single part, added from the OTHER direction: `BearS4D`
(`_S4DKernel`/`_S4DLayer`, `models_extended.py:140-200`) has no gate at all
— `_S4DLayer.forward` (`models_extended.py:189-200`) is a pure
FFT-convolution + static skip (`y = fft_conv(u, kernel) + D*u`), never
multiplied by any input-dependent term. Grafting a Mamba-3-style output gate
onto it, with dt/A/B/C left exactly as time-invariant as `BearS4D` already
makes them, isolates the gate's own contribution from selectivity — which is
the point of the R statistic in `graft_prereg_provenance.md` (uses
`bm3_frozen`, which has dt/A/B/C static AND has the gate, as the numerator's
upper reference).

### 6.2 Gate branch insertion point (in `s4d`'s forward)

`BearS4D.forward` (`models_extended.py:220-226`):

```python
def forward(self, x, return_kin=False):
    h = self.conv_embed(x).transpose(1, 2)  # (B, L', d_model)
    for layer, ln in zip(self.s4d_layers, self.layer_norms):
        h = h + layer(ln(h))                 # pre-norm residual
    h = self.norm(h).mean(1)                 # global avg pool
    logits = self.classifier(h)
    return (logits, []) if return_kin else logits
```

The single line `h = h + layer(ln(h))` (`models_extended.py:223`) is the
insertion point. `BearS4DPlusGate.forward` (`bm3_models.py`) overrides this
loop body only — every other line (`conv_embed`, `self.norm(h).mean(1)`,
`self.classifier(h)`) is called identically:

```python
for layer, ln, gate in zip(self.s4d_layers, self.layer_norms, self.gate_proj):
    ln_h = ln(h)
    y = layer(ln_h)          # untouched _S4DLayer.forward — same call BearS4D itself makes
    z = gate(ln_h)           # z's source projection (§6.3)
    h = h + y * F.silu(z)    # grafted gate branch: out = out * silu(z), Mamba-3-style (§1)
```

`y = layer(ln_h)` is the exact same call `BearS4D.forward` itself makes
(same `layer`/`ln` objects, same argument) — the SSM computation itself
(`_S4DKernel`/`_S4DLayer`) is untouched; only the residual line's
right-hand side changes, from `layer(ln(h))` to `layer(ln(h)) *
silu(gate(ln(h)))`. This mirrors where Mamba-3 itself applies its gate:
`out = out * silu(Z)`, applied to the SSM kernel's output before it is
consumed downstream (mamba3_siso_fwd.py:410-411, §5b above) — here, applied
to `_S4DLayer`'s output `y` before the residual add, the direct S4D analogue
of "before it is consumed downstream" since `_S4DLayer` has no separate
`out_proj` stage the way `Mamba3`/`FrozenSelectivityMamba3` do.

### 6.3 `z`'s source projection

New parameter: `self.gate_proj = nn.ModuleList([nn.Linear(d_model, d_model,
bias=True) for _ in range(n_layers)])` (`bm3_models.py`,
`BearS4DPlusGate.__init__`), one `nn.Linear` per layer. `z = gate(ln_h)` —
`gate_proj[i]` is applied to `ln_h = ln(h)`, the SAME tensor that is also
passed into `layer(ln_h)` as `_S4DLayer`'s own input `u`
(`models_extended.py:189`, `_S4DLayer.forward(self, u)`). This is the S4D
analogue of Mamba-3's `z`, which is sliced from the same `in_proj(u)` call
that also produces the `x`/`B`/`C` consumed by that layer's SSM (§1 above) —
same source tensor feeding both the gate and the SSM branch, not an
unrelated signal (e.g. gating from `y` itself would make `z` a function of
the SSM's own output rather than of the layer's input, which is not what
Mamba-3's `z` is; gating from a static parameter would not be
input-dependent at all and would not graft "a gate" in any meaningful
sense). `_S4DLayer` itself has no fused input projection to slice a `z`
channel out of the way `Mamba3.in_proj` does — `B`/`C`/`D`/`dt` in
`_S4DKernel` are free `nn.Parameter`s, not projections of `u` at all, which
is exactly the time-invariance property `s4d` ablates relative to `Mamba3`
(§1) — so a new, minimal `nn.Linear(d_model, d_model)` of that same layer
input is the smallest addition that gives `z` an analogous provenance.

### 6.4 S4D original — zero-change checklist

Every symbol `BearS4DPlusGate` touches is either (a) inherited unmodified
via `super().__init__()`/attribute access, or (b) new. Nothing in
`models_extended.py` is edited:

| `models_extended.py` symbol | touched by `s4d_plus_gate`? | how |
|---|---|---|
| `_S4DKernel` (class body, lines 140-179) | no | never referenced directly; reached only via `_S4DLayer.kernel_fn`, itself unreached directly |
| `_S4DLayer.__init__` (182-187) | no | `_S4DLayer` instances are built by `BearS4D.__init__` (inherited via `super().__init__()`), not reconstructed |
| `_S4DLayer.forward` (189-200) | no | called as `layer(ln_h)`, identical call shape/args to `BearS4D.forward`'s own `layer(ln(h))` |
| `BearS4D.__init__` (210-218) | no | called via `super().__init__(...)` with the same kwarg names; builds `conv_embed`/`s4d_layers`/`layer_norms`/`norm`/`classifier` exactly as it does for a plain `s4d` build |
| `BearS4D.forward` (220-226) | not called | `BearS4DPlusGate` overrides `forward` in the subclass; `BearS4D.forward` itself is dead code for this arm, not edited or monkeypatched |
| `BearS4D.conv_embed`/`.s4d_layers`/`.layer_norms`/`.norm`/`.classifier` (attributes) | reused, not replaced | same objects `super().__init__()` constructed; `BearS4DPlusGate.__init__` only adds `self.gate_proj`, never reassigns any of these |

No line of `models_extended.py` is edited, and no `BearS4D`/`_S4DLayer`
instance attribute is mutated or monkeypatched after construction — the only
new state is `BearS4DPlusGate.gate_proj`, added after `super().__init__()`
returns. Consequence, verified in `graft_unitcheck.json`: because
`gate_proj` is constructed strictly AFTER `super().__init__()` (which draws
the RNG stream in the exact same order `BearS4D.__init__` alone would),
every key `s4d_plus_gate` shares with `s4d`
(`conv_embed.{weight,bias}`, `s4d_layers.*`, `layer_norms.*`, `norm.*`,
`classifier.{weight,bias}`) is byte-identical under the same seed — not just
same name/shape, `torch.equal` true for every one — and the only new keys
are `gate_proj.{0..3}.{weight,bias}`.

### 6.5 Parameter accounting

`s4d` (this dir's `build_model()`, `d_model=64, d_state=128, n_layers=4,
n_classes=2`): **100,162** params (verified by direct count, matches the
task's reference value; also cross-checked against `_S4DKernel`'s own
per-layer arithmetic: `log_dt`(64) + `log_neg_A`(64×128=8192) +
`B`(8192) + `C`(8192) + `D`(64) = 24,704/layer × 4 = 98,816, plus
`conv_embed` `Conv1d(1,64,k=8,pad=3)` weight+bias = 512+64=576, plus 4×
`LayerNorm(64)` = 4×128=512, plus final `LayerNorm(64)`=128, plus
`classifier` `Linear(64,2)`=128+2=130 → 98,816+576+512+128+130=100,162 ✓).

`gate_proj` adds one `nn.Linear(64, 64, bias=True)` per layer:
`(64×64 + 64) = 4,160`/layer × 4 layers = **+16,640**.

`s4d_plus_gate` total: `100,162 + 16,640 = 116,802` — verified against
`build_model()`'s actual output in `graft_unitcheck.json` (also cross-checked
there via the full state_dict diff: every non-`gate_proj` key
`torch.equal`-identical to `s4d`, and `sum(new-key numel) == 16,640`).

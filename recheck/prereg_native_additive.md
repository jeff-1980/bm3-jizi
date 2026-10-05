# Pre-specification: gate -> additive substitution in the native host (exp. 3) and the reverse-direction additive control (exp. 4)

Paper 9, stage 4. Written 2026-10-05, **before any training cell of either grid**. No stage-4
result exists at the time of writing. The construction check (no data, no training) was run first
and is quoted as a design fact. This file is committed and pushed to `github.com/jeff-1980/bm3-jizi`
before the first cell; the remote commit time is the external time-stamp. sha256 and local mtime are
recorded in `prereg_native_additive.sha256`. After launch this file is not edited; departures go to
the DEVIATIONS section of the decision memo.

## 0. Fixed conditions (both experiments)
- Harness `xjtu_noisy_harness.py` imported unmodified (sampler, AdamW, cosine schedule, clip,
  batch 64, matched-SNR AWGN in training and evaluation, macro-F1 on the held-out condition).
- AWGN 0, -2, -6 dB; seeds 0-4; 50 epochs; horizontal channel; binary OR vs IR.
- **Primary rule: final epoch.** Best-epoch is reported alongside and never used for a decision.
- Paired differences: seed mean, two-sided 95 % t-interval (df 4), seeds with positive difference.
- Environment `bm3-repro` (torch 2.9 / triton 3.5), the build of every reference cell used below.
- GPU memory read at launch and logged. **Fuse: 15 GPU-hours summed over stage-4 cells (wall time,
  including time under contention); the driver stops and the numbers are reported as they stand.**
- Smoke runs (2 epochs) go to separate files and are excluded from every analysis.
- Bootstrap: the stage-3 sorted-order implementation (arm order fixed by sorting, so the output does
  not depend on PYTHONHASHSEED), 20,000 replicates, `numpy.random.default_rng(20261005)`, each arm's
  five seeds resampled with replacement independently per level, statistic recomputed per replicate,
  2.5/97.5 percentiles. In a replicate, a level whose denominator is below 2 pp is excluded from that
  replicate's mean over levels (the fraction of such replicates is reported).

## 1. Experiment 3 - native-host gate -> additive substitution (15 cells; decisive)

### Arm (`recheck/additive_native.py`)
`bm3_frozen_add`: bm3_frozen with the scan kernel called with Z = None and the layer output
`out_proj(y + silu(z))` (sum in float32), instead of the kernel's `y * silu(z)` (applied after the
D-skip, in float32). Same modules, same parameters (112,410), same initial weights.
Construction check `recheck/additive_native_unitcheck.py` passed before this file was written:
(1) in bm3_frozen the kernel output with Z equals the Z=None output times silu(z) - the donor
operator; (2) parameter difference 0 %; (3) state_dict bitwise identical to bm3_frozen at the same
seed; (4) the kernel is called with Z=None, the tensor entering out_proj equals y + silu(z) and not
y * silu(z), and z is input-dependent; (5) forward finite, every parameter has a finite gradient.
Snapshot: `recheck/unitcheck_stage4/`.

### Data and references
Original transition 37.5Hz11kN -> 40Hz10kN; eval fingerprint must equal `6c20b367522c...`
(asserted per cell). Reference cells, not re-run: `bm3_frozen` and `frozen_nogate` from the stage-1
re-run (`recheck/stage1/cells_recheck.jsonl`), same seeds, levels, epochs and build. Known at writing
(final epoch): bm3_frozen 83.1 / 87.3 / 83.1, frozen_nogate 74.0 / 71.7 / 60.5, denominator
9.1 / 15.6 / 22.6 pp.

### Statistic
Native rescue ratio rho(level) = (add - nogate) / (frozen - nogate), final epoch;
**rho = mean of the three per-level values**; seed-resampled 95 % interval as in section 0
(add, nogate, frozen resampled independently). Also reported: frozen - add and add - nogate paired
differences per level; rho under best-epoch.

### Decision rule
- **NOT RESCUED**: rho < 0.3 **and** interval upper bound < 0.5 -> multiplicative-gate specificity
  confirmed in the native host.
- **RESCUED**: rho >= 0.7 **and** interval lower bound > 0.5 -> any same-size branch suffices;
  gate specificity is withdrawn and the account is restructured.
- **Otherwise PARTIAL/INCONCLUSIVE**, written at the tier less favourable to gate specificity:
  RESCUED if the interval upper bound is >= 0.7, PARTIAL otherwise.

## 2. Experiment 4 - reverse-direction additive control (15 cells; descriptive)
`s4d_plus_branch` (stage-3 arm, unchanged code) on 40Hz10kN -> 37.5Hz11kN; eval fingerprint
`934248343b29...` (stage 3). References from stage 3 (`recheck/stage3/cells_stage3_exp2.jsonl`):
s4d, s4d_plus_gate_wm, bm3_frozen.
R2_branch = (branch - s4d)/(bm3_frozen - s4d), mean over levels; R2_gate = 0.911 [0.825, 1.040]
(stage 3, final). dR2 = R2_gate - R2_branch with shared resamples of s4d and bm3_frozen.
No hard gate. Descriptive labels fixed now: **branch ~ gate** if the dR2 interval contains 0 and
|dR2| < 0.2; **branch << gate** if dR2 > 0.2 with the interval excluding 0; otherwise
**unresolved**. Whatever the label, the manuscript states next to R2 that S4D sits near the trivial
floor in this direction (final 51-52 % vs 46.8 % majority-class macro-F1).

## 3. Title decision tree (fixed now; executed by tier after the memo)
The current manuscript (v6) title is *"Gating Benefits Depend on the Backbone: An Ablation Study of
a Mamba-3 Block under Matched-Noise Cross-Condition Drift"*. The tree below is the author's,
adopted verbatim:

| Exp. 3 tier | Title action |
|---|---|
| NOT RESCUED | Use "Necessary but Not Transplantable": necessary is gate-specific (confirmed in the native host); what transfers is generic capacity. Body text follows the section 6 wording "no gate-specific transfer". The subtitle is drafted to match the evidence and shown to the author. |
| RESCUED | Restructure the title around block structure / interaction; draft 2 candidates for the author; do not finalise. |
| PARTIAL (incl. INCONCLUSIVE) | Drop "Not Transplantable"; interaction wording; draft 2 candidates for the author; do not finalise. |

## 4. Not in scope
Independent bearings or a second dataset; final title choice under RESCUED/PARTIAL; local build of
the submission PDF.

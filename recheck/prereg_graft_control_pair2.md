# Pre-specification: width-matched graft control (exp. 1) and reverse transition (exp. 2)

Paper 9 (BM3 attribution study), stage 3. Written 2026-10-04, **before any training run of
either grid**. No result of either grid existed when this file was written. Construction checks
(no data, no training) and the pair-2 evaluation fingerprint (data only) were computed first and
are quoted here as design facts. This file is committed and pushed to
`github.com/jeff-1980/bm3-jizi` before the first cell is launched; the commit time on the remote is
the external time-stamp. Its sha256 and local mtime are recorded in
`prereg_graft_control_pair2.sha256` next to it. After the grids start, this file is not edited;
any departure is written to the DEVIATIONS section of the decision memo, not here.

## 0. Fixed conditions (both experiments)

- Harness `xjtu_noisy_harness.py` imported unmodified: sampler, AdamW, cosine schedule, gradient
  clip, batch 64, AWGN at matched SNR in training and evaluation, macro-F1 on the held-out condition.
- Levels: AWGN 0, -2, -6 dB. Seeds 0-4. 50 epochs. Single horizontal channel, binary OR vs IR.
- **Primary reporting rule: final epoch** (cosine LR annealed to 0; no evaluation data used to
  choose the epoch). Best-epoch (max over epochs on the evaluation condition, the harness's own
  rule) is reported in the same tables and never used for a decision.
- Paired differences: mean over seeds, two-sided 95 % t-interval (df = 4), and the number of seeds
  (of 5) with a positive difference. Pairing is by seed (same seed -> same noise realisation).
- Environment: conda env `bm3-repro` (torch 2.9 / triton 3.5), the same build as the stage-1 and
  stage-2 re-runs whose cells are used as references below.
- GPU memory is read at launch and logged. **Fuse: if the summed wall time of stage-3 cells reaches
  30 GPU-hours, the driver stops and the numbers are reported as they stand.**
- A 2-epoch smoke run of the new arms may be made to test the pipeline; it is written to a separate
  file (`smoke3.jsonl`) and is excluded from every analysis.

## 1. Experiment 1 - width-matched graft and additive control (30 cells)

### Question
Is the low recovery of the original graft (R = 0.174 best-epoch, 0.196 final) an artefact of the
graft being narrower and smaller than the donor gate (64-wide, 16,640 parameters vs 128-wide,
32,768 parameters in BM3)?

### Arms (`recheck/graft_control.py`)
Per layer, on the S4D host (conv stem, LayerNorms, final norm and classifier are BearS4D's own):

| arm | layer | params |
|---|---|---|
| `s4d_plus_gate_wm` | x = W_x ln h, z = W_z ln h (64->128, no bias); y = S4D_128(x) (d_state 64); h += W_out(y * silu(z)) (128->64, no bias) | 198,978 |
| `s4d_plus_branch` | identical modules and parameters; h += W_out(y + silu(z)) | 198,978 |

- Gate width 128 = bm3_frozen's d_inner, and the donor passes z into the scan kernel
  (is_outproj_norm False), i.e. the donor gate is y * silu(z) at width 128. z's projection has
  32,768 parameters across the four layers, equal to the donor's z slice.
- The 128-wide S4D layer uses d_state 64, so its parameter count and total state size
  (128 x 64) equal the host's 64-wide, d_state-128 layer: the SSM budget is unchanged.
- Both arms are **larger** than bm3_frozen (112,410) and than BM3 (177,938). This biases the test
  toward recovery, i.e. against the paper's L3 claim. The additive arm shares every parameter and is
  identical at initialisation, so it isolates the multiplicative interaction.
- Construction check `recheck/graft_control_unitcheck.py` passed before this file was written
  (donor shape; parameter difference 0.0 %; hooks show the tensor entering W_out equals
  y * silu(z) for the gate arm and y + silu(z) for the branch arm and not the other formula, with z
  input-dependent; state-dict diff against BearS4D shows host keys identical in name, shape and value
  and only the expected removed/added keys; the two arms identical at initialisation; forward finite
  and every parameter receives a finite gradient). Snapshot: `recheck/unitcheck_stage3/`.

### Data
Original transition 37.5Hz11kN -> 40Hz10kN. Evaluation fingerprint must equal
`6c20b367522ce5db59720120f8a7cd12792c7e070c6ae84bd1985511998727de` (asserted per cell).

### Reference cells (already exist, not re-run)
`s4d` and `bm3_frozen`: stage-1 re-run (`recheck/stage1/cells_recheck.jsonl`), same seeds, levels,
epochs and build. `frozen_clti`: stage-2 (`recheck/stage2/cells_stage2.jsonl`).
Final-epoch denominators known at writing: bm3_frozen - s4d = +15.5 / +19.8 / +16.4 pp;
frozen_clti - s4d = +21.9 / +24.0 / +21.2 pp.

### Statistics
- R_arm(level) = (mean_arm - mean_s4d) / (mean_bm3_frozen - mean_s4d), final epoch, per level;
  **R_arm = mean of the three per-level values** (the original definition).
- Seed-resampled 95 % interval: 20,000 replicates, `numpy.random.default_rng(20261004)`; in each
  replicate the five seeds of each arm (gate_wm, branch, s4d, bm3_frozen) are resampled with
  replacement independently, per level; R is recomputed; 2.5 / 97.5 percentiles.
- dR = R_gate_wm - R_branch, computed within each replicate with shared s4d and bm3_frozen resamples.
- Secondary (no decision): R' with frozen_clti in place of bm3_frozen in the denominator.
- Also reported: paired differences gate_wm - s4d, branch - s4d, gate_wm - branch per level.

### Decision rules
1. **L3 tier from R_gate_wm** (point estimate, final epoch):
   - R >= 0.7 -> **L3 dead**: the title and the host-dependence claim are withdrawn.
   - 0.3 <= R < 0.7 -> **L3 partial**: "a width-matched graft recovers part of the gap"; title
     retained only if it does not assert non-transplantability (it does not at present).
   - R < 0.3 -> **L3 survives under width matching**.
2. **INCONCLUSIVE**: if the 95 % interval of R_gate_wm contains 0.3 or 0.7, the verdict is
   INCONCLUSIVE between the tiers the interval touches, and the manuscript is written at the
   **least favourable tier for L3 among those touched** (the highest-R tier).
3. **Gate specificity**: a gate-specific statement ("the multiplicative interaction, not the added
   width/projections, carries the recovery") is made only if dR > 0.1 **and** its 95 % interval
   excludes 0. Otherwise the gate/branch comparison is reported descriptively, and any recovery is
   attributed to the block shape (projections, width) rather than to the gate.
4. If R_gate_wm >= 0.3 and the specificity rule fails, the manuscript states that the added
   Mamba-style block structure, not the gate, recovers part of the gap.

## 2. Experiment 2 - reverse transition (75 cells)

### STEP 0 (done before writing)
Per-condition window counts under the harness's OR/IR labelling and onset rule:
35Hz12kN: OR 4416, IR 0 (no inner-race bearing; 1_4 cage and 1_5 mixed are excluded) -> unusable.
37.5Hz11kN: OR 4624, IR 624 (IR = Bearing2_1). 40Hz10kN: OR 3504, IR 1744 (IR = Bearing3_3, 3_4).
**Chosen pair: 40Hz10kN -> 37.5Hz11kN** (train Bearing3_1/3_3/3_4/3_5; test Bearing2_1/2_2/2_4/2_5).

Limits carried into every statement: the reverse pair uses the **same eight bearings** with roles
swapped, so it tests the other transition direction, not independent bearings; evaluation IR comes
from a single bearing (Bearing2_1). Wording: "replicated on the reverse transition", never
"independent replication".

### Data
Evaluation fingerprint frozen here:
`934248343b29cf7c988616f4683eea40f9e4f0dee4bf94a8ae6da69d5ec9dba3` (asserted per cell).

### Arms
`frozen_clti`, `s4d`, `frozen_nogate`, `s4d_plus_gate_wm` (as specified), plus **`bm3_frozen`**,
added in this file before any run so that L2 is the single-variable contrast it is on the original
pair (bm3_frozen - frozen_nogate); without it the only base, frozen_clti, differs from
frozen_nogate in two further terms (trapezoidal weight, angles).

### Decision rules
1. **L1 on the reverse transition**: count levels at which frozen_clti - s4d (final) is positive
   with a 95 % t-interval excluding 0.
   - >= 2 of 3 -> L1 may be written as **replicated on the reverse transition**.
   - 1 of 3 -> **partial** replication.
   - 0 of 3 -> L1 is **narrowed back to the original transition** and the reverse result is reported
     as a non-replication in the manuscript.
2. **L2** (no hard gate): bm3_frozen - frozen_nogate per level, final epoch, with intervals and seed
   directions; report the number of levels at which the mean loss is >= 8 pp. frozen_clti -
   frozen_nogate reported descriptively.
3. **L3** (descriptive only): R2 = (gate_wm - s4d)/(bm3_frozen - s4d) per level and mean, with the
   same resampling; R2' with frozen_clti in the denominator. A per-level R is reported as n/a if its
   denominator is below 2 pp.

## 3. Outputs
`recheck/stage3/cells_stage3_exp1.jsonl`, `cells_stage3_exp2.jsonl` (per-epoch curves),
`analyze_stage3.py`, tables, and a decision memo that addresses every rule above in order, with a
DEVIATIONS section. The manuscript is changed only after the memo is written.

## 4. Declared design choices (made before any run)
- d_state 64 for the 128-wide S4D layer (SSM budget equal to the host's); the alternative
  (d_state 128, 297k parameters) was rejected as a further capacity inflation.
- The additive control uses y + silu(z) through the same W_out, so that it has the same parameters,
  the same nonlinearity and the same initial weights as the gate arm.
- bm3_frozen added to experiment 2 (reason above).

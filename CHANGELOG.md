# Changelog

## 2026-10-04 — follow-up grids, retitle, and the four missing modules

### Repository completeness
- Added `bearmamba3/` (dataset, model, auxiliary loss), `baselines/`,
  `models_extended.py` and `noise_utils.py`. Without them
  `xjtu_noisy_harness.py` could not be imported from a fresh checkout of the
  initial upload (commit 593cb22). `bearmamba3/` and `baselines/` contain only
  the files this study imports, not the whole upstream packages.
- Removed `paper/main.pdf` and added `paper/main_sandbox_build.pdf`: the
  available build environment lacks the TS1 (text companion) fonts, which were
  substituted at build time, so the committed PDF is faithful in layout but not
  in every symbol glyph. Build locally for the authoritative PDF.

### Stage-1 follow-up grid (`recheck/stage1/`, run 2026-10-01)
The harness reports, for each cell, the maximum over epochs of macro-F1
evaluated on the held-out operating condition. There is no separate validation
split, so the epoch is chosen with evaluation data. This was not stated in the
initial manuscript. 60 cells (BM3-frozen, S4D, frozen−gate, S4D+gate ×
{0, −2, −6} dB × 5 seeds × 50 epochs) were re-run with the training loop
unchanged, logging macro-F1 at every epoch.

- Absolute levels under best-epoch reporting are optimistic by 1.4–9.5 pp.
- The reduction is of similar size for the arms being contrasted, so the
  central contrasts keep their sign: BM3-frozen − S4D is +15.5 / +19.8 / +16.4 pp
  at 0 / −2 / −6 dB under final-epoch reporting.
- Two statements weakened: gate removal falls below plain S4D only at −6 dB
  under final-epoch reporting, and the graft recovery ratio's seed-resampled
  interval reaches the pre-specified 0.3 bound (R = 0.196, [0.09, 0.31]).

### Stage-2 follow-up grid (`recheck/stage2/`, run 2026-10-04)
Independent review established that the `bm3_frozen` arm freezes the
input-dependent parts of Δ, A, B and C but leaves two further input-dependent
tensors in Mamba-3's scan: the trapezoidal weight and the rotation angles.
`recheck/layered_freeze.py` adds the three arms that freeze those too
(`frozen_ctrap`, `frozen_cangle`, `frozen_clti`), each verified by capturing the
tensors the scan receives for two different inputs through the same module and
requiring bitwise equality for whatever the arm claims to have frozen
(`recheck/layered_unitcheck.py`). 60 cells: those three arms plus the original
`bm3_kin` arm × {0, −2, −6} dB × 5 seeds × 50 epochs.

Freezing the remaining terms does not reduce the advantage over S4D:

- With no tensor entering the scan dependent on the input, `frozen_clti` is
  +21.9, +24.0 and +21.2 pp above S4D at 0, −2 and −6 dB under final-epoch
  reporting (5/5 seeds, intervals excluding zero) — a larger margin than the
  partially frozen arm's +15.5, +19.8, +16.4 pp.
- No freezing step costs accuracy. BM3-frozen minus each of the three arms is
  negative at all three levels (−1.8 to −6.4 pp), i.e. the more heavily frozen
  arm is numerically ahead, with every interval including zero.
- Not a capacity effect: `frozen_clti` has 103,842 parameters against 112,410
  for `bm3_frozen` and 177,938 for BM3.
- At −6 dB `frozen_clti` also exceeds the full BM3 block by 4.6 pp
  ([−7.2, −1.9] for BM3 − `frozen_clti`, 0/5 seeds in BM3's favour). Reported,
  not built on.
- The re-run reproduces the original grids: the BM3 arm's best-epoch means
  (95.6, 95.8, 88.6) match the original 95.1, 95.4, 88.5 to within 0.5 pp.
- Δ(BM3 − frozen) under final-epoch reporting is +3.7, +0.9, +0.2 pp
  (best-epoch re-run +5.1, +3.5, +3.0), so the point estimates fall further
  below the 10 pp threshold rather than rising towards it.

The two readings this does not separate — input-dependence contributes nothing
measurable here, or what it contributes is recovered by learned constants under
this budget — lead to the same attribution and are both stated in the paper.

### Manuscript
Retitled from *"Necessary but Not Transplantable: Deep-Noise Robustness of
Mamba-Style Blocks Is a Component-Interaction Effect"*. The claim structure was
rescoped to what the interventions support:

- The first step is a statement about Δ, A, B, C only, not about input
  selectivity as a whole, and it is a failure to meet a confirmation criterion
  rather than a demonstration that freezing is cheap (no equivalence margin was
  pre-specified).
- Gate removal and gate addition are not matched interventions (128-wide with
  its own projection and 32,768 parameters in BM3; 64-wide with 16,640
  parameters in the S4D host), so "not transplantable" was replaced by a
  host-dependence reading, with a four-arm host-by-gate contrast reported.
- "Preregistered" was replaced by "pre-specified", with the strength of the
  supporting evidence stated; claims of byte-level identity were narrowed to
  what the unit checks actually compare; the 0.7 pp process figure is reported
  as a single repeat rather than a reproducibility floor.
- The data composition is now stated: training has three OR bearings and one IR
  bearing, evaluation two of each, and the five seeds do not resample bearings.

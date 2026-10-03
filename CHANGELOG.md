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
`recheck/layered_freeze.py` adds the arms that freeze those too, and the grid
also re-runs the original `bm3_kin` arm so that Δ(BM3 − frozen) no longer rests
on best-epoch values alone.

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

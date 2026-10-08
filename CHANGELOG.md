# Changelog

## 2026-10-08 — stage 5: reverse native additive substitution, final-epoch re-assessment, manuscript v8

Pre-specified in `recheck/prereg_stage5_metric_reverse.md` (sha256 f3b66a83…, committed in
9750930 before the first cell, together with the title branch rule; unchanged afterwards).
60 cells, 5.05 GPU-h (12 h fuse not reached). Final-epoch reporting is primary. Rule-by-rule
decisions with DEVIATIONS: `recheck/stage5_decision_memo.md`.

- Experiment 3 (reverse transition, 15 cells). `bm3_frozen_add` against the stage-3 reverse
  `bm3_frozen` / `frozen_nogate`: additive - gateless = +0.3 / -0.3 / +0.6 pp (all intervals
  include zero); rho_rev = 0.009 [-0.943, 0.362] -> NOT RESCUED. Both directions are now not
  rescued; title branch A ("Necessary in Its Native Block"; candidates A1/A2, final choice by the author).
- Experiment 2 (original transition, 45 cells). `frozen_noconv`, `frozen_As4d`, `s4d_wide` under
  final-epoch reporting: no change of sign against the best-epoch reference; S4D-wide capacity
  control holds (BM3-frozen - S4D-wide excludes zero at 3/3 levels). Tables 2-4 move to the
  final-epoch rule; best epoch becomes a sensitivity analysis.
- Analysis byte-identical across PYTHONHASHSEED 1 and 2. Manuscript numbers computed from
  per-cell values (`numbers_stage5.py`) and independently re-checked (`verify_v8.py`, 0 mismatches).
- Manuscript v8: title branch A; abstract, introduction, Tables 2-4, sensitivity section,
  native-host section (reverse block), discussion, boundaries, conclusion, highlights; Figures 3-4
  redrawn from final-epoch cells in `recheck/` (set `P9_RECHECK_DIR` to override); two discussion
  sentences downgraded to untested hypotheses; Appendix C compressed with the full incident text
  moved to `paper/supplementary_S1_incidents.tex`; draft `paper/cover_letter.tex` disclosing the
  related paper-8 manuscript.
- `docs/`: desk check of CWRU/PU as an independent dataset (no runs) and the Neurocomputing scope check.

## 2026-10-06 — stage 4: gate -> additive substitution in the native host, reverse additive control

Pre-specified in `recheck/prereg_native_additive.md` (sha256 cafcc974…, committed in
9371ba6 before the first cell, together with the title decision tree; unchanged afterwards).
30 cells, 4.1 GPU-h (15 h fuse not reached). Final-epoch reporting is primary.
Rule-by-rule decisions with DEVIATIONS: `recheck/stage4_decision_memo.md`.

- Experiment 3 (original transition, 15 cells). `bm3_frozen_add` replaces y * silu(z) by
  y + silu(z) inside BM3-frozen; parameters (112,410) and initial weights bitwise identical.
  Native rescue ratio rho = 0.024 [-1.159, 0.486] -> NOT RESCUED (rho < 0.3 and upper bound
  < 0.5). The additive arm is indistinguishable from removing z (+0.6 / -0.2 / +0.4 pp).
- Experiment 4 (reverse transition, 15 cells). `s4d_plus_branch` recovers R2 = 0.758
  [0.657, 0.884] against 0.911 for the width-matched gate; dR2 = 0.153 [0.057, 0.262]
  (best-epoch 0.037 [-0.096, 0.172]) -> pre-specified label "unresolved": a small,
  direction-dependent gate increment in the S4D host.
- Title decision tree executed (NOT RESCUED tier): the manuscript title returns to
  "Necessary but Not Transplantable"; the subtitle is a draft for the author to confirm.
- One launch was lost to a sandbox kernel restart before any cell was recorded and was
  relaunched unchanged.
- Stage-2 entry below: freezing-cost range corrected to -1.8..-6.3 pp (per-cell recomputation).

## 2026-10-05 — stage 3: width-matched graft control and reverse transition

Pre-specified in `recheck/prereg_graft_control_pair2.md` (sha256 720898b9…,
committed in a10f477 before the first cell; unchanged afterwards). 105 cells,
20.1 GPU-h (30 h fuse not reached). Final-epoch reporting is primary.
Decisions, rule by rule, with a DEVIATIONS section: `recheck/stage3_decision_memo.md`.

- Experiment 1 (original transition, 30 cells). `s4d_plus_gate_wm` rebuilds the
  graft in the donor's shape (x/z at width 128, S4D at width 128 with state 64,
  output projection); `s4d_plus_branch` is identical but additive
  (y + silu(z)). Both 198,978 parameters. R_gate_wm = 0.296 [0.159, 0.515]:
  the interval crosses 0.3, so the verdict is INCONCLUSIVE and the manuscript is
  written at the less favourable tier (partial recovery). R_branch = 0.408;
  dR = -0.112 [-0.263, 0.022], so no gate-specific statement is made.
- Experiment 2 (reverse transition 40Hz10kN -> 37.5Hz11kN, 75 cells; same eight
  bearings with roles swapped, eval hash 93424834…). frozen_clti - S4D =
  +26.3 / +25.6 / +22.2 pp, 3/3 levels with intervals excluding zero:
  replicated on the reverse transition. Gate removal from bm3_frozen costs
  18.4 / 22.0 / 24.0 pp (3/3 levels >= 8 pp). Width-matched graft R2 = 0.911,
  descriptive only (no additive control in this direction).
- `analyze_stage3.py`: the seeded bootstrap initially depended on Python's
  string-hash randomisation through set ordering; fixed to a sorted order and
  checked byte-identical across PYTHONHASHSEED values. Decisions unchanged.
- Manuscript: new subsections on the width-matched graft and the reverse
  transition; abstract, introduction, discussion, boundaries, conclusion,
  highlights, Figure 1 and Figure 4 title updated.

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
  negative at all three levels (−1.8 to −6.3 pp), i.e. the more heavily frozen
  arm is numerically ahead, with every interval including zero.
- Not a capacity effect: `frozen_clti` has 103,842 parameters against 112,410
  for `bm3_frozen` and 177,938 for BM3.
- At −6 dB `frozen_clti` also exceeds the full BM3 block by 4.5 pp
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

# Pre-specification: stage 5 - reverse-direction native additive substitution (exp. 3) and final-epoch re-assessment of the remaining Table 2-4 arms (exp. 2)

Paper 9, stage 5. Approved by the author on 2026-10-08 (task card `_tasks/p9-stage5-metric-unify-reverse-add-2026-10-08`).
Written **before any training cell of either grid**; no stage-5 result exists at the time of writing.
Pre-run checks (no training) quoted as design facts: the stage-4 construction check of `bm3_frozen_add`
re-passed in the current environment (`unitcheck_stage5/`); `build_model` parameter counts equal the
original grids (s4d_wide 178,498; frozen_noconv 112,026; frozen_As4d 112,410); eval fingerprints
recomputed: forward `6c20b367522c`, reverse `934248343b29` (equal to stages 1-4). Mamba source pinned to
the same upstream commit as stages 2-4 (`e9594ce`). This file is committed and pushed to
`github.com/jeff-1980/bm3-jizi` before the first cell; sha256 and local mtime are in
`prereg_stage5_metric_reverse.sha256`. It is not edited after launch; departures go to DEVIATIONS.

## 0. Fixed conditions
- Harness `xjtu_noisy_harness.py` imported unmodified; AWGN 0, -2, -6 dB; seeds 0-4; 50 epochs.
- **Primary rule: final epoch.** Best-epoch reported alongside, never used for a decision.
- Paired differences: seed mean, 95 % t-interval (df 4), seeds with positive difference.
- Bootstrap: sorted-arm-order implementation of stage 4 (`analyze_stage4.ratio_boot` logic),
  20,000 replicates, `numpy.random.default_rng(20261008)`; a level whose resampled denominator is
  < 2 pp is dropped from that replicate's mean (fraction reported). Output checked byte-identical under
  two PYTHONHASHSEED values.
- Manuscript numbers are computed directly from per-cell values, never re-rounded from an aggregate.
- **Fuse: 12 GPU-h** summed over stage-5 cells (wall time, contention included). GPU memory logged at launch.
- Order: experiment 3 first (it decides the title branch), then experiment 2 (frozen_noconv,
  frozen_As4d, s4d_wide). Smoke runs go to separate files and are excluded.

## 1. Experiment 3 - native additive substitution on the reverse transition (15 cells; decisive)
Arm `bm3_frozen_add` (stage-4 code, unchanged) on 40Hz10kN -> 37.5Hz11kN. References, not re-run:
`bm3_frozen`, `frozen_nogate` from stage 3 (`recheck/stage3/cells_stage3_exp2.jsonl`). Known at
writing (final): frozen 74.7 / 76.7 / 73.9, nogate 56.2 / 54.7 / 50.0; denominators 18.4 / 22.0 / 24.0 pp.

rho_rev = (add - nogate)/(frozen - nogate), per level, mean over levels, seed-resampled 95 % interval.
- **NOT RESCUED**: rho_rev < 0.3 and upper bound < 0.5.
- **RESCUED**: rho_rev >= 0.7 and lower bound > 0.5.
- otherwise **PARTIAL/INCONCLUSIVE**, written at the tier less favourable to the claim.
Writing rule (both directions): the conclusion is carried by the paired differences add - nogate and
frozen - add per level; rho is reported as a summary.

## 2. Experiment 2 - final-epoch re-assessment, original transition (45 cells)
Arms: `frozen_noconv` and `frozen_As4d` (Table 3 single-component removals), `s4d_wide` (capacity
control in the protocol section). Paired references (final epoch, same build, same seeds): `bm3_frozen`
and `s4d` from stage 1 (`recheck/stage1/cells_recheck.jsonl`).

Reference directions in the manuscript (best-epoch, original grids):
| contrast | 0 dB | -2 dB | -6 dB |
|---|---|---|---|
| frozen - frozen_noconv | -6.6 | -2.7 | +7.9 |
| frozen - frozen_As4d | -2.4 | -2.2 | -4.2 |
| s4d_wide - s4d | +1.8 | -1.7 | +0.5 |

Decision (descriptive):
- For frozen_noconv and frozen_As4d, a **flip** at a level is a sign change of the mean of
  frozen - arm between the reference (best-epoch, above) and the final-epoch value, counted only where
  both absolute values are >= 1.0 pp. No flip at any level -> the arm's entries in the main tables move
  to final epoch with unchanged wording. A flip at any level -> that arm's statement is downgraded and
  reported in the boundaries section; the favourable reporting rule is not chosen.
- For s4d_wide the manuscript claim is that this capacity increase does not explain the gap to the
  frozen block. It **holds** if bm3_frozen - s4d_wide (final) is positive with an interval excluding
  zero at >= 2 of 3 levels; otherwise it is downgraded. s4d_wide - s4d is reported alongside.
- If all three arms hold, Tables 2-4 are reported under the final-epoch rule for the 0/-2/-6 dB band,
  with best-epoch values as sensitivity analysis. Per-class grid and +10/+6/clean/-10 dB levels are not
  re-run and are labelled auxiliary (best-epoch) in the text.

## 3. Title branch (author's rule; final choice by the author)
The author ruled on 2026-10-08 that the absolute form "Not Transplantable" is dropped in every branch
(recorded as a deviation from the stage-4 decision tree, which did not take into account the same
batch's reverse result dR2 = 0.153 [0.057, 0.262], and because the body has avoided the phrase since
2026-10-05).

| Experiment 3 | Title family | Candidates (drafted now, before any result) |
|---|---|---|
| NOT RESCUED (so both directions) | "Necessary in Its Native Block" | A1: *Necessary in Its Native Block: The Multiplicative Gate, Not Input Selectivity, Underlies Deep-Noise Robustness of a Mamba-3 Block under Matched-Noise Cross-Condition Drift*; A2: *Necessary in Its Native Block: Selectivity and Gating in a Mamba-3 Block under Matched-Noise Cross-Condition Drift* |
| RESCUED or PARTIAL/INCONCLUSIVE | "Gating Benefits Depend on the Backbone" | B1: *Gating Benefits Depend on the Backbone: An Ablation Study of a Mamba-3 Block under Matched-Noise Cross-Condition Drift*; B2: *Gating Benefits Depend on the Backbone: Selectivity, Gating and Host Structure in a Mamba-3 Block under Matched-Noise Drift* |

Abstract, introduction, discussion, boundaries, conclusion and highlights follow the branch and are
checked against the verb-boundary table of the project card.

## 4. Declared choices and scope
- Cell count is 60 (15 + 45), not the approximate 45 in the task card: the card names three arms for
  experiment 2 (S4D-wide, conv-stem and heavy-tailed-A ablations), which is 45 cells. Experiment 2 runs
  after experiment 3 and is covered by the 12 GPU-h fuse; if the fuse trips, the remaining cells are
  reported as not run.
- Not in scope: independent-dataset runs (desk check only), SNR-mismatch experiments, per-class and
  out-of-band re-runs, final title choice, local PDF build.

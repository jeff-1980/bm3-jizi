# Pre-specification: stage 8 - core contrasts under independent training/evaluation noise, and same-host gate tests in the fully frozen block

Paper 9, stage 8. Approved by the author on 2026-10-09 in response to the simulated Neurocomputing review of
the same day (tiers 1 and 2, fuse raised to 15 GPU-h). Written **before any stage-8 training cell** (only
2-epoch pipeline smoke runs exist, in `results/smoke8.jsonl`, excluded from every analysis). Committed and
pushed to `github.com/jeff-1980/bm3-jizi` before the first grid cell; sha256 and mtime in
`prereg_stage8_indep_noise.sha256`. Not edited after launch; departures go to DEVIATIONS.

## 0. Why, and the single protocol change
In every earlier grid the noise stream was keyed on `[seed, idx]` and the training and evaluation wrappers received
the same seed, so window `idx` of the training set and window `idx` of the evaluation set carried the same base noise
vector (correlation 1.0 by construction; verified). A pre-run audit (no training) measured how much label information
this could carry: per-index train/test label agreement 0.549 on XJTU-SY (below the 0.628 chance level; a predictor
that only memorised noise would score 35.4 % macro-F1), 0.999 on PU disjoint and 1.000 on PU shared.
Stage 8 keys the stream on `[seed, split, idx]` (`indep_noise.py`; train split 101, test split 202). Training and
evaluation noise are independent (synthetic check: mean |corr| 0.016); all arms with a given seed see the same
evaluation noise (common evaluation bank); clean inputs are unchanged. Everything else - harness training code,
data, windows, z-scoring, SNR scaling, sampler, optimiser, schedule, 50 epochs, seeds 0-4, AWGN 0/-2/-6 dB - is as in
the earlier grids. The harness is imported unmodified.

## 1. Grids (150 cells; run order A, C, B; seed-major within each grid)
- **A** XJTU-SY 37.5 Hz/11 kN -> 40 Hz/10 kN (eval label sha `6c20b367522c`): `bm3_frozen`, `s4d`,
  `frozen_nogate`, `frozen_clti`, `bm3_frozen_add` (75 cells).
- **C** same transition: `frozen_clti_nogate` (z rows removed from frozen_clti's in_proj, 71,074 parameters) and
  `frozen_clti_add` (frozen_clti with y*silu(z) -> y+silu(z), 103,842 parameters, initial weights identical to
  frozen_clti) - `clti_gate.py` (30 cells). Construction check `stage8_unitcheck.py`: all 11 checks pass,
  including bitwise-equal initial weights to frozen_clti, the combination entering out_proj, and Q, K, ADT, DT,
  Trap, Angles bitwise input-independent at the kernel (V input-dependent).
- **B** PU shared bearings {KA04, KA16, KI04, KI14}, N15_M07_F10 -> N09_M07_F10 (eval label sha `addc1938057e`):
  `bm3_frozen`, `s4d`, `frozen_nogate` (45 cells).
- Fuse: cumulative stage-8 GPU time (including smoke) > 15 h stops the run; remaining cells reported as not run.

## 2. Decision rules (final epoch primary; best epoch reported alongside, never decisive)
Paired differences: seed mean, 95 % t-interval (df 4), seeds positive. Contrast label:
**HOLDS** if >= 2 of 3 levels have positive mean with interval excluding zero and no level is reversed;
**WEAKENED** if exactly 1 such level and no reversal; **DOES NOT HOLD** if 0 and no reversal;
**REVERSED** if any level has negative mean with interval excluding zero.
Ratios use a **paired-seed bootstrap** (the same resampled seed indices for every arm and level, all 5^5 = 3125
resamples enumerated; a level whose resampled denominator is < 2 pp is dropped from that resample) - this answers
review point 6; the independent-cell bootstrap of earlier stages is also reported for comparison.

Grid A (independent noise):
- **L1p** bm3_frozen - s4d;  **L1f** frozen_clti - s4d;  **L2** bm3_frozen - frozen_nogate.
- **L3b** native additive substitution: pointwise bm3_frozen_add - frozen_nogate, and
  rho = mean over levels of (add - nogate)/(frozen - nogate). NOT RESCUED if rho < 0.3 and paired upper bound < 0.5;
  RESCUED if rho >= 0.7 and paired lower bound > 0.5; otherwise INCONCLUSIVE; **not assessable** if L2 does not HOLD.
Grid C (same-host test, review point 3):
- **L2c** frozen_clti - frozen_clti_nogate (rule above).
- **L3c** rho_c = mean over levels of (clti_add - clti_nogate)/(frozen_clti - clti_nogate), rule as L3b.
- Descriptive: frozen_clti_nogate - s4d (does the fully frozen block without its gate fall to S4D?).
Grid B: **L1''** bm3_frozen - s4d; **L2''** bm3_frozen - frozen_nogate (rule above).
Descriptive throughout: independent-noise minus shared-noise means per arm and level (different noise draws, so
reported without a paired test).

## 3. Writing consequences (manuscript v11)
- Contrasts that HOLD under independent noise are reported with independent-noise values in the main tables; the
  shared-noise results of the same arms are kept in a clearly labelled appendix table ("earlier protocol: training and
  evaluation noise shared by index"). Arms not re-run keep their earlier values, labelled with that protocol.
- Any L1p, L1f, L2 or L1''/L2'' that is WEAKENED, DOES NOT HOLD or REVERSED: that claim is downgraded in the abstract,
  results and conclusion, and the title and abstract are returned to the author before being finalised.
- L3b NOT RESCUED keeps the native-host statement ("rescue below the pre-specified criterion", not "rescues nothing").
- Combined attribution in the fully frozen block ("its advantage over S4D depends on the multiplicative gate") is
  written only if L2c HOLDS and L3c is NOT RESCUED; otherwise the two findings (fully frozen advantage; gate role in
  the partially frozen host) are stated separately and the combined claim is removed.
- Independently of the outcome, v11 applies the review's text corrections: XJTU bearing identity (training and
  evaluation bearings disjoint within each direction), "scan dynamics and read/write coefficients" instead of "every
  tensor entering the scan", PU settings described as two protocols (not a single-factor change), paired-seed ratio
  intervals, threshold language instead of equivalence language, the reverse-direction S4D gate increment stated,
  and the minor items. Title candidate (review's neutral form) proposed; final title by the author.

## 4. Scope
150 grid cells. Not in scope: re-running the reverse transition, other noise types or levels, the per-class grid,
new datasets, final title, local PDF build.

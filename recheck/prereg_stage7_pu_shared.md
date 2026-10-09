# Pre-specification: stage 7 - PU diagnostic with shared bearings (bearing non-overlap vs condition drift)

Paper 9, stage 7. Approved by the author on 2026-10-09 (task card `_tasks/p9-stage7-pu-shared-bearing-diagnostic-2026-10-09`).
Written **before any stage-7 training cell, including the smoke check**. Design facts below come from data loading
only. This file is committed and pushed to `github.com/jeff-1980/bm3-jizi` before the first cell; sha256 and local
mtime are in `prereg_stage7_pu_shared.sha256`. It is not edited after launch; departures go to DEVIATIONS.

## 0. Question and single change
Stage 6 (disjoint bearings, 1500 -> 900 rpm) did not replicate and every arm transferred near the floor. All
XJTU-SY transitions share bearing identity between training and evaluation. This stage changes exactly one
factor relative to stage 6: training and evaluation use the **same** four bearings.
- Bearings: KA04, KA16 (outer race, 0), KI04, KI14 (inner race, 1) = the stage-6 training set, chosen in
  advance because it is class-balanced (2 + 2) and its source-condition data are identical to stage 6.
- Source N15_M07_F10 (training), target N09_M07_F10 (evaluation). Different recordings, so no window is shared
  (checked: 0 identical windows). Preprocessing identical to stage 6 (`pu_dataset.py`: vibration_1, 25.6 kHz,
  2048-sample non-overlapping windows, 20 recordings per bearing and condition).
- Window counts: train 4,005 (OR 2,001 / IR 2,004), test 4,003 (OR 2,001 / IR 2,002). Majority-class macro-F1
  on the test set: 33.3 %; a balanced random classifier scores about 50 %.
- Fingerprints (asserted at launch): label-only eval sha256 `addc1938057e...` (recomputed with the stage-6
  procedure); content train `3a40737977fb...` (identical to stage 6), test `908602165ad8...`.
- Arms: `s4d`, `bm3_frozen`, `frozen_nogate` (unchanged code; 100,162 / 112,410 / 79,642 parameters).
- AWGN 0, -2, -6 dB, matched train/test SNR; seeds 0-4; 50 epochs; 45 cells. Final epoch primary; best epoch
  alongside, never used for a decision. Paired differences: seed mean, 95 % t-interval (df 4), seeds positive.
- Fuse: smoke + grid GPU time > 12 h stops the run; remaining cells reported as not run.

## 1. Smoke check (runs first; excluded from analysis)
`s4d` and `bm3_frozen`, seed 99, clean / 0 dB / -6 dB, 50 epochs, final epoch. The grid is **not** run only if
the task is at ceiling (`s4d` >= 97 % at both 0 and -6 dB), because then no contrast is resolvable. The floor
question is not decided on one seed: it is decided by D1 on the full grid. The smoke report must state the
margin of each of the stage-6 criteria - learnable (best clean arm vs 60 %), not-floor (`bm3_frozen` at 0 dB vs
50 %), not-ceiling (`s4d` at 0 / -6 dB vs 97 %) - and compare them with the stage-6 margins (learnable +0.3 pp).

## 2. D1 - transferability gate (decided first, on the grid)
An arm-level is "off the floor" if its 5-seed final-epoch mean is >= 65 % **and** the one-sided 95 % lower
t-bound (df 4) of the mean is > 60 %. Rationale: a balanced random classifier scores ~50 % and the majority-class
predictor 33.3 %; stage 6 placed every arm-level mean at 47.7-56.9 % and no single cell above 60.2 %, so the bar
is set clearly above that band while remaining far below ceiling.
**D1 passes** if at least one arm is off the floor at >= 2 of the 3 noise levels. Otherwise **PLATFORM-LIMITED**:
this condition pair is not transferable on PU for any examined setting, the confound cannot be resolved here, and
branch F applies. Which arms pass at which levels is reported in either case.

## 3. D2 - effect replication (only if D1 passes)
Label per contrast: REPLICATED if >= 2 of 3 levels have positive mean with interval excluding zero; PARTIAL if
exactly 1; NOT REPLICATED if 0; any level with negative mean and interval excluding zero is a **reversal**.
- L1'': Delta(bm3_frozen - s4d).   - L2'': Delta(bm3_frozen - frozen_nogate).

## 4. Branches
- **R**: D1 passes, at least one of L1''/L2'' REPLICATED, and no reversal in either.
- **F**: D1 fails (PLATFORM-LIMITED), or both NOT REPLICATED, or any reversal.
- **INCONCLUSIVE**: every other combination (e.g. PARTIAL with no REPLICATED and no reversal); written at the F
  tier and returned to the author.

## 5. Writing consequences (manuscript v10)
- All branches: the abstract states that the disjoint-bearing PU setting did not replicate; Section 4.8 becomes
  "External validity: two PU settings" (disjoint vs shared bearings) with a table holding both; the boundaries
  section is rewritten per branch; the PU subsection and all labels stay.
- R: title A2 kept; the abstract and boundaries state the domain as cross-condition drift with shared bearing
  identity, shown on XJTU-SY and PU; the disjoint-bearing failure is reported alongside, worded after checking the
  stage-6 numbers against the verb-boundary table.
- F (and INCONCLUSIVE): two conditional title candidates and matching abstract / conclusion / highlights drafted
  and returned to the author; nothing finalised; the scope is narrowed to XJTU-SY using the stage-6 memo sentence.

## 6. Scope
45 grid cells + 6 smoke cells. Not in scope: load-change pairs, other datasets, re-running the disjoint grid,
`frozen_clti` / `bm3_frozen_add`, final title, local PDF build.

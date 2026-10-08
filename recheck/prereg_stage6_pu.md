# Pre-specification: stage 6 - replication on Paderborn (PU) real-damage bearings, disjoint train/test bearings

Paper 9, stage 6. Approved by the author on 2026-10-08 (task card `_tasks/p9-stage6-pu-replication-title-2026-10-08`).
Written **before any training cell on PU, including the smoke check**; no PU model has been trained at the
time of writing. Design facts quoted below come from data loading only (no training). This file is committed
and pushed to `github.com/jeff-1980/bm3-jizi` before the first cell; sha256 and local mtime are in
`prereg_stage6_pu.sha256`. It is not edited after launch; departures go to DEVIATIONS in the decision memo.

## 0. Data and fixed conditions
- Dataset class `recheck/pu_dataset.py` (`PUDatasetBinary`), same interface as the XJTU dataset, so the
  harness's noise wrapper, z-scoring, class-balanced sampler, optimiser, schedule and evaluation run unchanged
  (`xjtu_noisy_harness.py` imported unmodified).
- Real-damage bearings only (accelerated lifetime tests); outer race (KA) = 0, inner race (KI) = 1.
  Train: KA04, KA16, KI04, KI14 at N15_M07_F10 (1500 rpm, 0.7 Nm, 1000 N).
  Test: KA15, KA22, KI17, KI18, KI21 at N09_M07_F10 (900 rpm, same load and radial force).
  Train and test bearings are disjoint (asserted). Speed change in the same direction family as XJTU-SY.
- Channel `vibration_1` (64 kHz) resampled to 25.6 kHz (`resample_poly`, 2/5), so a 2048-sample window spans
  80 ms as in XJTU-SY; non-overlapping windows; all 20 recordings per bearing and condition.
- Window counts (data loading only): train 4,005 (OR 2,001 / IR 2,004); test 5,015 (OR 2,006 / IR 3,009;
  KA15 1,000, KA22 1,006, KI17 1,006, KI18 1,001, KI21 1,002). Majority-class macro-F1 on the test set = 37.5 %.
- Fingerprints (asserted by the driver at every launch): label-only eval sha256 `53d565d9520a...`; content
  sha256 (windows + labels) train `3a40737977fb...`, test `54402e68f661...`.
- Known caveat (stated in advance): KI04 and KI14 are documented as damaged in the same accelerated-lifetime
  run family and some literature treats them as near-duplicates; their recordings are not identical
  (checked: different signals). Both are training bearings, so this cannot leak into the test set.
- Arms (5): `bm3_frozen`, `frozen_clti`, `s4d`, `frozen_nogate`, `bm3_frozen_add` - the stage-2/4 code,
  unchanged; parameter counts 112,410 / 103,842 / 100,162 / 79,642 / 112,410 (equal to the XJTU runs).
- AWGN at 0, -2, -6 dB, matched train/test SNR as in XJTU; seeds 0-4; 50 epochs; 75 cells.
- **Primary rule: final epoch.** Best epoch reported alongside, never used for a decision.
- Paired differences: seed mean, 95 % t-interval (df 4), seeds with positive difference.
- Bootstrap for ratios: the sorted-order implementation of stages 4-5, 20,000 replicates,
  `numpy.random.default_rng(20261009)`; a level whose resampled denominator is < 2 pp is dropped from that
  replicate. Output must be byte-identical under PYTHONHASHSEED 1 and 2.
- Fuse: cumulative GPU time of smoke + grid > 12 h stops the run; remaining cells reported as not run.
- Absolute values are not compared with XJTU; only directions and intervals are judged.

## 1. Smoke check (runs first; excluded from every analysis)
Arms `s4d` and `bm3_frozen`, seed 99 (not a grid seed), conditions clean, 0 dB and -6 dB, 50 epochs, final epoch.
Proceed to the grid only if all three hold:
- (a) learnable: at least one arm reaches clean macro-F1 >= 60 %;
- (b) not at floor: `bm3_frozen` at 0 dB >= 50 % (majority-class floor 37.5 %);
- (c) not at ceiling: not (`s4d` >= 97 % at both 0 dB and -6 dB).
If any fails, the grid is **not run**, the outcome is reported to the author, and no alternative pair or
protocol is tried in this stage.

## 2. Decision rules (final epoch, the three noise levels)
A contrast is **REPLICATED** if at >= 2 of 3 levels the mean is positive and its 95 % interval excludes zero;
**PARTIAL** if exactly 1 of 3; **NOT REPLICATED** if 0 of 3; **REVERSED** if at >= 2 of 3 levels the mean is
negative with the interval excluding zero (REVERSED takes precedence).
- **L1' (full freeze, headline)**: Delta(frozen_clti - s4d). **L1' (partial freeze)**: Delta(bm3_frozen - s4d).
  Both are reported; the label of each is stated separately.
- **L2' (gate removal)**: Delta(bm3_frozen - frozen_nogate), same rule. The number of levels >= 8 pp is
  reported descriptively.
- **L3b' (native additive substitution)**: rho_PU = mean over levels of (add - nogate)/(frozen - nogate),
  seed-resampled 95 % interval. NOT RESCUED if rho < 0.3 and upper bound < 0.5; RESCUED if rho >= 0.7 and
  lower bound > 0.5; otherwise INCONCLUSIVE. If L2' is NOT REPLICATED or REVERSED, L3b' is labelled
  **not assessable** (no gate effect to rescue), and rho is reported descriptively only.
  The pointwise contrast add - nogate (mean, interval) carries the textual statement, as in stage 5.

## 3. Writing consequences (fixed now)
- All of L1' (full freeze), L2' REPLICATED and L3b' NOT RESCUED: the external-validity paragraph is upgraded
  to "two datasets, two transitions on XJTU-SY, disjoint train/test bearings on PU"; the abstract may add one
  sentence on the PU replication, with verbs bounded by the verb-boundary table.
- Any level PARTIAL / INCONCLUSIVE: reported level by level with that label; the abstract mentions PU only with
  the label attached.
- Any NOT REPLICATED / REVERSED / not assessable: reported as such in a dedicated subsection and in the
  boundaries section; claims are scoped back to XJTU-SY; **the decision whether to keep the PU section or
  re-scope the paper is returned to the author before any further manuscript change beyond reporting**.
- Title stays A2 unless an unfavourable branch is reached, in which case the title is also returned to the author.

## 4. Scope
- 75 grid cells + 6 smoke cells. Not in scope: other PU pairs, artificial-damage bearings, per-class
  analysis (except if available at no cost from stored predictions - none are stored, so not done),
  SNR-mismatch, final title change, local PDF build.

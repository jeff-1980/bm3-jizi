# Pre-specification: stage 9 - reverse transition under independent training/evaluation noise

Paper 9, stage 9. Task card `_tasks/p9-stage9-reverse-indep-noise-final-2026-10-10` (approved by the author on
2026-10-10). The card's default three arms were extended to four **before** this file was written: the
manuscript's reverse-direction statements cite both the partially frozen (bm3_frozen) and the fully frozen
(frozen_clti) block, and gate removal needs bm3_frozen, so frozen_clti cannot replace an arm. The author approved
the fourth arm (+15 cells) and the fuse below on 2026-10-10, before any stage-9 cell.
Written **before any stage-9 training cell, including the smoke check**. Committed and pushed to
`github.com/jeff-1980/bm3-jizi` before the first cell; sha256 and mtime in `prereg_stage9_reverse_indep.sha256`.
Not edited after launch; departures go to DEVIATIONS. **The fuse is fixed here and will not be changed after launch.**

## 0. Fixed conditions
- XJTU-SY reverse transition 40 Hz/10 kN -> 37.5 Hz/11 kN: training Bearing3_1, 3_3, 3_4, 3_5; evaluation
  Bearing2_1, 2_2, 2_4, 2_5 (disjoint); same split function, preprocessing and bearing roles as the stage-3 reverse
  grid; evaluation label sha `934248343b29` (asserted at launch).
- Noise protocol as stage 8: stream keyed on [seed, split, idx] (`indep_noise.py`, sha256 prefix fc44ccee0806),
  training and evaluation independent, one evaluation noise bank per seed shared by all arms.
- Harness `xjtu_noisy_harness.py` imported unmodified; AWGN 0, -2, -6 dB; seeds 0-4; 50 epochs.
- Arms (4): `s4d` (100,162 parameters), `bm3_frozen` (112,410), `frozen_nogate` (79,642), `frozen_clti` (103,842),
  the code used in stages 2-8, unchanged. 60 grid cells.
- **Primary rule: final epoch.** Best epoch reported alongside, never used for a decision.
- Paired differences: seed mean, 95 % t-interval (df 4), seeds with a positive difference.
- Analysis output must be byte-identical under PYTHONHASHSEED 1 and 2.
- **Fuse: cumulative stage-9 GPU time (smoke + grid) > 9 h stops the run**; remaining cells are reported as not run.
  Expected ~7.3 h on an unshared GPU (stage-3 reverse cells averaged 5.3-7.4 min).

## 1. Smoke check (runs first; excluded from every analysis)
Arms `s4d` and `bm3_frozen`, seed 99 (not a grid seed), 0 dB and -6 dB, 50 epochs, final epoch. Proceed to the grid
only if all hold: (a) every smoke cell finishes with 50 finite per-epoch values and the asserted label sha;
(b) not at the floor: `bm3_frozen` at 0 dB >= 60 % (stage-3 reverse mean 74.7 % under the earlier protocol);
(c) not at the ceiling: `s4d` < 97 % at both levels. Margins are reported. If any fails, **no grid cell is run** and
the outcome is returned to the author.

## 2. Decision rules (final epoch; the three levels)
For each contrast: **HOLDS** if >= 2 of 3 levels have a positive mean with interval excluding zero and no level is
reversed; **WEAKENED** if exactly 1 and no reversal; **DOES NOT HOLD** if 0 and no reversal; **REVERSED** if any level
has a negative mean with interval excluding zero.
- **L1f-rev** frozen_clti - s4d.  **L1p-rev** bm3_frozen - s4d.  **L2-rev** bm3_frozen - frozen_nogate.
- Descriptive: frozen_nogate - s4d; independent-noise minus earlier-protocol means per arm and level (stage-3
  reverse cells; different noise draws, no paired test).

## 3. Writing consequences (fixed now)
- HOLDS: the reverse-direction numbers in the manuscript are updated to the independent-noise values; structure
  unchanged.
- WEAKENED or DOES NOT HOLD: the reverse statement for that contrast is narrowed level by level ("resolved at ... only"),
  mirroring the stage-8 wording; "both transitions" is removed for that contrast.
- REVERSED (any contrast): **manuscript finalisation stops and the result is returned to the author**; the title
  (T2) is not finalised.
- Otherwise the manuscript is finalised as v11 with title T2 (author decision of 2026-10-10).

## 4. Scope
60 grid cells + 4 smoke cells. Not in scope: re-running component ablations, grafts or the per-class grid (author
decision: they keep the earlier-protocol caption), additive arms in the reverse direction, new datasets, title
change (unless REVERSED), local PDF build.

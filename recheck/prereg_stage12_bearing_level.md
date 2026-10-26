# Stage 12 (SASC revision): bearing-level description of the forward independent-noise grid

Written 2026-10-26, before any stage-12 cell is run.

**Why.** The stage-8 grids stored per-epoch macro-F1 only; no per-window predictions or checkpoints exist, so the class- and bearing-level composition of the main independent-noise results cannot be recovered. Two independent mock reviews asked whether the effects are carried by one evaluation bearing.

**What runs.** 20 cells: forward XJTU-SY (37.5 Hz/11 kN -> 40 Hz/10 kN), -6 dB, independent noise keyed on [seed, split, idx], arms frozen_clti, frozen_clti_nogate, frozen_clti_add, s4d, seeds 0-4, 50 epochs. Driver: recheck12_driver.py. Code path identical to recheck8_driver.py (stage 8, grids A and C); the only addition is that the final-epoch prediction of every evaluation window is saved with its bearing (Bearing3_1, 3_3, 3_4, 3_5).

**What is reported (descriptive only, no thresholds, no labels).**
1. Per arm and per evaluation bearing: final-epoch recall (fraction of that bearing's windows classified correctly), mean and range over seeds; window count and class of each bearing.
2. Per arm: summed confusion counts over seeds.
3. For each cell, stage-12 final macro-F1 next to the stage-8 value of the same arm and seed, and the mean absolute difference per arm. Stage-12 cells are re-runs, not bitwise reproductions (non-deterministic GPU kernels).

**How it is used.** As a descriptive table in the manuscript, stating that it comes from a re-run. If a gate-related difference at -6 dB is concentrated in one bearing, the text says so. No claim of the paper is upgraded or adjudicated on the basis of this grid, and the stage-8 numbers in the main tables are not replaced.

**Budget.** About 4-5 GPU-hours; no other grid is added.

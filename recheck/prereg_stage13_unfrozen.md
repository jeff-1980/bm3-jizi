# Stage 13 (TIMC revision): unfrozen Mamba-3 block under the final protocol

Written 2026-10-27, before any stage-13 cell is run (smoke check excluded).

**Why.** Stages 8 and 9 (independent training and evaluation noise, final-epoch reporting) compared frozen variants of the block with S4D but did not include the unfrozen block. The cost of freezing the scan coefficients, relative to the block as designed, was therefore not measured under the final protocol. A reviewer of the manuscript asked for this control.

**What runs.** 30 cells: arm `bm3_nokin` (unfrozen Mamba-3 block, cross-entropy only; same stem, depth, width, state size, optimiser, schedule, sampler and budget as `bm3_frozen`; harness `build_model`), XJTU-SY forward (37.5 Hz/11 kN -> 40 Hz/10 kN, eval hash 6c20b367...) and reverse (40 Hz/10 kN -> 37.5 Hz/11 kN, eval hash 93424834...), AWGN at 0, -2 and -6 dB, seeds 0-4, 50 epochs, independent noise keyed on [seed, split, idx]. Driver: `recheck13_driver.py`; its training loop is copied verbatim from `recheck9_driver.py`. The final-epoch prediction of every evaluation window is saved with its bearing. Smoke check: 2 epochs, one seed, excluded from analysis.

**Software stack.** Same as stage 12 (Python 3.12.3, torch 2.11.0+cu130, triton 3.6.0, mamba_ssm 2.3.2.post1, one RTX A5000), not the stage 1-9 stack. Stage 12 re-ran 20 stage-8 cells on this stack and differed from them by -0.98 to +0.26 pp (mean absolute difference <= 0.35 pp per arm); this is reported as the cross-stack drift. No frozen arm is re-run in stage 13.

**What is reported (fixed now).**
1. Per direction and SNR: mean (SD) final macro-F1 of `bm3_nokin` over five seeds, next to the stage-8/9 means of `frozen_clti` (fully frozen), `bm3_frozen` (partially frozen) and `s4d`.
2. Cost of freezing: paired by seed, `bm3_nokin` minus `frozen_clti` and minus `bm3_frozen`, mean with 95% paired t-interval (df = 4, t = 2.776). Same for `bm3_nokin` minus `s4d`.
3. Reading of item 2, stated now: a difference whose interval excludes zero is reported as a cost (or gain) of freezing at that level; otherwise "not resolved". Differences smaller than the cross-stack drift (1 pp) are not interpreted even if the interval excludes zero.
4. Per-bearing recall of `bm3_nokin`, descriptive.

**How it is used.** As an added row/column in the main tables and one sentence in the abstract on the cost of freezing. It does not change any stage-8/9 value or adjudication. Whatever the sign of the result, it is reported.

**Budget.** About 3 GPU-hours on one A5000; fuse at 5 GPU-hours. No other grid is added.

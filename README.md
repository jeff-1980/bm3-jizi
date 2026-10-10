# bm3-jizi — Reproducibility Package

Code, experimental data, pre-specification artifacts, and manuscript source for
*"What Carries Deep-Noise Robustness in a Mamba-3 Block? Input-Independent Scans and
Noise-Dependent Gating under Cross-Condition Shift"* (prepared for *Neurocomputing*;
manuscript v12, 2026-10-10).

The manuscript was rescoped several times after independent reviews; `CHANGELOG.md` lists
what changed and why, and `recheck/` holds the follow-up grids (stages 1-9). Since stage 10
the main text reports only results obtained with independent training and evaluation noise
(stages 8-9); everything run under the earlier shared-noise protocol is in Appendix D.
The manuscript's coverage table (Table 2) states which operation was run under which protocol.

## Layout

```
paper/                     LaTeX source, figures, and compiled PDF for the manuscript
  main.tex                 elsarticle preprint entry point
  sections/                Introduction, Background, Protocol, Results, Discussion, Boundaries, Conclusion
  appendix/                Appendix A (fairness protocol), B (confusion matrices + process floor), C (incidents + disclosure)
  figures/                 Fig.1-5 source scripts + rendered 300dpi pdf/png
  references.bib           Bibliography
  highlights.tex           Submission highlights
  main_sandbox_build.pdf   Manuscript built in a sandbox WITHOUT the TS1 (text companion)
                           fonts, which were substituted at build time. Layout is faithful,
                           a few symbol glyphs are not. Build locally (commands below) for
                           the authoritative PDF.

results/                   All 865 preregistered cells across five grids:
  fullgrid_*/               baseline grid (475 cells)
  frozensel_*/               frozen-selectivity grid (105 cells) + prereg/adjudication artifacts
  secondary_*/                component-ablation grid (175 cells) + prereg/adjudication artifacts
  graft_*/                    graft grid (35 cells) + prereg/adjudication artifacts
  perclass_*/                  per-class grid (75 cells) + confusion matrices
  preddump_*/                   evaluation-fingerprint / process-determinism verification runs
  predproc_*/                    prediction-dump availability checks

recheck/                   Follow-up grids (2026-10-01 / 2026-10-04), see CHANGELOG.md
  recheck_driver.py        Stage-1: re-runs 4 arms x {0,-2,-6} dB x 5 seeds x 50 epochs with
                           the harness training loop unchanged, logging macro-F1 at EVERY
                           epoch so the reported metric can also be read at the final epoch
                           (the harness's own metric is the max over epochs on the evaluation
                           condition, which uses evaluation data to pick the epoch)
  analyze_stage1.py        Stage-1 aggregation: per-rule means and paired contrasts
  stage1/                  Stage-1 outputs (60 cells with full per-epoch curves, 2 CSVs)
  layered_freeze.py        Stage-2 arms: the bm3_frozen arm freezes the input-dependent parts
                           of Delta/A/B/C but leaves Mamba-3's trapezoidal weight and rotation
                           angles input-dependent. frozen_ctrap / frozen_cangle / frozen_clti
                           freeze those too (clti = the scan's dynamics and read/write coefficients
                           no longer depend on the input; the value stream, the gate input, the
                           conv stem and the output gate are untouched)
  layered_unitcheck.py     Single-variable check for those arms, including the decisive
                           input-independence probe (capture Trap/Angles for two different
                           inputs through the same module and require bitwise equality for
                           whatever the arm claims to have frozen)
  layered_unitcheck_*/     Unit-check output snapshot
  recheck2_driver.py       Stage-2: the three layered arms plus the original bm3_kin arm
                           (so Delta(BM3 - frozen) is no longer rested on best-epoch values)
  analyze_stage2.py        Stage-2 aggregation under both reporting rules
  stage2/                  Stage-2 outputs (60 cells with per-epoch curves, 2 CSVs)
  prereg_graft_control_pair2.md  Stage-3 pre-specification (+ .sha256), committed before any run
  graft_control.py         Stage-3 arms: width-matched gate graft and same-size additive control
  graft_control_unitcheck.py, unitcheck_stage3/   Construction check and its snapshot
  recheck3_driver.py       Stage-3 driver (--pair 1 original, --pair 2 reverse transition)
  analyze_stage3.py        Stage-3 analysis implementing the pre-specified rules
  stage3_decision_memo.md  Rule-by-rule decisions and DEVIATIONS
  stage3/                  Stage-3 outputs (105 cells with per-epoch curves, tables, decisions)
  prereg_native_additive.md  Stage-4 pre-specification (+ .sha256) incl. the title decision tree
  additive_native.py       Stage-4 arm: gate replaced by an additive branch inside BM3-frozen
  additive_native_unitcheck.py, unitcheck_stage4/   Construction check and its snapshot
  recheck4_driver.py, analyze_stage4.py, stage4_decision_memo.md
  stage4/                  Stage-4 outputs (30 cells with per-epoch curves, tables, decisions)
  prereg_stage5_metric_reverse.md  Stage-5 pre-specification (+ .sha256) incl. the title branch rule
  recheck5_driver.py, analyze_stage5.py, stage5_decision_memo.md
  numbers_stage5.py, verify_v8.py   Manuscript numbers from per-cell values, and an independent re-check
  stage5/                  Stage-5 outputs (60 cells with per-epoch curves, tables, decisions)
  prereg_stage6_pu.md      Stage-6 pre-specification (+ .sha256): PU replication, smoke rule, decision rules
  pu_dataset.py, recheck6_driver.py, analyze_stage6.py, stage6_decision_memo.md
  stage6/                  Stage-6 outputs (75 PU cells + 6 smoke cells, tables, decisions, run log)
  prereg_stage7_pu_shared.md  Stage-7 pre-specification (+ .sha256): shared-bearing diagnostic, D1/D2, branches
  recheck7_driver.py, analyze_stage7.py, stage7_decision_memo.md
  stage7/                  Stage-7 outputs (45 PU cells + 6 smoke cells, tables, decisions, run log)
  prereg_stage8_indep_noise.md  Stage-8 pre-specification (+ .sha256): independent train/eval noise
  indep_noise.py           Noise wrapper keyed on (seed, split, idx)
  clti_gate.py, stage8_unitcheck.py, unitcheck_stage8/   Same-host gate arms and construction check
  recheck8_driver.py, analyze_stage8.py, stage8_decision_memo.md
  stage8/                  Stage-8 outputs (150 cells, tables, decisions, run logs)
  prereg_stage9_reverse_indep.md  Stage-9 pre-specification (+ .sha256): reverse transition, independent noise
  recheck9_driver.py, analyze_stage9.py, stage9_decision_memo.md
  stage9/                  Stage-9 outputs (60 cells + 4 smoke cells, tables, decisions, run log)

bearmamba3/                Dataset, model and auxiliary-loss modules imported by the harness
baselines/, models_extended.py, noise_utils.py
                           Remaining harness imports. These four were missing from the
                           initial upload, so `xjtu_noisy_harness.py` could not be imported
                           from a fresh checkout; added 2026-10-04. bearmamba3/ and
                           baselines/ carry only the files this study imports.

xjtu_noisy_harness.py      Training/eval harness (all arms, all noise conditions)
bm3_models.py               BM3 / S4D / S4D-wide model definitions
bm3_frozen.py                Frozen-selectivity and single-component-removal arms
arch_map.md                  Component-level architecture map (selective mechanism -> code, byte-identical-key verification)

*_aggregate.py               Per-grid aggregation scripts (frozensel/secondary/graft)
*_unitcheck.py                Per-grid state-dict key-diff / single-variable-discipline checkers
*_unitcheck_*/                 Unit-check output snapshots
smoke_*_check.py              Pre-launch smoke tests
launch_*.sh / status_*.sh      Grid launch and monitoring scripts
cmp_*.py, diag_*.py, peek_agg.py, check_fair.py   Ad hoc comparison/diagnostic scripts used during the study

prereg_provenance.md         Materialized preregistration trail (component-ablation grid)
graft_prereg_provenance.md   Materialized preregistration trail (graft grid)

REPORT_*.md                  Dated incident/decision reports referenced in Appendix C
*.log                        Raw run logs for the corresponding launch script
```

## Reproducing a result

Each grid directory under `results/` contains one subdirectory per
`arm__condition__seed`, each with `predictions.jsonl` (or `seed_N.json`) and a
`summary.json`. Every cell carries `eval_sha256` (full value in
`paper/appendix/appendixA.tex`, Section A.1); recomputing that hash over the
held-out test-label array and checking it against the value in each grid's
`fairness_report.json` is the fastest way to confirm that the same held-out
label vector was scored in every cell before comparing any two arms. The hash
covers the evaluation labels, not the input windows.

To regenerate a grid's headline numbers: run the corresponding
`*_aggregate.py` script against its `results/<grid>_*/` directory; the
thresholds were written down before each grid was launched, and the evidence
for that ordering is local: compare the pre-specification artifact's
modification time (`stat`) against the run-directory's timestamp in its folder
name, per the table in `paper/appendix/appendixA.tex` Section A.3. This is
internal pre-specification, not registration with an external time-stamped
registry, and `git clone` does not preserve the original mtimes.

## Environment and paths

Software used for stages 1-9: Python 3.12 (`bm3-repro` conda environment), torch 2.9.0+cu128,
triton 3.5.0, CUDA 12.8, Mamba source at upstream commit `e9594ce` (modules `mamba_ssm`,
including the Mamba-3 SISO kernels; the stage-2..9 arms call its `mamba3_siso_combined`).
The original grids (before stage 1) used torch 2.5 / cu121; stage 1 re-ran four arms on the
newer stack and reproduced the original values within about 1 pp (Appendix D.9 of the paper).

* The stage-1..9 drivers and arm modules take the harness directory from `P9_HARNESS_DIR`
  and default to the repository root (where `xjtu_noisy_harness.py` lives).
* `xjtu_noisy_harness.py` itself is the byte-identical file that produced every result and was
  deliberately not edited; it still contains three absolute paths near its top (`BM3_ROOT`,
  `EXT_DIR`, `DATA_ROOT`). To run it elsewhere, create those directories (or symlinks):
  `BM3_ROOT` must contain a `bearmamba3` package (this repository's `bearmamba3/`),
  `EXT_DIR` the `models_extended.py` / `noise_utils.py` / `baselines/` copies in this repository,
  and `DATA_ROOT` the XJTU-SY dataset (`XJTU-SY_Bearing_Datasets/`).
* Paderborn data: set `P9_PU_ROOT` to the folder with one sub-directory per bearing code.
* Datasets, the Mamba source tree and a GPU are external requirements and are not included.
  The analysis scripts (`recheck/analyze_stage*.py`) need only the released cells and numpy,
  scipy and pandas; they were re-run from a fresh clone for stages 8 and 9 and reproduced
  the archived decisions, contrasts and means byte for byte.

## Adjudication trail and disclosure

The mechanical analysis/review passes in this pipeline were implemented with
LLM-based coding agents; three caught adjudication incidents (two post-hoc
criterion substitutions, one stale-artifact false verification) are described
in full, with timelines, in `paper/appendix/appendixC.tex`. All numeric
thresholds and all rulings on these incidents were fixed or made by the human
author. The `REPORT_*.md` files at the repository root are the original,
dated incident/decision reports that appendix draws on.

## Manuscript build

```
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```
Requires a standard TeX Live install with `elsarticle`. Last verified build:
0 errors, 0 undefined references (sandbox build with substituted fonts; see main_sandbox_build.pdf note).

## License

Not yet specified. Contact the author before reuse.

# bm3-jizi — Reproducibility Package

Code, experimental data, preregistration artifacts, and manuscript source for
*"Necessary but Not Transplantable: Deep-Noise Robustness of Mamba-Style Blocks
Is a Component-Interaction Effect"* (submitted to *Neurocomputing*).

## Layout

```
paper/                     LaTeX source, figures, and compiled PDF for the manuscript
  main.tex                 elsarticle preprint entry point
  sections/                Introduction, Background, Protocol, Results, Discussion, Boundaries, Conclusion
  appendix/                Appendix A (fairness protocol), B (confusion matrices + process floor), C (incidents + disclosure)
  figures/                 Fig.1-5 source scripts + rendered 300dpi pdf/png
  references.bib           Bibliography
  highlights.tex           Submission highlights
  main.pdf                 Compiled manuscript (0 compile errors)

results/                   All 865 preregistered cells across five grids:
  fullgrid_*/               baseline grid (475 cells)
  frozensel_*/               frozen-selectivity grid (105 cells) + prereg/adjudication artifacts
  secondary_*/                component-ablation grid (175 cells) + prereg/adjudication artifacts
  graft_*/                    graft grid (35 cells) + prereg/adjudication artifacts
  perclass_*/                  per-class grid (75 cells) + confusion matrices
  preddump_*/                   evaluation-fingerprint / process-determinism verification runs
  predproc_*/                    prediction-dump availability checks

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
`fairness_report.json` is the fastest way to confirm you are looking at
byte-identical evaluation data before comparing any two arms.

To regenerate a grid's headline numbers: run the corresponding
`*_aggregate.py` script against its `results/<grid>_*/` directory; to verify
the preregistration predates the data, compare the prereg artifact's
modification time (`stat`) against the run-directory's timestamp in its
folder name, per the table in `paper/appendix/appendixA.tex` Section A.3.

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
0 errors, 0 undefined references, 26 pages.

## License

Not yet specified. Contact the author before reuse.

# Changelog

## 2026-10-11 — 50 references; real-font build (manuscript v14)

- Bibliography extended from 16 to 50 entries (all cited). The 34 additions were looked up by title in the
  arXiv API (29) and Crossref (5); the bibliographic fields come from those records (`paper/reference_lookup/`),
  not from memory. Each is cited where it supports a statement: S4/HiPPO/S5/LRU/Hyena/Mamba-2 and the
  gating lineage (LSTM, highway, GLU, SE, SiLU) in Section 2; copying/state-tracking/recall limits and
  MambaOut in Section 2.2; bearing-diagnosis reviews and CWRU benchmark, adversarial domain adaptation,
  corruption benchmarks in Section 2.3; AdamW, cosine schedule, BatchNorm and DomainBed in Section 3
  (AdamW and CosineAnnealingLR are what the harness uses); shortcut learning, seed-variance and ablation-practice
  papers in Sections 4 and 7. No result, number or conclusion changed.
- Build: `build_real.sh` + `genpk.sh` generate the TS1 bitmap fonts with Metafont from the local TeX tree, so the
  PDF no longer uses substituted fonts (replaces `main_sandbox_build.pdf`). 55 pages in the `preprint,12pt`
  option; 21 pages in `[final,5p,twocolumn]` with mathptmx (the `times` option's txfonts could not be
  built in this tree).

## 2026-10-10 — format check against Neurocomputing requirements (manuscript v13b)

- Keywords reduced from 8 to 7 (limit stated in the guide for authors: at most 7).
- Added DOI/arXiv identifiers to the five bibliography entries that had none (XJTU-SY: 10.1109/TR.2018.2882682;
  PHM Society paper: 10.36001/phme.2016.v3i1.1577; three arXiv preprints, identifiers matched to title and authors
  through the arXiv API). 16 entries, all cited, none missing.

## 2026-10-10 — stage 11: corrections from the third review of the day; manuscript v13

No new experiments.

- Fixed a transposition in Section 4.1: the share of the 3125 paired resamples in which at least one level was
  excluded from rho is 16.2% (507) for the fully frozen host and 0.5% (16) for the partially frozen host (the
  text had them swapped). The all-levels statistic (repository field paired_frac_dropped) is 0 for both and is
  named separately.
- Figure 1: step 3 is now the in-host replacement of the gate by an additive branch (the experiment whose numbers
  it shows); step 1 labels partial and full freeze; "additive = gateless" replaced by "not resolved". The graft onto
  S4D is described as an exploratory branch in Appendix D.
- Gateless-versus-S4D statements now specify host and protocol; "the gate costs most" replaced by a statement about
  the -6 dB loss with the cross-component ranking labelled earlier-protocol; the independent-noise grids are stated
  to contain no unfrozen BM3 arm.
- Contributions reduced to two scientific items plus supporting material; limitations merged into four themes.
- Section 4.1: cell allocation (105 XJTU-SY cells = 7 arms, 45 PU cells); Table 6 notes no multiple-comparison correction.
- "up to 18/20 pp" replaced by the range of the independent-noise results (12-28 pp).
- Data availability narrowed: analysis is recomputable from released cells; training needs data, GPU and the Mamba source.
- New recheck/harness_loader.py: loads the unmodified harness with its three absolute paths replaced by environment-
  controlled values (default: repository copies, which are byte-identical to the originals). All stage drivers use it
  and also add recheck/ to sys.path themselves. A 2-epoch smoke run of recheck9_driver.py from a clean copy of the
  repository imported bearmamba3 and models_extended from that copy and reproduced eval_sha 934248343b29. A full
  retraining from a clean environment has not been performed.

## 2026-10-10 — stage 10: evidence restructuring; manuscript v12 (title T2, scope of its noise-dependence claim stated)

No new experiments. Response to the simulated review of the same day (author decisions: keep T2 with
the noise-dependence statement restricted to the forward fully frozen host; move all earlier-protocol
results into an appendix; no further re-runs).

- Main text reports independent-noise results only (original transition, reverse, PU shared) plus a new
  coverage table (operation x host x transition x protocol) and a descriptive analysis of the change of
  the gate-removal cost with noise depth: only the forward fully frozen block shows it (+27.2 [19.2, 35.2] pp
  between 0 and -6 dB); forward partially frozen, reverse and PU shared do not. Added during review, not
  pre-specified.
- All earlier-protocol results (freezing, component removal, grafts, native additive on the reverse
  transition, PU disjoint, reporting-rule sensitivity, per-class) moved to Appendix D with protocol
  labels in every caption; Section 4.5 states the weight given to each.
- Additive substitution is now argued from the direct multiplicative-minus-additive contrasts
  (fully frozen +4.6/+7.9/+23.3 pp; partially frozen +9.0/+15.1/+26.8 pp), not from "indistinguishable
  from gate removal"; the pre-specified rescue test remains "not assessable". Resample drop fractions are
  reported for both definitions (16.2% / 0.5% at least one level; 0% all levels).
- Corrections: PU disjoint did include frozen-all; the statement "statistical tests were never
  adjudication criteria" now distinguishes original from later grids; the label-agreement inference was
  weakened; "all arms re-run" split into final-epoch and independent-noise re-runs; the abstract no longer
  suggests the reverse transition covers the fully frozen gate tests.
- Repository: stage-1..9 drivers and arm modules read the harness directory from P9_HARNESS_DIR (default:
  repository root) and the Paderborn root from P9_PU_ROOT (path edits only; no change to any computation
  or output); environment and path requirements documented in README; appendix parameter table now lists
  every arm.

## 2026-10-10 — stage 9: reverse transition under independent noise; manuscript v11 final (title T2)

Pre-specified in `recheck/prereg_stage9_reverse_indep.md` (sha256 cc5ba8fc…, committed in 9fadcfe before
the smoke check; unchanged). Fourth arm (frozen_clti) and the 9 h fuse approved by the author before the
pre-specification was written. 60 cells + 4 smoke cells, 5.27 GPU-h. Decisions: `recheck/stage9_decision_memo.md`.

- Reverse (40Hz10kN -> 37.5Hz11kN), independent noise, final epoch: frozen_clti - s4d +27.5/+26.7/+21.1 pp,
  bm3_frozen - s4d +25.5/+27.5/+20.0 pp, bm3_frozen - frozen_nogate +19.3/+22.7/+22.0 pp; all HOLDS,
  intervals excluding zero at every level, 5/5 seeds; no reversal. Arm means move -1.8..+1.7 pp vs the
  earlier protocol.
- Manuscript v11 final: title T2; Table 10 on independent noise (S4D+gate (wm) row kept on the earlier
  protocol, marked); direction qualifiers in abstract/conclusion; protocol summary for the parts left on
  the earlier protocol; paired-seed intervals for graft ratios; 645 follow-up cells.

## 2026-10-10 — stage 8: independent training/evaluation noise; manuscript v11-draft (title pending)

Pre-specified in `recheck/prereg_stage8_indep_noise.md` (sha256 b4cd7065…, committed in 5cefc74 before
the first grid cell; unchanged). 150 cells, 14.53 GPU-h. Decisions and DEVIATIONS:
`recheck/stage8_decision_memo.md`.

- Defect fixed: noise was keyed on (seed, idx) with the same seed for training and evaluation, so
  window idx shared its base noise vector across the two sets. `indep_noise.py` keys it on
  (seed, split, idx). Arm means move by at most 4 pp; no contrast changes direction.
- XJTU-SY: frozen_clti - s4d +21.3/+23.3/+20.3 pp and bm3_frozen - s4d HOLD; gate removal
  WEAKENED (resolved at -6 dB only) in both the partially and the fully frozen host; without
  its gate the fully frozen block beats S4D at 0/-2 dB and trails it at -6 dB. Additive
  substitution indistinguishable from gate removal in both hosts (not assessable by rule).
  Combined attribution (fully frozen advantage carried by the gate) not supported.
- PU shared: gate removal HOLDS; partial-freeze advantage WEAKENED.
- New arms `clti_gate.py` (frozen_clti_nogate, frozen_clti_add); 11/11 construction checks.
- Ratio intervals now also reported with a paired-seed bootstrap (`unitcheck_stage8/paired_bootstrap_check.json`).
- Manuscript v11-draft: new Section 4.1 and Tables 2-3 (independent noise); earlier-protocol
  tables labelled; XJTU bearing identity corrected (disjoint within each direction); scan
  wording ("dynamics and read/write coefficients"); PU described as two protocols; threshold
  instead of equivalence language; paired intervals; abstract 291 words. Title pending the
  author (pre-specified downgrade).

## 2026-10-09 — stage 7: PU shared-bearing diagnostic (branch R), manuscript v10

Pre-specified in `recheck/prereg_stage7_pu_shared.md` (sha256 98e5074f…, committed in 20395f3
before the smoke check and the grid; unchanged afterwards). Smoke margins: learnable +31.3 pp
(stage 6: +0.3), not-floor +31.0, not-ceiling +19.5. 45 grid cells, 3.05 GPU-h incl. smoke.
Decisions and DEVIATIONS: `recheck/stage7_decision_memo.md`.

- Single change from stage 6: training and evaluation use the same bearings {KA04, KA16, KI04,
  KI14} (1500 -> 900 rpm, different recordings). Arms s4d, bm3_frozen, frozen_nogate.
- D1 transferability gate: PASS (all arms off the floor at all levels; 75.5-86.6 % macro-F1).
- D2: L1'' bm3_frozen - s4d = +8.4 / +8.5 / +8.7 pp (REPLICATED, 2/3); L2'' bm3_frozen -
  frozen_nogate = +9.0 / +7.7 / +5.3 pp (REPLICATED, 3/3); no reversal -> branch R.
- Shared minus disjoint bearings raises every arm, S4D included, by 18.6-38.8 pp: bearing
  non-overlap breaks transfer for all examined architectures.
- Manuscript v10: title A2 kept; abstract, Section 4.8 (two PU settings, Table 9 extended),
  boundaries ("Bearing identity"), introduction, conclusion, highlights; scope = cross-condition
  drift with shared bearing identity (XJTU-SY and Paderborn).

## 2026-10-08 — stage 6: Paderborn replication (NOT REPLICATED), manuscript v9 draft pending author decision

Pre-specified in `recheck/prereg_stage6_pu.md` (sha256 502a2a69…, committed in d1b7337 before the
smoke check and the grid; unchanged afterwards). Smoke check passed its pre-specified criteria
(learnability by 0.3 pp); 75 grid cells, 5.40 GPU-h including the smoke check. Decisions and
DEVIATIONS: `recheck/stage6_decision_memo.md`.

- PU real-damage bearings, disjoint train {KA04, KA16, KI04, KI14} @ 1500 rpm and test
  {KA15, KA22, KI17, KI18, KI21} @ 900 rpm; arms bm3_frozen, frozen_clti, s4d, frozen_nogate,
  bm3_frozen_add. Final epoch: frozen_clti - s4d = -1.4 / -2.5 / -3.3 pp (NOT REPLICATED);
  bm3_frozen - s4d = -3.8 / -3.6 / -3.4 (NOT REPLICATED); gate removal bm3_frozen - frozen_nogate
  = -5.0 / -4.6 / -3.3 (NOT REPLICATED); native additive substitution not assessable.
  All arms end at 47.7-56.9 % macro-F1 (majority-class 37.5 %).
- Pre-specified writing branch: unfavourable; scope and title returned to the author. The manuscript
  draft adds a reporting subsection (Table 9) and a boundaries item restricting claims to XJTU-SY;
  title A2, corrected Saadatmand et al. citations and the cover letter (paper-8 status) are included.

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

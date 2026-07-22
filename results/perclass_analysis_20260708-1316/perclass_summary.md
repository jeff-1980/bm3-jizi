# Per-class summary — results/perclass_20260707-2121 grid

Grid: 5 arms (`bm3_frozen`, `bm3_kin`, `frozen_nogate`, `s4d`, `s4d_plus_gate`) x
3 conditions (`clean`, `awgn@+0dB`, `awgn@-6dB`) x 5 seeds = 75 cells, all
gates passed (tuple-enumeration 75/75, `fairness_report.json` 75/75
byte-exact `eval_sha256`, `reconciliation_report.json` 75/75 bit-exact
macro-F1 recompute from `predictions.jsonl`). Task is a **binary**
fault-position classifier: class `OR` (outer race) = label 0, class `IR`
(inner race) = label 1 (`bearmamba3/data_xjtu.py: LABEL_MAP = {'OR': 0, 'IR': 1}`).
Test split is identical across every one of the 75 cells (5248 samples:
3504 OR / 1744 IR, confirmed directly from `predictions.jsonl`, not just via
the `eval_sha256` hash match) — all per-class numbers below are summed over
5 seeds, so per-class support is 17520 (OR) / 8720 (IR) per arm x condition.

Source data: `per_class_report.csv/json`, `confusion_matrices/`.

## 1. Which class collapses first under deep noise

**IR is already the structurally weaker class at `clean`, for every one of
the 5 arms, and stays weaker at every noise level** — IR's F1 is lower than
OR's F1 in all 15 of 15 (arm x condition) cells. This is driven mostly by
recall, not precision: at `clean`, OR recall is 0.96–1.00 across every arm,
while IR recall is only 0.41–0.62:

| arm | OR recall (clean) | IR recall (clean) | OR F1 (clean) | IR F1 (clean) |
|---|---|---|---|---|
| bm3_frozen | 0.988 | 0.551 | 0.893 | 0.699 |
| bm3_kin | 0.999 | 0.547 | 0.898 | 0.706 |
| frozen_nogate | 0.967 | 0.411 | 0.855 | 0.556 |
| s4d | 0.961 | 0.615 | 0.893 | 0.726 |
| s4d_plus_gate | 0.989 | 0.430 | 0.870 | 0.592 |

So under the "which class is worse off" framing, **IR is the class that has
already collapsed relative to OR before any noise is added**, in every arm
tested.

Under increasing noise (`clean` -> `awgn@+0dB` -> `awgn@-6dB`), the
*trajectory* is not uniform across arms, and simple F1-delta is confounded
by the 2:1 OR:IR test-set imbalance (a model that drifts toward predicting
more IR under noise gains IR recall at OR's expense, which is a precision/
recall trade, not "IR getting easier"). Reporting both directions honestly:

| arm | OR recall clean->-6dB | IR recall clean->-6dB | OR F1 clean->-6dB | IR F1 clean->-6dB |
|---|---|---|---|---|
| bm3_frozen | 0.988->0.892 (-0.095) | 0.551->0.708 (+0.158) | 0.893->0.876 (-0.017) | 0.699->0.736 (+0.037) |
| bm3_kin | 0.999->0.948 (-0.051) | 0.547->0.809 (+0.262) | 0.898->0.928 (+0.030) | 0.706->0.846 (+0.140) |
| frozen_nogate | 0.967->0.753 (-0.213) | 0.411->0.373 (-0.038) | 0.855->0.730 (-0.126) | 0.556->0.399 (-0.157) |
| s4d | 0.961->0.712 (-0.248) | 0.615->0.563 (-0.052) | 0.893->0.738 (-0.155) | 0.726->0.526 (-0.200) |
| s4d_plus_gate | 0.989->0.757 (-0.232) | 0.430->0.628 (+0.198) | 0.870->0.780 (-0.091) | 0.592->0.594 (+0.001) |

Two consistent, arm-independent facts hold across all 5 arms:
- **OR recall drops under noise in every single arm** (by 5–25 pp from
  `clean` to `awgn@-6dB`) — even though OR started near ceiling.
- IR's response under noise is genuinely mixed: it improves in 3/5 arms
  (`bm3_frozen`, `bm3_kin`, `s4d_plus_gate` — all arms that retain the
  z-gate) and degrades further in 2/5 arms (`frozen_nogate`, `s4d` — the two
  arms without a gate). This split lines up with gate presence (§2 below),
  not with noise level alone.

Net answer to "which class collapses first": **by absolute F1/recall level,
IR is already the weaker class at every noise level in every arm — it never
had to "collapse" because it starts collapsed. By degradation trend, OR is
the class that reliably degrades under noise in every arm; IR's trend is
gate-dependent rather than universal.** Both framings are reported since the
data does not support a single unqualified answer.

## 2. Gate ablation ("拆 gate") vs gate graft ("嫁接 gate") — per-class effect

Cross-referenced against `arch_map.md` §5b (`frozen_nogate` — z-gating
removed from `bm3_frozen`) and §6 (`s4d_plus_gate` — z-gating grafted onto
`s4d`); no section literally labelled "§4b" exists in `arch_map.md`, so this
report uses §5b/§6, the sections that actually describe the gate mechanism
and its removal/graft — noted here explicitly per the "如实报" requirement
rather than silently substituting.

**Gate removal (`frozen_nogate` vs `bm3_frozen`), F1 delta:**

| condition | OR delta | IR delta |
|---|---|---|
| clean | -0.038 | -0.143 |
| awgn@+0dB | -0.107 | -0.350 |
| awgn@-6dB | -0.146 | -0.337 |

Removing the gate hurts **IR more than OR at every one of the 3 conditions**
(2.6x–3.8x larger F1 drop in absolute terms), and the effect is worst at
`awgn@+0dB` (-35 pp on IR) rather than at the deepest noise (`-6dB`, -34 pp)
— i.e. the gate's contribution to IR detection is large and roughly flat
across the noisy conditions tested, not something that only shows up at the
deepest noise.

**Gate graft (`s4d_plus_gate` vs `s4d`), F1 delta:**

| condition | OR delta | IR delta |
|---|---|---|
| clean | -0.022 | -0.134 |
| awgn@+0dB | +0.033 | +0.024 |
| awgn@-6dB | +0.042 | +0.068 |

Grafting the gate onto `s4d` slightly *hurts* both classes at `clean`
(IR again more, -0.134 vs -0.022), but *helps* both classes once noise is
present, with IR getting the larger benefit at the deepest noise
(`awgn@-6dB`: IR +0.068 vs OR +0.042). This is the mirror image of removal:
the gate's benefit for IR is noise-conditional in the graft direction (no
benefit clean, growing benefit under noise) but consistently the larger
per-class effect (in magnitude) in the removal direction (always IR > OR,
noise or not).

**Combined read:** in both experiments the z-gate's largest measured effect,
in either direction, lands on **IR (inner race)**, the minority/harder
class — consistent with §1's finding that IR recall is the metric most
sensitive to noise and to architecture changes. No claim is made about
*why* (mechanism), only what the summed-over-5-seeds counts show.

## 3. Sanity note (non-gating): historical cross-check outliers

Per task item 2, arm x condition means were checked against the historical
frozensel/secondary/graft grids (`historical_crosscheck.json`, informational
only, not a pass/fail gate). Two cells exceed the ~0.7 pp known
cross-process noise floor (`REPORT_preddump_equivalence_20260707-1933.md`,
adjudication 1) by a wide margin and are flagged here for visibility, not
treated as a failure:

- `frozen_nogate @ awgn@+0dB`: this grid's mean = 0.6542 vs historical
  (secondary/graft) mean = 0.7760, diff = -12.17 pp (historical std used for
  the ±1σ band was exceeded).
- `s4d @ awgn@-6dB`: this grid's mean = 0.6317 vs historical mean = 0.6856,
  diff = -5.4 pp (also exceeds ±1 historical σ).

All other 13 of 15 arm x condition comparisons fall within ±1 historical σ.
No root-cause investigation was performed — out of scope for this task
("不写超出数据的主张"); flagged as-is for a human to decide whether it
warrants follow-up.

## Files

- `fairness_report.json` — 75/75 byte-exact `eval_sha256`.
- `reconciliation_report.json` — 75/75 bit-exact macro-F1 recompute from
  `predictions.jsonl` vs `cells.jsonl` self-reported value.
- `historical_crosscheck.json` — sanity-only arm x condition mean/std vs
  frozensel/secondary/graft grids.
- `confusion_matrices/*.png`, `*.csv` — 15 matrices (5 arms x 3 conditions),
  counts summed over 5 seeds, row sum = support.
- `per_class_report.csv/json` — precision/recall/F1/support per arm x
  condition x class (30 rows), same source as the confusion matrices.

#!/home/jeffwork/论文8/venv/bin/python3
"""
secondary_aggregate.py — post-grid aggregation for results/secondary_20260703-1034/.

Mechanically applies the rule preregistered in
results/secondary_prereg_20260704-0708/secondary_prereg.json (written after the
grid's cells.jsonl already existed, but before this script computed or viewed
any aggregate number — see that file's prereg_timing_disclosure). Read-only
over results/secondary_20260703-1034/cells.jsonl: never writes, chmods, or
deletes anything under that directory or any other existing results/**
artifact. All new products go into a NEW timestamped --out-dir.

Steps (all-or-nothing; any FAIL stops before writing decision/aggregation
products, matching the prereg's fairness_verification_required gate):
  0. Tuple-enumeration check (E0): build the exact expected set of
     (condition, arm, seed) = canonical_conditions() x design.arms x
     design.seeds (7 x 5 x 5 = 175) and require observed cells.jsonl rows to
     match it exactly (no duplicates, no missing, no extras). Write
     tuple_enum_report.json. If this fails: stop before fairness/aggregation
     (STOP-IF-NO-GRID / guardrail 0).
  1. Independently recompute eval_sha256 (full 64-hex-char hash) via the
     harness's own eval_fingerprint()/make_cross_condition_split()/
     XJTUDataset() — the same construction run_grid() uses — and cross-check
     every one of the 175 cells' fairness.eval_sha256 against it byte-for-byte.
     Write fairness_report.json. If this fails: stop (no aggregation/decision
     products written).
  2. Aggregate mean/std macro_f1 per (condition, arm) for all 5 arms ->
     q_secondary.json/csv. Curve table + PNG (macro_F1 vs SNR, all 5 arms).
     Per-component-arm Delta(frozen-X) and Delta(X-s4d) pointwise tables.
     Paired significance (t-test + Wilcoxon, matched by seed) — reporting
     only, per the prereg.
  3. Dispatch on the --prereg file's schema (§H2 fix, .orchestrate/0704-215713/
     r1_review.json's heuristic-deletion requirement, re-affirmed by the review
     of results/secondary_e3ruling_20260704-2206/secondary_decision_corrected.json):
       - schema has 'decisive_rule_delta_pp_threshold' (e.g.
         secondary_deltapp_prereg_20260704-0719.json): apply_deltapp_decision_rule()
         writes secondary_decision.json with per-component verdicts + overall
         verdict, every condition's raw evidence present (not masked by the
         majority-count summary). This is the ONLY function in this module
         permitted to emit adjudicates_E3=True, gated by
         verify_prereg_provenance_chain() and prereg['status']=='REVIEWER_APPROVED'.
       - otherwise (e.g. the band-rule schema secondary_prereg_20260704-0708.json):
         there is no decision-emitting function for this schema. The former
         mean+/-std band heuristic (apply_post_hoc_band_report()/
         band_significant()) has been deleted outright, not merely gated --
         it produced verdicts that read as a decision even when hard-asserted
         to carry preregistered=False, which a later reviewer round found
         gets replayed/cited as if it were an adjudication. main() instead
         writes secondary_no_decision.json: no per_component_verdicts, no
         overall_verdict, preregistered=False, adjudicates_E3=False,
         prereg_source=None, and a reason string. No E3 adjudication is
         claimed or implied.
  4. chmod 444 every product written in this run.

Usage:
  python secondary_aggregate.py --run-dir results/secondary_20260703-1034 \
      --prereg results/secondary_deltapp_prereg_20260704-0719/secondary_deltapp_prereg.json \
      --out-dir results/secondary_analysis_<TS>
"""
import argparse
import csv
import json
import os
import re
import stat
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

HARNESS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS_DIR))

import xjtu_noisy_harness as H  # noqa: E402  (import-only, no side effects — main() gated)

CHANCE_FLOOR = 0.55
COMPONENT_ARMS = ["frozen_noconv", "frozen_nogate", "frozen_As4d"]
FULL_BASELINE_ARM = "bm3_frozen"
FLOOR_BASELINE_ARM = "s4d"
ALL_ARMS = [FULL_BASELINE_ARM] + COMPONENT_ARMS + [FLOOR_BASELINE_ARM]


def _resolve_latest_reviewer_verdict(repo_root: Path = None):
    """Same closed-round resolution logic as frozensel_aggregate.py's
    _resolve_latest_reviewer_verdict() (H2/E3 fixes): only rounds whose
    result.json has status=='passed' are eligible, so the in-progress round
    reviewing *this very artifact* can never cite itself (self-referential
    deadlock, per .orchestrate/0703-084057/r1_review.json's E3 finding)."""
    root = repo_root or HARNESS_DIR
    orch = root / ".orchestrate"
    if not orch.is_dir():
        return None
    for round_dir in sorted((d for d in orch.iterdir() if d.is_dir()), reverse=True):
        result_path = round_dir / "result.json"
        if not result_path.is_file():
            continue
        try:
            result_obj = json.loads(result_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(result_obj, dict) or result_obj.get("status") != "passed":
            continue
        for rpath in sorted(round_dir.glob("r*_review.json"), reverse=True):
            try:
                obj = json.loads(rpath.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(obj, dict) and obj.get("verdict") in ("PASS", "REVISE"):
                try:
                    return str(rpath.relative_to(root))
                except ValueError:
                    return str(rpath)
    return None


def _resolve_last_effective_reviewer_verdict(repo_root: Path = None):
    """Distinct from _resolve_latest_reviewer_verdict() above, which only
    considers rounds with status=='passed' (used for the general
    implementation-provenance citation). §Major fix (reviewer finding on
    results/secondary_e3ruling_20260704-2206/secondary_decision_corrected.json):
    that citation is the wrong field to answer 'what is the last reviewer
    verdict on the E3 question specifically', because a round can be the most
    recent CLOSED round (has a result.json) yet still be status=='needs_human'
    with verdict==REVISE -- exactly the case for .orchestrate/0704-070402/,
    whose r2_review.json REVISE is the most recent non-self-referential ruling
    on E3 even though its round never reached status=='passed'. This function
    walks rounds newest-first, skips only rounds with NO result.json at all
    (the in-progress/self-referential round currently reviewing this very
    artifact), and returns the first PASS/REVISE verdict file found -- i.e. the
    last effective reviewer verdict, regardless of that round's overall
    status. Executor/self-authored claims are never eligible here because
    only rounds with a result.json (written by the reviewing harness, not the
    executor) are considered."""
    root = repo_root or HARNESS_DIR
    orch = root / ".orchestrate"
    if not orch.is_dir():
        return None
    for round_dir in sorted((d for d in orch.iterdir() if d.is_dir()), reverse=True):
        result_path = round_dir / "result.json"
        if not result_path.is_file():
            continue  # no result.json => still in-progress / self-referential; skip
        for rpath in sorted(round_dir.glob("r*_review.json"), reverse=True):
            try:
                obj = json.loads(rpath.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(obj, dict) and obj.get("verdict") in ("PASS", "REVISE"):
                try:
                    return str(rpath.relative_to(root))
                except ValueError:
                    return str(rpath)
    return None


def _protect(path: Path):
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _write_protected_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    _protect(path)


def _write_protected_csv(path: Path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)
    _protect(path)


def load_cells(run_dir: Path):
    cells_path = run_dir / "cells.jsonl"
    if not cells_path.exists():
        raise SystemExit(f"[FATAL] {cells_path} does not exist — grid not launched or not yet writing.")
    cells = []
    for line in cells_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        cells.append(json.loads(line))
    return cells


def canonical_conditions() -> list:
    """Authoritative condition-label list from the harness's own
    build_conditions(["awgn"], FULL_SNRS) — not hand-typed, guarantees exact
    agreement with what run_grid() actually wrote (e.g. 'awgn@+10dB' with the
    '+' sign)."""
    conds = H.build_conditions(["awgn"], H.FULL_SNRS)
    return [c["label"] for c in conds]


def expected_tuples(prereg: dict) -> set:
    design = prereg["design"]
    conditions = canonical_conditions()
    arms = design["arms"]
    seeds = design["seeds"]
    return {(c, a, s) for c in conditions for a in arms for s in seeds}


def build_tuple_enum_report(run_dir: Path, cells: list, prereg: dict) -> dict:
    """Guardrail 0 (STOP-IF-NO-GRID): exact tuple-enumeration check, run
    BEFORE fairness/aggregation. 5 arms x 7 conditions x 5 seeds = 175."""
    expected = expected_tuples(prereg)
    observed_list = [(c["condition"], c["arm"], c["seed"]) for c in cells]
    counts = defaultdict(int)
    for t in observed_list:
        counts[t] += 1
    observed_set = set(observed_list)

    duplicates = sorted(
        [{"condition": c, "arm": a, "seed": s, "count": n} for (c, a, s), n in counts.items() if n > 1],
        key=lambda d: (d["condition"], d["arm"], d["seed"]),
    )
    missing = sorted(expected - observed_set)
    extra = sorted(observed_set - expected)

    n_expected = len(expected)
    n_observed = len(cells)
    n_duplicates = sum(n - 1 for n in counts.values() if n > 1)
    n_missing = len(missing)
    n_extra = len(extra)
    all_pass = (n_observed == n_expected and n_duplicates == 0 and n_missing == 0 and n_extra == 0)

    return {
        "run_dir": str(run_dir),
        "n_expected": n_expected,
        "n_observed": n_observed,
        "n_duplicates": n_duplicates,
        "n_missing": n_missing,
        "n_extra": n_extra,
        "all_pass": all_pass,
        "duplicate_tuples": duplicates,
        "missing_tuples": [{"condition": c, "arm": a, "seed": s} for c, a, s in missing],
        "extra_tuples": [{"condition": c, "arm": a, "seed": s} for c, a, s in extra],
        "rule": "exact set-equality of observed (condition, arm, seed) tuples in "
                "results/secondary_20260703-1034/cells.jsonl (only this directory — other "
                "secondary_* dirs contain no cells.jsonl and are ignored per the task's guardrail 0) "
                "against the full cross product of canonical_conditions() x prereg.design.arms x "
                "prereg.design.seeds (5 arms x 7 conditions x 5 seeds = 175). Any duplicate, missing, "
                "or extra tuple fails this check and stops the script before fairness_report.json / "
                "aggregation / decision products are written.",
    }


def independent_eval_sha() -> str:
    """Recompute eval_sha256 exactly as run_grid() does at startup,
    independently of anything stored in cells.jsonl."""
    train_bearings, test_bearings = H.make_cross_condition_split(H.TRAIN_COND, H.TEST_COND)
    base_test = H.XJTUDataset(H.DATA_ROOT, test_bearings, n_sensors=1)
    clean_test_ds = H.XJTUDatasetNoisy(base_test, "clean", None, 0)
    return H.eval_fingerprint(clean_test_ds)


def build_fairness_report(run_dir: Path, cells: list) -> dict:
    """Independently recompute the eval fingerprint and compare it, full
    64-hex-char hash, byte-for-byte, against every one of the 175 cells'
    stored fairness.eval_sha256 — not just a prefix."""
    expected = independent_eval_sha()
    mismatches = []
    hash_lengths_ok = True
    for c in cells:
        got = c.get("fairness", {}).get("eval_sha256", "")
        if len(got) != 64:
            hash_lengths_ok = False
        if got != expected:
            mismatches.append({
                "condition": c["condition"], "arm": c["arm"], "seed": c["seed"],
                "got_eval_sha256": got, "expected_eval_sha256": expected,
            })
    all_pass = hash_lengths_ok and not mismatches
    report = {
        "run_dir": str(run_dir),
        "expected_eval_sha256_full": expected,
        "expected_eval_sha256_len": len(expected),
        "n_cells_checked": len(cells),
        "n_cells_full_hash_present": sum(1 for c in cells if len(c.get("fairness", {}).get("eval_sha256", "")) == 64),
        "hash_lengths_all_64": hash_lengths_ok,
        "n_mismatches": len(mismatches),
        "mismatches": mismatches,
        "all_pass": all_pass,
        "sampled_cells_full_hash": [
            {"condition": c["condition"], "arm": c["arm"], "seed": c["seed"],
             "eval_sha256": c["fairness"]["eval_sha256"]}
            for c in cells[:5]
        ],
        "rule": "byte-for-byte equality of ALL 175 cells' eval_sha256 (full 64-hex-char hash) against "
                "an independently recomputed eval_sha256 (SHA256 of the cross-condition test-split "
                "labels via xjtu_noisy_harness.make_cross_condition_split()/XJTUDataset()/"
                "eval_fingerprint() — the exact construction run_grid() uses), per "
                "secondary_prereg_20260704-0708.json:fairness_verification_required.",
    }
    return report


def aggregate(cells: list, prereg: dict):
    design = prereg["design"]
    conditions = canonical_conditions()
    arms = design["arms"]

    by_ca = defaultdict(list)  # (condition, arm) -> [ (seed, f1) ]
    for c in cells:
        by_ca[(c["condition"], c["arm"])].append((c["seed"], c["best_macro_f1"]))

    agg = {}
    for cond in conditions:
        agg[cond] = {}
        for arm in arms:
            rows = sorted(by_ca.get((cond, arm), []))
            f1s = np.array([r[1] for r in rows], dtype=np.float64)
            seeds = [r[0] for r in rows]
            mean = float(f1s.mean()) if len(f1s) else None
            std = float(f1s.std(ddof=0)) if len(f1s) else None
            agg[cond][arm] = {
                "mean_macro_f1": mean,
                "std_macro_f1": std,
                "n_seeds_used": len(rows),
                "seeds": seeds,
                "at_chance": (mean is not None and mean <= CHANCE_FLOOR),
            }
    return agg


def snr_for_plot_map() -> dict:
    """condition label -> x-axis value. F1 fix already applied here (see
    REPORT_frozensel_20260703-0748.md): the harness's own build_conditions()
    leading {'label': 'clean', 'snr_db': None} entry must not overwrite the
    hardcoded clean->15.0 plot x-value."""
    m = {"clean": 15.0}
    for c in H.build_conditions(["awgn"], H.FULL_SNRS):
        if c["label"] == "clean":
            continue
        m[c["label"]] = c["snr_db"]
    return m


def write_q_secondary(out_dir: Path, agg: dict, design: dict):
    j = {"chance_floor": CHANCE_FLOOR, "conditions": agg}
    json_path = out_dir / "q_secondary.json"
    _write_protected_json(json_path, j)

    order = canonical_conditions()
    csv_path = out_dir / "q_secondary.csv"
    rows = []
    for cond in order:
        for arm in design["arms"]:
            e = agg[cond][arm]
            rows.append([cond, arm, e["mean_macro_f1"], e["std_macro_f1"],
                         e["n_seeds_used"], "|".join(map(str, e["seeds"])), e["at_chance"]])
    _write_protected_csv(csv_path, ["condition", "arm", "mean_macro_f1", "std_macro_f1",
                                     "n_seeds_used", "seeds", "at_chance"], rows)
    return json_path, csv_path


def write_curve_table(out_dir: Path, agg: dict, design: dict):
    csv_path = out_dir / "secondary_curve_table.csv"
    order = canonical_conditions()
    snr_for_plot = snr_for_plot_map()
    rows = []
    for arm in design["arms"]:
        for cond in order:
            e = agg[cond][arm]
            rows.append([arm, cond, snr_for_plot.get(cond), e["mean_macro_f1"], e["std_macro_f1"], e["n_seeds_used"]])
    _write_protected_csv(csv_path, ["arm", "condition", "snr_db_for_plot", "mean_macro_f1",
                                     "std_macro_f1", "n_seeds_used"], rows)
    return csv_path


def write_curve_figure(out_dir: Path, agg: dict, design: dict):
    png_path = out_dir / "secondary_curve.png"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = canonical_conditions()
    snr_for_plot = snr_for_plot_map()
    xs = [snr_for_plot[c] for c in order]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = {
        "bm3_frozen": "tab:orange",
        "frozen_noconv": "tab:red",
        "frozen_nogate": "tab:purple",
        "frozen_As4d": "tab:brown",
        "s4d": "tab:green",
    }
    for arm in design["arms"]:
        means = [agg[c][arm]["mean_macro_f1"] for c in order]
        stds = [agg[c][arm]["std_macro_f1"] for c in order]
        ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3,
                     label=arm, color=colors.get(arm))
    ax.axhline(CHANCE_FLOOR, color="gray", linestyle="--", linewidth=1, label=f"chance floor ({CHANCE_FLOOR:.0%})")
    ax.set_xlabel("AWGN SNR (dB); clean plotted at 15dB")
    ax.set_ylabel("macro F1 (mean +/- std across seeds)")
    ax.set_title("Secondary single-component ablations of bm3_frozen (all 5 arms)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _protect(png_path)
    return png_path


def paired_seed_f1(cells, condition, arm):
    out = {}
    for c in cells:
        if c["condition"] == condition and c["arm"] == arm:
            if c["seed"] in out:
                raise SystemExit(f"[FATAL] duplicate seed {c['seed']} for {condition}/{arm}")
            out[c["seed"]] = c["best_macro_f1"]
    return out


def paired_test(a_by_seed: dict, b_by_seed: dict):
    seeds = sorted(set(a_by_seed) & set(b_by_seed))
    if seeds != sorted(a_by_seed) or seeds != sorted(b_by_seed):
        raise SystemExit(f"[FATAL] seed sets do not match exactly: {a_by_seed.keys()} vs {b_by_seed.keys()}")
    a = np.array([a_by_seed[s] for s in seeds])
    b = np.array([b_by_seed[s] for s in seeds])
    diffs = a - b
    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
    except ValueError:
        w_stat, w_p = None, None
    return {
        "seeds": seeds,
        "n_pairs": len(seeds),
        "mean_diff_pp": round(float(diffs.mean()) * 100, 3),
        "std_diff_pp": round(float(diffs.std(ddof=1)) * 100, 3) if len(seeds) > 1 else None,
        "per_seed_diff_pp": {int(s): round(float(d) * 100, 3) for s, d in zip(seeds, diffs)},
        "paired_ttest": {"statistic": float(t_stat), "p_value": float(t_p)},
        "wilcoxon_signed_rank": (
            {"statistic": float(w_stat), "p_value": float(w_p)}
            if w_stat is not None else {"statistic": None, "p_value": None, "note": "all pairs identical"}
        ),
        "significant_p05": bool(t_p < 0.05) if t_p is not None else None,
    }


def build_component_deltas_and_significance(cells: list, agg: dict):
    """Per component arm X in COMPONENT_ARMS: Delta(frozen-X) pointwise,
    Delta(X-s4d) pointwise (both from the mean-level aggregate), plus a
    paired (same-seed) significance test for both deltas — REPORTING ONLY
    per the prereg; not used by apply_decision_rule()."""
    order = canonical_conditions()
    deltas = {}
    significance = {}
    for arm in COMPONENT_ARMS:
        deltas[arm] = {}
        significance[arm] = {}
        for cond in order:
            frz = agg[cond][FULL_BASELINE_ARM]
            xarm = agg[cond][arm]
            s4d = agg[cond][FLOOR_BASELINE_ARM]
            deltas[arm][cond] = {
                "delta_frozen_minus_X_pp": (
                    round((frz["mean_macro_f1"] - xarm["mean_macro_f1"]) * 100, 2)
                    if frz["mean_macro_f1"] is not None and xarm["mean_macro_f1"] is not None else None
                ),
                "delta_X_minus_s4d_pp": (
                    round((xarm["mean_macro_f1"] - s4d["mean_macro_f1"]) * 100, 2)
                    if xarm["mean_macro_f1"] is not None and s4d["mean_macro_f1"] is not None else None
                ),
            }
            frz_seeds = paired_seed_f1(cells, cond, FULL_BASELINE_ARM)
            x_seeds = paired_seed_f1(cells, cond, arm)
            s4d_seeds = paired_seed_f1(cells, cond, FLOOR_BASELINE_ARM)
            significance[arm][cond] = {
                "delta_frozen_minus_X": paired_test(frz_seeds, x_seeds),
                "delta_X_minus_s4d": paired_test(x_seeds, s4d_seeds),
            }
    return deltas, significance


def write_component_deltas(out_dir: Path, deltas: dict):
    order = canonical_conditions()
    json_path = out_dir / "secondary_component_deltas.json"
    _write_protected_json(json_path, {"conditions_order": order, "component_arms": COMPONENT_ARMS,
                                       "deltas": deltas})
    csv_path = out_dir / "secondary_component_deltas.csv"
    rows = []
    for arm in COMPONENT_ARMS:
        for cond in order:
            e = deltas[arm][cond]
            rows.append([arm, cond, e["delta_frozen_minus_X_pp"], e["delta_X_minus_s4d_pp"]])
    _write_protected_csv(csv_path, ["component_arm", "condition", "delta_frozen_minus_X_pp",
                                     "delta_X_minus_s4d_pp"], rows)
    return json_path, csv_path


def write_paired_significance(out_dir: Path, significance: dict):
    order = canonical_conditions()
    json_path = out_dir / "secondary_paired_significance.json"
    n_sig_frozen_x = {arm: sum(1 for c in order if significance[arm][c]["delta_frozen_minus_X"]["significant_p05"])
                       for arm in COMPONENT_ARMS}
    n_sig_x_s4d = {arm: sum(1 for c in order if significance[arm][c]["delta_X_minus_s4d"]["significant_p05"])
                    for arm in COMPONENT_ARMS}
    report = {
        "method": "paired (same-seed) comparison across the 5 matched seeds per condition; paired "
                  "t-test (scipy.stats.ttest_rel) and Wilcoxon signed-rank (scipy.stats.wilcoxon) as a "
                  "non-parametric cross-check. REPORTING ONLY per "
                  "secondary_prereg_20260704-0708.json:per_condition_aggregation."
                  "paired_significance_reporting_only — supplements but does not replace or feed into "
                  "secondary_decision.json's decisive_band_rule verdicts.",
        "alpha": 0.05,
        "conditions_order": order,
        "component_arms": COMPONENT_ARMS,
        "n_conditions_frozen_minus_X_significant_p05": n_sig_frozen_x,
        "n_conditions_X_minus_s4d_significant_p05": n_sig_x_s4d,
        "per_component_arm": significance,
    }
    _write_protected_json(json_path, report)

    csv_path = out_dir / "secondary_paired_significance.csv"
    rows = []
    for arm in COMPONENT_ARMS:
        for cond in order:
            for comp_key, comp_label in [("delta_frozen_minus_X", "frozen_minus_X"),
                                          ("delta_X_minus_s4d", "X_minus_s4d")]:
                e = significance[arm][cond][comp_key]
                rows.append([
                    arm, cond, comp_label, e["mean_diff_pp"], e["std_diff_pp"],
                    e["paired_ttest"]["statistic"], e["paired_ttest"]["p_value"],
                    e["wilcoxon_signed_rank"]["statistic"], e["wilcoxon_signed_rank"]["p_value"],
                    e["significant_p05"],
                ])
    _write_protected_csv(csv_path, ["component_arm", "condition", "comparison", "mean_diff_pp",
                                     "std_diff_pp", "paired_ttest_stat", "paired_ttest_p",
                                     "wilcoxon_stat", "wilcoxon_p", "significant_p05"], rows)
    return json_path, csv_path


def verify_prereg_provenance_chain(prereg: dict, prereg_path: Path, run_dir: Path):
    """Returns (ok: bool, reason: str). ok=True only if prereg['written_at']
    strictly precedes run_dir's own grid-start timestamp, parsed from the
    run_dir name's '_<YYYYMMDD-HHMM>' suffix -- the same convention every
    results/*_<TS>/ directory in this project already uses. This is the
    rule-date-before-grid-start chain required for a valid §E3 preregistered
    adjudication (see results/secondary_e3ruling_20260704-2206/prereg_provenance.md
    for the manually-verified derivation this function automates)."""
    written_at = prereg.get("written_at")
    if not written_at:
        return False, f"{prereg_path} has no written_at timestamp"
    m = re.search(r"_(\d{8})-(\d{4})$", run_dir.name)
    if not m:
        return False, f"run_dir name {run_dir.name!r} does not embed a _YYYYMMDD-HHMM grid-start timestamp"
    grid_start = datetime.strptime("".join(m.groups()), "%Y%m%d%H%M")
    try:
        prereg_time = datetime.strptime(written_at, "%Y-%m-%dT%H:%M")
    except ValueError:
        return False, f"written_at={written_at!r} is not in YYYY-MM-DDTHH:MM format"
    if prereg_time < grid_start:
        return True, f"prereg written_at={written_at} precedes grid start {grid_start.isoformat()}"
    return False, (f"prereg written_at={written_at} does NOT precede grid start {grid_start.isoformat()} "
                    f"for {run_dir} -- the grid already existed (or had already started) when the rule "
                    f"was authored, so this cannot be a valid preregistration for this run_dir")


def apply_deltapp_decision_rule(agg: dict, cells: list, prereg: dict, prereg_path: Path,
                                 run_dir: Path, out_dir: Path, deltas: dict):
    """Mechanically applies a preregistered Delta-pp threshold rule (schema:
    prereg['decisive_rule_delta_pp_threshold'], e.g.
    secondary_deltapp_prereg_20260704-0719.json). This is the ONLY function in
    this module permitted to emit adjudicates_E3=True, and only once every
    assertion below passes. The former mean+/-std band heuristic
    (band_significant()) has been deleted from this module entirely -- §H2 fix,
    .orchestrate/0704-215713/r1_review.json and the later review of
    results/secondary_e3ruling_20260704-2206/secondary_decision_corrected.json."""
    assert "decisive_rule_delta_pp_threshold" in prereg, (
        f"{prereg_path} is not a Delta-pp prereg (missing decisive_rule_delta_pp_threshold key); "
        "refusing to run the E3 Delta-pp adjudicator against the wrong rule schema")
    threshold = prereg["decisive_rule_delta_pp_threshold"]["DELTA_PP_THRESHOLD"]

    ok, chain_reason = verify_prereg_provenance_chain(prereg, prereg_path, run_dir)
    reviewer_approved = prereg.get("status") == "REVIEWER_APPROVED"
    can_adjudicate_e3 = bool(ok and reviewer_approved)

    design = prereg["design"]
    order = canonical_conditions()
    min_seeds = design["min_acceptable_seeds"]

    per_component_verdicts = {}
    per_component_evidence = {}
    for arm in COMPONENT_ARMS:
        evidence = []
        n_meaningful = 0
        n_frozen_wins = 0
        n_x_wins = 0
        n_frozen_losses = 0
        n_cells_pair = 0
        for cond in order:
            frz = agg[cond][FULL_BASELINE_ARM]
            xarm = agg[cond][arm]
            s4d = agg[cond][FLOOR_BASELINE_ARM]
            n_cells_pair += frz["n_seeds_used"] + xarm["n_seeds_used"]
            meaningful = (frz["mean_macro_f1"] is not None and xarm["mean_macro_f1"] is not None
                          and frz["mean_macro_f1"] > CHANCE_FLOOR and xarm["mean_macro_f1"] > CHANCE_FLOOR)
            delta = deltas[arm][cond]["delta_frozen_minus_X_pp"]
            if not meaningful:
                result = "excluded_at_chance"
            elif delta >= threshold:
                result = "frozen_wins"
            elif delta <= -threshold:
                result = "X_wins"
            else:
                result = "tie"
            if meaningful:
                n_meaningful += 1
                if result == "frozen_wins":
                    n_frozen_wins += 1
                elif result == "X_wins":
                    n_x_wins += 1
                    n_frozen_losses += 1
            evidence.append({
                "condition": cond,
                "meaningful": meaningful,
                "deltapp_result": result,
                "bm3_frozen": {"mean_macro_f1": frz["mean_macro_f1"], "std_macro_f1": frz["std_macro_f1"],
                               "n_seeds_used": frz["n_seeds_used"], "seeds": frz["seeds"], "at_chance": frz["at_chance"]},
                arm: {"mean_macro_f1": xarm["mean_macro_f1"], "std_macro_f1": xarm["std_macro_f1"],
                      "n_seeds_used": xarm["n_seeds_used"], "seeds": xarm["seeds"], "at_chance": xarm["at_chance"]},
                "s4d": {"mean_macro_f1": s4d["mean_macro_f1"], "std_macro_f1": s4d["std_macro_f1"],
                        "n_seeds_used": s4d["n_seeds_used"], "seeds": s4d["seeds"], "at_chance": s4d["at_chance"]},
                "delta_frozen_minus_X_pp": delta,
                "delta_X_minus_s4d_pp_context_only": deltas[arm][cond]["delta_X_minus_s4d_pp"],
            })

        if n_cells_pair < min_seeds * len(order) * 2:
            verdict = "INCONCLUSIVE"
            reason = f"only {n_cells_pair} bm3_frozen+{arm} cells present, below min ({min_seeds * len(order) * 2})"
        elif n_meaningful < 2:
            verdict = "INCONCLUSIVE"
            reason = f"only {n_meaningful} meaningful condition(s) (both arms > chance floor {CHANCE_FLOOR})"
        else:
            strict_majority = n_frozen_wins > (n_meaningful / 2)
            if strict_majority and n_frozen_losses == 0:
                verdict = "COMPONENT_CONTRIBUTES"
                reason = (f"bm3_frozen significantly beats {arm} (delta_frozen_minus_X_pp >= {threshold}) in "
                          f"{n_frozen_wins}/{n_meaningful} meaningful conditions (strict majority), and does "
                          f"not lose significantly to {arm} in any meaningful condition.")
            else:
                verdict = "COMPONENT_NOT_SHOWN_TO_CONTRIBUTE"
                reason = (f"bm3_frozen does not significantly beat {arm} in a strict majority of meaningful "
                          f"conditions ({n_frozen_wins}/{n_meaningful} frozen-wins; {n_x_wins}/{n_meaningful} "
                          f"{arm}-wins; ties fall through to this default outcome per tie_break).")

        per_component_verdicts[arm] = {
            "verdict": verdict,
            "reason": reason,
            "n_meaningful_conditions": n_meaningful,
            "n_frozen_significantly_wins": n_frozen_wins,
            "n_X_significantly_wins": n_x_wins,
            "n_cells_bm3_frozen_plus_X": n_cells_pair,
        }
        per_component_evidence[arm] = evidence

    contributes = [a for a in COMPONENT_ARMS if per_component_verdicts[a]["verdict"] == "COMPONENT_CONTRIBUTES"]
    inconclusive = [a for a in COMPONENT_ARMS if per_component_verdicts[a]["verdict"] == "INCONCLUSIVE"]
    if inconclusive:
        overall_verdict = "OVERALL_INCONCLUSIVE"
        overall_reason = f"component(s) {inconclusive} are INCONCLUSIVE at the per-component level."
    elif len(contributes) == 3:
        overall_verdict = "ALL_COMPONENTS_CONTRIBUTE"
        overall_reason = "all 3 component arms (frozen_noconv, frozen_nogate, frozen_As4d) are COMPONENT_CONTRIBUTES."
    elif len(contributes) == 0:
        overall_verdict = "NO_COMPONENTS_SHOWN_TO_CONTRIBUTE"
        overall_reason = "no component arm is COMPONENT_CONTRIBUTES."
    else:
        overall_verdict = "PARTIAL_COMPONENTS_CONTRIBUTE"
        overall_reason = f"component(s) {contributes} are COMPONENT_CONTRIBUTES; the rest are not."

    decision = {
        "decision_kind": "mechanical_deltapp_prereg_application",
        "prereg_id": prereg["prereg_id"],
        "prereg_path": str(prereg_path),
        "prereg_source": str(prereg_path) if ok else None,
        "responds_to_reviewer": prereg.get("responds_to_reviewer"),
        "reviewer_verdict_file": _resolve_latest_reviewer_verdict(),
        "reviewer_verdict_file_scope": "latest CLOSED round with status=='passed' -- general "
            "implementation-provenance citation only. NOT necessarily the last reviewer verdict on "
            "the E3 question; see last_effective_reviewer_verdict_file for that.",
        "last_effective_reviewer_verdict_file": _resolve_last_effective_reviewer_verdict(),
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "n_cells_total": len(cells),
        "n_cells_expected_full": design.get("cells_expected_full"),
        "chance_floor": CHANCE_FLOOR,
        "decisive_rule": f"Delta-pp threshold: bm3_frozen significantly beats X iff "
                          f"delta_frozen_minus_X_pp >= {threshold}; X significantly beats bm3_frozen iff "
                          f"delta_frozen_minus_X_pp <= -{threshold} ({prereg_path}:"
                          "decisive_rule_delta_pp_threshold). The mean+/-std band heuristic is NOT used "
                          "anywhere in this function.",
        "preregistered": ok,
        "adjudicates_E3": can_adjudicate_e3,
        "prereg_provenance_chain_reason": chain_reason,
        "prereg_reviewer_status": prereg.get("status"),
        "per_component_verdicts": per_component_verdicts,
        "overall_verdict": overall_verdict,
        "overall_reason": overall_reason,
        "per_component_evidence": per_component_evidence,
        "source_data": "q_secondary.json in this out_dir (mean_macro_f1 per condition x arm)",
    }

    # --- prevention assertions: refuse to emit an invalid or falsely-PASSing artifact ---
    allowed_component = {"COMPONENT_CONTRIBUTES", "COMPONENT_NOT_SHOWN_TO_CONTRIBUTE", "INCONCLUSIVE"}
    for arm in COMPONENT_ARMS:
        assert per_component_verdicts[arm]["verdict"] in allowed_component, \
            f"invalid per-component verdict enum: {per_component_verdicts[arm]['verdict']!r}"
    allowed_overall = {"ALL_COMPONENTS_CONTRIBUTE", "PARTIAL_COMPONENTS_CONTRIBUTE",
                       "NO_COMPONENTS_SHOWN_TO_CONTRIBUTE", "OVERALL_INCONCLUSIVE"}
    assert overall_verdict in allowed_overall, f"invalid overall verdict enum: {overall_verdict!r}"
    assert decision["prereg_source"] is not None or not decision["adjudicates_E3"], (
        "refusing to write: adjudicates_E3=True but prereg_source is missing")
    if not ok:
        assert decision["adjudicates_E3"] is False, (
            f"refusing to write: provenance chain invalid ({chain_reason}) but adjudicates_E3 was True")
    if not reviewer_approved:
        assert decision["adjudicates_E3"] is False, (
            f"refusing to write: prereg status={prereg.get('status')!r} is not REVIEWER_APPROVED "
            "but adjudicates_E3 was True")
    return decision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    prereg_path = Path(args.prereg)
    prereg = json.loads(prereg_path.read_text())
    design = prereg["design"]
    assert design["arms"] == ALL_ARMS, (design["arms"], ALL_ARMS)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)  # must be a brand-new dir (additive-only guardrail)

    cells = load_cells(run_dir)
    print(f"[LOAD] {len(cells)} cells from {run_dir}/cells.jsonl (read-only)")

    tuple_report = build_tuple_enum_report(run_dir, cells, prereg)
    tpath = out_dir / "tuple_enum_report.json"
    _write_protected_json(tpath, tuple_report)
    print(f"[TUPLE] n_expected={tuple_report['n_expected']} n_observed={tuple_report['n_observed']} "
          f"n_duplicates={tuple_report['n_duplicates']} n_missing={tuple_report['n_missing']} "
          f"n_extra={tuple_report['n_extra']} all_pass={tuple_report['all_pass']} -> {tpath}")

    if not tuple_report["all_pass"]:
        print("[STOP] tuple_enum_report.all_pass=False; refusing to proceed (guardrail 0 / E0).")
        sys.exit(4)

    fairness = build_fairness_report(run_dir, cells)
    fpath = out_dir / "fairness_report.json"
    _write_protected_json(fpath, fairness)
    print(f"[FAIRNESS] all_pass={fairness['all_pass']} n_mismatches={fairness['n_mismatches']} -> {fpath}")

    if not fairness["all_pass"]:
        print("[STOP] fairness_report.all_pass=False; refusing to aggregate or emit a decision.")
        sys.exit(3)

    agg = aggregate(cells, prereg)
    jpath, cpath = write_q_secondary(out_dir, agg, design)
    print(f"[AGG] wrote {jpath}, {cpath}")

    curve_csv = write_curve_table(out_dir, agg, design)
    print(f"[CURVE] wrote {curve_csv}")

    curve_png = write_curve_figure(out_dir, agg, design)
    print(f"[CURVE] wrote {curve_png}")

    deltas, significance = build_component_deltas_and_significance(cells, agg)
    dj, dc = write_component_deltas(out_dir, deltas)
    print(f"[DELTAS] wrote {dj}, {dc}")

    sj, sc = write_paired_significance(out_dir, significance)
    print(f"[SIGNIFICANCE] wrote {sj}, {sc} (reporting only)")

    is_deltapp_prereg = "decisive_rule_delta_pp_threshold" in prereg
    if is_deltapp_prereg:
        decision = apply_deltapp_decision_rule(agg, cells, prereg, prereg_path, run_dir, out_dir, deltas)
        dpath = out_dir / "secondary_decision.json"
        _write_protected_json(dpath, decision)
        print(f"[DECISION] overall_verdict={decision['overall_verdict']} -> {dpath}")
        for arm in COMPONENT_ARMS:
            v = decision["per_component_verdicts"][arm]
            print(f"[DECISION]   {arm}: {v['verdict']} ({v['reason']})")
    else:
        # §H2/§G2/§G3 fix (reviewer finding on results/secondary_e3ruling_20260704-2206/
        # secondary_decision_corrected.json): the mean+/-std band heuristic decision writer
        # (formerly apply_post_hoc_band_report()/band_significant()) has been deleted from this
        # module. apply_deltapp_decision_rule() is now the ONLY decision-emitting path. A prereg
        # lacking 'decisive_rule_delta_pp_threshold' cannot produce any per-component or overall
        # verdict here -- there is no fallback heuristic left to reproduce one. This is a
        # standalone no_decision record, not a decision artifact: no per_component_verdicts, no
        # overall_verdict, preregistered/adjudicates_E3 both hard-false.
        no_decision = {
            "decision_kind": "no_decision_non_deltapp_prereg",
            "prereg_id": prereg.get("prereg_id"),
            "prereg_path": str(prereg_path),
            "run_dir": str(run_dir),
            "out_dir": str(out_dir),
            "preregistered": False,
            "adjudicates_E3": False,
            "prereg_source": None,
            "per_component_verdicts": None,
            "overall_verdict": None,
            "reason": f"{prereg_path} does not define 'decisive_rule_delta_pp_threshold'. The "
                "mean+/-std band heuristic that used to fill this gap has been removed from "
                "secondary_aggregate.py; no decision-emitting rule exists for this prereg schema. "
                "No E3 adjudication was attempted or performed for this run. To obtain a decision "
                "for this run_dir, supply a prereg that defines decisive_rule_delta_pp_threshold "
                "(apply_deltapp_decision_rule() will then run and will independently determine "
                "whether it can also set adjudicates_E3=True, based on provenance-chain and "
                "reviewer-approval status).",
        }
        npath = out_dir / "secondary_no_decision.json"
        _write_protected_json(npath, no_decision)
        print(f"[NO-DECISION] prereg lacks decisive_rule_delta_pp_threshold; no verdicts emitted -> {npath}")


if __name__ == "__main__":
    main()

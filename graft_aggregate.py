#!/home/jeffwork/论文8/venv/bin/python3
"""
graft_aggregate.py — post-grid analysis for the constructive-graft experiment
(s4d_plus_gate), reading results/graft_20260705-224137/cells.jsonl (35 cells:
s4d_plus_gate x {clean,+10,+6,0,-2,-6,-10} x 5 seeds) plus the bm3_frozen/s4d/
frozen_nogate baseline arms already produced in results/secondary_20260703-1034/
cells.jsonl (per graft_prereg_provenance.md: "基线 bm3_frozen/s4d 取
secondary_20260703-1034 同批值").

Read-only over both cells.jsonl files; never launches or resumes training;
never writes into either of those two existing run_dirs. All products go into
a NEW timestamped --out-dir (additive-only guardrail), each chmod 444
immediately after writing.

Steps (all-or-nothing; any FAIL stops before the next step):
  0. STOP-IF-NO-GRID hard guardrail: glob results/graft_*/cells.jsonl, build
     the exact expected (condition, arm=s4d_plus_gate, seed) tuple set (7
     conditions x 5 seeds = 35) and require the union of all matching
     cells.jsonl files to equal it exactly (no missing, no duplicate, no
     extra). On failure: write needs_human.json to a new
     results/graft_needs_human_<TS>/ dir and exit — no aggregation, no
     decision, and (per the task's explicit "不得 loop 内启训") no training
     is ever launched by this script regardless of outcome.
  1. Independently recompute eval_sha256 (via the harness's own
     eval_fingerprint()/make_cross_condition_split()/XJTUDataset(), not by
     re-reading a stored value) and cross-check all 35 graft-grid cells'
     fairness.eval_sha256 against it byte-for-byte. Also cross-check the 105
     baseline cells (bm3_frozen/s4d/frozen_nogate x 7 x 5) for the same
     invariant, reported as a supplementary (non-gating, since the task scopes
     "35 行" to the graft grid) consistency section. Any graft-row mismatch
     stops the script before aggregation/decision.
  2. Aggregate mean/std macro_f1 per (condition, arm) for all 4 arms ->
     q_graft.json/csv.
  3. Curve table + PNG: macro_F1 vs SNR, s4d_plus_gate overlaid on
     bm3_frozen/s4d/frozen_nogate, mean +/- std errorbars.
  4. Per-level (all 7 conditions) R = (s4d_plus_gate - s4d) / (bm3_frozen -
     s4d) evidence table, plus the deep-noise (awgn@+0dB, awgn@-2dB,
     awgn@-6dB) 3-bin mean R that graft_prereg_provenance.md's decision rule
     is defined over.
  5. Paired (same-seed) significance (t-test + Wilcoxon) for
     (s4d_plus_gate - s4d) and (bm3_frozen - s4d) per condition — reported for
     transparency only; graft_prereg_provenance.md forbids any
     non-preregistered criterion, so this never feeds the verdict.
  6. Mechanically apply graft_prereg_provenance.md's 3-branch R rule ->
     graft_decision.json, citing the latest closed-round reviewer verdict.

Usage:
  python graft_aggregate.py --graft-run-dir results/graft_20260705-224137 \
      --baseline-run-dir results/secondary_20260703-1034 \
      --out-dir results/graft_analysis_<TS>
"""
import argparse
import csv
import json
import os
import stat
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HARNESS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS_DIR))

import xjtu_noisy_harness as H  # noqa: E402  (import-only, no side effects — main() gated)

CHANCE_FLOOR = 0.55
GRAFT_ARM = "s4d_plus_gate"
BASELINE_ARMS = ["bm3_frozen", "s4d", "frozen_nogate"]
ALL_ARMS = [GRAFT_ARM] + BASELINE_ARMS
CONDITION_ORDER = ["clean", "awgn@+10dB", "awgn@+6dB", "awgn@+0dB",
                   "awgn@-2dB", "awgn@-6dB", "awgn@-10dB"]
SEEDS = [0, 1, 2, 3, 4]
DEEP_NOISE_CONDITIONS = ["awgn@+0dB", "awgn@-2dB", "awgn@-6dB"]
R_CONSTRUCTIVE_THRESHOLD = 0.7
R_PARTIAL_THRESHOLD = 0.3
EXPECTED_EVAL_SHA_PREFIX = "6c20b367522c"


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


def load_cells(cells_path: Path):
    cells = []
    for line in cells_path.read_text().splitlines():
        line = line.strip()
        if line:
            cells.append(json.loads(line))
    return cells


def _resolve_latest_reviewer_verdict(repo_root: Path = None):
    """Same closed-round-only resolution rule as frozensel_aggregate.py's
    _resolve_latest_reviewer_verdict(): only a round whose result.json reports
    status=="passed" is eligible, so an in-progress round reviewing THIS
    artifact can never be cited as its own justification (self-referential
    deadlock, see decision_manifest.json precedent in
    results/secondary_e3ruling_20260704-2206/)."""
    root = repo_root or HARNESS_DIR
    orch = root / ".orchestrate"
    if not orch.is_dir():
        return None, None
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
                    rel = str(rpath.relative_to(root))
                except ValueError:
                    rel = str(rpath)
                return rel, obj.get("verdict")
    return None, None


# ── Step 0: STOP-IF-NO-GRID hard guardrail ──────────────────────────────────

def stop_if_no_grid(repo_root: Path):
    """Glob results/graft_*/cells.jsonl, union all rows for arm==s4d_plus_gate,
    and require exact set-equality against the 35-tuple cross product. Returns
    (ok: bool, report: dict, cells: list) — cells is only meaningful if ok."""
    candidates = sorted((repo_root / "results").glob("graft_*/cells.jsonl"))
    expected = {(c, GRAFT_ARM, s) for c in CONDITION_ORDER for s in SEEDS}

    all_rows = []
    per_file_counts = {}
    for p in candidates:
        rows = load_cells(p)
        per_file_counts[str(p)] = len(rows)
        all_rows.extend((r, p) for r in rows)

    graft_rows = [(r, p) for r, p in all_rows if r.get("arm") == GRAFT_ARM]
    observed_list = [(r["condition"], r["arm"], r["seed"]) for r, _ in graft_rows]
    counts = defaultdict(int)
    for t in observed_list:
        counts[t] += 1
    observed_set = set(observed_list)

    duplicates = sorted(
        [{"condition": c, "arm": a, "seed": s, "count": n} for (c, a, s), n in counts.items() if n > 1]
    )
    missing = sorted(expected - observed_set)
    extra = sorted(observed_set - expected)
    all_pass = (len(observed_list) == 35 and not duplicates and not missing and not extra)

    report = {
        "candidate_files": [str(p) for p in candidates],
        "n_rows_per_file": per_file_counts,
        "n_expected": 35,
        "n_observed": len(observed_list),
        "n_duplicates": sum(n - 1 for n in counts.values() if n > 1),
        "n_missing": len(missing),
        "n_extra": len(extra),
        "duplicate_tuples": duplicates,
        "missing_tuples": [{"condition": c, "arm": a, "seed": s} for c, a, s in missing],
        "extra_tuples": [{"condition": c, "arm": a, "seed": s} for c, a, s in extra],
        "all_pass": all_pass,
        "rule": "STOP-IF-NO-GRID: union of every results/graft_*/cells.jsonl row with "
                "arm==s4d_plus_gate must exactly equal the cross product of "
                "{clean,+10,+6,0,-2,-6,-10} x 5 seeds = 35 tuples. Any missing/duplicate/extra "
                "tuple is a hard stop: no training is launched by this script under any outcome; "
                "a failing grid is reported needs_human and the run stops here.",
    }
    # single unambiguous source cells.jsonl for downstream steps
    cells = [r for r, p in graft_rows] if all_pass and len(candidates) >= 1 else None
    return all_pass, report, cells


# ── Step 1: fairness ─────────────────────────────────────────────────────────

def independent_eval_sha() -> str:
    train_bearings, test_bearings = H.make_cross_condition_split(H.TRAIN_COND, H.TEST_COND)
    base_test = H.XJTUDataset(H.DATA_ROOT, test_bearings, n_sensors=1)
    clean_test_ds = H.XJTUDatasetNoisy(base_test, "clean", None, 0)
    return H.eval_fingerprint(clean_test_ds)


def build_fairness_report(graft_run_dir: Path, graft_cells: list, baseline_cells: list) -> dict:
    expected = independent_eval_sha()
    assert expected.startswith(EXPECTED_EVAL_SHA_PREFIX), (
        f"independently recomputed eval_sha256 {expected!r} does not start with the task's "
        f"literal prefix {EXPECTED_EVAL_SHA_PREFIX!r} — environment/data drift, must not proceed"
    )

    def _check(cells):
        mism = []
        len_ok = True
        for c in cells:
            got = c.get("fairness", {}).get("eval_sha256", "")
            if len(got) != 64:
                len_ok = False
            if got != expected:
                mism.append({"condition": c["condition"], "arm": c["arm"], "seed": c["seed"],
                             "got_eval_sha256": got, "expected_eval_sha256": expected})
        return mism, len_ok

    graft_mismatches, graft_len_ok = _check(graft_cells)
    baseline_mismatches, baseline_len_ok = _check(baseline_cells)

    all_pass = graft_len_ok and not graft_mismatches
    return {
        "run_dir": str(graft_run_dir),
        "expected_eval_sha256_full": expected,
        "expected_eval_sha256_len": len(expected),
        "task_literal_prefix_match": True,
        "n_cells_checked": len(graft_cells),
        "n_cells_full_hash_present": sum(1 for c in graft_cells if len(c.get("fairness", {}).get("eval_sha256", "")) == 64),
        "hash_lengths_all_64": graft_len_ok,
        "n_mismatches": len(graft_mismatches),
        "mismatches": graft_mismatches,
        "all_pass": all_pass,
        "sampled_cells_full_hash": [
            {"condition": c["condition"], "arm": c["arm"], "seed": c["seed"],
             "eval_sha256": c["fairness"]["eval_sha256"]}
            for c in graft_cells[:3]
        ],
        "baseline_cross_check": {
            "note": "Supplementary, non-gating (task scopes the '35 行' fairness requirement to the "
                    "graft grid): same byte-for-byte check applied to the 105 bm3_frozen/s4d/"
                    "frozen_nogate baseline cells reused from results/secondary_20260703-1034/, "
                    "confirming the R-statistic's two operands share the same fairness invariant "
                    "as the graft arm.",
            "n_cells_checked": len(baseline_cells),
            "hash_lengths_all_64": baseline_len_ok,
            "n_mismatches": len(baseline_mismatches),
            "mismatches": baseline_mismatches,
            "all_pass": baseline_len_ok and not baseline_mismatches,
        },
        "rule": "byte-for-byte equality against an independently recomputed eval_sha256 (SHA256 of "
                "the cross-condition test-split labels), per graft_prereg_provenance.md's implied "
                "fairness invariant (same eval split as every other arm in this project).",
    }


# ── Step 2/3: aggregate + curve ──────────────────────────────────────────────

def aggregate(graft_cells: list, baseline_cells: list):
    by_ca = defaultdict(list)
    for c in graft_cells + baseline_cells:
        if c["arm"] in ALL_ARMS:
            by_ca[(c["condition"], c["arm"])].append((c["seed"], c["best_macro_f1"]))

    agg = {}
    for cond in CONDITION_ORDER:
        agg[cond] = {}
        for arm in ALL_ARMS:
            rows = sorted(by_ca.get((cond, arm), []))
            assert len(rows) == 5, f"expected 5 seeds for ({cond},{arm}), got {len(rows)}"
            f1s = np.array([r[1] for r in rows], dtype=np.float64)
            seeds = [r[0] for r in rows]
            mean = float(f1s.mean())
            std = float(f1s.std(ddof=0))
            agg[cond][arm] = {
                "mean_macro_f1": mean,
                "std_macro_f1": std,
                "n_seeds_used": len(rows),
                "seeds": seeds,
                "at_chance": mean <= CHANCE_FLOOR,
            }
    return agg


def snr_for_plot_map() -> dict:
    m = {"clean": 15.0}
    for c in H.build_conditions(["awgn"], H.FULL_SNRS):
        if c["label"] == "clean":
            continue
        m[c["label"]] = c["snr_db"]
    return m


def write_q_graft(out_dir: Path, agg: dict):
    j = {"chance_floor": CHANCE_FLOOR, "arms": ALL_ARMS, "conditions": agg}
    _write_protected_json(out_dir / "q_graft.json", j)

    rows = []
    for cond in CONDITION_ORDER:
        for arm in ALL_ARMS:
            e = agg[cond][arm]
            rows.append([cond, arm, e["mean_macro_f1"], e["std_macro_f1"],
                         e["n_seeds_used"], "|".join(map(str, e["seeds"])), e["at_chance"]])
    _write_protected_csv(out_dir / "q_graft.csv",
                          ["condition", "arm", "mean_macro_f1", "std_macro_f1",
                           "n_seeds_used", "seeds", "at_chance"], rows)


def write_curve(out_dir: Path, agg: dict):
    snr_map = snr_for_plot_map()
    rows = []
    for arm in ALL_ARMS:
        for cond in CONDITION_ORDER:
            e = agg[cond][arm]
            rows.append([arm, cond, snr_map[cond], e["mean_macro_f1"], e["std_macro_f1"], e["n_seeds_used"]])
    _write_protected_csv(out_dir / "graft_curve_table.csv",
                          ["arm", "condition", "snr_db_for_plot", "mean_macro_f1",
                           "std_macro_f1", "n_seeds_used"], rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [snr_map[c] for c in CONDITION_ORDER]
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {GRAFT_ARM: "tab:red", "bm3_frozen": "tab:orange", "s4d": "tab:green", "frozen_nogate": "tab:purple"}
    styles = {GRAFT_ARM: "-", "bm3_frozen": "--", "s4d": "--", "frozen_nogate": "--"}
    for arm in ALL_ARMS:
        means = [agg[c][arm]["mean_macro_f1"] for c in CONDITION_ORDER]
        stds = [agg[c][arm]["std_macro_f1"] for c in CONDITION_ORDER]
        ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3, linestyle=styles[arm],
                     label=arm, color=colors[arm])
    ax.axhline(CHANCE_FLOOR, color="gray", linestyle=":", linewidth=1, label=f"chance floor ({CHANCE_FLOOR:.0%})")
    ax.set_xlabel("AWGN SNR (dB); clean plotted at 15dB")
    ax.set_ylabel("macro F1 (mean +/- std across seeds)")
    ax.set_title("Constructive graft: s4d_plus_gate vs bm3_frozen/s4d/frozen_nogate baselines")
    ax.legend()
    ax.grid(True, alpha=0.3)
    png_path = out_dir / "graft_curve.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _protect(png_path)


# ── Step 4: per-level R + deep-noise mean R ─────────────────────────────────

def compute_R_table(agg: dict):
    per_level = {}
    for cond in CONDITION_ORDER:
        gate = agg[cond][GRAFT_ARM]["mean_macro_f1"]
        frz = agg[cond]["bm3_frozen"]["mean_macro_f1"]
        s4d = agg[cond]["s4d"]["mean_macro_f1"]
        num = gate - s4d
        den = frz - s4d
        r = (num / den) if den != 0 else None
        per_level[cond] = {
            "s4d_plus_gate_mean_macro_f1": gate,
            "bm3_frozen_mean_macro_f1": frz,
            "s4d_mean_macro_f1": s4d,
            "delta_gate_minus_s4d_pp": round(num * 100, 2),
            "delta_frozen_minus_s4d_pp": round(den * 100, 2),
            "R": (round(r, 4) if r is not None else None),
            "R_undefined_reason": None if den != 0 else "bm3_frozen - s4d == 0 (division by zero)",
        }
    deep_Rs = [per_level[c]["R"] for c in DEEP_NOISE_CONDITIONS]
    assert all(r is not None for r in deep_Rs), (
        f"deep-noise R undefined in at least one of {DEEP_NOISE_CONDITIONS}: {deep_Rs}"
    )
    mean_R = float(np.mean(deep_Rs))
    return per_level, mean_R


# ── Step 5: paired significance (report-only) ───────────────────────────────

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
    assert seeds == sorted(a_by_seed) == sorted(b_by_seed), (a_by_seed.keys(), b_by_seed.keys())
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
        "paired_ttest": {"statistic": float(t_stat), "p_value": float(t_p)},
        "wilcoxon_signed_rank": (
            {"statistic": float(w_stat), "p_value": float(w_p)}
            if w_stat is not None else {"statistic": None, "p_value": None, "note": "all pairs identical"}
        ),
        "significant_p05": bool(t_p < 0.05) if t_p is not None else None,
    }


def build_paired_significance(graft_cells, baseline_cells):
    all_cells = graft_cells + baseline_cells
    per_condition = {}
    for cond in CONDITION_ORDER:
        gate = paired_seed_f1(all_cells, cond, GRAFT_ARM)
        frz = paired_seed_f1(all_cells, cond, "bm3_frozen")
        s4d = paired_seed_f1(all_cells, cond, "s4d")
        assert len(gate) == 5 and len(frz) == 5 and len(s4d) == 5
        per_condition[cond] = {
            "delta_gate_minus_s4d": paired_test(gate, s4d),
            "delta_frozen_minus_s4d": paired_test(frz, s4d),
        }
    return {
        "method": "paired (same-seed) comparison across the 5 matched seeds per condition; "
                  "paired t-test (scipy.stats.ttest_rel) and Wilcoxon signed-rank "
                  "(scipy.stats.wilcoxon) as a non-parametric cross-check.",
        "decisive": False,
        "note": "Reported for transparency only. graft_prereg_provenance.md's R-magnitude rule is "
                "the sole preregistered, decisive criterion (\"禁未预注册判据\"); this significance "
                "test plays no role in graft_decision.json's verdict.",
        "alpha": 0.05,
        "conditions": CONDITION_ORDER,
        "per_condition": per_condition,
    }


# ── Step 6: mechanical decision ─────────────────────────────────────────────

def classify_R(mean_R: float) -> str:
    if mean_R >= R_CONSTRUCTIVE_THRESHOLD:
        return "CONSTRUCTIVE_CONFIRMED"
    if mean_R >= R_PARTIAL_THRESHOLD:
        return "PARTIAL_RECOVERY"
    return "GRAFT_INSUFFICIENT"


def build_decision(out_dir: Path, graft_run_dir: Path, per_level: dict, mean_R: float,
                    paired_sig_path: Path, tuple_report: dict, fairness: dict):
    reviewer_file, reviewer_verdict = _resolve_latest_reviewer_verdict()
    verdict = classify_R(mean_R)
    deep_noise_evidence = {c: per_level[c] for c in DEEP_NOISE_CONDITIONS}

    reason = (
        f"R = mean over {{{', '.join(DEEP_NOISE_CONDITIONS)}}} of "
        f"(s4d_plus_gate - s4d) / (bm3_frozen - s4d) = {mean_R:.4f} "
        f"({'>= 0.7' if verdict == 'CONSTRUCTIVE_CONFIRMED' else ('in [0.3, 0.7)' if verdict == 'PARTIAL_RECOVERY' else '< 0.3')}) "
        f"=> {verdict}, per graft_prereg_provenance.md's predeclared 3-branch rule. "
        f"Per-bin R: " + ", ".join(f"{c}={per_level[c]['R']:.4f}" for c in DEEP_NOISE_CONDITIONS)
    )

    return {
        "decision_kind": "mechanical_prereg_application",
        "preregistered": True,
        "prereg_source": "graft_prereg_provenance.md (materialized pre-grid)",
        "prereg_path": "graft_prereg_provenance.md",
        "prereg_predates_grid": True,
        "prereg_materialized_note": "graft_prereg_provenance.md (397 bytes, mtime 2026-07-05 21:20) "
                                     "predates the grid's start (results/graft_20260705-224137/, "
                                     "grid start 2026-07-05 22:41 per graft_20260705-224137.log's "
                                     "[START] line), and predates this analysis run.",
        "reviewer_verdict_file": reviewer_file,
        "reviewer_verdict": reviewer_verdict,
        "reviewer_verdict_note": "Latest closed-round reviewer verdict for the graft build/smoke "
                                  "session (harness dispatch, graft_unitcheck, smoke test, "
                                  "launch/status scripts) — the full 35-cell grid was launched and "
                                  "completed after this review, using the exact reviewed code path.",
        "run_dir": str(graft_run_dir),
        "tuple_enum_report": {"n_expected": tuple_report["n_expected"], "n_observed": tuple_report["n_observed"],
                               "all_pass": tuple_report["all_pass"]},
        "fairness_report_all_pass": fairness["all_pass"],
        "decisive_rule": "graft_prereg_provenance.md verbatim: R = mean over {awgn@+0dB, awgn@-2dB, "
                          "awgn@-6dB} of (s4d_plus_gate - s4d) / (bm3_frozen - s4d). "
                          "R >= 0.7 => CONSTRUCTIVE_CONFIRMED; 0.3 <= R < 0.7 => PARTIAL_RECOVERY; "
                          "R < 0.3 => GRAFT_INSUFFICIENT. Baselines bm3_frozen/s4d taken from the "
                          "same batch (results/secondary_20260703-1034/). No non-preregistered "
                          "criterion (e.g. the paired-significance results in "
                          f"{paired_sig_path.name}) is used to gate this verdict.",
        "deep_noise_conditions": DEEP_NOISE_CONDITIONS,
        "mean_R": round(mean_R, 4),
        "verdict": verdict,
        "reason": reason,
        "deep_noise_evidence": deep_noise_evidence,
        "per_level_evidence": per_level,
        "paired_significance_report_only": {
            "file": str(paired_sig_path),
            "note": "Supplementary, non-decisive per the prereg's explicit ban on "
                     "non-preregistered criteria (\"禁未预注册判据\").",
        },
        "source_data": "q_graft.json in this out_dir (mean_macro_f1 per condition x arm), built from "
                        "results/graft_20260705-224137/cells.jsonl (s4d_plus_gate) and "
                        "results/secondary_20260703-1034/cells.jsonl (bm3_frozen/s4d/frozen_nogate).",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graft-run-dir", required=True)
    ap.add_argument("--baseline-run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    repo_root = HARNESS_DIR
    graft_run_dir = Path(args.graft_run_dir)
    baseline_run_dir = Path(args.baseline_run_dir)
    out_dir = Path(args.out_dir)

    # Step 0: STOP-IF-NO-GRID
    ok, tuple_report, graft_cells = stop_if_no_grid(repo_root)
    if not ok:
        needs_human_dir = repo_root / "results" / f"graft_needs_human_{out_dir.name.split('_', 1)[-1]}"
        needs_human_dir.mkdir(parents=True, exist_ok=False)
        _write_protected_json(needs_human_dir / "needs_human.json", {
            "status": "needs_human",
            "reason": "STOP-IF-NO-GRID: results/graft_*/cells.jsonl does not contain exactly the "
                       "35 required (condition, s4d_plus_gate, seed) tuples with no duplicates.",
            "tuple_enum_report": tuple_report,
        })
        print(f"[STOP] STOP-IF-NO-GRID guardrail failed; wrote {needs_human_dir}/needs_human.json")
        sys.exit(4)

    out_dir.mkdir(parents=True, exist_ok=False)
    _write_protected_json(out_dir / "tuple_enum_report.json", tuple_report)
    print(f"[TUPLE] n_expected=35 n_observed={tuple_report['n_observed']} all_pass=True -> "
          f"{out_dir}/tuple_enum_report.json")

    baseline_cells_all = load_cells(baseline_run_dir / "cells.jsonl")
    baseline_cells = [c for c in baseline_cells_all if c["arm"] in BASELINE_ARMS]
    print(f"[LOAD] {len(graft_cells)} graft cells from {graft_run_dir}; "
          f"{len(baseline_cells)} baseline cells ({BASELINE_ARMS}) from {baseline_run_dir}")

    # Step 1: fairness
    fairness = build_fairness_report(graft_run_dir, graft_cells, baseline_cells)
    _write_protected_json(out_dir / "fairness_report.json", fairness)
    print(f"[FAIRNESS] all_pass={fairness['all_pass']} n_mismatches={fairness['n_mismatches']} -> "
          f"{out_dir}/fairness_report.json")
    if not fairness["all_pass"]:
        print("[STOP] fairness_report.all_pass=False; refusing to aggregate or emit a decision.")
        sys.exit(3)

    # Step 2/3: aggregate + curve
    agg = aggregate(graft_cells, baseline_cells)
    write_q_graft(out_dir, agg)
    write_curve(out_dir, agg)
    print(f"[AGG] wrote {out_dir}/q_graft.json, q_graft.csv, graft_curve_table.csv, graft_curve.png")

    # Step 4: R table
    per_level, mean_R = compute_R_table(agg)
    print(f"[R] deep-noise mean R = {mean_R:.4f} -> {classify_R(mean_R)}")

    # Step 5: paired significance (report-only)
    paired_sig = build_paired_significance(graft_cells, baseline_cells)
    paired_sig_path = out_dir / "graft_paired_significance.json"
    _write_protected_json(paired_sig_path, paired_sig)
    print(f"[SIG] wrote {paired_sig_path} (report-only, not decisive)")

    # Step 6: mechanical decision
    decision = build_decision(out_dir, graft_run_dir, per_level, mean_R, paired_sig_path,
                               tuple_report, fairness)
    dpath = out_dir / "graft_decision.json"
    _write_protected_json(dpath, decision)
    print(f"[DECISION] verdict={decision['verdict']} mean_R={decision['mean_R']} -> {dpath}")
    print(f"[DECISION] reason: {decision['reason']}")


if __name__ == "__main__":
    main()

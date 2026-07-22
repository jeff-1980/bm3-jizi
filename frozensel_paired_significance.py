#!/home/jeffwork/论文8/venv/bin/python3
"""
frozensel_paired_significance.py — supplementary paired-significance analysis
for results/frozensel_20260702-1827/, extending frozensel_aggregate.py.

frozensel_aggregate.py's frozensel_decision.json only reports
delta_kin_minus_frozen_pp and delta_kin_minus_s4d_pp (mean-level, non-overlapping
mean+/-std band as its significance test, scoped to the kin-vs-frozen claim per
the prereg's decision_rule). This script additionally computes, per condition:
  - delta_frozen_minus_s4d (not previously reported anywhere)
  - a real paired-sample significance test (paired t-test + Wilcoxon signed-rank)
    for BOTH delta_kin_minus_frozen and delta_frozen_minus_s4d, matched by seed
    (same seed = same split/init/noise-instance across arms, per the fairness
    iron law), rather than the mean+/-std overlap heuristic.

Read-only over cells.jsonl; does not touch any existing results/** file.
Writes into a NEW timestamped directory (additive-only guardrail) rather than
into results/frozensel_20260702-1827/ (whose product files are already
written and chmod 444 by frozensel_aggregate.py).

Usage:
  python frozensel_paired_significance.py --run-dir results/frozensel_20260702-1827 \
      --out-dir results/frozensel_pairedsig_<TS>
"""
import argparse
import csv
import json
import os
import stat
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def _protect(path: Path):
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _write_protected_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    _protect(path)


def load_cells(run_dir: Path):
    cells = []
    for line in (run_dir / "cells.jsonl").read_text().splitlines():
        line = line.strip()
        if line:
            cells.append(json.loads(line))
    return cells


CONDITION_ORDER = ["clean", "awgn@+10dB", "awgn@+6dB", "awgn@+0dB",
                   "awgn@-2dB", "awgn@-6dB", "awgn@-10dB"]


def paired_seed_f1(cells, condition, arm):
    """seed -> best_macro_f1 for one (condition, arm), asserting one row per seed."""
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
    except ValueError as e:
        # all-zero differences (identical arrays) -> wilcoxon undefined
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    cells = load_cells(run_dir)
    conds_present = sorted(set(c["condition"] for c in cells))
    assert set(conds_present) == set(CONDITION_ORDER), (conds_present, CONDITION_ORDER)

    per_condition = {}
    for cond in CONDITION_ORDER:
        kin = paired_seed_f1(cells, cond, "bm3_kin")
        frz = paired_seed_f1(cells, cond, "bm3_frozen")
        s4d = paired_seed_f1(cells, cond, "s4d")
        assert len(kin) == 5 and len(frz) == 5 and len(s4d) == 5, (cond, len(kin), len(frz), len(s4d))

        per_condition[cond] = {
            "delta_kin_minus_frozen": paired_test(kin, frz),
            "delta_frozen_minus_s4d": paired_test(frz, s4d),
        }

    n_sig_kin_frozen = sum(1 for v in per_condition.values() if v["delta_kin_minus_frozen"]["significant_p05"])
    n_sig_frozen_s4d = sum(1 for v in per_condition.values() if v["delta_frozen_minus_s4d"]["significant_p05"])

    report = {
        "source_run_dir": str(run_dir),
        "source_cells_file": str(run_dir / "cells.jsonl"),
        "method": "paired (same-seed) comparison across the 5 matched seeds per condition; "
                  "paired t-test (scipy.stats.ttest_rel) and Wilcoxon signed-rank "
                  "(scipy.stats.wilcoxon) as a non-parametric cross-check. "
                  "Supplements (does not replace) frozensel_decision.json's prereg-mandated "
                  "non-overlapping mean+/-std significance rule, which is scoped only to "
                  "kin-vs-frozen and does not report frozen-vs-s4d.",
        "alpha": 0.05,
        "conditions": CONDITION_ORDER,
        "n_conditions_kin_vs_frozen_significant_p05": n_sig_kin_frozen,
        "n_conditions_frozen_vs_s4d_significant_p05": n_sig_frozen_s4d,
        "per_condition": per_condition,
    }
    _write_protected_json(out_dir / "frozensel_paired_significance.json", report)

    csv_path = out_dir / "frozensel_paired_significance.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "comparison", "mean_diff_pp", "std_diff_pp",
                    "paired_ttest_stat", "paired_ttest_p", "wilcoxon_stat", "wilcoxon_p",
                    "significant_p05"])
        for cond in CONDITION_ORDER:
            for comp_key, comp_label in [("delta_kin_minus_frozen", "kin_minus_frozen"),
                                          ("delta_frozen_minus_s4d", "frozen_minus_s4d")]:
                e = per_condition[cond][comp_key]
                w.writerow([
                    cond, comp_label, e["mean_diff_pp"], e["std_diff_pp"],
                    e["paired_ttest"]["statistic"], e["paired_ttest"]["p_value"],
                    e["wilcoxon_signed_rank"]["statistic"], e["wilcoxon_signed_rank"]["p_value"],
                    e["significant_p05"],
                ])
    _protect(csv_path)

    print(f"[OK] wrote {out_dir}/frozensel_paired_significance.json and .csv")
    print(f"[SUMMARY] kin-vs-frozen significant (p<0.05) in {n_sig_kin_frozen}/7 conditions; "
          f"frozen-vs-s4d significant (p<0.05) in {n_sig_frozen_s4d}/7 conditions")


if __name__ == "__main__":
    main()

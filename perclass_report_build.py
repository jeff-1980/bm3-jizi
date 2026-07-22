#!/usr/bin/env python3
"""
perclass_report_build.py — builds the per-class fairness/reconciliation/
confusion-matrix/summary report for the results/perclass_2* grid.

Read-only against results/perclass_20260707-2121/ (never writes there).
All outputs go under a new timestamped results/perclass_analysis_<ts>/
directory (created by the caller before this script runs), each artifact
chmod 444 immediately after being written.

Does NOT launch any training. Does NOT modify xjtu_noisy_harness.py or any
other existing implementation file.
"""
import csv
import json
import os
import stat
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT   = Path(__file__).resolve().parent
GRID_DIR    = REPO_ROOT / "results" / "perclass_20260707-2121"
OUT_DIR     = REPO_ROOT / "results" / "perclass_analysis_20260708-1316"
CM_DIR      = OUT_DIR / "confusion_matrices"

EXPECTED_EVAL_SHA = "6c20b367522ce5db59720120f8a7cd12792c7e070c6ae84bd1985511998727de"
N_CLASSES   = 2
CLASS_NAMES = {0: "OR", 1: "IR"}  # from bearmamba3/data_xjtu.py LABEL_MAP = {'OR': 0, 'IR': 1}

ARMS       = ["bm3_frozen", "bm3_kin", "frozen_nogate", "s4d", "s4d_plus_gate"]
CONDITIONS = ["clean", "awgn@+0dB", "awgn@-6dB"]
SEEDS      = [0, 1, 2, 3, 4]

COND_SLUG = {"clean": "clean", "awgn@+0dB": "awgn_p0dB", "awgn@-6dB": "awgn_m6dB"}

HIST_SOURCES = {
    "frozensel": REPO_ROOT / "results" / "frozensel_20260702-1827" / "q_frozensel.json",
    "secondary": REPO_ROOT / "results" / "secondary_analysis_20260704-0708" / "q_secondary.json",
    "graft":     REPO_ROOT / "results" / "graft_analysis_20260706-0811" / "q_graft.json",
}


def protect(path: Path):
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    protect(path)


def write_text(path: Path, text: str):
    path.write_text(text)
    protect(path)


# ── 0. STOP-IF-NO-GRID: tuple enumeration hard gate ────────────────────────

def load_cells():
    path = GRID_DIR / "cells.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return rows


def tuple_enum_report(rows):
    seen = defaultdict(int)
    for r in rows:
        seen[(r["arm"], r["condition"], r["seed"])] += 1
    expected = {(a, c, s) for a in ARMS for c in CONDITIONS for s in SEEDS}
    observed = set(seen.keys())
    duplicates = sorted([list(k) for k, v in seen.items() if v > 1])
    missing = sorted([list(k) for k in expected - observed])
    extra = sorted([list(k) for k in observed - expected])

    missing_pred = []
    for r in rows:
        pf = GRID_DIR / r["predictions_file"]
        if not pf.is_file():
            missing_pred.append([r["arm"], r["condition"], r["seed"], str(pf)])

    n_expected = len(expected)
    n_observed = len(rows)
    all_pass = (n_observed == n_expected and not duplicates and not missing
                and not extra and not missing_pred)

    report = {
        "candidate_files": [str(GRID_DIR / "cells.jsonl")],
        "n_rows": n_observed,
        "n_expected": n_expected,
        "arms": ARMS,
        "conditions": CONDITIONS,
        "seeds": SEEDS,
        "n_duplicates": len(duplicates),
        "duplicate_tuples": duplicates,
        "n_missing": len(missing),
        "missing_tuples": missing,
        "n_extra": len(extra),
        "extra_tuples": extra,
        "n_missing_predictions_jsonl": len(missing_pred),
        "missing_predictions_jsonl": missing_pred,
        "all_pass": all_pass,
        "rule": ("STOP-IF-NO-GRID: results/perclass_2*/cells.jsonl must exactly "
                 "enumerate the 5 arms x 3 conditions x 5 seeds = 75 tuple grid "
                 "(no missing/duplicate/extra), and every row's predictions_file "
                 "must exist on disk. Any failure is a hard stop: no training is "
                 "launched by this script under any outcome."),
    }
    return report, all_pass


# ── 1. fairness_report.json — 75-row byte-exact eval_sha check ─────────────

def fairness_report(rows):
    all_rows = []
    n_mismatch = 0
    for r in rows:
        sha = r.get("fairness", {}).get("eval_sha256")
        match = (sha == EXPECTED_EVAL_SHA)
        if not match:
            n_mismatch += 1
        all_rows.append({
            "arm": r["arm"], "condition": r["condition"], "seed": r["seed"],
            "eval_sha256": sha, "len": len(sha) if sha else 0, "match": match,
        })
    report = {
        "run_dir": str(GRID_DIR.relative_to(REPO_ROOT)),
        "expected_eval_sha256_full": EXPECTED_EVAL_SHA,
        "expected_eval_sha256_len": len(EXPECTED_EVAL_SHA),
        "n_cells_checked": len(rows),
        "n_rows_reported": len(all_rows),
        "hash_lengths_all_64": all(x["len"] == 64 for x in all_rows),
        "n_mismatches": n_mismatch,
        "mismatches": [x for x in all_rows if not x["match"]],
        "all_pass": (n_mismatch == 0 and len(rows) == 75),
        "all_75_cells_full_hash": all_rows,
        "rule": ("byte-for-byte equality of every one of the 75 grid rows' "
                 "fairness.eval_sha256 against the task-specified literal "
                 "6c20b367522c... (SHA256 of the cross-condition test-split "
                 "labels, identical across arms/conditions/seeds by "
                 "construction — see xjtu_noisy_harness.py eval_fingerprint)."),
    }
    return report


# ── 2. same-source reconciliation: recompute macro-F1 from predictions.jsonl ─

def eval_macro_f1_from_predictions(y_true: np.ndarray, y_pred: np.ndarray):
    """Exact re-implementation of xjtu_noisy_harness.eval_macro_f1's tp/fn/fp
    -> per_f1 -> macro_f1 arithmetic (same operation order, same dtypes),
    applied to already-dumped predictions instead of a live loader pass."""
    tp = np.zeros(N_CLASSES, dtype=np.int64)
    fn = np.zeros(N_CLASSES, dtype=np.int64)
    fp = np.zeros(N_CLASSES, dtype=np.int64)
    for c in range(N_CLASSES):
        tp[c] = int(((y_pred == c) & (y_true == c)).sum())
        fn[c] = int(((y_pred != c) & (y_true == c)).sum())
        fp[c] = int(((y_pred == c) & (y_true != c)).sum())
    per_f1 = []
    for c in range(N_CLASSES):
        prec = tp[c] / max(tp[c] + fp[c], 1)
        rec  = tp[c] / max(tp[c] + fn[c], 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-8)
        per_f1.append(float(f1))
    macro_f1 = float(np.mean(per_f1))
    return tp, fn, fp, per_f1, macro_f1


def load_predictions(pred_file: Path):
    y_true, y_pred = [], []
    for line in pred_file.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        y_true.append(d["y_true"])
        y_pred.append(d["y_pred"])
    return np.array(y_true, dtype=np.int64), np.array(y_pred, dtype=np.int64)


def reconciliation_report(rows):
    per_cell = []
    n_bitexact = 0
    cell_cache = {}  # (arm,cond,seed) -> (tp,fn,fp,per_f1,macro_f1,y_true,y_pred)
    for r in rows:
        pf = GRID_DIR / r["predictions_file"]
        y_true, y_pred = load_predictions(pf)
        tp, fn, fp, per_f1, macro_f1 = eval_macro_f1_from_predictions(y_true, y_pred)
        reported = r["best_macro_f1"]
        bitexact = (macro_f1 == reported)
        if bitexact:
            n_bitexact += 1
        key = (r["arm"], r["condition"], r["seed"])
        cell_cache[key] = {
            "tp": tp, "fn": fn, "fp": fp, "per_f1": per_f1, "macro_f1": macro_f1,
            "y_true": y_true, "y_pred": y_pred, "n_samples": len(y_true),
        }
        per_cell.append({
            "arm": r["arm"], "condition": r["condition"], "seed": r["seed"],
            "reported_best_macro_f1": reported,
            "recomputed_macro_f1_from_predictions_jsonl": macro_f1,
            "abs_diff": abs(macro_f1 - reported),
            "bitexact_match": bitexact,
            "n_samples_in_predictions_jsonl": len(y_true),
        })
    report = {
        "run_dir": str(GRID_DIR.relative_to(REPO_ROOT)),
        "method": ("For each of the 75 cells, recompute tp/fn/fp per class from "
                   "predictions.jsonl's (y_true, y_pred) columns using the exact "
                   "same tp/fn/fp -> per-class-F1 -> macro-F1 formula as "
                   "xjtu_noisy_harness.eval_macro_f1 (same source, same dumped "
                   "epoch's predictions — predictions.jsonl is written only at "
                   "the best epoch, see harness train_cell/_write_predictions_jsonl), "
                   "then compare to cells.jsonl's self-reported best_macro_f1 for "
                   "bit-for-bit (Python float ==) equality."),
        "n_cells": len(rows),
        "n_bitexact_match": n_bitexact,
        "n_mismatch": len(rows) - n_bitexact,
        "all_pass_75_of_75": (n_bitexact == 75 and len(rows) == 75),
        "per_cell": per_cell,
    }
    return report, cell_cache


# ── 3. confusion matrices: 5 arms x 3 conditions, summed across seeds ──────

def build_confusion_matrices(rows, cell_cache):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrices = {}  # (arm,cond) -> 2x2 int64 array [true, pred]
    for r in rows:
        key = (r["arm"], r["condition"], r["seed"])
        c = cell_cache[key]
        y_true, y_pred = c["y_true"], c["y_pred"]
        cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
        for t in range(N_CLASSES):
            for p in range(N_CLASSES):
                cm[t, p] += int(((y_true == t) & (y_pred == p)).sum())
        mk = (r["arm"], r["condition"])
        matrices[mk] = matrices.get(mk, np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)) + cm

    csv_rows = []
    index = []
    for arm in ARMS:
        for cond in CONDITIONS:
            cm = matrices[(arm, cond)]
            support = cm.sum(axis=1)
            slug = f"{arm}__{COND_SLUG[cond]}"
            csv_path = CM_DIR / f"{slug}.csv"
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["arm", "condition", "true_class", "pred_OR", "pred_IR", "row_sum_support"])
                for t in range(N_CLASSES):
                    w.writerow([arm, cond, CLASS_NAMES[t], int(cm[t, 0]), int(cm[t, 1]), int(support[t])])
            protect(csv_path)

            fig, ax = plt.subplots(figsize=(4, 3.6))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1]); ax.set_xticklabels(["OR", "IR"])
            ax.set_yticks([0, 1]); ax.set_yticklabels(["OR", "IR"])
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            ax.set_title(f"{arm} | {cond}\n(summed over {len(SEEDS)} seeds)", fontsize=9)
            vmax = cm.max()
            for t in range(N_CLASSES):
                for p in range(N_CLASSES):
                    val = cm[t, p]
                    color = "white" if val > vmax * 0.5 else "black"
                    ax.text(p, t, str(val), ha="center", va="center", color=color, fontsize=11)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            png_path = CM_DIR / f"{slug}.png"
            fig.savefig(png_path, dpi=150)
            plt.close(fig)
            protect(png_path)

            index.append({
                "arm": arm, "condition": cond,
                "csv": str(csv_path.relative_to(REPO_ROOT)),
                "png": str(png_path.relative_to(REPO_ROOT)),
                "support_OR": int(support[0]), "support_IR": int(support[1]),
                "total_support": int(support.sum()),
            })
            csv_rows.append(index[-1])

    write_json(CM_DIR / "confusion_matrices_index.json", {
        "n_matrices": len(index), "n_seeds_summed": len(SEEDS),
        "class_names": CLASS_NAMES, "matrices": index,
    })
    return matrices


# ── 4. per-class precision/recall/F1 (+support), per arm per condition ─────

def per_class_report(matrices):
    rows_out = []
    for arm in ARMS:
        for cond in CONDITIONS:
            cm = matrices[(arm, cond)]
            for c in range(N_CLASSES):
                tp = int(cm[c, c])
                fn = int(cm[c, :].sum() - tp)
                fp = int(cm[:, c].sum() - tp)
                support = tp + fn
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-8)
                rows_out.append({
                    "arm": arm, "condition": cond, "class": CLASS_NAMES[c],
                    "precision": prec, "recall": rec, "f1": f1, "support": support,
                    "tp": tp, "fp": fp, "fn": fn,
                })
    return rows_out


def write_per_class(rows_out):
    json_path = OUT_DIR / "per_class_report.json"
    write_json(json_path, {
        "n_seeds_summed": len(SEEDS),
        "class_names": CLASS_NAMES,
        "method": ("Precision/recall/F1/support per (arm, condition, class), "
                   "computed from the same seed-summed 2x2 confusion matrices "
                   "as confusion_matrices/ (§3) — support = row sum = TP+FN for "
                   "that class, summed over 5 seeds."),
        "rows": rows_out,
    })

    csv_path = OUT_DIR / "per_class_report.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "condition", "class", "precision", "recall", "f1", "support", "tp", "fp", "fn"])
        for r in rows_out:
            w.writerow([r["arm"], r["condition"], r["class"],
                        f"{r['precision']:.6f}", f"{r['recall']:.6f}", f"{r['f1']:.6f}",
                        r["support"], r["tp"], r["fp"], r["fn"]])
    protect(csv_path)
    return json_path, csv_path


# ── 2b. historical cross-check (sanity only, NOT a gate) ───────────────────

def historical_crosscheck(rows):
    our = defaultdict(list)  # (arm,cond) -> [macro_f1 per seed]
    for r in rows:
        our[(r["arm"], r["condition"])].append(r["best_macro_f1"])

    hist = {}
    for name, path in HIST_SOURCES.items():
        if path.is_file():
            hist[name] = json.loads(path.read_text())

    comparisons = []
    for arm in ARMS:
        for cond in CONDITIONS:
            vals = our[(arm, cond)]
            our_mean = float(np.mean(vals))
            our_std = float(np.std(vals))
            matches = []
            for name, d in hist.items():
                c = d.get("conditions", {}).get(cond, {})
                if arm in c:
                    hmean = c[arm]["mean_macro_f1"]
                    hstd = c[arm]["std_macro_f1"]
                    diff = our_mean - hmean
                    within_1std = (hstd > 0 and abs(diff) <= hstd) or (hstd == 0 and diff == 0)
                    matches.append({
                        "source": name,
                        "source_path": str(HIST_SOURCES[name].relative_to(REPO_ROOT)),
                        "historical_mean_macro_f1": hmean, "historical_std_macro_f1": hstd,
                        "diff_our_minus_historical": diff,
                        "diff_in_historical_std_units": (diff / hstd) if hstd > 0 else None,
                        "within_1_historical_std": within_1std,
                    })
            comparisons.append({
                "arm": arm, "condition": cond,
                "our_mean_macro_f1": our_mean, "our_std_macro_f1": our_std,
                "our_n_seeds": len(vals),
                "historical_matches": matches,
            })
    report = {
        "status": "SANITY_ONLY_NOT_A_GATE",
        "rule": ("Per task spec item 2: compare each arm x condition mean against "
                 "the historical frozensel/secondary/graft grids for the same "
                 "arm x same condition. Deviation within +/-1 historical std is "
                 "reported for information only; it is NOT a pass/fail gate here. "
                 "Cross-process noise floor of ~0.7pp between independent runs is "
                 "a known baseline per REPORT_preddump_equivalence_20260707-1933.md "
                 "adjudication 1."),
        "historical_sources": {k: str(v.relative_to(REPO_ROOT)) for k, v in HIST_SOURCES.items() if v.is_file()},
        "comparisons": comparisons,
    }
    return report


# ── main ─────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CM_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_cells()

    te_report, gate_ok = tuple_enum_report(rows)
    write_json(OUT_DIR / "tuple_enum_report.json", te_report)
    print(f"[GATE] tuple_enum all_pass={gate_ok}  n_rows={te_report['n_rows']}")
    if not gate_ok:
        needs_human = {
            "status": "needs_human",
            "reason": "STOP-IF-NO-GRID hard gate failed — see tuple_enum_report.json",
        }
        write_json(OUT_DIR / "NEEDS_HUMAN.json", needs_human)
        print("[STOP] grid incomplete; wrote NEEDS_HUMAN.json; aborting (no training launched).")
        sys.exit(1)

    fr = fairness_report(rows)
    write_json(OUT_DIR / "fairness_report.json", fr)
    print(f"[FAIRNESS] all_pass={fr['all_pass']}  n_rows={fr['n_rows_reported']}  n_mismatches={fr['n_mismatches']}")
    if not fr["all_pass"]:
        print("[STOP] fairness_report mismatches found; aborting further analysis.")
        sys.exit(1)

    recon_report, cell_cache = reconciliation_report(rows)
    write_json(OUT_DIR / "reconciliation_report.json", recon_report)
    print(f"[RECONCILE] bitexact={recon_report['n_bitexact_match']}/{recon_report['n_cells']}")

    hist_report = historical_crosscheck(rows)
    write_json(OUT_DIR / "historical_crosscheck.json", hist_report)
    print("[HIST] sanity cross-check written (non-gating)")

    matrices = build_confusion_matrices(rows, cell_cache)
    print(f"[CM] built {len(matrices)} confusion matrices -> {CM_DIR}")

    pc_rows = per_class_report(matrices)
    pc_json, pc_csv = write_per_class(pc_rows)
    print(f"[PERCLASS] wrote {pc_json.name}, {pc_csv.name} ({len(pc_rows)} rows)")

    print("[DONE] core artifacts written under", OUT_DIR)
    print("[NOTE] perclass_summary.md is authored separately from this "
          "script's json/csv outputs (per_class_report.json, "
          "historical_crosscheck.json) and chmod 444'd by hand.")


if __name__ == "__main__":
    main()

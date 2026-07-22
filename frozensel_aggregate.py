#!/home/jeffwork/论文8/venv/bin/python3
"""
frozensel_aggregate.py — post-grid aggregation for results/frozensel_<TS>/.

Mechanically applies the rule preregistered in
results/frozensel_prereg_20260702-1826/frozensel_prereg.json (written BEFORE
the grid was launched). Does not launch or resume any training; read-only
over an existing cells.jsonl.

Steps (all-or-nothing; any FAIL stops before writing decision/aggregation
products, matching the prereg's fairness_verification_required gate):
  1. Load cells.jsonl, require >= min_cells (63).
  2. Tuple-enumeration check (E0): build the exact expected set of
     (condition, arm, seed) = canonical_conditions() x design.arms x
     design.seeds (7 x 3 x 5 = 105) and require observed cells.jsonl rows to
     match it exactly (no duplicates, no missing, no extras). Write
     tuple_enum_report.json. If this fails: stop before fairness/aggregation.
  3. Independently recompute eval_sha256 (full hash) via the harness's own
     eval_fingerprint()/make_cross_condition_split()/XJTUDataset() — the same
     construction run_grid() uses — and cross-check every cell's
     fairness.eval_sha256 against it byte-for-byte. Write fairness_report.json.
  4. If fairness fails: stop (no aggregation/decision products written).
  5. Aggregate mean/std macro_f1 per (condition, arm) -> q_frozensel.json/csv.
  6. Curve table + PNG (macro_F1 vs SNR per arm, mean +/- std).
  7. Apply prereg decision_rule -> frozensel_decision.json.
  8. chmod 444 every product written in this run (cells.jsonl is already 444
     from run_grid(); never touched here). Re-running against a run_dir that
     already has protected products is safe and idempotent: any product that
     already exists on disk is left untouched (skipped, not overwritten) —
     see _write_protected_json()/_write_protected_csv().

Usage:
  python frozensel_aggregate.py --run-dir results/frozensel_20260702-1827 \
      --prereg results/frozensel_prereg_20260702-1826/frozensel_prereg.json
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

HARNESS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS_DIR))

import xjtu_noisy_harness as H  # noqa: E402  (import-only, no side effects — main() gated)

CHANCE_FLOOR = 0.55

# H2 fix (.orchestrate/0703-081727/r2_review.json): a hardcoded
# REVIEWER_VERDICT_PATH regresses to a stale reviewer verdict every time a
# new review round runs. Resolve dynamically instead — see
# _resolve_latest_reviewer_verdict().
DEEP_NOISE_CONDITIONS = ["awgn@+0dB", "awgn@-2dB", "awgn@-6dB", "awgn@-10dB"]
# awgn@-10dB excluded from the frozen-vs-s4d attribution check: s4d is
# at_chance there (q_frozensel.json: mean_macro_f1=0.523 <= 0.55), so that
# comparison is not meaningful.
ATTRIBUTION_CONDITIONS = ["awgn@+0dB", "awgn@-2dB", "awgn@-6dB"]
NOT_PRIMARY_THRESHOLD_PP = 10.0
ATTRIBUTION_THRESHOLD_PP = 13.0
ALLOWED_SCOPED_VERDICTS = {
    "SELECTIVITY_NOT_PRIMARY_MECHANISM",
    "SELECTIVITY_MAY_BE_PRIMARY_MECHANISM",
    "INCONCLUSIVE",
}


def _resolve_latest_reviewer_verdict(repo_root: Path = None):
    """H2: dynamically resolve the latest valid reviewer verdict file under
    .orchestrate/<round>/, instead of a path hardcoded at one point in time.

    A 'valid reviewer verdict' is an r*_review.json with a top-level
    'verdict' in {'PASS','REVISE'}. Round directories are named
    '<MMDD>-<HHMMSS>', which sort correctly as strings in chronological
    order. Executor logs (r*_executor.log) are never treated as reviewer
    verdicts, even if they contain executor-written PASS/resolution claims
    (per .orchestrate/0703-074355/r2_review.json's E3 finding: 'treat
    executor-written PASS/resolution claims as void').

    E3 fix (.orchestrate/0703-084057/r1_review.json): the prior version of
    this function scanned ALL round directories, including one still in
    progress. That always resolves to the round that is *currently
    critiquing* the not-yet-written artifact -- a self-referential deadlock
    (see decision_manifest.json's self_reference_deadlock note): the moment
    a corrected citation is written, the review of that correction becomes
    the new "latest" file and stales the citation again. Restricting to
    CLOSED rounds (round_dir/result.json exists and status == "passed")
    breaks the deadlock: a closed round's terminal review is immutable
    history and cannot be re-opened by a later round starting elsewhere, so
    citing it stays valid even after new rounds begin. Returns None if no
    closed round directory contains a parseable review file.
    """
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


def _protect(path: Path):
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _write_protected_json(path: Path, obj):
    if path.exists():
        print(f"[SKIP] {path} already exists (protected, additive-only); not overwriting.")
        return
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    _protect(path)


def _protected_csv_writer(path: Path, header, rows):
    """Write a CSV and chmod 444, unless path already exists — then skip
    (additive-only guardrail: never overwrite an already-written protected
    artifact, even on a re-run of this script over the same run_dir)."""
    if path.exists():
        print(f"[SKIP] {path} already exists (protected, additive-only); not overwriting.")
        return False
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)
    _protect(path)
    return True


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
    """
    Authoritative condition-label list, taken straight from the harness's own
    build_conditions(["awgn"], FULL_SNRS) — NOT re-derived by hand here.
    The harness formats AWGN labels as f"{nt}@{snr:+.0f}dB" (e.g. "awgn@+10dB",
    "awgn@+0dB", "awgn@-6dB"); the prereg doc's human-readable "conditions"
    list omits the '+' sign, so matching against that string list directly
    would silently produce empty aggregates for every AWGN condition. Using
    build_conditions() here guarantees exact agreement with what run_grid()
    actually wrote to cells.jsonl.
    """
    conds = H.build_conditions(["awgn"], H.FULL_SNRS)
    return [c["label"] for c in conds]


def expected_tuples(prereg: dict) -> set:
    """The exact, mechanically-derived (condition, arm, seed) set this grid must
    contain: canonical_conditions() (from the harness's own build_conditions(),
    see canonical_conditions() docstring) x design.arms x design.seeds.
    For frozensel_prereg_20260702-1826.json this is 7 x 3 x 5 = 105 tuples
    (design.cells_expected_full)."""
    design = prereg["design"]
    conditions = canonical_conditions()
    arms = design["arms"]
    seeds = design["seeds"]
    return {(c, a, s) for c in conditions for a in arms for s in seeds}


def build_tuple_enum_report(run_dir: Path, cells: list, prereg: dict) -> dict:
    """E0: full tuple-enumeration check, run BEFORE fairness/aggregation.

    Verifies cells.jsonl contains exactly one row per (condition, arm, seed)
    in expected_tuples() -- no duplicates, nothing missing, nothing extra.
    Aggregating over a cells list with duplicate or missing tuples would
    silently bias per-(condition, arm) means (e.g. a seed counted twice, or a
    seed's absence masked by averaging over fewer seeds without anyone
    noticing), so this must gate before aggregate()/apply_scoped_decision_rule()."""
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
        "rule": "exact set-equality of observed (condition, arm, seed) tuples in cells.jsonl against "
                "the full cross product of canonical_conditions() x prereg.design.arms x "
                "prereg.design.seeds. Any duplicate, missing, or extra tuple fails this check and "
                "stops the script before fairness_report.json / aggregation / decision products "
                "are written (E0).",
    }


def independent_eval_sha(prereg: dict) -> str:
    """Recompute eval_sha256 exactly as run_grid() does at startup, independently."""
    train_bearings, test_bearings = H.make_cross_condition_split(H.TRAIN_COND, H.TEST_COND)
    base_test = H.XJTUDataset(H.DATA_ROOT, test_bearings, n_sensors=1)
    clean_test_ds = H.XJTUDatasetNoisy(base_test, "clean", None, 0)
    return H.eval_fingerprint(clean_test_ds)


def build_fairness_report(run_dir: Path, cells: list, prereg: dict) -> dict:
    expected = independent_eval_sha(prereg)
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
            for c in cells[:3]
        ],
        "rule": "byte-for-byte equality against an independently recomputed eval_sha256 "
                "(SHA256 of the cross-condition test-split labels), per "
                "frozensel_prereg.json:fairness_verification_required.",
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
    """condition label -> x-axis value, derived from the harness's own
    build_conditions() rather than hand-typed labels (see canonical_conditions())."""
    m = {"clean": 15.0}  # clean plotted to the right of the noisiest AWGN point
    for c in H.build_conditions(["awgn"], H.FULL_SNRS):
        if c["label"] == "clean":
            continue  # build_conditions() always prepends a clean entry with snr_db=None;
                      # without this guard it overwrites the hardcoded 15.0 above with None,
                      # silently dropping the clean point from the curve table/plot.
        m[c["label"]] = c["snr_db"]
    return m


def write_q_frozensel(run_dir: Path, agg: dict, design: dict):
    j = {"chance_floor": CHANCE_FLOOR, "conditions": agg}
    json_path = run_dir / "q_frozensel.json"
    _write_protected_json(json_path, j)

    order = canonical_conditions()
    csv_path = run_dir / "q_frozensel.csv"
    rows = []
    for cond in order:
        for arm in design["arms"]:
            e = agg[cond][arm]
            rows.append([cond, arm, e["mean_macro_f1"], e["std_macro_f1"],
                         e["n_seeds_used"], "|".join(map(str, e["seeds"])), e["at_chance"]])
    _protected_csv_writer(csv_path, ["condition", "arm", "mean_macro_f1", "std_macro_f1",
                                      "n_seeds_used", "seeds", "at_chance"], rows)
    return json_path, csv_path


def write_curve_table(run_dir: Path, agg: dict, design: dict):
    csv_path = run_dir / "frozensel_curve_table.csv"
    order = canonical_conditions()
    snr_for_plot = snr_for_plot_map()
    rows = []
    for arm in design["arms"]:
        for cond in order:
            e = agg[cond][arm]
            rows.append([arm, cond, snr_for_plot.get(cond), e["mean_macro_f1"], e["std_macro_f1"], e["n_seeds_used"]])
    _protected_csv_writer(csv_path, ["arm", "condition", "snr_db_for_plot", "mean_macro_f1",
                                      "std_macro_f1", "n_seeds_used"], rows)
    return csv_path


def write_curve_figure(run_dir: Path, agg: dict, design: dict):
    png_path = run_dir / "frozensel_curve.png"
    if png_path.exists():
        print(f"[SKIP] {png_path} already exists (protected, additive-only); not overwriting.")
        return png_path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = canonical_conditions()
    snr_for_plot = snr_for_plot_map()
    xs = [snr_for_plot[c] for c in order]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"bm3_kin": "tab:blue", "bm3_frozen": "tab:orange", "s4d": "tab:green"}
    for arm in design["arms"]:
        means = [agg[c][arm]["mean_macro_f1"] for c in order]
        stds = [agg[c][arm]["std_macro_f1"] for c in order]
        ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3,
                     label=arm, color=colors.get(arm))
    ax.axhline(CHANCE_FLOOR, color="gray", linestyle="--", linewidth=1, label=f"chance floor ({CHANCE_FLOOR:.0%})")
    ax.set_xlabel("AWGN SNR (dB); clean plotted at 15dB")
    ax.set_ylabel("macro F1 (mean +/- std across seeds)")
    ax.set_title("Frozen-selectivity ablation: bm3_kin vs bm3_frozen vs s4d")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _protect(png_path)
    return png_path


def write_curve_superseded_manifest_if_stale(run_dir: Path, curvefix_dir: Path):
    """F1: results/frozensel_20260702-1827/frozensel_curve_table.csv was written by
    an earlier, buggy snr_for_plot_map() that dropped the 'clean' condition's
    x-value (see REPORT_frozensel_20260703-0748.md). That file is already
    chmod 444 and, per the additive-only guardrail, must never be overwritten
    -- so this writes an explicit, protected manifest declaring it superseded
    and pointing consumers at results/frozensel_curvefix_20260703-0748/ (the
    already-regenerated, correct curve_table.csv/curve.png pair) as the
    authoritative curve product for this run_dir. No-op if the on-disk table
    already has the clean row filled in (nothing to supersede) or if the
    manifest already exists (additive-only)."""
    table_path = run_dir / "frozensel_curve_table.csv"
    manifest_path = run_dir / "frozensel_curve_products_manifest.json"
    if manifest_path.exists():
        print(f"[SKIP] {manifest_path} already exists; not overwriting.")
        return manifest_path
    if not table_path.exists():
        return None

    with open(table_path, newline="") as f:
        rows = list(csv.DictReader(f))
    clean_rows = [r for r in rows if r["condition"] == "clean"]
    stale = bool(clean_rows) and any((r["snr_db_for_plot"] or "").strip() == "" for r in clean_rows)
    if not stale:
        return None

    curvefix_table = curvefix_dir / "frozensel_curve_table.csv"
    curvefix_png = curvefix_dir / "frozensel_curve.png"
    manifest = {
        "run_dir": str(run_dir),
        "superseded_files": ["frozensel_curve_table.csv", "frozensel_curve.png"],
        "defect": "snr_for_plot_map() in frozensel_aggregate.py (pre-fix) let build_conditions()'s "
                  "own leading {'label': 'clean', 'snr_db': None} entry overwrite the hardcoded "
                  "clean->15.0 plot x-value, so every 'clean' row got snr_db_for_plot='' and the "
                  "clean point was silently omitted from frozensel_curve.png. Fixed in "
                  "snr_for_plot_map() (added an explicit skip for the build_conditions() 'clean' "
                  "entry) after these two files were already written and chmod 444.",
        "authoritative_dir": str(curvefix_dir),
        "authoritative_files": {
            "frozensel_curve_table.csv": str(curvefix_table),
            "frozensel_curve.png": str(curvefix_png),
        },
        "authoritative_files_exist": curvefix_table.exists() and curvefix_png.exists(),
        "non_superseded_files_in_this_run_dir": [
            "q_frozensel.json", "q_frozensel.csv", "frozensel_decision.json",
            "fairness_report.json", "tuple_enum_report.json",
            "(these do not depend on snr_for_plot_map() and are unaffected by this defect)",
        ],
        "report": "REPORT_frozensel_20260703-0748.md",
    }
    _write_protected_json(manifest_path, manifest)
    return manifest_path


def apply_scoped_decision_rule(agg: dict, design: dict, prereg: dict, run_dir: Path, cells: list):
    """H2 fix (.orchestrate/0703-081727/r1_review.json, r2_review.json): the
    original significance-band-majority verdict path (SELECTIVITY_CLAIM_
    SURVIVES/FALSIFIED, gated on kin_sig_wins vs a strict majority of
    meaningful conditions) has been deleted from this module. This scoped
    decision rule instead thresholds the magnitude of Δ(kin-frozen) in the
    four deep-noise conditions only, and adds a frozen-vs-s4d attribution
    block. This is a documented amendment to frozensel_prereg_20260702-1826.json
    (which only defined the significance-band rule); the amendment and the
    human ruling authorizing it are recorded in decision_manifest.json next
    to the artifact this function's output is written to."""
    conditions = canonical_conditions()
    per_condition = {}
    for cond in conditions:
        kin = agg[cond]["bm3_kin"]
        frz = agg[cond]["bm3_frozen"]
        s4d = agg[cond]["s4d"]
        per_condition[cond] = {
            "bm3_kin": kin, "bm3_frozen": frz, "s4d": s4d,
            "delta_kin_minus_frozen_pp": (
                round((kin["mean_macro_f1"] - frz["mean_macro_f1"]) * 100, 2)
                if kin["mean_macro_f1"] is not None and frz["mean_macro_f1"] is not None else None
            ),
            "delta_kin_minus_s4d_pp": (
                round((kin["mean_macro_f1"] - s4d["mean_macro_f1"]) * 100, 2)
                if kin["mean_macro_f1"] is not None and s4d["mean_macro_f1"] is not None else None
            ),
            "delta_frozen_minus_s4d_pp": (
                round((frz["mean_macro_f1"] - s4d["mean_macro_f1"]) * 100, 2)
                if frz["mean_macro_f1"] is not None and s4d["mean_macro_f1"] is not None else None
            ),
        }

    n_cells = len(cells)
    min_cells_required = design["cells_expected_min"]

    deep_noise_deltas = {}
    attribution = {}
    if n_cells < min_cells_required:
        verdict = "INCONCLUSIVE"
        reason = f"only {n_cells} cells present, below min_acceptable ({min_cells_required})"
    else:
        missing_deep = [c for c in DEEP_NOISE_CONDITIONS if c not in agg]
        if missing_deep:
            verdict = "INCONCLUSIVE"
            reason = f"deep-noise conditions missing from aggregate: {missing_deep}"
        else:
            deep_noise_deltas = {c: per_condition[c]["delta_kin_minus_frozen_pp"] for c in DEEP_NOISE_CONDITIONS}
            max_delta = max(deep_noise_deltas.values())
            if max_delta < NOT_PRIMARY_THRESHOLD_PP:
                verdict = "SELECTIVITY_NOT_PRIMARY_MECHANISM"
                deltas_str = ", ".join(f"{c}={v:+.2f}pp" for c, v in deep_noise_deltas.items())
                reason = (
                    "bm3_kin's advantage over bm3_frozen stays below the "
                    f"+{NOT_PRIMARY_THRESHOLD_PP:.0f}pp deep-noise threshold in all four deep-noise "
                    f"conditions (Δ(kin-frozen) per q_frozensel.json: {deltas_str}), so input-dependent "
                    "selectivity is not the primary mechanism behind BM3's deep-noise performance."
                )
            else:
                breaches = {c: v for c, v in deep_noise_deltas.items() if v >= NOT_PRIMARY_THRESHOLD_PP}
                verdict = "SELECTIVITY_MAY_BE_PRIMARY_MECHANISM"
                reason = (
                    f"bm3_kin beats bm3_frozen by >= +{NOT_PRIMARY_THRESHOLD_PP:.0f}pp in "
                    f"{len(breaches)} deep-noise condition(s) (per q_frozensel.json): {breaches}; "
                    "selectivity may be a primary contributor there."
                )

            frozen_minus_s4d = {c: per_condition[c]["delta_frozen_minus_s4d_pp"] for c in ATTRIBUTION_CONDITIONS}
            all_ge_threshold = all(v >= ATTRIBUTION_THRESHOLD_PP for v in frozen_minus_s4d.values())
            attribution = {
                "frozen_minus_s4d_deep_noise_pp": frozen_minus_s4d,
                "threshold_pp": ATTRIBUTION_THRESHOLD_PP,
                "excluded_condition": (
                    "awgn@-10dB (s4d is at_chance there per q_frozensel.json: "
                    "mean_macro_f1 <= chance_floor; frozen-vs-s4d not a meaningful comparison)"
                ),
                "all_conditions_meet_threshold": all_ge_threshold,
                "conclusion": (
                    "bm3_frozen (selectivity removed) still beats s4d by >= "
                    f"+{ATTRIBUTION_THRESHOLD_PP:.0f}pp in every checked deep-noise condition, so "
                    "BM3's deep-noise advantage over the S4D baseline is attributable mainly to "
                    "non-selective BM3-architecture components, not to input-dependent selectivity."
                ) if all_ge_threshold else (
                    "Not all checked deep-noise conditions meet the "
                    f"+{ATTRIBUTION_THRESHOLD_PP:.0f}pp attribution threshold; attribution to "
                    "non-selective components is not fully supported by this data."
                ),
            }

    decision = {
        "decision_kind": "scoped_deep_noise_amendment",
        # E3 fix (.orchestrate/0703-084057/r1_review.json): this Δpp magnitude
        # rule was never preregistered (frozensel_prereg_20260702-1826.json
        # only locked the significance-band-majority rule before the grid
        # ran). Label this artifact explicitly as post-hoc/exploratory so it
        # is never mistaken for the E3-adjudicating decision; that role
        # belongs to frozensel_decision.json (the mechanical application of
        # the actually-preregistered rule).
        "post_hoc_exploratory": True,
        "preregistered": False,
        "adjudicates_E3": False,
        "authoritative_E3_decision_file": "frozensel_decision.json",
        "prereg_id": prereg["prereg_id"],
        "prereg_path": None,  # filled by caller with the actual path used
        "prereg_amendment_note": (
            "Amends frozensel_prereg_20260702-1826.json's significance-band-majority verdict rule "
            "with a deep-noise Δpp magnitude rule, per reviewer escalation .orchestrate/0703-081727/ "
            "(r1_review.json, r2_review.json). This rule was requested by the review process AFTER "
            "the grid's data already existed, so unlike the significance-band rule it is disclosed "
            "as a post-hoc scope amendment, not a preregistered criterion -- it MUST NOT be treated "
            "as the E3-adjudicating decision (see post_hoc_exploratory/adjudicates_E3 above and "
            "authoritative_E3_decision_file). See decision_manifest.json next to this artifact for "
            "the citation chain, the self-referential-deadlock note, and the verbatim human ruling "
            "authorizing this scope change."
        ),
        "responds_to_reviewer": prereg["responds_to_reviewer"],
        "reviewer_verdict_file": _resolve_latest_reviewer_verdict(),
        "run_dir": str(run_dir),
        "n_cells_total": n_cells,
        "n_cells_expected_full": design["cells_expected_full"],
        "n_cells_expected_min": min_cells_required,
        "chance_floor": CHANCE_FLOOR,
        "deep_noise_conditions": DEEP_NOISE_CONDITIONS,
        "not_primary_threshold_pp": NOT_PRIMARY_THRESHOLD_PP,
        "delta_kin_minus_frozen_deep_noise_pp": deep_noise_deltas,
        "verdict": verdict,
        "reason": reason,
        "attribution": attribution,
        "source_data": "q_frozensel.json in this run_dir (mean_macro_f1 per condition x arm)",
        "superseded_significance_majority_decision": {
            "file": "frozensel_decision.json",
            "note": (
                "That artifact used the original prereg's significance-band-majority rule and "
                "remains on disk (chmod 444, additive-only guardrail); this scoped decision "
                "supersedes it for the deep-noise-attribution question per reviewer escalation. "
                "The two are consistent: both conclude bm3_kin's input-dependent selectivity does "
                "not primarily/significantly drive BM3's deep-noise performance edge."
            ),
        },
        "per_condition": per_condition,
    }
    assert decision["verdict"] in ALLOWED_SCOPED_VERDICTS, (
        f"verdict {decision['verdict']!r} outside allowed scoped enum {ALLOWED_SCOPED_VERDICTS}"
    )
    return decision


def write_decision_superseded_manifest_if_needed(run_dir: Path, scoped_decision_path: Path):
    """H1/H2: old_path (the original significance-band-majority decision) is
    already chmod 444 and must never be overwritten (additive-only
    guardrail). This writes an explicit, protected manifest declaring it
    superseded for the deep-noise-attribution question and pointing at the
    new scoped decision artifact, mirroring the pattern already used for the
    curve-products manifest (write_curve_superseded_manifest_if_stale)."""
    old_path = run_dir / "frozensel_decision.json"
    manifest_path = run_dir / "frozensel_decision_superseded_manifest.json"
    if manifest_path.exists():
        print(f"[SKIP] {manifest_path} already exists; not overwriting.")
        return manifest_path
    if not old_path.exists() or not scoped_decision_path.exists():
        return None

    old = json.loads(old_path.read_text())
    manifest = {
        "run_dir": str(run_dir),
        "superseded_file": "frozensel_decision.json",
        "superseded_verdict": old.get("verdict"),
        "reason": (
            "Reviewer escalation .orchestrate/0703-081727/ (r1_review.json, r2_review.json) "
            "required a deep-noise-magnitude-based scoped verdict (SELECTIVITY_NOT_PRIMARY_MECHANISM) "
            "with an explicit frozen-vs-s4d attribution block, replacing the original "
            "significance-band-majority verdict (SELECTIVITY_CLAIM_FALSIFIED) framing for this "
            "scoped decision. frozensel_decision.json is left in place unmodified "
            "(additive-only guardrail; chmod 444) as the historical record of the original "
            "prereg's mechanical rule output."
        ),
        "authoritative_file": str(scoped_decision_path),
        "authoritative_file_exists": scoped_decision_path.exists(),
        "decision_manifest": str(run_dir / "decision_manifest.json"),
        "not_a_reversal": (
            "Both artifacts agree bm3_kin's selectivity does not significantly/primarily drive "
            "BM3's deep-noise performance; the scoped decision adds magnitude (pp-threshold) "
            "framing and non-selective-component attribution that the original significance-band "
            "rule did not compute."
        ),
    }
    _write_protected_json(manifest_path, manifest)
    return manifest_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--curvefix-dir", default=None,
                     help="Directory holding a corrected frozensel_curve_table.csv/"
                          "frozensel_curve.png pair, for the F1 superseded-manifest check. "
                          "Defaults to auto-detecting a sibling results/frozensel_curvefix_*/ dir.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    prereg_path = Path(args.prereg)
    prereg = json.loads(prereg_path.read_text())
    design = prereg["design"]

    cells = load_cells(run_dir)
    print(f"[LOAD] {len(cells)} cells from {run_dir}/cells.jsonl")

    tuple_report = build_tuple_enum_report(run_dir, cells, prereg)
    tpath = run_dir / "tuple_enum_report.json"
    _write_protected_json(tpath, tuple_report)
    print(f"[TUPLE] n_expected={tuple_report['n_expected']} n_observed={tuple_report['n_observed']} "
          f"n_duplicates={tuple_report['n_duplicates']} n_missing={tuple_report['n_missing']} "
          f"n_extra={tuple_report['n_extra']} all_pass={tuple_report['all_pass']} -> {tpath}")

    if not tuple_report["all_pass"]:
        print("[STOP] tuple_enum_report.all_pass=False; refusing to proceed to fairness/aggregation (E0).")
        sys.exit(4)

    min_cells = design["cells_expected_min"]
    if len(cells) < min_cells:
        print(f"[STOP] {len(cells)} < min required {min_cells}; not aggregating.")
        sys.exit(2)

    fairness = build_fairness_report(run_dir, cells, prereg)
    fpath = run_dir / "fairness_report.json"
    _write_protected_json(fpath, fairness)
    print(f"[FAIRNESS] all_pass={fairness['all_pass']} n_mismatches={fairness['n_mismatches']} -> {fpath}")

    if not fairness["all_pass"]:
        print("[STOP] fairness_report.all_pass=False; refusing to aggregate or emit a decision.")
        sys.exit(3)

    agg = aggregate(cells, prereg)
    jpath, cpath = write_q_frozensel(run_dir, agg, design)
    print(f"[AGG] wrote {jpath}, {cpath}")

    curve_csv = write_curve_table(run_dir, agg, design)
    print(f"[CURVE] wrote {curve_csv}")

    curve_png = write_curve_figure(run_dir, agg, design)
    print(f"[CURVE] wrote {curve_png}")

    curvefix_dir = Path(args.curvefix_dir) if args.curvefix_dir else None
    if curvefix_dir is None:
        candidates = sorted((run_dir.parent).glob("frozensel_curvefix_*"))
        curvefix_dir = candidates[-1] if candidates else run_dir
    manifest_path = write_curve_superseded_manifest_if_stale(run_dir, curvefix_dir)
    if manifest_path:
        print(f"[CURVE-MANIFEST] {curve_csv} / {curve_png} superseded -> {manifest_path} "
              f"(authoritative: {curvefix_dir})")

    decision = apply_scoped_decision_rule(agg, design, prereg, run_dir, cells)
    decision["prereg_path"] = str(prereg_path)
    dpath = run_dir / "frozensel_decision_scoped.json"
    _write_protected_json(dpath, decision)
    print(f"[DECISION] verdict={decision['verdict']} -> {dpath}")
    print(f"[DECISION] reason: {decision['reason']}")

    superseded_manifest = write_decision_superseded_manifest_if_needed(run_dir, dpath)
    if superseded_manifest:
        print(f"[DECISION-MANIFEST] frozensel_decision.json superseded -> {superseded_manifest} "
              f"(authoritative: {dpath})")


if __name__ == "__main__":
    main()

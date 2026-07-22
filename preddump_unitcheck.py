"""
preddump_unitcheck.py — smoke reconciliation for the --dump_predictions
switch added to xjtu_noisy_harness.py (task step 2: "单格 clean 2ep×1seed，
开 dump").

Guardrail-0 note: trains exactly ONE cell (arm=s4d, condition=clean, seed=0,
2 epochs) — within both the epoch and cell bounds ("loop 内禁 >2ep/>2cell
训练"). Unlike smoke_secondary_check.py / smoke_graft_check.py (which
reimplement their own train loop to avoid the full-grid defaults hardcoded
in xjtu_noisy_harness.main()), this script calls
xjtu_noisy_harness.run_grid() DIRECTLY with a single-cell conditions/arms/
seeds list — the actual harness code path that --dump_predictions patches —
because the entire point of this check is proving the predictions.jsonl
side-channel and the self-reported best_macro_f1 come from the same
in-training eval pass ("旁路与指标同源"). A reimplemented loop would not
exercise run_grid()'s new dump-write branch or train_cell()'s new
best-epoch-capture branch at all.

Checks (all_pass true only if all pass):
  1. eval_sha256 in the written cells.jsonl row == the literal 64-hex hash
     already established by every prior run in this repo (frozensel/
     secondary/graft), byte-for-byte (full-string equality, not just the
     task's 12-char prefix).
  2. predictions.jsonl line count == the eval split's support (independently
     recomputed via make_cross_condition_split + XJTUDataset, not read back
     from the harness).
  3. sample_id column is exactly range(support) with no gaps/duplicates.
  4. macro-F1 recomputed from predictions.jsonl's y_true/y_pred (using the
     identical tp/fn/fp -> per-class-F1 -> mean formula as
     eval_macro_f1()) equals the cell's self-reported best_macro_f1 in
     cells.jsonl bit-for-bit (float equality, not a tolerance band) — this
     is the key gate proving the dump and the metric are the same source.
  5. predictions.jsonl was chmod 444 by the harness.

Writes:
  - results/preddump_smoke_<ts>/            (raw harness output: cells.jsonl,
                                              noise_verification.json,
                                              s4d__clean__seed0/predictions.jsonl)
    — produced by calling the real xjtu_noisy_harness.run_grid(), not this
    script's own training loop.
  - results/preddump_unitcheck_<ts>/preddump_unitcheck.json  (chmod 444)
    — this check's report.

Does not touch any existing results/ directory (both output dirs above are
freshly timestamped). Does not call xjtu_noisy_harness.main() (avoids
argparse / the 5-seed/50-epoch full-grid defaults).
"""
import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import xjtu_noisy_harness as H

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPECTED_EVAL_SHA_FULL = "6c20b367522ce5db59720120f8a7cd12792c7e070c6ae84bd1985511998727de"
N_EPOCHS = 2
SEED = 0
ARM = "s4d"
CONDITION_LABEL = "clean"


def recompute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Identical formula to xjtu_noisy_harness.eval_macro_f1()'s per_f1/macro_f1
    computation, applied to the dumped y_true/y_pred instead of a live loader."""
    tp = np.zeros(H.N_CLASSES, dtype=np.int64)
    fn = np.zeros(H.N_CLASSES, dtype=np.int64)
    fp = np.zeros(H.N_CLASSES, dtype=np.int64)
    for c in range(H.N_CLASSES):
        tp[c] = int(((y_pred == c) & (y_true == c)).sum())
        fn[c] = int(((y_pred != c) & (y_true == c)).sum())
        fp[c] = int(((y_pred == c) & (y_true != c)).sum())
    per_f1 = []
    for c in range(H.N_CLASSES):
        prec = tp[c] / max(tp[c] + fp[c], 1)
        rec = tp[c] / max(tp[c] + fn[c], 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        per_f1.append(float(f1))
    return float(np.mean(per_f1))


def main():
    print(f"[INIT] device={device}")
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    smoke_run_dir = HERE / "results" / f"preddump_smoke_{ts}"

    conditions = [{"noise_type": "clean", "snr_db": None, "label": CONDITION_LABEL}]
    print(f"[RUN] run_grid(dump_predictions=True) arm={ARM} seed={SEED} "
          f"n_epochs={N_EPOCHS} run_dir={smoke_run_dir}")
    H.run_grid(smoke_run_dir, conditions, [ARM], [SEED], N_EPOCHS, device,
               verbose=True, dump_predictions=True)

    cells_path = smoke_run_dir / "cells.jsonl"
    cell_lines = cells_path.read_text().splitlines()
    report = {
        "device": str(device),
        "arm": ARM,
        "condition": CONDITION_LABEL,
        "seed": SEED,
        "n_epochs": N_EPOCHS,
        "smoke_run_dir": str(smoke_run_dir.relative_to(HERE)),
        "eval_sha256_expected_full": EXPECTED_EVAL_SHA_FULL,
    }

    checks = {}
    checks["exactly_one_cell_written"] = (len(cell_lines) == 1)
    cell = json.loads(cell_lines[0])
    report["cell"] = cell

    # 1. eval_sha256 byte-for-byte
    eval_sha_full = cell["fairness"]["eval_sha256"]
    checks["eval_sha256_full_match"] = (eval_sha_full == EXPECTED_EVAL_SHA_FULL)

    # Independently recompute support (test-set size) — not read back from harness.
    train_bearings, test_bearings = H.make_cross_condition_split(H.TRAIN_COND, H.TEST_COND)
    base_test = H.XJTUDataset(H.DATA_ROOT, test_bearings, n_sensors=1)
    support = len(base_test)
    report["support_independently_computed"] = support

    # predictions.jsonl location
    checks["predictions_file_field_present"] = ("predictions_file" in cell)
    pred_path = smoke_run_dir / cell["predictions_file"]
    checks["predictions_file_exists"] = pred_path.exists()

    pred_mode = oct(os.stat(pred_path).st_mode & 0o777) if pred_path.exists() else None
    checks["predictions_file_chmod_444"] = (pred_mode == "0o444")
    report["predictions_file"] = str(pred_path.relative_to(HERE))
    report["predictions_file_mode"] = pred_mode

    rows = [json.loads(line) for line in pred_path.read_text().splitlines()]
    report["predictions_row_count"] = len(rows)
    checks["predictions_rowcount_eq_support"] = (len(rows) == support)

    sample_ids = [r["sample_id"] for r in rows]
    checks["sample_id_is_exact_range"] = (sample_ids == list(range(len(rows))))

    y_true = np.array([r["y_true"] for r in rows], dtype=np.int64)
    y_pred = np.array([r["y_pred"] for r in rows], dtype=np.int64)
    has_logits = all("logits" in r for r in rows)
    checks["logits_present"] = has_logits

    recomputed_macro_f1 = recompute_macro_f1(y_true, y_pred)
    self_reported_macro_f1 = cell["best_macro_f1"]
    report["recomputed_macro_f1"] = recomputed_macro_f1
    report["self_reported_best_macro_f1"] = self_reported_macro_f1
    report["macro_f1_abs_diff"] = abs(recomputed_macro_f1 - self_reported_macro_f1)
    checks["macro_f1_bitexact_match"] = (recomputed_macro_f1 == self_reported_macro_f1)

    report["checks"] = checks
    report["all_pass"] = all(checks.values())

    print(json.dumps({"checks": checks,
                       "eval_sha256_full": eval_sha_full,
                       "support": support,
                       "predictions_row_count": len(rows),
                       "recomputed_macro_f1": recomputed_macro_f1,
                       "self_reported_best_macro_f1": self_reported_macro_f1},
                      indent=2))

    out_dir = HERE / "results" / f"preddump_unitcheck_{ts}"
    out_dir.mkdir(parents=True, exist_ok=False)
    out_path = out_dir / "preddump_unitcheck.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    os.chmod(out_path, 0o444)
    print(f"[written] {out_path} (chmod 444)")

    if not report["all_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

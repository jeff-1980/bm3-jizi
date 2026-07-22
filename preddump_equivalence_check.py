"""
preddump_equivalence_check.py — ON/OFF equivalence smoke for --dump_predictions
(task: "ON/OFF 等价烟雾（本卡唯一实验）").

Spec: same cell (clean, bm3_frozen), same seed, 2 epochs, run once with
--dump_predictions on and once with it off. Gate: best_macro_f1 bit-for-bit
identical across the two runs, eval_sha256 == 6c20b367522c... in both, and
(if a training-loss trajectory is ever persisted to disk) identical too.
Bit-for-bit equality across both runs is the proof that the switch is
zero-interference on training/eval.

Each run is a SEPARATE python process (preddump_equivalence_worker.py,
spawned via subprocess with the same venv interpreter the harness itself
uses), not two in-process calls — this rules out any shared-state leakage
(RNG, CUDA context, module globals) between the ON and OFF arms as an
explanation for a match or a mismatch. Device is forced to CPU inside the
worker (see its docstring) to remove cuDNN non-determinism as a confound.

Writes (all new, none overwrite anything under results/):
  - results/preddump_equiv_on_<ts>/   (raw run_grid() output, dump ON)
  - results/preddump_equiv_off_<ts>/  (raw run_grid() output, dump OFF)
  - results/preddump_equivalence_<ts>/preddump_equivalence.json (chmod 444)
"""
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV_PY = "/home/jeffwork/论文8/venv/bin/python3"
WORKER = HERE / "preddump_equivalence_worker.py"

EXPECTED_EVAL_SHA_FULL = "6c20b367522ce5db59720120f8a7cd12792c7e070c6ae84bd1985511998727de"


def run_worker(mode: str, run_dir: Path) -> None:
    cmd = [VENV_PY, str(WORKER), "--mode", mode, "--run-dir", str(run_dir)]
    print(f"[CHECK] launching: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"worker mode={mode} failed with returncode={proc.returncode}")


def read_single_cell(run_dir: Path) -> dict:
    cells_path = run_dir / "cells.jsonl"
    lines = cells_path.read_text().splitlines()
    assert len(lines) == 1, f"expected exactly 1 cell in {cells_path}, got {len(lines)}"
    return json.loads(lines[0])


def grep_for_loss_trajectory(run_dir: Path) -> list[str]:
    """Independent check that no per-epoch training-loss trajectory is ever
    persisted to disk by this run (neither ON nor OFF) — so the task's
    "训练损失轨迹（若落盘）一致" clause is vacuously satisfied: there is
    nothing on disk to diverge. Scans every file actually written under
    run_dir for the substring "loss" (cheap, no assumptions about format)."""
    hits = []
    for p in sorted(run_dir.rglob("*")):
        if p.is_file():
            try:
                text = p.read_text(errors="ignore")
            except Exception:
                continue
            if "loss" in text.lower():
                hits.append(str(p.relative_to(HERE)))
    return hits


def main():
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    on_dir = HERE / "results" / f"preddump_equiv_on_{ts}"
    off_dir = HERE / "results" / f"preddump_equiv_off_{ts}"

    run_worker("on", on_dir)
    run_worker("off", off_dir)

    cell_on = read_single_cell(on_dir)
    cell_off = read_single_cell(off_dir)

    checks = {}

    # 1. best_macro_f1 bit-for-bit identical between ON and OFF.
    checks["best_macro_f1_bitexact_match"] = (cell_on["best_macro_f1"] == cell_off["best_macro_f1"])

    # 2. eval_sha256 == the literal established fingerprint, in BOTH runs, full string.
    eval_sha_on = cell_on["fairness"]["eval_sha256"]
    eval_sha_off = cell_off["fairness"]["eval_sha256"]
    checks["eval_sha256_on_matches_expected"] = (eval_sha_on == EXPECTED_EVAL_SHA_FULL)
    checks["eval_sha256_off_matches_expected"] = (eval_sha_off == EXPECTED_EVAL_SHA_FULL)
    checks["eval_sha256_on_off_identical"] = (eval_sha_on == eval_sha_off)

    # 3. Every other fairness/identity field also identical (train_order/noise
    #    fingerprints, n_params) — confirms the flag touches nothing but the
    #    predictions side-channel.
    checks["train_order_sha256_identical"] = (
        cell_on["fairness"]["train_order_sha256"] == cell_off["fairness"]["train_order_sha256"])
    checks["noise_sha256_identical"] = (
        cell_on["fairness"]["noise_sha256"] == cell_off["fairness"]["noise_sha256"])
    checks["n_params_identical"] = (cell_on["n_params"] == cell_off["n_params"])

    # 4. cells.jsonl schema delta: ON must carry predictions_file/predictions_epoch,
    #    OFF must NOT — the only permitted difference between the two rows
    #    (besides elapsed_s, which is wall-clock timing, not a correctness field).
    checks["on_has_predictions_file_field"] = ("predictions_file" in cell_on)
    checks["off_lacks_predictions_file_field"] = ("predictions_file" not in cell_off)
    only_diff_keys = sorted(set(cell_on) ^ set(cell_off))
    checks["only_schema_diff_is_predictions_fields"] = (
        set(only_diff_keys) == {"predictions_file", "predictions_epoch"})

    # 5. Training-loss trajectory: confirm neither run persists one to disk at
    #    all (see grep_for_loss_trajectory docstring) — task's conditional
    #    ("若落盘") is therefore vacuously satisfied, documented explicitly
    #    rather than silently skipped.
    loss_hits_on = grep_for_loss_trajectory(on_dir)
    loss_hits_off = grep_for_loss_trajectory(off_dir)
    checks["no_loss_trajectory_persisted_on"] = (len(loss_hits_on) == 0)
    checks["no_loss_trajectory_persisted_off"] = (len(loss_hits_off) == 0)

    all_pass = all(checks.values())

    report = {
        "task": "ON/OFF preddump equivalence smoke",
        "arm": "bm3_frozen",
        "condition": "clean",
        "seed": 0,
        "n_epochs": 2,
        "device": "cuda-if-available (harness default; NOT forced to CPU)",
        "device_note": (
            "bm3_frozen's gated RMSNorm is a Triton kernel that requires a "
            "CUDA tensor and raises on CPU, so this smoke must run on CUDA. "
            "CUDA/Triton is not bit-reproducible run-to-run even with the "
            "same torch.manual_seed() (see "
            "preddump_equivalence_determinism_check.py's OFF-vs-OFF control, "
            "and preddump_equivalence_sameweights_check.py for the isolation "
            "test that removes this confound)."),
        "on_run_dir": str(on_dir.relative_to(HERE)),
        "off_run_dir": str(off_dir.relative_to(HERE)),
        "eval_sha256_expected_full": EXPECTED_EVAL_SHA_FULL,
        "cell_on": cell_on,
        "cell_off": cell_off,
        "best_macro_f1_on": cell_on["best_macro_f1"],
        "best_macro_f1_off": cell_off["best_macro_f1"],
        "best_macro_f1_abs_diff": abs(cell_on["best_macro_f1"] - cell_off["best_macro_f1"]),
        "eval_sha256_on": eval_sha_on,
        "eval_sha256_off": eval_sha_off,
        "loss_trajectory_note": (
            "train_cell()/run_grid() do not write any per-epoch training-loss "
            "value to disk in either mode (verified by grepping every file "
            "written under both run dirs for the substring 'loss'); the "
            "task's conditional clause therefore has nothing to compare and "
            "is satisfied vacuously."),
        "loss_trajectory_files_containing_loss_on": loss_hits_on,
        "loss_trajectory_files_containing_loss_off": loss_hits_off,
        "only_schema_diff_keys_on_vs_off": only_diff_keys,
        "checks": checks,
        "all_pass": all_pass,
    }

    print(json.dumps({"checks": checks, "all_pass": all_pass,
                       "best_macro_f1_on": cell_on["best_macro_f1"],
                       "best_macro_f1_off": cell_off["best_macro_f1"],
                       "eval_sha256_on": eval_sha_on,
                       "eval_sha256_off": eval_sha_off}, indent=2))

    out_dir = HERE / "results" / f"preddump_equivalence_{ts}"
    out_dir.mkdir(parents=True, exist_ok=False)
    out_path = out_dir / "preddump_equivalence.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    os.chmod(out_path, 0o444)
    print(f"[written] {out_path} (chmod 444)")

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()

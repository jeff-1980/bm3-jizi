"""
preddump_equivalence_determinism_check.py — pre-check for the ON/OFF
--dump_predictions equivalence smoke.

bm3_frozen must run on CUDA (its gated RMSNorm is a Triton kernel that
rejects CPU tensors — see preddump_equivalence_worker.py's docstring), so
the ON-vs-OFF comparison in preddump_equivalence_check.py cannot use CPU to
sidestep GPU nondeterminism. Before trusting a bit-for-bit ON-vs-OFF match
(or treating a mismatch as evidence against the switch), this script
establishes the GPU's own noise floor: run the SAME cell (bm3_frozen,
clean, seed=0, 2 epochs, dump_predictions=OFF both times) twice, as two
independent processes, and check whether best_macro_f1 is bit-for-bit
reproducible on this hardware absent any --dump_predictions involvement at
all. If OFF-vs-OFF already reproduces bit-for-bit, an ON-vs-OFF mismatch
could only be attributed to the switch; if OFF-vs-OFF does NOT reproduce,
GPU-level nondeterminism (not the switch) is the explanation for any
ON-vs-OFF diff, and the equivalence check must be interpreted accordingly.

Writes results/preddump_equiv_determinism_<ts>/{run_a,run_b}/ (raw
run_grid() output) and results/preddump_equiv_determinism_<ts>/determinism_check.json
(chmod 444). Does not touch any existing results/ directory.
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


def run_worker(mode: str, run_dir: Path) -> None:
    cmd = [VENV_PY, str(WORKER), "--mode", mode, "--run-dir", str(run_dir)]
    print(f"[DETCHECK] launching: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"worker mode={mode} failed with returncode={proc.returncode}")


def read_single_cell(run_dir: Path) -> dict:
    cells_path = run_dir / "cells.jsonl"
    lines = cells_path.read_text().splitlines()
    assert len(lines) == 1
    return json.loads(lines[0])


def main():
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = HERE / "results" / f"preddump_equiv_determinism_{ts}"
    run_a_dir = out_dir / "run_a"
    run_b_dir = out_dir / "run_b"

    run_worker("off", run_a_dir)
    run_worker("off", run_b_dir)

    cell_a = read_single_cell(run_a_dir)
    cell_b = read_single_cell(run_b_dir)

    checks = {
        "best_macro_f1_bitexact_match": (cell_a["best_macro_f1"] == cell_b["best_macro_f1"]),
        "eval_sha256_identical": (cell_a["fairness"]["eval_sha256"] == cell_b["fairness"]["eval_sha256"]),
        "n_params_identical": (cell_a["n_params"] == cell_b["n_params"]),
    }
    all_pass = all(checks.values())

    report = {
        "purpose": "establish GPU noise floor before trusting ON-vs-OFF comparison",
        "arm": "bm3_frozen", "condition": "clean", "seed": 0, "n_epochs": 2,
        "mode_both_runs": "off",
        "run_a_dir": str(run_a_dir.relative_to(HERE)),
        "run_b_dir": str(run_b_dir.relative_to(HERE)),
        "best_macro_f1_a": cell_a["best_macro_f1"],
        "best_macro_f1_b": cell_b["best_macro_f1"],
        "best_macro_f1_abs_diff": abs(cell_a["best_macro_f1"] - cell_b["best_macro_f1"]),
        "checks": checks,
        "gpu_is_bitexact_deterministic": all_pass,
    }
    print(json.dumps(report, indent=2))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "determinism_check.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    os.chmod(out_path, 0o444)
    print(f"[written] {out_path} (chmod 444)")


if __name__ == "__main__":
    main()

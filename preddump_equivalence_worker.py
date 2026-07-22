"""
preddump_equivalence_worker.py — ON/OFF worker process for the
--dump_predictions equivalence smoke (task: "ON/OFF 等价烟雾").

Trains exactly ONE cell (arm=bm3_frozen, condition=clean, seed=0, 2 epochs)
via xjtu_noisy_harness.run_grid() — the real grid-runner code path, not a
reimplemented loop — with dump_predictions on or off per --mode. Each
invocation of this script is a fresh Python process (spawned by
preddump_equivalence_check.py), so there is zero shared in-process state
(RNG, module globals, CUDA context) between the ON and OFF runs.

Device: cuda-if-available (harness default), NOT forced to CPU. bm3_frozen's
gated RMSNorm (mamba_ssm.ops.triton.layernorm_gated) is a Triton kernel that
requires a CUDA tensor — it raises on CPU ("Pointer argument ... cpu tensor?"),
so bm3_frozen cannot run this smoke on CPU at all. Empirical determinism of
this GPU/model combination across repeated runs is checked separately by
preddump_equivalence_determinism_check.py (OFF-vs-OFF, same seed) before the
ON-vs-OFF comparison is trusted as a pure test of the --dump_predictions
switch; see that script's docstring and REPORT_preddump_equivalence_*.md for
the result.

Writes: <run_dir>/cells.jsonl (+ noise_verification.json, + predictions.jsonl
subdir when mode=on) via the unmodified run_grid()/train_cell() code paths.
Does not touch any existing results/ directory (run_dir is a fresh
timestamped path passed in by the caller).
"""
import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import xjtu_noisy_harness as H

ARM = "bm3_frozen"
CONDITION_LABEL = "clean"
SEED = 0
N_EPOCHS = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["on", "off"], required=True)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dump_predictions = (args.mode == "on")
    run_dir = Path(args.run_dir)

    conditions = [{"noise_type": "clean", "snr_db": None, "label": CONDITION_LABEL}]
    print(f"[WORKER mode={args.mode}] device={device} arm={ARM} seed={SEED} "
          f"n_epochs={N_EPOCHS} dump_predictions={dump_predictions} run_dir={run_dir}")
    H.run_grid(run_dir, conditions, [ARM], [SEED], N_EPOCHS, device,
               verbose=True, dump_predictions=dump_predictions)
    print(f"[WORKER mode={args.mode}] done")


if __name__ == "__main__":
    main()

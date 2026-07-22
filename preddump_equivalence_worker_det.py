"""
preddump_equivalence_worker_det.py — deterministic-mode variant of
preddump_equivalence_worker.py, written to close the reviewer's §E1'
finding on `.orchestrate/0707-191044/r1_review.json` (2026-07-07 19:35
round): the prior ON/OFF gate used no CUDA determinism flags at all, so
the observed non-bit-exactness could not be distinguished from "flags
were simply never set" vs. "this GPU/kernel combination is fundamentally
non-reproducible". This script adds the standard PyTorch determinism knobs
BEFORE any CUDA op runs, then trains the same single cell
(bm3_frozen/clean/seed=0/2 epochs) via the unmodified
`xjtu_noisy_harness.run_grid()` code path, identically to the non-det
worker. `xjtu_noisy_harness.py` itself is not touched (harness-external
scratch script only, same as preddump_equivalence_worker.py).

Determinism knobs applied (all in-process, before importing/calling
anything CUDA-related):
  - CUBLAS_WORKSPACE_CONFIG=:4096:8 (env var, must be set before the first
    cuBLAS handle is created — set via subprocess env by the caller, see
    preddump_equivalence_determinism_check_det.py)
  - torch.backends.cudnn.deterministic = True
  - torch.backends.cudnn.benchmark = False
  - torch.use_deterministic_algorithms(True, warn_only=False) — raises
    immediately (loud failure, not silent) if any op in the actual forward
    /backward graph lacks a deterministic CUDA implementation, which is
    itself diagnostic information about why bit-exactness might still be
    unreachable.

Writes: <run_dir>/cells.jsonl (+ noise_verification.json, + predictions.jsonl
subdir when mode=on), exactly like the non-det worker. Does not touch any
existing results/ directory.
"""
import argparse
import sys
from pathlib import Path

import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=False)

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
    print(f"[WORKER-DET mode={args.mode}] device={device} arm={ARM} seed={SEED} "
          f"n_epochs={N_EPOCHS} dump_predictions={dump_predictions} run_dir={run_dir} "
          f"cudnn.deterministic={torch.backends.cudnn.deterministic} "
          f"cudnn.benchmark={torch.backends.cudnn.benchmark}")
    H.run_grid(run_dir, conditions, [ARM], [SEED], N_EPOCHS, device,
               verbose=True, dump_predictions=dump_predictions)
    print(f"[WORKER-DET mode={args.mode}] done")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Behavioral (not just textual) proof that xjtu_noisy_harness.py's eval_macro_f1
with collect_predictions=False (the default / --dump_predictions=False path)
computes byte-identical output to the reconstructed pre-edit eval_macro_f1
(which has no collect_predictions parameter at all).

This does NOT touch the real XJTU dataset, the real BearMamba3/S4D models,
or run any training loop — it is a pure unit check of the eval function in
isolation with a tiny synthetic model + synthetic batch, so it consumes zero
epochs/cells against the session's training cap. It only imports the two
harness files as modules (importing executes the same top-level dependency
imports the harness already does at every invocation; no training runs).

Usage: run from the repo root with the project venv:
  /home/jeffwork/论文8/venv/bin/python3 results/preddump_diff_20260707-1214/eval_macro_f1_equivalence_check.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent  # results/preddump_diff_.../  ->  repo root
CURRENT_PATH  = HERE / "xjtu_noisy_harness.CURRENT_COPY.py"
PREEDIT_PATH  = HERE / "xjtu_noisy_harness.PREEDIT_reconstructed.py"

# bm3_frozen / bm3_models / noise_utils are plain top-level modules that live
# in the repo root; when xjtu_noisy_harness.py is run directly as a script,
# Python auto-adds its own directory to sys.path[0], which is what makes
# `import bm3_frozen` etc. resolve. Loading these two copies via importlib
# from a results/ subdirectory does not get that auto-add, so it's done here
# explicitly — this does not change which files get imported (still the repo
# root's bm3_frozen.py/bm3_models.py/noise_utils.py), only makes them findable.
sys.path.insert(0, str(REPO_ROOT))


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TinyModel(nn.Module):
    """Fixed-weight linear classifier — deterministic in eval() mode, no randomness."""
    def __init__(self, n_in=8, n_classes=2):
        super().__init__()
        self.fc = nn.Linear(n_in, n_classes)

    def forward(self, x):
        return self.fc(x)


def make_fixed_loader(n_in=8, n_classes=2, n_batches=5, batch_size=16, seed=1234):
    g = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(n_batches):
        x = torch.randn(batch_size, n_in, generator=g)
        labels = torch.randint(0, n_classes, (batch_size,), generator=g)
        rpm = torch.zeros(batch_size)
        batches.append((x, labels, rpm))
    return batches  # a plain list is iterable exactly like a DataLoader in this loop


def main():
    current  = _load_module(CURRENT_PATH, "xjtu_harness_current_copy")
    preedit  = _load_module(PREEDIT_PATH, "xjtu_harness_preedit_reconstructed")

    device = torch.device("cpu")
    torch.manual_seed(0)
    model = TinyModel(n_in=8, n_classes=current.N_CLASSES)
    model.eval()

    loader = make_fixed_loader(n_in=8, n_classes=current.N_CLASSES)

    # Pre-edit reconstructed: no collect_predictions kwarg exists at all.
    per_f1_pre, macro_f1_pre = preedit.eval_macro_f1(model, loader, device)

    # Current file, dump_predictions=False path == collect_predictions default False.
    ret_current_default = current.eval_macro_f1(model, loader, device)
    assert len(ret_current_default) == 2, \
        f"current eval_macro_f1() default call returned {len(ret_current_default)}-tuple, expected 2 (unchanged default arity)"
    per_f1_cur_default, macro_f1_cur_default = ret_current_default

    # Current file, explicit collect_predictions=False (same as dump_predictions=False's call site).
    ret_current_explicit_false = current.eval_macro_f1(model, loader, device, collect_predictions=False)
    per_f1_cur_explicit, macro_f1_cur_explicit = ret_current_explicit_false

    per_f1_match_default  = bool(np.array_equal(per_f1_pre, per_f1_cur_default))
    macro_f1_match_default = bool(macro_f1_pre == macro_f1_cur_default)
    per_f1_match_explicit  = bool(np.array_equal(per_f1_pre, per_f1_cur_explicit))
    macro_f1_match_explicit = bool(macro_f1_pre == macro_f1_cur_explicit)

    # Also confirm collect_predictions=True does NOT change tp/fn/fp-derived per_f1/macro_f1
    # (same source array pass, per report's "same source" claim) — 5-tuple, first two entries equal.
    ret_current_true = current.eval_macro_f1(model, loader, device, collect_predictions=True)
    assert len(ret_current_true) == 5, \
        f"current eval_macro_f1(collect_predictions=True) returned {len(ret_current_true)}-tuple, expected 5"
    per_f1_true, macro_f1_true, y_true, y_pred, logits = ret_current_true
    per_f1_match_dumpon  = bool(np.array_equal(per_f1_pre, per_f1_true))
    macro_f1_match_dumpon = bool(macro_f1_pre == macro_f1_true)

    result = {
        "check": "eval_macro_f1 dump-off/default behavioral equivalence vs reconstructed pre-edit baseline",
        "model": "TinyModel(nn.Linear, fixed weights, eval() mode, deterministic synthetic batches)",
        "note_no_training_no_real_dataset": True,
        "per_f1_match_default_call": per_f1_match_default,
        "macro_f1_match_default_call": macro_f1_match_default,
        "per_f1_match_explicit_collect_predictions_false": per_f1_match_explicit,
        "macro_f1_match_explicit_collect_predictions_false": macro_f1_match_explicit,
        "per_f1_match_dumpon_same_source_as_dumpoff": per_f1_match_dumpon,
        "macro_f1_match_dumpon_same_source_as_dumpoff": macro_f1_match_dumpon,
        "preedit_per_f1": per_f1_pre.tolist(),
        "preedit_macro_f1": macro_f1_pre,
        "current_default_per_f1": per_f1_cur_default.tolist(),
        "current_default_macro_f1": macro_f1_cur_default,
        "current_dumpon_per_f1": per_f1_true.tolist(),
        "current_dumpon_macro_f1": macro_f1_true,
        "all_pass": bool(
            per_f1_match_default and macro_f1_match_default and
            per_f1_match_explicit and macro_f1_match_explicit and
            per_f1_match_dumpon and macro_f1_match_dumpon
        ),
    }
    out_path = HERE / "eval_macro_f1_equivalence_check.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""
preddump_equivalence_sameweights_check.py — decisive isolation test for the
ON/OFF --dump_predictions equivalence smoke.

Why this script exists: preddump_equivalence_check.py ran the literal task
ask (same cell, same seed, 2ep, dump ON in one process / OFF in another) and
found best_macro_f1 differs between the two runs (0.6799057... vs
0.6819760...). preddump_equivalence_determinism_check.py independently
showed this is NOT caused by --dump_predictions: two OFF-vs-OFF runs of the
identical cell differ by *more* (0.0068736...) than the ON-vs-OFF diff
(0.0020703...). The root cause is that bm3_frozen's gated RMSNorm
(mamba_ssm.ops.triton.layernorm_gated) is a Triton kernel whose backward
pass is not bit-reproducible across separate CUDA process launches, even
with an identical torch.manual_seed() — a pre-existing hardware/kernel
property of this arm, unrelated to the flag this card added.

Cross-process comparison therefore cannot isolate the flag's effect from
that pre-existing GPU noise. This script does what CAN isolate it: train
bm3_frozen for the SAME 2 epochs ONCE (one process, one set of weights),
then call the harness's real eval_macro_f1() TWICE on that single frozen
model/loader — once with collect_predictions=False (the OFF code path)
and once with collect_predictions=True (the ON code path) — and diff the
returned macro_f1 bit-for-bit. Since both calls run against the identical
already-trained weights with no intervening backward pass, any GPU kernel
non-determinism from training is fully absent from this comparison; only
the code delta introduced by the flag (an extra bookkeeping accumulation
inside the SAME tp/fn/fp loop, see eval_macro_f1()'s docstring) can affect
the result.

This reuses only real harness functions/objects (set_seed, build_model,
make_sampler_weights, XJTUDataset(Noisy), eval_macro_f1) — the outer 2-epoch
training loop below is copied from train_cell()'s body (unavoidable: it does
not return the trained model object, only a result dict) rather than a
from-scratch reimplementation of training semantics. No line of
xjtu_noisy_harness.py is modified.

Writes results/preddump_equiv_sameweights_<ts>/sameweights_check.json
(chmod 444). Trains exactly ONE cell (bm3_frozen/clean/seed0), 2 epochs.
"""
import datetime
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import xjtu_noisy_harness as H

ARM = "bm3_frozen"
SEED = 0
N_EPOCHS = 2


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SAMEWEIGHTS] device={device} arm={ARM} seed={SEED} n_epochs={N_EPOCHS}")

    H.set_seed(SEED)
    train_bearings, test_bearings = H.make_cross_condition_split(H.TRAIN_COND, H.TEST_COND)
    base_train = H.XJTUDataset(H.DATA_ROOT, train_bearings, n_sensors=1)
    base_test = H.XJTUDataset(H.DATA_ROOT, test_bearings, n_sensors=1)
    train_ds = H.XJTUDatasetNoisy(base_train, "clean", None, SEED)
    test_ds = H.XJTUDatasetNoisy(base_test, "clean", None, SEED)
    eval_sha = H.eval_fingerprint(test_ds)
    print(f"[SAMEWEIGHTS] eval_sha256={eval_sha}")

    # --- Training loop copied verbatim from train_cell()'s body (up to the
    # point of returning the trained model, which train_cell() does not
    # expose) so this test can evaluate the SAME model twice afterward. ---
    H.set_seed(SEED)
    model = H.build_model(ARM, device)
    weights_list = H.make_sampler_weights(train_ds._labels)
    n_samples = len(train_ds)
    g = torch.Generator()
    g.manual_seed(SEED)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights_list, num_samples=n_samples, replacement=True, generator=g,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=H.BATCH_SIZE, sampler=sampler,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=H.LR, weight_decay=H.WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    param_dtype = next(model.parameters()).dtype

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        for x, labels, rpm in train_loader:
            x = x.to(device, dtype=param_dtype)
            labels = labels.to(device)
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = F.cross_entropy(logits.float(), labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), H.GRAD_CLIP)
            optimizer.step()
        scheduler.step()
        print(f"    ep={epoch}/{N_EPOCHS} trained")

    # --- Decisive comparison: SAME frozen model, SAME loader, real
    # eval_macro_f1() called twice — OFF branch then ON branch. ---
    per_f1_off, macro_f1_off = H.eval_macro_f1(model, test_loader, device,
                                                collect_predictions=False)
    per_f1_on, macro_f1_on, y_true, y_pred, logits = H.eval_macro_f1(
        model, test_loader, device, collect_predictions=True)

    checks = {
        "macro_f1_bitexact_match": (macro_f1_off == macro_f1_on),
        "per_f1_bitexact_match": (per_f1_off.tolist() == per_f1_on.tolist()),
        "predictions_rowcount_eq_test_loader_len": (len(y_true) == len(test_ds)),
    }
    all_pass = all(checks.values())

    report = {
        "purpose": (
            "isolate the --dump_predictions code delta from cross-process "
            "GPU/Triton training nondeterminism: same trained weights, "
            "same eval loader, real eval_macro_f1() called with "
            "collect_predictions=False then True"),
        "arm": ARM, "condition": "clean", "seed": SEED, "n_epochs": N_EPOCHS,
        "device": str(device),
        "eval_sha256": eval_sha,
        "macro_f1_off": macro_f1_off,
        "macro_f1_on": macro_f1_on,
        "macro_f1_abs_diff": abs(macro_f1_off - macro_f1_on),
        "per_f1_off": per_f1_off.tolist(),
        "per_f1_on": per_f1_on.tolist(),
        "checks": checks,
        "all_pass": all_pass,
    }
    print(json.dumps({"checks": checks, "all_pass": all_pass,
                       "macro_f1_off": macro_f1_off, "macro_f1_on": macro_f1_on},
                      indent=2))

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = HERE / "results" / f"preddump_equiv_sameweights_{ts}"
    out_dir.mkdir(parents=True, exist_ok=False)
    out_path = out_dir / "sameweights_check.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    os.chmod(out_path, 0o444)
    print(f"[written] {out_path} (chmod 444)")

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()

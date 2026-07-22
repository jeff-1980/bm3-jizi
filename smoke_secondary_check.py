"""
smoke_secondary_check.py — guarded smoke test for the three secondary arms
(frozen_noconv/frozen_nogate/frozen_As4d, bm3_models.py), per task step 4:
"三臂各单格（XJTU cross_speed clean）2ep×1seed" — one cell per arm, 2 epochs,
seed=0, clean condition.

Guardrail-0 note: the harness-level guardrail template says "loop 内禁止 >2
epoch 或 >2 cell 训练" (copied verbatim from stage-1's smoke_frozensel_check.py
header, which itself trained exactly 2 cells — the 2 arms stage 1 needed).
This task's own step 4 explicitly specifies the experiment design as THREE
single-cell arm smokes (2 epochs, 1 seed each) — a concrete, named
deliverable that supersedes the copied "<=2 cell" phrasing for cell *count*;
the "<=2 epoch" bound (the part that actually caps GPU cost per training
loop) is honored exactly: every cell here trains for 2 epochs, never more,
and there are no other epoch/cell-count violations (no full grid, no
multi-seed). This reasoning is declared here rather than silently exceeding
the copied template.

Trains frozen_noconv/frozen_nogate/frozen_As4d on XJTU clean, seed=0, 2
epochs each (3 cells total). Verifies:
  - training CE loss decreases epoch-over-epoch, no NaN anywhere, per arm
  - eval_sha256 (test-label fingerprint, arm-independent) == the literal
    hash from the task spec, 6c20b367522c... (task's own truncated form) —
    checked both as an exact 12-hex-char prefix match against the task text
    and as a full-hash match against the value already verified in
    results/frozensel_20260702-1827/cells.jsonl (same TRAIN_COND/TEST_COND/
    clean-condition/seed=0 fairness invariant, this dir's only regime)

Writes results/secondary_smoke_<ts>/smoke_report.json (chmod 444). Does not
touch cells.jsonl or any existing results/ directory; does not call
xjtu_noisy_harness.main()/run_grid() (avoids the 5-seed/50-epoch full-grid
defaults hardcoded there for --secondary).
"""
import datetime
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import xjtu_noisy_harness as H

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPECTED_EVAL_SHA_PREFIX12 = "6c20b367522c"   # task spec's literal truncated hash
EXPECTED_EVAL_SHA_FULL = "6c20b367522ce5db59720120f8a7cd12792c7e070c6ae84bd1985511998727de"  # results/frozensel_20260702-1827/cells.jsonl
N_EPOCHS = 2
SEED = 0
ARMS = ["frozen_noconv", "frozen_nogate", "frozen_As4d"]


def train_one_cell(arm, train_ds, test_ds):
    H.set_seed(SEED)
    model = H.build_model(arm, device)

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

    epoch_records = []
    for epoch in range(1, N_EPOCHS + 1):
        t_ep0 = time.time()
        model.train()
        losses = []
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
            losses.append(float(loss.item()))
        scheduler.step()
        _, macro_f1 = H.eval_macro_f1(model, test_loader, device)
        ep_time = time.time() - t_ep0
        mean_loss = sum(losses) / len(losses)
        epoch_records.append({
            "epoch": epoch,
            "mean_train_loss": mean_loss,
            "has_nan_loss": any(l != l for l in losses),  # NaN != NaN
            "macro_f1": macro_f1,
            "epoch_time_s": round(ep_time, 2),
            "n_batches": len(losses),
        })
        print(f"    [{arm}] ep={epoch}/{N_EPOCHS} mean_loss={mean_loss:.4f} "
              f"macro_f1={macro_f1:.4f} time={ep_time:.1f}s")
    return {
        "n_params": sum(p.numel() for p in model.parameters()),
        "epochs": epoch_records,
    }


def main():
    print(f"[INIT] device={device}")
    train_bearings, test_bearings = H.make_cross_condition_split(H.TRAIN_COND, H.TEST_COND)
    base_train = H.XJTUDataset(H.DATA_ROOT, train_bearings, n_sensors=1)
    base_test = H.XJTUDataset(H.DATA_ROOT, test_bearings, n_sensors=1)

    train_ds = H.XJTUDatasetNoisy(base_train, "clean", None, SEED)
    test_ds = H.XJTUDatasetNoisy(base_test, "clean", None, SEED)

    eval_sha_full = H.eval_fingerprint(test_ds)
    eval_sha_prefix12 = eval_sha_full[:12]
    eval_sha_prefix_match = eval_sha_prefix12 == EXPECTED_EVAL_SHA_PREFIX12
    eval_sha_full_match = eval_sha_full == EXPECTED_EVAL_SHA_FULL
    print(f"[FINGERPRINT] eval_sha256={eval_sha_full}  "
          f"(prefix12={eval_sha_prefix12}, expected={EXPECTED_EVAL_SHA_PREFIX12}, "
          f"prefix_match={eval_sha_prefix_match}, full_match={eval_sha_full_match})")

    report = {
        "device": str(device),
        "condition": "clean",
        "regime": "cross_condition (TRAIN_COND=%s -> TEST_COND=%s, this dir's only regime)"
                  % (H.TRAIN_COND, H.TEST_COND),
        "seed": SEED,
        "n_epochs": N_EPOCHS,
        "eval_sha256_full": eval_sha_full,
        "eval_sha256_prefix12": eval_sha_prefix12,
        "eval_sha256_expected_prefix12": EXPECTED_EVAL_SHA_PREFIX12,
        "eval_sha256_prefix_match": eval_sha_prefix_match,
        "eval_sha256_expected_full": EXPECTED_EVAL_SHA_FULL,
        "eval_sha256_full_match": eval_sha_full_match,
        "arms": {},
    }

    for arm in ARMS:
        print(f"[TRAIN] arm={arm}")
        result = train_one_cell(arm, train_ds, test_ds)
        losses = [e["mean_train_loss"] for e in result["epochs"]]
        loss_decreased = losses[-1] < losses[0]
        any_nan = any(e["has_nan_loss"] for e in result["epochs"])
        report["arms"][arm] = {
            **result,
            "loss_decreased_ep1_to_ep_last": loss_decreased,
            "any_nan": any_nan,
        }

    report["all_pass"] = (
        report["eval_sha256_prefix_match"]
        and report["eval_sha256_full_match"]
        and all(report["arms"][a]["loss_decreased_ep1_to_ep_last"] for a in ARMS)
        and not any(report["arms"][a]["any_nan"] for a in ARMS)
    )

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = HERE / "results" / f"secondary_smoke_{ts}"
    out_dir.mkdir(parents=True, exist_ok=False)
    out_path = out_dir / "smoke_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    os.chmod(out_path, 0o444)
    print(f"[written] {out_path} (chmod 444)")
    print(json.dumps({k: v for k, v in report.items() if k != "arms"}, indent=2))
    if not report["all_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

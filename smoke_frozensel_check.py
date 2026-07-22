"""
smoke_frozensel_check.py — one-off guarded smoke test for bm3_frozen (task
guardrail 0b: <=2 epochs, <=2 cells trained in this session).

Trains bm3_frozen AND bm3_kin (for throughput comparison) on XJTU clean,
seed=0, 2 epochs each (2 cells total). Verifies:
  - training CE loss decreases epoch-over-epoch, no NaN anywhere
  - eval_sha256 (test-label fingerprint, arm-independent) == 6c20b367522ce5db
    (the project-level fair-comparison hash also stamped into every cell of
    ablation_nokin_20260630-0958 and capctrl_s4dwide_20260701-2237)
  - bm3_frozen per-epoch wall time is the same order of magnitude as bm3_kin

Writes results/frozensel_smoke_<ts>/smoke_report.json (chmod 444). Does not
touch cells.jsonl or any existing results/ directory; does not call
xjtu_noisy_harness.main()/run_grid() (avoids the 5-seed/50-epoch full-grid
defaults hardcoded there for --ablation/--capctrl-style flags).
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
EXPECTED_EVAL_SHA16 = "6c20b367522ce5db"
N_EPOCHS = 2
SEED = 0
ARMS = ["bm3_frozen", "bm3_kin"]


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

    is_bm3 = (arm == "bm3_kin")
    param_dtype = next(model.parameters()).dtype

    epoch_records = []
    for epoch in range(1, N_EPOCHS + 1):
        t_ep0 = time.time()
        model.train()
        losses = []
        for x, labels, rpm in train_loader:
            x = x.to(device, dtype=param_dtype)
            labels = labels.to(device)
            rpm = rpm.to(device)
            if is_bm3:
                out, kin = model(x, return_kin=True)
                l_ce = F.cross_entropy(out.float(), labels)
                l_kin = H.kinematic_loss(kin, rpm, H.FS_EFF, variant="cover",
                                         bearing_kwargs=H.BEARING_KW)
                loss = l_ce + H.LAMBDA_KIN * l_kin
            else:
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
    eval_sha16 = eval_sha_full[:16]
    eval_sha_match = eval_sha16 == EXPECTED_EVAL_SHA16
    print(f"[FINGERPRINT] eval_sha256={eval_sha_full}  "
          f"(prefix16={eval_sha16}, expected={EXPECTED_EVAL_SHA16}, match={eval_sha_match})")

    report = {
        "device": str(device),
        "condition": "clean",
        "regime": "cross_condition (TRAIN_COND=%s -> TEST_COND=%s, this dir's only regime)"
                  % (H.TRAIN_COND, H.TEST_COND),
        "seed": SEED,
        "n_epochs": N_EPOCHS,
        "eval_sha256_full": eval_sha_full,
        "eval_sha256_prefix16": eval_sha16,
        "eval_sha256_expected_prefix16": EXPECTED_EVAL_SHA16,
        "eval_sha256_match": eval_sha_match,
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

    t_frozen = sum(e["epoch_time_s"] for e in report["arms"]["bm3_frozen"]["epochs"])
    t_bm3 = sum(e["epoch_time_s"] for e in report["arms"]["bm3_kin"]["epochs"])
    ratio = t_frozen / t_bm3 if t_bm3 > 0 else float("inf")
    report["throughput_same_order_of_magnitude"] = 0.2 <= ratio <= 5.0
    report["bm3_frozen_vs_bm3_kin_time_ratio"] = round(ratio, 3)

    report["all_pass"] = (
        report["eval_sha256_match"]
        and all(report["arms"][a]["loss_decreased_ep1_to_ep_last"] for a in ARMS)
        and not any(report["arms"][a]["any_nan"] for a in ARMS)
        and report["throughput_same_order_of_magnitude"]
    )

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = HERE / "results" / f"frozensel_smoke_{ts}"
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

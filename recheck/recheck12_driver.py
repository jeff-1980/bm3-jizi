"""
Paper-9 stage-12 (SASC revision): bearing-level description of the forward independent-noise grid.

The stage-8 grids stored per-epoch macro-F1 only, not per-window predictions, so class- and bearing-level
results of the main grid cannot be recovered. This driver re-runs a 20-cell subset of stage-8 grids A and C
(forward XJTU-SY 37.5Hz11kN -> 40Hz10kN, -6 dB, arms frozen_clti, frozen_clti_nogate, frozen_clti_add, s4d,
seeds 0-4) with the stage-8 code path unchanged (same build, sampler, optimiser, independent noise keyed on
[seed, split, idx]) and additionally writes the final-epoch prediction of every evaluation window together
with the bearing it came from. It is descriptive only (prereg_stage12_bearing_level.md); no adjudication.
Cells are not bitwise reproductions of stage 8 (non-deterministic GPU kernels); stage-8 final values are
recorded next to the new ones for comparison.
"""
import sys, json, time, argparse, os
from pathlib import Path
SRC = os.environ.get("P9_HARNESS_DIR") or str(Path(__file__).resolve().parents[1])
sys.path.insert(0, SRC)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness_loader
h = harness_loader.load(SRC)
import numpy as np, torch, torch.nn.functional as F
import layered_freeze as lf
import additive_native as an
import clti_gate as cg2
import indep_noise as ino

KW_BASE = dict(d_model=64, d_state=128, n_layers=4, n_sensors=1, n_classes=h.N_CLASSES,
               conv_stride=2, use_batchnorm=False, dtype=torch.bfloat16)
ARMS = ["frozen_clti", "frozen_clti_nogate", "frozen_clti_add", "s4d"]
IND = ino.make(h)
OUT = "p9_recheck/results/cells_stage12_bearing.jsonl"
PRED = "p9_recheck/results/preds_stage12"


def build(arm, device):
    if arm in lf.ARMS:
        return lf.build_arm(arm, device, **KW_BASE)
    if arm in an.ARMS:
        return an.build_arm(arm, device, **KW_BASE)
    if arm in cg2.ARMS:
        return cg2.build_arm(arm, device, **KW_BASE)
    return h.build_model(arm, device)


def train_cell(arm, train_ds, test_ds, seed, n_epochs, device):
    # identical to recheck8_driver.train_cell_hist (no bm3_kin arm here), plus final-epoch predictions
    t0 = time.time()
    h.set_seed(seed)
    model = build(arm, device)
    weights_list = h.make_sampler_weights(train_ds._labels)
    g = torch.Generator(); g.manual_seed(seed)
    sampler = torch.utils.data.WeightedRandomSampler(weights=weights_list, num_samples=len(train_ds),
                                                     replacement=True, generator=g)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=h.BATCH_SIZE, sampler=sampler,
                                               num_workers=0, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False,
                                              num_workers=0, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=h.LR, weight_decay=h.WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    pdt = next(model.parameters()).dtype
    hist, losses = [], []
    for epoch in range(1, n_epochs + 1):
        model.train(); ls = []
        for x, labels, rpm in train_loader:
            x = x.to(device, dtype=pdt); labels = labels.to(device)
            logits = model(x)
            if isinstance(logits, tuple): logits = logits[0]
            loss = F.cross_entropy(logits.float(), labels)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), h.GRAD_CLIP); opt.step()
            ls.append(float(loss))
        sched.step()
        if epoch < n_epochs:
            _, mf1 = h.eval_macro_f1(model, test_loader, device)
        else:
            _, mf1, y_true, y_pred, _ = h.eval_macro_f1(model, test_loader, device, collect_predictions=True)
        hist.append(mf1); losses.append(float(np.mean(ls)))
    hist = np.array(hist)
    return dict(hist=hist.tolist(), train_loss=losses, final=float(hist[-1]),
                n_params=sum(p.numel() for p in model.parameters()),
                elapsed_s=round(time.time() - t0, 1)), np.asarray(y_true), np.asarray(y_pred)


def bearing_of_window(test_bearings):
    ids = []
    for b in test_bearings:  # XJTUDataset concatenates windows bearing by bearing, in list order
        ids += [b] * len(h.XJTUDataset(h.DATA_ROOT, [b], n_sensors=1))
    return np.array(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr", type=float, default=-6.0)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--stage8", default="recheck/stage8", help="directory with the released stage-8 JSONL")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    out = Path(OUT); out.parent.mkdir(parents=True, exist_ok=True); Path(PRED).mkdir(parents=True, exist_ok=True)
    ref = {}
    for f in ["cells_stage8_A_xjtu.jsonl", "cells_stage8_C_xjtu_clti.jsonl"]:
        for l in open(Path(a.stage8) / f):
            r = json.loads(l); ref[(r["arm"], r["snr_db"], r["seed"])] = r
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            c = json.loads(l); done.add((c["arm"], c["seed"]))
    dev = torch.device("cuda")
    trb, teb = h.make_cross_condition_split("37.5Hz11kN", "40Hz10kN")
    base_tr = h.XJTUDataset(h.DATA_ROOT, trb, n_sensors=1); base_te = h.XJTUDataset(h.DATA_ROOT, teb, n_sensors=1)
    assert not set(trb) & set(teb)
    bear = bearing_of_window(teb)
    assert len(bear) == len(base_te._labels), (len(bear), len(base_te._labels))
    eval_sha = h.eval_fingerprint(h.XJTUDatasetNoisy(base_te, "clean", None, 0))
    assert eval_sha.startswith("6c20b367522c"), eval_sha
    print(f"[INIT] test bearings {list(teb)}  windows {len(bear)}  eval_sha {eval_sha[:12]}", flush=True)
    label = f"awgn@{a.snr:+.0f}dB"
    with open(out, "a") as f:
        for seed in seeds:
            tr = IND(base_tr, "awgn", a.snr, seed, split="train")
            te = IND(base_te, "awgn", a.snr, seed, split="test")
            for arm in ARMS:
                if (arm, seed) in done: continue
                print(f"[RUN] {label} {arm} seed={seed}", flush=True)
                r, yt, yp = train_cell(arm, tr, te, seed, a.epochs, dev)
                assert np.array_equal(yt, np.asarray(base_te._labels))
                per_bearing = {}
                for b in teb:
                    m = bear == b
                    per_bearing[b] = dict(label=int(yt[m][0]), n=int(m.sum()),
                                          n_correct=int((yp[m] == yt[m]).sum()),
                                          recall=float((yp[m] == yt[m]).mean()))
                np.savez_compressed(Path(PRED) / f"{arm}_seed{seed}.npz", y_true=yt, y_pred=yp, bearing=bear)
                s8 = ref.get((arm, a.snr, seed), {}).get("final")
                rec = dict(stage=12, data="xjtu", noise_protocol="independent[seed,split,idx]", condition=label,
                           snr_db=a.snr, arm=arm, seed=seed, epochs=a.epochs, eval_sha256=eval_sha,
                           per_bearing=per_bearing, stage8_final=s8, **r)
                f.write(json.dumps(rec) + "\n"); f.flush()
                print(f"      final={r['final']*100:.2f} (stage8 {s8*100 if s8 else float('nan'):.2f}) "
                      + " ".join(f"{b}:{v['recall']:.2f}" for b, v in per_bearing.items())
                      + f" {r['elapsed_s']:.0f}s", flush=True)
    print("[DONE]", flush=True)


main()

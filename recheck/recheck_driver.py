"""
Paper-9 selection-rule recheck. Re-runs a subset of the original XJTU cross-speed AWGN grid with the ORIGINAL harness code
(imported unmodified from the read-only source dir), recording the full per-epoch target-domain macro-F1 history so the
reported metric can be computed under rules that do not pick the epoch on the test set:
    final = macro-F1 at last epoch (cosine LR -> 0); last5 = mean of last 5 epochs; oracle = max over epochs (original rule).
Training loop identical to harness train_cell; only the evaluation record is extended. Seed-major order, resumable.
"""
import sys, json, time, argparse, importlib.util
from pathlib import Path
SRC = "/home/jeffwork/exp/bm3-defense/xjtu_noisy_defense_20260627"
sys.path.insert(0, SRC)
spec = importlib.util.spec_from_file_location("h", f"{SRC}/xjtu_noisy_harness.py")
h = importlib.util.module_from_spec(spec); sys.modules["h"] = h; spec.loader.exec_module(h)
import numpy as np, torch, torch.nn.functional as F

def train_cell_hist(arm, train_ds, test_ds, seed, n_epochs, device):
    t0 = time.time()
    h.set_seed(seed)
    model = h.build_model(arm, device)
    weights_list = h.make_sampler_weights(train_ds._labels)
    g = torch.Generator(); g.manual_seed(seed)
    sampler = torch.utils.data.WeightedRandomSampler(weights=weights_list, num_samples=len(train_ds), replacement=True, generator=g)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=h.BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
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
        _, mf1 = h.eval_macro_f1(model, test_loader, device)
        hist.append(mf1); losses.append(float(np.mean(ls)))
    hist = np.array(hist)
    return dict(hist=hist.tolist(), train_loss=losses, oracle=float(hist.max()), oracle_epoch=int(hist.argmax()) + 1,
                final=float(hist[-1]), last5=float(hist[-5:].mean()),
                n_params=sum(p.numel() for p in model.parameters()), elapsed_s=round(time.time() - t0, 1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="bm3_frozen,s4d,frozen_nogate,s4d_plus_gate")
    ap.add_argument("--snrs", default="0,-2,-6")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--out", default="p9_recheck/results/cells_recheck.jsonl")
    a = ap.parse_args()
    arms = a.arms.split(","); snrs = [float(s) for s in a.snrs.split(",")]; seeds = [int(s) for s in a.seeds.split(",")]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            try: c = json.loads(l); done.add((c["condition"], c["arm"], c["seed"]))
            except Exception: pass
    dev = torch.device("cuda")
    trb, teb = h.make_cross_condition_split(h.TRAIN_COND, h.TEST_COND)
    base_tr = h.XJTUDataset(h.DATA_ROOT, trb, n_sensors=1); base_te = h.XJTUDataset(h.DATA_ROOT, teb, n_sensors=1)
    eval_sha = h.eval_fingerprint(h.XJTUDatasetNoisy(base_te, "clean", None, 0))
    print("[INIT] eval_sha", eval_sha[:12], flush=True)
    assert eval_sha.startswith("6c20b367522c"), "eval fingerprint differs from the original grids"
    conds = [c for c in h.build_conditions(["awgn"], h.FULL_SNRS) if c["snr_db"] in snrs]
    total = len(conds) * len(arms) * len(seeds); n = len(done)
    with open(out, "a") as f:
        for seed in seeds:
            for cond in conds:
                tr = h.XJTUDatasetNoisy(base_tr, cond["noise_type"], cond["snr_db"], seed)
                te = h.XJTUDatasetNoisy(base_te, cond["noise_type"], cond["snr_db"], seed)
                assert h.eval_fingerprint(te) == eval_sha
                for arm in arms:
                    if (cond["label"], arm, seed) in done: continue
                    n += 1; print(f"[RUN] {cond['label']} {arm} seed={seed} ({n}/{total})", flush=True)
                    r = train_cell_hist(arm, tr, te, seed, a.epochs, dev)
                    rec = dict(condition=cond["label"], noise_type=cond["noise_type"], snr_db=cond["snr_db"], arm=arm, seed=seed,
                               epochs=a.epochs, eval_sha256=eval_sha, **r)
                    f.write(json.dumps(rec) + "\n"); f.flush()
                    print(f"      oracle={r['oracle']*100:.2f} (ep{r['oracle_epoch']}) final={r['final']*100:.2f} last5={r['last5']*100:.2f} {r['elapsed_s']:.0f}s", flush=True)
    print("[DONE]", flush=True)
main()

"""
Paper-9 stage-6 grid on Paderborn (PU) real-damage bearings (pre-specified in prereg_stage6_pu.md before any run).
Dataset: pu_dataset.PUDatasetBinary, train {KA04, KA16, KI04, KI14} @ N15_M07_F10 -> test {KA15, KA22, KI17,
KI18, KI21} @ N09_M07_F10; fingerprints asserted against the prereg. Noise, z-scoring, sampler, optimiser,
schedule and evaluation are the harness's own (imported unmodified); the per-epoch record is as in stages 1-5.
--smoke runs the pre-specified ceiling/floor check (s4d, bm3_frozen; seed 99; clean, 0, -6 dB), excluded from analysis.
A cumulative-GPU-time fuse (default 12 h, smoke + grid) stops the run.
"""
import sys, json, time, argparse, importlib.util
from pathlib import Path
SRC = __import__("os").environ.get("P9_HARNESS_DIR") or str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root holds xjtu_noisy_harness.py
sys.path.insert(0, SRC)
spec = importlib.util.spec_from_file_location("h", f"{SRC}/xjtu_noisy_harness.py")
h = importlib.util.module_from_spec(spec); sys.modules["h"] = h; spec.loader.exec_module(h)
import numpy as np, torch, torch.nn.functional as F
import layered_freeze as lf
import graft_control as gc
import additive_native as an
from bearmamba3.kinematic_loss import kinematic_loss

KW_BASE = dict(d_model=64, d_state=128, n_layers=4, n_sensors=1, n_classes=h.N_CLASSES,
               conv_stride=2, use_batchnorm=False, dtype=torch.bfloat16)


def build(arm, device):
    if arm in lf.ARMS:
        return lf.build_arm(arm, device, **KW_BASE)
    if arm in gc.ARMS:
        return gc.build_arm(arm, device, n_classes=h.N_CLASSES)
    if arm in an.ARMS:
        return an.build_arm(arm, device, **KW_BASE)
    return h.build_model(arm, device)


import pu_dataset as pu
PU = dict(label_sha="53d565d9520a", content_train="3a40737977fb", content_test="54402e68f661")
OUT = "p9_recheck/results/cells_stage6_pu.jsonl"
SMOKE_OUT = "p9_recheck/results/smoke6_pu.jsonl"
ARMS6 = "bm3_frozen,frozen_clti,s4d,frozen_nogate,bm3_frozen_add"
FUSE_FILES = [OUT, SMOKE_OUT]
def gpu_hours_used():
    t = 0.0
    for f in FUSE_FILES:
        if Path(f).exists():
            for l in Path(f).read_text().splitlines():
                try: t += json.loads(l)["elapsed_s"]
                except Exception: pass
    return t / 3600


def train_cell_hist(arm, train_ds, test_ds, seed, n_epochs, device):
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
    is_bm3 = (arm == "bm3_kin")          # harness train_cell: only bm3_kin carries L_kin
    hist, losses = [], []
    for epoch in range(1, n_epochs + 1):
        model.train(); ls = []
        for x, labels, rpm in train_loader:
            x = x.to(device, dtype=pdt); labels = labels.to(device); rpm = rpm.to(device)
            if is_bm3:
                out, kin = model(x, return_kin=True)
                loss = F.cross_entropy(out.float(), labels) + h.LAMBDA_KIN * kinematic_loss(
                    kin, rpm, h.FS_EFF, variant="cover", bearing_kwargs=h.BEARING_KW)
            else:
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
    return dict(hist=hist.tolist(), train_loss=losses, oracle=float(hist.max()),
                oracle_epoch=int(hist.argmax()) + 1, final=float(hist[-1]),
                last5=float(hist[-5:].mean()),
                n_params=sum(p.numel() for p in model.parameters()),
                elapsed_s=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="pre-specified ceiling/floor smoke (seed 99, excluded)")
    ap.add_argument("--arms", default=None)
    ap.add_argument("--snrs", default="0,-2,-6")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--fuse-hours", type=float, default=12.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.smoke:
        arms, seeds, conds = ["s4d", "bm3_frozen"], [99], [("clean", None), ("awgn", 0.0), ("awgn", -6.0)]
        out = Path(a.out or SMOKE_OUT)
    else:
        arms = (a.arms or ARMS6).split(","); seeds = [int(s) for s in a.seeds.split(",")]
        conds = [("awgn", float(x)) for x in a.snrs.split(",")]
        out = Path(a.out or OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            try:
                c = json.loads(l); done.add((c["condition"], c["arm"], c["seed"]))
            except Exception:
                pass
    dev = torch.device("cuda")
    free, total = torch.cuda.mem_get_info()
    print(f"[INIT] PU {pu.TRAIN_COND} -> {pu.TEST_COND}  GPU free {free/2**20:.0f}/{total/2**20:.0f} MiB  "
          f"stage-6 GPU-h so far {gpu_hours_used():.2f}", flush=True)
    base_tr = pu.PUDatasetBinary(pu.TRAIN_BEARINGS, pu.TRAIN_COND, cache_dir="p9_recheck/pu_cache")
    base_te = pu.PUDatasetBinary(pu.TEST_BEARINGS, pu.TEST_COND, cache_dir="p9_recheck/pu_cache")
    assert not set(base_tr.bearings) & set(base_te.bearings)
    eval_sha = h.eval_fingerprint(h.XJTUDatasetNoisy(base_te, "clean", None, 0))
    ct, ce = pu.content_fingerprint(base_tr), pu.content_fingerprint(base_te)
    print("[INIT] eval_sha", eval_sha[:12], "content", ct[:12], ce[:12], flush=True)
    assert eval_sha.startswith(PU["label_sha"]) and ct.startswith(PU["content_train"]) and ce.startswith(PU["content_test"])
    total = len(conds) * len(arms) * len(seeds); n = len(done)
    with open(out, "a") as f:
        for seed in seeds:
            for nt, snr in conds:
                label = "clean" if nt == "clean" else f"awgn@{snr:+.0f}dB"
                tr = h.XJTUDatasetNoisy(base_tr, nt, snr, seed)
                te = h.XJTUDatasetNoisy(base_te, nt, snr, seed)
                for arm in arms:
                    if (label, arm, seed) in done: continue
                    if gpu_hours_used() >= a.fuse_hours:
                        print(f"[FUSE] stage-6 GPU time {gpu_hours_used():.2f} h >= {a.fuse_hours} h; stopping", flush=True)
                        return
                    n += 1
                    print(f"[RUN] {label} {arm} seed={seed} ({n}/{total})", flush=True)
                    r = train_cell_hist(arm, tr, te, seed, a.epochs, dev)
                    rec = dict(dataset="PU", train_cond=pu.TRAIN_COND, test_cond=pu.TEST_COND, condition=label,
                               noise_type=nt, snr_db=snr, arm=arm, seed=seed, epochs=a.epochs, smoke=a.smoke,
                               eval_sha256=eval_sha, content_train=ct, content_test=ce, **r)
                    f.write(json.dumps(rec) + "\n"); f.flush()
                    print(f"      oracle={r['oracle']*100:.2f} (ep{r['oracle_epoch']}) "
                          f"final={r['final']*100:.2f} {r['elapsed_s']:.0f}s", flush=True)
    print("[DONE]", flush=True)


main()

"""
Paper-9 stage-9 grid: XJTU-SY reverse transition (40Hz10kN -> 37.5Hz11kN) under INDEPENDENT train/evaluation
noise (pre-specified in prereg_stage9_reverse_indep.md before any run). Same split, preprocessing and bearing
roles as the stage-3 reverse grid; noise stream keyed on [seed, split, idx] (indep_noise.py, as in stage 8);
harness imported unmodified. --smoke runs the pre-specified smoke check (excluded from analysis).
Fuse (cumulative stage-9 GPU time, smoke + grid) is fixed in the pre-specification.
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
import clti_gate as cg2
import indep_noise as ino
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
    if arm in cg2.ARMS:
        return cg2.build_arm(arm, device, **KW_BASE)
    return h.build_model(arm, device)


import pu_dataset as pu
IND = ino.make(h)
GRIDS = {
    "D": dict(data="xjtu_rev", arms=None, sha="934248343b29",
              out="p9_recheck/results/cells_stage9_D_xjtu_rev.jsonl"),
}
SMOKE_OUT = "p9_recheck/results/smoke9.jsonl"
FUSE_FILES = [g["out"] for g in GRIDS.values()] + [SMOKE_OUT]


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


def noise_bank_sha(te):
    import hashlib
    hh = hashlib.sha256()
    for i in range(0, len(te), max(1, len(te) // 64)):
        hh.update(te[i][0].numpy().tobytes())
    return hh.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="D", choices=list(GRIDS))
    ap.add_argument("--arms", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--snrs", default="0,-2,-6")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--fuse-hours", type=float, required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    G = GRIDS[a.grid]; arms = a.arms.split(","); out = Path(a.out or (SMOKE_OUT if a.smoke else G["out"]))
    if a.smoke:
        a.seeds, a.snrs = "99", "0,-6"
    seeds = [int(x) for x in a.seeds.split(",")]; snrs = [float(x) for x in a.snrs.split(",")]
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            try: c = json.loads(l); done.add((c.get("grid"), c["condition"], c["arm"], c["seed"]))
            except Exception: pass
    dev = torch.device("cuda")
    free, tot = torch.cuda.mem_get_info()
    if G["data"] == "xjtu_rev":
        trb, teb = h.make_cross_condition_split("40Hz10kN", "37.5Hz11kN")
        base_tr = h.XJTUDataset(h.DATA_ROOT, trb, n_sensors=1); base_te = h.XJTUDataset(h.DATA_ROOT, teb, n_sensors=1)
        assert not set(trb) & set(teb)
    else:
        raise ValueError(G["data"])
    eval_sha = h.eval_fingerprint(h.XJTUDatasetNoisy(base_te, "clean", None, 0))
    print(f"[INIT] grid {a.grid} ({G['data']})  GPU free {free/2**20:.0f}/{tot/2**20:.0f} MiB  "
          f"stage-9 GPU-h so far {gpu_hours_used():.2f}  eval_sha {eval_sha[:12]}", flush=True)
    assert eval_sha.startswith(G["sha"]), eval_sha
    total = len(snrs) * len(arms) * len(seeds); n = len(done)
    with open(out, "a") as f:
        for seed in seeds:
            for snr in snrs:
                label = f"awgn@{snr:+.0f}dB"
                tr = IND(base_tr, "awgn", snr, seed, split="train")
                te = IND(base_te, "awgn", snr, seed, split="test")
                bank = noise_bank_sha(te)
                for arm in arms:
                    if (a.grid, label, arm, seed) in done: continue
                    if gpu_hours_used() >= a.fuse_hours:
                        print(f"[FUSE] stage-9 GPU time {gpu_hours_used():.2f} h >= {a.fuse_hours} h; stopping", flush=True)
                        return
                    n += 1
                    print(f"[RUN] {a.grid} {label} {arm} seed={seed} ({n}/{total})", flush=True)
                    r = train_cell_hist(arm, tr, te, seed, a.epochs, dev)
                    rec = dict(stage=9, grid=a.grid, smoke=a.smoke, data=G["data"], noise_protocol="independent[seed,split,idx]",
                               condition=label, noise_type="awgn", snr_db=snr, arm=arm, seed=seed, epochs=a.epochs,
                               eval_sha256=eval_sha, eval_noise_bank_sha256=bank, **r)
                    f.write(json.dumps(rec) + "\n"); f.flush()
                    print(f"      oracle={r['oracle']*100:.2f} (ep{r['oracle_epoch']}) final={r['final']*100:.2f} {r['elapsed_s']:.0f}s", flush=True)
    print("[DONE]", flush=True)


main()

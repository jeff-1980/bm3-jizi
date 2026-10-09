"""
Paper-9 stage-8 grids under INDEPENDENT train/evaluation noise (pre-specified in prereg_stage8_indep_noise.md
before any run). Noise stream keyed on [seed, split, idx] (indep_noise.py); harness imported unmodified.
  --grid A  XJTU-SY 37.5Hz11kN -> 40Hz10kN; bm3_frozen, s4d, frozen_nogate, frozen_clti, bm3_frozen_add
  --grid C  XJTU-SY same transition; frozen_clti_nogate, frozen_clti_add (same-host gate tests)
  --grid B  PU shared bearings {KA04, KA16, KI04, KI14} N15_M07_F10 -> N09_M07_F10; bm3_frozen, s4d, frozen_nogate
Fuse: cumulative GPU time over all stage-8 files > fuse (default 15 h) stops the run.
"""
import sys, json, time, argparse, importlib.util
from pathlib import Path
SRC = "/home/jeffwork/exp/bm3-defense/xjtu_noisy_defense_20260627"
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
    "A": dict(data="xjtu", arms="bm3_frozen,s4d,frozen_nogate,frozen_clti,bm3_frozen_add", sha="6c20b367522c",
              out="p9_recheck/results/cells_stage8_A_xjtu.jsonl"),
    "C": dict(data="xjtu", arms="frozen_clti_nogate,frozen_clti_add", sha="6c20b367522c",
              out="p9_recheck/results/cells_stage8_C_xjtu_clti.jsonl"),
    "B": dict(data="pu_shared", arms="bm3_frozen,s4d,frozen_nogate", sha="addc1938057e",
              out="p9_recheck/results/cells_stage8_B_pu_shared.jsonl"),
}
FUSE_FILES = [g["out"] for g in GRIDS.values()] + ["p9_recheck/results/smoke8.jsonl"]


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
    ap.add_argument("--grid", required=True, choices=list(GRIDS))
    ap.add_argument("--snrs", default="0,-2,-6")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--fuse-hours", type=float, default=25.0)   # raised from 15 h by the author on 2026-10-09 after launch (GPU shared with an external job)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    G = GRIDS[a.grid]; arms = G["arms"].split(","); out = Path(a.out or G["out"])
    seeds = [int(s) for s in a.seeds.split(",")]; snrs = [float(x) for x in a.snrs.split(",")]
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            try: c = json.loads(l); done.add((c.get("grid"), c["condition"], c["arm"], c["seed"]))
            except Exception: pass
    dev = torch.device("cuda")
    free, tot = torch.cuda.mem_get_info()
    if G["data"] == "xjtu":
        trb, teb = h.make_cross_condition_split("37.5Hz11kN", "40Hz10kN")
        base_tr = h.XJTUDataset(h.DATA_ROOT, trb, n_sensors=1); base_te = h.XJTUDataset(h.DATA_ROOT, teb, n_sensors=1)
        assert not set(trb) & set(teb)
    else:
        base_tr = pu.PUDatasetBinary(pu.TRAIN_BEARINGS, pu.TRAIN_COND, cache_dir="p9_recheck/pu_cache")
        base_te = pu.PUDatasetBinary(pu.TRAIN_BEARINGS, pu.TEST_COND, cache_dir="p9_recheck/pu_cache")
    eval_sha = h.eval_fingerprint(h.XJTUDatasetNoisy(base_te, "clean", None, 0))
    print(f"[INIT] grid {a.grid} ({G['data']})  GPU free {free/2**20:.0f}/{tot/2**20:.0f} MiB  "
          f"stage-8 GPU-h so far {gpu_hours_used():.2f}  eval_sha {eval_sha[:12]}", flush=True)
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
                        print(f"[FUSE] stage-8 GPU time {gpu_hours_used():.2f} h >= {a.fuse_hours} h; stopping", flush=True)
                        return
                    n += 1
                    print(f"[RUN] {a.grid} {label} {arm} seed={seed} ({n}/{total})", flush=True)
                    r = train_cell_hist(arm, tr, te, seed, a.epochs, dev)
                    rec = dict(stage=8, grid=a.grid, data=G["data"], noise_protocol="independent[seed,split,idx]",
                               condition=label, noise_type="awgn", snr_db=snr, arm=arm, seed=seed, epochs=a.epochs,
                               eval_sha256=eval_sha, eval_noise_bank_sha256=bank, **r)
                    f.write(json.dumps(rec) + "\n"); f.flush()
                    print(f"      oracle={r['oracle']*100:.2f} (ep{r['oracle_epoch']}) final={r['final']*100:.2f} {r['elapsed_s']:.0f}s", flush=True)
    print("[DONE]", flush=True)


main()

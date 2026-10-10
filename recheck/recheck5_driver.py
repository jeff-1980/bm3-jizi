"""
Paper-9 stage-5 grid (pre-specified in prereg_stage5_metric_reverse.md before any run).

  --pair 1  experiment 2: frozen_noconv, frozen_As4d, s4d_wide on the original transition
            (37.5Hz11kN -> 40Hz10kN); eval fingerprint must equal 6c20b367522c.
  --pair 2  experiment 3: bm3_frozen_add on the reverse transition (40Hz10kN -> 37.5Hz11kN);
            eval fingerprint must equal 934248343b29 (stage 3).

Training loop, sampler, optimiser, schedule, noise and evaluation are the harness's own
(imported unmodified); only the per-epoch evaluation record is extended, as in stages 1-2.
A cumulative-GPU-time fuse (default 12 h, summed over every stage-5 results file) stops the run.
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


PAIRS = {1: dict(train="37.5Hz11kN", test="40Hz10kN", sha="6c20b367522c",
                 arms="frozen_noconv,frozen_As4d,s4d_wide",
                 out="p9_recheck/results/cells_stage5_exp2.jsonl"),
         2: dict(train="40Hz10kN", test="37.5Hz11kN", sha=None,   # filled from the prereg below
                 arms="bm3_frozen_add",
                 out="p9_recheck/results/cells_stage5_exp3.jsonl")}
FUSE_FILES = ["p9_recheck/results/cells_stage5_exp2.jsonl", "p9_recheck/results/cells_stage5_exp3.jsonl"]


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
    ap.add_argument("--pair", type=int, required=True, choices=[1, 2])
    ap.add_argument("--pair2-sha", default=None, help="eval fingerprint frozen in the prereg")
    ap.add_argument("--arms", default=None)
    ap.add_argument("--snrs", default="0,-2,-6")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--fuse-hours", type=float, default=12.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    P = dict(PAIRS[a.pair])
    if a.pair == 2:
        assert a.pair2_sha, "pair 2 needs the pre-specified fingerprint"
        P["sha"] = a.pair2_sha
    arms = (a.arms or P["arms"]).split(","); snrs = [float(s) for s in a.snrs.split(",")]
    seeds = [int(s) for s in a.seeds.split(",")]
    out = Path(a.out or P["out"]); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            try:
                c = json.loads(l); done.add((c["condition"], c["arm"], c["seed"]))
            except Exception:
                pass
    dev = torch.device("cuda")
    free, total = torch.cuda.mem_get_info()
    print(f"[INIT] pair {a.pair} {P['train']} -> {P['test']}  GPU free {free/2**20:.0f}/{total/2**20:.0f} MiB  "
          f"stage-5 GPU-h so far {gpu_hours_used():.2f}", flush=True)
    trb, teb = h.make_cross_condition_split(P["train"], P["test"])
    base_tr = h.XJTUDataset(h.DATA_ROOT, trb, n_sensors=1)
    base_te = h.XJTUDataset(h.DATA_ROOT, teb, n_sensors=1)
    eval_sha = h.eval_fingerprint(h.XJTUDatasetNoisy(base_te, "clean", None, 0))
    print("[INIT] eval_sha", eval_sha[:12], flush=True)
    assert eval_sha.startswith(P["sha"]), f"eval fingerprint {eval_sha[:12]} differs from the pre-specified {P['sha']}"
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
                    if gpu_hours_used() >= a.fuse_hours:
                        print(f"[FUSE] stage-5 GPU time {gpu_hours_used():.2f} h >= {a.fuse_hours} h; stopping", flush=True)
                        return
                    n += 1
                    print(f"[RUN] {cond['label']} {arm} seed={seed} ({n}/{total})", flush=True)
                    r = train_cell_hist(arm, tr, te, seed, a.epochs, dev)
                    rec = dict(pair=a.pair, train_cond=P["train"], test_cond=P["test"], condition=cond["label"], noise_type=cond["noise_type"],
                               snr_db=cond["snr_db"], arm=arm, seed=seed, epochs=a.epochs,
                               eval_sha256=eval_sha, **r)
                    f.write(json.dumps(rec) + "\n"); f.flush()
                    print(f"      oracle={r['oracle']*100:.2f} (ep{r['oracle_epoch']}) "
                          f"final={r['final']*100:.2f} last5={r['last5']*100:.2f} "
                          f"{r['elapsed_s']:.0f}s", flush=True)
    print("[DONE]", flush=True)


main()

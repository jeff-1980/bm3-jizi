"""
Paper-9 stage-13 (TIMC revision): unfrozen Mamba-3 block under the final protocol.

Stages 8 and 9 compared frozen variants of the block with S4D under independent training and evaluation noise but
did not include the unfrozen block, so the cost of freezing was not measured under the final protocol. This driver
runs the unfrozen block trained with cross-entropy only (harness arm bm3_nokin: same stem, depth, width, state size
and optimiser as bm3_frozen; no kinematic loss) on both XJTU-SY transitions, 0/-2/-6 dB, seeds 0-4, 50 epochs,
final-epoch reporting, with the stage-8/9 code path (build, sampler, optimiser, independent noise keyed on
[seed, split, idx]; train_cell_hist copied verbatim from recheck9_driver.py). It also saves the final-epoch
prediction of every evaluation window with its bearing. Pre-specification: prereg_stage13_unfrozen.md.
Arms other than bm3_nokin are allowed (--arms) only for the smoke check.
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



IND = ino.make(h)
DIRS = {
    "fwd": dict(src="37.5Hz11kN", dst="40Hz10kN", sha="6c20b367522c"),
    "rev": dict(src="40Hz10kN", dst="37.5Hz11kN", sha="934248343b29"),
}
OUT = "p9_recheck/results/cells_stage13_unfrozen.jsonl"
PRED = "p9_recheck/results/preds_stage13"


def bearing_of_window(test_bearings):
    ids = []
    for b in test_bearings:  # XJTUDataset concatenates windows bearing by bearing, in list order
        ids += [b] * len(h.XJTUDataset(h.DATA_ROOT, [b], n_sensors=1))
    return np.array(ids)


_LAST = {}
_orig_eval = h.eval_macro_f1


def _eval_keep(model, loader, device, **kw):
    # same metric as the harness; additionally keeps the predictions of the latest call
    acc, mf1, yt, yp, extra = _orig_eval(model, loader, device, collect_predictions=True)
    _LAST["yt"], _LAST["yp"] = np.asarray(yt), np.asarray(yp)
    return acc, mf1


h.eval_macro_f1 = _eval_keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="fwd,rev")
    ap.add_argument("--arms", default="bm3_nokin")
    ap.add_argument("--snrs", default="0,-2,-6")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--pred", default=PRED)
    ap.add_argument("--fuse-hours", type=float, default=5.0)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]; snrs = [float(x) for x in a.snrs.split(",")]
    arms = a.arms.split(",")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); Path(a.pred).mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            c = json.loads(l); done.add((c["direction"], c["condition"], c["arm"], c["seed"]))
    dev = torch.device("cuda")
    with open(out, "a") as f:
        for dn in a.dirs.split(","):
            D = DIRS[dn]
            trb, teb = h.make_cross_condition_split(D["src"], D["dst"])
            base_tr = h.XJTUDataset(h.DATA_ROOT, trb, n_sensors=1); base_te = h.XJTUDataset(h.DATA_ROOT, teb, n_sensors=1)
            assert not set(trb) & set(teb)
            bear = bearing_of_window(teb)
            assert len(bear) == len(base_te._labels)
            eval_sha = h.eval_fingerprint(h.XJTUDatasetNoisy(base_te, "clean", None, 0))
            print(f"[INIT] {dn} test bearings {list(teb)} windows {len(bear)} eval_sha {eval_sha[:12]}", flush=True)
            assert eval_sha.startswith(D["sha"]), eval_sha
            for seed in seeds:
                for snr in snrs:
                    label = f"awgn@{snr:+.0f}dB"
                    tr = IND(base_tr, "awgn", snr, seed, split="train")
                    te = IND(base_te, "awgn", snr, seed, split="test")
                    bank = noise_bank_sha(te)
                    for arm in arms:
                        if (dn, label, arm, seed) in done: continue
                        used = sum(json.loads(l)["elapsed_s"] for l in out.read_text().splitlines()) / 3600 if out.exists() else 0.0
                        if used >= a.fuse_hours:
                            print(f"[FUSE] stage-13 GPU time {used:.2f} h >= {a.fuse_hours} h; stopping", flush=True); return
                        print(f"[RUN] {dn} {label} {arm} seed={seed}", flush=True)
                        r = train_cell_hist(arm, tr, te, seed, a.epochs, dev)
                        yt, yp = _LAST["yt"], _LAST["yp"]
                        assert np.array_equal(yt, np.asarray(base_te._labels))
                        per_bearing = {b: dict(label=int(yt[bear == b][0]), n=int((bear == b).sum()),
                                               recall=float((yp[bear == b] == yt[bear == b]).mean())) for b in teb}
                        np.savez_compressed(Path(a.pred) / f"{dn}_{arm}_snr{snr:+.0f}_seed{seed}.npz",
                                            y_true=yt, y_pred=yp, bearing=bear)
                        rec = dict(stage=13, direction=dn, data="xjtu" if dn == "fwd" else "xjtu_rev",
                                   noise_protocol="independent[seed,split,idx]", condition=label, noise_type="awgn",
                                   snr_db=snr, arm=arm, seed=seed, epochs=a.epochs, eval_sha256=eval_sha,
                                   eval_noise_bank_sha256=bank, per_bearing=per_bearing, **r)
                        f.write(json.dumps(rec) + "\n"); f.flush()
                        print(f"      final={r['final']*100:.2f} " + " ".join(f"{b}:{v['recall']:.2f}" for b, v in per_bearing.items())
                              + f" {r['elapsed_s']:.0f}s", flush=True)
    print("[DONE]", flush=True)


main()

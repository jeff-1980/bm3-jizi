"""
Construction checks for stage 8 (clti_gate.py arms and indep_noise.py). No training.
Writes a read-only JSON snapshot; exits non-zero on any failure.

Arms (frozen_clti_nogate, frozen_clti_add) against frozen_clti at the same seed:
  1. param_match     add == frozen_clti; nogate == frozen_clti - n_layers * d_inner * d_model (z rows).
  2. init_identical  add: state_dict bitwise equal to frozen_clti. nogate: all keys but in_proj bitwise
                     equal, and in_proj equal to frozen_clti's in_proj rows [d_inner:] (x rows).
  3. combine_path    kernel called with Z=None; tensor entering out_proj == y + silu(z) (add, and
                     differs from y * silu(z)) or == y (nogate); z input-dependent (add).
  4. coeffs_const    Q, K, ADT, DT, Trap, Angles captured at the kernel for two different inputs are
                     bitwise equal (scan dynamics and read/write coefficients input-independent; V is not).
  5. fwd_bwd         finite output; every parameter receives a finite gradient.
Noise wrapper (synthetic windows):
  6. noise_indep     train/test noise at the same (seed, idx): |corr| < 0.05 (harness wrapper: corr ~ 1);
                     the evaluation noise is identical for two wrapper instances with the same seed
                     (common bank) and differs across seeds; clean path unchanged vs harness.
"""
import json, os, sys, time, importlib.util
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from einops import rearrange
SRC = __import__("os").environ.get("P9_HARNESS_DIR") or str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root holds xjtu_noisy_harness.py
sys.path.insert(0, SRC); sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))  # recheck/ (robust to PYTHONSAFEPATH)
import harness_loader
h = harness_loader.load(SRC)
import layered_freeze as lf, clti_gate as cg, indep_noise as ino
KW = dict(d_model=64, d_state=128, n_layers=4, n_sensors=1, n_classes=h.N_CLASSES,
          conv_stride=2, use_batchnorm=False, dtype=torch.bfloat16)
dev = torch.device("cuda"); C = {}

def build(arm, seed=0):
    torch.manual_seed(seed)
    return lf.build_arm("frozen_clti", dev, **KW) if arm == "frozen_clti" else cg.build_arm(arm, dev, **KW)

ref = build("frozen_clti"); L0 = ref.mamba_layers[0]; di, dm = L0.d_inner, L0.d_model; nl = len(ref.mamba_layers)
npar = lambda m: sum(p.numel() for p in m.parameters())
for arm in cg.ARMS:
    m = build(arm); sr, sm = ref.state_dict(), m.state_dict()
    exp_n = npar(ref) if arm == "frozen_clti_add" else npar(ref) - nl * di * dm
    C[f"{arm}:param_match"] = dict(ref=npar(ref), arm=npar(m), expected=exp_n, **{"pass": npar(m) == exp_n})
    if arm == "frozen_clti_add":
        ok = set(sr) == set(sm) and all(torch.equal(sr[k], sm[k]) for k in sr)
    else:
        ks = [k for k in sr if not k.endswith("in_proj.weight")]
        ok = set(sm) == set(sr) and all(torch.equal(sr[k], sm[k]) for k in ks) and \
             all(torch.equal(sm[k], sr[k][di:]) for k in sr if k.endswith("in_proj.weight"))
    C[f"{arm}:init_identical"] = {"pass": bool(ok)}
    # combine path + coefficient capture
    m.eval(); orig = cg.mamba3_siso_combined; cap = []
    def spy(**kw):
        y = orig(**kw); cap.append({k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in kw.items()} | {"_y": y.detach()}); return y
    cg.mamba3_siso_combined = spy
    zs, gs = {}, {}; hooks = []
    for i, L in enumerate(m.mamba_layers):
        hooks.append(L.in_proj.register_forward_hook(lambda mod, a_, o, i=i: zs.__setitem__(i, o.detach().float())))
        hooks.append(L.out_proj.register_forward_pre_hook(lambda mod, a_, i=i: gs.__setitem__(i, a_[0].detach().float())))
    runs = []
    for t in range(2):
        cap.clear(); zs.clear(); gs.clear(); torch.manual_seed(100 + t)
        with torch.no_grad(): m(torch.randn(2, 1, 2048, device=dev))
        runs.append((list(cap), dict(zs), dict(gs)))
    for hk in hooks: hk.remove()
    cg.mamba3_siso_combined = orig
    cap0, zs0, gs0 = runs[0]
    own, other = [], []
    for i in range(nl):
        y = rearrange(cap0[i]["_y"], "b l h p -> b l (h p)").float()
        if arm == "frozen_clti_add":
            z = zs0[i][..., :di]
            own.append(torch.allclose(gs0[i], (y + F.silu(z)).to(torch.bfloat16).float(), atol=2e-2, rtol=2e-2))
            other.append(not torch.allclose(gs0[i], (y * F.silu(z)).to(torch.bfloat16).float(), atol=2e-2, rtol=2e-2))
        else:
            own.append(torch.allclose(gs0[i], y.to(torch.bfloat16).float(), atol=1e-6)); other.append(zs0[i].shape[-1] == di)
    znone = all(c["Z"] is None for c in cap0)
    C[f"{arm}:combine_path"] = dict(kernel_Z_None=znone, out_proj_input_matches=all(own), contrast_ok=all(other),
                                    **{"pass": znone and all(own) and all(other)})
    keys = ["Q", "K", "ADT", "DT", "Trap", "Angles"]
    const = {k: all(torch.equal(runs[0][0][i][k], runs[1][0][i][k]) for i in range(nl)) for k in keys}
    vdep = all(not torch.equal(runs[0][0][i]["V"], runs[1][0][i]["V"]) for i in range(nl))
    C[f"{arm}:coeffs_const"] = dict(**const, V_input_dependent=vdep, **{"pass": all(const.values()) and vdep})
    m.train(); o = m(torch.randn(4, 1, 2048, device=dev)); o.float().sum().backward()
    gr = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in m.parameters())
    C[f"{arm}:fwd_bwd"] = dict(shape=list(o.shape), **{"pass": list(o.shape) == [4, h.N_CLASSES] and bool(torch.isfinite(o).all()) and gr})

# 6. noise wrapper on synthetic windows
class Syn:
    def __init__(self, n, seed):
        r = np.random.default_rng(seed); self._windows = [r.standard_normal(2048).astype(np.float32) for _ in range(n)]
        self._labels = [i % 2 for i in range(n)]; self._rpms = [0.0] * n
A, Bd = Syn(64, 1), Syn(64, 2); IN = ino.make(h)
def pert(ds, i): x = ds._windows[i]; z = (x - x.mean()) / x.std(); return ds[i][0].numpy().ravel() - z
co_old = np.mean([np.corrcoef(pert(h.XJTUDatasetNoisy(A, "awgn", -6.0, 3), i), pert(h.XJTUDatasetNoisy(Bd, "awgn", -6.0, 3), i))[0, 1] for i in range(64)])
tr, te = IN(A, "awgn", -6.0, 3, split="train"), IN(Bd, "awgn", -6.0, 3, split="test")
co_new = np.mean([abs(np.corrcoef(pert(tr, i), pert(te, i))[0, 1]) for i in range(64)])
te2 = IN(Bd, "awgn", -6.0, 3, split="test"); te_s4 = IN(Bd, "awgn", -6.0, 4, split="test")
bank = all(torch.equal(te[i][0], te2[i][0]) for i in range(64)); seeddiff = not torch.equal(te[0][0], te_s4[0][0])
clean = all(torch.equal(IN(A, "clean", None, 3, split="train")[i][0], h.XJTUDatasetNoisy(A, "clean", None, 3)[i][0]) for i in range(8))
C["noise_indep"] = dict(harness_mean_corr=float(co_old), new_mean_abs_corr=float(co_new), common_eval_bank=bank,
                        differs_across_seeds=seeddiff, clean_unchanged=clean,
                        **{"pass": co_old > 0.99 and co_new < 0.05 and bank and seeddiff and clean})
out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / "unitcheck_stage8"); out.mkdir(parents=True, exist_ok=True)
rep = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "torch": torch.__version__, "checks": C, "all_pass": all(v["pass"] for v in C.values())}
p = out / "stage8_unitcheck.json"
if p.exists(): os.chmod(p, 0o644)
p.write_text(json.dumps(rep, indent=1, default=float)); os.chmod(p, 0o444)
print(json.dumps({k: v["pass"] for k, v in C.items()})); print("params", {k: v.get("arm") for k, v in C.items() if "param" in k}, "ref", npar(ref))
print("noise corr harness", round(co_old, 6), "new", round(co_new, 4)); print("all_pass:", rep["all_pass"])
sys.exit(0 if rep["all_pass"] else 1)

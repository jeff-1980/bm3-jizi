"""
Construction check for bm3_frozen_add (additive_native.py). No data, no training. Writes a
read-only JSON snapshot; exits non-zero on any failure.

  1. donor_operator   in bm3_frozen, the kernel output with Z=z equals the kernel output with
                      Z=None multiplied by silu(z) (bf16 tolerance): the donor combination is
                      y * silu(z) applied after the D-skip.
  2. param_match      parameter count of bm3_frozen_add equals bm3_frozen (difference 0 %).
  3. init_identical   at the same seed, state_dict of bm3_frozen_add equals bm3_frozen in keys,
                      shapes and values (bitwise).
  4. combine_path     the kernel is called with Z=None; the tensor entering out_proj equals
                      y + silu(z) and differs from y * silu(z); z changes with the input.
  5. fwd_bwd          finite output (batch, n_classes); every parameter receives a finite gradient.
"""
import argparse, json, os, sys, time, importlib.util
from pathlib import Path
import torch
import torch.nn.functional as F
from einops import rearrange

SRC = os.environ.get("P9_HARNESS_DIR") or str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, SRC); sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))  # recheck/ (robust to PYTHONSAFEPATH)
import harness_loader
h = harness_loader.load(SRC)
import additive_native as an
import bm3_frozen as bf

KW = dict(d_model=64, d_state=128, n_layers=4, n_sensors=1, n_classes=h.N_CLASSES,
          conv_stride=2, use_batchnorm=False, dtype=torch.bfloat16)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=None); a = ap.parse_args()
    dev = torch.device("cuda")
    out = Path(a.out or Path(__file__).resolve().parent / f"additive_native_unitcheck_{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    C = {}

    # 1. donor operator, by calling the kernel twice with the frozen layer's own tensors
    torch.manual_seed(0); ref = h.build_model("bm3_frozen", dev).eval()
    grabbed = []
    orig = bf.mamba3_siso_combined
    def spy(**kw):
        grabbed.append(kw); return orig(**kw)
    bf.mamba3_siso_combined = spy
    with torch.no_grad():
        ref(torch.randn(2, 1, 2048, device=dev))
    bf.mamba3_siso_combined = orig
    ok = []
    with torch.no_grad():
        for kw in grabbed:
            yz = orig(**kw).float()
            k2 = dict(kw); k2["Z"] = None
            y0 = orig(**k2).float()
            exp = y0 * F.silu(kw["Z"].float())
            ok.append(torch.allclose(yz, exp, atol=3e-2, rtol=3e-2) and
                      float((yz - exp).abs().max()) < float((yz - y0).abs().max()))
    C["donor_operator"] = {"layers": len(grabbed), "pass": bool(grabbed) and all(ok)}

    # 2-3. parameters and initial weights
    torch.manual_seed(0); base = h.build_model("bm3_frozen", dev)
    torch.manual_seed(0); m = an.build_arm("bm3_frozen_add", dev, **KW)
    nb, na = (sum(p.numel() for p in x.parameters()) for x in (base, m))
    C["param_match"] = {"bm3_frozen": nb, "bm3_frozen_add": na, "rel_diff": abs(na - nb) / nb, "pass": na == nb}
    sb, sa = base.state_dict(), m.state_dict()
    same = set(sb) == set(sa) and all(sb[k].shape == sa[k].shape and torch.equal(sb[k], sa[k]) for k in sb)
    C["init_identical"] = {"n_keys": len(sb), "pass": bool(same)}

    # 4. combination path
    m.eval(); calls = []; rec = []
    orig2 = an.mamba3_siso_combined
    def spy2(**kw):
        y = orig2(**kw); calls.append(kw["Z"] is None); rec.append({"y": rearrange(y, "b l h p -> b l (h p)").detach().float()}); return y
    an.mamba3_siso_combined = spy2
    hooks = []
    for i, L in enumerate(m.mamba_layers):
        hooks.append(L.in_proj.register_forward_hook(      # in_proj emits [z, x, trap, angles]
            lambda mod, a_, o, i=i, d=L.d_inner: zs.__setitem__(i, o[..., :d].detach().float())))
        hooks.append(L.out_proj.register_forward_pre_hook(lambda mod, a_, i=i: gs.__setitem__(i, a_[0].detach().float())))
    res = []
    for trial in range(2):
        zs, gs = {}, {}; calls.clear(); rec.clear()
        with torch.no_grad():
            m(torch.randn(2, 1, 2048, device=dev))
        res.append((dict(zs), dict(gs), list(rec), list(calls)))
    for hk in hooks: hk.remove()
    an.mamba3_siso_combined = orig2
    zs, gs, rec0, calls0 = res[0]
    d = m.mamba_layers[0].d_inner
    own, other = [], []
    for i in range(len(m.mamba_layers)):
        z = zs[i][..., :d]; y = rec0[i]["y"]; g = gs[i]
        own.append(torch.allclose(g, (y + F.silu(z)).to(torch.bfloat16).float(), atol=2e-2, rtol=2e-2))
        other.append(not torch.allclose(g, (y * F.silu(z)).to(torch.bfloat16).float(), atol=2e-2, rtol=2e-2))
    zdep = all(not torch.equal(res[0][0][i], res[1][0][i]) for i in range(len(m.mamba_layers)))
    C["combine_path"] = {"kernel_called_with_Z_None": all(calls0), "matches_y_plus_silu_z": all(own),
                         "differs_from_y_times_silu_z": all(other), "z_input_dependent": zdep,
                         "pass": all(calls0) and all(own) and all(other) and zdep}

    # 5. forward/backward
    m.train(); o = m(torch.randn(4, 1, 2048, device=dev)); o.float().sum().backward()
    gr = {k: p.grad is not None and bool(torch.isfinite(p.grad).all()) for k, p in m.named_parameters()}
    C["fwd_bwd"] = {"out_shape": list(o.shape), "finite": bool(torch.isfinite(o).all()),
                    "all_params_finite_grad": all(gr.values()),
                    "pass": list(o.shape) == [4, h.N_CLASSES] and bool(torch.isfinite(o).all()) and all(gr.values())}

    rep = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "torch": torch.__version__, "checks": C,
           "all_pass": all(v["pass"] for v in C.values())}
    p = out / "additive_native_unitcheck.json"; p.write_text(json.dumps(rep, indent=1)); os.chmod(p, 0o444)
    print(json.dumps({k: v["pass"] for k, v in C.items()}), "params", nb, na)
    print("all_pass:", rep["all_pass"], "->", p)
    sys.exit(0 if rep["all_pass"] else 1)


main()

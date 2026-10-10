"""
Construction check for the stage-3 arms (graft_control.py). All checks are mechanical; writes a
read-only JSON snapshot and exits non-zero on any failure. No data is loaded and nothing is trained.

  1. donor_shape      bm3_frozen's layer has d_inner == graft_control.D_INNER and passes the gate
                      into the scan kernel (is_outproj_norm False), i.e. the donor gate is
                      y * silu(z) at width d_inner.
  2. param_match      s4d_plus_gate_wm and s4d_plus_branch differ by < 2 % in parameter count.
  3. combine_path     per layer, forward hooks capture y (scan output), z (in_z output) and the
                      tensor entering out_proj. Requires out_in == y * silu(z) for the gate arm and
                      out_in == y + silu(z) for the branch arm, and that each does NOT match the
                      other arm's formula. Also requires z to change with the input.
  4. state_dict_diff  against a plain BearS4D built under the same seed: conv_embed, layer_norms,
                      norm and classifier keys are identical in name, shape and value; the only
                      removed keys are the 64-wide S4D layer tensors and the only added keys are
                      in_x / in_z / out weights and the 128-wide S4D layer tensors. The two arms
                      are identical to each other in keys, shapes and values at the same seed.
  5. fwd_bwd          forward is finite with shape (batch, n_classes); every parameter gets a
                      finite gradient.
"""
import argparse, json, os, sys, time, importlib.util
from pathlib import Path
import torch
import torch.nn.functional as F

SRC = os.environ.get("P9_HARNESS_DIR") or str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, SRC)
sys.path.insert(0, str(Path(__file__).resolve().parent))
spec = importlib.util.spec_from_file_location("h", f"{SRC}/xjtu_noisy_harness.py")
h = importlib.util.module_from_spec(spec); sys.modules["h"] = h; spec.loader.exec_module(h)
import graft_control as gc
from models_extended import BearS4D

FORMULA = {"mult": lambda y, z: y * F.silu(z), "add": lambda y, z: y + F.silu(z)}


def build(arm, dev, seed=0):
    torch.manual_seed(seed)
    return gc.build_arm(arm, dev)


def capture(model, x):
    rec = []
    hooks = []
    for i, layer in enumerate(model.s4d_layers):
        slot = {}
        rec.append(slot)
        hooks.append(layer.ssm.register_forward_hook(lambda m, a, o, s=slot: s.__setitem__("y", o.detach().float())))
        hooks.append(layer.in_z.register_forward_hook(lambda m, a, o, s=slot: s.__setitem__("z", o.detach().float())))
        hooks.append(layer.out.register_forward_pre_hook(lambda m, a, s=slot: s.__setitem__("g", a[0].detach().float())))
    with torch.no_grad():
        model(x)
    for hk in hooks:
        hk.remove()
    return rec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=None); a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out or Path(__file__).resolve().parent / f"graft_control_unitcheck_{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    rep = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "torch": torch.__version__, "checks": {}}
    C = rep["checks"]

    L = h.build_model("bm3_frozen", dev).mamba_layers[0]
    C["donor_shape"] = {"d_inner": int(L.d_inner), "is_outproj_norm": bool(L.is_outproj_norm),
                        "pass": int(L.d_inner) == gc.D_INNER and not bool(L.is_outproj_norm)}

    n = {arm: sum(p.numel() for p in build(arm, dev).parameters()) for arm in gc.ARMS}
    ref = {arm: sum(p.numel() for p in h.build_model(arm, dev).parameters())
           for arm in ("s4d", "bm3_frozen", "s4d_plus_gate", "s4d_wide")}
    rel = abs(n["s4d_plus_gate_wm"] - n["s4d_plus_branch"]) / n["s4d_plus_gate_wm"]
    C["param_match"] = {"params": n, "reference_params": ref, "rel_diff": rel, "pass": rel < 0.02}

    cp = {}
    for arm, mode in gc.ARMS.items():
        m = build(arm, dev).eval()
        x1 = torch.randn(3, 1, 2048, device=dev); x2 = torch.randn(3, 1, 2048, device=dev)
        r1, r2 = capture(m, x1), capture(m, x2)
        other = "add" if mode == "mult" else "mult"
        own = all(torch.allclose(s["g"], FORMULA[mode](s["y"], s["z"]), atol=1e-5, rtol=1e-4) for s in r1)
        not_other = all(not torch.allclose(s["g"], FORMULA[other](s["y"], s["z"]), atol=1e-3, rtol=1e-3) for s in r1)
        z_dep = all(not torch.equal(s1["z"], s2["z"]) for s1, s2 in zip(r1, r2))
        cp[arm] = {"mode": mode, "matches_own_formula": own, "differs_from_other_formula": not_other,
                   "z_input_dependent": z_dep, "layers": len(r1), "pass": own and not_other and z_dep}
    C["combine_path"] = {**cp, "pass": all(v["pass"] for v in cp.values())}

    torch.manual_seed(0); base = BearS4D(d_model=64, d_state=128, n_layers=4, n_sensors=1,
                                         n_classes=h.N_CLASSES, conv_stride=2).to(dev)
    bsd = base.state_dict()
    sd = {arm: build(arm, dev).state_dict() for arm in gc.ARMS}
    w = sd["s4d_plus_gate_wm"]
    shared = sorted(set(bsd) & set(w))
    removed = sorted(set(bsd) - set(w)); added = sorted(set(w) - set(bsd))
    host_ok = all(k.split(".")[0] in ("conv_embed", "layer_norms", "norm", "classifier") for k in shared) and \
        all(bsd[k].shape == w[k].shape and torch.equal(bsd[k], w[k]) for k in shared)
    removed_ok = all(k.startswith("s4d_layers.") and (".kernel_fn." in k or k.endswith(".D")) and ".ssm." not in k for k in removed)
    added_ok = all(k.startswith("s4d_layers.") and any(t in k for t in (".in_x.weight", ".in_z.weight", ".out.weight", ".ssm.")) for k in added)
    b = sd["s4d_plus_branch"]
    twins = set(w) == set(b) and all(w[k].shape == b[k].shape and torch.equal(w[k], b[k]) for k in w)
    C["state_dict_diff"] = {"n_shared_host_keys": len(shared), "removed": removed, "added": added,
                            "host_keys_identical": host_ok, "removed_only_s4d64": removed_ok,
                            "added_only_expected": added_ok, "arms_identical_at_init": twins,
                            "pass": host_ok and removed_ok and added_ok and twins and len(shared) > 0}

    fb = {}
    for arm in gc.ARMS:
        m = build(arm, dev).train()
        o = m(torch.randn(4, 1, 2048, device=dev))
        o.float().sum().backward()
        grads = {k: (p.grad is not None and bool(torch.isfinite(p.grad).all())) for k, p in m.named_parameters()}
        fb[arm] = {"out_shape": list(o.shape), "finite": bool(torch.isfinite(o).all()),
                   "all_params_have_finite_grad": all(grads.values()),
                   "pass": list(o.shape) == [4, h.N_CLASSES] and bool(torch.isfinite(o).all()) and all(grads.values())}
    C["fwd_bwd"] = {**fb, "pass": all(v["pass"] for v in fb.values())}

    rep["all_pass"] = all(v["pass"] for v in C.values())
    p = out / "graft_control_unitcheck.json"
    p.write_text(json.dumps(rep, indent=1)); os.chmod(p, 0o444)
    print(json.dumps({k: v["pass"] for k, v in C.items()}), "params", n, "rel_diff", rel)
    print("all_pass:", rep["all_pass"], "->", p)
    sys.exit(0 if rep["all_pass"] else 1)


main()

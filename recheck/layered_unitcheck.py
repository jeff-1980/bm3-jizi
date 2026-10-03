"""
Single-variable discipline check for the stage-2 layered-freeze arms (layered_freeze.py).

Four checks per arm, all decided mechanically; writes a JSON snapshot and exits non-zero
on any failure.

  1. forward_backward   forward runs, output shape is (batch, n_classes), no NaN, and every
                        newly introduced constant parameter receives a gradient.
  2. input_independence  the decisive check. The call into mamba3_siso_combined is wrapped and
                        the Trap / Angles tensors it receives are captured for two different
                        random inputs through the SAME module. A tensor that the arm claims to
                        have frozen must be bitwise identical across the two inputs; a tensor
                        the arm leaves selective must differ.
  3. state_dict_diff     key-level diff against bm3_frozen: the only permitted differences are
                        the shrunk in_proj slice and the added trap_const / angle_const keys.
                        Every other key must match in name and shape.
  4. param_table         parameter counts and the delta against bm3_frozen.
"""
import argparse, json, os, sys, time, importlib.util
from pathlib import Path
import torch

SRC = os.environ.get("P9_HARNESS_DIR", str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, SRC)
sys.path.insert(0, str(Path(__file__).resolve().parent))
spec = importlib.util.spec_from_file_location("h", f"{SRC}/xjtu_noisy_harness.py")
h = importlib.util.module_from_spec(spec); sys.modules["h"] = h; spec.loader.exec_module(h)
import layered_freeze as lf

KW = dict(d_model=64, d_state=128, n_layers=4, n_sensors=1, n_classes=2, conv_stride=2,
          use_batchnorm=False, dtype=torch.bfloat16)
EXPECT = {"frozen_ctrap":  dict(trap="frozen", angles="selective"),
          "frozen_cangle": dict(trap="selective", angles="frozen"),
          "frozen_clti":   dict(trap="frozen", angles="frozen")}


def capture(model, dev, n=2):
    """Trap/Angles tensors passed to the scan, for n independent random inputs."""
    orig = lf.mamba3_siso_combined
    grabbed = []
    def spy(**kw):
        grabbed.append({k: kw[k].detach().float().cpu().clone() for k in ("Trap", "Angles")})
        return orig(**kw)
    lf.mamba3_siso_combined = spy
    runs = []
    try:
        with torch.no_grad():
            for i in range(n):
                grabbed.clear()
                torch.manual_seed(1000 + i)
                model(torch.randn(2, 1, 2048, device=dev))
                runs.append([dict(g) for g in grabbed])
    finally:
        lf.mamba3_siso_combined = orig
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or f"recheck/layered_unitcheck_{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    base = h.build_model("bm3_frozen", dev)
    base_sd = {k: tuple(v.shape) for k, v in base.state_dict().items()}
    n_base = sum(p.numel() for p in base.parameters())
    rep = {"base_arm": "bm3_frozen", "base_params": n_base, "torch": torch.__version__, "arms": {}}

    for arm, expect in EXPECT.items():
        torch.manual_seed(0)
        m = lf.build_arm(arm, dev, **KW)
        r = {"expected": expect}

        x = torch.randn(4, 1, 2048, device=dev)
        o = m(x); o = o[0] if isinstance(o, tuple) else o
        o.float().sum().backward()
        consts = {k: (p.grad is not None) for k, p in m.named_parameters()
                  if k.split(".")[-1] in ("trap_const", "angle_const", "A_const", "B_const", "C_const")}
        r["forward_backward"] = {
            "out_shape": list(o.shape), "finite": bool(torch.isfinite(o).all()),
            "n_const_params": len(consts), "all_const_params_have_grad": all(consts.values()),
            "pass": list(o.shape) == [4, 2] and bool(torch.isfinite(o).all()) and all(consts.values()),
        }

        torch.manual_seed(0)
        mp = lf.build_arm(arm, dev, **KW)
        runs = capture(mp, dev)
        obs, ok = {}, True
        for key in ("Trap", "Angles"):
            same = all(torch.equal(l[key], r2[key]) for l, r2 in zip(runs[0], runs[1]))
            name = "trap" if key == "Trap" else "angles"
            obs[name] = "frozen" if same else "selective"
            ok &= (obs[name] == expect[name])
        r["input_independence"] = {"observed": obs, "n_layers_probed": len(runs[0]), "pass": ok}

        sd = {k: tuple(v.shape) for k, v in m.state_dict().items()}
        added = sorted(set(sd) - set(base_sd)); removed = sorted(set(base_sd) - set(sd))
        reshaped = sorted(k for k in set(sd) & set(base_sd) if sd[k] != base_sd[k])
        allowed_added = all(k.split(".")[-1] in ("trap_const", "angle_const") for k in added)
        allowed_reshaped = all(k.endswith("in_proj.weight") for k in reshaped)
        r["state_dict_diff"] = {
            "added": added, "removed": removed, "reshaped": reshaped,
            "in_proj_shape": [sd[k] for k in reshaped][:1],
            "pass": allowed_added and allowed_reshaped and not removed,
        }

        n = sum(p.numel() for p in m.parameters())
        r["param_table"] = {"params": n, "delta_vs_bm3_frozen": n - n_base}
        r["pass"] = all(r[k]["pass"] for k in
                        ("forward_backward", "input_independence", "state_dict_diff"))
        rep["arms"][arm] = r

    rep["all_pass"] = all(v["pass"] for v in rep["arms"].values())
    p = out / "layered_unitcheck.json"
    p.write_text(json.dumps(rep, indent=1)); os.chmod(p, 0o444)
    print(json.dumps({k: {"pass": v["pass"], "observed": v["input_independence"]["observed"],
                          "params": v["param_table"]} for k, v in rep["arms"].items()}, indent=1))
    print("all_pass:", rep["all_pass"], "->", p)
    sys.exit(0 if rep["all_pass"] else 1)


main()

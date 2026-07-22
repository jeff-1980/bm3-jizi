"""
frozensel_unitcheck.py — Unit validation for FrozenSelectivityMamba3 /
BearMamba3Frozen, run on THIS directory's harness/model wiring (not the CWRU
directory's artifacts — models are built via xjtu_noisy_harness.build_model(),
the exact code path xjtu_noisy_harness.py itself uses to train/eval cells).

Checks (machine-readable pass/fail, all_pass true only if all pass):
  1. forward_pass        — forward pass succeeds, output shape correct, no NaN
  2. selectivity_removed — in_proj no longer emits B/C/dd_dt/dd_A (shrunk by
                            exactly the selective slice sizes) and the
                            replacement constants (A_const/B_const/C_const)
                            exist on the layer
  3. backward_grads      — after backward, A_const/B_const/C_const/dt_bias
                            grads are all not None
  4. state_dict_diff      — key-by-key state_dict diff, bm3_frozen's Mamba
                            layer vs bm3_nokin's (built via this dir's
                            build_model('bm3_kin', ...) — bm3_kin/bm3_nokin
                            share the same BearMamba3 architecture, only the
                            training loss path differs, see
                            xjtu_noisy_harness.py:348). Every diff key is
                            mapped to its arch_map.md justification; every
                            non-selective key must be byte-identical in
                            shape; the total param delta must equal the
                            arch_map.md §4 arithmetic exactly.
  5. param_table          — per-arm parameter count for every arm in
                            build_model() (bm3_kin, bm3_nokin, bm3_frozen,
                            cnn1d, tcn, cnnlstm, s4d, s4d_wide)

GUARDRAILS: writes only to a new timestamped output dir; chmod 444 on the
JSON; does not touch any existing results/ file; does not modify
xjtu_noisy_harness.py beyond the additive bm3_frozen wiring already made
(see report).
"""
import datetime
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import xjtu_noisy_harness as H
from mamba_ssm.modules.mamba3 import Mamba3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Keys expected to differ between a selective Mamba3 layer (bm3_kin/bm3_nokin)
# and its FrozenSelectivityMamba3 counterpart (bm3_frozen) — the selective
# components themselves. Mapped 1:1 to arch_map.md §2/§3 reasoning. Any other
# key must match name+shape exactly, or check 4 fails.
SELECTIVE_ALLOWED_DIFF_KEYS = {
    "in_proj.weight": "arch_map.md §2 — in_proj rebuilt smaller: [z,x,trap,angle] only, B/C/dd_dt/dd_A slices removed",
    "A_const":        "arch_map.md §2 — new constant replacing dd_A (Mamba3's A has no static analogue to inherit)",
    "B_const":        "arch_map.md §2 — new constant replacing the removed B in_proj slice",
    "C_const":        "arch_map.md §2 — new constant replacing the removed C in_proj slice",
}


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def state_shapes(m):
    return {k: tuple(v.shape) for k, v in m.state_dict().items()}


def main():
    torch.manual_seed(0)
    result = {"device": str(device), "checks": {}}

    # ── 1. forward pass: shape + NaN, model built via THIS dir's build_model ──
    m = H.build_model("bm3_frozen", device)
    batch, seqlen = 4, 2048
    x = torch.randn(batch, 1, seqlen, device=device)  # n_sensors=1, matches kw_base
    logits = m(x)
    expected_shape = (batch, H.N_CLASSES)
    fwd_ok = (tuple(logits.shape) == expected_shape) and not torch.isnan(logits).any().item()
    result["checks"]["forward_pass"] = {
        "built_via": "xjtu_noisy_harness.build_model('bm3_frozen', device)",
        "output_shape": list(logits.shape),
        "expected_shape": list(expected_shape),
        "shape_ok": tuple(logits.shape) == expected_shape,
        "has_nan": bool(torch.isnan(logits).any().item()),
        "passed": fwd_ok,
    }
    assert fwd_ok, f"FAIL forward_pass: {result['checks']['forward_pass']}"

    # ── 2. selectivity machinery physically removed ──────────────────────────
    layer0 = m.mamba_layers[0]
    ref_layer = Mamba3(d_model=64, d_state=128, is_mimo=False, mimo_rank=4,
                       rope_fraction=0.5, chunk_size=64, dtype=torch.bfloat16).to(device)
    expected_selective_width = (
        2 * ref_layer.d_state * ref_layer.num_bc_heads * ref_layer.mimo_rank  # B, C
        + 2 * ref_layer.nheads                                                # dd_dt, dd_A
    )
    actual_shrink = ref_layer.in_proj.weight.shape[0] - layer0.in_proj.weight.shape[0]
    in_proj_shrunk_correctly = actual_shrink == expected_selective_width
    has_constants = all(hasattr(layer0, n) for n in ("A_const", "B_const", "C_const"))
    no_selective_proj = in_proj_shrunk_correctly and has_constants
    result["checks"]["selectivity_removed"] = {
        "ref_unfrozen_in_proj_out_features": int(ref_layer.in_proj.weight.shape[0]),
        "frozen_in_proj_out_features": int(layer0.in_proj.weight.shape[0]),
        "expected_selective_slice_width(B+C+dd_dt+dd_A)": int(expected_selective_width),
        "actual_shrink": int(actual_shrink),
        "in_proj_shrunk_correctly": in_proj_shrunk_correctly,
        "has_A_const_B_const_C_const": has_constants,
        "passed": no_selective_proj,
    }
    assert no_selective_proj, f"FAIL selectivity_removed: {result['checks']['selectivity_removed']}"

    # ── 3. backward: constants are learning ───────────────────────────────────
    loss = logits.float().sum()
    loss.backward()
    grads_ok = {
        "A_const.grad is not None": layer0.A_const.grad is not None,
        "B_const.grad is not None": layer0.B_const.grad is not None,
        "C_const.grad is not None": layer0.C_const.grad is not None,
        "dt_bias.grad is not None": layer0.dt_bias.grad is not None,
    }
    result["checks"]["backward_grads"] = {**grads_ok, "passed": all(grads_ok.values())}

    # ── 4. state_dict diff, bm3_frozen vs bm3_nokin (both built via build_model) ──
    bm3_nokin = H.build_model("bm3_kin", device)  # bm3_kin == bm3_nokin architecture (loss path only differs)
    nokin_layer_shapes = state_shapes(bm3_nokin.mamba_layers[0])
    frozen_layer_shapes = state_shapes(layer0)
    shared_keys = set(nokin_layer_shapes) & set(frozen_layer_shapes)
    unexpected_missing = set(nokin_layer_shapes) - set(frozen_layer_shapes) - set(SELECTIVE_ALLOWED_DIFF_KEYS)
    unexpected_added = set(frozen_layer_shapes) - set(nokin_layer_shapes) - set(SELECTIVE_ALLOWED_DIFF_KEYS)
    shared_mismatched_shape = {
        k: {"bm3_nokin": nokin_layer_shapes[k], "bm3_frozen": frozen_layer_shapes[k]}
        for k in shared_keys - set(SELECTIVE_ALLOWED_DIFF_KEYS)
        if nokin_layer_shapes[k] != frozen_layer_shapes[k]
    }
    non_selective_keys = shared_keys - set(SELECTIVE_ALLOWED_DIFF_KEYS)
    non_selective_zero_diff = not shared_mismatched_shape
    diff_is_selectivity_only = (
        not unexpected_missing and not unexpected_added and non_selective_zero_diff
    )

    per_key_diff = []
    for k in sorted(set(nokin_layer_shapes) | set(frozen_layer_shapes)):
        in_nokin = k in nokin_layer_shapes
        in_frozen = k in frozen_layer_shapes
        if in_nokin and in_frozen and nokin_layer_shapes[k] == frozen_layer_shapes[k]:
            status = "identical"
        elif k in SELECTIVE_ALLOWED_DIFF_KEYS:
            status = "selective_diff (expected)"
        else:
            status = "UNEXPECTED_DIFF"
        per_key_diff.append({
            "key": k,
            "bm3_nokin_shape": list(nokin_layer_shapes.get(k, ())) if in_nokin else None,
            "bm3_frozen_shape": list(frozen_layer_shapes.get(k, ())) if in_frozen else None,
            "status": status,
            "reason": SELECTIVE_ALLOWED_DIFF_KEYS.get(k, "arch_map.md §3 S4 — non-selective, must match" if status == "identical" else None),
        })

    # Full-model param delta (not just one layer) — must equal the arch_map.md
    # §4 arithmetic: -260*d_model + 258, per layer, x n_layers=4 = -65528.
    n_bm3_frozen_full = count_params(m)
    n_bm3_nokin_full = count_params(bm3_nokin)
    full_delta = n_bm3_frozen_full - n_bm3_nokin_full
    expected_full_delta = 4 * ((-(128 + 128 + 2 + 2) * 64) + (2 + 128 + 128))  # arch_map.md §4
    delta_explained = full_delta == expected_full_delta
    frozen_has_fewer_params = n_bm3_frozen_full < n_bm3_nokin_full

    arch_diff_passed = diff_is_selectivity_only and frozen_has_fewer_params and delta_explained
    result["checks"]["state_dict_diff"] = {
        "description": (
            "bm3_frozen (FrozenSelectivityMamba3) vs bm3_nokin (Mamba3, both "
            "built via THIS dir's xjtu_noisy_harness.build_model()) must be "
            "the SAME layer family: every state_dict key outside "
            "SELECTIVE_ALLOWED_DIFF_KEYS matches name+shape exactly (zero "
            "diff on non-selective keys), and the full-model param delta "
            "equals the arch_map.md §4 arithmetic exactly."
        ),
        "selective_allowed_diff_keys": SELECTIVE_ALLOWED_DIFF_KEYS,
        "per_layer_key_diff": per_key_diff,
        "non_selective_keys_checked": sorted(non_selective_keys),
        "non_selective_zero_diff": non_selective_zero_diff,
        "unexpected_missing_keys": sorted(unexpected_missing),
        "unexpected_added_keys": sorted(unexpected_added),
        "full_model_param_delta": full_delta,
        "expected_param_delta(arch_map_section4)": expected_full_delta,
        "delta_explained": delta_explained,
        "frozen_has_fewer_params": frozen_has_fewer_params,
        "passed": arch_diff_passed,
    }
    assert arch_diff_passed, f"FAIL state_dict_diff: non_selective_zero_diff={non_selective_zero_diff} delta_explained={delta_explained} frozen_has_fewer_params={frozen_has_fewer_params}"

    # ── 5. per-arm parameter table (every arm this dir's build_model knows) ──
    arms = ["bm3_kin", "bm3_frozen", "cnn1d", "tcn", "cnnlstm", "s4d", "s4d_wide"]
    param_table = {}
    for arm in arms:
        mm = H.build_model(arm, device)
        param_table[arm] = count_params(mm)
    param_table["bm3_nokin (== bm3_kin architecture)"] = param_table["bm3_kin"]
    result["checks"]["param_table"] = {"params": param_table, "passed": True}

    all_passed = (
        result["checks"]["forward_pass"]["passed"]
        and result["checks"]["selectivity_removed"]["passed"]
        and result["checks"]["backward_grads"]["passed"]
        and result["checks"]["state_dict_diff"]["passed"]
        and result["checks"]["param_table"]["passed"]
    )
    result["ALL_PASS"] = all_passed

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = HERE / f"frozensel_unitcheck_{ts}"
    out_dir.mkdir(parents=True, exist_ok=False)
    out_path = out_dir / "frozensel_unitcheck.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    os.chmod(out_path, 0o444)
    print(f"[written] {out_path}  (chmod 444)")
    print(json.dumps({k: v for k, v in result.items() if k != "checks"} |
                      {"checks_summary": {k: v["passed"] for k, v in result["checks"].items()}}, indent=2))
    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()

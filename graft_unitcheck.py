"""
graft_unitcheck.py — unit validation for the constructive graft arm
(s4d_plus_gate, bm3_models.py/arch_map.md §6), built via THIS directory's
xjtu_noisy_harness.build_model() — the exact code path the harness itself
uses to train/eval cells.

NEW artifact — does NOT reuse or overwrite secondary_unitcheck.py/.json.
The reference baseline here is `s4d` (not `bm3_frozen`, unlike
secondary_unitcheck.py), because s4d_plus_gate is a graft ONTO s4d, not an
ablation of bm3_frozen — see arch_map.md §6.

Checks (machine-readable pass/fail, all_pass true only if all pass):
  1. forward_pass    — forward succeeds, output shape correct, no NaN
  2. backward_grads  — after backward, every parameter's grad is not None
  3. state_dict_diff — FULL-MODEL key-by-key state_dict diff vs s4d: every
                        diff key must be a gate_proj.* key (the ONLY new
                        keys this graft introduces); every OTHER key must be
                        name+shape identical AND tensor-value identical
                        (max_abs_diff == 0, float64) — i.e. every S4D
                        original key is byte-for-byte equal, not just
                        same-shaped (arch_map.md §6.4).
  4. param_table     — s4d vs s4d_plus_gate param counts, cross-checked
                        against arch_map.md §6.5's hand arithmetic
                        (100,162 + 16,640 = 116,802).

GUARDRAILS: writes only to a new timestamped output dir (graft_unitcheck_TS/,
distinct from secondary_unitcheck_TS/); chmod 444 on the JSON; does not touch
any existing results/ file, secondary_unitcheck.py/.json, or bm3_kin/s4d's
existing implementation; does not run any full-grid training (unit checks
build models and run single forward/backward calls only, no train loop, no
epoch loop).
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_LAYERS = 4  # matches kw_base in xjtu_noisy_harness.build_model()
ARM = "s4d_plus_gate"
BASE_ARM = "s4d"

# Every key allowed to differ from `s4d`: gate_proj's weight/bias, one per
# layer — the ONLY new state BearS4DPlusGate introduces (arch_map.md §6.4).
ARM_ALLOWED_DIFFS = {
    f"gate_proj.{i}.weight": "arch_map.md §6.3 — new per-layer nn.Linear(d_model,d_model): z's source projection, applied to the same ln(h) that also feeds _S4DLayer"
    for i in range(N_LAYERS)
}
ARM_ALLOWED_DIFFS.update({
    f"gate_proj.{i}.bias": "arch_map.md §6.3 — new per-layer nn.Linear(d_model,d_model) bias"
    for i in range(N_LAYERS)
})

# Expected full-model param delta vs s4d (100,162), from arch_map.md §6.5:
# one nn.Linear(64,64,bias=True) per layer = (64*64+64) = 4160/layer x 4.
EXPECTED_PARAM_DELTA = N_LAYERS * (64 * 64 + 64)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def _tensor_max_abs_diff(a, b):
    """max |a-b| computed in float64 (lossless upcast for bf16/fp32/fp16),
    or an exact torch.equal for non-floating tensors. Returns
    (max_abs_diff_float_or_None, values_equal_bool).
    """
    a, b = a.detach(), b.detach()
    if torch.is_floating_point(a) and torch.is_floating_point(b):
        diff = float((a.double() - b.double()).abs().max().item())
        return diff, diff == 0.0
    equal = torch.equal(a, b)
    return (0.0 if equal else None), equal


def check_forward_backward(arm, model):
    batch, seqlen = 4, 2048
    x = torch.randn(batch, 1, seqlen, device=device)
    logits = model(x)
    expected_shape = (batch, H.N_CLASSES)
    shape_ok = tuple(logits.shape) == expected_shape
    has_nan = bool(torch.isnan(logits).any().item())
    fwd_ok = shape_ok and not has_nan
    fwd_result = {
        "built_via": f"xjtu_noisy_harness.build_model({arm!r}, device)",
        "output_shape": list(logits.shape),
        "expected_shape": list(expected_shape),
        "shape_ok": shape_ok,
        "has_nan": has_nan,
        "passed": fwd_ok,
    }
    assert fwd_ok, f"FAIL forward_pass[{arm}]: {fwd_result}"

    loss = logits.float().sum()
    loss.backward()
    none_grads = [n for n, p in model.named_parameters() if p.grad is None]
    grads_ok = len(none_grads) == 0
    bwd_result = {
        "n_params_checked": sum(1 for _ in model.named_parameters()),
        "params_with_none_grad": none_grads,
        "passed": grads_ok,
    }
    assert grads_ok, f"FAIL backward_grads[{arm}]: {bwd_result}"
    return fwd_result, bwd_result


def check_state_dict_diff(arm, arm_model, base_model):
    """Full-model state_dict diff vs s4d, checking both shape AND tensor
    VALUE for every key present (with matching shape) in both models.
    Shape-only agreement is not sufficient: it would not catch e.g.
    gate_proj's construction accidentally consuming RNG draws BEFORE the
    shared s4d submodules were built (which would silently perturb their
    values while leaving names/shapes untouched) — see arch_map.md §6.4 for
    why BearS4DPlusGate's __init__ order avoids exactly that.
    """
    allowed = ARM_ALLOWED_DIFFS
    base_sd = base_model.state_dict()
    arm_sd = arm_model.state_dict()
    all_keys = sorted(set(base_sd) | set(arm_sd))

    unexpected_missing = []     # in base (s4d), not in arm, not allowed
    unexpected_added = []       # in arm, not in base, not allowed
    unexpected_shape_diff = []  # in both, shapes differ, not allowed
    unexpected_value_diff = []  # in both, same shape, VALUES differ, not allowed
    per_key = []
    for k in all_keys:
        in_base = k in base_sd
        in_arm = k in arm_sd
        base_shape = tuple(base_sd[k].shape) if in_base else None
        arm_shape = tuple(arm_sd[k].shape) if in_arm else None
        max_abs_diff = None

        if in_base and in_arm and base_shape == arm_shape:
            max_abs_diff, values_equal = _tensor_max_abs_diff(base_sd[k], arm_sd[k])
            if values_equal:
                status = "identical"
                reason = None
            elif k in allowed:
                status = "allowed_diff (expected, name+shape match but value differs)"
                reason = allowed[k]
            else:
                status = "UNEXPECTED_VALUE_DIFF"
                reason = None
                unexpected_value_diff.append(k)
        elif k in allowed:
            status = "allowed_diff (expected — new gate_proj key, not present in s4d)"
            reason = allowed[k]
        elif in_base and not in_arm:
            status = "UNEXPECTED_MISSING"
            reason = None
            unexpected_missing.append(k)
        elif in_arm and not in_base:
            status = "UNEXPECTED_ADDED"
            reason = None
            unexpected_added.append(k)
        else:
            status = "UNEXPECTED_SHAPE_DIFF"
            reason = None
            unexpected_shape_diff.append(k)
        per_key.append({
            "key": k,
            f"{BASE_ARM}_shape": list(base_shape) if base_shape else None,
            f"{arm}_shape": list(arm_shape) if arm_shape else None,
            "max_abs_diff": max_abs_diff,
            "status": status,
            "reason": reason,
        })

    diff_is_gate_only = (
        not unexpected_missing and not unexpected_added
        and not unexpected_shape_diff and not unexpected_value_diff
    )
    n_allowed_diff_keys_seen = sum(1 for k in allowed if k in base_sd or k in arm_sd)
    all_allowed_keys_present = n_allowed_diff_keys_seen == len(allowed)
    # Every non-allowed shared key must be values-identical (S4D original
    # keys byte-for-byte equal, arch_map.md §6.4's core claim).
    shared_nonallowed_keys = [k for k in all_keys if k not in allowed and k in base_sd and k in arm_sd]
    all_shared_nonallowed_identical = all(
        pk["status"] == "identical"
        for pk in per_key
        if pk["key"] in shared_nonallowed_keys
    )

    passed = diff_is_gate_only and all_allowed_keys_present and all_shared_nonallowed_identical
    result = {
        "description": (
            f"{BASE_ARM} vs {arm} (both built via THIS dir's "
            "xjtu_noisy_harness.build_model()): every state_dict key outside "
            "ARM_ALLOWED_DIFFS (gate_proj.*) must match name+shape AND "
            "tensor value exactly (max_abs_diff == 0, float64) — proof that "
            "the graft ADDS a gate branch without touching any S4D original "
            "key (arch_map.md §6.4)."
        ),
        "allowed_diff_keys": allowed,
        "all_allowed_keys_accounted_for": all_allowed_keys_present,
        "all_shared_nonallowed_keys_byte_identical": all_shared_nonallowed_identical,
        "per_key_diff": per_key,
        "unexpected_missing_keys": unexpected_missing,
        "unexpected_added_keys": unexpected_added,
        "unexpected_shape_diff_keys": unexpected_shape_diff,
        "unexpected_value_diff_keys": unexpected_value_diff,
        "diff_is_gate_only": diff_is_gate_only,
        "passed": passed,
    }
    assert passed, (
        f"FAIL state_dict_diff[{arm}]: unexpected_missing={unexpected_missing} "
        f"unexpected_added={unexpected_added} unexpected_shape_diff={unexpected_shape_diff} "
        f"unexpected_value_diff={unexpected_value_diff} "
        f"all_allowed_keys_accounted_for={all_allowed_keys_present} "
        f"all_shared_nonallowed_keys_byte_identical={all_shared_nonallowed_identical}"
    )
    return result


def main():
    torch.manual_seed(0)
    result = {"device": str(device), "arm": ARM, "base_arm": BASE_ARM, "checks": {}}

    torch.manual_seed(0)
    base_model = H.build_model(BASE_ARM, device)
    n_base = count_params(base_model)

    print(f"[CHECK] arm={ARM} vs base={BASE_ARM}")
    torch.manual_seed(0)
    model = H.build_model(ARM, device)

    fwd_result, bwd_result = check_forward_backward(ARM, model)
    diff_result = check_state_dict_diff(ARM, model, base_model)

    n_arm = count_params(model)
    full_delta = n_arm - n_base
    delta_explained = full_delta == EXPECTED_PARAM_DELTA

    result["checks"] = {
        "forward_pass": fwd_result,
        "backward_grads": bwd_result,
        "state_dict_diff": diff_result,
    }
    result["n_params"] = {BASE_ARM: n_base, ARM: n_arm, "delta": full_delta}
    result["expected_param_delta(arch_map_section6_5)"] = EXPECTED_PARAM_DELTA
    result["delta_explained"] = delta_explained
    assert delta_explained, (
        f"FAIL param_delta[{ARM}]: actual={full_delta} expected={EXPECTED_PARAM_DELTA}"
    )

    # ── shared: per-arm parameter table (every arm this dir's build_model knows) ──
    all_arms = ["bm3_kin", "bm3_frozen", "frozen_noconv", "frozen_nogate",
                "frozen_As4d", "cnn1d", "tcn", "cnnlstm", "s4d", "s4d_wide",
                "s4d_plus_gate"]
    param_table = {}
    for arm in all_arms:
        mm = H.build_model(arm, device)
        param_table[arm] = count_params(mm)
    param_table["bm3_nokin (== bm3_kin architecture)"] = param_table["bm3_kin"]
    result["param_table"] = {
        "params": param_table,
        "s4d_expected": 100162,
        "s4d_plus_gate_expected": 116802,
        "s4d_matches_expected": param_table["s4d"] == 100162,
        "s4d_plus_gate_matches_expected": param_table["s4d_plus_gate"] == 116802,
    }
    result["param_table"]["passed"] = (
        result["param_table"]["s4d_matches_expected"]
        and result["param_table"]["s4d_plus_gate_matches_expected"]
    )
    assert result["param_table"]["passed"], (
        f"FAIL param_table: s4d={param_table['s4d']} (expect 100162), "
        f"s4d_plus_gate={param_table['s4d_plus_gate']} (expect 116802)"
    )

    result["ALL_PASS"] = (
        fwd_result["passed"] and bwd_result["passed"]
        and diff_result["passed"] and delta_explained
        and result["param_table"]["passed"]
    )
    print(f"  ALL_PASS={result['ALL_PASS']}  n_params[{ARM}]={n_arm}  "
          f"delta_vs_{BASE_ARM}={full_delta}")

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = HERE / f"graft_unitcheck_{ts}"
    out_dir.mkdir(parents=True, exist_ok=False)
    out_path = out_dir / "graft_unitcheck.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    os.chmod(out_path, 0o444)
    print(f"[written] {out_path}  (chmod 444)")
    print(json.dumps({
        "ALL_PASS": result["ALL_PASS"],
        "param_table": param_table,
    }, indent=2))
    if not result["ALL_PASS"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

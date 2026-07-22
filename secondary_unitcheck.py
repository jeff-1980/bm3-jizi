"""
secondary_unitcheck.py — unit validation for the three secondary
single-variable ablations of bm3_frozen (frozen_noconv/frozen_nogate/
frozen_As4d, bm3_models.py), built via THIS directory's
xjtu_noisy_harness.build_model() — the exact code path the harness itself
uses to train/eval cells.

Per arm, checks (machine-readable pass/fail, all_pass true only if all pass):
  1. forward_pass    — forward succeeds, output shape correct, no NaN
  2. backward_grads  — after backward, every parameter's grad is not None
  3. state_dict_diff — FULL-MODEL key-by-key state_dict diff vs bm3_frozen:
                        every diff key (missing/added/shape-changed/VALUE-
                        changed) is mapped to its arch_map.md §5
                        justification; every OTHER key must be name+shape
                        identical AND tensor-value identical (max_abs_diff
                        == 0, computed in float64 — single-variable proof
                        that only the targeted part differs, not just that
                        its name/shape looks untouched)
Plus one shared check:
  4. param_table — per-arm parameter count for every arm build_model() knows
                   (original arms + the three new ones), cross-checked
                   against arch_map.md §5d's hand arithmetic.

GUARDRAILS: writes only to a new timestamped output dir; chmod 444 on the
JSON; does not touch any existing results/ file or bm3_kin/bm3_frozen/s4d's
existing implementation; does not run any full-grid training (unit checks
build models and run single forward/backward calls only, no train loop).
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

# Per-arm: which full-model state_dict keys are EXPECTED to differ from
# bm3_frozen, and why — mapped 1:1 to arch_map.md §5a/§5b/§5c. Any other key
# must match name+shape exactly, or state_dict_diff fails for that arm.
ARM_ALLOWED_DIFFS = {
    "frozen_noconv": {
        "conv_embed.weight": "arch_map.md §5a — Conv1d stem removed entirely, replaced by StridedLinearStem (no nn.Conv1d/F.conv1d anywhere in forward path); this Conv1d key no longer exists",
        "conv_embed.bias":   "arch_map.md §5a — Conv1d stem removed entirely, replaced by StridedLinearStem; this Conv1d key no longer exists",
        "conv_embed.proj.weight": "arch_map.md §5a — StridedLinearStem's nn.Linear channel-lift weight: shape (64,1,7)->(64,1), mathematically the same per-timestep map as the removed kernel_size=1 Conv1d, built without any Conv1d call",
        "conv_embed.proj.bias":   "arch_map.md §5a — StridedLinearStem's nn.Linear bias (same shape (64,), new object)",
    },
    "frozen_nogate": {
        f"mamba_layers.{i}.in_proj.weight": "arch_map.md §5b — in_proj rebuilt smaller: z's d_inner-wide slice dropped, [x,trap,angle] only"
        for i in range(N_LAYERS)
    },
    "frozen_As4d": {
        **{f"mamba_layers.{i}.A_const": "arch_map.md §5c — A_const removed, replaced by A_log (S4D-style reparameterization)"
           for i in range(N_LAYERS)},
        **{f"mamba_layers.{i}.A_log": "arch_map.md §5c — new parameter: A_head = -exp(A_log) replaces -heavy_tail_activation(A_const)"
           for i in range(N_LAYERS)},
    },
}

# Expected full-model param delta vs bm3_frozen (112,410), from arch_map.md §5d.
EXPECTED_PARAM_DELTA = {
    "frozen_noconv": (64 + 64) - (448 + 64),          # conv_embed weight+bias delta
    "frozen_nogate": N_LAYERS * (162 - 290) * 64,     # in_proj out_features shrink x d_model, x4 layers
    "frozen_As4d":   N_LAYERS * (2 - 2),              # A_const(2,) -> A_log(2,): zero delta
}


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def _tensor_max_abs_diff(a, b):
    """max |a-b| computed in float64 (lossless upcast for bf16/fp32/fp16),
    or an exact torch.equal for non-floating tensors (e.g. integer buffers).
    Returns (max_abs_diff_float_or_None, values_equal_bool).
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
    """Full-model state_dict diff vs bm3_frozen, checking both shape AND
    tensor VALUE for every key present (with matching shape) in both
    models. Shape-only agreement is not sufficient evidence of a
    single-variable ablation — two independently-initialized layers can
    have identical name+shape while holding completely different numbers
    (this is exactly what silently broke frozen_nogate/frozen_As4d prior to
    the bm3_models.py fix that made mamba_layers replacement copy
    non-target parameters verbatim instead of re-randomizing them).
    """
    allowed = ARM_ALLOWED_DIFFS[arm]
    base_sd = base_model.state_dict()
    arm_sd = arm_model.state_dict()
    all_keys = sorted(set(base_sd) | set(arm_sd))

    unexpected_missing = []     # in base, not in arm, not allowed
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
            status = "allowed_diff (expected)"
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
            "bm3_frozen_shape": list(base_shape) if base_shape else None,
            f"{arm}_shape": list(arm_shape) if arm_shape else None,
            "max_abs_diff": max_abs_diff,
            "status": status,
            "reason": reason,
        })

    diff_is_single_variable = (
        not unexpected_missing and not unexpected_added
        and not unexpected_shape_diff and not unexpected_value_diff
    )
    n_allowed_diff_keys_seen = sum(1 for k in allowed if k in base_sd or k in arm_sd)
    all_allowed_keys_present = n_allowed_diff_keys_seen == len(allowed)

    passed = diff_is_single_variable and all_allowed_keys_present
    result = {
        "description": (
            f"bm3_frozen vs {arm} (both built via THIS dir's "
            "xjtu_noisy_harness.build_model()): every state_dict key outside "
            "ARM_ALLOWED_DIFFS[arm] must match name+shape AND tensor value "
            "exactly (max_abs_diff == 0, float64) — single-variable proof "
            "that only the documented part (arch_map.md §5) differs, not "
            "just that its name/shape looks untouched."
        ),
        "allowed_diff_keys": allowed,
        "all_allowed_keys_accounted_for": all_allowed_keys_present,
        "per_key_diff": per_key,
        "unexpected_missing_keys": unexpected_missing,
        "unexpected_added_keys": unexpected_added,
        "unexpected_shape_diff_keys": unexpected_shape_diff,
        "unexpected_value_diff_keys": unexpected_value_diff,
        "diff_is_single_variable": diff_is_single_variable,
        "passed": passed,
    }
    assert passed, (
        f"FAIL state_dict_diff[{arm}]: unexpected_missing={unexpected_missing} "
        f"unexpected_added={unexpected_added} unexpected_shape_diff={unexpected_shape_diff} "
        f"unexpected_value_diff={unexpected_value_diff} "
        f"all_allowed_keys_accounted_for={all_allowed_keys_present}"
    )
    return result


def main():
    torch.manual_seed(0)
    result = {"device": str(device), "arms": {}}

    bm3_frozen_model = H.build_model("bm3_frozen", device)
    n_bm3_frozen = count_params(bm3_frozen_model)

    secondary_arms = ["frozen_noconv", "frozen_nogate", "frozen_As4d"]

    for arm in secondary_arms:
        print(f"[CHECK] arm={arm}")
        torch.manual_seed(0)
        model = H.build_model(arm, device)

        fwd_result, bwd_result = check_forward_backward(arm, model)
        diff_result = check_state_dict_diff(arm, model, bm3_frozen_model)

        n_arm = count_params(model)
        full_delta = n_arm - n_bm3_frozen
        expected_delta = EXPECTED_PARAM_DELTA[arm]
        delta_explained = full_delta == expected_delta

        result["arms"][arm] = {
            "checks": {
                "forward_pass": fwd_result,
                "backward_grads": bwd_result,
                "state_dict_diff": diff_result,
            },
            "n_params": n_arm,
            "n_params_bm3_frozen": n_bm3_frozen,
            "param_delta_vs_bm3_frozen": full_delta,
            "expected_param_delta(arch_map_section5d)": expected_delta,
            "delta_explained": delta_explained,
        }
        result["arms"][arm]["ALL_PASS"] = (
            fwd_result["passed"] and bwd_result["passed"]
            and diff_result["passed"] and delta_explained
        )
        assert delta_explained, (
            f"FAIL param_delta[{arm}]: actual={full_delta} expected={expected_delta}"
        )
        print(f"  ALL_PASS={result['arms'][arm]['ALL_PASS']}  n_params={n_arm}  "
              f"delta_vs_bm3_frozen={full_delta}")

    # ── shared: per-arm parameter table (every arm this dir's build_model knows) ──
    all_arms = ["bm3_kin", "bm3_frozen", "frozen_noconv", "frozen_nogate",
                "frozen_As4d", "cnn1d", "tcn", "cnnlstm", "s4d", "s4d_wide"]
    param_table = {}
    for arm in all_arms:
        mm = H.build_model(arm, device)
        param_table[arm] = count_params(mm)
    param_table["bm3_nokin (== bm3_kin architecture)"] = param_table["bm3_kin"]
    result["param_table"] = {"params": param_table, "passed": True}

    result["ALL_PASS"] = (
        all(result["arms"][a]["ALL_PASS"] for a in secondary_arms)
        and result["param_table"]["passed"]
    )

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = HERE / f"secondary_unitcheck_{ts}"
    out_dir.mkdir(parents=True, exist_ok=False)
    out_path = out_dir / "secondary_unitcheck.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    os.chmod(out_path, 0o444)
    print(f"[written] {out_path}  (chmod 444)")
    print(json.dumps({
        "ALL_PASS": result["ALL_PASS"],
        "per_arm_ALL_PASS": {a: result["arms"][a]["ALL_PASS"] for a in secondary_arms},
        "param_table": param_table,
    }, indent=2))
    if not result["ALL_PASS"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

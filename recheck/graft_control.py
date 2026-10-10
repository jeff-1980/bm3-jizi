"""
Paper-9 stage-3 arms: a width-matched gate graft and its parameter-identical additive control.

The original graft (bm3_models.BearS4DPlusGate) adds a 64-wide gate, z = Linear(64->64)(ln h),
directly on each S4D layer's 64-wide output. In the donor block the gate is 128-wide: Mamba-3
projects the layer input to x and z of width d_inner = 128 (expand = 2), runs the scan on x,
multiplies by silu(z), and projects back with out_proj (128 -> 64). Removal and addition were
therefore not matched in width or parameters. These arms rebuild the graft in the donor's shape:

  s4d_plus_gate_wm   per layer:  x = W_x ln_h,  z = W_z ln_h          (64 -> 128 each, no bias)
                                 y = S4D_128(x)                        (width 128, d_state 64)
                                 h = h + W_out( y * silu(z) )          (128 -> 64, no bias)
  s4d_plus_branch    identical modules and parameters; the only change is the combination:
                                 h = h + W_out( y + silu(z) )          (additive, non-multiplicative)

The S4D layer at width 128 uses d_state = 64 so that its own parameter count and total state size
(128 x 64 = 8192) equal the host's (64 x 128): the SSM budget is unchanged and only redistributed
across the wider channel axis. The arms are larger than bm3_frozen (they carry Mamba-style in/out
projections on top of an unchanged SSM budget), which biases the test TOWARD recovery, i.e.
against the paper's L3 claim; the additive control shares every parameter, so a gate-specific
reading requires the multiplicative arm to beat it.

conv_embed, layer_norms, norm and classifier are BearS4D's own modules, built by BearS4D.__init__.
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

SRC = __import__("os").environ.get("P9_HARNESS_DIR") or str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root holds xjtu_noisy_harness.py
if SRC not in sys.path:
    sys.path.insert(0, SRC)
from models_extended import BearS4D, _S4DLayer

D_INNER = 128          # = bm3_frozen layer d_inner (asserted in the unit check)
D_STATE_WM = 64        # 128 x 64 = 64 x 128: SSM parameter/state budget equal to the host's


class _WideGatedLayer(nn.Module):
    def __init__(self, d_model, d_inner, d_state, mode):
        super().__init__()
        assert mode in ("mult", "add")
        self.mode = mode
        self.in_x = nn.Linear(d_model, d_inner, bias=False)
        self.in_z = nn.Linear(d_model, d_inner, bias=False)
        self.ssm = _S4DLayer(d_inner, d_state)
        self.out = nn.Linear(d_inner, d_model, bias=False)

    def combine(self, y, z):
        g = F.silu(z)
        return y * g if self.mode == "mult" else y + g

    def forward(self, u):
        y = self.ssm(self.in_x(u))
        z = self.in_z(u)
        return self.out(self.combine(y, z))


class BearS4DWideGraft(BearS4D):
    def __init__(self, mode, d_model=64, d_state=128, n_layers=4, n_sensors=1, n_classes=2,
                 conv_stride=2, d_inner=D_INNER, d_state_wm=D_STATE_WM, **kwargs):
        super().__init__(d_model=d_model, d_state=d_state, n_layers=n_layers, n_sensors=n_sensors,
                         n_classes=n_classes, conv_stride=conv_stride, **kwargs)
        # replaces the 64-wide S4D layers; conv_embed / layer_norms / norm / classifier untouched
        self.s4d_layers = nn.ModuleList(
            [_WideGatedLayer(d_model, d_inner, d_state_wm, mode) for _ in range(n_layers)])
        self.mode = mode
    # BearS4D.forward is reused unchanged: h = h + layer(ln(h)) per layer


ARMS = {"s4d_plus_gate_wm": "mult", "s4d_plus_branch": "add"}


def build_arm(arm, device, d_model=64, n_layers=4, n_sensors=1, n_classes=2, conv_stride=2):
    return BearS4DWideGraft(ARMS[arm], d_model=d_model, d_state=128, n_layers=n_layers,
                            n_sensors=n_sensors, n_classes=n_classes,
                            conv_stride=conv_stride).to(device)

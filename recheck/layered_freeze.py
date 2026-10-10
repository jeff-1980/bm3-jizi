"""
Paper-9 stage-2 arms: layered freezing of the remaining input-dependent scan terms.

The existing bm3_frozen arm (bm3_frozen.FrozenSelectivityMamba3) removes the input-dependent
parts of Delta, A, B, C but KEEPS two further input-dependent tensors that Mamba-3's scan
consumes: the trapezoidal weight `trap` and the rotation `angles` (mamba3.Mamba3.forward
passes both straight from in_proj into mamba3_siso_combined). This module adds the three
arms that close that gap, by exactly the same device the frozen arm uses for Delta/A/B/C:
delete the corresponding in_proj output slice and replace the tensor with a learned
parameter broadcast over (batch, position).

  frozen_ctrap   trap constant,   angles input-dependent
  frozen_cangle  angles constant, trap   input-dependent
  frozen_clti    both constant  -> no tensor entering the scan depends on the input
                                   (the conv stem and the output gate are untouched)

Everything else - conv stem, gate, norms, projections, depth, head - is inherited from
BearMamba3Frozen unchanged, so each arm differs from bm3_frozen in exactly the targeted
slice. Initialisation follows the existing frozen arm's convention of a neutral zeros
init for the newly introduced constants (A_const is initialised the same way there);
zero is also the mean of the deleted projection's output at initialisation.

Note on `trap`: Mamba3.forward passes trap to the kernel without an activation (the
sigmoid in Mamba3._preprocess is on the incremental-decode path only), so trap_const is
likewise passed raw.
"""
import math, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

SRC = __import__("os").environ.get("P9_HARNESS_DIR") or str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root holds xjtu_noisy_harness.py
if SRC not in sys.path:
    sys.path.insert(0, SRC)
from mamba_ssm.modules.mamba3 import heavy_tail_activation
from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
from bm3_frozen import FrozenSelectivityMamba3, BearMamba3Frozen


class LayeredFrozenMamba3(FrozenSelectivityMamba3):
    """FrozenSelectivityMamba3 with `trap` and/or `angles` also made input-independent."""

    const_trap: bool = False
    const_angles: bool = False

    def __init__(self, *args, **kwargs):
        self.const_trap = bool(kwargs.pop("const_trap", False))
        self.const_angles = bool(kwargs.pop("const_angles", False))
        super().__init__(*args, **kwargs)
        work_dtype = self.in_proj.weight.dtype
        d = 2 * self.d_inner
        if not self.const_trap:
            d += self.nheads
        if not self.const_angles:
            d += self.num_rope_angles
        self.in_proj = nn.Linear(self.d_model, d, bias=False,
                                 device=self.in_proj.weight.device, dtype=work_dtype)
        if self.const_trap:
            self.trap_const = nn.Parameter(torch.zeros(self.nheads, dtype=torch.float32))
        if self.const_angles:
            self.angle_const = nn.Parameter(torch.zeros(self.num_rope_angles, dtype=torch.float32))

    def forward(self, u, seq_idx=None, cu_seqlens=None, inference_params=None):
        assert inference_params is None, "forward-only arm (no incremental decode)"
        batch, seqlen, _ = u.shape

        sizes = [self.d_inner, self.d_inner]
        if not self.const_trap:
            sizes.append(self.nheads)
        if not self.const_angles:
            sizes.append(self.num_rope_angles)
        parts = list(torch.split(self.in_proj(u), sizes, dim=-1))
        z, x = parts[0], parts[1]
        rest = parts[2:]
        trap_proj = None if self.const_trap else rest.pop(0)
        angle_proj = None if self.const_angles else rest.pop(0)

        z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)

        if self.const_trap:
            trap = self.trap_const.view(1, -1, 1).expand(batch, -1, seqlen).contiguous().to(x.dtype)
        else:
            trap = rearrange(trap_proj, "b l h -> b h l")

        # dt, A: per-head constants (inherited from FrozenSelectivityMamba3)
        DT_head = F.softplus(self.dt_bias)
        A_head = torch.clamp(-heavy_tail_activation(self.A_const), max=-self.A_floor)
        DT = DT_head.view(1, -1, 1).expand(batch, -1, seqlen).contiguous().to(torch.float32)
        ADT = (A_head * DT_head).view(1, -1, 1).expand(batch, -1, seqlen).contiguous().to(torch.float32)

        # B, C: constants through the same RMSNorms (inherited convention)
        B = self.B_norm(self.B_const.view(1, 1, self.mimo_rank, self.num_bc_heads, self.d_state)
                        .expand(batch, seqlen, -1, -1, -1))
        C = self.C_norm(self.C_const.view(1, 1, self.mimo_rank, self.num_bc_heads, self.d_state)
                        .expand(batch, seqlen, -1, -1, -1))

        if self.const_angles:
            angles = (self.angle_const.view(1, 1, 1, -1)
                      .expand(batch, seqlen, self.nheads, -1).to(torch.float32))
        else:
            angles = angle_proj.unsqueeze(-2).expand(-1, -1, self.nheads, -1).to(torch.float32)

        y = mamba3_siso_combined(
            Q=C.squeeze(2), K=B.squeeze(2), V=x,
            ADT=ADT, DT=DT, Trap=trap,
            Q_bias=self.C_bias.squeeze(1), K_bias=self.B_bias.squeeze(1),
            Angles=angles, D=self.D,
            Z=z if not self.is_outproj_norm else None,
            chunk_size=self.chunk_size, Input_States=None,
            return_final_states=False, cu_seqlens=cu_seqlens,
        )
        y = rearrange(y, "b l h p -> b l (h p)")
        if self.is_outproj_norm:
            y = self.norm(y, rearrange(z, "b l h p -> b l (h p)"))
        return self.out_proj(y.to(x.dtype))


def _layered_backbone(const_trap, const_angles, **kw):
    """BearMamba3Frozen whose Mamba layers are LayeredFrozenMamba3.

    The layer class is swapped inside bm3_frozen's namespace for the duration of the
    constructor call, so every layer keyword (d_state, is_mimo, mimo_rank, rope_fraction,
    chunk_size, dtype) is taken from BearMamba3Frozen.__init__ itself rather than restated
    here: the new arms cannot drift from the bm3_frozen configuration they ablate.
    """
    import bm3_frozen as _bf

    class _Patched(LayeredFrozenMamba3):
        def __init__(self, *a, **k):
            k.setdefault("const_trap", const_trap)
            k.setdefault("const_angles", const_angles)
            super().__init__(*a, **k)

    orig = _bf.FrozenSelectivityMamba3
    _bf.FrozenSelectivityMamba3 = _Patched
    try:
        m = _bf.BearMamba3Frozen(**kw)
    finally:
        _bf.FrozenSelectivityMamba3 = orig
    n = sum(isinstance(l, LayeredFrozenMamba3) for l in m.mamba_layers)
    assert n == len(m.mamba_layers) and n > 0, f"layer swap incomplete: {n}/{len(m.mamba_layers)}"
    return m


ARMS = {
    "frozen_ctrap":  dict(const_trap=True,  const_angles=False),
    "frozen_cangle": dict(const_trap=False, const_angles=True),
    "frozen_clti":   dict(const_trap=True,  const_angles=True),
}


def build_arm(arm, device, **kw):
    return _layered_backbone(**ARMS[arm], **kw).to(device)

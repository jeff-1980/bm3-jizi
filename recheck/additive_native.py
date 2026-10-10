"""
Paper-9 stage-4 arm: the multiplicative gate of the frozen Mamba-3 block replaced by an
additive combination, inside the native host.

bm3_frozen (bm3_frozen.FrozenSelectivityMamba3) passes z to the fused scan kernel, which applies
out = out * silu(z) after the D-skip term, in float32, before storing the output
(mamba3_siso_fwd.py, "Apply Z-gating if present"). The output projection then maps
d_inner -> d_model.

  bm3_frozen_add   identical modules, parameters and initial weights; the kernel is called with
                   Z=None (no gating in the kernel), and the layer output is
                   out_proj( y + silu(z) ), with y the kernel output and the sum taken in float32.

This is the stage-3 gate -> additive substitution (graft_control.py) applied in the opposite
direction, in the donor block. Against frozen_nogate (z slice removed, Z=None) and bm3_frozen
(y * silu(z)) it isolates the combination operator at fixed width, projections and parameters.

The layer class is swapped inside bm3_frozen's namespace for the duration of the constructor
call (as in layered_freeze.py), so the constructor, its random-number consumption and therefore
the initial weights are those of bm3_frozen.
"""
import sys
import torch
import torch.nn.functional as F
from einops import rearrange

SRC = __import__("os").environ.get("P9_HARNESS_DIR") or str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root holds xjtu_noisy_harness.py
if SRC not in sys.path:
    sys.path.insert(0, SRC)
from mamba_ssm.modules.mamba3 import heavy_tail_activation
from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
from bm3_frozen import FrozenSelectivityMamba3


class AdditiveFrozenMamba3(FrozenSelectivityMamba3):
    def forward(self, u, seq_idx=None, cu_seqlens=None, inference_params=None):
        assert inference_params is None
        assert not self.is_outproj_norm, "additive arm requires is_outproj_norm=False (BM3's config)"
        batch, seqlen, _ = u.shape
        z, x, trap, angles = torch.split(
            self.in_proj(u), [self.d_inner, self.d_inner, self.nheads, self.num_rope_angles], dim=-1)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        trap = rearrange(trap, "b l h -> b h l")
        DT_head = F.softplus(self.dt_bias)
        A_head = torch.clamp(-heavy_tail_activation(self.A_const), max=-self.A_floor)
        DT = DT_head.view(1, -1, 1).expand(batch, -1, seqlen).contiguous().to(torch.float32)
        ADT = (A_head * DT_head).view(1, -1, 1).expand(batch, -1, seqlen).contiguous().to(torch.float32)
        B = self.B_norm(self.B_const.view(1, 1, self.mimo_rank, self.num_bc_heads, self.d_state)
                        .expand(batch, seqlen, -1, -1, -1))
        C = self.C_norm(self.C_const.view(1, 1, self.mimo_rank, self.num_bc_heads, self.d_state)
                        .expand(batch, seqlen, -1, -1, -1))
        angles = angles.unsqueeze(-2).expand(-1, -1, self.nheads, -1).to(torch.float32)
        y = mamba3_siso_combined(
            Q=C.squeeze(2), K=B.squeeze(2), V=x, ADT=ADT, DT=DT, Trap=trap,
            Q_bias=self.C_bias.squeeze(1), K_bias=self.B_bias.squeeze(1),
            Angles=angles, D=self.D, Z=None,          # no gating inside the kernel
            chunk_size=self.chunk_size, Input_States=None,
            return_final_states=False, cu_seqlens=cu_seqlens)
        y = rearrange(y, "b l h p -> b l (h p)")
        g = y.float() + F.silu(z.float())             # additive combination (float32, as the kernel's gate)
        return self.out_proj(g.to(x.dtype))


def build_arm(arm, device, **kw):
    assert arm == "bm3_frozen_add"
    import bm3_frozen as _bf
    orig = _bf.FrozenSelectivityMamba3
    _bf.FrozenSelectivityMamba3 = AdditiveFrozenMamba3
    try:
        m = _bf.BearMamba3Frozen(**kw)
    finally:
        _bf.FrozenSelectivityMamba3 = orig
    assert all(type(l) is AdditiveFrozenMamba3 for l in m.mamba_layers) and len(m.mamba_layers) > 0
    return m.to(device)


ARMS = {"bm3_frozen_add": "add"}

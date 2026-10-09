"""
Paper-9 stage-8 arms: gate removal and gate -> additive substitution inside the FULLY frozen host
(frozen_clti: Delta, A, B, C, trap and angles all learned constants; layered_freeze.py).

  frozen_clti_nogate  z slice removed from in_proj; the kernel is called with Z=None; out_proj(y).
                      The surviving in_proj rows [x] are SLICED from frozen_clti's own in_proj at
                      the same seed (as bm3_models.BearMamba3FrozenNoGate does for bm3_frozen), so
                      every remaining parameter is identical to frozen_clti at initialisation.
  frozen_clti_add     identical modules, parameters and initial weights to frozen_clti; the kernel
                      is called with Z=None and the layer output is out_proj(y + silu(z)), the sum in
                      float32 (as additive_native.py does for bm3_frozen).

The layer class is swapped inside bm3_frozen's namespace for the constructor call (as in
layered_freeze.py), so the configuration is inherited from BearMamba3Frozen and the random-number
consumption, hence the initial weights, are those of frozen_clti.
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

SRC = "/home/jeffwork/exp/bm3-defense/xjtu_noisy_defense_20260627"
if SRC not in sys.path:
    sys.path.insert(0, SRC)
from mamba_ssm.modules.mamba3 import heavy_tail_activation
from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
from layered_freeze import LayeredFrozenMamba3


class CltiGateMamba3(LayeredFrozenMamba3):
    mode: str = "mult"

    def __init__(self, *args, **kwargs):
        mode = kwargs.pop("mode", "mult")
        kwargs["const_trap"] = True; kwargs["const_angles"] = True
        super().__init__(*args, **kwargs)
        self.mode = mode
        assert not self.is_outproj_norm, "requires is_outproj_norm=False (BM3's config)"
        if mode == "nogate":
            old = self.in_proj
            assert old.out_features == 2 * self.d_inner
            # create the smaller Linear without consuming random numbers, so later layers and the
            # head receive exactly frozen_clti's initial weights
            cpu_state = torch.get_rng_state()
            cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=False,
                                     device=old.weight.device, dtype=old.weight.dtype)
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
            with torch.no_grad():
                self.in_proj.weight.copy_(old.weight[self.d_inner:])   # drop z rows, keep x rows
        else:
            assert mode == "add"

    def forward(self, u, seq_idx=None, cu_seqlens=None, inference_params=None):
        assert inference_params is None
        batch, seqlen, _ = u.shape
        if self.mode == "nogate":
            x = self.in_proj(u); z = None
        else:
            z, x = torch.split(self.in_proj(u), [self.d_inner, self.d_inner], dim=-1)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        trap = self.trap_const.view(1, -1, 1).expand(batch, -1, seqlen).contiguous().to(x.dtype)
        DT_head = F.softplus(self.dt_bias)
        A_head = torch.clamp(-heavy_tail_activation(self.A_const), max=-self.A_floor)
        DT = DT_head.view(1, -1, 1).expand(batch, -1, seqlen).contiguous().to(torch.float32)
        ADT = (A_head * DT_head).view(1, -1, 1).expand(batch, -1, seqlen).contiguous().to(torch.float32)
        B = self.B_norm(self.B_const.view(1, 1, self.mimo_rank, self.num_bc_heads, self.d_state)
                        .expand(batch, seqlen, -1, -1, -1))
        C = self.C_norm(self.C_const.view(1, 1, self.mimo_rank, self.num_bc_heads, self.d_state)
                        .expand(batch, seqlen, -1, -1, -1))
        angles = self.angle_const.view(1, 1, 1, -1).expand(batch, seqlen, self.nheads, -1).to(torch.float32)
        y = mamba3_siso_combined(
            Q=C.squeeze(2), K=B.squeeze(2), V=x, ADT=ADT, DT=DT, Trap=trap,
            Q_bias=self.C_bias.squeeze(1), K_bias=self.B_bias.squeeze(1),
            Angles=angles, D=self.D, Z=None,
            chunk_size=self.chunk_size, Input_States=None,
            return_final_states=False, cu_seqlens=cu_seqlens)
        y = rearrange(y, "b l h p -> b l (h p)")
        if self.mode == "add":
            return self.out_proj((y.float() + F.silu(z.float())).to(x.dtype))
        return self.out_proj(y.to(x.dtype))


ARMS = {"frozen_clti_nogate": "nogate", "frozen_clti_add": "add"}


def build_arm(arm, device, **kw):
    import bm3_frozen as _bf
    mode = ARMS[arm]

    class _Patched(CltiGateMamba3):
        def __init__(self, *a, **k):
            k.setdefault("mode", mode)
            super().__init__(*a, **k)

    orig = _bf.FrozenSelectivityMamba3
    _bf.FrozenSelectivityMamba3 = _Patched
    try:
        m = _bf.BearMamba3Frozen(**kw)
    finally:
        _bf.FrozenSelectivityMamba3 = orig
    assert all(isinstance(l, CltiGateMamba3) for l in m.mamba_layers) and len(m.mamba_layers) > 0
    return m.to(device)

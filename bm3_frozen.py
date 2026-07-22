"""
bm3_frozen.py — FrozenSelectivityMamba3 / BearMamba3Frozen, ported into this
directory from ../extended_baselines_noisy_20260626-2235/models_extended.py
(r1's CWRU implementation, lines 233-389 as of 2026-07-02).

Port note: mamba_ssm.modules.mamba3.Mamba3 is byte-identical between this
directory and the CWRU directory — both harnesses (xjtu_noisy_harness.py here,
smoke_harness.py there) run under the SAME venv shebang
(/home/jeffwork/论文8/venv/bin/python3), i.e. the SAME site-packages install
(/home/jeffwork/论文8/venv/lib/python3.12/site-packages/mamba_ssm/modules/mamba3.py).
Verified via `python -c "import mamba_ssm.modules.mamba3 as m; print(m.__file__)"`
under that venv. No base-class version diff exists, so this is a literal copy,
not an adaptation — see arch_map.md for the full diff record (empty diff).

Class bodies below are unchanged from the CWRU source (see docstrings for the
mechanism-removal rationale). Only this module's own docstring and imports are
local.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm.modules.mamba3 import Mamba3, heavy_tail_activation
from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined


class FrozenSelectivityMamba3(Mamba3):
    """Mamba-3 layer (mamba_ssm.modules.mamba3.Mamba3 — the SAME layer family
    BearMamba3 actually uses) with input-selectivity removed.

    Mamba-3's in_proj emits, in order, [z, x, B, C, dd_dt, dd_A, trap, angle].
    "Input selectivity" in the classic S6 sense is exactly {B, C, dd_dt, dd_A}
    — the parts of in_proj that make dt/A/B/C functions of the *current input
    token*. This class removes precisely those four output slices (shrinking
    in_proj accordingly) and replaces dt/A/B/C with learned constants that do
    NOT depend on u:
      - dt: DT = softplus(dt_bias) — dt_bias is inherited unmodified from
        Mamba3.__init__; only dd_dt (the input-dependent additive term) is
        removed, so this is the direct Mamba-3 analogue of the same freeze
        classic FrozenSelectivityMamba applies to dt.
      - A: dd_A had no static analogue in Mamba3 (A is *always* input-projected
        there, unlike classic Mamba's static A_log) — a new A_const parameter
        is introduced to play that role, through the same
        -heavy_tail_activation(...) clamp Mamba3 already uses for A.
      - B, C: replaced by B_const/C_const, broadcast over (batch, seqlen) and
        passed through the SAME B_norm/C_norm RMSNorm modules Mamba3 uses.

    Everything else — RoPE (angle stays input-dependent, computed from the
    shrunk in_proj: RoPE is a positional mechanic, not per-token selectivity),
    the trapezoidal integration correction (trap, likewise data-dependent and
    kept), B_bias/C_bias, D skip, out_proj, the chunked SSD scan kernel
    (mamba3_siso_combined) — is inherited unchanged and used exactly as
    Mamba3.forward() uses it. SISO only (is_mimo=False): matches BearMamba3's
    actual configuration in this repo (xjtu_noisy_harness.py build_model()
    always passes is_mimo=False for bm3_kin/bm3_nokin).
    """

    def __init__(self, *args, **kwargs):
        assert not kwargs.get("is_mimo", False), (
            "FrozenSelectivityMamba3 supports SISO (is_mimo=False) only — "
            "matches BearMamba3's actual configuration in this repo."
        )
        super().__init__(*args, **kwargs)
        work_dtype = self.in_proj.weight.dtype
        d_in_proj_frozen = 2 * self.d_inner + self.nheads + self.num_rope_angles
        self.in_proj = nn.Linear(self.d_model, d_in_proj_frozen, bias=False,
                                 device=self.in_proj.weight.device,
                                 dtype=work_dtype)
        # A_const has no inherited initialization to reuse (Mamba3's A is
        # always input-projected, never a static parameter) — init to zeros:
        # heavy_tail_activation(0) == 1, so A starts at -1, a neutral decay
        # rate comparable in magnitude to Mamba3's own dt-range init.
        self.A_const = nn.Parameter(torch.zeros(self.nheads, dtype=torch.float32))
        # B_const/C_const replace the removed B/C projections; shape matches
        # the (mimo_rank, num_bc_heads, d_state) block Mamba3 rearranges B/C
        # into before B_norm/C_norm. Stored in the model's working dtype
        # (matching B_norm/C_norm's weight dtype and what Q/K normally are).
        self.B_const = nn.Parameter(
            torch.ones(self.mimo_rank, self.num_bc_heads, self.d_state, dtype=work_dtype))
        self.C_const = nn.Parameter(
            (torch.randn(self.mimo_rank, self.num_bc_heads, self.d_state, dtype=torch.float32)
             / math.sqrt(self.d_state)).to(work_dtype))

    def forward(self, u, seq_idx=None, cu_seqlens=None, inference_params=None):
        assert inference_params is None, (
            "FrozenSelectivityMamba3 is forward-only (training/eval); "
            "incremental decode (step()) is not supported."
        )
        batch, seqlen, _ = u.shape

        zxtrapangle = self.in_proj(u)
        z, x, trap, angles = torch.split(
            zxtrapangle,
            [self.d_inner, self.d_inner, self.nheads, self.num_rope_angles],
            dim=-1,
        )
        z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        trap = rearrange(trap, "b l h -> b h l")

        # Non-selective dt, A: true per-head constants (no dependence on u),
        # broadcast to every (batch, position) — this IS "selectivity removed".
        DT_head = F.softplus(self.dt_bias)                        # (nheads,)
        A_head = -heavy_tail_activation(self.A_const)              # (nheads,)
        A_head = torch.clamp(A_head, max=-self.A_floor)
        DT = DT_head.view(1, -1, 1).expand(batch, -1, seqlen).contiguous().to(torch.float32)
        ADT = (A_head * DT_head).view(1, -1, 1).expand(batch, -1, seqlen).contiguous().to(torch.float32)

        # Non-selective B, C: true constants, broadcast over batch/seqlen, then
        # through the same RMSNorm modules Mamba3 uses (the only other
        # transform applied to B/C besides the now-removed projection).
        B_bcast = self.B_const.view(1, 1, self.mimo_rank, self.num_bc_heads, self.d_state).expand(
            batch, seqlen, -1, -1, -1)
        C_bcast = self.C_const.view(1, 1, self.mimo_rank, self.num_bc_heads, self.d_state).expand(
            batch, seqlen, -1, -1, -1)
        B = self.B_norm(B_bcast)
        C = self.C_norm(C_bcast)

        # Data-dependent RoPE angle — positional mechanic, not selectivity;
        # cast to float32 as required by the SISO kernel (matches Mamba3.forward).
        angles = angles.unsqueeze(-2).expand(-1, -1, self.nheads, -1).to(torch.float32)

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
            z = rearrange(z, "b l h p -> b l (h p)")
            y = self.norm(y, z)
        return self.out_proj(y.to(x.dtype))


class BearMamba3Frozen(nn.Module):
    """bm3_frozen backbone: identical conv stem / pre-norm residual / head to
    BearMamba3 (bearmamba3/model.py) with is_mimo=False (== bm3_nokin's actual
    config in this repo), with each layer's Mamba3(...) replaced by
    FrozenSelectivityMamba3(...) — the SAME layer family/kernel, differing
    only in the selective components (in_proj size + *_const parameters) and
    class name. See frozensel_unitcheck.py for the structural diff check
    against bm3_nokin that verifies this claim key-by-key.
    """
    def __init__(self, d_model=64, d_state=128, n_layers=4, n_sensors=1,
                 n_classes=4, conv_stride=2, use_batchnorm=False,
                 dtype=torch.bfloat16, **kwargs):
        super().__init__()
        self.conv_stride = conv_stride
        self.conv_embed = nn.Conv1d(n_sensors, d_model, kernel_size=7,
                                    stride=conv_stride, padding=3)
        self.bn_embed = nn.BatchNorm1d(d_model) if use_batchnorm else None
        self.mamba_layers = nn.ModuleList([
            FrozenSelectivityMamba3(d_model=d_model, d_state=d_state,
                                    is_mimo=False, mimo_rank=4, rope_fraction=0.5,
                                    chunk_size=64, dtype=dtype)
            for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model, dtype=dtype) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)
        self.dtype = dtype

    def forward(self, x, return_kin=False):
        h = self.conv_embed(x.to(self.conv_embed.weight.dtype))
        if self.bn_embed is not None:
            h = self.bn_embed(h.float()).to(self.conv_embed.weight.dtype)
        h = h.transpose(1, 2).to(self.dtype)          # (B, L', D)
        for layer, ln in zip(self.mamba_layers, self.layer_norms):
            h = h + layer(ln(h))                       # pre-norm residual
        h = self.norm(h.float()).mean(dim=1)
        logits = self.classifier(h)
        return (logits, []) if return_kin else logits

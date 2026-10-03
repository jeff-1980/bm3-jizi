"""
models_extended.py — New baseline models for extended fair comparison.

Models added (all ~100K params, same conv_embed as BM3/CNN1D/Transformer):
  BearTCN     : Temporal Convolutional Network (Bai et al., 2018) with causal
                dilated convolutions and exponentially growing dilation.
  BearCNNLSTM : Hybrid CNN (local features) + LSTM (temporal modeling).
  BearS4D     : Diagonal State Space Model (S4D-Real, Gu et al., 2022) — a
                non-Mamba, time-invariant SSM with analytically computed kernel.
                Key difference from Mamba: A,B,C are time-invariant (not input-selective).

All models:
  - Same conv_embed: Conv1d(n_sensors, d_model, kernel=8, stride=conv_stride, pad=3)
  - Same pre-norm residual depth (4 layers)
  - Return logits; accept return_kin=False (no kinematic loss, lam=0)
  - ~100K parameters with default d_model=64, n_layers=4
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm.modules.mamba3 import Mamba3, heavy_tail_activation
from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined


# ══════════════════════════════════════════════════════════════════════════════
# 1. TCN — Temporal Convolutional Network
# ══════════════════════════════════════════════════════════════════════════════

class _CausalDilatedConv1d(nn.Module):
    """Left-pad + Conv1d for causal dilated temporal convolution."""
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              dilation=dilation, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, (self.left_pad, 0)))


class _TemporalBlock(nn.Module):
    """Two causal dilated convs + BN + GELU + residual (same channel dim)."""
    def __init__(self, d_model, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            _CausalDilatedConv1d(d_model, kernel_size, dilation),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            _CausalDilatedConv1d(d_model, kernel_size, dilation),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x) + x  # residual (no projection needed, same dim)


class BearTCN(nn.Module):
    """TCN baseline (~101K params).

    conv_embed → 4 TemporalBlocks (dilation 1,2,4,8) → global avg pool → classifier.
    """
    def __init__(self, d_model=64, n_layers=4, n_sensors=1, n_classes=4,
                 conv_stride=2, kernel_size=3, **kwargs):
        super().__init__()
        self.conv_embed = nn.Conv1d(n_sensors, d_model, kernel_size=8,
                                    stride=conv_stride, padding=3)
        self.blocks = nn.ModuleList([
            _TemporalBlock(d_model, kernel_size=kernel_size, dilation=2 ** i)
            for i in range(n_layers)
        ])
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x, return_kin=False):
        h = self.conv_embed(x)
        for blk in self.blocks:
            h = blk(h)
        h = h.mean(-1)              # global average pool
        logits = self.classifier(h)
        return (logits, []) if return_kin else logits


# ══════════════════════════════════════════════════════════════════════════════
# 2. CNN-LSTM
# ══════════════════════════════════════════════════════════════════════════════

class _ConvPool1D(nn.Module):
    """Two Conv+BN+GELU followed by MaxPool(2) — same structure as BearCNN1D block."""
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model), nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model), nn.GELU(),
            nn.MaxPool1d(2),
        )

    def forward(self, x):
        return self.net(x)


class BearCNNLSTM(nn.Module):
    """CNN-LSTM baseline (~117K params).

    conv_embed → 2 ConvPool blocks → 2-layer LSTM → last hidden → classifier.
    Conv blocks provide local features; LSTM models temporal dependencies.
    """
    def __init__(self, d_model=64, n_layers=4, n_sensors=1, n_classes=4,
                 conv_stride=2, **kwargs):
        super().__init__()
        n_conv = max(1, n_layers // 2)  # = 2 for n_layers=4
        n_lstm = max(1, n_layers - n_conv)  # = 2 for n_layers=4
        self.conv_embed = nn.Conv1d(n_sensors, d_model, kernel_size=8,
                                    stride=conv_stride, padding=3)
        self.conv_blocks = nn.ModuleList([_ConvPool1D(d_model) for _ in range(n_conv)])
        self.lstm = nn.LSTM(d_model, d_model, num_layers=n_lstm, batch_first=True,
                            dropout=0.1 if n_lstm > 1 else 0.0)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x, return_kin=False):
        h = self.conv_embed(x)              # (B, d, L//stride)
        for blk in self.conv_blocks:
            h = blk(h)                      # (B, d, L // 2^n_conv)
        h = h.permute(0, 2, 1)             # (B, L', d) for LSTM
        _, (hn, _) = self.lstm(h)           # hn: (n_lstm, B, d)
        h = hn[-1]                          # (B, d) last hidden state
        logits = self.classifier(h)
        return (logits, []) if return_kin else logits


# ══════════════════════════════════════════════════════════════════════════════
# 3. S4D — Diagonal State Space Model (non-Mamba SSM variant)
# ══════════════════════════════════════════════════════════════════════════════

class _S4DKernel(nn.Module):
    """Analytically computed SSM kernel using real diagonal A (S4D-Real).

    Continuous-time SSM: dx/dt = Ax + Bu,  y = Cx + Du
    A = diag(a_1, ..., a_N), a_n < 0 (stability), parameterized as -exp(log_neg_A).
    ZOH discretization: Ā = exp(Δ·A),  B̄ = (Ā - 1) / A · B
    Kernel: k[t] = sum_n  C_n · Ā_n^t · B̄_n   (causal impulse response)
    """
    def __init__(self, d_model, d_state=128, dt_min=1e-3, dt_max=0.1):
        super().__init__()
        # Step size: one scalar per feature, learned in log space
        log_dt = torch.linspace(math.log(dt_min), math.log(dt_max), d_model)
        self.log_dt = nn.Parameter(log_dt)

        # Diagonal A: parameterized as -exp(log_neg_A), initialized evenly in log scale
        log_neg_A_1d = torch.linspace(math.log(dt_min), math.log(0.5), d_state)
        self.log_neg_A = nn.Parameter(
            log_neg_A_1d.unsqueeze(0).expand(d_model, -1).clone()
        )  # (d_model, d_state)

        # Input / output projections
        scale = d_state ** -0.5
        self.B = nn.Parameter(torch.randn(d_model, d_state) * scale)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * scale)

    def forward(self, L: int):
        dt = torch.exp(self.log_dt).unsqueeze(1)          # (d_model, 1)
        A = -torch.exp(self.log_neg_A)                     # (d_model, d_state), <0
        A_bar = torch.exp(dt * A)                          # (d_model, d_state), (0,1)
        B_bar = (A_bar - 1.0) / A * self.B               # ZOH: (d_model, d_state)

        t_idx = torch.arange(L, device=dt.device, dtype=torch.float32)  # (L,)
        log_Abar = torch.log(A_bar.clamp(min=1e-30))      # (d_model, d_state)
        # A_bar_t[d, n, l] = A_bar[d,n]^l
        exponents = log_Abar.unsqueeze(-1) * t_idx.view(1, 1, -1)  # (d_model, d_state, L)
        A_bar_t = torch.exp(exponents)                    # (d_model, d_state, L)

        CB_bar = self.C * B_bar                           # (d_model, d_state)
        kernel = torch.einsum("dn,dnl->dl", CB_bar, A_bar_t)  # (d_model, L)
        return kernel


class _S4DLayer(nn.Module):
    """S4D SSM layer: causal FFT convolution with analytically-computed kernel + skip."""
    def __init__(self, d_model, d_state=128):
        super().__init__()
        self.kernel_fn = _S4DKernel(d_model, d_state)
        self.D = nn.Parameter(torch.ones(d_model))   # skip connection

    def forward(self, u):
        # u: (B, L, d_model)
        B, L, D = u.shape
        kernel = self.kernel_fn(L)                    # (d_model, L)

        u_t = u.permute(0, 2, 1).float()             # (B, d_model, L)
        fft_len = 2 * L                               # zero-pad for linear conv
        U_f = torch.fft.rfft(u_t, n=fft_len)
        K_f = torch.fft.rfft(kernel.float(), n=fft_len)
        y = torch.fft.irfft(U_f * K_f.unsqueeze(0), n=fft_len)[..., :L]  # (B, d_model, L)
        y = y + self.D.view(1, -1, 1) * u_t
        return y.to(u.dtype).permute(0, 2, 1)        # (B, L, d_model)


class BearS4D(nn.Module):
    """S4D (Diagonal SSM) baseline (~102K params) — non-Mamba SSM variant.

    Key difference from Mamba (BM3): A, B, C are TIME-INVARIANT (not input-selective).
    Uses the analytically-computed causal convolution kernel from S4D-Real.
    Pre-norm residual architecture mirrors BearMamba3 for fair comparison.
    """
    def __init__(self, d_model=64, d_state=128, n_layers=4, n_sensors=1,
                 n_classes=4, conv_stride=2, **kwargs):
        super().__init__()
        self.conv_embed = nn.Conv1d(n_sensors, d_model, kernel_size=8,
                                    stride=conv_stride, padding=3)
        self.s4d_layers = nn.ModuleList([_S4DLayer(d_model, d_state) for _ in range(n_layers)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x, return_kin=False):
        h = self.conv_embed(x).transpose(1, 2)  # (B, L', d_model)
        for layer, ln in zip(self.s4d_layers, self.layer_norms):
            h = h + layer(ln(h))                 # pre-norm residual
        h = self.norm(h).mean(1)                 # global avg pool
        logits = self.classifier(h)
        return (logits, []) if return_kin else logits


# ══════════════════════════════════════════════════════════════════════════════
# 4. FrozenSelectivityMamba3 — Mamba-3 layer with input-selectivity removed
# ══════════════════════════════════════════════════════════════════════════════

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
    actual configuration in this repo (smoke_harness.py build_model() always
    passes is_mimo=False for bm3/bm3_nokin).
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


# ══════════════════════════════════════════════════════════════════════════════
# 5. BearMamba3Frozen — BM3 skeleton with FrozenSelectivityMamba3 in place of Mamba3
# ══════════════════════════════════════════════════════════════════════════════

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

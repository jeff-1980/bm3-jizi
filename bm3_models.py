"""
bm3_models.py — secondary single-variable ablations built on the frozen base
(BearMamba3Frozen / FrozenSelectivityMamba3, bm3_frozen.py). Each class here
changes EXACTLY ONE additional part beyond stage-1 selectivity-removal:

  frozen_noconv -> BearMamba3FrozenNoConv   : outer conv_embed stem only
  frozen_nogate -> BearMamba3FrozenNoGate   : z-gating (mamba3_siso_combined Z) only
  frozen_As4d   -> BearMamba3FrozenAS4D     : A's functional form (A_const -> A_log) only

Location/removal-method/feasibility rationale for each arm: arch_map.md §5.
Structural verification (state_dict diff vs bm3_frozen, forward/backward,
param table): secondary_unitcheck.py / secondary_unitcheck.json.

Also holds the constructive graft arm (arch_map.md §6):

  s4d_plus_gate -> BearS4DPlusGate : grafts a Mamba-3-style output gate onto
                                     the never-selective BearS4D baseline
                                     (models_extended.py), which itself is
                                     NOT modified (zero edits — see
                                     arch_map.md §6 for the checklist).
Structural verification vs s4d: graft_unitcheck.py / graft_unitcheck.json.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm.modules.mamba3 import heavy_tail_activation
from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined

from bm3_frozen import FrozenSelectivityMamba3, BearMamba3Frozen
from models_extended import BearS4D


# ── Shared: non-target-parameter preservation for the mamba_layers-replacing
#    arms (frozen_nogate, frozen_As4d) ──────────────────────────────────────

# Every FrozenSelectivityMamba3 parameter EXCEPT in_proj/out_proj/B_norm/
# C_norm (handled separately below, since they're nn.Module/nn.Linear
# submodules rather than bare nn.Parameters).
_SHARED_SCALAR_PARAMS = ("dt_bias", "A_const", "B_const", "C_const",
                         "B_bias", "C_bias", "D")


def _copy_nontarget_frozen_params(dst_layer, src_layer, exclude=()):
    """Copy every FrozenSelectivityMamba3 parameter that is NOT the one part
    a given secondary arm intentionally changes, verbatim (`.copy_`, not a
    fresh random init) from `src_layer` (the untouched bm3_frozen-style layer
    BearMamba3Frozen.__init__ already built for this model instance) into
    `dst_layer` (the arm's replacement layer instance).

    Without this, constructing dst_layer via its own __init__ re-runs
    Mamba3.__init__'s random initialization, which consumes a DIFFERENT
    slice of the torch RNG stream than src_layer did (src_layer was built
    earlier, inside BearMamba3Frozen.__init__, before dst_layer exists) —
    silently perturbing every "non-target" parameter's VALUE even though its
    name/shape still matches bm3_frozen's. That numeric drift is exactly
    what broke single-arm isolation for frozen_nogate/frozen_As4d (see
    secondary_unitcheck.py's value-diff check, arch_map.md §5).
    """
    with torch.no_grad():
        for name in _SHARED_SCALAR_PARAMS:
            if name in exclude:
                continue
            getattr(dst_layer, name).copy_(getattr(src_layer, name))
        dst_layer.B_norm.load_state_dict(src_layer.B_norm.state_dict())
        dst_layer.C_norm.load_state_dict(src_layer.C_norm.state_dict())
        dst_layer.out_proj.weight.copy_(src_layer.out_proj.weight)


# ── Arm 1: frozen_noconv ────────────────────────────────────────────────────

class StridedLinearStem(nn.Module):
    """Non-convolutional replacement for BearMamba3Frozen's outer conv_embed
    stem, used by frozen_noconv. A kernel_size=1 Conv1d has zero
    cross-timestep receptive field, so it is already a per-timestep linear
    channel lift applied at every conv_stride-th input position — this
    module makes that explicit (strided slicing, no convolution, + a plain
    nn.Linear) so frozen_noconv's forward path contains no nn.Conv1d/
    F.conv1d call anywhere, not merely a kernel_size=1 one.

    Mathematically identical output to
    nn.Conv1d(n_sensors, d_model, kernel_size=1, stride=conv_stride,
    padding=0): out[:, d, t] = sum_c weight[d, c] * x[:, c, t*stride] +
    bias[d]. Exposes .weight/.bias so BearMamba3Frozen.forward's
    `self.conv_embed.weight.dtype` dtype-cast keeps working unmodified.
    See arch_map.md §5a.
    """
    def __init__(self, n_sensors, d_model, stride, device=None, dtype=None):
        super().__init__()
        self.stride = stride
        self.proj = nn.Linear(n_sensors, d_model, bias=True,
                              device=device, dtype=dtype)

    @property
    def weight(self):
        return self.proj.weight

    @property
    def bias(self):
        return self.proj.bias

    def forward(self, x):
        # x: (batch, n_sensors, L_in) — same input convention as Conv1d.
        x_strided = x[:, :, ::self.stride]         # (batch, n_sensors, L_out)
        h = self.proj(x_strided.transpose(1, 2))   # (batch, L_out, d_model)
        return h.transpose(1, 2)                    # (batch, d_model, L_out)


class BearMamba3FrozenNoConv(BearMamba3Frozen):
    """BearMamba3Frozen with the outer conv_embed stem replaced by
    StridedLinearStem: removes ONLY the stem's cross-timestep local mixing
    (no nn.Conv1d anywhere in the forward path — see StridedLinearStem's
    docstring for why a kernel_size=1 Conv1d and this stride-slice+Linear
    module are mathematically the same map). Output length is provably
    identical to the original kernel=7/pad=3 stem for any L_in (arch_map.md
    §5a). mamba_layers (FrozenSelectivityMamba3), layer_norms, norm,
    classifier: byte-identical to BearMamba3Frozen.
    """
    def __init__(self, d_model=64, d_state=128, n_layers=4, n_sensors=1,
                 n_classes=4, conv_stride=2, use_batchnorm=False,
                 dtype=torch.bfloat16, **kwargs):
        super().__init__(d_model=d_model, d_state=d_state, n_layers=n_layers,
                          n_sensors=n_sensors, n_classes=n_classes,
                          conv_stride=conv_stride, use_batchnorm=use_batchnorm,
                          dtype=dtype, **kwargs)
        self.conv_embed = StridedLinearStem(n_sensors, d_model, conv_stride,
                                            dtype=dtype)


# ── Arm 2: frozen_nogate ─────────────────────────────────────────────────────

class FrozenNoGateMamba3(FrozenSelectivityMamba3):
    """FrozenSelectivityMamba3 with z-gating additionally removed.

    Mamba-3's output gate is not a separate nn.Module — the fused kernel
    itself applies `out = out * silu(Z)` when Z is not None
    (mamba3_siso_fwd.py:410-411, guarded by compile-time HAS_Z). This class
    (a) drops z's d_inner-wide slice from in_proj — frozen's in_proj already
    emits only [z, x, trap, angle] (bm3_frozen.py); z is the ONLY slice
    removed here — and (b) calls the kernel with Z=None, so HAS_Z=False and
    the gating multiply (and the z_block load) never execute: not zeroed,
    physically absent from the compute graph. See arch_map.md §5b.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        work_dtype = self.in_proj.weight.dtype
        d_in_proj_nogate = self.d_inner + self.nheads + self.num_rope_angles
        self.in_proj = nn.Linear(self.d_model, d_in_proj_nogate, bias=False,
                                 device=self.in_proj.weight.device,
                                 dtype=work_dtype)

    def forward(self, u, seq_idx=None, cu_seqlens=None, inference_params=None):
        assert inference_params is None, (
            "FrozenNoGateMamba3 is forward-only (training/eval); "
            "incremental decode (step()) is not supported."
        )
        assert not self.is_outproj_norm, (
            "FrozenNoGateMamba3 requires is_outproj_norm=False (BearMamba3's "
            "actual configuration in this repo) — the is_outproj_norm=True "
            "path re-applies gating via self.norm(y, z), which would silently "
            "reintroduce the gate this arm removes."
        )
        batch, seqlen, _ = u.shape

        xtrapangle = self.in_proj(u)
        x, trap, angles = torch.split(
            xtrapangle,
            [self.d_inner, self.nheads, self.num_rope_angles],
            dim=-1,
        )
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        trap = rearrange(trap, "b l h -> b h l")

        DT_head = F.softplus(self.dt_bias)                        # (nheads,)
        A_head = -heavy_tail_activation(self.A_const)              # (nheads,)
        A_head = torch.clamp(A_head, max=-self.A_floor)
        DT = DT_head.view(1, -1, 1).expand(batch, -1, seqlen).contiguous().to(torch.float32)
        ADT = (A_head * DT_head).view(1, -1, 1).expand(batch, -1, seqlen).contiguous().to(torch.float32)

        B_bcast = self.B_const.view(1, 1, self.mimo_rank, self.num_bc_heads, self.d_state).expand(
            batch, seqlen, -1, -1, -1)
        C_bcast = self.C_const.view(1, 1, self.mimo_rank, self.num_bc_heads, self.d_state).expand(
            batch, seqlen, -1, -1, -1)
        B = self.B_norm(B_bcast)
        C = self.C_norm(C_bcast)

        angles = angles.unsqueeze(-2).expand(-1, -1, self.nheads, -1).to(torch.float32)

        y = mamba3_siso_combined(
            Q=C.squeeze(2), K=B.squeeze(2), V=x,
            ADT=ADT, DT=DT, Trap=trap,
            Q_bias=self.C_bias.squeeze(1), K_bias=self.B_bias.squeeze(1),
            Angles=angles, D=self.D,
            Z=None,   # gate removed: HAS_Z=False in the kernel, no silu(Z) multiply at all
            chunk_size=self.chunk_size, Input_States=None,
            return_final_states=False, cu_seqlens=cu_seqlens,
        )
        y = rearrange(y, "b l h p -> b l (h p)")
        return self.out_proj(y.to(x.dtype))


class BearMamba3FrozenNoGate(BearMamba3Frozen):
    """bm3_frozen backbone with each layer's FrozenSelectivityMamba3 replaced
    by FrozenNoGateMamba3 (z-gating additionally removed). Conv stem /
    pre-norm depth / classifier: byte-identical to BearMamba3Frozen.

    Every non-target parameter (dt_bias, A_const, B_const, C_const, B_bias,
    C_bias, D, B_norm, C_norm, out_proj) is copied VERBATIM from the
    already-built bm3_frozen-style layer super().__init__() produced, not
    re-randomized — see _copy_nontarget_frozen_params's docstring for why a
    fresh FrozenNoGateMamba3(...) construction would otherwise silently
    perturb these values. in_proj (the target: z's slice removed) is built
    by SLICING that same original layer's in_proj weight rather than a fresh
    random Linear, so even the surviving [x, trap, angle] rows stay
    byte-identical to bm3_frozen (arch_map.md §5b).
    """
    def __init__(self, d_model=64, d_state=128, n_layers=4, n_sensors=1,
                 n_classes=4, conv_stride=2, use_batchnorm=False,
                 dtype=torch.bfloat16, **kwargs):
        super().__init__(d_model=d_model, d_state=d_state, n_layers=n_layers,
                          n_sensors=n_sensors, n_classes=n_classes,
                          conv_stride=conv_stride, use_batchnorm=use_batchnorm,
                          dtype=dtype, **kwargs)
        new_layers = []
        for layer in self.mamba_layers:
            new_layer = FrozenNoGateMamba3(d_model=d_model, d_state=d_state,
                               is_mimo=False, mimo_rank=4, rope_fraction=0.5,
                               chunk_size=64, dtype=dtype)
            _copy_nontarget_frozen_params(new_layer, layer)
            with torch.no_grad():
                new_layer.in_proj.weight.copy_(layer.in_proj.weight[layer.d_inner:])
            new_layers.append(new_layer)
        self.mamba_layers = nn.ModuleList(new_layers)


# ── Arm 3: frozen_As4d ───────────────────────────────────────────────────────

class FrozenAS4DMamba3(FrozenSelectivityMamba3):
    """FrozenSelectivityMamba3 with A_const's heavy_tail_activation
    parameterization replaced by a classic S4D-style real-diagonal form:
    A_head = -exp(A_log) instead of clamp(-heavy_tail_activation(A_const),
    max=-A_floor). Only this ONE line's functional form changes — dt/B/C,
    in_proj, the kernel call shape, out_proj: byte-identical to
    FrozenSelectivityMamba3. Feasibility judgment + rejected alternative:
    arch_map.md §5c.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.A_const
        # S4D-real/HiPPO-style arithmetic-progression init: A_log =
        # log([1..nheads]) -> A_head = -[1..nheads]. Same order of magnitude
        # as bm3_frozen's own A_const=0 init (-> A_head=-1, arch_map.md §2).
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, self.nheads + 1, dtype=torch.float32)))

    def forward(self, u, seq_idx=None, cu_seqlens=None, inference_params=None):
        assert inference_params is None, (
            "FrozenAS4DMamba3 is forward-only (training/eval); "
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

        DT_head = F.softplus(self.dt_bias)                        # (nheads,)
        # S4D-style A: real, negative, no heavy-tail activation, no A_floor
        # clamp needed (exp(.) > 0 unconditionally, so -exp(.) < 0 always).
        A_head = -torch.exp(self.A_log)                            # (nheads,)
        DT = DT_head.view(1, -1, 1).expand(batch, -1, seqlen).contiguous().to(torch.float32)
        ADT = (A_head * DT_head).view(1, -1, 1).expand(batch, -1, seqlen).contiguous().to(torch.float32)

        B_bcast = self.B_const.view(1, 1, self.mimo_rank, self.num_bc_heads, self.d_state).expand(
            batch, seqlen, -1, -1, -1)
        C_bcast = self.C_const.view(1, 1, self.mimo_rank, self.num_bc_heads, self.d_state).expand(
            batch, seqlen, -1, -1, -1)
        B = self.B_norm(B_bcast)
        C = self.C_norm(C_bcast)

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


class BearMamba3FrozenAS4D(BearMamba3Frozen):
    """bm3_frozen backbone with each layer's FrozenSelectivityMamba3 replaced
    by FrozenAS4DMamba3 (A's parameterization only). Conv stem / pre-norm
    depth / classifier: byte-identical to BearMamba3Frozen.

    Every non-target parameter (dt_bias, B_const, C_const, B_bias, C_bias,
    D, B_norm, C_norm, out_proj, in_proj — this arm doesn't touch in_proj's
    shape/split at all) is copied VERBATIM from the already-built
    bm3_frozen-style layer super().__init__() produced, not re-randomized —
    see _copy_nontarget_frozen_params's docstring. Only A_const (target:
    dropped) / A_log (target: new) are exempt — A_log intentionally keeps
    FrozenAS4DMamba3.__init__'s deterministic arithmetic-progression init,
    independent of the source layer's A_const value by design (arch_map.md
    §5c).
    """
    def __init__(self, d_model=64, d_state=128, n_layers=4, n_sensors=1,
                 n_classes=4, conv_stride=2, use_batchnorm=False,
                 dtype=torch.bfloat16, **kwargs):
        super().__init__(d_model=d_model, d_state=d_state, n_layers=n_layers,
                          n_sensors=n_sensors, n_classes=n_classes,
                          conv_stride=conv_stride, use_batchnorm=use_batchnorm,
                          dtype=dtype, **kwargs)
        new_layers = []
        for layer in self.mamba_layers:
            new_layer = FrozenAS4DMamba3(d_model=d_model, d_state=d_state,
                             is_mimo=False, mimo_rank=4, rope_fraction=0.5,
                             chunk_size=64, dtype=dtype)
            _copy_nontarget_frozen_params(new_layer, layer, exclude=("A_const",))
            with torch.no_grad():
                new_layer.in_proj.weight.copy_(layer.in_proj.weight)
            new_layers.append(new_layer)
        self.mamba_layers = nn.ModuleList(new_layers)


# ── Arm 4 (graft, constructive): s4d_plus_gate ──────────────────────────────

class BearS4DPlusGate(BearS4D):
    """BearS4D (never-selective S4D baseline, models_extended.py) with ONE
    part grafted on: an input-dependent output gate borrowed from Mamba-3's
    z-gate mechanism (`out = out * silu(Z)`, arch_map.md §1/§5b), applied to
    each `_S4DLayer`'s SSM output. `_S4DKernel`/`_S4DLayer`/`BearS4D` in
    models_extended.py have ZERO edits — see arch_map.md §6 for the
    exhaustive zero-change checklist. This class only subclasses BearS4D,
    reuses the exact submodules `super().__init__()` already built
    (`conv_embed`, `s4d_layers`, `layer_norms`, `norm`, `classifier` —
    untouched, not reconstructed), and overrides `forward` to splice in the
    gate multiply plus one new per-layer projection.

    Insertion point (arch_map.md §6): `BearS4D.forward`'s per-layer residual
    line `h = h + layer(ln(h))` (models_extended.py:223) is the ONLY
    behavior this class changes, and it changes it in a subclass override,
    not by editing BearS4D itself.

    z's source projection (arch_map.md §6): a new per-layer
    `nn.Linear(d_model, d_model, bias=True)` applied to the SAME pre-SSM
    tensor (`ln(h)`, the layer-normed residual stream) that `_S4DLayer`
    itself also consumes as `u` — mirroring Mamba-3's z, which is sliced
    from the same `in_proj(u)` call that also produces x/B/C for that layer
    (arch_map.md §1). `_S4DLayer` has no single fused input projection to
    slice a z channel from (B/C/D/dt there are free `nn.Parameter`s, not
    projections of the per-token input at all — that time-invariance is
    exactly what S4D ablates relative to Mamba-3), so a new small `nn.Linear`
    of the same layer input is the smallest change that gives z an analogous
    "projection of this layer's input" provenance instead of inventing an
    unrelated source (e.g. gating from a global/static parameter, or from
    `y` itself, would not carry the property being grafted: an
    INPUT-DEPENDENT gate).
    """
    def __init__(self, d_model=64, d_state=128, n_layers=4, n_sensors=1,
                 n_classes=4, conv_stride=2, **kwargs):
        super().__init__(d_model=d_model, d_state=d_state, n_layers=n_layers,
                          n_sensors=n_sensors, n_classes=n_classes,
                          conv_stride=conv_stride, **kwargs)
        # New key only: gate_proj.{i}.weight/bias, {i} in range(n_layers).
        # Constructed AFTER super().__init__() so it consumes the RNG stream
        # strictly after conv_embed/s4d_layers/layer_norms/norm/classifier —
        # every one of those keys is therefore byte-identical to a plain
        # BearS4D built under the same seed (verified in graft_unitcheck.json).
        self.gate_proj = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=True) for _ in range(n_layers)
        ])

    def forward(self, x, return_kin=False):
        h = self.conv_embed(x).transpose(1, 2)   # (B, L', d_model) — identical call to BearS4D.forward
        for layer, ln, gate in zip(self.s4d_layers, self.layer_norms, self.gate_proj):
            ln_h = ln(h)
            y = layer(ln_h)          # untouched _S4DLayer.forward — same call BearS4D itself makes
            z = gate(ln_h)           # z's source projection: Linear(ln_h), same input as the SSM's u
            h = h + y * F.silu(z)    # grafted gate branch: out = out * silu(z) (Mamba-3-style, §1)
        h = self.norm(h).mean(1)
        logits = self.classifier(h)
        return (logits, []) if return_kin else logits

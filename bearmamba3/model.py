"""
bearmamba3/model.py — BearMamba-3 骨干
关键设计:
  - 卷积嵌入（非 Patch，依据 IR-attractor 论文的 OR 盲点结论）
  - Mamba-3 层（SISO/MIMO 可切换，用于消融）
  - forward 同时返回各层 (theta, DT) 激活 → 激活级 L_kin（见侦察报告结论三）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.modules.mamba3 import Mamba3


class BearMamba3(nn.Module):
    def __init__(self, d_model=64, d_state=128, n_layers=4,
                 n_sensors=3, n_classes=4, conv_stride=2,
                 is_mimo=True, mimo_rank=4, rope_fraction=0.5,
                 use_batchnorm=False,
                 dtype=torch.bfloat16):
        super().__init__()
        self.conv_stride = conv_stride
        self.conv_embed = nn.Conv1d(n_sensors, d_model, kernel_size=7,
                                    stride=conv_stride, padding=3)
        # BN after conv_embed — fair-comparison ablation vs 1D-CNN (which uses BN in all conv blocks)
        # Runs in float32, cast back to model dtype before Mamba layers
        self.bn_embed = nn.BatchNorm1d(d_model) if use_batchnorm else None
        chunk = 64 // mimo_rank if is_mimo else 64  # 官方推荐
        self.mamba_layers = nn.ModuleList([
            Mamba3(d_model=d_model, d_state=d_state,
                   is_mimo=is_mimo, mimo_rank=mimo_rank,
                   rope_fraction=rope_fraction, chunk_size=chunk,
                   dtype=dtype)
            for _ in range(n_layers)
        ])
        # 每层前置 LN，与模型 dtype 一致（pre-norm residual，与官方 Mamba Block 一致）
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model, dtype=dtype) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)
        self.dtype = dtype

    @staticmethod
    def _extract_theta_dt(layer: Mamba3, u: torch.Tensor):
        """复刻 mamba3.py forward 的 in_proj split,取 angles/DT。
        与 kernel 内部完全一致: theta = tanh(angle)*pi, DT = softplus(dd_dt+bias)。
        注意: 这里复用 layer.in_proj 的同一次线性变换结果会更省算力,
        但官方 forward 未暴露中间量,为保持官方 kernel 不动,接受一次冗余投影
        (仅训练时计算,推理可关)。"""
        proj = layer.in_proj(u)
        sizes = [layer.d_inner, layer.d_inner,
                 layer.d_state * layer.num_bc_heads * layer.mimo_rank,
                 layer.d_state * layer.num_bc_heads * layer.mimo_rank,
                 layer.nheads, layer.nheads, layer.nheads,
                 layer.num_rope_angles]
        *_, dd_dt, _, _, angles = torch.split(proj, sizes, dim=-1)
        DT = F.softplus(dd_dt.float() + layer.dt_bias.float())   # (B, L, H)
        theta = torch.tanh(angles.float()) * torch.pi            # (B, L, K)
        return theta, DT

    def forward(self, x, return_kin=False):
        """x: (B, n_sensors, L) 原始振动波形"""
        h = self.conv_embed(x.to(self.conv_embed.weight.dtype))  # (B, D, L')
        if self.bn_embed is not None:
            h = self.bn_embed(h.float()).to(self.conv_embed.weight.dtype)
        h = h.transpose(1, 2).to(self.dtype)                     # (B, L', D)

        kin = []  # [(theta, DT), ...] per layer
        for i, (layer, ln) in enumerate(zip(self.mamba_layers, self.layer_norms)):
            h_in = ln(h)                         # pre-norm; same tensor sent to layer and _extract_theta_dt
            if return_kin:
                kin.append(self._extract_theta_dt(layer, h_in))
            h = h + layer(h_in)                  # residual connection

        h = self.norm(h.float()).mean(dim=1)
        logits = self.classifier(h)
        return (logits, kin) if return_kin else logits

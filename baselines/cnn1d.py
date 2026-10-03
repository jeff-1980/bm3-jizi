"""
baselines/cnn1d.py — 1D-CNN 基线（~101K 参数）

与 BearMamba-3 的对照协议:
  - conv_embed 层：Conv1d, kernel=8, stride=conv_stride, padding=3（无 BN/GELU，与 BM3 一致）
  - 后续 4 个双卷积块：标准 1D-CNN 结构（Conv1d + BatchNorm1d + GELU × 2 + MaxPool）
    ⚠️ BatchNorm 是 1D-CNN 的标准配置，BM3 使用 LayerNorm；两者归一化策略不同，
    BN 在批内统计归一化对低 SNR 场景有额外稳定作用，这是两模型在 CWRU -8dB 表现
    差距（10pp+）的主要机制来源之一（详见论文 Discussion §BN vs LN）。
  - 全局平均池化 → 线性分类头（非 Patch，与本组嵌入约定一致）
  - 训练协议与 BearMamba-3 完全一致（同 train.py）
  - 不支持 return_kin（纯 CE 基线，无物理先验）
"""
import torch
import torch.nn as nn


class BearCNN1D(nn.Module):
    def __init__(self, d_model: int = 64, n_layers: int = 4,
                 n_sensors: int = 1, n_classes: int = 4,
                 conv_stride: int = 2, **kwargs):
        super().__init__()
        self.conv_embed = nn.Conv1d(n_sensors, d_model, kernel_size=8,
                                    stride=conv_stride, padding=3)
        blocks = []
        for _ in range(n_layers):
            blocks.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
                nn.MaxPool1d(2),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x, return_kin: bool = False):
        x = self.conv_embed(x)          # (B, d, L/stride)
        for blk in self.blocks:
            x = blk(x)                  # (B, d, L/2^n_layers)
        x = x.mean(-1)                  # (B, d) global avg pool
        logits = self.classifier(x)
        if return_kin:
            return logits, None
        return logits

"""
bearmamba3/kinematic_loss.py — 运动学瞬时频率对齐正则（修正版 L_kin）

物理通路（侦察报告结论二/三）:
  f_{h,k}(t) = tanh(angle_k(t)) · DT_h(t) · fs_eff / 2   [Hz]
  (theta = tanh(angle)·π，代入得 f = theta·DT·fs_eff/(2π) = tanh·DT·fs_eff/2)
  L_kin 对每个样本、每个时刻生效,
  目标频率按该样本的 RPM 计算 → 天然支持变转速。

两种变体（建议消融对比）:
  - cover:  每个故障频率必须有状态与之谐振（推荐主项,不压缩表达力）
  - attract: 每个状态被拉向最近的故障频率（原计划方向,作消融对照）
"""
import torch


def compute_fault_freqs(rpm, n_balls=9, d=0.3126, D=1.537,
                        contact_angle_deg=0.0, n_harmonics=3,
                        device="cuda"):
    """标准轴承特征频率（默认 CWRU SKF 6205 驱动端参数）。
    rpm: 标量或 (B,) 张量 → 返回 (B, n_freqs) 或 (n_freqs,)
    频率集合: {fr, FTF, BPFO, 2·BSF, BPFI} × 1..n_harmonics 次谐波
    """
    rpm = torch.as_tensor(rpm, dtype=torch.float32, device=device)
    fr = rpm / 60.0
    ratio = d / D
    cos_a = torch.cos(torch.tensor(contact_angle_deg * torch.pi / 180, device=device))
    FTF  = (fr / 2) * (1 - ratio * cos_a)
    BPFO = (n_balls / 2) * fr * (1 - ratio * cos_a)
    BPFI = (n_balls / 2) * fr * (1 + ratio * cos_a)
    BSF2 = (D / d) * fr * (1 - (ratio * cos_a) ** 2)   # 2×BSF（滚动体每转两次冲击）
    base = torch.stack([fr, FTF, BPFO, BSF2, BPFI], dim=-1)        # (..., 5)
    harm = torch.arange(1, n_harmonics + 1, device=device, dtype=torch.float32)
    return (base.unsqueeze(-1) * harm).flatten(-2)                  # (..., 5*H)


def instantaneous_freqs(kin, fs_eff):
    """kin: model 返回的 [(theta(B,L,K), DT(B,L,H)), ...] → (B, n_layers·H·K, L)"""
    fs = []
    for theta, DT in kin:
        f = theta.unsqueeze(2) * DT.unsqueeze(-1) * fs_eff / (2 * torch.pi)  # (B,L,H,K)
        fs.append(f.abs().flatten(2))                                        # (B,L,H*K)
    return torch.cat(fs, dim=-1).transpose(1, 2)                             # (B,S,L)


def kinematic_loss(kin, rpm, fs_eff, variant="cover",
                   bearing_kwargs=None, eps=1e-6):
    """
    kin:     BearMamba3(x, return_kin=True) 的第二个返回值
    rpm:     (B,) 每个样本的转速 → 变转速时逐样本目标
    variant: "cover" | "attract" | "both"
    返回标量 loss（相对误差度量,跨数据集时 λ 不需重调量纲）
    """
    bearing_kwargs = bearing_kwargs or {}
    f_state = instantaneous_freqs(kin, fs_eff)                 # (B, S, L)
    f_fault = compute_fault_freqs(rpm, device=f_state.device,
                                  **bearing_kwargs)            # (B, J)

    # 相对距离矩阵: (B, S, J)（对时间取均值后再比,降低显存; 也可逐 t）
    f_bar = f_state.mean(dim=-1)                               # (B, S) 时间平均瞬时频率
    dist = (f_bar.unsqueeze(-1) - f_fault.unsqueeze(1)).abs() / (f_fault.unsqueeze(1) + eps)

    losses = {}
    losses["cover"] = dist.min(dim=1).values.mean()    # 每个故障频率找最近状态
    losses["attract"] = dist.min(dim=2).values.mean()  # 每个状态找最近故障频率
    if variant == "both":
        return losses["cover"] + 0.1 * losses["attract"]
    return losses[variant]


# ------------------------- 训练循环用法示例 -------------------------
# logits, kin = model(x, return_kin=True)            # x: (B, 3, L), rpm: (B,)
# L_ce  = F.cross_entropy(logits, labels)
# L_kin = kinematic_loss(kin, rpm, fs_eff=12000/model.conv_stride, variant="cover")
# loss  = L_ce + lambda_kin * L_kin                  # λ 扫描: {1e-3, 1e-2, 1e-1, 1.0}

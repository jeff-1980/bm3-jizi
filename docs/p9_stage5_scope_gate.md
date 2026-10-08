# 论文9# 范围门（mssp-rejection-checklist 第 0 条）正式核查：Neurocomputing

2026-10-08，对象为稿件 v8（标题分支待实验 3 确定）。

## 期刊范围（官方页面，2026-10-08 检索）
- Aims & Scope 欢迎"加深对神经网络与学习系统理解"的理论性贡献，列举的方向包括 architectures、learning methods、analysis of network dynamics。
- Guide for Authors 写明常规投稿必须与神经网络或学习系统直接相关，并且不接收"纯动力系统、纯控制、纯图像处理，或与神经网络无直接关系的现有算法的简单组合"。
- 应用方向包括 signal processing。

## 逐条核查

| 第 0 条自查动作 | 本稿情况 | 结论 |
|---|---|---|
| 编辑会把本稿读成本领域贡献，还是又一篇通用 ML 模型？ | 本稿不提出新模型，研究的是选择性 SSM（Mamba-3 块）各组件对鲁棒性的归因：选择性扫描、乘性门控、卷积前端、状态参数化。这正对应 "analysis of network dynamics / architectures" 这类理解性贡献。轴承振动只是受控测试平台 | **通过** |
| 标题 / 摘要 / cover letter 是否扣住刊物关键词 | 标题含 Mamba-3、gate/selectivity；摘要以 selective state-space models 开篇。cover letter 将显式写入 "understanding of neural network architectures" 和 "learning systems" | 通过；cover letter 已按此起草 |
| 是否是"通用深度模型只应用到轴承" | 不是。没有提出新架构，结论是关于部件的归因，并在两个迁移方向、两个宿主上做了对称干预 | **通过** |
| 是否触及"纯现有算法组合"条款 | 本稿是受控消融研究，不是算法组合。需要防止编辑把它读成"应用报告"，所以引言首段要点出 SSM 社区的开放问题（选择性是否为鲁棒性来源） | 通过；引言首段已满足 |
| 与 P8 的范围是否冲突 | P8 投 MST（传感器配置，测量学叙事），P9 投 Neurocomputing（网络部件归因）。目标刊不同，贡献也不重叠 | 通过；需在 cover letter 披露 |

## 风险与剩余动作
1. **风险等级：绿灯偏黄**。主要风险不在范围，而在"单一应用数据集"会被认为对一般性的理解贡献有限。独立数据集复现（PU，见桌查备忘）可以降低这一风险。
2. cover letter 首段写清"贡献类型 = understanding of learning systems / analysis of architectures"，不使用 "fault-diagnosis method" 的叙事。
3. Highlights 首条以 SSM 部件结论开头，不以轴承任务开头（v8 已是这样）。

**结论：范围门通过，不需要调整叙事；cover letter 已按上述要点起草（见 v8 包内 `cover_letter.tex`）。**

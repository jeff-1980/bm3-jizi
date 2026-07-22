# predproc_availability.md — 逐样本预测 / 混淆矩阵可得性探查

- checked_at: 2026-07-06T09:34 (local)
- checked_by: automated-task-run (xjtu_noisy_defense_20260627)
- scope: 四批 run — `fullgrid_fixed_20260628-2248`（"fixed"）、`frozensel_20260702-1827`（"frozensel" 主网格）、
  `secondary_20260703-1034`、`graft_20260705-224137`
- guardrail: 只读探查，未修改/删除任何 `results/` 既有文件；未训练。

## 1. 四批 run 的 `cells.jsonl` 字段探查

对四个目录逐一 `head -1 cells.jsonl` 并列出 JSON 键：

| run dir | 字段 |
|---|---|
| `results/fullgrid_fixed_20260628-2248/cells.jsonl` | `condition, noise_type, snr_db, arm, seed, best_macro_f1, n_params, elapsed_s, fairness{eval_sha256, train_order_sha256, noise_sha256}` |
| `results/frozensel_20260702-1827/cells.jsonl` | 同上（字段集合逐行一致） |
| `results/secondary_20260703-1034/cells.jsonl` | 同上 |
| `results/graft_20260705-224137/cells.jsonl` | 同上 |

**结论：四批 run 的 `cells.jsonl` 每行只落盘一个标量 `best_macro_f1`（该 cell 训练过程中各 epoch macro-F1 的最大值），
不含：**
- 逐样本 `y_true` / `y_pred`
- 每类 `logits` 或概率
- 混淆矩阵（4x4 计数）
- 甚至连 per-class F1 数组 / TP-FP-FN 计数向量都没有落盘（这些量在训练过程中被计算但只用于求 `macro_f1` 标量后丢弃，见下文 §2）

同目录下的其余产物（`decision.json`, `q1.json`/`q2.json`, `summary_table.csv`, `fairness_report.json`,
`*_curve.png`, `*_curve_table.csv`, `noise_verification.json`, `tuple_enum_report.json` 等）均为对
`best_macro_f1` 标量的二次聚合/统计产物，同样不含逐样本预测或混淆矩阵。

对全仓库 grep 未发现任何 `confusion`、`y_pred`、`y_true`、`per_sample`、`logits_cache` 等字样出现在任何 `.py` 文件中
（`grep -rl "confusion" --include="*.py" .` 无匹配）。

## 2. `xjtu_noisy_harness.py` 的 eval 函数分析

关键函数：`eval_macro_f1`（`xjtu_noisy_harness.py:305-330`）与调用它的 `train_cell`（`:349-415`）。

```
eval_macro_f1(model, loader, device):
    ... 用 tp/fn/fp (per-class, N_CLASSES 长度的 np 数组) 计算 per_f1 ...
    return per_f1, float(mean(per_f1))          # 只返回 (array, scalar)
```

`train_cell` 的训练循环（`:380-406`）每个 epoch 调一次 `eval_macro_f1`，只取回 `macro_f1` 标量参与
`best_f1 = max(best_f1, macro_f1)`；`per_f1` 数组和函数内部算出的 `tp/fn/fp` 计数在函数返回后立刻被丢弃，
**从未写入磁盘，也从未作为 `train_cell` 返回字典的一部分**（`train_cell` 返回值只有
`best_macro_f1 / elapsed_s / n_params`，见 `:411-415`）。也就是说，哪怕只是"重新算一次混淆矩阵"这种最小改动，
现有代码路径里也没有可以直接拦截的中间产物——per-class 计数在当前实现里根本不落盘、不返回。

### 2.1 是否存在 checkpoint / 可确定性重放的已存权重？

全仓库检索 `torch.save` / `state_dict()` 落盘 / `*.pt` / `*.pth` / `*checkpoint*` / `*ckpt*`：
**无任何匹配**（`bm3_models.py:65-66` 的 `load_state_dict` 是架构移植验证脚本里 in-memory 的权重拷贝，
不是训练权重的落盘保存）。`train_cell` 全程只在内存中训练模型，训练结束后模型对象随函数返回而丢弃，
**没有为任一 (arm, condition, seed) 保存过任何 checkpoint**。

### 2.2 即使重训，是否"确定性可复现"？

- `set_seed(seed)`（`:335-339`）会设置 `random`/`numpy`/`torch.manual_seed`/`torch.cuda.manual_seed_all`，
  `WeightedRandomSampler` 也用独立 generator 以同一 seed 播种（`:359-364`），这保证了**采样顺序**层面的可控性。
- 但代码中**没有**设置 `torch.backends.cudnn.deterministic = True` / `torch.use_deterministic_algorithms(True)`
  / 关闭 `cudnn.benchmark`（全文 grep `deterministic|benchmark|cudnn` 只命中两处注释性英文单词，非实际调用）。
- 更关键的是，模型（`bm3_kin`/`bm3_frozen`/`frozen_nogate`/`s4d_plus_gate` 等）依赖
  `mamba_ssm.ops.triton.mamba3.mamba3_siso_combined`（Triton 自定义核，`bm3_frozen.py:25`, `bm3_models.py:28`）。
  这类选择性扫描（selective-scan）Triton/CUDA 核的反向传播通常使用原子加（atomic add）归约，
  在 GPU 上**不是逐 bit 确定的**，即使固定所有随机种子，两次重训得到的权重轨迹也会有细微差异
  （历史上 mamba_ssm 系核心即以反向不确定著称）。

**结论：即使用相同 seed 重新跑一遍训练+eval，也不是"确定性 replay 已有实验"，而是一次新的、结果会有
（通常很小但非零）随机涨落的独立训练**。所有四批 run 共享同一个 `eval_sha256` 前缀
`6c20b367522ce5db...`（`fullgrid_fixed` 存的是截断到16位的 `6c20b367522ce5db`，
`frozensel`/`secondary`/`graft` 三批存的是完整64位 `6c20b367522ce5db59720120f8a7cd12792c7e070c6ae84bd1985511998727de`，
两者前16位一致，确认四批 run 用的是同一个 eval 集合/顺序）——重跑可以保持同一个 eval_sha（即同一测试集/样本顺序），
但不能保证与原 `cells.jsonl` 里记录的 `best_macro_f1` 逐 bit 复现。

## 3. 分支判定

- 预测是否已落盘：**否**（§1，只有标量 macro-F1）。
- 是否有 checkpoint 可确定性重放：**否**（§2.1 无 checkpoint；§2.2 即使同 seed 重训也非确定性 replay，
  受 Triton 核反向不确定性影响）。

→ **命中分支 B：预测未存 且 无 checkpoint 可确定性重放。**

按任务指令：**停，写 `needs_human`**，本次任务不进入步骤 1-3（混淆矩阵/per-class 报告/小结），
不在 loop 内启动任何重训。详见同目录 `results/predproc_needs_human_20260706-0934/needs_human.json`
中的最小成本估计与人工决策请求。

# 论文9# stage-5 判定备忘（逐条对照预注册）

- **预注册**：`recheck/prereg_stage5_metric_reverse.md`，sha256 `f3b66a835e313f8d90d11bcb6ddd0f4eb84769ed36a3f86299e84299fd289e1e`。
- **三点时间链**：
  1. 本地写定：mtime 2026-10-08 09:56:11 +0800。
  2. 推送：提交 `9750930`，提交时间 09:56:11 +0800。远端 HEAD 在第一个 cell 之前已核对。
  3. 网格启动：10:00:52。
- 跑完后复核 sha256，与写定时一致。
- **数据**：
  - 实验③：15 cell，eval 指纹 `934248343b29`，全部一致。
  - 实验②：45 cell，eval 指纹 `6c20b367522c`，全部一致。
  - 每个 cell 都是 50 epoch，逐 epoch 曲线完整。
  - 实验②三臂的参数量与原网格逐个相同：s4d_wide 178,498、frozen_noconv 112,026、frozen_As4d 112,410。
- **GPU 用时**：实验③ 1.45 h，实验② 3.60 h，合计 5.05 h，未触发 12 h 熔断。
- **分析**：脚本为 `recheck/analyze_stage5.py`，排序 bootstrap，rng 20261008，20,000 次重抽样。PYTHONHASHSEED=1 与 2 下，ratios、decisions、contrasts 三个输出逐字节一致。
- **稿件数字**：`recheck/numbers_stage5.py` 直接从逐 cell 值计算。`recheck/verify_v8.py` 独立重算表 2、3、4、表 7 反向块和敏感性均值表，结果为 0 处不符。

## 实验③：反向原生加性替换（决定性，决定标题分支）

宏 F1（%，5 种子均值，末 epoch / best-epoch）：

| 臂 | 0 dB | −2 dB | −6 dB |
|---|---|---|---|
| bm3_frozen（stage-3 反向） | 74.7 / 81.1 | 76.7 / 84.0 | 73.9 / 80.8 |
| bm3_frozen_add | 56.5 / 64.3 | 54.4 / 62.2 | 50.5 / 55.7 |
| frozen_nogate（stage-3 反向） | 56.2 / 63.8 | 54.7 / 62.0 | 50.0 / 57.1 |

配对差（末 epoch，方括号为 95% t 区间，末尾为正向种子数）：
- 加性 − 无门控：+0.3 [−0.6, 1.2] 3/5；−0.3 [−1.0, 0.4] 2/5；+0.6 [−0.2, 1.4] 5/5。三档区间都含 0，两臂逐点重合。
- 乘性 − 加性：+18.1 [1.2, 35.0] 5/5；+22.3 [3.5, 41.0] 5/5；+23.4 [6.8, 40.0] 5/5。

ρ_rev（汇总）：
- 末 epoch：0.009，95% 区间 [−0.943, 0.362]；各档 0.017、−0.014、0.023。
- best-epoch：−0.005，区间 [−1.206, 0.392]。
- 重抽样中有 0.65% 因分母小于 2 pp 剔除了一档，比例远低于 stage-4 的 7.8%。

**判定：NOT RESCUED**（ρ < 0.3 且上界 0.362 < 0.5）。加上 stage-4 的正向结果，两个方向都是 NOT RESCUED，所以按预注册走**标题分支 A**（"Necessary in Its Native Block" 系）。

## 实验②：末 epoch 复核（Table 2–4 其余臂）

| 对比（末 epoch，pp） | 0 dB | −2 dB | −6 dB | 参照 best-epoch（原网格） | flip |
|---|---|---|---|---|---|
| frozen − frozen_noconv | −13.2 [−24.1, −2.2] 0/5 | −7.5 [−14.6, −0.3] 0/5 | +5.9 [−4.0, 15.8] 3/5 | −6.6 / −2.7 / +7.9 | 无 |
| frozen − frozen_As4d | −1.9 [−9.2, 5.3] 2/5 | −2.6 [−8.6, 3.4] 2/5 | −4.8 [−16.4, 6.9] 2/5 | −2.4 / −2.2 / −4.2 | 无 |
| bm3_frozen − s4d_wide | +15.9 [5.0, 26.8] 5/5 | +20.8 [13.2, 28.4] 5/5 | +16.0 [4.6, 27.4] 4/5 | — | 3/3 档区间不含 0，**holds** |
| s4d_wide − s4d | −0.4 [−4.7, 4.0] | −1.0 [−7.7, 5.7] | +0.4 [−4.1, 4.9] | +1.8 / −1.7 / +0.5 | 仅作报告 |

**判定**：三臂都 holds，`exp2_all_hold = true`。表 2–4 改为末 epoch 主口径，best-epoch 降为敏感性分析。per-class 网格以及 clean、+10、+6、−10 dB 档标为辅助（best-epoch）。

另有一处方向不变、但末 epoch 下更明确：conv stem 的移除在 0 和 −2 dB 的区间现在不含 0，即移除 stem 在这两档更好。稿件写成"regime trade-off 被锐化，未翻转"。

## 稿件 v8 对照项目卡 §6 verb-boundary 表

| 主张 | v8 措辞 | 边界标签 |
|---|---|---|
| L1 | "does not require any input-dependent term in its scan"；冻结 Δ,A,B,C 的代价改为"at most about 4 pp"（末 epoch） | full，未越界 |
| L2 | "necessary"；"to roughly the level of S4D"，删去了 "or below"（末 epoch 下无门控臂与 S4D 的差三档区间都含 0） | full；−2 dB 档区间宽，已注明 |
| L3（原生宿主） | "the additive and gateless arms coincide at every level in both directions"，以配对差承载结论，ρ 只作汇总 | 原生宿主门控特异性由单方向升为**双向 full** |
| L3（S4D 宿主） | "recovers no more than an additive branch … only a small, direction-dependent increment" | context-specific，未变 |
| per-class | 现象部分保留，机理部分仍为假设 | 已降级（v8 讨论节两句改为未测假设） |

**标题提醒**：候选 A1 的 "Underlies" 是强动词，叠加 "Not Input Selectivity" 的对比。证据是一个块、一个测试平台，A1 读起来像一般性因果结论。A2（"Selectivity and Gating in a Mamba-3 Block"）更稳。按预注册，最终由作者选；v8 主稿暂放 A1，A2 写在 main.tex 注释里。

## DEVIATIONS

1. **"Not Transplantable" 全面弃用**：这是作者 10-08 的裁决，预注册第 3 节已写明，相对 stage-4 决策树为偏离，理由见预注册。
2. **cell 数为 60，不是任务卡写的约 45**：任务卡的实验② 列了三臂，按 3 档 × 5 种子算是 45 cell，加上实验③ 15 cell 共 60。预注册第 4 节已声明。
3. **结果查看顺序**：实验③ 的结果在实验② 跑完之前已查看并告知作者。实验② 的判定规则在预注册中已固定，不受影响。
4. **冒烟运行**：`smoke5.jsonl`、`smoke5_pair2.jsonl` 均为 2 epoch，不进入任何分析。
5. **桌查备忘里的四臂建议与任务卡不同**：PU 最小网格建议用 bm3_frozen 替换 frozen_clti，因为 ρ 与单变量 L2 都需要 bm3_frozen 作宿主。这只是建议，未开跑。
6. **附录 C 压缩**：从约 1,320 词压到约 320 词，事故全文移到补充材料 S1（`supplementary_S1_incidents.tex`），内容未删。主稿没有任何数字依赖附录 C。

# Prereg Provenance Chain — `results/secondary_20260703-1034` (component-ablation §E3)

This document materializes, with verbatim quotes and independently checked
timestamps, the "rules-before-data" chain for the secondary (conv/gate/A)
component-ablation adjudication. It supersedes nothing and does not modify
any existing `results/**` artifact; it is a new, standalone record.

**Correction to prior investigation**: an earlier round
(`.orchestrate/0704-215713/r1_executor.log`) concluded "No file, directory, or
string matching 'vault', 'vault card', or `bm3-secondary-ablation-decide`
exists anywhere in this repository" and, based on that, refused to write this
document. That search was scoped to the git repository only. The vault card
and its corroborating log entry live outside the repository, on the same
machine, at the paths quoted below — they were not searched. This document
was produced by locating and reading those files directly.

## 1. The vault card (verbatim)

Path: `/mnt/c/Users/ThinkPad/Obsidian Vault/故障诊断Wiki/_tasks/bm3-secondary-ablation-decide.md`

Frontmatter (verbatim):
```
type: orchestrate-task
title: BM3 二级消融 — conv/gate/A 部件裁决（stage-2 r2：网格 175/175 就绪，只读聚合）
status: needs_human
gate: manual
approved: true
dry_run: false
workdir: /home/jeffwork/exp/bm3-defense/xjtu_noisy_defense_20260627
max_rounds: 2
protect: results/**
created: 2026-07-03
revised: r2（2026-07-04：权威网格钉死 secondary_20260703-1034，175=5臂×7档×5seed 基线同批新跑；0703-0840 两轮 REVISE 系网格未就绪时抢跑，过期；用户误起的 secondary_20260704-0702 重复 run 已杀已删，枚举时如仍见其残留一律忽略）
depends_on: bm3-secondary-ablation-implement
rounds: 2
verdict: REVISE
guardrail_breach: false
logdir: /home/jeffwork/exp/bm3-defense/xjtu_noisy_defense_20260627/.orchestrate/0704-070402
```

The decisive rule, verbatim, from the card's `## 预注册判定规则（跑前锁死，防事后换尺）` section:

> 对每个部件臂 X ∈ {noconv, nogate, As4d}，看深噪声区（0/−2/−6dB）Δ(bm3_frozen − X)——"拆掉该部件损失多少鲁棒性"：
> - **主凶（PRIMARY_DRIVER）**：Δ(frozen−X) 深噪三档均 ≥ +8pp 且 X 塌向 s4d（X 与 s4d 深噪差 ≤ +5pp）。
> - **次要贡献（CONTRIBUTOR）**：Δ(frozen−X) 深噪三档均值 ∈ [+3, +8)pp。
> - **无关（NOT_DRIVER）**：深噪三档 |Δ| < +3pp。
> - **组合情形**：无单一 PRIMARY 而多臂 CONTRIBUTOR ⇒ `DISTRIBUTED_ACROSS_COMPONENTS`（部件协同，正当结局非失败）。
> - 判据仅用预注册 Δpp 门槛；**禁止显著性多数票或任何未预注册标准**（重演 frozensel executor 越界即直接 REVISE）。

**r2 revision transparency disclosure** (verbatim, from the `revised:` frontmatter
field quoted above): the r2 revision on 2026-07-04 did not change the decisive
rule text above ("原样未动" per the corroborating log entry, §2 below). It
only (a) pinned the authoritative grid directory to
`results/secondary_20260703-1034` (175 = 5 arms × 7 conditions × 5 seeds), (b)
disclosed that two earlier REVISE rounds at `0703-0840` had raced ahead of the
grid before it was ready and are expired, and (c) disclosed that an
accidentally-started duplicate run `secondary_20260704-0702` was killed and
deleted, and that any residue of it found during tuple enumeration should be
ignored. All three disclosures are carried forward here for transparency.

File stat (`stat`, checked 2026-07-05): last modified 2026-07-04 07:24:19
local time — this is when the r2 revision (grid-pinning + disclosures above)
was written back into the card by `vault_task_watcher.py` after the
`.orchestrate/0704-070402` round completed. This modify time reflects the r2
*revision*, not the original r1 creation; the `created: 2026-07-03` field and
the independent log.md narrative (§2) are the evidence for the original
authoring date, since this file has no git history (`Obsidian Vault` is not a
git repository).

## 2. Corroborating log.md entry (verbatim, dated 2026-07-03)

Path: `/mnt/c/Users/ThinkPad/Obsidian Vault/故障诊断Wiki/log.md`, entry
`## [2026-07-03] card | 二级消融 conv/gate/A 部件定位（两段式卡链，用户拍板补做）`:

> - **触发**：用户拍板补二级消融（§7 建议项），把"非选择性结构"从排除法收敛到具体部件。目标=论文从"证伪选择性归因"升级为"证伪+定位"（二区→冲一区）。
> - **设计核心**：三新臂全部在 **bm3_frozen 底座**上单部件再拆（非从 kin 拆）——选择性已被 frozen 排除，从 frozen 拆少一个混杂变量，单变量更纯。基线 bm3_frozen/s4d 复用 frozensel_20260702-1827 免重跑。
>   - `frozen_noconv`（去 causal conv1d 分支）/ `frozen_nogate`（去 SiLU 门控）/ `frozen_As4d`（A 参数化换 S4D 版）。
>   - **A 臂设可行性闸**：arch_map 须先判 A 参数化能否与扫描机制单独隔离；不可隔离则标 INFEASIBLE，不强造语义不清的臂。
> - **预注册裁决规则**（stage-2 锁死）：对每臂看深噪三档 Δ(frozen−X)——≥+8pp 且塌向 s4d = PRIMARY_DRIVER；[+3,+8) = CONTRIBUTOR；<+3 = NOT_DRIVER；无单点主凶但多臂 CONTRIBUTOR = DISTRIBUTED_ACROSS_COMPONENTS（协同，正当结局）。**禁显著性多数票/未预注册判据**（明写防重演 frozensel executor 越界）。
> - **落库**：[[_tasks/bm3-secondary-ablation-implement]]（stage-1，approved:true，实现+烟雾，护栏0+workdir自证+单变量 diff 白名单）+ [[_tasks/bm3-secondary-ablation-decide]]（stage-2，approved:false，只读聚合+部件裁决）。

This rule text is word-for-word consistent with the card's `## 预注册判定规则`
section quoted in §1 (same thresholds: +8pp/collapse, [+3,+8), <+3pp,
DISTRIBUTED_ACROSS_COMPONENTS). Two independently-authored documents (a task
card and a running research log) agreeing on identical numeric thresholds is
strong evidence this is a real prior decision, not a rule reverse-engineered
from the data.

The next entry in the same log, `## [2026-07-03] stage1-pass | 二级消融三臂实现全 PASS，A 臂做成（非 INFEASIBLE）`,
records (verbatim, relevant lines):

> - **stage-1 结局**：[[_tasks/bm3-secondary-ablation-implement]] verdict=PASS（2 轮，09:28）。
> - **满量**：三新臂 × 7 档 × 5 seed = 105 格（基线 frozen/s4d 复用），用户 tmux launch_secondary.sh。

i.e. stage-1 (implementation) converged to PASS at **09:28** on 2026-07-03,
*after* which the user was to launch the full grid via `launch_secondary.sh`.
The card (with its decisive rule) already existed at that point — the card
entry documenting its creation (§2, above) appears earlier in the same log,
before this 09:28 entry.

## 3. Grid start timestamp

`results/secondary_20260703-1034/` — directory name suffix gives grid start
**2026-07-03T10:34**, confirmed by the run log's first line:

```
[START] device=cuda  run_dir=results/secondary_20260703-1034
```

(`secondary_20260703-1034.log:1`). Grid completion: `[DONE] cells.jsonl
written (175 cells)` (`secondary_20260703-1034.log:356`), `cells.jsonl` mtime
`2026-07-04 03:06:01`.

## 4. Chain

card created 2026-07-03 (date; corroborated by log.md's same-day narrative
entry, itself preceding the 09:28 stage-1-PASS entry) → stage-1 PASS at
2026-07-03T09:28 → user launches grid → grid starts 2026-07-03T10:34 → grid
completes 2026-07-04T03:06.

**The rule precedes the data.** The card's decisive rule (§1) and its
log.md corroboration (§2) both predate grid start by at least ~1 hour
(09:28 → 10:34), independent of the exact time-of-day the card was first
created earlier that same morning.

## 5. Two other prereg-shaped files exist and are VOID for this adjudication

- `results/secondary_prereg_20260704-0708/secondary_prereg.json` — written
  2026-07-04T07:08, **after** grid start and grid completion; defines a
  mean±std non-overlapping-band heuristic, not the card's Δpp rule; the file
  itself discloses `grid_data_already_existed_at_write_time=true`. **VOID.**
  This is not "the" preregistration — it is a different, later, invalid rule
  an executor wrote instead of using the card's rule that already existed.
- `results/secondary_deltapp_prereg_20260704-0719/secondary_deltapp_prereg.json`
  — written 2026-07-04T07:19, also post-dates the grid; uses a flat 10pp
  threshold (not the card's 8pp/[3,8)/collapse-to-s4d structure);
  `status=DRAFT_PENDING_REVIEWER_APPROVAL`; its own text prohibits retroactive
  application to this run_dir. **VOID.**

Both are declared void not because preregistration-in-general is invalid, but
because a valid, earlier preregistration (the vault card, §1) already existed
and neither of these files is it — they were separate, later, unauthorized
substitute rules.

## 6. Why the automated reviewer never saw the card (structural, not an error)

`.orchestrate/0704-070402/r1_review.json` and `r2_review.json` both correctly
returned `REVISE`, reasoning entirely from files under `results/**`, and never
mention the card's actual Δpp thresholds. This is not reviewer error — the
review harness (`/home/jeffwork/exp/physics-mechanism/orchestrate.py`,
`run_codex_review()`, ~line 258-263) builds the reviewer's prompt as:

```
"You are a rigorous code/experiment reviewer for a PHM research project. "
"Review the current state of this working directory against the rubric "
"below. Be specific and actionable. Output ONLY JSON matching the schema.\n\n"
f"=== RUBRIC ===\n{rubric}\n"
```

The reviewer receives only the card's `## RUBRIC` section (which states the
abstract requirement "E3 裁决用预注册 Δpp 门槛" without the concrete numbers)
plus whatever it finds inspecting the workdir
`/home/jeffwork/exp/bm3-defense/xjtu_noisy_defense_20260627`. It never
receives the card's `## TASK` section (where the concrete 8pp/[3,8)pp/collapse
rule lives), and its workdir scope excludes the external vault path
`/mnt/c/Users/ThinkPad/Obsidian Vault/`. The reviewer could not, by
construction, have found or applied the card's rule. Its `REVISE` verdicts
correctly identified that the executor's *artifacts* were not a valid
preregistered adjudication — that finding stands. See `e3_human_ruling.json`
for how this is reconciled without contradicting the reviewer.

## Cross-references

- `/mnt/c/Users/ThinkPad/Obsidian Vault/故障诊断Wiki/_tasks/bm3-secondary-ablation-decide.md`
- `/mnt/c/Users/ThinkPad/Obsidian Vault/故障诊断Wiki/log.md` (2026-07-03 entries)
- `results/secondary_20260703-1034/` (`secondary_20260703-1034.log`, `cells.jsonl`)
- `results/secondary_prereg_20260704-0708/secondary_prereg.json` (void)
- `results/secondary_deltapp_prereg_20260704-0719/secondary_deltapp_prereg.json` (void)
- `.orchestrate/0704-070402/r1_review.json`, `.orchestrate/0704-070402/r2_review.json`
- `/home/jeffwork/exp/physics-mechanism/orchestrate.py` (`run_codex_review`, review-prompt construction)
- `results/secondary_adjudication_20260705-1610/secondary_decision.json` (this chain's mechanical adjudication)
- `results/secondary_adjudication_20260705-1610/e3_human_ruling.json`
- `results/secondary_e3ruling_20260704-2206/prereg_provenance.md` (prior, more conservative provenance doc — not superseded, kept as-is; that document correctly stated no valid prereg was *found in the repo*, which was true at the time, since the vault card was outside its search scope)

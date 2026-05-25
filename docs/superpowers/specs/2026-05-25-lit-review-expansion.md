# 文献调研扩展（4 个补充方向）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 通过 4 个并行 subagent，为 SafeComm-PSched 算法的各核心组件识别可整合的新技术方案，输出 4 份 ~300-400 行格式标准化的文献调研报告，并更新综合报告。

**Architecture:** 4 个 subagent 完全并行执行，各自使用 `local-research-skills:literature-review` 技能，独立写入 `docs/lit_review/05~08_.md`；全部完成后主线程汇总更新 `00_synthesis_report.md`。

**Tech Stack:** local-research-skills:literature-review（网络搜索 + 学术数据库）、Markdown 格式输出、git commit 每个文件

---

## 文件结构

| 操作 | 文件路径 | 负责方 |
|------|----------|--------|
| 创建 | `docs/lit_review/05_multi_objective_pareto_marl.md` | Agent A |
| 创建 | `docs/lit_review/06_cbf_mpc_shielding_safe_marl.md` | Agent B |
| 创建 | `docs/lit_review/07_graph_attention_scalable_marl.md` | Agent C |
| 创建 | `docs/lit_review/08_sim2real_uav_marl.md` | Agent D |
| 修改 | `docs/lit_review/00_synthesis_report.md` | 主线程（Task 2） |

---

## Task 1：4 个方向并行文献调研（单消息同时派发）

> **执行说明**：此 Task 的 4 个 subagent 必须在同一条消息中同时启动（Agent 工具并行调用），不得串行。

### Task 1A：Multi-objective / Pareto MARL 调研

**文件：**
- 创建：`docs/lit_review/05_multi_objective_pareto_marl.md`

- [ ] **步骤 1A-1：启动 subagent，使用如下完整提示词**

```
你是一名研究助理，任务是对 "Multi-objective / Pareto MARL" 方向进行系统性文献调研，并将结果写入 /root/autodl-tmp/safecomm-marl/docs/lit_review/05_multi_objective_pareto_marl.md。

背景信息：
本项目（SafeComm-MARL）开发了一个安全感知通信调度框架（SafeComm-PSched），使用单约束 Lagrangian 方法（MAPPO + 单个 λ_s）处理碰撞安全约束。核心问题：多目标/Pareto 方法能否替代或增强这个 Lagrangian 设计？

调研关键词（用于数据库检索）：
- "multi-objective multi-agent reinforcement learning"
- "Pareto MARL" / "MOMARL"
- "multi-objective PPO" / "multi-objective constrained RL"
- "Pareto front reinforcement learning safety"
- "scalarization multi-agent RL"
- "reward shaping vs Lagrangian constrained RL"
- "multi-objective safe reinforcement learning 2022 2023 2024"

核心聚焦问题（报告必须回答）：
1. 现有多目标/Pareto MARL 方法与单约束 Lagrangian 相比，有哪些实质性的理论或实践优势？
2. 在 UAV 编队（任务性能 vs 碰撞安全）这类二目标场景中，Pareto 前沿方法 vs Lagrangian，谁更适合？为什么？
3. 是否存在可操作的混合方案（如 Pareto 权重搜索替代 Lagrangian 乘子自适应、多目标 actor-critic）？
4. 在通信受限场景下，多目标优化是否比单目标 Lagrangian 更难还是更容易扩展？

与 SafeComm 关联组件：Lagrangian 安全优化模块（safecomm_psched.py 中的 λ_s 更新规则）

输出格式要求（严格遵守，参考现有文件 docs/lit_review/01_safe_marl.md 的格式）：
1. 概述：方向定义、与 SafeComm 的关联（~1 页）
2. 搜索策略表：关键词 + 数据来源 + 说明
3. 文献分类表：按子方向分组（如：纯多目标 RL、Pareto 前沿方法、约束 RL vs 多目标 RL 比较、应用到机器人/UAV 的工作），含论文名/作者/年份/核心贡献/与 SafeComm 相关度
4. 主要论文详评（至少 6 篇，每篇格式：核心方法、通信假设（如适用）、理论保证、应用场景、与本研究的关系）
5. 与 SafeComm-PSched 整合分析：
   - 可直接替换 Lagrangian 的方案（Drop-in replacement）
   - 可增强 Lagrangian 的方案（Augmentation）
   - 暂不适用但值得跟踪的工作
6. 关键发现汇总：4-5 条可操作结论，明确说明对 SafeComm 设计的影响

报告长度：~300-400 行 Markdown

完成后请用 git 提交：
git add docs/lit_review/05_multi_objective_pareto_marl.md
git commit -m "docs: 添加方向05文献调研——Multi-objective/Pareto MARL"
```

- [ ] **步骤 1A-2：确认文件已创建且 ≥300 行**

```bash
wc -l /root/autodl-tmp/safecomm-marl/docs/lit_review/05_multi_objective_pareto_marl.md
```
预期：`≥300 /root/autodl-tmp/safecomm-marl/docs/lit_review/05_multi_objective_pareto_marl.md`

---

### Task 1B：CBF / MPC / Shielding for Safe MARL 调研

**文件：**
- 创建：`docs/lit_review/06_cbf_mpc_shielding_safe_marl.md`

- [ ] **步骤 1B-1：启动 subagent，使用如下完整提示词**

```
你是一名研究助理，任务是对 "CBF / MPC / Shielding for Safe MARL" 方向进行系统性文献调研，并将结果写入 /root/autodl-tmp/safecomm-marl/docs/lit_review/06_cbf_mpc_shielding_safe_marl.md。

背景信息：
本项目（SafeComm-MARL）的安全感知调度器使用距离型 CBF 值 h_ij(t) = (||p_i - p_j|| - d_min)^2 来评估危险度，并据此优先分配通信链路（TopK 调度）。SafeComm 目前使用 Lagrangian 软约束，而非硬 CBF 约束层。核心问题：CBF、MPC、Shielding 是否提供了比 Lagrangian 更强的安全保证，且在通信受限场景下可行？

调研关键词（用于数据库检索）：
- "control barrier function multi-agent reinforcement learning"
- "CBF safe multi-agent" / "neural CBF MARL"
- "model predictive control safe reinforcement learning"
- "MPC safety layer neural network policy"
- "runtime shielding multi-agent RL"
- "safe exploration MARL CBF"
- "distributionally robust CBF"
- "CBF communication constraint" / "CBF partial observation"
- "multi-agent shielding LTL safety"
- "safe MARL hard constraint 2022 2023 2024"

核心聚焦问题（报告必须回答）：
1. CBF 硬约束层、MPC 安全层、运行时 Shielding 各自在 MARL 中提供了什么级别的安全保证（硬/软/概率）？与 Lagrangian 相比的本质区别？
2. CBF 在多智能体场景下通常需要邻居状态（h_ij 依赖 p_j）——现有工作如何在通信受限/部分观测时处理这个问题？（估计？退化？保守近似？）
3. 对 SafeComm-PSched 的具体意义：CBF 值既用于调度（scheduler.py），也可用于硬约束层——是否有工作将调度决策与 CBF 安全层联合优化？
4. 哪些工作在实际 UAV 或多机器人系统上验证了 CBF/MPC/Shielding？迁移到真实系统的主要挑战？

与 SafeComm 关联组件：安全感知调度器 scheduler.py（CBF 值计算）、safecomm_psched.py（安全约束处理）

输出格式要求（严格遵守，参考现有文件 docs/lit_review/01_safe_marl.md 的格式）：
1. 概述：三种安全机制（CBF/MPC/Shield）的比较定义、与 SafeComm 的关联（~1 页）
2. 搜索策略表：关键词 + 数据来源 + 说明
3. 文献分类表：按子方向分组（CBF for MARL、MPC 安全层、运行时 Shielding、通信受限下的安全方法），含论文名/作者/年份/核心贡献/与 SafeComm 相关度
4. 主要论文详评（至少 6 篇，每篇格式：核心方法、通信假设、理论保证、应用场景、与本研究的关系）
5. 与 SafeComm-PSched 整合分析：
   - CBF 值与调度器的联合设计可能性
   - 替换 Lagrangian 为硬 CBF 层的可行性分析
   - 通信受限下的 CBF 近似方案
6. 关键发现汇总：4-5 条可操作结论，明确说明对 SafeComm 设计的影响

报告长度：~300-400 行 Markdown

完成后请用 git 提交：
git add docs/lit_review/06_cbf_mpc_shielding_safe_marl.md
git commit -m "docs: 添加方向06文献调研——CBF/MPC/Shielding for Safe MARL"
```

- [ ] **步骤 1B-2：确认文件已创建且 ≥300 行**

```bash
wc -l /root/autodl-tmp/safecomm-marl/docs/lit_review/06_cbf_mpc_shielding_safe_marl.md
```
预期：`≥300 /root/autodl-tmp/safecomm-marl/docs/lit_review/06_cbf_mpc_shielding_safe_marl.md`

---

### Task 1C：Graph / Attention / Scalable MARL 调研

**文件：**
- 创建：`docs/lit_review/07_graph_attention_scalable_marl.md`

- [ ] **步骤 1C-1：启动 subagent，使用如下完整提示词**

```
你是一名研究助理，任务是对 "Graph / Attention / Scalable MARL" 方向进行系统性文献调研，并将结果写入 /root/autodl-tmp/safecomm-marl/docs/lit_review/07_graph_attention_scalable_marl.md。

背景信息：
本项目（SafeComm-MARL）的消息编码与聚合模块（MessageEncoder）使用单头注意力（d_msg=64）聚合来自 TopK 邻居的消息，通信图由安全感知调度器动态决定（每步最多 k 条活跃链路）。核心问题：现有 GNN/Attention 架构是否提供比单头注意力更好的通信拓扑感知和消息聚合质量？

调研关键词（用于数据库检索）：
- "graph neural network multi-agent reinforcement learning"
- "DGN deep graph network multi-agent" / "MAAC multi-agent actor-critic attention"
- "MAGIC multi-agent graph attention communication"
- "scalable multi-agent reinforcement learning graph"
- "dynamic graph topology multi-agent RL"
- "communication graph learning MARL"
- "multi-head attention multi-agent policy"
- "graph convolutional network cooperative MARL"
- "mean field theory scalable MARL"
- "scalable MARL large number agents 2022 2023 2024"

核心聚焦问题（报告必须回答）：
1. DGN、MAAC、MAGIC 等代表性工作与单头注意力相比，在通信拓扑感知和消息聚合上的实质改进是什么（不仅是描述，要分析原理）？
2. 哪些图架构在智能体数量 n>8 时表现出明确的扩展性优势（实验数据支持）？
3. 是否有方法同时支持动态图拓扑（类似 TopK 每步切换）且不引入训练不稳定性？
4. Mean Field 近似在大规模 MARL 中的作用——能否与 SafeComm 的通信受限设置兼容？
5. 对 SafeComm 的 MessageEncoder 设计：是否有现成的 drop-in 替换方案比单头注意力显著更好？

与 SafeComm 关联组件：MessageEncoder 网络（algorithms/networks.py）、通信图结构

输出格式要求（严格遵守，参考现有文件 docs/lit_review/01_safe_marl.md 的格式）：
1. 概述：GNN/Attention/Scalable MARL 的关系图谱、与 SafeComm 的关联（~1 页）
2. 搜索策略表：关键词 + 数据来源 + 说明
3. 文献分类表：按子方向分组（图神经网络 MARL、多头注意力方法、可扩展架构、动态拓扑方法），含论文名/作者/年份/核心贡献/与 SafeComm 相关度
4. 主要论文详评（至少 6 篇，每篇格式：核心方法、通信拓扑假设、可扩展性、应用场景、与本研究的关系）
5. 与 SafeComm-PSched 整合分析：
   - 可直接替换单头注意力的方案（Drop-in replacement，并给出改动估计）
   - 对动态 TopK 拓扑的兼容性分析
   - n>8 时推荐的架构选择
6. 关键发现汇总：4-5 条可操作结论，明确说明对 MessageEncoder 设计的影响

报告长度：~300-400 行 Markdown

完成后请用 git 提交：
git add docs/lit_review/07_graph_attention_scalable_marl.md
git commit -m "docs: 添加方向07文献调研——Graph/Attention/Scalable MARL"
```

- [ ] **步骤 1C-2：确认文件已创建且 ≥300 行**

```bash
wc -l /root/autodl-tmp/safecomm-marl/docs/lit_review/07_graph_attention_scalable_marl.md
```
预期：`≥300 /root/autodl-tmp/safecomm-marl/docs/lit_review/07_graph_attention_scalable_marl.md`

---

### Task 1D：Simulation-to-Reality Stack for UAV MARL 调研

**文件：**
- 创建：`docs/lit_review/08_sim2real_uav_marl.md`

- [ ] **步骤 1D-1：启动 subagent，使用如下完整提示词**

```
你是一名研究助理，任务是对 "Simulation-to-Reality Stack for UAV MARL" 方向进行系统性文献调研，并将结果写入 /root/autodl-tmp/safecomm-marl/docs/lit_review/08_sim2real_uav_marl.md。

背景信息：
本项目（SafeComm-MARL）分两阶段仿真：Phase 1 使用轻量双积分器仿真（envs/uav_formation_env.py），Phase 2 迁移到 PyBullet 四旋翼仿真（envs/pybullet_uav_env.py），最终目标是部署到真实 UAV 硬件。核心问题：从双积分器到 PyBullet 到真实 UAV 的迁移过程中，主要障碍是什么？现有文献提供了哪些可操作的解决方案？

调研关键词（用于数据库检索）：
- "sim-to-real UAV reinforcement learning"
- "domain randomization UAV quadrotor"
- "simulation to reality transfer multi-robot MARL"
- "PyBullet UAV quadrotor reinforcement learning"
- "domain randomization multi-agent"
- "adaptive domain randomization UAV"
- "system identification neural network UAV"
- "reality gap UAV neural policy"
- "zero-shot sim2real transfer quadrotor"
- "Gazebo ROS UAV reinforcement learning sim2real 2022 2023 2024"

核心聚焦问题（报告必须回答）：
1. 多 UAV MARL 的 Sim2Real 迁移的主要障碍是什么（动力学差距、通信延迟建模、传感器噪声、多机气动耦合）？各障碍的相对重要性如何？
2. 域随机化（DR）、自适应 DR（ADR/ARC）、对抗性强化学习（RARL）、系统辨识（SysID）在 UAV 场景中的适用性比较——哪个方法在多机场景效果最好？
3. 从双积分器仿真到 PyBullet 四旋翼仿真，需要增加哪些物理参数的随机化（电机推力曲线、转动惯量、空气阻力系数）？现有最佳实践是什么？
4. 通信延迟和丢包在真实 UAV 群中不可避免——哪些工作在 Sim2Real 过程中显式建模了通信不确定性？
5. 从 PyBullet 到真实 UAV 硬件，现有工作的典型实验流程（仿真精度要求、硬件测试协议、回退机制）？

与 SafeComm 关联组件：envs/pybullet_uav_env.py（Phase 2 仿真）、training configs（超参数随机化范围）

输出格式要求（严格遵守，参考现有文件 docs/lit_review/01_safe_marl.md 的格式）：
1. 概述：Sim2Real 流程的层次定义（双积分器→PyBullet→真实 UAV）、与 SafeComm 的关联（~1 页）
2. 搜索策略表：关键词 + 数据来源 + 说明
3. 文献分类表：按子方向分组（域随机化方法、自适应 DR、系统辨识、通信不确定性建模、多机 Sim2Real），含论文名/作者/年份/核心贡献/与 SafeComm 相关度
4. 主要论文详评（至少 6 篇，每篇格式：核心方法、随机化策略/仿真配置、实验结果（迁移成功率/性能保留率）、应用场景、与本研究的关系）
5. 与 SafeComm-PSched 整合分析：
   - Phase 1→Phase 2 迁移的推荐随机化参数列表
   - Phase 2→真实硬件的推荐验证流程
   - 通信延迟/丢包的仿真建模方案
6. 关键发现汇总：4-5 条可操作结论，明确说明对 envs/pybullet_uav_env.py 和 configs/ 的设计影响

报告长度：~300-400 行 Markdown

完成后请用 git 提交：
git add docs/lit_review/08_sim2real_uav_marl.md
git commit -m "docs: 添加方向08文献调研——Sim2Real UAV MARL"
```

- [ ] **步骤 1D-2：确认文件已创建且 ≥300 行**

```bash
wc -l /root/autodl-tmp/safecomm-marl/docs/lit_review/08_sim2real_uav_marl.md
```
预期：`≥300 /root/autodl-tmp/safecomm-marl/docs/lit_review/08_sim2real_uav_marl.md`

---

## Task 2：更新综合报告

> **前置条件**：Task 1 的全部 4 个 subagent 均已完成并提交。

**文件：**
- 修改：`docs/lit_review/00_synthesis_report.md`

- [ ] **步骤 2-1：读取 4 个新报告的关键发现**

```bash
grep -A 30 "关键发现" /root/autodl-tmp/safecomm-marl/docs/lit_review/05_multi_objective_pareto_marl.md
grep -A 30 "关键发现" /root/autodl-tmp/safecomm-marl/docs/lit_review/06_cbf_mpc_shielding_safe_marl.md
grep -A 30 "关键发现" /root/autodl-tmp/safecomm-marl/docs/lit_review/07_graph_attention_scalable_marl.md
grep -A 30 "关键发现" /root/autodl-tmp/safecomm-marl/docs/lit_review/08_sim2real_uav_marl.md
```

- [ ] **步骤 2-2：在 `00_synthesis_report.md` 末尾追加新方向汇总章节**

在文件末尾（`*报告生成时间*` 行之前）添加以下章节，内容根据步骤 2-1 的实际读取结果填写：

```markdown
---

## 三、补充方向调研摘要（2026-05-25 扩展）

### 方向5：Multi-objective / Pareto MARL
（从 05_.md 的关键发现汇总中提取 3-4 条，聚焦对 Lagrangian 模块的影响）

### 方向6：CBF / MPC / Shielding for Safe MARL
（从 06_.md 的关键发现汇总中提取 3-4 条，聚焦对调度器 CBF 值计算的影响）

### 方向7：Graph / Attention / Scalable MARL
（从 07_.md 的关键发现汇总中提取 3-4 条，聚焦对 MessageEncoder 设计的影响）

### 方向8：Simulation-to-Reality Stack
（从 08_.md 的关键发现汇总中提取 3-4 条，聚焦对 Phase 2 仿真配置的影响）

### 更新后的设计建议矩阵

| SafeComm 组件 | 当前设计 | 文献支持的改进方向 | 优先级 |
|--------------|---------|-----------------|--------|
| Lagrangian 模块 | 单约束 λ_s | （来自方向5）| 中 |
| 安全调度器 CBF 值 | 距离型 h_ij | （来自方向6）| 高 |
| MessageEncoder | 单头注意力 | （来自方向7）| 中 |
| Phase 2 仿真 | PyBullet 四旋翼 | （来自方向8）| 低（后期） |
```

- [ ] **步骤 2-3：提交更新**

```bash
git add docs/lit_review/00_synthesis_report.md
git commit -m "docs: 更新综合报告——补充4个新方向关键发现"
```

- [ ] **步骤 2-4：验证最终文件状态**

```bash
ls -la /root/autodl-tmp/safecomm-marl/docs/lit_review/
git log --oneline -6
```

预期：共 9 个文件（00~08），最近 5 条提交对应 4 个新文件 + 1 个综合报告更新。

---

## 成功标准核查

- [ ] `05_multi_objective_pareto_marl.md` ≥300 行，含 ≥6 篇详评 + SafeComm 整合分析
- [ ] `06_cbf_mpc_shielding_safe_marl.md` ≥300 行，含 ≥6 篇详评 + SafeComm 整合分析
- [ ] `07_graph_attention_scalable_marl.md` ≥300 行，含 ≥6 篇详评 + SafeComm 整合分析
- [ ] `08_sim2real_uav_marl.md` ≥300 行，含 ≥6 篇详评 + SafeComm 整合分析
- [ ] `00_synthesis_report.md` 包含新章节"补充方向调研摘要"和更新后的设计建议矩阵
- [ ] git log 显示 5 条新提交

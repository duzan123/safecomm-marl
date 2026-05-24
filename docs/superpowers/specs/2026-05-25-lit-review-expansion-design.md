# 文献调研扩展规格：4 个补充方向

**日期：** 2026-05-25  
**版本：** v1.0  
**关联规格：** `2026-05-25-safecomm-marl-design.md`

---

## 1. 背景与目标

当前仓库已完成 4 个基础方向的文献调研（`docs/lit_review/01~04_.md`，共约 50 篇），覆盖 Safe MARL、通信受限 MARL、安全×通信交叉、多 UAV 编队控制。

本次调研扩展面向 **4 个补充技术方向**，目标是：

> **识别可整合进 SafeComm-PSched 各核心组件的新技术方案**，并为论文 Related Work 提供系统化覆盖。

---

## 2. 调研方向与核心聚焦问题

### 方向 05：Multi-objective / Pareto MARL

**输出文件：** `docs/lit_review/05_multi_objective_pareto_marl.md`

**核心聚焦问题：**
- 现有多目标/Pareto MARL 方法（如 MOMARL、帕累托前沿优化、多目标 PPO/SAC）能否替代或增强 SafeComm-PSched 当前的单约束 Lagrangian 设计？
- 与 Lagrangian 单约束方法相比，多目标方法在 UAV 编队（任务性能 vs 安全）场景下的实际优劣是什么？
- 是否存在可操作的混合方案（例如：用 Pareto 权重搜索替代 Lagrangian 乘子自适应）？

**与 SafeComm 的关联组件：** Lagrangian 安全优化模块（`algorithms/safecomm_psched.py` §Lagrangian 部分）

---

### 方向 06：CBF / MPC / Shielding for Safe MARL

**输出文件：** `docs/lit_review/06_cbf_mpc_shielding_safe_marl.md`

**核心聚焦问题：**
- CBF（Control Barrier Function）、MPC 安全层、运行时 Shielding 在多智能体场景下，相比 Lagrangian 软约束，提供了什么样的硬/概率安全保证？
- 现有 CBF-MARL、Shield-MARL 工作在通信受限场景下是否有实现可行性（CBF 通常需要邻居状态——如何在通信预算下估计）？
- 对 SafeComm-PSched 的调度器（CBF 值计算）和安全约束模块有何具体可整合之处？

**与 SafeComm 的关联组件：** 安全感知调度器（`algorithms/scheduler.py`）、CBF 值计算逻辑

---

### 方向 07：Graph / Attention / Scalable MARL

**输出文件：** `docs/lit_review/07_graph_attention_scalable_marl.md`

**核心聚焦问题：**
- 现有图神经网络（GNN）和注意力机制在可扩展 MARL 中（如 GPG、DGN、MAAC、MAGIC 等），相比 SafeComm 当前的 TopK + 单头注意力设计，在通信拓扑感知和消息聚合质量上有哪些具体改进？
- 哪些图架构在智能体数量 $n$ 增大时（$n > 8$）表现出更好的扩展性？
- 是否有方法同时考虑图拓扑的动态变化（类似 TopK 切换）而不引入不稳定性？

**与 SafeComm 的关联组件：** 消息编码与聚合网络（`algorithms/networks.py` MessageEncoder）

---

### 方向 08：Simulation-to-Reality Stack for UAV MARL

**输出文件：** `docs/lit_review/08_sim2real_uav_marl.md`

**核心聚焦问题：**
- 多 UAV MARL 的 Sim2Real 迁移主要障碍是什么（动力学差距、通信延迟建模、感知噪声、多机耦合）？
- 域随机化（Domain Randomization）、自适应方法（Adaptive DR、RARL）、系统辨识（System Identification）在 UAV MARL 场景中的适用性和最佳实践？
- 从双积分器仿真（Phase 1）过渡到 PyBullet 四旋翼（Phase 2）再到真实硬件，现有文献提供了哪些可操作的迁移流程？

**与 SafeComm 的关联组件：** `envs/pybullet_uav_env.py`（Phase 2）、训练配置（`configs/`）

---

## 3. 执行方案

### 3.1 并行策略

采用 **4 subagent 完全并行**，各方向互相独立无依赖关系，通过 `local-research-skills:literature-review` 技能执行。

```
主线程
  ├── Agent A → 方向 05（Multi-obj）
  ├── Agent B → 方向 06（CBF/MPC/Shield）
  ├── Agent C → 方向 07（Graph/Attention）
  └── Agent D → 方向 08（Sim2Real）
        ↓（全部完成后）
  更新 00_synthesis_report.md（补充矩阵行）
```

### 3.2 每个 subagent 的输入规格

每个 agent 接收：
- **调研方向名称**（用于数据库检索关键词）
- **核心聚焦问题**（引导调研深度和侧重）
- **与 SafeComm 组件的对应关系**（确保输出具有可操作性）
- **输出格式要求**：与现有 `01~04_.md` 一致（系统化分类表 + 主要论文详评 + 整合建议，~300-400 行）

### 3.3 输出格式规范（与现有文件一致）

每个报告必须包含：

1. **概述**：方向定义、与 SafeComm 的关联
2. **文献分类表**：按子方向分组，含论文名/作者/年份/核心贡献/与 SafeComm 相关度
3. **主要论文详评**（5-8 篇）：方法、结果、局限、整合可能性
4. **与 SafeComm-PSched 整合分析**：
   - 可直接替换的组件（Drop-in replacement）
   - 可增强的组件（Augmentation）
   - 暂不适用但值得跟踪的工作
5. **关键发现汇总**：3-5 条可操作结论

---

## 4. 后处理：更新综合报告

所有 4 个 subagent 完成后，将新方向的关键发现整合进 `docs/lit_review/00_synthesis_report.md`，扩展研究现状矩阵和创新定位分析。

---

## 5. 成功标准

- [ ] `05~08_.md` 四个文件各 ~300-400 行，格式与现有文件一致
- [ ] 每个文件包含 ≥5 篇具体论文详评
- [ ] 每个文件包含针对 SafeComm-PSched 具体组件的整合建议
- [ ] `00_synthesis_report.md` 更新反映新方向发现

# SafeComm-VoI 项目总体协调计划

> **给 Claude 的使用说明**：本文档是 Claude 的项目协调手册。Codex 执行具体代码任务，Claude 负责阶段审查、架构决策、阻塞诊断。遇到阶段切换、架构疑问、任务阻塞时，请将本文档与相关上下文一起提供给 Claude。

**算法**：SafeComm-VoI（Safety-aware Value-of-Information Multi-Agent RL）  
**目标期刊**：NeurIPS / ICML / ICLR  
**问题**：CDecPOSG-CB — 通信受限多 UAV 编队控制中安全约束与通信预算的联合优化  
**设计规格**：`docs/superpowers/specs/2026-05-26-safecomm-voi-design.md`

---

## 1. 阶段路线图

| 阶段 | 名称 | 状态 | 前置条件 | 详细计划文件 |
|------|------|------|----------|------------|
| Phase 1 | Scheduler + Prototype | ⬜ 待完成 | — | `docs/superpowers/plans/2026-05-26-safecomm-voi-phase1.md` |
| Phase 2 | Full SafeComm-VoI | ⬜ 待完成 | Phase 1 ✅ | `docs/superpowers/plans/2026-05-26-safecomm-voi-phase2.md` |
| Phase 3 | PyBullet + Domain Rand. | ⬜ 待完成 | Phase 2 ✅ | `docs/superpowers/plans/2026-05-26-safecomm-voi-phase3.md` |
| Phase 4 | 多任务 + N≥8 扩展 | ⬜ 待完成 | Phase 3 ✅ | `docs/superpowers/plans/2026-05-26-safecomm-voi-phase4.md` |

**阶段切换规则**：每个阶段的门控标准（第3节）全部通过后，Claude 出具 ✅ 结论，方可开始下一阶段。

---

## 2. Codex 工作流程

每次给 Codex 一个任务时，使用以下提示模板：

```
### 任务上下文

项目：SafeComm-VoI，Python 3.10 + PyTorch，UAV 编队 MARL
当前阶段：Phase [X]
工作目录：/root/autodl-tmp/safecomm-marl/

已完成的前置任务：
- [列出本任务依赖的已完成任务]

架构约束（必须遵守）：
见 docs/superpowers/specs/2026-05-26-safecomm-voi-master-coordination.md 第4节

### 当前任务

[直接复制对应 Phase 计划文件中的完整任务文本，包含步骤和代码]

### 成功标准

- [复制该任务的测试命令和预期输出]
- git commit -m "feat: ..." 完成提交
```

**重要**：给 Codex 的任务文本必须完整复制计划文件中的任务内容（含代码片段），不要只给任务标题。

---

## 3. 各阶段门控标准

Claude 在阶段切换前逐条核查以下清单。全部通过才出具 ✅ 结论。

### Phase 1 → Phase 2 门控

**自动化测试**：
- [ ] `pytest tests/test_scheduler.py -v` 全部通过（含 `test_voi_priority_formula`、`test_budget_constraint`）
- [ ] `pytest tests/test_env.py -v` 全部通过
- [ ] `pytest tests/test_algorithm.py::test_phase1_smoke -v` 通过

**行为验证**（读代码 + 询问用户）：
- [ ] 每步确保 `|E_t| ≤ k`：`scheduler.select_edges()` 使用 TopK，输出大小有界
- [ ] β=0 vs β>0：在多 agent 拥挤场景中，β>0 时 AoI 老化 pair 的优先级高于 β=0
- [ ] λ_s 更新方向：`J_safe > d_safe` 时 λ_s 严格上升（从 train.py Lagrangian 更新逻辑确认）

**Claude 读文件确认**：
- `algorithms/scheduler.py`：VoI 公式与设计规格 §2.1 数学一致
- `algorithms/safecomm_psched.py`：Lagrangian 更新逻辑（`λ_s ← max(0, λ_s + α_s(J_safe - d_safe))`）

---

### Phase 2 → Phase 3 门控

**自动化测试**：
- [ ] `pytest tests/test_hocbf_qp.py -v` 全部通过（含不可行性处理、slack 激活测试）
- [ ] `pytest tests/test_baselines.py -v` 全部通过（6 个 baseline）
- [ ] `pytest tests/ -v` 全套通过（Phase 1 测试不得退化）

**行为验证**：
- [ ] RolloutBuffer 同时存储 `a_pi` 和 `a_star`：`buffer.int_costs` shape 为 `(T, n_agents)`
- [ ] λ_int 仅在 safe_gate 激活时更新（EMA 平滑，非单 rollout 触发）
- [ ] evaluate.py 支持 `--shield on` 和 `--shield off` 两种模式

**Claude 读文件确认**：
- `algorithms/hocbf_qp.py`：HOCBF 约束公式（`2*r^T*a + 2‖v‖² + α1*dh + α2*ψ1 - m_acc + ξ ≥ 0`）
- `algorithms/safecomm_psched.py` collect_rollout：dual-action 分支正确，`a_pi` 和 `a_star` 不混用
- `algorithms/safecomm_psched.py` update()：MGDA 两阶段顺序正确，safety gate EMA 实现正确

---

### Phase 3 → Phase 4 门控

**自动化测试**：
- [ ] `pytest tests/test_pybullet_env.py -v` 全部通过（≥14 个测试）
- [ ] `pytest tests/ -v` 全套通过（Phase 1-2 测试不得退化）

**行为验证**：
- [ ] PyBullet 环境 obs_dim 与 UAVFormationEnv 一致（相同 `K_obs` 时 obs shape 相同）
- [ ] `envs/pybullet_uav_env.py` 封装 `gym-pybullet-drones` 的 Aviary 类，未手写四旋翼 RPM/推力/力矩动力学
- [ ] DR 包装器：Scheduler 仍接收 real positions（未噪声化），噪声只注入 agent obs
- [ ] Phase 2 训练好的 checkpoint 在 PyBullet 环境中可加载并运行 100 步无 error

**Claude 读文件确认**：
- `envs/domain_randomization.py`：噪声注入点在 `step()` 返回的 obs，不影响 `get_physical_graph()` 或 CBF 计算

---

### Phase 4 完成门控

**自动化测试**：
- [ ] `pytest tests/test_pursuit_env.py -v` 全部通过
- [ ] `pytest tests/test_scalable_critic.py -v` 全部通过（N=4, 6, 8 三种规模）
- [ ] `pytest tests/ -v` 全套通过

**行为验证**：
- [ ] ScalableCritic N-agnostic：相同权重对 N=4 和 N=8 均可前向推断，输出 shape `(batch, 1)`
- [ ] `train_multitask.py` 在 formation + pursuit 交替训练中运行 200 步无 error

**Claude 读文件确认**：
- `algorithms/scalable_critic.py`：接口 `critic(obs_all: Tensor[B, N, obs_dim]) → Tensor[B, 1]` 与现有主类兼容

---

## 4. 架构约束清单

Codex 实现时必须遵守以下约束。如遇到与约束矛盾的技术决策，需先咨询 Claude。

**约束 1：双动作 rollout（核心不变项）**  
环境执行 `a_star`（QP 过滤后），PPO log probability 使用 `a_pi`（策略原始输出）。两者必须分开存储在 RolloutBuffer 中，不可合并或混用。

**约束 2：CBF 缓冲区单调性**  
`d_buf_ij = v_max × Δt_comm_ij`，随通信缺失时间严格单调增。QP 使用 `d_eff = d_min + d_buf`；调度器 `h_sched` 也使用同一 `d_buf`。两处必须使用完全相同的公式。

**约束 3：VoI 优先级公式**  
`priority_ij = (1 + β × min(Δt_comm/τ_ref, τ_max)) / max(h_sched_ij, ε)`  
`h_sched_ij = max(‖p_i − p̂_j‖ − d_min − d_buf_ij, 0)²`  
h_sched clip 到 0（保证碰撞区内优先级最高）。公式不得简化或修改。

**约束 4：通信历史更新时机**  
仅在被选入 `E_t` 的 pair 才更新 `(p̂_j, v̂_j, Δt_comm = 0)`；未选入的 pair 只递增 `Δt_comm += dt`。不得在其他任何时刻修改通信历史。

**约束 5：QP slack 变量保留**  
`ξ_i ≥ 0` 必须保留在所有 QP 实现中。优化变量固定为 4 维 `[a_x, a_y, a_z, ξ]`。禁止以任何理由改为硬约束（无 slack）模式。

**约束 6：Safety gate 用 EMA，不用单 rollout**  
`J_safe_ema ← (1-α) × J_safe_ema + α × J_safe_current`  
`safe_gate = (J_safe_ema ≤ d_safe + δ_s)`  
λ_int 的更新条件必须基于此 EMA 值，禁止直接用单次 rollout 的 J_safe 触发。

**约束 7：MGDA 两阶段顺序**  
第一级：(g_task, λ_s·g_safe)；第二级：(g_ts, λ_int·g_int)。顺序不可颠倒。每级仅在对应 λ > threshold 时激活 MGDA，否则退化为普通梯度差。

**约束 8：集中式 Critic 接口**  
Phase 1-3 的 Critic 接口：`critic(obs_all: Tensor[B, N×obs_dim]) → Tensor[B, 1]`。  
Phase 4 的 ScalableCritic 必须提供 `forward_flat()` 方法兼容此接口。

**约束 9：参数共享**  
Actor、MessageEncoder 所有 agent 共享权重（CTDE）。`log_sigma` 为共享可训练标量参数。不得为不同 agent 实例化独立网络。

**约束 10：测试文件只增不减**  
各 Phase 的测试文件（`test_scheduler.py`、`test_hocbf_qp.py` 等）在后续 Phase 中不得修改或删除，只允许追加新测试。

**约束 11：evaluate.py 双模式**  
评估脚本必须始终支持 `--shield on/off`（或等价参数）两种模式，用于 Shield Independence Eval。不得移除任一模式。

**约束 12：观测向量格式**  
UAV 状态向量固定为 `[p_x, p_y, p_z, v_x, v_y, v_z]`（6维），相对观测使用相同格式。`obs_dim = 6 + K_obs × 6`。在所有环境、网络、测试中保持一致。

---

## 5. Claude 审查协议

### 场景 A：阶段切换审查

触发：Codex 完成某阶段所有任务后。

Claude 操作步骤：
1. 要求用户提供 `pytest tests/ -v` 的完整输出
2. 逐条核查第3节对应阶段的门控清单
3. 读取关键实现文件（见门控标准中"Claude 读文件确认"条目），验证数学公式与设计规格一致
4. 出具结论：
   - **✅ 通过**：列出所有通过项，明确"可开始 Phase [N+1]"
   - **❌ 未通过**：列出具体问题，给出 Codex 可直接执行的修复指令

### 场景 B：架构疑问解答

触发：Codex 遇到设计决策，不确定该怎么做。

Claude 操作步骤：
1. 检查第4节约束清单，判断是否有明确约束覆盖该问题
2. **有约束**：引用对应约束条款，给出明确答案（不需要讨论备选方案）
3. **无约束**：查阅 `docs/superpowers/specs/2026-05-26-safecomm-voi-design.md`，做出决策，**将新决策追加到第4节约束清单**，并更新本文档

### 场景 C：阻塞诊断

触发：Codex 遇到无法解决的错误或技术问题。

Claude 操作步骤：
1. 收集：完整错误信息、相关代码片段（10-30行）、已尝试的方案
2. 定位根因类别：
   - 数学公式实现错误（对照设计规格公式）
   - 接口不匹配（对照约束清单）
   - 测试设计问题（测试预期与实现逻辑不一致）
   - 环境/依赖问题
3. 给出可直接粘贴给 Codex 的修复指令

---

## 6. 当前状态快照

> 每完成一个阶段后更新此表。

| 阶段 | 状态 | 完成日期 | 备注 |
|------|------|----------|------|
| Phase 1 | ⬜ 待完成 | — | 详细计划已就绪 |
| Phase 2 | ⬜ 待完成 | — | 依赖 Phase 1 |
| Phase 3 | ⬜ 待完成 | — | 依赖 Phase 2 |
| Phase 4 | ⬜ 待完成 | — | 依赖 Phase 3 |

**当前活跃阶段**：Phase 1  
**最后更新**：2026-05-26

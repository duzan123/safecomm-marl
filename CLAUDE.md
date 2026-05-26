# CLAUDE.md

本文件为 Claude Code 在此仓库中工作时提供指导。所有交流（回复、提问、注释说明等）均使用**简体中文**。

## 用户偏好

- 默认使用简体中文回复用户。
- 新增项目文档、调研报告、计划文档和说明性注释默认使用简体中文。
- 论文标题、会议/期刊名、术语、DOI、arXiv 编号、BibTeX 字段和引用元数据可保留英文原文，以保证可检索性和引用准确性。

---

## 项目概览

**safecomm-marl** — 面向安全通信的多智能体强化学习（MARL）框架，应用于通信受限多 UAV 编队控制场景。

**核心问题**：现有安全 MARL（MACPO、MAPPO-Lag）假设全通信；现有通信受限 MARL（DGN、SchedNet）不考虑安全约束——二者的联合优化是 NeurIPS/ICML/ICLR 层面的实质性空白。

**解决方案**：提出 **SafeComm-VoI**（Safety-aware Value-of-Information MARL）算法——VoI 感知调度器（AoI 老化 × CBF 危险度）优先分配通信预算，HOCBF-QP 硬盾 + 双约束 Lagrangian（λ_s + λ_int）+ 两阶段 MGDA 实现安全与通信干预成本的联合优化。

**目标期刊**：NeurIPS / ICML / ICLR

---

## 当前状态

- **阶段**：设计 + 实施计划完成，代码尚未实现（Phase 1 待执行）
- 文献调研（8 个方向，共约 50 篇，已核验）已完成 → `docs/lit_review_verified_2026-05-26/`
- 系统设计规格已完成 → `docs/superpowers/plans/2026-05-26-safecomm-voi-design.md`
- 4 阶段详细实施计划已完成 → `docs/superpowers/plans/`
- 总体协调手册已完成（Claude 审查用）→ `docs/superpowers/plans/2026-05-26-safecomm-voi-master-coordination.md`
- 10 个相关开源仓库已克隆 → `external/`（DGN、InforMARL、MAGIC、MACPO、gcbfplus、gym-pybullet-drones、macbf、on-policy、safe-control-gym、sched_net）
- 源代码尚未创建，依赖项和工具链未建立

**协作模型**：Claude 负责阶段审查、架构决策、阻塞诊断；Codex 负责具体代码实现。阶段切换需 Claude 审查通过后方可进入下一阶段。

---

## 问题形式化：CDecPOSG-CB

**Constrained Decentralized Partially Observable Stochastic Game with Communication Budget**

- **状态空间** $S \subset \mathbb{R}^{6n}$：$n$ 个 UAV 的位置 + 速度 + 偏航角（每 UAV 6 维）
- **动作空间** $A_i \subset \mathbb{R}^3$：速度增量控制（$\Delta v_x, \Delta v_y, \Delta v_z$）
- **安全约束**：$\mathbb{E}[\sum_t C^\text{safe}] \leq d$，其中 $C^\text{safe}$ 为碰撞惩罚
- **通信硬约束**：每步全局最多 $k$ 条活跃通信链路

---

## 算法：SafeComm-VoI

**核心组件：**

1. **VoI 感知调度器**（无需训练，构造性保证通信预算 $|E_t| \leq k$）
   - 优先级：$\text{priority}_{ij} = \frac{1 + \beta \cdot \min(\Delta t_\text{comm}/\tau_\text{ref},\ \tau_\text{max})}{\max(h^\text{sched}_{ij},\ \varepsilon)}$
   - 调度 CBF：$h^\text{sched}_{ij} = \max(\|p_i - \hat{p}_j\| - d_\text{min} - d_\text{buf,ij},\ 0)^2$；$d_\text{buf} = v_\text{max} \cdot \Delta t_\text{comm}$
   - 通信图：$E_t = \text{TopK}(E^\text{phys}_t,\ k,\ \text{by priority})$

2. **HOCBF-QP 硬盾**（运行时安全过滤，每 agent 独立 QP）
   - 优化变量：$[a_x, a_y, a_z, \xi]$（4 维，含 slack $\xi \geq 0$）
   - 约束：$2r^T a + 2\|v\|^2 + \alpha_1 \dot{h} + \alpha_2 \psi_1 - m_\text{acc} + \xi \geq 0$
   - 双动作不变项：环境执行 $a^*$（QP 输出），PPO log prob 使用 $a^\pi$（策略输出），两者分开存储

3. **消息编码与聚合**：单头注意力聚合邻居消息（$d_\text{msg} = 64$），参数共享

4. **MAPPO 策略网络**：参数共享 Actor + 集中式 Critic（CTDE）

5. **双约束 Lagrangian**：
   - $\lambda_s$：安全约束（始终激活）
   - $\lambda_\text{int}$：干预成本约束（Safety Gate EMA 激活：$J^\text{safe}_\text{ema} \leq d_\text{safe} + \delta_s$）

6. **两阶段 MGDA 梯度协调**：
   - 第一级：$(g_\text{task},\ \lambda_s \cdot g_\text{safe})$
   - 第二级：$(g_{ts},\ \lambda_\text{int} \cdot g_\text{int})$

---

## 计划代码架构

```
safecomm-marl/
├── envs/
│   ├── uav_formation_env.py       # 轻量双积分器仿真（Phase 1-2）
│   ├── pybullet_uav_env.py        # PyBullet 四旋翼仿真（Phase 3）
│   ├── domain_randomization.py    # DR 包装器，噪声注入 obs（Phase 3）
│   └── pursuit_env.py             # 追逃任务环境（Phase 4）
├── algorithms/
│   ├── safecomm_psched.py         # 核心算法主文件（SafeComm-VoI）
│   ├── scheduler.py               # VoI 感知调度器（TopK + AoI + CBF）
│   ├── hocbf_qp.py                # HOCBF-QP 硬盾（Phase 2）
│   ├── networks.py                # Actor / Critic / MessageEncoder
│   ├── scalable_critic.py         # Transformer N-agnostic Critic（Phase 4）
│   └── ppo_utils.py               # GAE、PPO clip、RolloutBuffer
├── baselines/
│   ├── mappo.py
│   ├── macpo.py
│   ├── mappo_lag.py
│   ├── dgn_mappo.py
│   ├── liu2023.py                 # Liu et al. NeurIPS 2023
│   └── cbf_ppo.py                 # CBF-PPO（Qin RA-L 2023）
├── configs/
│   ├── default.yaml               # 默认超参数
│   └── ablation/                  # 消融实验配置
├── train.py                       # 单任务训练入口
├── train_multitask.py             # 多任务训练入口（Phase 4）
├── evaluate.py                    # 评估入口（支持 --shield on/off）
└── tests/
    ├── test_scheduler.py          # Phase 1
    ├── test_env.py                # Phase 1
    ├── test_algorithm.py          # Phase 1
    ├── test_hocbf_qp.py           # Phase 2
    ├── test_baselines.py          # Phase 2
    ├── test_pybullet_env.py       # Phase 3
    ├── test_pursuit_env.py        # Phase 4
    └── test_scalable_critic.py    # Phase 4
```

---

## 构建与运行命令

> 代码实现后在此处补充：
> - 安装依赖：`pip install -r requirements.txt`
> - 训练：`python train.py --config configs/default.yaml`
> - 评估：`python evaluate.py --checkpoint <path>`
> - 单测：`pytest tests/test_scheduler.py -v`

---

## 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `N` | 4（→8） | UAV 数量（Phase 1-3 为 4，Phase 4 扩展至 8） |
| `k` | 4 | 通信预算（每步最多活跃链路数） |
| `d_safe` | 0.1 | 安全约束阈值 |
| `d_int` | 0.05 | 干预成本约束阈值 |
| `d_min` | 0.5 m | 最小安全距离 |
| `d_msg` | 64 | 消息向量维度 |
| 隐藏层 | [256, 256] | Actor / Critic MLP 规模 |
| `obs_dim` | 6 + K_obs×6 | 观测维度（状态 6 维 + K 邻居各 6 维） |
| `β` | 0.5 | VoI 调度器 AoI 老化权重 |
| `τ_ref` | 5 步 | AoI 参考时间窗口 |
| `v_max` | 2.0 m/s | CBF 缓冲区计算用最大速度 |

---

## 六个 Baseline

| 方法 | 安全约束 | 通信约束 |
|------|---------|---------|
| MAPPO | 无 | 无（任务上界） |
| MACPO | CMDP | 无 |
| MAPPO-Lag | Lagrangian | 无 |
| DGN + MAPPO | 无 | 图约束 |
| Liu et al. NeurIPS 2023 | CMDP | 信道质量自适应 |
| CBF-PPO（Qin RA-L 2023） | CBF 硬约束 | 无 |

---

## 文档索引

### 文献调研（已核验版本）
- `docs/lit_review_verified_2026-05-26/00_synthesis_report.md` — 综合报告
- `docs/lit_review_verified_2026-05-26/01_safe_marl.md` — 安全 MARL 方向
- `docs/lit_review_verified_2026-05-26/02_comm_constrained_marl.md` — 通信受限 MARL 方向
- `docs/lit_review_verified_2026-05-26/03_safety_comm_intersection.md` — 安全×通信交叉方向
- `docs/lit_review_verified_2026-05-26/04_uav_formation.md` — 多 UAV 编队控制方向
- `docs/lit_review_verified_2026-05-26/05_multi_objective_pareto_marl.md` — 多目标 Pareto MARL
- `docs/lit_review_verified_2026-05-26/06_cbf_mpc_shielding_safe_marl.md` — CBF/MPC/Shielding 安全方法
- `docs/lit_review_verified_2026-05-26/07_graph_attention_scalable_marl.md` — 图注意力可扩展 MARL
- `docs/lit_review_verified_2026-05-26/08_sim2real_uav_marl.md` — Sim2Real UAV MARL
- `docs/archive/lit_review/` — 旧版文献调研（已归档，仅供参考）

### 设计规格
- `docs/archive/2026-05-25-safecomm-marl-design.md` — 初版设计规格（SafeComm-PSched，已过时）
- `docs/superpowers/plans/2026-05-26-safecomm-voi-design.md` — **当前设计规格**（SafeComm-VoI，算法伪代码、实验设计、理论贡献）
- `docs/superpowers/plans/2026-05-26-safecomm-voi-master-coordination.md` — **Claude 协调手册**（阶段门控标准、12 项架构约束、审查协议）

### 实施计划（供 Codex 执行）
- `docs/superpowers/plans/2026-05-26-safecomm-voi-phase1.md` — Phase 1：VoI 调度器 + 原型
- `docs/superpowers/plans/2026-05-26-safecomm-voi-phase2.md` — Phase 2：完整 SafeComm-VoI（HOCBF-QP + 双约束）
- `docs/superpowers/plans/2026-05-26-safecomm-voi-phase3.md` — Phase 3：PyBullet + 域随机化
- `docs/superpowers/plans/2026-05-26-safecomm-voi-phase4.md` — Phase 4：多任务 + N≥8 扩展

# CLAUDE.md

本文件为 Claude Code 在此仓库中工作时提供指导。所有交流（回复、提问、注释说明等）均使用**简体中文**。

---

## 项目概览

**safecomm-marl** — 面向安全通信的多智能体强化学习（MARL）框架，应用于通信受限多 UAV 编队控制场景。

**核心问题**：现有安全 MARL（MACPO、MAPPO-Lag）假设全通信；现有通信受限 MARL（DGN、SchedNet）不考虑安全约束——二者的联合优化是 NeurIPS/ICML/ICLR 层面的实质性空白。

**解决方案**：提出 **SafeComm-PSched** 算法——安全感知调度器（Safety-Priority Scheduler）按 CBF 危险度优先分配通信预算，MAPPO + 单 Lagrangian 处理安全约束，两者通过 CBF 值自然耦合。

**目标期刊**：NeurIPS / ICML / ICLR

---

## 当前状态

- **阶段**：设计完成，代码尚未实现
- 文献调研（4 个方向，共约 50 篇）已完成 → `docs/lit_review/`
- 系统设计规格已完成 → `docs/superpowers/specs/2026-05-25-safecomm-marl-design.md`
- 源代码尚未创建，依赖项和工具链未建立

---

## 问题形式化：CDecPOSG-CB

**Constrained Decentralized Partially Observable Stochastic Game with Communication Budget**

- **状态空间** $S \subset \mathbb{R}^{6n}$：$n$ 个 UAV 的位置 + 速度 + 偏航角（每 UAV 6 维）
- **动作空间** $A_i \subset \mathbb{R}^3$：速度增量控制（$\Delta v_x, \Delta v_y, \Delta v_z$）
- **安全约束**：$\mathbb{E}[\sum_t C^\text{safe}] \leq d$，其中 $C^\text{safe}$ 为碰撞惩罚
- **通信硬约束**：每步全局最多 $k$ 条活跃通信链路

---

## 算法：SafeComm-PSched

**核心组件：**

1. **安全感知调度器**（无需训练，构造性保证通信预算）
   - CBF 值：$h_{ij}(t) = (\|p_i - p_j\| - d_\text{min})^2$
   - 通信图：$E_t = \text{TopK}(E^\text{phys}_t,\ k,\ \text{by}\ 1/h_{ij})$

2. **消息编码与聚合**：单头注意力聚合邻居消息（$d_\text{msg} = 64$）

3. **MAPPO 策略网络**：参数共享 Actor + 集中式 Critic + Cost Critic

4. **Lagrangian 安全优化**：$\lambda_s \leftarrow \max(0,\ \lambda_s + \alpha_\lambda \cdot (\hat{J}_\text{safe} - d))$

---

## 计划代码架构

```
safecomm-marl/
├── envs/
│   ├── uav_formation_env.py     # 轻量 2D/3D 仿真（双积分器，Phase 1）
│   └── pybullet_uav_env.py      # PyBullet 四旋翼仿真（Phase 2）
├── algorithms/
│   ├── safecomm_psched.py       # 核心算法主文件
│   ├── scheduler.py             # 安全感知调度器（TopK + CBF）
│   ├── networks.py              # Actor / Critic / CostCritic / MessageEncoder
│   └── ppo_utils.py             # GAE、PPO clip 工具函数
├── baselines/
│   ├── mappo.py
│   ├── macpo.py
│   ├── mappo_lag.py
│   └── dgn_mappo.py
├── configs/
│   ├── default.yaml             # 默认超参数
│   └── ablation/                # 消融实验配置
├── train.py                     # 训练入口
├── evaluate.py                  # 评估入口
└── tests/
    ├── test_scheduler.py
    ├── test_env.py
    └── test_algorithm.py
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
| `N` | 4–8 | UAV 数量（Phase 1） |
| `k` | 4 | 通信预算（每步最多活跃链路数） |
| `d` | 0.1 | 安全约束阈值 |
| `d_min` | 0.5 m | 最小安全距离 |
| `d_msg` | 64 | 消息向量维度 |
| 隐藏层 | [256, 256] | Actor / Critic MLP 规模 |

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

- `docs/lit_review/00_synthesis_report.md` — 文献调研综合报告（约 50 篇）
- `docs/lit_review/01_safe_marl.md` — 安全 MARL 方向
- `docs/lit_review/02_comm_constrained_marl.md` — 通信受限 MARL 方向
- `docs/lit_review/03_safety_comm_intersection.md` — 安全×通信交叉方向
- `docs/lit_review/04_uav_formation.md` — 多 UAV 编队控制方向
- `docs/superpowers/specs/2026-05-25-safecomm-marl-design.md` — 系统设计规格（算法伪代码、实验设计、理论贡献）

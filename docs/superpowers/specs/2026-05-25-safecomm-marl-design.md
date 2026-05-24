# SafeComm-MARL 系统设计规格

**日期：** 2026-05-25  
**版本：** v1.0  
**目标期刊：** NeurIPS / ICML / ICLR  

---

## 1. 概述

SafeComm-MARL 是首个将**安全约束**（CMDP/碰撞避免）与**通信约束**（带宽硬预算）**联合建模并优化**的多智能体强化学习框架，应用于多 UAV 编队控制场景。

**核心问题：** 现有安全 MARL 假设全通信（MACPO, MAPPO-Lag），现有通信受限 MARL 不考虑安全约束（DGN, SchedNet）——两者的联合优化是一个实质性空白。

**解决方案：** 安全感知调度器（Safety-Priority Scheduler）将通信硬预算按安全风险优先级分配，MAPPO + 单 Lagrangian 处理安全约束，两者通过 CBF 值自然耦合。

---

## 2. 问题形式化：CDecPOSG-CB

**定义（Constrained Decentralized Partially Observable Stochastic Game with Communication Budget）：**

$$\mathcal{M} = \langle N,\ S,\ \{O_i\},\ \{A_i\},\ P,\ R,\ G^\text{phys},\ C^\text{safe},\ k,\ d \rangle$$

| 符号 | 类型 | 含义 |
|------|------|------|
| $N = \{1,\ldots,n\}$ | 集合 | $n$ 个 UAV 智能体 |
| $S \subset \mathbb{R}^{6n}$ | 连续空间 | 联合状态（位置 + 速度 + 偏航角，每 UAV 6 维） |
| $O_i \subset \mathbb{R}^{d_o}$ | 连续空间 | 智能体 $i$ 的本地观测（自身全状态 + 感知范围内邻居相对状态） |
| $A_i \subset \mathbb{R}^{3}$ | 连续空间 | 控制动作（$\Delta v_x, \Delta v_y, \Delta v_z$） |
| $P(s' \mid s, a)$ | 转移函数 | UAV 双积分器动力学（Phase 1）/ 四旋翼动力学（Phase 2） |
| $R(s, a) \in \mathbb{R}$ | 奖励函数 | 编队误差奖励 + 目标达到奖励（见 §5.2） |
| $G^\text{phys}_t = (V, E^\text{phys}_t)$ | 时变图 | 物理可连接图：$(i,j) \in E^\text{phys}$ 当且仅当 $\|p_i - p_j\| \leq R_\text{comm}$ |
| $C^\text{safe}(s, a) \geq 0$ | 成本函数 | 碰撞惩罚：$\mathbb{1}[\min_{j \neq i} \|p_i - p_j\| < d_\text{min}]$ |
| $k \in \mathbb{Z}^+$ | 超参数 | **通信硬预算**：每步全局最多 $k$ 条活跃通信链路 |
| $d \geq 0$ | 超参数 | 安全约束阈值：$\mathbb{E}[\sum_t C^\text{safe}] \leq d$ |

**目标（原始约束优化问题）：**

$$\max_{\pi, \text{enc}} \quad J_\text{task}(\pi, E) = \mathbb{E}_\tau \left[ \sum_{t=0}^{T} \gamma^t R(s_t, a_t) \right]$$

$$\text{s.t.} \quad J_\text{safe}(\pi) = \mathbb{E}_\tau \left[ \sum_{t=0}^{T} C^\text{safe}(s_t, a_t) \right] \leq d$$

$$\quad\quad\quad |E_t| \leq k \quad \forall t \in \{0, \ldots, T\}$$

**关键区别于现有工作：**

| 维度 | Liu NeurIPS 2023 | 本工作 |
|------|-----------------|--------|
| 通信建模 | 信道质量（外生参数） | 带宽预算（内生硬约束） |
| 通信策略 | 被动置信度加权 | 主动安全感知调度 |
| 优化框架 | 软约束 | 硬通信 + 软安全（Lagrangian） |

---

## 3. 算法：SafeComm-PSched

**算法名称：** SafeComm-PSched（Safety-Priority Scheduled MARL）  
**骨干网络：** MAPPO（参数共享）  
**核心创新：** 安全感知调度器将通信预算按 CBF 危险度优先分配

### 3.1 安全感知调度器

**CBF 值定义（距离型）：**

$$h_{ij}(t) = \left( \|p_i(t) - p_j(t)\| - d_\text{min} \right)^2 \geq 0$$

当 $h_{ij} \to 0$ 时，$(i, j)$ 对接近碰撞边界，通信最紧迫。

**优先级与通信图激活：**

$$\text{priority}_{ij}(t) = \frac{1}{\max(h_{ij}(t),\ \varepsilon)}, \quad \varepsilon = 10^{-6}$$

$$E_t = \operatorname{TopK}\!\Big(\{(i,j) \in E^\text{phys}_t\},\ k,\ \text{by } \text{priority}_{ij}\Big)$$

**性质：** 调度器无需训练，构造性保证 $|E_t| \leq k$（通信硬约束始终满足）。

### 3.2 网络架构

```
输入层
  o_i ∈ ℝ^{d_obs}       本地观测向量
  m_i = {enc_j(o_j) : j ∈ N_i(E_t)}    邻居消息集合

消息编码器（共享参数）
  enc: ℝ^{d_obs} → ℝ^{d_msg}    2层 MLP，d_msg = 64

消息聚合（注意力）
  agg_i = Σ_{j∈N_i} α_ij · enc_j(o_j)
  α_ij = softmax(q_i^T k_j / √d_msg)    单头注意力

Actor（共享参数）
  π_i(· | [o_i; agg_i]) : ℝ^{d_obs+d_msg} → N(μ, σ²)    2层 MLP

Critic（集中式，训练时用）
  V_θ(s₁,...,sₙ) → ℝ    任务价值，2层 MLP

Cost Critic（集中式，训练时用）
  V_φ^c(s₁,...,sₙ) → ℝ    安全成本价值，2层 MLP
```

**超参数默认值：** $d_\text{obs}=18$（自身状态 6 维 + 2 个可见邻居各 6 维相对状态，N=4 默认配置），$d_\text{msg}=64$，Actor/Critic 隐藏层 $[256, 256]$

### 3.3 训练算法（伪代码）

```
初始化：π, enc, V_θ, V_φ^c, λ_s = 0, 学习率 α_π, α_λ

for 迭代 t = 1, 2, ..., T_max:

  # 1. 数据收集
  for 每个环境步 τ:
    计算 CBF 值 h_ij(t) 和物理图 G^phys_t
    激活通信图 E_t = TopK(E^phys_t, k, by 1/h_ij)
    各 UAV 接收消息 m_i，聚合得 agg_i
    执行动作 a_i ~ π_i(·|o_i, agg_i)
    收集奖励 r_t 和安全成本 c_t^safe

  # 2. 计算优势函数
  A_π  = GAE(r, V_θ)       任务优势
  A_c  = GAE(c^safe, V_φ^c)  安全成本优势
  
  # 3. PPO 策略更新（含安全惩罚）
  L_PPO = E[clip(r(π/π_old), 1±ε) · A_π] - λ_s · E[A_c]
  更新 π, enc, V_θ, V_φ^c（最大化 L_PPO）

  # 4. Lagrangian 乘子更新
  Ĵ_safe = E_τ[Σ_t c_t^safe]
  λ_s ← max(0, λ_s + α_λ · (Ĵ_safe - d))

  # 5. 日志记录
  记录 J_task, J_safe, λ_s, |E_t|/k（带宽利用率）

输出：π*, enc*, λ_s*
```

---

## 4. 理论贡献（目标）

### 命题 1（通信硬约束满足）
调度器构造性保证任意时刻 $|E_t| \leq k$。  
*证明：TopK 操作从 $|E^\text{phys}_t|$ 条边中最多选 $k$ 条，直接可行。*

### 定理 1（安全约束收敛）
在标准 CMDP 假设下（有界奖励、Slater 条件成立），Lagrangian 算法满足：

$$J_\text{safe}(\pi^T) \leq d + O(1/\sqrt{T})$$

*证明：利用 Lagrangian 对偶收敛理论（Altman 1999，Achiam 2017），MAPPO-Lag 的已有结论直接移植。*

### 命题 2（安全-通信 Pareto 权衡）
在固定策略类 $\Pi$ 上，最优安全代价关于通信预算 $k$ 单调不增：

$$k_1 \leq k_2 \implies J_\text{safe}^*(k_1) \geq J_\text{safe}^*(k_2)$$

*直觉：通信越多，协调信息越丰富，越容易避免碰撞。*

---

## 5. 实验设计

### 5.1 仿真环境

| 阶段 | 环境 | UAV 数量 | 目的 |
|------|------|---------|------|
| Phase 1（Week 7–10） | 自制轻量 2D/3D 仿真（双积分器） | N = 4–8 | 算法验证，快速迭代 |
| Phase 2（Week 11–13） | PyBullet + 四旋翼动力学模型 | N = 4–16 | 物理真实性验证 |
| Phase 3（Week 14，可选） | Gazebo / AirSim | N = 8–12 | 展示实验 |

### 5.2 奖励函数设计

$$R(s, a) = -w_f \cdot \text{FE}(s) + w_s \cdot \mathbb{1}[\text{成功}] - w_c \cdot \mathbb{1}[\text{碰撞}]$$

| 项 | 默认权重 | 含义 |
|----|---------|------|
| $\text{FE}(s) = \frac{1}{n}\sum_i \|p_i - p_i^*\|$ | $w_f = 1.0$ | 编队误差（越小越好） |
| 成功到达目标 | $w_s = 5.0$ | 稀疏成功奖励 |
| 碰撞惩罚 | $w_c = 10.0$ | 仅用于 MAPPO 无约束基线 |

**注意：** SafeComm-PSched 通过 Lagrangian 处理碰撞，奖励函数中不含碰撞惩罚项（分离关注点）。

### 5.3 Baseline 方案（6 个）

| 方法 | 安全 | 通信约束 | 说明 |
|------|------|---------|------|
| MAPPO | 无 | 无 | 任务性能上界 |
| MACPO | CMDP | 无 | 安全 MARL 主基线 |
| MAPPO-Lag | Lagrangian | 无 | 软安全基线 |
| DGN + MAPPO | 无 | 图约束 | 通信受限基线 |
| Liu et al. NeurIPS 2023 | CMDP | 信道质量自适应 | 最相关对比 |
| CBF-PPO（Qin RA-L 2023） | CBF 硬约束 | 无 | 硬安全基线 |
| **SafeComm-PSched（本方法）** | Lagrangian | 硬预算 k | 提出方法 |

### 5.4 评估指标

| 类别 | 指标 | 单位 |
|------|------|------|
| 任务性能 | 编队误差 Formation Error | 米 |
| 任务性能 | 任务成功率 Success Rate | % |
| 安全性 | 碰撞率 Collision Rate | 次/集 |
| 安全性 | 约束违反率 CVR | % |
| 通信效率 | 每步消息数 MPS | 条/步 |
| 通信效率 | 带宽利用率 BU | $|E_t|/k$ |
| 样本效率 | 训练曲线 | 步数 |

### 5.5 消融实验

| 消融项 | 说明 |
|-------|------|
| w/o 安全调度 → 随机 Top-k | 验证安全感知调度的必要性 |
| w/o 通信约束（k = n(n-1)） | 全通信下的安全 MARL 上界 |
| k 敏感性分析 | k = 0, 2, 4, 8, ∞，绘制 Pareto 曲线 |
| 消息维度 d_msg | 32, 64, 128 |

---

## 6. 代码架构

```
safecomm-marl/
├── envs/
│   ├── uav_formation_env.py     # 自制轻量仿真（Phase 1）
│   └── pybullet_uav_env.py      # PyBullet 仿真（Phase 2）
├── algorithms/
│   ├── safecomm_psched.py       # 核心算法（主文件）
│   ├── scheduler.py             # 安全感知调度器
│   ├── networks.py              # Actor, Critic, CostCritic, MessageEncoder
│   └── ppo_utils.py             # GAE, PPO clip 工具
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
    ├── test_scheduler.py        # 调度器单元测试
    ├── test_env.py              # 环境接口测试
    └── test_algorithm.py        # 算法集成测试
```

---

## 7. 风险与应对

| 风险 | 概率 | 缓解策略 |
|------|------|---------|
| Lagrangian 训练不稳定（λ 振荡） | 中 | 添加 λ 梯度裁剪；使用双时间尺度更新（λ 更新频率低于 π） |
| 通信图稀疏时性能退化 > 20% | 中 | 将 k 作为课程学习的变量（先 k 大后 k 小）；验证 Pareto 曲线 |
| PyBullet 四旋翼动力学 CBF 值不稳定 | 低 | Phase 1 先用质点模型验证，Phase 2 再适配；可引入缓冲距离 ε_cbf |
| 理论证明过难（Theorem 1 条件不满足） | 中 | 退而求其次：提供经验性安全曲线 + Slater 条件验证 |

---

## 8. 下一步

- [ ] **实现 Phase 1 仿真环境**（`envs/uav_formation_env.py`，双积分器，N=6）
- [ ] **实现调度器**（`algorithms/scheduler.py`，TopK + CBF，单元测试覆盖）
- [ ] **实现 SafeComm-PSched 核心**（`algorithms/safecomm_psched.py`）
- [ ] **实现 MAPPO 和 MAPPO-Lag 基线**（用于早期对比验证）
- [ ] **运行第一个训练实验**（N=4，验证 J_safe ≤ d 约束满足）

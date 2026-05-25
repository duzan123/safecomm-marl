# SafeComm-MARL 系统设计规格 v2.0

**日期：** 2026-05-25  
**版本：** v2.0（基于8个文献调研方向更新）  
**前版：** `docs/superpowers/specs/2026-05-25-safecomm-marl-design.md`（v1.0）  
**目标期刊：** NeurIPS / ICML / ICLR  

---

### v2.0 主要更新摘要

| 组件 | v1.0 | v2.0 | 依据 |
|------|------|------|------|
| 消息聚合 | 单头注意力 | H=2 双头注意力 | 方向7（图注意力MARL） |
| Lagrangian 训练 | 基础 PPO-Lag | + MGDA 梯度调和 | 方向5（Multi-obj MARL） |
| 调度器 CBF 近似 | 直接距离型 | + 保守缓冲公式 | 方向6（CBF理论） |
| Proposition 2 | 直觉单调性 | + Wang公式量化证明 | 方向6（CBF理论） |
| Proposition 3 | 无 | CHV 约束 Pareto 前沿框架 | 方向5（Multi-obj MARL） |
| Phase 2 仿真 | 待建 | 7类DR参数规范 | 方向8（Sim2Real） |
| k 消融实验 | 性能对比 | + CHV约束Pareto前沿 | 方向5（Multi-obj MARL） |
| 风险矩阵 | 4行 | 增加RARL禁用、延迟CBF风险 | 方向8（Sim2Real） |

---

## 1. 概述

SafeComm-MARL 是首个将**安全约束**（CMDP/碰撞避免）与**通信约束**（带宽硬预算）**联合建模并优化**的多智能体强化学习框架，应用于多 UAV 编队控制场景。

**核心问题：** 现有安全 MARL 假设全通信（MACPO, MAPPO-Lag），现有通信受限 MARL 不考虑安全约束（DGN, SchedNet）——两者的联合优化是一个实质性空白。

**解决方案：** 安全感知调度器（Safety-Priority Scheduler）将通信硬预算按安全风险优先级分配，MAPPO + 单 Lagrangian 处理安全约束，两者通过 CBF 值自然耦合。

相比 v1.0，本版本基于 8 个方向的系统性文献调研（约 70 篇论文），对核心算法组件进行了三项可操作改进：
1. 将消息聚合升级为 **H=2 双头注意力**，使两个头分别学习"安全距离"和"编队角色"两类正交信息；
2. 在 Lagrangian 训练中引入 **MGDA 梯度调和**，防止 λ_s 增大时压制任务梯度导致振荡；
3. 在调度器 CBF 值计算中加入 **Wang 保守缓冲公式**（Wang et al., TAC 2017），使 Proposition 2 获得可量化的理论支撑。

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

## 3. 算法：SafeComm-PSched（v2.0）

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

#### 3.1.1 通信受限下的保守 CBF 近似（v2.0 新增）

当步骤 $t$ 未分配链路 $(i, j)$（即 $(i, j) \notin E_t$），调度器无法获得 $j$ 的实时位置。
此时采用保守缓冲公式（Wang et al., TAC 2017）计算保守 CBF 下界：

$$d_\text{buffer}(i,j,t) = v_\text{max} \cdot \Delta t_\text{comm}(i,j,t)$$

其中 $\Delta t_\text{comm}(i,j,t)$ 为智能体 $i$ 上次接收 $j$ 状态至当前步的时间间隔。

**保守 CBF 值定义：**

$$h^\text{cons}_{ij}(t) = \Big( \|p_i(t) - \hat{p}_j(t)\| - d_\text{min} - d_\text{buffer}(i,j,t) \Big)^2$$

其中 $\hat{p}_j(t) = p_j(t_\text{last})$（上次通信时 $j$ 的已知位置）。

**性质：** 由于 $h^\text{cons}_{ij} \leq h_{ij}$（真实值），调度器基于保守估计的优先级更高，即通信中断时间越长，该对优先级越高，自然形成"优先恢复通信"的调度偏好。这一性质为 Proposition 2 提供了构造性证明基础。

### 3.2 网络架构（v2.0）

```
输入层
  o_i ∈ ℝ^{d_obs}       本地观测向量
  m_i = {enc_j(o_j) : j ∈ N_i(E_t)}    邻居消息集合

消息编码器（共享参数）
  enc: ℝ^{d_obs} → ℝ^{d_msg}    2层 MLP，d_msg = 64

消息聚合（H=2 双头注意力，v2.0）
  # 每头维度 d_head = d_msg / H = 32
  for h in range(H=2):
    q_h = W_q^h · o_i          ∈ ℝ^{d_head}          # query
    K_h = W_k^h · {enc_j}     ∈ ℝ^{|N_i| × d_head}   # keys
    V_h = W_v^h · {enc_j}     ∈ ℝ^{|N_i| × d_head}   # values
    α_h = softmax(q_h^T K_h / √d_head)
    head_h = Σ_j α_{h,j} · V_{h,j}
  agg_i = W_o · concat(head_0, head_1)  ∈ ℝ^{d_msg}

Actor（共享参数）
  π_i(· | [o_i; agg_i]) : ℝ^{d_obs+d_msg} → N(μ, σ²)    2层 MLP

Critic（集中式，训练时用）
  V_θ(s₁,...,sₙ) → ℝ    任务价值，2层 MLP

Cost Critic（集中式，训练时用）
  V_φ^c(s₁,...,sₙ) → ℝ    安全成本价值，2层 MLP
```

**设计决策（重要）：** 消息聚合跳数固定为 1，**禁止使用多跳 GCN**。  
理由：DGN 风格的 2 层图卷积允许信息经中间节点间接传播，隐式软化了带宽硬约束——与 CDecPOSG-CB 的通信预算理论贡献语义冲突，且破坏消融实验中"通信预算效果"的可解释性。

**超参数默认值：** $d_\text{obs}=18$，H=2（固定），$d_\text{head}=32$，$d_\text{msg}=64$，Actor/Critic 隐藏层 $[256, 256]$

### 3.3 训练算法（v2.0，含 MGDA）

```
初始化：π, enc, V_θ, V_φ^c, λ_s = 0, 学习率 α_π, α_λ, λ_threshold = 0.1

for 迭代 t = 1, 2, ..., T_max:

  # 1. 数据收集
  for 每个环境步 τ:
    计算 CBF 值 h_ij(t)（含保守近似，§3.1.1）和物理图 G^phys_t
    激活通信图 E_t = TopK(E^phys_t, k, by 1/h_ij)
    各 UAV 接收消息 m_i，经 H=2 双头注意力聚合得 agg_i
    执行动作 a_i ~ π_i(·|o_i, agg_i)
    收集奖励 r_t 和安全成本 c_t^safe

  # 2. 计算优势函数
  A_π  = GAE(r, V_θ)         任务优势
  A_c  = GAE(c^safe, V_φ^c)  安全成本优势
  
  # 3. PPO 策略更新（含 MGDA 梯度调和，v2.0）
  
  # 3a. 计算两路梯度
  L_task = E[clip(r(π/π_old), 1±ε) · A_π]
  L_safe = λ_s · E[A_c]
  
  ∇_task = grad(L_task, θ)        # 任务梯度
  ∇_safe = grad(L_safe, θ)        # 安全惩罚梯度
  
  # 3b. MGDA 梯度调和（仅在 λ_s > λ_threshold 时激活）
  if λ_s > λ_threshold:
    w_task, w_safe = MGDA_solve(∇_task, ∇_safe)   # Frank-Wolfe近似，~20行
    ∇_combined = w_task · ∇_task + w_safe · ∇_safe
  else:
    ∇_combined = ∇_task - ∇_safe   # 初始阶段直接合并
  
  # 3c. 参数更新
  θ ← θ + α_π · ∇_combined
  更新 V_θ, V_φ^c（标准均方误差梯度）

  # 4. Lagrangian 乘子更新（双时间尺度：更新频率低于 π）
  Ĵ_safe = E_τ[Σ_t c_t^safe]
  λ_s ← max(0, λ_s + α_λ · (Ĵ_safe - d))

  # 5. 日志记录
  记录 J_task, J_safe, λ_s, |E_t|/k（带宽利用率）, w_task, w_safe

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

*证明：利用 Lagrangian 对偶收敛理论（Altman 1999，Achiam 2017），MAPPO-Lag 的已有结论直接移植。MGDA 调和在 Slater 条件下不影响原始-对偶收敛性。*

### 命题 2（安全-通信 Pareto 权衡，量化版，v2.0）

在固定策略类 $\Pi$ 和单步最大速度 $v_\text{max}$ 下：

**(a) 单调性：**

$$k_1 \leq k_2 \implies J_\text{safe}^*(k_1) \geq J_\text{safe}^*(k_2)$$

**(b) 量化界（Wang et al., TAC 2017）：**

对于平均通信间隔 $\bar{\Delta t}_\text{comm}(k) = T/k$，所需安全缓冲距离满足：

$$d_\text{buffer}(k) = v_\text{max} \cdot \bar{\Delta t}_\text{comm}(k) = \frac{v_\text{max} \cdot T}{k}$$

即通信预算 $k$ 越小，所需安全裕量越大，安全代价单调增大：

$$\Delta J_\text{safe}(k_1, k_2) \geq \Omega\!\left(\frac{1}{k_1} - \frac{1}{k_2}\right)$$

*证明思路：利用保守 CBF 缓冲公式（§3.1.1），$k$ 减小 → 平均通信间隔增大 → $d_\text{buffer}$ 增大 → CBF 可行域收窄 → 最优安全代价单调不减。*

### 命题 3（约束 Pareto 前沿的 CHV 量化框架，v2.0 新增）

定义约束 Hypervolume（CHV）指标：

$$\text{CHV}(\pi, r) = \text{Vol}\!\left(\{(J_\text{task}, -J_\text{safe}) : J_\text{safe} \leq d\} \setminus \text{Dominated}(r)\right)$$

其中 $r = (r_\text{task}, r_\text{safe})$ 为参考点（最差性能点）。

**应用：** 不同 $k$ 值下的 $\{J_\text{task}, J_\text{safe}\}$ 散点即为约束 Pareto 前沿，CHV 度量安全感知调度策略相对随机 TopK 的 Pareto 改进幅度：

$$\Delta\text{CHV} = \text{CHV}(\text{SafeComm-PSched}) - \text{CHV}(\text{Random-TopK}) > 0$$

该命题将消融实验（§5.5）中的 $k$ 敏感性分析形式化为约束 Pareto 前沿的可视化与量化比较。

---

## 5. 实验设计

### 5.1 仿真环境

| 阶段 | 环境 | UAV 数量 | 目的 |
|------|------|---------|------|
| Phase 1（Week 7–10） | 自制轻量 2D/3D 仿真（双积分器） | N = 4–8 | 算法验证，快速迭代 |
| Phase 2（Week 11–13） | PyBullet + 四旋翼动力学模型 | N = 4–16 | 物理真实性验证，域随机化 |
| Phase 3（Week 14，可选） | Gazebo / AirSim | N = 8–12 | 展示实验，提升说服力 |

#### Phase 1 补充规范

- 安全约束从宽松开始（课程学习）：$d_\text{min} = 0.3\text{m}$，$d = 0.5$ 次/集
- 验证目标：安全约束满足（$J_\text{safe} \leq d$）、$k$ 消融曲线形态

#### Phase 2 域随机化规范（v2.0，方向8文献支持）

**配置文件：** `configs/phase2_dr.yaml`

| 随机化参数 | 范围 | 分布 | 优先级 | 理由 |
|-----------|------|------|--------|------|
| 推力系数（thrust_coeff） | [0.82, 1.18] | Uniform | 高 | 电机磨损最主要因素 |
| 电机时间常数（motor_time_constant） | [0.010, 0.050]s | Uniform | 高 | 影响响应速度 |
| **相对位置传感噪声**（relative_pos_noise） | σ = 0.03m | Gaussian | **最高** | CBF 值 $h_{ij}\to 0$ 时误差被非线性放大，直接影响调度决策 |
| 下洗气流扰动（downwash_thrust） | [-0.03, 0.03] | Uniform | 中 | 多机编队特有气动耦合 |
| **通信延迟**（comm_delay） | [0, 40]ms | Uniform | **高** | SafeComm 专有：>40ms 导致 CBF 值过期，级联失效 |
| **消息丢包率**（message_dropout） | [0%, 30%] | Uniform | **高** | SafeComm 专有：影响调度决策的信息完整性 |
| 通信预算抖动（comm_budget_jitter） | ±1 链路 | 15%概率 | 中 | 模拟真实带宽波动 |

**明确禁用：** RARL（对抗性强化学习）——四旋翼场景已有多篇文献报告 RARL 导致策略崩塌，应坚持 Uniform DR；资源充裕时使用 ADR/DROPO。

#### 渐进约束收紧（课程迁移）

| 阶段 | $d_\text{min}$ | Lagrangian $d$ | 说明 |
|------|--------------|--------------|------|
| Phase 1 | 0.3m | 0.5 次/集 | 宽松初始化，加速学习 |
| Phase 2 | 0.5m | 0.2 次/集 | 收紧，接近真实部署 |
| 真实飞行（可选） | 0.8m | 0.1 次/集 | 保守安全裕量 |

Lagrangian 框架与渐进收紧天然兼容：分阶段调整 $d$ 参数，无需重新设计算法结构。

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
| w/o 安全调度 → 随机 Top-k | 验证安全感知调度的必要性（**主消融**） |
| w/o 通信约束（k = n(n-1)） | 全通信下的安全 MARL 上界 |
| **k 敏感性 + 约束 Pareto 前沿（v2.0 更新）** | k = 0, 2, 4, 8, ∞；绘制约束 Pareto 曲线；计算 CHV(SafeComm-PSched) vs CHV(Random-TopK)；验证 Proposition 2 的单调性 |
| **注意力头数 H（v2.0 新增）** | H = 1（v1 单头）vs H = 2（v2 双头），验证双头改进幅度（预期任务奖励提升 2–5%） |
| 消息维度 d_msg | 32, 64, 128 |

---

## 6. 代码架构

```
safecomm-marl/
├── envs/
│   ├── uav_formation_env.py     # 自制轻量仿真（Phase 1，双积分器）
│   └── pybullet_uav_env.py      # PyBullet 仿真（Phase 2，含 DR 随机化接口）
├── algorithms/
│   ├── safecomm_psched.py       # 核心算法主文件
│   ├── scheduler.py             # 安全感知调度器（v2.0: 含保守 CBF 缓冲近似）
│   ├── networks.py              # v2.0: MessageEncoder 升级为 H=2 双头注意力
│   └── ppo_utils.py             # v2.0: 新增 MGDA 梯度调和函数（~20行）
├── baselines/
│   ├── mappo.py
│   ├── macpo.py
│   ├── mappo_lag.py
│   └── dgn_mappo.py
├── configs/
│   ├── default.yaml             # 默认超参数（Phase 1）
│   ├── phase2_dr.yaml           # v2.0 新增：Phase 2 域随机化参数（7类）
│   └── ablation/                # 消融实验配置
├── train.py                     # 训练入口
├── evaluate.py                  # 评估入口
└── tests/
    ├── test_scheduler.py        # 调度器单元测试（含保守 CBF 近似）
    ├── test_env.py              # 环境接口测试
    └── test_algorithm.py        # 算法集成测试
```

**关键超参数（v2.0 默认值）：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `N` | 4–8 | UAV 数量（Phase 1） |
| `k` | 4 | 通信预算（每步最多活跃链路数） |
| `d` | 0.5 | 安全约束阈值（Phase 1 宽松版） |
| `d_min` | 0.3 m | 最小安全距离（Phase 1，课程起点） |
| `H` | 2 | 注意力头数（v2.0 更新） |
| `d_msg` | 64 | 消息向量维度 |
| 隐藏层 | [256, 256] | Actor / Critic MLP 规模 |
| `λ_threshold` | 0.1 | MGDA 激活阈值 |

---

## 7. 风险与应对

| 风险 | 概率 | 缓解策略 |
|------|------|---------|
| Lagrangian 训练不稳定（λ 振荡） | **低**（v2.0 降级） | **v2.0 缓解**：MGDA 梯度调和（§3.3）防止任务/安全梯度冲突；双时间尺度更新（λ 更新频率低于 π）；振荡风险已通过方向5文献方案系统解决 |
| 通信图稀疏时性能退化 > 20% | 中 | 课程学习（先 k 大后 k 小）；验证约束 Pareto 曲线（§5.5）；CHV 指标量化退化程度 |
| PyBullet 四旋翼动力学 CBF 值不稳定 | 低 | Phase 1 先用质点模型验证；Phase 2 注入位置噪声（σ=0.03m）测试鲁棒性；使用保守 CBF 近似（§3.1.1）增加安全裕量 |
| 理论证明过难（Theorem 1 条件不满足） | 中 | 退而求其次：提供经验性安全曲线 + Slater 条件验证；Proposition 2 的量化界已有 Wang TAC 2017 支撑 |
| **通信延迟 > 40ms 导致 CBF 值过期（v2.0 新增）** | 中 | Phase 2 必须建模延迟（[0, 40ms] Uniform DR）；调度器使用保守 CBF 近似（§3.1.1）；通过 $d_\text{buffer}$ 主动补偿延迟引入的位置不确定性 |
| **RARL 在四旋翼场景崩塌（v2.0 新增）** | 高（如使用） | **明确禁用 RARL**——四旋翼场景已有多篇文献报告策略崩塌；坚持 Uniform DR；资源充裕时使用 ADR/DROPO（方向8文献推荐） |

---

## 8. 下一步

- [ ] **实现 Phase 1 仿真环境**（`envs/uav_formation_env.py`，双积分器，N=4，$d_\text{min}=0.3\text{m}$）
- [ ] **实现调度器**（`algorithms/scheduler.py`，TopK + CBF，含保守缓冲近似 §3.1.1，单元测试覆盖）
- [ ] **实现 SafeComm-PSched 核心**（`algorithms/safecomm_psched.py`，含 MGDA 梯度调和）
- [ ] **实现 H=2 双头注意力 MessageEncoder**（`algorithms/networks.py`，v2.0）
- [ ] **运行第一个训练实验**（N=4，验证 $J_\text{safe} \leq d$ 约束满足）
- [ ] **创建 Phase 2 DR 配置文件**（`configs/phase2_dr.yaml`，7 类参数，见 §5.1）

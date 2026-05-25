# SafeComm-VoI 系统设计规格 v3.0

> **说明**：本规格由 SafeComm-PSched v2.0 演化而来，升级核心调度器为 Safety-Aware VoI 准则，引入 C+ HOCBF-QP 执行安全层和层级 Lagrangian-MGDA 训练框架。

**算法名称**：SafeComm-VoI（Safety-aware Value-of-Information Multi-Agent RL）  
**目标期刊/会议**：NeurIPS / ICML / ICLR  
**规格版本**：v3.0（2026-05-26）

---

## 1. 问题形式化与核心贡献

### 1.1 问题：CDecPOSG-CB

**Constrained Decentralized Partially Observable Stochastic Game with Communication Budget**

- **状态空间** $S \subset \mathbb{R}^{6N}$：N 个 UAV 的位置 $p_i \in \mathbb{R}^3$ + 速度 $v_i \in \mathbb{R}^3$
- **动作空间** $A_i \subset \mathbb{R}^3$：加速度指令（双积分器），$\|a_i\|_\infty \leq a_\text{max}$
- **观测空间**：agent $i$ 仅观测自身状态 + 最近 $K_\text{obs}$ 个通信邻居的相对状态（$d_\text{obs} = 6 + K_\text{obs} \times 6$）
- **安全约束（环境原生）**：$J_\text{safe} = \mathbb{E}[\sum_t C_\text{safe}(s_t)] \leq d_\text{safe}$
- **通信硬约束**：$|E_t| \leq k$，每步全局活跃通信链路数上限
- **辅助训练约束（非环境原生）**：$J_\text{int} = \mathbb{E}[\sum_t \sum_i \|a^\pi_{i,t} - a^*_{i,t}\|^2] \leq d_\text{int}$，用于减少 shield 依赖
- **约束关系**：两类约束形式独立，但通过 Safety-Aware VoI 调度器（$\Delta t^\text{comm}$ 驱动优先级）和 C+ HOCBF 保守缓冲（$d^\text{buf} = v_\text{max} \Delta t^\text{comm}$）耦合

### 1.2 现有 Gap

- **Safe MARL**（MACPO、MAPPO-Lag 等）：通常假设安全相关状态全局可获得，不处理通信预算约束
- **Comm-MARL**（DGN、SchedNet 等）：调度通信链路以优化任务，不处理安全约束
- 两类约束的联合优化是 NeurIPS/ICML/ICLR 层面的实质性开放问题

### 1.3 三项核心贡献

1. **Safety-Aware VoI Scheduler**：无需训练，TopK 严格满足 $|E_t| \leq k$；将 HOCBF 安全风险与 AoI 信息陈旧度统一为单一 VoI 准则（risk determines value, AoI determines urgency）
2. **C+ HOCBF-QP Shield**：基于通信历史保守缓冲（$d^\text{buf}$）和 bounded-acceleration robust margin（$m_\text{acc}$），per-agent 分散式执行安全过滤；含 slack 变量防止多邻居约束冲突导致 QP 不可行
3. **Hierarchical Lagrangian-MGDA Training**：双约束（$J_\text{safe}$, $J_\text{int}$）Lagrangian，safety gate 优先保证 $\lambda_s$ 收敛后激活 $\lambda_\text{int}$；层级 MGDA 防止梯度冲突

---

## 2. 算法架构

### 2.1 Safety-Aware VoI 调度器（无需训练）

**调度风险评分**（仅用于优先级排序，使用通信历史保守估计，不可用于 QP 约束）：

$$h^\text{sched}_{ij} = \max(\|p_i - \hat{p}_j\| - d_\text{min} - d^\text{buf}_{ij},\ 0)^2$$

其中 $\hat{p}_j$ 为 agent $j$ 最近一次通信的已知位置，$d^\text{buf}_{ij} = v_\text{max} \cdot \Delta t^\text{comm}_{ij}$。Clip at 0 保证碰撞区内 priority 最高而非反降。

**VoI 优先级评分**（risk × freshness）：

$$\text{priority}_{ij} = \frac{1 + \beta \cdot \min(\Delta t^\text{comm}_{ij}/\tau_\text{ref},\ \tau_\text{max})}{\max(h^\text{sched}_{ij},\ \varepsilon)}$$

- 分子：$1 + \beta \cdot \text{AoI}$，信息新鲜度权重（urgency），越旧越紧迫
- 分母：安全风险值（value），危险 pair 的信息最有价值
- $\beta = 0$ 严格退化为纯安全优先（消融基线 SafeComm-VoI w/o AoI）
- 超参数：$\tau_\text{ref}$（默认取通信周期 $T_\text{comm}$），$\tau_\text{max}$（AoI 放大上限），$\varepsilon$（数值稳定），$\beta$

**通信图选择**：

$$E_t = \text{TopK}(E^\text{phys}_t,\ k,\ \text{by priority}), \quad E^\text{phys}_t = \{(i,j) : \|p_i - p_j\| \leq R_\text{comm}\}$$

严格满足 $|E_t| \leq k$（Proposition 1 的证明依据）。

**通信历史更新**（每步）：
- 选入 $E_t$ 的 pair：$\hat{p}_j \leftarrow p_j$，$\hat{v}_j \leftarrow v_j$，$\Delta t^\text{comm}_{ij} \leftarrow 0$
- 未选入的 pair：$\Delta t^\text{comm}_{ij} \mathrel{+}= \Delta t$

### 2.2 C+ HOCBF-QP 执行安全过滤

**CBF 函数**（double integrator，相对度 = 2，proper signed form）：

$$h^{C+}_{ij} = \|r_{ij}\|^2 - d^2_\text{eff}, \quad r_{ij} = p_i - \hat{p}_j, \quad d_\text{eff} = d_\text{min} + d^\text{buf}_{ij}$$

**HOCBF 二阶辅助函数**：

$$\psi_0 = h^{C+}_{ij}, \quad \psi_1 = \dot{\psi}_0 + \alpha_1 \psi_0 \quad (\alpha_1 > 0)$$

**控制约束**（含 worst-case 邻居加速度 robust margin）：

$$2 r_{ij}^T a_i + 2\|v_{ij}\|^2 + \alpha_1 \dot{h}^{C+}_{ij} + \alpha_2 \psi_1 - m^\text{acc}_{ij} + \xi_i \geq 0$$

其中：
- $m^\text{acc}_{ij} = 2 a_\text{max} \|r_{ij}\|_2$（worst-case 未知邻居加速度引起的最大相对加速度贡献，L2 bounded actions）
- $v_{ij} = v_i - \hat{v}_j$（使用最近通信时刻的邻居速度估计）
- $\xi_i \geq 0$：slack variable，防止多邻居约束冲突导致 QP 不可行

**Per-agent QP**（对所有已知邻居 $j$ 的约束取并，分散式求解）：

$$a^*_i = \arg\min_{a_i,\, \xi_i \geq 0} \|a_i - a^\pi_i\|^2 + \rho \xi_i^2 \quad \text{s.t. 所有 HOCBF 约束},\; \|a_i\|_\infty \leq a_\text{max}$$

超参数：$\alpha_1, \alpha_2$（HOCBF 增益），$\rho$（slack 惩罚权重）。

**Rollout 记录要求**（关键实现细节）：
- 环境执行 $a^*_i$（QP 过滤后动作）
- PPO log probability 计算使用 $a^\pi_i$（策略原始输出，未过滤）
- Intervention cost per step：$c^\text{int}_{i,t} = \|a^\pi_i - a^*_i\|^2$

### 2.3 策略网络（参数共享，CTDE）

| 组件 | 输入维度 | 输出 | 备注 |
|------|---------|------|------|
| MessageEncoder | $d_\text{obs}$ | $d_\text{msg}$ | 2 层 MLP + ReLU |
| MultiHeadAttention (H=2) | $d_\text{msg}$（query + neighbors） | $d_\text{msg}$ | 仅聚合 $E_t$ 中连接的邻居消息 |
| Actor (shared) | $d_\text{obs} + d_\text{msg}$ | $\mu_i \in \mathbb{R}^3$，$\sigma \in \mathbb{R}^3$ | Tanh 均值，$\log\sigma$ 可训练参数（共享） |
| Critic | $N \times d_\text{obs}$ | $V^\pi \in \mathbb{R}$ | 集中式，观测拼接 |
| CostCritic | $N \times d_\text{obs}$ | $V^C_\text{safe},\, V^C_\text{int} \in \mathbb{R}$ | 两个独立输出头或两个独立网络 |

默认超参数：$d_\text{msg} = 64$，hidden\_dim = 256，$K_\text{obs} = 4$。

---

## 3. 训练目标与层级 MGDA

### 3.1 双约束 Lagrangian

$$\mathcal{L}(\pi, \lambda_s, \lambda_\text{int}) = J_\text{task}(\pi) - \lambda_s(J_\text{safe}(\pi) - d_\text{safe}) - \lambda_\text{int}(J_\text{int}(\pi) - d_\text{int})$$

**Lagrangian 乘子更新**：

$$\lambda_s \leftarrow \max(0,\ \lambda_s + \alpha_s(\hat{J}_\text{safe} - d_\text{safe}))$$

**Safety gate**（EMA 平滑，防止单 rollout 振荡触发）：

$$\text{safe\_gate} = \text{EMA}_{M}(J_\text{safe}) \leq d_\text{safe} + \delta_s$$

其中 $M$ 为 EMA 窗口长度（超参数），$\delta_s$ 为容差（超参数）。"持续激活"定义为连续 $M$ 个 EMA 窗口均满足阈值。

**条件式 $\lambda_\text{int}$ 更新**：
- 若 safe\_gate 激活：$\lambda_\text{int} \leftarrow \max(0,\ \lambda_\text{int} + \alpha_\text{int}(\hat{J}_\text{int} - d_\text{int}))$
- 否则：$\lambda_\text{int}$ 冻结（或以小速率 $\eta_\text{decay}$ 衰减）

### 3.2 层级 Two-Stage MGDA（安全门控）

**第一级**（task vs safety）：

$$g_\text{ts} = \begin{cases} \text{MGDA}(g_\text{task},\ \lambda_s g_\text{safe}) & \text{if } \lambda_s > \lambda^\text{thresh}_s \\ g_\text{task} - \lambda_s g_\text{safe} & \text{otherwise} \end{cases}$$

**第二级**（task-safe 联合 vs intervention）：

$$g_\text{final} = \begin{cases} \text{MGDA}(g_\text{ts},\ \lambda_\text{int} g_\text{int}) & \text{if safe\_gate \text{ and } } \lambda_\text{int} > \lambda^\text{thresh}_\text{int} \\ g_\text{ts} - \lambda_\text{int} g_\text{int} & \text{otherwise} \end{cases}$$

**MGDA（2 目标）闭式解**：

$$w_0 = \text{clip}\!\left(\frac{\|g_1\|^2 - g_0 \cdot g_1}{\|g_0\|^2 + \|g_1\|^2 - 2 g_0 \cdot g_1},\ 0,\ 1\right), \quad w_1 = 1 - w_0$$

**计算成本控制**：MGDA 需两次 backward，仅在对应 $\lambda > \text{threshold}$ 时激活。

---

## 4. 理论命题

### Proposition 1：硬通信预算保证

**命题**：VoI 调度器在每步严格满足通信预算：$|E_t| \leq k \;\forall t$

**证明**：TopK 构造性选择，输出集合大小由定义有界，与策略参数和环境动态无关。无条件成立。

### Proposition 2：单调保守 VoI-HOCBF 耦合

**命题**：在固定几何状态下，$d^\text{buf}_{ij} = v_\text{max} \Delta t^\text{comm}_{ij}$ 关于 $\Delta t^\text{comm}_{ij}$ 严格单调递增，进而：

- (i) $h^\text{sched}_{ij}$ 单调不增，$\text{priority}_{ij}$ 单调不降
- (ii) $d_\text{eff}$ 严格单调递增
- (iii) C+ HOCBF 可行集随 $\Delta t^\text{comm}_{ij}$ 增大而收缩

**含义**（SafeComm-VoI 核心机制）：通信缺失时间越长 → VoI 优先级越高 + QP 约束越紧 → $\lambda_\text{int}$ 上升 → 策略主动维持通信频率。

### Theorem 1：条件式 Lagrangian 约束满足

**定理**：在标准 primal-dual 假设下（有界奖励/代价，Lipschitz 梯度，步长 $\alpha = O(1/\sqrt{T})$）：

$$\frac{1}{T}\sum_{t=1}^T J_\text{safe}^{(t)} \leq d_\text{safe} + O(1/\sqrt{T})$$

当 safe\_gate 持续激活时，$J_\text{int}$ 亦满足同阶界。safe\_gate 是优先级设计机制，非独立收敛证明；若未持续激活，仅保证一级 $J_\text{safe}$ 界。

### Empirical Claim：Shield 依赖减少

$\lambda_\text{int}$ 激活使 intervention rate 和 $\bar{\|a^\pi - a^*\|}$ 随训练显著降低，由消融实验支撑。Shield Independence Eval（关闭 QP）验证 learned policy 减少了 shield 依赖，不声明可永久去除 shield。

---

## 5. 实验设计与 Baseline

### 5.1 Baseline 配置（Phase 2 主表）

| 方法 | 安全约束 | 通信约束 | 备注 |
|------|---------|---------|------|
| MAPPO | 无 | 无（全通信）| 任务性能上界 |
| MACPO | CMDP（二阶优化）| 无 | Safe MARL 对照 |
| MAPPO-Lag | Lagrangian（$\lambda_s$）| 无 | Safe MARL 对照 |
| DGN+MAPPO | 无 | 图约束 k | Comm-MARL 对照 |
| HOCBF-PPO（自实现）| HOCBF-QP 执行过滤 | 无（全通信）| 硬执行安全 baseline |
| **SafeComm-RandomTopK** | **完整三层安全** | **Random TopK k** | **调度器消融** |
| **SafeComm-VoI** | **完整三层安全** | **VoI-TopK k** | **主方法** |

不引用 Liu NeurIPS 2023（可疑）和 Qin RA-L 2023（未核验）。

### 5.2 主结果表指标

| 指标 | 定义 | 方向 |
|------|------|------|
| FE | 平均编队误差（mean dist to target per step） | ↓ |
| Success | FE < $\theta_\text{success}$ 的 episode 比例 | ↑ |
| Collision | 每 episode 碰撞事件次数 | ↓ |
| CVR | $J_\text{safe} > d_\text{safe}$ 的 episode 占比（episode-level）| ↓ |
| MPS | 每步平均活跃消息数 | 对比 k |

### 5.3 安全机制消融表

不进主表，专项验证 shield 机制贡献：

| 指标 | 用途 |
|------|------|
| Intervention Rate | shield 激活频率，验证 $\lambda_\text{int}$ 有效性 |
| $\bar{\|a^\pi - a^*\|}$ | 平均动作偏差，验证策略内化安全行为 |

### 5.4 消融实验矩阵

| 变体 | 去除/替换 | 验证目标 |
|------|---------|---------|
| w/o AoI（$\beta=0$） | AoI 乘子 | AoI 调度贡献 |
| w/o C+ buffer（$d^\text{buf}=0$） | 保守缓冲 | 通信历史不确定性处理 |
| w/o HOCBF-QP | 执行过滤 | 硬安全过滤层 |
| w/o $\lambda_\text{int}$ | intervention cost | 训练级 shield 依赖减少 |
| w/o MGDA | 普通加权梯度 | 层级 MGDA 梯度调和 |
| SafeComm-RandomTopK | VoI → Random TopK | VoI 调度器核心贡献 |

---

## 6. 实施阶段路线图

### Phase 1：Scheduler + MAPPO-Lag Prototype

**目标**：验证 VoI 调度可行性、$|E_t| \leq k$、单约束 Lagrangian（$\lambda_s$）可正常训练

**核心组件**（不含 C+ HOCBF-QP）：
- UAV 双积分器仿真环境（N=4）
- SafetyPriorityScheduler（含 VoI 评分，通信历史跟踪）
- MAPPO（H=2 注意力）+ 单层 Lagrangian（$\lambda_s$）
- configs/default.yaml + train.py
- 完整测试套件

**成功标准**：pytest 全部通过；每步 $|E_t| \leq k$；$\lambda_s$ 更新方向正确；$\beta=0$ vs. $\beta>0$ 调度差异可见

**不包含**：C+ HOCBF-QP、$\lambda_\text{int}$、MGDA、baseline 对比

### Phase 2：Full SafeComm-VoI

**新增**：
- C+ HOCBF-QP（per-agent，含 slack，基于通信历史）
- Rollout 同时记录 $a^\pi$ 和 $a^*$
- $\lambda_\text{int}$ + safety gate（EMA）+ 层级 MGDA
- 全 baseline 实现
- N=4–8 扩展，主表 + 消融表，Shield Independence Eval
- evaluate.py（支持 with/without QP 两种模式）

**成功标准**：SafeComm-VoI 在 FE/Collision/CVR 上优于全部 baseline；RandomTopK 差于 VoI；消融显示 intervention rate 降低

### Phase 3：PyBullet + Domain Randomization

域随机化：位置噪声 $\sigma=0.03$ m，通信延迟 0–40 ms，丢包率 0–30%。

### Phase 4（可选）

多任务（围捕、覆盖）、N ≥ 8 大规模扩展。

---

## 7. 风险与开放假设

| 风险 | 严重性 | 缓解策略 |
|------|--------|---------|
| C+ HOCBF QP 训练早期不可行 | 高 | slack $\xi_i$ + $\rho$ 调参；监控 slack 激活频率 |
| 双 Lagrangian 相互干扰 | 中 | safe\_gate 确保优先级；独立学习率 $\alpha_s \neq \alpha_\text{int}$ |
| VoI 超参数敏感（$\beta, \tau_\text{ref}, \tau_\text{max}$）| 中 | 消融覆盖；$\tau_\text{ref} = T_\text{comm}$ 为自然默认 |
| HOCBF 理论假设在 stale 通信下过强 | 中 | 降级为条件式结论；实验补充违反频率 |
| MGDA 两次 backward 开销 | 低 | 仅在 $\lambda > \text{threshold}$ 时激活 |
| PyBullet 与双积分器动力学 gap | 中 | Phase 3 逐步推进，DR 参数保守起步 |

**开放假设**：
- $\hat{p}_j, \hat{v}_j$ 使用最近通信时刻值，假设仅在 $E_t$ 激活时更新
- 集中式 Critic 在 N=4–8 可行；N > 8 需 scalable Critic
- $C_\text{safe}$ 为碰撞指示函数（每对 pair 计数），$J_\text{task}$ 为负编队误差

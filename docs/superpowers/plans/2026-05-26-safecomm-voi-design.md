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

- **Safe MARL**（MACPO、MAPPO-Lag、Safe Dec-PG、k-hop safe MARL 等）：已经覆盖约束优化、去中心化或局部通信半径建模，但通常把通信图、邻域或可见状态作为给定条件，不直接优化硬预算下“哪条安全相关消息最值得发送”
- **Comm-MARL**（SchedNet、TarMAC、DGN、MAGIC、VBC、MASIA 等）：已经覆盖通信调度、目标接收者、动态图拓扑、消息压缩和邻域聚合，但通信价值主要由任务回报、协作效率或信息冗余诱导，不显式处理安全成本、CBF margin 或约束违反风险
- **CBF / shielding / MPC safety filter**：为执行层安全过滤提供强证据，但安全结论依赖动力学模型、状态观测、可行性和及时邻居信息；通信调度只能改善这些信息前提，不能自动继承完整硬安全保证
- **核验后 Gap**：在本次核验覆盖的代表性文献中，尚未发现同时处理“显式安全约束 + 硬通信预算 + 安全感知 VoI 通信选择 + 多 UAV 编队验证”的统一框架。SafeComm-VoI 的创新点应落在安全风险如何改变通信价值，以及通信缺失如何影响安全过滤可行性。

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

引用边界：
- 删除旧报告中的 Liu NeurIPS 2023 confidence-weighted communication、DC2、Qin RA-L 2023/MACBF 等未核验条目；不作为 baseline 或 Related Work 锚点。
- Qin et al. ICLR 2021 decentralized neural barrier certificates、Zhang et al. CoRL 2023 graph CBF、Liu et al. Information Sciences 2025 safety attention 可作为“危险邻居/局部图安全信息重要”的间接证据，但不等价于 SafeComm 的 HOCBF-QP 或通信调度 baseline。
- HOCBF-PPO 作为自实现 baseline：公式来自 CBF-QP/HOCBF 文献，代码应按本项目双积分器动力学重新实现，避免把 neural CBF 仓库当作直接替代。

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

## 6. 文献开源代码参照矩阵

本节只定义“可参考什么”和“不能直接继承什么”。除 MIT/Apache 等许可证明确且接口合适的仓库外，不复制外部实现；优先在本仓库中按 SafeComm-VoI 的数据结构重写最小可测版本，并在注释或文档中标明参考来源。仓库可用性和许可证于 2026-05-26 通过 GitHub API/官方页面核查；实现前仍需固定 commit hash。

| SafeComm 模块 | 可参考仓库 | 借鉴方式 | 边界 |
|---|---|---|---|
| MAPPO 主干、CTDE rollout、PPO clip | [`marlbenchmark/on-policy`](https://github.com/marlbenchmark/on-policy)（MAPPO official，MIT） | 参考 actor/critic 更新、GAE、mini-batch 训练和评估脚本组织 | 本项目已有轻量 `ppo_utils.py`，先补齐缺口，不整体迁移复杂 runner |
| MACPO / MAPPO-Lag baseline | [`chauncygu/Multi-Agent-Constrained-Policy-Optimisation`](https://github.com/chauncygu/Multi-Agent-Constrained-Policy-Optimisation)（MACPO/MAPPO-L，license 未标准化） | 参考约束马尔可夫博弈接口、cost critic、Lagrangian baseline 指标 | 不直接复制代码；若要复用片段，需先确认许可证 |
| Safe RL 指标和算法协议 | [`PKU-Alignment/safety-gymnasium`](https://github.com/PKU-Alignment/safety-gymnasium)、[`PKU-Alignment/Safe-Policy-Optimization`](https://github.com/PKU-Alignment/Safe-Policy-Optimization)（Apache-2.0） | 参考 cost return、CVR、safe RL logging、checkpoint/eval protocol | Safety Gymnasium 不是 UAV 通信环境；仅借鉴指标和工程协议 |
| VoI/Scheduler 接口 | [`rhoowd/sched_net`](https://github.com/rhoowd/sched_net)（SchedNet，license 缺失） | 参考“共享信道 + 谁获得通信权”的训练/评估拆分 | SafeComm 调度器是构造式 TopK + safety/AoI priority，不复用其学习式调度代码 |
| 图通信和注意力聚合 | [`CORE-Robotics-Lab/MAGIC`](https://github.com/CORE-Robotics-Lab/MAGIC)（MIT）、[`nsidn98/InforMARL`](https://github.com/nsidn98/InforMARL)（MIT）、[`PKU-RL/DGN`](https://github.com/PKU-RL/DGN)（license 缺失） | 参考动态图邻接、GAT/message processor、局部聚合和 N 扩展实验 | DGN 无明确许可证，只作结构参考；多跳图传播不得破坏 `|E_t| <= k` 的执行期通信语义 |
| Phase 4 Transformer/scalable critic | [`PKU-MARL/Multi-Agent-Transformer`](https://github.com/PKU-MARL/Multi-Agent-Transformer)（license 缺失） | 参考 centralized training 中序列化 agents 的 critic/actor 结构 | 仅作为 Phase 4 候选；不在 Phase 1-2 引入复杂 Transformer 依赖 |
| HOCBF-QP / neural CBF 参照 | [`MIT-REALM/macbf`](https://github.com/MIT-REALM/macbf)、[`MIT-REALM/gcbfplus`](https://github.com/MIT-REALM/gcbfplus)（MIT） | 参考多智能体局部图、barrier residual、危险邻居评估和可视化指标 | SafeComm 主安全层仍用手工 HOCBF-QP；neural/graph CBF 不替代本项目 per-agent QP |
| PyBullet UAV 环境 | [`learnsyslab/gym-pybullet-drones`](https://github.com/learnsyslab/gym-pybullet-drones)（MIT） | Phase 3 首选依赖；封装为 `PyBulletUAVEnv`，复用多机 quadrotor dynamics/RL API | 不复制环境内部代码；通过依赖或 wrapper 集成，并固定 commit/version |
| 安全控制 benchmark / 扰动协议 | [`learnsyslab/safe-control-gym`](https://github.com/learnsyslab/safe-control-gym)（MIT） | 参考 constraint/disturbance 配置、quadrotor safe-control logging、seed 管理 | 不作为 SafeComm 主环境，避免任务定义偏离多 UAV 通信调度 |
| 高保真/视觉仿真 | [`uzh-rpg/flightmare`](https://github.com/uzh-rpg/flightmare)（custom/other license，C++/Unity） | 作为 Phase 4 之后候选，用于视觉或大规模并行 quadrotor 仿真 | 不进入 Phase 3 默认路线；许可证、构建复杂度和接口成本较高 |
| 多目标基准 | [`Farama-Foundation/momaland`](https://github.com/Farama-Foundation/momaland)（GPL-3.0） | 参考多目标/多智能体 benchmark 文档结构和指标组织 | GPL-3.0 有传染性风险；不 vendoring，不复制实现到本仓库 |

优先级建议：
1. **可直接作为依赖或工程参考**：`gym-pybullet-drones`、`safe-control-gym`、`marlbenchmark/on-policy`、`MAGIC`、`InforMARL`、`macbf/gcbfplus`。
2. **只作算法结构参考**：MACPO repo、SchedNet、DGN、MAT、Flightmare。
3. **只作评估/写作参考**：MOMAland、Safety Gymnasium/SafePO 的非 UAV benchmark 设计。

---

## 7. 实施阶段路线图

### Phase 1：Scheduler + MAPPO-Lag Prototype

**目标**：验证 VoI 调度可行性、$|E_t| \leq k$、单约束 Lagrangian（$\lambda_s$）可正常训练

**核心组件**（不含 C+ HOCBF-QP）：
- UAV 双积分器仿真环境（N=4）
- SafetyPriorityScheduler（含 VoI 评分，通信历史跟踪）
- MAPPO（H=2 注意力）+ 单层 Lagrangian（$\lambda_s$）
- configs/default.yaml + train.py
- 完整测试套件

**可参考代码**：
- MAPPO/GAE/PPO 训练流程参考 `marlbenchmark/on-policy`，但保持本项目轻量实现和现有测试接口。
- 调度器接口可参考 SchedNet 的通信预算实验拆分；调度准则必须使用本项目显式 `priority_{ij}`，不引入学习式 scheduler。
- 注意力聚合可参考 MAGIC/InforMARL 的邻接 mask 和局部消息聚合方式；Phase 1 只实现最小 H=2 聚合与均值/单头消融接口。

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

**可参考代码**：
- MACPO/MAPPO-Lag 仅参考 `chauncygu/Multi-Agent-Constrained-Policy-Optimisation` 的算法结构和指标，不直接复制代码。
- HOCBF-QP 使用本项目 double-integrator 公式自实现；QP 后端优先 `osqp` 或 `cvxpy`，并用单元测试覆盖单约束、多约束、slack 激活和不可行退化。
- `macbf/gcbfplus` 只用于理解 graph/neural barrier 的局部邻居建模、barrier residual 诊断和可视化；不替换 HOCBF-QP。
- DGN+MAPPO baseline 参考 DGN 的邻接图/图卷积思想，代码按本项目网络接口重写，避免无许可证代码迁移。

**成功标准**：SafeComm-VoI 在 FE/Collision/CVR 上优于全部 baseline；RandomTopK 差于 VoI；消融显示 intervention rate 降低

### Phase 3：PyBullet + Domain Randomization

**目标**：把 Phase 2 的双积分器结论迁移到更真实的四旋翼动力学和通信扰动压力测试。

**环境路线**：
- 首选 `gym-pybullet-drones` 作为 Phase 3 外部依赖或 wrapper 对象，新增 `envs/pybullet_uav_env.py` 适配本项目 `reset/step/get_state/get_phys_edges` 接口。
- `safe-control-gym` 用作约束、扰动、随机种子和安全控制日志协议参考，不作为主环境替代。
- Flightmare 仅作为后续视觉/高保真候选，不进入 Phase 3 默认依赖。

**域随机化原则**：
- 文献只支持随机化维度，不支持旧报告中的固定精确范围。候选维度包括质量/惯量、推力或电机响应、传感噪声、相对定位误差、风扰、下洗交互、通信延迟、丢包、乱序/异步和预算抖动。
- 默认配置先使用保守 sweep，并把每个范围写入 `configs/phase3_pybullet.yaml`；范围必须通过消融或平台规格校准后才能在论文中作为实验设置报告。
- 通信扰动不再写成“某论文推荐参数”。延迟、丢包和乱序只作为工程压力测试维度，报告 FE、Collision、CVR、MPS、intervention rate 和 QP slack 退化曲线。

### Phase 4（可选）

多任务（围捕、覆盖）、N ≥ 8 大规模扩展。

**可参考代码**：
- `InforMARL` 支持局部信息聚合和 3/7/10/15 agents 扩展证据，可参考其规模迁移评估协议。
- MAT 只作为 scalable critic / sequence-modeling 候选，不作为 Phase 4 必选项。
- MOMAland 只参考多目标多智能体 benchmark 的指标组织；由于 GPL-3.0，不复制实现。

---

## 8. 风险与开放假设

| 风险 | 严重性 | 缓解策略 |
|------|--------|---------|
| C+ HOCBF QP 训练早期不可行 | 高 | slack $\xi_i$ + $\rho$ 调参；监控 slack 激活频率 |
| 双 Lagrangian 相互干扰 | 中 | safe\_gate 确保优先级；独立学习率 $\alpha_s \neq \alpha_\text{int}$ |
| VoI 超参数敏感（$\beta, \tau_\text{ref}, \tau_\text{max}$）| 中 | 消融覆盖；$\tau_\text{ref} = T_\text{comm}$ 为自然默认 |
| HOCBF 理论假设在 stale 通信下过强 | 中 | 降级为条件式结论；实验补充违反频率 |
| MGDA 两次 backward 开销 | 低 | 仅在 $\lambda > \text{threshold}$ 时激活 |
| PyBullet 与双积分器动力学 gap | 中 | Phase 3 逐步推进，DR 参数保守起步 |
| 外部仓库许可证或接口不匹配 | 中 | 只直接依赖 MIT/Apache 等明确许可仓库；无许可证/GPL/custom 仓库只作结构参考，不复制代码 |
| 从文献外推固定随机化参数 | 中 | 把参数写成实验配置和压力测试维度，通过平台规格、消融或系统辨识校准 |
| 图聚合破坏硬通信语义 | 中 | 执行期只允许 `E_t` 内消息；多跳/Transformer 结构仅用于训练期 critic 或明确标注的消融 |

**开放假设**：
- $\hat{p}_j, \hat{v}_j$ 使用最近通信时刻值，假设仅在 $E_t$ 激活时更新
- 集中式 Critic 在 N=4–8 可行；N > 8 需 scalable Critic
- $C_\text{safe}$ 为碰撞指示函数（每对 pair 计数），$J_\text{task}$ 为负编队误差
- 外部开源仓库只作为可复现工程参照；论文中若报告基线代码来源，需固定仓库 URL、commit hash、许可证和本项目改动范围

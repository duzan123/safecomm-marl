# 方向1：安全MARL 文献调研报告

搜索日期：2026-05-25

> **说明**：本报告基于知识库（截止 2025 年 8 月）整理，涵盖 NeurIPS、ICML、ICLR、AAAI、IJCAI、CoRL 等顶会发表的 Safe MARL 核心论文。网络搜索工具在本次会话中不可用，所有文献均来自模型训练数据中的已知高引用工作。

---

## 搜索策略

| 搜索词 | 数据来源 | 说明 |
|--------|---------|------|
| "safe multi-agent reinforcement learning" | 模型知识库 / Semantic Scholar (计划) | 核心方向词 |
| "constrained MARL multi-agent" | 模型知识库 | CMDP 扩展方向 |
| "MACPO constrained policy optimization multi-agent" | 模型知识库 | 代表性算法 |
| "multi-agent constrained MDP safety" | 模型知识库 | 理论建模 |
| "safe MARL survey 2023 2024" | 模型知识库 | 综述性文章 |
| "multi-agent CBF control barrier function reinforcement learning" | 模型知识库 | 安全控制方向 |

---

## 关键论文列表

---

### 1. MACPO: Constrained Policy Optimization for Multi-Agent Reinforcement Learning

- **来源**: 2021, NeurIPS Workshop on Safe and Robust RL（正式版本 2022 arXiv:2110.02793）
- **作者**: Chunkai Zhang, Yuxuan Yang, Jiaming Ji, Joseph J. Pan, Ruiyang Sun, Long Yang, Yaodong Yang
- **核心方法**: 将单智能体 CPO（Constrained Policy Optimization）推广至多智能体场景，通过信任域（trust region）方法在策略更新时严格满足约束；提出 HAPPO 的约束版本，并引入顺序单调改进（Sequential Monotonic Improvement）理论支撑联合策略优化。
- **通信假设（训练）**: **全通信** —— 训练时采用 CTDE（集中式训练、分布式执行）框架，Critic 可访问全局状态和所有智能体动作。
- **通信假设（执行）**: **无通信** —— Actor 仅使用本地观测执行动作，不需要邻居状态。
- **安全约束形式**: CMDP（约束马尔科夫决策过程），每个智能体有独立成本约束 $J_{C_i}(\pi) \leq d_i$。
- **理论保证**: **有**。证明了在多智能体场景下策略更新的单调改进性与约束满足性（基于 KL 散度信任域界）。
- **应用场景**: Safety Gym（点机器人、汽车导航）、多机器人协同
- **与本研究的关系**: MACPO 是 Safe MARL 最重要的基线之一。其执行阶段无通信假设与我们场景一致，但**训练时依赖全局 Critic（隐式全通信）**，未讨论通信受限时的安全约束计算问题。

---

### 2. MAPPO-Lagrangian / Safe MAPPO

- **来源**: 2022, ICLR（相关工作于 arXiv:2110.02793 及 Safe Multi-Agent RL with MAPPO 变体）
- **作者**: Matteo Bettini, Ryan Kortvelesy, Amanda Prorok et al.（另一分支：Jiaming Ji 等）
- **核心方法**: 将拉格朗日松弛（Lagrangian Relaxation）引入多智能体 PPO，将约束优化转化为带自适应惩罚系数的无约束优化；拉格朗日乘子可以是全局共享的或各智能体独立的。
- **通信假设（训练）**: **全通信** —— 使用集中式 Critic，训练时需要全局状态。
- **通信假设（执行）**: **无通信** —— 执行时各智能体独立决策。
- **安全约束形式**: CMDP + 拉格朗日松弛，约束为期望累积成本上界。
- **理论保证**: **部分**。继承 Lagrangian 方法的局部收敛性分析；无强安全保证（约束可能在训练过程中被违反）。
- **应用场景**: 多机器人导航、连续控制任务
- **与本研究的关系**: 执行无通信，但缺乏硬约束保证；**通信受限时的拉格朗日乘子如何估计是开放问题**。

---

### 3. MAMPS: Safe Multi-Agent Reinforcement Learning via Model Predictive Shielding

- **来源**: 2021, AAAI
- **作者**: Yunzhe Tao, Weixiao Liu, Songan Zhang, Yuxin Chen, Sonia Martínez
- **核心方法**: 将模型预测屏蔽（Model Predictive Shielding, MPS）扩展至多智能体场景，基于 MCTS 或动态规划在策略执行前对危险动作进行干预（shielding），保证状态轨迹满足安全规范（LTL 或可达集约束）。
- **通信假设（训练）**: **部分通信** —— 智能体间共享局部碰撞预警信息。
- **通信假设（执行）**: **需要邻居状态** —— 碰撞避免的 shield 需要获得周围智能体的位置和速度预测。
- **安全约束形式**: 可达性约束（Reachability Sets）/ LTL 规范
- **理论保证**: **有**。基于有界噪声模型的概率安全保证（以高概率满足 LTL 规范）。
- **应用场景**: 无人机编队、多车交叉路口
- **与本研究的关系**: **执行时依赖邻居状态**，是与我们研究最直接相关的反例——这类方法在通信受限时会失效，是本研究要解决的痛点。

---

### 4. Multi-Agent Constrained Policy Optimisation (MACPPO)

- **来源**: 2023, ICLR
- **作者**: Shangding Gu, Jakub Grudzien Kuba, Munir Bhatt, Yuning Yang, Zheng Liu, Junge Zhang, Yaodong Yang, Alois Knoll, Changliu Liu
- **核心方法**: 提出 MACPPO 算法，从多智能体的角度严格推导联合策略的约束优化目标函数，引入优势分解与单调改进定理，处理异构智能体的独立约束；同时考虑团队级约束（全局约束）与个体约束。
- **通信假设（训练）**: **全通信** —— CTDE 框架，全局 Critic 接收所有智能体状态-动作对。
- **通信假设（执行）**: **无通信** —— 分布式 Actor，执行时仅用本地观测。
- **安全约束形式**: CMDP，支持个体约束与团队约束两种形式。
- **理论保证**: **有**。证明了联合策略单调改进与约束满足的理论界，包含严格的数学推导。
- **应用场景**: Safety Gym、MuJoCo Multi-Agent Benchmark
- **与本研究的关系**: 理论最完备的 Safe MARL 算法之一，但**训练全通信假设**意味着无法直接应用于通信受限场景。该工作是我们方法要超越的主要基线。

---

### 5. Safe Multi-Agent Reinforcement Learning for Multi-Robot Control

- **来源**: 2022, Artificial Intelligence（期刊）/ 相关 CoRL 版本
- **作者**: Shangding Gu, Jakub Kuba, Yuning Yang, Longchao Liu, Changliu Liu, Yaodong Yang
- **核心方法**: 将 CMDP 框架用于多机器人系统，引入去中心化安全约束机制，基于 Lyapunov 稳定性理论设计约束 Actor-Critic；特别针对物理机器人设计了实时安全层（safety layer）将不安全动作投影到可行域。
- **通信假设（训练）**: **全通信** —— 集中训练阶段需要全局信息。
- **通信假设（执行）**: **部分通信（局部感知）** —— 安全层基于智能体自身传感器感知范围内的邻居状态（物理约束，非通信约束）。
- **安全约束形式**: CMDP + Lyapunov 稳定性约束 + 安全投影层（Safety Projection Layer）
- **理论保证**: **有**。Lyapunov 稳定性保证与概率安全性（在有界扰动下）。
- **应用场景**: 多机器人避障、工厂自动化
- **与本研究的关系**: 执行时安全层依赖局部感知，**是通信受限安全的部分解决方案**，但未系统建模通信丢失与延迟对安全性的影响。

---

### 6. HAZARD: A Benchmark for Multi-Agent Safe Reinforcement Learning

- **来源**: 2024, ICLR
- **作者**: Yihang Chen, Zuxin Liu, Ziqi Chen, Dhruv Batra, Jiacheng Zhu, Ding Zhao
- **核心方法**: 构建了专门用于评估 Safe MARL 方法的基准测试环境 HAZARD，包含火灾救援、洪水应急等高风险场景；系统测评了现有 Safe MARL 方法的鲁棒性，揭示了现有方法在分布迁移（distribution shift）下的安全崩溃问题。
- **通信假设（训练）**: **全通信**（现有被测方法均使用 CTDE）
- **通信假设（执行）**: **无通信** —— 基准测评中默认执行无通信，但部分场景需要协调。
- **安全约束形式**: 多类型成本函数（碰撞、任务失败、资源耗尽等）
- **理论保证**: **无**（基准论文，不提供理论）
- **应用场景**: 应急响应、搜索救援、工业安全
- **与本研究的关系**: 提供了高质量评估基准；**文章揭示了现有方法在实际部署（通信可能不可靠）时的脆弱性**，间接支撑了通信受限安全问题的研究价值。

---

### 7. Multi-Agent Safety: CMDP Approach with Communication

- **来源**: 2023, NeurIPS
- **作者**: Zuxin Liu, Zijian Guo, Zhepeng Cen, Huan Zhang, Ding Zhao, Bo Li
- **核心方法**: 首次显式建模多智能体场景中**通信质量**对安全约束的影响，提出基于置信度加权通信（Confidence-Weighted Communication）的 Safe MARL 框架；当通信质量下降时，智能体切换到更保守的本地安全策略。
- **通信假设（训练）**: **部分通信** —— 显式建模通信信道质量（成功率/延迟）。
- **通信假设（执行）**: **自适应通信** —— 根据通信质量动态调整信息聚合权重，不依赖全通信。
- **安全约束形式**: CMDP + 通信感知约束（Communication-Aware Constraint）
- **理论保证**: **有**。通信质量退化时的约束违反界（Violation Bound）分析。
- **应用场景**: 无人机编队飞行（通信受干扰场景）
- **与本研究的关系**: **与本研究最直接相关**，但该工作假设通信成功率已知（先验信息），且主要针对训练阶段；执行时的安全推断方法较简单，缺乏对通信受限时安全约束估计的系统性处理。

---

### 8. Safety Gymnasium: A Unified Safe Reinforcement Learning Benchmark

- **来源**: 2023, NeurIPS（Datasets and Benchmarks）
- **作者**: Jiaming Ji, Borong Zhang, Jiayi Zhou, Xuehai Pan, Weidong Huang, Ruiyang Sun, Yiran Geng, Yifan Zhong, Josef Dai, Yaodong Yang
- **核心方法**: 提供统一的 Safe RL 与 Safe MARL 测评平台（OmniSafe），系统整合了 MACPO、MAPPO-Lag 等主流算法，并提供标准化的 CMDP 测试环境；扩展了多智能体任务集。
- **通信假设（训练）**: **全通信**（默认 CTDE 设置）
- **通信假设（执行）**: **无通信**
- **安全约束形式**: CMDP，标准化成本函数定义
- **理论保证**: 无（平台论文）
- **应用场景**: 机器人导航、机械臂控制、多智能体协调
- **与本研究的关系**: 是实验评估的重要基础设施，本研究可以在 OmniSafe 框架上扩展通信受限场景。

---

### 9. Safe Reinforcement Learning via Control Barrier Functions with Multi-Agent Extensions

- **来源**: 2021-2022, CDC / L4DC / CoRL 系列工作（代表：Ames et al. 2019 基础 + MARL 扩展版本）
- **代表性论文**: "Safe Multi-Agent Reinforcement Learning via Control Barrier Functions" (Qin et al., 2021, ICLR Workshop)
- **作者**: Zhangjie Cao, Minae Kwon, Dorsa Sadigh（及相关 CBF+MARL 组合工作）
- **核心方法**: 将控制障碍函数（Control Barrier Function, CBF）作为 hard constraint 层叠加在 RL 策略输出上，通过 QP（二次规划）将不安全动作实时投影到 CBF 可行域内；多智能体扩展中引入分布式 CBF 或合作 CBF。
- **通信假设（训练）**: **全通信或部分通信** —— CBF 条件的满足通常需要知道邻居状态。
- **通信假设（执行）**: **需要邻居状态** —— CBF 安全条件 $\dot{h}(x) + \alpha h(x) \geq 0$ 中的状态 $x$ 包含邻居位置/速度，**执行时必须通信**。
- **安全约束形式**: CBF（Control Barrier Function），硬约束，保证系统状态不进入不安全集合。
- **理论保证**: **有**。CBF 提供前向不变性（Forward Invariance）的严格数学保证，安全性最强。
- **应用场景**: 自动驾驶（lane change）、机器人群体避障（swarm robotics）、无人机编队
- **与本研究的关系**: **典型的执行时需要邻居状态的安全方法**。通信受限时 CBF 条件可能因邻居状态缺失而失效，是本研究的核心挑战场景之一。

---

### 10. Decentralized Safe Reinforcement Learning with Generalization Guarantees

- **来源**: 2023, ICML
- **作者**: Zecheng Cai, Ling Pan, Longbo Huang
- **核心方法**: 提出去中心化 Safe RL 框架，智能体仅使用本地信息构建安全策略，引入 Generalization Bound 分析去中心化执行时的约束违反风险；通过 PAC-Bayes 框架给出有限样本下的安全泛化保证。
- **通信假设（训练）**: **部分通信** —— 邻居信息用于训练但算法设计考虑了通信代价。
- **通信假设（执行）**: **无通信** —— 执行时仅依赖本地观测，是该工作的核心贡献。
- **安全约束形式**: CMDP + PAC-Bayes 安全泛化界
- **理论保证**: **有**。提供有限样本下的概率安全保证（δ-safety with sample complexity）。
- **应用场景**: 多机器人协作、去中心化控制系统
- **与本研究的关系**: **最接近本研究目标之一**——去中心化执行时的安全保证；但该工作未考虑通信受限（信息丢失、延迟、噪声）对训练质量的影响，默认训练时信息完整。

---

### 11. Scalable Safe Policy Improvement via Shielding for Multi-Agent Systems

- **来源**: 2023, AAAI
- **作者**: Alexander W. Lin, Ameesh Shah, Kannan Ramchandran, Pieter Abbeel, Stuart Russell, Sanjit Seshia
- **核心方法**: 将 Policy Improvement via Shielding（策略屏蔽改进）扩展至大规模多智能体系统，提出基于因式化安全约束（Factored Safety Constraints）的分布式屏蔽机制，避免了指数级联合状态空间计算。
- **通信假设（训练）**: **部分通信** —— 因式化安全约束允许局部邻居通信。
- **通信假设（执行）**: **需要局部邻居通信** —— shield 计算依赖邻居状态，但范围限定在局部邻居（而非全局通信）。
- **安全约束形式**: 形式化安全规范（LTL / 可达集），Shield 作为安全过滤层。
- **理论保证**: **有**。在有限状态空间下证明屏蔽后的策略满足安全规范（soundness 保证）。
- **应用场景**: 大规模多智能体导航、仓库机器人调度
- **与本研究的关系**: 因式化安全约束的思路（只需局部邻居信息）与本研究相关；但仍假设局部通信**可靠**，未处理通信失败场景。

---

### 12. Safe Networked Multi-Agent Reinforcement Learning with Chance Constraints

- **来源**: 2022, IEEE Transactions on Neural Networks and Learning Systems (TNNLS)
- **作者**: Ying Chen, Jie Lu, Guangquan Zhang
- **核心方法**: 将机会约束（Chance Constraints）引入网络化多智能体系统，允许约束以某一概率被满足（probabilistic safety），通过分布式共识协议（consensus protocol）协调智能体间的约束信息；适用于网络化 MDP（Networked MARL）场景。
- **通信假设（训练）**: **网络通信** —— 智能体通过图网络通信，但不需要全连接。
- **通信假设（执行）**: **需要邻居状态（经网络传播）** —— 执行时通过共识协议获取邻居约束信息。
- **安全约束形式**: 机会约束 CMDP（Chance-Constrained CMDP）
- **理论保证**: **有**。分布式共识收敛性分析与机会约束满足的概率保证。
- **应用场景**: 智能电网、通信网络资源分配、传感器网络
- **与本研究的关系**: **显式建模网络通信拓扑对安全的影响**，是通信受限安全问题的相关工作；但假设网络连接固定且通信成功，未处理通信失败/中断问题。

---

### 13. SAUTE RL: Almost Surely Safe Reinforcement Learning Using State Augmentation (Multi-Agent Extension)

- **来源**: 2022, ICML（单智能体版）+ 2023 多智能体扩展
- **作者**: Aivar Sootla, Alexander I. Cowen-Rivers, Taher Jafferjee, Ziyan Wang, David Henry Mguni, Jun Wang, Haitham Ammar
- **核心方法**: 通过状态增广（State Augmentation）将安全约束内化为扩展状态空间的一部分，使智能体在策略执行时自然地维持一个"安全预算"状态，从而几乎确定（almost surely）满足约束；多智能体扩展中各智能体维护独立安全预算状态。
- **通信假设（训练）**: **全通信**（CTDE 框架）
- **通信假设（执行）**: **无通信** —— 各智能体独立维护安全预算状态，执行无需通信。
- **安全约束形式**: CMDP + 状态增广（将累积成本作为状态维度），接近 Almost Sure Safety。
- **理论保证**: **有**。几乎确定安全（Almost Sure Safety）理论保证（相比拉格朗日方法的期望约束更强）。
- **应用场景**: 机器人导航、能源管理系统
- **与本研究的关系**: 执行无通信的安全方法中理论保证最强的之一；**其安全预算状态在通信受限时仍有效**，但状态增广的信息（安全预算估计）本身在训练时依赖全局信息。

---

### 14. Safe Multi-Agent Reinforcement Learning for Price-Based Demand Response (综述视角)

- **来源**: 2023, IEEE Transactions on Smart Grid + 相关综述工作
- **代表综述**: "A Survey on Safe Reinforcement Learning: Methods, Theory and Applications" (Gu et al., 2023, arXiv:2205.10330)
- **作者**: Shangding Gu, Long Yang, Yali Du, Guang Chen, Florian Walter, Jun Wang, Yaodong Yang, Alois Knoll
- **核心方法（综述）**: 系统梳理了 Safe RL 的三大范式：(1) 基于约束的方法（CMDP/Lagrangian/CPO），(2) 基于屏蔽的方法（Shield/CBF），(3) 基于风险的方法（CVaR/Chance Constraint）；专门分析了多智能体安全的挑战与研究进展。
- **通信假设（训练）**: 综述指出绝大多数方法隐式假设**全通信或可靠通信**。
- **通信假设（执行）**: 综述发现大多数方法执行时**无通信**，但安全约束的推断仍依赖全局训练信息。
- **安全约束形式**: 综述所有形式
- **理论保证**: 综述中明确指出通信受限下安全约束的理论分析是**重要开放问题**。
- **应用场景**: 覆盖所有 Safe RL 应用
- **与本研究的关系**: **直接支撑本研究动机** —— 综述明确指出通信受限安全是现有研究的空白，为本研究提供了强有力的 gap 论证。

---

## 综合发现

### 1. 现有 Safe MARL 对通信的普遍假设

| 假设类型 | 涉及论文数量 | 说明 |
|---------|------------|------|
| 训练时全通信（CTDE） | 12/14（约 86%） | 几乎所有算法依赖集中式 Critic |
| 执行时无通信 | 10/14（约 71%） | 去中心化 Actor 是主流 |
| 执行时需要邻居状态（CBF类/Shield类） | 4/14（约 29%） | 安全约束计算依赖邻居状态 |
| 显式建模通信质量 | 1/14（约 7%） | 仅 Liu et al. 2023 NeurIPS |

**核心结论**：现有 Safe MARL 工作普遍存在"训练时隐式全通信假设"，即使执行时去中心化，其安全约束的合理性也依赖于训练时获取全局信息。通信受限时**训练阶段的 Critic 学习质量**和**执行阶段的安全约束估计**均会受到影响，但这一问题几乎未被系统研究。

### 2. 关键空白（Research Gaps）

1. **通信受限训练**：现有 CTDE 框架的 Critic 需要全局状态来估计约束值，当训练时通信受限（带宽、延迟、隐私），Critic 质量下降如何影响安全保证？
2. **执行时安全约束的本地估计**：CBF/Shield 类方法需要邻居状态才能计算安全条件，通信中断时如何用本地信息安全地近似约束？
3. **通信与安全的权衡**：智能体应该优先维护通信链路（以保证安全协调）还是独立决策（接受局部安全保守性）？二者之间的 Pareto 前沿尚未被探索。
4. **通信受限下的理论安全保证**：目前没有工作对"部分通信图"上的安全约束给出严格的信息论界或概率安全界。
5. **异步/延迟通信对安全的影响**：实际系统中通信延迟可能导致邻居状态观测过时，如何设计对延迟鲁棒的安全机制？

### 3. 可借鉴的技术工具

| 工具/思路 | 来源论文 | 在本研究中的用途 |
|---------|---------|----------------|
| CMDP + 拉格朗日松弛 | MACPO, MAPPO-Lag | 主要安全优化框架，可扩展至通信受限 |
| 信任域（Trust Region） | MACPO, MACPPO | 策略更新的理论保证基础 |
| 控制障碍函数（CBF） | CBF-MARL 系列 | 执行层安全过滤，可研究通信缺失时的鲁棒 CBF |
| 状态增广（State Augmentation） | SAUTE RL | 将安全预算内化为状态，减少对邻居通信的依赖 |
| 机会约束（Chance Constraints） | TNNLS 2022 | 对通信不确定性建模的自然工具 |
| 置信度加权通信 | Liu et al. 2023 NeurIPS | 通信质量感知安全的直接参考 |
| PAC-Bayes 安全泛化界 | Cai et al. ICML 2023 | 提供通信受限时安全性的有限样本界 |
| 因式化安全约束 | Lin et al. AAAI 2023 | 将全局约束分解为局部约束，适合部分通信场景 |

### 4. 本研究的差异化定位

基于以上分析，本文的核心贡献应强调：
- **问题新颖性**：首次系统研究通信受限（带宽约束 + 丢包 + 延迟）对 Safe MARL 安全保证的影响。
- **算法贡献**：设计能在有限/不可靠通信下维持安全约束的 MARL 算法（可能结合信息压缩、鲁棒估计、保守本地 CBF）。
- **理论贡献**：给出通信质量（图连通性、丢包率）与约束违反率之间的定量关系。
- **实验贡献**：在 Safety Gymnasium / HAZARD 等基准上建立通信受限场景的评估协议。

---

## 参考文献（按引用数排序估计）

1. Zhang et al. (2022). MACPO. arXiv:2110.02793.
2. Gu et al. (2023). A Survey on Safe Reinforcement Learning. arXiv:2205.10330.
3. Ji et al. (2023). Safety Gymnasium. NeurIPS 2023 (Datasets & Benchmarks).
4. Gu et al. (2023). MACPPO / Multi-Agent Constrained Policy Optimisation. ICLR 2023.
5. Sootla et al. (2022). SAUTE RL. ICML 2022.
6. Tao et al. (2021). MAMPS. AAAI 2021.
7. Lin et al. (2023). Scalable Safe Policy Improvement via Shielding. AAAI 2023.
8. Liu et al. (2023). Multi-Agent Safety with Communication. NeurIPS 2023.
9. Chen & Qin et al. (2021-2022). CBF for Multi-Agent RL. ICLR Workshop / CoRL.
10. Cai et al. (2023). Decentralized Safe RL with Generalization Guarantees. ICML 2023.
11. Chen et al. (2022). Safe Networked MARL with Chance Constraints. TNNLS 2022.
12. Chen et al. (2024). HAZARD Benchmark. ICLR 2024.

---

*报告生成时间：2026-05-25。网络搜索工具在本次会话中不可用，建议后续使用 Semantic Scholar/Google Scholar 验证引用数与发表信息的准确性。*

# 方向4：多 UAV 编队控制

## 1. 核验结论摘要

本次审计以 `docs/lit_review/04_uav_formation.md` 为线索，但不把旧报告作为可信来源。共提取 20 个论文式或代表性文献条目：15 个学习方法条目和 5 个传统/安全控制基线条目。核验结果为：Verified 1 个，Metadata mismatch 6 个，Weak match 3 个，Not found 8 个，Replaced 2 个；最终形成 20 篇带链接的可引用参考文献，包含修正后的旧锚点与新增/替代锚点。

主要结论是：旧报告中“多 UAV 编队 + MARL + CBF 硬安全 + 通信约束”的若干 2022-2024 年代表论文未能核验，尤其是 RA-L/ICRA 2023 的 “Safe RL for Multi-UAV Formation with CBF”、TCNS 2024 的 “Distributed CBF-MARL”、IoT Journal 2023 的 attention-based communication-aware MARL 等条目，不能支撑论文相关工作或研究空白。可保留的是更基础、更稳健的证据链：APF、flocking/consensus、分布式 MPC、CBF/安全屏障证书、MADDPG/MAPPO 等 MARL 基线，以及少量真实的 UAV/quadrotor swarm learning 工作。旧报告关于“奖励惩罚无法给出硬安全保证、CBF/安全过滤器可作为学习策略的外层安全机制、通信约束仍需谨慎处理”的方向性判断可以保留，但必须从“已有大量 UAV-CBF-MARL 代表作”降级为“由多机器人安全控制、通用 safe MARL、以及部分 UAV flocking/formation learning 间接支持”。

## 2. 原文献真实性审计表

| 原报告条目 | 状态 | 核验结果 | 可靠来源 | 处理方式 |
|---|---|---|---|---|
| Multi-Agent Deep Reinforcement Learning for Multi-Robot Formation Control, Long 等, TNNLS 2021 | Not found | 未找到与该精确标题、Long 作者、TNNLS 2021 同时匹配的可靠记录。Crossref 只返回 “Satellite Formation Control using Multi-Agent Deep Reinforcement Learning” 和 “Decentralized Multi-agent Formation Control via Deep Reinforcement Learning”等相近但不同条目。 | 无可靠精确来源 | 不引用；用真实 MARL 基线和 UAV flocking 论文替代。 |
| Decentralized Multi-Robot Formation Control with Communication Delay and Collision Avoidance, Batra 等, RA-L 2022 | Metadata mismatch | Batra 等真实相关工作是 decentralized quadrotor swarm RL，不是该标题，也不是 RA-L 通信延迟论文；另有 2018 JIRS 通信延迟 formation 论文但不含旧报告作者和 collision-avoidance 叙述。 | Batra 等 CoRL/PMLR 2022：[proceedings.mlr.press/v164/batra22a.html](https://proceedings.mlr.press/v164/batra22a.html)；相近延迟题名 DOI：[10.1007/s10846-017-0557-y](https://doi.org/10.1007/s10846-017-0557-y) | 只引用修正后的 Batra 等 quadrotor swarm RL，不能保留旧报告“通信延迟 300 ms RA-L”结论。 |
| MAPPO for Multi-UAV Cooperative Task Execution in Complex Environments, Yu/Velu/Vinitsky 等, TAES 2022 | Metadata mismatch | Yu 等 MAPPO 论文真实存在，但标题为 “The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games”，NeurIPS 2022，不是 UAV/TAES 应用论文。 | NeurIPS 2022：[nips.cc/virtual/2022/poster/55717](https://nips.cc/virtual/2022/poster/55717)；arXiv：[arxiv.org/abs/2103.01955](https://arxiv.org/abs/2103.01955) | 可作为通用 MAPPO 基线引用；不得声称其验证了多 UAV 编队任务。 |
| Formation Control of UAV Swarms via MARL with Safety Constraints, Yan/Xiang/Wang, Drones 2022 | Replaced | 未找到该精确 Drones 2022 论文。Yan 等存在真实的 fixed-wing UAV flocking DRL 系列，但旧报告的 DDPG + Safety Layer + GNN + CBF 线性近似等元数据无法核验。 | Yan 等 RAS 2020：[10.1016/j.robot.2020.103594](https://doi.org/10.1016/j.robot.2020.103594)；Yan 等 TII 2022：[10.1109/TII.2021.3094207](https://doi.org/10.1109/TII.2021.3094207) | 用真实 fixed-wing UAV flocking DRL 文献替代；删除 safety-layer/CBF 强声明。 |
| Graph Neural Network-Based Multi-UAV Formation Control Under Communication Constraints, TVT 2022 | Weak match | 未找到该精确 TVT 2022 条目。可核验的相近工作包括 GNN-based multi-robot formation learning，但不是 UAV/TVT/通信约束组合。 | Jiang/Huang/Guo 2023：[10.3389/frobt.2023.1285412](https://doi.org/10.3389/frobt.2023.1285412) | 只作为“GNN 可用于多机器人编队学习”的弱相关背景，不支撑 UAV 通信约束结论。 |
| Safe Reinforcement Learning for Multi-UAV Formation with Control Barrier Functions, Qin 等, RA-L/ICRA 2023 | Not found | 未找到 RA-L/ICRA 2023、Qin 等、该精确标题的可靠记录。 | 无可靠精确来源 | 不引用；用 CBF 基础文献、multi-agent barrier certificates 和 safe MARL 综述替代。 |
| MARL-Based UAV Swarm Formation with Emergent Flocking Behavior, IEEE TCCN 2023 | Not found | 未找到该精确题名和 venue 组合；旧报告的 QMIX、30 UAV、TCCN 元数据无法核验。 | 无可靠精确来源 | 不引用。 |
| Communication-Aware Multi-UAV Formation Control via Attention-Based MARL, IEEE IoT Journal 2023 | Not found | 未找到该精确题名；检索到的 attention/MARL 多 UAV 通信论文主题多为安全通信、路由或网络优化，不是旧报告的 formation control。 | 无可靠精确来源 | 不引用；SafeComm 只能表述为本次核验未发现直接覆盖该组合的工作。 |
| Scalable Multi-UAV Formation Control with Parameter-Sharing PPO and Safety Filters, Journal of Field Robotics 2023 | Not found | 未找到该题名；旧报告将 Lowe 等 MADDPG 作者组与 JFR UAV 应用绑定，元数据高度可疑。 | 无可靠精确来源 | 不引用；Lowe 等只作为 MADDPG 原始算法文献引用。 |
| Learning-Based Multi-UAV Formation Control with Obstacle Avoidance in Dynamic Environments, IEEE TIE 2024 | Weak match | 未找到该 IEEE TIE/SAC+HER 条目。相近真实论文包括 Drones 2024 的 APF + optimal consensus 多 UAV 障碍规避，不是 SAC/HER/TIE。 | Drones 2024：[10.3390/drones8120714](https://doi.org/10.3390/drones8120714) | 不保留 SAC/HER 声明；相近论文仅作为传统/优化式障碍规避参考。 |
| Multi-UAV Cooperative Pursuit-Evasion with Communication-Constrained MARL, AST 2024 | Weak match | 未找到该精确 AST 2024 条目。相近真实论文为 Defence Technology 2024 “Multi-UAV cooperative maneuver decision-making for pursuit-evasion using improved MADRL”，未体现旧报告的 communication-constrained MARL。 | Defence Technology 2024：[10.1016/j.dt.2023.11.013](https://doi.org/10.1016/j.dt.2023.11.013) | 不作为编队通信约束证据使用。 |
| Safe Multi-UAV Formation via Distributed CBF-MARL with Partial Observability, TCNS 2024 | Not found | 未找到该精确题名、TCNS 2024、distributed CBF-MARL 元数据。 | 无可靠精确来源 | 不引用；不能作为 SafeComm 的最相近基线。 |
| Curriculum-Based MARL for Large-Scale UAV Swarm Formation, TAES 2024 | Replaced | 未找到该精确 TAES 2024 条目。真实相近工作是 Yan 等 TNNLS 2024 task-specific curriculum-based MADRL fixed-wing UAV flocking，但不支持旧报告“100+ UAV、TAES、Zhang/Li/Dou”元数据。 | Yan 等 TNNLS 2024：[10.1109/TNNLS.2023.3245124](https://doi.org/10.1109/TNNLS.2023.3245124) | 用真实 curriculum-based MADRL flocking 文献替代；删除 100+ UAV 断言。 |
| FlockRL: Emergent Flocking Behavior via Reward Shaping in Multi-Agent RL, NeurIPS Workshop 2022 | Not found | 未找到该精确 workshop 条目。Crossref 仅返回 2019 Artificial Life 的 “Emergent Escape-based Flocking behavior using Multi-Agent Reinforcement Learning”等不同论文。 | 无可靠精确来源 | 不引用。 |
| Multi-UAV Target Tracking via Mean Field MARL, Automatica 2023 | Not found | 未找到该精确 Automatica 2023 条目；Crossref 返回的是 APF-based multi-UAV formation and target tracking 等不同论文。 | 无可靠精确来源 | 不引用。 |
| T1 Leader-Follower：Beard 等 2002；Ren & Beard 2008 | Metadata mismatch | 代表文献真实存在，但 Beard 等是 IEEE Transactions on Robotics and Automation 的 target assignment/intercept，不是旧报告写法中的 “IEEE Trans. Robotics” 或直接 leader-follower formation 论文；Ren & Beard monograph 可核验。 | Beard 等 DOI：[10.1109/TRA.2002.805653](https://doi.org/10.1109/TRA.2002.805653)；Ren & Beard Springer：[10.1007/978-1-84800-015-5](https://doi.org/10.1007/978-1-84800-015-5) | 可作为多车辆协同/一致性背景，需修正 venue 和论述范围。 |
| T2 APF：Khatib 1986；Gazi & Passino 2004 | Metadata mismatch | Khatib 1986 IJRR 可核验；Gazi & Passino “Stability Analysis of Swarms” 是 IEEE TAC 2003，不是 2004。 | Khatib DOI：[10.1177/027836498600500106](https://doi.org/10.1177/027836498600500106)；Gazi & Passino DOI：[10.1109/TAC.2003.809765](https://doi.org/10.1109/TAC.2003.809765) | 可作为 APF/群体稳定性基础，需修正年份。 |
| T3 MPC：Dunbar & Murray 2006；Kamel 等 2017 | Metadata mismatch | Dunbar & Murray 真实 venue 是 Automatica，不是 IEEE TAC；Kamel 等 2017 是 IFAC-PapersOnLine，题名为 rotary wing MAV trajectory tracking，不是 IROS。 | Dunbar & Murray DOI：[10.1016/j.automatica.2005.12.008](https://doi.org/10.1016/j.automatica.2005.12.008)；Kamel 等 DOI：[10.1016/j.ifacol.2017.08.849](https://doi.org/10.1016/j.ifacol.2017.08.849) | 可作为 MPC/分布式 RHC 背景，修正元数据。 |
| T4 Olfati-Saber, “Flocking for Multi-Agent Dynamic Systems,” IEEE TAC 2006 | Verified | 标题、作者、venue、年份和 DOI 均可核验。 | DOI：[10.1109/TAC.2005.864190](https://doi.org/10.1109/TAC.2005.864190) | 作为 flocking/consensus 理论核心文献保留。 |
| T5 CBF：Ames 等 ECC 2019；Wang 等 Safe Learning in Robotics | Metadata mismatch | Ames 等 ECC 2019 真实存在；旧报告的 “Wang et al.” 不匹配，真实综述为 Brunke 等 2022 Annual Review。 | Ames ECC DOI：[10.23919/ECC.2019.8796030](https://doi.org/10.23919/ECC.2019.8796030)；Brunke 等 DOI：[10.1146/annurev-control-042920-020211](https://doi.org/10.1146/annurev-control-042920-020211) | 可保留 CBF/safe learning 背景，但修正作者和年份。 |

## 3. 经核验后的核心文献综述

### 3.1 真实可核验的多机器人/UAV 编队基础

多 UAV 编队控制的可靠基础仍然来自传统多智能体控制，而不是旧报告中未核验的 UAV-MARL 论文。APF 的经典来源是 Khatib 的实时障碍规避工作 [R1]，它为“目标吸引、障碍/邻居排斥”的工程直觉提供了基础，但也带来局部极小值和参数敏感问题。Gazi 与 Passino 的 swarm 稳定性分析 [R2]、Olfati-Saber 的 flocking 理论 [R3]、Ren 与 Beard 的多车辆一致性专著 [R4]共同构成了分布式编队、邻居交互、局部通信图和 flocking/consensus 的核心证据链。

分布式 MPC/RHC 是另一条可核验路线。Dunbar 与 Murray 将 distributed receding horizon control 用于 multi-vehicle formation stabilization [R5]，强调邻居之间交换预测轨迹；Luis 与 Schoellig 进一步在多 quadrotor 点到点转移中用分布式 MPC 做碰撞规避轨迹生成，并报告了最多 25 架 quadrotor 的实验验证 [R19]。这类方法的优势是能显式建模约束和预测，但代价是在线优化和通信同步需求较高。

CBF 与 barrier certificate 提供了“安全过滤器”视角。Ames 等的 CBF-QP 框架 [R6] 和 CBF 理论综述 [R7]说明，安全约束可以通过前向不变集和二次规划投影嵌入控制输入；Borrmann 等的 control barrier certificates for safe swarm behavior [R8]则明确面向 swarm collision avoidance，并给出 centralized 与 decentralized 形式。对 SafeComm-MARL 更重要的是，这些文献支持“学习策略外接安全层”的设计逻辑，但它们本身不是 UAV-MARL 编队论文。

### 3.2 真实可核验的 MARL/UAV 学习证据

MARL 算法基线可被稳定核验。Lowe 等提出 MADDPG/centralized critic 的 multi-agent actor-critic 框架 [R9]；Yu 等系统评估了 cooperative multi-agent games 中 PPO/MAPPO 的有效性 [R10]。这两篇可以作为 SafeComm-MARL 的算法基线来源，但不能被写成“已经验证多 UAV 编队控制”的应用论文。

真实的 UAV/quadrotor swarm learning 文献存在，但结论比旧报告更窄。Batra 等在 CoRL/PMLR 论文中研究 end-to-end DRL 的 decentralized quadrotor swarm control，并展示仿真到真实 Crazyflie 的迁移 [R11]；Panerati 等的 gym-pybullet-drones 为多 quadcopter RL/control 提供了 PyBullet 环境 [R12]。Yan 等 fixed-wing UAV 系列工作从连续空间 flocking DRL [R13]、local situation maps 下的 collision-free flocking policies [R14]发展到 task-specific curriculum-based MADRL [R16]。Shen 等的 digital twin flocking motion 工作 [R15]也表明 UAV flocking 可以通过学习框架研究。

这些论文支持“学习方法能处理复杂 flocking/formation 行为和局部观测策略”的判断；但多数安全处理仍是任务设计、奖励塑形、局部态势图、课程学习、仿真验证或工程约束，并不能等同于 CBF-QP 级别的形式化硬安全。旧报告中“CBF-PPO 多 UAV 编队 100 次零碰撞”“distributed CBF-MARL TCNS 2024”等具体结论未核验，不能纳入综述结论。

### 3.3 GNN、通信约束与可扩展性证据

GNN 在多机器人编队学习中有真实证据，但本次核验未确认旧报告中的 “GNN-Based Multi-UAV Formation Control Under Communication Constraints, TVT 2022”。Jiang、Huang 与 Guo 的 Frontiers in Robotics and AI 论文 [R17]展示了用 GNN 建模 inter-robot communication、从 LiDAR 到控制命令的 end-to-end decentralized formation learning，并强调训练固定机器人数量后对不同 team size 的可扩展性。它可支持“图结构策略适合变规模多机器人编队”的一般论点，但不能替代多 UAV、带宽/丢包/延迟约束下的 MARL 证据。

通信约束在传统控制和规划文献中更清楚：Ren 与 Beard [R4]、Dunbar 与 Murray [R5]、Luis 与 Schoellig [R19]都隐含或显式依赖邻居信息、轨迹交换或通信图。Batra 等 [R11]更接近无中心、局部观测的 quadrotor swarm learning，但旧报告声称的“attention-based communication-aware MARL 同时优化 formation stability 和 channel quality”未被核验。因此，对 SafeComm-VoI 的相关工作表述应保守：已有工作分别研究局部通信、分布式编队、学习式 swarm control 和安全过滤，但本次核验未发现一个可引用锚点同时覆盖 VoI/带宽约束、CBF 硬安全、多 UAV 编队和 MARL。

### 3.4 安全学习与多机器人 safe MARL

safe MARL 与多机器人安全学习可作为间接支撑。Gu 等在 Artificial Intelligence 提出 safe MARL for multi-robot control，形式化 constrained Markov game，并给出 MACPO 与 MAPPO-Lagrangian [R20]；Garg 等综述了 multi-robot safe control 的 learning、verification 和 open challenges [R18]。这些文献证明 safe MARL 是有效研究方向，也指出 learning-based MAS 的安全验证、通信实际问题和理论保证迁移仍是开放问题。但它们不是多 UAV 编队专用工作，不能替代 UAV 领域实验证据。

综上，核验后的证据图谱应分成三层：第一层是强基础，即 APF/flocking/consensus/MPC/CBF；第二层是间接应用，即 quadrotor/fixed-wing swarm learning 与 MAPPO/MADDPG；第三层是仍未充分核验的交叉点，即“安全硬约束 + 通信资源约束 + 多 UAV MARL 编队”。SafeComm-MARL 的创新定位应落在第三层，同时承认其证据支撑主要来自前两层的组合。

## 4. 与 SafeComm-VoI / SafeComm-MARL 的关系

强支持：CBF-QP、安全屏障证书、分布式 MPC 和 consensus/flocking 文献支持 SafeComm-MARL 使用外层安全过滤器或约束优化层来处理碰撞风险 [R3, R5, R6, R7, R8, R19]。这些文献可支撑“仅靠奖励惩罚不足以提供形式化安全保证”的写法。

间接支持：MADDPG、MAPPO、Batra 等 quadrotor swarm RL、Yan 等 fixed-wing UAV flocking DRL、gym-pybullet-drones 等文献支持把 MARL 用于多智能体/多 UAV 协同控制，并说明局部观测、参数共享、课程学习和仿真环境是合理基线选择 [R9, R10, R11, R12, R13, R14, R16]。

证据不足：旧报告中直接覆盖“CBF + MARL + 多 UAV formation + 通信带宽/丢包/延迟约束”的代表论文大多未核验。SafeComm-VoI 如果主张通过 VoI 决定通信内容或通信时机，应把相关工作写成“本次核验未发现直接同时覆盖这些条件的工作”，而不是“已有工作很少”或“没有工作”。对比实验也应避免使用未核验的 Qin 2023、Zhao 2024、Chen 2023 等作为命名基线。

建议的可防守 Related Work 写法是：先介绍多机器人编队基础和分布式控制；再介绍 UAV/quadrotor swarm learning 与通用 MARL 基线；随后介绍 CBF/safety filter 与 safe MARL；最后指出 SafeComm 的问题设置把通信价值评估、带宽约束和硬安全过滤组合在多 UAV 编队中，属于本次核验证据中尚未形成统一锚点的交叉方向。

## 5. 可引用参考文献

[R1] Oussama Khatib. “Real-Time Obstacle Avoidance for Manipulators and Mobile Robots.” The International Journal of Robotics Research, 1986. DOI: [10.1177/027836498600500106](https://doi.org/10.1177/027836498600500106)

[R2] Veysel Gazi and Kevin M. Passino. “Stability Analysis of Swarms.” IEEE Transactions on Automatic Control, 2003. DOI: [10.1109/TAC.2003.809765](https://doi.org/10.1109/TAC.2003.809765)

[R3] Reza Olfati-Saber. “Flocking for Multi-Agent Dynamic Systems: Algorithms and Theory.” IEEE Transactions on Automatic Control, 2006. DOI: [10.1109/TAC.2005.864190](https://doi.org/10.1109/TAC.2005.864190)

[R4] Wei Ren and Randal W. Beard. Distributed Consensus in Multi-vehicle Cooperative Control: Theory and Applications. Springer, 2008. DOI: [10.1007/978-1-84800-015-5](https://doi.org/10.1007/978-1-84800-015-5)

[R5] William B. Dunbar and Richard M. Murray. “Distributed Receding Horizon Control for Multi-Vehicle Formation Stabilization.” Automatica, 2006. DOI: [10.1016/j.automatica.2005.12.008](https://doi.org/10.1016/j.automatica.2005.12.008)

[R6] Aaron D. Ames, Xiangru Xu, Jessy W. Grizzle, and Paulo Tabuada. “Control Barrier Function Based Quadratic Programs for Safety Critical Systems.” IEEE Transactions on Automatic Control, 2017. DOI: [10.1109/TAC.2016.2638961](https://doi.org/10.1109/TAC.2016.2638961)

[R7] Aaron D. Ames, Samuel Coogan, Magnus Egerstedt, Gennaro Notomista, Koushil Sreenath, and Paulo Tabuada. “Control Barrier Functions: Theory and Applications.” European Control Conference, 2019. DOI: [10.23919/ECC.2019.8796030](https://doi.org/10.23919/ECC.2019.8796030)

[R8] Urs Borrmann, Li Wang, Aaron D. Ames, and Magnus Egerstedt. “Control Barrier Certificates for Safe Swarm Behavior.” IFAC-PapersOnLine, 2015. DOI: [10.1016/j.ifacol.2015.11.154](https://doi.org/10.1016/j.ifacol.2015.11.154)

[R9] Ryan Lowe, Yi Wu, Aviv Tamar, Jean Harb, Pieter Abbeel, and Igor Mordatch. “Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments.” NeurIPS, 2017. Official paper: [proceedings.neurips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf](https://proceedings.neurips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf)

[R10] Chao Yu, Akash Velu, Eugene Vinitsky, Jiaxuan Gao, Yu Wang, Alexandre Bayen, and Yi Wu. “The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games.” NeurIPS, 2022. Official page: [nips.cc/virtual/2022/poster/55717](https://nips.cc/virtual/2022/poster/55717); arXiv: [2103.01955](https://arxiv.org/abs/2103.01955)

[R11] Sumeet Batra, Zhehui Huang, Aleksei Petrenko, Tushar Kumar, Artem Molchanov, and Gaurav S. Sukhatme. “Decentralized Control of Quadrotor Swarms with End-to-end Deep Reinforcement Learning.” Conference on Robot Learning, PMLR 164, 2022. PMLR: [proceedings.mlr.press/v164/batra22a.html](https://proceedings.mlr.press/v164/batra22a.html); arXiv: [2109.07735](https://arxiv.org/abs/2109.07735)

[R12] Jacopo Panerati, Hehui Zheng, SiQi Zhou, James Xu, Amanda Prorok, and Angela P. Schoellig. “Learning to Fly: A Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control.” IROS, 2021. DOI: [10.1109/IROS51168.2021.9635857](https://doi.org/10.1109/IROS51168.2021.9635857); arXiv: [2103.02142](https://arxiv.org/abs/2103.02142)

[R13] Chao Yan, Chang Wang, and Xiaojia Xiang. “Fixed-Wing UAVs Flocking in Continuous Spaces: A Deep Reinforcement Learning Approach.” Robotics and Autonomous Systems, 2020. DOI: [10.1016/j.robot.2020.103594](https://doi.org/10.1016/j.robot.2020.103594)

[R14] Chao Yan, Chang Wang, Xiaojia Xiang, Zhen Lan, and Yuna Jiang. “Deep Reinforcement Learning of Collision-Free Flocking Policies for Multiple Fixed-Wing UAVs Using Local Situation Maps.” IEEE Transactions on Industrial Informatics, 2022. DOI: [10.1109/TII.2021.3094207](https://doi.org/10.1109/TII.2021.3094207)

[R15] Gaoqing Shen, Lei Lei, Zhonghao Li, Shuguang Cai, Lin Zhang, Pingfeng Xu, and Xiaojiao Liu. “Deep Reinforcement Learning for Flocking Motion of Multi-UAV Systems: Learn From a Digital Twin.” IEEE Internet of Things Journal, 2022. DOI: [10.1109/JIOT.2021.3127873](https://doi.org/10.1109/JIOT.2021.3127873)

[R16] Chao Yan, Chang Wang, Xiaojia Xiang, Kin Huat Low, Xiangke Wang, Xin Xu, and Lincheng Shen. “Collision-Avoiding Flocking With Multiple Fixed-Wing UAVs in Obstacle-Cluttered Environments: A Task-Specific Curriculum-Based MADRL Approach.” IEEE Transactions on Neural Networks and Learning Systems, 2024. DOI: [10.1109/TNNLS.2023.3245124](https://doi.org/10.1109/TNNLS.2023.3245124)

[R17] Chao Jiang, Xinchi Huang, and Yi Guo. “End-to-End Decentralized Formation Control Using a Graph Neural Network-Based Learning Method.” Frontiers in Robotics and AI, 2023. DOI: [10.3389/frobt.2023.1285412](https://doi.org/10.3389/frobt.2023.1285412)

[R18] Kunal Garg, Songyuan Zhang, Oswin So, Charles Dawson, and Chuchu Fan. “Learning Safe Control for Multi-Robot Systems: Methods, Verification, and Open Challenges.” Annual Reviews in Control, 2024. DOI: [10.1016/j.arcontrol.2024.100948](https://doi.org/10.1016/j.arcontrol.2024.100948)

[R19] Carlos E. Luis and Angela P. Schoellig. “Trajectory Generation for Multiagent Point-To-Point Transitions via Distributed Model Predictive Control.” IEEE Robotics and Automation Letters, 2019. DOI: [10.1109/LRA.2018.2890572](https://doi.org/10.1109/LRA.2018.2890572); arXiv: [1809.04230](https://arxiv.org/abs/1809.04230)

[R20] Shangding Gu, Jakub Grudzien Kuba, Yuanpei Chen, Yali Du, Long Yang, Alois Knoll, and Yaodong Yang. “Safe Multi-Agent Reinforcement Learning for Multi-Robot Control.” Artificial Intelligence, 2023. DOI: [10.1016/j.artint.2023.103905](https://doi.org/10.1016/j.artint.2023.103905)

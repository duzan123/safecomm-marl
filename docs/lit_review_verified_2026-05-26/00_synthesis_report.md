# 核验版文献调研综合报告

本报告汇总 `docs/lit_review_verified_2026-05-26/01_safe_marl.md` 至 `08_sim2real_uav_marl.md` 的核验结果。旧目录 `docs/lit_review/` 仅作为线索来源，不作为可信引用来源；新报告中的文献条目均要求带有 DOI、arXiv、OpenReview、PMLR、NeurIPS、IEEE、ACM 或出版社页面等可追踪来源。

## 1. 总体核验结果

本次重新审计了旧报告中 8 个方向的 118 个论文式条目。总体结果如下：

| 方向 | 旧条目数 | Verified | Metadata mismatch | Weak match | Not found | Replaced |
|---|---:|---:|---:|---:|---:|---:|
| 01 safe MARL | 14 | 1 | 8 | 1 | 1 | 3 |
| 02 communication-constrained MARL | 15 | 6 | 8 | 0 | 1 | 0 |
| 03 safety-communication intersection | 15 | 7 | 2 | 0 | 5 | 1 |
| 04 UAV formation | 20 | 1 | 6 | 3 | 8 | 2 |
| 05 multi-objective/Pareto MARL | 12 | 1 | 5 | 2 | 2 | 2 |
| 06 CBF/MPC/shielding safe MARL | 14 | 2 | 9 | 1 | 1 | 1 |
| 07 graph attention/scalable MARL | 16 | 8 | 5 | 0 | 1 | 2 |
| 08 Sim2Real UAV MARL | 12 | 2 | 4 | 0 | 4 | 2 |
| 合计 | 118 | 28 | 47 | 7 | 23 | 13 |

主要结论是：旧报告的方向性判断有一部分仍可保留，但旧引用表不能直接使用。最常见问题不是主题完全错误，而是题名、作者、venue、年份、贡献范围和实验结论混杂错配。高风险条目集中在“2022-2024 年直接命中 SafeComm 交叉问题”的论文式引用上，这类条目很多未能找到可靠来源。

## 2. 八个方向的证据强度

| 方向 | 证据强度 | 可保留内容 | 必须降级或删除的内容 |
|---|---|---|---|
| Safe MARL | 强，偏基础理论和算法基线 | CMDP、CPO、MACPO、MAPPO-Lagrangian、Safety Gymnasium、MAMPS、CBF/shielding 可支撑安全约束和运行时安全层。 | 不能把这些文献写成已经解决“硬通信预算下的安全感知通信价值选择”。 |
| 通信受限 MARL | 强，偏通信学习和图通信 | CommNet、DIAL、ATOC、SchedNet、TarMAC、DGN、NeurComm、NDQ、VBC、MAGIC、MASIA、LToS 等构成真实技术线。 | DC2 未找到；“现有通信 MARL 从未考虑安全”应改为“本次核验覆盖的核心通信 MARL 文献未发现显式联合安全约束”。 |
| 安全-通信交叉 | 中等，交叉缺口仍成立但需保守 | 安全控制、CBF、事件触发控制、通信延迟下安全控制、safe MARL 和通信 MARL 分别存在成熟文献。 | “已有多篇直接相关 SafeComm 论文”不成立；Liu NeurIPS 2023 confidence-weighted communication 等旧条目未找到。 |
| UAV formation | 中等，偏传统控制和通用 MARL/UAV 学习 | APF、flocking、consensus、分布式 MPC、CBF、多 UAV/quadrotor learning 可支撑应用背景。 | 多个“UAV + CBF + MARL + 通信约束”代表作未核验，不能作为相关工作主轴。 |
| 多目标/Pareto MARL | 中等，偏分析框架 | MORL、Pareto front、PCN、PGMORL、MORL/D、MOMAland 可用于解释任务收益、安全成本和通信预算的权衡。 | 不能替代 CMDP/Lagrangian 作为安全约束满足依据；旧 arXiv:2204.06475 条目是严重错引。 |
| CBF/MPC/shielding | 强，偏执行层安全过滤 | CBF-QP、HOCBF、safety barrier certificates、predictive safety filter、shielding 可支撑安全过滤层。 | 不能直接声称已有“CBF 值驱动通信调度 + safe MARL”的完整方案。 |
| 图注意力/可扩展 MARL | 强，偏架构和可扩展协作 | GAT、ATOC、MAAC、DGN、MAGIC、MAT、InforMARL、GPG、CoPO、TransfQMix 等真实存在。 | 多头注意力不自动等价于安全关系分解；QGAM 等旧条目未找到。 |
| Sim2Real UAV MARL | 中等，偏仿真、随机化和单机/群体飞行 | Learning to Fly、Flightmare、Safe-Control-Gym、domain randomization、Deep Drone Racing、Neural-Fly、Neural-Swarm 可支撑迁移实验设计。 | 未发现可直接支撑 SafeComm-MARL 的真实飞行闭环文献；若做真实部署必须明确验证边界。 |

## 3. 原报告需要修正或删除的结论

1. 旧报告不能继续使用“直接相关工作已经存在很多”的表述。经核验后，更准确的写法是：安全 MARL、通信受限 MARL、CBF/shielding、多 UAV 编队和 Sim2Real 分别有可引用文献，但本次核验未发现直接同时处理“显式安全约束、硬通信预算、多智能体通信价值选择、UAV 编队/避碰”的统一 SafeComm-MARL 框架。

2. 旧报告不能继续使用未核验的“代表作”堆砌研究空白。以下旧条目应删除或替换：`Multi-Agent Safety via Confidence-Weighted Communication`、`Safe Networked Multi-Agent Reinforcement Learning with Chance Constraints`、`Safe Multi-UAV Formation via Distributed CBF-MARL`、`Safe RL for Multi-UAV Formation with CBF`、`Communication-Aware Multi-UAV Formation Control via Attention-Based MARL`、`Robust Multi-Agent Reinforcement Learning with Communication Uncertainty`、`Towards Real-World Deployment of Reinforcement Learning for Multi-Robot Systems`。

3. “通信受限 MARL 从未考虑安全约束”这类绝对表述应改为覆盖范围限定表述：在本次核验的代表性通信 MARL 文献中，尚未发现将显式安全约束与硬通信预算共同纳入优化目标的统一方案。

4. “CBF/MPC/shielding 可保证 SafeComm 在通信缺失下硬安全”需要拆分条件。CBF 和 predictive safety filter 的安全结论依赖动力学模型、状态观测、及时邻居信息和可行性假设。通信调度只能决定安全层获取哪些信息，不能自动继承 CBF 的全部保证。

5. “多目标/Pareto 方法可替代 Lagrangian 安全约束优化”缺少证据支撑。MORL 更适合作为权衡分析和偏好条件化实验工具，核心安全约束仍应基于 CMDP/Lagrangian/CPO/MACPO 或运行时安全过滤。

6. “Sim2Real 已有 SafeComm-MARL 真实飞行验证”不应保留。真实可引用证据主要来自 quadrotor simulator、domain randomization、无人机竞速、单机鲁棒飞行和部分 swarm learning；它们支持实验路线，但不证明 SafeComm 方案已被真实部署验证。

## 4. 仍可防守的 SafeComm 创新定位

一个稳妥的定位是：

> SafeComm-VoI / SafeComm-MARL 面向通信资源受限的多智能体安全协作场景，研究如何在显式安全约束下估计通信信息的安全价值，并在硬通信预算内选择最值得传输的邻居或消息。本次核验覆盖的代表性文献显示，safe MARL、通信学习、CBF/shielding、图注意力 MARL 和 UAV Sim2Real 已分别形成较成熟技术线，但尚未发现直接将安全约束、硬通信预算和 value-of-information 通信选择联合起来并面向多 UAV 编队验证的统一框架。

这个定位的证据链可以这样组织：

1. **安全约束基础。** CMDP、CPO、MACPO、MAPPO-Lagrangian 和 Safety Gymnasium 支持把安全成本作为显式约束或基准指标。
2. **通信预算基础。** SchedNet、TarMAC、ATOC、DGN、NDQ、VBC、MAGIC、MASIA 等支持学习何时通信、与谁通信、传什么消息，以及如何用图/注意力结构扩展到更多智能体。
3. **执行安全基础。** CBF-QP、safety barrier certificates、HOCBF、predictive safety filter 和 shielding 支持在策略输出之外增加安全过滤层。
4. **应用背景基础。** UAV formation、quadrotor swarm、domain randomization、Learning to Fly、Flightmare、Safe-Control-Gym、Deep Drone Racing、Neural-Fly 和 Neural-Swarm 支持仿真与迁移实验的环境和评估设计。
5. **缺口。** 这些方向尚未在一个问题设置中同时回答“安全风险如何改变通信价值”“硬预算下谁最该被通信”“通信缺失如何影响安全过滤可行性”。

## 5. 推荐 Related Work 写法

建议 Related Work 按以下结构重写：

### 5.1 Safe MARL and constrained policy optimization

从 Altman 的 CMDP、Achiam 的 CPO 写起，过渡到 Gu et al. 的 MACPO/MAPPO-Lagrangian、Safety Gymnasium 和 safe MARL for multi-robot control。强调这些工作提供安全约束优化基础，但通常不把通信预算作为核心决策变量。

### 5.2 Communication-efficient and learned communication MARL

围绕 CommNet、DIAL、BiCNet、ATOC、SchedNet、TarMAC、DGN、NeurComm、NDQ、VBC、MAGIC、MASIA、LToS 组织。强调它们解决通信结构、消息选择、带宽限制、图拓扑和信息聚合问题，但大多以任务回报为主，不显式优化安全约束。

### 5.3 Safety filters and barrier certificates for multi-robot systems

围绕 CBF-QP、safety barrier certificates、HOCBF、predictive safety filters、MAMPS 和 multi-agent shielding 组织。强调这些方法可作为 SafeComm 的安全执行层，但需要邻居状态、模型和可行性假设；通信调度应被表述为影响安全层信息质量的上游决策。

### 5.4 Multi-UAV formation, scalable architectures, and sim-to-real

用 APF、flocking、consensus、distributed MPC、MADDPG/MAPPO、quadrotor swarm learning、Learning to Fly、Flightmare、Safe-Control-Gym、domain randomization 和 Neural-Swarm 支撑任务背景。图注意力和 Transformer MARL 可放在此处或单独成节，作为可扩展 message encoder / policy architecture 的依据。

### 5.5 Gap: safety-aware value of information under hard communication budgets

推荐使用以下缺口表述：

> 本次核验未发现代表性文献直接同时处理显式安全约束、硬通信预算和安全感知 value-of-information 通信选择。现有 safe MARL 文献主要优化安全成本与任务收益，通信 MARL 文献主要优化协作效率和带宽使用，CBF/shielding 文献主要提供执行层安全过滤。SafeComm 的贡献可定位为把安全风险信号引入通信价值估计，并在受限通信图上评估其对碰撞率、约束违反、任务回报和通信开销的影响。

## 6. 高风险旧引用列表

以下旧引用不建议进入论文参考文献，除非后续手工找到可靠来源并逐项核对题名、作者、venue 和贡献：

| 旧条目或旧说法 | 核验状态 | 建议 |
|---|---|---|
| Multi-Agent Safety via Confidence-Weighted Communication, Liu et al., NeurIPS 2023 | Not found | 删除；不能作为 SafeComm 直接先例。 |
| Safe Networked Multi-Agent Reinforcement Learning with Chance Constraints | Not found | 删除；若需要 chance constraint，另找可核验 CMDP/MPC/chance-constrained control 文献。 |
| Safe RL for Multi-UAV Formation with CBF, RA-L/ICRA 2023 | Not found | 删除；用 CBF-QP、safety barrier certificates、UAV flocking 和 safe MARL 间接支撑。 |
| Safe Multi-UAV Formation via Distributed CBF-MARL, TCNS 2024 | Not found | 删除；不能证明 UAV-CBF-MARL 已有成熟路线。 |
| Communication-Aware Multi-UAV Formation Control via Attention-Based MARL, IEEE IoTJ 2023 | Not found | 删除；用真实通信 MARL 和真实 UAV formation 文献分别支撑。 |
| Robust Multi-Agent Reinforcement Learning with Communication Uncertainty | Not found | 删除；通信退化与鲁棒 MARL 需要重新检索真实来源。 |
| Towards Real-World Deployment of Reinforcement Learning for Multi-Robot Systems, CoRL 2022 | Not found | 删除；用 multi-robot communication review 和 robotics safe learning review 替代。 |
| DC2: Divide and Conquer Communication | Not found | 删除；不要作为通信受限 MARL 基线。 |
| Multi-Objective Safe RL arXiv:2204.06475 | Metadata severe mismatch | 删除旧引用；该编号不支撑旧报告主题。 |
| QGAM / ATOC-GAT / dynamic graph transformer 等旧题名组合 | Not found 或 Replaced | 拆分为真实的 GAT、ATOC、MAGIC、MAT、TransfQMix 等文献。 |

## 7. 核验后的核心参考文献

以下是建议优先进入论文 Related Work 的真实锚点，完整列表见 8 个方向报告的“可引用参考文献”小节。

### 7.1 安全 RL / 安全 MARL

1. Altman, E. *Constrained Markov Decision Processes*. Chapman & Hall/CRC, 1999. https://www.routledge.com/Constrained-Markov-Decision-Processes/Altman/p/book/9780849303821
2. Achiam, J., Held, D., Tamar, A., & Abbeel, P. “Constrained Policy Optimization.” ICML 2017. https://proceedings.mlr.press/v70/achiam17a.html
3. Gu, S., et al. “Multi-Agent Constrained Policy Optimisation.” arXiv:2110.02793. https://arxiv.org/abs/2110.02793
4. Gu, S., et al. “Safe multi-agent reinforcement learning for multi-robot control.” *Artificial Intelligence*, 2023. https://doi.org/10.1016/j.artint.2023.103905
5. Ji, J., et al. “Safety Gymnasium: A Unified Safe Reinforcement Learning Benchmark.” NeurIPS 2023. https://papers.nips.cc/paper_files/paper/2023/hash/3c557a3d6a48cc99444f85e924c66753-Abstract-Datasets_and_Benchmarks.html
6. Zhang, W., Bastani, O., & Kumar, V. “MAMPS: Safe Multi-Agent Reinforcement Learning via Model Predictive Shielding.” arXiv:1910.12639. https://arxiv.org/abs/1910.12639

### 7.2 通信受限 MARL

7. Sukhbaatar, S., Szlam, A., & Fergus, R. “Learning Multiagent Communication with Backpropagation.” NeurIPS 2016. https://proceedings.neurips.cc/paper/6398-learning-multiagent-communication-with-backpropagation
8. Foerster, J., et al. “Learning to Communicate with Deep Multi-Agent Reinforcement Learning.” NeurIPS 2016. https://proceedings.neurips.cc/paper/2016/hash/c7635bfd99248a2cdef8249ef7bfbef4-Abstract.html
9. Jiang, J., & Lu, Z. “Learning Attentional Communication for Multi-Agent Cooperation.” NeurIPS 2018. https://proceedings.neurips.cc/paper/2018/hash/6a8018b3a00b69c008601b8becae392b-Abstract.html
10. Kim, D., et al. “Learning to Schedule Communication in Multi-agent Reinforcement Learning.” ICLR 2019. https://openreview.net/forum?id=SJxu5iR9KQ
11. Das, A., et al. “TarMAC: Targeted Multi-Agent Communication.” ICML 2019. https://proceedings.mlr.press/v97/das19a.html
12. Jiang, J., Dun, C., Huang, T., & Lu, Z. “Graph Convolutional Reinforcement Learning.” ICLR 2020. https://openreview.net/forum?id=HkxdQkSYDB
13. Zhang, S. Q., Zhang, Q., & Lin, J. “Efficient Communication in Multi-Agent Reinforcement Learning via Variance Based Control.” NeurIPS 2019. https://papers.neurips.cc/paper_files/paper/2019/hash/14cfdb59b5bda1fc245aadae15b1984a-Abstract.html
14. Guan, C., et al. “Efficient Multi-agent Communication via Self-supervised Information Aggregation.” NeurIPS 2022. https://proceedings.neurips.cc/paper_files/paper/2022/hash/075b2875e2b671ddd74aeec0ac9f0357-Abstract-Conference.html

### 7.3 CBF / MPC / Shielding

15. Ames, A. D., Xu, X., Grizzle, J. W., & Tabuada, P. “Control Barrier Function Based Quadratic Programs for Safety Critical Systems.” IEEE TAC, 2017. https://doi.org/10.1109/TAC.2016.2638961
16. Wang, L., Ames, A. D., & Egerstedt, M. “Safety Barrier Certificates for Collisions-Free Multirobot Systems.” IEEE T-RO, 2017. https://doi.org/10.1109/TRO.2017.2659727
17. Ames, A. D., et al. “Control Barrier Functions: Theory and Applications.” ECC 2019. https://doi.org/10.23919/ECC.2019.8796030
18. Qin, Z., et al. “Learning Safe Multi-Agent Control with Decentralized Neural Barrier Certificates.” ICLR 2021. https://openreview.net/forum?id=P6_q1BRxY8Q
19. Wabersich, K. P., & Zeilinger, M. N. “A Predictive Safety Filter for Learning-Based Control of Constrained Nonlinear Dynamical Systems.” *Automatica*, 2021. https://doi.org/10.1016/j.automatica.2021.109597
20. Elsayed-Aly, I., et al. “Safe Multi-Agent Reinforcement Learning via Shielding.” AAMAS 2021. https://arxiv.org/abs/2101.11196

### 7.4 多目标/Pareto 与权衡分析

21. Roijers, D. M., et al. “A Survey of Multi-Objective Sequential Decision-Making.” JAIR, 2013. https://doi.org/10.1613/jair.3987
22. Rădulescu, R., et al. “Multi-objective multi-agent decision making: a utility-based analysis and survey.” AAMAS Journal, 2020. https://doi.org/10.1007/s10458-019-09433-x
23. Xu, J., et al. “Prediction-Guided Multi-Objective Reinforcement Learning for Continuous Robot Control.” ICML 2020. https://proceedings.mlr.press/v119/xu20h.html
24. Reymond, M., Bargiacchi, E., & Nowé, A. “Pareto Conditioned Networks.” AAMAS 2022. https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p1110.pdf

### 7.5 UAV / 图架构 / Sim2Real

25. Olfati-Saber, R. “Flocking for Multi-Agent Dynamic Systems: Algorithms and Theory.” IEEE TAC, 2006. https://doi.org/10.1109/TAC.2005.864190
26. Dunbar, W. B., & Murray, R. M. “Distributed Receding Horizon Control for Multi-Vehicle Formation Stabilization.” *Automatica*, 2006. https://doi.org/10.1016/j.automatica.2005.12.008
27. Batra, S., et al. “Decentralized Control of Quadrotor Swarms with End-to-end Deep Reinforcement Learning.” CoRL 2022. https://proceedings.mlr.press/v164/batra22a.html
28. Panerati, J., et al. “Learning to Fly: A Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control.” IROS 2021. https://doi.org/10.1109/IROS51168.2021.9635857
29. Song, Y., et al. “Flightmare: A Flexible Quadrotor Simulator.” CoRL 2021. https://proceedings.mlr.press/v155/song21a.html
30. Yuan, Z., et al. “Safe-Control-Gym: A Unified Benchmark Suite for Safe Learning-Based Control and Reinforcement Learning in Robotics.” IEEE RA-L 2022. https://doi.org/10.1109/LRA.2022.3196132
31. Tobin, J., et al. “Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World.” IROS 2017. https://doi.org/10.1109/IROS.2017.8202133
32. Loquercio, A., et al. “Deep Drone Racing: From Simulation to Reality With Domain Randomization.” IEEE T-RO, 2020. https://doi.org/10.1109/TRO.2019.2942989
33. Shi, G., Hönig, W., Yue, Y., & Chung, S.-J. “Neural-Swarm: Decentralized Close-Proximity Multirotor Control Using Learned Interactions.” ICRA 2020. https://doi.org/10.1109/ICRA40945.2020.9196800
34. Veličković, P., et al. “Graph Attention Networks.” ICLR 2018. https://openreview.net/forum?id=rJXMpikCZ
35. Wen, M., et al. “Multi-Agent Reinforcement Learning is a Sequence Modeling Problem.” NeurIPS 2022. https://proceedings.neurips.cc/paper_files/paper/2022/hash/69413f87e5a34897cd010ca698097d0a-Abstract-Conference.html
36. Nayak, S., et al. “Scalable Multi-Agent Reinforcement Learning through Intelligent Information Aggregation.” ICML 2023. https://proceedings.mlr.press/v202/nayak23a.html


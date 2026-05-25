# 方向3：安全 × 通信交叉地带

## 1. 核验结论摘要

检索与核验日期：2026-05-26。旧报告仅作为审计线索；最终结论仅使用 DOI/Crossref、出版社、arXiv、OpenReview、NeurIPS Proceedings、AAMAS 官方论文集等可追溯来源。

本方向从旧报告中抽取 15 个论文式或准论文式条目。核验结果为：Verified 7 条，Metadata mismatch 2 条，Weak match 0 条，Not found 5 条，Replaced 1 条；另新增 11 篇可核验锚点/相邻文献用于重写综述。

旧报告中最关键的三个“直接交叉”条目未通过核验：`Multi-Agent Safety via Confidence-Weighted Communication / Liu NeurIPS 2023`、`Safe Networked Multi-Agent RL with Chance Constraints`、`Safe Multi-UAV via Distributed CBF without Communication / Zhao et al. TCNS 2024` 均未找到可靠来源支撑。旧报告关于“Liu et al. 2023 是最直接相关工作”的论断应删除。

经核验后，研究缺口仍可保守保留，但证据表述必须降级：本次核验未发现直接同时将安全约束和通信硬预算作为 MARL 联合优化目标的工作。更准确的定位是：安全 MARL、通信受限 MARL、CBF/事件触发安全控制和通信退化下的分布式安全控制分别存在成熟或相邻文献，但它们尚未形成面向 SafeComm-MARL 的统一问题设置、硬预算通信调度和形式化安全约束联合保证。

## 2. 原文献真实性审计表

| 原报告条目 | 状态 | 核验结果 | 可靠来源 | 处理方式 |
|---|---|---|---|---|
| Multi-Agent Safety via Confidence-Weighted Communication / Liu et al., NeurIPS 2023 | Not found | 精确标题、作者组合、NeurIPS Proceedings、OpenReview 与 arXiv 定向检索均未确认该论文。 | NeurIPS Proceedings 检索无命中；OpenReview/arXiv 定向检索无命中。 | 删除，不用于支撑“最相关工作”或研究缺口。 |
| Safe Networked Multi-Agent RL with Chance Constraints / Chen, Lu, Zhang, TNNLS 2022 | Not found | 未找到与该标题、作者、venue 同时匹配的 TNNLS 论文。 | Crossref/IEEE/TNNLS 题名与作者组合检索无可靠命中。 | 删除；机会约束 safe MARL 相关论证需另找真实文献。 |
| Safe Multi-UAV via Distributed CBF without Communication / Zhao, Xu, Liu, TCNS 2024 | Not found | 未找到该标题、作者组合和 TCNS 2024 元数据；旧报告中的 ORCA、无通信、12% 性能差等细节无来源。 | IEEE/Crossref/arXiv 定向检索无可靠命中。 | 删除；分布式 CBF 改用 Wang et al. 2017、Mestres et al. 2024、Ballotta & Talak 2025 等真实文献。 |
| Ames et al. “Event-triggered control for safety-critical systems” / L4DC 2020 | Metadata mismatch | 未找到旧题名与 L4DC 2020 的精确匹配；相近且可核验的是 Taylor et al. 2021 L-CSS 和 Xiao et al. 2022 IEEE TAC。 | Taylor et al. DOI: https://doi.org/10.1109/LCSYS.2020.3005101；Xiao et al. DOI: https://doi.org/10.1109/TAC.2022.3202088 | 用修正后的事件触发安全控制文献支撑，不保留旧元数据。 |
| Verginis & Dimarogonas “Event-triggered multi-robot formation control” / IFAC 2019 | Metadata mismatch | 找到相近主题但元数据不同：Yi, Wei, Dimarogonas, Johansson 2017 IFAC-PapersOnLine 论文，而非 Verginis & Dimarogonas 2019。 | https://doi.org/10.1016/j.ifacol.2017.08.1444 | 用修正后的 Yi et al. 2017 引用。 |
| distributed CBF limited communication | Replaced | 原报告未给出具体题名/作者；未能确认唯一旧条目。可核验相邻工作主要处理通信延迟、连通性维护或分布式安全导航，而非 MARL 通信硬预算。 | Ballotta & Talak 2025: https://doi.org/10.1109/TVT.2025.3546857；Mestres et al. 2024: https://arxiv.org/abs/2402.06195；Yang et al. 2024: https://arxiv.org/abs/2410.05798 | 作为相邻主题替换，不作为旧条目引用。 |
| MACPO / MAPPO-Lag | Verified | MACPO/MAPPO-Lagrangian 可在 Gu et al. 的 safe MARL 工作中核验；处理多智能体安全约束，但不处理通信预算。 | https://arxiv.org/abs/2110.02793；扩展期刊版：https://doi.org/10.1016/j.artint.2023.103905 | 保留为“安全 MARL 侧”锚点。 |
| CommNet | Verified | `Learning Multiagent Communication with Backpropagation`，NeurIPS 2016；学习连续通信，但无安全约束或硬通信预算。 | https://proceedings.neurips.cc/paper/2016/hash/55b1927fdafef39c48e5b73b5d61ea60-Abstract.html | 保留为通信 MARL 基础文献。 |
| DGN | Verified | `Graph Convolutional Reinforcement Learning`，ICLR 2020/arXiv；图卷积处理动态邻接关系，但无安全约束。 | https://arxiv.org/abs/1810.09202 | 保留为图通信/邻域聚合锚点。 |
| SchedNet | Verified | `Learning to Schedule Communication in Multi-agent Reinforcement Learning`，ICLR 2019；显式带宽/共享介质调度，但无安全约束。 | https://openreview.net/forum?id=SJxu5iR9KQ；https://arxiv.org/abs/1902.01554 | 保留为通信硬预算侧核心文献。 |
| VBC | Verified | `Efficient Communication in Multi-Agent Reinforcement Learning via Variance Based Control`，arXiv 2019；降低通信开销，但不处理安全。 | https://arxiv.org/abs/1909.02682 | 保留为通信效率相邻文献。 |
| MAGIC | Verified | `Multi-Agent Graph-Attention Communication and Teaming`，AAMAS 2021；学习何时与谁通信，但无形式化安全约束。 | https://www.ifaamas.org/Proceedings/aamas2021/pdfs/p964.pdf | 保留为动态图注意力通信文献。 |
| ATOC | Verified | `Learning Attentional Communication for Multi-Agent Cooperation`，NeurIPS 2018/arXiv；学习注意力通信，但不处理安全。 | https://arxiv.org/abs/1805.07733 | 保留为选择性通信文献。 |
| Chen et al. 2023 IoTJ（UAV 通信感知） | Not found | 条目缺少题名；按 Chen + 2023 + IEEE Internet of Things Journal + UAV + communication-aware/MARL 检索，未确认唯一论文。 | Crossref/IEEE 定向检索未确认唯一匹配。 | 删除；不能支撑应用场景论断。 |
| Zhao et al. 2024 TCNS（分布式 CBF-MARL） | Not found | 与上文 “Safe Multi-UAV via Distributed CBF without Communication” 疑似同源；未找到可核验 TCNS 2024 条目。 | Crossref/IEEE/arXiv 定向检索无可靠命中。 | 删除；不进入参考文献。 |

## 3. 经核验后的核心文献综述

### 3.1 安全控制与 CBF：安全保证成熟，但通常不学习通信预算

CBF 和 barrier certificate 文献为多机器人安全提供了强支撑。Wang, Ames, Egerstedt 的 IEEE T-RO 2017 论文将 safety barrier certificates 用于多机器人无碰撞控制，Ames et al. ECC 2019 对 CBF 理论和应用作了系统总结。这类工作适合支撑 SafeComm-MARL 的安全层或 shield 设计，但其核心对象是连续控制可行集和安全不变性，不是学习式通信调度。

事件触发控制提供了“资源使用与系统状态挂钩”的理论路线。Tabuada 2007 是事件触发实时调度的经典基础；Dimarogonas, Frazzoli, Johansson 2012 将事件触发扩展到多智能体系统；Taylor et al. 2021 和 Xiao et al. 2022 进一步把事件触发与 safety-critical/CBF 控制结合。这些结果强烈支持“安全风险上升时才提高控制/通信频率”的设计直觉，但它们大多依赖模型、控制律和触发条件设计，不是 MARL 中端到端学习通信策略。

### 3.2 安全 MARL：约束优化存在，但通信通常是理想化或未建模

Gu et al. 的 MACPO/MAPPO-Lagrangian 系列将多智能体安全学习建模为 constrained Markov game 或多智能体约束优化，并给出 safe MARL benchmark 和约束策略优化方法。它适合支撑 SafeComm-MARL 的 CMDP/Lagrangian 安全优化部分，但该路线默认智能体间的信息结构或训练范式，并未把“谁通信、何时通信、通信多少”作为硬预算约束来优化。

因此，MACPO 一类文献可以作为 SafeComm-MARL 的安全约束基线，而不能作为通信受限安全协同的直接先例。

### 3.3 通信 MARL：预算、调度和动态图已有进展，但缺少硬安全约束

CommNet 代表早期可微通信机制，强调通过反向传播学习消息表示；ATOC、DGN、MAGIC 等工作进一步引入注意力、图结构或动态交互建模，使智能体能够选择更有价值的通信对象或邻域信息。SchedNet 是本方向最重要的通信预算锚点之一，因为它明确建模有限带宽和共享通信介质，并学习哪些智能体获得广播机会。VBC 则从消息方差角度减少通信开销。

这些文献支持 SafeComm-VoI 的“信息价值驱动通信”思想，也能作为通信节省和调度基线。但是，它们的优化目标主要是任务回报、协作效率或通信开销，未把安全约束作为 CMDP/CBF 式硬约束处理。

### 3.4 安全 × 通信的真实相邻工作：控制侧更接近，MARL 侧仍不充分

Ballotta and Talak 2025 直接研究通信延迟下的多机器人安全分布式控制，并用 distributed CBF 与图神经网络学习安全控制器，说明通信退化会显著影响安全控制。Mestres et al. 2024 给出 CBF-based optimal controllers 的分布式安全导航框架。Yang et al. 2024 用 CBF/GP 学习通信质量并维护多机器人连通性。这些是“安全与通信退化/连通性”交叉最接近的真实锚点，但它们属于控制或学习增强控制，不是通信硬预算 MARL。

2026 年 Ocean Engineering 的 SMART-CMARL 工作（Draz et al.）是应用侧新的相邻证据：它在 USV 协同导航中同时讨论 bandwidth-efficient communication、消息调度、COLREG 合规与碰撞规避。但从可核验摘要和出版社元数据看，其安全更多体现为规则/惩罚/任务设计，尚不能等同于 CBF 或 CMDP 形式的硬安全约束联合优化；因此它削弱“完全无人触及应用交叉”的说法，但不推翻 SafeComm-MARL 的形式化研究缺口。

### 3.5 保守研究缺口

经核验后，更稳妥的结论是：安全约束、通信预算和多智能体学习分别有扎实基础，控制理论侧已有通信延迟、事件触发、连通性维护与安全的交叉工作；通信 MARL 侧已有有限带宽/调度机制；安全 MARL 侧已有约束策略优化。缺口不应写成“绝对空白”，而应写成：本次核验未发现直接同时将安全约束和通信硬预算作为 MARL 联合优化目标的工作。

## 4. 与 SafeComm-VoI / SafeComm-MARL 的关系

强支持：SchedNet、VBC、MAGIC、ATOC、DGN 支持“通信是可学习、可调度、可稀疏化的决策变量”；其中 SchedNet 最接近硬预算通信设置。

强支持：CBF、安全 barrier certificates、事件触发安全控制支持“安全风险可用于触发更密集控制/通信”的理论路线，适合作为 SafeComm-VoI 中 VoI 计算的安全项或 SafeComm-MARL 的 shield 层。

间接支持：MACPO/MAPPO-Lagrangian 支持把多智能体安全建模为约束优化，但需要扩展通信预算约束和可学习通信策略。

直接相邻但不足：Ballotta and Talak 2025、Yang et al. 2024、Draz et al. 2026 说明安全与通信条件在机器人/USV 系统中确实耦合；但它们没有形成“通信硬预算 + 形式化安全约束 + MARL 联合优化”的统一范式。

建议写法：SafeComm-MARL 可定位为在 constrained MARL 中显式加入通信预算约束，并用安全风险/价值信息驱动通信调度；SafeComm-VoI 可定位为连接事件触发安全控制与通信受限 MARL 的机制设计。论文中必须避免引用旧报告未核验的 Liu/Chen/Zhao 条目。

## 5. 可引用参考文献

1. Li Wang, Aaron D. Ames, Magnus Egerstedt. “Safety Barrier Certificates for Collisions-Free Multirobot Systems.” IEEE Transactions on Robotics, 2017. https://doi.org/10.1109/TRO.2017.2659727
2. Aaron D. Ames, Samuel Coogan, Magnus Egerstedt, Gennaro Notomista, Koushil Sreenath, Paulo Tabuada. “Control Barrier Functions: Theory and Applications.” ECC 2019. https://doi.org/10.23919/ECC.2019.8796030
3. Paulo Tabuada. “Event-Triggered Real-Time Scheduling of Stabilizing Control Tasks.” IEEE Transactions on Automatic Control, 2007. https://doi.org/10.1109/TAC.2007.904277
4. Dimos V. Dimarogonas, Emilio Frazzoli, Karl H. Johansson. “Distributed Event-Triggered Control for Multi-Agent Systems.” IEEE Transactions on Automatic Control, 2012. https://doi.org/10.1109/TAC.2011.2174666
5. Andrew J. Taylor, Pio Ong, Jorge Cortes, Aaron D. Ames. “Safety-Critical Event Triggered Control via Input-to-State Safe Barrier Functions.” IEEE Control Systems Letters, 2021. https://doi.org/10.1109/LCSYS.2020.3005101；arXiv: https://arxiv.org/abs/2003.06963
6. Wei Xiao, Calin Belta, Christos G. Cassandras. “Event-Triggered Control for Safety-Critical Systems With Unknown Dynamics.” IEEE Transactions on Automatic Control, 2022. https://doi.org/10.1109/TAC.2022.3202088
7. Xinlei Yi, Jieqiang Wei, Dimos V. Dimarogonas, Karl H. Johansson. “Formation Control for Multi-Agent Systems with Connectivity Preservation and Event-Triggered Controllers.” IFAC-PapersOnLine, 2017. https://doi.org/10.1016/j.ifacol.2017.08.1444
8. Luca Ballotta, Rajat Talak. “Safe Distributed Control of Multi-Robot Systems With Communication Delays.” IEEE Transactions on Vehicular Technology, 2025. https://doi.org/10.1109/TVT.2025.3546857；arXiv: https://arxiv.org/abs/2402.09382
9. Pol Mestres, Carlos Nieto-Granda, Jorge Cortés. “Distributed Safe Navigation of Multi-Agent Systems using Control Barrier Function-Based Optimal Controllers.” arXiv, 2024. https://arxiv.org/abs/2402.06195
10. Yupeng Yang, Yiwei Lyu, Yanze Zhang, Ian Gao, Wenhao Luo. “Integrating Online Learning and Connectivity Maintenance for Communication-Aware Multi-Robot Coordination.” IROS 2024/arXiv. https://arxiv.org/abs/2410.05798
11. Shangding Gu, Jakub Grudzien Kuba, Munning Wen, Ruiqing Chen, Ziyan Wang, Zheng Tian, Jun Wang, Alois Knoll, Yaodong Yang. “Multi-Agent Constrained Policy Optimisation.” arXiv, 2021/2022. https://arxiv.org/abs/2110.02793；扩展期刊版 “Safe multi-agent reinforcement learning for multi-robot control.” Artificial Intelligence, 2023. https://doi.org/10.1016/j.artint.2023.103905
12. Daewoo Kim, Sangwoo Moon, David Hostallero, Wan Ju Kang, Taeyoung Lee, Kyunghwan Son, Yung Yi. “Learning to Schedule Communication in Multi-agent Reinforcement Learning.” ICLR 2019. https://openreview.net/forum?id=SJxu5iR9KQ；arXiv: https://arxiv.org/abs/1902.01554
13. Sainbayar Sukhbaatar, Arthur Szlam, Rob Fergus. “Learning Multiagent Communication with Backpropagation.” NeurIPS 2016. https://proceedings.neurips.cc/paper/2016/hash/55b1927fdafef39c48e5b73b5d61ea60-Abstract.html
14. Jiechuan Jiang, Chen Dun, Tiejun Huang, Zongqing Lu. “Graph Convolutional Reinforcement Learning.” ICLR 2020/arXiv. https://arxiv.org/abs/1810.09202
15. Sai Qian Zhang, Qi Zhang, Jieyu Lin. “Efficient Communication in Multi-Agent Reinforcement Learning via Variance Based Control.” arXiv, 2019. https://arxiv.org/abs/1909.02682
16. Yaru Niu, Rohan Paleja, Matthew Gombolay. “Multi-Agent Graph-Attention Communication and Teaming.” AAMAS 2021. https://www.ifaamas.org/Proceedings/aamas2021/pdfs/p964.pdf
17. Jiechuan Jiang, Zongqing Lu. “Learning Attentional Communication for Multi-Agent Cooperation.” NeurIPS 2018/arXiv. https://arxiv.org/abs/1805.07733
18. Umar Draz, Sana Yasin, Tariq Ali, Mohammad Hijji, Muhammad Ayaz, El-Hadi M. Aggoune. “Communication-aware multi-agent reinforcement learning for cooperative navigation of multiple USVs in dynamic ocean environments.” Ocean Engineering, 2026. https://doi.org/10.1016/j.oceaneng.2025.123626

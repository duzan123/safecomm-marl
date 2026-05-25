# 方向1：安全 MARL

## 1. 核验结论摘要

核验日期：2026-05-26。旧报告仅作为审计输入，最终综述只使用 DOI、出版社/论文集、arXiv、OpenReview、PMLR、NeurIPS/ICLR/AAAI/AAMAS 官方或准官方页面可追溯的文献。

本方向共审计旧报告中的 14 个论文式条目。核验结果为：Verified 1 条，Metadata mismatch 8 条，Weak match 1 条，Not found 1 条，Replaced 3 条。另补入 5 类真实锚点：Altman 的 CMDP 专著、Achiam et al. 的 CPO、Yu et al. 的 MAPPO、Lu et al. 的 Safe Dec-PG、Ying et al. 的通信半径截断 safe MARL。

旧报告中可以保留但需要改写的结论是：安全 MARL 常以 CMDP/约束马尔可夫博弈建模，MACPO/MAPPO-Lagrangian 是重要基线，CBF/shielding 类方法能提供更强的运行时安全过滤，Safety Gymnasium 可作为 safe RL/MARL 评测基础设施。需要降级的结论是：旧报告给出的“训练全通信比例”“显式建模通信质量比例”等统计没有可靠抽样依据；HAZARD 不是经核验的“Multi-Agent Safe RL benchmark”；SAUTE RL 是单智能体 safe RL 方法，不能直接作为多智能体扩展证据。需要删除或替换的结论是：旧报告中关于“Multi-Agent Safety: CMDP Approach with Communication”“Safe Networked MARL with Chance Constraints”“Decentralized Safe RL with Generalization Guarantees”“Scalable Safe Policy Improvement via Shielding for Multi-Agent Systems”的具体题名、作者和 venue 断言，本次未能按原样核验。

保守结论：本次核验发现，已有工作已经覆盖了去中心化安全策略梯度、网络化/局部通信半径下的安全 MARL、CBF/shielding 的去中心化安全过滤等方向；但未发现直接以 SafeComm-VoI 方式系统建模“安全约束违反风险、通信价值、带宽/丢包/延迟预算”三者权衡的核心论文。因此 SafeComm-MARL 的 gap 可以表述为“现有 safe MARL 已考虑去中心化或局部通信，但对通信价值与安全约束估计之间的联合建模仍不充分”，不应表述为“没有任何通信受限 safe MARL 工作”。

## 2. 原文献真实性审计表

| 原报告条目 | 状态 | 核验结果 | 可靠来源 | 处理方式 |
|---|---|---|---|---|
| MACPO: Constrained Policy Optimization for Multi-Agent Reinforcement Learning | Metadata mismatch | 旧报告题名、作者和发表状态错误。可核验论文为 Gu et al. 的 *Multi-Agent Constrained Policy Optimisation*，arXiv:2110.02793；其期刊扩展为 *Safe multi-agent reinforcement learning for multi-robot control*，Artificial Intelligence 2023。 | arXiv: https://arxiv.org/abs/2110.02793；AIJ/DOI: https://doi.org/10.1016/j.artint.2023.103905 | 使用修正后的 Gu et al. MACPO / MAPPO-Lagrangian 元数据；不保留旧作者与旧题名。 |
| MAPPO-Lagrangian / Safe MAPPO | Metadata mismatch | MAPPO-Lagrangian 可在 Gu et al. MACPO/AIJ 工作中核验；MAPPO 本身是 Yu et al. NeurIPS 2022 的合作 MARL 基线。未核验到旧报告所称 Bettini/Prorok 等 ICLR 2022 “Safe MAPPO”独立论文。 | Gu et al. arXiv: https://arxiv.org/abs/2110.02793；Yu et al. NeurIPS: https://papers.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html | 作为 MACPO/AIJ 中的 MAPPO-Lagrangian 基线引用；“Safe MAPPO”不作为独立文献支撑。 |
| MAMPS: Safe Multi-Agent Reinforcement Learning via Model Predictive Shielding | Metadata mismatch | 论文存在，但旧报告的 AAAI 2021、作者列表错误。真实条目为 Wenbo Zhang, Osbert Bastani, Vijay Kumar，arXiv:1910.12639，2019。 | arXiv: https://arxiv.org/abs/1910.12639 | 使用 arXiv 2019 元数据；只作为 model predictive shielding 的代表。 |
| Multi-Agent Constrained Policy Optimisation (MACPPO) | Metadata mismatch | 旧报告把 MACPO 写成 MACPPO，并称 ICLR 2023，作者也不一致。可核验条目仍是 Gu et al. *Multi-Agent Constrained Policy Optimisation* 及其 AIJ 扩展。 | arXiv: https://arxiv.org/abs/2110.02793；AIJ/DOI: https://doi.org/10.1016/j.artint.2023.103905 | 与 MACPO 条目合并处理，不单列 MACPPO。 |
| Safe Multi-Agent Reinforcement Learning for Multi-Robot Control | Metadata mismatch | 论文存在于 Artificial Intelligence 319:103905, 2023；旧报告年份、部分作者和“Lyapunov/safety projection layer”描述不可靠。该文核心是 constrained Markov game、MACPO、MAPPO-Lagrangian 和 Safe MAMuJoCo/Safe MARobosuite/Safe MAIG。 | ScienceDirect/DOI: https://doi.org/10.1016/j.artint.2023.103905 | 使用修正后的 AIJ 2023 期刊版本。 |
| HAZARD: A Benchmark for Multi-Agent Safe Reinforcement Learning | Metadata mismatch | HAZARD 论文存在，但真实题名为 *HAZARD Challenge: Embodied Decision Making in Dynamically Changing Environments*，ICLR 2024；作者和主题均与旧报告不一致，且不是专门的 multi-agent safe RL 基准。 | ICLR Proceedings: https://proceedings.iclr.cc/paper_files/paper/2024/hash/d937cb3fe2851ed0ab9af5e38f885077-Abstract-Conference.html；OpenReview: https://openreview.net/forum?id=n6mLhaBahJ | 不用于支撑 Safe MARL 核心结论；仅可作为动态灾害 embodied benchmark 的旁支背景。 |
| Multi-Agent Safety: CMDP Approach with Communication | Replaced | 未找到旧报告所述题名、作者和 NeurIPS 2023 条目。真实相关替代文献包括 Lu et al. AAAI 2021 的 Safe Dec-PG，以及 Ying et al. NeurIPS 2023 的通信半径截断 primal-dual safe MARL。 | Lu et al. AAAI: https://doi.org/10.1609/aaai.v35i10.17062；Ying et al. NeurIPS: https://proceedings.neurips.cc/paper_files/paper/2023/hash/72a1ec14aed36985ffba175e0bba3fec-Abstract-Conference.html | 用 Lu et al. 和 Ying et al. 替代；删除旧条目的具体断言。 |
| Safety Gymnasium: A Unified Safe Reinforcement Learning Benchmark | Verified | 题名、NeurIPS 2023 Datasets and Benchmarks、主要作者和安全 RL/MARL 基准定位可核验。需注意 arXiv 页面写作 “Safety-Gymnasium”，作者名 Dai 在不同页面有 Juntao/Josef 变体。 | NeurIPS: https://papers.nips.cc/paper_files/paper/2023/hash/3c557a3d6a48cc99444f85e924c66753-Abstract-Datasets_and_Benchmarks.html；arXiv: https://arxiv.org/abs/2310.12567 | 可引用。 |
| Safe Reinforcement Learning via Control Barrier Functions with Multi-Agent Extensions | Weak match | 旧报告给出的代表性题名和作者组合未能精确核验。真实相关工作包括 Qin et al. ICLR 2021 的 decentralized neural barrier certificates、Cai et al. arXiv 2021 的 decentralized multiple CBFs，以及 Ames et al. TAC 2017 的 CBF-QP 基础。 | Qin et al. OpenReview: https://openreview.net/forum?id=P6_q1BRxY8Q；Cai et al. arXiv: https://arxiv.org/abs/2103.12553；Ames et al. DOI: https://doi.org/10.1109/TAC.2016.2638961 | 旧条目不直接引用；用核验后的 CBF 文献重写。 |
| Decentralized Safe Reinforcement Learning with Generalization Guarantees | Replaced | 未找到旧报告所称题名、作者和 ICML 2023 条目。可替代的真实证据是 Lu et al. AAAI 2021 的去中心化安全 PG；“generalization guarantees”更接近 Qin et al. ICLR 2021 的多智能体 barrier 泛化实验/保证，但不是该旧题名。 | Lu et al. AAAI: https://doi.org/10.1609/aaai.v35i10.17062；Qin et al. OpenReview: https://openreview.net/forum?id=P6_q1BRxY8Q | 用真实去中心化 safe MARL 与 CBF 泛化文献替代。 |
| Scalable Safe Policy Improvement via Shielding for Multi-Agent Systems | Replaced | 未找到旧报告题名与作者。真实相关工作包括 Elsayed-Aly et al. AAMAS 2021 的 safe MARL via shielding，以及 Melcer et al. NeurIPS 2022 的 shield decentralization。 | AAMAS/作者 PDF: https://www.cs.virginia.edu/~lufeng/papers/aamas2021.pdf；NeurIPS: https://proceedings.neurips.cc/paper_files/paper/2022/hash/57444e14ecd9e2c8f603b4f012ce3811-Abstract-Conference.html | 用 verified shielding 文献替代；不使用旧题名。 |
| Safe Networked Multi-Agent Reinforcement Learning with Chance Constraints | Not found | 未找到旧报告给出的题名、作者、TNNLS 2022 记录。搜索到的 chance constraints 多为控制/MPC或能源系统语境，不能确认该旧条目存在。 | 定向检索未得到 IEEE/TNNLS/Crossref/出版社可核验记录；可对照 Lu et al. AAAI 网络化 D-CMDP: https://doi.org/10.1609/aaai.v35i10.17062 | 不引用，不支撑结论。 |
| SAUTE RL: Almost Surely Safe Reinforcement Learning Using State Augmentation (Multi-Agent Extension) | Metadata mismatch | SAUTE RL 真实存在，ICML 2022/PMLR，单智能体 safe RL；旧报告所称 2023 多智能体扩展未核验到。 | PMLR: https://proceedings.mlr.press/v162/sootla22a.html；arXiv: https://arxiv.org/abs/2202.06558 | 作为单智能体 almost-sure safe RL 状态增广思想引用；不声称已有多智能体扩展。 |
| Safe Multi-Agent Reinforcement Learning for Price-Based Demand Response / Gu et al. survey | Metadata mismatch | 可核验综述为 Gu et al. *A Review of Safe Reinforcement Learning: Methods, Theory and Applications*，arXiv:2205.10330。旧报告题名、年份和“price-based demand response”混合不可靠。 | arXiv: https://arxiv.org/abs/2205.10330 | 使用真实综述作为 safe RL 背景；不把它当成通信受限 safe MARL 的直接证据。 |

## 3. 经核验后的核心文献综述

安全 MARL 的建模主线来自 CMDP。Altman 的 *Constrained Markov Decision Processes* 将“最大化收益，同时约束一个或多个成本”形式化为 occupation measure、线性规划与拉格朗日方法等基础工具。Achiam et al. 的 CPO 将这一思想带入深度 RL，通过 trust-region 更新给出近似约束满足的迭代保证。多智能体场景中，Gu et al. 将问题提升为 constrained Markov game：每个智能体或团队既优化回报，又受到个体/联合成本约束影响；MACPO 继承 CPO 和多智能体 trust-region 思路，MAPPO-Lagrangian 则以拉格朗日松弛方式给出更实用的近似实现。

从“算法基线”角度看，MAPPO 本身不是安全算法，但它是合作 MARL 中经常使用的强基线；MAPPO-Lagrangian 的意义在于把这一工程上有效的 actor-critic 结构接入成本约束。Gu et al. 的 AIJ 2023 版本进一步给出 Safe MAMuJoCo、Safe MARobosuite 和 Safe MAIG 等多机器人安全基准，并报告 MACPO/MAPPO-Lagrangian 在 reward-safety 权衡上的表现。可防守的写法是：MACPO/MAPPO-Lagrangian 是当前 safe MARL 中约束优化路线的重要起点；但它们的理论和实验设置并不等价于“通信受限、丢包、延迟下仍有安全保证”。

去中心化与网络化 safe MARL 已有真实工作，旧报告不能声称该方向空白。Lu et al. 在 AAAI 2021 中把多智能体安全学习建模为 distributed CMDP，假设智能体通过 peer-to-peer 网络与邻居交换信息，并提出 Safe Dec-PG，给出可量化收敛分析。Ying et al. 在 NeurIPS 2023 中进一步研究 general utilities 下的 safe MARL，使用 shadow reward 和 k-hop neighbor truncation，把通信半径 k 纳入算法和误差项。这两篇文献对 SafeComm-MARL 特别关键：它们证明“局部通信/网络拓扑”已经进入 safe MARL 理论文献，因此 SafeComm 的新颖性需要落在“通信价值选择、带宽/丢包/延迟建模、对安全约束估计的影响”上，而不是简单落在“安全 + 多智能体 + 通信”三词交叉。

另一条主线是运行时安全过滤。MAMPS 使用 model predictive shielding，在 learned policy 不可证明安全时切换到 backup policy；其安全性依赖模型预测和可验证的备份策略。CBF 路线以 Ames et al. 的 CBF-QP 为基础，通过控制输入约束保证安全集合前向不变；在多智能体中，Qin et al. 学习 decentralized neural barrier certificates，Cai et al. 将 decentralized multiple CBFs 接入 MADDPG-CBF。Shielding 路线中，Elsayed-Aly et al. 给出 centralized/factored shielding 的 safe MARL 框架，Melcer et al. 则研究将集中式 shield 分解为去中心化 shield。上述方法都对 SafeComm 有间接支持：如果安全过滤器需要邻居状态、局部模型或 shield 状态，那么通信受限会影响过滤器可用信息；但这些论文通常并不把“通信是否值得发送”作为优化目标。

基准方面，Safety Gymnasium 是可核验的 NeurIPS 2023 Datasets and Benchmarks 论文，包含安全 RL 的环境套件和 SafePO 算法库，并覆盖 single-agent 与 multi-agent 安全任务。它比旧报告中的 HAZARD 更适合作为本方向实验平台锚点。HAZARD 的真实 ICLR 2024 论文是 embodied decision making in dynamically changing environments，包含 fire、flood、wind 等灾害场景，但它不是专门的 multi-agent safe RL benchmark；因此可以作为“动态灾害环境”灵感，不应作为 safe MARL 主证据。

SAUTE RL 提供了另一种安全语义：通过状态增广把安全预算纳入状态，从而逼近 almost-sure constraint satisfaction。该工作在 ICML 2022 可核验，但本次未发现其多智能体扩展的可靠论文。因此它可以作为 SafeComm-MARL 设计中的“本地安全预算/约束状态”思想来源，不能写成已有的 multi-agent SAUTE baseline。

综上，核验后的文献图谱更像三层结构：第一层是 CMDP/CPO/MACPO/MAPPO-Lagrangian 的约束优化；第二层是 Safe Dec-PG、k-hop primal-dual 等去中心化/网络化 safe MARL；第三层是 MAMPS、CBF、shield decentralization 等运行时安全过滤。SafeComm-VoI 应在第二层和第三层之间定位：在局部通信、延迟或丢包存在时，选择哪些信息最能降低约束估计误差和运行时安全风险。

## 4. 与 SafeComm-VoI / SafeComm-MARL 的关系

强支持的关系有三点。第一，MACPO/MAPPO-Lagrangian 和 AIJ 2023 safe multi-robot control 论文说明，safe MARL 的主流可用基线仍是 CMDP/约束马尔可夫博弈 + policy optimization，因此 SafeComm-MARL 可以沿用成本约束、拉格朗日乘子、cost critic 等接口。第二，Lu et al. 与 Ying et al. 说明，去中心化网络、邻居通信、k-hop 截断已经是 safe MARL 的合理理论对象；SafeComm 不能忽略这些工作。第三，CBF/shielding 文献说明，运行时安全常依赖局部邻居状态或模型信息，因此通信缺失确实会影响安全过滤质量。

间接支持的关系是：Safety Gymnasium 和 Safe MAMuJoCo/Safe MARobosuite/Safe MAIG 可为实验提供成本函数、约束违反率和多智能体任务接口；SAUTE RL 可启发“安全预算状态”的建模；MAMPS/shield decentralization 可启发“必要时覆盖策略动作”的安全层。

证据不足或需要谨慎表述的部分是：本次核验未发现一个已被可靠来源收录的工作直接提出 SafeComm-VoI，即在 safe MARL 中显式估计“某条消息对未来约束违反概率或成本上界的边际价值”，并同时处理带宽、丢包和延迟。旧报告中“现有工作几乎都依赖全通信”“只有 1 篇显式建模通信质量”等比例结论没有可靠统计支撑，应删除。更稳妥的 gap 写法是：已有去中心化 safe MARL 多把通信图或通信半径作为给定条件，已有 CBF/shielding 多把局部状态可得性作为安全过滤前提；SafeComm-VoI 可研究当通信资源本身受限且不可靠时，如何选择、压缩或调度安全相关信息，以降低约束估计误差和运行时风险。

## 5. 可引用参考文献

1. Eitan Altman. *Constrained Markov Decision Processes*. Chapman & Hall/CRC, 1999. Routledge: https://www.routledge.com/Constrained-Markov-Decision-Processes/Altman/p/book/9780849303821
2. Joshua Achiam, David Held, Aviv Tamar, Pieter Abbeel. “Constrained Policy Optimization.” ICML 2017, PMLR 70:22-31. https://proceedings.mlr.press/v70/achiam17a.html
3. Shangding Gu, Jakub Grudzien Kuba, Munning Wen, Ruiqing Chen, Ziyan Wang, Zheng Tian, Jun Wang, Alois Knoll, Yaodong Yang. “Multi-Agent Constrained Policy Optimisation.” arXiv:2110.02793, 2021. https://arxiv.org/abs/2110.02793
4. Shangding Gu, Jakub Grudzien Kuba, Yuanpei Chen, Yali Du, Long Yang, Alois Knoll, Yaodong Yang. “Safe multi-agent reinforcement learning for multi-robot control.” *Artificial Intelligence*, 319:103905, 2023. https://doi.org/10.1016/j.artint.2023.103905
5. Chao Yu, Akash Velu, Eugene Vinitsky, Jiaxuan Gao, Yu Wang, Alexandre Bayen, Yi Wu. “The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games.” NeurIPS 2022 Datasets and Benchmarks. https://papers.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html
6. Songtao Lu, Kaiqing Zhang, Tianyi Chen, Tamer Basar, Lior Horesh. “Decentralized Policy Gradient Descent Ascent for Safe Multi-Agent Reinforcement Learning.” AAAI 2021. https://doi.org/10.1609/aaai.v35i10.17062
7. Donghao Ying, Yunkai Zhang, Yuhao Ding, Alec Koppel, Javad Lavaei. “Scalable Primal-Dual Actor-Critic Method for Safe Multi-Agent RL with General Utilities.” NeurIPS 2023. https://proceedings.neurips.cc/paper_files/paper/2023/hash/72a1ec14aed36985ffba175e0bba3fec-Abstract-Conference.html
8. Jiaming Ji, Borong Zhang, Jiayi Zhou, Xuehai Pan, Weidong Huang, Ruiyang Sun, Yiran Geng, Yifan Zhong, Josef/Juntao Dai, Yaodong Yang. “Safety Gymnasium: A Unified Safe Reinforcement Learning Benchmark.” NeurIPS 2023 Datasets and Benchmarks. https://papers.nips.cc/paper_files/paper/2023/hash/3c557a3d6a48cc99444f85e924c66753-Abstract-Datasets_and_Benchmarks.html
9. Aivar Sootla, Alexander I. Cowen-Rivers, Taher Jafferjee, Ziyan Wang, David H. Mguni, Jun Wang, Haitham Ammar. “Saute RL: Almost Surely Safe Reinforcement Learning Using State Augmentation.” ICML 2022, PMLR 162:20423-20443. https://proceedings.mlr.press/v162/sootla22a.html
10. Shangding Gu, Long Yang, Yali Du, Guang Chen, Florian Walter, Jun Wang, Yaodong Yang, Alois Knoll. “A Review of Safe Reinforcement Learning: Methods, Theory and Applications.” arXiv:2205.10330, 2022. https://arxiv.org/abs/2205.10330
11. Wenbo Zhang, Osbert Bastani, Vijay Kumar. “MAMPS: Safe Multi-Agent Reinforcement Learning via Model Predictive Shielding.” arXiv:1910.12639, 2019. https://arxiv.org/abs/1910.12639
12. Aaron D. Ames, Xiangru Xu, Jessy W. Grizzle, Paulo Tabuada. “Control Barrier Function Based Quadratic Programs for Safety Critical Systems.” *IEEE Transactions on Automatic Control*, 62(8):3861-3876, 2017. https://doi.org/10.1109/TAC.2016.2638961
13. Zengyi Qin, Kaiqing Zhang, Yuxiao Chen, Jingkai Chen, Chuchu Fan. “Learning Safe Multi-Agent Control with Decentralized Neural Barrier Certificates.” ICLR 2021. https://openreview.net/forum?id=P6_q1BRxY8Q
14. Zhiyuan Cai, Huanhui Cao, Wenjie Lu, Lin Zhang, Hao Xiong. “Safe Multi-Agent Reinforcement Learning through Decentralized Multiple Control Barrier Functions.” arXiv:2103.12553, 2021. https://arxiv.org/abs/2103.12553
15. Ingy Elsayed-Aly, Suda Bharadwaj, Christopher Amato, Ruediger Ehlers, Ufuk Topcu, Lu Feng. “Safe Multi-Agent Reinforcement Learning via Shielding.” AAMAS 2021. https://www.cs.virginia.edu/~lufeng/papers/aamas2021.pdf
16. Daniel Melcer, Christopher Amato, Stavros Tripakis. “Shield Decentralization for Safe Multi-Agent Reinforcement Learning.” NeurIPS 2022. https://proceedings.neurips.cc/paper_files/paper/2022/hash/57444e14ecd9e2c8f603b4f012ce3811-Abstract-Conference.html

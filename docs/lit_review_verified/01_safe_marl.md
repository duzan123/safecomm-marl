# 方向1：安全 MARL

## 1. 核验结论摘要

- 原报告文献数量：14 条主文献条目。
- Verified 数量：2。
- Metadata mismatch 数量：6。
- Weak match 数量：2。
- Not found 数量：2。
- Replaced 数量：2。
- Replaced 或新增文献数量：8。新增或替换纳入的可引用核心文献包括 Altman 的 CMDP 专著、Achiam et al. 的 CPO、Yu et al. 的 MAPPO、Gu et al. 的 Safe RL 综述、Lu et al. 的 Safe Dec-PG、Xiao et al. 的 MBDS、Bianchi et al. 的 factored multi-agent safe policy improvement，以及 Guo et al. 的通信延迟安全 MADRL 应用。
- 对原报告核心结论的影响：原报告“Safe MARL 主流方法训练阶段常依赖 CTDE、全局状态或集中式 critic”的方向基本成立，但证据需要收紧。已核验文献中同时存在去中心化优化、局部 CBF、动态 shielding、通信延迟电网控制等工作，因此不能声称“没有任何工作研究通信或局部信息”。更准确的结论是：核心 Safe MARL 基线多在训练中使用全局状态、联合动作或集中式价值函数；已有去中心化/网络化安全工作通常仍假设可靠邻居通信、全局状态可观测、固定通信图或领域特定同步机制，对带宽受限、丢包、延迟与安全约束保证之间的统一理论关系仍不充分。

## 2. 原文献真实性审计表

| 原条目 | 状态 | 核验结果 | 可靠来源 | 处理 |
|---|---|---|---|---|
| MACPO: Constrained Policy Optimization for Multi-Agent Reinforcement Learning | Metadata mismatch | 未找到该英文题名对应的正式论文。可核验的核心来源是 Gu et al. 的 "Multi-Agent Constrained Policy Optimisation"，其中提出 MACPO 和 MAPPO-Lagrangian。原报告作者、题名和 workshop/年份表述不可靠。 | https://arxiv.org/abs/2110.02793；https://dblp.org/rec/journals/corr/abs-2110-02793 | 不按原题名引用；并入已核验的 MACPO/MAPPO-L 条目。 |
| MAPPO-Lagrangian / Safe MAPPO | Metadata mismatch | MAPPO-Lagrangian 是 Gu et al. "Multi-Agent Constrained Policy Optimisation" 中的算法，不是独立 ICLR 论文；“Safe MAPPO”作为泛称可理解，但不可作为独立文献引用。 | https://arxiv.org/abs/2110.02793；https://github.com/chauncygu/multi-agent-constrained-policy-optimisation | 作为 MACPO 论文中的算法讨论，不列为独立核心文献。 |
| MAMPS: Safe Multi-Agent Reinforcement Learning via Model Predictive Shielding | Metadata mismatch | 题名真实，但原报告作者和 AAAI 2021 来源错误。可核验版本为 Wenbo Zhang, Osbert Bastani, Vijay Kumar 的 arXiv:1910.12639。 | https://arxiv.org/abs/1910.12639 | 保留为 shielding 类文献，但纠正作者和来源。 |
| Multi-Agent Constrained Policy Optimisation (MACPPO) | Metadata mismatch | 真实题名是 "Multi-Agent Constrained Policy Optimisation"，算法名为 MACPO，不是 MACPPO；未核验到 ICLR 2023 正式发表。相关扩展期刊版为 "Safe multi-agent reinforcement learning for multi-robot control"，AIJ 2023。 | https://arxiv.org/abs/2110.02793；https://doi.org/10.1016/j.artint.2023.103905；https://pair-lab.ai/publication/aij/ | 以 arXiv MACPO 和 AIJ 期刊版作为可引用来源。 |
| Safe Multi-Agent Reinforcement Learning for Multi-Robot Control | Verified | 题名、主要作者和 AIJ 期刊来源可核验；原报告年份写作 2022 不准确，正式期刊信息为 Artificial Intelligence, 2023。 | https://doi.org/10.1016/j.artint.2023.103905；https://pair-lab.ai/publication/aij/ | 保留并纠正年份、venue 和 DOI。 |
| HAZARD: A Benchmark for Multi-Agent Safe Reinforcement Learning | Weak match | 存在 ICLR 2024 "HAZARD Challenge: Embodied Decision Making in Dynamically Changing Environments"，但它是 embodied decision-making benchmark，不是题名所称的 Multi-Agent Safe RL benchmark。 | https://arxiv.org/abs/2401.12975；https://embodied-agi.cs.umass.edu/hazard/；https://openreview.net/pdf/58c946ebc8e9d80791fa263544f59b7be48edeef.pdf | 不作为 Safe MARL 核心证据；仅可作为动态危险环境 benchmark 背景。 |
| Multi-Agent Safety: CMDP Approach with Communication | Not found | 未找到与该题名、作者组合或“Confidence-Weighted Communication”描述一致的 NeurIPS 2023 论文。 | 无可追溯来源；已检索题名、通信质量、安全 MARL、Zuxin Liu 相关关键词。 | 不引用；仅保留在审计表。 |
| Safety Gymnasium: A Unified Safe Reinforcement Learning Benchmark | Verified | NeurIPS 2023 Datasets and Benchmarks 论文可核验，确实包含单智能体和多智能体安全任务，并提供 SafePO/Safety-Gymnasium 基准。 | https://papers.nips.cc/paper_files/paper/2023/hash/3c557a3d6a48cc99444f85e924c66753-Abstract-Datasets_and_Benchmarks.html；https://proceedings.neurips.cc/paper_files/paper/2023/file/3c557a3d6a48cc99444f85e924c66753-Paper-Datasets_and_Benchmarks.pdf | 保留为基准文献。 |
| Safe Reinforcement Learning via Control Barrier Functions with Multi-Agent Extensions | Weak match | 原条目是泛化描述，未给出可核验的准确论文。可核验的相近工作包括 "Safe Multi-Agent Reinforcement Learning through Decentralized Multiple Control Barrier Functions"。 | https://arxiv.org/abs/2103.12553；https://dblp.org/rec/journals/corr/abs-2103-12553.html | 用可核验的 decentralized multiple CBF 论文替代泛称。 |
| Decentralized Safe Reinforcement Learning with Generalization Guarantees | Replaced | 未找到该精确题名或原报告所述 PAC-Bayes 多智能体 ICML 2023 文献。可核验的去中心化安全 MARL 代表是 Lu et al. AAAI 2021 Safe Dec-PG。 | https://ojs.aaai.org/index.php/AAAI/article/view/17062；https://doi.org/10.1609/aaai.v35i10.17062 | 替换为 Lu et al. 的去中心化安全策略梯度文献。 |
| Scalable Safe Policy Improvement via Shielding for Multi-Agent Systems | Replaced | 未找到原题名和 AAAI 2023 元数据。相近且可核验的方向包括 AAMAS 2023 model-based dynamic shielding，以及 ICML 2024 factored multi-agent safe policy improvement。 | https://arxiv.org/abs/2304.06281；https://www.ifaamas.org/Proceedings/aamas2023/pdfs/p1587.pdf；https://proceedings.mlr.press/v235/bianchi24b.html | 替换为 Xiao et al. 2023 和 Bianchi et al. 2024。 |
| Safe Networked Multi-Agent Reinforcement Learning with Chance Constraints | Not found | 未找到该精确题名、TNNLS 2022 元数据或原报告作者组合。检索到的相关可核验工作多为网络化/通信延迟安全 MADRL，而非原题名机会约束论文。 | 无可追溯来源对应原条目；相关替代来源：https://doi.org/10.1016/j.apenergy.2023.121648 | 不引用原条目；通信延迟应用另作为新增文献。 |
| SAUTE RL: Almost Surely Safe Reinforcement Learning Using State Augmentation (Multi-Agent Extension) | Metadata mismatch | SAUTE RL 本身真实且发表于 ICML 2022/PMLR，但核验来源支持的是单智能体 Safe RL；未找到原报告所称“2023 多智能体扩展”。 | https://proceedings.mlr.press/v162/sootla22a.html；https://arxiv.org/abs/2202.06558 | 保留 SAUTE RL 作为状态增广安全思想，不声称已核验多智能体扩展。 |
| Safe Multi-Agent Reinforcement Learning for Price-Based Demand Response | Metadata mismatch | 题名可核验，作者为 Hannah Markgraf and Matthias Althoff；可引用正式来源为 IEEE PES ISGT EUROPE 2023，不是原报告所称 IEEE Transactions on Smart Grid。 | https://doi.org/10.1109/isgteurope56780.2023.10407281；https://ieeexplore.ieee.org/document/10407281；https://openalex.org/W4391341675 | 可作为能源需求响应应用文献，但需按 IEEE 会议元数据引用。 |

## 3. 经核验后的核心文献综述

Safe MARL 的理论基础主要来自 CMDP 和安全策略优化。Altman 的 CMDP 专著给出“最大化回报并满足成本约束”的经典建模语言；Achiam et al. 的 CPO 将约束策略优化和 trust region 更新结合，提供近似逐迭代约束满足的单智能体 Safe RL 基线。Gu et al. 将这一思路扩展到 constrained Markov game，提出 MACPO 和 MAPPO-Lagrangian；AIJ 期刊版进一步把 Safe MAMuJoCo、Safe MARobosuite、Safe MAIG 等多机器人安全基准纳入同一框架。这里最关键的事实是：这些方法的理论和实现通常需要联合状态、联合动作、团队成本或集中式 critic 来估计回报与约束优势，因此它们适合作为 SafeComm-MARL 的强基线，但不能直接回答训练通信受限时 critic 如何保持安全估计的问题。

MAPPO 本身不是安全算法，但 Yu et al. 系统展示 PPO/MAPPO 在合作 MARL 中是强基线，这解释了为什么 MAPPO-Lagrangian 会成为 Safe MARL 中自然的工程起点。它的价值在于稳定、易复现和 CTDE 训练范式成熟；局限也相同：集中式训练阶段通常假设可访问全局状态或全局观测拼接，通信受限时需要重新定义 critic 输入、代价估计和拉格朗日乘子更新。

另一条路线是 shielding、CBF 和安全过滤。MAMPS 通过 model predictive shielding 在执行前替换无法保证安全的动作；decentralized multiple CBF 工作把控制障碍函数作为局部安全 shield；Xiao et al. 的 MBDS 用可动态拆分、合并和重计算的分布式 shield 降低多智能体 LTL shielding 的保守性与合成开销；Bianchi et al. 则在 factored multi-agent MDP 中研究可扩展的 safe policy improvement。这类方法的共同优点是安全约束更接近“执行层硬约束”，但它们通常需要模型、局部邻居状态、因式化结构或可靠的 shield 监控信息。对于 SafeComm-MARL，这些工作提示了一个可行方向：在通信缺失时使用局部保守 shield 或因式化约束，但必须显式处理邻居状态缺失、延迟和过期信息。

去中心化与网络化安全 MARL 已有可核验证据，但假设与 SafeComm-MARL 并不完全相同。Lu et al. 的 Safe Dec-PG 不依赖中心控制器，通过 peer-to-peer 网络共享信息并处理耦合安全约束；不过其模型描述中仍假设状态和动作全局可观测，局部隐私主要体现在奖励和成本。Guo et al. 的电网控制论文显式考虑通信延迟，并用状态同步模块缓解实时决策中的延迟影响，但这是分布式能源 Volt-Var 控制的领域特定方案。它们说明“通信相关安全 MARL”并非空白，同时也说明通用的带宽、丢包、延迟与安全违反率之间的理论联系仍有空间。

基准与基础设施方面，Safety-Gymnasium 是可引用的 Safe RL/Safe MARL 基准来源，覆盖单智能体和多智能体安全任务，并提供一组 SafePO 算法实现。HAZARD Challenge 虽经核验为 ICLR 2024 benchmark，但它面向动态灾害环境中的 embodied decision making，不应被当作 Multi-Agent Safe RL benchmark 引用。SAUTE RL 提供几乎必然安全的状态增广思想，但已核验来源是单智能体 ICML 2022 论文；它可以启发 SafeComm-MARL 维护安全预算状态，却不能作为“已有多智能体 SAUTE 扩展”的证据。

## 4. 与 SafeComm-MARL 的关系

已核验文献对通信、集中训练和全局状态的假设并不一致，需要分层表述。

MACPO、MAPPO-Lagrangian、MAPPO 和 Safe multi-agent RL for multi-robot control 这组核心策略优化文献，主要采用 CTDE 或等价的集中训练设置；训练时常需要全局状态、联合动作、团队成本或集中式 critic，执行时则多为去中心化 actor。它们通常不要求执行时全通信，但训练阶段对全局信息的依赖很强。

Lu et al. 的 Safe Dec-PG 明确不是集中式训练：它使用 peer-to-peer 通信网络、无中心控制器，并允许奖励和约束局部/私有。但该工作仍在 D-CMDP 建模中假设状态和动作全局可观测，且没有把丢包、带宽预算或随机延迟作为核心安全变量。因此它是 SafeComm-MARL 的重要相关工作，而不是已经解决通信受限安全保证的终点。

CBF、MAMPS、MBDS 和 scalable safe policy improvement 更接近执行层安全机制。它们不一定假设全局全通信，但通常需要局部邻居状态、动态模型、shield 合成模型、因式化转移/价值函数或可靠监控信息。对 SafeComm-MARL 来说，关键问题不是“执行时有没有通信”这么简单，而是邻居信息缺失或延迟时，安全过滤器如何保持可行且不过度保守。

Safety-Gymnasium 和 AIJ Safe MARL benchmark 可作为实验平台来源，但默认仿真环境往往能提供集中训练所需的全局状态或成本信号；要用于 SafeComm-MARL，需要额外加入通信图、带宽、丢包、延迟和信息过期机制。Guo et al. 的通信延迟电网论文说明领域应用中已经有人显式处理延迟，但它的同步机制和安全投影依赖电力系统结构，不能直接泛化为通用 Safe MARL 理论。

因此，SafeComm-MARL 的合理定位是：在已存在的 constrained policy optimization、decentralized safe PG、shield/CBF 和通信延迟应用之间，系统研究受限/不可靠通信如何影响约束价值估计、执行层安全过滤和约束违反概率。应避免过度声称“没有任何工作考虑通信”，但可以明确指出：通用 Safe MARL 中同时覆盖通信预算、丢包/延迟、局部观测安全估计和可验证约束保证的工作仍然不足。

## 5. 可引用参考文献

1. Eitan Altman. "Constrained Markov Decision Processes". Chapman & Hall/CRC, 1999；CRC Press/Routledge eBook record, 2021. DOI: 10.1201/9781315140223。来源链接：https://doi.org/10.1201/9781315140223；https://openalex.org/W1518931405
2. Joshua Achiam, David Held, Aviv Tamar, Pieter Abbeel. "Constrained Policy Optimization". ICML, PMLR 70:22-31, 2017. 来源链接：https://proceedings.mlr.press/v70/achiam17a.html
3. Shai Shalev-Shwartz, Shaked Shammah, Amnon Shashua. "Safe, Multi-Agent, Reinforcement Learning for Autonomous Driving". arXiv:1610.03295, 2016. 来源链接：https://arxiv.org/abs/1610.03295
4. Chao Yu, Akash Velu, Eugene Vinitsky, Jiaxuan Gao, Yu Wang, Alexandre Bayen, Yi Wu. "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games". NeurIPS 2022 Datasets and Benchmarks, 2022. 来源链接：https://papers.nips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html；https://arxiv.org/abs/2103.01955
5. Shangding Gu, Jakub Grudzien Kuba, Munning Wen, Ruiqing Chen, Ziyan Wang, Zheng Tian, Jun Wang, Alois Knoll, Yaodong Yang. "Multi-Agent Constrained Policy Optimisation". arXiv:2110.02793, 2021. 来源链接：https://arxiv.org/abs/2110.02793
6. Shangding Gu, Jakub Grudzien Kuba, Yuanpei Chen, Yali Du, Long Yang, Alois Knoll, Yaodong Yang. "Safe multi-agent reinforcement learning for multi-robot control". Artificial Intelligence, 319:103905, 2023. DOI: 10.1016/j.artint.2023.103905。来源链接：https://doi.org/10.1016/j.artint.2023.103905
7. Jiaming Ji, Borong Zhang, Jiayi Zhou, Xuehai Pan, Weidong Huang, Ruiyang Sun, Yiran Geng, Yifan Zhong, Josef Dai, Yaodong Yang. "Safety Gymnasium: A Unified Safe Reinforcement Learning Benchmark". NeurIPS 2023 Datasets and Benchmarks, 2023. 来源链接：https://papers.nips.cc/paper_files/paper/2023/hash/3c557a3d6a48cc99444f85e924c66753-Abstract-Datasets_and_Benchmarks.html
8. Aivar Sootla, Alexander I. Cowen-Rivers, Taher Jafferjee, Ziyan Wang, David H. Mguni, Jun Wang, Haitham Ammar. "Saute RL: Almost Surely Safe Reinforcement Learning Using State Augmentation". ICML, PMLR 162:20423-20443, 2022. 来源链接：https://proceedings.mlr.press/v162/sootla22a.html
9. Wenbo Zhang, Osbert Bastani, Vijay Kumar. "MAMPS: Safe Multi-Agent Reinforcement Learning via Model Predictive Shielding". arXiv:1910.12639, 2019. 来源链接：https://arxiv.org/abs/1910.12639
10. Zhiyuan Cai, Huanhui Cao, Wenjie Lu, Lin Zhang, Hao Xiong. "Safe Multi-Agent Reinforcement Learning through Decentralized Multiple Control Barrier Functions". arXiv:2103.12553, 2021. 来源链接：https://arxiv.org/abs/2103.12553
11. Songtao Lu, Kaiqing Zhang, Tianyi Chen, Tamer Başar, Lior Horesh. "Decentralized Policy Gradient Descent Ascent for Safe Multi-Agent Reinforcement Learning". AAAI, 35(10):8767-8775, 2021. DOI: 10.1609/aaai.v35i10.17062。来源链接：https://ojs.aaai.org/index.php/AAAI/article/view/17062
12. Wenli Xiao, Yiwei Lyu, John M. Dolan. "Model-based Dynamic Shielding for Safe and Efficient Multi-Agent Reinforcement Learning". AAMAS, 1587-1596, 2023. 来源链接：https://www.ifaamas.org/Proceedings/aamas2023/pdfs/p1587.pdf；https://arxiv.org/abs/2304.06281
13. Federico Bianchi, Edoardo Zorzi, Alberto Castellini, Thiago D. Simão, Matthijs T. J. Spaan, Alessandro Farinelli. "Scalable Safe Policy Improvement for Factored Multi-Agent MDPs". ICML, PMLR 235:3952-3973, 2024. 来源链接：https://proceedings.mlr.press/v235/bianchi24b.html
14. Guodong Guo, Mengfan Zhang, Yanfeng Gong, Qianwen Xu. "Safe multi-agent deep reinforcement learning for real-time decentralized control of inverter based renewable energy resources considering communication delay". Applied Energy, 349:121648, 2023. DOI: 10.1016/j.apenergy.2023.121648。来源链接：https://doi.org/10.1016/j.apenergy.2023.121648
15. Hannah Markgraf, Matthias Althoff. "Safe Multi-Agent Reinforcement Learning for Price-Based Demand Response". 2023 IEEE PES Innovative Smart Grid Technologies Europe (ISGT EUROPE), 2023. DOI: 10.1109/isgteurope56780.2023.10407281。来源链接：https://doi.org/10.1109/isgteurope56780.2023.10407281；https://ieeexplore.ieee.org/document/10407281；https://openalex.org/W4391341675
16. Shangding Gu, Long Yang, Yali Du, Guang Chen, Florian Walter, Jun Wang, Alois Knoll. "A Review of Safe Reinforcement Learning: Methods, Theory and Applications". arXiv:2205.10330, 2022. 来源链接：https://arxiv.org/abs/2205.10330

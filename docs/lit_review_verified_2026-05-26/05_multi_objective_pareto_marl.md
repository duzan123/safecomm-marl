# 方向5：Multi-objective / Pareto MARL

## 1. 核验结论摘要

核验日期：2026-05-26。旧报告共抽取 12 个论文式候选条目。本次定向核验结果为：Verified 1 条，Metadata mismatch 5 条，Weak match 2 条，Not found 2 条，Replaced 2 条；另新增 6 篇可防守的核心锚点文献，主要用于补足 CMDP、约束 RL、MORL 综述和 MOMARL 基准。

总体结论需要明显降级：旧报告中“多目标/Pareto 方法可作为 SafeComm 单约束 Lagrangian 的替代方案”的表述缺少证据支撑。经核验后的文献更支持以下保守定位：CMDP/Lagrangian 与 CPO 一类约束优化文献直接支持安全约束满足；MORL、Pareto front、PGMORL、MORL/D、PCN、MOMAland 主要支持权衡分析、策略族生成、偏好条件化和评估可视化，不能单独保证 $J_\mathrm{safe}\le d$。因此，多目标/Pareto 文献可用于解释 SafeComm-VoI / SafeComm-MARL 的通信预算、安全成本和任务收益之间的权衡，但不能作为替换 Lagrangian 的核心依据。

旧报告中仍可保留的部分是：将不同通信预算 $k$ 下的任务收益和安全成本画成 Pareto-style 曲线是合理的；使用 hypervolume、coverage set、偏好条件化策略作为分析工具也有文献基础。需要删除或更正的部分是：`arXiv:2204.06475`、Safety-Aware MORL ICRA 2023、Distributional Multi-Objective MARL AAMAS 2021、Constrained MDP multi-objective perspectives JMLR 2021、Multi-Agent Pareto Q-Learning 2003 JMLR、Hypernetworks for MORL 2022/2023 等未核验或元数据错误的断言。

## 2. 原文献真实性审计表

| 原报告条目 | 状态 | 核验结果 | 可靠来源 | 处理方式 |
|---|---|---|---|---|
| A Multi-Objective Deep Reinforcement Learning Framework | Metadata mismatch | 论文存在，作者基本正确，但旧报告年份/卷号写成 2019、EAAI 78；可靠记录为 Engineering Applications of Artificial Intelligence, 96, 103915, 2020。 | [arXiv:1803.02965](https://arxiv.org/abs/1803.02965)；[DOI:10.1016/j.engappai.2020.103915](https://doi.org/10.1016/j.engappai.2020.103915) | 使用修正后的元数据；只作为 MODRL 框架和标量化/非标量化工具参考。 |
| MORL/D: Multi-Objective Reinforcement Learning based on Decomposition | Metadata mismatch | 题名存在，但旧报告把作者和 venue 错归给 Xu 等、TNNLS。可核验版本为 Felten, Talbi, Danoy 的 OLA 2022；后续有 JAIR 2024 taxonomy/framework 论文。 | [Felten 个人页/OLA 2022](https://ffelten.github.io/publication/2022-07-MORLD)；[JAIR 2024 DOI:10.1613/jair.1.15702](https://doi.org/10.1613/jair.1.15702) | 使用修正后的 MORL/D 文献；不沿用旧作者和 TNNLS 说法。 |
| Pareto Multi-Task Learning | Verified | 题名、作者和 NeurIPS 2019 venue 与官方记录一致。该文是多任务学习，不是 RL/MARL 文献。 | [NeurIPS 2019](https://papers.nips.cc/paper_files/paper/2019/hash/685bfde03eb646c27ed565881917c71c-Abstract.html)；[arXiv:1912.12854](https://arxiv.org/abs/1912.12854) | 可用于“多目标梯度冲突/多任务 Pareto 解”间接类比；不能作为 MARL 安全约束证据。 |
| PGMORL: Prediction-Guided Multi-Objective Reinforcement Learning | Metadata mismatch | 论文存在，venue 为 ICML 2020 PMLR；旧报告作者名有多处错误。正确作者为 Jie Xu, Yunsheng Tian, Pingchuan Ma, Daniela Rus, Shinjiro Sueda, Wojciech Matusik。 | [PMLR v119](https://proceedings.mlr.press/v119/xu20h.html) | 使用修正后的元数据；用于 Pareto 前沿探索和 hypervolume-guided 策略发现。 |
| Multi-Objective Safe Reinforcement Learning arXiv:2204.06475 | Not found | `arXiv:2204.06475` 实际为天体物理论文 “The contribution of magnetized galactic outflows to extragalactic Faraday rotation”，不是 RL。未找到旧条目所述 2022 arXiv Safe RL 论文。 | [arXiv:2204.06475](https://arxiv.org/abs/2204.06475)；相邻真实工作：[Horie et al. 2019 DOI](https://doi.org/10.1007/s10015-019-00523-3) | 旧条目不得引用；安全多目标关系改由 Horie et al. 2019、Altman 1999、Achiam et al. 2017 支撑。 |
| Pareto Conditioned Networks | Metadata mismatch | 论文存在，作者正确，但旧报告 venue 写成 NeurIPS 2022；可靠记录为 AAMAS 2022。 | [AAMAS 2022 PDF](https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p1110.pdf)；[arXiv:2204.05036](https://arxiv.org/abs/2204.05036) | 使用修正后的元数据；用于偏好/目标条件化策略族，不作为约束满足保证。 |
| Safety-Aware Multi-Objective Reinforcement Learning | Not found | 未找到旧报告所述 “2023 ICRA、Gu/OmniSafe 团队” 论文。可核验的相近主题是 Horie et al. 2019 的 MOSafeRL，以及 CPO/CMDP 约束 RL 文献。 | 相邻真实工作：[Horie et al. 2019 DOI](https://doi.org/10.1007/s10015-019-00523-3)；[CPO PMLR](https://proceedings.mlr.press/v70/achiam17a.html) | 删除旧条目；若讨论安全约束，应引用 CMDP/CPO 或 Horie et al. 2019。 |
| Distributional Multi-Objective MARL | Weak match | 未核验到旧报告所称 “DMOMARL, AAMAS 2021, Röpke/Roijers/Nowé/Rădulescu” 条目。存在相近但不同的单智能体 distributional MORL/多目标策略优化工作。 | [Abdolmaleki et al. ICML 2020](https://proceedings.mlr.press/v119/abdolmaleki20a.html)；[DPMORL NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/hash/32285dd184dbfc33cb2d1f0db53c23c5-Abstract-Conference.html) | 旧条目不用于论证；distributional 方向只作为尾部风险/分布偏好间接扩展。 |
| MOMALAND: A Suite of Benchmarks for Multi-Objective Multi-Agent Decision Making | Metadata mismatch | MOMAland 存在，但旧报告题名、年份和 venue 不准。核验版本为 “MOMAland: A Set of Benchmarks for Multi-Objective Multi-Agent Reinforcement Learning”，arXiv 2024，OpenReview/JDMLR 2026。 | [OpenReview JDMLR 2026](https://openreview.net/forum?id=vzHLRK0sSp)；[arXiv:2407.16312](https://arxiv.org/abs/2407.16312)；[Farama repo](https://github.com/Farama-Foundation/momaland) | 使用修正后的元数据；可作为 MOMARL 基准和评估协议参考。 |
| Constrained MDP multi-objective perspectives | Replaced | 未找到旧报告所称 “2021 JMLR、Le Cleac'h/Schwager/Manchester” 综述。可靠替代为 Altman CMDP 专著和 Gattami et al. AISTATS/PMLR 2021。 | [Altman CMDP DOI:10.1201/9781315140223](https://doi.org/10.1201/9781315140223)；[Gattami et al. PMLR 2021](https://proceedings.mlr.press/v130/gattami21a.html) | 用替代文献支撑 CMDP 与 constrained/multi-objective MDP 关系。 |
| Multi-Agent Pareto Q-Learning | Weak match | 未找到旧报告所称 “2003 JMLR Multi-Agent Pareto Q-Learning”。相近工作包括 2005 general-sum games 的 Pareto-Q learning、2014 JMLR 单智能体 Pareto Q-learning，以及 2020 MOMADM 综述。 | [Springer CEEMAS 2005](https://link.springer.com/chapter/10.1007/11559221_64)；[JMLR 2014 Pareto Q-learning](https://jmlr.csail.mit.edu/papers/v15/vanmoffaert14a.html)；[JAAMAS 2020 DOI](https://doi.org/10.1007/s10458-019-09433-x) | 不保留旧元数据；若需要 Q-learning/Pareto 集，引用 Van Moffaert & Nowé 2014 或 Rădulescu et al. 2020 综述。 |
| Hypernetworks for Multi-Objective Reinforcement Learning | Replaced | 未找到旧报告所称 “2022 NeurIPS Workshop/2023 ICLR 投稿、Sirbu 等” 的 MORL 超网络论文。真实强相关替代是 Navon et al. ICLR 2021 的 Pareto HyperNetworks，但它是一般多目标优化/多任务学习，不是 RL。 | [OpenReview ICLR 2021](https://openreview.net/forum?id=NjF772F4ZZR)；[arXiv:2010.04104](https://arxiv.org/abs/2010.04104) | 只作为偏好到模型参数映射的间接方法参考；不作为 MORL/MARL 直接证据。 |

## 3. 经核验后的核心文献综述

### 3.1 从 CMDP 到多目标视角：直接支持“保留约束优化”，不支持纯 MORL 替代

Altman 的 CMDP 专著为“最大化任务收益，同时满足成本约束”提供了经典模型基础；Achiam et al. 的 CPO 则将这一思想推进到深度策略优化，并强调在训练过程中近似满足约束。这两类文献直接支撑 SafeComm-VoI / SafeComm-MARL 中用 Lagrangian 或 primal-dual 机制处理安全成本约束的选择。

Gattami et al. 2021 明确把 constrained MDP 与 multi-objective MDP 放在同一学习框架中讨论，并给出可收敛的 Q-learning 型求解。该类结果支持一个保守但重要的解释：单约束 CMDP 可被看作多目标问题的一个约束化版本，但“多目标”本身并不自动给出硬约束满足。对 SafeComm 来说，这意味着安全成本若是论文的核心约束，Lagrangian/CMDP 仍是主证据；Pareto 语言适合解释不同可行策略的权衡位置。

### 3.2 MORL/Pareto 前沿算法：支持权衡曲线和策略族，不提供安全约束保证

Roijers et al. 2013 的 JAIR 综述和 Hayes et al. 2022 的实践指南说明，多目标序贯决策常见输出是单策略、凸覆盖集或 Pareto front，具体取决于效用函数、scalarization 和决策者偏好是否已知。Nguyen et al. 2020 的 MODRL 框架、Van Moffaert & Nowé 2014 的 Pareto Q-learning、Felten et al. 的 MORL/D、Xu et al. 2020 的 PGMORL 都属于帮助构造或近似 Pareto 前沿/覆盖集的方法。

这些方法对 SafeComm 的价值主要是实验分析：固定算法框架后，扫通信预算 $k$、安全阈值 $d$ 或偏好权重，可得到任务收益、通信开销、安全成本之间的 Pareto-style 曲线；PGMORL 和 MORL/D 中使用的 hypervolume/coverage 语言可用于更系统地报告“权衡面覆盖质量”。但这些方法通常以偏好或 scalarization 为输入，输出策略族或近似前沿；它们不能替代 Lagrangian 来保证某个策略满足 $J_\mathrm{safe}\le d$。

### 3.3 偏好条件化与超网络：支持部署灵活性，证据强度为间接

PCN 通过把目标回报/期望折中条件输入网络来学习多种非支配策略，适合在部署后根据用户偏好选择策略。Navon et al. 的 Pareto HyperNetworks 则说明可用超网络把偏好向量映射为模型参数，但该文不是 RL/MARL 方法。因此，在 SafeComm 场景中，PCN/PHN 类文献只能间接支持一个扩展想法：把通信预算 $k$、安全阈值 $d$ 或偏好向量作为策略条件输入，训练一个 budget-conditioned 或 safety-conditioned policy。

这种扩展的目标是提升部署灵活性，而不是替换约束优化。若系统要求每个部署点都满足安全约束，仍需要 CMDP、CPO、Lagrangian、shielding 或后验可行性过滤来验证策略是否安全。

### 3.4 MOMARL 基准和多智能体分类：支持定位问题，不直接支持 SafeComm 算法正确性

Rădulescu et al. 2020 的 multi-objective multi-agent decision making 综述为 MOMARL/MOMADM 提供了 reward/utility 维度的分类，区分 team reward、individual reward、team utility、individual utility 等设置。MOMAland 进一步提供了标准化 multi-objective multi-agent reinforcement learning 环境和 PettingZoo 风格 API，可作为未来对比或迁移测试平台。

这些文献能帮助 SafeComm 说明自身属于“多智能体、团队协作、任务收益与安全/通信成本冲突”的问题族；但 MOMAland 是基准论文，Rădulescu et al. 是分类综述，它们不证明 SafeComm 的通信调度、VoI 估计或安全约束机制有效。若用于论文 related work，应写成“定位和评估参考”，而不是“算法替代方案”。

### 3.5 安全、多目标与分布式风险：目前只支持扩展视角

Horie et al. 2019 明确讨论 MORL 与 SafeRL 的关系，并提出 MOSafeRL 思路；它可作为“安全可被看作收益与风险/成功概率之间的多目标权衡”的早期证据。Abdolmaleki et al. 2020 和 Cai et al. 2023 的 distributional MORL 工作则提示：若 SafeComm 未来从期望碰撞成本扩展到 CVaR、尾部风险或多变量回报分布，多目标分布式建模会有价值。

但这些文献目前只能支持扩展讨论。旧报告中“Distributional Multi-Objective MARL 直接适合 UAV 安全尾部风险”的说法证据不足，因为本次未核验到旧条目所称的 DMOMARL AAMAS 2021。若 SafeComm 当前只使用期望安全成本约束，引用 CPO/CMDP 比引用 distributional MORL 更直接。

## 4. 与 SafeComm-VoI / SafeComm-MARL 的关系

| 证据方向 | 支持替换 Lagrangian 吗 | 对 SafeComm 的可用写法 | 证据强度 |
|---|---|---|---|
| CMDP、CPO、constrained MDP 学习 | 不支持替换；反而支持保留约束优化 | SafeComm 的安全成本应建模为 CMDP 约束，Lagrangian/primal-dual 是合理的求解机制。 | 强支持 |
| MORL/Pareto front/PGMORL/MORL/D | 不支持替换 | 用于分析不同 $k$、$d$ 或偏好权重下的任务收益-安全成本-通信开销权衡；可报告 Pareto curve、coverage、hypervolume。 | 间接支持 |
| PCN、Pareto HyperNetworks | 不支持替换 | 可作为未来扩展：把安全阈值或通信预算作为策略条件输入，生成可切换策略族。仍需外部约束验证。 | 间接支持 |
| MOMARL 综述、MOMAland | 不支持替换 | 用于说明 SafeComm 位于 MOMARL 与安全/通信约束交叉区；MOMAland 可作补充 benchmark 风格参考。 | 间接支持 |
| Horie MOSafeRL 与 distributional MORL | 不支持替换 | Horie 可用于说明 SafeRL 与 MORL 的概念关系；distributional MORL 只适合未来尾部风险/CVaR 扩展。 | 证据有限 |

建议在 SafeComm 论文中采用以下保守表述：多目标/Pareto 视角提供了分析通信预算、安全风险和任务性能权衡的语言；SafeComm 的核心约束满足仍应由 CMDP/Lagrangian 或 CPO 类约束 RL 文献支撑。本次核验未发现可直接替换 SafeComm 单约束 Lagrangian 的真实、成熟、可核验 MOMARL/Pareto 方法。

## 5. 可引用参考文献

1. Altman, E. (1999). *Constrained Markov Decision Processes*. Chapman & Hall/CRC. [DOI:10.1201/9781315140223](https://doi.org/10.1201/9781315140223)
2. Achiam, J., Held, D., Tamar, A., & Abbeel, P. (2017). Constrained Policy Optimization. ICML 2017, PMLR 70:22-31. [PMLR](https://proceedings.mlr.press/v70/achiam17a.html)
3. Gattami, A., Bai, Q., & Aggarwal, V. (2021). Reinforcement Learning for Constrained Markov Decision Processes. AISTATS 2021, PMLR 130:2656-2664. [PMLR](https://proceedings.mlr.press/v130/gattami21a.html)
4. Roijers, D. M., Vamplew, P., Whiteson, S., & Dazeley, R. (2013). A Survey of Multi-Objective Sequential Decision-Making. *Journal of Artificial Intelligence Research*, 48, 67-113. [JAIR](https://jair.org/index.php/jair/article/view/10836)；[DOI:10.1613/jair.3987](https://doi.org/10.1613/jair.3987)
5. Hayes, C. F., Rădulescu, R., Bargiacchi, E., et al. (2022). A practical guide to multi-objective reinforcement learning and planning. *Autonomous Agents and Multi-Agent Systems*, 36, 26. [Springer](https://link.springer.com/article/10.1007/s10458-022-09552-y)
6. Rădulescu, R., Mannion, P., Roijers, D. M., & Nowé, A. (2020). Multi-objective multi-agent decision making: a utility-based analysis and survey. *Autonomous Agents and Multi-Agent Systems*, 34, 10. [DOI:10.1007/s10458-019-09433-x](https://doi.org/10.1007/s10458-019-09433-x)
7. Nguyen, T. T., Nguyen, N. D., Vamplew, P., Nahavandi, S., Dazeley, R., & Lim, C. P. (2020). A Multi-Objective Deep Reinforcement Learning Framework. *Engineering Applications of Artificial Intelligence*, 96, 103915. [arXiv](https://arxiv.org/abs/1803.02965)；[DOI:10.1016/j.engappai.2020.103915](https://doi.org/10.1016/j.engappai.2020.103915)
8. Van Moffaert, K., & Nowé, A. (2014). Multi-Objective Reinforcement Learning using Sets of Pareto Dominating Policies. *Journal of Machine Learning Research*, 15, 3663-3692. [JMLR](https://jmlr.csail.mit.edu/papers/v15/vanmoffaert14a.html)
9. Felten, F., Talbi, E.-G., & Danoy, G. (2022). MORL/D: Multi-Objective Reinforcement Learning based on Decomposition. OLA 2022. [作者页](https://ffelten.github.io/publication/2022-07-MORLD)；扩展版：[JAIR 2024 DOI:10.1613/jair.1.15702](https://doi.org/10.1613/jair.1.15702)
10. Xu, J., Tian, Y., Ma, P., Rus, D., Sueda, S., & Matusik, W. (2020). Prediction-Guided Multi-Objective Reinforcement Learning for Continuous Robot Control. ICML 2020, PMLR 119:10607-10616. [PMLR](https://proceedings.mlr.press/v119/xu20h.html)
11. Lin, X., Zhen, H.-L., Li, Z., Zhang, Q.-F., & Kwong, S. (2019). Pareto Multi-Task Learning. NeurIPS 2019. [NeurIPS](https://papers.nips.cc/paper_files/paper/2019/hash/685bfde03eb646c27ed565881917c71c-Abstract.html)；[arXiv](https://arxiv.org/abs/1912.12854)
12. Reymond, M., Bargiacchi, E., & Nowé, A. (2022). Pareto Conditioned Networks. AAMAS 2022, 1110-1118. [AAMAS PDF](https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p1110.pdf)；[arXiv](https://arxiv.org/abs/2204.05036)
13. Horie, N., Matsui, T., Moriyama, K., Mutoh, A., & Inuzuka, N. (2019). Multi-objective safe reinforcement learning: the relationship between multi-objective reinforcement learning and safe reinforcement learning. *Artificial Life and Robotics*, 24, 352-359. [DOI:10.1007/s10015-019-00523-3](https://doi.org/10.1007/s10015-019-00523-3)
14. Felten, F., Ucak, U., Azmani, H., Peng, G., Röpke, W., Baier, H., Mannion, P., Roijers, D. M., Terry, J. K., Talbi, E.-G., Danoy, G., Nowé, A., & Rădulescu, R. (2026). MOMAland: A Set of Benchmarks for Multi-Objective Multi-Agent Reinforcement Learning. *Journal of Data-centric Machine Learning Research*. [OpenReview](https://openreview.net/forum?id=vzHLRK0sSp)；[arXiv:2407.16312](https://arxiv.org/abs/2407.16312)
15. Abdolmaleki, A., Huang, S. H., Hasenclever, L., et al. (2020). A distributional view on multi-objective policy optimization. ICML 2020, PMLR 119:11-22. [PMLR](https://proceedings.mlr.press/v119/abdolmaleki20a.html)
16. Cai, X.-Q., Zhang, P., Zhao, L., Bian, J., Sugiyama, M., & Llorens, A. (2023). Distributional Pareto-Optimal Multi-Objective Reinforcement Learning. NeurIPS 2023. [NeurIPS](https://papers.neurips.cc/paper_files/paper/2023/hash/32285dd184dbfc33cb2d1f0db53c23c5-Abstract-Conference.html)
17. Navon, A., Shamsian, A., Fetaya, E., & Chechik, G. (2021). Learning the Pareto Front with Hypernetworks. ICLR 2021. [OpenReview](https://openreview.net/forum?id=NjF772F4ZZR)；[arXiv](https://arxiv.org/abs/2010.04104)

# 方向2：通信受限 MARL

核验日期：2026-05-26

## 1. 核验结论摘要

- 原报告文献数量：按原“关键论文列表”统计为 15 篇；原报告另列 2 条安全/约束 MARL 边界工作，未计入方向2核心条目数。
- Verified 数量：5。
- Metadata mismatch 数量：7。
- Weak match 数量：0。
- Not found 数量：0。
- Replaced 或新增文献数量：7，其中 3 条替换不可验证或不可引用原条目，4 条作为经核验补充文献。
- 对原报告核心结论的影响：总体结论“通信受限 MARL 主要优化任务奖励、协作性能或通信效率，几乎不直接优化安全约束”基本成立。但原报告把“稀疏通信、低带宽、鲁棒通信”过快等同于“安全相关通信”，需要收窄表述：这些方法通常只处理通信预算、拓扑、信道或消息效率，不能推出对状态约束、碰撞避免、约束违反概率或 CBF/CMDP 安全指标有保证。另有若干条目存在标题、作者、venue 或年份错误，必须以审计表中的修正版为准。

## 2. 原文献真实性审计表

| 原条目 | 状态 | 核验结果 | 可靠来源 | 处理 |
|---|---|---|---|---|
| Learning Multiagent Communication with Backpropagation (CommNet) | Verified | 论文题名、作者、NIPS/NeurIPS 2016 元数据可核验。 | NeurIPS: https://papers.nips.cc/paper/6398-learning-multiagent-communication-with-backpropagation；arXiv: https://arxiv.org/abs/1605.07736 | 保留为基础可学习连续通信文献。 |
| Learning to Communicate with Deep Multi-Agent Reinforcement Learning (DIAL) | Verified | 论文题名、作者、NIPS/NeurIPS 2016 元数据可核验。 | NeurIPS: https://papers.neurips.cc/paper/6042-learning-to-communicate-with-deep-multi-agent-reinforcement-learning；arXiv: https://arxiv.org/abs/1605.06676 | 保留为可微通信与离散化执行参考。 |
| Multiagent Bidirectionally-Coordinated Nets for Learning to Play StarCraft Combat | Metadata mismatch | 真实论文为 arXiv 预印本，题名通常为 Multiagent Bidirectionally-Coordinated Nets: Emergence of Human-level Coordination in Learning to Play StarCraft Combat Games；未核实到原报告所称 ICML Workshop 正式记录。 | arXiv: https://arxiv.org/abs/1703.10069；DBLP: https://dblp.org/rec/journals/corr/PengYWYTLW17 | 降级为背景文献，仅用于说明隐式序列通信/协调，不作为正式会议核心证据。 |
| Learning Attentional Communication for Multi-Agent Cooperation (ATOC) | Verified | 论文题名、作者、NeurIPS 2018 元数据可核验。 | NeurIPS: https://papers.nips.cc/paper/7956-learning-attentional-communication-for-multi-agent-cooperation；PDF: https://proceedings.neurips.cc/paper/2018/file/6a8018b3a00b69c008601b8becae392b-Paper.pdf | 保留为注意力门控和稀疏通信代表。 |
| Learning to Schedule Communication in Multi-Agent Systems (SchedNet) | Metadata mismatch | 方法和作者真实，但正式题名为 Learning to Schedule Communication in Multi-agent Reinforcement Learning，venue 为 ICLR 2019。 | OpenReview: https://openreview.net/forum?id=SJxu5iR9KQ；PDF: https://openreview.net/pdf?id=HUAnBToP_a | 以 ICLR 2019 修正题名保留，是显式带宽/共享信道调度代表。 |
| Graph Convolutional Reinforcement Learning (DGN) | Verified | 论文题名、作者、ICLR 2020 元数据可核验。 | OpenReview: https://openreview.net/forum?id=HkxdQkSYDB；OpenReview 记录副本: https://openreview.net/forum?id=S8icDSeqfvy | 保留为图拓扑/邻域传播通信代表。 |
| Networked Multi-Agent Reinforcement Learning with Communication | Metadata mismatch | 未找到该精确题名的 ICML 2020 论文；可核验的对应理论线索是 Fully Decentralized Multi-Agent Reinforcement Learning with Networked Agents，ICML 2018。 | ICML 2018: https://icml.cc/virtual/2018/poster/2269；arXiv: https://arxiv.org/abs/1802.08757 | 以 Fully Decentralized Multi-Agent Reinforcement Learning with Networked Agents 替换原题名，用作固定/时变网络通信理论参考。 |
| TarMAC: Targeted Multi-Agent Communication | Metadata mismatch | 题名、ICML 2019 可核验，但原报告作者列表错误；真实作者为 Abhishek Das、Theophile Gervet、Joshua Romoff、Dhruv Batra、Devi Parikh、Mike Rabbat、Joelle Pineau。 | PMLR: https://proceedings.mlr.press/v97/das19a.html；arXiv: https://arxiv.org/abs/1810.11187 | 修正作者后保留为定向通信代表。 |
| Learning Individually Inferred Communication for Multi-Agent Cooperation (I2C) | Verified | 论文题名、作者、NeurIPS 2020 元数据可核验。 | NeurIPS: https://papers.nips.cc/paper_files/paper/2020/hash/fb2fcd534b0ff3bbed73cc51df620323-Abstract.html；arXiv: https://arxiv.org/abs/2006.06455 | 保留为因果/个体推断通信选择代表。 |
| NeurComm: Neighbor Routing and Message Passing for Robust Multi-Agent Communication | Replaced | 原题名不可核验；可核验替代论文为 Multi-agent Reinforcement Learning for Networked System Control，ICLR 2020，其中提出 NeurComm 通信协议，但原报告题名与年份不成立。 | OpenReview: https://openreview.net/forum?id=Syx7A3NFvH；arXiv: https://arxiv.org/abs/2004.01339；DBLP: https://dblp.org/rec/conf/iclr/ChuCK20 | 原题名不作为论据；替换为 Multi-agent Reinforcement Learning for Networked System Control。 |
| MASIA: Multi-Agent Shared Information Aggregation | Replaced | 原题名不可核验；可核验替代论文为 Efficient Multi-agent Communication via Self-supervised Information Aggregation，NeurIPS 2022，其方法缩写为 MASIA，但原报告 AAAI 2023、作者 Yun 等元数据不成立。 | NeurIPS: https://proceedings.neurips.cc/paper_files/paper/2022/hash/075b2875e2b671ddd74aeec0ac9f0357-Abstract-Conference.html；DBLP: https://dblp.org/rec/conf/nips/GuanCYWYZ022 | 原题名不作为论据；替换为 Efficient Multi-agent Communication via Self-supervised Information Aggregation。 |
| NDQ: Nearly Decomposable Value Function for Cooperation | Metadata mismatch | 方法 NDQ 真实，但正式题名为 Learning Nearly Decomposable Value Functions Via Communication Minimization，venue 为 ICLR 2020；原报告“NeurIPS Workshop / ICML 2020”不准确。 | OpenReview: https://openreview.net/forum?id=HJx-3grYDB；arXiv: https://arxiv.org/abs/1910.05366 | 修正题名和 venue 后保留。 |
| Multi-Agent Graph-Attention Communication (MAGIC) | Metadata mismatch | 论文真实，正式题名为 Multi-Agent Graph-Attention Communication and Teaming，AAMAS 2021；原报告写 AAAI 且首作者名误写为 Yifan Niu，真实为 Yaru Niu。 | AAMAS PDF: https://www.ifaamas.org/Proceedings/aamas2021/pdfs/p964.pdf；代码页: https://github.com/CORE-Robotics-Lab/MAGIC | 修正后保留为动态图注意力通信代表。 |
| DC2: Divide and Conquer Communication | Replaced | 未找到与该题名、NeurIPS 2023、所述复杂度主张相匹配的可靠论文。可核验的相近方向是 Learning Structured Communication for Multi-Agent Reinforcement Learning，AAMAS 2023，使用层次结构/分组通信。 | OpenReview: https://openreview.net/forum?id=BklWt24tvH；arXiv: https://arxiv.org/abs/2002.04235；AAMAS PDF: https://www.ifaamas.org/Proceedings/aamas2023/pdfs/p436.pdf | 原条目不作为论据；以 LSC 替换。 |
| Efficient Communication in Multi-Agent Reinforcement Learning via Variance Based Control (VBC) | Metadata mismatch | 论文真实，但 venue/year 为 NeurIPS 2019，不是原报告所称 2022 NeurIPS。 | NeurIPS: https://papers.neurips.cc/paper/8586-efficient-communication-in-multi-agent-reinforcement-learning-via-variance-based-control；arXiv: https://arxiv.org/abs/1909.02682 | 修正年份后保留为通信开销控制代表。 |

## 3. 经核验后的核心文献综述

早期可学习通信工作主要证明“通信内容可以作为策略的一部分端到端学习”。CommNet 采用连续消息和全连接聚合，DIAL/RIAL 将通信通道纳入深度 MARL 训练，并在执行时处理离散化或噪声信道。这一类工作奠定了可微通信范式，但默认通信可用性较强，主要目标是提升合作任务奖励，而不是满足显式通信预算或安全约束。

显式通信预算与调度方法把通信资源作为一等约束。SchedNet 直接建模共享通信介质下每步只有有限数量智能体可广播的场景，因此是“通信预算”最清楚的代表之一。VBC 通过控制消息方差减少通信开销，NDQ 用互信息和熵正则学习“必要时通信”的近似可分解价值函数，Sparse Discrete Communication Learning 用可变长离散消息和消息长度惩罚鼓励短消息。这些方法能够降低消息数量、长度或冗余度，但其优化信号仍主要来自任务回报和通信效率。

稀疏拓扑和目标选择方法强调“何时、向谁、传什么”。ATOC 使用注意力机制选择通信发起者和通信组；TarMAC 通过签名、键和值实现有目标的多轮通信；I2C 用因果影响估计推断需要通信的对象；IC3Net 通过门控学习何时通信，并扩展到合作、半合作和竞争场景。这些工作比全广播更接近真实受限通信，但稀疏化标准通常是任务相关性、收益或信息效率，不是安全风险。

图结构通信把通信限制映射为邻接关系或动态图。DGN 在多智能体环境中使用图卷积传播邻域信息；MAGIC 用图注意力调度“何时”和“向谁”通信；Multi-agent Reinforcement Learning for Networked System Control 在预设网络邻居之间传递消息；Zhang 等的 fully decentralized networked MARL 研究则从理论上处理稀疏、可能时变的网络连接。LSC 和 Efficient Multi-agent Communication via Self-supervised Information Aggregation 进一步从层次结构和消息聚合角度改善大规模通信效率。它们对 SafeComm-MARL 很有价值，因为安全约束常天然依赖局部邻域、拓扑连通性和风险传播，但原论文一般没有把通信图学习与状态安全约束联合优化。

安全 MARL 的核验文献显示，安全约束和通信约束目前多是分离处理。MACPO/MAPPO-Lagrangian 把多智能体安全学习建模为 constrained Markov game，关注奖励改进和约束满足；Safety Gymnasium 提供含单智能体与多智能体安全任务的基准。这些工作有助于定义约束代价、可行性和评估协议，但并不解决受限通信下安全信息如何调度、压缩或可靠传递的问题。

## 4. 与 SafeComm-MARL 的关系

全连接可微通信类方法，代表为 CommNet 和 DIAL，优化的是任务奖励以及通信协议学习能力；它们通常不优化通信预算，也不优化安全约束。对 SafeComm-MARL 的作用主要是提供端到端消息学习机制，不能直接证明受限通信下的安全性。

带宽调度、消息压缩和事件/稀疏触发类方法，代表为 SchedNet、VBC、NDQ、Efficient Multi-agent Communication via Self-supervised Information Aggregation、Sparse Discrete Communication Learning，明确或间接优化通信预算、消息长度、消息冗余或聚合效率。它们适合作为 SafeComm-MARL 的通信资源层，但不能把“少通信”直接解释为“安全通信”：如果安全关键消息被压缩、丢弃或低优先级调度，系统安全反而可能恶化。

目标选择和动态图通信类方法，代表为 ATOC、TarMAC、I2C、IC3Net、MAGIC、DGN、LSC，优化的是通信对象、通信拓扑或任务相关信息流。它们可改造成安全感知通信策略，例如把约束余量、碰撞风险、CBF 值、代价 critic 或邻域可达性作为注意力/调度输入。但原始方法大多没有以约束违反率、状态约束或安全可行域为目标，也没有提供安全保证。

网络化 MARL 理论类方法，代表为 Fully Decentralized Multi-Agent Reinforcement Learning with Networked Agents 和 Multi-agent Reinforcement Learning for Networked System Control，显式处理固定或时变网络邻居通信，部分工作给出收敛或稳定训练分析。它们对 SafeComm-MARL 的理论建模有帮助，但安全约束仍需另行引入，例如 CMDP/Constrained Markov Game、Lagrangian、CPO/MACPO 或 CBF 约束。

安全 MARL 类方法，代表为 MACPO 和 Safety Gymnasium，优化或评估安全约束，但通常不以通信预算为核心变量。SafeComm-MARL 的空白正在于把这两条线合并：在受限通信图、有限带宽、丢包/延迟或消息压缩条件下，同时优化任务奖励、通信预算和安全约束，并明确安全关键消息的优先级与可验证约束效果。

## 5. 可引用参考文献

- Sainbayar Sukhbaatar, Arthur Szlam, Rob Fergus. Learning Multiagent Communication with Backpropagation. Advances in Neural Information Processing Systems 29 (NIPS 2016), 2016. 来源链接：https://papers.nips.cc/paper/6398-learning-multiagent-communication-with-backpropagation；https://arxiv.org/abs/1605.07736
- Jakob N. Foerster, Ioannis Alexandros Assael, Nando de Freitas, Shimon Whiteson. Learning to Communicate with Deep Multi-Agent Reinforcement Learning. Advances in Neural Information Processing Systems 29 (NIPS 2016), 2016. 来源链接：https://papers.neurips.cc/paper/6042-learning-to-communicate-with-deep-multi-agent-reinforcement-learning；https://arxiv.org/abs/1605.06676
- Peng Peng, Ying Wen, Yaodong Yang, Quan Yuan, Zhenkun Tang, Haitao Long, Jun Wang. Multiagent Bidirectionally-Coordinated Nets: Emergence of Human-level Coordination in Learning to Play StarCraft Combat Games. arXiv, 2017. 来源链接：https://arxiv.org/abs/1703.10069
- Jiechuan Jiang, Zongqing Lu. Learning Attentional Communication for Multi-Agent Cooperation. Advances in Neural Information Processing Systems 31 (NeurIPS 2018), 2018. 来源链接：https://papers.nips.cc/paper/7956-learning-attentional-communication-for-multi-agent-cooperation；https://arxiv.org/abs/1805.07733
- Daewoo Kim, Sangwoo Moon, David Hostallero, Wan Ju Kang, Taeyoung Lee, Kyunghwan Son, Yung Yi. Learning to Schedule Communication in Multi-agent Reinforcement Learning. International Conference on Learning Representations (ICLR), 2019. 来源链接：https://openreview.net/forum?id=SJxu5iR9KQ；https://arxiv.org/abs/1902.01554
- Jiechuan Jiang, Chen Dun, Tiejun Huang, Zongqing Lu. Graph Convolutional Reinforcement Learning. International Conference on Learning Representations (ICLR), 2020. 来源链接：https://openreview.net/forum?id=HkxdQkSYDB
- Kaiqing Zhang, Zhuoran Yang, Han Liu, Tong Zhang, Tamer Başar. Fully Decentralized Multi-Agent Reinforcement Learning with Networked Agents. International Conference on Machine Learning (ICML), 2018. 来源链接：https://icml.cc/virtual/2018/poster/2269；https://arxiv.org/abs/1802.08757
- Abhishek Das, Theophile Gervet, Joshua Romoff, Dhruv Batra, Devi Parikh, Mike Rabbat, Joelle Pineau. TarMAC: Targeted Multi-Agent Communication. Proceedings of the 36th International Conference on Machine Learning, PMLR 97:1538-1546, 2019. 来源链接：https://proceedings.mlr.press/v97/das19a.html；https://arxiv.org/abs/1810.11187
- Ziluo Ding, Tiejun Huang, Zongqing Lu. Learning Individually Inferred Communication for Multi-Agent Cooperation. Advances in Neural Information Processing Systems 33 (NeurIPS 2020), 2020. 来源链接：https://papers.nips.cc/paper_files/paper/2020/hash/fb2fcd534b0ff3bbed73cc51df620323-Abstract.html；https://arxiv.org/abs/2006.06455
- Tianshu Chu, Sandeep Chinchali, Sachin Katti. Multi-agent Reinforcement Learning for Networked System Control. International Conference on Learning Representations (ICLR), 2020. 来源链接：https://openreview.net/forum?id=Syx7A3NFvH；https://arxiv.org/abs/2004.01339
- Cong Guan, Feng Chen, Lei Yuan, Chenghe Wang, Hao Yin, Zongzhang Zhang, Yang Yu. Efficient Multi-agent Communication via Self-supervised Information Aggregation. Advances in Neural Information Processing Systems 35 (NeurIPS 2022), 2022. 来源链接：https://proceedings.neurips.cc/paper_files/paper/2022/hash/075b2875e2b671ddd74aeec0ac9f0357-Abstract-Conference.html
- Tonghan Wang, Jianhao Wang, Chongyi Zheng, Chongjie Zhang. Learning Nearly Decomposable Value Functions Via Communication Minimization. International Conference on Learning Representations (ICLR), 2020. 来源链接：https://openreview.net/forum?id=HJx-3grYDB；https://arxiv.org/abs/1910.05366
- Yaru Niu, Rohan Paleja, Matthew Gombolay. Multi-Agent Graph-Attention Communication and Teaming. International Conference on Autonomous Agents and Multiagent Systems (AAMAS), 2021. 来源链接：https://www.ifaamas.org/Proceedings/aamas2021/pdfs/p964.pdf
- Junjie Sheng, Xiangfeng Wang, Bo Jin, Wenhao Li, Jun Wang, Junchi Yan, Tsung-Hui Chang, Hongyuan Zha. Learning Structured Communication for Multi-Agent Reinforcement Learning. International Conference on Autonomous Agents and Multiagent Systems (AAMAS), 2023. 来源链接：https://www.ifaamas.org/Proceedings/aamas2023/pdfs/p436.pdf；https://arxiv.org/abs/2002.04235
- Sai Qian Zhang, Qi Zhang, Jieyu Lin. Efficient Communication in Multi-Agent Reinforcement Learning via Variance Based Control. Advances in Neural Information Processing Systems 32 (NeurIPS 2019), 2019. 来源链接：https://papers.neurips.cc/paper/8586-efficient-communication-in-multi-agent-reinforcement-learning-via-variance-based-control；https://arxiv.org/abs/1909.02682
- Amanpreet Singh, Tushar Jain, Sainbayar Sukhbaatar. Learning when to Communicate at Scale in Multiagent Cooperative and Competitive Tasks. International Conference on Learning Representations (ICLR), 2019. 来源链接：https://openreview.net/forum?id=rye7knCqK7；https://arxiv.org/abs/1812.09755
- Benjamin Freed, Rohan James, Guillaume Sartoretti, Howie Choset. Sparse Discrete Communication Learning for Multi-Agent Cooperation Through Backpropagation. IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2020. 来源链接：https://publications.ri.cmu.edu/sparse-discrete-communication-learning-for-multi-agent-cooperation-through-backpropagation；DOI：https://doi.org/10.1109/IROS45743.2020.9341079
- Benjamin Freed, Guillaume Sartoretti, Jiaheng Hu, Howie Choset. Communication Learning via Backpropagation in Discrete Channels with Unknown Noise. Proceedings of the AAAI Conference on Artificial Intelligence, 34(05):7160-7168, 2020. 来源链接：https://ojs.aaai.org/index.php/AAAI/article/view/6205；DOI：https://doi.org/10.1609/aaai.v34i05.6205
- Shangding Gu, Jakub Grudzien Kuba, Munning Wen, Ruiqing Chen, Ziyan Wang, Zheng Tian, Jun Wang, Alois Knoll, Yaodong Yang. Multi-Agent Constrained Policy Optimisation. arXiv, 2021. 来源链接：https://arxiv.org/abs/2110.02793
- Jiaming Ji, Borong Zhang, Jiayi Zhou, Xuehai Pan, Weidong Huang, Ruiyang Sun, Yiran Geng, Yifan Zhong, Juntao Dai, Yaodong Yang. Safety-Gymnasium: A Unified Safe Reinforcement Learning Benchmark. Advances in Neural Information Processing Systems 36 (NeurIPS 2023) Datasets and Benchmarks Track, 2023. 来源链接：https://papers.nips.cc/paper_files/paper/2023/hash/3c557a3d6a48cc99444f85e924c66753-Abstract-Datasets_and_Benchmarks.html；https://arxiv.org/abs/2310.12567

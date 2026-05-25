# 方向2：通信受限 MARL

## 1. 核验结论摘要

本方向以旧报告中的 15 个通信受限 MARL 候选条目为审计对象，旧报告只作为线索，不作为可信来源。核验结果为：Verified 6 篇，Metadata mismatch 8 篇，Weak match 0 篇，Not found 1 篇，Replaced 0 篇；另新增 3 篇可用于边界定位或补强论证的真实文献：Learning to Share in Networked MARL、MACPO、Safety-Gymnasium。

可以保留的结论是：通信受限 MARL 确实围绕全连接通信、带宽/共享信道、稀疏对象选择、动态图拓扑、邻域网络和消息压缩等问题发展出一条清晰技术线。需要降级的结论是：旧报告中“现有通信受限 MARL 从未考虑安全约束”应改为“本次核验覆盖的核心通信 MARL 文献未发现将通信约束与显式安全约束联合建模”。需要删除或修正的内容包括：DC2 条目未找到可靠来源；Networked MARL、NeurComm、MASIA、NDQ、MAGIC、VBC、TarMAC、BiCNet 的作者、标题、年份或 venue 存在不同程度错误。

## 2. 原文献真实性审计表

| 原报告条目 | 状态 | 核验结果 | 可靠来源 | 处理方式 |
|---|---|---|---|---|
| CommNet: Learning Multiagent Communication with Backpropagation | Verified | 标题、作者、NeurIPS/NIPS 2016 基本一致；Arthur Szlam 在官方页小写显示，不影响识别。 | [NeurIPS Proceedings](https://proceedings.neurips.cc/paper/6398-learning-multiagent-communication-with-backpropagation) | 保留，作为可学习连续通信的基础文献。 |
| DIAL: Learning to Communicate with Deep Multi-Agent Reinforcement Learning | Verified | 标题、作者、NeurIPS/NIPS 2016 一致；官方作者为 Jakob Foerster, Ioannis Alexandros Assael, Nando de Freitas, Shimon Whiteson。 | [NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2016/hash/c7635bfd99248a2cdef8249ef7bfbef4-Abstract.html) | 保留，作为可微/带噪通信通道的基础文献。 |
| BiCNet: Multiagent Bidirectionally-Coordinated Nets for Learning to Play StarCraft Combat | Metadata mismatch | 论文存在，但可靠来源显示为 arXiv/CoRR 2017，标题后续为 “Emergence of Human-level Coordination...”，未核验到旧报告所称 ICML Workshop 正式来源。 | [arXiv:1703.10069](https://arxiv.org/abs/1703.10069), [dblp](https://dblp.org/rec/journals/corr/PengYWYTLW17) | 使用修正后的 arXiv 元数据；不作为正式会议论文引用。 |
| ATOC: Learning Attentional Communication for Multi-Agent Cooperation | Verified | 标题、作者 Jiang & Lu、NeurIPS 2018 一致。 | [NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2018/hash/6a8018b3a00b69c008601b8becae392b-Abstract.html) | 保留，作为注意力式动态通信门控文献。 |
| SchedNet: Learning to Schedule Communication in Multi-Agent Systems | Verified | 旧报告标题略写为 “Systems”，可靠来源题名为 “Learning to Schedule Communication in Multi-agent Reinforcement Learning”，作者和 ICLR 2019 一致。 | [OpenReview](https://openreview.net/forum?id=SJxu5iR9KQ), [ICLR 2019 poster](https://iclr.cc/virtual/2019/poster/931) | 保留，引用时使用官方完整标题。 |
| DGN: Graph Convolutional Reinforcement Learning | Verified | 标题、作者、ICLR 2020 一致。 | [OpenReview](https://openreview.net/forum?id=HkxdQkSYDB) | 保留，作为图结构邻域信息传播文献。 |
| Networked Multi-Agent Reinforcement Learning with Communication | Metadata mismatch | 未找到旧标题与旧 venue 的精确匹配；旧作者组合可靠匹配到 ICML 2018 论文 “Fully Decentralized Multi-Agent Reinforcement Learning with Networked Agents”。 | [PMLR](https://proceedings.mlr.press/v80/zhang18n.html) | 改用修正后的 Zhang et al. ICML 2018 文献，不使用旧标题。 |
| TarMAC: Targeted Multi-Agent Communication | Metadata mismatch | 标题和 ICML 2019 正确，但旧作者列有误；PMLR 作者为 Abhishek Das, Théophile Gervet, Joshua Romoff, Dhruv Batra, Devi Parikh, Mike Rabbat, Joelle Pineau。 | [PMLR](https://proceedings.mlr.press/v97/das19a.html) | 使用修正后的作者和 PMLR 元数据。 |
| I2C: Learning Individually Inferred Communication for Multi-Agent Cooperation | Verified | 标题、作者 Ding/Huang/Lu、NeurIPS 2020 一致。 | [NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2020/hash/fb2fcd534b0ff3bbed73cc51df620323-Abstract.html) | 保留，作为因果推断式点对点通信选择文献。 |
| NeurComm: Neighbor Routing and Message Passing for Robust Multi-Agent Communication | Metadata mismatch | 未找到旧标题；可靠来源显示 NeurComm 是 Chu/Chinchali/Katti 在 ICLR 2020 “Multi-agent Reinforcement Learning for Networked System Control” 中提出的可微通信协议。 | [OpenReview](https://openreview.net/forum?id=Syx7A3NFvH), [dblp](https://dblp.org/rec/conf/iclr/ChuCK20) | 使用修正后的论文标题、年份和 venue；删除“2022 ICLR/Neighbor Routing”表述。 |
| MASIA: Multi-Agent Shared Information Aggregation | Metadata mismatch | 旧作者 Yun、AAAI 2023 未核验到；可靠来源为 NeurIPS 2022 “Efficient Multi-agent Communication via Self-supervised Information Aggregation”，MASIA 指 Multi-Agent communication via Self-supervised Information Aggregation。 | [NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/075b2875e2b671ddd74aeec0ac9f0357-Abstract-Conference.html), [OpenReview](https://openreview.net/forum?id=n4wnZAdBavx) | 使用修正后的 Guan et al. NeurIPS 2022 文献。 |
| NDQ: Nearly Decomposable Value Function for Cooperation | Metadata mismatch | 旧标题、venue 有误；可靠来源为 ICLR 2020 “Learning Nearly Decomposable Value Functions Via Communication Minimization”。 | [OpenReview](https://openreview.net/forum?id=HJx-3grYDB) | 使用修正后的 ICLR 2020 元数据。 |
| MAGIC: Multi-Agent Graph-Attention Communication | Metadata mismatch | 论文存在于 AAMAS 2021；旧报告将第一作者写作 Yifan Niu，可靠来源为 Yaru Niu、Rohan Paleja、Matthew Gombolay。 | [IFAAMAS/AAMAS PDF](https://www.ifaamas.org/Proceedings/aamas2021/pdfs/p964.pdf), [AAMAS 2021 contents](https://www.ifaamas.org/Proceedings/aamas2021/forms/contents.htm) | 使用修正后的作者和完整标题 “Multi-Agent Graph-Attention Communication and Teaming”。 |
| DC2: Divide and Conquer Communication | Not found | 用精确标题、DC2+MARL、DC2+SMAC、NeurIPS 2023 等组合检索，未找到可靠来源能支撑旧报告条目。 | 未找到 DOI、arXiv、NeurIPS/OpenReview/PMLR/ACM/IEEE/Springer 可靠匹配 | 不进入参考文献，不支撑任何结论。 |
| VBC: Efficient Communication in Multi-Agent Reinforcement Learning via Variance Based Control | Metadata mismatch | 论文存在，但旧报告年份写作 2022 有误；可靠来源显示为 NeurIPS 2019，作者 Sai Qian Zhang, Qi Zhang, Jieyu Lin。 | [NeurIPS Proceedings](https://papers.neurips.cc/paper_files/paper/2019/hash/14cfdb59b5bda1fc245aadae15b1984a-Abstract.html), [arXiv:1909.02682](https://arxiv.org/abs/1909.02682) | 使用修正后的 NeurIPS 2019 元数据。 |

## 3. 经核验后的核心文献综述

早期可学习通信 MARL 主要验证“通信内容可以端到端学习”，但不直接建模现实通信约束。CommNet 让多智能体通过连续向量通信并联合学习策略，DIAL/RIAL 进一步把通信动作纳入深度多智能体强化学习，其中 DIAL 在训练时允许梯度穿过带噪通信通道，在执行时再离散化消息。这两类工作为后续通信学习提供了基础范式，但默认通信可用性较强，未处理带宽竞争、链路拓扑、安全成本或状态约束。

显式通信约束的第一条主线是带宽和共享媒介约束。SchedNet 明确假设通信带宽有限且智能体共享通信媒介，每步只有受限数量智能体可以广播消息，因此学习“谁获得广播权、如何编码消息、如何利用消息”。VBC 从通信效率出发，用消息方差控制训练和执行中的消息交换，官方摘要报告通信开销降低而性能保持或提升。NDQ 将通信最小化和价值函数分解结合，使智能体大部分时间独立行动、必要时通信；其目标是降低通信频率而不牺牲合作性能。MASIA 不直接做硬调度，而是学习自监督消息聚合，把多条队友消息压缩为更相关的紧凑表征。这些工作真实处理了通信成本或通信冗余，但其约束是通信预算、调度或信息效率，不是安全成本、碰撞风险、CBF 约束或 CMDP 约束。

第二条主线是“何时与谁通信”的稀疏拓扑学习。ATOC 用注意力机制学习是否发起通信，解决大规模全广播信息混杂问题。TarMAC 学习消息内容与目标接收者，实现有目标的多轮通信。I2C 用因果推断思想估计一个智能体对另一个智能体行动的影响，从而学习点对点通信先验。MAGIC 进一步把“何时通信、向谁通信、如何处理消息”统一到动态图注意力通信协议中，使用动态图和 GAT 处理通信信号。这些方法提供了 SafeComm-VoI 可借鉴的通信价值估计机制，但它们的“价值”仍主要由任务奖励或协作效率定义，而不是由安全裕度、风险传播或约束违反概率定义。

第三条主线是固定或时变网络拓扑下的邻域通信。Zhang et al. 的 ICML 2018 工作从理论上研究 networked agents：智能体通过可能稀疏和时变的通信网络接收邻居消息，并给出去中心化 actor-critic 的收敛保证。Chu et al. 的 ICLR 2020 工作把 networked MARL 用于交通信号控制和协同自适应巡航控制，并提出 NeurComm 可微通信协议。DGN 使用图卷积捕获多智能体环境中动态变化的邻居关系。LToS 在 NeurIPS 2022 中研究 networked MARL 的受限邻居交互，并学习邻居间奖励共享以改善全局协作。这类文献对 SafeComm-MARL 中“受限通信图上的策略优化”最有直接启发，但仍没有把图通信与形式化安全约束联合起来。

边界上，安全 MARL 文献确实处理约束，但通常不是通信受限通信学习问题。MACPO 将安全 MARL 表述为 constrained Markov game，并提出 MACPO/MAPPO-Lagrangian；Safety-Gymnasium 提供包含单智能体和多智能体安全任务的 SafeRL benchmark。这些工作可支撑 SafeComm 的安全约束侧背景，但不能替代通信受限 MARL 证据：它们并没有解决带宽、共享媒介、动态通信拓扑或消息价值调度问题。

因此，本次核验后的保守结论是：真实通信 MARL 工作中，SchedNet、VBC、NDQ、MASIA、ATOC、TarMAC、I2C、MAGIC、DGN、NeurComm、Zhang et al. 2018 和 LToS 都处理了某种通信约束或通信效率问题；但在这些核心文献中，未发现将通信预算/拓扑/调度与显式安全约束联合建模的工作。该结论是本次核验范围内的证据结论，不应写成全领域绝对断言。

## 4. 与 SafeComm-VoI / SafeComm-MARL 的关系

对 SafeComm-VoI 的强支持来自 SchedNet、I2C、TarMAC、MAGIC、VBC 和 NDQ：这些工作证明通信并非必须全广播，可以学习调度、目标接收者、点对点因果重要性、动态图注意力或事件式压缩。但它们的通信价值基本由任务回报、信息冗余或协作效率诱导，缺少“安全价值”的显式定义。SafeComm-VoI 可以把 VoI 从“提升任务回报的信息价值”扩展为“降低约束违反风险的信息价值”，例如用安全裕度、约束松弛量、预测碰撞风险或 CBF 余量参与通信优先级计算。

对 SafeComm-MARL 的强支持来自 networked MARL 与图通信文献。Zhang et al. 2018、DGN、NeurComm、LToS 和 MAGIC 都说明受限邻域图或动态图通信是可学习 MARL 的合理建模对象。SafeComm-MARL 可以沿用这些工作中的邻接图、动态图注意力和邻域消息传递结构，但需要新增安全约束优化层：通信图不仅传任务相关信息，还要优先传播可能影响约束可行性的局部状态。

证据不足之处也必须明确：本次核验没有找到 DC2 旧条目的可靠来源；也没有在上述通信 MARL 核心文献中发现完整的“通信受限 + 安全约束 + 多智能体强化学习”闭环方案。因此，论文写作中应避免宣称“首个”或“完全没有任何工作”，更稳妥的表述是：“在本次核验覆盖的代表性通信受限 MARL 和安全 MARL 文献中，尚未发现同时学习安全感知通信调度并处理显式安全约束的统一框架。”

## 5. 可引用参考文献

1. Sainbayar Sukhbaatar, Arthur Szlam, Rob Fergus. 2016. Learning Multiagent Communication with Backpropagation. NeurIPS/NIPS 2016. [https://proceedings.neurips.cc/paper/6398-learning-multiagent-communication-with-backpropagation](https://proceedings.neurips.cc/paper/6398-learning-multiagent-communication-with-backpropagation)
2. Jakob Foerster, Ioannis Alexandros Assael, Nando de Freitas, Shimon Whiteson. 2016. Learning to Communicate with Deep Multi-Agent Reinforcement Learning. NeurIPS/NIPS 2016. [https://proceedings.neurips.cc/paper/2016/hash/c7635bfd99248a2cdef8249ef7bfbef4-Abstract.html](https://proceedings.neurips.cc/paper/2016/hash/c7635bfd99248a2cdef8249ef7bfbef4-Abstract.html)
3. Peng Peng, Ying Wen, Yaodong Yang, Quan Yuan, Zhenkun Tang, Haitao Long, Jun Wang. 2017. Multiagent Bidirectionally-Coordinated Nets: Emergence of Human-level Coordination in Learning to Play StarCraft Combat Games. arXiv:1703.10069. [https://arxiv.org/abs/1703.10069](https://arxiv.org/abs/1703.10069)
4. Jiechuan Jiang, Zongqing Lu. 2018. Learning Attentional Communication for Multi-Agent Cooperation. NeurIPS 2018. [https://proceedings.neurips.cc/paper/2018/hash/6a8018b3a00b69c008601b8becae392b-Abstract.html](https://proceedings.neurips.cc/paper/2018/hash/6a8018b3a00b69c008601b8becae392b-Abstract.html)
5. Daewoo Kim, Sangwoo Moon, David Hostallero, Wan Ju Kang, Taeyoung Lee, Kyunghwan Son, Yung Yi. 2019. Learning to Schedule Communication in Multi-agent Reinforcement Learning. ICLR 2019. [https://openreview.net/forum?id=SJxu5iR9KQ](https://openreview.net/forum?id=SJxu5iR9KQ)
6. Abhishek Das, Théophile Gervet, Joshua Romoff, Dhruv Batra, Devi Parikh, Mike Rabbat, Joelle Pineau. 2019. TarMAC: Targeted Multi-Agent Communication. ICML 2019, PMLR 97:1538-1546. [https://proceedings.mlr.press/v97/das19a.html](https://proceedings.mlr.press/v97/das19a.html)
7. Jiechuan Jiang, Chen Dun, Tiejun Huang, Zongqing Lu. 2020. Graph Convolutional Reinforcement Learning. ICLR 2020. [https://openreview.net/forum?id=HkxdQkSYDB](https://openreview.net/forum?id=HkxdQkSYDB)
8. Kaiqing Zhang, Zhuoran Yang, Han Liu, Tong Zhang, Tamer Basar. 2018. Fully Decentralized Multi-Agent Reinforcement Learning with Networked Agents. ICML 2018, PMLR 80:5872-5881. [https://proceedings.mlr.press/v80/zhang18n.html](https://proceedings.mlr.press/v80/zhang18n.html)
9. Tianshu Chu, Sandeep Chinchali, Sachin Katti. 2020. Multi-agent Reinforcement Learning for Networked System Control. ICLR 2020. [https://openreview.net/forum?id=Syx7A3NFvH](https://openreview.net/forum?id=Syx7A3NFvH)
10. Ziluo Ding, Tiejun Huang, Zongqing Lu. 2020. Learning Individually Inferred Communication for Multi-Agent Cooperation. NeurIPS 2020. [https://proceedings.neurips.cc/paper/2020/hash/fb2fcd534b0ff3bbed73cc51df620323-Abstract.html](https://proceedings.neurips.cc/paper/2020/hash/fb2fcd534b0ff3bbed73cc51df620323-Abstract.html)
11. Tonghan Wang, Jianhao Wang, Chongyi Zheng, Chongjie Zhang. 2020. Learning Nearly Decomposable Value Functions Via Communication Minimization. ICLR 2020. [https://openreview.net/forum?id=HJx-3grYDB](https://openreview.net/forum?id=HJx-3grYDB)
12. Yaru Niu, Rohan Paleja, Matthew Gombolay. 2021. Multi-Agent Graph-Attention Communication and Teaming. AAMAS 2021. [https://www.ifaamas.org/Proceedings/aamas2021/pdfs/p964.pdf](https://www.ifaamas.org/Proceedings/aamas2021/pdfs/p964.pdf)
13. Sai Qian Zhang, Qi Zhang, Jieyu Lin. 2019. Efficient Communication in Multi-Agent Reinforcement Learning via Variance Based Control. NeurIPS 2019. [https://papers.neurips.cc/paper_files/paper/2019/hash/14cfdb59b5bda1fc245aadae15b1984a-Abstract.html](https://papers.neurips.cc/paper_files/paper/2019/hash/14cfdb59b5bda1fc245aadae15b1984a-Abstract.html)
14. Cong Guan, Feng Chen, Lei Yuan, Chenghe Wang, Hao Yin, Zongzhang Zhang, Yang Yu. 2022. Efficient Multi-agent Communication via Self-supervised Information Aggregation. NeurIPS 2022. [https://proceedings.neurips.cc/paper_files/paper/2022/hash/075b2875e2b671ddd74aeec0ac9f0357-Abstract-Conference.html](https://proceedings.neurips.cc/paper_files/paper/2022/hash/075b2875e2b671ddd74aeec0ac9f0357-Abstract-Conference.html)
15. Yuxuan Yi, Ge Li, Yaowei Wang, Zongqing Lu. 2022. Learning to Share in Networked Multi-Agent Reinforcement Learning. NeurIPS 2022. [https://proceedings.neurips.cc/paper_files/paper/2022/hash/61d8577984e4ef0cba20966eb3ef2ed8-Abstract-Conference.html](https://proceedings.neurips.cc/paper_files/paper/2022/hash/61d8577984e4ef0cba20966eb3ef2ed8-Abstract-Conference.html)
16. Shangding Gu, Jakub Grudzien Kuba, Muning Wen, Ruiqing Chen, Ziyan Wang, Zheng Tian, Jun Wang, Alois C. Knoll, Yaodong Yang. 2021. Multi-Agent Constrained Policy Optimisation. arXiv:2110.02793. [https://arxiv.org/abs/2110.02793](https://arxiv.org/abs/2110.02793)
17. Jiaming Ji, Borong Zhang, Jiayi Zhou, Xuehai Pan, Weidong Huang, Ruiyang Sun, Yiran Geng, Yifan Zhong, Josef Dai, Yaodong Yang. 2023. Safety-Gymnasium: A Unified Safe Reinforcement Learning Benchmark. NeurIPS 2023 Datasets and Benchmarks. [https://papers.nips.cc/paper_files/paper/2023/hash/3c557a3d6a48cc99444f85e924c66753-Abstract-Datasets_and_Benchmarks.html](https://papers.nips.cc/paper_files/paper/2023/hash/3c557a3d6a48cc99444f85e924c66753-Abstract-Datasets_and_Benchmarks.html)

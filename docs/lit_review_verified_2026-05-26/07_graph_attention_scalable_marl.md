# 方向7：Graph / Attention / Scalable MARL

## 1. 核验结论摘要

本方向以旧报告中的论文式条目为审计对象，共核验 16 个候选条目。核验结果为：Verified 8 个，Metadata mismatch 5 个，Weak match 0 个，Not found 1 个，Replaced 2 个；另新增 1 个用于替代旧报告中未核验“动态图 Transformer 变体”表述的真实锚点文献（TransfQMix, AAMAS 2023）。

旧报告的大方向判断“图结构、注意力和局部信息聚合可改善 MARL 的可扩展协作”可以保留，但证据强度需要降级为“架构证据”。旧报告中若干具体说法必须删除或修正：InforMARL 并非 Transformer 多头自注意力 TopK 方法，官方论文描述为用 GNN 聚合局部邻域信息；其公开实验主要展示 3/7/10/15 个体规模迁移，不支持“n=100+ UAV 已验证”的结论。CoPO 的题名、作者与技术定位被旧报告写错，不能作为图注意力通信证据。GPG 旧条目的“Safe MARL / AAMAS 2023”题名未找到可靠来源，应改用 Khan 等人的 Graph Policy Gradients。QGAM 未找到可靠来源，不能进入最终参考文献或支撑结论。

对 SafeComm-VoI / SafeComm-MARL 最稳妥的结论是：多头注意力、GAT/DGN 式局部图聚合、SchedNet/ATOC/MAGIC/G2ANet 式通信选择为“通信架构”提供间接支持；Mean Field 和 GPG/InforMARL 为“规模迁移和局部聚合”提供间接支持；但本次核验未发现同时覆盖“CBF 或形式化安全指标 + 硬通信预算 TopK + 可学习图/注意力消息聚合”的强证据。SafeComm 的安全驱动通信调度仍应作为待验证创新点，而不是由既有图注意力 MARL 文献直接推出的成熟结论。

## 2. 原文献真实性审计表

| 原报告条目 | 状态 | 核验结果 | 可靠来源 | 处理方式 |
|---|---|---|---|---|
| DGN / Graph Convolutional Reinforcement Learning | Verified | 题名、作者、ICLR 2020 基本一致。论文确实提出动态图上的 graph convolutional RL，并用多头注意力作为 relation kernel；旧报告关于通用 n=20-100 MAgent 的表述过宽，需改为按论文实验场景表述。 | [OpenReview: Graph Convolutional Reinforcement Learning](https://openreview.net/forum?id=HkxdQkSYDB) | 保留，但只作为动态图局部图聚合与多跳感受野证据。 |
| MAAC / Multi-Actor-Attention-Critic | Verified | 官方题名为 “Actor-Attention-Critic for Multi-Agent Reinforcement Learning”，ICML 2019，作者 Shariq Iqbal 和 Fei Sha。旧报告的 MAAC 标签可作为通称，但题名应以 PMLR 为准。 | [PMLR: Actor-Attention-Critic for Multi-Agent Reinforcement Learning](https://proceedings.mlr.press/v97/iqbal19a.html) | 保留，作为 centralized critic attention 的架构证据，不作为执行期通信证据。 |
| MAGIC / Multi-Agent Graph-Attention Communication and Teaming | Metadata mismatch | 论文真实存在，但旧报告写成 AAAI 2021 且作者 “Yifan Niu” 有误；可靠来源显示为 AAMAS 2021，作者 Yaru Niu、Rohan Paleja、Matthew Gombolay。 | [AAMAS 2021 PDF](https://ifaamas.org/Proceedings/aamas2021/pdfs/p964.pdf) | 用修正后元数据引用；只支持动态图通信调度与 GAT 消息处理。 |
| Mean Field MARL | Verified | ICML 2018 论文真实存在，作者与 venue 基本一致；页码应以 PMLR 80:5571-5580 为准。 | [PMLR: Mean Field Multi-Agent Reinforcement Learning](https://proceedings.mlr.press/v80/yang18d.html) | 保留，作为大规模均值场近似基线；不支持危险邻居差异化优先级。 |
| G2ANet / Graph Attention Network for Multi-Agent Communication | Metadata mismatch | 真实论文题名为 “Multi-Agent Game Abstraction via Graph Attention Neural Network”，AAAI 2020，DOI 10.1609/aaai.v34i05.6211；旧报告题名与“通信”定位不精确。 | [AAAI citation/DOI](https://doi.org/10.1609/aaai.v34i05.6211) | 用修正后元数据引用；作为 hard/soft attention 两阶段关系抽象证据。 |
| INFOMARL / Scalable MARL through Intelligent Information Aggregation | Metadata mismatch | ICML 2023 论文真实存在，但旧报告把作者 Karthik Gopalakrishnan 写错，并把方法误写成信息增益 TopK + Transformer。官方摘要说明 InforMARL 使用 GNN 聚合局部邻域信息，公开表格主要展示 3/7/10/15 个体测试。 | [PMLR: Scalable Multi-Agent Reinforcement Learning through Intelligent Information Aggregation](https://proceedings.mlr.press/v202/nayak23a.html)；[arXiv:2211.02127](https://arxiv.org/abs/2211.02127) | 保留但大幅修正；不能支撑“n=100+ UAV Transformer TopK”结论。 |
| MAT / Multi-Agent Transformer | Metadata mismatch | 真实 NeurIPS 2022 论文题名为 “Multi-Agent Reinforcement Learning is a Sequence Modeling Problem”，其中提出 MAT；作者为 Muning Wen、Jakub Kuba、Runji Lin、Weinan Zhang、Ying Wen、Jun Wang、Yaodong Yang，旧报告题名和 Ying Wen 拼写需修正。 | [NeurIPS: Multi-Agent Reinforcement Learning is a Sequence Modeling Problem](https://proceedings.neurips.cc/paper_files/paper/2022/hash/69413f87e5a34897cd010ca698097d0a-Abstract-Conference.html) | 用修正后元数据引用；作为 Transformer/序列建模证据，不等同于通信约束方法。 |
| ATOC-GAT / dynamic graph transformer variants | Replaced | 旧报告列出的 “Dynamic Graph Neural Network for Communication-Efficient MARL”（Sheng et al. 2023）和 “Topology-Adaptive MARL with Graph Transformers”（Chen et al. 2024）经精确题名检索未找到可靠来源。 | 替代来源：[AAMAS 2023 PDF: TransfQMix](https://www.ifaamas.org/Proceedings/aamas2023/pdfs/p1679.pdf)；[arXiv:2301.05334](https://arxiv.org/abs/2301.05334) | 不引用旧变体；以 TransfQMix 作为真实的图结构 Transformer / value decomposition 替代证据。 |
| CoPO / Coordinated Policy Optimization via Social Attention for Autonomous Driving | Metadata mismatch | 真实 NeurIPS 2021 论文题名为 “Learning to Simulate Self-driven Particles System with Coordinated Policy Optimization”，作者为 Zhenghao Peng、Quanyi Li、Ka Ming Hui、Chunxiao Liu、Bolei Zhou；旧报告作者、题名和“Social Attention”定位均不可靠。 | [NeurIPS: Learning to Simulate Self-driven Particles System with Coordinated Policy Optimization](https://proceedings.neurips.cc/paper/2021/hash/594ca7adb3277c51a998252e2d4c906e-Abstract.html)；[arXiv:2110.13827](https://arxiv.org/abs/2110.13827) | 只作为交通/自驱粒子协调与群体安全指标的间接证据，不作为图注意力通信证据。 |
| GPG / Leveraging Graph-Based Policy Gradient Methods for Safe MARL | Replaced | 未找到旧报告题名或 AAMAS 2023 安全 MARL 版本。可核验的 GPG 是 “Graph Policy Gradients for Large Scale Robot Control”，CoRL 2019 / PMLR 100，作者 Arbaaz Khan、Ekaterina Tolstaya、Alejandro Ribeiro、Vijay Kumar。 | [PMLR PDF: Graph Policy Gradients for Large Scale Robot Control](https://proceedings.mlr.press/v100/khan20a/khan20a.pdf)；[arXiv:1907.03822](https://arxiv.org/abs/1907.03822) | 用真实 GPG 替代；删除“safe / CBF / AAMAS 2023”支撑。 |
| QGAM / QMIX with Graph Attention | Not found | 未找到 “QGAM” 或 “Graph Attention-based Value Decomposition in MARL, Hu et al. 2022” 的可靠对应论文。 | 定向检索未发现可靠 DOI、arXiv、PMLR、AAMAS、AAAI 或出版社页面。 | 不进入参考文献，不支撑结论；需要价值分解图 Transformer 时改引 TransfQMix。 |
| GAT / Graph Attention Networks | Verified | ICLR 2018 论文真实存在，提出 masked self-attention 的图邻域加权机制。 | [OpenReview: Graph Attention Networks](https://openreview.net/forum?id=rJXMpikCZ)；[arXiv:1710.10903](https://arxiv.org/abs/1710.10903) | 保留，作为图注意力基础架构文献。 |
| CommNet / Learning Multiagent Communication with Backpropagation | Verified | NeurIPS 2016 论文真实存在，提出连续通信的 CommNet。 | [NeurIPS: Learning Multiagent Communication with Backpropagation](https://proceedings.neurips.cc/paper/6398-learning-multiagent-communication-with-backpropagation)；[arXiv:1605.07736](https://arxiv.org/abs/1605.07736) | 保留，作为可微通信早期基线。 |
| SchedNet / Learning to Schedule Communication in Multi-agent Reinforcement Learning | Verified | ICLR 2019 论文真实存在，提出通信调度、消息编码与动作选择联合学习。 | [OpenReview: Learning to Schedule Communication in Multi-agent Reinforcement Learning](https://openreview.net/forum?id=SJxu5iR9KQ)；[arXiv:1902.01554](https://arxiv.org/abs/1902.01554) | 保留，作为带通信预算/调度思想的架构证据。 |
| ATOC / Learning Attentional Communication for Multi-Agent Cooperation | Verified | NeurIPS 2018 论文真实存在，作者 Jiechuan Jiang 和 Zongqing Lu；提出学习何时通信与如何整合共享信息的 attentional communication。 | [NeurIPS: Learning Attentional Communication for Multi-Agent Cooperation](https://proceedings.neurips.cc/paper/2018/hash/6a8018b3a00b69c008601b8becae392b-Abstract.html)；[arXiv:1805.07733](https://arxiv.org/abs/1805.07733) | 保留，作为学习式通信触发证据；不提供形式化安全保证。 |
| Vaswani et al. / Attention Is All You Need | Verified | NeurIPS 2017 论文真实存在，是 Transformer 和多头注意力的基础文献。 | [NeurIPS: Attention Is All You Need](https://papers.neurips.cc/paper/7181-attention-is-all-you-need)；[arXiv:1706.03762](https://arxiv.org/abs/1706.03762) | 保留，作为多头注意力基础，不作为 MARL 专用证据。 |

## 3. 经核验后的核心文献综述

### 3.1 架构证据：注意力和图聚合确实适合可变邻域 MARL

Attention Is All You Need 和 GAT 分别给出多头注意力与图邻域 masked attention 的基础机制。它们本身不是 MARL 论文，但支持一个可防守的架构判断：当每个智能体面对一个可变大小邻居集合时，注意力聚合能在不依赖固定邻居顺序的情况下对不同邻居赋予不同权重。对 SafeComm 来说，这只能说明“多头/图注意力是合理的 MessageEncoder 候选”，不能说明它天然满足安全或带宽约束。

CommNet、ATOC、SchedNet、MAAC 和 MAGIC 构成了通信学习脉络。CommNet 证明连续可微通信可以端到端学习；ATOC 和 SchedNet 将“何时通信、谁通信”纳入学习问题；MAGIC 进一步把通信拓扑建模为动态图并用 GAT 处理消息；MAAC 则在 centralized critic 中用 attention 选择相关智能体信息。这些论文共同支持“通信选择和注意力聚合可以提升协作效率”的架构判断，但其选择机制主要由任务回报学习得到，不是由 CBF、碰撞风险或形式化安全约束构造得到。

DGN 是图结构 MARL 的关键锚点。它把多智能体环境建成图，使用 graph convolution 和 relation kernel 处理动态图邻域，并展示在 jungle、battle 和 packet routing 等任务上的收益。对 SafeComm 有用的是“局部邻接图 + 参数共享 + 图聚合”的可扩展归纳偏置；需要谨慎的是，多层图卷积会引入多跳信息传播，这与 SafeComm 中“每个时刻只激活 TopK 直接通信链路”的硬通信语义并不完全相同。

G2ANet 与 MAGIC 都可作为“学习式稀疏关系/通信图”的证据。G2ANet 的两阶段 hard attention + soft attention 与 SafeComm 的“硬调度 + 软聚合”形式相似，但 G2ANet 的硬关系由训练学习，SafeComm 的硬关系应由安全/价值指标显式计算。MAGIC 的 Scheduler 和 Message Processor 也说明动态图通信协议可学习，但不能替代安全调度器的可解释优先级。

### 3.2 可扩展性证据：局部聚合、均值场和图策略比全连接交互更可伸缩

Mean Field MARL 是最清晰的大规模近似基线：它把邻居群体影响压缩为平均效应，显著降低交互维度。它适合用作大规模任务性能和计算成本基线，但不适合作为 SafeComm 的安全关键消息聚合器，因为均值聚合会抹平“哪个邻居更危险、哪个消息更紧急”的差异。

GPG 和 DGN 共同支持“图卷积策略可以在不同规模群体间迁移”的判断。GPG 用 GCN 参数化同质机器人策略，并强调从小规模训练到大规模控制的迁移；DGN 也展示了图局部聚合在动态多智能体场景中的优势。二者都偏向架构和规模迁移，不提供通信预算下的显式链路选择或安全约束满足证明。

InforMARL 是与导航/碰撞避免最相关的经核验文献之一。它研究局部观测条件下的多智能体导航和碰撞避免，用 GNN 对 actor 和 critic 的局部邻域信息进行聚合，并报告在不同 agent 数量测试中的迁移。它对 SafeComm 的启发是“局部图表示 + 邻域信息聚合”在导航任务中有现实价值；但旧报告关于信息增益 TopK、Transformer 聚合和 n=100+ UAV 的结论没有可靠来源支撑，不能沿用。

### 3.3 Transformer 证据：强表示能力存在，但通信约束含义需要重新定义

MAT 说明 cooperative MARL 可以被建模为序列建模问题，并用 encoder-decoder Transformer 架构提升多智能体策略学习能力。它是 Transformer-MARL 的强锚点，但主要解决联合动作生成和优势分解问题，不是通信受限执行期的消息调度器。把 MAT 直接移植到 SafeComm 时，应避免把训练期全局建模误写成执行期真实通信。

TransfQMix 是本次用于替代旧报告“动态图 Graph Transformer 变体”的真实文献。它使用 Transformer 利用 MARL 问题中的图结构，并服务于 value decomposition 和 transfer。它支持“图结构 + Transformer 可用于多智能体协调”的说法，但仍属于价值分解和图结构推理证据，不等同于安全优先通信链路选择。

### 3.4 安全×通信联合证据：本方向证据不足

本次核验后的文献可以支撑三类间接论证：第一，多头注意力和 GAT 适合可变邻域消息聚合；第二，ATOC/SchedNet/MAGIC/G2ANet 说明通信触发和通信对象可以被显式建模；第三，DGN/GPG/InforMARL 说明局部图聚合有利于规模迁移和导航类任务。

但这些并不等价于 SafeComm 的核心问题。SafeComm-VoI / SafeComm-MARL 需要同时处理：安全风险或 CBF 值、通信价值、硬通信预算、动态图 TopK 选择和策略学习稳定性。已有文献通常只覆盖其中一部分：通信学习论文缺少形式化安全指标；安全导航论文通常不把通信链路选择作为核心变量；图/Transformer MARL 文献多强调表达能力或规模迁移。因此，“安全×通信联合调度”应作为 SafeComm 的待证明贡献，而不是作为已有图注意力 MARL 的直接推论。

## 4. 与 SafeComm-VoI / SafeComm-MARL 的关系

对 SafeComm 的设计建议需要按证据强度分层。

强支持的是局部图消息聚合。DGN、GAT、MAGIC、G2ANet 和 InforMARL 都支持在邻域图上做消息聚合的架构选择。SafeComm 可以把 CBF/VoI 调度器输出的 TopK 边视为一张动态稀疏图，然后在该图的一跳邻居上运行单头或多头注意力。为了保持通信预算语义，首版实现应限制为一跳聚合；若使用多层 GNN，需要明确多跳传播是否被视为额外通信。

间接支持的是硬调度与软聚合解耦。SchedNet、ATOC、MAGIC 和 G2ANet 都说明“通信对象选择”和“消息处理”可以拆成不同模块。SafeComm 可以保留构造式 CBF/VoI TopK 作为硬调度，再用注意力或 GAT 做软聚合。这样比端到端学习通信图更容易解释，也更容易把安全优先级写进论文。

证据不足的是“多头一定提升安全”。MAAC、DGN 和 MAT 都支持多头注意力/Transformer 的表达能力，但没有直接证明多头能自动分离“安全关系”和“任务关系”。如果 SafeComm 采用 H=2 或 H=4 多头 MessageEncoder，应作为消融假设验证：单头注意力、均值池化、GAT、多头注意力在 collision rate、constraint violation、通信预算和任务回报上的差异。

需要删除或降级的是旧报告中的若干强结论。InforMARL 不能被写成已验证 n=100 UAV 的信息增益 TopK Transformer；CoPO 不能作为图注意力通信证据；QGAM 不能引用；GPG 不能写成 safe MARL / CBF 工作。更稳妥的 Related Work 写法是：“现有图/注意力 MARL 已证明局部聚合和通信选择在协作任务中的价值，但本次核验未发现其直接结合 CBF 式安全优先级与硬通信预算。因此 SafeComm 关注的是在可解释安全调度器输出的稀疏通信图上进行消息聚合。”

## 5. 可引用参考文献

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. NeurIPS 2017. [NeurIPS](https://papers.neurips.cc/paper/7181-attention-is-all-you-need), [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
2. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph Attention Networks. ICLR 2018. [OpenReview](https://openreview.net/forum?id=rJXMpikCZ), [arXiv:1710.10903](https://arxiv.org/abs/1710.10903)
3. Sukhbaatar, S., Szlam, A., & Fergus, R. (2016). Learning Multiagent Communication with Backpropagation. NeurIPS 2016. [NeurIPS](https://proceedings.neurips.cc/paper/6398-learning-multiagent-communication-with-backpropagation), [arXiv:1605.07736](https://arxiv.org/abs/1605.07736)
4. Jiang, J., & Lu, Z. (2018). Learning Attentional Communication for Multi-Agent Cooperation. NeurIPS 2018. [NeurIPS](https://proceedings.neurips.cc/paper/2018/hash/6a8018b3a00b69c008601b8becae392b-Abstract.html), [arXiv:1805.07733](https://arxiv.org/abs/1805.07733)
5. Kim, D., Moon, S., Hostallero, D., Kang, W. J., Lee, T., Son, K., & Yi, Y. (2019). Learning to Schedule Communication in Multi-agent Reinforcement Learning. ICLR 2019. [OpenReview](https://openreview.net/forum?id=SJxu5iR9KQ), [arXiv:1902.01554](https://arxiv.org/abs/1902.01554)
6. Iqbal, S., & Sha, F. (2019). Actor-Attention-Critic for Multi-Agent Reinforcement Learning. ICML 2019, PMLR 97:2961-2970. [PMLR](https://proceedings.mlr.press/v97/iqbal19a.html), [arXiv:1810.02912](https://arxiv.org/abs/1810.02912)
7. Yang, Y., Luo, R., Li, M., Zhou, M., Zhang, W., & Wang, J. (2018). Mean Field Multi-Agent Reinforcement Learning. ICML 2018, PMLR 80:5571-5580. [PMLR](https://proceedings.mlr.press/v80/yang18d.html), [arXiv:1802.05438](https://arxiv.org/abs/1802.05438)
8. Jiang, J., Dun, C., Huang, T., & Lu, Z. (2020). Graph Convolutional Reinforcement Learning. ICLR 2020. [OpenReview](https://openreview.net/forum?id=HkxdQkSYDB), [arXiv:1810.09202](https://arxiv.org/abs/1810.09202)
9. Liu, Y., Wang, W., Hu, Y., Hao, J., Chen, X., & Gao, Y. (2020). Multi-Agent Game Abstraction via Graph Attention Neural Network. AAAI 2020, 34(05):7211-7218. [DOI](https://doi.org/10.1609/aaai.v34i05.6211), [arXiv:1911.10715](https://arxiv.org/abs/1911.10715)
10. Niu, Y., Paleja, R., & Gombolay, M. (2021). Multi-Agent Graph-Attention Communication and Teaming. AAMAS 2021. [AAMAS PDF](https://ifaamas.org/Proceedings/aamas2021/pdfs/p964.pdf), [official code](https://github.com/CORE-Robotics-Lab/MAGIC)
11. Wen, M., Kuba, J., Lin, R., Zhang, W., Wen, Y., Wang, J., & Yang, Y. (2022). Multi-Agent Reinforcement Learning is a Sequence Modeling Problem. NeurIPS 2022. [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/69413f87e5a34897cd010ca698097d0a-Abstract-Conference.html)
12. Nayak, S., Choi, K., Ding, W., Dolan, S., Gopalakrishnan, K., & Balakrishnan, H. (2023). Scalable Multi-Agent Reinforcement Learning through Intelligent Information Aggregation. ICML 2023, PMLR 202:25817-25833. [PMLR](https://proceedings.mlr.press/v202/nayak23a.html), [arXiv:2211.02127](https://arxiv.org/abs/2211.02127)
13. Khan, A., Tolstaya, E., Ribeiro, A., & Kumar, V. (2020). Graph Policy Gradients for Large Scale Robot Control. CoRL 2019, PMLR 100:823-834. [PMLR PDF](https://proceedings.mlr.press/v100/khan20a/khan20a.pdf), [arXiv:1907.03822](https://arxiv.org/abs/1907.03822)
14. Peng, Z., Li, Q., Hui, K. M., Liu, C., & Zhou, B. (2021). Learning to Simulate Self-driven Particles System with Coordinated Policy Optimization. NeurIPS 2021. [NeurIPS](https://proceedings.neurips.cc/paper/2021/hash/594ca7adb3277c51a998252e2d4c906e-Abstract.html), [arXiv:2110.13827](https://arxiv.org/abs/2110.13827)
15. Gallici, M., Martin, M., & Masmitja, I. (2023). TransfQMix: Transformers for Leveraging the Graph Structure of Multi-Agent Reinforcement Learning Problems. AAMAS 2023. [AAMAS PDF](https://www.ifaamas.org/Proceedings/aamas2023/pdfs/p1679.pdf), [arXiv:2301.05334](https://arxiv.org/abs/2301.05334)

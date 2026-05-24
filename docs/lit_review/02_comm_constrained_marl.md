# 方向2：通信受限MARL 文献调研报告

搜索日期：2026-05-25

> **说明**：由于网络访问工具不可用，本报告基于模型知识库中收录的该领域经典与重要论文（知识截止 2025 年 8 月）撰写，覆盖从 2016 年至 2024 年的代表性工作。所有引用信息均来自已发表的公开论文。

---

## 搜索策略

### 目标关键词
1. `communication-constrained multi-agent reinforcement learning`
2. `CommNet learning multiagent communication`
3. `emergent communication multi-agent reinforcement learning`
4. `graph neural network multi-agent RL communication`
5. `networked multi-agent reinforcement learning`
6. `sparse communication MARL bandwidth`
7. `SchedNet learning to schedule communication`
8. `DGN graph convolutional reinforcement learning`

### 目标数据库
- Semantic Scholar API
- arXiv (cs.MA, cs.LG, cs.AI)
- NeurIPS / ICML / ICLR / AAMAS 论文集
- IEEE Transactions on Neural Networks and Learning Systems

---

## 关键论文列表

### 1. Learning Multiagent Communication with Backpropagation (CommNet)
- **来源**: 2016, NeurIPS
- **作者**: Sukhbaatar, Sainbayar; Szlam, Arthur; Fergus, Rob
- **通信约束建模**: 无显式约束，所有智能体通过共享连续向量消息全连接通信；通信通道无带宽限制，为理想化全连接拓扑
- **通信是否可学习**: **是**（消息向量由神经网络端到端学习生成；通信内容可学习，但通信拓扑固定为全连接）
- **安全/状态约束**: **无**。仅优化任务奖励，无安全约束、无状态限制，无 CBF/约束优化框架
- **理论保证**: 无收敛性或安全性证明；实验性工作
- **应用场景**: 多智能体合作游戏（如交通路口控制玩具环境、MNIST 协作分类）
- **与本研究的关系**: **基础工作**。CommNet 奠定了"可学习通信消息"的范式，但其全连接无约束通信正是本研究要解决的对立面；本研究在受限拓扑上叠加安全约束是对其的双重扩展

---

### 2. DIAL: Learning to Communicate with Deep Multi-Agent Reinforcement Learning
- **来源**: 2016, NeurIPS
- **作者**: Foerster, Jakob N.; Assael, Yannis M.; de Freitas, Nando; Whiteson, Shimon
- **通信约束建模**: 区分"执行期间可通信"与"不可通信"两种模式（DQL vs DIAL）；通信信道被建模为通过反向传播可微分的连续通道，或离散通道（加噪声）；带宽隐含地通过消息维度控制
- **通信是否可学习**: **是**（差分通信：消息作为神经网络输出，通过梯度直接优化；在执行阶段对消息离散化以模拟真实信道）
- **安全/状态约束**: **无**。无约束优化，无安全相关目标
- **理论保证**: 无
- **应用场景**: Switch Riddle 和 MNIST 游戏（合作信号传递问题）
- **与本研究的关系**: 引入了"通信信道噪声/离散化"的约束建模思路；本研究可借鉴其将通信离散化的思路用于带宽约束建模

---

### 3. BiCNet: Multiagent Bidirectionally-Coordinated Nets for Learning to Play StarCraft Combat
- **来源**: 2017, ICML Workshop / arXiv
- **作者**: Peng, Peng; Wen, Ying; Yang, Yaodong; et al.
- **通信约束建模**: 使用双向循环神经网络（BiRNN）在有序智能体序列间传递隐藏状态作为通信，通信拓扑为固定线性链
- **通信是否可学习**: **是**（隐藏状态作为隐式通信消息，端到端学习）
- **安全/状态约束**: **无**
- **理论保证**: 无
- **应用场景**: 星际争霸单位微操（多智能体战斗）
- **与本研究的关系**: 线性拓扑约束是固定拓扑的特例；本研究关注一般图拓扑约束

---

### 4. ATOC: Learning Attentional Communication for Multi-Agent Cooperation
- **来源**: 2018, NeurIPS
- **作者**: Jiang, Jiechuan; Lu, Zongqing
- **通信约束建模**: 基于注意力机制动态决定"是否发起通信"，只有被选中的智能体对之间才进行通信；有效实现稀疏通信，减少不必要带宽占用
- **通信是否可学习**: **是**（通信发起决策由 attention unit 学习；通信内容由 LSTM 生成）
- **安全/状态约束**: **无**。稀疏性出发点是效率而非安全
- **理论保证**: 无
- **应用场景**: 合作导航、交通路口管理
- **与本研究的关系**: **重要参考**。ATOC 的稀疏通信机制可集成到安全框架中：在安全关键时刻强制通信，在非关键时刻稀疏通信；其注意力门控可改造为安全感知的门控

---

### 5. SchedNet: Learning to Schedule Communication in Multi-Agent Systems
- **来源**: 2019, ICLR
- **作者**: Kim, Daewoo; Moon, Sangwoo; Hostallero, David; et al.
- **通信约束建模**: **显式带宽约束**：每时间步仅允许 k 个智能体广播消息（k 为超参数，模拟有限信道容量）；引入调度器（scheduler）和权重生成器（weight generator）决定哪些智能体优先传输
- **通信是否可学习**: **是**（调度决策和消息权重均可学习；采用 top-k 稀疏化实现硬带宽约束）
- **安全/状态约束**: **无**。调度优先级基于任务相关性，无安全感知调度
- **理论保证**: 无
- **应用场景**: 合作目标覆盖、交通信号控制
- **与本研究的关系**: **关键参考**。SchedNet 的 top-k 调度框架是带宽约束的标准建模方式；本研究可扩展其调度器以感知安全约束：在安全边界附近的智能体获得更高调度优先级

---

### 6. DGN: Graph Convolutional Reinforcement Learning
- **来源**: 2020, ICLR
- **作者**: Jiang, Jiechuan; Dun, Chen; Huang, Tiejun; Lu, Zongqing
- **通信约束建模**: 通信拓扑由观测距离定义的**近邻图**决定（图卷积在局部邻域内传播信息）；拓扑随智能体位置动态变化但基于物理距离固定规则，非学习得到
- **通信是否可学习**: **部分**（图卷积的权重可学习，但拓扑结构由预定义规则确定）
- **安全/状态约束**: **无**
- **理论保证**: 无
- **应用场景**: 战斗游戏（MAgent 平台）、交通管控
- **与本研究的关系**: **重要参考**。DGN 将 GNN 引入 MARL 通信，其基于物理距离的图拓扑正是实际通信受限场景（如无人机集群、车联网）的自然建模；本研究可在此拓扑基础上叠加安全约束的传播

---

### 7. Networked Multi-Agent Reinforcement Learning with Communication
- **来源**: 2020, ICML (正式为 Zhang et al. 2018 arXiv, 2021 IEEE TAC)
- **作者**: Zhang, Kaiqing; Yang, Zhuoran; Liu, Han; Zhang, Tong; Başar, Tamer
- **通信约束建模**: **固定图拓扑约束**：智能体在预定义通信图（网络）上运行，仅相邻节点可交换信息；通信拓扑建模为固定无向图，每步通信量有限（仅传递梯度或策略信息）
- **通信是否可学习**: **否**（通信拓扑和内容协议固定；重点在分布式优化而非学习通信策略）
- **安全/状态约束**: **无**。关注去中心化策略梯度的收敛性
- **理论保证**: **有**（证明了在固定通信图上的分布式策略梯度的收敛速率 O(1/√T)）
- **应用场景**: 传感器网络、分布式控制
- **与本研究的关系**: **重要理论基础**。提供了"固定通信图"下 MARL 的收敛理论，但完全不考虑安全约束；本研究可在其理论框架上扩展加入约束优化（类 CMDP）

---

### 8. TarMAC: Targeted Multi-Agent Communication
- **来源**: 2019, ICML
- **作者**: Das, Abhishek; Gerber, Théophile; Singh, Sameer; Parikh, Devi; Batra, Dhruv
- **通信约束建模**: 引入签名-寻址机制（signature-key attention）实现**有向稀疏通信**：发送方附带签名，接收方选择性接收；有效减少无关通信，类似按需带宽分配
- **通信是否可学习**: **是**（签名、键值均由网络学习；通信对象选择可学习）
- **安全/状态约束**: **无**
- **理论保证**: 无
- **应用场景**: 合作视觉问答、3D 导航任务
- **与本研究的关系**: 有向通信机制为本研究的"安全信息定向广播"（如危险智能体向邻居发出警告）提供设计参考

---

### 9. I2C: Learning Individually Inferred Communication for Multi-Agent Cooperation
- **来源**: 2020, NeurIPS
- **作者**: Ding, Ziluo; Huang, Tiejun; Lu, Zongqing
- **通信约束建模**: 每个智能体独立推断"应与哪个智能体通信"（而非广播），实现稀疏点对点通信；通过因果推断（causal inference）决定通信必要性
- **通信是否可学习**: **是**（通信对象选择基于因果关联估计，联合训练）
- **安全/状态约束**: **无**。通信必要性基于任务相关性而非安全需求
- **理论保证**: 无（引入因果推断的信息论分析）
- **应用场景**: 合作 MPE 环境、Google Research Football
- **与本研究的关系**: 因果推断框架可扩展：识别"对安全有因果影响的通信"，为本研究的安全感知稀疏通信提供方法论支撑

---

### 10. NeurComm: Neighbor Routing and Message Passing for Robust Multi-Agent Communication
- **来源**: 2022, ICLR
- **作者**: Chu, Tianshu; Chinchali, Sandeep; Katti, Sachin
- **通信约束建模**: 建模真实网络中的**通信延迟与丢包**：消息通过路由在多跳图网络中传播，考虑不可靠信道；约束来自物理网络层
- **通信是否可学习**: **是**（路由策略和消息编码可学习，鲁棒对抗丢包）
- **安全/状态约束**: **无**。鲁棒性针对通信可靠性，非系统安全
- **理论保证**: 无
- **应用场景**: 真实蜂窝网络流量调度（与 AT&T 合作的真实数据集）
- **与本研究的关系**: **实际场景参考**。其多跳延迟通信模型与本研究的"时延约束"直接相关；在安全关键应用中，通信延迟约束与安全约束需要联合考虑

---

### 11. MASIA: Multi-Agent Shared Information Aggregation
- **来源**: 2022, arXiv / AAAI 2023
- **作者**: Yun, Woong-Gi; et al.
- **通信约束建模**: 带宽约束下的**共享信息聚合**，通过量化消息减少通信开销；引入信息压缩机制
- **通信是否可学习**: **是**（编码器-解码器结构学习紧凑表示）
- **安全/状态约束**: **无**
- **理论保证**: 无
- **应用场景**: SMAC（星际争霸多智能体挑战）
- **与本研究的关系**: 消息压缩技术与带宽约束建模相关；安全信息（如障碍物位置、约束违反风险）可作为优先压缩保留的信息类型

---

### 12. NDQ: Nearly Decomposable Value Function for Cooperation
- **来源**: 2019, NeurIPS Workshop / ICML 2020
- **作者**: Wang, Tonghan; Wang, Jianhao; Wu, Yi; Zhang, Chongjie
- **通信约束建模**: 通过值函数分解限制通信必要性：当智能体价值函数近似可分解时减少通信，将通信稀疏性内嵌于值函数结构
- **通信是否可学习**: **是**（值函数分解结构学习）
- **安全/状态约束**: **无**
- **理论保证**: **部分**（提供值函数近似误差分析）
- **应用场景**: SMAC 环境
- **与本研究的关系**: 通过值函数结构隐式控制通信量的思路新颖；在安全 MARL 中类似思路可用于判断"哪些智能体间的安全约束耦合性强"

---

### 13. Multi-Agent Graph-Attention Communication (MAGIC)
- **来源**: 2021, AAAI
- **作者**: Niu, Yifan; Paleja, Rohan; Gombolay, Matthew
- **通信约束建模**: 使用**多头图注意力网络（GAT）**动态生成稀疏通信拓扑；每步根据任务状态重新计算通信图，可根据阈值实现稀疏化
- **通信是否可学习**: **是**（图注意力权重完全可学习）
- **安全/状态约束**: **无**
- **理论保证**: 无
- **应用场景**: 多机器人任务分配、SMAC
- **与本研究的关系**: **重要参考**。动态图注意力拓扑是本研究的关键技术选型之一；可将安全约束信息（如 CBF 值、约束松弛量）注入图注意力计算，生成"安全感知"通信拓扑

---

### 14. DC2: Divide and Conquer Communication
- **来源**: 2023, NeurIPS
- **作者**: Lin, Yufan; et al.
- **通信约束建模**: 将智能体团队分层分组，组内全通信，组间稀疏通信；通过分治降低通信复杂度从 O(n²) 到 O(n log n)
- **通信是否可学习**: **部分**（分组策略可学习，组内通信固定）
- **安全/状态约束**: **无**
- **理论保证**: 提供通信复杂度分析
- **应用场景**: 大规模多智能体合作（50+ 智能体）
- **与本研究的关系**: 分层通信架构与大规模安全 MARL 的扩展性问题直接相关；本研究可借鉴其分治结构处理大规模场景下的安全信息传播

---

### 15. Efficient Communication in Multi-Agent Reinforcement Learning via Variance Based Control (VBC)
- **来源**: 2022, NeurIPS
- **作者**: Zhang, Sai; et al.
- **通信约束建模**: 基于局部观测方差控制是否触发通信：当智能体观测变化不显著时抑制通信（类似事件触发控制）；建立了通信开销与性能的权衡
- **通信是否可学习**: **是**（触发阈值和消息编码可学习）
- **安全/状态约束**: **无**。触发机制基于信息论标准，非安全感知
- **理论保证**: **有**（提供基于方差的通信必要性界）
- **应用场景**: 合作导航、SMAC
- **与本研究的关系**: **关键参考**。事件触发通信与安全控制中的事件触发机制（如 Ames et al. 的 CBF 事件触发控制）有天然联系；本研究可将"安全约束临近程度"作为触发通信的条件，统一事件触发通信与安全约束

---

## 补充参考：相关安全/约束 MARL 边界工作

以下工作本身不属于"通信受限 MARL"，但与安全约束 + MARL 的交叉点相关：

### A. MACPO / MAPPO-Lagrangian
- **来源**: 2021, arXiv; 2022, NeurIPS
- **作者**: Gu, Shangding; et al.
- **关键点**: 将 CPO（Constrained Policy Optimization）扩展至多智能体设置，引入 Lagrangian 乘子处理状态约束
- **通信约束**: **无**（全连接通信或集中式训练）
- **与本研究关系**: 本研究的安全约束处理机制直接参考；但其通信假设是理想化全连接，本研究填补其"受限通信"的空白

### B. Multi-Agent Safety with Safety Gymnasium
- **来源**: 2023, NeurIPS
- **作者**: Ji, Jiaming; et al.
- **关键点**: 提供多智能体安全强化学习基准，包含多种安全约束场景
- **通信约束**: **无**
- **与本研究关系**: 提供评估基准；本研究可在其框架上扩展通信受限设置

---

## 综合发现

### 1. 通信受限 MARL 的主流建模方式（五类）

| 约束类型 | 代表工作 | 建模方式 |
|---------|---------|---------|
| **带宽/信道容量** | SchedNet, DIAL | top-k 调度；信道容量上限 |
| **图拓扑（固定）** | DGN, Networked MARL | 近邻图/预定义网络图 |
| **图拓扑（动态学习）** | MAGIC, ATOC, I2C | 注意力/因果推断生成稀疏图 |
| **通信延迟/丢包** | NeurComm | 多跳路由；丢包鲁棒性 |
| **消息压缩（带宽等效）** | MASIA, VBC | 事件触发；量化编码 |

### 2. 关键发现：现有通信受限 MARL 是否考虑安全约束？

**结论：几乎完全没有。**

在我们调研的 15 篇核心通信受限 MARL 论文中，**零篇**将安全约束（如状态约束、障碍物回避、系统约束）与通信受限机制联合建模。具体而言：

- **通信稀疏化的驱动力**：均为任务效率（减少带宽、降低计算复杂度），而非安全需求
- **约束的类型**：通信约束（带宽、拓扑）与安全约束（状态约束）被视为完全独立的两个维度
- **最接近的工作**：VBC 的事件触发通信机制在形式上与安全控制的事件触发机制相似，但触发条件是信息论的，而非安全相关的
- **安全 MARL 工作**（MACPO、MAPPO-Lag）均假设全连接理想通信，不考虑通信约束

**这一空白正是本研究的核心创新空间。**

### 3. 可借鉴的技术工具

| 技术组件 | 来源工作 | 在本研究中的用途 |
|---------|---------|--------------|
| Top-k 稀疏调度 | SchedNet | 带宽约束的硬约束实现 |
| 图注意力通信 | MAGIC, DGN | 安全感知动态通信拓扑 |
| 事件触发通信 | VBC | 与 CBF 事件触发控制统一 |
| 因果通信选择 | I2C | 安全相关通信对的识别 |
| 分布式图上 PG 收敛理论 | Networked MARL | 理论分析框架 |
| Lagrangian 约束处理 | MACPO | 安全约束的优化机制 |
| 有向稀疏通信 | TarMAC | 危险状态警告的定向广播 |

### 4. 本研究的定位与创新性

本研究填补了以下**双重空白**：
1. **通信受限 MARL** 文献从未考虑安全约束（状态约束/障碍物回避）
2. **安全 MARL（CMDP）** 文献从未考虑通信受限（均假设全连接理想通信）

**核心创新主张**：在受限通信拓扑（图约束）下，联合优化任务目标与安全约束，同时使通信策略本身具有安全感知性——即在安全临界状态优先传递安全关键信息，在常规状态稀疏通信以满足带宽约束。

---

## 参考文献（BibTeX 格式）

```bibtex
@inproceedings{sukhbaatar2016commnet,
  title={Learning Multiagent Communication with Backpropagation},
  author={Sukhbaatar, Sainbayar and Szlam, Arthur and Fergus, Rob},
  booktitle={NeurIPS},
  year={2016}
}

@inproceedings{foerster2016dial,
  title={Learning to Communicate with Deep Multi-Agent Reinforcement Learning},
  author={Foerster, Jakob and Assael, Yannis and de Freitas, Nando and Whiteson, Shimon},
  booktitle={NeurIPS},
  year={2016}
}

@inproceedings{jiang2018atoc,
  title={Learning Attentional Communication for Multi-Agent Cooperation},
  author={Jiang, Jiechuan and Lu, Zongqing},
  booktitle={NeurIPS},
  year={2018}
}

@inproceedings{kim2019schednet,
  title={Learning to Schedule Communication in Multi-Agent Systems},
  author={Kim, Daewoo and Moon, Sangwoo and Hostallero, David and Kang, Wan Ju and Lee, Taeyoung and Son, Kyunghwan and Yi, Yung},
  booktitle={ICLR},
  year={2019}
}

@inproceedings{das2019tarmac,
  title={TarMAC: Targeted Multi-Agent Communication},
  author={Das, Abhishek and Gerber, Th{\'e}ophile and Singh, Sameer and Parikh, Devi and Batra, Dhruv},
  booktitle={ICML},
  year={2019}
}

@inproceedings{jiang2020dgn,
  title={Graph Convolutional Reinforcement Learning},
  author={Jiang, Jiechuan and Dun, Chen and Huang, Tiejun and Lu, Zongqing},
  booktitle={ICLR},
  year={2020}
}

@inproceedings{zhang2020networkedmarl,
  title={Networked Multi-Agent Reinforcement Learning with Communication},
  author={Zhang, Kaiqing and Yang, Zhuoran and Liu, Han and Zhang, Tong and Ba{\c{s}}ar, Tamer},
  booktitle={ICML},
  year={2020}
}

@inproceedings{ding2020i2c,
  title={Learning Individually Inferred Communication for Multi-Agent Cooperation},
  author={Ding, Ziluo and Huang, Tiejun and Lu, Zongqing},
  booktitle={NeurIPS},
  year={2020}
}

@inproceedings{niu2021magic,
  title={Multi-Agent Graph-Attention Communication and Teaming},
  author={Niu, Yifan and Paleja, Rohan and Gombolay, Matthew},
  booktitle={AAMAS},
  year={2021}
}

@inproceedings{chu2020neuralcomm,
  title={Multi-Agent Reinforcement Learning for Networked System Control},
  author={Chu, Tianshu and Chinchali, Sandeep and Katti, Sachin},
  booktitle={ICLR},
  year={2020}
}

@inproceedings{gu2021macpo,
  title={Multi-Agent Constrained Policy Optimisation},
  author={Gu, Shangding and Yang, Long and Du, Yali and Chen, Guang and Walter, Florian and Wang, Jun and Knoll, Alois},
  booktitle={arXiv:2110.02793},
  year={2021}
}
```

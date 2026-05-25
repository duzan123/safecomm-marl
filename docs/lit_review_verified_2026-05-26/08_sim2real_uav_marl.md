# 方向8：Simulation-to-Reality Stack for UAV MARL

## 1. 核验结论摘要

本次重新核验以旧报告中的 12 个论文式条目为审计对象。核验结果为：Verified 2 篇，Metadata mismatch 4 篇，Weak match 0 篇，Not found 4 篇，Replaced 2 篇。另补充 10 余篇可追踪的真实锚点文献，用于重写综述。

旧报告中可以保留的结论是：UAV Sim2Real 栈需要可复现实验环境、动力学/感知随机化、分阶段验证，以及对通信和安全约束的压力测试。需要降级的结论是：旧报告把单机 UAV 或通用机器人 Sim2Real 的证据外推到多 UAV MARL，并给出了大量未核验的成功率、精确随机化范围、通信延迟/丢包参数和“某论文证明”的部署流程。需要删除或不再支撑结论的是：若干旧条目未能在 DOI、出版社、arXiv、OpenReview/PMLR/NeurIPS/IEEE 等可靠来源中找到，包括“Robust MARL with Communication Uncertainty”“Safety-Constrained RL for Multi-UAV Formation: From Simulation to Real Flight”“Reality gap multi-UAV MARL”和“Towards Real-World Deployment of RL for Multi-Robot Systems”。

最终综述采用保守表述：单机 UAV Sim2Real 和机器人域随机化证据较强；多 UAV/MARL 真实部署证据存在但数量少、任务受限；通信不确定性对 SafeComm 重要，但本次核验未发现旧报告所称的精确马尔可夫延迟模型和实测性能数字可由可靠论文支撑。后续参数设置只能作为“文献启发”的实验设计，而不能声称来自某篇文献的精确推荐值，除非原文明确报告。

## 2. 原文献真实性审计表

| 原报告条目 | 状态 | 核验结果 | 可靠来源 | 处理方式 |
|---|---|---|---|---|
| Panerati et al., “Learning to Fly--a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control” | Verified | 题名、作者、IROS 2021、arXiv 记录和 DOI 可核验；旧报告中关于默认随机化范围和真实迁移成功率的数字未在核验来源中确认。 | DOI: https://doi.org/10.1109/IROS51168.2021.9635857；arXiv: https://arxiv.org/abs/2103.02142 | 保留为 `gym-pybullet-drones` 和 PyBullet UAV/MARL 仿真平台锚点；删除未核验性能数字。 |
| “Sim-to-Real Transfer of Active Suspension Control for Quadrupedal Robots / Domain Randomization for Robust Quadrotor Policy Transfer”，含 Tobin et al. IROS 2017 | Metadata mismatch | Tobin et al. 的视觉域随机化论文真实存在，但不是四旋翼动力学随机化，也不支持旧报告列出的 UAV 参数范围和 DJI/编队结果；“active suspension/quadruped”表述与条目混杂。 | Tobin DOI: https://doi.org/10.1109/IROS.2017.8202133；arXiv: https://arxiv.org/abs/1703.06907；Peng dynamics randomization DOI: https://doi.org/10.1109/ICRA.2018.8460528 | 用 Tobin 作为视觉域随机化基础，用 Peng et al. 作为动力学随机化基础；不保留旧报告 UAV 结果。 |
| Loquercio et al., “Learning High-Speed Flight in the Wild” | Verified | Science Robotics 2021 论文真实存在，作者、题名、DOI 和 arXiv 可核验；旧报告中的“多保真课程”和若干精确随机化参数未作为原文明确结论引用。 | DOI: https://doi.org/10.1126/scirobotics.abg5810；arXiv: https://arxiv.org/abs/2110.05113 | 保留为单机高速飞行、模拟训练到真实复杂环境的强锚点；仅引用原文支持的零样本/仿真训练事实。 |
| “Domain Randomization for Simulation-Based Policy Optimization with Application to Quadrotor Attitude Control / Domain Randomization for Robust Quadrotor Policy Transfer”，Eschmann et al. RA-L 2022 | Replaced | 按精确标题、作者和 venue 检索未找到与旧条目一致的 RA-L 2022 论文；旧报告中的五种 DR 策略对比、角度误差和 RARL 结论无可靠来源支撑。 | 未找到旧条目的可靠来源；替代：ADR PMLR https://proceedings.mlr.press/v100/mehta20a.html；Deep Drone Racing DOI https://doi.org/10.1109/TRO.2019.2942989 | 不再引用该旧条目；用 ADR、DROPO、Deep Drone Racing 和综述文献支撑域随机化讨论。 |
| “Scalable Multi-Agent Reinforcement Learning through Intelligent Information Aggregation / MASIA” | Metadata mismatch | 旧报告把两篇不同论文混合：InforMARL 是 Nayak et al. ICML 2023/PMLR；MASIA 是 Guan et al. NeurIPS 2022/OpenReview。旧报告作者、venue 和通信 dropout 数字不匹配。 | InforMARL PMLR: https://proceedings.mlr.press/v202/nayak23a.html；MASIA OpenReview: https://openreview.net/forum?id=n4wnZAdBavx；NeurIPS 页面: https://neurips.cc/virtual/2022/poster/53730 | 分开引用：InforMARL 支撑局部信息聚合和扩展性，MASIA 支撑消息聚合；不把它们作为 UAV Sim2Real 证据。 |
| O'Connell et al., “Neural Fly: Enabling Robust Quadrotor Tracking with In-Context Learning” | Metadata mismatch | 真实题名为 “Neural-Fly enables rapid learning for agile flight in strong winds”，Science Robotics 2022；旧标题和“In-Context Learning”表述不准确。 | DOI: https://doi.org/10.1126/scirobotics.abm6597；arXiv: https://arxiv.org/abs/2205.06908 | 以修正后的 Neural-Fly 引用，用于气动扰动、强风和快速适应；不保留旧标题。 |
| “Multi-UAV Sim-to-Real with Curriculum Learning and Safety-Aware Transfer”，Xue et al. TRO 2023 | Replaced | 未找到与该题名、作者、TRO 2023 完全匹配的论文；旧报告的 Crazyflie N=4 成功率、CBF 课程和碰撞率数字无可靠来源。 | 替代多 UAV 真实部署来源：Chen et al. RAS DOI https://doi.org/10.1016/j.robot.2023.104489；Xie et al. arXiv https://arxiv.org/abs/2410.18495 | 不引用旧条目；用真实多 UAV MARL/编队论文替代，并标注证据强度低于已发表期刊/会议论文。 |
| “Robust Multi-Agent Reinforcement Learning with Communication Uncertainty”，Wang et al. NeurIPS 2023 | Not found | 未在 NeurIPS/OpenReview/arXiv/出版社页面中找到与旧题名、作者和 2023 venue 匹配的论文；旧报告的马尔可夫延迟、丢包率和性能保留数字不可用。 | 无可靠来源；通信现实差距替代背景：Gielis et al. DOI https://doi.org/10.1007/s43154-022-00090-9 | 不用于支撑结论；通信建模改写为工程压力测试需求，而非引用旧论文参数。 |
| Tiboni et al. “DROPO / Adaptive Domain Randomization via Regret Minimization for Zero-Shot Sim-to-Real Transfer” | Metadata mismatch | DROPO 真实存在，但题名、作者、venue 与旧报告不一致。真实论文为 Tiboni, Arndt, Kyrki，Robotics and Autonomous Systems 2023；不是 CoRL 2022，也不是 UAV 论文。 | DOI: https://doi.org/10.1016/j.robot.2023.104432；arXiv: https://arxiv.org/abs/2201.08434 | 以修正后 DROPO 引用为离线域随机化/系统辨识启发；不声称其验证 UAV。 |
| “Safety-Constrained Reinforcement Learning for Multi-UAV Formation: From Simulation to Real Flight”，Liu et al. T-ITS 2024 | Not found | 未找到与旧题名、作者、T-ITS 2024 匹配的可靠来源；旧报告关于逐步收紧安全距离、DJI Phantom 实飞和 0% 碰撞率的说法不可支撑。 | 无可靠来源；安全学习替代背景：Safe-Control-Gym DOI https://doi.org/10.1109/LRA.2022.3196132；Safe learning review DOI https://doi.org/10.1146/annurev-control-042920-020211 | 删除旧结论；保留“分阶段安全验证”作为 SafeComm 工程建议，但不引用旧论文数字。 |
| “From Simulation to Reality: Bridging the Gap for Multi-Agent Reinforcement Learning in UAV Swarms”，Peng et al. IROS 2023 | Not found | 未找到与旧题名、作者、IROS 2023 匹配的可靠来源；旧报告中的“inter-agent gap”量化数字不可用。 | 无可靠来源；多 UAV 替代来源见 Chen et al. DOI https://doi.org/10.1016/j.robot.2023.104489、Xie et al. https://arxiv.org/abs/2410.18495 | 不用于论证；把“多机交互差距”改写为由 Neural-Swarm 和多 UAV 任务间接提示的风险。 |
| “Towards Real-World Deployment of Reinforcement Learning for Multi-Robot Systems”，Gu et al. CoRL 2022 | Not found | 未找到与旧题名、作者和 CoRL 2022 匹配的论文；旧报告的 REAL 协议、85% 首次部署成功率和 70% 故障减少不可支撑。 | 无可靠来源；多机器人通信部署综述：Gielis et al. DOI https://doi.org/10.1007/s43154-022-00090-9 | 删除旧论文式引用；用综述和安全学习文献支持部署流程建议。 |

## 3. 经核验后的核心文献综述

### 3.1 UAV Sim2Real 栈的仿真器基础

UAV Sim2Real 研究首先需要可复现、可扩展且能暴露物理差距的仿真环境。Panerati et al. 的 `gym-pybullet-drones` 是本方向最直接的开源锚点：论文提出基于 Bullet/PyBullet 的 Gym 风格多四旋翼环境，支持单机和多机控制/RL 接口，并示范了轨迹跟踪、下洗影响和单/多智能体稳定任务。它适合 SafeComm-MARL 的 Phase 2 级别物理仿真，但核验来源不支持旧报告给出的“默认随机化范围”和“真实迁移成功率”数字。

Flightmare 提供另一类重要栈：它将 Unity 渲染和四旋翼动力学解耦，PMLR 页面明确报告其多模态传感器、RL API 和并行四旋翼仿真能力。若 SafeComm 后续涉及视觉、复杂三维场景或大批量采样，Flightmare 是比纯 PyBullet 更偏感知/渲染的候选；若目标是通信调度、相对状态和安全约束，PyBullet/gym-pybullet-drones 更贴近低层动力学与碰撞压力测试。

Safe-Control-Gym 则不是 UAV Sim2Real 部署论文，而是安全学习/控制 benchmark。它提供安全约束、扰动注入和符号动力学接口，适合作为 SafeComm 中“约束、扰动、可重复压力测试”的设计参照。

### 3.2 域随机化和现实差距：可引用方法，但参数需重新校准

Tobin et al. 的 IROS 2017 论文是视觉域随机化的经典来源，核心思想是通过模拟器视觉外观随机化让真实图像成为训练分布中的一种变体。Peng et al. 的 ICRA 2018 论文把类似思想用于机器人控制的动力学随机化，说明随机化动力学可帮助策略适应模型误差。两者可以支撑“随机化是 Sim2Real 常用策略”，但不能支撑旧报告中的四旋翼质量、惯量、推力系数、通信延迟等精确范围。

后续方法关注如何减少人工调随机化分布。Active Domain Randomization 通过选择更有信息量的环境变化来训练更稳健的策略；DROPO 使用少量离线真实轨迹估计随机化分布，实质上把系统辨识和域随机化结合起来。它们对 SafeComm 的启发是：Phase 2 随机化范围应当先作为设计假设，再通过仿真压力测试和少量真实/硬件在环数据校准，而不是直接照搬旧报告的数值表。

Muratore et al. 的 Frontiers 综述提供了静态、 adaptive 和 adversarial domain randomization 的分类背景。对 SafeComm 写作而言，应使用“域随机化用于覆盖现实差距”“自适应/离线方法可减少人工调参”这类稳健表述，避免写成“某固定范围已被证明最优”。

### 3.3 单 UAV Sim2Real 的强证据

单机无人机的 Sim2Real 证据明显强于多 UAV MARL。Sadeghi and Levine 的 CAD2RL 在 RSS 2017 展示了仅用合成图像训练的单图像飞行策略迁移；Loquercio et al. 的 Deep Drone Racing 在 IEEE T-RO 2020 通过域随机化实现无人机竞速场景的仿真到真实迁移；Loquercio et al. 的 Science Robotics 2021 进一步展示了高速度、机载感知和仿真训练策略在自然/人造复杂场景中的真实飞行。

Neural-Fly 不是传统的“训练一次后零样本部署”论文，而是利用学习到的气动表示和在线适应来处理强风飞行。它对 SafeComm 的意义是：气动扰动、风、载荷和个体差异可能不能只靠固定随机化一次性解决；若目标走向真实 UAV，可能需要系统辨识、在线残差/扰动估计或保守控制器补偿。

这些文献支持 SafeComm 使用“逐级保真度、随机化、真实部署前压力测试”的叙事，但它们多为单机或感知/控制任务，不足以证明多 UAV 通信调度策略可以直接零样本部署。

### 3.4 多 UAV/MARL 证据：存在真实部署线索，但仍是薄弱环节

多 UAV 场景中的现实差距不仅来自单机动力学，还来自相互作用、相对定位、通信和编队几何。Neural-Swarm 在 ICRA 2020 研究近距离多旋翼的分散控制和学习到的交互项，可作为“多机相互作用会影响真实控制”的锚点；但它不是 SafeComm 式通信调度 MARL。

Chen et al. 的 Robotics and Autonomous Systems 2023 论文是较接近多 UAV MARL 与真实 Crazyflie 验证的文献：其任务是多 UAV 对负载抓取点的协同导航，采用 MADDPG、示范学习和课程/注意机制，并报告从 CoppeliaSim 到真实 Crazyflie 的实验。该文可支撑“多 UAV MARL 真实验证存在，但任务特定、规模有限”。Xie et al. 的 2024/2025 arXiv 预印本研究多 UAV 编队、静/动态障碍物和 RL，并报告仿真及真实环境实验；因其是预印本，证据强度应低于已发表期刊/会议论文。

InforMARL 和 MASIA 分别支撑“局部/邻域信息聚合可改善可扩展 MARL”和“消息聚合可提升通信 MARL 表征效率”。但两者不是 UAV Sim2Real 论文，不能作为真实 UAV 迁移证据。它们更适合作为 SafeComm 通信选择、邻域聚合和可扩展策略网络的算法背景。

### 3.5 通信和安全：应作为部署风险建模，而非引用旧报告参数

Gielis, Shankar and Prorok 的多机器人通信综述明确指出，真实机器人网络会遇到消息丢失、异步、乱序接收、去中心化 mesh 拓扑和可靠性不足等问题；这为 SafeComm 在仿真中引入通信压力测试提供了可靠依据。但是，本次核验未找到旧报告所称的 NeurIPS 2023 “Robust MARL with Communication Uncertainty”，也不能引用其固定延迟均值、丢包率或性能保留率。

安全方面，Brunke et al. 的 Annual Review 综述和 Safe-Control-Gym 支持“安全学习需要不确定性、约束、扰动和可比较 benchmark”的总体论点。对 SafeComm 来说，CBF/Lagrangian/安全回退机制可以作为设计方向，但旧报告声称的具体安全距离课程、碰撞率和真实飞行结果应删除。

## 4. 与 SafeComm-VoI / SafeComm-MARL 的关系

**强支持：仿真栈和随机化原则。** `gym-pybullet-drones`、Flightmare、Tobin/Peng 的域随机化、DROPO/ADR 和机器人 Sim2Real 综述支持 SafeComm 采用“先仿真、再随机化、再压力测试/校准”的路线。参数层面只能写成文献启发：质量、惯量、推力/电机响应、传感噪声、风扰、相对定位误差、下洗交互和通信扰动都应纳入候选随机化维度，但具体范围需要由平台规格、系统辨识和消融实验决定。

**间接支持：多 UAV 相互作用和 MARL 结构。** Neural-Swarm、Chen et al. 和 Xie et al. 支持多 UAV 任务中相互作用、课程学习、注意机制和真实验证的必要性；InforMARL/MASIA 支持局部信息聚合和通信消息压缩的算法动机。它们可以帮助 SafeComm 解释“为什么通信选择和邻域信息价值重要”，但不能证明 SafeComm 的通信调度策略已经被现有文献覆盖。

**强支持：通信现实差距需要单独建模。** 多机器人通信综述给出可靠依据：真实链路不是同步、可靠、全连接的抽象通道。SafeComm 的实验应将延迟、丢包、异步/乱序、带宽预算抖动作为压力测试维度，并报告策略在这些维度下的安全约束违反率、任务回报和通信开销。不要把旧报告的马尔可夫延迟或 Gilbert-Elliott 参数写成文献给出的标准值；可将它们作为工程建模选项。

**证据不足：多 UAV SafeComm-MARL 端到端 Sim2Real。** 本次核验未发现直接覆盖“安全约束 + VoI/通信调度 + 多 UAV MARL + PyBullet/真实硬件迁移”的论文。因此 SafeComm 可以保守表述为：现有研究分别覆盖 UAV 仿真器、单机 Sim2Real、域随机化、多机相互作用、通信 MARL 和安全学习，但这些要素在 SafeComm-VoI / SafeComm-MARL 的组合场景中仍缺少系统验证。

建议在论文中采用如下防守性写法：

1. Phase 1 到 Phase 2：从简化动力学到 PyBullet/Flightmare 的迁移，使用文献启发的随机化维度和系统性消融，不引用未核验的精确范围。
2. Phase 2 压力测试：固定随机种子、分离动力学扰动和通信扰动，报告碰撞/约束违反率、编队误差、通信预算利用率和退化曲线。
3. 硬件前验证：硬件在环或日志回放优先；真实飞行只作为未来工作或小规模验证，除非已有真实实验数据。
4. SafeComm 创新定位：不是“首次 UAV Sim2Real”，而是“在安全约束和通信价值选择同时存在时，系统评估通信受限 MARL 的仿真到真实准备度”。

## 5. 可引用参考文献

1. Jacopo Panerati, Hehui Zheng, SiQi Zhou, James Xu, Amanda Prorok, Angela P. Schoellig. “Learning to Fly--a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control.” IROS 2021. DOI: https://doi.org/10.1109/IROS51168.2021.9635857；arXiv: https://arxiv.org/abs/2103.02142
2. Yunlong Song, Selim Naji, Elia Kaufmann, Antonio Loquercio, Davide Scaramuzza. “Flightmare: A Flexible Quadrotor Simulator.” CoRL/PMLR 2021. https://proceedings.mlr.press/v155/song21a.html；arXiv: https://arxiv.org/abs/2009.00563
3. Zhaocong Yuan, Adam W. Hall, Siqi Zhou, Lukas Brunke, Melissa Greeff, Jacopo Panerati, Angela P. Schoellig. “Safe-Control-Gym: A Unified Benchmark Suite for Safe Learning-Based Control and Reinforcement Learning in Robotics.” IEEE RA-L 2022. DOI: https://doi.org/10.1109/LRA.2022.3196132
4. Josh Tobin, Rachel Fong, Alex Ray, Jonas Schneider, Wojciech Zaremba, Pieter Abbeel. “Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World.” IROS 2017. DOI: https://doi.org/10.1109/IROS.2017.8202133；arXiv: https://arxiv.org/abs/1703.06907
5. Xue Bin Peng, Marcin Andrychowicz, Wojciech Zaremba, Pieter Abbeel. “Sim-to-Real Transfer of Robotic Control with Dynamics Randomization.” ICRA 2018. DOI: https://doi.org/10.1109/ICRA.2018.8460528；arXiv: https://arxiv.org/abs/1710.06537
6. Bhairav Mehta, Manfred Diaz, Florian Golemo, Christopher J. Pal, Liam Paull. “Active Domain Randomization.” CoRL/PMLR 2020. https://proceedings.mlr.press/v100/mehta20a.html；arXiv: https://arxiv.org/abs/1904.04762
7. Gabriele Tiboni, Karol Arndt, Ville Kyrki. “DROPO: Sim-to-real transfer with offline domain randomization.” Robotics and Autonomous Systems 2023. DOI: https://doi.org/10.1016/j.robot.2023.104432；arXiv: https://arxiv.org/abs/2201.08434
8. Fabio Muratore, Felix Treede, Michael Gienger, Jan Peters. “Domain Randomization for Simulation-Based Policy Optimization with Transferability Assessment.” CoRL/PMLR 2018. https://proceedings.mlr.press/v87/muratore18a.html
9. Fereshteh Sadeghi, Sergey Levine. “CAD2RL: Real Single-Image Flight Without a Single Real Image.” RSS 2017. DOI: https://doi.org/10.15607/RSS.2017.XIII.034
10. Antonio Loquercio, Elia Kaufmann, René Ranftl, Alexey Dosovitskiy, Vladlen Koltun, Davide Scaramuzza. “Deep Drone Racing: From Simulation to Reality With Domain Randomization.” IEEE T-RO 2020. DOI: https://doi.org/10.1109/TRO.2019.2942989；arXiv: https://arxiv.org/abs/1905.09727
11. Antonio Loquercio, Elia Kaufmann, René Ranftl, Matthias Müller, Vladlen Koltun, Davide Scaramuzza. “Learning high-speed flight in the wild.” Science Robotics 2021. DOI: https://doi.org/10.1126/scirobotics.abg5810；arXiv: https://arxiv.org/abs/2110.05113
12. Michael O'Connell, Guanya Shi, Xichen Shi, Kamyar Azizzadenesheli, Anima Anandkumar, Yisong Yue, Soon-Jo Chung. “Neural-Fly enables rapid learning for agile flight in strong winds.” Science Robotics 2022. DOI: https://doi.org/10.1126/scirobotics.abm6597；arXiv: https://arxiv.org/abs/2205.06908
13. Guanya Shi, Wolfgang Hönig, Yisong Yue, Soon-Jo Chung. “Neural-Swarm: Decentralized Close-Proximity Multirotor Control Using Learned Interactions.” ICRA 2020. DOI: https://doi.org/10.1109/ICRA40945.2020.9196800
14. Jingyu Chen, Ruidong Ma, John Oyekan. “A deep multi-agent reinforcement learning framework for autonomous aerial navigation to grasping points on loads.” Robotics and Autonomous Systems 2023. DOI: https://doi.org/10.1016/j.robot.2023.104489
15. Yuqing Xie, Chao Yu, Hongzhi Zang, Feng Gao, Wenhao Tang, Jingyi Huang, Jiayu Chen, Botian Xu, Yi Wu, Yu Wang. “Multi-UAV Formation Control with Static and Dynamic Obstacle Avoidance via Reinforcement Learning.” arXiv 2024/2025. https://arxiv.org/abs/2410.18495
16. Siddharth Nayak, Kenneth Choi, Wenqi Ding, Sydney Dolan, Karthik Gopalakrishnan, Hamsa Balakrishnan. “Scalable Multi-Agent Reinforcement Learning through Intelligent Information Aggregation.” ICML/PMLR 2023. https://proceedings.mlr.press/v202/nayak23a.html；arXiv: https://arxiv.org/abs/2211.02127
17. Cong Guan, Feng Chen, Lei Yuan, Chenghe Wang, Hao Yin, Zongzhang Zhang, Yang Yu. “Efficient Multi-agent Communication via Self-supervised Information Aggregation.” NeurIPS 2022. OpenReview: https://openreview.net/forum?id=n4wnZAdBavx；NeurIPS: https://neurips.cc/virtual/2022/poster/53730；arXiv extended version: https://arxiv.org/abs/2302.09605
18. Jennifer Gielis, Ajay Shankar, Amanda Prorok. “A Critical Review of Communications in Multi-robot Systems.” Current Robotics Reports 2022. DOI: https://doi.org/10.1007/s43154-022-00090-9
19. Lukas Brunke, Melissa Greeff, Adam W. Hall, Zhaocong Yuan, Siqi Zhou, Jacopo Panerati, Angela P. Schoellig. “Safe Learning in Robotics: From Learning-Based Control to Safe Reinforcement Learning.” Annual Review of Control, Robotics, and Autonomous Systems 2022. DOI: https://doi.org/10.1146/annurev-control-042920-020211
20. Joschka Boedecker, Josiah Hanna, Fabio Muratore, et al. “Robot Learning From Randomized Simulations: A Review.” Frontiers in Robotics and AI 2022. DOI: https://doi.org/10.3389/frobt.2022.799893

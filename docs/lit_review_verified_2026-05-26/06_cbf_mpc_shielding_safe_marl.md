# 方向6：CBF / MPC / Shielding for Safe MARL

核验日期：2026-05-26

## 1. 核验结论摘要

本方向从旧报告中抽取 14 个论文式候选条目重新核验。结果为：Verified 2 篇，Metadata mismatch 9 项，Weak match 1 项，Not found 1 项，Replaced 1 项；另补充 6 篇经核验的真实锚点文献，用于支撑 CBF/HOCBF、predictive safety filter、multi-agent shielding、graph/neural CBF 与 safe MARL 的重写综述。

旧报告的主题判断需要明显降级：CBF/CBF-QP/HOCBF 对“执行层即时安全过滤”和“多机器人避碰”有强支持；MPC/predictive safety filter 对“基于模型的安全过滤”有强支持但通信和在线优化成本更高；shielding 对有限状态或抽象模型下的规范满足有强支持。旧报告中关于“CBF 值驱动通信调度”“稀疏通信下 MACBF 保守回退”“无通信 UAV swarms TCNS 2024”“Hasanbeig hybrid multi-agent shielding”的若干具体说法未被可靠来源支持，不能作为结论依据。对 SafeComm-VoI / SafeComm-MARL 而言，现有文献只能间接支持“安全风险信号可用于优先关注危险邻居”，尚不能直接证明“学习通信调度 + CBF 安全层”的完整方案已有或已被验证。

## 2. 原文献真实性审计表

| 原报告条目 | 状态 | 核验结果 | 可靠来源 | 处理方式 |
|---|---|---|---|---|
| Ames et al. “Safe Reinforcement Learning via Control Barrier Functions / CBF 基础框架”，旧称 CDC 2019 + TAC 2019 | Metadata mismatch | 相关真实论文是 Ames et al. “Control Barrier Functions: Theory and Applications”，ECC 2019；另有 Ames et al. “Control Barrier Function Based Quadratic Programs for Safety Critical Systems”，IEEE TAC 2017。旧报告把 venue 和题名混合了。 | ECC 2019 DOI: https://doi.org/10.23919/ECC.2019.8796030；TAC 2017 DOI: https://doi.org/10.1109/TAC.2016.2638961；arXiv: https://arxiv.org/abs/1903.11199 | 使用修正后的 ECC 2019 与 TAC 2017 作为 CBF 理论和 CBF-QP 锚点。 |
| Qin et al. “Multi-Agent Safety via Control Barrier Functions”，旧称 RA-L/ICRA 2023，作者 Zhengyu Qin/Xiaoxiao Su/Banghua Zhu/Changliu Liu | Replaced | 未找到与旧题名、作者和 venue 同时匹配的可靠记录。最接近的真实高质量工作是 Qin et al. ICLR 2021 “Learning Safe Multi-agent Control with Decentralized Neural Barrier Certificates”，以及 Zhang/Garg/Fan CoRL 2023 graph CBF。 | OpenReview ICLR 2021: https://openreview.net/forum?id=P6_q1BRxY8Q；PMLR CoRL 2023: https://proceedings.mlr.press/v229/zhang23h.html | 删除旧元数据；用 Qin 2021 和 Zhang 2023 支撑 decentralized/neural/graph CBF 论述。 |
| “Learning Safe Policies with Expert Guidance”，旧称 Könighofer/Bloem/Pranger/Hahn，NeurIPS 2023，Shielding + MARL | Metadata mismatch | 题名真实存在，但为 Huang/Wu/Precup/Cai 的 NeurIPS 2018 论文，主题是 expert guidance 下的安全策略学习，不是 Könighofer 等人的 shielding/MARL。 | NeurIPS 2018: https://papers.nips.cc/paper_files/paper/2018/hash/a89b71bb5227c75d463dd82a03115738-Abstract.html | 不用于 shielding/MARL 论证；shielding 改用 Alshiekh 2018、ElSayed-Aly 2021 和 Bharadwaj 2019。 |
| Wabersich & Zeilinger predictive safety filter，旧称 2018 CDC + 2021 IEEE TAC | Metadata mismatch | “A predictive safety filter for learning-based control of constrained nonlinear dynamical systems” 真实发表于 Automatica 129, 2021；2018 相关论文为 “Linear Model Predictive Safety Certification for Learning-Based Control”，CDC 2018。旧报告的 IEEE TAC 信息错误。 | Automatica DOI: https://doi.org/10.1016/j.automatica.2021.109597；arXiv: https://arxiv.org/abs/1812.05506；CDC 2018 DOI: https://doi.org/10.1109/CDC.2018.8619829 | 使用 Automatica 2021 作为 predictive safety filter 核心来源，CDC 2018 作为前身。 |
| Dawson et al. “Safe Nonlinear Control Using Robust Neural Lyapunov-Barrier Functions”，旧称 RA-L/ICRA 2023 | Metadata mismatch | 真实记录为 CoRL 2021 论文，PMLR volume 164 于 2022 发布；不是 RA-L/ICRA 2023。 | PMLR: https://proceedings.mlr.press/v164/dawson22a.html；arXiv: https://arxiv.org/abs/2109.06697 | 使用修正后的 CoRL/PMLR 元数据；仅支撑 neural CLBF/rCLBF，不支撑通信调度。 |
| “MACBF: Multi-Agent CBF for Safe Multi-Robot Systems”，旧称 Qin et al. T-RO/ICRA 2021，并声称 20% 稀疏通信和保守 fallback | Metadata mismatch | 找到真实相近论文 Qin et al. ICLR 2021 “Learning Safe Multi-agent Control with Decentralized Neural Barrier Certificates”。旧报告的 T-RO/ICRA venue、MACBF 标题、稀疏通信比例和 fallback 细节未在可靠来源中核验。 | OpenReview ICLR 2021: https://openreview.net/forum?id=P6_q1BRxY8Q | 使用 Qin 2021 的真实表述：联合学习 policy 与 decentralized neural barrier certificates；删除未核验的稀疏通信数字和 fallback 说法。 |
| Elsayed-Aly et al. “Safe Exploration in Multi-Agent Systems via Shielding”，旧称 AAMAS 2021 | Metadata mismatch | 真实题名是 “Safe Multi-Agent Reinforcement Learning via Shielding”，AAMAS 2021；作者为 Ingy ElSayed-Aly, Suda Bharadwaj, Christopher Amato, Rüdiger Ehlers, Ufuk Topcu, Lu Feng。 | AAMAS accepted papers: https://aamas2021.soton.ac.uk/programme/accepted-papers/；arXiv: https://arxiv.org/abs/2101.11196；PDF: https://www.cs.virginia.edu/~lufeng/papers/aamas2021.pdf | 用修正后的题名和作者支撑 multi-agent shielding。 |
| “Safe MARL with Decentralized CBF for UAV Swarms”，旧称 Zhao et al., IEEE TCNS 2024 | Not found | 用精确题名、关键词、作者组合和 venue 检索，未找到可靠匹配；旧报告中“完全无通信 UAV swarm + CBF + ORCA fallback + TCNS 2024”的组合不能核验。 | 定向检索未发现可靠来源；相近 UAV/CBF 应用如 multi-UAV target search 另有独立论文：https://doi.org/10.1177/01423312231158677 | 不进入可引用参考文献；不支撑结论。 |
| Dalal et al. “Safe Exploration in Continuous Action Spaces”，旧称 ICML Workshop 2018 / NeurIPS 2021 完整版 | Metadata mismatch | 真实可核验来源为 arXiv/CoRR 2018；未核验到 NeurIPS 2021 完整版。论文提出连续动作空间 safety layer，并非多智能体通信调度工作。 | arXiv: https://arxiv.org/abs/1801.08757；arXiv DOI: https://doi.org/10.48550/arXiv.1801.08757 | 作为 safety layer 早期来源引用，删除 NeurIPS 2021 说法。 |
| Wang/Ames/Egerstedt “Distributed CBF for Multi-Agent Systems with Partial Information”，旧称 L4DC/TAC 2017 | Metadata mismatch | 真实核心论文是 “Safety Barrier Certificates for Collisions-Free Multirobot Systems”，IEEE T-RO 2017。旧报告的 L4DC/TAC、partial information、通信延迟缓冲公式等表述未核验。 | DOI: https://doi.org/10.1109/TRO.2017.2659727；CaltechAUTHORS: https://authors.library.caltech.edu/records/tshzw-g4v69 | 使用 T-RO 2017 支撑多机器人 CBF-QP/去中心化避碰；删除未核验的通信延迟公式。 |
| Cheng/Khojasteh probabilistic CBF，旧称 Cheng NeurIPS 2019、Khojasteh CDC/L4DC 2020 且 Khojasteh 作者含 Martins | Metadata mismatch | Cheng et al. 真实发表于 AAAI 2019；Khojasteh et al. 真实发表于 L4DC/PMLR 2020，作者为 Mohammad Javad Khojasteh, Vikas Dhiman, Massimo Franceschetti, Nikolay Atanasov。旧报告 venue 和作者有误。 | Cheng AAAI DOI: https://doi.org/10.1609/aaai.v33i01.33013387；AAAI page: https://ojs.aaai.org/index.php/AAAI/article/view/4213；Khojasteh PMLR: https://proceedings.mlr.press/v120/khojasteh20a.html | 使用修正后的 AAAI 2019 和 PMLR 2020；仅支撑不确定性/概率 CBF。 |
| Hasanbeig et al. “Shielded Reinforcement Learning for Hybrid Multi-Agent Systems”，旧称 AAAI Workshop 2021 | Weak match | 未找到旧题名的可靠记录。真实相近条目包括 Hasanbeig 等参与的 “Shielding Atari Games with Bounded Prescience”，AAMAS 2021，以及 Hasanbeig et al. “DeepSynth: Automata Synthesis for Automatic Task Segmentation in Deep Reinforcement Learning”，AAAI 2021；二者均不等同于旧条目。 | AAMAS PDF: https://www.ifaamas.org/Proceedings/aamas2021/pdfs/p1507.pdf；arXiv: https://arxiv.org/abs/2101.08153；DBLP DeepSynth: https://dblp.org/rec/conf/aaai/HasanbeigJAMK21 | 不作为 hybrid multi-agent shielding 证据；如需引用 Hasanbeig，只能按真实题名限域引用。 |
| Gu et al. safe RL survey，旧称 “A Survey on Safe Reinforcement Learning: Methods, Theory and Applications”，arXiv:2205.10330 | Verified | 真实题名为 “A Review of Safe Reinforcement Learning: Methods, Theory and Applications”，作者和 arXiv 号基本匹配；旧报告把 Survey/Review 题名写法略有出入。 | arXiv: https://arxiv.org/abs/2205.10330；arXiv DOI: https://doi.org/10.48550/arXiv.2205.10330 | 可作为背景综述引用，但不支撑本方向的具体算法贡献。 |
| Ames/Xu/Grizzle/Tabuada “Control Barrier Function Based Quadratic Programs for Safety Critical Systems”，IEEE TAC 2017 | Verified | 题名、作者、IEEE TAC 2017、卷期页码和 DOI 均可核验。旧报告参考文献中页码写为 3861-3876，真实为 IEEE TAC 62(8):3861-3876。 | DOI: https://doi.org/10.1109/TAC.2016.2638961；arXiv: https://arxiv.org/abs/1609.06408 | 作为 CBF-QP 强证据引用。 |

## 3. 经核验后的核心文献综述

### 3.1 CBF/CBF-QP/HOCBF：执行层安全过滤的强证据

CBF 文献的主干是“把安全集合的前向不变性转化为对控制输入的即时约束”。Ames/Xu/Grizzle/Tabuada 的 IEEE TAC 2017 论文给出 CBF 与 CLF-CBF-QP 的标准框架；Ames/Coogan/Egerstedt/Notomista/Sreenath/Tabuada 的 ECC 2019 论文进一步系统化了 CBF 理论和应用。对 SafeComm-MARL 这类碰撞避免问题，CBF-QP 的价值在于它是执行时过滤器：在模型、状态和可行性假设成立时，它比期望约束或训练期惩罚更接近“每一步不越界”的需求。

多机器人方向上，Wang/Ames/Egerstedt 的 T-RO 2017 “Safety Barrier Certificates for Collisions-Free Multirobot Systems” 是可靠锚点：论文用 QP 最小修改 nominal controller，并讨论集中式到去中心化的多机器人 collision-free 控制。这个证据支持“多智能体避碰可用 CBF/QP 实时执行过滤”，但并不支持旧报告中关于通信延迟缓冲公式、partial information 或 L4DC/TAC 的具体说法。

HOCBF 用于相对阶高于一的约束。Xiao/Belta 的 IEEE TAC 2022 “High-Order Control Barrier Functions” 可作为 HOCBF 锚点；Nguyen/Sreenath 的 ACC 2016 exponential CBF 也是真实前身。对 UAV/quadrotor 这类高阶动力学，HOCBF/ECBF 比一阶 CBF 更贴近真实执行约束，因此“CBF/HOCBF 对执行安全强支持”是可防守的；但它仍要求状态、动力学和约束相对阶处理准确。

### 3.2 Neural / graph CBF 与 safe MARL：存在直接交叉，但通信调度仍是间接证据

Qin/Zhang/Chen/Chen/Fan 的 ICLR 2021 “Learning Safe Multi-agent Control with Decentralized Neural Barrier Certificates” 是真实的 safe multi-agent control + neural barrier certificate 工作。它支持“可联合学习多智能体控制策略和去中心化 barrier certificate，并在局部/邻居结构上扩展到更多智能体”的论述。旧报告把它改写为 “MACBF T-RO/ICRA 2021” 并加入稀疏通信比例和保守 fallback，这些细节没有核验来源，必须删除。

Zhang/Garg/Fan 的 CoRL 2023 “Neural Graph Control Barrier Functions Guided Distributed Collision-avoidance Multi-agent Control” 进一步把 CBF certificate 与图结构结合，强调 large-scale distributed collision avoidance、local information、GNN generalization 和点云/LiDAR 输入。它对 SafeComm 的启发比旧报告中的虚构 MACBF 更可靠：危险邻居的局部图结构确实是 CBF-based distributed control 的核心输入。但该论文并未提出通信预算下的主动通信调度，因此对“学习通信调度”的支持只能标为间接。

Liu/Liu/Yu 的 Information Sciences 2025 “Safe robust multi-agent reinforcement learning with neural control barrier functions and safety attention mechanism” 是核验到的更直接 safe MARL + neural CBF + attention 工作。它明确处理 time-varying observable agents、partial observability 和 safety attention。该文献可以支持“安全注意力/危险邻居加权是 safe MARL 中合理的局部信息处理方式”；但它仍不是 SafeComm 式“在通信预算下选择哪些链路传输”的直接来源。

Dawson/Qin/Gao/Fan 的 CoRL/PMLR 2022 robust neural Lyapunov-barrier functions 和 Yang/Jiang/Liu/Chen/Li 的 RA-L 2023 neural barrier certificate 工作说明，学习型 barrier certificate 可以降低手工设计负担，并与安全 RL 结合。不过这类工作通常把安全证书用于控制或策略更新，不直接解决多智能体通信链路选择。

### 3.3 MPC / predictive safety filter：强安全过滤证据，但通信和计算成本更高

Wabersich/Zeilinger 的 Automatica 2021 predictive safety filter 是 model predictive safety filter 的核心来源：安全过滤器检查学习控制器给出的输入是否可安全执行，不安全时用 MPC 形式的 backup trajectory 修改输入。它支持“安全层可作为 RL 外挂模块，且可在模型不确定下给出概率安全分析”的结论。

与 CBF-QP 相比，predictive safety filter/MPC 的优势是可显式处理未来约束和终端安全集合；代价是每步解有限时域优化问题。在多智能体场景中，如果安全约束依赖其他智能体未来轨迹，通信需求会从“当前邻居状态”上升到“邻居预测轨迹或假设模型”。本次核验没有把这一点作为某篇分布式 MPC 论文的直接结论，而是从 predictive safety filter 的算法结构对 SafeComm 场景作保守推断。

Dalal et al. 的 arXiv 2018 “Safe Exploration in Continuous Action Spaces” 是 safety layer 早期工作，通过学习线性化约束模型并解析修正连续动作。它可用于说明“把策略输出投影到安全可行动作集合”是 safe RL 中真实存在的路线；但由于其可核验版本为 arXiv，且不是 MARL/通信调度论文，证据强度低于 CBF-QP 和 Wabersich/Zeilinger。

### 3.4 Shielding：形式化硬约束强，但依赖抽象模型和状态可见性

Alshiekh/Bloem/Ehlers/Könighofer/Niekum/Topcu 的 AAAI 2018 “Safe Reinforcement Learning via Shielding” 是 runtime shielding 的核心来源。该路线用 temporal logic specification 合成 shield，监控 learner action，并在可能违反规范时修正动作。它对“训练期和执行期都可插入安全屏蔽层”有强支持，尤其适合离散或可抽象系统。

ElSayed-Aly/Bharadwaj/Amato/Ehlers/Topcu/Feng 的 AAMAS 2021 “Safe Multi-Agent Reinforcement Learning via Shielding” 是本方向真正的 multi-agent shielding 锚点。它提出 centralized shielding 和 factored shielding，目标是在 MARL 学习过程中保证不访问 unsafe states，并通过因式化提高可扩展性。这支持“multi-agent shielding 可提供模型内硬安全”的结论；同时也提示它依赖联合状态/因式状态、规范和 shield synthesis，不自然解决连续 UAV 动力学下的通信预算优化。

Bharadwaj/Bloem/Dimitrova/Könighofer/Topcu 的 ACC 2019 “Synthesis of Minimum-Cost Shields for Multi-agent Systems” 是多智能体 shield synthesis 的控制/形式化方法锚点，可用于补充 AAMAS 2021 的背景。Hasanbeig 相关旧条目不能支撑 hybrid multi-agent shielding 结论；若要引用 Hasanbeig，只能限于真实题名，如 AAMAS 2021 Atari shielding 或 AAAI 2021 DeepSynth。

## 4. 与 SafeComm-VoI / SafeComm-MARL 的关系

**强支持：执行层安全过滤。** CBF-QP、HOCBF、safety barrier certificates、predictive safety filter 和 shielding 都证明了一个共同事实：在模型、状态和规范假设满足时，学习策略之外可以叠加独立安全层。SafeComm-MARL 若要增强碰撞避免可信度，可把通信调度策略与执行层 CBF/HOCBF filter 或 predictive safety filter 分开陈述：学习模块负责性能/通信效率，安全 filter 负责最后一层动作修正。

**间接支持：危险邻居优先。** 多机器人 CBF、decentralized neural barrier certificates、graph CBF 和 safety attention 都表明，邻居关系、局部图结构和危险程度是安全控制的关键输入。SafeComm-VoI 如果用 CBF margin、time-to-collision、barrier residual 或 uncertainty margin 估计通信价值，可以引用这些工作作为“风险相关局部信息重要”的间接证据。但必须避免写成“已有文献证明 CBF 值驱动通信调度最优”，本次核验没有发现这样的直接来源。

**证据不足：学习通信调度的直接先例。** 本次核验未发现可靠文献同时满足三点：多智能体 RL、CBF/MPC/shielding 执行安全、通信预算下主动学习或优化链路调度。因此 SafeComm 的创新定位可以保守写成：“将安全证书/风险 margin 用作通信价值信号，并在通信受限 MARL 中主动分配邻居信息，本次核验未发现有同一机制的直接覆盖工作。”

**设计建议。** 对 SafeComm-VoI / SafeComm-MARL，建议把保证层级拆清楚：CBF/HOCBF filter 在准确动力学和及时邻居状态下给出强执行安全；通信调度只决定哪些邻居状态被及时更新，因此对硬安全是信息前提而非自动保证；若邻居状态缺失，应把保证降为不确定性/概率/保守估计下的条件保证。MPC filter 可作为高安全但高通信/计算开销对照；shielding 可作为离散规范或简化网格环境对照，不宜直接当作连续 UAV 主基线。

## 5. 可引用参考文献

1. Ames, A. D., Xu, X., Grizzle, J. W., & Tabuada, P. (2017). Control Barrier Function Based Quadratic Programs for Safety Critical Systems. IEEE Transactions on Automatic Control, 62(8), 3861-3876. https://doi.org/10.1109/TAC.2016.2638961
2. Ames, A. D., Coogan, S., Egerstedt, M., Notomista, G., Sreenath, K., & Tabuada, P. (2019). Control Barrier Functions: Theory and Applications. 2019 18th European Control Conference (ECC), 3420-3431. https://doi.org/10.23919/ECC.2019.8796030
3. Wang, L., Ames, A. D., & Egerstedt, M. (2017). Safety Barrier Certificates for Collisions-Free Multirobot Systems. IEEE Transactions on Robotics, 33(3), 661-674. https://doi.org/10.1109/TRO.2017.2659727
4. Xiao, W., & Belta, C. (2022). High-Order Control Barrier Functions. IEEE Transactions on Automatic Control, 67(7), 3655-3662. https://doi.org/10.1109/TAC.2021.3105491
5. Nguyen, Q., & Sreenath, K. (2016). Exponential Control Barrier Functions for Enforcing High Relative-Degree Safety-Critical Constraints. 2016 American Control Conference (ACC), 322-328. https://doi.org/10.1109/ACC.2016.7524935
6. Wabersich, K. P., & Zeilinger, M. N. (2021). A Predictive Safety Filter for Learning-Based Control of Constrained Nonlinear Dynamical Systems. Automatica, 129, 109597. https://doi.org/10.1016/j.automatica.2021.109597
7. Wabersich, K. P., & Zeilinger, M. N. (2018). Linear Model Predictive Safety Certification for Learning-Based Control. 2018 IEEE Conference on Decision and Control (CDC). https://doi.org/10.1109/CDC.2018.8619829
8. Dalal, G., Dvijotham, K., Vecerik, M., Hester, T., Paduraru, C., & Tassa, Y. (2018). Safe Exploration in Continuous Action Spaces. arXiv:1801.08757. https://arxiv.org/abs/1801.08757
9. Cheng, R., Orosz, G., Murray, R. M., & Burdick, J. W. (2019). End-to-End Safe Reinforcement Learning through Barrier Functions for Safety-Critical Continuous Control Tasks. Proceedings of the AAAI Conference on Artificial Intelligence, 33(01), 3387-3395. https://doi.org/10.1609/aaai.v33i01.33013387
10. Khojasteh, M. J., Dhiman, V., Franceschetti, M., & Atanasov, N. (2020). Probabilistic Safety Constraints for Learned High Relative Degree System Dynamics. Proceedings of the 2nd Conference on Learning for Dynamics and Control, PMLR 120:781-792. https://proceedings.mlr.press/v120/khojasteh20a.html
11. Cheng, R., Khojasteh, M. J., Ames, A. D., & Burdick, J. W. (2020). Safe Multi-Agent Interaction through Robust Control Barrier Functions with Learned Uncertainties. 59th IEEE Conference on Decision and Control (CDC 2020). https://arxiv.org/abs/2004.05273
12. Qin, Z., Zhang, K., Chen, Y., Chen, J., & Fan, C. (2021). Learning Safe Multi-agent Control with Decentralized Neural Barrier Certificates. ICLR 2021. https://openreview.net/forum?id=P6_q1BRxY8Q
13. Dawson, C., Qin, Z., Gao, S., & Fan, C. (2022). Safe Nonlinear Control Using Robust Neural Lyapunov-Barrier Functions. Proceedings of the 5th Conference on Robot Learning, PMLR 164:1724-1735. https://proceedings.mlr.press/v164/dawson22a.html
14. Zhang, S., Garg, K., & Fan, C. (2023). Neural Graph Control Barrier Functions Guided Distributed Collision-avoidance Multi-agent Control. Proceedings of The 7th Conference on Robot Learning, PMLR 229:2373-2392. https://proceedings.mlr.press/v229/zhang23h.html
15. Liu, S., Liu, L., & Yu, Z. (2025). Safe Robust Multi-Agent Reinforcement Learning with Neural Control Barrier Functions and Safety Attention Mechanism. Information Sciences, 690, 121567. https://doi.org/10.1016/j.ins.2024.121567
16. Yang, Y., Jiang, Y., Liu, Y., Chen, J., & Li, S. E. (2023). Model-Free Safe Reinforcement Learning Through Neural Barrier Certificate. IEEE Robotics and Automation Letters, 8(3), 1295-1302. https://doi.org/10.1109/LRA.2023.3238656
17. Alshiekh, M., Bloem, R., Ehlers, R., Könighofer, B., Niekum, S., & Topcu, U. (2018). Safe Reinforcement Learning via Shielding. Proceedings of the AAAI Conference on Artificial Intelligence, 32(1). https://doi.org/10.1609/aaai.v32i1.11797
18. ElSayed-Aly, I., Bharadwaj, S., Amato, C., Ehlers, R., Topcu, U., & Feng, L. (2021). Safe Multi-Agent Reinforcement Learning via Shielding. AAMAS 2021. https://arxiv.org/abs/2101.11196
19. Bharadwaj, S., Bloem, R., Dimitrova, R., Könighofer, B., & Topcu, U. (2019). Synthesis of Minimum-Cost Shields for Multi-Agent Systems. 2019 American Control Conference (ACC), 1048-1055. https://doi.org/10.23919/ACC.2019.8815233
20. Gu, S., Yang, L., Du, Y., Chen, G., Walter, F., Wang, J., & Knoll, A. (2024 revision). A Review of Safe Reinforcement Learning: Methods, Theory and Applications. arXiv:2205.10330. https://arxiv.org/abs/2205.10330

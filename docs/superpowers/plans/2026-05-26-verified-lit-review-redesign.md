# Verified Literature Review Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new verified SafeComm literature-review set under `docs/lit_review_verified_2026-05-26/` without modifying the original `docs/lit_review/` reports.

**Architecture:** The work is document-first and source-driven. Each direction report is produced independently from its matching original file, using a common audit table, conservative verification labels, verified thematic synthesis, and traceable references; the synthesis report is written only after all eight direction reports exist.

**Tech Stack:** Markdown, shell tools (`rg`, `wc`, `git diff`, `git status`), web literature sources, Crossref/DOI, arXiv, OpenReview, PMLR, official conference/proceedings pages, publisher pages, Semantic Scholar, OpenAlex.

---

## File Structure

- Create: `docs/lit_review_verified_2026-05-26/`
  - New output directory. It contains only the 9 verified Markdown reports for this review pass.
- Create: `docs/lit_review_verified_2026-05-26/01_safe_marl.md`
  - Audits and rewrites safe RL/MARL, CMDP, Lagrangian, CPO/MACPO, safe benchmark, and safe policy literature.
- Create: `docs/lit_review_verified_2026-05-26/02_comm_constrained_marl.md`
  - Audits and rewrites learned communication, bandwidth-limited communication, graph communication, scheduling, and communication-efficiency MARL literature.
- Create: `docs/lit_review_verified_2026-05-26/03_safety_comm_intersection.md`
  - Audits and rewrites literature that jointly or partially connects safety constraints with communication limits, including networked safe control.
- Create: `docs/lit_review_verified_2026-05-26/04_uav_formation.md`
  - Audits and rewrites multi-UAV or multi-robot formation, flocking, collision avoidance, and MARL-control literature.
- Create: `docs/lit_review_verified_2026-05-26/05_multi_objective_pareto_marl.md`
  - Audits and rewrites multi-objective RL/MARL, Pareto, constrained RL, and safe multi-objective literature.
- Create: `docs/lit_review_verified_2026-05-26/06_cbf_mpc_shielding_safe_marl.md`
  - Audits and rewrites CBF, MPC safety filters, safety certificates, shielding, and safe learning/control literature.
- Create: `docs/lit_review_verified_2026-05-26/07_graph_attention_scalable_marl.md`
  - Audits and rewrites graph neural MARL, attention-based MARL, transformer MARL, mean-field MARL, and scalable aggregation literature.
- Create: `docs/lit_review_verified_2026-05-26/08_sim2real_uav_marl.md`
  - Audits and rewrites UAV/quadcopter sim-to-real, domain randomization, PyBullet/Gazebo/AirSim, communication uncertainty, and deployment workflow literature.
- Create: `docs/lit_review_verified_2026-05-26/00_synthesis_report.md`
  - Summarizes evidence strength, high-risk original citations, revised SafeComm positioning, and recommended Related Work framing.
- Read only: `docs/lit_review/*.md`
  - Original reports are inputs only and must not be modified.
- Read: `docs/superpowers/specs/2026-05-26-verified-lit-review-redesign.md`
  - Governing design spec.

## Common Direction Report Template

Every direction report must use this exact section structure:

```markdown
# 方向X：方向名称

## 1. 核验结论摘要

## 2. 原文献真实性审计表

## 3. 经核验后的核心文献综述

## 4. 与 SafeComm-VoI / SafeComm-MARL 的关系

## 5. 可引用参考文献
```

Use these verification labels in every audit table: `Verified`, `Metadata mismatch`, `Weak match`, `Not found`, `Replaced`.

Use this audit table schema:

```markdown
| 原报告条目 | 状态 | 核验结果 | 可靠来源 | 处理方式 |
|---|---|---|---|---|
```

Use this source priority: DOI/Crossref or publisher page first; arXiv second; OpenReview/PMLR/conference pages third; IEEE/ACM/Springer/ScienceDirect/JMLR/AAAI/AAMAS/IJCAI pages fourth; Semantic Scholar/OpenAlex fifth. Google Scholar is discovery-only and cannot be the sole source.

## Task 1: Prepare Output Directory and Guard Original Inputs

**Files:**
- Create: `docs/lit_review_verified_2026-05-26/`
- Read: `docs/lit_review/*.md`
- Read: `docs/superpowers/specs/2026-05-26-verified-lit-review-redesign.md`

- [ ] **Step 1: Create the output directory**

Run:

```bash
mkdir -p docs/lit_review_verified_2026-05-26
```

Expected: command exits with code 0.

- [ ] **Step 2: Print the expected output files**

Run:

```bash
printf '%s\n' \
  docs/lit_review_verified_2026-05-26/00_synthesis_report.md \
  docs/lit_review_verified_2026-05-26/01_safe_marl.md \
  docs/lit_review_verified_2026-05-26/02_comm_constrained_marl.md \
  docs/lit_review_verified_2026-05-26/03_safety_comm_intersection.md \
  docs/lit_review_verified_2026-05-26/04_uav_formation.md \
  docs/lit_review_verified_2026-05-26/05_multi_objective_pareto_marl.md \
  docs/lit_review_verified_2026-05-26/06_cbf_mpc_shielding_safe_marl.md \
  docs/lit_review_verified_2026-05-26/07_graph_attention_scalable_marl.md \
  docs/lit_review_verified_2026-05-26/08_sim2real_uav_marl.md
```

Expected: exactly 9 paths are printed.

- [ ] **Step 3: Verify old reports are unchanged before writing**

Run:

```bash
git diff -- docs/lit_review
```

Expected: no output.

- [ ] **Step 4: Confirm the design spec is available**

Run:

```bash
test -f docs/superpowers/specs/2026-05-26-verified-lit-review-redesign.md && sed -n '1,140p' docs/superpowers/specs/2026-05-26-verified-lit-review-redesign.md
```

Expected: the command prints the verified literature-review redesign spec beginning with `# 核验版文献调研重建设计`.

## Task 2: Direction 1, Safe MARL

**Files:**
- Read: `docs/lit_review/01_safe_marl.md`
- Create: `docs/lit_review_verified_2026-05-26/01_safe_marl.md`

- [ ] **Step 1: Extract original citation candidates**

Run:

```bash
rg -n "^(### [0-9]+\\.|[0-9]+\\. |### [A-Z]\\.)" docs/lit_review/01_safe_marl.md
```

Expected: output includes MACPO, MAPPO-Lagrangian, MAMPS, MACPPO, HAZARD, Safety Gymnasium, CBF extensions, SAUTE RL, and Safe Networked MARL entries.

- [ ] **Step 2: Verify high-risk original entries with exact searches**

Use web search and source pages for these exact queries:

```text
"MACPO: Constrained Policy Optimization for Multi-Agent Reinforcement Learning"
"MAPPO-Lagrangian" "Safe MAPPO"
"MAMPS" "Safe Multi-Agent Reinforcement Learning via Model Predictive Shielding"
"Multi-Agent Constrained Policy Optimisation" "arXiv:2110.02793"
"HAZARD" "multi-agent safe reinforcement learning"
"Safe Networked Multi-Agent Reinforcement Learning with Chance Constraints"
"Decentralized Safe Reinforcement Learning with Generalization Guarantees"
"Scalable Safe Policy Improvement via Shielding for Multi-Agent Systems"
```

Expected: each entry is labeled `Verified`, `Metadata mismatch`, `Weak match`, `Not found`, or `Replaced`, with at least one source URL recorded for all `Verified`, corrected, or replacement papers.

- [ ] **Step 3: Add verified anchor references**

Use reliable source links for these anchor papers when relevant:

```text
Achiam et al. "Constrained Policy Optimization" ICML 2017 PMLR
Zhang et al. "Multi-Agent Constrained Policy Optimisation" arXiv:2110.02793
Yu et al. "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" NeurIPS 2022
Ji et al. "Safety Gymnasium" NeurIPS 2023 Datasets and Benchmarks
Sootla et al. "SAUTE RL: Almost Surely Safe Reinforcement Learning Using State Augmentation" ICML 2022 PMLR
Gu et al. "A Survey of Safe Reinforcement Learning: Methods, Theory and Applications" arXiv:2205.10330
Altman "Constrained Markov Decision Processes" 1999
```

Expected: retained references have traceable links from PMLR, arXiv, OpenReview, NeurIPS, publisher pages, Semantic Scholar, or OpenAlex.

- [ ] **Step 4: Write the direction report**

Create `docs/lit_review_verified_2026-05-26/01_safe_marl.md` using the common template. The report must explicitly separate safe-MARL claims that are supported by verified literature from original claims that depended on unverified entries.

- [ ] **Step 5: Verify the report**

Run:

```bash
rg -n "Verified|Metadata mismatch|Weak match|Not found|Replaced" docs/lit_review_verified_2026-05-26/01_safe_marl.md
rg -n "https?://|doi.org|arxiv.org|openreview.net|proceedings.mlr.press|semanticscholar.org|openalex.org|ieeexplore.ieee.org|dl.acm.org" docs/lit_review_verified_2026-05-26/01_safe_marl.md
git diff -- docs/lit_review
```

Expected: status labels and source links are present; `git diff -- docs/lit_review` has no output.

- [ ] **Step 6: Commit Direction 1**

Run:

```bash
git add docs/lit_review_verified_2026-05-26/01_safe_marl.md
git commit -m "docs: add verified safe marl review"
```

Expected: commit contains only `docs/lit_review_verified_2026-05-26/01_safe_marl.md`.

## Task 3: Direction 2, Communication-Constrained MARL

**Files:**
- Read: `docs/lit_review/02_comm_constrained_marl.md`
- Create: `docs/lit_review_verified_2026-05-26/02_comm_constrained_marl.md`

- [ ] **Step 1: Extract original citation candidates**

Run:

```bash
rg -n "^(### [0-9]+\\.|### [A-Z]\\.|@inproceedings|@article|@misc)" docs/lit_review/02_comm_constrained_marl.md
```

Expected: output includes CommNet, DIAL, BiCNet, ATOC, SchedNet, DGN, Networked MARL, TarMAC, I2C, NeurComm, MASIA, NDQ, MAGIC, DC2, VBC, MACPO, and Safety Gymnasium.

- [ ] **Step 2: Verify original communication entries with exact searches**

Use web search and source pages for these exact queries:

```text
"Learning Multiagent Communication with Backpropagation"
"Learning to Communicate with Deep Multi-Agent Reinforcement Learning"
"Multiagent Bidirectionally-Coordinated Nets for Learning to Play StarCraft Combat"
"Learning Attentional Communication for Multi-Agent Cooperation"
"Learning to Schedule Communication in Multi-Agent Systems"
"Graph Convolutional Reinforcement Learning" "ICLR 2020"
"Networked Multi-Agent Reinforcement Learning in Continuous Spaces"
"Networked Multi-Agent Reinforcement Learning with Communication"
"TarMAC: Targeted Multi-Agent Communication"
"Learning Individually Inferred Communication for Multi-Agent Cooperation"
"Multi-Agent Graph-Attention Communication and Teaming"
"Efficient Communication in Multi-Agent Reinforcement Learning via Variance Based Control"
"Divide and Conquer Communication for Multi-Agent Reinforcement Learning"
```

Expected: each original entry receives a verification label; communication papers with incorrect venues or titles are corrected before use.

- [ ] **Step 3: Add verified anchor references**

Use reliable source links for these anchor papers when relevant:

```text
Sukhbaatar et al. "Learning Multiagent Communication with Backpropagation" NeurIPS 2016
Foerster et al. "Learning to Communicate with Deep Multi-Agent Reinforcement Learning" NeurIPS 2016
Jiang and Lu "Learning Attentional Communication for Multi-Agent Cooperation" NeurIPS 2018
Kim et al. "Learning to Schedule Communication in Multi-Agent Systems" ICLR 2019
Das et al. "TarMAC: Targeted Multi-Agent Communication" ICML 2019
Jiang et al. "Graph Convolutional Reinforcement Learning" ICLR 2020
Ding et al. "Learning Individually Inferred Communication for Multi-Agent Cooperation" NeurIPS 2020
Niu et al. "Multi-Agent Graph-Attention Communication and Teaming" AAMAS 2021
```

Expected: the verified synthesis can describe learned message content, dynamic communication, top-k scheduling, and graph attention without relying on unverified original entries.

- [ ] **Step 4: Write the direction report**

Create `docs/lit_review_verified_2026-05-26/02_comm_constrained_marl.md` using the common template. The SafeComm relation section must state which verified works handle communication constraints but do not handle safety constraints.

- [ ] **Step 5: Verify the report**

Run:

```bash
rg -n "Verified|Metadata mismatch|Weak match|Not found|Replaced" docs/lit_review_verified_2026-05-26/02_comm_constrained_marl.md
rg -n "https?://|doi.org|arxiv.org|openreview.net|proceedings.mlr.press|semanticscholar.org|openalex.org|dl.acm.org" docs/lit_review_verified_2026-05-26/02_comm_constrained_marl.md
git diff -- docs/lit_review
```

Expected: status labels and source links are present; old reports remain unchanged.

- [ ] **Step 6: Commit Direction 2**

Run:

```bash
git add docs/lit_review_verified_2026-05-26/02_comm_constrained_marl.md
git commit -m "docs: add verified communication marl review"
```

Expected: commit contains only `docs/lit_review_verified_2026-05-26/02_comm_constrained_marl.md`.

## Task 4: Direction 3, Safety and Communication Intersection

**Files:**
- Read: `docs/lit_review/03_safety_comm_intersection.md`
- Create: `docs/lit_review_verified_2026-05-26/03_safety_comm_intersection.md`

- [ ] **Step 1: Extract original citation candidates**

Run:

```bash
rg -n "^(### [0-9]+\\.|### [A-Z]\\.|\\| MACPO|\\| CommNet|\\| VBC|\\| MAGIC|\\| Chen|\\| Zhao)" docs/lit_review/03_safety_comm_intersection.md
```

Expected: output includes the alleged Liu NeurIPS 2023 entry, safe networked MARL, distributed CBF without communication, event-triggered control, and indirect related works.

- [ ] **Step 2: Verify direct-intersection claims first**

Use web search and source pages for these exact queries:

```text
"Multi-Agent Safety via Confidence-Weighted Communication"
"safe communication multi-agent reinforcement learning" "NeurIPS 2023"
"communication-aware safe reinforcement learning" "multi-agent"
"Safe Networked Multi-Agent Reinforcement Learning with Chance Constraints"
"event-triggered control for safety-critical systems" "control barrier function"
"Event-triggered multi-robot formation control" "Dimarogonas"
"Safety Barrier Certificates for Collisions-Free Multirobot Systems"
"distributed control barrier function limited communication multi-agent"
```

Expected: if no reliable source confirms an original direct-intersection paper, it is labeled `Not found` or `Weak match` and cannot support the gap claim.

- [ ] **Step 3: Add verified adjacent references**

Use reliable source links for these adjacent but real bodies of work when relevant:

```text
Wang et al. "Safety Barrier Certificates for Collisions-Free Multirobot Systems" IEEE Transactions on Robotics 2017
Ames et al. "Control Barrier Functions: Theory and Applications" ECC 2019
Tabuada "Event-triggered real-time scheduling of stabilizing control tasks" IEEE TAC 2007
Dimarogonas/Frazzoli/Johansson event-triggered multi-agent agreement or multi-robot control papers
Zhang et al. "Multi-Agent Constrained Policy Optimisation" arXiv:2110.02793
Kim et al. "Learning to Schedule Communication in Multi-Agent Systems" ICLR 2019
```

Expected: the report can discuss evidence-adjacent safety communication ideas while conservatively stating that this review did not verify a direct hard-safety plus hard-communication-budget MARL method.

- [ ] **Step 4: Write the direction report**

Create `docs/lit_review_verified_2026-05-26/03_safety_comm_intersection.md` using the common template. The SafeComm relation section must use conservative wording for the research gap: `本次核验未发现直接同时将安全约束和通信硬预算作为 MARL 联合优化目标的工作。`

- [ ] **Step 5: Verify the report**

Run:

```bash
rg -n "Verified|Metadata mismatch|Weak match|Not found|Replaced" docs/lit_review_verified_2026-05-26/03_safety_comm_intersection.md
rg -n "本次核验未发现|https?://|doi.org|arxiv.org|openreview.net|proceedings.mlr.press|semanticscholar.org|openalex.org|ieeexplore.ieee.org" docs/lit_review_verified_2026-05-26/03_safety_comm_intersection.md
git diff -- docs/lit_review
```

Expected: conservative gap wording and source links are present; old reports remain unchanged.

- [ ] **Step 6: Commit Direction 3**

Run:

```bash
git add docs/lit_review_verified_2026-05-26/03_safety_comm_intersection.md
git commit -m "docs: add verified safety communication intersection review"
```

Expected: commit contains only `docs/lit_review_verified_2026-05-26/03_safety_comm_intersection.md`.

## Task 5: Direction 4, Multi-UAV Formation

**Files:**
- Read: `docs/lit_review/04_uav_formation.md`
- Create: `docs/lit_review_verified_2026-05-26/04_uav_formation.md`

- [ ] **Step 1: Extract original citation candidates**

Run:

```bash
rg -n "^(### [0-9]+\\.|### T[0-9]+\\.|\\[[0-9]+\\])" docs/lit_review/04_uav_formation.md
```

Expected: output includes 15 learning-method entries and traditional baselines such as leader-follower, APF, MPC, Olfati-Saber flocking, and CBF.

- [ ] **Step 2: Verify high-risk UAV/MARL entries first**

Use web search and source pages for these exact queries:

```text
"Multi-Agent Deep Reinforcement Learning for Multi-Robot Formation Control"
"Decentralized Multi-Robot Formation Control with Communication Delay and Collision Avoidance"
"Formation Control of UAV Swarms via Multi-Agent Reinforcement Learning with Safety Constraints"
"Graph Neural Network-Based Multi-UAV Formation Control Under Communication Constraints"
"Safe Reinforcement Learning for Multi-UAV Formation with Control Barrier Functions"
"Communication-Aware Multi-UAV Formation Control via Attention-Based MARL"
"Safe Multi-UAV Formation via Distributed CBF-MARL with Partial Observability"
"Curriculum-Based MARL for Large-Scale UAV Swarm Formation"
"Multi-UAV Target Tracking via Mean Field MARL"
```

Expected: invented, weakly matched, or metadata-mismatched UAV application papers are clearly labeled and excluded from evidence if not verified.

- [ ] **Step 3: Add verified anchor references**

Use reliable source links for these anchor papers when relevant:

```text
Olfati-Saber "Flocking for Multi-Agent Dynamic Systems: Algorithms and Theory" IEEE TAC 2006
Ren and Beard "Distributed Consensus in Multi-vehicle Cooperative Control" Springer book or publisher page
Khatib "Real-Time Obstacle Avoidance for Manipulators and Mobile Robots" IJRR 1986
Dunbar and Murray "Distributed Receding Horizon Control for Multi-Vehicle Formation Stabilization" Automatica or IEEE records
Ames et al. "Control Barrier Functions: Theory and Applications" ECC 2019
Yu et al. "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" NeurIPS 2022
Lowe et al. "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments" NeurIPS 2017
Panerati et al. "Learning to Fly--a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-Agent Quadcopter Control" IROS 2021
```

Expected: the verified review can support conservative baseline and environment choices even if many original UAV-specific papers are not retained.

- [ ] **Step 4: Write the direction report**

Create `docs/lit_review_verified_2026-05-26/04_uav_formation.md` using the common template. The SafeComm relation section must distinguish verified multi-robot/UAV formation foundations from unverified UAV-MARL application claims.

- [ ] **Step 5: Verify the report**

Run:

```bash
rg -n "Verified|Metadata mismatch|Weak match|Not found|Replaced" docs/lit_review_verified_2026-05-26/04_uav_formation.md
rg -n "https?://|doi.org|arxiv.org|openreview.net|proceedings.mlr.press|semanticscholar.org|openalex.org|ieeexplore.ieee.org|link.springer.com" docs/lit_review_verified_2026-05-26/04_uav_formation.md
git diff -- docs/lit_review
```

Expected: status labels and source links are present; old reports remain unchanged.

- [ ] **Step 6: Commit Direction 4**

Run:

```bash
git add docs/lit_review_verified_2026-05-26/04_uav_formation.md
git commit -m "docs: add verified uav formation review"
```

Expected: commit contains only `docs/lit_review_verified_2026-05-26/04_uav_formation.md`.

## Task 6: Direction 5, Multi-Objective and Pareto MARL

**Files:**
- Read: `docs/lit_review/05_multi_objective_pareto_marl.md`
- Create: `docs/lit_review_verified_2026-05-26/05_multi_objective_pareto_marl.md`

- [ ] **Step 1: Extract original citation candidates**

Run:

```bash
rg -n "^(### [0-9]+\\.|[0-9]+\\. )" docs/lit_review/05_multi_objective_pareto_marl.md
```

Expected: output includes MORL framework, MORL/D, Pareto multi-task learning, PGMORL, MO-Safe RL, PCN, MOMALAND, Pareto Q-learning, and safe RL survey entries.

- [ ] **Step 2: Verify original multi-objective entries**

Use web search and source pages for these exact queries:

```text
"A Multi-Objective Deep Reinforcement Learning Framework" "Engineering Applications of Artificial Intelligence"
"MORL/D: Multiobjective Reinforcement Learning Based on Decomposition"
"Pareto Multi-Task Learning" "NeurIPS 2019"
"Prediction-Guided Multi-Objective Reinforcement Learning" "ICML 2020"
"Multi-Objective Safe Reinforcement Learning" "arXiv:2204.06475"
"Pareto Conditioned Networks" "NeurIPS 2022"
"MOMALAND: A Suite of Benchmarks for Multi-Objective Multi-Agent Decision Making"
"Multi-Agent Multi-Objective Reinforcement Learning under Incomplete Information" "JAIR"
"On the Usefulness of Distributional Information for Multi-Objective Multi-Agent Reinforcement Learning"
```

Expected: papers with wrong venues, years, or titles are corrected; unverified “safety-aware MORL” variants are excluded unless a reliable source confirms them.

- [ ] **Step 3: Add verified anchor references**

Use reliable source links for these anchor papers when relevant:

```text
Altman "Constrained Markov Decision Processes" 1999
Achiam et al. "Constrained Policy Optimization" ICML 2017
Roijers et al. "A Survey of Multi-Objective Sequential Decision-Making" JAIR 2013
Lin et al. "Pareto Multi-Task Learning" NeurIPS 2019
Xu et al. "Prediction-Guided Multi-Objective Reinforcement Learning" ICML 2020
Felten et al. "A Toolkit for Reliable Benchmarking and Research in Multi-Objective Multi-Agent Decision Making" or MOMALAND source if verified
```

Expected: the verified review can discuss Lagrangian vs Pareto framing without inventing direct UAV-specific multi-objective MARL evidence.

- [ ] **Step 4: Write the direction report**

Create `docs/lit_review_verified_2026-05-26/05_multi_objective_pareto_marl.md` using the common template. The SafeComm relation section must state whether multi-objective literature directly supports replacing Lagrangian or only supports analysis/visualization of trade-offs.

- [ ] **Step 5: Verify the report**

Run:

```bash
rg -n "Verified|Metadata mismatch|Weak match|Not found|Replaced" docs/lit_review_verified_2026-05-26/05_multi_objective_pareto_marl.md
rg -n "https?://|doi.org|arxiv.org|openreview.net|proceedings.mlr.press|semanticscholar.org|openalex.org|jair.org|link.springer.com" docs/lit_review_verified_2026-05-26/05_multi_objective_pareto_marl.md
git diff -- docs/lit_review
```

Expected: status labels and source links are present; old reports remain unchanged.

- [ ] **Step 6: Commit Direction 5**

Run:

```bash
git add docs/lit_review_verified_2026-05-26/05_multi_objective_pareto_marl.md
git commit -m "docs: add verified multi-objective marl review"
```

Expected: commit contains only `docs/lit_review_verified_2026-05-26/05_multi_objective_pareto_marl.md`.

## Task 7: Direction 6, CBF, MPC, and Shielding for Safe MARL

**Files:**
- Read: `docs/lit_review/06_cbf_mpc_shielding_safe_marl.md`
- Create: `docs/lit_review_verified_2026-05-26/06_cbf_mpc_shielding_safe_marl.md`

- [ ] **Step 1: Extract original citation candidates**

Run:

```bash
rg -n "^(### [0-9]+\\.|[0-9]+\\. )" docs/lit_review/06_cbf_mpc_shielding_safe_marl.md
```

Expected: output includes Ames CBF, Qin multi-agent safety CBF, shielding, Wabersich and Zeilinger safety filter, Dawson neural CBF, MACBF, safety barrier certificates, probabilistic CBF, and safe RL survey entries.

- [ ] **Step 2: Verify original CBF/MPC/shielding entries**

Use web search and source pages for these exact queries:

```text
"Control Barrier Functions: Theory and Applications" "Ames" "2019"
"Control Barrier Function Based Quadratic Programs for Safety Critical Systems"
"Safety Barrier Certificates for Collisions-Free Multirobot Systems"
"Multi-Agent Safety via Control Barrier Functions" "Qin"
"A Predictive Safety Filter for Learning-Based Control of Constrained Nonlinear Dynamical Systems"
"Safe Nonlinear Control Using Robust Neural Lyapunov-Barrier Functions"
"End-to-End Safe Reinforcement Learning through Barrier Functions for Safety-Critical Continuous Control Tasks"
"Safe Exploration in Continuous Action Spaces"
"Safe Multi-Agent Reinforcement Learning via Shielding"
"Learning Safe Policies with Expert Guidance"
```

Expected: entries with wrong venue or conflated authors are corrected; CBF-MARL claims are downgraded if only CBF/control rather than MARL is verified.

- [ ] **Step 3: Add verified anchor references**

Use reliable source links for these anchor papers when relevant:

```text
Ames et al. "Control Barrier Function Based Quadratic Programs for Safety Critical Systems" IEEE TAC 2017
Ames et al. "Control Barrier Functions: Theory and Applications" ECC 2019
Wang, Ames, and Egerstedt "Safety Barrier Certificates for Collisions-Free Multirobot Systems" IEEE T-RO 2017
Wabersich and Zeilinger "A Predictive Safety Filter for Learning-Based Control of Constrained Nonlinear Dynamical Systems" Automatica or IEEE TAC source
Cheng et al. "End-to-End Safe Reinforcement Learning through Barrier Functions for Safety-Critical Continuous Control Tasks" AAAI or arXiv/source
Dalal et al. "Safe Exploration in Continuous Action Spaces" source page
Alshiekh et al. "Safe Reinforcement Learning via Shielding" AAAI 2018
```

Expected: verified references support a careful distinction between hard CBF-style execution filters, predictive safety filters, shielding, and Lagrangian/CMDP methods.

- [ ] **Step 4: Write the direction report**

Create `docs/lit_review_verified_2026-05-26/06_cbf_mpc_shielding_safe_marl.md` using the common template. The SafeComm relation section must explicitly mark CBF/HOCBF support as strong for execution safety but indirect for learned communication scheduling unless a direct scheduling source is verified.

- [ ] **Step 5: Verify the report**

Run:

```bash
rg -n "Verified|Metadata mismatch|Weak match|Not found|Replaced" docs/lit_review_verified_2026-05-26/06_cbf_mpc_shielding_safe_marl.md
rg -n "https?://|doi.org|arxiv.org|openreview.net|proceedings.mlr.press|semanticscholar.org|openalex.org|ieeexplore.ieee.org|aaai.org|link.springer.com" docs/lit_review_verified_2026-05-26/06_cbf_mpc_shielding_safe_marl.md
git diff -- docs/lit_review
```

Expected: status labels and source links are present; old reports remain unchanged.

- [ ] **Step 6: Commit Direction 6**

Run:

```bash
git add docs/lit_review_verified_2026-05-26/06_cbf_mpc_shielding_safe_marl.md
git commit -m "docs: add verified cbf mpc shielding review"
```

Expected: commit contains only `docs/lit_review_verified_2026-05-26/06_cbf_mpc_shielding_safe_marl.md`.

## Task 8: Direction 7, Graph, Attention, and Scalable MARL

**Files:**
- Read: `docs/lit_review/07_graph_attention_scalable_marl.md`
- Create: `docs/lit_review_verified_2026-05-26/07_graph_attention_scalable_marl.md`

- [ ] **Step 1: Extract original citation candidates**

Run:

```bash
rg -n "^(### [0-9]+\\.|[0-9]+\\. )" docs/lit_review/07_graph_attention_scalable_marl.md
```

Expected: output includes DGN, MAAC, MAGIC, Mean Field MARL, G2ANet, INFOMARL, MAT, CoPO, GPG, QGAM, CommNet, SchedNet, ATOC, and Transformer entries.

- [ ] **Step 2: Verify original graph/attention/scalability entries**

Use web search and source pages for these exact queries:

```text
"Graph Convolutional Reinforcement Learning" "ICLR 2020"
"Actor-Attention-Critic for Multi-Agent Reinforcement Learning" "ICML 2019"
"Multi-Agent Graph-Attention Communication and Teaming"
"Mean Field Multi-Agent Reinforcement Learning" "ICML 2018"
"Multi-Agent Game Abstraction via Graph Attention Neural Network"
"Scalable Multi-Agent Reinforcement Learning through Intelligent Information Aggregation"
"Multi-Agent Transformer" "NeurIPS 2022"
"Learning to Simulate Self-Driven Particles System with Coordinated Policy Optimization"
"Leveraging Graph-Based Policy Gradient Methods for Safe Multi-Agent Reinforcement Learning"
"Graph Attention Networks" "ICLR 2018"
```

Expected: title/venue mismatches are corrected; unverified “dynamic graph transformer” placeholder-style entries are excluded unless confirmed by reliable sources.

- [ ] **Step 3: Add verified anchor references**

Use reliable source links for these anchor papers when relevant:

```text
Vaswani et al. "Attention Is All You Need" NeurIPS 2017
Velickovic et al. "Graph Attention Networks" ICLR 2018
Iqbal and Sha "Actor-Attention-Critic for Multi-Agent Reinforcement Learning" ICML 2019 PMLR
Jiang et al. "Graph Convolutional Reinforcement Learning" ICLR 2020
Yang et al. "Mean Field Multi-Agent Reinforcement Learning" ICML 2018 PMLR
Wen et al. "Multi-Agent Transformer" NeurIPS 2022
Nayak et al. "Scalable Multi-Agent Reinforcement Learning through Intelligent Information Aggregation" ICML 2023 PMLR
```

Expected: the verified review can justify multi-head attention and local graph aggregation as plausible architecture support, while avoiding unsupported claims about exact SafeComm gains.

- [ ] **Step 4: Write the direction report**

Create `docs/lit_review_verified_2026-05-26/07_graph_attention_scalable_marl.md` using the common template. The SafeComm relation section must distinguish “architecture evidence” from “safety-communication joint evidence”.

- [ ] **Step 5: Verify the report**

Run:

```bash
rg -n "Verified|Metadata mismatch|Weak match|Not found|Replaced" docs/lit_review_verified_2026-05-26/07_graph_attention_scalable_marl.md
rg -n "https?://|doi.org|arxiv.org|openreview.net|proceedings.mlr.press|semanticscholar.org|openalex.org|dl.acm.org|aaai.org" docs/lit_review_verified_2026-05-26/07_graph_attention_scalable_marl.md
git diff -- docs/lit_review
```

Expected: status labels and source links are present; old reports remain unchanged.

- [ ] **Step 6: Commit Direction 7**

Run:

```bash
git add docs/lit_review_verified_2026-05-26/07_graph_attention_scalable_marl.md
git commit -m "docs: add verified graph attention marl review"
```

Expected: commit contains only `docs/lit_review_verified_2026-05-26/07_graph_attention_scalable_marl.md`.

## Task 9: Direction 8, Sim-to-Real UAV MARL

**Files:**
- Read: `docs/lit_review/08_sim2real_uav_marl.md`
- Create: `docs/lit_review_verified_2026-05-26/08_sim2real_uav_marl.md`

- [ ] **Step 1: Extract original citation candidates**

Run:

```bash
rg -n "^(### [0-9]+\\.|[0-9]+\\. )" docs/lit_review/08_sim2real_uav_marl.md
```

Expected: output includes gym-pybullet-drones, domain randomization, high-speed flight, robust quadrotor policy transfer, information aggregation, Neural Fly, multi-UAV sim-to-real, communication uncertainty, DROPO, and deployment workflow entries.

- [ ] **Step 2: Verify original Sim2Real entries**

Use web search and source pages for these exact queries:

```text
"Learning to Fly--a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-Agent Quadcopter Control"
"Domain randomization for transferring deep neural networks from simulation to the real world"
"Learning high-speed flight in the wild"
"Domain randomization for robust quadrotor policy transfer"
"Neural-Fly enables rapid learning for agile flight in strong winds"
"DROPO: Sim-to-real transfer with offline domain randomization"
"Robust Multi-Agent Reinforcement Learning with Communication Uncertainty"
"Multi-UAV sim-to-real with curriculum learning and safety-aware transfer"
"Safety-constrained reinforcement learning for multi-UAV formation from simulation to real flight"
"From simulation to reality bridging the gap for multi-agent reinforcement learning in UAV swarms"
```

Expected: unverified multi-UAV deployment papers are labeled `Not found` or `Weak match`; only real UAV/sim-to-real sources support final recommendations.

- [ ] **Step 3: Add verified anchor references**

Use reliable source links for these anchor papers when relevant:

```text
Panerati et al. "Learning to Fly--a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-Agent Quadcopter Control" IROS 2021
Tobin et al. "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World" IROS 2017
Loquercio et al. "Learning high-speed flight in the wild" Science Robotics 2021
O'Connell et al. "Neural-Fly enables rapid learning for agile flight in strong winds" Science Robotics 2022
Mehta et al. "Active Domain Randomization" CoRL 2019 if relevant and verified
Mandlekar et al. or OpenAI domain randomization robot sim-to-real sources if UAV-specific sources are limited
Panerati et al. gym-pybullet-drones GitHub/paper page for environment details
```

Expected: the verified review supports PyBullet/gym-pybullet-drones, uniform/adaptive domain randomization, and cautious communication uncertainty modeling without inventing unverified numerical transfer rates.

- [ ] **Step 4: Write the direction report**

Create `docs/lit_review_verified_2026-05-26/08_sim2real_uav_marl.md` using the common template. The SafeComm relation section must mark domain randomization parameters as literature-informed but avoid claiming exact values unless the source explicitly reports them.

- [ ] **Step 5: Verify the report**

Run:

```bash
rg -n "Verified|Metadata mismatch|Weak match|Not found|Replaced" docs/lit_review_verified_2026-05-26/08_sim2real_uav_marl.md
rg -n "https?://|doi.org|arxiv.org|openreview.net|proceedings.mlr.press|semanticscholar.org|openalex.org|science.org|ieeexplore.ieee.org|roboticsproceedings.org" docs/lit_review_verified_2026-05-26/08_sim2real_uav_marl.md
git diff -- docs/lit_review
```

Expected: status labels and source links are present; old reports remain unchanged.

- [ ] **Step 6: Commit Direction 8**

Run:

```bash
git add docs/lit_review_verified_2026-05-26/08_sim2real_uav_marl.md
git commit -m "docs: add verified uav sim2real review"
```

Expected: commit contains only `docs/lit_review_verified_2026-05-26/08_sim2real_uav_marl.md`.

## Task 10: Write Verified Synthesis Report

**Files:**
- Read: `docs/lit_review_verified_2026-05-26/01_safe_marl.md`
- Read: `docs/lit_review_verified_2026-05-26/02_comm_constrained_marl.md`
- Read: `docs/lit_review_verified_2026-05-26/03_safety_comm_intersection.md`
- Read: `docs/lit_review_verified_2026-05-26/04_uav_formation.md`
- Read: `docs/lit_review_verified_2026-05-26/05_multi_objective_pareto_marl.md`
- Read: `docs/lit_review_verified_2026-05-26/06_cbf_mpc_shielding_safe_marl.md`
- Read: `docs/lit_review_verified_2026-05-26/07_graph_attention_scalable_marl.md`
- Read: `docs/lit_review_verified_2026-05-26/08_sim2real_uav_marl.md`
- Create: `docs/lit_review_verified_2026-05-26/00_synthesis_report.md`

- [ ] **Step 1: Confirm all 8 direction reports exist**

Run:

```bash
for f in \
  docs/lit_review_verified_2026-05-26/01_safe_marl.md \
  docs/lit_review_verified_2026-05-26/02_comm_constrained_marl.md \
  docs/lit_review_verified_2026-05-26/03_safety_comm_intersection.md \
  docs/lit_review_verified_2026-05-26/04_uav_formation.md \
  docs/lit_review_verified_2026-05-26/05_multi_objective_pareto_marl.md \
  docs/lit_review_verified_2026-05-26/06_cbf_mpc_shielding_safe_marl.md \
  docs/lit_review_verified_2026-05-26/07_graph_attention_scalable_marl.md \
  docs/lit_review_verified_2026-05-26/08_sim2real_uav_marl.md; do
    test -f "$f" && echo "OK $f"
  done
```

Expected: 8 lines beginning with `OK`.

- [ ] **Step 2: Extract verification summaries and high-risk entries**

Run:

```bash
rg -n "## 1\\. 核验结论摘要|Not found|Weak match|Metadata mismatch|Replaced|强支持|间接支持|证据不足" docs/lit_review_verified_2026-05-26/0[1-8]_*.md
```

Expected: output shows summary sections, audit status labels, and SafeComm evidence-strength wording across the 8 reports.

- [ ] **Step 3: Write the synthesis report**

Create `docs/lit_review_verified_2026-05-26/00_synthesis_report.md` with these exact sections:

```markdown
# 核验版文献调研综合报告

## 1. 总体核验结果
## 2. 八个方向的证据强度
## 3. 原报告需要修正或删除的结论
## 4. 仍可防守的 SafeComm 创新定位
## 5. 推荐 Related Work 写法
## 6. 高风险旧引用列表
## 7. 核验后的核心参考文献
```

The synthesis must use conservative gap wording: `本次核验未发现直接覆盖...` rather than absolute claims such as `没有任何工作`.

- [ ] **Step 4: Verify synthesis report**

Run:

```bash
rg -n "总体核验结果|八个方向的证据强度|本次核验未发现|高风险旧引用|https?://|doi.org|arxiv.org|openreview.net|proceedings.mlr.press" docs/lit_review_verified_2026-05-26/00_synthesis_report.md
git diff -- docs/lit_review
```

Expected: synthesis sections, conservative gap wording, source links, and no old-report changes.

- [ ] **Step 5: Commit synthesis report**

Run:

```bash
git add docs/lit_review_verified_2026-05-26/00_synthesis_report.md
git commit -m "docs: add verified literature synthesis report"
```

Expected: commit contains only `docs/lit_review_verified_2026-05-26/00_synthesis_report.md`.

## Task 11: Final Verification and Completion Summary

**Files:**
- Read: `docs/lit_review_verified_2026-05-26/*.md`
- Read only: `docs/lit_review/*.md`

- [ ] **Step 1: Verify output file count and names**

Run:

```bash
find docs/lit_review_verified_2026-05-26 -maxdepth 1 -type f -name '*.md' | sort
```

Expected: exactly these 9 files are printed:

```text
docs/lit_review_verified_2026-05-26/00_synthesis_report.md
docs/lit_review_verified_2026-05-26/01_safe_marl.md
docs/lit_review_verified_2026-05-26/02_comm_constrained_marl.md
docs/lit_review_verified_2026-05-26/03_safety_comm_intersection.md
docs/lit_review_verified_2026-05-26/04_uav_formation.md
docs/lit_review_verified_2026-05-26/05_multi_objective_pareto_marl.md
docs/lit_review_verified_2026-05-26/06_cbf_mpc_shielding_safe_marl.md
docs/lit_review_verified_2026-05-26/07_graph_attention_scalable_marl.md
docs/lit_review_verified_2026-05-26/08_sim2real_uav_marl.md
```

- [ ] **Step 2: Verify required sections are present in all reports**

Run:

```bash
for f in docs/lit_review_verified_2026-05-26/0[1-8]_*.md; do
  echo "$f"
  rg -n "^## 1\\. 核验结论摘要|^## 2\\. 原文献真实性审计表|^## 3\\. 经核验后的核心文献综述|^## 4\\. 与 SafeComm-VoI / SafeComm-MARL 的关系|^## 5\\. 可引用参考文献" "$f"
done
```

Expected: each of the 8 direction reports prints all 5 required section headings.

- [ ] **Step 3: Verify all direction reports include audit labels and links**

Run:

```bash
for f in docs/lit_review_verified_2026-05-26/0[1-8]_*.md; do
  rg -q "Verified|Metadata mismatch|Weak match|Not found|Replaced" "$f" && \
  rg -q "https?://|doi.org|arxiv.org|openreview.net|proceedings.mlr.press|semanticscholar.org|openalex.org|ieeexplore.ieee.org|dl.acm.org" "$f" && \
  echo "OK $f"
done
```

Expected: 8 lines beginning with `OK`.

- [ ] **Step 4: Verify original reports remain untouched**

Run:

```bash
git diff -- docs/lit_review
```

Expected: no output.

- [ ] **Step 5: Inspect recent commits**

Run:

```bash
git log --oneline -12
```

Expected: recent commits include the design spec, 8 direction reports, and synthesis report commits for this verified literature-review work.

- [ ] **Step 6: Prepare completion summary**

In the final response, list:

```text
Created directory: docs/lit_review_verified_2026-05-26/
Created reports: 00_synthesis_report.md through 08_sim2real_uav_marl.md
Verification run: output count, required sections, audit labels/source links, and git diff -- docs/lit_review
Known limitations: any source categories that were unavailable or any areas where only indirect evidence was found
```

Do not claim a citation is verified unless a reliable source link is present in the relevant report.

# Verified Literature Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create nine verified literature review reports in `docs/lit_review_verified/` by auditing the original citations, removing or downgrading unverifiable entries, adding traceable core literature, and rewriting conclusions around verified evidence.

**Architecture:** The work is document-first and source-driven. Each direction report is produced independently from its matching original file, using a common audit table, verification labels, verified synthesis sections, and traceable references; the synthesis report is produced after all eight direction reports exist.

**Tech Stack:** Markdown, shell tools (`rg`, `find`, `wc`, `git diff`), web literature sources, DOI/Crossref, arXiv, OpenReview, official proceedings and publisher pages, Semantic Scholar, OpenAlex.

---

## File Structure

- Create: `docs/lit_review_verified/`
  - Contains only the nine requested Markdown reports.
- Create: `docs/lit_review_verified/01_safe_marl.md`
  - Audits and rewrites safe multi-agent reinforcement learning literature.
- Create: `docs/lit_review_verified/02_comm_constrained_marl.md`
  - Audits and rewrites communication-constrained MARL literature.
- Create: `docs/lit_review_verified/03_safety_comm_intersection.md`
  - Audits and rewrites the intersection of safety constraints and communication constraints.
- Create: `docs/lit_review_verified/04_uav_formation.md`
  - Audits and rewrites multi-UAV formation control literature.
- Create: `docs/lit_review_verified/05_multi_objective_pareto_marl.md`
  - Audits and rewrites multi-objective, Pareto, and constrained RL/MARL literature.
- Create: `docs/lit_review_verified/06_cbf_mpc_shielding_safe_marl.md`
  - Audits and rewrites CBF, MPC safety filter, and shielding literature for safe RL/MARL.
- Create: `docs/lit_review_verified/07_graph_attention_scalable_marl.md`
  - Audits and rewrites graph, attention, transformer, and scalable MARL literature.
- Create: `docs/lit_review_verified/08_sim2real_uav_marl.md`
  - Audits and rewrites simulation-to-reality literature for UAV and multi-agent RL.
- Create: `docs/lit_review_verified/00_synthesis_report.md`
  - Summarizes the evidence strength, high-risk original citations, and defensible SafeComm-MARL positioning after all eight direction reports are complete.
- Do not modify: `docs/lit_review/*.md`
  - Original reports are read-only inputs.

## Shared Verification Protocol

Use this protocol in every direction task.

- [ ] **Step 1: Extract original citation candidates**

Use the exact extraction command listed inside the direction task being executed.

Expected: output includes the original paper headings and reference-list entries for the selected source file.

- [ ] **Step 2: Verify each original entry against reliable sources**

For each original entry, search by exact title first, then by title plus lead author, then by title plus venue. Prefer sources in this order:

```text
1. DOI/Crossref or DOI resolver
2. arXiv record
3. OpenReview, official conference proceedings, or conference page
4. IEEE, ACM, Springer, ScienceDirect, PMLR, AAAI, IJCAI, AAMAS, RSS, CoRL, ICRA, IROS, NeurIPS, ICML, ICLR
5. Semantic Scholar or OpenAlex
```

Expected: each original entry receives one label: `Verified`, `Metadata mismatch`, `Weak match`, `Not found`, or `Replaced`.

- [ ] **Step 3: Apply conservative replacement rules**

Use these rules:

```text
Verified: may be cited and used as evidence.
Metadata mismatch: may be cited only with corrected metadata.
Weak match: may appear only in the audit table; do not use it as evidence.
Not found: may appear only in the audit table and high-risk list.
Replaced: original entry is excluded from evidence; cite the verified replacement instead.
```

Expected: no substantive claim in the rewritten review depends on `Weak match` or `Not found` entries.

- [ ] **Step 4: Use the common direction-report structure**

Each direction report must contain these exact sections:

```markdown
# 方向X：标题与对应原文件一致的中文方向名

## 1. 核验结论摘要

## 2. 原文献真实性审计表

## 3. 经核验后的核心文献综述

## 4. 与 SafeComm-MARL 的关系

## 5. 可引用参考文献
```

Expected: the report contains the five sections above, concrete audit counts, and traceable links for all cited references.

## Task 1: Create Output Directory and Guard Original Inputs

**Files:**
- Create: `docs/lit_review_verified/`
- Read: `docs/lit_review/*.md`
- Do not modify: `docs/lit_review/*.md`

- [ ] **Step 1: Create the output directory**

Run:

```bash
mkdir -p docs/lit_review_verified
```

Expected: command exits successfully.

- [ ] **Step 2: Record the expected output file list**

Run:

```bash
printf '%s\n' \
  docs/lit_review_verified/00_synthesis_report.md \
  docs/lit_review_verified/01_safe_marl.md \
  docs/lit_review_verified/02_comm_constrained_marl.md \
  docs/lit_review_verified/03_safety_comm_intersection.md \
  docs/lit_review_verified/04_uav_formation.md \
  docs/lit_review_verified/05_multi_objective_pareto_marl.md \
  docs/lit_review_verified/06_cbf_mpc_shielding_safe_marl.md \
  docs/lit_review_verified/07_graph_attention_scalable_marl.md \
  docs/lit_review_verified/08_sim2real_uav_marl.md
```

Expected: exactly the nine target paths are printed.

- [ ] **Step 3: Verify original reports are unchanged before writing new reports**

Run:

```bash
git diff -- docs/lit_review
```

Expected: no output.

- [ ] **Step 4: Do not commit an empty directory**

Run:

```bash
git status --short docs/lit_review_verified
```

Expected: no tracked changes are required for the empty directory. The first committed file in `docs/lit_review_verified/` will be a completed direction report.

## Task 2: Direction 1, Safe MARL

**Files:**
- Read: `docs/lit_review/01_safe_marl.md`
- Create: `docs/lit_review_verified/01_safe_marl.md`

- [ ] **Step 1: Extract original entries**

Run:

```bash
rg -n "^(### [0-9]+\\.|[0-9]+\\. )" docs/lit_review/01_safe_marl.md
```

Expected: output includes original entries such as MACPO, MAPPO-Lagrangian, MAMPS, MACPPO, HAZARD, Safety Gymnasium, SAUTE RL, and Safe Networked MARL.

- [ ] **Step 2: Verify original high-risk entries first**

Search these exact strings and record results:

```text
"MACPO: Constrained Policy Optimization for Multi-Agent Reinforcement Learning"
"MAPPO-Lagrangian Safe MAPPO"
"MAMPS Safe Multi-Agent Reinforcement Learning via Model Predictive Shielding"
"Multi-Agent Constrained Policy Optimisation"
"HAZARD Benchmark multi-agent safe reinforcement learning"
"Safe Networked Multi-Agent Reinforcement Learning with Chance Constraints"
"SAUTE RL Almost Surely Safe Reinforcement Learning Using State Augmentation"
```

Expected: entries with no reliable source are marked `Weak match` or `Not found`; entries with different authors, titles, years, or venues are marked `Metadata mismatch`.

- [ ] **Step 3: Add verified core safe-RL and safe-MARL references**

Use traceable sources for these anchor works where relevant:

```text
Achiam et al., "Constrained Policy Optimization", ICML 2017, PMLR.
Zhang et al., "Multi-Agent Constrained Policy Optimisation", arXiv:2110.02793 or verified venue.
Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games", NeurIPS 2022.
Ji et al., "Safety Gymnasium", NeurIPS 2023 Datasets and Benchmarks or arXiv.
Sootla et al., "SAUTE RL: Almost Surely Safe Reinforcement Learning Using State Augmentation", ICML 2022 or PMLR.
Gu et al., "A Survey of Safe Reinforcement Learning: Methods, Theory and Applications", arXiv:2205.10330.
Altman, "Constrained Markov Decision Processes", 1999, publisher record.
```

Expected: all retained references have a DOI, arXiv, PMLR, OpenReview, publisher, Semantic Scholar, or OpenAlex link.

- [ ] **Step 4: Write the report**

Create `docs/lit_review_verified/01_safe_marl.md` with the common direction-report structure. The SafeComm-MARL relation section must explicitly state whether verified safe-MARL literature assumes full communication, centralized training, global state, or unconstrained communication.

Expected: report contains an audit table, verified synthesis, and reference list with source links.

- [ ] **Step 5: Verify the report locally**

Run:

```bash
rg -n "Verified|Metadata mismatch|Weak match|Not found|Replaced" docs/lit_review_verified/01_safe_marl.md
rg -n "https?://|doi.org|arxiv.org|openreview.net|proceedings.mlr.press|semanticscholar.org|openalex.org" docs/lit_review_verified/01_safe_marl.md
git diff -- docs/lit_review
```

Expected: status labels and source links are present; `git diff -- docs/lit_review` has no output.

- [ ] **Step 6: Commit Direction 1**

Run:

```bash
git add docs/lit_review_verified/01_safe_marl.md
git commit -m "docs: verify safe marl literature"
```

Expected: commit contains only `docs/lit_review_verified/01_safe_marl.md`.

## Task 3: Direction 2, Communication-Constrained MARL

**Files:**
- Read: `docs/lit_review/02_comm_constrained_marl.md`
- Create: `docs/lit_review_verified/02_comm_constrained_marl.md`

- [ ] **Step 1: Extract original entries**

Run:

```bash
rg -n "^(### [0-9]+\\.|@in|@article|@misc)" docs/lit_review/02_comm_constrained_marl.md
```

Expected: output includes CommNet, DIAL, BiCNet, ATOC, SchedNet, DGN, Networked MARL, TarMAC, I2C, NeurComm, MASIA, NDQ, MAGIC, DC2, and VBC.

- [ ] **Step 2: Verify original communication entries**

Search these exact strings and record source links:

```text
"Learning Multiagent Communication with Backpropagation"
"Learning to Communicate with Deep Multi-Agent Reinforcement Learning"
"Multiagent Bidirectionally-Coordinated Nets for Learning to Play StarCraft Combat"
"Learning Attentional Communication for Multi-Agent Cooperation"
"Learning to Schedule Communication in Multi-Agent Systems"
"Graph Convolutional Reinforcement Learning"
"Networked Multi-Agent Reinforcement Learning in Continuous Spaces"
"Targeted Multi-Agent Communication"
"Learning Individually Inferred Communication for Multi-Agent Cooperation"
"Multi-Agent Graph-Attention Communication and Teaming"
"Efficient Communication in Multi-Agent Reinforcement Learning via Variance Based Control"
```

Expected: entries that exist under different titles or venues are marked `Metadata mismatch`; entries that cannot be verified are excluded from evidence.

- [ ] **Step 3: Add verified core communication-MARL references**

Use traceable sources for these anchor works where relevant:

```text
Sukhbaatar et al., "Learning Multiagent Communication with Backpropagation", NeurIPS 2016.
Foerster et al., "Learning to Communicate with Deep Multi-Agent Reinforcement Learning", NeurIPS 2016.
Jiang and Lu, "Learning Attentional Communication for Multi-Agent Cooperation", NeurIPS 2018.
Kim et al., "Learning to Schedule Communication in Multi-Agent Systems", ICLR 2019 OpenReview.
Das et al., "TarMAC: Targeted Multi-Agent Communication", ICML 2019 PMLR.
Jiang et al., "Graph Convolutional Reinforcement Learning", ICLR 2020 OpenReview.
Ding et al., "Learning Individually Inferred Communication for Multi-Agent Cooperation", NeurIPS 2020.
Niu et al., "Multi-Agent Graph-Attention Communication and Teaming", AAMAS 2021.
```

Expected: the rewritten review distinguishes learned message content, learned communication topology, scheduled communication, and bandwidth-limited communication.

- [ ] **Step 4: Write the report**

Create `docs/lit_review_verified/02_comm_constrained_marl.md`. The SafeComm-MARL relation section must explicitly state whether each communication method optimizes safety constraints, communication budgets, or only task reward.

Expected: report contains audit table, thematic synthesis, and references with source links.

- [ ] **Step 5: Verify the report locally**

Run:

```bash
rg -n "Verified|Metadata mismatch|Weak match|Not found|Replaced" docs/lit_review_verified/02_comm_constrained_marl.md
rg -n "https?://|doi.org|arxiv.org|openreview.net|proceedings.mlr.press|semanticscholar.org|openalex.org" docs/lit_review_verified/02_comm_constrained_marl.md
git diff -- docs/lit_review
```

Expected: status labels and source links are present; original reports remain unchanged.

- [ ] **Step 6: Commit Direction 2**

Run:

```bash
git add docs/lit_review_verified/02_comm_constrained_marl.md
git commit -m "docs: verify communication constrained marl literature"
```

Expected: commit contains only `docs/lit_review_verified/02_comm_constrained_marl.md`.

## Task 4: Direction 3, Safety and Communication Intersection

**Files:**
- Read: `docs/lit_review/03_safety_comm_intersection.md`
- Create: `docs/lit_review_verified/03_safety_comm_intersection.md`

- [ ] **Step 1: Extract original entries**

Run:

```bash
rg -n "^(### [0-9]+\\.|[0-9]+\\. )" docs/lit_review/03_safety_comm_intersection.md
```

Expected: output includes Confidence-Weighted Communication, Safe Networked MARL with Chance Constraints, Safe Multi-UAV via Distributed CBF without Communication, and event-triggered control entries.

- [ ] **Step 2: Verify original intersection claims**

Search these exact strings and record source links:

```text
"Multi-Agent Safety via Confidence-Weighted Communication"
"Safe Networked Multi-Agent Reinforcement Learning with Chance Constraints"
"Safe Multi-UAV via Distributed CBF without Communication"
"event-triggered control multi-agent systems communication constraints safety"
"communication constraints control barrier functions multi agent"
```

Expected: unverifiable direct-intersection papers are marked `Not found` or `Weak match`; event-triggered control literature is treated as related control theory, not direct safe-MARL evidence.

- [ ] **Step 3: Add verified intersection and boundary references**

Use traceable sources for these anchor areas where relevant:

```text
Verified safe-MARL references from Direction 1.
Verified communication-MARL references from Direction 2.
Wang et al., "Safety Barrier Certificates for Collisions-Free Multirobot Systems", IEEE Transactions on Robotics, 2017.
Ames et al., "Control Barrier Function Based Quadratic Programs for Safety Critical Systems", IEEE TAC, 2017.
Dimarogonas et al. or Tabuada event-triggered multi-agent control references with DOI or publisher links.
Networked control references on communication-limited safety or event-triggered consensus with DOI or publisher links.
```

Expected: the report separates direct evidence from adjacent evidence and avoids claiming uniqueness as a proven fact.

- [ ] **Step 4: Write the report**

Create `docs/lit_review_verified/03_safety_comm_intersection.md`. The SafeComm-MARL relation section must state the strongest defensible claim using wording such as "we found limited verified literature directly optimizing both safety constraints and communication budgets in MARL" rather than "no work exists."

Expected: direct and indirect evidence are clearly separated.

- [ ] **Step 5: Verify the report locally**

Run:

```bash
rg -n "Verified|Metadata mismatch|Weak match|Not found|Replaced" docs/lit_review_verified/03_safety_comm_intersection.md
rg -n "https?://|doi.org|arxiv.org|openreview.net|proceedings.mlr.press|semanticscholar.org|openalex.org" docs/lit_review_verified/03_safety_comm_intersection.md
git diff -- docs/lit_review
```

Expected: source links and conservative gap language are present; original reports remain unchanged.

- [ ] **Step 6: Commit Direction 3**

Run:

```bash
git add docs/lit_review_verified/03_safety_comm_intersection.md
git commit -m "docs: verify safety communication intersection literature"
```

Expected: commit contains only `docs/lit_review_verified/03_safety_comm_intersection.md`.

## Task 5: Direction 4, Multi-UAV Formation

**Files:**
- Read: `docs/lit_review/04_uav_formation.md`
- Create: `docs/lit_review_verified/04_uav_formation.md`

- [ ] **Step 1: Extract original entries**

Run:

```bash
rg -n "^(### [0-9]+\\.|\\[[0-9]+\\]|[0-9]+\\. )" docs/lit_review/04_uav_formation.md
```

Expected: output includes claimed UAV formation MARL papers and classic formation-control baselines.

- [ ] **Step 2: Verify high-risk UAV formation entries**

Search these exact strings and record source links:

```text
"Safe RL for Multi-UAV Formation with CBF"
"Safe Multi-UAV via Distributed CBF-MARL"
"Formation Control of UAV Swarms with Safety Constraints Drones 2022"
"GNN-Based Multi-UAV Formation TVT 2022"
"Curriculum-Based MARL for UAV Swarm TAES 2024"
"Communication-Aware Multi-UAV via Attention MARL Internet of Things Journal 2023"
"Multi-UAV Pursuit with Communication-Constrained MARL Aerospace Science and Technology 2024"
```

Expected: generic or invented-sounding entries with no reliable match are marked `Not found` or `Weak match`.

- [ ] **Step 3: Add verified UAV and formation-control references**

Use traceable sources for these anchor works where relevant:

```text
Olfati-Saber, "Flocking for Multi-Agent Dynamic Systems: Algorithms and Theory", IEEE TAC, 2006.
Dunbar and Murray, "Distributed Receding Horizon Control for Multi-Vehicle Formation Stabilization", Automatica or IEEE source.
Ren and Beard, "Distributed Consensus in Multi-vehicle Cooperative Control", publisher record.
Khatib, "Real-Time Obstacle Avoidance for Manipulators and Mobile Robots", IJRR, 1986.
Kamel et al., UAV trajectory planning or MPC references with IEEE links.
Verified MAPPO and communication-MARL references when used as UAV-formation baselines.
Verified multi-UAV RL papers only when title and venue are confirmed by reliable sources.
```

Expected: the report distinguishes classical control baselines from RL/MARL UAV formation work.

- [ ] **Step 4: Write the report**

Create `docs/lit_review_verified/04_uav_formation.md`. The SafeComm-MARL relation section must state which UAV formation claims are supported by verified UAV-specific work and which are extrapolated from general MARL or classical formation control.

Expected: report contains audit table, thematic synthesis, and conservative mapping to SafeComm-MARL.

- [ ] **Step 5: Verify the report locally**

Run:

```bash
rg -n "Verified|Metadata mismatch|Weak match|Not found|Replaced" docs/lit_review_verified/04_uav_formation.md
rg -n "https?://|doi.org|arxiv.org|openreview.net|proceedings.mlr.press|semanticscholar.org|openalex.org" docs/lit_review_verified/04_uav_formation.md
git diff -- docs/lit_review
```

Expected: source links are present and original reports remain unchanged.

- [ ] **Step 6: Commit Direction 4**

Run:

```bash
git add docs/lit_review_verified/04_uav_formation.md
git commit -m "docs: verify uav formation literature"
```

Expected: commit contains only `docs/lit_review_verified/04_uav_formation.md`.

## Task 6: Direction 5, Multi-Objective and Pareto MARL

**Files:**
- Read: `docs/lit_review/05_multi_objective_pareto_marl.md`
- Create: `docs/lit_review_verified/05_multi_objective_pareto_marl.md`

- [ ] **Step 1: Extract original entries**

Run:

```bash
rg -n "^(### [0-9]+\\.|[0-9]+\\. )" docs/lit_review/05_multi_objective_pareto_marl.md
```

Expected: output includes MODRL, MORL/D, Pareto multi-task learning, PGMORL, multi-objective safe RL, Pareto conditioned networks, MOMALAND, CMDP, and CPO.

- [ ] **Step 2: Verify original multi-objective entries**

Search these exact strings and record source links:

```text
"A Multi-Objective Deep Reinforcement Learning Framework"
"MORL/D Multiobjective Reinforcement Learning Based on Decomposition"
"Pareto Multi-Task Learning"
"Prediction-Guided Multi-Objective Reinforcement Learning"
"Multi-Objective Safe Reinforcement Learning arXiv 2204.06475"
"Pareto Conditioned Networks"
"Distributional Multi-Objective Multi-Agent Reinforcement Learning"
"MOMALAND A Suite of Benchmarks for Multi-Objective Multi-Agent Decision Making"
"Constrained Markov Decision Processes Altman 1999"
"Constrained Policy Optimization Achiam 2017"
```

Expected: title, author, venue, and year mismatches are captured in the audit table.

- [ ] **Step 3: Add verified core multi-objective and constrained references**

Use traceable sources for these anchor works where relevant:

```text
Roijers et al., "A Survey of Multi-Objective Sequential Decision-Making", JAIR or verified source.
Yang et al. or Xu et al., verified MORL/D paper with DOI or publisher page.
Sener and Koltun, "Multi-Task Learning as Multi-Objective Optimization", NeurIPS 2018.
Xu et al., "Prediction-Guided Multi-Objective Reinforcement Learning", ICML 2020 or verified venue.
Felten et al., "A Toolkit for Reliable Benchmarking and Research in Multi-Objective Reinforcement Learning" or verified MORL benchmark work.
MOMALAND benchmark paper with arXiv, OpenReview, or proceedings page.
Altman, "Constrained Markov Decision Processes", 1999.
Achiam et al., "Constrained Policy Optimization", ICML 2017.
```

Expected: the review clarifies the difference between multi-objective optimization, constrained optimization, scalarized rewards, Pareto fronts, and Lagrangian CMDP methods.

- [ ] **Step 4: Write the report**

Create `docs/lit_review_verified/05_multi_objective_pareto_marl.md`. The SafeComm-MARL relation section must state whether Lagrangian treatment is supported as a constrained-optimization choice and avoid claiming Pareto optimality unless supported by verified theory.

Expected: report contains audit table, verified synthesis, and corrected claims.

- [ ] **Step 5: Verify the report locally**

Run:

```bash
rg -n "Verified|Metadata mismatch|Weak match|Not found|Replaced" docs/lit_review_verified/05_multi_objective_pareto_marl.md
rg -n "https?://|doi.org|arxiv.org|openreview.net|proceedings.mlr.press|semanticscholar.org|openalex.org" docs/lit_review_verified/05_multi_objective_pareto_marl.md
git diff -- docs/lit_review
```

Expected: labels and links are present; original reports remain unchanged.

- [ ] **Step 6: Commit Direction 5**

Run:

```bash
git add docs/lit_review_verified/05_multi_objective_pareto_marl.md
git commit -m "docs: verify multi objective marl literature"
```

Expected: commit contains only `docs/lit_review_verified/05_multi_objective_pareto_marl.md`.

## Task 7: Direction 6, CBF, MPC, and Shielding for Safe MARL

**Files:**
- Read: `docs/lit_review/06_cbf_mpc_shielding_safe_marl.md`
- Create: `docs/lit_review_verified/06_cbf_mpc_shielding_safe_marl.md`

- [ ] **Step 1: Extract original entries**

Run:

```bash
rg -n "^(### [0-9]+\\.|[0-9]+\\. )" docs/lit_review/06_cbf_mpc_shielding_safe_marl.md
```

Expected: output includes CBF theory, multi-agent CBF, predictive safety filters, neural Lyapunov-barrier functions, MACBF, shielding, Dalal safety layer, and probabilistic safety constraints.

- [ ] **Step 2: Verify original CBF and shielding entries**

Search these exact strings and record source links:

```text
"Control Barrier Functions: Theory and Applications"
"Multi-Agent Safety via Control Barrier Functions"
"Learning Safe Policies with Expert Guidance"
"A Predictive Safety Filter for Learning-Based Control of Constrained Nonlinear Dynamical Systems"
"Safe Nonlinear Control Using Robust Neural Lyapunov-Barrier Functions"
"Learning Safe Unlabeled Multi-Robot Planning with Motion Constraints"
"Safe Multi-Agent Reinforcement Learning via Shielding"
"Distributed Control Barrier Functions for Safe Multi-UAV Formation without Communication"
"Safe Exploration in Continuous Action Spaces"
"Safety Barrier Certificates for Collisions-Free Multirobot Systems"
"End-to-End Safe Reinforcement Learning through Barrier Functions"
"Probabilistic Safety Constraints for Learned High Relative Degree System Dynamics"
```

Expected: claims of IEEE venue, year, and author lists are checked against reliable metadata.

- [ ] **Step 3: Add verified CBF, safety-filter, and shielding references**

Use traceable sources for these anchor works where relevant:

```text
Ames et al., "Control Barrier Function Based Quadratic Programs for Safety Critical Systems", IEEE TAC, 2017.
Ames et al., "Control Barrier Functions: Theory and Applications", ECC or verified proceedings.
Wang et al., "Safety Barrier Certificates for Collisions-Free Multirobot Systems", IEEE Transactions on Robotics, 2017.
Wabersich and Zeilinger, "A Predictive Safety Filter for Learning-Based Control of Constrained Nonlinear Dynamical Systems", Automatica or IEEE TAC verified source.
Cheng et al., "End-to-End Safe Reinforcement Learning through Barrier Functions", AAAI or verified source.
Dalal et al., "Safe Exploration in Continuous Action Spaces", arXiv or workshop record.
Alshiekh et al., "Safe Reinforcement Learning via Shielding", AAAI 2018.
Elsayed-Aly et al., verified safe multi-agent shielding work if traceable.
```

Expected: the rewritten review distinguishes continuous CBF-QP safety filters, MPC/predictive safety filters, formal shielding, and learned safety critics.

- [ ] **Step 4: Write the report**

Create `docs/lit_review_verified/06_cbf_mpc_shielding_safe_marl.md`. The SafeComm-MARL relation section must mark any claim about communication-aware CBF scheduling as a proposed positioning unless directly supported by verified literature.

Expected: report contains corrected metadata and does not reuse unverified formulas as literature-backed claims.

- [ ] **Step 5: Verify the report locally**

Run:

```bash
rg -n "Verified|Metadata mismatch|Weak match|Not found|Replaced" docs/lit_review_verified/06_cbf_mpc_shielding_safe_marl.md
rg -n "https?://|doi.org|arxiv.org|openreview.net|proceedings.mlr.press|semanticscholar.org|openalex.org" docs/lit_review_verified/06_cbf_mpc_shielding_safe_marl.md
git diff -- docs/lit_review
```

Expected: source links and status labels are present; original reports remain unchanged.

- [ ] **Step 6: Commit Direction 6**

Run:

```bash
git add docs/lit_review_verified/06_cbf_mpc_shielding_safe_marl.md
git commit -m "docs: verify cbf mpc shielding literature"
```

Expected: commit contains only `docs/lit_review_verified/06_cbf_mpc_shielding_safe_marl.md`.

## Task 8: Direction 7, Graph, Attention, and Scalable MARL

**Files:**
- Read: `docs/lit_review/07_graph_attention_scalable_marl.md`
- Create: `docs/lit_review_verified/07_graph_attention_scalable_marl.md`

- [ ] **Step 1: Extract original entries**

Run:

```bash
rg -n "^(### [0-9]+\\.|[0-9]+\\. )" docs/lit_review/07_graph_attention_scalable_marl.md
```

Expected: output includes DGN, MAAC, MAGIC, mean-field MARL, G2ANet, INFOMARL, MAT, CoPO, GPG, QGAM, GAT, CommNet, SchedNet, ATOC, and Transformers.

- [ ] **Step 2: Verify original graph and attention entries**

Search these exact strings and record source links:

```text
"Graph Convolutional Reinforcement Learning"
"Actor-Attention-Critic for Multi-Agent Reinforcement Learning"
"Multi-Agent Graph-Attention Communication and Teaming"
"Mean Field Multi-Agent Reinforcement Learning"
"Multi-Agent Game Abstraction via Graph Attention Neural Network"
"Scalable Multi-Agent Reinforcement Learning through Intelligent Information Aggregation"
"Multi-Agent Transformer"
"Learning to Simulate Self-Driven Particles System with Coordinated Policy Optimization"
"Leveraging Graph-Based Policy Gradient Methods for Safe Multi-Agent Reinforcement Learning"
"Graph Attention Networks"
"Attention Is All You Need"
```

Expected: title and venue mismatches are captured in the audit table.

- [ ] **Step 3: Add verified graph and scalable MARL references**

Use traceable sources for these anchor works where relevant:

```text
Jiang et al., "Graph Convolutional Reinforcement Learning", ICLR 2020 OpenReview.
Iqbal and Sha, "Actor-Attention-Critic for Multi-Agent Reinforcement Learning", ICML 2019 PMLR.
Niu et al., "Multi-Agent Graph-Attention Communication and Teaming", AAMAS 2021.
Yang et al., "Mean Field Multi-Agent Reinforcement Learning", ICML 2018 PMLR.
Liu et al., "Multi-Agent Game Abstraction via Graph Attention Neural Network", AAAI 2020.
Nayak et al., "Scalable Multi-Agent Reinforcement Learning through Intelligent Information Aggregation", ICML 2023 PMLR.
Wen et al., "Multi-Agent Transformer", NeurIPS 2022.
Veličković et al., "Graph Attention Networks", ICLR 2018 OpenReview.
Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
```

Expected: the report distinguishes graph neural networks, attention aggregation, transformer-based centralized critics, learned communication topology, and scalability evidence.

- [ ] **Step 4: Write the report**

Create `docs/lit_review_verified/07_graph_attention_scalable_marl.md`. The SafeComm-MARL relation section must state whether multi-hop graph propagation conflicts with strict communication-budget semantics and identify that as an interpretation unless directly proven by a cited paper.

Expected: report contains verified source links and avoids unsupported performance predictions.

- [ ] **Step 5: Verify the report locally**

Run:

```bash
rg -n "Verified|Metadata mismatch|Weak match|Not found|Replaced" docs/lit_review_verified/07_graph_attention_scalable_marl.md
rg -n "https?://|doi.org|arxiv.org|openreview.net|proceedings.mlr.press|semanticscholar.org|openalex.org" docs/lit_review_verified/07_graph_attention_scalable_marl.md
git diff -- docs/lit_review
```

Expected: source links and status labels are present; original reports remain unchanged.

- [ ] **Step 6: Commit Direction 7**

Run:

```bash
git add docs/lit_review_verified/07_graph_attention_scalable_marl.md
git commit -m "docs: verify graph attention scalable marl literature"
```

Expected: commit contains only `docs/lit_review_verified/07_graph_attention_scalable_marl.md`.

## Task 9: Direction 8, Simulation-to-Reality for UAV MARL

**Files:**
- Read: `docs/lit_review/08_sim2real_uav_marl.md`
- Create: `docs/lit_review_verified/08_sim2real_uav_marl.md`

- [ ] **Step 1: Extract original entries**

Run:

```bash
rg -n "^(### [0-9]+\\.|[0-9]+\\. )" docs/lit_review/08_sim2real_uav_marl.md
```

Expected: output includes gym-pybullet-drones, domain randomization, high-speed flight, INFOMARL, Neural-Fly, DROPO, multi-UAV sim-to-real, communication uncertainty, and real-world deployment entries.

- [ ] **Step 2: Verify high-risk sim-to-real entries**

Search these exact strings and record source links:

```text
"Learning to Fly—a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-Agent Quadcopter Control"
"Domain Randomization for Simulation-Based Policy Optimization with Application to Quadrotor Attitude Control"
"Multi-UAV sim-to-real with curriculum learning and safety-aware transfer"
"Robust Multi-Agent Reinforcement Learning with Communication Uncertainty"
"Safety-constrained reinforcement learning for multi-UAV formation from simulation to real flight"
"From Simulation to Reality Bridging the Gap for Multi-Agent Reinforcement Learning in UAV Swarms"
"Towards Real-World Deployment of Reinforcement Learning for Multi-Robot Systems"
```

Expected: invented or untraceable multi-UAV sim-to-real claims are marked `Not found` or `Weak match`.

- [ ] **Step 3: Add verified sim-to-real and UAV RL references**

Use traceable sources for these anchor works where relevant:

```text
Panerati et al., "Learning to Fly in Seconds", gym-pybullet-drones paper, IROS or arXiv verified source.
Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World", IROS 2017.
Loquercio et al., "Learning High-Speed Flight in the Wild", Science Robotics, 2021.
O'Connell et al., "Neural-Fly Enables Rapid Learning for Agile Flight in Strong Winds", Science Robotics, 2022.
Mehta et al., "DROPO: Sim-to-Real Transfer with Offline Domain Randomization", Robotics and Autonomous Systems, 2022.
Peng et al. or other verified UAV sim-to-real works only when metadata and source links are reliable.
Verified multi-agent sim-to-real work only when it explicitly involves multiple agents or multi-robot systems.
```

Expected: the report separates quadrotor sim-to-real, multi-robot sim-to-real, and communication-uncertainty robustness.

- [ ] **Step 4: Write the report**

Create `docs/lit_review_verified/08_sim2real_uav_marl.md`. The SafeComm-MARL relation section must downgrade any parameter-specific claims, such as exact noise magnitudes or delay thresholds, unless those values are directly supported by verified sources or explicitly marked as engineering recommendations.

Expected: report contains audit table, verified synthesis, and source-backed deployment risks.

- [ ] **Step 5: Verify the report locally**

Run:

```bash
rg -n "Verified|Metadata mismatch|Weak match|Not found|Replaced" docs/lit_review_verified/08_sim2real_uav_marl.md
rg -n "https?://|doi.org|arxiv.org|openreview.net|proceedings.mlr.press|semanticscholar.org|openalex.org" docs/lit_review_verified/08_sim2real_uav_marl.md
git diff -- docs/lit_review
```

Expected: source links and status labels are present; original reports remain unchanged.

- [ ] **Step 6: Commit Direction 8**

Run:

```bash
git add docs/lit_review_verified/08_sim2real_uav_marl.md
git commit -m "docs: verify sim2real uav marl literature"
```

Expected: commit contains only `docs/lit_review_verified/08_sim2real_uav_marl.md`.

## Task 10: Synthesis Report

**Files:**
- Read: `docs/lit_review_verified/01_safe_marl.md`
- Read: `docs/lit_review_verified/02_comm_constrained_marl.md`
- Read: `docs/lit_review_verified/03_safety_comm_intersection.md`
- Read: `docs/lit_review_verified/04_uav_formation.md`
- Read: `docs/lit_review_verified/05_multi_objective_pareto_marl.md`
- Read: `docs/lit_review_verified/06_cbf_mpc_shielding_safe_marl.md`
- Read: `docs/lit_review_verified/07_graph_attention_scalable_marl.md`
- Read: `docs/lit_review_verified/08_sim2real_uav_marl.md`
- Create: `docs/lit_review_verified/00_synthesis_report.md`

- [ ] **Step 1: Extract verification counts and high-risk entries**

Run:

```bash
rg -n "Verified|Metadata mismatch|Weak match|Not found|Replaced|核验结论摘要|高风险|未找到|疑似" docs/lit_review_verified/0[1-8]_*.md
```

Expected: output gives enough material to compute overall counts and identify high-risk original citations.

- [ ] **Step 2: Write synthesis report structure**

Create `docs/lit_review_verified/00_synthesis_report.md` with these exact sections:

```markdown
# 经核验文献综合报告

## 1. 总体真实性审计

## 2. 八个方向的证据强度

## 3. 原报告中需要修正的结论

## 4. SafeComm-MARL 可保留的创新定位

## 5. 推荐 Related Work 写法

## 6. 高风险引用清单

## 7. 经核验核心参考文献
```

Expected: the synthesis report uses the section names above.

- [ ] **Step 3: Classify evidence strength by direction**

Use this scale:

```text
Strong: core claims supported by multiple verified papers from reliable venues.
Moderate: core claims supported, but direct SafeComm-MARL alignment is partial.
Weak: adjacent literature exists, but direct support for original claims is limited.
High risk: many original entries are unverified or replaced.
```

Expected: each of the eight directions receives one evidence-strength label and a concrete reason.

- [ ] **Step 4: Rewrite SafeComm-MARL positioning**

State only claims supported by verified reports. Use conservative language for gap claims:

```text
Verified literature supports studying safe MARL and communication-constrained MARL as mature but mostly separate threads.
Verified literature supports CBF, safety filters, shielding, and communication scheduling as relevant adjacent methods.
The strongest defensible gap is limited verified work directly integrating safety constraints, explicit communication budgets, and UAV/MARL deployment assumptions in one framework.
Claims that no prior work exists must be avoided unless an exhaustive search protocol supports them.
```

Expected: synthesis avoids absolute novelty claims that exceed the verification evidence.

- [ ] **Step 5: Verify the synthesis locally**

Run:

```bash
rg -n "Strong|Moderate|Weak|High risk|Verified|Metadata mismatch|Weak match|Not found|Replaced" docs/lit_review_verified/00_synthesis_report.md
rg -n "https?://|doi.org|arxiv.org|openreview.net|proceedings.mlr.press|semanticscholar.org|openalex.org" docs/lit_review_verified/00_synthesis_report.md
git diff -- docs/lit_review
```

Expected: evidence-strength labels, audit labels, and source links are present; original reports remain unchanged.

- [ ] **Step 6: Commit synthesis report**

Run:

```bash
git add docs/lit_review_verified/00_synthesis_report.md
git commit -m "docs: synthesize verified literature review"
```

Expected: commit contains only `docs/lit_review_verified/00_synthesis_report.md`.

## Task 11: Final Acceptance Verification

**Files:**
- Read: `docs/lit_review_verified/*.md`
- Verify unchanged: `docs/lit_review/*.md`

- [ ] **Step 1: Verify exactly nine reports exist**

Run:

```bash
find docs/lit_review_verified -maxdepth 1 -type f -name '*.md' | sort
test "$(find docs/lit_review_verified -maxdepth 1 -type f -name '*.md' | wc -l)" -eq 9
```

Expected: the printed list contains exactly:

```text
docs/lit_review_verified/00_synthesis_report.md
docs/lit_review_verified/01_safe_marl.md
docs/lit_review_verified/02_comm_constrained_marl.md
docs/lit_review_verified/03_safety_comm_intersection.md
docs/lit_review_verified/04_uav_formation.md
docs/lit_review_verified/05_multi_objective_pareto_marl.md
docs/lit_review_verified/06_cbf_mpc_shielding_safe_marl.md
docs/lit_review_verified/07_graph_attention_scalable_marl.md
docs/lit_review_verified/08_sim2real_uav_marl.md
```

- [ ] **Step 2: Verify original reports are unchanged**

Run:

```bash
git diff -- docs/lit_review
```

Expected: no output.

- [ ] **Step 3: Verify every report has required sections**

Run:

```bash
for f in docs/lit_review_verified/0[1-8]_*.md; do
  printf '%s\n' "$f"
  rg -n "^## 1\\. 核验结论摘要|^## 2\\. 原文献真实性审计表|^## 3\\. 经核验后的核心文献综述|^## 4\\. 与 SafeComm-MARL 的关系|^## 5\\. 可引用参考文献" "$f"
done
rg -n "^## 1\\. 总体真实性审计|^## 2\\. 八个方向的证据强度|^## 3\\. 原报告中需要修正的结论|^## 4\\. SafeComm-MARL 可保留的创新定位|^## 5\\. 推荐 Related Work 写法|^## 6\\. 高风险引用清单|^## 7\\. 经核验核心参考文献" docs/lit_review_verified/00_synthesis_report.md
```

Expected: every direction report prints five required section headings; synthesis prints seven required section headings.

- [ ] **Step 4: Verify audit labels and source links exist**

Run:

```bash
rg -n "Verified|Metadata mismatch|Weak match|Not found|Replaced" docs/lit_review_verified/*.md
rg -n "https?://|doi.org|arxiv.org|openreview.net|proceedings.mlr.press|semanticscholar.org|openalex.org" docs/lit_review_verified/*.md
```

Expected: every report has audit labels and source links.

- [ ] **Step 5: Scan for unresolved markers and unsupported absolute novelty language**

Run:

```bash
rg -n "(\\bT[A-Z]{2}\\b|F[A-Z]{4}|source[[:space:]]+needed|citation[[:space:]]+needed|[?][?]|没有任何工作|首次|首个|完全空白|证明不存在)" docs/lit_review_verified/*.md
```

Expected: no unresolved markers; any absolute novelty phrase found is removed or rewritten conservatively.

- [ ] **Step 6: Commit final verification cleanup if files changed**

If Step 5 required edits, run:

```bash
git add docs/lit_review_verified/*.md
git commit -m "docs: finalize verified literature review reports"
```

Expected: commit contains only cleanup changes in `docs/lit_review_verified/*.md`.

- [ ] **Step 7: Report completion evidence**

Run:

```bash
git status --short
git log --oneline -5
```

Expected: `docs/lit_review/` is unchanged, `docs/lit_review_verified/` reports are committed or intentionally left staged for review, and remaining unrelated untracked files are explicitly reported.

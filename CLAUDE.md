# CLAUDE.md

本文件为 Claude Code 在此仓库中工作时提供指导。所有交流（回复、提问、注释说明等）均使用**简体中文**。

---

## ⚡ 新 Agent 快速定向（首次进入此仓库必读）

> **如果你是刚被分配到这个仓库的 Claude 或 Codex agent，请先读完本节再做任何操作。**

### 项目一句话

这是一个 MARL 学术研究项目，目标发表 NeurIPS/ICML/ICLR。正在实现 **SafeComm-VoI** 算法（安全感知 VoI 通信调度 + HOCBF-QP + 双约束 Lagrangian + 两阶段 MGDA）。

### 当前进度快照（2026-05-29）

| 阶段 | 状态 | 说明 |
|------|------|------|
| Phase 1：VoI 调度器 + 原型 | ✅ **已完成** | 80 tests passing，commit `4f1e825` |
| Phase 2：完整 SafeComm-VoI | 🔄 **下一步执行** | 提示词已就绪，等待 Codex 执行 |
| Phase 3：PyBullet + 域随机化 | ⬜ 待开始 | 依赖 Phase 2 |
| Phase 4：多任务 + N≥8 扩展 | ⬜ 待开始 | 依赖 Phase 3 |

### Phase 1 已实现的文件（不得修改）

```
envs/uav_formation_env.py       ← 双积分器 UAV 环境（11 tests）
algorithms/scheduler.py         ← VoI 感知调度器（8 tests）
algorithms/networks.py          ← Actor/Critic/MessageEncoder（Phase 2 会扩展）
algorithms/ppo_utils.py         ← RolloutBuffer/GAE/PPO clip（Phase 2 会扩展）
algorithms/safecomm_psched.py   ← SafeCommVoI 主类（Phase 2 会扩展）
configs/default.yaml            ← 默认超参数
train.py                        ← 训练入口（Phase 2 会添加 WandB）
tests/test_env.py               ← 11 tests ✅
tests/test_ppo_utils.py         ← 全部通过 ✅
tests/test_networks.py          ← 全部通过 ✅
tests/test_scheduler.py         ← 8 tests ✅
tests/test_algorithm.py         ← 含 test_phase1_smoke ✅
```

### 立即要做的事：执行 Phase 2

**Phase 2 Codex 综合执行提示词文件**：
`docs/superpowers/plans/2026-05-26-phase2-codex-prompts.md`

该文件包含一个可直接发给 Codex 的综合提示词，Codex 读取后会自主调度子 agent 按顺序完成 Phase 2 的 7 个 Task：

| Task | 内容 | 新增文件 |
|------|------|---------|
| 1 | C+ HOCBF-QP Shield | `algorithms/hocbf_qp.py`, `tests/test_hocbf_qp.py` |
| 2 | IntCostCritic + RolloutBuffer 扩展 | 修改 `networks.py`, `ppo_utils.py` |
| 3 | 双动作 collect_rollout（a^π vs a^*） | 修改 `safecomm_psched.py` |
| 4 | λ_int + Safety Gate EMA + 层级 MGDA | 修改 `safecomm_psched.py` |
| 5 | 5 个 Baseline 包装器 | `baselines/` 目录 |
| 6 | MACPO baseline + test_baselines.py | `baselines/macpo.py`, `tests/test_baselines.py` |
| 7 | evaluate.py + 消融配置 + WandB 集成 | `evaluate.py`, `configs/ablation/`, 修改 `train.py` |

**Phase 2 门控标准（完成后请 Claude 审查）**：
见 `docs/superpowers/plans/2026-05-26-safecomm-voi-master-coordination.md` 第3节"Phase 2 → Phase 3 门控"。

### 新服务器环境搭建

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 WandB（API Key 不在 git 中，必须手动创建 .env）
echo "WANDB_API_KEY=<向用户索取>" > .env

# 3. 验证 Phase 1 环境正常
pytest tests/ -v
# 预期：80 tests passing

# 4. external/ 参考仓库（634MB，git 只追踪 commit hash，需手动重新克隆）
# 参见仓库根目录的迁移说明，或向用户确认是否需要
```

### 协作模型

- **Claude**：阶段门控审查、架构决策、阻塞诊断（不写代码）
- **Codex**：具体代码实现，每个 Task 后运行测试并提交
- **阶段切换**：Codex 完成某阶段所有 Task → 将 `pytest tests/ -v` 完整输出发给 Claude → Claude 逐条审查门控标准 → 出具 ✅/❌ 结论

### 关键架构约束（共 13 条，不得违反）

完整约束见 `docs/superpowers/plans/2026-05-26-safecomm-voi-master-coordination.md` 第4节。最重要的 5 条：

1. **双动作不变项**：buffer 存 a^π，环境执行 a^*，严禁混淆
2. **CBF 缓冲区**：d_buf = v_max × dt_comm，调度器和 QP 使用完全相同公式
3. **QP slack 保留**：ξ ≥ 0 不可去掉，优化变量固定 4 维
4. **Safety gate 用 EMA**：λ_int 更新条件基于 EMA，不用单次 rollout
5. **WandB 强制**：Phase 2 起所有正式训练必须初始化 WandB，API key 仅存 .env

---

## 用户偏好

- 默认使用简体中文回复用户。
- 新增项目文档、调研报告、计划文档和说明性注释默认使用简体中文。
- 论文标题、会议/期刊名、术语、DOI、arXiv 编号、BibTeX 字段和引用元数据可保留英文原文，以保证可检索性和引用准确性。

---

## 项目概览

**safecomm-marl** — 面向安全通信的多智能体强化学习（MARL）框架，应用于通信受限多 UAV 编队控制场景。

**核心问题**：现有安全 MARL（MACPO、MAPPO-Lag）假设全通信；现有通信受限 MARL（DGN、SchedNet）不考虑安全约束——二者的联合优化是 NeurIPS/ICML/ICLR 层面的实质性空白。

**解决方案**：提出 **SafeComm-VoI**（Safety-aware Value-of-Information MARL）算法——VoI 感知调度器（AoI 老化 × CBF 危险度）优先分配通信预算，HOCBF-QP 硬盾 + 双约束 Lagrangian（λ_s + λ_int）+ 两阶段 MGDA 实现安全与通信干预成本的联合优化。

**目标期刊**：NeurIPS / ICML / ICLR

---

## 当前状态

- **阶段**：Phase 1 ✅ 已完成（门控通过），Phase 2 🔄 待执行（提示词已就绪）
- 文献调研（8 个方向，共约 50 篇，已核验）已完成 → `docs/lit_review_verified_2026-05-26/`
- 系统设计规格已完成 → `docs/superpowers/plans/2026-05-26-safecomm-voi-design.md`
- 4 阶段详细实施计划已完成 → `docs/superpowers/plans/`
- 总体协调手册已完成（Claude 审查用）→ `docs/superpowers/plans/2026-05-26-safecomm-voi-master-coordination.md`
- **Phase 2 Codex 执行提示词已就绪** → `docs/superpowers/plans/2026-05-26-phase2-codex-prompts.md`
- Phase 1 源代码已完成：80 tests passing（见顶部"新 Agent 快速定向"节）
- 10 个相关开源仓库已克隆 → `external/`（git 只追踪 commit hash，新服务器需重新克隆）

**协作模型**：Claude 负责阶段审查、架构决策、阻塞诊断；Codex 负责具体代码实现。阶段切换需 Claude 审查通过后方可进入下一阶段。

---

## 问题形式化：CDecPOSG-CB

**Constrained Decentralized Partially Observable Stochastic Game with Communication Budget**

- **状态空间** $S \subset \mathbb{R}^{6n}$：$n$ 个 UAV 的位置 + 速度 + 偏航角（每 UAV 6 维）
- **动作空间** $A_i \subset \mathbb{R}^3$：速度增量控制（$\Delta v_x, \Delta v_y, \Delta v_z$）
- **安全约束**：$\mathbb{E}[\sum_t C^\text{safe}] \leq d$，其中 $C^\text{safe}$ 为碰撞惩罚
- **通信硬约束**：每步全局最多 $k$ 条活跃通信链路

---

## 算法：SafeComm-VoI

**核心组件：**

1. **VoI 感知调度器**（无需训练，构造性保证通信预算 $|E_t| \leq k$）
   - 优先级：$\text{priority}_{ij} = \frac{1 + \beta \cdot \min(\Delta t_\text{comm}/\tau_\text{ref},\ \tau_\text{max})}{\max(h^\text{sched}_{ij},\ \varepsilon)}$
   - 调度 CBF：$h^\text{sched}_{ij} = \max(\|p_i - \hat{p}_j\| - d_\text{min} - d_\text{buf,ij},\ 0)^2$；$d_\text{buf} = v_\text{max} \cdot \Delta t_\text{comm}$
   - 通信图：$E_t = \text{TopK}(E^\text{phys}_t,\ k,\ \text{by priority})$

2. **HOCBF-QP 硬盾**（运行时安全过滤，每 agent 独立 QP）
   - 优化变量：$[a_x, a_y, a_z, \xi]$（4 维，含 slack $\xi \geq 0$）
   - 约束：$2r^T a + 2\|v\|^2 + \alpha_1 \dot{h} + \alpha_2 \psi_1 - m_\text{acc} + \xi \geq 0$
   - 双动作不变项：环境执行 $a^*$（QP 输出），PPO log prob 使用 $a^\pi$（策略输出），两者分开存储

3. **消息编码与聚合**：单头注意力聚合邻居消息（$d_\text{msg} = 64$），参数共享

4. **MAPPO 策略网络**：参数共享 Actor + 集中式 Critic（CTDE）

5. **双约束 Lagrangian**：
   - $\lambda_s$：安全约束（始终激活）
   - $\lambda_\text{int}$：干预成本约束（Safety Gate EMA 激活：$J^\text{safe}_\text{ema} \leq d_\text{safe} + \delta_s$）

6. **两阶段 MGDA 梯度协调**：
   - 第一级：$(g_\text{task},\ \lambda_s \cdot g_\text{safe})$
   - 第二级：$(g_{ts},\ \lambda_\text{int} \cdot g_\text{int})$

---

## 计划代码架构

```
safecomm-marl/
├── envs/
│   ├── uav_formation_env.py       # 轻量双积分器仿真（Phase 1-2）
│   ├── pybullet_uav_env.py        # PyBullet 四旋翼仿真（Phase 3）
│   ├── domain_randomization.py    # DR 包装器，噪声注入 obs（Phase 3）
│   └── pursuit_env.py             # 追逃任务环境（Phase 4）
├── algorithms/
│   ├── safecomm_psched.py         # 核心算法主文件（SafeComm-VoI）
│   ├── scheduler.py               # VoI 感知调度器（TopK + AoI + CBF）
│   ├── hocbf_qp.py                # HOCBF-QP 硬盾（Phase 2）
│   ├── networks.py                # Actor / Critic / MessageEncoder
│   ├── scalable_critic.py         # Transformer N-agnostic Critic（Phase 4）
│   └── ppo_utils.py               # GAE、PPO clip、RolloutBuffer
├── baselines/
│   ├── mappo.py
│   ├── macpo.py
│   ├── mappo_lag.py
│   ├── dgn_mappo.py
│   ├── liu2023.py                 # Liu et al. NeurIPS 2023
│   └── cbf_ppo.py                 # CBF-PPO（Qin RA-L 2023）
├── configs/
│   ├── default.yaml               # 默认超参数
│   └── ablation/                  # 消融实验配置
├── train.py                       # 单任务训练入口
├── train_multitask.py             # 多任务训练入口（Phase 4）
├── evaluate.py                    # 评估入口（支持 --shield on/off）
└── tests/
    ├── test_scheduler.py          # Phase 1
    ├── test_env.py                # Phase 1
    ├── test_algorithm.py          # Phase 1
    ├── test_hocbf_qp.py           # Phase 2
    ├── test_baselines.py          # Phase 2
    ├── test_pybullet_env.py       # Phase 3
    ├── test_pursuit_env.py        # Phase 4
    └── test_scalable_critic.py    # Phase 4
```

---

## 构建与运行命令

```bash
# 安装依赖
pip install -r requirements.txt

# 配置 WandB（首次使用）
export WANDB_API_KEY=$(cat .env | grep WANDB_API_KEY | cut -d= -f2)
# 或直接：export WANDB_API_KEY=<your_key>

# 训练（含 WandB 自动记录）
python train.py --config configs/default.yaml

# 评估（with/without QP Shield）
python evaluate.py --checkpoint checkpoints/final.pt --episodes 20
python evaluate.py --checkpoint checkpoints/final.pt --no_qp

# 单测
pytest tests/test_scheduler.py -v
pytest tests/ -v
```

---

## WandB 训练追踪

所有正式训练必须通过 WandB 追踪，本地开发测试（smoke test）可通过 `WANDB_MODE=disabled` 跳过。

- **账号**：1974693174@qq.com
- **项目名**：`safecomm-marl`
- **API Key**：存储在 `.env` 文件中（已加入 `.gitignore`，**禁止写入任何 git 追踪文件**）
- **本地设置**：
  ```bash
  echo "WANDB_API_KEY=<your_key>" > .env   # .env 已在 .gitignore 中
  ```
- **train.py 集成要求**（Phase 2 起）：
  ```python
  import wandb, os
  from dotenv import load_dotenv
  load_dotenv()  # 读取 .env
  wandb.init(project="safecomm-marl", config=cfg, name=run_name)
  # 每次迭代后：
  wandb.log({...metrics...}, step=agent.total_steps)
  wandb.finish()
  ```
- **WandB run 命名规范**：`{AlgoName}-N{n_agents}-k{k}-{YYYYMMDD_HHMMSS}`
  - 示例：`SafeCommVoI-N4-k4-20260527_143022`
- **必须记录的 metrics**：
  - 每次迭代：`actor_loss`、`critic_loss`、`lambda_s`、`J_safe`、`approx_kl`、`mean_episode_cost`、`mean_formation_error`、`mean_bandwidth_util`
  - Phase 2 起额外：`lambda_int`、`safe_gate_active`、`J_int`、`mean_xi`
- **smoke test 中禁用 WandB**：在测试代码顶部设置 `os.environ["WANDB_MODE"] = "disabled"`

---

## 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `N` | 4（→8） | UAV 数量（Phase 1-3 为 4，Phase 4 扩展至 8） |
| `k` | 4 | 通信预算（每步最多活跃链路数） |
| `d_safe` | 0.1 | 安全约束阈值 |
| `d_int` | 0.05 | 干预成本约束阈值 |
| `d_min` | 0.5 m | 最小安全距离 |
| `d_msg` | 64 | 消息向量维度 |
| 隐藏层 | [256, 256] | Actor / Critic MLP 规模 |
| `obs_dim` | 6 + K_obs×6 | 观测维度（状态 6 维 + K 邻居各 6 维） |
| `β` | 0.5 | VoI 调度器 AoI 老化权重 |
| `τ_ref` | 5 步 | AoI 参考时间窗口 |
| `v_max` | 2.0 m/s | CBF 缓冲区计算用最大速度 |

---

## 六个 Baseline

| 方法 | 安全约束 | 通信约束 |
|------|---------|---------|
| MAPPO | 无 | 无（任务上界） |
| MACPO | CMDP | 无 |
| MAPPO-Lag | Lagrangian | 无 |
| DGN + MAPPO | 无 | 图约束 |
| Liu et al. NeurIPS 2023 | CMDP | 信道质量自适应 |
| CBF-PPO（Qin RA-L 2023） | CBF 硬约束 | 无 |

---

## 文档索引

### 文献调研（已核验版本）
- `docs/lit_review_verified_2026-05-26/00_synthesis_report.md` — 综合报告
- `docs/lit_review_verified_2026-05-26/01_safe_marl.md` — 安全 MARL 方向
- `docs/lit_review_verified_2026-05-26/02_comm_constrained_marl.md` — 通信受限 MARL 方向
- `docs/lit_review_verified_2026-05-26/03_safety_comm_intersection.md` — 安全×通信交叉方向
- `docs/lit_review_verified_2026-05-26/04_uav_formation.md` — 多 UAV 编队控制方向
- `docs/lit_review_verified_2026-05-26/05_multi_objective_pareto_marl.md` — 多目标 Pareto MARL
- `docs/lit_review_verified_2026-05-26/06_cbf_mpc_shielding_safe_marl.md` — CBF/MPC/Shielding 安全方法
- `docs/lit_review_verified_2026-05-26/07_graph_attention_scalable_marl.md` — 图注意力可扩展 MARL
- `docs/lit_review_verified_2026-05-26/08_sim2real_uav_marl.md` — Sim2Real UAV MARL
- `docs/archive/lit_review/` — 旧版文献调研（已归档，仅供参考）

### 设计规格
- `docs/archive/2026-05-25-safecomm-marl-design.md` — 初版设计规格（SafeComm-PSched，已过时）
- `docs/superpowers/plans/2026-05-26-safecomm-voi-design.md` — **当前设计规格**（SafeComm-VoI，算法伪代码、实验设计、理论贡献）
- `docs/superpowers/plans/2026-05-26-safecomm-voi-master-coordination.md` — **Claude 协调手册**（阶段门控标准、12 项架构约束、审查协议）

### 实施计划（供 Codex 执行）
- `docs/superpowers/plans/2026-05-26-safecomm-voi-phase1.md` — Phase 1：VoI 调度器 + 原型（✅ 已完成）
- `docs/superpowers/plans/2026-05-26-safecomm-voi-phase2.md` — Phase 2：完整 SafeComm-VoI（HOCBF-QP + 双约束）（🔄 下一步）
- `docs/superpowers/plans/2026-05-26-safecomm-voi-phase3.md` — Phase 3：PyBullet + 域随机化
- `docs/superpowers/plans/2026-05-26-safecomm-voi-phase4.md` — Phase 4：多任务 + N≥8 扩展

### Codex 执行提示词
- `docs/superpowers/plans/2026-05-26-phase1-codex-prompts.md` — Phase 1 分 Task 提示词（已完成，仅供参考）
- `docs/superpowers/plans/2026-05-26-phase2-codex-prompts.md` — **Phase 2 综合执行提示词**（将文件内容直接发给 Codex，Codex 自主调度子 agent 完成所有 Task）

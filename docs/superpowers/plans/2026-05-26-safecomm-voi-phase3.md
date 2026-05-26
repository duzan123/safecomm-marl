# SafeComm-VoI Phase 3 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Phase 2 训练好的 SafeCommVoI 策略迁移到 PyBullet 四旋翼仿真环境，并引入域随机化，验证从双积分器到高保真动力学 wrapper 的迁移能力。

**Architecture:** 三个核心组件——(1) `PyBulletUAVEnv`：封装 `external/gym-pybullet-drones` 的 `VelocityAviary` 或 `MultiHoverAviary`，暴露与 Phase 1/2 一致的 `reset/step/state/get_physical_graph/compute_cbf_values` 接口；(2) `DomainRandomizationWrapper`：只在 agent observation 和通信消息层注入噪声、延迟和丢包，不能污染 scheduler/QP 使用的真实状态；(3) `configs/pybullet.yaml`：记录外部依赖 commit、Aviary 类型、DR sweep 和迁移训练参数。

**Tech Stack:** Python 3.10+, NumPy, PyTorch, gymnasium, pybullet, gym-pybullet-drones, PyYAML, pytest

**外部代码复用边界（Phase 3）：**
- `external/gym-pybullet-drones`（MIT，commit `90b178a`）是本阶段唯一直接环境依赖。不要手写四旋翼刚体动力学、RPM 动力学或 PID 低层控制；这些由 `gym-pybullet-drones` 提供。
- `VelocityAviary` 优先用于 SafeComm 动作适配，因为它接受多机速度目标；`MultiHoverAviary` 可作为多机 hover 对照。二者都必须通过本项目 adapter 转换成 Phase 1/2 的 obs、reward、cost 和通信图。
- `external/safe-control-gym`（MIT，commit `6b5391d`）只参考 constraint/disturbance、seed 管理和 safe-control logging，不作为主环境替代，不引入其重依赖栈。
- 域随机化范围来自工程压力测试，不写成某篇文献给出的固定推荐值；论文中报告前需要通过消融或平台规格校准。

---

## 文件结构

| 文件 | 操作 | 说明 |
|------|------|------|
| `envs/pybullet_uav_env.py` | 新建 | `gym-pybullet-drones` adapter，保持 SafeComm MARL 接口 |
| `envs/domain_randomization.py` | 新建 | 观测噪声、通信延迟、丢包、预算抖动 wrapper |
| `configs/pybullet.yaml` | 新建 | PyBullet wrapper、Aviary、DR 和迁移训练配置 |
| `requirements.txt` 或环境说明 | 修改 | 增加 `gym-pybullet-drones` 安装说明，优先使用 `external/gym-pybullet-drones` editable 安装 |
| `train.py` | 修改 | 新增 `--env pybullet` 和 `--domain_rand` |
| `evaluate.py` | 修改 | 支持 `--env pybullet`、`--domain_rand`、`--gui` |
| `tests/test_pybullet_env.py` | 新建 | adapter 接口、动作映射、状态真实性、DR wrapper 测试 |

**不修改的文件：** `algorithms/scheduler.py`、`algorithms/hocbf_qp.py`、`algorithms/safecomm_psched.py` 的核心算法逻辑。Phase 3 只替换环境，不改变 SafeComm 算法定义。

---

## Task 1：封装 gym-pybullet-drones（envs/pybullet_uav_env.py）

**Files:**
- Create: `envs/pybullet_uav_env.py`
- Test: `tests/test_pybullet_env.py`

### 步骤

- [ ] **Step 1: 写失败测试**

`tests/test_pybullet_env.py` 至少覆盖：
- `PyBulletUAVEnv(n_agents=4, gui=False, seed=42)` 可创建、`close()` 可重复调用。
- `reset()` 返回 `(obs_list, info)`，`len(obs_list) == n_agents`，每个 obs shape 为 `6 + K_obs * 6`。
- `step(actions)` 接受 `(n_agents, 3)` 的 SafeComm 加速度/速度增量动作，返回 `(obs_list, rewards, costs, done, info)`。
- `state` 返回 `[pos_0, vel_0, ..., pos_n, vel_n]` 的 `6n` 向量，来源必须是底层 Aviary 的真实 kinematic state。
- `get_physical_graph()` 返回所有距离小于 `R_comm` 的 pair，且 `|E_phys|` 不受通信预算 `k` 裁剪。
- `compute_cbf_values()` 使用真实位置计算 `||p_i-p_j||^2-d_min^2`，不使用噪声观测。
- adapter 在 `aviary_type="velocity"` 和 `aviary_type="multi_hover"` 下至少能 reset；训练默认使用 `velocity`。

测试文件中对外部依赖使用 `pytest.importorskip("gym_pybullet_drones")`，避免在未安装依赖时产生误导性失败。

- [ ] **Step 2: 实现 adapter，而非重写动力学**

`PyBulletUAVEnv` 的职责限定为：
- 初始化 `gym_pybullet_drones.envs.VelocityAviary.VelocityAviary` 或 `gym_pybullet_drones.envs.MultiHoverAviary.MultiHoverAviary`。
- 维护目标编队 `goal_positions`、通信半径 `R_comm`、安全距离 `d_min`、episode 计数和 SafeComm reward/cost。
- 将 SafeComm `(n, 3)` 动作映射到底层 Aviary：
  - `VelocityAviary`：将策略动作裁剪为期望速度方向，并附加速度比例列，形成 `(n, 4)` action。
  - `MultiHoverAviary`：仅作为 hover 对照，按底层 action type 进行最小映射，不作为主训练路线。
- 从底层 `obs` 或 `_getDroneStateVector(i)` 提取位置和速度，构造与 `UAVFormationEnv` 一致的局部观测。
- 在 `info` 中记录 `formation_error`、`min_pairwise_dist`、`collision_count`、`aviary_type`、`gym_pybullet_drones_commit`。

禁止事项：
- 不实现自定义 RPM、推力、力矩、四元数积分、碰撞几何或低层 PID。
- 不直接修改 `external/gym-pybullet-drones` 源码；需要改行为时在 adapter 中覆盖。
- 不让 observation noise 影响 `state`、`get_physical_graph()` 或 CBF/QP 输入。

- [ ] **Step 3: 依赖安装和 smoke test**

推荐安装方式：

```bash
cd /root/autodl-tmp/safecomm-marl
python -m pip install -e external/gym-pybullet-drones
python -m pytest tests/test_pybullet_env.py::TestPyBulletUAVEnv -v
```

成功标准：adapter 测试通过，且运行 100 step 不报错、不产生 NaN，`info["aviary_type"] == "velocity"`。

---

## Task 2：域随机化包装器（envs/domain_randomization.py）

**Files:**
- Create: `envs/domain_randomization.py`
- Test: `tests/test_pybullet_env.py`

### 步骤

- [ ] **Step 4: 实现 DomainRandomizationWrapper**

wrapper 支持以下扰动：
- `obs_noise_sigma`：只加到 agent observation 中的位置/相对位置分量。
- `velocity_noise_sigma`：只加到 observation 的速度/相对速度分量。
- `comm_delay_max_steps`：对通信消息使用 per-edge delay buffer。
- `packet_loss_rate`：以概率丢弃通信消息或返回最近一次有效消息。
- `budget_jitter`：可选压力测试，将有效通信预算在 `[k-jitter, k+jitter]` 内扰动，但必须显式记录，不作为主结果默认设置。

wrapper 必须透传底层真实状态接口：
- `state`
- `positions`
- `velocities`
- `get_physical_graph()`
- `compute_cbf_values()`
- `close()`

调度器和 HOCBF-QP 使用真实状态；DR 只影响 agent policy 可见观测和通信消息。这个边界要通过单元测试固定：加噪后 `obs_list` 变化，但 `env.state` 和 `get_physical_graph()` 不变。

- [ ] **Step 5: DR 测试**

新增测试：
- `test_obs_noise_changes_obs_not_state`
- `test_packet_loss_keeps_shape`
- `test_delay_buffer_is_bounded`
- `test_scheduler_uses_clean_physical_graph`
- `test_wrapper_close_closes_base_env`

---

## Task 3：配置、训练和评估接入

**Files:**
- Create: `configs/pybullet.yaml`
- Modify: `train.py`
- Modify: `evaluate.py`

### 步骤

- [ ] **Step 6: 创建 PyBullet 配置**

`configs/pybullet.yaml` 必须包含：

```yaml
env:
  type: "pybullet"
  backend: "gym_pybullet_drones"
  backend_path: "external/gym-pybullet-drones"
  backend_commit: "90b178a"
  aviary_type: "velocity"
  n_agents: 4
  ctrl_freq: 30
  pyb_freq: 240
  max_steps: 400
  d_min: 0.5
  R_comm: 5.0
  v_max: 3.0
  a_max: 2.0
  K_obs_neighbors: 4
  gui: false
  seed: 42

domain_rand:
  enabled: true
  obs_noise_sigma: 0.03
  velocity_noise_sigma: 0.02
  comm_delay_max_steps: 1
  packet_loss_rate: 0.10
  budget_jitter: 0

training:
  total_steps: 500_000
  pretrained_checkpoint: null
  checkpoint_dir: "checkpoints/phase3"
```

随机化范围可通过 ablation/sweep 扩展，但默认配置必须保守，先保证迁移评估稳定。

- [ ] **Step 7: 修改 train.py**

新增环境选择逻辑：
- `env.type == "double_integrator"`：沿用 Phase 1/2。
- `env.type == "pybullet"`：创建 `PyBulletUAVEnv`，再按 `domain_rand.enabled` 包装 `DomainRandomizationWrapper`。
- `--env pybullet` 可覆盖配置。
- `--domain_rand` 强制启用 DR，`--no_domain_rand` 强制关闭 DR。
- `--checkpoint` 或 `training.pretrained_checkpoint` 加载 Phase 2 checkpoint 微调。

- [ ] **Step 8: 修改 evaluate.py**

评估必须支持：
- `--env pybullet`
- `--gui`
- `--domain_rand`
- `--shield on/off`
- 输出 FE、Success、Collision、CVR、MPS、intervention rate、QP slack 均值。

---

## Task 4：集成验证

### 自动化验证

```bash
cd /root/autodl-tmp/safecomm-marl
python -m pytest tests/test_pybullet_env.py -v
python train.py --config configs/pybullet.yaml --env pybullet --total_steps 1000 --device cpu
python evaluate.py --config configs/pybullet.yaml --env pybullet --episodes 2 --shield on
```

### 行为验证

- [ ] `PyBulletUAVEnv.obs_dim == UAVFormationEnv.obs_dim`（相同 `K_obs` 下）
- [ ] `PyBulletUAVEnv.state` 来自真实底层 kinematic state，不来自加噪 observation
- [ ] scheduler 输入仍是真实 `get_physical_graph()`，DR 不破坏 `|E_t| <= k`
- [ ] Phase 2 checkpoint 可加载并运行 100 step
- [ ] `--shield off` 可用于 Shield Independence Eval
- [ ] 运行结束后 PyBullet 连接被关闭，无悬挂 GUI/client

---

## Phase 3 成功标准

| 指标 | 验收条件 |
|------|---------|
| 外部复用 | `gym-pybullet-drones` 通过 adapter 使用，未手写四旋翼动力学 |
| 接口一致性 | `PyBulletUAVEnv` 与 Phase 1/2 环境接口一致 |
| 测试通过率 | `tests/test_pybullet_env.py` 全部通过 |
| 迁移训练 | Phase 2 checkpoint 可加载并在 PyBullet wrapper 中继续训练 |
| 域随机化 | 噪声只影响 observation/通信消息，不影响 scheduler/QP 真实状态 |
| 评估完整性 | PyBullet 模式可报告 FE、Collision、CVR、MPS、intervention、slack |

---

## 文献和代码依据

- `external/gym-pybullet-drones` 对应 Panerati et al., IROS 2021 “Learning to Fly”，是 Phase 3 动力学和多机 quadrotor API 的直接工程依据。
- `external/safe-control-gym` 对应 Safe-Control-Gym，作为扰动、约束、logging 和 seed 管理参考，不作为主环境。
- Domain randomization 只支撑“随机化维度选择”，不支撑默认参数的论文级最优性；默认参数是保守工程起点。

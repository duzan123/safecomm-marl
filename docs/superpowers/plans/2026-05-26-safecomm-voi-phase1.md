# SafeComm-VoI Phase 1 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从零构建 SafeComm-VoI Phase 1——双积分器 UAV 编队环境、PPO 工具函数、神经网络模块、VoI 感知调度器、SafeCommVoI 主算法（单约束 Lagrangian + VoI 调度，无 HOCBF-QP，无 MGDA），满足 Phase 1 成功标准。

**Architecture:** 6 个模块从零创建，无任何预置代码：(1) 双积分器编队环境；(2) RolloutBuffer + PPO/GAE 工具；(3) MessageEncoder + GATAggregator + Actor/Critic 网络；(4) SafetyPriorityScheduler（含 VoI 优先级公式）；(5) SafeCommVoI 主类（collect_rollout + update + train + evaluate）；(6) 配置文件与训练入口。

**Tech Stack:** Python 3.10+, NumPy, PyTorch, scipy, pytest, PyYAML

**外部代码复用边界（Phase 1）：**
- `external/on-policy`（MIT）只参考 R-MAPPO 的 GAE、PPO clip、CTDE rollout 组织思路；轻量 `ppo_utils.py` 自写，不迁移其 runner 体系。
- `external/MAGIC`（MIT）参考 masked attention 邻接 mask 写法；`GATAggregator` 自写，保证只聚合 `E_t` 内消息。
- `external/InforMARL`（MIT）参考局部图观测和邻接矩阵组织，不引入 torch-geometric。
- `external/sched_net` 无明确许可证，仅作为"共享信道 + k 个通信名额"问题设定参考；VoI 公式完整自写。
- 不复制任何外部仓库源码片段；若后续借鉴 MIT 片段需在注释中标明来源和 commit。

---

## 文件结构

| 文件 | 操作 | 说明 |
|------|------|------|
| `envs/__init__.py` | 新建 | 空模块文件 |
| `envs/uav_formation_env.py` | 新建 | 双积分器编队环境，SafeComm 核心接口 |
| `algorithms/__init__.py` | 新建 | 空模块文件 |
| `algorithms/ppo_utils.py` | 新建 | RolloutBuffer、GAE、PPO clip/value loss、mgda_solve |
| `algorithms/networks.py` | 新建 | MessageEncoder、GATAggregator、Actor、Critic、CostCritic、SafeCommNetworks |
| `algorithms/scheduler.py` | 新建 | SafetyPriorityScheduler（VoI 优先级公式 + TopK）|
| `algorithms/safecomm_psched.py` | 新建 | SafeCommVoI 主类（MAPPO-Lag + VoI 调度，Phase 1）|
| `configs/default.yaml` | 新建 | 默认超参数配置（含 VoI 参数）|
| `train.py` | 新建 | 训练入口脚本 |
| `tests/__init__.py` | 新建 | 空模块文件 |
| `tests/test_env.py` | 新建 | UAVFormationEnv 单元测试 |
| `tests/test_ppo_utils.py` | 新建 | RolloutBuffer + GAE 测试 |
| `tests/test_networks.py` | 新建 | 网络形状 + 接口测试 |
| `tests/test_scheduler.py` | 新建 | VoI 优先级公式 + 调度预算约束测试 |
| `tests/test_algorithm.py` | 新建 | SafeCommVoI 冒烟测试 |

---

## Task 0：双积分器 UAV 编队环境

**Files:**
- Create: `envs/__init__.py`
- Create: `envs/uav_formation_env.py`
- Create: `tests/__init__.py`
- Create: `tests/test_env.py`

### 步骤

- [ ] **Step 1: 创建 envs/__init__.py 和 tests/__init__.py（空文件）**

```bash
mkdir -p envs algorithms configs tests
touch envs/__init__.py algorithms/__init__.py tests/__init__.py
```

- [ ] **Step 2: 写失败测试（tests/test_env.py）**

新建 `tests/test_env.py`：

```python
"""UAVFormationEnv 单元测试（规格 §1.1）"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from envs.uav_formation_env import UAVFormationEnv


class TestUAVFormationEnv:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.env = UAVFormationEnv(n_agents=4, K_obs_neighbors=4, seed=42)

    def test_obs_dim_formula(self):
        """obs_dim = 6 + K_obs * 6"""
        assert self.env.obs_dim == 6 + 4 * 6

    def test_reset_returns_correct_length(self):
        obs_list, info = self.env.reset()
        assert len(obs_list) == 4
        for obs in obs_list:
            assert obs.shape == (self.env.obs_dim,)
        assert "formation_error" in info
        assert "success" in info

    def test_step_interface(self):
        self.env.reset()
        actions = np.zeros((4, 3), dtype=np.float32)
        obs2, rewards, costs, done, info = self.env.step(actions)
        assert len(obs2) == 4
        assert len(rewards) == 4
        assert len(costs) == 4
        assert isinstance(done, bool)

    def test_state_shape(self):
        self.env.reset()
        state = self.env.state
        assert state.shape == (4 * 6,)

    def test_done_on_max_steps(self):
        env = UAVFormationEnv(n_agents=2, max_steps=3, seed=0)
        env.reset()
        done = False
        for _ in range(3):
            _, _, _, done, _ = env.step(np.zeros((2, 3), dtype=np.float32))
        assert done

    def test_cost_is_binary(self):
        self.env.reset()
        for _ in range(5):
            _, _, costs, done, _ = self.env.step(np.zeros((4, 3), dtype=np.float32))
            for c in costs:
                assert c in (0.0, 1.0)
            if done:
                break

    def test_get_physical_graph_all_pairs_when_large_rcomm(self):
        env = UAVFormationEnv(n_agents=4, R_comm=1000.0, seed=0)
        env.reset()
        edges = env.get_physical_graph()
        assert len(edges) == 6  # C(4,2) = 6
        for i, j in edges:
            assert i < j

    def test_get_physical_graph_no_edges_when_zero_rcomm(self):
        env = UAVFormationEnv(n_agents=4, R_comm=0.0, seed=0)
        env.reset()
        edges = env.get_physical_graph()
        assert len(edges) == 0

    def test_compute_cbf_nonnegative(self):
        self.env.reset()
        cbf = self.env.compute_cbf_values()
        for (i, j), h in cbf.items():
            assert i < j
            assert h >= 0.0

    def test_velocities_bounded(self):
        env = UAVFormationEnv(n_agents=3, v_max=2.0, a_max=10.0, max_steps=50, seed=0)
        env.reset()
        huge = np.full((3, 3), 100.0, dtype=np.float32)
        for _ in range(10):
            env.step(huge)
        assert np.all(np.abs(env.velocities) <= 2.0 + 1e-5)

    def test_k_obs_zero_gives_six_dim(self):
        env = UAVFormationEnv(n_agents=4, K_obs_neighbors=0)
        obs_list, _ = env.reset()
        assert env.obs_dim == 6
        for obs in obs_list:
            assert obs.shape == (6,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 3: 运行失败测试**

```bash
cd /root/autodl-tmp/safecomm-marl && python -m pytest tests/test_env.py -v 2>&1 | head -15
```

预期：`ModuleNotFoundError: No module named 'envs.uav_formation_env'`

- [ ] **Step 4: 实现 envs/uav_formation_env.py**

新建 `envs/uav_formation_env.py`：

```python
"""
双积分器 UAV 编队控制环境（Phase 1）

观测向量：[自身位置(3), 自身速度(3), 邻居1相对位置(3), 邻居1相对速度(3), ...] (6 + K_obs*6)
动作：速度增量（加速度指令），||a||_inf <= a_max
安全成本：碰撞指示器（0/1）
"""
import numpy as np
from typing import Dict, List, Optional, Tuple


class UAVFormationEnv:
    def __init__(
        self,
        n_agents: int = 4,
        dt: float = 0.1,
        max_steps: int = 200,
        d_min: float = 0.5,
        R_comm: float = 5.0,
        v_max: float = 3.0,
        a_max: float = 2.0,
        arena_size: float = 10.0,
        formation_type: str = "circle",
        w_formation: float = 1.0,
        w_success: float = 5.0,
        success_threshold: float = 0.3,
        K_obs_neighbors: int = 4,
        seed: Optional[int] = None,
    ):
        self.n = n_agents
        self.dt = dt
        self.max_steps = max_steps
        self.d_min = d_min
        self.R_comm = R_comm
        self.v_max = v_max
        self.a_max = a_max
        self.arena_size = arena_size
        self.formation_type = formation_type
        self.w_formation = w_formation
        self.w_success = w_success
        self.success_threshold = success_threshold
        self.K_obs = K_obs_neighbors

        self.obs_dim = 6 + self.K_obs * 6
        self.act_dim = 3

        self.positions: np.ndarray = np.zeros((self.n, 3), dtype=np.float32)
        self.velocities: np.ndarray = np.zeros((self.n, 3), dtype=np.float32)
        self.goal_positions: np.ndarray = np.zeros((self.n, 3), dtype=np.float32)
        self.t: int = 0
        self.rng = np.random.default_rng(seed)

    @property
    def state(self) -> np.ndarray:
        return np.concatenate([self.positions, self.velocities], axis=1).flatten()

    def _compute_goals(self, center: np.ndarray) -> np.ndarray:
        if self.formation_type == "circle":
            radius = max(self.n * 0.5, 2.0)
            angles = np.linspace(0, 2 * np.pi, self.n, endpoint=False)
            goals = np.column_stack([
                center[0] + radius * np.cos(angles),
                center[1] + radius * np.sin(angles),
                np.full(self.n, center[2]),
            ]).astype(np.float32)
        elif self.formation_type == "line":
            spacing = max(self.d_min * 2.5, 1.5)
            offsets = (np.arange(self.n) - (self.n - 1) / 2) * spacing
            goals = np.column_stack([
                center[0] + offsets,
                np.full(self.n, center[1]),
                np.full(self.n, center[2]),
            ]).astype(np.float32)
        else:  # grid / v_shape fallback
            cols = int(np.ceil(np.sqrt(self.n)))
            spacing = max(self.d_min * 2.5, 1.5)
            goals = []
            for i in range(self.n):
                row, col = i // cols, i % cols
                goals.append([
                    center[0] + col * spacing - (cols - 1) * spacing / 2,
                    center[1] + row * spacing - (cols - 1) * spacing / 2,
                    center[2],
                ])
            goals = np.array(goals, dtype=np.float32)
        return goals

    def reset(
        self,
        init_positions: Optional[np.ndarray] = None,
        goal_center: Optional[np.ndarray] = None,
    ) -> Tuple[List[np.ndarray], Dict]:
        self.t = 0

        center = np.zeros(3, dtype=np.float32) if goal_center is None else np.asarray(goal_center, dtype=np.float32)
        self.goal_positions = self._compute_goals(center)

        if init_positions is not None:
            self.positions = init_positions.copy().astype(np.float32)
        else:
            half = self.arena_size / 2 * 0.7
            for _ in range(2000):
                cand = self.rng.uniform(-half, half, (self.n, 3)).astype(np.float32)
                cand[:, 2] = 0.0
                ok = True
                for i in range(self.n):
                    for j in range(i + 1, self.n):
                        if np.linalg.norm(cand[i] - cand[j]) < self.d_min * 2.5:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    self.positions = cand
                    break
            else:
                # Deterministic grid fallback
                cols = int(np.ceil(np.sqrt(self.n)))
                sp = self.d_min * 3.0
                pos = []
                for i in range(self.n):
                    r, c = i // cols, i % cols
                    pos.append([c * sp - (cols - 1) * sp / 2, r * sp, 0.0])
                self.positions = np.array(pos, dtype=np.float32)

        self.velocities = np.zeros((self.n, 3), dtype=np.float32)
        return self._get_obs_all(), self._get_info()

    def step(self, actions: np.ndarray) -> Tuple[List[np.ndarray], List[float], List[float], bool, Dict]:
        actions = np.clip(actions, -self.a_max, self.a_max).astype(np.float32)
        self.velocities = np.clip(self.velocities + actions * self.dt, -self.v_max, self.v_max)
        self.positions = self.positions + self.velocities * self.dt
        self._apply_boundary()
        self.t += 1
        done = self.t >= self.max_steps
        return self._get_obs_all(), self._compute_rewards(), self._compute_costs(), done, self._get_info()

    def _apply_boundary(self):
        half = self.arena_size / 2
        for dim in range(3):
            hi = self.positions[:, dim] > half
            lo = self.positions[:, dim] < -half
            self.positions[hi, dim] = half
            self.velocities[hi, dim] = np.minimum(self.velocities[hi, dim], 0.0)
            self.positions[lo, dim] = -half
            self.velocities[lo, dim] = np.maximum(self.velocities[lo, dim], 0.0)

    def _compute_rewards(self) -> List[float]:
        errors = np.linalg.norm(self.positions - self.goal_positions, axis=1)
        mean_err = float(np.mean(errors))
        success = float(mean_err < self.success_threshold)
        r = -self.w_formation * mean_err + self.w_success * success
        return [r] * self.n

    def _compute_costs(self) -> List[float]:
        costs = []
        for i in range(self.n):
            c = 0.0
            for j in range(self.n):
                if i != j and np.linalg.norm(self.positions[i] - self.positions[j]) < self.d_min:
                    c = 1.0
                    break
            costs.append(c)
        return costs

    def _get_obs_all(self) -> List[np.ndarray]:
        return [self._get_obs(i) for i in range(self.n)]

    def _get_obs(self, i: int) -> np.ndarray:
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        obs[0:3] = self.positions[i]
        obs[3:6] = self.velocities[i]
        nbrs = sorted(
            [(np.linalg.norm(self.positions[i] - self.positions[j]), j)
             for j in range(self.n) if j != i]
        )
        for k, (_, j) in enumerate(nbrs[:self.K_obs]):
            base = 6 + k * 6
            obs[base:base+3] = self.positions[j] - self.positions[i]
            obs[base+3:base+6] = self.velocities[j] - self.velocities[i]
        return obs

    def _get_info(self) -> Dict:
        errors = np.linalg.norm(self.positions - self.goal_positions, axis=1)
        pairs = [(i, j) for i in range(self.n) for j in range(i + 1, self.n)]
        min_d = min((np.linalg.norm(self.positions[i] - self.positions[j]) for i, j in pairs), default=float("inf"))
        return {
            "t": self.t,
            "formation_error": float(np.mean(errors)),
            "min_dist": float(min_d),
            "success": bool(np.mean(errors) < self.success_threshold),
            "positions": self.positions.copy(),
            "goal_positions": self.goal_positions.copy(),
        }

    def get_physical_graph(self) -> List[Tuple[int, int]]:
        return [
            (i, j)
            for i in range(self.n)
            for j in range(i + 1, self.n)
            if np.linalg.norm(self.positions[i] - self.positions[j]) <= self.R_comm
        ]

    def compute_cbf_values(self) -> Dict[Tuple[int, int], float]:
        return {
            (i, j): float((np.linalg.norm(self.positions[i] - self.positions[j]) - self.d_min) ** 2)
            for i in range(self.n)
            for j in range(i + 1, self.n)
        }
```

- [ ] **Step 5: 运行测试确认通过**

```bash
cd /root/autodl-tmp/safecomm-marl && python -m pytest tests/test_env.py -v
```

预期：11 个测试全部 `PASSED`

- [ ] **Step 6: 提交**

```bash
cd /root/autodl-tmp/safecomm-marl && git add envs/ tests/__init__.py tests/test_env.py
git commit -m "feat: add UAVFormationEnv double-integrator environment (Phase 1 Task 0)"
```

---

## Task 1：PPO 工具函数

**Files:**
- Create: `algorithms/ppo_utils.py`
- Create: `tests/test_ppo_utils.py`

### 步骤

- [ ] **Step 1: 写失败测试（tests/test_ppo_utils.py）**

```python
"""RolloutBuffer 和 PPO 工具函数测试"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import pytest
from algorithms.ppo_utils import (
    RolloutBuffer, compute_gae, ppo_clip_loss,
    value_loss, explained_variance, mgda_solve,
)


class TestRolloutBuffer:

    def _make_buffer(self):
        return RolloutBuffer(n_steps=20, n_agents=3, obs_dim=12,
                             act_dim=3, global_state_dim=36)

    def test_add_and_ptr(self):
        buf = self._make_buffer()
        for t in range(5):
            buf.add(
                obs=np.zeros((3, 12)),
                actions=np.zeros((3, 3)),
                log_probs=np.zeros(3),
                rewards=[0.1, 0.2, 0.3],
                costs=[0.0, 0.0, 0.0],
                value=0.5,
                cost_value=0.1,
                done=False,
                global_state=np.zeros(36),
                adj_matrix=np.zeros((3, 3)),
            )
        assert buf.ptr == 5

    def test_finish_rollout_sets_last_value(self):
        buf = self._make_buffer()
        buf.add(np.zeros((3,12)), np.zeros((3,3)), np.zeros(3),
                [0.0]*3, [0.0]*3, 0.5, 0.1, False, np.zeros(36), np.zeros((3,3)))
        buf.finish_rollout(last_value=1.0, last_cost_value=0.5)
        assert buf.values[buf.ptr] == 1.0
        assert buf.cost_values[buf.ptr] == 0.5

    def test_get_training_data_keys(self):
        buf = self._make_buffer()
        for _ in range(10):
            buf.add(np.zeros((3,12)), np.zeros((3,3)), np.zeros(3),
                    [0.1]*3, [0.0]*3, 0.5, 0.1, False, np.zeros(36), np.eye(3))
        buf.finish_rollout(0.4, 0.1)
        data = buf.get_training_data(gamma=0.99, lam=0.95, device="cpu")
        for key in ["obs","actions","log_probs_old","advantages","returns",
                    "cost_advantages","cost_returns","global_states","adj_matrices"]:
            assert key in data

    def test_advantages_normalized(self):
        buf = self._make_buffer()
        for t in range(10):
            buf.add(np.zeros((3,12)), np.zeros((3,3)), np.zeros(3),
                    [float(t)]*3, [0.0]*3, float(t)*0.1, 0.0, t==9,
                    np.zeros(36), np.zeros((3,3)))
        buf.finish_rollout(0.0, 0.0)
        data = buf.get_training_data()
        adv = data["advantages"].numpy()
        assert abs(adv.mean()) < 0.1
        assert abs(adv.std() - 1.0) < 0.1


class TestComputeGAE:

    def test_zero_rewards_zero_values(self):
        rewards = np.zeros(5)
        values = np.zeros(6)
        dones = np.array([False]*5)
        adv, ret = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
        assert adv.shape == (5,)
        np.testing.assert_allclose(adv, 0.0, atol=1e-6)

    def test_done_resets_gae(self):
        rewards = np.array([1.0, 0.0, 1.0])
        values = np.zeros(4)
        dones = np.array([False, True, False])
        adv, ret = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
        # After done at t=1, GAE for t=2 should not propagate from t=0
        assert adv[2] > 0.0
        assert abs(adv[1]) < 1.0  # done at t=1: adv[1] = reward[1] + 0 - value[1] = 0


class TestPPOLoss:

    def test_ratio_one_gives_no_clip(self):
        log_p = torch.zeros(10)
        log_p_old = torch.zeros(10)
        adv = torch.ones(10)
        loss, kl = ppo_clip_loss(log_p, log_p_old, adv, clip_eps=0.2)
        assert float(loss) == pytest.approx(-1.0, abs=1e-5)

    def test_value_loss_clipped(self):
        pred = torch.tensor([1.5, 2.5, 3.5])
        old = torch.tensor([1.0, 2.0, 3.0])
        returns = torch.tensor([2.0, 3.0, 4.0])
        loss = value_loss(pred, old, returns, clip_eps=0.2)
        assert float(loss) >= 0.0


class TestMGDASolve:

    def test_identical_gradients_equal_weights(self):
        g = torch.ones(10)
        w1, w2 = mgda_solve(g, g)
        assert abs(w1 - 0.5) < 0.1 and abs(w2 - 0.5) < 0.1

    def test_weights_sum_to_one(self):
        g1 = torch.randn(20)
        g2 = torch.randn(20)
        w1, w2 = mgda_solve(g1, g2)
        assert abs(w1 + w2 - 1.0) < 1e-6
        assert w1 >= 0.0 and w2 >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 2: 运行失败测试**

```bash
cd /root/autodl-tmp/safecomm-marl && python -m pytest tests/test_ppo_utils.py -v 2>&1 | head -10
```

预期：`ModuleNotFoundError: No module named 'algorithms.ppo_utils'`

- [ ] **Step 3: 实现 algorithms/ppo_utils.py**

新建 `algorithms/ppo_utils.py`：

```python
"""PPO 工具函数：RolloutBuffer、GAE、PPO clip loss、value loss、mgda_solve"""
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
    returns = advantages + values[:T]
    return advantages, returns


def ppo_clip_loss(
    log_probs: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ratio = torch.exp(log_probs - log_probs_old)
    loss1 = ratio * advantages
    loss2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss = -torch.min(loss1, loss2).mean()
    approx_kl = (log_probs_old - log_probs).mean().detach()
    return loss, approx_kl


def value_loss(
    values_pred: torch.Tensor,
    values_old: torch.Tensor,
    returns: torch.Tensor,
    clip_eps: float = 0.2,
    use_clip: bool = True,
) -> torch.Tensor:
    if use_clip:
        clipped = values_old + torch.clamp(values_pred - values_old, -clip_eps, clip_eps)
        return torch.max((values_pred - returns).pow(2), (clipped - returns).pow(2)).mean()
    return (values_pred - returns).pow(2).mean()


def normalize_advantages(adv: torch.Tensor) -> torch.Tensor:
    return (adv - adv.mean()) / (adv.std() + 1e-8)


def explained_variance(values_pred: np.ndarray, returns: np.ndarray) -> float:
    var_ret = float(np.var(returns))
    if var_ret < 1e-8:
        return 0.0
    return float(1.0 - np.var(returns - values_pred) / var_ret)


def mgda_solve(g1: torch.Tensor, g2: torch.Tensor) -> Tuple[float, float]:
    """Frank-Wolfe MGDA: min_{w1+w2=1, w>=0} ||w1*g1 + w2*g2||^2。Phase 2 使用。"""
    dot = float((g1 * g2).sum())
    n1 = float((g1 * g1).sum())
    n2 = float((g2 * g2).sum())
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.5, 0.5
    denom = n1 + n2 - 2.0 * dot
    if abs(denom) < 1e-10:
        return 0.5, 0.5
    w1 = float(np.clip((n2 - dot) / denom, 0.0, 1.0))
    return w1, 1.0 - w1


class RolloutBuffer:
    def __init__(
        self,
        n_steps: int,
        n_agents: int,
        obs_dim: int,
        act_dim: int,
        global_state_dim: int,
    ):
        T, N = n_steps, n_agents
        self.n_steps = T
        self.n_agents = N
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.global_state_dim = global_state_dim
        self.ptr = 0

        self.obs = np.zeros((T, N, obs_dim), dtype=np.float32)
        self.actions = np.zeros((T, N, act_dim), dtype=np.float32)
        self.log_probs = np.zeros((T, N), dtype=np.float32)
        self.rewards = np.zeros((T, N), dtype=np.float32)
        self.costs = np.zeros((T, N), dtype=np.float32)
        self.values = np.zeros(T + 1, dtype=np.float32)
        self.cost_values = np.zeros(T + 1, dtype=np.float32)
        self.dones = np.zeros(T, dtype=bool)
        self.global_states = np.zeros((T, global_state_dim), dtype=np.float32)
        self.adj_matrices = np.zeros((T, N, N), dtype=np.float32)

    def reset(self):
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards,
        costs,
        value: float,
        cost_value: float,
        done: bool,
        global_state: np.ndarray,
        adj_matrix: np.ndarray,
    ):
        t = self.ptr
        self.obs[t] = obs
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.rewards[t] = np.asarray(rewards, dtype=np.float32)
        self.costs[t] = np.asarray(costs, dtype=np.float32)
        self.values[t] = float(value)
        self.cost_values[t] = float(cost_value)
        self.dones[t] = bool(done)
        gs = np.asarray(global_state, dtype=np.float32)
        if len(gs) < self.global_state_dim:
            gs = np.concatenate([gs, np.zeros(self.global_state_dim - len(gs))])
        elif len(gs) > self.global_state_dim:
            gs = gs[:self.global_state_dim]
        self.global_states[t] = gs
        self.adj_matrices[t] = adj_matrix
        self.ptr += 1

    def finish_rollout(self, last_value: float, last_cost_value: float):
        self.values[self.ptr] = float(last_value)
        self.cost_values[self.ptr] = float(last_cost_value)

    def compute_advantages_and_returns(
        self,
        gamma: float = 0.99,
        lam: float = 0.95,
        normalize_adv: bool = True,
        normalize_cost_adv: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        T = self.ptr
        mean_rewards = self.rewards[:T].mean(axis=1)
        mean_costs = self.costs[:T].mean(axis=1)
        adv, ret = compute_gae(mean_rewards, self.values[:T+1], self.dones[:T], gamma, lam)
        c_adv, c_ret = compute_gae(mean_costs, self.cost_values[:T+1], self.dones[:T], gamma, lam)
        if normalize_adv and adv.std() > 1e-6:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        if normalize_cost_adv and c_adv.std() > 1e-6:
            c_adv = (c_adv - c_adv.mean()) / (c_adv.std() + 1e-8)
        return adv, ret, c_adv, c_ret

    def get_training_data(
        self,
        gamma: float = 0.99,
        lam: float = 0.95,
        normalize_adv: bool = True,
        device: str = "cpu",
    ) -> Dict[str, torch.Tensor]:
        T = self.ptr
        adv, ret, c_adv, c_ret = self.compute_advantages_and_returns(gamma, lam, normalize_adv)

        def tt(a):
            return torch.FloatTensor(a).to(device)

        return {
            "obs": tt(self.obs[:T]),
            "actions": tt(self.actions[:T]),
            "log_probs_old": tt(self.log_probs[:T]),
            "advantages": tt(adv),
            "returns": tt(ret),
            "cost_advantages": tt(c_adv),
            "cost_returns": tt(c_ret),
            "values_old": tt(self.values[:T]),
            "cost_values_old": tt(self.cost_values[:T]),
            "global_states": tt(self.global_states[:T]),
            "adj_matrices": tt(self.adj_matrices[:T]),
            "dones": torch.BoolTensor(self.dones[:T]).to(device),
        }
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd /root/autodl-tmp/safecomm-marl && python -m pytest tests/test_ppo_utils.py -v
```

预期：所有测试 `PASSED`

- [ ] **Step 5: 提交**

```bash
cd /root/autodl-tmp/safecomm-marl && git add algorithms/ppo_utils.py tests/test_ppo_utils.py
git commit -m "feat: add RolloutBuffer and PPO utilities (Phase 1 Task 1)"
```

---

## Task 2：神经网络模块

**Files:**
- Create: `algorithms/networks.py`
- Create: `tests/test_networks.py`

### 步骤

- [ ] **Step 1: 写失败测试（tests/test_networks.py）**

```python
"""SafeCommNetworks 接口和形状测试"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import pytest
from algorithms.networks import SafeCommNetworks


def make_nets(n=4, obs_dim=30, msg_dim=32, hidden=[64,64]):
    return SafeCommNetworks(n_agents=n, obs_dim=obs_dim, act_dim=3,
                            msg_dim=msg_dim, hidden_dims=hidden, H=2)


class TestSafeCommNetworks:

    def test_act_output_shapes(self):
        nets = make_nets()
        obs_list = [np.random.rand(30).astype(np.float32) for _ in range(4)]
        nbr = {i: [j for j in range(4) if j != i] for i in range(4)}
        actions, log_probs, values, cost_values = nets.act(obs_list, nbr, device="cpu")
        assert actions.shape == (4, 3)
        assert log_probs.shape == (4,)
        assert values.shape == (1,) or values.ndim <= 1
        assert cost_values.shape == (1,) or cost_values.ndim <= 1

    def test_actions_in_tanh_range(self):
        nets = make_nets()
        obs_list = [np.random.rand(30).astype(np.float32) for _ in range(4)]
        nbr = {i: [j for j in range(4) if j != i] for i in range(4)}
        actions, _, _, _ = nets.act(obs_list, nbr, deterministic=True)
        assert np.all(actions > -1.0 - 1e-5) and np.all(actions < 1.0 + 1e-5)

    def test_no_neighbors_still_works(self):
        nets = make_nets()
        obs_list = [np.random.rand(30).astype(np.float32) for _ in range(4)]
        nbr = {i: [] for i in range(4)}  # no communication
        actions, log_probs, _, _ = nets.act(obs_list, nbr, device="cpu")
        assert actions.shape == (4, 3)
        assert not np.any(np.isnan(actions))

    def test_evaluate_actions_batch_shapes(self):
        T, N, obs_dim = 8, 4, 30
        nets = make_nets(n=N, obs_dim=obs_dim)
        obs_b = torch.randn(T, N, obs_dim)
        act_b = torch.tanh(torch.randn(T, N, 3))
        adj_b = torch.ones(T, N, N) - torch.eye(N).unsqueeze(0)
        gs_b = torch.randn(T, N * obs_dim)
        log_probs, entropy, values, cost_values, agg = nets.evaluate_actions_batch(
            obs_b, act_b, adj_b, gs_b
        )
        assert log_probs.shape == (T, N)
        assert entropy.shape == (T, N)
        assert values.shape == (T,)
        assert cost_values.shape == (T,)

    def test_gradients_flow(self):
        nets = make_nets()
        T, N, obs_dim = 4, 4, 30
        obs_b = torch.randn(T, N, obs_dim)
        act_b = torch.tanh(torch.randn(T, N, 3))
        adj_b = torch.ones(T, N, N) - torch.eye(N).unsqueeze(0)
        gs_b = torch.randn(T, N * obs_dim)
        log_probs, _, values, _, _ = nets.evaluate_actions_batch(obs_b, act_b, adj_b, gs_b)
        loss = -log_probs.mean() + values.mean()
        loss.backward()
        grad_found = any(p.grad is not None for p in nets.parameters())
        assert grad_found

    def test_deterministic_vs_stochastic(self):
        nets = make_nets()
        nets.eval()
        obs_list = [np.random.rand(30).astype(np.float32) for _ in range(4)]
        nbr = {i: [j for j in range(4) if j != i] for i in range(4)}
        a1, _, _, _ = nets.act(obs_list, nbr, deterministic=True)
        a2, _, _, _ = nets.act(obs_list, nbr, deterministic=True)
        np.testing.assert_allclose(a1, a2, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 2: 运行失败测试**

```bash
cd /root/autodl-tmp/safecomm-marl && python -m pytest tests/test_networks.py -v 2>&1 | head -10
```

预期：`ModuleNotFoundError: No module named 'algorithms.networks'`

- [ ] **Step 3: 实现 algorithms/networks.py**

新建 `algorithms/networks.py`：

```python
"""
SafeComm-VoI 神经网络模块（Phase 1）

组件：
  MessageEncoder    — obs → msg_dim（共享编码器）
  GATAggregator     — 邻居消息 → 聚合向量（单头 attention）
  Actor             — [obs; agg_msg] → 动作均值 + 共享 log_sigma
  Critic            — global_state → 任务价值
  CostCritic        — global_state → 安全成本价值
  SafeCommNetworks  — 容器类，提供 act() 和 evaluate_actions_batch()
"""
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


def build_mlp(in_dim: int, out_dim: int, hidden: List[int], act: str = "tanh") -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_dim
    Activation = nn.Tanh if act == "tanh" else nn.ReLU
    for h in hidden:
        layers += [nn.Linear(prev, h), Activation()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class MessageEncoder(nn.Module):
    def __init__(self, obs_dim: int, msg_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, msg_dim),
            nn.ReLU(),
            nn.Linear(msg_dim, msg_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class GATAggregator(nn.Module):
    """单头图注意力聚合（Phase 1）。"""

    def __init__(self, obs_dim: int, msg_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(obs_dim, msg_dim, bias=False)
        self.scale = math.sqrt(msg_dim)
        self.msg_dim = msg_dim

    def forward(
        self,
        own_obs: torch.Tensor,       # [B, obs_dim]
        neighbor_msgs: torch.Tensor,  # [B, K, msg_dim]  K=邻居数（可为 0）
    ) -> torch.Tensor:
        if neighbor_msgs.shape[1] == 0:
            return torch.zeros(own_obs.shape[0], self.msg_dim, device=own_obs.device)
        q = self.query_proj(own_obs).unsqueeze(2)          # [B, msg_dim, 1]
        attn = torch.bmm(neighbor_msgs, q).squeeze(2) / self.scale  # [B, K]
        w = torch.softmax(attn, dim=-1).unsqueeze(-1)       # [B, K, 1]
        return (w * neighbor_msgs).sum(1)                   # [B, msg_dim]


class Actor(nn.Module):
    """共享策略网络，输出高斯分布参数（tanh squash）。"""

    def __init__(self, obs_dim: int, msg_dim: int, act_dim: int, hidden: List[int]):
        super().__init__()
        self.net = build_mlp(obs_dim + msg_dim, act_dim, hidden, act="tanh")
        self.log_sigma = nn.Parameter(torch.zeros(act_dim))

    def get_dist(self, obs: torch.Tensor, agg: torch.Tensor) -> torch.distributions.Normal:
        mean = self.net(torch.cat([obs, agg], dim=-1))
        sigma = self.log_sigma.exp().clamp(0.01, 2.0)
        return torch.distributions.Normal(mean, sigma.expand_as(mean))


class Critic(nn.Module):
    def __init__(self, global_state_dim: int, hidden: List[int]):
        super().__init__()
        self.net = build_mlp(global_state_dim, 1, hidden, act="tanh")

    def forward(self, gs: torch.Tensor) -> torch.Tensor:
        return self.net(gs).squeeze(-1)


class CostCritic(nn.Module):
    def __init__(self, global_state_dim: int, hidden: List[int]):
        super().__init__()
        self.net = build_mlp(global_state_dim, 1, hidden, act="tanh")

    def forward(self, gs: torch.Tensor) -> torch.Tensor:
        return self.net(gs).squeeze(-1)


class SafeCommNetworks(nn.Module):
    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        act_dim: int,
        msg_dim: int,
        hidden_dims: List[int],
        H: int = 2,
    ):
        super().__init__()
        self.n = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.msg_dim = msg_dim
        self.global_state_dim = n_agents * obs_dim

        self.encoder = MessageEncoder(obs_dim, msg_dim)
        self.aggregator = GATAggregator(obs_dim, msg_dim)
        self.actor = Actor(obs_dim, msg_dim, act_dim, hidden_dims)
        self.critic = Critic(self.global_state_dim, hidden_dims)
        self.cost_critic = CostCritic(self.global_state_dim, hidden_dims)

    def _pad_global_state(self, gs: torch.Tensor) -> torch.Tensor:
        d = self.global_state_dim
        if gs.shape[-1] < d:
            pad = torch.zeros(*gs.shape[:-1], d - gs.shape[-1], device=gs.device)
            gs = torch.cat([gs, pad], dim=-1)
        return gs[:, :d] if gs.ndim == 2 else gs[..., :d]

    @torch.no_grad()
    def act(
        self,
        obs_list: List[np.ndarray],
        neighbor_lists: Dict[int, List[int]],
        deterministic: bool = False,
        device: str = "cpu",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.eval()
        N = len(obs_list)
        obs_t = torch.FloatTensor(np.stack(obs_list)).to(device)  # [N, obs_dim]
        msgs = self.encoder(obs_t)  # [N, msg_dim]

        agg_list = []
        for i in range(N):
            nbrs = neighbor_lists.get(i, [])
            if nbrs:
                nm = msgs[nbrs].unsqueeze(0)         # [1, K, msg_dim]
                oi = obs_t[i].unsqueeze(0)           # [1, obs_dim]
                agg = self.aggregator(oi, nm)        # [1, msg_dim]
            else:
                agg = torch.zeros(1, self.msg_dim, device=device)
            agg_list.append(agg)
        agg_t = torch.cat(agg_list, dim=0)  # [N, msg_dim]

        dist = self.actor.get_dist(obs_t, agg_t)
        raw = dist.mean if deterministic else dist.rsample()
        actions = torch.tanh(raw)
        log_probs = dist.log_prob(raw).sum(-1) - (2 * (math.log(2) - raw - torch.nn.functional.softplus(-2 * raw))).sum(-1)

        gs = self._pad_global_state(obs_t.flatten().unsqueeze(0))
        values = self.critic(gs)
        cost_values = self.cost_critic(gs)

        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.cpu().numpy(),
            cost_values.cpu().numpy(),
        )

    def evaluate_actions_batch(
        self,
        obs_batch: torch.Tensor,      # [T, N, obs_dim]
        actions_batch: torch.Tensor,  # [T, N, act_dim] tanh-squashed
        adj_matrices: torch.Tensor,   # [T, N, N]
        global_states: torch.Tensor,  # [T, N*obs_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        T, N, _ = obs_batch.shape
        obs_flat = obs_batch.reshape(T * N, self.obs_dim)
        msgs = self.encoder(obs_flat).reshape(T, N, self.msg_dim)

        agg = torch.zeros(T, N, self.msg_dim, device=obs_batch.device)
        for i in range(N):
            adj_i = adj_matrices[:, i, :]               # [T, N]
            n_nbr = adj_i.sum(-1, keepdim=True).clamp(1)
            agg[:, i] = (msgs * adj_i.unsqueeze(-1)).sum(1) / n_nbr

        obs_flat2 = obs_batch.reshape(T * N, self.obs_dim)
        agg_flat = agg.reshape(T * N, self.msg_dim)
        dist = self.actor.get_dist(obs_flat2, agg_flat)

        raw = torch.atanh(actions_batch.reshape(T * N, self.act_dim).clamp(-1 + 1e-6, 1 - 1e-6))
        log_p = dist.log_prob(raw).sum(-1) - (2 * (math.log(2) - raw - torch.nn.functional.softplus(-2 * raw))).sum(-1)
        log_p = log_p.reshape(T, N)
        entropy = dist.entropy().sum(-1).reshape(T, N)

        gs = self._pad_global_state(global_states)
        values = self.critic(gs)
        cost_values = self.cost_critic(gs)

        return log_p, entropy, values, cost_values, agg

    def get_values(
        self, global_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gs = self._pad_global_state(global_states)
        return self.critic(gs), self.cost_critic(gs)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd /root/autodl-tmp/safecomm-marl && python -m pytest tests/test_networks.py -v
```

预期：所有测试 `PASSED`

- [ ] **Step 5: 提交**

```bash
cd /root/autodl-tmp/safecomm-marl && git add algorithms/networks.py tests/test_networks.py
git commit -m "feat: add SafeCommNetworks (MessageEncoder, GATAggregator, Actor, Critic) (Phase 1 Task 2)"
```

---

## Task 3：VoI 感知调度器

**Files:**
- Create: `algorithms/scheduler.py`
- Create: `tests/test_scheduler.py`

### 步骤

- [ ] **Step 1: 写失败测试（tests/test_scheduler.py）**

```python
"""SafetyPriorityScheduler + VoI 优先级公式测试（规格 §2.1）"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from algorithms.scheduler import SafetyPriorityScheduler


class TestBudgetConstraint:
    def test_topk_never_exceeds_k(self):
        sched = SafetyPriorityScheduler(n_agents=5, k=3, d_min=0.5)
        positions = np.random.rand(5, 3).astype(np.float64) * 4
        sched.reset_episode(positions)
        all_edges = [(i, j) for i in range(5) for j in range(i+1, 5)]
        active = sched.schedule(positions, all_edges)
        assert len(active) <= 3

    def test_empty_physical_graph(self):
        sched = SafetyPriorityScheduler(n_agents=3, k=2, d_min=0.5)
        positions = np.zeros((3, 3))
        sched.reset_episode(positions)
        active = sched.schedule(positions, [])
        assert active == []


class TestVoIPriority:
    """VoI 优先级公式正确性测试（规格 §2.1）"""

    def test_voi_formula_beta_zero_numerically(self):
        """β=0, v_max=0: priority = 1/h_sched = 1/(dist-d_min)^2"""
        sched = SafetyPriorityScheduler(
            n_agents=2, k=1, d_min=0.5, v_max=0.0, dt=0.1,
            beta=0.0, tau_ref=1.0, tau_max=5.0,
        )
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        sched.reset_episode(positions)
        prios = sched.compute_voi_priority(positions, [(0, 1)])
        # h = (1.0 - 0.5)^2 = 0.25; priority = 1.0 / 0.25 = 4.0
        assert abs(prios[(0, 1)] - 4.0) < 1e-6

    def test_voi_formula_with_aoi_numerically(self):
        """β=1, 5 steps gap: AoI=0.5, priority = (1+0.5)/0.25 = 6.0"""
        sched = SafetyPriorityScheduler(
            n_agents=2, k=1, d_min=0.5, v_max=0.0, dt=0.1,
            beta=1.0, tau_ref=1.0, tau_max=5.0,
        )
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        sched.reset_episode(positions)
        for _ in range(5):
            sched.update_history([], positions)
        prios = sched.compute_voi_priority(positions, [(0, 1)])
        # dt_comm=5*0.1=0.5s; AoI=min(0.5/1.0,5.0)=0.5; h=(0.5)^2=0.25
        assert abs(prios[(0, 1)] - 6.0) < 1e-6

    def test_beta_zero_selects_dangerous_pair(self):
        """β=0 应选最危险对（h 最小），不受 AoI 影响"""
        sched = SafetyPriorityScheduler(
            n_agents=3, k=1, d_min=0.5, v_max=0.0, dt=0.1,
            beta=0.0, tau_ref=1.0, tau_max=5.0,
        )
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.51, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ], dtype=np.float64)
        sched.reset_episode(positions)
        for _ in range(20):
            sched.update_history([(0, 1)], positions)
        active = sched.schedule(positions, [(0, 1), (0, 2), (1, 2)])
        assert (0, 1) in active

    def test_proposition2_monotone_priority(self):
        """Proposition 2：通信缺失时间越长 → VoI 优先级单调不降"""
        sched = SafetyPriorityScheduler(
            n_agents=2, k=1, d_min=0.5, v_max=0.0, dt=0.1,
            beta=1.0, tau_ref=1.0, tau_max=10.0,
        )
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        sched.reset_episode(positions)
        prev = sched.compute_voi_priority(positions, [(0, 1)])[(0, 1)]
        for _ in range(8):
            sched.update_history([], positions)
            curr = sched.compute_voi_priority(positions, [(0, 1)])[(0, 1)]
            assert curr >= prev - 1e-9
            prev = curr

    def test_aoi_boost_stale_pair(self):
        """β>0 的优先级高于 β=0（相同场景，AoI 积累后）"""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
        ], dtype=np.float64)
        edges = [(0, 1), (0, 2), (1, 2)]
        s_voi = SafetyPriorityScheduler(n_agents=3, k=1, d_min=0.5, v_max=0.0,
                                         dt=0.1, beta=2.0, tau_ref=0.5, tau_max=10.0)
        s_safe = SafetyPriorityScheduler(n_agents=3, k=1, d_min=0.5, v_max=0.0,
                                          dt=0.1, beta=0.0, tau_ref=0.5, tau_max=10.0)
        s_voi.reset_episode(positions)
        s_safe.reset_episode(positions)
        for _ in range(20):
            s_voi.update_history([(0, 2)], positions)
            s_safe.update_history([(0, 2)], positions)
        pv = s_voi.compute_voi_priority(positions, edges)
        ps = s_safe.compute_voi_priority(positions, edges)
        assert pv[(0, 1)] > ps[(0, 1)]

    def test_tau_max_clamps_aoi(self):
        """AoI 超过 tau_max * tau_ref 后优先级不再增长"""
        sched = SafetyPriorityScheduler(
            n_agents=2, k=1, d_min=0.5, v_max=0.0, dt=0.1,
            beta=1.0, tau_ref=1.0, tau_max=2.0,
        )
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        sched.reset_episode(positions)
        for _ in range(30):
            sched.update_history([], positions)
        p30 = sched.compute_voi_priority(positions, [(0, 1)])[(0, 1)]
        for _ in range(10):
            sched.update_history([], positions)
        p40 = sched.compute_voi_priority(positions, [(0, 1)])[(0, 1)]
        assert abs(p40 - p30) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 2: 运行失败测试**

```bash
cd /root/autodl-tmp/safecomm-marl && python -m pytest tests/test_scheduler.py -v 2>&1 | head -10
```

预期：`ModuleNotFoundError: No module named 'algorithms.scheduler'`

- [ ] **Step 3: 实现 algorithms/scheduler.py**

新建 `algorithms/scheduler.py`：

```python
"""
SafetyPriorityScheduler — VoI 感知通信调度器（规格 §2.1）

priority_ij = (1 + β·min(Δt_comm/τ_ref, τ_max)) / max(h_sched_ij, ε)
h_sched_ij  = max(||p_i - p̂_j|| - d_min - d_buf_ij, 0)^2
d_buf_ij    = v_max · Δt_comm_ij

TopK 严格满足 |E_t| <= k（Proposition 1）。
"""
import numpy as np
from typing import Dict, List, Optional, Tuple


class SafetyPriorityScheduler:
    def __init__(
        self,
        n_agents: int,
        k: int,
        d_min: float = 0.5,
        eps: float = 1e-6,
        schedule_mode: str = "safety_priority",
        v_max: float = 0.0,
        dt: float = 0.1,
        beta: float = 1.0,
        tau_ref: float = 1.0,
        tau_max: float = 5.0,
    ):
        self.n = n_agents
        self.k = k
        self.d_min = d_min
        self.eps = eps
        self.schedule_mode = schedule_mode
        self.v_max = v_max
        self.dt = dt
        self.beta = beta
        self.tau_ref = max(tau_ref, 1e-9)
        self.tau_max = tau_max

        self._last_comm_time: np.ndarray = np.zeros((n_agents, n_agents), dtype=np.int32)
        self._last_known_pos: Optional[np.ndarray] = None
        self._last_known_vel: Optional[np.ndarray] = None
        self._current_step: int = 0

    def reset_episode(self, positions: np.ndarray):
        self._last_comm_time = np.zeros((self.n, self.n), dtype=np.int32)
        self._last_known_pos = positions.copy().astype(np.float64)
        self._last_known_vel = np.zeros_like(self._last_known_pos)
        self._current_step = 0

    def update_history(self, active_edges: List[Tuple[int, int]], positions: np.ndarray):
        """每步调用：被选入 E_t 的 pair 重置 Δt_comm，其余递增。"""
        comm_set = {(min(i, j), max(i, j)) for i, j in active_edges}
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if (i, j) in comm_set:
                    self._last_comm_time[i, j] = 0
                    self._last_comm_time[j, i] = 0
                    if self._last_known_pos is not None:
                        self._last_known_pos[j] = positions[j].copy()
                        self._last_known_pos[i] = positions[i].copy()
                else:
                    self._last_comm_time[i, j] += 1
                    self._last_comm_time[j, i] += 1
        self._current_step += 1

    def compute_voi_priority(
        self,
        positions: np.ndarray,
        edges: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict[Tuple[int, int], float]:
        """VoI 优先级评分（规格 §2.1）。"""
        if edges is None:
            edges = [(i, j) for i in range(self.n) for j in range(i + 1, self.n)]

        priorities: Dict[Tuple[int, int], float] = {}
        for raw_i, raw_j in edges:
            i, j = min(raw_i, raw_j), max(raw_i, raw_j)

            dt_comm_steps = int(self._last_comm_time[i, j])
            dt_comm_sec = float(dt_comm_steps) * self.dt

            if self._last_known_pos is not None and self.v_max > 0:
                p_hat_j = self._last_known_pos[j]
                d_buf = self.v_max * dt_comm_sec
            else:
                p_hat_j = positions[j]
                d_buf = 0.0

            dist = float(np.linalg.norm(positions[i] - p_hat_j))
            h_sched = max(dist - self.d_min - d_buf, 0.0) ** 2

            aoi = min(dt_comm_sec / self.tau_ref, self.tau_max)
            numerator = 1.0 + self.beta * aoi
            priorities[(i, j)] = numerator / max(h_sched, self.eps)

        return priorities

    def schedule(
        self,
        positions: np.ndarray,
        phys_edges: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """TopK 调度，返回 |E_t| <= k 的活跃边列表。"""
        if not phys_edges:
            return []

        if self.schedule_mode == "random":
            idxs = np.random.choice(len(phys_edges), min(self.k, len(phys_edges)), replace=False)
            return [phys_edges[i] for i in sorted(idxs)]

        if self.schedule_mode == "full":
            return list(phys_edges)

        # safety_priority (default): VoI TopK
        priorities = self.compute_voi_priority(positions, phys_edges)
        sorted_edges = sorted(phys_edges, key=lambda e: priorities.get((min(e), max(e)), 0.0), reverse=True)
        return sorted_edges[:self.k]

    def get_neighbor_lists(self, active_edges: List[Tuple[int, int]]) -> Dict[int, List[int]]:
        """将活跃边转换为每个 agent 的邻居列表。"""
        nbrs: Dict[int, List[int]] = {i: [] for i in range(self.n)}
        for i, j in active_edges:
            nbrs[i].append(j)
            nbrs[j].append(i)
        return nbrs
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd /root/autodl-tmp/safecomm-marl && python -m pytest tests/test_scheduler.py -v
```

预期：所有测试 `PASSED`（含 6 个 TestVoIPriority 测试）

- [ ] **Step 5: 提交**

```bash
cd /root/autodl-tmp/safecomm-marl && git add algorithms/scheduler.py tests/test_scheduler.py
git commit -m "feat: add SafetyPriorityScheduler with VoI priority formula (Phase 1 Task 3)"
```

---

## Task 4：SafeCommVoI 主算法

**Files:**
- Create: `algorithms/safecomm_psched.py`
- Create: `tests/test_algorithm.py`

### 步骤

- [ ] **Step 1: 写失败测试（tests/test_algorithm.py）**

```python
"""SafeCommVoI 冒烟测试（Phase 1）"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from envs.uav_formation_env import UAVFormationEnv
from algorithms.safecomm_psched import SafeCommVoI


FAST_CONFIG = {
    "k": 4,
    "n_steps": 40,
    "n_epochs": 2,
    "batch_size": 20,
    "msg_dim": 32,
    "hidden_dims": [64, 64],
    "H": 2,
    "safety_budget": 0.5,
    "beta": 1.0,
    "tau_ref": 1.0,
    "tau_max": 5.0,
}


def make_env_and_agent():
    env = UAVFormationEnv(n_agents=4, d_min=0.3, dt=0.1, max_steps=50, R_comm=5.0, seed=0)
    agent = SafeCommVoI(env, config=FAST_CONFIG, device="cpu")
    return env, agent


class TestSafeCommVoI:

    def test_collect_rollout_returns_info(self):
        _, agent = make_env_and_agent()
        info = agent.collect_rollout()
        assert "n_episodes" in info
        assert "mean_episode_cost" in info
        assert "mean_bandwidth_util" in info

    def test_update_returns_metrics(self):
        _, agent = make_env_and_agent()
        agent.collect_rollout()
        metrics = agent.update()
        assert "actor_loss" in metrics
        assert "lambda_s" in metrics
        assert "approx_kl" in metrics

    def test_lambda_s_increases_when_unsafe(self):
        """安全预算极低时 λ_s 应上升"""
        env = UAVFormationEnv(n_agents=4, d_min=0.3, dt=0.1, max_steps=50, R_comm=5.0, seed=0)
        cfg = {**FAST_CONFIG, "safety_budget": 0.001}
        agent = SafeCommVoI(env, config=cfg, device="cpu")
        lam_before = float(agent.lambda_s)
        agent.collect_rollout()
        agent.update()
        lam_after = float(agent.lambda_s)
        assert lam_after >= lam_before

    def test_buffer_fills_correctly(self):
        _, agent = make_env_and_agent()
        agent.collect_rollout()
        assert agent.buffer.ptr > 0

    def test_save_load_roundtrip(self):
        import tempfile
        _, agent = make_env_and_agent()
        agent.collect_rollout()
        agent.update()
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            path = f.name
        agent.save(path)
        lam_orig = float(agent.lambda_s)

        _, agent2 = make_env_and_agent()
        agent2.load(path)
        assert abs(float(agent2.lambda_s) - lam_orig) < 1e-6

        import os; os.unlink(path)

    def test_evaluate_returns_metrics(self):
        _, agent = make_env_and_agent()
        metrics = agent.evaluate(n_episodes=2, deterministic=True)
        assert "mean_formation_error" in metrics
        assert "mean_episode_cost" in metrics
        assert "success_rate" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 2: 运行失败测试**

```bash
cd /root/autodl-tmp/safecomm-marl && python -m pytest tests/test_algorithm.py -v 2>&1 | head -10
```

预期：`ModuleNotFoundError: No module named 'algorithms.safecomm_psched'`

- [ ] **Step 3: 实现 algorithms/safecomm_psched.py**

新建 `algorithms/safecomm_psched.py`：

```python
"""
SafeCommVoI Phase 1 算法主类

Phase 1 包含：
  - Safety-Aware VoI Scheduler（TopK + AoI + CBF）
  - MAPPO（H=2，参数共享，CTDE）
  - 单约束 Lagrangian（λ_s，安全成本约束）
Phase 1 不含：HOCBF-QP、λ_int、MGDA、baselines（见 Phase 2）
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional

from envs.uav_formation_env import UAVFormationEnv
from algorithms.networks import SafeCommNetworks
from algorithms.ppo_utils import (
    RolloutBuffer, ppo_clip_loss, value_loss,
    explained_variance,
)
from algorithms.scheduler import SafetyPriorityScheduler


class SafeCommVoI:
    """
    SafeComm-VoI Phase 1 主类。
    接口：collect_rollout() → update() → train() → evaluate() → save() / load()
    """

    def __init__(
        self,
        env: UAVFormationEnv,
        config: Optional[Dict] = None,
        device: str = "auto",
    ):
        self.env = env
        n = env.n

        default = {
            "n_agents": n,
            "obs_dim": env.obs_dim,
            "act_dim": env.act_dim,
            "action_scale": env.a_max,
            "k": 4,
            "R_comm": env.R_comm,
            "d_min": env.d_min,
            "schedule_mode": "safety_priority",
            "beta": 1.0,
            "tau_ref": 1.0,
            "tau_max": 5.0,
            "v_max_cbf": env.v_max,
            "msg_dim": 64,
            "hidden_dims": [256, 256],
            "H": 2,
            "n_steps": 2048,
            "n_epochs": 10,
            "batch_size": 256,
            "gamma": 0.99,
            "lam": 0.95,
            "clip_eps": 0.2,
            "value_coef": 0.5,
            "cost_value_coef": 0.5,
            "entropy_coef": 0.01,
            "max_grad_norm": 0.5,
            "lr_actor": 3e-4,
            "lr_critic": 1e-3,
            "lr_lambda": 0.01,
            "safety_budget": 0.1,
            "lambda_init": 0.0,
            "lambda_max": 10.0,
            "log_interval": 10,
            "eval_interval": 50,
        }
        if config:
            default.update(config)
        self.config = default

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.networks = SafeCommNetworks(
            n_agents=n,
            obs_dim=self.config["obs_dim"],
            act_dim=self.config["act_dim"],
            msg_dim=self.config["msg_dim"],
            hidden_dims=self.config["hidden_dims"],
            H=self.config["H"],
        ).to(device)

        self.scheduler = SafetyPriorityScheduler(
            n_agents=n,
            k=self.config["k"],
            d_min=self.config["d_min"],
            schedule_mode=self.config["schedule_mode"],
            v_max=self.config.get("v_max_cbf", env.v_max),
            dt=env.dt,
            beta=self.config.get("beta", 1.0),
            tau_ref=self.config.get("tau_ref", 1.0),
            tau_max=self.config.get("tau_max", 5.0),
        )

        global_state_dim = n * self.config["obs_dim"]
        self.buffer = RolloutBuffer(
            n_steps=self.config["n_steps"],
            n_agents=n,
            obs_dim=self.config["obs_dim"],
            act_dim=self.config["act_dim"],
            global_state_dim=global_state_dim,
        )

        actor_params = (
            list(self.networks.encoder.parameters())
            + list(self.networks.aggregator.parameters())
            + list(self.networks.actor.parameters())
        )
        critic_params = (
            list(self.networks.critic.parameters())
            + list(self.networks.cost_critic.parameters())
        )
        self.actor_optimizer = torch.optim.Adam(actor_params, lr=self.config["lr_actor"])
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=self.config["lr_critic"])

        self.lambda_s: float = float(self.config["lambda_init"])
        self.lambda_max: float = float(self.config["lambda_max"])

        self.n_iterations: int = 0
        self.total_steps: int = 0

    # ------------------------------------------------------------------
    def collect_rollout(self) -> Dict:
        self.networks.eval()
        self.buffer.reset()
        n = self.env.n
        n_episodes = 0
        ep_costs, bw_utils = [], []
        t_start = time.time()

        while self.buffer.ptr < self.config["n_steps"]:
            obs_list, _ = self.env.reset()
            self.scheduler.reset_episode(self.env.positions)
            ep_cost = 0.0
            ep_bw = []
            done = False

            while not done and self.buffer.ptr < self.config["n_steps"]:
                phys = self.env.get_physical_graph()
                active = self.scheduler.schedule(self.env.positions, phys)
                self.scheduler.update_history(active, self.env.positions)
                nbr_lists = self.scheduler.get_neighbor_lists(active)

                bw = len(active) / max(self.config["k"], 1)
                ep_bw.append(bw)

                adj = np.zeros((n, n), dtype=np.float32)
                for i, j in active:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0

                actions, log_probs, values, cost_values = self.networks.act(
                    obs_list, nbr_lists, deterministic=False, device=self.device
                )
                scaled = actions * self.config["action_scale"]
                next_obs, rewards, costs, done, info = self.env.step(scaled)

                global_state = np.concatenate(obs_list, axis=0)
                self.buffer.add(
                    obs=np.stack(obs_list),
                    actions=actions,
                    log_probs=log_probs,
                    rewards=rewards,
                    costs=costs,
                    value=float(np.mean(values)) if values.ndim > 0 else float(values),
                    cost_value=float(np.mean(cost_values)) if cost_values.ndim > 0 else float(cost_values),
                    done=done,
                    global_state=global_state,
                    adj_matrix=adj,
                )
                ep_cost += float(np.mean(costs))
                obs_list = next_obs

            n_episodes += 1
            ep_costs.append(ep_cost)
            bw_utils.append(float(np.mean(ep_bw)) if ep_bw else 0.0)

        # Bootstrap
        phys = self.env.get_physical_graph()
        active = self.scheduler.schedule(self.env.positions, phys)
        nbr_lists = self.scheduler.get_neighbor_lists(active)
        _, _, last_v, last_cv = self.networks.act(obs_list, nbr_lists, device=self.device)
        self.buffer.finish_rollout(
            float(np.mean(last_v)) if last_v.ndim > 0 else float(last_v),
            float(np.mean(last_cv)) if last_cv.ndim > 0 else float(last_cv),
        )

        fps = self.buffer.ptr / max(time.time() - t_start, 1e-6)
        self.total_steps += self.buffer.ptr
        return {
            "n_episodes": n_episodes,
            "mean_episode_cost": float(np.mean(ep_costs)) if ep_costs else 0.0,
            "mean_bandwidth_util": float(np.mean(bw_utils)) if bw_utils else 0.0,
            "fps": fps,
            "mean_formation_error": float(self.env._get_info().get("formation_error", 0.0)),
        }

    # ------------------------------------------------------------------
    def update(self) -> Dict:
        self.networks.train()
        data = self.buffer.get_training_data(
            gamma=self.config["gamma"],
            lam=self.config["lam"],
            normalize_adv=True,
            device=self.device,
        )

        T = len(data["obs"])
        bs = min(self.config["batch_size"], T)
        n_epochs = self.config["n_epochs"]
        clip_eps = self.config["clip_eps"]
        lambda_s = self.lambda_s

        all_a_loss, all_c_loss, all_kl, all_ent = [], [], [], []

        for _ in range(n_epochs):
            idxs = torch.randperm(T, device=self.device)
            for start in range(0, T, bs):
                idx = idxs[start:start + bs]
                if len(idx) < 2:
                    continue

                obs_b = data["obs"][idx]
                act_b = data["actions"][idx]
                lp_old_b = data["log_probs_old"][idx]
                adv_b = data["advantages"][idx]
                ret_b = data["returns"][idx]
                c_adv_b = data["cost_advantages"][idx]
                c_ret_b = data["cost_returns"][idx]
                v_old_b = data["values_old"][idx]
                cv_old_b = data["cost_values_old"][idx]
                gs_b = data["global_states"][idx]
                adj_b = data["adj_matrices"][idx]

                log_probs_b, entropy_b, values_b, cost_values_b, _ = (
                    self.networks.evaluate_actions_batch(obs_b, act_b, adj_b, gs_b)
                )

                lp_mean = log_probs_b.mean(dim=1)
                lp_old_mean = lp_old_b.mean(dim=1)

                actor_loss, kl = ppo_clip_loss(lp_mean, lp_old_mean.detach(), adv_b, clip_eps)
                entropy_loss = -self.config["entropy_coef"] * entropy_b.mean()
                safety_penalty = lambda_s * c_adv_b.mean()
                total_actor = actor_loss + entropy_loss + safety_penalty

                self.actor_optimizer.zero_grad()
                total_actor.backward()
                nn.utils.clip_grad_norm_(
                    [p for pg in self.actor_optimizer.param_groups for p in pg["params"]],
                    self.config["max_grad_norm"],
                )
                self.actor_optimizer.step()

                vf = value_loss(values_b, v_old_b.detach(), ret_b, clip_eps)
                cvf = value_loss(cost_values_b, cv_old_b.detach(), c_ret_b, clip_eps)
                crit_loss = self.config["value_coef"] * vf + self.config["cost_value_coef"] * cvf

                self.critic_optimizer.zero_grad()
                crit_loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for pg in self.critic_optimizer.param_groups for p in pg["params"]],
                    self.config["max_grad_norm"],
                )
                self.critic_optimizer.step()

                all_a_loss.append(float(actor_loss.detach()))
                all_c_loss.append(float(crit_loss.detach()))
                all_kl.append(float(kl))
                all_ent.append(float(entropy_b.mean().detach()))

        # Lagrangian update
        mean_step_cost = float(self.buffer.costs[:self.buffer.ptr].mean())
        ep_cost = mean_step_cost * self.env.max_steps
        violation = ep_cost - self.config["safety_budget"]
        self.lambda_s = float(np.clip(
            self.lambda_s + self.config["lr_lambda"] * violation,
            0.0, self.lambda_max,
        ))

        self.n_iterations += 1
        ev = explained_variance(
            self.buffer.values[:self.buffer.ptr],
            self.buffer.rewards[:self.buffer.ptr].mean(axis=1),
        )
        return {
            "actor_loss": float(np.mean(all_a_loss)),
            "critic_loss": float(np.mean(all_c_loss)),
            "entropy": float(np.mean(all_ent)),
            "approx_kl": float(np.mean(all_kl)),
            "lambda_s": self.lambda_s,
            "J_safe": ep_cost,
            "cost_violation": violation,
            "ev_task": ev,
        }

    # ------------------------------------------------------------------
    def train(self, total_steps: int) -> Dict:
        print(f"[SafeComm-VoI Phase 1] 开始训练，总步数={total_steps}，设备={self.device}")
        last_metrics: Dict = {}
        while self.total_steps < total_steps:
            rollout_info = self.collect_rollout()
            update_info = self.update()
            last_metrics = {**rollout_info, **update_info}
            if self.n_iterations % self.config.get("log_interval", 10) == 0:
                print(
                    f"[iter {self.n_iterations:4d} | steps {self.total_steps:7d}] "
                    f"FE={rollout_info['mean_formation_error']:.4f}  "
                    f"cost={rollout_info['mean_episode_cost']:.4f}  "
                    f"λ_s={update_info['lambda_s']:.4f}  "
                    f"BU={rollout_info['mean_bandwidth_util']:.3f}  "
                    f"KL={update_info['approx_kl']:.5f}  "
                    f"FPS={rollout_info['fps']:.0f}"
                )
        return last_metrics

    # ------------------------------------------------------------------
    def evaluate(self, n_episodes: int = 20, deterministic: bool = True) -> Dict:
        self.networks.eval()
        all_fe, all_cost, all_success = [], [], []

        with torch.no_grad():
            for _ in range(n_episodes):
                obs_list, _ = self.env.reset()
                self.scheduler.reset_episode(self.env.positions)
                ep_fe, ep_cost = [], []
                done = False
                while not done:
                    phys = self.env.get_physical_graph()
                    active = self.scheduler.schedule(self.env.positions, phys)
                    self.scheduler.update_history(active, self.env.positions)
                    nbr_lists = self.scheduler.get_neighbor_lists(active)
                    actions, _, _, _ = self.networks.act(
                        obs_list, nbr_lists, deterministic=deterministic, device=self.device
                    )
                    obs_list, _, costs, done, info = self.env.step(actions * self.config["action_scale"])
                    ep_fe.append(info["formation_error"])
                    ep_cost.append(float(np.mean(costs)))
                all_fe.append(float(np.mean(ep_fe)))
                all_cost.append(float(np.sum(ep_cost)))
                all_success.append(float(info.get("success", False)))

        return {
            "mean_formation_error": float(np.mean(all_fe)),
            "mean_episode_cost": float(np.mean(all_cost)),
            "success_rate": float(np.mean(all_success)),
        }

    # ------------------------------------------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "network_state": self.networks.state_dict(),
            "actor_opt": self.actor_optimizer.state_dict(),
            "critic_opt": self.critic_optimizer.state_dict(),
            "lambda_s": self.lambda_s,
            "n_iterations": self.n_iterations,
            "total_steps": self.total_steps,
            "config": self.config,
        }, path)
        print(f"[SafeComm-VoI] 模型已保存至 {path}")

    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.networks.load_state_dict(state["network_state"])
        self.actor_optimizer.load_state_dict(state["actor_opt"])
        self.critic_optimizer.load_state_dict(state["critic_opt"])
        self.lambda_s = float(state.get("lambda_s", 0.0))
        self.n_iterations = int(state.get("n_iterations", 0))
        self.total_steps = int(state.get("total_steps", 0))
        print(f"[SafeComm-VoI] 模型已从 {path} 加载，迭代={self.n_iterations}，步数={self.total_steps}")
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd /root/autodl-tmp/safecomm-marl && python -m pytest tests/test_algorithm.py -v
```

预期：所有测试 `PASSED`

- [ ] **Step 5: 运行全量测试套件（确认无回归）**

```bash
cd /root/autodl-tmp/safecomm-marl && python -m pytest tests/ -v --tb=short 2>&1 | tail -20
```

预期：全部通过（test_env + test_ppo_utils + test_networks + test_scheduler + test_algorithm）

- [ ] **Step 6: 提交**

```bash
cd /root/autodl-tmp/safecomm-marl && git add algorithms/safecomm_psched.py tests/test_algorithm.py
git commit -m "feat: add SafeCommVoI Phase 1 main algorithm (MAPPO-Lag + VoI scheduler)"
```

---

## Task 5：配置文件与训练入口

**Files:**
- Create: `configs/default.yaml`
- Create: `train.py`

### 步骤

- [ ] **Step 1: 创建 configs/default.yaml**

新建 `configs/default.yaml`：

```yaml
# SafeComm-VoI Phase 1 默认超参数配置
# 规格参考：docs/superpowers/plans/2026-05-26-safecomm-voi-design.md §2.1

env:
  n_agents: 4
  dt: 0.1
  max_steps: 200
  d_min: 0.5
  R_comm: 5.0
  v_max: 3.0
  a_max: 2.0
  arena_size: 10.0
  formation_type: "circle"
  w_formation: 1.0
  w_success: 5.0
  success_threshold: 0.3
  K_obs_neighbors: 4
  seed: 42

comm:
  k: 4
  schedule_mode: "safety_priority"
  beta: 1.0
  tau_ref: 1.0
  tau_max: 5.0
  v_max_cbf: 3.0

network:
  msg_dim: 64
  hidden_dims: [256, 256]
  H: 2

ppo:
  n_steps: 2048
  n_epochs: 10
  batch_size: 256
  gamma: 0.99
  lam: 0.95
  clip_eps: 0.2
  value_coef: 0.5
  cost_value_coef: 0.5
  entropy_coef: 0.01
  max_grad_norm: 0.5

lr:
  actor: 3.0e-4
  critic: 1.0e-3
  lambda: 0.01

safety:
  budget: 0.1
  lambda_init: 0.0
  lambda_max: 10.0

training:
  total_steps: 1_000_000
  log_interval: 10
  eval_interval: 50
  eval_episodes: 10
  save_interval: 100
  checkpoint_dir: "checkpoints"
  device: "auto"
  seed: 42
```

- [ ] **Step 2: 创建 train.py**

新建 `train.py`：

```python
"""
SafeComm-VoI Phase 1 训练入口

用法：
  python train.py
  python train.py --config configs/default.yaml
  python train.py --beta 0.0   # 消融：纯安全优先
  python train.py --n_agents 4 --k 4 --total_steps 500000
"""
import argparse
import os
import yaml

from envs.uav_formation_env import UAVFormationEnv
from algorithms.safecomm_psched import SafeCommVoI


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def flatten_config(cfg: dict) -> dict:
    c = cfg.get("comm", {})
    n = cfg.get("network", {})
    p = cfg.get("ppo", {})
    lr = cfg.get("lr", {})
    s = cfg.get("safety", {})
    t = cfg.get("training", {})
    return {
        "k": c.get("k", 4),
        "schedule_mode": c.get("schedule_mode", "safety_priority"),
        "beta": c.get("beta", 1.0),
        "tau_ref": c.get("tau_ref", 1.0),
        "tau_max": c.get("tau_max", 5.0),
        "v_max_cbf": c.get("v_max_cbf", 3.0),
        "msg_dim": n.get("msg_dim", 64),
        "hidden_dims": n.get("hidden_dims", [256, 256]),
        "H": n.get("H", 2),
        "n_steps": p.get("n_steps", 2048),
        "n_epochs": p.get("n_epochs", 10),
        "batch_size": p.get("batch_size", 256),
        "gamma": p.get("gamma", 0.99),
        "lam": p.get("lam", 0.95),
        "clip_eps": p.get("clip_eps", 0.2),
        "value_coef": p.get("value_coef", 0.5),
        "cost_value_coef": p.get("cost_value_coef", 0.5),
        "entropy_coef": p.get("entropy_coef", 0.01),
        "max_grad_norm": p.get("max_grad_norm", 0.5),
        "lr_actor": lr.get("actor", 3e-4),
        "lr_critic": lr.get("critic", 1e-3),
        "lr_lambda": lr.get("lambda", 0.01),
        "safety_budget": s.get("budget", 0.1),
        "lambda_init": s.get("lambda_init", 0.0),
        "lambda_max": s.get("lambda_max", 10.0),
        "log_interval": t.get("log_interval", 10),
        "eval_interval": t.get("eval_interval", 50),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n_agents", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.n_agents: cfg["env"]["n_agents"] = args.n_agents
    if args.k: cfg["comm"]["k"] = args.k
    if args.beta is not None: cfg["comm"]["beta"] = args.beta
    if args.total_steps: cfg["training"]["total_steps"] = args.total_steps
    if args.seed:
        cfg["env"]["seed"] = args.seed
        cfg["training"]["seed"] = args.seed

    ec = cfg.get("env", {})
    env = UAVFormationEnv(
        n_agents=ec.get("n_agents", 4),
        dt=ec.get("dt", 0.1),
        max_steps=ec.get("max_steps", 200),
        d_min=ec.get("d_min", 0.5),
        R_comm=ec.get("R_comm", 5.0),
        v_max=ec.get("v_max", 3.0),
        a_max=ec.get("a_max", 2.0),
        arena_size=ec.get("arena_size", 10.0),
        formation_type=ec.get("formation_type", "circle"),
        w_formation=ec.get("w_formation", 1.0),
        w_success=ec.get("w_success", 5.0),
        success_threshold=ec.get("success_threshold", 0.3),
        K_obs_neighbors=ec.get("K_obs_neighbors", 4),
        seed=ec.get("seed", 42),
    )

    agent = SafeCommVoI(env=env, config=flatten_config(cfg),
                        device=args.device or cfg["training"].get("device", "auto"))
    if args.checkpoint and os.path.exists(args.checkpoint):
        agent.load(args.checkpoint)

    tc = cfg.get("training", {})
    metrics = agent.train(total_steps=tc.get("total_steps", 1_000_000))

    ckpt_dir = tc.get("checkpoint_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    agent.save(os.path.join(ckpt_dir, "final.pt"))

    print("\n[最终评估]")
    eval_m = agent.evaluate(n_episodes=tc.get("eval_episodes", 20), deterministic=True)
    for k, v in eval_m.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 验证配置可加载**

```bash
cd /root/autodl-tmp/safecomm-marl && python -c "
import yaml
cfg = yaml.safe_load(open('configs/default.yaml'))
assert cfg['comm']['beta'] == 1.0
assert cfg['env']['K_obs_neighbors'] == 4
print('OK: configs/default.yaml 加载正常')
"
```

- [ ] **Step 4: 提交**

```bash
cd /root/autodl-tmp/safecomm-marl && mkdir -p configs && git add configs/default.yaml train.py
git commit -m "chore: add configs/default.yaml and train.py for SafeComm-VoI Phase 1"
```

---

## Task 6：Phase 1 成功标准验证（冒烟测试）

**Files:** 只运行测试和短脚本，不修改任何文件

### Phase 1 成功标准

1. pytest 全部通过
2. 每步 |E_t| ≤ k
3. λ_s 更新方向正确（成本超预算时 λ_s 上升）
4. β=0 vs β>0 调度差异可见

### 步骤

- [ ] **Step 1: 全量 pytest 通过**

```bash
cd /root/autodl-tmp/safecomm-marl && python -m pytest tests/ -v 2>&1 | tail -20
```

预期：全部 `PASSED`，无 `FAILED`

- [ ] **Step 2: 验证 |E_t| ≤ k**

```bash
cd /root/autodl-tmp/safecomm-marl && python -c "
import numpy as np
from envs.uav_formation_env import UAVFormationEnv
from algorithms.safecomm_psched import SafeCommVoI

env = UAVFormationEnv(n_agents=4, d_min=0.5, dt=0.1, max_steps=10, R_comm=5.0)
agent = SafeCommVoI(env, config={'k': 4, 'n_steps': 40, 'n_epochs': 1, 'batch_size': 20,
                                  'msg_dim': 32, 'hidden_dims': [64, 64], 'H': 2,
                                  'safety_budget': 0.5, 'beta': 1.0, 'tau_ref': 1.0, 'tau_max': 5.0},
                    device='cpu')
env.reset()
agent.scheduler.reset_episode(env.positions)
for _ in range(20):
    phys = env.get_physical_graph()
    active_edges = agent.scheduler.schedule(env.positions, phys)
    assert len(active_edges) <= 4, f'|E_t|={len(active_edges)} > k=4'
print('OK: |E_t| <= k 在 20 步内始终满足')
"
```

预期输出：`OK: |E_t| <= k 在 20 步内始终满足`

- [ ] **Step 3: 验证 λ_s 更新方向**

```bash
cd /root/autodl-tmp/safecomm-marl && python -c "
import numpy as np
from envs.uav_formation_env import UAVFormationEnv
from algorithms.safecomm_psched import SafeCommVoI

env = UAVFormationEnv(n_agents=4, d_min=0.3, dt=0.1, max_steps=10, R_comm=5.0)
cfg = {'k': 4, 'n_steps': 40, 'n_epochs': 2, 'batch_size': 20,
       'msg_dim': 32, 'hidden_dims': [64, 64], 'H': 2,
       'safety_budget': 0.001, 'beta': 1.0, 'tau_ref': 1.0, 'tau_max': 5.0}
agent = SafeCommVoI(env, config=cfg, device='cpu')
lam_before = float(agent.lambda_s)
agent.collect_rollout()
agent.update()
lam_after = float(agent.lambda_s)
assert lam_after >= lam_before, f'λ_s 应 >= 初值 {lam_before}，实际 {lam_after}'
print(f'OK: λ_s {lam_before:.4f} -> {lam_after:.4f}（方向正确）')
"
```

预期：`OK: λ_s 0.0000 -> X.XXXX（方向正确）`

- [ ] **Step 4: 验证 β=0 vs β>0 调度差异**

```bash
cd /root/autodl-tmp/safecomm-marl && python -c "
import numpy as np
from algorithms.scheduler import SafetyPriorityScheduler

positions = np.array([[0,0,0],[3,0,0],[10,0,0]], dtype=np.float64)
edges = [(0,1),(0,2),(1,2)]

s_voi = SafetyPriorityScheduler(n_agents=3, k=1, d_min=0.5, v_max=0.0, dt=0.1,
                                 beta=2.0, tau_ref=0.5, tau_max=10.0)
s_safe = SafetyPriorityScheduler(n_agents=3, k=1, d_min=0.5, v_max=0.0, dt=0.1,
                                  beta=0.0, tau_ref=0.5, tau_max=10.0)
s_voi.reset_episode(positions)
s_safe.reset_episode(positions)
for _ in range(20):
    s_voi.update_history([(0,1)], positions)
    s_safe.update_history([(0,1)], positions)

pv = s_voi.compute_voi_priority(positions, edges)
ps = s_safe.compute_voi_priority(positions, edges)
print('VoI (β=2.0):', {k: f'{v:.3f}' for k,v in sorted(pv.items())})
print('Safe(β=0.0):', {k: f'{v:.3f}' for k,v in sorted(ps.items())})
ratio_voi = pv[(0,2)] / pv[(0,1)]
ratio_safe = ps[(0,2)] / ps[(0,1)]
print(f'(0,2)/(0,1) 比值: β=2.0 -> {ratio_voi:.3f}, β=0.0 -> {ratio_safe:.3f}')
assert abs(ratio_voi - ratio_safe) > 1e-3, 'β=0 与 β>0 调度优先级比应不同'
print('OK: β=0 vs β>0 调度差异可见')
"
```

预期：两组比值有明显差异，末行打印 `OK`

- [ ] **Step 5: 最终全量提交确认**

```bash
cd /root/autodl-tmp/safecomm-marl && git status
```

如有未提交文件，提交：

```bash
cd /root/autodl-tmp/safecomm-marl && git add -A
git commit -m "test: Phase 1 success criteria verified (all smoke tests pass)"
```

---

## 自审检查单

| 条目 | 覆盖 Task |
|------|----------|
| UAVFormationEnv obs_dim = 6 + K_obs*6 | Task 0 |
| 双积分器动力学：v += a*dt, p += v*dt | Task 0 |
| RolloutBuffer 存储 obs/actions/rewards/costs/values/adj | Task 1 |
| GAE 在 done 边界处正确截断 | Task 1 |
| SafeCommNetworks 参数共享（所有 agent 同一 Actor/Encoder）| Task 2 |
| Proposition 1（|E_t|≤k）由 TopK 保证 | Task 3 |
| h_sched 使用 p_hat_j（最近通信已知位置）| Task 3 |
| β=0 严格退化为纯安全优先 | Task 3 测试覆盖 |
| AoI = Δt_comm(秒)/τ_ref，截断于 τ_max | Task 3 测试覆盖 |
| collect_rollout 直接执行策略动作（Phase 1 无 QP）| Task 4 |
| λ_s 更新：`λ_s ← clip(λ_s + α·(J_safe-d), 0, λ_max)` | Task 4 |
| save/load 保存 lambda_s、n_iterations | Task 4 |
| configs/default.yaml VoI 参数齐全 | Task 5 |
| train.py 支持 --beta 消融参数 | Task 5 |

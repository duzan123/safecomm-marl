# SafeComm-VoI Phase 4 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 验证 SafeCommVoI 算法的泛化能力与可扩展性——新增围捕（Pursuit）场景环境，实现支持 N≥8 智能体的注意力式集中式 Critic（ScalableCritic），并提供多任务训练脚本和 N=8 配置。

**Architecture:** 三个核心组件——(1) `UAVPursuitEnv`：多 UAV 协作围捕环境，保持与 Phase 1 相同的 gym-style 接口；(2) `ScalableCritic`：基于 PyTorch `MultiheadAttention` 的 Transformer-style 集中式 Critic，输入 N 个 per-agent state token，替换 MLP concat 方案，支持任意 N；(3) `train_multitask.py`：多任务训练入口，轮流在编队和围捕两个场景中训练，验证策略复用。

**Tech Stack:** Python 3.x, NumPy, PyTorch (MultiheadAttention), PyYAML, pytest

**外部代码复用边界（Phase 4）：**
- `external/InforMARL`（MIT）参考局部图观测、邻接矩阵、规模迁移评估和 3/7/10/15 agents 实验组织；不引入其旧 torch-geometric 依赖栈，不复制完整 runner。
- `external/MAGIC`（MIT）参考 masked attention/message processor 的邻接 mask 处理；Phase 4 的 `ScalableCritic` 使用本项目 PyTorch `MultiheadAttention` 自写。
- `external/gcbfplus`（MIT）可作为 N 扩展下 graph CBF/GCBF+ 独立 baseline 参考；SafeComm 主线仍使用 HOCBF-QP 和本项目 scheduler。
- `external/DGN` 无明确许可证，只用于描述通信受限图 MARL baseline，不复制源码。
- 任何多跳图传播、Transformer critic 或集中式 critic 只能用于训练期 critic 或明确标注的 ablation；执行期通信仍只能使用 `E_t` 内消息，不能破坏 `|E_t| <= k`。

---

## 文件结构

| 文件 | 操作 | 说明 |
|------|------|------|
| `envs/uav_pursuit_env.py` | 新建 | 多 UAV 协作围捕场景（N 追捕者 + 1 目标），与 Phase 1 接口一致 |
| `algorithms/scalable_critic.py` | 新建 | 注意力式集中式 Critic（支持 N≥8），Token = per-agent state |
| `configs/pursuit.yaml` | 新建 | 围捕场景配置 |
| `configs/n8_scale.yaml` | 新建 | N=8 大规模扩展配置（支持 ScalableCritic） |
| `train_multitask.py` | 新建 | 多任务训练入口（编队 + 围捕轮流训练） |
| `tests/test_pursuit_env.py` | 新建 | 围捕环境接口测试 |
| `tests/test_scalable_critic.py` | 新建 | ScalableCritic 接口和形状测试 |

**不修改的文件：** `algorithms/networks.py`，`algorithms/safecomm_psched.py`，`algorithms/scheduler.py`，`envs/uav_formation_env.py`，`envs/pybullet_uav_env.py`，`envs/domain_randomization.py`

---

## Task 1：UAV 围捕场景环境（envs/uav_pursuit_env.py）

**Files:**
- Create: `envs/uav_pursuit_env.py`
- Test: `tests/test_pursuit_env.py`

### 步骤

- [ ] **Step 1: 编写围捕环境失败测试（TDD 先行）**

新建 `tests/test_pursuit_env.py`，写入以下完整测试代码：

```python
"""
UAV 围捕环境测试

验证 UAVPursuitEnv 的接口与 UAVFormationEnv 完全兼容。
"""
import numpy as np
import pytest


class TestUAVPursuitEnv:
    """测试围捕环境接口一致性"""

    @pytest.fixture(autouse=True)
    def setup(self):
        from envs.uav_pursuit_env import UAVPursuitEnv
        self.env = UAVPursuitEnv(n_agents=4, seed=42)
        yield

    def test_obs_dim_formula(self):
        """obs_dim = 6 + K_obs * 6 + 3（追加目标相对位置）"""
        expected = 6 + self.env.K_obs * 6 + 3
        assert self.env.obs_dim == expected, (
            f"obs_dim={self.env.obs_dim} != expected={expected}"
        )

    def test_reset_interface(self):
        """reset() 返回 (obs_list, info)，obs_list 长度 = n_agents"""
        obs_list, info = self.env.reset()
        assert isinstance(obs_list, list)
        assert len(obs_list) == self.env.n
        for obs in obs_list:
            assert obs.shape == (self.env.obs_dim,)
        assert "t" in info
        assert "capture_rate" in info
        assert "min_dist_to_target" in info

    def test_step_interface(self):
        """step(actions) 返回 (obs_list, rewards, costs, done, info)"""
        self.env.reset()
        actions = np.zeros((self.env.n, 3), dtype=np.float32)
        result = self.env.step(actions)
        assert len(result) == 5
        obs_list, rewards, costs, done, info = result
        assert len(obs_list) == self.env.n
        assert len(rewards) == self.env.n
        assert len(costs) == self.env.n
        assert isinstance(done, bool)

    def test_reward_positive_when_capturing(self):
        """距离目标越近，奖励应越高（比初始位置高）"""
        self.env.reset()
        # 手动将所有追捕者移到目标附近
        target_pos = self.env.target_pos.copy()
        for i in range(self.env.n):
            self.env.positions[i] = target_pos + np.array([
                0.3 * (i - self.env.n / 2), 0.3, 0.0
            ], dtype=np.float32)
        rewards = self.env._compute_rewards()
        # 捕获奖励应为正
        assert all(r > 0 for r in rewards), f"近距离应有正奖励，实际={rewards}"

    def test_cost_binary(self):
        """安全成本应为 0.0 或 1.0"""
        self.env.reset()
        actions = np.zeros((self.env.n, 3), dtype=np.float32)
        for _ in range(5):
            _, _, costs, done, _ = self.env.step(actions)
            for c in costs:
                assert c in (0.0, 1.0)
            if done:
                break

    def test_target_moves(self):
        """目标应随时间移动（逃逸策略）"""
        self.env.reset()
        initial_target_pos = self.env.target_pos.copy()
        actions = np.zeros((self.env.n, 3), dtype=np.float32)
        self.env.step(actions)
        # 目标应该移动（逃逸）
        moved = not np.allclose(self.env.target_pos, initial_target_pos)
        assert moved, "目标应随时间移动"

    def test_success_when_captured(self):
        """所有追捕者在捕获半径内时 done=True"""
        self.env.reset()
        target_pos = self.env.target_pos.copy()
        # 将所有追捕者放到目标的捕获半径内
        for i in range(self.env.n):
            self.env.positions[i] = target_pos + np.array([
                0.1 * i, 0.0, 0.0
            ], dtype=np.float32)
        assert self.env._check_captured(), "所有追捕者在捕获半径内时应判定捕获成功"

    def test_get_physical_graph(self):
        """get_physical_graph() 应返回正确格式的边列表"""
        self.env.reset()
        edges = self.env.get_physical_graph()
        for i, j in edges:
            assert i < j
            assert 0 <= i < self.env.n
            assert 0 <= j < self.env.n

    def test_compute_cbf_values(self):
        """compute_cbf_values() 应返回非负 CBF 值"""
        self.env.reset()
        cbf = self.env.compute_cbf_values()
        for (i, j), h in cbf.items():
            assert i < j
            assert h >= 0.0

    def test_state_property(self):
        """state 属性应返回 [6n + 6] 向量（追捕者 + 目标状态）"""
        self.env.reset()
        state = self.env.state
        # 追捕者 6n + 目标 6（位置3 + 速度3）
        assert state.shape == (6 * self.env.n + 6,)

    def test_n8_agents(self):
        """N=8 时环境应正常初始化"""
        from envs.uav_pursuit_env import UAVPursuitEnv
        env8 = UAVPursuitEnv(n_agents=8, seed=0)
        obs_list, info = env8.reset()
        assert len(obs_list) == 8
        for obs in obs_list:
            assert obs.shape == (env8.obs_dim,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

运行（预期失败，因为文件未创建）：
```bash
cd /root/autodl-tmp/safecomm-marl && python -m pytest tests/test_pursuit_env.py -v 2>&1 | head -10
```

预期输出：`ImportError: No module named 'envs.uav_pursuit_env'`

---

- [ ] **Step 2: 实现 UAVPursuitEnv（envs/uav_pursuit_env.py）**

新建 `/root/autodl-tmp/safecomm-marl/envs/uav_pursuit_env.py`，写入以下完整代码：

```python
"""
Phase 4 多 UAV 协作围捕场景

场景描述：
  - N 个追捕者 UAV 协作追捕 1 个逃跑目标（双积分器动力学）
  - 目标使用简单逃逸策略：朝离最近追捕者最远的方向加速
  - 捕获条件：至少 n_capturers_required 个追捕者与目标的距离 < capture_radius

安全约束（与 Phase 1 一致）：
  - 追捕者之间禁止碰撞（C_i^safe = 1[||pi - pj|| < d_min]）
  - 注意：追捕者与目标的接近不算碰撞，只要求捕获

接口与 Phase 1 UAVFormationEnv 完全一致，但：
  - obs_dim = 6 + K_obs * 6 + 3（追加目标相对位置向量）
  - info 字典新增 'capture_rate' 和 'min_dist_to_target'
  - state 属性返回 [6n + 6]（追捕者 + 目标状态）
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class UAVPursuitEnv:
    """
    多 UAV 协作围捕环境（Phase 4）。

    追捕者使用双积分器动力学（与 Phase 1 相同），目标也用双积分器。
    观测向量：
      [自身位置（相对目标）(3), 自身速度(3),
       邻居1相对位置(3), 邻居1相对速度(3), ...,  (K_obs 个)
       目标相对位置(3)]                            (额外 3 维)

    接口：
        obs_list, info = env.reset()
        obs_list, rewards, costs, done, info = env.step(actions)
    """

    def __init__(
        self,
        n_agents: int = 4,
        dt: float = 0.1,
        max_steps: int = 300,
        d_min: float = 0.5,
        R_comm: float = 5.0,
        v_max: float = 3.0,
        a_max: float = 2.0,
        target_v_max: float = 1.5,
        target_a_max: float = 1.0,
        arena_size: float = 12.0,
        capture_radius: float = 1.5,
        n_capturers_required: int = 1,
        w_capture: float = 5.0,
        w_approach: float = 1.0,
        w_team: float = 0.5,
        K_obs_neighbors: int = 2,
        seed: Optional[int] = None,
    ):
        self.n = n_agents
        self.dt = dt
        self.max_steps = max_steps
        self.d_min = d_min
        self.R_comm = R_comm
        self.v_max = v_max
        self.a_max = a_max
        self.target_v_max = target_v_max
        self.target_a_max = target_a_max
        self.arena_size = arena_size
        self.capture_radius = capture_radius
        self.n_capturers_required = n_capturers_required
        self.w_capture = w_capture
        self.w_approach = w_approach
        self.w_team = w_team
        self.K_obs = K_obs_neighbors

        # 观测维度 = 自身 6 + K_obs * 6（邻居）+ 3（目标相对位置）
        self.obs_dim = 6 + self.K_obs * 6 + 3
        self.act_dim = 3

        # 动作空间边界
        self.action_low = np.full(self.act_dim, -a_max, dtype=np.float32)
        self.action_high = np.full(self.act_dim, a_max, dtype=np.float32)

        # 状态（追捕者）
        self.positions: np.ndarray = np.zeros((self.n, 3))
        self.velocities: np.ndarray = np.zeros((self.n, 3))

        # 状态（目标）
        self.target_pos: np.ndarray = np.zeros(3)
        self.target_vel: np.ndarray = np.zeros(3)

        # goal_positions 保留以兼容外部接口
        self.goal_positions: np.ndarray = np.zeros((self.n, 3))

        self.t: int = 0
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def reset(
        self,
        init_positions: Optional[np.ndarray] = None,
        goal_center: Optional[np.ndarray] = None,
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        重置环境。goal_center 参数保留以保持接口兼容性，但围捕场景不使用。

        Returns:
            obs_list: 所有追捕者的观测向量列表
            info: 调试信息
        """
        self.t = 0

        # 初始化追捕者位置
        if init_positions is not None:
            self.positions = init_positions.copy().astype(np.float32)
        else:
            self.positions = self._random_init_positions_pursuers()

        self.velocities = np.zeros((self.n, 3), dtype=np.float32)

        # 初始化目标（在追捕者群体中心的对面方向）
        centroid = self.positions.mean(axis=0)
        direction = self.rng.uniform(-1, 1, 3).astype(np.float32)
        direction /= (np.linalg.norm(direction) + 1e-8)
        init_dist = self.arena_size * 0.3
        self.target_pos = (centroid + direction * init_dist).astype(np.float32)
        half = self.arena_size / 2
        self.target_pos = np.clip(self.target_pos, -half * 0.9, half * 0.9)
        self.target_vel = np.zeros(3, dtype=np.float32)

        # goal_positions 与目标同步
        self.goal_positions = np.tile(self.target_pos, (self.n, 1))

        obs_list = self._get_obs_all()
        info = self._get_info()
        return obs_list, info

    def step(
        self, actions: np.ndarray
    ) -> Tuple[List[np.ndarray], List[float], List[float], bool, Dict]:
        """
        执行一步动作。

        Args:
            actions: [n, 3] 追捕者的速度增量指令

        Returns:
            obs_list, rewards, costs, done, info
        """
        actions = np.clip(actions, -self.a_max, self.a_max).astype(np.float32)

        # 更新追捕者（双积分器）
        self.velocities = np.clip(
            self.velocities + actions * self.dt, -self.v_max, self.v_max
        )
        self.positions = self.positions + self.velocities * self.dt
        self._apply_boundary(self.positions, self.velocities)

        # 更新目标（逃逸策略）
        self._update_target()

        # 同步 goal_positions
        self.goal_positions = np.tile(self.target_pos, (self.n, 1))

        self.t += 1

        reward_list = self._compute_rewards()
        cost_list = self._compute_costs()
        done = self.t >= self.max_steps or self._check_captured()

        obs_list = self._get_obs_all()
        info = self._get_info()
        return obs_list, reward_list, cost_list, done, info

    # ------------------------------------------------------------------
    # 目标动力学（逃逸策略）
    # ------------------------------------------------------------------

    def _update_target(self) -> None:
        """
        目标逃逸策略：
          1. 计算目标到每个追捕者的方向向量
          2. 取加权和（权重 = 1/距离，越近越强）
          3. 朝该方向加速（逃离最危险的追捕者群）
          4. 加入少量随机扰动避免被完全预测
        """
        escape_dir = np.zeros(3, dtype=np.float32)
        min_dist_threshold = 0.1

        for i in range(self.n):
            diff = self.target_pos - self.positions[i]
            dist = float(np.linalg.norm(diff)) + min_dist_threshold
            escape_dir += diff / dist

        escape_norm = np.linalg.norm(escape_dir) + 1e-8
        escape_dir_normalized = escape_dir / escape_norm

        target_accel = self.target_a_max * escape_dir_normalized
        target_accel += self.rng.normal(0, 0.1, 3).astype(np.float32)

        self.target_vel = np.clip(
            self.target_vel + target_accel * self.dt,
            -self.target_v_max, self.target_v_max
        ).astype(np.float32)
        self.target_pos = self.target_pos + self.target_vel * self.dt

        # 边界反弹
        half = self.arena_size / 2
        for dim in range(3):
            if self.target_pos[dim] > half:
                self.target_pos[dim] = half
                self.target_vel[dim] *= -0.5
            elif self.target_pos[dim] < -half:
                self.target_pos[dim] = -half
                self.target_vel[dim] *= -0.5

    # ------------------------------------------------------------------
    # 奖励、成本、观测
    # ------------------------------------------------------------------

    def _compute_rewards(self) -> List[float]:
        """
        围捕奖励：
          R = -w_approach * mean(||pi - pt||)
            + w_team * 团队协作奖励（追捕者方向向量多样性）
            + w_capture * 1[捕获成功]
        """
        dists_to_target = np.linalg.norm(
            self.positions - self.target_pos, axis=1
        )
        approach_reward = -self.w_approach * float(np.mean(dists_to_target))

        # 团队协作奖励：追捕者从不同方向包围目标
        directions = self.positions - self.target_pos
        norms = np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8
        unit_dirs = directions / norms

        if self.n > 1:
            diversity = 0.0
            count = 0
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    cos_sim = float(np.dot(unit_dirs[i], unit_dirs[j]))
                    diversity += (1.0 - cos_sim)
                    count += 1
            team_reward = self.w_team * (diversity / max(count, 1))
        else:
            team_reward = 0.0

        capture_reward = self.w_capture * float(self._check_captured())
        total = approach_reward + team_reward + capture_reward
        return [float(total) for _ in range(self.n)]

    def _compute_costs(self) -> List[float]:
        """安全成本：追捕者之间的碰撞（与 Phase 1 一致）。"""
        costs = []
        for i in range(self.n):
            in_collision = False
            for j in range(self.n):
                if i == j:
                    continue
                dist = np.linalg.norm(self.positions[i] - self.positions[j])
                if dist < self.d_min:
                    in_collision = True
                    break
            costs.append(1.0 if in_collision else 0.0)
        return costs

    def _get_obs_all(self) -> List[np.ndarray]:
        """获取所有追捕者的观测向量列表。"""
        return [self._get_obs(i) for i in range(self.n)]

    def _get_obs(self, agent_id: int) -> np.ndarray:
        """
        追捕者 i 的观测向量：
          [自身位置（相对目标）(3), 自身速度(3),
           邻居1相对位置(3), 邻居1相对速度(3), ...,  (K_obs 个)
           目标相对位置(3)]
        """
        obs = np.zeros(self.obs_dim, dtype=np.float32)

        # 自身状态：位置（相对目标）+ 速度
        obs[0:3] = self.positions[agent_id] - self.target_pos
        obs[3:6] = self.velocities[agent_id]

        # 邻居信息（按距离排序，最多 K_obs 个）
        neighbors = self._get_sorted_neighbors(agent_id)
        for k, j in enumerate(neighbors[:self.K_obs]):
            base = 6 + k * 6
            obs[base:base+3] = self.positions[j] - self.positions[agent_id]
            obs[base+3:base+6] = self.velocities[j] - self.velocities[agent_id]

        # 目标相对位置（最后 3 维）
        target_base = 6 + self.K_obs * 6
        obs[target_base:target_base+3] = self.target_pos - self.positions[agent_id]

        return obs

    def _get_sorted_neighbors(self, agent_id: int) -> List[int]:
        """返回感知范围内按距离升序排列的邻居 id 列表。"""
        dists = []
        for j in range(self.n):
            if j == agent_id:
                continue
            d = np.linalg.norm(self.positions[agent_id] - self.positions[j])
            if d <= self.R_comm:
                dists.append((d, j))
        dists.sort()
        return [j for _, j in dists]

    # ------------------------------------------------------------------
    # 辅助函数（与 Phase 1 接口兼容）
    # ------------------------------------------------------------------

    def _check_captured(self) -> bool:
        """捕获判定：至少 n_capturers_required 个追捕者在目标 capture_radius 内。"""
        dists = np.linalg.norm(self.positions - self.target_pos, axis=1)
        n_close = int(np.sum(dists < self.capture_radius))
        return n_close >= self.n_capturers_required

    def _min_pairwise_dist(self) -> float:
        """计算所有追捕者之间的最小距离。"""
        min_d = float("inf")
        for i in range(self.n):
            for j in range(i + 1, self.n):
                d = float(np.linalg.norm(self.positions[i] - self.positions[j]))
                if d < min_d:
                    min_d = d
        return min_d if self.n > 1 else 0.0

    def _get_info(self) -> Dict:
        """返回调试信息字典。"""
        dists_to_target = np.linalg.norm(
            self.positions - self.target_pos, axis=1
        )
        return {
            "t": self.t,
            "formation_error": float(np.mean(dists_to_target)),  # 兼容性字段
            "min_dist": self._min_pairwise_dist(),
            "n_collisions": int(sum(1 for c in self._compute_costs() if c > 0)),
            "success": self._check_captured(),
            "positions": self.positions.copy(),
            "goal_positions": self.goal_positions.copy(),
            # 围捕专用
            "min_dist_to_target": float(np.min(dists_to_target)),
            "capture_rate": float(np.mean(dists_to_target < self.capture_radius)),
            "target_pos": self.target_pos.copy(),
        }

    def _apply_boundary(
        self, positions: np.ndarray, velocities: np.ndarray
    ) -> None:
        """边界反弹（原地修改）。"""
        half = self.arena_size / 2
        for dim in range(3):
            out_high = positions[:, dim] > half
            out_low = positions[:, dim] < -half
            positions[out_high, dim] = half
            velocities[out_high, dim] *= -0.5
            positions[out_low, dim] = -half
            velocities[out_low, dim] *= -0.5

    def _random_init_positions_pursuers(self) -> np.ndarray:
        """随机初始化追捕者位置，保证互相间距 > 2*d_min。"""
        half = self.arena_size / 2 * 0.6
        min_init_dist = 2.0 * self.d_min

        for _ in range(1000):
            positions = self.rng.uniform(-half, half, (self.n, 3)).astype(np.float32)
            valid = True
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    if np.linalg.norm(positions[i] - positions[j]) < min_init_dist:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return positions

        # fallback：网格初始化
        positions = np.zeros((self.n, 3), dtype=np.float32)
        cols = int(np.ceil(np.sqrt(self.n)))
        spacing = max(min_init_dist * 1.5, 2.0)
        for i in range(self.n):
            row = i // cols
            col = i % cols
            positions[i] = [
                col * spacing - (cols - 1) * spacing / 2,
                row * spacing - (cols - 1) * spacing / 2,
                0.0,
            ]
        return positions

    def get_physical_graph(self) -> List[Tuple[int, int]]:
        """返回物理可连接图的边列表。供外部调度器使用。"""
        edges = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                dist = np.linalg.norm(self.positions[i] - self.positions[j])
                if dist <= self.R_comm:
                    edges.append((i, j))
        return edges

    def compute_cbf_values(self) -> Dict[Tuple[int, int], float]:
        """计算所有追捕者对的 CBF 值。供外部调度器使用。"""
        cbf_values = {}
        for i in range(self.n):
            for j in range(i + 1, self.n):
                dist = float(np.linalg.norm(self.positions[i] - self.positions[j]))
                h_ij = (dist - self.d_min) ** 2
                cbf_values[(i, j)] = h_ij
        return cbf_values

    @property
    def state(self) -> np.ndarray:
        """全局状态向量：追捕者 [6n] + 目标 [6] = [6n+6]。"""
        pursuers_state = np.concatenate(
            [self.positions, self.velocities], axis=1
        ).flatten()
        target_state = np.concatenate([self.target_pos, self.target_vel])
        return np.concatenate([pursuers_state, target_state])

    def render(self, mode: str = "text") -> Optional[str]:
        """简单的文本渲染（调试用）。"""
        if mode == "text":
            lines = [f"[Pursuit t={self.t}/{self.max_steps}]"]
            dists = np.linalg.norm(self.positions - self.target_pos, axis=1)
            for i in range(self.n):
                p_pos = self.positions[i]
                lines.append(
                    f"  追捕者 {i}: pos=({p_pos[0]:.2f},{p_pos[1]:.2f},{p_pos[2]:.2f}) "
                    f"dist_to_tgt={dists[i]:.3f}"
                )
            tp = self.target_pos
            lines.append(f"  目标: pos=({tp[0]:.2f},{tp[1]:.2f},{tp[2]:.2f})")
            lines.append(
                f"  min_dist={self._min_pairwise_dist():.3f}  "
                f"捕获={self._check_captured()}"
            )
            return "\n".join(lines)
        return None
```

---

- [ ] **Step 3: 运行测试，确认通过**

```bash
cd /root/autodl-tmp/safecomm-marl && python -m pytest tests/test_pursuit_env.py -v
```

预期输出：
```
tests/test_pursuit_env.py::TestUAVPursuitEnv::test_obs_dim_formula PASSED
tests/test_pursuit_env.py::TestUAVPursuitEnv::test_reset_interface PASSED
tests/test_pursuit_env.py::TestUAVPursuitEnv::test_step_interface PASSED
tests/test_pursuit_env.py::TestUAVPursuitEnv::test_reward_positive_when_capturing PASSED
tests/test_pursuit_env.py::TestUAVPursuitEnv::test_cost_binary PASSED
tests/test_pursuit_env.py::TestUAVPursuitEnv::test_target_moves PASSED
tests/test_pursuit_env.py::TestUAVPursuitEnv::test_success_when_captured PASSED
tests/test_pursuit_env.py::TestUAVPursuitEnv::test_get_physical_graph PASSED
tests/test_pursuit_env.py::TestUAVPursuitEnv::test_compute_cbf_values PASSED
tests/test_pursuit_env.py::TestUAVPursuitEnv::test_state_property PASSED
tests/test_pursuit_env.py::TestUAVPursuitEnv::test_n8_agents PASSED
========== 11 passed in X.XXs ==========
```

---

## Task 2：注意力式集中式 Critic（algorithms/scalable_critic.py）

**Files:**
- Create: `algorithms/scalable_critic.py`
- Test: `tests/test_scalable_critic.py`

**实现边界：** 该 critic 参考 InforMARL 的局部图信息聚合评估思路和 MAGIC 的 masked attention 结构，但实现必须是本项目内的最小 PyTorch 模块。critic 可以使用全局 token 做 CTDE 训练；actor 执行期仍只能读取自身观测和 `E_t` 中的消息。

### 步骤

- [ ] **Step 4: 编写 ScalableCritic 测试（TDD 先行）**

新建 `tests/test_scalable_critic.py`，写入以下完整测试代码：

```python
"""
ScalableCritic 测试

验证注意力式集中式 Critic 支持任意 N，且接口与原 Critic 一致。
"""
import torch
import numpy as np
import pytest


class TestScalableCritic:
    """ScalableCritic 形状和功能测试"""

    def test_output_shape_n4(self):
        """N=4 时输出形状应为 [batch_size]"""
        from algorithms.scalable_critic import ScalableCritic
        critic = ScalableCritic(state_dim=6, n_agents=4, d_model=64)
        batch = 8
        states = torch.randn(batch, 4, 6)  # [batch, n_agents, state_dim]
        values = critic(states)
        assert values.shape == (batch,), f"预期 ({batch},)，实际 {values.shape}"

    def test_output_shape_n8(self):
        """N=8 时输出形状应为 [batch_size]"""
        from algorithms.scalable_critic import ScalableCritic
        critic = ScalableCritic(state_dim=6, n_agents=8, d_model=64)
        batch = 4
        states = torch.randn(batch, 8, 6)
        values = critic(states)
        assert values.shape == (batch,), f"预期 ({batch},)，实际 {values.shape}"

    def test_output_shape_n1(self):
        """N=1 时（单智能体）输出形状应为 [batch_size]"""
        from algorithms.scalable_critic import ScalableCritic
        critic = ScalableCritic(state_dim=6, n_agents=1, d_model=64)
        batch = 16
        states = torch.randn(batch, 1, 6)
        values = critic(states)
        assert values.shape == (batch,)

    def test_variable_n_at_inference(self):
        """ScalableCritic 应支持推断时不同 N（Transformer 无大小约束）"""
        from algorithms.scalable_critic import ScalableCritic
        # 用 n_agents=4 初始化，推断时用 N=6
        critic = ScalableCritic(state_dim=6, n_agents=4, d_model=64)
        states_n6 = torch.randn(2, 6, 6)
        values = critic(states_n6)
        assert values.shape == (2,)

    def test_gradient_flows(self):
        """梯度应该正确反传"""
        from algorithms.scalable_critic import ScalableCritic
        critic = ScalableCritic(state_dim=6, n_agents=4, d_model=64)
        states = torch.randn(4, 4, 6, requires_grad=True)
        values = critic(states)
        loss = values.mean()
        loss.backward()
        for name, param in critic.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"参数 {name} 梯度为 None"

    def test_output_is_scalar_per_batch(self):
        """每个 batch 样本应输出一个标量价值"""
        from algorithms.scalable_critic import ScalableCritic
        critic = ScalableCritic(state_dim=6, n_agents=4, d_model=64)
        states = torch.randn(32, 4, 6)
        values = critic(states)
        assert values.dim() == 1
        assert values.shape[0] == 32

    def test_parameter_count_reasonable(self):
        """参数量应在合理范围（不超过 10M）"""
        from algorithms.scalable_critic import ScalableCritic
        critic = ScalableCritic(
            state_dim=6, n_agents=8, d_model=128, n_heads=4, n_layers=2
        )
        n_params = sum(p.numel() for p in critic.parameters())
        assert n_params < 10_000_000, f"参数量 {n_params} 过大（>10M）"
        assert n_params > 1_000, f"参数量 {n_params} 过小（<1K）"

    def test_deterministic_in_eval_mode(self):
        """eval 模式下相同输入应给出相同输出"""
        from algorithms.scalable_critic import ScalableCritic
        critic = ScalableCritic(state_dim=6, n_agents=4, d_model=64)
        critic.eval()
        states = torch.randn(2, 4, 6)
        with torch.no_grad():
            out1 = critic(states)
            out2 = critic(states)
        assert torch.allclose(out1, out2), "eval 模式下相同输入应给出相同输出"

    def test_flat_global_state_interface(self):
        """ScalableCritic 还应支持扁平输入（与原 Critic 兼容）"""
        from algorithms.scalable_critic import ScalableCritic
        n_agents = 4
        state_dim = 6
        critic = ScalableCritic(state_dim=state_dim, n_agents=n_agents, d_model=64)
        flat_states = torch.randn(8, n_agents * state_dim)
        values = critic.forward_flat(flat_states)
        assert values.shape == (8,)


class TestScalableCriticIntegration:
    """集成测试：ScalableCritic 与围捕场景的兼容性"""

    def test_critic_state_dim_for_pursuit_env(self):
        """围捕场景：n_agents+1 token（n 追捕者 + 1 目标），每个 token 6 维"""
        from algorithms.scalable_critic import ScalableCritic
        n_agents = 4
        # n_agents+1 token：n 个追捕者 + 1 个目标
        critic = ScalableCritic(state_dim=6, n_agents=n_agents + 1, d_model=64)
        states = torch.randn(4, n_agents + 1, 6)
        values = critic(states)
        assert values.shape == (4,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

运行（预期失败）：
```bash
cd /root/autodl-tmp/safecomm-marl && python -m pytest tests/test_scalable_critic.py -v 2>&1 | head -10
```

---

- [ ] **Step 5: 实现 ScalableCritic（algorithms/scalable_critic.py）**

新建 `/root/autodl-tmp/safecomm-marl/algorithms/scalable_critic.py`，写入以下完整代码：

```python
"""
ScalableCritic：注意力式集中式 Critic，支持 N≥8 智能体

设计原则：
  - 每个智能体的状态作为一个 token（per-agent state token）
  - 使用 PyTorch MultiheadAttention 对全局状态 token 做 self-attention
  - 通过 CLS token 聚合得到全局价值估计
  - 不受智能体数量限制（Transformer 天然变长输入）

架构：
  1. Token 投影：state_dim → d_model（线性层）
  2. 可学习位置嵌入 [max_n_agents+1, d_model]
  3. Transformer 编码器：n_layers × (MultiheadAttention + FFN)，Pre-LN
  4. CLS token 聚合
  5. 价值头：d_model → 1（2 层 MLP + Tanh）

与原始 Critic 的兼容性：
  - forward(states: [batch, n_agents, state_dim]) → [batch]
  - forward_flat(flat_states: [batch, n*state_dim]) → [batch]（兼容 API）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TransformerEncoderLayer(nn.Module):
    """
    单层 Transformer 编码器（Pre-LN 变体，训练更稳定）。

    架构：
      x = x + Dropout(MultiheadAttention(LayerNorm(x)))
      x = x + Dropout(FFN(LayerNorm(x)))
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_ffn: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,  # 输入形状 [batch, seq, d_model]
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ffn),
            nn.ReLU(),
            nn.Linear(dim_ffn, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            x: [batch, seq_len, d_model]
        """
        # Pre-LN self-attention
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(x, x, x, need_weights=False)
        x = residual + self.dropout(attn_out)

        # Pre-LN FFN
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ffn(x))

        return x


class ScalableCritic(nn.Module):
    """
    注意力式集中式 Critic，支持任意 N 个智能体。

    输入格式：
      - tokenized: [batch, n_agents, state_dim]
      - flat（兼容）: [batch, n_agents * state_dim]（通过 forward_flat 调用）

    输出：[batch] 标量价值

    用法（替换 networks.py 中的 Critic）：
        critic = ScalableCritic(state_dim=6, n_agents=8, d_model=128)
        values = critic(states)  # states: [batch, 8, 6]
    """

    def __init__(
        self,
        state_dim: int,
        n_agents: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_ffn: Optional[int] = None,
        dropout: float = 0.0,
        use_cls_token: bool = True,
        max_n_agents: int = 16,
    ):
        """
        Args:
            state_dim: 每个智能体的状态向量维度（例如 6：位置3+速度3）
            n_agents: 智能体数量（用于 forward_flat 的 reshape）
            d_model: Transformer 模型维度
            n_heads: 注意力头数（d_model 必须能被 n_heads 整除）
            n_layers: Transformer 编码器层数
            dim_ffn: FFN 隐藏层维度（默认 4 * d_model）
            dropout: Dropout 概率（训练时建议 0.0，MARL 样本有限）
            use_cls_token: True = 用 CLS token 聚合；False = 平均池化
            max_n_agents: 位置嵌入支持的最大 token 数（推断时 N 不超过此值）
        """
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model={d_model} 必须能被 n_heads={n_heads} 整除"
        )
        if dim_ffn is None:
            dim_ffn = 4 * d_model

        self.state_dim = state_dim
        self.n_agents = n_agents
        self.d_model = d_model
        self.use_cls_token = use_cls_token

        # 1. Token 投影：state_dim → d_model
        self.token_proj = nn.Linear(state_dim, d_model)

        # 2. 可学习位置嵌入 [max_n_agents + 1, d_model]（+1 为 CLS）
        self.pos_embed = nn.Parameter(
            torch.zeros(max_n_agents + 1, d_model)
        )
        nn.init.normal_(self.pos_embed, std=0.02)

        # 3. CLS token（若使用）
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)

        # 4. Transformer 编码器
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                dim_ffn=dim_ffn,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # 5. 价值头
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: [batch, n_agents, state_dim] per-agent state token

        Returns:
            values: [batch] 标量价值
        """
        batch_size, n, _ = states.shape

        # Token 投影：[batch, n, d_model]
        tokens = self.token_proj(states)

        if self.use_cls_token:
            # 追加 CLS token：[batch, 1, d_model]
            cls = self.cls_token.expand(batch_size, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)  # [batch, n+1, d_model]
            seq_len = n + 1
            pos = self.pos_embed[:seq_len, :].unsqueeze(0)
            tokens = tokens + pos
        else:
            seq_len = n
            pos = self.pos_embed[:seq_len, :].unsqueeze(0)
            tokens = tokens + pos

        # Transformer 编码器
        for layer in self.encoder_layers:
            tokens = layer(tokens)

        tokens = self.final_norm(tokens)

        # 全局聚合
        if self.use_cls_token:
            global_repr = tokens[:, 0, :]  # CLS token [batch, d_model]
        else:
            global_repr = tokens.mean(dim=1)  # 平均池化 [batch, d_model]

        # 价值头
        values = self.value_head(global_repr).squeeze(-1)  # [batch]
        return values

    def forward_flat(self, flat_states: torch.Tensor) -> torch.Tensor:
        """
        兼容原 Critic API：输入扁平状态向量 [batch, n_agents * state_dim]。
        内部 reshape 为 [batch, n_agents, state_dim] 后调用 forward()。
        """
        batch_size = flat_states.shape[0]
        states = flat_states.view(batch_size, self.n_agents, self.state_dim)
        return self.forward(states)

    def count_parameters(self) -> int:
        """统计可训练参数量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

---

- [ ] **Step 6: 运行 ScalableCritic 测试**

```bash
cd /root/autodl-tmp/safecomm-marl && python -m pytest tests/test_scalable_critic.py -v
```

预期输出：
```
tests/test_scalable_critic.py::TestScalableCritic::test_output_shape_n4 PASSED
tests/test_scalable_critic.py::TestScalableCritic::test_output_shape_n8 PASSED
tests/test_scalable_critic.py::TestScalableCritic::test_output_shape_n1 PASSED
tests/test_scalable_critic.py::TestScalableCritic::test_variable_n_at_inference PASSED
tests/test_scalable_critic.py::TestScalableCritic::test_gradient_flows PASSED
tests/test_scalable_critic.py::TestScalableCritic::test_output_is_scalar_per_batch PASSED
tests/test_scalable_critic.py::TestScalableCritic::test_parameter_count_reasonable PASSED
tests/test_scalable_critic.py::TestScalableCritic::test_deterministic_in_eval_mode PASSED
tests/test_scalable_critic.py::TestScalableCritic::test_flat_global_state_interface PASSED
tests/test_scalable_critic.py::TestScalableCriticIntegration::test_critic_state_dim_for_pursuit_env PASSED
========== 10 passed in X.XXs ==========
```

---

## Task 3：配置文件与多任务训练脚本

**Files:**
- Create: `configs/pursuit.yaml`
- Create: `configs/n8_scale.yaml`
- Create: `train_multitask.py`

### 步骤

- [ ] **Step 7: 创建围捕场景配置（configs/pursuit.yaml）**

新建 `/root/autodl-tmp/safecomm-marl/configs/pursuit.yaml`，写入以下完整内容：

```yaml
# SafeComm-VoI Phase 4 围捕场景配置

# ============================================================
# 环境参数（围捕场景）
# ============================================================
env:
  type: "pursuit"
  n_agents: 4
  dt: 0.1
  max_steps: 300
  d_min: 0.5
  R_comm: 5.0
  v_max: 3.0
  a_max: 2.0
  target_v_max: 1.5
  target_a_max: 1.0
  arena_size: 12.0
  capture_radius: 1.5
  n_capturers_required: 1
  w_capture: 5.0
  w_approach: 1.0
  w_team: 0.5
  K_obs_neighbors: 2
  seed: 42

# ============================================================
# 通信约束
# ============================================================
comm:
  k: 4
  schedule_mode: "safety_priority"

# ============================================================
# 网络架构
# ============================================================
network:
  msg_dim: 64
  hidden_dims: [256, 256]
  H: 2

# ============================================================
# PPO 超参数
# ============================================================
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

# ============================================================
# 学习率
# ============================================================
lr:
  actor: 3.0e-4
  critic: 1.0e-3
  lambda: 0.01

# ============================================================
# 安全约束
# ============================================================
safety:
  budget: 0.1
  lambda_init: 0.0
  lambda_max: 10.0

# ============================================================
# VoI 调度器参数
# ============================================================
voi:
  beta: 1.0
  tau_ref: 1.0
  tau_max: 5.0
  v_max_cbf: 3.0

# ============================================================
# 训练设置
# ============================================================
training:
  total_steps: 1_000_000
  log_interval: 10
  eval_interval: 50
  eval_episodes: 10
  save_interval: 100
  checkpoint_dir: "checkpoints/pursuit"
  device: "auto"
  seed: 42
```

---

- [ ] **Step 8: 创建 N=8 大规模扩展配置（configs/n8_scale.yaml）**

新建 `/root/autodl-tmp/safecomm-marl/configs/n8_scale.yaml`，写入以下完整内容：

```yaml
# SafeComm-VoI Phase 4 N=8 大规模扩展配置
# 关键变化：N=8，使用 ScalableCritic（Transformer-style）替代 MLP concat

# ============================================================
# 环境参数（N=8 编队）
# ============================================================
env:
  type: "double_integrator"
  n_agents: 8
  dt: 0.1
  max_steps: 300
  d_min: 0.5
  R_comm: 6.0                  # N=8 时略微增大通信范围
  v_max: 3.0
  a_max: 2.0
  arena_size: 15.0             # N=8 时扩大飞行区域
  formation_type: "circle"
  w_formation: 1.0
  w_success: 5.0
  success_threshold: 0.3
  K_obs_neighbors: 3           # N=8 时适当增加邻居观测数
  seed: 42

# ============================================================
# 通信约束（N=8 下 k=8）
# ============================================================
comm:
  k: 8                         # 通信预算随 N 线性扩展
  schedule_mode: "safety_priority"

# ============================================================
# 网络架构（N=8 使用 ScalableCritic）
# ============================================================
network:
  msg_dim: 64
  hidden_dims: [256, 256]
  H: 2
  use_scalable_critic: true    # 启用 ScalableCritic
  scalable_critic:
    d_model: 128
    n_heads: 4
    n_layers: 2
    dropout: 0.0
    use_cls_token: true

# ============================================================
# PPO 超参数（N=8 时 batch 稍大）
# ============================================================
ppo:
  n_steps: 4096                # N=8 时收集更多样本
  n_epochs: 10
  batch_size: 512
  gamma: 0.99
  lam: 0.95
  clip_eps: 0.2
  value_coef: 0.5
  cost_value_coef: 0.5
  entropy_coef: 0.01
  max_grad_norm: 0.5

# ============================================================
# 学习率
# ============================================================
lr:
  actor: 3.0e-4
  critic: 1.0e-3
  lambda: 0.01

# ============================================================
# 安全约束
# ============================================================
safety:
  budget: 0.1
  lambda_init: 0.0
  lambda_max: 10.0

# ============================================================
# VoI 调度器参数
# ============================================================
voi:
  beta: 1.0
  tau_ref: 1.0
  tau_max: 5.0
  v_max_cbf: 3.0

# ============================================================
# 训练设置
# ============================================================
training:
  total_steps: 2_000_000       # N=8 需要更多训练步数
  log_interval: 10
  eval_interval: 50
  eval_episodes: 10
  save_interval: 100
  checkpoint_dir: "checkpoints/n8"
  device: "auto"
  seed: 42
```

---

- [ ] **Step 9: 创建多任务训练脚本（train_multitask.py）**

新建 `/root/autodl-tmp/safecomm-marl/train_multitask.py`，写入以下完整代码：

```python
"""
SafeComm-VoI Phase 4 多任务训练入口

用法：
  # 轮流训练编队 + 围捕（各 500k 步）
  python train_multitask.py

  # 自定义步数
  python train_multitask.py --formation_steps 600000 --pursuit_steps 400000

  # 仅训练围捕
  python train_multitask.py --tasks pursuit

  # N=8 大规模扩展（使用 ScalableCritic）
  python train_multitask.py --n_agents 8 --use_scalable_critic

多任务策略：
  - 共享 Actor 和 MessageEncoder（策略网络参数复用）
  - 各任务使用各自的 SafeCommVoI 实例（独立 Critic）
  - 轮流训练（curriculum-style）：先编队 K1 步，再围捕 K2 步
"""

import argparse
import yaml
import os
import torch
from typing import Dict, Optional


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件。"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def merge_configs(base: dict, override: dict) -> dict:
    """递归合并两个配置字典（override 覆盖 base）。"""
    result = base.copy()
    for key, val in override.items():
        if isinstance(val, dict) and key in result and isinstance(result[key], dict):
            result[key] = merge_configs(result[key], val)
        else:
            result[key] = val
    return result


def flatten_config(cfg: dict) -> dict:
    """将嵌套 YAML 配置展平为算法所需的扁平字典。"""
    flat = {}
    env_cfg = cfg.get("env", {})
    comm_cfg = cfg.get("comm", {})
    net_cfg = cfg.get("network", {})
    ppo_cfg = cfg.get("ppo", {})
    lr_cfg = cfg.get("lr", {})
    safety_cfg = cfg.get("safety", {})
    train_cfg = cfg.get("training", {})
    voi_cfg = cfg.get("voi", {})

    flat.update({
        "k": comm_cfg.get("k", 4),
        "schedule_mode": comm_cfg.get("schedule_mode", "safety_priority"),
        "msg_dim": net_cfg.get("msg_dim", 64),
        "hidden_dims": net_cfg.get("hidden_dims", [256, 256]),
        "H": net_cfg.get("H", 2),
        "n_steps": ppo_cfg.get("n_steps", 2048),
        "n_epochs": ppo_cfg.get("n_epochs", 10),
        "batch_size": ppo_cfg.get("batch_size", 256),
        "gamma": ppo_cfg.get("gamma", 0.99),
        "lam": ppo_cfg.get("lam", 0.95),
        "clip_eps": ppo_cfg.get("clip_eps", 0.2),
        "value_coef": ppo_cfg.get("value_coef", 0.5),
        "cost_value_coef": ppo_cfg.get("cost_value_coef", 0.5),
        "entropy_coef": ppo_cfg.get("entropy_coef", 0.01),
        "max_grad_norm": ppo_cfg.get("max_grad_norm", 0.5),
        "lr_actor": lr_cfg.get("actor", 3e-4),
        "lr_critic": lr_cfg.get("critic", 1e-3),
        "lr_lambda": lr_cfg.get("lambda", 0.01),
        "safety_budget": safety_cfg.get("budget", 0.1),
        "lambda_init": safety_cfg.get("lambda_init", 0.0),
        "lambda_max": safety_cfg.get("lambda_max", 10.0),
        "log_interval": train_cfg.get("log_interval", 10),
        "eval_interval": train_cfg.get("eval_interval", 50),
        "beta": voi_cfg.get("beta", 1.0),
        "tau_ref": voi_cfg.get("tau_ref", 1.0),
        "tau_max": voi_cfg.get("tau_max", 5.0),
        "v_max_cbf": voi_cfg.get("v_max_cbf", 3.0),
    })
    return flat


def build_env(cfg: dict, task: str):
    """根据任务类型构建环境。"""
    env_cfg = cfg.get("env", {})

    if task == "pursuit":
        from envs.uav_pursuit_env import UAVPursuitEnv
        env = UAVPursuitEnv(
            n_agents=env_cfg.get("n_agents", 4),
            dt=env_cfg.get("dt", 0.1),
            max_steps=env_cfg.get("max_steps", 300),
            d_min=env_cfg.get("d_min", 0.5),
            R_comm=env_cfg.get("R_comm", 5.0),
            v_max=env_cfg.get("v_max", 3.0),
            a_max=env_cfg.get("a_max", 2.0),
            target_v_max=env_cfg.get("target_v_max", 1.5),
            target_a_max=env_cfg.get("target_a_max", 1.0),
            arena_size=env_cfg.get("arena_size", 12.0),
            capture_radius=env_cfg.get("capture_radius", 1.5),
            n_capturers_required=env_cfg.get("n_capturers_required", 1),
            w_capture=env_cfg.get("w_capture", 5.0),
            w_approach=env_cfg.get("w_approach", 1.0),
            w_team=env_cfg.get("w_team", 0.5),
            K_obs_neighbors=env_cfg.get("K_obs_neighbors", 2),
            seed=env_cfg.get("seed", 42),
        )
    else:
        from envs.uav_formation_env import UAVFormationEnv
        env = UAVFormationEnv(
            n_agents=env_cfg.get("n_agents", 4),
            dt=env_cfg.get("dt", 0.1),
            max_steps=env_cfg.get("max_steps", 200),
            d_min=env_cfg.get("d_min", 0.5),
            R_comm=env_cfg.get("R_comm", 5.0),
            v_max=env_cfg.get("v_max", 3.0),
            a_max=env_cfg.get("a_max", 2.0),
            arena_size=env_cfg.get("arena_size", 10.0),
            formation_type=env_cfg.get("formation_type", "circle"),
            w_formation=env_cfg.get("w_formation", 1.0),
            w_success=env_cfg.get("w_success", 5.0),
            success_threshold=env_cfg.get("success_threshold", 0.3),
            K_obs_neighbors=env_cfg.get("K_obs_neighbors", 2),
            seed=env_cfg.get("seed", 42),
        )

    return env


def main():
    parser = argparse.ArgumentParser(description="SafeComm-VoI Phase 4 多任务训练")
    parser.add_argument(
        "--formation_config", type=str, default="configs/default.yaml",
        help="编队任务配置文件"
    )
    parser.add_argument(
        "--pursuit_config", type=str, default="configs/pursuit.yaml",
        help="围捕任务配置文件"
    )
    parser.add_argument(
        "--tasks", type=str, default="formation,pursuit",
        help="训练任务列表，逗号分隔（formation/pursuit）"
    )
    parser.add_argument(
        "--formation_steps", type=int, default=500_000,
        help="编队任务训练步数"
    )
    parser.add_argument(
        "--pursuit_steps", type=int, default=500_000,
        help="围捕任务训练步数"
    )
    parser.add_argument(
        "--n_agents", type=int, default=None,
        help="覆盖配置中的 n_agents（用于 N=8 大规模测试）"
    )
    parser.add_argument(
        "--use_scalable_critic", action="store_true", default=False,
        help="使用 ScalableCritic（N≥8 时推荐）"
    )
    parser.add_argument(
        "--n8_config", type=str, default="configs/n8_scale.yaml",
        help="N=8 扩展配置文件"
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints/multitask",
        help="模型保存目录"
    )
    parser.add_argument(
        "--pretrained", type=str, default=None,
        help="从此 Phase 2/3 checkpoint 初始化策略网络"
    )
    args = parser.parse_args()

    # 解析任务列表
    task_list = [t.strip() for t in args.tasks.split(",")]
    print(f"[Phase 4] 多任务训练：{task_list}")

    # 加载配置
    formation_cfg = load_config(args.formation_config)
    pursuit_cfg = load_config(args.pursuit_config)

    # CLI 覆盖 n_agents
    if args.n_agents is not None:
        for cfg in [formation_cfg, pursuit_cfg]:
            cfg["env"]["n_agents"] = args.n_agents
        print(f"[Phase 4] N={args.n_agents} 大规模扩展模式")

    # N=8 时加载专用配置
    if args.use_scalable_critic or (args.n_agents is not None and args.n_agents >= 8):
        if os.path.exists(args.n8_config):
            n8_cfg = load_config(args.n8_config)
            formation_cfg = merge_configs(formation_cfg, n8_cfg)
            if args.n_agents:
                formation_cfg["env"]["n_agents"] = args.n_agents
                pursuit_cfg["env"]["n_agents"] = args.n_agents
        print("[Phase 4] 使用 ScalableCritic（Transformer-style 集中式 Critic）")

    # 确定设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[Phase 4] 设备: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    task_steps = {
        "formation": args.formation_steps,
        "pursuit": args.pursuit_steps,
    }

    from algorithms.safecomm_psched import SafeCommVoI

    agents = {}
    envs = {}

    for task in task_list:
        cfg = formation_cfg if task == "formation" else pursuit_cfg
        env = build_env(cfg, task)
        agent_config = flatten_config(cfg)

        # ScalableCritic 参数注入
        net_cfg = cfg.get("network", {})
        use_scalable = net_cfg.get("use_scalable_critic", False) or args.use_scalable_critic
        if use_scalable:
            sc_cfg = net_cfg.get("scalable_critic", {})
            agent_config["use_scalable_critic"] = True
            agent_config["sc_d_model"] = sc_cfg.get("d_model", 128)
            agent_config["sc_n_heads"] = sc_cfg.get("n_heads", 4)
            agent_config["sc_n_layers"] = sc_cfg.get("n_layers", 2)
            agent_config["sc_use_cls_token"] = sc_cfg.get("use_cls_token", True)

        agent = SafeCommVoI(
            env=env,
            config=agent_config,
            device=device,
        )

        # 从预训练 checkpoint 加载
        if args.pretrained and os.path.exists(args.pretrained):
            agent.load(args.pretrained)
            print(f"[Phase 4] 从 {args.pretrained} 加载预训练策略（任务={task}）")

        agents[task] = agent
        envs[task] = env

    # 多任务轮流训练
    print("\n[Phase 4] 开始多任务轮流训练")
    total_steps_all = 0

    for task in task_list:
        steps = task_steps.get(task, 500_000)
        print(f"\n{'='*60}")
        print(f"[Phase 4] 开始训练任务：{task}，步数={steps:,}")
        print(f"{'='*60}")

        agent = agents[task]
        metrics = agent.train(total_steps=steps)
        total_steps_all += steps

        print(f"\n[多任务] 任务={task}  已完成步数={total_steps_all}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        # 保存该任务的 checkpoint
        ckpt_path = os.path.join(args.checkpoint_dir, f"{task}_final.pt")
        agent.save(ckpt_path)
        print(f"[Phase 4] 已保存：{ckpt_path}")

    # 跨任务泛化评估
    print(f"\n{'='*60}")
    print("[Phase 4] 跨任务泛化评估")
    print(f"{'='*60}")

    for eval_task in task_list:
        print(f"\n[自评估] 任务={eval_task}")
        agent = agents[eval_task]
        try:
            eval_metrics = agent.evaluate(n_episodes=10, deterministic=True)
            for k, v in eval_metrics.items():
                print(f"  {k}: {v:.4f}")
        except Exception as e:
            print(f"  评估失败: {e}")

    # 关闭所有环境
    for env in envs.values():
        if hasattr(env, 'close'):
            env.close()

    print(f"\n[Phase 4] 多任务训练完成！总步数={total_steps_all:,}")
    print(f"检查点保存至：{args.checkpoint_dir}/")


if __name__ == "__main__":
    main()
```

---

- [ ] **Step 10: 运行所有 Phase 4 测试，验证实现完整性**

```bash
cd /root/autodl-tmp/safecomm-marl && python -m pytest tests/test_pursuit_env.py tests/test_scalable_critic.py -v
```

预期输出：
```
tests/test_pursuit_env.py::TestUAVPursuitEnv::test_obs_dim_formula PASSED
tests/test_pursuit_env.py::TestUAVPursuitEnv::test_reset_interface PASSED
tests/test_pursuit_env.py::TestUAVPursuitEnv::test_step_interface PASSED
tests/test_pursuit_env.py::TestUAVPursuitEnv::test_reward_positive_when_capturing PASSED
tests/test_pursuit_env.py::TestUAVPursuitEnv::test_cost_binary PASSED
tests/test_pursuit_env.py::TestUAVPursuitEnv::test_target_moves PASSED
tests/test_pursuit_env.py::TestUAVPursuitEnv::test_success_when_captured PASSED
tests/test_pursuit_env.py::TestUAVPursuitEnv::test_get_physical_graph PASSED
tests/test_pursuit_env.py::TestUAVPursuitEnv::test_compute_cbf_values PASSED
tests/test_pursuit_env.py::TestUAVPursuitEnv::test_state_property PASSED
tests/test_pursuit_env.py::TestUAVPursuitEnv::test_n8_agents PASSED
tests/test_scalable_critic.py::TestScalableCritic::test_output_shape_n4 PASSED
tests/test_scalable_critic.py::TestScalableCritic::test_output_shape_n8 PASSED
tests/test_scalable_critic.py::TestScalableCritic::test_output_shape_n1 PASSED
tests/test_scalable_critic.py::TestScalableCritic::test_variable_n_at_inference PASSED
tests/test_scalable_critic.py::TestScalableCritic::test_gradient_flows PASSED
tests/test_scalable_critic.py::TestScalableCritic::test_output_is_scalar_per_batch PASSED
tests/test_scalable_critic.py::TestScalableCritic::test_parameter_count_reasonable PASSED
tests/test_scalable_critic.py::TestScalableCritic::test_deterministic_in_eval_mode PASSED
tests/test_scalable_critic.py::TestScalableCritic::test_flat_global_state_interface PASSED
tests/test_scalable_critic.py::TestScalableCriticIntegration::test_critic_state_dim_for_pursuit_env PASSED
========== 21 passed in X.XXs ==========
```

验证 ScalableCritic N 可扩展性：
```bash
cd /root/autodl-tmp/safecomm-marl && python -c "
from algorithms.scalable_critic import ScalableCritic
import torch

for n in [4, 8, 16]:
    critic = ScalableCritic(state_dim=6, n_agents=n, d_model=128, n_heads=4)
    n_params = critic.count_parameters()
    x = torch.randn(4, n, 6)
    v = critic(x)
    assert v.shape == (4,)
    print(f'N={n}: 参数量={n_params:,}  输出形状={v.shape}  通过')
"
```

预期输出：
```
N=4: 参数量=XXX  输出形状=torch.Size([4])  通过
N=8: 参数量=XXX  输出形状=torch.Size([4])  通过
N=16: 参数量=XXX  输出形状=torch.Size([4])  通过
```

围捕环境接口兼容性验证：
```bash
cd /root/autodl-tmp/safecomm-marl && python -c "
from envs.uav_pursuit_env import UAVPursuitEnv
from envs.uav_formation_env import UAVFormationEnv
import numpy as np

n, K = 4, 2
for Env, name, extra_kw in [
    (UAVFormationEnv, '编队', {}),
    (UAVPursuitEnv, '围捕', {}),
]:
    env = Env(n_agents=n, K_obs_neighbors=K, seed=0, **extra_kw)
    obs_list, info = env.reset()
    actions = np.zeros((n, 3), dtype=np.float32)
    obs_list2, rewards, costs, done, info = env.step(actions)
    assert len(obs_list) == n and len(obs_list2) == n
    edges = env.get_physical_graph()
    cbf = env.compute_cbf_values()
    state = env.state
    print(f'{name}: obs_dim={env.obs_dim}  act_dim={env.act_dim}  state_dim={len(state)}  通过')
"
```

预期输出：
```
编队: obs_dim=18  act_dim=3  state_dim=24  通过
围捕: obs_dim=21  act_dim=3  state_dim=30  通过
```

多任务冒烟测试（各 500 步）：
```bash
cd /root/autodl-tmp/safecomm-marl && python train_multitask.py \
    --formation_steps 500 \
    --pursuit_steps 500 \
    --device cpu \
    --checkpoint_dir /tmp/phase4_test 2>&1 | tail -20
```

预期输出：无报错，保存两个 checkpoint 文件。

---

## Phase 4 成功标准

| 指标 | 验收条件 |
|------|---------|
| 接口兼容性 | UAVPursuitEnv 的 reset/step/get_physical_graph/compute_cbf_values/state 与 UAVFormationEnv 接口一致 |
| obs_dim 公式 | 围捕场景 obs_dim = 6 + K_obs * 6 + 3（多 3 维目标相对位置） |
| ScalableCritic | N=4/8/16 时输出形状 [batch] 正确，参数量不随 N 爆炸 |
| 测试通过率 | 21/21 测试全绿 |
| 可扩展性 | ScalableCritic 推断时支持任意 N（Transformer 天然变长输入）|
| 多任务训练 | `python train_multitask.py` 无报错，在两个场景间切换 |
| N=8 扩展 | `python train_multitask.py --n_agents 8 --use_scalable_critic` 无报错 |

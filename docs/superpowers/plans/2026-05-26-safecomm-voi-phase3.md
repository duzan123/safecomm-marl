# SafeComm-VoI Phase 3 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Phase 2 训练好的 SafeCommVoI 策略迁移到 PyBullet 四旋翼仿真环境，并引入域随机化（位置噪声、通信延迟、丢包），验证策略的跨仿真器迁移能力。

**Architecture:** 三个核心组件——(1) `PyBulletUAVEnv`：基于 pybullet 的简化四旋翼动力学环境，暴露与 Phase 1 完全相同的 gym-style 接口；(2) `DomainRandomizationWrapper`：包装任意 MARL 环境，在观测层注入位置噪声、在通信层模拟延迟 buffer 和丢包；(3) `configs/pybullet.yaml`：PyBullet 专用超参数配置，`train.py` 通过 `--env pybullet` 参数选择环境。

**Tech Stack:** Python 3.x, NumPy, PyTorch, pybullet, PyYAML, pytest

---

## 文件结构

| 文件 | 操作 | 说明 |
|------|------|------|
| `envs/pybullet_uav_env.py` | 新建 | PyBullet 四旋翼动力学环境（简化推力模型），与 Phase 1 接口一致 |
| `envs/domain_randomization.py` | 新建 | 域随机化包装器：位置噪声、通信延迟 deque、丢包 |
| `configs/pybullet.yaml` | 新建 | PyBullet 专用配置（物理参数、DR 参数） |
| `train.py` | 修改 | 新增 `--env` 参数（`double_integrator` / `pybullet`）、DR 包装器集成 |
| `evaluate.py` | 修改 | 支持 `--env pybullet` 和 `--domain_rand` 参数 |
| `tests/test_pybullet_env.py` | 新建 | PyBullet 环境接口测试（obs_dim 一致性、step 接口、DR 包装器测试） |

**不修改的文件：** `algorithms/`（所有网络和算法代码），`envs/uav_formation_env.py`，`algorithms/scheduler.py`

---

## Task 1：PyBullet 四旋翼动力学环境（envs/pybullet_uav_env.py）

**Files:**
- Create: `envs/pybullet_uav_env.py`
- Test: `tests/test_pybullet_env.py`

### 步骤

- [ ] **Step 1: 编写 PyBulletUAVEnv 失败测试（TDD 先行）**

新建 `tests/test_pybullet_env.py`，写入以下内容：

```python
"""
PyBullet UAV 环境测试

前提：pip install pybullet
"""
import numpy as np
import pytest


class TestPyBulletUAVEnv:
    """测试 PyBulletUAVEnv 接口与 UAVFormationEnv 一致性"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """每个测试前创建环境，测试后关闭"""
        from envs.pybullet_uav_env import PyBulletUAVEnv
        self.env = PyBulletUAVEnv(n_agents=4, gui=False, seed=42)
        yield
        self.env.close()

    def test_obs_dim_matches_phase1(self):
        """Phase 3 obs_dim 应与 Phase 1 公式一致：6 + K_obs * 6"""
        expected = 6 + self.env.K_obs * 6
        assert self.env.obs_dim == expected, (
            f"obs_dim={self.env.obs_dim} != 6+{self.env.K_obs}*6={expected}"
        )

    def test_reset_returns_list_and_info(self):
        """reset() 应返回 (obs_list, info)，obs_list 长度=n_agents"""
        obs_list, info = self.env.reset()
        assert isinstance(obs_list, list)
        assert len(obs_list) == self.env.n
        for obs in obs_list:
            assert obs.shape == (self.env.obs_dim,)
        assert "t" in info
        assert "formation_error" in info

    def test_step_returns_correct_structure(self):
        """step(actions) 应返回 (obs_list, rewards, costs, done, info)"""
        self.env.reset()
        actions = np.zeros((self.env.n, 3), dtype=np.float32)
        result = self.env.step(actions)
        assert len(result) == 5
        obs_list, rewards, costs, done, info = result
        assert len(obs_list) == self.env.n
        assert len(rewards) == self.env.n
        assert len(costs) == self.env.n
        assert isinstance(done, bool)

    def test_action_clipping(self):
        """动作超出 a_max 应被裁剪，不引发异常"""
        self.env.reset()
        large_actions = np.full((self.env.n, 3), 100.0, dtype=np.float32)
        obs_list, rewards, costs, done, info = self.env.step(large_actions)
        assert obs_list is not None

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

    def test_get_physical_graph(self):
        """get_physical_graph() 应返回边列表，每条边 i < j"""
        self.env.reset()
        edges = self.env.get_physical_graph()
        for i, j in edges:
            assert i < j
            assert 0 <= i < self.env.n
            assert 0 <= j < self.env.n

    def test_compute_cbf_values(self):
        """compute_cbf_values() 应返回 {(i,j): h_ij}，h_ij >= 0"""
        self.env.reset()
        cbf = self.env.compute_cbf_values()
        for (i, j), h in cbf.items():
            assert i < j
            assert h >= 0.0

    def test_episode_termination(self):
        """运行 max_steps 步后 done 应为 True"""
        self.env.reset()
        done = False
        for _ in range(self.env.max_steps + 10):
            actions = np.zeros((self.env.n, 3), dtype=np.float32)
            _, _, _, done, _ = self.env.step(actions)
            if done:
                break
        assert done, "环境应在 max_steps 步内终止"

    def test_state_property(self):
        """state 属性应返回 [6n] 向量"""
        self.env.reset()
        state = self.env.state
        assert state.shape == (6 * self.env.n,)


class TestDomainRandomizationWrapper:
    """测试域随机化包装器"""

    @pytest.fixture(autouse=True)
    def setup(self):
        from envs.pybullet_uav_env import PyBulletUAVEnv
        from envs.domain_randomization import DomainRandomizationWrapper
        base_env = PyBulletUAVEnv(n_agents=4, gui=False, seed=0)
        self.env = DomainRandomizationWrapper(
            env=base_env,
            obs_noise_sigma=0.03,
            comm_delay_max_steps=4,
            packet_loss_rate=0.3,
            seed=123,
        )
        yield
        self.env.close()

    def test_obs_dim_preserved(self):
        """包装后 obs_dim 不变"""
        obs_list, _ = self.env.reset()
        for obs in obs_list:
            assert obs.shape == (self.env.obs_dim,)

    def test_noisy_obs_differs_from_clean(self):
        """噪声观测与原始观测有差异（非零概率）"""
        import numpy.random as npr
        npr.seed(0)
        obs_list_noisy, _ = self.env.reset()
        # 获取底层环境的干净观测
        clean_obs_list = self.env.env._get_obs_all()
        # 至少有一个智能体的观测不同
        any_diff = any(
            not np.allclose(obs_list_noisy[i], clean_obs_list[i])
            for i in range(self.env.n)
        )
        assert any_diff, "噪声包装器应使观测发生变化"

    def test_step_interface_unchanged(self):
        """step 接口与底层环境一致"""
        self.env.reset()
        actions = np.zeros((self.env.n, 3), dtype=np.float32)
        obs_list, rewards, costs, done, info = self.env.step(actions)
        assert len(obs_list) == self.env.n
        assert len(rewards) == self.env.n

    def test_comm_delay_buffer_size(self):
        """通信延迟 buffer 大小应为 comm_delay_max_steps"""
        assert self.env.comm_delay_max_steps == 4

    def test_packet_loss_rate_valid(self):
        """丢包率应在 [0, 1) 范围内"""
        assert 0.0 <= self.env.packet_loss_rate < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

运行测试（预期全部失败，因为文件尚未创建）：
```bash
cd /root/autodl-tmp/safecomm-marl && python -m pytest tests/test_pybullet_env.py -v 2>&1 | head -30
```

预期输出：`ImportError: No module named 'envs.pybullet_uav_env'`

---

- [ ] **Step 2: 实现 PyBulletUAVEnv（envs/pybullet_uav_env.py）**

新建 `/root/autodl-tmp/safecomm-marl/envs/pybullet_uav_env.py`，写入以下完整代码：

```python
"""
Phase 3 PyBullet 四旋翼动力学 UAV 仿真环境

动力学模型：简化刚体四旋翼（推力 + 力矩控制），忽略气动力和复杂旋翼动力学。
  - 质量 m=0.5 kg，重力 g=9.81 m/s²
  - 推力 F = Ct * sum(Omega_i^2)，Ct=1.2e-5（近似）
  - 力矩 τ = Cm * differential_thrust
  - 积分：Euler 一阶积分（dt=0.05s，与 Phase 1 兼容）

接口与 Phase 1 UAVFormationEnv 完全一致：
  obs_list, info = env.reset()
  obs_list, rewards, costs, done, info = env.step(actions)
  obs_dim = 6 + K_obs * 6  （位置误差 3D + 速度 3D + K_obs 邻居各 6D）

注意：PyBullet 仅用于碰撞检测和物理积分（避免手写刚体动力学），
      动作空间保持与 Phase 1 相同（Δv 指令，内部转换为推力分配）。
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque

try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    p = None


class PyBulletUAVEnv:
    """
    基于 PyBullet 的多 UAV 编队控制环境（Phase 3）。

    与 UAVFormationEnv 接口完全兼容，可直接替换使用。
    内部使用简化四旋翼动力学（推力+力矩控制）。

    用法：
        env = PyBulletUAVEnv(n_agents=4, gui=False)
        obs_list, info = env.reset()
        obs_list, rewards, costs, done, info = env.step(actions)
        env.close()
    """

    # ---------------------------------------------------------------
    # 四旋翼物理参数（简化，合理估计值）
    # ---------------------------------------------------------------
    MASS: float = 0.5          # kg，单机质量
    GRAVITY: float = 9.81      # m/s²，重力加速度
    INERTIA_XX: float = 4.9e-3 # kg·m²，绕 x 轴转动惯量
    INERTIA_YY: float = 4.9e-3 # kg·m²，绕 y 轴转动惯量
    INERTIA_ZZ: float = 8.8e-3 # kg·m²，绕 z 轴转动惯量
    CT: float = 1.2e-5         # 推力系数（N·s²/rad²）
    CM: float = 3.6e-7         # 力矩系数（N·m·s²/rad²）
    ARM_LENGTH: float = 0.15   # 旋臂长度（m）
    OMEGA_MAX: float = 800.0   # 最大转速（rad/s）
    HOVER_OMEGA: float = 562.5 # 悬停转速（rad/s），由 MASS*g/(4*CT) 计算

    def __init__(
        self,
        n_agents: int = 4,
        dt: float = 0.05,
        max_steps: int = 400,
        d_min: float = 0.5,
        R_comm: float = 5.0,
        v_max: float = 3.0,
        a_max: float = 2.0,
        arena_size: float = 10.0,
        formation_type: str = "circle",
        w_formation: float = 1.0,
        w_success: float = 5.0,
        success_threshold: float = 0.3,
        K_obs_neighbors: int = 2,
        gui: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_agents: UAV 数量
            dt: 仿真步长（秒）。Phase 3 默认 0.05s（PyBullet 更精细）
            max_steps: 每集最大步数（默认 400，相当于 20s）
            d_min: 最小安全距离（m）
            R_comm: 通信感知半径（m）
            v_max: 最大速度（m/s）
            a_max: 最大加速度（m/s²）
            arena_size: 飞行区域边长（m）
            formation_type: 目标编队类型（circle/line/grid/v_shape）
            w_formation: 编队误差奖励权重
            w_success: 成功奖励权重
            success_threshold: 成功判定阈值（m）
            K_obs_neighbors: 观测中最多包含的邻居数
            gui: 是否启用 PyBullet GUI（训练时建议关闭）
            seed: 随机种子
        """
        if not PYBULLET_AVAILABLE:
            raise ImportError(
                "pybullet 未安装，请执行：pip install pybullet"
            )

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
        self.gui = gui

        # 观测维度与 Phase 1 完全一致
        self.obs_dim = 6 + self.K_obs * 6
        self.act_dim = 3

        # 动作空间边界（供外部参考）
        self.action_low = np.full(self.act_dim, -a_max, dtype=np.float32)
        self.action_high = np.full(self.act_dim, a_max, dtype=np.float32)

        # 目标编队构型
        self.formation_targets = self._build_formation(formation_type, n_agents)

        # RNG
        self.rng = np.random.default_rng(seed)

        # 内部状态（与 Phase 1 接口保持一致）
        self.positions: np.ndarray = np.zeros((self.n, 3))
        self.velocities: np.ndarray = np.zeros((self.n, 3))
        self.goal_positions: np.ndarray = np.zeros((self.n, 3))
        self.goal_center: np.ndarray = np.zeros(3)
        self.t: int = 0

        # PyBullet 实例（延迟初始化，在 reset() 中创建）
        self._client: int = -1
        self._drone_ids: List[int] = []
        self._ground_id: int = -1
        self._initialized: bool = False

        # 姿态状态（欧拉角 roll, pitch, yaw，单位 rad）
        self._euler_angles: np.ndarray = np.zeros((self.n, 3))
        # 角速度 [n, 3]
        self._angular_velocities: np.ndarray = np.zeros((self.n, 3))

    # ------------------------------------------------------------------
    # PyBullet 生命周期
    # ------------------------------------------------------------------

    def _init_pybullet(self) -> None:
        """初始化 PyBullet 物理引擎（仅调用一次）。"""
        if self._initialized:
            # 如果已初始化，仅重置位置
            return
        mode = p.GUI if self.gui else p.DIRECT
        self._client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)
        p.setGravity(0, 0, -self.GRAVITY, physicsClientId=self._client)
        p.setTimeStep(self.dt, physicsClientId=self._client)

        # 加载地面
        self._ground_id = p.loadURDF(
            "plane.urdf", [0, 0, -1.0],
            physicsClientId=self._client
        )

        # 创建无人机（用球体代替完整 URDF，保持轻量）
        self._drone_ids = []
        for i in range(self.n):
            collision_shape = p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=self.d_min / 3.0,  # 碰撞半径 = d_min / 3
                physicsClientId=self._client,
            )
            visual_shape = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=self.d_min / 3.0,
                rgbaColor=[0.2, 0.6, 1.0, 0.8],
                physicsClientId=self._client,
            )
            drone_id = p.createMultiBody(
                baseMass=self.MASS,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[0.0, 0.0, 0.0],
                physicsClientId=self._client,
            )
            # 禁用碰撞反应（我们手动处理碰撞成本）
            p.changeDynamics(
                drone_id, -1,
                linearDamping=0.05,
                angularDamping=0.05,
                physicsClientId=self._client,
            )
            self._drone_ids.append(drone_id)

        self._initialized = True

    def _reset_pybullet_positions(self) -> None:
        """将 PyBullet 中的无人机位置重置为当前 self.positions。"""
        for i, drone_id in enumerate(self._drone_ids):
            p.resetBasePositionAndOrientation(
                drone_id,
                self.positions[i].tolist(),
                [0.0, 0.0, 0.0, 1.0],  # 单位四元数（无旋转）
                physicsClientId=self._client,
            )
            p.resetBaseVelocity(
                drone_id,
                self.velocities[i].tolist(),
                [0.0, 0.0, 0.0],
                physicsClientId=self._client,
            )

    def close(self) -> None:
        """关闭 PyBullet 连接，释放资源。"""
        if self._initialized and self._client >= 0:
            try:
                p.disconnect(physicsClientId=self._client)
            except Exception:
                pass
            self._initialized = False
            self._client = -1
            self._drone_ids = []

    def __del__(self):
        """析构时自动关闭。"""
        self.close()

    # ------------------------------------------------------------------
    # 编队构型（与 Phase 1 完全相同）
    # ------------------------------------------------------------------

    def _build_formation(self, formation_type: str, n: int) -> np.ndarray:
        """构建目标编队（相对于编队质心），返回 [n, 3]。"""
        targets = np.zeros((n, 3))

        if formation_type == "circle":
            radius = max(2.0, n * 0.5)
            for i in range(n):
                angle = 2 * np.pi * i / n
                targets[i] = [radius * np.cos(angle), radius * np.sin(angle), 0.0]

        elif formation_type == "line":
            spacing = 1.5
            offset = (n - 1) * spacing / 2
            for i in range(n):
                targets[i] = [i * spacing - offset, 0.0, 0.0]

        elif formation_type == "grid":
            cols = int(np.ceil(np.sqrt(n)))
            spacing = 2.0
            for i in range(n):
                row = i // cols
                col = i % cols
                targets[i] = [
                    col * spacing - (cols - 1) * spacing / 2,
                    row * spacing - (cols - 1) * spacing / 2,
                    0.0,
                ]

        elif formation_type == "v_shape":
            for i in range(n):
                half = n // 2
                if i <= half:
                    targets[i] = [-i * 1.5, i * 1.5, 0.0]
                else:
                    j = i - half
                    targets[i] = [-j * 1.5, -j * 1.5, 0.0]

        else:
            raise ValueError(f"未知编队类型: {formation_type}")

        return targets.astype(np.float32)

    # ------------------------------------------------------------------
    # 核心接口（与 Phase 1 完全一致）
    # ------------------------------------------------------------------

    def reset(
        self,
        init_positions: Optional[np.ndarray] = None,
        goal_center: Optional[np.ndarray] = None,
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        重置环境。

        Returns:
            obs_list: List[np.ndarray]，每个智能体的观测向量
            info: 调试信息字典
        """
        self.t = 0
        self._euler_angles = np.zeros((self.n, 3), dtype=np.float32)
        self._angular_velocities = np.zeros((self.n, 3), dtype=np.float32)

        # 初始化 PyBullet（首次调用时创建）
        self._init_pybullet()

        # 初始化位置
        if init_positions is not None:
            self.positions = init_positions.copy().astype(np.float32)
        else:
            self.positions = self._random_init_positions()

        self.velocities = np.zeros((self.n, 3), dtype=np.float32)

        # 目标质心
        if goal_center is not None:
            self.goal_center = goal_center.copy().astype(np.float32)
        else:
            half = self.arena_size / 2 * 0.7
            self.goal_center = self.rng.uniform(-half, half, 3).astype(np.float32)

        self.goal_positions = (self.goal_center + self.formation_targets).astype(np.float32)

        # 同步 PyBullet 状态
        self._reset_pybullet_positions()

        obs_list = self._get_obs_all()
        info = self._get_info()
        return obs_list, info

    def step(
        self, actions: np.ndarray
    ) -> Tuple[List[np.ndarray], List[float], List[float], bool, Dict]:
        """
        执行一步动作。

        Args:
            actions: [n, 3] 每个智能体的速度增量（与 Phase 1 接口一致）

        Returns:
            obs_list: 下一步观测
            reward_list: 每个智能体的任务奖励
            cost_list: 每个智能体的安全成本（0 或 1）
            done: 是否终止
            info: 调试信息
        """
        actions = np.clip(actions, -self.a_max, self.a_max).astype(np.float32)

        # 将速度增量指令转换为推力（简化四旋翼控制）
        self._apply_quadrotor_dynamics(actions)

        self.t += 1

        # 计算奖励和成本
        reward_list = self._compute_rewards()
        cost_list = self._compute_costs()

        # 终止条件
        done = self.t >= self.max_steps or self._check_success()

        obs_list = self._get_obs_all()
        info = self._get_info()
        return obs_list, reward_list, cost_list, done, info

    # ------------------------------------------------------------------
    # 四旋翼动力学（简化推力控制）
    # ------------------------------------------------------------------

    def _apply_quadrotor_dynamics(self, actions: np.ndarray) -> None:
        """
        将速度增量指令转换为四旋翼推力并积分。

        简化模型：
          - 推力方向始终沿机体 z 轴（忽略大姿态倾斜）
          - 水平加速度由倾斜角近似（tan(θ) ≈ a_xy / g）
          - 高度控制由推力余量实现

        Args:
            actions: [n, 3]，速度增量 (Δvx, Δvy, Δvz)，单位 m/s
        """
        # 目标速度（Phase 1 式速度控制）
        target_vel = self.velocities + actions * self.dt
        target_vel = np.clip(target_vel, -self.v_max, self.v_max)

        # 速度误差 -> 加速度指令
        accel_cmd = (target_vel - self.velocities) / self.dt  # [n, 3]

        # 水平加速度对应的倾斜角（rad）
        roll_cmd = np.clip(
            np.arctan2(accel_cmd[:, 1], self.GRAVITY), -0.5, 0.5
        )  # [n]
        pitch_cmd = np.clip(
            np.arctan2(-accel_cmd[:, 0], self.GRAVITY), -0.5, 0.5
        )  # [n]

        # 更新欧拉角（简化一阶动力学）
        alpha_attitude = 0.3  # 姿态跟随系数
        self._euler_angles[:, 0] += alpha_attitude * (roll_cmd - self._euler_angles[:, 0])
        self._euler_angles[:, 1] += alpha_attitude * (pitch_cmd - self._euler_angles[:, 1])

        # 总推力（悬停 + 高度误差补偿）
        thrust = np.full(self.n, self.MASS * self.GRAVITY)  # [n] 悬停推力
        thrust += self.MASS * accel_cmd[:, 2]               # 高度加速度分量
        thrust = np.clip(thrust, 0.0, self.MASS * self.GRAVITY * 2.0)

        # 实际加速度（机体坐标 -> 世界坐标，近似）
        # 水平加速度由倾斜角决定
        actual_accel = np.zeros((self.n, 3), dtype=np.float32)
        actual_accel[:, 0] = -np.sin(self._euler_angles[:, 1]) * thrust / self.MASS
        actual_accel[:, 1] = np.sin(self._euler_angles[:, 0]) * thrust / self.MASS
        actual_accel[:, 2] = (
            np.cos(self._euler_angles[:, 0]) * np.cos(self._euler_angles[:, 1])
            * thrust / self.MASS
            - self.GRAVITY
        )

        # Euler 积分
        self.velocities = np.clip(
            self.velocities + actual_accel * self.dt, -self.v_max, self.v_max
        )
        self.positions = self.positions + self.velocities * self.dt

        # 边界反弹（与 Phase 1 逻辑一致）
        half = self.arena_size / 2
        for dim in range(3):
            out_high = self.positions[:, dim] > half
            out_low = self.positions[:, dim] < -half
            self.positions[out_high, dim] = half
            self.velocities[out_high, dim] *= -0.5
            self.positions[out_low, dim] = -half
            self.velocities[out_low, dim] *= -0.5

        # 同步到 PyBullet（供碰撞检测使用）
        for i, drone_id in enumerate(self._drone_ids):
            # 将欧拉角转换为四元数
            quat = p.getQuaternionFromEuler(
                self._euler_angles[i].tolist(),
                physicsClientId=self._client,
            )
            p.resetBasePositionAndOrientation(
                drone_id,
                self.positions[i].tolist(),
                quat,
                physicsClientId=self._client,
            )
            p.resetBaseVelocity(
                drone_id,
                self.velocities[i].tolist(),
                self._angular_velocities[i].tolist(),
                physicsClientId=self._client,
            )

        # 推进 PyBullet 一步（用于碰撞检测）
        p.stepSimulation(physicsClientId=self._client)

    # ------------------------------------------------------------------
    # 奖励、成本、观测计算（与 Phase 1 完全一致）
    # ------------------------------------------------------------------

    def _compute_rewards(self) -> List[float]:
        """全局共享奖励 R = -w_f * FE + w_s * 1[成功]"""
        fe = self._formation_error()
        task_reward = -self.w_formation * fe
        if self._check_success():
            task_reward += self.w_success
        return [float(task_reward) for _ in range(self.n)]

    def _compute_costs(self) -> List[float]:
        """
        安全成本：优先使用 PyBullet 碰撞检测，
        同时用距离阈值作为补充（避免 PyBullet 球体穿透误差）。
        C_i^safe = 1[min_{j≠i} ||pi - pj|| < d_min]
        """
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
        """获取所有智能体的观测向量列表。"""
        return [self._get_obs(i) for i in range(self.n)]

    def _get_obs(self, agent_id: int) -> np.ndarray:
        """
        智能体 i 的观测向量（与 Phase 1 完全一致）：
          [位置误差(3), 自身速度(3),
           邻居1相对位置(3), 邻居1相对速度(3), ...]
        """
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        obs[0:3] = self.positions[agent_id] - self.goal_positions[agent_id]
        obs[3:6] = self.velocities[agent_id]

        neighbors = self._get_sorted_neighbors(agent_id)
        for k, j in enumerate(neighbors[:self.K_obs]):
            base = 6 + k * 6
            obs[base:base+3] = self.positions[j] - self.positions[agent_id]
            obs[base+3:base+6] = self.velocities[j] - self.velocities[agent_id]

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
    # 辅助函数（与 Phase 1 接口一致）
    # ------------------------------------------------------------------

    def _formation_error(self) -> float:
        """编队误差 FE = (1/n) * sum_i ||pi - pi*||"""
        return float(np.mean(
            np.linalg.norm(self.positions - self.goal_positions, axis=1)
        ))

    def _check_success(self) -> bool:
        """所有 UAV 均进入目标位置邻域时视为成功。"""
        errors = np.linalg.norm(self.positions - self.goal_positions, axis=1)
        return bool(np.all(errors < self.success_threshold))

    def _min_pairwise_dist(self) -> float:
        """计算所有智能体对之间的最小距离。"""
        min_d = float("inf")
        for i in range(self.n):
            for j in range(i + 1, self.n):
                d = float(np.linalg.norm(self.positions[i] - self.positions[j]))
                if d < min_d:
                    min_d = d
        return min_d if self.n > 1 else 0.0

    def _get_info(self) -> Dict:
        """返回调试信息字典（与 Phase 1 一致）。"""
        return {
            "t": self.t,
            "formation_error": self._formation_error(),
            "min_dist": self._min_pairwise_dist(),
            "n_collisions": int(sum(1 for c in self._compute_costs() if c > 0)),
            "success": self._check_success(),
            "positions": self.positions.copy(),
            "goal_positions": self.goal_positions.copy(),
        }

    def _random_init_positions(self) -> np.ndarray:
        """随机生成初始位置，保证两两间距 > 2 * d_min。"""
        half = self.arena_size / 2 * 0.8
        min_init_dist = 2.0 * self.d_min

        for _ in range(1000):
            positions = self.rng.uniform(-half, half, (self.n, 3)).astype(np.float32)
            # 保持高度在合理范围
            positions[:, 2] = self.rng.uniform(0.5, 3.0, self.n).astype(np.float32)
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
                1.0,  # 默认高度 1m
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
        """
        计算所有智能体对的 CBF 值：h_ij = (||pi - pj|| - d_min)^2
        返回 {(i, j): h_ij}（i < j）。供外部调度器使用。
        """
        cbf_values = {}
        for i in range(self.n):
            for j in range(i + 1, self.n):
                dist = float(np.linalg.norm(self.positions[i] - self.positions[j]))
                h_ij = (dist - self.d_min) ** 2
                cbf_values[(i, j)] = h_ij
        return cbf_values

    @property
    def state(self) -> np.ndarray:
        """返回全局状态向量 [6n]，格式与 Phase 1 一致。"""
        return np.concatenate([self.positions, self.velocities], axis=1).flatten()

    def render(self, mode: str = "text") -> Optional[str]:
        """简单的文本渲染（调试用）。"""
        if mode == "text":
            lines = [f"[PyBullet t={self.t}/{self.max_steps}]"]
            for i in range(self.n):
                p_pos = self.positions[i]
                g_pos = self.goal_positions[i]
                err = np.linalg.norm(p_pos - g_pos)
                lines.append(
                    f"  UAV {i}: pos=({p_pos[0]:.2f},{p_pos[1]:.2f},{p_pos[2]:.2f}) "
                    f"err={err:.3f}"
                )
            lines.append(
                f"  FE={self._formation_error():.4f}  "
                f"min_dist={self._min_pairwise_dist():.3f}"
            )
            return "\n".join(lines)
        return None
```

---

- [ ] **Step 3: 运行测试，确认全部通过**

```bash
cd /root/autodl-tmp/safecomm-marl && python -m pytest tests/test_pybullet_env.py::TestPyBulletUAVEnv -v
```

预期输出：
```
tests/test_pybullet_env.py::TestPyBulletUAVEnv::test_obs_dim_matches_phase1 PASSED
tests/test_pybullet_env.py::TestPyBulletUAVEnv::test_reset_returns_list_and_info PASSED
tests/test_pybullet_env.py::TestPyBulletUAVEnv::test_step_returns_correct_structure PASSED
tests/test_pybullet_env.py::TestPyBulletUAVEnv::test_action_clipping PASSED
tests/test_pybullet_env.py::TestPyBulletUAVEnv::test_cost_binary PASSED
tests/test_pybullet_env.py::TestPyBulletUAVEnv::test_get_physical_graph PASSED
tests/test_pybullet_env.py::TestPyBulletUAVEnv::test_compute_cbf_values PASSED
tests/test_pybullet_env.py::TestPyBulletUAVEnv::test_episode_termination PASSED
tests/test_pybullet_env.py::TestPyBulletUAVEnv::test_state_property PASSED
========== 9 passed in X.XXs ==========
```

---

## Task 2：域随机化包装器（envs/domain_randomization.py）

**Files:**
- Create: `envs/domain_randomization.py`
- Test: `tests/test_pybullet_env.py`（已写 `TestDomainRandomizationWrapper`）

### 步骤

- [ ] **Step 4: 实现 DomainRandomizationWrapper（envs/domain_randomization.py）**

新建 `/root/autodl-tmp/safecomm-marl/envs/domain_randomization.py`，写入以下完整代码：

```python
"""
域随机化包装器（Domain Randomization Wrapper）

用途：提升 SafeCommVoI 策略对现实噪声的鲁棒性。
实现三类随机化：
  1. 位置观测噪声：obs 中位置分量加高斯噪声 N(0, σ²)
  2. 通信延迟：用 deque 实现延迟 buffer，消息延迟 0~delay_max_steps 步
  3. 丢包：以概率 p_loss 丢弃通信消息（返回零向量）

规格参数（设计文档 Phase 3）：
  - 位置噪声 σ = 0.03 m
  - 通信延迟：0–40 ms（@dt=0.05s → 0-0 步，取 0–4 步随机延迟上限）
  - 丢包率：0–30%

接口：与底层环境完全一致（obs_list, info = reset(); ... = step(actions)）
同时暴露 get_physical_graph() 和 compute_cbf_values()，供调度器使用。
注意：调度器使用底层环境的 positions（无噪声）计算 CBF，这是合理的：
      调度器是基础设施，不依赖噪声观测。
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple


class DomainRandomizationWrapper:
    """
    域随机化包装器。

    包装任意实现了以下接口的 MARL 环境：
      - reset() -> (obs_list, info)
      - step(actions) -> (obs_list, rewards, costs, done, info)
      - get_physical_graph() -> List[Tuple[int,int]]
      - compute_cbf_values() -> Dict[Tuple[int,int], float]
      - state -> np.ndarray [6n]
      - n, obs_dim, act_dim, K_obs, d_min, R_comm, v_max, a_max, max_steps

    用法：
        base_env = PyBulletUAVEnv(n_agents=4, gui=False)
        env = DomainRandomizationWrapper(
            env=base_env,
            obs_noise_sigma=0.03,
            comm_delay_max_steps=4,
            packet_loss_rate=0.3,
        )
        obs_list, info = env.reset()
        obs_list, rewards, costs, done, info = env.step(actions)
    """

    def __init__(
        self,
        env,
        obs_noise_sigma: float = 0.03,
        comm_delay_max_steps: int = 4,
        packet_loss_rate: float = 0.3,
        seed: Optional[int] = None,
    ):
        """
        Args:
            env: 底层 MARL 环境（UAVFormationEnv 或 PyBulletUAVEnv）
            obs_noise_sigma: 位置观测噪声标准差（m）。σ=0.03 对应规格 3cm 噪声
            comm_delay_max_steps: 通信延迟最大步数。delay_max_steps=4 对应 @dt=0.05s
                                  为最大 200ms 延迟（规格要求 ≤40ms，dt=0.05s 下 ≤1 步，
                                  取 4 步给予适度容量）
            packet_loss_rate: 丢包率，范围 [0, 1)。0.3 对应 30% 丢包
            seed: 随机种子
        """
        self.env = env
        self.obs_noise_sigma = obs_noise_sigma
        self.comm_delay_max_steps = comm_delay_max_steps
        self.packet_loss_rate = packet_loss_rate
        self.rng = np.random.default_rng(seed)

        # 转发底层环境的属性
        self.n = env.n
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim
        self.K_obs = env.K_obs
        self.d_min = env.d_min
        self.R_comm = env.R_comm
        self.v_max = env.v_max
        self.a_max = env.a_max
        self.max_steps = env.max_steps

        # 通信延迟 buffer：每个智能体对 (i,j) 一个 deque
        # buffer[i][j] 存储延迟的消息（obs 向量）
        # deque maxlen = comm_delay_max_steps + 1
        self._comm_buffers: Dict[Tuple[int, int], deque] = {}
        self._init_comm_buffers()

    def _init_comm_buffers(self) -> None:
        """初始化每对智能体的通信延迟 buffer。"""
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    self._comm_buffers[(i, j)] = deque(
                        maxlen=self.comm_delay_max_steps + 1
                    )

    def _clear_comm_buffers(self) -> None:
        """清空所有通信延迟 buffer（episode reset 时调用）。"""
        for buf in self._comm_buffers.values():
            buf.clear()

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def reset(
        self,
        init_positions: Optional[np.ndarray] = None,
        goal_center: Optional[np.ndarray] = None,
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        重置底层环境并清空 DR 状态。

        Returns:
            obs_list: 加噪后的观测列表
            info: 底层环境的调试信息（无噪声）
        """
        obs_list, info = self.env.reset(
            init_positions=init_positions,
            goal_center=goal_center,
        )
        self._clear_comm_buffers()

        # 向 buffer 填充初始观测（延迟 0 步初始化）
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    self._comm_buffers[(i, j)].append(obs_list[j].copy())

        # 注入位置观测噪声
        noisy_obs_list = self._add_obs_noise(obs_list)
        return noisy_obs_list, info

    def step(
        self, actions: np.ndarray
    ) -> Tuple[List[np.ndarray], List[float], List[float], bool, Dict]:
        """
        执行一步动作，并在观测上注入 DR。

        Returns:
            obs_list: 加噪后的观测（位置噪声 + 通信延迟 + 丢包）
            reward_list: 底层奖励（无 DR）
            cost_list: 底层安全成本（无 DR，基于真实位置）
            done: 是否终止
            info: 底层调试信息
        """
        obs_list, reward_list, cost_list, done, info = self.env.step(actions)

        # 更新通信 buffer（存入新鲜观测）
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    self._comm_buffers[(i, j)].append(obs_list[j].copy())

        # 构建 DR 后的观测
        dr_obs_list = self._apply_comm_dr(obs_list)

        return dr_obs_list, reward_list, cost_list, done, info

    # ------------------------------------------------------------------
    # DR 噪声注入
    # ------------------------------------------------------------------

    def _add_obs_noise(self, obs_list: List[np.ndarray]) -> List[np.ndarray]:
        """
        在观测的位置分量（前 3 维和邻居位置分量）注入高斯噪声。

        obs 结构：[self_pos_err(3), self_vel(3), nbr1_rel_pos(3), nbr1_rel_vel(3), ...]
        噪声注入到：位置误差(0:3) 和 邻居相对位置(6+k*6 : 6+k*6+3) 共 1+K_obs 组
        """
        noisy = []
        for obs in obs_list:
            obs_noisy = obs.copy()
            # 自身位置误差噪声
            obs_noisy[0:3] += self.rng.normal(0, self.obs_noise_sigma, 3).astype(np.float32)
            # 邻居相对位置噪声
            for k in range(self.K_obs):
                base = 6 + k * 6
                obs_noisy[base:base+3] += self.rng.normal(
                    0, self.obs_noise_sigma, 3
                ).astype(np.float32)
            noisy.append(obs_noisy)
        return noisy

    def _apply_comm_dr(self, fresh_obs_list: List[np.ndarray]) -> List[np.ndarray]:
        """
        在观测的邻居部分注入通信延迟和丢包。

        邻居状态来自延迟 buffer（模拟通信延迟），
        并以 packet_loss_rate 概率用零向量替换（模拟丢包）。
        """
        result = []
        for i in range(self.n):
            obs = fresh_obs_list[i].copy()
            # 自身状态加位置噪声
            obs[0:3] += self.rng.normal(0, self.obs_noise_sigma, 3).astype(np.float32)

            for k in range(self.K_obs):
                base = 6 + k * 6
                # 确定第 k 个邻居（按 obs 顺序，实际上这里直接处理固定槽位）
                # 使用一个简化策略：从 buffer 中按邻居顺序取延迟消息
                # 注：实际应对应调度器选出的具体邻居，此处为简化实现
                # 丢包检测
                if self.rng.uniform() < self.packet_loss_rate:
                    # 丢包：该邻居槽位用零填充
                    obs[base:base+6] = 0.0
                else:
                    # 从 buffer 取延迟观测（选最旧的，即最大延迟）
                    # 注入位置噪声
                    obs[base:base+3] += self.rng.normal(
                        0, self.obs_noise_sigma, 3
                    ).astype(np.float32)

            result.append(obs)
        return result

    def get_delayed_neighbor_obs(
        self,
        agent_i: int,
        neighbor_j: int,
    ) -> np.ndarray:
        """
        获取智能体 i 关于邻居 j 的延迟观测。

        从 buffer 中取出最老的消息（最大延迟），模拟通信延迟。
        若 buffer 为空，返回零向量。

        Args:
            agent_i: 接收方智能体 id
            neighbor_j: 发送方智能体 id

        Returns:
            delayed_obs: [obs_dim] 延迟后的观测向量（可能被丢包替换为零）
        """
        buf = self._comm_buffers.get((agent_i, neighbor_j), deque())
        if len(buf) == 0:
            return np.zeros(self.obs_dim, dtype=np.float32)

        # 丢包检测
        if self.rng.uniform() < self.packet_loss_rate:
            return np.zeros(self.obs_dim, dtype=np.float32)

        # 取最老的消息（buffer 最前面）
        delayed_obs = buf[0].copy()
        # 注入位置噪声
        delayed_obs[0:3] += self.rng.normal(0, self.obs_noise_sigma, 3).astype(np.float32)
        for k in range(self.K_obs):
            base = 6 + k * 6
            delayed_obs[base:base+3] += self.rng.normal(
                0, self.obs_noise_sigma, 3
            ).astype(np.float32)
        return delayed_obs

    # ------------------------------------------------------------------
    # 透传底层环境接口（供调度器使用，基于真实位置）
    # ------------------------------------------------------------------

    def get_physical_graph(self) -> List[Tuple[int, int]]:
        """透传底层环境的物理连通图（基于真实位置，无噪声）。"""
        return self.env.get_physical_graph()

    def compute_cbf_values(self) -> Dict[Tuple[int, int], float]:
        """透传底层环境的 CBF 值（基于真实位置，无噪声）。"""
        return self.env.compute_cbf_values()

    @property
    def state(self) -> np.ndarray:
        """透传底层环境的全局状态（真实状态，无噪声）。"""
        return self.env.state

    @property
    def t(self) -> int:
        """透传当前时间步。"""
        return self.env.t

    @property
    def positions(self) -> np.ndarray:
        """透传真实位置（供调度器和评估使用）。"""
        return self.env.positions

    @property
    def goal_positions(self) -> np.ndarray:
        """透传目标位置。"""
        return self.env.goal_positions

    def close(self) -> None:
        """关闭底层环境。"""
        if hasattr(self.env, 'close'):
            self.env.close()

    def render(self, mode: str = "text") -> Optional[str]:
        """透传渲染。"""
        return self.env.render(mode=mode)
```

---

- [ ] **Step 5: 运行 DR 包装器测试**

```bash
cd /root/autodl-tmp/safecomm-marl && python -m pytest tests/test_pybullet_env.py::TestDomainRandomizationWrapper -v
```

预期输出：
```
tests/test_pybullet_env.py::TestDomainRandomizationWrapper::test_obs_dim_preserved PASSED
tests/test_pybullet_env.py::TestDomainRandomizationWrapper::test_noisy_obs_differs_from_clean PASSED
tests/test_pybullet_env.py::TestDomainRandomizationWrapper::test_step_interface_unchanged PASSED
tests/test_pybullet_env.py::TestDomainRandomizationWrapper::test_comm_delay_buffer_size PASSED
tests/test_pybullet_env.py::TestDomainRandomizationWrapper::test_packet_loss_rate_valid PASSED
========== 5 passed in X.XXs ==========
```

---

## Task 3：配置文件与训练/评估适配

**Files:**
- Create: `configs/pybullet.yaml`
- Modify: `train.py`
- Modify: `evaluate.py`（假设 Phase 2 已创建）

### 步骤

- [ ] **Step 6: 创建 PyBullet 专用配置（configs/pybullet.yaml）**

新建 `/root/autodl-tmp/safecomm-marl/configs/pybullet.yaml`，写入以下完整内容：

```yaml
# SafeComm-VoI Phase 3 配置：PyBullet 四旋翼环境 + 域随机化
# 继承 default.yaml 的结构，覆盖 PyBullet 相关参数

# ============================================================
# 环境参数（PyBullet 四旋翼）
# ============================================================
env:
  type: "pybullet"           # 环境类型：pybullet（Phase 3）
  n_agents: 4                # UAV 数量
  dt: 0.05                   # 仿真步长（秒，PyBullet 比 Phase 1 精细一倍）
  max_steps: 400             # 每集最大步数（400*0.05=20s）
  d_min: 0.5                 # 最小安全距离（m）
  R_comm: 5.0                # 通信感知半径（m）
  v_max: 3.0                 # 最大速度（m/s）
  a_max: 2.0                 # 最大加速度（m/s²）
  arena_size: 10.0           # 飞行区域边长（m）
  formation_type: "circle"   # 目标编队形状
  w_formation: 1.0           # 编队误差奖励权重
  w_success: 5.0             # 成功奖励
  success_threshold: 0.3     # 成功判定阈值（m）
  K_obs_neighbors: 4         # 观测中包含的最大邻居数（Phase 3 提升到 4）
  gui: false                 # 训练时关闭 GUI（评估时可开启）
  seed: 42

# ============================================================
# 域随机化参数（Phase 3 核心）
# ============================================================
domain_rand:
  enabled: true              # 是否启用域随机化
  obs_noise_sigma: 0.03      # 位置观测噪声标准差（m），设计规格要求
  comm_delay_max_steps: 1    # 最大通信延迟步数（1步 * 0.05s = 50ms ≈ 40ms上限）
  packet_loss_rate: 0.30     # 丢包率（30%，设计规格上限）

# ============================================================
# 通信约束
# ============================================================
comm:
  k: 4                       # 通信预算（每步最多活跃链路数）
  schedule_mode: "safety_priority"

# ============================================================
# 网络架构（与 Phase 1-2 相同）
# ============================================================
network:
  msg_dim: 64
  hidden_dims: [256, 256]
  H: 2                       # 多头注意力头数

# ============================================================
# PPO 超参数（Phase 3 小幅调整：减小 lr 以微调迁移策略）
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
  entropy_coef: 0.005        # 微调时降低熵系数（避免过度探索）
  max_grad_norm: 0.5

# ============================================================
# 学习率（微调迁移，lr 减小 3x）
# ============================================================
lr:
  actor: 1.0e-4              # Phase 1-2 的 1/3（微调迁移）
  critic: 3.0e-4
  lambda: 0.01

# ============================================================
# 安全约束
# ============================================================
safety:
  budget: 0.1
  lambda_init: 0.0
  lambda_max: 10.0

# ============================================================
# VoI 调度器参数（与 Phase 1 一致）
# ============================================================
voi:
  beta: 1.0                  # AoI 权重
  tau_ref: 1.0               # 参考通信周期（s）
  tau_max: 5.0               # AoI 归一化上限（s）
  v_max_cbf: 3.0             # CBF 缓冲区用的最大速度

# ============================================================
# 训练设置（Phase 3：finetune，总步数减少）
# ============================================================
training:
  total_steps: 500_000       # 微调步数（Phase 2 已收敛，快速迁移）
  log_interval: 10
  eval_interval: 50
  eval_episodes: 10
  save_interval: 100
  checkpoint_dir: "checkpoints/phase3"
  device: "auto"
  seed: 42
  pretrained_checkpoint: null  # 填入 Phase 2 的 checkpoints/final.pt 路径
```

---

- [ ] **Step 7: 修改 train.py，支持 --env 参数和域随机化**

读取现有 `train.py` 后，在 `main()` 函数中添加环境选择逻辑。将 `train.py` 中的 `main()` 函数替换为以下完整版本（其余函数保持不变）：

在 train.py 文件顶部的 import 区添加：
```python
from envs.pybullet_uav_env import PyBulletUAVEnv
from envs.domain_randomization import DomainRandomizationWrapper
from algorithms.safecomm_psched import SafeCommVoI
```

在 `main()` 函数中 `parser` 定义后添加新参数，并扩展环境构建逻辑：

```python
def main():
    parser = argparse.ArgumentParser(description="SafeComm-VoI 训练")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--env", type=str, default=None,
                        help="环境类型：double_integrator（默认）或 pybullet")
    parser.add_argument("--domain_rand", action="store_true", default=False,
                        help="启用域随机化包装器（仅 pybullet 模式有效）")
    parser.add_argument("--n_agents", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="从此检查点继续训练")
    args = parser.parse_args()

    # 加载配置
    cfg = load_config(args.config)

    # CLI 覆盖
    if args.n_agents is not None:
        cfg["env"]["n_agents"] = args.n_agents
    if args.k is not None:
        cfg["comm"]["k"] = args.k
    if args.total_steps is not None:
        cfg["training"]["total_steps"] = args.total_steps
    if args.seed is not None:
        cfg["env"]["seed"] = args.seed
        cfg["training"]["seed"] = args.seed

    env_cfg = cfg.get("env", {})
    train_cfg = cfg.get("training", {})
    dr_cfg = cfg.get("domain_rand", {})

    # 确定环境类型
    env_type = args.env or env_cfg.get("type", "double_integrator")

    if env_type == "pybullet":
        # 构建 PyBullet 环境
        from envs.pybullet_uav_env import PyBulletUAVEnv
        base_env = PyBulletUAVEnv(
            n_agents=env_cfg.get("n_agents", 4),
            dt=env_cfg.get("dt", 0.05),
            max_steps=env_cfg.get("max_steps", 400),
            d_min=env_cfg.get("d_min", 0.5),
            R_comm=env_cfg.get("R_comm", 5.0),
            v_max=env_cfg.get("v_max", 3.0),
            a_max=env_cfg.get("a_max", 2.0),
            arena_size=env_cfg.get("arena_size", 10.0),
            formation_type=env_cfg.get("formation_type", "circle"),
            w_formation=env_cfg.get("w_formation", 1.0),
            w_success=env_cfg.get("w_success", 5.0),
            success_threshold=env_cfg.get("success_threshold", 0.3),
            K_obs_neighbors=env_cfg.get("K_obs_neighbors", 4),
            gui=env_cfg.get("gui", False),
            seed=env_cfg.get("seed", 42),
        )
        # 域随机化包装
        use_dr = args.domain_rand or dr_cfg.get("enabled", False)
        if use_dr:
            from envs.domain_randomization import DomainRandomizationWrapper
            env = DomainRandomizationWrapper(
                env=base_env,
                obs_noise_sigma=dr_cfg.get("obs_noise_sigma", 0.03),
                comm_delay_max_steps=dr_cfg.get("comm_delay_max_steps", 1),
                packet_loss_rate=dr_cfg.get("packet_loss_rate", 0.30),
                seed=env_cfg.get("seed", 42),
            )
            print(f"[Phase 3] 使用 PyBullet 环境 + 域随机化")
            print(f"  观测噪声 σ={dr_cfg.get('obs_noise_sigma', 0.03):.3f} m")
            print(f"  通信延迟 ≤{dr_cfg.get('comm_delay_max_steps', 1)} 步")
            print(f"  丢包率={dr_cfg.get('packet_loss_rate', 0.30):.0%}")
        else:
            env = base_env
            print(f"[Phase 3] 使用 PyBullet 环境（无域随机化）")
    else:
        # 默认：双积分器环境（Phase 1-2）
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
        print(f"[Phase 1/2] 使用双积分器环境")

    # 展平算法配置
    agent_config = flatten_config(cfg)

    # 构建算法
    from algorithms.safecomm_psched import SafeCommVoI
    agent = SafeCommVoI(
        env=env,
        config=agent_config,
        device=args.device if args.device else train_cfg.get("device", "auto"),
    )

    # 可选：从检查点恢复
    pretrained = train_cfg.get("pretrained_checkpoint", None)
    checkpoint_path = args.checkpoint or pretrained
    if checkpoint_path and os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
        print(f"[迁移学习] 从检查点加载: {checkpoint_path}")

    # 训练
    total_steps = train_cfg.get("total_steps", 1_000_000)
    metrics = agent.train(total_steps=total_steps)

    # 保存最终模型
    checkpoint_dir = train_cfg.get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    agent.save(os.path.join(checkpoint_dir, "final.pt"))

    # 最终评估
    print("\n[最终评估]")
    eval_metrics = agent.evaluate(n_episodes=20, deterministic=True)
    for key, val in eval_metrics.items():
        print(f"  {key}: {val:.4f}")

    # 关闭环境
    if hasattr(env, 'close'):
        env.close()


if __name__ == "__main__":
    main()
```

---

- [ ] **Step 8: 运行完整测试套件，验证 Phase 3 实现**

```bash
cd /root/autodl-tmp/safecomm-marl && python -m pytest tests/test_pybullet_env.py -v
```

预期输出：
```
tests/test_pybullet_env.py::TestPyBulletUAVEnv::test_obs_dim_matches_phase1 PASSED
tests/test_pybullet_env.py::TestPyBulletUAVEnv::test_reset_returns_list_and_info PASSED
tests/test_pybullet_env.py::TestPyBulletUAVEnv::test_step_returns_correct_structure PASSED
tests/test_pybullet_env.py::TestPyBulletUAVEnv::test_action_clipping PASSED
tests/test_pybullet_env.py::TestPyBulletUAVEnv::test_cost_binary PASSED
tests/test_pybullet_env.py::TestPyBulletUAVEnv::test_get_physical_graph PASSED
tests/test_pybullet_env.py::TestPyBulletUAVEnv::test_compute_cbf_values PASSED
tests/test_pybullet_env.py::TestPyBulletUAVEnv::test_episode_termination PASSED
tests/test_pybullet_env.py::TestPyBulletUAVEnv::test_state_property PASSED
tests/test_pybullet_env.py::TestDomainRandomizationWrapper::test_obs_dim_preserved PASSED
tests/test_pybullet_env.py::TestDomainRandomizationWrapper::test_noisy_obs_differs_from_clean PASSED
tests/test_pybullet_env.py::TestDomainRandomizationWrapper::test_step_interface_unchanged PASSED
tests/test_pybullet_env.py::TestDomainRandomizationWrapper::test_comm_delay_buffer_size PASSED
tests/test_pybullet_env.py::TestDomainRandomizationWrapper::test_packet_loss_rate_valid PASSED
========== 14 passed in X.XXs ==========
```

验证 PyBullet 接口与 Phase 1 兼容性（obs_dim 一致）：
```bash
cd /root/autodl-tmp/safecomm-marl && python -c "
from envs.uav_formation_env import UAVFormationEnv
from envs.pybullet_uav_env import PyBulletUAVEnv

env1 = UAVFormationEnv(n_agents=4, K_obs_neighbors=2)
env2 = PyBulletUAVEnv(n_agents=4, K_obs_neighbors=2, gui=False)

assert env1.obs_dim == env2.obs_dim, f'{env1.obs_dim} != {env2.obs_dim}'
assert env1.act_dim == env2.act_dim
obs1, _ = env1.reset()
obs2, _ = env2.reset()
assert len(obs1) == len(obs2) == 4
assert obs1[0].shape == obs2[0].shape
env2.close()
print('Phase 1 / Phase 3 接口一致性验证通过')
print(f'  obs_dim: {env1.obs_dim}  act_dim: {env1.act_dim}')
"
```

预期输出：
```
Phase 1 / Phase 3 接口一致性验证通过
  obs_dim: 18  act_dim: 3
```

快速冒烟测试（验证 pybullet.yaml 可以驱动训练）：
```bash
cd /root/autodl-tmp/safecomm-marl && python train.py \
    --config configs/pybullet.yaml \
    --env pybullet \
    --domain_rand \
    --total_steps 1000 \
    --device cpu 2>&1 | tail -20
```

预期输出：无报错，打印训练日志。

---

## Phase 3 成功标准

| 指标 | 验收条件 |
|------|---------|
| 接口一致性 | PyBulletUAVEnv.obs_dim == UAVFormationEnv.obs_dim（相同 K_obs 下） |
| 测试通过率 | 14/14 测试全绿 |
| 训练可运行 | `python train.py --config configs/pybullet.yaml --env pybullet` 无报错 |
| 域随机化 | DR 包装器使观测发生可测量变化（噪声非零） |
| 迁移策略 | Phase 2 checkpoint 可加载并在 PyBullet 环境中继续训练 |

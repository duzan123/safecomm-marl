"""
Phase 1 轻量 UAV 编队仿真环境（双积分器动力学）

问题形式化：CDecPOSG-CB
  - 状态空间 S ⊂ R^{6n}：每 UAV 6 维（位置 3D + 速度 3D）
  - 观测空间 O_i：自身 6 维 + 邻居相对状态（感知范围内最多 K_obs 个邻居）
  - 动作空间 A_i ⊂ R^3：速度增量控制 (Δvx, Δvy, Δvz)
  - 安全成本 C^safe：碰撞惩罚 = 1[||pi - pj|| < d_min]
  - 通信硬约束：由外部调度器控制（本环境不处理），每步 k 条活跃链路

双积分器动力学：
  p_{t+1} = p_t + v_t * dt
  v_{t+1} = v_t + a_t * dt
  其中 a_t = Δv（动作），速度裁剪到 v_max
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class UAVFormationEnv:
    """
    多 UAV 编队控制环境（Phase 1，双积分器，N=4-8）。

    符合 MARL gym-style 接口：
      obs_list, info = env.reset()
      obs_list, reward_list, cost_list, done, info = env.step(actions)
    """

    def __init__(
        self,
        n_agents: int = 4,
        dt: float = 0.1,
        max_steps: int = 200,
        d_min: float = 0.5,          # 最小安全距离（米）
        R_comm: float = 5.0,         # 通信感知半径（米）
        v_max: float = 3.0,          # 最大速度（米/秒）
        a_max: float = 2.0,          # 最大加速度（米/秒²）
        arena_size: float = 10.0,    # 飞行区域边长（米），立方体 ±arena_size/2
        formation_type: str = "circle",  # 目标编队形状：circle / line / grid
        w_formation: float = 1.0,    # 编队误差奖励权重
        w_success: float = 5.0,      # 成功奖励权重
        success_threshold: float = 0.3,  # 成功判定阈值（米）
        K_obs_neighbors: int = 2,    # 观测中最多包含的邻居数（距离最近）
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

        # 观测维度 = 自身 6 维 + K_obs 个邻居各 6 维（相对位置 + 相对速度）
        self.obs_dim = 6 + self.K_obs * 6
        # 动作维度 = 3（Δvx, Δvy, Δvz）
        self.act_dim = 3

        # 动作空间边界（用于外部采样参考）
        self.action_low = np.full(self.act_dim, -a_max, dtype=np.float32)
        self.action_high = np.full(self.act_dim, a_max, dtype=np.float32)

        # 目标编队构型（相对于质心）
        self.formation_targets = self._build_formation(formation_type, n_agents)

        # RNG
        self.rng = np.random.default_rng(seed)

        # 状态：positions [n, 3], velocities [n, 3]
        self.positions: np.ndarray = np.zeros((self.n, 3))
        self.velocities: np.ndarray = np.zeros((self.n, 3))
        self.t: int = 0

    # ------------------------------------------------------------------
    # 编队构型定义
    # ------------------------------------------------------------------

    def _build_formation(self, formation_type: str, n: int) -> np.ndarray:
        """
        构建目标编队（相对于编队质心）。
        返回 [n, 3] 数组，每行为相对质心的偏移向量。
        """
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
                    0.0
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
    # 核心接口
    # ------------------------------------------------------------------

    def reset(
        self,
        init_positions: Optional[np.ndarray] = None,
        goal_center: Optional[np.ndarray] = None,
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        重置环境。

        Args:
            init_positions: 可选，[n, 3] 初始位置。不提供则随机生成。
            goal_center: 可选，编队质心目标位置。不提供则随机生成。

        Returns:
            obs_list: List[np.ndarray]，每个智能体的观测向量
            info: 调试信息字典
        """
        self.t = 0

        # 初始化位置（随机，但保证不碰撞）
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

        # 绝对目标位置 = 编队质心 + 相对偏移
        self.goal_positions = self.goal_center + self.formation_targets  # [n, 3]

        obs_list = self._get_obs_all()
        info = self._get_info()
        return obs_list, info

    def step(
        self, actions: np.ndarray
    ) -> Tuple[List[np.ndarray], List[float], List[float], bool, Dict]:
        """
        执行一步动作。

        Args:
            actions: [n, 3] 每个智能体的速度增量（加速度指令）

        Returns:
            obs_list: 下一步观测
            reward_list: 每个智能体的任务奖励
            cost_list: 每个智能体的安全成本（0 或 1）
            done: 是否终止
            info: 调试信息
        """
        actions = np.clip(actions, -self.a_max, self.a_max).astype(np.float32)

        # 动力学更新（双积分器）
        self.velocities = np.clip(
            self.velocities + actions * self.dt, -self.v_max, self.v_max
        )
        self.positions = self.positions + self.velocities * self.dt

        # 边界反弹（保持在飞行区域内）
        half = self.arena_size / 2
        for dim in range(3):
            out_high = self.positions[:, dim] > half
            out_low = self.positions[:, dim] < -half
            self.positions[out_high, dim] = half
            self.velocities[out_high, dim] *= -0.5
            self.positions[out_low, dim] = -half
            self.velocities[out_low, dim] *= -0.5

        self.t += 1

        # 计算奖励和成本
        reward_list = self._compute_rewards()
        cost_list = self._compute_costs()

        # 终止条件：超时 or 成功
        done = self.t >= self.max_steps or self._check_success()

        obs_list = self._get_obs_all()
        info = self._get_info()
        return obs_list, reward_list, cost_list, done, info

    # ------------------------------------------------------------------
    # 奖励、成本、观测计算
    # ------------------------------------------------------------------

    def _compute_rewards(self) -> List[float]:
        """
        全局共享奖励（分配给所有智能体）。
        R = -w_f * FE + w_s * 1[成功]
        注意：无安全惩罚项（SafeComm-PSched 通过 Lagrangian 处理安全）
        """
        fe = self._formation_error()
        task_reward = -self.w_formation * fe

        if self._check_success():
            task_reward += self.w_success

        return [float(task_reward) for _ in range(self.n)]

    def _compute_costs(self) -> List[float]:
        """
        安全成本：每个智能体是否与任何其他智能体碰撞。
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
        智能体 i 的观测向量：
          [自身位置(3), 自身速度(3),
           邻居1相对位置(3), 邻居1相对速度(3),
           邻居2相对位置(3), 邻居2相对速度(3), ...]
        不足 K_obs 个邻居时用零填充。
        """
        obs = np.zeros(self.obs_dim, dtype=np.float32)

        # 自身状态：位置 + 速度（相对目标位置）
        obs[0:3] = self.positions[agent_id] - self.goal_positions[agent_id]
        obs[3:6] = self.velocities[agent_id]

        # 邻居：按距离排序，取最近的 K_obs 个（在感知范围内）
        neighbors = self._get_sorted_neighbors(agent_id)

        for k, j in enumerate(neighbors[:self.K_obs]):
            base = 6 + k * 6
            obs[base:base+3] = self.positions[j] - self.positions[agent_id]  # 相对位置
            obs[base+3:base+6] = self.velocities[j] - self.velocities[agent_id]  # 相对速度

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
    # 辅助函数
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

    def _get_info(self) -> Dict:
        """返回调试信息字典。"""
        return {
            "t": self.t,
            "formation_error": self._formation_error(),
            "min_dist": self._min_pairwise_dist(),
            "n_collisions": int(sum(1 for c in self._compute_costs() if c > 0)),
            "success": self._check_success(),
            "positions": self.positions.copy(),
            "goal_positions": self.goal_positions.copy(),
        }

    def _min_pairwise_dist(self) -> float:
        """计算所有智能体对之间的最小距离。"""
        min_d = float("inf")
        for i in range(self.n):
            for j in range(i + 1, self.n):
                d = float(np.linalg.norm(self.positions[i] - self.positions[j]))
                if d < min_d:
                    min_d = d
        return min_d if self.n > 1 else 0.0

    def _random_init_positions(self) -> np.ndarray:
        """
        随机生成初始位置，保证两两间距 > 2 * d_min（避免初始碰撞）。
        最多尝试 1000 次，失败则使用网格初始化。
        """
        half = self.arena_size / 2 * 0.8
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
                0.0
            ]
        return positions

    def get_physical_graph(self) -> List[Tuple[int, int]]:
        """
        返回物理可连接图的边列表：(i, j) 当且仅当 ||pi - pj|| <= R_comm。
        供外部调度器（scheduler.py）使用。
        """
        edges = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                dist = np.linalg.norm(self.positions[i] - self.positions[j])
                if dist <= self.R_comm:
                    edges.append((i, j))
        return edges

    def compute_cbf_values(self) -> Dict[Tuple[int, int], float]:
        """
        计算所有智能体对的 CBF 值：
        h_ij(t) = (||pi(t) - pj(t)|| - d_min)^2

        返回字典 {(i, j): h_ij}（i < j）。
        供外部调度器使用。
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
        """返回全局状态向量 [n, 6] -> 展平为 [6n]。"""
        return np.concatenate([self.positions, self.velocities], axis=1).flatten()

    def render(self, mode: str = "text") -> Optional[str]:
        """简单的文本渲染（调试用）。"""
        if mode == "text":
            lines = [f"[t={self.t}/{self.max_steps}]"]
            for i in range(self.n):
                p = self.positions[i]
                g = self.goal_positions[i]
                err = np.linalg.norm(p - g)
                lines.append(
                    f"  UAV {i}: pos=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f}) "
                    f"err={err:.3f}"
                )
            lines.append(f"  FE={self._formation_error():.4f}  "
                         f"min_dist={self._min_pairwise_dist():.3f}")
            return "\n".join(lines)
        return None

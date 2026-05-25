"""
安全感知通信调度器（Safety-Priority Scheduler）

核心设计：
  - 无需训练，构造性保证通信硬预算 |E_t| <= k
  - 按 CBF 危险度优先分配通信预算
  - CBF 值：h_ij(t) = (||pi(t) - pj(t)|| - d_min)^2
  - 优先级：priority_ij = 1 / max(h_ij, eps)（h 越小 = 越危险 → 优先级越高）
  - 通信图：E_t = TopK(E^phys_t, k, by priority_ij)

命题 1（通信硬约束满足）：
  TopK 操作从 |E^phys_t| 条边中最多选 k 条，直接可行。
  无论 E^phys_t 大小如何，|E_t| <= min(|E^phys_t|, k) <= k。
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class SafetyPriorityScheduler:
    """
    安全感知通信调度器。

    使用方式：
        scheduler = SafetyPriorityScheduler(n_agents=4, k=4, d_min=0.5)
        E_t = scheduler.schedule(positions, phys_edges)
        # 返回 List[Tuple[int, int]]，长度 <= k
    """

    def __init__(
        self,
        n_agents: int,
        k: int,
        d_min: float = 0.5,
        eps: float = 1e-6,
        schedule_mode: str = "safety_priority",  # safety_priority | random | full
        v_max: float = 0.0,    # ADD: max velocity for conservative buffer (0 = disabled)
        dt: float = 0.1,       # ADD: timestep for buffer calculation
    ):
        """
        Args:
            n_agents: UAV 数量
            k: 通信预算（每步最多活跃链路数）
            d_min: 最小安全距离（米），与 CBF 计算一致
            eps: CBF 值下界，防止除零（eps = 1e-6）
            schedule_mode:
                "safety_priority" — 标准 SafeComm-PSched 调度（按 CBF 危险度）
                "random"          — 随机 Top-k（消融实验用）
                "full"            — 全通信（k >= |E^phys|，等价于不限制）
            v_max: 最大速度（m/s），用于保守 CBF 缓冲区计算（0 = 禁用）
            dt: 时间步长（s），用于缓冲区计算
        """
        self.n = n_agents
        self.k = k
        self.d_min = d_min
        self.eps = eps
        self.schedule_mode = schedule_mode
        self.v_max = v_max
        self.dt = dt
        # Communication history (only used when v_max > 0)
        self._last_known_pos: Optional[np.ndarray] = None   # [n, 3]
        self._last_comm_time: Optional[np.ndarray] = None   # [n, n] symmetric, steps since last comm
        self._current_step: int = 0

    # ------------------------------------------------------------------
    # 通信历史管理（保守 CBF 支持）
    # ------------------------------------------------------------------

    def reset_episode(self, positions: np.ndarray) -> None:
        """Reset communication history at episode start.

        Call this at the start of each episode.
        """
        n = positions.shape[0]
        self._last_known_pos = positions.copy()
        self._last_comm_time = np.zeros((n, n), dtype=np.float64)
        self._current_step = 0

    def update_history(
        self,
        active_edges: List[Tuple[int, int]],
        positions: np.ndarray,
    ) -> None:
        """Update communication history after each scheduling step.

        Call this AFTER schedule_step() with the selected active_edges.
        Increments step counter; resets comm timer for connected pairs.
        """
        if self._last_known_pos is None:
            return
        self._current_step += 1
        # Increment all timers
        self._last_comm_time += 1
        # Reset timer and update position for connected pairs
        for i, j in active_edges:
            self._last_known_pos[j] = positions[j].copy()
            self._last_known_pos[i] = positions[i].copy()
            self._last_comm_time[i, j] = 0
            self._last_comm_time[j, i] = 0

    # ------------------------------------------------------------------
    # 核心调度接口
    # ------------------------------------------------------------------

    def schedule(
        self,
        positions: np.ndarray,
        phys_edges: Optional[List[Tuple[int, int]]] = None,
    ) -> List[Tuple[int, int]]:
        """
        执行通信调度，返回激活的通信边集合。

        Args:
            positions: [n, 3] 当前 UAV 位置
            phys_edges: 物理可连接边列表（可选）。
                        若为 None，则自动根据 R_comm 计算（但此方法不需要 R_comm，
                        建议由环境提供 phys_edges）。
                        若为空列表，表示无任何物理连接。

        Returns:
            active_edges: 激活的通信边列表，|active_edges| <= k
                          每条边 (i, j) 满足 i < j
        """
        if phys_edges is None:
            # 默认：所有智能体对都可以通信（全连通情况）
            phys_edges = [
                (i, j)
                for i in range(self.n)
                for j in range(i + 1, self.n)
            ]

        if len(phys_edges) == 0:
            return []

        if len(phys_edges) <= self.k or self.schedule_mode == "full":
            return list(phys_edges)

        if self.schedule_mode == "random":
            return self._random_topk(phys_edges)

        # 默认：safety_priority
        return self._safety_priority_topk(positions, phys_edges)

    def _safety_priority_topk(
        self,
        positions: np.ndarray,
        phys_edges: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """按 CBF 危险度优先选 Top-k 边。使用保守 CBF 近似（§3.1.1）。"""
        # Use conservative CBF if history is initialized
        if self._last_known_pos is not None and self.v_max > 0:
            cbf_values = self.compute_conservative_cbf_values(positions, phys_edges)
        else:
            cbf_values = self.compute_cbf_values(positions, phys_edges)

        priorities = []
        for edge in phys_edges:
            i, j = min(edge), max(edge)
            h_ij = cbf_values.get((i, j), self.eps)
            priority = 1.0 / max(h_ij, self.eps)
            priorities.append((priority, (i, j)))

        # 按优先级降序排序，取前 k 条
        priorities.sort(key=lambda x: x[0], reverse=True)
        return [edge for _, edge in priorities[:self.k]]

    def _random_topk(
        self,
        phys_edges: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """
        随机选 k 条边（消融实验：验证安全感知调度的必要性）。
        """
        indices = np.random.choice(len(phys_edges), size=self.k, replace=False)
        return [phys_edges[i] for i in indices]

    # ------------------------------------------------------------------
    # CBF 计算工具
    # ------------------------------------------------------------------

    def compute_cbf_matrix(self, positions: np.ndarray) -> np.ndarray:
        """
        计算所有智能体对的 CBF 值矩阵。

        Returns:
            cbf_matrix: [n, n] 对称矩阵，cbf_matrix[i, j] = h_ij
                        对角线为 0，h_ij = (||pi - pj|| - d_min)^2
        """
        n = positions.shape[0]
        cbf_matrix = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(i + 1, n):
                dist = float(np.linalg.norm(positions[i] - positions[j]))
                h_ij = (dist - self.d_min) ** 2
                cbf_matrix[i, j] = h_ij
                cbf_matrix[j, i] = h_ij

        return cbf_matrix

    def compute_cbf_values(
        self,
        positions: np.ndarray,
        edges: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict[Tuple[int, int], float]:
        """
        计算指定边集合（或全部对）的 CBF 值字典。

        Returns:
            {(i, j): h_ij}（i < j）
        """
        if edges is None:
            edges = [
                (i, j)
                for i in range(self.n)
                for j in range(i + 1, self.n)
            ]

        cbf_dict = {}
        for i, j in edges:
            dist = float(np.linalg.norm(positions[i] - positions[j]))
            h_ij = (dist - self.d_min) ** 2
            cbf_dict[(i, j)] = h_ij

        return cbf_dict

    def compute_conservative_cbf_values(
        self,
        positions: np.ndarray,
        edges: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict[Tuple[int, int], float]:
        """Compute conservative CBF values using Wang TAC 2017 buffer formula.

        h_cons_ij = max(||p_i - p_j_hat|| - d_min - d_buffer, 0)^2
        d_buffer = v_max * dt_comm (time gap in steps * dt)

        Falls back to standard CBF if history not initialized.
        """
        if self._last_known_pos is None or self.v_max <= 0:
            return self.compute_cbf_values(positions, edges)

        if edges is None:
            edges = [(i, j) for i in range(self.n) for j in range(i + 1, self.n)]

        cbf_dict = {}
        for i, j in edges:
            p_j_hat = self._last_known_pos[j]
            dt_comm = float(self._last_comm_time[i, j]) * self.dt
            d_buffer = self.v_max * dt_comm
            dist = float(np.linalg.norm(positions[i] - p_j_hat))
            conservative_dist = max(dist - d_buffer, 0.0)
            h_cons = max(conservative_dist - self.d_min, 0.0) ** 2
            cbf_dict[(i, j)] = h_cons
        return cbf_dict

    def compute_priority_dict(
        self,
        positions: np.ndarray,
        edges: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict[Tuple[int, int], float]:
        """
        计算优先级字典 priority_ij = 1 / max(h_ij, eps)。

        Returns:
            {(i, j): priority_ij}
        """
        cbf_dict = self.compute_cbf_values(positions, edges)
        return {
            edge: 1.0 / max(h, self.eps)
            for edge, h in cbf_dict.items()
        }

    # ------------------------------------------------------------------
    # 邻居列表构建（供网络聚合使用）
    # ------------------------------------------------------------------

    def build_neighbor_lists(
        self,
        active_edges: List[Tuple[int, int]],
    ) -> Dict[int, List[int]]:
        """
        从激活边集合构建每个智能体的邻居列表。

        Args:
            active_edges: List[Tuple[int, int]]，激活的通信边

        Returns:
            neighbor_lists: {agent_id: [neighbor_ids]}
                            注意：边 (i,j) 对 i 和 j 都加入对方邻居
        """
        neighbor_lists: Dict[int, List[int]] = {i: [] for i in range(self.n)}
        for i, j in active_edges:
            neighbor_lists[i].append(j)
            neighbor_lists[j].append(i)
        return neighbor_lists

    def get_adjacency_matrix(
        self,
        active_edges: List[Tuple[int, int]],
    ) -> np.ndarray:
        """
        从激活边集合构建邻接矩阵 [n, n]（0/1 对称矩阵）。
        """
        adj = np.zeros((self.n, self.n), dtype=np.float32)
        for i, j in active_edges:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
        return adj

    # ------------------------------------------------------------------
    # 统计信息
    # ------------------------------------------------------------------

    def bandwidth_utilization(self, active_edges: List[Tuple[int, int]]) -> float:
        """带宽利用率 = |E_t| / k"""
        return len(active_edges) / max(self.k, 1)

    def schedule_step(
        self,
        positions: np.ndarray,
        phys_edges: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict:
        """
        完整调度步骤，返回调度结果 + 统计信息。

        Returns:
            {
                "active_edges": List[Tuple[int,int]],
                "neighbor_lists": Dict[int, List[int]],
                "adjacency_matrix": np.ndarray [n, n],
                "cbf_values": Dict[Tuple[int,int], float],
                "bandwidth_utilization": float,
                "n_active": int,
            }
        """
        active_edges = self.schedule(positions, phys_edges)
        cbf_values = self.compute_conservative_cbf_values(positions, phys_edges)
        neighbor_lists = self.build_neighbor_lists(active_edges)
        adj_matrix = self.get_adjacency_matrix(active_edges)
        bu = self.bandwidth_utilization(active_edges)

        return {
            "active_edges": active_edges,
            "neighbor_lists": neighbor_lists,
            "adjacency_matrix": adj_matrix,
            "cbf_values": cbf_values,
            "bandwidth_utilization": bu,
            "n_active": len(active_edges),
        }


# ------------------------------------------------------------------
# 便利工厂函数
# ------------------------------------------------------------------

def make_scheduler(
    n_agents: int,
    k: int,
    mode: str = "safety_priority",
    d_min: float = 0.5,
) -> SafetyPriorityScheduler:
    """
    创建调度器的便利工厂函数。

    Args:
        n_agents: UAV 数量
        k: 通信预算
        mode: "safety_priority" | "random" | "full"
        d_min: 最小安全距离
    """
    return SafetyPriorityScheduler(
        n_agents=n_agents,
        k=k,
        d_min=d_min,
        schedule_mode=mode,
    )

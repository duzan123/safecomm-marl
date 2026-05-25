"""
调度器单元测试

覆盖：
  1. 命题 1：通信硬约束始终满足（|E_t| <= k）
  2. CBF 值计算正确性
  3. 优先级排序：危险对（h 小）优先获得通信链路
  4. 随机调度模式
  5. 全通信模式（k 足够大）
  6. 孤立节点（无邻居）处理
  7. 邻居列表与邻接矩阵构建
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from algorithms.scheduler import SafetyPriorityScheduler, make_scheduler


class TestCBFValues:
    """CBF 值计算正确性测试"""

    def test_cbf_zero_at_boundary(self):
        """当两 UAV 距离恰好等于 d_min 时，h_ij = 0"""
        scheduler = SafetyPriorityScheduler(n_agents=2, k=1, d_min=0.5)
        positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float64)
        cbf = scheduler.compute_cbf_values(positions)
        assert abs(cbf[(0, 1)]) < 1e-10, "h_ij 应在边界距离处为 0"

    def test_cbf_positive_outside_boundary(self):
        """距离 > d_min 时 h_ij > 0"""
        scheduler = SafetyPriorityScheduler(n_agents=2, k=1, d_min=0.5)
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        cbf = scheduler.compute_cbf_values(positions)
        assert cbf[(0, 1)] > 0, "安全距离外 h_ij > 0"

    def test_cbf_formula(self):
        """验证 h_ij = (||pi - pj|| - d_min)^2"""
        scheduler = SafetyPriorityScheduler(n_agents=2, k=1, d_min=0.5)
        p = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float64)
        cbf = scheduler.compute_cbf_values(p)
        expected = (1.5 - 0.5) ** 2  # = 1.0
        assert abs(cbf[(0, 1)] - expected) < 1e-9

    def test_cbf_matrix_symmetry(self):
        """CBF 矩阵应对称"""
        scheduler = SafetyPriorityScheduler(n_agents=4, k=4, d_min=0.5)
        positions = np.random.rand(4, 3)
        mat = scheduler.compute_cbf_matrix(positions)
        assert np.allclose(mat, mat.T), "CBF 矩阵应对称"
        assert np.all(np.diag(mat) == 0), "对角线应为 0"


class TestHardConstraint:
    """命题 1：通信硬约束满足性测试"""

    def test_hard_budget_all_connected(self):
        """全连通图下，激活边数 <= k"""
        for n in [4, 6, 8]:
            for k in [1, 2, 4]:
                scheduler = SafetyPriorityScheduler(n_agents=n, k=k, d_min=0.5)
                positions = np.random.rand(n, 3) * 10
                phys_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
                active = scheduler.schedule(positions, phys_edges)
                assert len(active) <= k, (
                    f"n={n}, k={k}: |E_t|={len(active)} > k 违反通信硬约束"
                )

    def test_hard_budget_sparse_graph(self):
        """稀疏物理图下，若 |E^phys| < k，则全部激活"""
        scheduler = SafetyPriorityScheduler(n_agents=4, k=10, d_min=0.5)
        positions = np.random.rand(4, 3)
        phys_edges = [(0, 1), (1, 2)]  # 只有 2 条边
        active = scheduler.schedule(positions, phys_edges)
        assert len(active) == 2, "稀疏图应全部激活（不足 k 条）"

    def test_hard_budget_empty_graph(self):
        """空物理图：不激活任何边"""
        scheduler = SafetyPriorityScheduler(n_agents=4, k=4, d_min=0.5)
        positions = np.random.rand(4, 3)
        active = scheduler.schedule(positions, phys_edges=[])
        assert len(active) == 0

    def test_hard_budget_random_trials(self):
        """100 次随机测试，确保硬约束始终满足"""
        rng = np.random.default_rng(42)
        for _ in range(100):
            n = rng.integers(3, 9)
            k = rng.integers(1, n * (n-1) // 2 + 1)
            scheduler = SafetyPriorityScheduler(n_agents=int(n), k=int(k), d_min=0.5)
            positions = rng.uniform(-5, 5, (int(n), 3))
            phys_edges = [
                (i, j) for i in range(int(n)) for j in range(i+1, int(n))
            ]
            active = scheduler.schedule(positions, phys_edges)
            assert len(active) <= k, f"随机测试硬约束违反: |E_t|={len(active)}, k={k}"


class TestSafetyPriority:
    """安全优先级排序正确性测试"""

    def test_dangerous_pair_selected(self):
        """
        最危险的一对（距离最近，h 最小）应当在 k=1 时被优先选中。

        配置：3 个 UAV，一对距离很近（危险），其余较远（安全）
        """
        scheduler = SafetyPriorityScheduler(n_agents=3, k=1, d_min=0.5)
        positions = np.array([
            [0.0, 0.0, 0.0],   # UAV 0
            [0.51, 0.0, 0.0],  # UAV 1，距离 0 很近（h ≈ 0.0001）
            [5.0, 0.0, 0.0],   # UAV 2，距离 0 很远（h = 20.25）
        ], dtype=np.float64)
        phys_edges = [(0, 1), (0, 2), (1, 2)]
        active = scheduler.schedule(positions, phys_edges)
        assert len(active) == 1
        # 应选中 (0, 1) 这对危险的链路
        assert (0, 1) in active, f"危险对 (0,1) 应被优先选中，实际选中 {active}"

    def test_priority_order(self):
        """验证优先级字典排序与 CBF 值的对应关系（h 越小 → priority 越大）"""
        scheduler = SafetyPriorityScheduler(n_agents=3, k=2, d_min=0.5)
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.6, 0.0, 0.0],  # 距离 0.6，h=(0.1)^2=0.01
            [3.0, 0.0, 0.0],  # 距离 3.0，h=(2.5)^2=6.25
        ], dtype=np.float64)
        edges = [(0, 1), (0, 2), (1, 2)]
        priorities = scheduler.compute_priority_dict(positions, edges)

        # (0,1) 最危险，优先级应最高
        assert priorities[(0, 1)] > priorities[(0, 2)]
        assert priorities[(0, 1)] > priorities[(1, 2)]


class TestScheduleModes:
    """不同调度模式测试"""

    def test_full_mode_activates_all(self):
        """full 模式应激活所有物理边"""
        scheduler = make_scheduler(n_agents=4, k=2, mode="full")
        positions = np.random.rand(4, 3)
        phys_edges = [(0,1), (1,2), (2,3), (0,3)]
        active = scheduler.schedule(positions, phys_edges)
        assert set(active) == set(phys_edges)

    def test_random_mode_budget(self):
        """随机模式也应满足 |E_t| <= k"""
        scheduler = make_scheduler(n_agents=6, k=3, mode="random")
        positions = np.random.rand(6, 3)
        phys_edges = [(i, j) for i in range(6) for j in range(i+1, 6)]
        active = scheduler.schedule(positions, phys_edges)
        assert len(active) <= 3

    def test_random_mode_randomness(self):
        """随机模式的结果应在多次调用中有所变化（非退化常量）"""
        np.random.seed(0)
        scheduler = make_scheduler(n_agents=6, k=3, mode="random")
        positions = np.random.rand(6, 3)
        phys_edges = [(i, j) for i in range(6) for j in range(i+1, 6)]

        results = set()
        for _ in range(20):
            active = scheduler.schedule(positions, phys_edges)
            results.add(frozenset(active))
        # 20 次随机应不全相同
        assert len(results) > 1, "随机调度在 20 次试验中应产生不同结果"


class TestNeighborLists:
    """邻居列表与邻接矩阵构建测试"""

    def test_neighbor_list_bidirectional(self):
        """边 (i,j) 应同时加入 j 的邻居和 i 的邻居"""
        scheduler = SafetyPriorityScheduler(n_agents=4, k=4, d_min=0.5)
        active_edges = [(0, 1), (1, 2), (0, 3)]
        nbrs = scheduler.build_neighbor_lists(active_edges)
        assert 1 in nbrs[0] and 0 in nbrs[1]
        assert 2 in nbrs[1] and 1 in nbrs[2]
        assert 3 in nbrs[0] and 0 in nbrs[3]

    def test_adjacency_matrix_symmetry(self):
        """邻接矩阵应对称"""
        scheduler = SafetyPriorityScheduler(n_agents=4, k=4, d_min=0.5)
        active_edges = [(0, 1), (2, 3)]
        adj = scheduler.get_adjacency_matrix(active_edges)
        assert adj.shape == (4, 4)
        assert np.allclose(adj, adj.T), "邻接矩阵应对称"
        assert adj[0, 1] == 1.0 and adj[1, 0] == 1.0
        assert adj[2, 3] == 1.0 and adj[3, 2] == 1.0
        assert adj[0, 2] == 0.0

    def test_isolated_agent_empty_neighbors(self):
        """孤立节点（无激活边）的邻居列表应为空"""
        scheduler = SafetyPriorityScheduler(n_agents=4, k=4, d_min=0.5)
        active_edges = [(0, 1)]  # UAV 2 和 3 孤立
        nbrs = scheduler.build_neighbor_lists(active_edges)
        assert nbrs[2] == []
        assert nbrs[3] == []


class TestBandwidthUtilization:
    """带宽利用率计算测试"""

    def test_utilization_range(self):
        """带宽利用率应在 [0, 1] 范围内"""
        scheduler = SafetyPriorityScheduler(n_agents=4, k=4, d_min=0.5)
        positions = np.random.rand(4, 3)
        phys_edges = [(0, 1), (1, 2)]
        active = scheduler.schedule(positions, phys_edges)
        bu = scheduler.bandwidth_utilization(active)
        assert 0.0 <= bu <= 1.0

    def test_full_utilization(self):
        """激活 k 条边时，带宽利用率应为 1.0"""
        scheduler = SafetyPriorityScheduler(n_agents=6, k=3, d_min=0.5)
        # 确保物理图有足够多边
        positions = np.random.rand(6, 3) * 10
        phys_edges = [(i, j) for i in range(6) for j in range(i+1, 6)]
        active = scheduler.schedule(positions, phys_edges)
        bu = scheduler.bandwidth_utilization(active)
        if len(active) == 3:  # 恰好选了 k 条
            assert abs(bu - 1.0) < 1e-9


class TestScheduleStep:
    """完整调度步骤返回值测试"""

    def test_schedule_step_keys(self):
        """schedule_step 应返回所有预期的键"""
        scheduler = SafetyPriorityScheduler(n_agents=4, k=4, d_min=0.5)
        positions = np.random.rand(4, 3)
        result = scheduler.schedule_step(positions)
        expected_keys = {
            "active_edges", "neighbor_lists", "adjacency_matrix",
            "cbf_values", "bandwidth_utilization", "n_active"
        }
        assert expected_keys.issubset(result.keys())

    def test_schedule_step_consistency(self):
        """邻接矩阵应与激活边集合一致"""
        scheduler = SafetyPriorityScheduler(n_agents=4, k=4, d_min=0.5)
        positions = np.random.rand(4, 3)
        result = scheduler.schedule_step(positions)
        adj = result["adjacency_matrix"]
        for i, j in result["active_edges"]:
            assert adj[i, j] == 1.0 and adj[j, i] == 1.0, (
                f"边 ({i},{j}) 在邻接矩阵中应为 1"
            )


# ==============================================================================
# Conservative CBF Buffer Tests (v2.0, Wang TAC 2017)
# ==============================================================================

class TestConservativeCBF:
    def test_conservative_cbf_equals_standard_at_t0(self):
        """At t=0 (no comm gap), conservative CBF should equal standard CBF."""
        sched = SafetyPriorityScheduler(n_agents=4, k=4, d_min=0.5, v_max=1.0, dt=0.1)
        positions = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
        ])
        sched.reset_episode(positions)
        cons = sched.compute_conservative_cbf_values(positions, [(0, 1)])
        std = sched.compute_cbf_values(positions, [(0, 1)])
        # At t=0, dt_comm=0, d_buffer=0, so conservative = standard
        assert abs(cons[(0, 1)] - std[(0, 1)]) < 1e-6

    def test_conservative_cbf_decreases_with_comm_gap(self):
        """Longer comm gap -> larger buffer -> smaller conservative CBF -> higher priority."""
        sched = SafetyPriorityScheduler(n_agents=4, k=4, d_min=0.5, v_max=1.0, dt=0.1)
        positions = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        ])
        sched.reset_episode(positions)
        h_at_t0 = sched.compute_conservative_cbf_values(positions, [(0, 1)])[(0, 1)]

        # Simulate 3 steps without agent 0-1 communicating
        for _ in range(3):
            sched.update_history([], positions)  # no active edges

        h_at_t3 = sched.compute_conservative_cbf_values(positions, [(0, 1)])[(0, 1)]
        # After 3 steps gap, buffer = 1.0 * 3 * 0.1 = 0.3m
        # h_cons should be smaller (more dangerous)
        assert h_at_t3 <= h_at_t0

    def test_update_history_resets_timer(self):
        """Communicating resets the comm timer to 0."""
        sched = SafetyPriorityScheduler(n_agents=4, k=4, d_min=0.5, v_max=1.0, dt=0.1)
        positions = np.zeros((4, 3))
        sched.reset_episode(positions)

        # Advance time
        for _ in range(5):
            sched.update_history([], positions)

        # Now communicate (0,1)
        sched.update_history([(0, 1)], positions)
        assert sched._last_comm_time[0, 1] == 0
        assert sched._last_comm_time[1, 0] == 0

    def test_conservative_cbf_disabled_without_v_max(self):
        """v_max=0 disables conservative buffer, falls back to standard CBF."""
        sched_no_vmax = SafetyPriorityScheduler(n_agents=4, k=4, d_min=0.5, v_max=0.0, dt=0.1)
        sched_vmax = SafetyPriorityScheduler(n_agents=4, k=4, d_min=0.5, v_max=1.0, dt=0.1)
        positions = np.zeros((4, 3))
        positions[1] = np.array([2.0, 0.0, 0.0])
        sched_no_vmax.reset_episode(positions)
        sched_vmax.reset_episode(positions)

        h_no_vmax = sched_no_vmax.compute_conservative_cbf_values(positions, [(0, 1)])[(0, 1)]
        h_std = sched_no_vmax.compute_cbf_values(positions, [(0, 1)])[(0, 1)]
        assert abs(h_no_vmax - h_std) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

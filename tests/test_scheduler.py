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

    @pytest.mark.parametrize("mode", ["safety_priority", "random"])
    def test_budgeted_modes_return_empty_when_k_zero(self, mode):
        sched = SafetyPriorityScheduler(n_agents=3, k=0, d_min=0.5, schedule_mode=mode)
        positions = np.zeros((3, 3))
        sched.reset_episode(positions)
        active = sched.schedule(positions, [(0, 1), (1, 2)])
        assert active == []

    def test_full_mode_bypasses_k_zero_budget(self):
        sched = SafetyPriorityScheduler(n_agents=4, k=0, d_min=0.5, schedule_mode="full")
        positions = np.zeros((4, 3))
        phys_edges = [(0, 1), (1, 2), (2, 3)]
        sched.reset_episode(positions)
        active = sched.schedule(positions, phys_edges)
        assert active == phys_edges

    def test_topk_returns_all_edges_when_k_covers_graph(self):
        sched = SafetyPriorityScheduler(n_agents=4, k=10, d_min=0.5)
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ], dtype=np.float64)
        phys_edges = [(0, 1), (0, 2), (2, 3)]
        sched.reset_episode(positions)
        active = sched.schedule(positions, phys_edges)
        assert len(active) == len(phys_edges)
        assert set(active) == set(phys_edges)

    def test_random_mode_respects_k_and_seed(self):
        positions = np.zeros((5, 3))
        phys_edges = [(i, j) for i in range(5) for j in range(i + 1, 5)]
        sched_a = SafetyPriorityScheduler(
            n_agents=5, k=3, d_min=0.5, schedule_mode="random", seed=123
        )
        sched_b = SafetyPriorityScheduler(
            n_agents=5, k=3, d_min=0.5, schedule_mode="random", seed=123
        )
        sched_a.reset_episode(positions)
        sched_b.reset_episode(positions)

        active_a = sched_a.schedule(positions, phys_edges)
        active_b = sched_b.schedule(positions, phys_edges)

        assert len(active_a) <= 3
        assert active_a == active_b
        assert set(active_a).issubset(set(phys_edges))


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


class TestCommunicationHistory:
    def test_schedule_does_not_mutate_history(self):
        sched = SafetyPriorityScheduler(
            n_agents=3, k=1, d_min=0.5, v_max=0.0, dt=0.1,
            beta=1.0, tau_ref=1.0, tau_max=5.0,
        )
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ], dtype=np.float64)
        edges = [(0, 1), (0, 2), (1, 2)]
        sched.reset_episode(positions)
        sched.update_history([], positions)

        dt_before = dict(sched._dt_comm)
        prios_before = sched.compute_voi_priority(positions, edges)

        sched.schedule(positions, edges)

        assert sched._dt_comm == dt_before
        assert sched.compute_voi_priority(positions, edges) == prios_before

    def test_update_history_resets_active_and_ages_inactive_pairs(self):
        sched = SafetyPriorityScheduler(n_agents=3, k=1, d_min=0.5, dt=0.25)
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ], dtype=np.float64)
        moved_positions = positions.copy()
        moved_positions[1] = np.array([1.5, 0.0, 0.0])
        sched.reset_episode(positions)

        sched.update_history([], positions)
        sched.update_history([(1, 0)], moved_positions)

        assert sched._dt_comm[(0, 1)] == 0.0
        assert sched._dt_comm[(0, 2)] == pytest.approx(0.5)
        assert sched._dt_comm[(1, 2)] == pytest.approx(0.5)
        np.testing.assert_allclose(sched._last_pos_j[(0, 1)], moved_positions[1])


class TestEdgeNormalization:
    def test_reverse_and_duplicate_edges_are_normalized_for_schedule(self):
        sched = SafetyPriorityScheduler(n_agents=3, k=5, d_min=0.5)
        positions = np.zeros((3, 3))
        sched.reset_episode(positions)

        active = sched.schedule(positions, [(1, 0), (0, 1), (2, 1), (1, 2)])

        assert len(active) == 2
        assert set(active) == {(0, 1), (1, 2)}

    def test_neighbor_lists_are_deduped_normalized_and_bidirectional(self):
        sched = SafetyPriorityScheduler(n_agents=3, k=2, d_min=0.5)

        neighbors = sched.get_neighbor_lists([(1, 0), (0, 1), (2, 1), (1, 2)])

        assert neighbors == {0: [1], 1: [0, 2], 2: [1]}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

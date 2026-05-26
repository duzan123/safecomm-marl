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

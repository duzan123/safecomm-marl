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

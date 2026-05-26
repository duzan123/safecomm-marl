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

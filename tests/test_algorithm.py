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
        """安全预算超限时 λ_s 按 primal-dual 公式上升"""
        env = UAVFormationEnv(n_agents=4, d_min=0.3, dt=0.1, max_steps=50, R_comm=5.0, seed=0)
        cfg = {**FAST_CONFIG, "safety_budget": 0.5}
        agent = SafeCommVoI(env, config=cfg, device="cpu")
        lam_before = float(agent.lambda_s)
        agent.collect_rollout()
        agent.buffer.costs[: agent.buffer.ptr] = 1.0
        metrics = agent.update()
        lam_after = float(agent.lambda_s)
        expected = np.clip(
            lam_before
            + agent.config["lr_lambda"] * (metrics["J_safe"] - agent.config["safety_budget"]),
            0.0,
            agent.config["lambda_max"],
        )
        assert metrics["J_safe"] > agent.config["safety_budget"]
        assert lam_after > lam_before
        assert lam_after == pytest.approx(expected)

    def test_collect_rollout_fills_buffer_and_scheduler_semantics(self, monkeypatch):
        _, agent = make_env_and_agent()
        schedule_calls = []
        update_calls = []
        original_schedule = agent.scheduler.schedule
        original_update_history = agent.scheduler.update_history

        def schedule_spy(positions, phys_edges):
            active = original_schedule(positions, phys_edges)
            schedule_calls.append((len(list(phys_edges)), len(active)))
            return active

        def update_history_spy(active_edges, positions, velocities=None):
            update_calls.append(len(list(active_edges)))
            return original_update_history(active_edges, positions, velocities)

        monkeypatch.setattr(agent.scheduler, "schedule", schedule_spy)
        monkeypatch.setattr(agent.scheduler, "update_history", update_history_spy)

        agent.collect_rollout()
        assert agent.buffer.ptr == agent.config["n_steps"]
        assert agent.buffer._finished
        data = agent.buffer.get_training_data(device="cpu")
        assert data["obs"].shape[0] == agent.config["n_steps"]
        assert len(schedule_calls) == agent.config["n_steps"]
        assert len(update_calls) == agent.config["n_steps"]
        for _, n_active in schedule_calls:
            assert n_active <= agent.config["k"]
        adj = agent.buffer.adj_matrices[: agent.buffer.ptr]
        np.testing.assert_allclose(adj, np.transpose(adj, (0, 2, 1)))
        assert np.all(np.diagonal(adj, axis1=1, axis2=2) == 0.0)
        assert np.all(adj.sum(axis=(1, 2)) / 2.0 <= agent.config["k"])

    def test_save_load_roundtrip(self):
        import tempfile
        _, agent = make_env_and_agent()
        agent.collect_rollout()
        agent.update()
        lam_orig = float(agent.lambda_s)
        n_iter_orig = int(agent.n_iterations)
        total_steps_orig = int(agent.total_steps)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            agent.save(path)

            _, agent2 = make_env_and_agent()
            agent2.load(path)
            assert abs(float(agent2.lambda_s) - lam_orig) < 1e-6
            assert agent2.n_iterations == n_iter_orig
            assert agent2.total_steps == total_steps_orig
        finally:
            os.unlink(path)

    def test_load_rejects_mismatched_checkpoint_config(self):
        import tempfile
        _, agent = make_env_and_agent()
        agent.collect_rollout()
        agent.update()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            agent.save(path)
            env2 = UAVFormationEnv(n_agents=4, d_min=0.3, dt=0.1, max_steps=50, R_comm=5.0, seed=0)
            incompatible_cfg = {**FAST_CONFIG, "n_steps": FAST_CONFIG["n_steps"] + 1}
            agent2 = SafeCommVoI(env2, config=incompatible_cfg, device="cpu")
            with pytest.raises(ValueError, match="checkpoint config"):
                agent2.load(path)
        finally:
            os.unlink(path)

    @pytest.mark.parametrize(
        "bad_key,bad_value",
        [
            ("n_steps", 0),
            ("batch_size", 0),
            ("n_epochs", 0),
            ("lr_actor", 0.0),
            ("action_scale", 0.0),
        ],
    )
    def test_invalid_config_rejected(self, bad_key, bad_value):
        env = UAVFormationEnv(n_agents=4, d_min=0.3, dt=0.1, max_steps=50, R_comm=5.0, seed=0)
        cfg = {**FAST_CONFIG, bad_key: bad_value}
        with pytest.raises(ValueError, match=bad_key):
            SafeCommVoI(env, config=cfg, device="cpu")

    def test_evaluate_returns_metrics(self):
        _, agent = make_env_and_agent()
        metrics = agent.evaluate(n_episodes=2, deterministic=True)
        assert "mean_formation_error" in metrics
        assert "mean_episode_cost" in metrics
        assert "success_rate" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

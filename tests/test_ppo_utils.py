"""RolloutBuffer 和 PPO 工具函数测试"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import pytest
from algorithms.ppo_utils import (
    RolloutBuffer, compute_gae, ppo_clip_loss,
    value_loss, explained_variance, mgda_solve,
)


class TestRolloutBuffer:

    def _make_buffer(self):
        return RolloutBuffer(n_steps=20, n_agents=3, obs_dim=12,
                             act_dim=3, global_state_dim=36)

    def _add_zero_step(self, buf, reward=0.1, cost=0.0, done=False):
        buf.add(
            obs=np.zeros((3, 12)),
            actions=np.zeros((3, 3)),
            log_probs=np.zeros(3),
            rewards=[reward] * 3,
            costs=[cost] * 3,
            value=0.5,
            cost_value=0.1,
            done=done,
            global_state=np.zeros(36),
            adj_matrix=np.eye(3),
        )

    def test_add_and_ptr(self):
        buf = self._make_buffer()
        for t in range(5):
            buf.add(
                obs=np.zeros((3, 12)),
                actions=np.zeros((3, 3)),
                log_probs=np.zeros(3),
                rewards=[0.1, 0.2, 0.3],
                costs=[0.0, 0.0, 0.0],
                value=0.5,
                cost_value=0.1,
                done=False,
                global_state=np.zeros(36),
                adj_matrix=np.zeros((3, 3)),
            )
        assert buf.ptr == 5

    def test_finish_rollout_sets_last_value(self):
        buf = self._make_buffer()
        buf.add(np.zeros((3,12)), np.zeros((3,3)), np.zeros(3),
                [0.0]*3, [0.0]*3, 0.5, 0.1, False, np.zeros(36), np.zeros((3,3)))
        buf.finish_rollout(last_value=1.0, last_cost_value=0.5)
        assert buf.values[buf.ptr] == 1.0
        assert buf.cost_values[buf.ptr] == 0.5

    def test_get_training_data_keys(self):
        buf = self._make_buffer()
        for _ in range(10):
            self._add_zero_step(buf)
        buf.finish_rollout(0.4, 0.1)
        data = buf.get_training_data(gamma=0.99, lam=0.95, device="cpu")
        for key in ["obs","actions","log_probs_old","advantages","returns",
                    "cost_advantages","cost_returns","global_states","adj_matrices"]:
            assert key in data
        assert data["obs"].shape == (10, 3, 12)
        assert data["actions"].shape == (10, 3, 3)
        assert data["log_probs_old"].shape == (10, 3)
        assert data["advantages"].shape == (10,)
        assert data["returns"].shape == (10,)
        assert data["adj_matrices"].shape == (10, 3, 3)
        assert data["obs"].dtype == torch.float32
        assert data["dones"].dtype == torch.bool

    def test_advantages_normalized(self):
        buf = self._make_buffer()
        for t in range(10):
            buf.add(np.zeros((3,12)), np.zeros((3,3)), np.zeros(3),
                    [float(t)]*3, [0.0]*3, float(t)*0.1, 0.0, t==9,
                    np.zeros(36), np.zeros((3,3)))
        buf.finish_rollout(0.0, 0.0)
        data = buf.get_training_data()
        adv = data["advantages"].numpy()
        assert abs(adv.mean()) < 0.1
        assert abs(adv.std() - 1.0) < 0.1

    def test_get_training_data_requires_finish_rollout(self):
        buf = self._make_buffer()
        self._add_zero_step(buf)
        with pytest.raises(RuntimeError):
            buf.get_training_data()

    def test_reset_allows_repeated_rollouts(self):
        buf = self._make_buffer()
        self._add_zero_step(buf, reward=1.0)
        buf.finish_rollout(0.0, 0.0)
        first = buf.get_training_data()
        assert first["obs"].shape[0] == 1

        buf.reset()
        assert buf.ptr == 0
        self._add_zero_step(buf, reward=2.0, done=True)
        buf.finish_rollout(0.0, 0.0)
        second = buf.get_training_data()
        assert buf.ptr == 1
        assert second["obs"].shape[0] == 1
        assert second["rewards"][0, 0].item() == pytest.approx(2.0)

    def test_reset_clears_finished_state(self):
        buf = self._make_buffer()
        self._add_zero_step(buf)
        buf.finish_rollout(0.0, 0.0)
        buf.reset()
        self._add_zero_step(buf)
        with pytest.raises(RuntimeError):
            buf.get_training_data()

    def test_capacity_exceeded_raises(self):
        buf = RolloutBuffer(n_steps=1, n_agents=3, obs_dim=12,
                            act_dim=3, global_state_dim=36)
        self._add_zero_step(buf)
        with pytest.raises(RuntimeError):
            self._add_zero_step(buf)

    def test_shape_mismatch_raises(self):
        buf = self._make_buffer()
        with pytest.raises(ValueError):
            buf.add(np.zeros((3,11)), np.zeros((3,3)), np.zeros(3),
                    [0.1]*3, [0.0]*3, 0.5, 0.1, False, np.zeros(36), np.eye(3))


class TestComputeGAE:

    def test_zero_rewards_zero_values(self):
        rewards = np.zeros(5)
        values = np.zeros(6)
        dones = np.array([False]*5)
        adv, ret = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
        assert adv.shape == (5,)
        np.testing.assert_allclose(adv, 0.0, atol=1e-6)

    def test_done_resets_gae(self):
        rewards = np.array([1.0, 0.0, 1.0])
        values = np.zeros(4)
        dones = np.array([False, True, False])
        adv, ret = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
        np.testing.assert_allclose(adv, np.array([1.0, 0.0, 1.0], dtype=np.float32))
        np.testing.assert_allclose(ret, np.array([1.0, 0.0, 1.0], dtype=np.float32))


class TestPPOLoss:

    def test_ratio_one_gives_no_clip(self):
        log_p = torch.zeros(10)
        log_p_old = torch.zeros(10)
        adv = torch.ones(10)
        loss, kl = ppo_clip_loss(log_p, log_p_old, adv, clip_eps=0.2)
        assert float(loss) == pytest.approx(-1.0, abs=1e-5)

    def test_time_step_advantages_broadcast_to_agents(self):
        log_p = torch.zeros(4, 3)
        log_p_old = torch.zeros(4, 3)
        adv = torch.ones(4)
        loss, kl = ppo_clip_loss(log_p, log_p_old, adv, clip_eps=0.2)
        assert float(loss) == pytest.approx(-1.0, abs=1e-5)

    def test_value_loss_clipped(self):
        pred = torch.tensor([1.5, 2.5, 3.5])
        old = torch.tensor([1.0, 2.0, 3.0])
        returns = torch.tensor([2.0, 3.0, 4.0])
        loss = value_loss(pred, old, returns, clip_eps=0.2)
        assert float(loss) >= 0.0

    def test_value_loss_column_vectors_match_flat_vectors(self):
        pred_col = torch.tensor([[1.5], [2.5], [3.5]])
        old_col = torch.tensor([[1.0], [2.0], [3.0]])
        returns = torch.tensor([2.0, 3.0, 4.0])
        column_loss = value_loss(pred_col, old_col, returns, clip_eps=0.2)
        flat_loss = value_loss(pred_col.squeeze(-1), old_col.squeeze(-1), returns, clip_eps=0.2)
        assert float(column_loss) == pytest.approx(float(flat_loss), abs=1e-6)

    def test_value_loss_shape_mismatch_raises_value_error(self):
        with pytest.raises(ValueError):
            value_loss(torch.zeros(3, 1), torch.zeros(3), torch.zeros(2))


class TestExplainedVariance:

    def test_perfect_prediction_is_one(self):
        returns = torch.tensor([1.0, 2.0, 3.0])
        pred = returns.clone()
        assert explained_variance(pred, returns) == pytest.approx(1.0)

    def test_near_zero_return_variance_is_handled(self):
        returns = torch.ones(3)
        pred = torch.zeros(3)
        assert explained_variance(pred, returns) == 0.0


class TestMGDASolve:

    def test_identical_gradients_equal_weights(self):
        g = torch.ones(10)
        w1, w2 = mgda_solve(g, g)
        assert abs(w1 - 0.5) < 0.1 and abs(w2 - 0.5) < 0.1

    def test_weights_sum_to_one(self):
        g1 = torch.randn(20)
        g2 = torch.randn(20)
        w1, w2 = mgda_solve(g1, g2)
        assert abs(w1 + w2 - 1.0) < 1e-6
        assert w1 >= 0.0 and w2 >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

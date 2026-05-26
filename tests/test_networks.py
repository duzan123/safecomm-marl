"""SafeCommNetworks 接口和形状测试"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import pytest
from algorithms.networks import SafeCommNetworks


def make_nets(n=4, obs_dim=30, msg_dim=32, hidden=[64,64]):
    return SafeCommNetworks(n_agents=n, obs_dim=obs_dim, act_dim=3,
                            msg_dim=msg_dim, hidden_dims=hidden, H=2)


class TestSafeCommNetworks:

    def test_act_output_shapes(self):
        nets = make_nets()
        obs_list = [np.random.rand(30).astype(np.float32) for _ in range(4)]
        nbr = {i: [j for j in range(4) if j != i] for i in range(4)}
        actions, log_probs, values, cost_values = nets.act(obs_list, nbr, device="cpu")
        assert actions.shape == (4, 3)
        assert log_probs.shape == (4,)
        assert values.shape == (1,) or values.ndim <= 1
        assert cost_values.shape == (1,) or cost_values.ndim <= 1

    def test_actions_in_tanh_range(self):
        nets = make_nets()
        obs_list = [np.random.rand(30).astype(np.float32) for _ in range(4)]
        nbr = {i: [j for j in range(4) if j != i] for i in range(4)}
        actions, _, _, _ = nets.act(obs_list, nbr, deterministic=True)
        assert np.all(actions > -1.0 - 1e-5) and np.all(actions < 1.0 + 1e-5)

    def test_no_neighbors_still_works(self):
        nets = make_nets()
        obs_list = [np.random.rand(30).astype(np.float32) for _ in range(4)]
        nbr = {i: [] for i in range(4)}  # no communication
        actions, log_probs, _, _ = nets.act(obs_list, nbr, device="cpu")
        assert actions.shape == (4, 3)
        assert not np.any(np.isnan(actions))

    def test_evaluate_actions_batch_shapes(self):
        T, N, obs_dim = 8, 4, 30
        nets = make_nets(n=N, obs_dim=obs_dim)
        obs_b = torch.randn(T, N, obs_dim)
        act_b = torch.tanh(torch.randn(T, N, 3))
        adj_b = torch.ones(T, N, N) - torch.eye(N).unsqueeze(0)
        gs_b = torch.randn(T, N * obs_dim)
        log_probs, entropy, values, cost_values, agg = nets.evaluate_actions_batch(
            obs_b, act_b, adj_b, gs_b
        )
        assert log_probs.shape == (T, N)
        assert entropy.shape == (T, N)
        assert values.shape == (T,)
        assert cost_values.shape == (T,)

    def test_gradients_flow(self):
        nets = make_nets()
        T, N, obs_dim = 4, 4, 30
        obs_b = torch.randn(T, N, obs_dim)
        act_b = torch.tanh(torch.randn(T, N, 3))
        adj_b = torch.ones(T, N, N) - torch.eye(N).unsqueeze(0)
        gs_b = torch.randn(T, N * obs_dim)
        log_probs, _, values, _, _ = nets.evaluate_actions_batch(obs_b, act_b, adj_b, gs_b)
        loss = -log_probs.mean() + values.mean()
        loss.backward()
        grad_found = any(p.grad is not None for p in nets.parameters())
        assert grad_found

    def test_deterministic_vs_stochastic(self):
        nets = make_nets()
        nets.eval()
        obs_list = [np.random.rand(30).astype(np.float32) for _ in range(4)]
        nbr = {i: [j for j in range(4) if j != i] for i in range(4)}
        a1, _, _, _ = nets.act(obs_list, nbr, deterministic=True)
        a2, _, _, _ = nets.act(obs_list, nbr, deterministic=True)
        np.testing.assert_allclose(a1, a2, atol=1e-5)

    def test_log_sigma_is_single_shared_parameter(self):
        nets = make_nets()
        assert isinstance(nets.actor.log_sigma, torch.nn.Parameter)
        assert nets.actor.log_sigma.shape == (3,)
        named_log_sigma = [name for name, _ in nets.named_parameters() if name.endswith("log_sigma")]
        assert named_log_sigma == ["actor.log_sigma"]

    def test_get_values_pads_and_trims_global_state(self):
        nets = make_nets(n=4, obs_dim=30)
        short_gs = torch.randn(2, 117)
        long_gs = torch.randn(2, 125)

        short_values, short_costs = nets.get_values(short_gs)
        long_values, long_costs = nets.get_values(long_gs)

        assert short_values.shape == (2,)
        assert short_costs.shape == (2,)
        assert long_values.shape == (2,)
        assert long_costs.shape == (2,)

    def test_batch_no_neighbors_returns_zero_aggregate(self):
        T, N, obs_dim = 3, 4, 30
        nets = make_nets(n=N, obs_dim=obs_dim)
        obs_b = torch.randn(T, N, obs_dim)
        act_b = torch.tanh(torch.randn(T, N, 3))
        adj_b = torch.zeros(T, N, N)
        gs_b = torch.randn(T, N * obs_dim)

        _, _, _, _, agg = nets.evaluate_actions_batch(obs_b, act_b, adj_b, gs_b)

        assert agg.shape == (T, N, nets.msg_dim)
        assert torch.allclose(agg, torch.zeros_like(agg))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

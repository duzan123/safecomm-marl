import math
import torch
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.networks import (
    MessageEncoder,
    MultiHeadAttentionAggregator,
    Actor,
    Critic,
    CostCritic,
    SafeCommNetworks,
)


def test_message_encoder_shape():
    enc = MessageEncoder(obs_dim=18, msg_dim=64)
    x = torch.randn(4, 18)
    out = enc(x)
    assert out.shape == (4, 64)


def test_multi_head_aggregator_with_neighbors():
    agg = MultiHeadAttentionAggregator(d_msg=64, H=2)
    query = torch.randn(64)
    neighbors = torch.randn(3, 64)
    out = agg(query, neighbors)
    assert out.shape == (64,)


def test_multi_head_aggregator_no_neighbors():
    agg = MultiHeadAttentionAggregator(d_msg=64, H=2)
    query = torch.randn(64)
    neighbors = torch.zeros(0, 64)
    out = agg(query, neighbors)
    assert out.shape == (64,)
    assert torch.all(out == 0)


def test_multi_head_aggregator_one_neighbor():
    agg = MultiHeadAttentionAggregator(d_msg=64, H=2)
    query = torch.randn(64)
    neighbor = torch.randn(1, 64)
    out = agg(query, neighbor)
    assert out.shape == (64,)


def test_h2_d_head():
    """d_head should be d_msg // H."""
    agg = MultiHeadAttentionAggregator(d_msg=64, H=2)
    assert agg.d_head == 32
    assert agg.H == 2


def test_h1_differs_from_h2():
    """H=2 and H=1 aggregators produce different outputs (different init)."""
    torch.manual_seed(42)
    agg2 = MultiHeadAttentionAggregator(d_msg=64, H=2)
    agg1 = MultiHeadAttentionAggregator(d_msg=64, H=1)
    query = torch.randn(64)
    neighbors = torch.randn(3, 64)
    out2 = agg2(query, neighbors)
    out1 = agg1(query, neighbors)
    assert not torch.allclose(out2, out1)


def test_multi_head_invalid_dims():
    """d_msg not divisible by H should raise AssertionError."""
    with pytest.raises(AssertionError):
        MultiHeadAttentionAggregator(d_msg=64, H=3)


def test_actor_output_shape():
    actor = Actor(obs_dim=18, act_dim=3, msg_dim=64, hidden_dims=[64, 64])
    obs = torch.randn(4, 18)
    agg = torch.randn(4, 64)
    dist = actor(obs, agg)
    actions, log_probs = actor.get_action(obs, agg)
    assert actions.shape == (4, 3)
    assert log_probs.shape == (4,)


def test_critic_output_shape():
    critic = Critic(global_state_dim=4 * 18, hidden_dims=[64, 64])
    global_obs = torch.randn(8, 72)
    out = critic(global_obs)
    assert out.shape == (8,)


def test_cost_critic_output_shape():
    cc = CostCritic(global_state_dim=4 * 18, hidden_dims=[64, 64])
    global_obs = torch.randn(8, 72)
    out = cc(global_obs)
    assert out.shape == (8,)


def test_safe_comm_networks_h2():
    """SafeCommNetworks with H=2 should use MultiHeadAttentionAggregator."""
    nets = SafeCommNetworks(obs_dim=18, act_dim=3, n_agents=4, msg_dim=64, H=2, hidden_dims=[64, 64])
    assert isinstance(nets.aggregator, MultiHeadAttentionAggregator)
    assert nets.aggregator.H == 2


def test_safe_comm_networks_h1():
    """SafeCommNetworks with H=1 should use single-head MessageAggregator."""
    from algorithms.networks import MessageAggregator
    nets = SafeCommNetworks(obs_dim=18, act_dim=3, n_agents=4, msg_dim=64, H=1, hidden_dims=[64, 64])
    assert isinstance(nets.aggregator, MessageAggregator)

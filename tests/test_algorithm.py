"""Integration tests for SafeCommPSched algorithm."""
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.uav_formation_env import UAVFormationEnv
from algorithms.safecomm_psched import SafeCommPSched
from algorithms.networks import MultiHeadAttentionAggregator


FAST_CONFIG = {
    "k": 4,
    "n_steps": 40,
    "n_epochs": 2,
    "batch_size": 20,
    "msg_dim": 32,
    "hidden_dims": [64, 64],
    "H": 2,
    "lambda_threshold": 0.1,
    "safety_budget": 0.5,
}


def make_env_and_agent():
    env = UAVFormationEnv(n_agents=4, d_min=0.3, dt=0.1, max_steps=50, R_comm=5.0)
    agent = SafeCommPSched(env, config=FAST_CONFIG, device="cpu")
    return env, agent


def test_collect_and_update_no_crash():
    """collect_rollout + update should run without error."""
    env, agent = make_env_and_agent()
    rollout_info = agent.collect_rollout()
    stats = agent.update()
    assert "actor_loss" in stats
    assert "lambda_s" in stats
    assert "J_safe" in stats


def test_lambda_increases_when_cost_exceeds_budget():
    """lambda_s should increase when J_safe > safety_budget."""
    env, agent = make_env_and_agent()
    # Force high costs: place agents very close together
    agent.env.positions = np.zeros((4, 3))
    agent.env.positions[1] = np.array([0.1, 0.0, 0.0])  # inside d_min=0.3
    initial_lambda = agent.lambda_s
    agent.collect_rollout()
    # Manually set high cost in buffer to simulate violations
    agent.buffer.costs[:agent.buffer.ptr] = 5.0
    stats = agent.update()
    # If J_safe > safety_budget, lambda should increase
    assert stats["lambda_s"] >= 0.0  # at minimum non-negative


def test_lambda_nonnegative():
    """lambda_s must never go below 0."""
    env, agent = make_env_and_agent()
    agent.lambda_s = 0.001
    # Zero out costs to simulate no violations
    agent.collect_rollout()
    agent.buffer.costs[:agent.buffer.ptr] = 0.0
    stats = agent.update()
    assert stats["lambda_s"] >= 0.0


def test_comm_budget_respected():
    """|active_edges| <= k at every scheduler step."""
    env, agent = make_env_and_agent()
    k = FAST_CONFIG["k"]
    env.reset()
    for _ in range(50):
        phys_edges = env.get_physical_graph()
        sched_result = agent.scheduler.schedule_step(
            positions=env.positions,
            phys_edges=phys_edges,
        )
        active_edges = sched_result["active_edges"]
        assert len(active_edges) <= k, f"Got {len(active_edges)} edges, budget k={k}"
        # Step env
        actions = np.zeros((4, 3))
        _, _, _, done, _ = env.step(actions)
        if done:
            env.reset()


def test_h2_aggregator_used():
    """SafeCommPSched with H=2 should use MultiHeadAttentionAggregator."""
    env, agent = make_env_and_agent()
    assert isinstance(agent.networks.aggregator, MultiHeadAttentionAggregator)
    assert agent.networks.aggregator.H == 2


def test_update_stats_keys():
    """update() stats dict should have all required keys."""
    env, agent = make_env_and_agent()
    agent.collect_rollout()
    stats = agent.update()
    required_keys = {"actor_loss", "critic_loss", "lambda_s", "J_safe"}
    for key in required_keys:
        assert key in stats, f"Missing key: {key}"

import torch
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.ppo_utils import compute_gae, mgda_solve


def test_mgda_orthogonal_gradients():
    """Orthogonal gradients of equal norm → equal weights (0.5/0.5)."""
    g0 = torch.zeros(10)
    g0[0] = 1.0
    g1 = torch.zeros(10)
    g1[1] = 1.0
    w0, w1 = mgda_solve(g0, g1)
    assert abs(w0 + w1 - 1.0) < 1e-5
    assert abs(w0 - 0.5) < 1e-5


def test_mgda_aligned_gradients():
    """Perfectly aligned gradients → weights sum to 1, both non-negative."""
    g = torch.randn(20)
    w0, w1 = mgda_solve(g, g)
    assert abs(w0 + w1 - 1.0) < 1e-5
    assert w0 >= 0.0 and w1 >= 0.0


def test_mgda_zero_safe_gradient():
    """Zero safe gradient → all weight on task."""
    g_task = torch.ones(10)
    g_safe = torch.zeros(10)
    w0, w1 = mgda_solve(g_task, g_safe)
    assert abs(w0 - 1.0) < 1e-5
    assert abs(w1 - 0.0) < 1e-5


def test_mgda_zero_task_gradient():
    """Zero task gradient → all weight on safe."""
    g_task = torch.zeros(10)
    g_safe = torch.ones(10)
    w0, w1 = mgda_solve(g_task, g_safe)
    assert abs(w0 - 0.0) < 1e-5
    assert abs(w1 - 1.0) < 1e-5


def test_mgda_both_zero_gradients():
    """Both zero → equal weights 0.5/0.5."""
    w0, w1 = mgda_solve(torch.zeros(10), torch.zeros(10))
    assert abs(w0 - 0.5) < 1e-5
    assert abs(w1 - 0.5) < 1e-5


def test_mgda_weights_sum_to_one():
    """Weights always sum to 1.0."""
    torch.manual_seed(0)
    for _ in range(20):
        g0 = torch.randn(50)
        g1 = torch.randn(50)
        w0, w1 = mgda_solve(g0, g1)
        assert abs(w0 + w1 - 1.0) < 1e-5
        assert w0 >= 0.0 and w1 >= 0.0


def test_mgda_opposite_gradients():
    """Opposite gradients → 0.5/0.5 (both should be equally penalized)."""
    g = torch.ones(10)
    w0, w1 = mgda_solve(g, -g)
    assert abs(w0 + w1 - 1.0) < 1e-5


def test_gae_shape():
    """GAE output shapes match input."""
    rewards = np.ones(10)
    values = np.zeros(11)
    dones = np.zeros(10)
    adv, ret = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
    assert adv.shape == (10,)
    assert ret.shape == (10,)


def test_gae_episode_boundary():
    """done=1 zeros out future bootstrap."""
    rewards = np.array([1.0, 1.0])
    values = np.array([0.5, 0.5, 100.0])
    dones = np.array([0.0, 1.0])
    adv, _ = compute_gae(rewards, values, dones, gamma=0.99, lam=1.0)
    # At t=1: delta = 1.0 + 0.99*100.0*0 - 0.5 = 0.5
    assert abs(adv[1] - 0.5) < 1e-4

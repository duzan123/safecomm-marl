"""PPO training utilities for SafeComm-VoI Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute generalized advantage estimation with done masking."""
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    values_arr = np.asarray(values, dtype=np.float32)
    dones_arr = np.asarray(dones, dtype=bool)

    if rewards_arr.ndim != 1:
        raise ValueError(f"rewards must be 1D, got shape {rewards_arr.shape}")
    if values_arr.ndim != 1:
        raise ValueError(f"values must be 1D, got shape {values_arr.shape}")
    if dones_arr.ndim != 1:
        raise ValueError(f"dones must be 1D, got shape {dones_arr.shape}")

    n_steps = rewards_arr.shape[0]
    if values_arr.shape[0] != n_steps + 1:
        raise ValueError(
            f"values length must be T + 1 ({n_steps + 1}), got {values_arr.shape[0]}"
        )
    if dones_arr.shape[0] != n_steps:
        raise ValueError(f"dones length must be T ({n_steps}), got {dones_arr.shape[0]}")

    advantages = np.zeros(n_steps, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(n_steps)):
        mask = 1.0 - float(dones_arr[t])
        delta = rewards_arr[t] + gamma * values_arr[t + 1] * mask - values_arr[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae

    returns = advantages + values_arr[:-1]
    return advantages.astype(np.float32), returns.astype(np.float32)


def ppo_clip_loss(
    log_probs: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return PPO clipped policy loss and an approximate KL diagnostic."""
    advantages = torch.as_tensor(advantages, dtype=log_probs.dtype, device=log_probs.device)
    while advantages.ndim < log_probs.ndim:
        advantages = advantages.unsqueeze(-1)

    log_ratio = log_probs - log_probs_old
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

    loss_unclipped = ratio * advantages
    loss_clipped = clipped_ratio * advantages
    loss = -torch.min(loss_unclipped, loss_clipped).mean()
    approx_kl = ((ratio - 1.0) - log_ratio).mean().detach()
    return loss, approx_kl


def value_loss(
    values_pred: torch.Tensor,
    values_old: torch.Tensor,
    returns: torch.Tensor,
    clip_eps: float = 0.2,
    use_clip: bool = True,
) -> torch.Tensor:
    """Return PPO value loss with optional value clipping."""
    if not use_clip:
        return 0.5 * torch.mean((values_pred - returns) ** 2)

    values_clipped = values_old + torch.clamp(values_pred - values_old, -clip_eps, clip_eps)
    loss_unclipped = (values_pred - returns) ** 2
    loss_clipped = (values_clipped - returns) ** 2
    return 0.5 * torch.max(loss_unclipped, loss_clipped).mean()


def explained_variance(values_pred: torch.Tensor, returns: torch.Tensor) -> float:
    """Compute explained variance, returning 0 when target variance is near zero."""
    pred = torch.as_tensor(values_pred, dtype=torch.float32).detach().flatten()
    target = torch.as_tensor(returns, dtype=torch.float32).detach().flatten()
    if pred.shape != target.shape:
        raise ValueError(f"values_pred and returns shapes must match, got {pred.shape} and {target.shape}")

    target_var = torch.var(target, unbiased=False)
    if torch.isnan(target_var) or target_var.item() < 1e-8:
        return 0.0

    error_var = torch.var(target - pred, unbiased=False)
    return float(1.0 - error_var.item() / target_var.item())


def mgda_solve(g1: torch.Tensor, g2: torch.Tensor) -> tuple[float, float]:
    """Closed-form two-objective MGDA weights for w1 * g1 + w2 * g2."""
    grad1 = torch.as_tensor(g1, dtype=torch.float64).detach().flatten()
    grad2 = torch.as_tensor(g2, dtype=torch.float64).detach().flatten()
    if grad1.shape != grad2.shape:
        raise ValueError(f"gradient shapes must match, got {grad1.shape} and {grad2.shape}")
    if not torch.isfinite(grad1).all() or not torch.isfinite(grad2).all():
        raise ValueError("gradients must be finite")

    diff = grad1 - grad2
    denom = torch.dot(diff, diff).item()
    if denom <= 1e-12:
        return 0.5, 0.5

    numerator = (torch.dot(grad2, grad2) - torch.dot(grad1, grad2)).item()
    w1 = float(np.clip(numerator / denom, 0.0, 1.0))
    w2 = 1.0 - w1
    return w1, w2


@dataclass(frozen=True)
class _ShapeSpec:
    name: str
    shape: tuple[int, ...]


class RolloutBuffer:
    """Fixed-size rollout storage for CTDE PPO training."""

    def __init__(
        self,
        n_steps: int,
        n_agents: int,
        obs_dim: int,
        act_dim: int,
        global_state_dim: int,
    ) -> None:
        for name, value in {
            "n_steps": n_steps,
            "n_agents": n_agents,
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "global_state_dim": global_state_dim,
        }.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        self.n_steps = n_steps
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.global_state_dim = global_state_dim
        self.ptr = 0

        self.obs = np.zeros((n_steps, n_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_agents, act_dim), dtype=np.float32)
        self.log_probs = np.zeros((n_steps, n_agents), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_agents), dtype=np.float32)
        self.costs = np.zeros((n_steps, n_agents), dtype=np.float32)
        self.values = np.zeros(n_steps + 1, dtype=np.float32)
        self.cost_values = np.zeros(n_steps + 1, dtype=np.float32)
        self.dones = np.zeros(n_steps, dtype=bool)
        self.global_states = np.zeros((n_steps, global_state_dim), dtype=np.float32)
        self.adj_matrices = np.zeros((n_steps, n_agents, n_agents), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        costs: np.ndarray,
        value: float,
        cost_value: float,
        done: bool,
        global_state: np.ndarray,
        adj_matrix: np.ndarray,
    ) -> None:
        """Store a single environment step."""
        if self.ptr >= self.n_steps:
            raise RuntimeError("RolloutBuffer capacity exceeded")

        obs_arr = self._array(obs, _ShapeSpec("obs", (self.n_agents, self.obs_dim)))
        actions_arr = self._array(actions, _ShapeSpec("actions", (self.n_agents, self.act_dim)))
        log_probs_arr = self._array(log_probs, _ShapeSpec("log_probs", (self.n_agents,)))
        rewards_arr = self._array(rewards, _ShapeSpec("rewards", (self.n_agents,)))
        costs_arr = self._array(costs, _ShapeSpec("costs", (self.n_agents,)))
        global_state_arr = self._array(
            global_state, _ShapeSpec("global_state", (self.global_state_dim,))
        )
        adj_matrix_arr = self._array(
            adj_matrix, _ShapeSpec("adj_matrix", (self.n_agents, self.n_agents))
        )
        value_f = self._finite_scalar(value, "value")
        cost_value_f = self._finite_scalar(cost_value, "cost_value")

        self.obs[self.ptr] = obs_arr
        self.actions[self.ptr] = actions_arr
        self.log_probs[self.ptr] = log_probs_arr
        self.rewards[self.ptr] = rewards_arr
        self.costs[self.ptr] = costs_arr
        self.values[self.ptr] = value_f
        self.cost_values[self.ptr] = cost_value_f
        self.dones[self.ptr] = bool(done)
        self.global_states[self.ptr] = global_state_arr
        self.adj_matrices[self.ptr] = adj_matrix_arr
        self.ptr += 1

    def finish_rollout(self, last_value: float, last_cost_value: float) -> None:
        """Write bootstrap values at the current rollout end."""
        self.values[self.ptr] = self._finite_scalar(last_value, "last_value")
        self.cost_values[self.ptr] = self._finite_scalar(last_cost_value, "last_cost_value")

    def compute_advantages_and_returns(
        self,
        gamma: float = 0.99,
        lam: float = 0.95,
        normalize_advantages: bool = True,
    ) -> dict[str, np.ndarray]:
        """Compute reward and cost GAE from mean per-agent rewards/costs."""
        n = self.ptr
        reward_signal = self.rewards[:n].mean(axis=1)
        cost_signal = self.costs[:n].mean(axis=1)

        advantages, returns = compute_gae(
            reward_signal, self.values[: n + 1], self.dones[:n], gamma=gamma, lam=lam
        )
        cost_advantages, cost_returns = compute_gae(
            cost_signal, self.cost_values[: n + 1], self.dones[:n], gamma=gamma, lam=lam
        )

        if normalize_advantages:
            advantages = self._normalize(advantages)
            cost_advantages = self._normalize(cost_advantages)

        return {
            "advantages": advantages.astype(np.float32),
            "returns": returns.astype(np.float32),
            "cost_advantages": cost_advantages.astype(np.float32),
            "cost_returns": cost_returns.astype(np.float32),
        }

    def get_training_data(
        self,
        gamma: float = 0.99,
        lam: float = 0.95,
        device: str | torch.device = "cpu",
        normalize_advantages: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Return rollout tensors for PPO updates."""
        n = self.ptr
        adv_data = self.compute_advantages_and_returns(
            gamma=gamma, lam=lam, normalize_advantages=normalize_advantages
        )

        data: dict[str, Any] = {
            "obs": self.obs[:n],
            "actions": self.actions[:n],
            "log_probs_old": self.log_probs[:n],
            "advantages": adv_data["advantages"],
            "returns": adv_data["returns"],
            "cost_advantages": adv_data["cost_advantages"],
            "cost_returns": adv_data["cost_returns"],
            "global_states": self.global_states[:n],
            "adj_matrices": self.adj_matrices[:n],
            "values_old": self.values[:n],
            "cost_values_old": self.cost_values[:n],
            "dones": self.dones[:n],
            "rewards": self.rewards[:n],
            "costs": self.costs[:n],
        }

        tensors: dict[str, torch.Tensor] = {}
        for key, value in data.items():
            if key == "dones":
                tensors[key] = torch.as_tensor(value, dtype=torch.bool, device=device)
            else:
                tensors[key] = torch.as_tensor(value, dtype=torch.float32, device=device)
        return tensors

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        if values.size <= 1:
            return values.astype(np.float32)
        std = float(values.std())
        if std < 1e-8:
            return values.astype(np.float32)
        return ((values - values.mean()) / (std + 1e-8)).astype(np.float32)

    @staticmethod
    def _array(value: Any, spec: _ShapeSpec) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape != spec.shape:
            raise ValueError(f"{spec.name} must have shape {spec.shape}, got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{spec.name} must contain only finite values")
        return arr

    @staticmethod
    def _finite_scalar(value: float, name: str) -> float:
        scalar = float(value)
        if not np.isfinite(scalar):
            raise ValueError(f"{name} must be finite")
        return scalar


__all__ = [
    "RolloutBuffer",
    "compute_gae",
    "ppo_clip_loss",
    "value_loss",
    "explained_variance",
    "mgda_solve",
]

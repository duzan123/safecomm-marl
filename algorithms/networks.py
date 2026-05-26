"""Neural network modules for SafeComm-VoI Phase 1."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dims: Sequence[int],
    activation: type[nn.Module] = nn.Tanh,
) -> nn.Sequential:
    """Build a feed-forward MLP with the project's small-network defaults."""
    layers: list[nn.Module] = []
    prev_dim = in_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation())
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, out_dim))
    return nn.Sequential(*layers)


class MessageEncoder(nn.Module):
    """Shared two-layer message encoder: observation -> message embedding."""

    def __init__(self, obs_dim: int, msg_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, msg_dim),
            nn.ReLU(),
            nn.Linear(msg_dim, msg_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class GATAggregator(nn.Module):
    """Masked multi-head attention aggregator over active communication edges."""

    def __init__(self, obs_dim: int, msg_dim: int, H: int = 2) -> None:
        super().__init__()
        if H <= 0:
            raise ValueError("H must be a positive integer")

        self.obs_dim = obs_dim
        self.msg_dim = msg_dim
        self.H = int(H)
        self.attn_dim = int(math.ceil(msg_dim / self.H) * self.H)
        self.head_dim = self.attn_dim // self.H

        self.query_proj = nn.Linear(obs_dim, self.attn_dim, bias=False)
        self.key_proj = nn.Linear(msg_dim, self.attn_dim, bias=False)
        self.value_proj = nn.Linear(msg_dim, self.attn_dim, bias=False)
        self.out_proj = nn.Linear(self.attn_dim, msg_dim, bias=False)

    def forward(
        self,
        own_obs: torch.Tensor,
        messages: torch.Tensor,
        adj_matrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Aggregate messages.

        With ``adj_matrix`` this accepts batched graph inputs:
        ``own_obs[..., N, obs_dim]``, ``messages[..., N, msg_dim]``,
        ``adj_matrix[..., N, N]`` and returns ``[..., N, msg_dim]``.

        Without ``adj_matrix`` it accepts per-node neighbor messages:
        ``own_obs[B, obs_dim]``, ``messages[B, K, msg_dim]`` and returns
        ``[B, msg_dim]``. ``K == 0`` returns exact zeros.
        """
        if adj_matrix is None:
            return self._aggregate_neighbor_messages(own_obs, messages)
        return self._aggregate_adjacency(own_obs, messages, adj_matrix)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(*tensor.shape[:-1], self.H, self.head_dim)

    def _aggregate_neighbor_messages(
        self,
        own_obs: torch.Tensor,
        neighbor_msgs: torch.Tensor,
    ) -> torch.Tensor:
        if neighbor_msgs.shape[-2] == 0:
            return own_obs.new_zeros(*own_obs.shape[:-1], self.msg_dim)

        q = self._split_heads(self.query_proj(own_obs))  # [B, H, D]
        k = self._split_heads(self.key_proj(neighbor_msgs))  # [B, K, H, D]
        v = self._split_heads(self.value_proj(neighbor_msgs))  # [B, K, H, D]

        scores = torch.einsum("bhd,bkhd->bhk", q, k) / math.sqrt(self.head_dim)
        weights = torch.softmax(scores, dim=-1)
        context = torch.einsum("bhk,bkhd->bhd", weights, v).reshape(
            *own_obs.shape[:-1], self.attn_dim
        )
        return self.out_proj(context)

    def _aggregate_adjacency(
        self,
        own_obs: torch.Tensor,
        messages: torch.Tensor,
        adj_matrix: torch.Tensor,
    ) -> torch.Tensor:
        q = self._split_heads(self.query_proj(own_obs))  # [..., N, H, D]
        k = self._split_heads(self.key_proj(messages))  # [..., N, H, D]
        v = self._split_heads(self.value_proj(messages))  # [..., N, H, D]

        scores = torch.einsum("...ihd,...jhd->...hij", q, k) / math.sqrt(self.head_dim)
        mask = adj_matrix.to(dtype=torch.bool, device=scores.device).unsqueeze(-3)
        masked_scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        weights = torch.softmax(masked_scores, dim=-1) * mask.to(dtype=scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        context = torch.einsum("...hij,...jhd->...ihd", weights, v).reshape(
            *messages.shape[:-1], self.attn_dim
        )
        return self.out_proj(context)


class Actor(nn.Module):
    """Shared Gaussian policy network with tanh-squashed actions."""

    def __init__(
        self,
        obs_dim: int,
        msg_dim: int,
        act_dim: int,
        hidden_dims: Sequence[int],
    ) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim + msg_dim, act_dim, hidden_dims)
        self.log_sigma = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor, agg_msg: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, agg_msg], dim=-1))

    def get_dist(self, obs: torch.Tensor, agg_msg: torch.Tensor) -> torch.distributions.Normal:
        mean = self.forward(obs, agg_msg)
        std = self.log_sigma.clamp(min=-5.0, max=2.0).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)


class Critic(nn.Module):
    """Centralized task-value critic over concatenated observations."""

    def __init__(self, global_state_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        self.net = build_mlp(global_state_dim, 1, hidden_dims)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state).squeeze(-1)


class CostCritic(nn.Module):
    """Centralized safety-cost critic over concatenated observations."""

    def __init__(self, global_state_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        self.net = build_mlp(global_state_dim, 1, hidden_dims)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state).squeeze(-1)


class SafeCommNetworks(nn.Module):
    """Container for shared actor, message modules, and centralized critics."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        act_dim: int,
        msg_dim: int,
        hidden_dims: Sequence[int],
        H: int = 2,
    ) -> None:
        super().__init__()
        for name, value in {
            "n_agents": n_agents,
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "msg_dim": msg_dim,
        }.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        self.n_agents = n_agents
        self.n = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.msg_dim = msg_dim
        self.global_state_dim = n_agents * obs_dim

        self.encoder = MessageEncoder(obs_dim, msg_dim)
        self.aggregator = GATAggregator(obs_dim, msg_dim, H=H)
        self.actor = Actor(obs_dim, msg_dim, act_dim, hidden_dims)
        self.critic = Critic(self.global_state_dim, hidden_dims)
        self.cost_critic = CostCritic(self.global_state_dim, hidden_dims)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _as_float_tensor(
        self,
        value: torch.Tensor | np.ndarray | Sequence[float],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        target_device = self.device if device is None else device
        if isinstance(value, torch.Tensor):
            return value.to(device=target_device, dtype=torch.float32)
        return torch.as_tensor(value, dtype=torch.float32, device=target_device)

    def _pad_global_state(self, global_state: torch.Tensor) -> torch.Tensor:
        """Pad or trim the last dimension to the configured critic input width."""
        if global_state.shape[-1] == self.global_state_dim:
            return global_state
        if global_state.shape[-1] > self.global_state_dim:
            return global_state[..., : self.global_state_dim]

        pad_shape = (*global_state.shape[:-1], self.global_state_dim - global_state.shape[-1])
        return torch.cat([global_state, global_state.new_zeros(pad_shape)], dim=-1)

    def _prepare_global_states(
        self,
        global_states: torch.Tensor | np.ndarray | Sequence[float],
    ) -> torch.Tensor:
        gs = self._as_float_tensor(global_states)
        if gs.ndim == 1:
            gs = gs.unsqueeze(0)
        elif gs.ndim > 2:
            gs = gs.reshape(gs.shape[0], -1)
        return self._pad_global_state(gs)

    def _neighbors_to_adj(
        self,
        neighbor_lists: Mapping[int, Sequence[int]],
        n_agents: int,
        device: torch.device,
    ) -> torch.Tensor:
        adj = torch.zeros(n_agents, n_agents, dtype=torch.float32, device=device)
        for i in range(n_agents):
            for j in neighbor_lists.get(i, ()):
                if 0 <= int(j) < n_agents and int(j) != i:
                    adj[i, int(j)] = 1.0
        return adj

    def _squashed_log_prob(
        self,
        dist: torch.distributions.Normal,
        raw_actions: torch.Tensor,
    ) -> torch.Tensor:
        log_prob = dist.log_prob(raw_actions).sum(dim=-1)
        correction = 2.0 * (
            math.log(2.0) - raw_actions - F.softplus(-2.0 * raw_actions)
        )
        return log_prob - correction.sum(dim=-1)

    def _atanh(self, actions: torch.Tensor) -> torch.Tensor:
        clamped = actions.clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
        return 0.5 * (torch.log1p(clamped) - torch.log1p(-clamped))

    @torch.no_grad()
    def act(
        self,
        obs_list: Sequence[np.ndarray],
        neighbor_lists: Mapping[int, Sequence[int]],
        deterministic: bool = False,
        device: str | torch.device = "cpu",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample or choose one action per agent and evaluate centralized critics."""
        target_device = torch.device(device)
        self.to(target_device)

        obs_np = np.asarray(obs_list, dtype=np.float32)
        if obs_np.ndim != 2 or obs_np.shape[1] != self.obs_dim:
            raise ValueError(f"obs_list must have shape [N, {self.obs_dim}], got {obs_np.shape}")

        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=target_device)
        n_agents = obs.shape[0]
        messages = self.encoder(obs)
        adj = self._neighbors_to_adj(neighbor_lists, n_agents, target_device)
        agg_msgs = self.aggregator(obs.unsqueeze(0), messages.unsqueeze(0), adj.unsqueeze(0)).squeeze(0)

        dist = self.actor.get_dist(obs, agg_msgs)
        raw_actions = dist.mean if deterministic else dist.rsample()
        actions = torch.tanh(raw_actions)
        log_probs = self._squashed_log_prob(dist, raw_actions)

        global_state = self._pad_global_state(obs.reshape(1, -1))
        values = self.critic(global_state)
        cost_values = self.cost_critic(global_state)

        return (
            actions.detach().cpu().numpy(),
            log_probs.detach().cpu().numpy(),
            values.detach().cpu().numpy(),
            cost_values.detach().cpu().numpy(),
        )

    def evaluate_actions_batch(
        self,
        obs_batch: torch.Tensor,
        actions_batch: torch.Tensor,
        adj_matrices: torch.Tensor,
        global_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate tanh-squashed actions for PPO updates."""
        obs_batch = self._as_float_tensor(obs_batch)
        actions_batch = self._as_float_tensor(actions_batch)
        adj_matrices = self._as_float_tensor(adj_matrices)
        global_states = self._as_float_tensor(global_states)

        if obs_batch.ndim != 3 or obs_batch.shape[-1] != self.obs_dim:
            raise ValueError(
                f"obs_batch must have shape [T, N, {self.obs_dim}], got {tuple(obs_batch.shape)}"
            )
        if actions_batch.shape[:2] != obs_batch.shape[:2] or actions_batch.shape[-1] != self.act_dim:
            raise ValueError(
                "actions_batch must have shape [T, N, act_dim] matching obs_batch, "
                f"got {tuple(actions_batch.shape)}"
            )
        if adj_matrices.shape != obs_batch.shape[:2] + obs_batch.shape[1:2]:
            raise ValueError(
                "adj_matrices must have shape [T, N, N] matching obs_batch, "
                f"got {tuple(adj_matrices.shape)}"
            )

        t_steps, n_agents, _ = obs_batch.shape
        obs_flat = obs_batch.reshape(t_steps * n_agents, self.obs_dim)
        messages = self.encoder(obs_flat).reshape(t_steps, n_agents, self.msg_dim)
        agg_msgs = self.aggregator(obs_batch, messages, adj_matrices)

        dist = self.actor.get_dist(
            obs_flat,
            agg_msgs.reshape(t_steps * n_agents, self.msg_dim),
        )
        raw_actions = self._atanh(actions_batch.reshape(t_steps * n_agents, self.act_dim))
        log_probs = self._squashed_log_prob(dist, raw_actions).reshape(t_steps, n_agents)
        entropy = dist.entropy().sum(dim=-1).reshape(t_steps, n_agents)

        critic_input = self._prepare_global_states(global_states)
        values = self.critic(critic_input)
        cost_values = self.cost_critic(critic_input)

        return log_probs, entropy, values, cost_values, agg_msgs

    def get_values(
        self,
        global_states: torch.Tensor | np.ndarray | Sequence[float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        critic_input = self._prepare_global_states(global_states)
        return self.critic(critic_input), self.cost_critic(critic_input)

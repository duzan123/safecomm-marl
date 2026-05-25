"""
SafeComm-PSched 神经网络模块

网络架构（参见设计规格 §3.2）：

  MessageEncoder（共享参数）：
    enc: R^{d_obs} → R^{d_msg}    2层 MLP

  MessageAggregator（单头注意力）：
    agg_i = Σ_{j∈N_i} α_ij · enc_j(o_j)
    α_ij = softmax(q_i^T k_j / √d_msg)

  Actor（共享参数）：
    π_i(· | [o_i; agg_i]) : R^{d_obs+d_msg} → 正态分布 N(μ, σ²)

  Critic（集中式，训练时用）：
    V_θ(s₁,...,sₙ) → R    任务价值

  CostCritic（集中式，训练时用）：
    V_φ^c(s₁,...,sₙ) → R    安全成本价值

所有 MLP 使用 2 层 [256, 256] 隐藏层 + Tanh 激活。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, List, Optional, Tuple


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int],
    activation: str = "tanh",
    output_activation: Optional[str] = None,
) -> nn.Sequential:
    """
    构建多层感知机（MLP）。

    Args:
        input_dim: 输入维度
        output_dim: 输出维度
        hidden_dims: 隐藏层维度列表，例如 [256, 256]
        activation: 隐藏层激活函数（"tanh" | "relu" | "elu"）
        output_activation: 输出层激活函数（None = 线性）

    Returns:
        nn.Sequential MLP 模块
    """
    act_fn = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
    }[activation]

    layers = []
    in_dim = input_dim
    for h_dim in hidden_dims:
        layers.extend([nn.Linear(in_dim, h_dim), act_fn()])
        in_dim = h_dim

    layers.append(nn.Linear(in_dim, output_dim))

    if output_activation == "tanh":
        layers.append(nn.Tanh())
    elif output_activation == "relu":
        layers.append(nn.ReLU())
    elif output_activation == "softplus":
        layers.append(nn.Softplus())

    return nn.Sequential(*layers)


# ==============================================================================
# MessageEncoder：消息编码器（参数共享）
# ==============================================================================

class MessageEncoder(nn.Module):
    """
    将智能体的本地观测编码为消息向量。

    enc: R^{d_obs} → R^{d_msg}    2层 MLP，参数共享
    """

    def __init__(
        self,
        obs_dim: int,
        msg_dim: int = 64,
        hidden_dims: List[int] = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        self.obs_dim = obs_dim
        self.msg_dim = msg_dim

        self.encoder = build_mlp(
            input_dim=obs_dim,
            output_dim=msg_dim,
            hidden_dims=hidden_dims,
            activation="tanh",
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: [..., obs_dim] 观测向量

        Returns:
            msg: [..., msg_dim] 消息向量
        """
        return self.encoder(obs)


# ==============================================================================
# MessageAggregator：单头注意力聚合
# ==============================================================================

class MessageAggregator(nn.Module):
    """
    单头注意力消息聚合。

    agg_i = Σ_{j∈N_i} α_ij · enc_j(o_j)
    α_ij = softmax(q_i^T k_j / √d_msg)

    Query/Key 投影使用线性层（参数共享）。
    """

    def __init__(self, msg_dim: int = 64):
        super().__init__()
        self.msg_dim = msg_dim
        self.scale = msg_dim ** (-0.5)

        # Query 和 Key 投影
        self.query_proj = nn.Linear(msg_dim, msg_dim, bias=False)
        self.key_proj = nn.Linear(msg_dim, msg_dim, bias=False)

    def forward(
        self,
        self_msg: torch.Tensor,
        neighbor_msgs: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            self_msg: [batch_size, msg_dim] 或 [msg_dim]，自身消息（用作 Query）
            neighbor_msgs: List of [batch_size, msg_dim] 或 [msg_dim]，邻居消息

        Returns:
            agg: 与 self_msg 同形状，聚合后的消息向量
                  若无邻居，返回全零向量
        """
        if len(neighbor_msgs) == 0:
            return torch.zeros_like(self_msg)

        # Stack: [n_neighbors, batch, msg_dim] 或 [n_neighbors, msg_dim]
        if self_msg.dim() == 1:
            # 非批量模式：[msg_dim]
            query = self.query_proj(self_msg)  # [msg_dim]
            keys = torch.stack([self.key_proj(m) for m in neighbor_msgs], dim=0)  # [n_nbr, msg_dim]
            # 注意力得分
            scores = torch.mv(keys, query) * self.scale  # [n_nbr]
            attn = F.softmax(scores, dim=0)  # [n_nbr]
            # 聚合
            msgs_stack = torch.stack(neighbor_msgs, dim=0)  # [n_nbr, msg_dim]
            agg = torch.einsum("n,nd->d", attn, msgs_stack)  # [msg_dim]
        else:
            # 批量模式：[batch, msg_dim]
            query = self.query_proj(self_msg)  # [batch, msg_dim]
            keys = torch.stack([self.key_proj(m) for m in neighbor_msgs], dim=1)  # [batch, n_nbr, msg_dim]
            # 注意力得分：[batch, n_nbr]
            scores = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1) * self.scale
            attn = F.softmax(scores, dim=-1)  # [batch, n_nbr]
            # 聚合
            msgs_stack = torch.stack(neighbor_msgs, dim=1)  # [batch, n_nbr, msg_dim]
            agg = torch.bmm(attn.unsqueeze(1), msgs_stack).squeeze(1)  # [batch, msg_dim]

        return agg


# ==============================================================================
# MultiHeadAttentionAggregator：H 头注意力聚合（v2.0）
# ==============================================================================

class MultiHeadAttentionAggregator(nn.Module):
    """
    H 头注意力消息聚合（v2.0，H=2）。

    每头维度 d_head = d_msg / H
    for h in range(H):
        q_h = W_q^h · enc_i       ∈ R^{d_head}
        K_h = W_k^h · {enc_j}     ∈ R^{|N_i| × d_head}
        V_h = W_v^h · {enc_j}     ∈ R^{|N_i| × d_head}
        α_h = softmax(q_h^T K_h / √d_head)
        head_h = Σ_j α_{h,j} · V_{h,j}
    agg_i = W_o · concat(head_0, ..., head_{H-1}) ∈ R^{d_msg}
    """

    def __init__(self, d_msg: int = 64, H: int = 2):
        super().__init__()
        assert d_msg % H == 0, f"d_msg={d_msg} must be divisible by H={H}"
        self.H = H
        self.d_msg = d_msg
        self.d_head = d_msg // H

        self.W_q = nn.ModuleList(
            [nn.Linear(d_msg, self.d_head, bias=False) for _ in range(H)]
        )
        self.W_k = nn.ModuleList(
            [nn.Linear(d_msg, self.d_head, bias=False) for _ in range(H)]
        )
        self.W_v = nn.ModuleList(
            [nn.Linear(d_msg, self.d_head, bias=False) for _ in range(H)]
        )
        self.W_o = nn.Linear(H * self.d_head, d_msg)

    def forward(
        self,
        query_enc: torch.Tensor,                         # (d_msg,) or (batch, d_msg)
        neighbor_encs: "List[torch.Tensor] | torch.Tensor",  # list of (d_msg,) or (num_nbr, d_msg)
    ) -> torch.Tensor:
        import math

        # Normalise neighbor_encs: accept either a list of tensors or a pre-stacked tensor
        if isinstance(neighbor_encs, list):
            if len(neighbor_encs) == 0:
                return torch.zeros(self.d_msg, device=query_enc.device)
            neighbor_encs = torch.stack(neighbor_encs, dim=0)  # (num_nbr, d_msg)
        else:
            if neighbor_encs.shape[0] == 0:
                return torch.zeros(self.d_msg, device=query_enc.device)

        scale = math.sqrt(self.d_head)
        heads = []
        for h in range(self.H):
            q_h = self.W_q[h](query_enc)        # (d_head,)
            K_h = self.W_k[h](neighbor_encs)    # (num_nbr, d_head)
            V_h = self.W_v[h](neighbor_encs)    # (num_nbr, d_head)
            attn = F.softmax(K_h @ q_h / scale, dim=0)  # (num_nbr,)
            head_h = (attn.unsqueeze(-1) * V_h).sum(dim=0)  # (d_head,)
            heads.append(head_h)

        return self.W_o(torch.cat(heads, dim=-1))  # (d_msg,)

    def forward_batch(
        self,
        msgs: torch.Tensor,          # (batch, n_agents, d_msg)
        adj_matrices: torch.Tensor,  # (batch, n_agents, n_agents)
    ) -> torch.Tensor:               # (batch, n_agents, d_msg)
        """Batched multi-head attention for PPO update phase."""
        import math
        batch_size, n_agents, _ = msgs.shape
        scale = math.sqrt(self.d_head)
        heads = []
        for h in range(self.H):
            Q = self.W_q[h](msgs)   # (batch, n, d_head)
            K = self.W_k[h](msgs)   # (batch, n, d_head)
            V = self.W_v[h](msgs)   # (batch, n, d_head)
            # scores: (batch, n, n)
            scores = torch.bmm(Q, K.transpose(1, 2)) / scale
            # mask out non-neighbors and isolated agents
            mask = (adj_matrices == 0)
            scores = scores.masked_fill(mask, float("-inf"))
            all_masked = mask.all(dim=-1, keepdim=True)
            scores = scores.masked_fill(all_masked.expand_as(scores), 0.0)
            attn = F.softmax(scores, dim=-1)  # (batch, n, n)
            attn = attn.masked_fill(all_masked.expand_as(attn), 0.0)
            head_h = torch.bmm(attn, V)  # (batch, n, d_head)
            heads.append(head_h)
        # concat and project
        concat = torch.cat(heads, dim=-1)  # (batch, n, H*d_head)
        return self.W_o(concat)            # (batch, n, d_msg)


# ==============================================================================
# Actor：策略网络（参数共享）
# ==============================================================================

class Actor(nn.Module):
    """
    MAPPO Actor 网络（参数共享，连续动作空间）。

    输入：[o_i; agg_i] ∈ R^{d_obs + d_msg}
    输出：动作分布 N(μ, σ²I)，σ 通过 log_std 参数学习

    π_i(· | [o_i; agg_i]) : R^{d_obs+d_msg} → N(μ, σ²)
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        msg_dim: int = 64,
        hidden_dims: List[int] = None,
        log_std_init: float = 0.0,
        log_std_min: float = -3.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.msg_dim = msg_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        input_dim = obs_dim + msg_dim

        # 均值头
        self.mu_net = build_mlp(
            input_dim=input_dim,
            output_dim=act_dim,
            hidden_dims=hidden_dims,
            activation="tanh",
            output_activation="tanh",  # 输出有界 [-1, 1]，训练时乘以 a_max
        )

        # 可学习的 log_std（状态无关）
        self.log_std = nn.Parameter(
            torch.full((act_dim,), log_std_init)
        )

    def forward(
        self,
        obs: torch.Tensor,
        agg_msg: torch.Tensor,
    ) -> Normal:
        """
        Args:
            obs: [..., obs_dim]
            agg_msg: [..., msg_dim]

        Returns:
            dist: Normal 分布对象
        """
        x = torch.cat([obs, agg_msg], dim=-1)
        mu = self.mu_net(x)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std).expand_as(mu)
        return Normal(mu, std)

    def get_action(
        self,
        obs: torch.Tensor,
        agg_msg: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样动作。

        Returns:
            action: [..., act_dim]，裁剪到 [-1, 1]
            log_prob: [...] 动作对数概率
        """
        dist = self.forward(obs, agg_msg)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        agg_msg: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        评估给定动作的对数概率和分布熵（PPO 更新时使用）。

        Returns:
            log_prob: [...] 对数概率
            entropy: [...] 分布熵
        """
        dist = self.forward(obs, agg_msg)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


# ==============================================================================
# Critic：集中式价值网络
# ==============================================================================

class Critic(nn.Module):
    """
    集中式任务价值网络 V_θ(s₁,...,sₙ)。

    输入：全局状态 s = concat(s₁,...,sₙ) ∈ R^{n * state_dim}
    输出：标量价值 V ∈ R
    """

    def __init__(
        self,
        global_state_dim: int,
        hidden_dims: List[int] = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.value_net = build_mlp(
            input_dim=global_state_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation="tanh",
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_state: [..., global_state_dim]

        Returns:
            value: [..., 1] 或 [...] 价值估计
        """
        return self.value_net(global_state).squeeze(-1)


# ==============================================================================
# CostCritic：集中式安全成本价值网络
# ==============================================================================

class CostCritic(nn.Module):
    """
    集中式安全成本价值网络 V_φ^c(s₁,...,sₙ)。

    与 Critic 结构相同，但专门估计安全成本回报。
    分离任务和成本估计有利于训练稳定性。
    """

    def __init__(
        self,
        global_state_dim: int,
        hidden_dims: List[int] = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.cost_value_net = build_mlp(
            input_dim=global_state_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation="tanh",
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_state: [..., global_state_dim]

        Returns:
            cost_value: [...] 安全成本价值估计
        """
        return self.cost_value_net(global_state).squeeze(-1)


# ==============================================================================
# SafeCommNetworks：统一管理所有网络模块
# ==============================================================================

class SafeCommNetworks(nn.Module):
    """
    SafeComm-PSched 所有网络模块的容器类。

    包含：
      - encoder: MessageEncoder（共享参数）
      - aggregator: MessageAggregator（单头注意力）
      - actor: Actor（共享参数）
      - critic: Critic（集中式任务价值）
      - cost_critic: CostCritic（集中式安全成本价值）
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        n_agents: int,
        global_state_dim: Optional[int] = None,
        msg_dim: int = 64,
        hidden_dims: List[int] = None,
        H: int = 2,
    ):
        """
        Args:
            obs_dim: 单个智能体观测维度
            act_dim: 动作维度
            n_agents: 智能体数量
            global_state_dim: 全局状态维度（Critic 输入）。
                              若为 None，则使用 n_agents * obs_dim。
            msg_dim: 消息向量维度（默认 64）
            hidden_dims: 所有 MLP 的隐藏层（默认 [256, 256]）
            H: 注意力头数（默认 2，使用 MultiHeadAttentionAggregator；1 则使用单头 MessageAggregator）
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        if global_state_dim is None:
            global_state_dim = n_agents * obs_dim

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_agents = n_agents
        self.msg_dim = msg_dim
        self.global_state_dim = global_state_dim
        self.H = H

        self.encoder = MessageEncoder(
            obs_dim=obs_dim,
            msg_dim=msg_dim,
            hidden_dims=hidden_dims,
        )
        if H > 1:
            self.aggregator = MultiHeadAttentionAggregator(d_msg=msg_dim, H=H)
        else:
            self.aggregator = MessageAggregator(msg_dim=msg_dim)
        self.actor = Actor(
            obs_dim=obs_dim,
            act_dim=act_dim,
            msg_dim=msg_dim,
            hidden_dims=hidden_dims,
        )
        self.critic = Critic(
            global_state_dim=global_state_dim,
            hidden_dims=hidden_dims,
        )
        self.cost_critic = CostCritic(
            global_state_dim=global_state_dim,
            hidden_dims=hidden_dims,
        )

    # ------------------------------------------------------------------
    # 前向推断（单步，无批量）
    # ------------------------------------------------------------------

    def act(
        self,
        obs_list: List[np.ndarray],
        neighbor_lists: Dict[int, List[int]],
        deterministic: bool = False,
        device: str = "cpu",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        推断所有智能体的动作（推断模式，无梯度）。

        Args:
            obs_list: 长度为 n_agents 的观测列表，每个 [obs_dim]
            neighbor_lists: {agent_id: [neighbor_ids]}
            deterministic: 是否使用确定性策略（测试时用）
            device: 计算设备

        Returns:
            actions: [n_agents, act_dim] numpy 数组
            log_probs: [n_agents] numpy 数组
            values: [n_agents] 任务价值（共享全局价值）numpy 数组
            cost_values: [n_agents] 安全成本价值 numpy 数组
        """
        with torch.no_grad():
            # 转换为 tensor
            obs_tensors = [
                torch.FloatTensor(o).to(device) for o in obs_list
            ]  # List of [obs_dim]

            # 编码所有观测
            all_msgs = [self.encoder(o) for o in obs_tensors]  # List of [msg_dim]

            # 聚合消息
            agg_msgs = []
            for i in range(self.n_agents):
                nbr_msgs = [all_msgs[j] for j in neighbor_lists.get(i, [])]
                agg = self.aggregator(all_msgs[i], nbr_msgs)
                agg_msgs.append(agg)

            # 采样动作
            actions_list = []
            log_probs_list = []
            for i in range(self.n_agents):
                action, log_prob = self.actor.get_action(
                    obs_tensors[i], agg_msgs[i], deterministic=deterministic
                )
                actions_list.append(action.cpu().numpy())
                log_probs_list.append(log_prob.cpu().numpy())

            # 全局状态（Critic 输入）
            global_state = torch.cat(obs_tensors, dim=-1)  # [n * obs_dim]
            if global_state.shape[-1] < self.global_state_dim:
                # 如果实际状态维度比初始化时小，用零填充（容错）
                pad = torch.zeros(
                    self.global_state_dim - global_state.shape[-1], device=device
                )
                global_state = torch.cat([global_state, pad], dim=-1)

            value = self.critic(global_state.unsqueeze(0)).squeeze().cpu().numpy()
            cost_value = self.cost_critic(global_state.unsqueeze(0)).squeeze().cpu().numpy()

            actions = np.stack(actions_list, axis=0)
            log_probs = np.array(log_probs_list)
            values = np.full(self.n_agents, float(value))
            cost_values = np.full(self.n_agents, float(cost_value))

        return actions, log_probs, values, cost_values

    def get_values(
        self,
        global_states: torch.Tensor,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量计算价值估计（训练时使用）。

        Args:
            global_states: [batch_size, global_state_dim]

        Returns:
            values: [batch_size] 任务价值
            cost_values: [batch_size] 安全成本价值
        """
        values = self.critic(global_states)
        cost_values = self.cost_critic(global_states)
        return values, cost_values

    def evaluate_actions_batch(
        self,
        obs_batch: torch.Tensor,         # [batch, n_agents, obs_dim]
        actions_batch: torch.Tensor,     # [batch, n_agents, act_dim]
        adj_matrices: torch.Tensor,      # [batch, n_agents, n_agents]
        global_states: torch.Tensor,     # [batch, global_state_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        批量评估动作（PPO 更新阶段）。

        Args:
            obs_batch: [batch, n_agents, obs_dim]
            actions_batch: [batch, n_agents, act_dim]
            adj_matrices: [batch, n_agents, n_agents] 邻接矩阵
            global_states: [batch, global_state_dim]

        Returns:
            log_probs: [batch, n_agents]
            entropy: [batch, n_agents]
            values: [batch] 任务价值
            cost_values: [batch] 安全成本价值
            all_msgs_agg: [batch, n_agents, msg_dim] 聚合消息（可用于调试）
        """
        batch_size = obs_batch.shape[0]

        # 编码所有智能体观测: [batch, n_agents, msg_dim]
        obs_flat = obs_batch.view(batch_size * self.n_agents, self.obs_dim)
        msgs_flat = self.encoder(obs_flat)  # [batch*n, msg_dim]
        msgs = msgs_flat.view(batch_size, self.n_agents, self.msg_dim)

        # 注意力聚合（批量）—— 支持多头和单头
        # Dispatch to appropriate batch aggregation
        if hasattr(self.aggregator, 'forward_batch'):
            agg_msgs = self.aggregator.forward_batch(msgs, adj_matrices)
        else:
            # Single-head fallback
            queries = self.aggregator.query_proj(msgs)
            keys = self.aggregator.key_proj(msgs)
            scores = torch.bmm(queries, keys.transpose(1, 2)) * self.aggregator.scale
            mask = (adj_matrices == 0)
            scores = scores.masked_fill(mask, float("-inf"))
            all_masked = mask.all(dim=-1, keepdim=True)
            scores = scores.masked_fill(all_masked.expand_as(scores), 0.0)
            attn = F.softmax(scores, dim=-1)
            attn = attn.masked_fill(all_masked.expand_as(attn), 0.0)
            agg_msgs = torch.bmm(attn, msgs)

        # 评估动作
        obs_flat2 = obs_batch.view(batch_size * self.n_agents, self.obs_dim)
        agg_flat = agg_msgs.view(batch_size * self.n_agents, self.msg_dim)
        acts_flat = actions_batch.view(batch_size * self.n_agents, self.act_dim)

        log_probs_flat, entropy_flat = self.actor.evaluate_actions(
            obs_flat2, agg_flat, acts_flat
        )

        log_probs = log_probs_flat.view(batch_size, self.n_agents)
        entropy = entropy_flat.view(batch_size, self.n_agents)

        # 价值估计
        values, cost_values = self.get_values(global_states)

        return log_probs, entropy, values, cost_values, agg_msgs

    def count_parameters(self) -> Dict[str, int]:
        """统计各模块参数量。"""
        def count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "encoder": count(self.encoder),
            "aggregator": count(self.aggregator),
            "actor": count(self.actor),
            "critic": count(self.critic),
            "cost_critic": count(self.cost_critic),
            "total": count(self),
        }

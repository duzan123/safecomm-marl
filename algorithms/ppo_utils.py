"""
PPO 工具函数模块

包含：
  - compute_gae: 广义优势估计（GAE，Schulman 2016）
  - compute_returns: 折现回报计算
  - RolloutBuffer: 轨迹经验回放缓冲区
  - normalize_advantages: 优势函数标准化
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


# ==============================================================================
# GAE（广义优势估计）
# ==============================================================================

def compute_gae(
    rewards: np.ndarray,      # [T] 或 [T, n_agents]
    values: np.ndarray,       # [T+1] 或 [T+1, n_agents]
    dones: np.ndarray,        # [T] bool 数组
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算广义优势估计（GAE-λ）和折现回报。

    A_t = Σ_{l=0}^{T-t-1} (γλ)^l * δ_{t+l}
    δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)

    Args:
        rewards: [T] 奖励序列（或 [T, n_agents] 多智能体）
        values: [T+1] 价值估计（最后一个是 bootstrap 值）
        dones: [T] 终止标志（True 表示该步之后重置）
        gamma: 折扣因子
        lam: GAE λ 参数

    Returns:
        advantages: [T] 优势函数
        returns: [T] 折现回报（= advantages + values[:-1]）
    """
    T = len(rewards)
    advantages = np.zeros_like(rewards, dtype=np.float64)

    gae = 0.0
    for t in reversed(range(T)):
        not_done = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]
        gae = delta + gamma * lam * not_done * gae
        advantages[t] = gae

    returns = advantages + values[:-1]
    return advantages.astype(np.float32), returns.astype(np.float32)


def compute_returns(
    rewards: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float = 0.99,
) -> np.ndarray:
    """
    计算折现回报（无 GAE，简单版本）。

    G_t = r_t + γ * G_{t+1} * (1 - done_t)

    Args:
        rewards: [T]
        dones: [T]
        last_value: bootstrap 价值（最后一步之后的价值估计）
        gamma: 折扣因子

    Returns:
        returns: [T]
    """
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    R = last_value
    for t in reversed(range(T)):
        R = rewards[t] + gamma * R * (1.0 - float(dones[t]))
        returns[t] = R
    return returns


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    对优势函数进行零均值单位方差标准化。

    Args:
        advantages: 任意形状的 tensor
        eps: 数值稳定性常数

    Returns:
        标准化后的 advantages
    """
    mean = advantages.mean()
    std = advantages.std()
    return (advantages - mean) / (std + eps)


# ==============================================================================
# PPO Clip 损失
# ==============================================================================

def ppo_clip_loss(
    log_probs_new: torch.Tensor,   # [batch] 或 [batch, n_agents]
    log_probs_old: torch.Tensor,   # [batch] 或 [batch, n_agents]
    advantages: torch.Tensor,      # [batch] 或 [batch, n_agents]
    clip_eps: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算 PPO-clip 目标函数（最大化方向）。

    L^CLIP = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
    其中 r_t = π_new(a|s) / π_old(a|s) = exp(log_π_new - log_π_old)

    Args:
        log_probs_new: 新策略的动作对数概率
        log_probs_old: 旧策略的动作对数概率（detach）
        advantages: 优势函数估计
        clip_eps: PPO clip 参数（默认 0.2）

    Returns:
        clip_loss: 标量，PPO clip 损失（取负号用于梯度下降）
        approx_kl: 近似 KL 散度（监控用）
    """
    ratio = torch.exp(log_probs_new - log_probs_old)
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)

    surrogate1 = ratio * advantages
    surrogate2 = clipped_ratio * advantages

    clip_loss = -torch.min(surrogate1, surrogate2).mean()
    approx_kl = ((log_probs_old - log_probs_new) ** 2 / 2).mean()

    return clip_loss, approx_kl


def value_loss(
    values_pred: torch.Tensor,     # [batch]
    values_old: torch.Tensor,      # [batch]（值函数 clip 时使用）
    returns: torch.Tensor,         # [batch]
    clip_eps: float = 0.2,
    use_clip: bool = True,
) -> torch.Tensor:
    """
    计算价值函数损失（带可选的 value clipping）。

    L^VF = E[(V(s) - G_t)^2]
    或带 clip：L^VF = E[max((V-G)^2, (V_clip - G)^2)]

    Args:
        values_pred: 当前网络输出的价值
        values_old: 旧网络输出的价值（用于 clip）
        returns: 折现回报 G_t
        clip_eps: value clip 范围
        use_clip: 是否使用 value clipping

    Returns:
        vf_loss: 价值函数损失标量
    """
    if use_clip:
        values_clipped = values_old + torch.clamp(
            values_pred - values_old, -clip_eps, clip_eps
        )
        vf_loss1 = (values_pred - returns) ** 2
        vf_loss2 = (values_clipped - returns) ** 2
        vf_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()
    else:
        vf_loss = 0.5 * ((values_pred - returns) ** 2).mean()

    return vf_loss


# ==============================================================================
# RolloutBuffer：轨迹缓冲区
# ==============================================================================

class RolloutBuffer:
    """
    MAPPO 轨迹缓冲区，支持多智能体数据存储。

    数据布局（每条样本均对应一个时间步）：
      - obs:           [T, n_agents, obs_dim]
      - actions:       [T, n_agents, act_dim]
      - log_probs:     [T, n_agents]
      - rewards:       [T, n_agents]（全局共享奖励，广播到每个智能体）
      - costs:         [T, n_agents]（安全成本）
      - values:        [T+1] 全局价值估计（最后一个是 bootstrap）
      - cost_values:   [T+1] 安全成本价值估计
      - dones:         [T] bool
      - global_states: [T, global_state_dim]
      - adj_matrices:  [T, n_agents, n_agents]（通信图邻接矩阵）
    """

    def __init__(
        self,
        n_steps: int,
        n_agents: int,
        obs_dim: int,
        act_dim: int,
        global_state_dim: int,
    ):
        self.n_steps = n_steps
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.global_state_dim = global_state_dim

        self.reset()

    def reset(self):
        """清空缓冲区。"""
        T, N = self.n_steps, self.n_agents

        self.obs = np.zeros((T, N, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((T, N, self.act_dim), dtype=np.float32)
        self.log_probs = np.zeros((T, N), dtype=np.float32)
        self.rewards = np.zeros((T, N), dtype=np.float32)
        self.costs = np.zeros((T, N), dtype=np.float32)
        self.values = np.zeros(T + 1, dtype=np.float32)
        self.cost_values = np.zeros(T + 1, dtype=np.float32)
        self.dones = np.zeros(T, dtype=np.bool_)
        self.global_states = np.zeros((T, self.global_state_dim), dtype=np.float32)
        self.adj_matrices = np.zeros((T, N, N), dtype=np.float32)

        self.ptr = 0  # 当前写入位置
        self.full = False

    def add(
        self,
        obs: np.ndarray,                      # [n_agents, obs_dim]
        actions: np.ndarray,                  # [n_agents, act_dim]
        log_probs: np.ndarray,                # [n_agents]
        rewards: "List[float] | np.ndarray",  # [n_agents]
        costs: "List[float] | np.ndarray",    # [n_agents]
        value: float,                         # 全局价值估计
        cost_value: float,                    # 安全成本价值估计
        done: bool,
        global_state: np.ndarray,             # [global_state_dim]
        adj_matrix: np.ndarray,               # [n_agents, n_agents]
    ):
        """
        添加一步数据到缓冲区。
        """
        t = self.ptr
        assert t < self.n_steps, "缓冲区已满，请先调用 reset()"

        self.obs[t] = obs
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.rewards[t] = rewards
        self.costs[t] = costs
        self.values[t] = value
        self.cost_values[t] = cost_value
        self.dones[t] = done
        self.global_states[t] = global_state
        self.adj_matrices[t] = adj_matrix

        self.ptr += 1
        if self.ptr >= self.n_steps:
            self.full = True

    def finish_rollout(self, last_value: float, last_cost_value: float):
        """
        完成轨迹收集，存储 bootstrap 值（用于 GAE 计算）。

        Args:
            last_value: 最后一步之后的价值估计
            last_cost_value: 最后一步之后的安全成本价值估计
        """
        self.values[self.ptr] = last_value
        self.cost_values[self.ptr] = last_cost_value

    def compute_advantages_and_returns(
        self,
        gamma: float = 0.99,
        lam: float = 0.95,
        normalize_adv: bool = True,
        normalize_cost_adv: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        计算任务和安全成本的优势函数与折现回报。

        Returns:
            advantages: [T]
            returns: [T]
            cost_advantages: [T]
            cost_returns: [T]
        """
        T = self.ptr  # 实际步数

        # 使用每个智能体的平均奖励作为全局奖励（参数共享）
        mean_rewards = self.rewards[:T].mean(axis=1)  # [T]
        mean_costs = self.costs[:T].mean(axis=1)       # [T]

        advantages, returns = compute_gae(
            rewards=mean_rewards,
            values=self.values[:T + 1],
            dones=self.dones[:T],
            gamma=gamma,
            lam=lam,
        )

        cost_advantages, cost_returns = compute_gae(
            rewards=mean_costs,
            values=self.cost_values[:T + 1],
            dones=self.dones[:T],
            gamma=gamma,
            lam=lam,
        )

        if normalize_adv and advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if normalize_cost_adv and cost_advantages.std() > 1e-6:
            cost_advantages = (cost_advantages - cost_advantages.mean()) / (
                cost_advantages.std() + 1e-8
            )

        return advantages, returns, cost_advantages, cost_returns

    def get_training_data(
        self,
        gamma: float = 0.99,
        lam: float = 0.95,
        normalize_adv: bool = True,
        device: str = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """
        获取训练所需的全部数据（转换为 torch.Tensor）。

        Returns:
            字典，包含所有训练所需张量
        """
        T = self.ptr
        advantages, returns, cost_advantages, cost_returns = (
            self.compute_advantages_and_returns(gamma, lam, normalize_adv)
        )

        def to_tensor(arr):
            return torch.FloatTensor(arr).to(device)

        return {
            "obs": to_tensor(self.obs[:T]),                    # [T, n, obs_dim]
            "actions": to_tensor(self.actions[:T]),            # [T, n, act_dim]
            "log_probs_old": to_tensor(self.log_probs[:T]),    # [T, n]
            "advantages": to_tensor(advantages),               # [T]
            "returns": to_tensor(returns),                     # [T]
            "cost_advantages": to_tensor(cost_advantages),     # [T]
            "cost_returns": to_tensor(cost_returns),           # [T]
            "values_old": to_tensor(self.values[:T]),          # [T]
            "cost_values_old": to_tensor(self.cost_values[:T]), # [T]
            "global_states": to_tensor(self.global_states[:T]), # [T, gs_dim]
            "adj_matrices": to_tensor(self.adj_matrices[:T]),  # [T, n, n]
            "dones": torch.BoolTensor(self.dones[:T]).to(device),
        }

    def is_full(self) -> bool:
        return self.full or (self.ptr >= self.n_steps)

    def __len__(self) -> int:
        return self.ptr


# ==============================================================================
# 统计工具
# ==============================================================================

def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    计算价值函数的解释方差（EV）。
    EV = 1 - Var(y_true - y_pred) / Var(y_true)
    完美预测 EV=1，随机预测 EV=0，差于随机 EV<0。
    """
    var_y = np.var(y_true)
    if var_y < 1e-10:
        return float("nan")
    return float(1.0 - np.var(y_true - y_pred) / var_y)


# ==============================================================================
# MGDA（多梯度下降算法）
# ==============================================================================

def mgda_solve(
    grad_task: torch.Tensor,
    grad_safe: torch.Tensor,
) -> tuple:
    """Closed-form MGDA for 2 objectives (Multiple Gradient Descent Algorithm).

    Finds (w_task, w_safe) on the probability simplex minimizing
    ||w_task * g_task + w_safe * g_safe||^2.

    Closed-form solution:
        w_task = clip((||g1||^2 - g0·g1) / (||g0||^2 + ||g1||^2 - 2*g0·g1), 0, 1)
        w_safe = 1 - w_task

    Edge cases: zero gradients → assign all weight to non-zero; both zero → 0.5/0.5.
    """
    g0 = grad_task.detach().float().flatten()
    g1 = grad_safe.detach().float().flatten()

    g0_sq = float((g0 * g0).sum())
    g1_sq = float((g1 * g1).sum())
    g01 = float((g0 * g1).sum())

    if g0_sq < 1e-10 and g1_sq < 1e-10:
        return 0.5, 0.5
    if g0_sq < 1e-10:
        return 0.0, 1.0
    if g1_sq < 1e-10:
        return 1.0, 0.0

    denom = g0_sq + g1_sq - 2.0 * g01
    if abs(denom) < 1e-10:
        return 0.5, 0.5

    w0 = (g1_sq - g01) / denom
    w0 = max(0.0, min(1.0, float(w0)))
    return w0, 1.0 - w0

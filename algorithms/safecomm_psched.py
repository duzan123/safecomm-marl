"""
SafeComm-PSched：Safety-Priority Scheduled MARL

核心算法主文件，实现设计规格 §3.3 的训练伪代码。

算法流程：
  for 每次迭代:
    1. 数据收集：CBF 计算 → 调度通信图 → 聚合消息 → 采样动作 → 收集奖励/成本
    2. 计算优势：GAE(r, V_θ) 和 GAE(c^safe, V_φ^c)
    3. PPO 更新：L_PPO = clip 损失 - λ_s * 成本优势项
    4. Lagrangian 乘子更新：λ_s ← max(0, λ_s + α_λ * (Ĵ_safe - d))
    5. 日志记录

Lagrangian 安全优化（定理 1，安全约束收敛）：
  J_safe(π^T) ≤ d + O(1/√T)
"""

import os
import time
import numpy as np
import torch
import torch.optim as optim
from typing import Dict, List, Optional, Tuple

from algorithms.networks import SafeCommNetworks
from algorithms.scheduler import SafetyPriorityScheduler
from algorithms.ppo_utils import (
    RolloutBuffer,
    ppo_clip_loss,
    value_loss,
    normalize_advantages,
    explained_variance,
    mgda_solve,
)
from envs.uav_formation_env import UAVFormationEnv


class SafeCommPSched:
    """
    SafeComm-PSched 算法主类。

    使用方式：
        agent = SafeCommPSched(env, config)
        agent.train(total_steps=1_000_000)
    """

    def __init__(
        self,
        env: UAVFormationEnv,
        config: Optional[Dict] = None,
        device: str = "auto",
    ):
        """
        Args:
            env: UAVFormationEnv 实例
            config: 超参数字典（None 则使用默认值）
            device: 计算设备（"auto" | "cpu" | "cuda"）
        """
        # 默认超参数
        default_config = {
            # 环境参数
            "n_agents": env.n,
            "obs_dim": env.obs_dim,
            "act_dim": env.act_dim,
            "action_scale": env.a_max,  # 将 [-1,1] 映射到 [-a_max, a_max]
            # 通信参数
            "k": 4,
            "R_comm": env.R_comm,
            "d_min": env.d_min,
            "schedule_mode": "safety_priority",
            # 网络参数
            "msg_dim": 64,
            "hidden_dims": [256, 256],
            # PPO 超参数
            "n_steps": 2048,           # 每次迭代收集的时间步数
            "n_epochs": 10,            # 每次迭代 PPO 更新轮数
            "batch_size": 256,         # mini-batch 大小
            "gamma": 0.99,             # 折扣因子
            "lam": 0.95,               # GAE λ
            "clip_eps": 0.2,           # PPO clip 参数
            "value_coef": 0.5,         # 价值损失系数
            "cost_value_coef": 0.5,    # 安全成本价值损失系数
            "entropy_coef": 0.01,      # 熵奖励系数
            "max_grad_norm": 0.5,      # 梯度裁剪
            # 学习率
            "lr_actor": 3e-4,
            "lr_critic": 1e-3,
            "lr_lambda": 0.01,         # Lagrangian 乘子学习率（双时间尺度：慢于 π）
            # 安全约束
            "safety_budget": 0.1,      # d（每集期望安全成本上限）
            "lambda_init": 0.0,        # Lagrangian 乘子初值
            "lambda_max": 10.0,        # λ 上界（防止过度惩罚）
            # 日志
            "log_interval": 10,        # 每多少次迭代打印一次
            "eval_interval": 50,       # 每多少次迭代评估一次
            # v2.0 新增
            "lambda_threshold": 0.1,   # MGDA 激活阈值（v2.0）
            "H": 2,                    # 注意力头数（v2.0）
            "v_max_cbf": 1.0,          # CBF 保守缓冲最大速度（v2.0）
            "dt": 0.1,                  # 时间步长（CBF 缓冲计算用）
        }

        if config is not None:
            default_config.update(config)
        self.config = default_config

        # 设备
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.env = env
        n = self.config["n_agents"]
        obs_dim = self.config["obs_dim"]

        # 全局状态维度 = n_agents * obs_dim（集中式 Critic）
        self.global_state_dim = n * obs_dim

        # 初始化网络
        self.networks = SafeCommNetworks(
            obs_dim=obs_dim,
            act_dim=self.config["act_dim"],
            n_agents=n,
            global_state_dim=self.global_state_dim,
            msg_dim=self.config["msg_dim"],
            hidden_dims=self.config["hidden_dims"],
            H=self.config["H"],
        ).to(self.device)

        # 初始化调度器（无需训练）
        self.scheduler = SafetyPriorityScheduler(
            n_agents=n,
            k=self.config["k"],
            d_min=self.config["d_min"],
            schedule_mode=self.config["schedule_mode"],
            v_max=self.config.get("v_max_cbf", self.config.get("v_max", 1.0)),
            dt=self.config.get("dt", 0.1),
        )

        # 优化器
        actor_params = (
            list(self.networks.encoder.parameters())
            + list(self.networks.aggregator.parameters())
            + list(self.networks.actor.parameters())
        )
        critic_params = (
            list(self.networks.critic.parameters())
            + list(self.networks.cost_critic.parameters())
        )

        self.actor_optimizer = optim.Adam(
            actor_params, lr=self.config["lr_actor"]
        )
        self.critic_optimizer = optim.Adam(
            critic_params, lr=self.config["lr_critic"]
        )

        # Lagrangian 乘子（可学习标量）
        self.lambda_s = self.config["lambda_init"]
        self.lambda_max = self.config["lambda_max"]

        # 经验缓冲区
        self.buffer = RolloutBuffer(
            n_steps=self.config["n_steps"],
            n_agents=n,
            obs_dim=obs_dim,
            act_dim=self.config["act_dim"],
            global_state_dim=self.global_state_dim,
        )

        # 训练统计
        self.total_steps = 0
        self.n_iterations = 0
        self.metrics_history: List[Dict] = []

    # ------------------------------------------------------------------
    # 数据收集
    # ------------------------------------------------------------------

    def collect_rollout(self) -> Dict:
        """
        收集一批轨迹数据（n_steps 步）。

        Returns:
            rollout_info: 本次采集的统计信息
        """
        self.buffer.reset()
        self.networks.eval()

        obs_list, info = self.env.reset()
        self.scheduler.reset_episode(self.env.positions)
        episode_rewards = []
        episode_costs = []
        episode_formations = []
        n_episodes = 0
        ep_reward = 0.0
        ep_cost = 0.0

        n_steps = self.config["n_steps"]
        n = self.config["n_agents"]

        # 通信带宽利用率统计
        bw_utils = []

        for step in range(n_steps):
            # 1. 获取物理连接图
            phys_edges = self.env.get_physical_graph()

            # 2. 安全感知调度
            sched_result = self.scheduler.schedule_step(
                positions=self.env.positions,
                phys_edges=phys_edges,
            )
            active_edges = sched_result["active_edges"]
            self.scheduler.update_history(active_edges, self.env.positions)
            neighbor_lists = sched_result["neighbor_lists"]
            adj_matrix = sched_result["adjacency_matrix"]
            bw_utils.append(sched_result["bandwidth_utilization"])

            # 3. 策略推断（无梯度）
            actions, log_probs, values, cost_values = self.networks.act(
                obs_list=obs_list,
                neighbor_lists=neighbor_lists,
                deterministic=False,
                device=self.device,
            )

            # 4. 将 [-1,1] 动作缩放到 [-a_max, a_max]
            scaled_actions = actions * self.config["action_scale"]

            # 5. 全局状态（Critic 输入）
            global_state = np.concatenate(obs_list, axis=0)  # [n * obs_dim]
            if len(global_state) < self.global_state_dim:
                pad = np.zeros(self.global_state_dim - len(global_state))
                global_state = np.concatenate([global_state, pad])

            # 6. 环境步进
            next_obs_list, reward_list, cost_list, done, step_info = self.env.step(
                scaled_actions
            )

            # 7. 存入缓冲区
            self.buffer.add(
                obs=np.stack(obs_list, axis=0),
                actions=actions,  # 存储未缩放的 [-1,1] 动作（用于 log_prob 计算）
                log_probs=log_probs,
                rewards=reward_list,
                costs=cost_list,
                value=float(values[0]),
                cost_value=float(cost_values[0]),
                done=done,
                global_state=global_state,
                adj_matrix=adj_matrix,
            )

            ep_reward += float(np.mean(reward_list))
            ep_cost += float(np.mean(cost_list))

            if done:
                episode_rewards.append(ep_reward)
                episode_costs.append(ep_cost)
                episode_formations.append(step_info["formation_error"])
                ep_reward = 0.0
                ep_cost = 0.0
                n_episodes += 1
                obs_list, _ = self.env.reset()
                self.scheduler.reset_episode(self.env.positions)
            else:
                obs_list = next_obs_list

            self.total_steps += 1

        # 收集最后一步的 bootstrap 价值
        phys_edges = self.env.get_physical_graph()
        sched_result = self.scheduler.schedule_step(self.env.positions, phys_edges)
        _, _, last_values, last_cost_values = self.networks.act(
            obs_list=obs_list,
            neighbor_lists=sched_result["neighbor_lists"],
            deterministic=False,
            device=self.device,
        )
        self.buffer.finish_rollout(
            last_value=float(last_values[0]),
            last_cost_value=float(last_cost_values[0]),
        )

        rollout_info = {
            "mean_episode_reward": np.mean(episode_rewards) if episode_rewards else ep_reward,
            "mean_episode_cost": np.mean(episode_costs) if episode_costs else ep_cost,
            "mean_formation_error": np.mean(episode_formations) if episode_formations else 0.0,
            "n_episodes": n_episodes,
            "mean_bandwidth_util": float(np.mean(bw_utils)),
        }
        return rollout_info

    # ------------------------------------------------------------------
    # PPO 更新（含 Lagrangian）
    # ------------------------------------------------------------------

    def update(self) -> Dict:
        """
        执行 PPO 更新步骤（n_epochs 轮 mini-batch 更新）。

        Returns:
            update_info: 更新统计信息
        """
        self.networks.train()

        data = self.buffer.get_training_data(
            gamma=self.config["gamma"],
            lam=self.config["lam"],
            normalize_adv=True,
            device=self.device,
        )

        T = len(data["obs"])
        batch_size = min(self.config["batch_size"], T)
        n_epochs = self.config["n_epochs"]
        clip_eps = self.config["clip_eps"]
        lambda_s = self.lambda_s

        # 统计
        all_actor_losses = []
        all_critic_losses = []
        all_entropy = []
        all_approx_kl = []
        all_cost_violations = []

        for epoch in range(n_epochs):
            indices = torch.randperm(T)

            for start in range(0, T, batch_size):
                idx = indices[start:start + batch_size]
                if len(idx) < 2:
                    continue

                obs_b = data["obs"][idx]                      # [B, n, obs_dim]
                actions_b = data["actions"][idx]               # [B, n, act_dim]
                log_probs_old_b = data["log_probs_old"][idx]   # [B, n]
                advantages_b = data["advantages"][idx]         # [B]
                returns_b = data["returns"][idx]               # [B]
                cost_adv_b = data["cost_advantages"][idx]      # [B]
                cost_returns_b = data["cost_returns"][idx]     # [B]
                values_old_b = data["values_old"][idx]         # [B]
                cost_values_old_b = data["cost_values_old"][idx]
                global_states_b = data["global_states"][idx]  # [B, gs_dim]
                adj_b = data["adj_matrices"][idx]              # [B, n, n]

                # 前向
                log_probs_b, entropy_b, values_b, cost_values_b, _ = (
                    self.networks.evaluate_actions_batch(
                        obs_batch=obs_b,
                        actions_batch=actions_b,
                        adj_matrices=adj_b,
                        global_states=global_states_b,
                    )
                )

                # 策略损失（所有智能体取均值）
                log_probs_mean = log_probs_b.mean(dim=1)    # [B]
                log_probs_old_mean = log_probs_old_b.mean(dim=1)  # [B]

                # PPO clip（任务）
                actor_loss, approx_kl = ppo_clip_loss(
                    log_probs_new=log_probs_mean,
                    log_probs_old=log_probs_old_mean.detach(),
                    advantages=advantages_b,
                    clip_eps=clip_eps,
                )

                # Lagrangian 安全惩罚：- λ_s * E[A_c]
                # 注意：cost_adv 已归一化，乘以 lambda_s 后加入 actor loss
                safety_penalty = lambda_s * cost_adv_b.mean()

                # 熵奖励
                entropy_mean = entropy_b.mean()
                entropy_loss = -self.config["entropy_coef"] * entropy_mean

                total_actor_loss = actor_loss + safety_penalty + entropy_loss

                # 价值损失
                vf_loss = value_loss(
                    values_pred=values_b,
                    values_old=values_old_b.detach(),
                    returns=returns_b,
                    clip_eps=clip_eps,
                    use_clip=True,
                )
                cost_vf_loss = value_loss(
                    values_pred=cost_values_b,
                    values_old=cost_values_old_b.detach(),
                    returns=cost_returns_b,
                    clip_eps=clip_eps,
                    use_clip=True,
                )
                total_critic_loss = (
                    self.config["value_coef"] * vf_loss
                    + self.config["cost_value_coef"] * cost_vf_loss
                )

                # 反向传播（Actor）—— 含 MGDA 梯度调和（v2.0）
                actor_params = (
                    list(self.networks.encoder.parameters())
                    + list(self.networks.aggregator.parameters())
                    + list(self.networks.actor.parameters())
                )
                if lambda_s > self.config.get("lambda_threshold", 0.1):
                    # MGDA：分别计算任务梯度和安全梯度
                    loss_task_only = actor_loss + entropy_loss
                    loss_safe_only = safety_penalty

                    self.actor_optimizer.zero_grad()
                    loss_task_only.backward(retain_graph=True)
                    g_task = torch.cat([
                        p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
                        for p in actor_params
                    ])

                    self.actor_optimizer.zero_grad()
                    loss_safe_only.backward(retain_graph=True)
                    g_safe = torch.cat([
                        p.grad.flatten() if p.grad is not None else torch.zeros(p.numel())
                        for p in actor_params
                    ])

                    w_task, w_safe = mgda_solve(g_task, g_safe)
                    combined_loss = w_task * loss_task_only + w_safe * loss_safe_only

                    self.actor_optimizer.zero_grad()
                    combined_loss.backward()
                else:
                    self.actor_optimizer.zero_grad()
                    total_actor_loss.backward(retain_graph=True)

                torch.nn.utils.clip_grad_norm_(actor_params, self.config["max_grad_norm"])
                self.actor_optimizer.step()

                # 反向传播（Critic）
                self.critic_optimizer.zero_grad()
                total_critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.networks.critic.parameters()]
                    + [p for p in self.networks.cost_critic.parameters()],
                    self.config["max_grad_norm"],
                )
                self.critic_optimizer.step()

                all_actor_losses.append(float(total_actor_loss.detach()))
                all_critic_losses.append(float(total_critic_loss.detach()))
                all_entropy.append(float(entropy_mean.detach()))
                all_approx_kl.append(float(approx_kl.detach()))

        # Lagrangian 乘子更新（双时间尺度：每次迭代只更新一次）
        mean_episode_cost = float(data["costs"].mean()) if "costs" in data else 0.0
        # 使用本次收集的平均每步成本估计 J_safe
        mean_step_cost = float(self.buffer.costs[:self.buffer.ptr].mean())
        # 估计每集平均成本（近似）
        estimated_episode_cost = mean_step_cost * self.env.max_steps

        cost_violation = estimated_episode_cost - self.config["safety_budget"]
        all_cost_violations.append(cost_violation)

        # λ ← max(0, λ + α_λ * (Ĵ_safe - d))，裁剪到 [0, lambda_max]
        self.lambda_s = float(np.clip(
            self.lambda_s + self.config["lr_lambda"] * cost_violation,
            0.0,
            self.lambda_max,
        ))

        update_info = {
            "actor_loss": np.mean(all_actor_losses),
            "critic_loss": np.mean(all_critic_losses),
            "entropy": np.mean(all_entropy),
            "approx_kl": np.mean(all_approx_kl),
            "lambda_s": self.lambda_s,
            "J_safe": estimated_episode_cost,
            "estimated_episode_cost": estimated_episode_cost,
            "cost_violation": cost_violation,
            "ev_task": explained_variance(
                self.buffer.values[:self.buffer.ptr],
                self.buffer.rewards[:self.buffer.ptr].mean(axis=1),
            ),
        }
        return update_info

    # ------------------------------------------------------------------
    # 训练主循环
    # ------------------------------------------------------------------

    def train(
        self,
        total_steps: int,
        log_dir: Optional[str] = None,
    ) -> List[Dict]:
        """
        训练主循环。

        Args:
            total_steps: 总训练步数
            log_dir: 日志保存目录（None 则不保存）

        Returns:
            metrics_history: 训练指标历史列表
        """
        print(f"[SafeComm-PSched] 开始训练，总步数={total_steps}，设备={self.device}")
        print(f"  n_agents={self.config['n_agents']}, k={self.config['k']}, "
              f"safety_budget={self.config['safety_budget']}")

        n_params = self.networks.count_parameters()
        print(f"  参数量: {n_params}")

        start_time = time.time()
        log_interval = self.config["log_interval"]

        while self.total_steps < total_steps:
            iter_start = time.time()

            # 数据收集
            rollout_info = self.collect_rollout()

            # PPO + Lagrangian 更新
            update_info = self.update()

            self.n_iterations += 1
            iter_time = time.time() - iter_start

            # 合并统计
            metrics = {
                "iteration": self.n_iterations,
                "total_steps": self.total_steps,
                "iter_time": iter_time,
                **rollout_info,
                **update_info,
            }
            self.metrics_history.append(metrics)

            # 打印日志
            if self.n_iterations % log_interval == 0:
                elapsed = time.time() - start_time
                fps = self.total_steps / max(elapsed, 1)
                print(
                    f"[iter {self.n_iterations:4d} | steps {self.total_steps:7d}] "
                    f"FE={rollout_info['mean_formation_error']:.4f}  "
                    f"cost={rollout_info['mean_episode_cost']:.4f}  "
                    f"λ={update_info['lambda_s']:.4f}  "
                    f"BU={rollout_info['mean_bandwidth_util']:.3f}  "
                    f"KL={update_info['approx_kl']:.5f}  "
                    f"FPS={fps:.0f}"
                )

        print(f"[SafeComm-PSched] 训练完成，总步数={self.total_steps}，"
              f"用时={time.time()-start_time:.1f}s")
        return self.metrics_history

    # ------------------------------------------------------------------
    # 评估
    # ------------------------------------------------------------------

    def evaluate(
        self,
        n_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
    ) -> Dict:
        """
        评估当前策略。

        Returns:
            eval_metrics: 评估指标字典（参见设计规格 §5.4）
        """
        self.networks.eval()

        all_episode_rewards = []
        all_episode_costs = []
        all_formation_errors = []
        all_success = []
        all_collision_rates = []
        all_bw_utils = []
        all_min_dists = []

        with torch.no_grad():
            for ep in range(n_episodes):
                obs_list, _ = self.env.reset()
                ep_reward = 0.0
                ep_cost = 0.0
                ep_bw_utils = []
                done = False

                while not done:
                    phys_edges = self.env.get_physical_graph()
                    sched_result = self.scheduler.schedule_step(
                        self.env.positions, phys_edges
                    )
                    actions, _, _, _ = self.networks.act(
                        obs_list=obs_list,
                        neighbor_lists=sched_result["neighbor_lists"],
                        deterministic=deterministic,
                        device=self.device,
                    )
                    scaled_actions = actions * self.config["action_scale"]
                    obs_list, rewards, costs, done, info = self.env.step(scaled_actions)

                    ep_reward += float(np.mean(rewards))
                    ep_cost += float(np.mean(costs))
                    ep_bw_utils.append(sched_result["bandwidth_utilization"])

                    if render:
                        print(self.env.render())

                all_episode_rewards.append(ep_reward)
                all_episode_costs.append(ep_cost)
                all_formation_errors.append(info["formation_error"])
                all_success.append(info["success"])
                all_collision_rates.append(ep_cost / max(self.env.t, 1))
                all_bw_utils.append(float(np.mean(ep_bw_utils)))
                all_min_dists.append(info["min_dist"])

        eval_metrics = {
            # 任务性能
            "mean_reward": float(np.mean(all_episode_rewards)),
            "std_reward": float(np.std(all_episode_rewards)),
            "formation_error": float(np.mean(all_formation_errors)),
            "success_rate": float(np.mean(all_success)),
            # 安全性
            "collision_rate": float(np.mean(all_collision_rates)),
            "constraint_violation_rate": float(
                np.mean([c > self.config["safety_budget"] for c in all_episode_costs])
            ),
            # 通信效率
            "bandwidth_utilization": float(np.mean(all_bw_utils)),
            "mean_min_dist": float(np.mean(all_min_dists)),
            # Lagrangian 状态
            "lambda_s": self.lambda_s,
        }
        return eval_metrics

    # ------------------------------------------------------------------
    # 模型保存与加载
    # ------------------------------------------------------------------

    def save(self, path: str):
        """保存模型权重和训练状态。"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {
            "networks": self.networks.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "lambda_s": self.lambda_s,
            "total_steps": self.total_steps,
            "n_iterations": self.n_iterations,
            "config": self.config,
        }
        torch.save(state, path)
        print(f"[SafeComm-PSched] 模型已保存至 {path}")

    def load(self, path: str):
        """加载模型权重和训练状态。"""
        state = torch.load(path, map_location=self.device)
        self.networks.load_state_dict(state["networks"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.lambda_s = state["lambda_s"]
        self.total_steps = state["total_steps"]
        self.n_iterations = state["n_iterations"]
        print(f"[SafeComm-PSched] 模型已从 {path} 加载，"
              f"迭代={self.n_iterations}，步数={self.total_steps}")

"""SafeComm-VoI Phase 1 main algorithm."""

from __future__ import annotations

import os
import time
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from algorithms.networks import SafeCommNetworks
from algorithms.ppo_utils import ppo_clip_loss, value_loss
from algorithms.ppo_utils import RolloutBuffer
from algorithms.scheduler import SafetyPriorityScheduler


class SafeCommVoI:
    """MAPPO-Lagrangian prototype with Safety-aware VoI communication scheduling.

    Phase 1 deliberately has no HOCBF-QP shield, no MGDA, and no dual-action
    rollout path. The policy action is scaled and sent directly to the env.
    """

    def __init__(
        self,
        env: Any,
        config: dict[str, Any] | None = None,
        device: str | torch.device = "auto",
    ) -> None:
        self.env = env
        self.n_agents = self._num_agents(env)
        self.obs_dim = int(env.obs_dim)
        self.act_dim = int(env.act_dim)
        self.global_state_dim = self.n_agents * self.obs_dim
        self.device = self._resolve_device(device)

        defaults = self._default_config(env)
        self.config = {**defaults, **(config or {})}
        self.config["hidden_dims"] = list(self.config["hidden_dims"])

        seed = self.config.get("seed")
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

        self.networks = SafeCommNetworks(
            n_agents=self.n_agents,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            msg_dim=int(self.config["msg_dim"]),
            hidden_dims=self.config["hidden_dims"],
            H=int(self.config["H"]),
        ).to(self.device)

        self.scheduler = self._build_scheduler()

        self.buffer = RolloutBuffer(
            n_steps=int(self.config["n_steps"]),
            n_agents=self.n_agents,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            global_state_dim=self.global_state_dim,
        )

        self._actor_params = list(self.networks.encoder.parameters())
        self._actor_params += list(self.networks.aggregator.parameters())
        self._actor_params += list(self.networks.actor.parameters())
        self._critic_params = list(self.networks.critic.parameters())
        self._critic_params += list(self.networks.cost_critic.parameters())

        self.actor_optimizer = torch.optim.Adam(
            self._actor_params, lr=float(self.config["lr_actor"])
        )
        self.critic_optimizer = torch.optim.Adam(
            self._critic_params, lr=float(self.config["lr_critic"])
        )

        self.lambda_s = float(self.config["lambda_init"])
        self.n_iterations = 0
        self.total_steps = 0

    def collect_rollout(self) -> dict[str, float]:
        """Collect one fixed-size rollout using scheduled communication edges."""
        start_time = time.time()
        self.networks.eval()
        self.buffer.reset()

        episode_costs: list[float] = []
        episode_errors: list[float] = []
        bandwidth_utils: list[float] = []
        formation_errors: list[float] = []

        obs_list, _ = self._reset_env_and_scheduler()
        n_episodes = 1
        ep_cost = 0.0
        ep_error = 0.0
        ep_steps = 0
        done = False

        while self.buffer.ptr < self.buffer.n_steps:
            phys_edges = self.env.get_physical_graph()
            active_edges = self.scheduler.schedule(self.env.positions, phys_edges)
            self.scheduler.update_history(
                active_edges, self.env.positions, self.env.velocities
            )
            neighbor_lists = self.scheduler.get_neighbor_lists(active_edges)
            adj_matrix = self._adjacency_from_edges(active_edges)

            policy_actions, log_probs, values, cost_values = self.networks.act(
                obs_list,
                neighbor_lists,
                deterministic=False,
                device=self.device,
            )
            exec_actions = (
                np.asarray(policy_actions, dtype=np.float32)
                * float(self.config["action_scale"])
            )

            obs_array = np.stack(obs_list).astype(np.float32)
            global_state = np.concatenate(obs_list).astype(np.float32)
            next_obs, rewards, costs, done, info = self.env.step(exec_actions)

            self.buffer.add(
                obs=obs_array,
                actions=np.asarray(policy_actions, dtype=np.float32),
                log_probs=np.asarray(log_probs, dtype=np.float32),
                rewards=np.asarray(rewards, dtype=np.float32),
                costs=np.asarray(costs, dtype=np.float32),
                value=self._scalar(values),
                cost_value=self._scalar(cost_values),
                done=done,
                global_state=global_state,
                adj_matrix=adj_matrix,
            )

            step_cost = float(np.mean(costs))
            formation_error = float(info.get("formation_error", 0.0))
            ep_cost += step_cost
            ep_error += formation_error
            ep_steps += 1
            formation_errors.append(formation_error)
            bandwidth_utils.append(self._bandwidth_util(active_edges, phys_edges))
            self.total_steps += 1
            obs_list = next_obs

            if done:
                episode_costs.append(ep_cost)
                episode_errors.append(ep_error / max(ep_steps, 1))
                if self.buffer.ptr >= self.buffer.n_steps:
                    break
                obs_list, _ = self._reset_env_and_scheduler()
                n_episodes += 1
                ep_cost = 0.0
                ep_error = 0.0
                ep_steps = 0
                done = False

        if ep_steps > 0 and not done:
            episode_costs.append(ep_cost)
            episode_errors.append(ep_error / max(ep_steps, 1))

        last_value, last_cost_value = (0.0, 0.0)
        if not done:
            global_state = np.concatenate(obs_list).astype(np.float32)
            with torch.no_grad():
                values, cost_values = self.networks.get_values(global_state)
            last_value = self._scalar(values)
            last_cost_value = self._scalar(cost_values)
        self.buffer.finish_rollout(last_value, last_cost_value)

        elapsed = max(time.time() - start_time, 1e-8)
        return {
            "n_episodes": float(n_episodes),
            "mean_episode_cost": float(np.mean(episode_costs)) if episode_costs else 0.0,
            "mean_bandwidth_util": float(np.mean(bandwidth_utils)) if bandwidth_utils else 0.0,
            "fps": float(self.buffer.ptr / elapsed),
            "mean_formation_error": float(np.mean(formation_errors)) if formation_errors else 0.0,
        }

    def update(self) -> dict[str, float]:
        """Run PPO-Lagrangian updates on the most recent finished rollout."""
        if self.buffer.ptr == 0:
            raise RuntimeError("collect_rollout() must add samples before update()")

        self.networks.train()
        data = self.buffer.get_training_data(
            gamma=float(self.config["gamma"]),
            lam=float(self.config["lam"]),
            device=self.device,
            normalize_advantages=bool(self.config["normalize_advantages"]),
        )

        n_steps = int(data["obs"].shape[0])
        batch_size = min(int(self.config["batch_size"]), n_steps)
        n_epochs = int(self.config["n_epochs"])

        actor_losses: list[float] = []
        critic_losses: list[float] = []
        entropies: list[float] = []
        approx_kls: list[float] = []

        for _ in range(n_epochs):
            permutation = torch.randperm(n_steps, device=self.device)
            for start in range(0, n_steps, batch_size):
                idx = permutation[start : start + batch_size]
                minibatch = {key: value[idx] for key, value in data.items()}

                log_probs, entropy, values, cost_values, _ = self.networks.evaluate_actions_batch(
                    minibatch["obs"],
                    minibatch["actions"],
                    minibatch["adj_matrices"],
                    minibatch["global_states"],
                )

                actor_loss, approx_kl = ppo_clip_loss(
                    log_probs,
                    minibatch["log_probs_old"],
                    minibatch["advantages"],
                    clip_eps=float(self.config["clip_eps"]),
                )
                safety_loss = self._cost_surrogate_loss(
                    log_probs,
                    minibatch["log_probs_old"],
                    minibatch["cost_advantages"],
                )
                entropy_mean = entropy.mean()
                total_actor_loss = (
                    actor_loss
                    + float(self.lambda_s) * safety_loss
                    - float(self.config["entropy_coef"]) * entropy_mean
                )

                self.actor_optimizer.zero_grad(set_to_none=True)
                total_actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    self._actor_params, float(self.config["max_grad_norm"])
                )
                self.actor_optimizer.step()

                critic_loss = (
                    float(self.config["value_coef"])
                    * value_loss(
                        values,
                        minibatch["values_old"],
                        minibatch["returns"],
                        clip_eps=float(self.config["clip_eps"]),
                    )
                    + float(self.config["cost_value_coef"])
                    * value_loss(
                        cost_values,
                        minibatch["cost_values_old"],
                        minibatch["cost_returns"],
                        clip_eps=float(self.config["clip_eps"]),
                    )
                )

                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(
                    self._critic_params, float(self.config["max_grad_norm"])
                )
                self.critic_optimizer.step()

                actor_losses.append(float(total_actor_loss.detach().cpu()))
                critic_losses.append(float(critic_loss.detach().cpu()))
                entropies.append(float(entropy_mean.detach().cpu()))
                approx_kls.append(float(approx_kl.detach().cpu()))

        mean_step_cost = float(self.buffer.costs[: self.buffer.ptr].mean())
        j_safe = mean_step_cost * float(getattr(self.env, "max_steps", self.buffer.ptr))
        self.lambda_s = float(
            np.clip(
                self.lambda_s
                + float(self.config["lr_lambda"])
                * (j_safe - float(self.config["safety_budget"])),
                0.0,
                float(self.config["lambda_max"]),
            )
        )
        self.n_iterations += 1

        return {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "lambda_s": float(self.lambda_s),
            "J_safe": float(j_safe),
        }

    def train(self, total_steps: int) -> dict[str, float]:
        """Collect rollouts and update until ``total_steps`` environment steps."""
        target_steps = int(total_steps)
        metrics: dict[str, float] = {}
        while self.total_steps < target_steps:
            rollout_info = self.collect_rollout()
            update_metrics = self.update()
            metrics = {**rollout_info, **update_metrics}
            if bool(self.config["verbose"]):
                print(
                    f"iter={self.n_iterations} steps={self.total_steps} "
                    f"J_safe={metrics['J_safe']:.3f} lambda_s={self.lambda_s:.3f}"
                )
        return metrics

    def evaluate(
        self, n_episodes: int = 20, deterministic: bool = True
    ) -> dict[str, float]:
        """Run policy evaluation without updating parameters."""
        was_training = self.networks.training
        self.networks.eval()

        episode_costs: list[float] = []
        formation_errors: list[float] = []
        successes: list[float] = []

        for _ in range(int(n_episodes)):
            obs_list, info = self._reset_env_and_scheduler()
            done = False
            ep_cost = 0.0
            ep_errors: list[float] = [float(info.get("formation_error", 0.0))]
            last_info = info

            while not done:
                phys_edges = self.env.get_physical_graph()
                active_edges = self.scheduler.schedule(self.env.positions, phys_edges)
                self.scheduler.update_history(
                    active_edges, self.env.positions, self.env.velocities
                )
                neighbor_lists = self.scheduler.get_neighbor_lists(active_edges)
                policy_actions, _, _, _ = self.networks.act(
                    obs_list,
                    neighbor_lists,
                    deterministic=deterministic,
                    device=self.device,
                )
                exec_actions = (
                    np.asarray(policy_actions, dtype=np.float32)
                    * float(self.config["action_scale"])
                )
                obs_list, _, costs, done, last_info = self.env.step(exec_actions)
                ep_cost += float(np.mean(costs))
                ep_errors.append(float(last_info.get("formation_error", 0.0)))

            episode_costs.append(ep_cost)
            formation_errors.append(float(np.mean(ep_errors)) if ep_errors else 0.0)
            successes.append(1.0 if bool(last_info.get("success", False)) else 0.0)

        if was_training:
            self.networks.train()

        return {
            "mean_formation_error": float(np.mean(formation_errors)) if formation_errors else 0.0,
            "mean_episode_cost": float(np.mean(episode_costs)) if episode_costs else 0.0,
            "success_rate": float(np.mean(successes)) if successes else 0.0,
        }

    def save(self, path: str | os.PathLike[str]) -> None:
        """Persist network, optimizer, Lagrangian, and progress state."""
        path = os.fspath(path)
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(
            {
                "network_state": self.networks.state_dict(),
                "actor_optimizer_state": self.actor_optimizer.state_dict(),
                "critic_optimizer_state": self.critic_optimizer.state_dict(),
                "lambda_s": float(self.lambda_s),
                "n_iterations": int(self.n_iterations),
                "total_steps": int(self.total_steps),
                "config": dict(self.config),
            },
            path,
        )

    def load(self, path: str | os.PathLike[str]) -> None:
        """Restore a checkpoint saved by :meth:`save`."""
        checkpoint = torch.load(os.fspath(path), map_location=self.device)
        self.networks.load_state_dict(checkpoint["network_state"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state"])
        self.lambda_s = float(checkpoint["lambda_s"])
        self.n_iterations = int(checkpoint.get("n_iterations", 0))
        self.total_steps = int(checkpoint.get("total_steps", 0))
        self.config = {**self.config, **dict(checkpoint.get("config", {}))}
        self.config["hidden_dims"] = list(self.config["hidden_dims"])
        self.scheduler = self._build_scheduler()

    def _reset_env_and_scheduler(self) -> tuple[list[np.ndarray], dict[str, Any]]:
        obs_list, info = self.env.reset()
        self.scheduler.reset_episode(self.env.positions, self.env.velocities)
        return obs_list, info

    def _build_scheduler(self) -> SafetyPriorityScheduler:
        return SafetyPriorityScheduler(
            n_agents=self.n_agents,
            k=int(self.config["k"]),
            d_min=float(getattr(self.env, "d_min", 0.5)),
            eps=float(self.config["scheduler_eps"]),
            schedule_mode=str(self.config["schedule_mode"]),
            v_max=float(self.config["v_max_cbf"]),
            dt=float(getattr(self.env, "dt", 0.1)),
            beta=float(self.config["beta"]),
            tau_ref=float(self.config["tau_ref"]),
            tau_max=float(self.config["tau_max"]),
            seed=self.config.get("seed"),
        )

    def _adjacency_from_edges(self, edges: Iterable[Sequence[int]]) -> np.ndarray:
        adj = np.zeros((self.n_agents, self.n_agents), dtype=np.float32)
        for edge in edges:
            i, j = int(edge[0]), int(edge[1])
            if i == j:
                continue
            adj[i, j] = 1.0
            adj[j, i] = 1.0
        return adj

    def _bandwidth_util(
        self,
        active_edges: Sequence[Sequence[int]],
        phys_edges: Sequence[Sequence[int]],
    ) -> float:
        if str(self.config["schedule_mode"]) == "full":
            denom = max(len(phys_edges), 1)
        elif int(self.config["k"]) > 0:
            denom = int(self.config["k"])
        else:
            denom = 1
        return float(len(active_edges) / denom)

    def _cost_surrogate_loss(
        self,
        log_probs: torch.Tensor,
        log_probs_old: torch.Tensor,
        cost_advantages: torch.Tensor,
    ) -> torch.Tensor:
        advantages = cost_advantages.to(dtype=log_probs.dtype, device=log_probs.device)
        while advantages.ndim < log_probs.ndim:
            advantages = advantages.unsqueeze(-1)
        log_ratio = log_probs - log_probs_old
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(
            ratio,
            1.0 - float(self.config["clip_eps"]),
            1.0 + float(self.config["clip_eps"]),
        )
        return torch.max(ratio * advantages, clipped_ratio * advantages).mean()

    @staticmethod
    def _scalar(value: Any) -> float:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            raise ValueError("value must contain at least one scalar")
        return float(arr[0])

    @staticmethod
    def _resolve_device(device: str | torch.device) -> torch.device:
        if str(device) == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @staticmethod
    def _default_config(env: Any) -> dict[str, Any]:
        n_agents = SafeCommVoI._num_agents(env)
        max_edges = n_agents * (n_agents - 1) // 2
        return {
            "k": min(4, max_edges),
            "action_scale": float(getattr(env, "a_max", 1.0)),
            "n_steps": 1024,
            "n_epochs": 4,
            "batch_size": 256,
            "msg_dim": 64,
            "hidden_dims": [256, 256],
            "H": 2,
            "gamma": 0.99,
            "lam": 0.95,
            "clip_eps": 0.2,
            "value_coef": 0.5,
            "cost_value_coef": 0.5,
            "entropy_coef": 0.01,
            "max_grad_norm": 0.5,
            "lr_actor": 3e-4,
            "lr_critic": 3e-4,
            "lr_lambda": 1e-2,
            "safety_budget": 0.5,
            "lambda_init": 0.0,
            "lambda_max": 100.0,
            "beta": 1.0,
            "tau_ref": 1.0,
            "tau_max": 5.0,
            "v_max_cbf": float(getattr(env, "v_max", 0.0)),
            "schedule_mode": "safety_priority",
            "scheduler_eps": 1e-6,
            "normalize_advantages": True,
            "seed": None,
            "verbose": False,
        }

    @staticmethod
    def _num_agents(env: Any) -> int:
        if hasattr(env, "n_agents"):
            return int(env.n_agents)
        if hasattr(env, "n"):
            return int(env.n)
        raise AttributeError("env must expose n_agents or n")


__all__ = ["SafeCommVoI"]

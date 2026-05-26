"""Double-integrator UAV formation environment for SafeComm-VoI Phase 1."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


class UAVFormationEnv:
    """A lightweight multi-agent UAV formation environment.

    State per agent is ``[p_x, p_y, p_z, v_x, v_y, v_z]``. Observations expose
    the agent's absolute state followed by nearest-neighbor relative states.
    """

    def __init__(
        self,
        n_agents: int = 4,
        dt: float = 0.1,
        max_steps: int = 200,
        d_min: float = 0.5,
        R_comm: float = 5.0,
        v_max: float = 3.0,
        a_max: float = 2.0,
        arena_size: float = 10.0,
        formation_radius: Optional[float] = None,
        goal_altitude: float = 1.0,
        w_formation: float = 1.0,
        w_success: float = 5.0,
        success_threshold: float = 0.3,
        K_obs_neighbors: int = 4,
        action_scale: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        if n_agents <= 0:
            raise ValueError("n_agents must be positive")
        if dt <= 0:
            raise ValueError("dt must be positive")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if K_obs_neighbors < 0:
            raise ValueError("K_obs_neighbors must be non-negative")

        self.n_agents = int(n_agents)
        self.n = self.n_agents
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.d_min = float(d_min)
        self.R_comm = float(R_comm)
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        self.arena_size = float(arena_size)
        self.formation_radius = formation_radius
        self.goal_altitude = float(goal_altitude)
        self.w_formation = float(w_formation)
        self.w_success = float(w_success)
        self.success_threshold = float(success_threshold)
        self.K_obs = int(K_obs_neighbors)
        self.K_obs_neighbors = self.K_obs
        self.action_scale = float(action_scale)

        self.obs_dim = 6 + self.K_obs * 6
        self.act_dim = 3

        self.rng = np.random.default_rng(seed)
        self.positions = np.zeros((self.n_agents, 3), dtype=np.float32)
        self.velocities = np.zeros((self.n_agents, 3), dtype=np.float32)
        self.goal_positions = self._make_goal_positions()
        self.t = 0

    @property
    def state(self) -> np.ndarray:
        """Return flattened true state as agent-wise position plus velocity."""
        return np.concatenate((self.positions, self.velocities), axis=1).reshape(-1).astype(np.float32)

    def reset(self, seed: Optional[int] = None) -> Tuple[List[np.ndarray], Dict[str, object]]:
        """Reset the environment and return per-agent observations plus info."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.t = 0
        self.goal_positions = self._make_goal_positions()

        noise = self.rng.uniform(-0.4, 0.4, size=(self.n_agents, 3)).astype(np.float32)
        noise[:, 2] *= 0.25
        self.positions = self.goal_positions + noise
        self.positions = np.clip(self.positions, -self.arena_size, self.arena_size).astype(np.float32)
        self.velocities = np.zeros((self.n_agents, 3), dtype=np.float32)

        return self._get_obs_list(), self._get_info()

    def step(
        self, actions: np.ndarray
    ) -> Tuple[List[np.ndarray], List[float], List[float], bool, Dict[str, object]]:
        """Advance double-integrator dynamics by one step."""
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.n_agents, self.act_dim):
            raise ValueError(f"actions must have shape {(self.n_agents, self.act_dim)}, got {actions.shape}")

        exec_actions = np.nan_to_num(actions * self.action_scale, nan=0.0, posinf=self.a_max, neginf=-self.a_max)
        exec_actions = np.clip(exec_actions, -self.a_max, self.a_max)

        self.velocities = np.clip(
            self.velocities + exec_actions * self.dt,
            -self.v_max,
            self.v_max,
        ).astype(np.float32)
        self.positions = (self.positions + self.velocities * self.dt).astype(np.float32)
        self._apply_bounds()

        self.t += 1

        info = self._get_info()
        costs = self._collision_costs()
        rewards = self._rewards(costs, bool(info["success"]))
        done = self.t >= self.max_steps
        return self._get_obs_list(), rewards, costs, done, info

    def get_physical_graph(self) -> List[Tuple[int, int]]:
        """Return all true-position communication pairs within ``R_comm``."""
        if self.R_comm <= 0.0:
            return []

        edges: List[Tuple[int, int]] = []
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                dist = float(np.linalg.norm(self.positions[i] - self.positions[j]))
                if dist <= self.R_comm:
                    edges.append((i, j))
        return edges

    def compute_cbf_values(self) -> Dict[Tuple[int, int], float]:
        """Return non-negative pairwise CBF margins from true positions."""
        cbf_values: Dict[Tuple[int, int], float] = {}
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                dist = float(np.linalg.norm(self.positions[i] - self.positions[j]))
                cbf_values[(i, j)] = float(max(dist - self.d_min, 0.0) ** 2)
        return cbf_values

    def _get_obs(self, agent_idx: int) -> np.ndarray:
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        obs[:3] = self.positions[agent_idx]
        obs[3:6] = self.velocities[agent_idx]

        if self.K_obs == 0:
            return obs

        neighbor_ids = [j for j in range(self.n_agents) if j != agent_idx]
        neighbor_ids.sort(key=lambda j: float(np.linalg.norm(self.positions[j] - self.positions[agent_idx])))

        for slot, neighbor_idx in enumerate(neighbor_ids[: self.K_obs]):
            start = 6 + slot * 6
            obs[start : start + 3] = self.positions[neighbor_idx] - self.positions[agent_idx]
            obs[start + 3 : start + 6] = self.velocities[neighbor_idx] - self.velocities[agent_idx]
        return obs

    def _get_obs_list(self) -> List[np.ndarray]:
        return [self._get_obs(i) for i in range(self.n_agents)]

    def _make_goal_positions(self) -> np.ndarray:
        if self.formation_radius is None:
            radius = max(2.0, self.n_agents * max(self.d_min, 0.1) / np.pi)
        else:
            radius = float(self.formation_radius)

        angles = np.linspace(0.0, 2.0 * np.pi, self.n_agents, endpoint=False)
        goals = np.zeros((self.n_agents, 3), dtype=np.float32)
        goals[:, 0] = radius * np.cos(angles)
        goals[:, 1] = radius * np.sin(angles)
        goals[:, 2] = self.goal_altitude
        return goals

    def _formation_errors(self) -> np.ndarray:
        return np.linalg.norm(self.positions - self.goal_positions, axis=1).astype(np.float32)

    def _get_info(self) -> Dict[str, object]:
        errors = self._formation_errors()
        formation_error = float(np.mean(errors))
        return {
            "formation_error": formation_error,
            "success": bool(formation_error <= self.success_threshold),
            "positions": self.positions.copy(),
            "goal_positions": self.goal_positions.copy(),
            "min_dist": self._min_pair_distance(),
            "t": self.t,
        }

    def _collision_costs(self) -> List[float]:
        costs = np.zeros(self.n_agents, dtype=np.float32)
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                dist = float(np.linalg.norm(self.positions[i] - self.positions[j]))
                if dist < self.d_min:
                    costs[i] = 1.0
                    costs[j] = 1.0
        return [float(c) for c in costs]

    def _rewards(self, costs: List[float], success: bool) -> List[float]:
        errors = self._formation_errors()
        rewards = -self.w_formation * errors - 10.0 * np.asarray(costs, dtype=np.float32)
        if success:
            rewards = rewards + self.w_success
        return [float(r) for r in rewards]

    def _min_pair_distance(self) -> float:
        min_dist = float("inf")
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                dist = float(np.linalg.norm(self.positions[i] - self.positions[j]))
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    def _apply_bounds(self) -> None:
        before_clip = self.positions.copy()
        self.positions = np.clip(self.positions, -self.arena_size, self.arena_size).astype(np.float32)
        hit_bound = before_clip != self.positions
        self.velocities[hit_bound] = 0.0

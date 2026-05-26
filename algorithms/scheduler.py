"""Safety-aware VoI communication scheduler."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Edge = Tuple[int, int]


class SafetyPriorityScheduler:
    """TopK scheduler using the Safety-aware VoI priority from the design spec."""

    _VALID_MODES = {"safety_priority", "full", "random"}

    def __init__(
        self,
        n_agents: int,
        k: int,
        d_min: float = 0.5,
        eps: float = 1e-6,
        schedule_mode: str = "safety_priority",
        v_max: float = 0.0,
        dt: float = 0.1,
        beta: float = 1.0,
        tau_ref: float = 1.0,
        tau_max: float = 5.0,
        seed: Optional[int] = None,
    ) -> None:
        self.n_agents = self._validate_int("n_agents", n_agents, min_value=1)
        self.k = self._validate_int("k", k, min_value=0)
        self.d_min = self._validate_float("d_min", d_min, min_value=0.0, inclusive=True)
        self.eps = self._validate_float("eps", eps, min_value=0.0, inclusive=False)
        if schedule_mode not in self._VALID_MODES:
            raise ValueError(
                f"schedule_mode must be one of {sorted(self._VALID_MODES)}, "
                f"got {schedule_mode!r}"
            )
        self.schedule_mode = schedule_mode
        self.v_max = self._validate_float("v_max", v_max, min_value=0.0, inclusive=True)
        self.dt = self._validate_float("dt", dt, min_value=0.0, inclusive=False)
        self.beta = self._validate_float("beta", beta, min_value=0.0, inclusive=True)
        self.tau_ref = self._validate_float("tau_ref", tau_ref, min_value=0.0, inclusive=False)
        self.tau_max = self._validate_float("tau_max", tau_max, min_value=0.0, inclusive=True)
        self._rng = np.random.default_rng(seed)

        self._dt_comm: Dict[Edge, float] = {}
        self._last_pos_j: Dict[Edge, np.ndarray] = {}
        self._last_vel_j: Dict[Edge, np.ndarray] = {}
        self._initialized = False

    def reset_episode(
        self, positions: np.ndarray, velocities: Optional[np.ndarray] = None
    ) -> None:
        """Initialize pair-wise communication history from the current true state."""
        positions_arr = self._validate_state_array("positions", positions)
        velocities_arr = self._coerce_velocities(velocities)

        self._dt_comm = {}
        self._last_pos_j = {}
        self._last_vel_j = {}
        for edge in self._all_pairs():
            _, j = edge
            self._dt_comm[edge] = 0.0
            self._last_pos_j[edge] = positions_arr[j].copy()
            self._last_vel_j[edge] = velocities_arr[j].copy()
        self._initialized = True

    def update_history(
        self,
        active_edges: Iterable[Sequence[int]],
        positions: np.ndarray,
        velocities: Optional[np.ndarray] = None,
    ) -> None:
        """Advance all pair ages and refresh only pairs selected for communication."""
        self._require_initialized()
        positions_arr = self._validate_state_array("positions", positions)
        velocities_arr = self._coerce_velocities(velocities)
        active_set = set(self._normalize_edges(active_edges))

        for edge in self._all_pairs():
            if edge in active_set:
                _, j = edge
                self._dt_comm[edge] = 0.0
                self._last_pos_j[edge] = positions_arr[j].copy()
                self._last_vel_j[edge] = velocities_arr[j].copy()
            else:
                self._dt_comm[edge] += self.dt

    def compute_voi_priority(
        self, positions: np.ndarray, edges: Optional[Iterable[Sequence[int]]] = None
    ) -> Dict[Edge, float]:
        """Compute strict Safety-aware VoI priorities for the requested edges."""
        self._require_initialized()
        positions_arr = self._validate_state_array("positions", positions)
        normalized_edges = self._all_pairs() if edges is None else self._normalize_edges(edges)

        priorities: Dict[Edge, float] = {}
        for edge in normalized_edges:
            i, _ = edge
            dt_comm = self._dt_comm[edge]
            d_buf = self.v_max * dt_comm
            dist = float(np.linalg.norm(positions_arr[i] - self._last_pos_j[edge]))
            h_sched = max(dist - self.d_min - d_buf, 0.0) ** 2
            aoi = min(dt_comm / self.tau_ref, self.tau_max)
            numerator = 1.0 + self.beta * aoi
            priorities[edge] = numerator / max(h_sched, self.eps)
        return priorities

    def schedule(self, positions: np.ndarray, phys_edges: Iterable[Sequence[int]]) -> List[Edge]:
        """Select active communication edges without mutating communication history."""
        edges = self._normalize_edges(phys_edges)
        if not edges:
            return []

        if self.schedule_mode == "full":
            return edges

        if self.k == 0:
            return []

        if self.schedule_mode == "random":
            if len(edges) <= self.k:
                return edges
            chosen = self._rng.choice(len(edges), size=self.k, replace=False)
            return [edges[int(idx)] for idx in chosen]

        priorities = self.compute_voi_priority(positions, edges)
        ranked = sorted(edges, key=lambda edge: (-priorities[edge], edge))
        return ranked[: self.k]

    def get_neighbor_lists(self, active_edges: Iterable[Sequence[int]]) -> Dict[int, List[int]]:
        """Return undirected neighbor lists for every agent."""
        neighbors = {i: [] for i in range(self.n_agents)}
        for i, j in self._normalize_edges(active_edges):
            neighbors[i].append(j)
            neighbors[j].append(i)
        return neighbors

    def _normalize_edges(self, edges: Iterable[Sequence[int]]) -> List[Edge]:
        seen = set()
        normalized: List[Edge] = []
        for edge in edges:
            pair = self._normalize_pair(edge)
            if pair not in seen:
                seen.add(pair)
                normalized.append(pair)
        return normalized

    def _normalize_pair(self, edge: Sequence[int]) -> Edge:
        if len(edge) != 2:
            raise ValueError(f"edge must contain exactly two agent indices, got {edge!r}")
        a, b = edge
        i = self._validate_int("edge index", a, min_value=0)
        j = self._validate_int("edge index", b, min_value=0)
        if i >= self.n_agents or j >= self.n_agents:
            raise ValueError(f"edge indices must be < n_agents ({self.n_agents}), got {edge!r}")
        if i == j:
            raise ValueError(f"self edges are not allowed, got {edge!r}")
        return (i, j) if i < j else (j, i)

    def _all_pairs(self) -> List[Edge]:
        return [
            (i, j)
            for i in range(self.n_agents)
            for j in range(i + 1, self.n_agents)
        ]

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("reset_episode must be called before using the scheduler")

    def _validate_state_array(self, name: str, value: np.ndarray) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)
        expected_shape = (self.n_agents, 3)
        if arr.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} must contain only finite values")
        return arr

    def _coerce_velocities(self, velocities: Optional[np.ndarray]) -> np.ndarray:
        if velocities is None:
            return np.zeros((self.n_agents, 3), dtype=np.float64)
        return self._validate_state_array("velocities", velocities)

    @staticmethod
    def _validate_int(name: str, value: int, min_value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer")
        value = int(value)
        if value < min_value:
            raise ValueError(f"{name} must be >= {min_value}")
        return value

    @staticmethod
    def _validate_float(
        name: str, value: float, min_value: float, inclusive: bool
    ) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite real number")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if inclusive:
            valid = value >= min_value
            op = ">="
        else:
            valid = value > min_value
            op = ">"
        if not valid:
            raise ValueError(f"{name} must be {op} {min_value}")
        return value

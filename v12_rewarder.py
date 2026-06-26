"""Potential-based reward shaping for routing.

Two potentials are supported:
  - "distance": Phi(s) = -sum of distances to upcoming-gate qubits.
        Used by V15 and V16. Biases toward "bring upcoming gates close,"
        which is right for QFT but wrong for parallel circuits.
  - "critical_path": Phi(s) = -ASAP-critical-path of the routed-so-far prefix.
        Used by V17. Aligned with the actual evaluation objective. Per-step
        shaping equals the unoptimized critical-path increase the action
        causes, regardless of circuit family. This is the same signal
        SABRE-MS uses classically (lambda * max(finish[q1], finish[q2])),
        which is empirically the strongest classical proxy for makespan.

By Ng et al. 1999, both choices preserve the optimal policy. Empirically
they bias the policy differently during finite training because they shape
the loss landscape differently. Critical-path Phi is family-agnostic and
aligned with the eval metric.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
from qgym.envs.routing.routing_rewarders import SwapQualityRewarder
from qgym.envs.routing.routing_state import RoutingState


class PotentialShapedRewarder(SwapQualityRewarder):
    def __init__(
        self,
        gate_durations: dict[str, int],
        illegal_action_penalty: float = -10.0,
        penalty_per_swap: float = -2.0,
        reward_per_surpass: float = 2.0,
        good_swap_reward: float = 2.0,
        # Potential-based shaping
        potential_lambda: float = 1.0,
        gamma: float = 1.0,
        potential_kind: str = "distance",  # "distance" or "critical_path"
        # Terminal makespan
        final_makespan_weight: float = 1.0,
        optimize_terminal: bool = True,
        realistic_hardware: Any = None,
    ) -> None:
        super().__init__(
            illegal_action_penalty=illegal_action_penalty,
            penalty_per_swap=penalty_per_swap,
            reward_per_surpass=reward_per_surpass,
            good_swap_reward=good_swap_reward,
        )
        self._gate_durations = dict(gate_durations)
        self._lambda = float(potential_lambda)
        self._gamma = float(gamma)
        if potential_kind not in ("distance", "critical_path"):
            raise ValueError(f"potential_kind must be 'distance' or 'critical_path', got {potential_kind!r}")
        self._potential_kind = potential_kind
        self._final_makespan_weight = float(final_makespan_weight)
        self._optimize_terminal = bool(optimize_terminal)
        self._realistic_hardware = realistic_hardware
        self._dist_cache: np.ndarray | None = None
        self._dist_cache_id: int | None = None

    def _get_distance_matrix(self, state: RoutingState) -> np.ndarray:
        """Cached all-pairs shortest-path distance on the connection graph."""
        cg = state.connection_graph
        if self._dist_cache is None or self._dist_cache_id != id(cg):
            n = state.n_qubits
            d = dict(nx.all_pairs_shortest_path_length(cg))
            mat = np.zeros((n, n), dtype=np.float32)
            for u in range(n):
                for v in range(n):
                    mat[u, v] = float(d[u].get(v, n + 1))
            self._dist_cache = mat
            self._dist_cache_id = id(cg)
        return self._dist_cache

    def _potential_distance(self, state: RoutingState) -> float:
        """Phi(s) = -sum of distances. We return -sum_of_dist so HIGHER is BETTER."""
        circuit = state.interaction_circuit
        position = int(state.position)
        max_reach = state.max_observation_reach
        end = min(position + max_reach, len(circuit))
        if end <= position:
            return 0.0
        dist = self._get_distance_matrix(state)
        mapping = state.mapping
        total = 0.0
        for k in range(position, end):
            lq1, lq2 = circuit[k]
            p1 = int(mapping[int(lq1)])
            p2 = int(mapping[int(lq2)])
            total += float(dist[p1, p2])
        return -total  # negative so Phi is bounded above by 0 (all adjacent)

    def _potential_critical_path(self, state: RoutingState) -> float:
        """Phi(s) = -ASAP-critical-path of routed-so-far prefix.

        Negative so Phi is bounded above by 0 (empty prefix). Per-step shaping
        r_shape = gamma*Phi(s') - Phi(s) is non-positive and equals minus the
        increase in critical path that the action caused: 0 if the SWAP was
        on idle qubits, large-negative if it extended the critical path.
        This is the SABRE-MS makespan term, in RL reward form.
        """
        from pipeline import asap_makespan, routing_to_circuit_prefix
        prefix = routing_to_circuit_prefix(state)
        if not prefix:
            return 0.0
        return -float(asap_makespan(prefix, self._gate_durations))

    def _potential(self, state: RoutingState) -> float:
        if self._potential_kind == "critical_path":
            return self._potential_critical_path(state)
        return self._potential_distance(state)

    def compute_reward(
        self, *, old_state: RoutingState, action: int, new_state: RoutingState,
    ) -> float:
        # Base reward from SwapQualityRewarder
        reward = super().compute_reward(
            old_state=old_state, action=action, new_state=new_state
        )

        if self._is_illegal(action, old_state):
            return reward

        # Potential-based shaping: r_shape = gamma * Phi(s') - Phi(s)
        # = -(gamma*sum_new) + sum_old. Positive if sum_new < sum_old (got closer).
        if self._lambda != 0.0:
            phi_old = self._potential(old_state)
            phi_new = self._potential(new_state)
            reward += self._lambda * (self._gamma * phi_new - phi_old)

        # Terminal makespan signal
        if self._final_makespan_weight != 0.0 and new_state.is_done():
            from pipeline import routing_to_circuit, asap_makespan
            full = routing_to_circuit(new_state)
            if self._optimize_terminal:
                from optimize import optimize_circuit
                full = optimize_circuit(full, new_state.n_qubits)
            if self._realistic_hardware is not None:
                from realistic_scheduler import realistic_makespan
                makespan = realistic_makespan(full, self._realistic_hardware)
            else:
                makespan = asap_makespan(full, self._gate_durations)
            reward -= self._final_makespan_weight * makespan

        return reward

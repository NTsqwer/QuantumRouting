"""V19 observation wrapper: V14-lite features + V18 per-edge features.

V18 dropped V14-lite's qubit_finish_times and interaction_gates_ahead in favor
of per-edge features. On linear5/qft this broke the policy (V18 det = 43.04 vs
V14-lite 19.66). Hypothesis: QFT needs the long-horizon lookahead V14-lite
preserved.

V19 keeps everything useful:
  - qgym defaults (mapping, is_legal_surpass, interaction_gates_ahead)
  - V14-lite additions (qubit_finish_times, critical_path_so_far)
  - V18 additions (per-edge dist_now/after/max_finish, front_layer_dist)

Drops: connection_graph (constant per topology).
"""

from __future__ import annotations

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium.spaces import Box, Dict as DictSpace

from pipeline import routing_to_circuit_prefix


SCALE_CYCLES = 50.0
CNOT_DUR = 2
SWAP_DUR = 6


def _finish_times(routed_gates, n_qubits):
    free = np.zeros(n_qubits, dtype=np.int_)
    for g in routed_gates:
        d = SWAP_DUR if g.name == "swap" else CNOT_DUR
        if g.q2 is None or g.q1 == g.q2:
            free[g.q1] += d
        else:
            start = max(int(free[g.q1]), int(free[g.q2]))
            free[g.q1] = start + d
            free[g.q2] = start + d
    return free


class V19ObservationWrapper(gym.ObservationWrapper):
    """V14-lite features + V18 per-edge features, all in one observation."""

    def __init__(self, env, max_makespan=500, extended_set_size=10,
                 extended_weight=0.5):
        super().__init__(env)
        self._n_qubits = env._state.connection_graph.number_of_nodes()
        self._n_edges = env._state.n_connections
        self._max_makespan = int(max_makespan)
        self._extended_set_size = int(extended_set_size)
        self._extended_weight = float(extended_weight)
        self._edges = list(env._state.connection_graph.edges())
        d = dict(nx.all_pairs_shortest_path_length(env._state.connection_graph))
        self._dist = np.zeros((self._n_qubits, self._n_qubits), dtype=np.float32)
        for u in range(self._n_qubits):
            for v in range(self._n_qubits):
                self._dist[u, v] = float(d[u].get(v, self._n_qubits + 1))

        norm_high = float(self._max_makespan / SCALE_CYCLES)
        n_e = self._n_edges

        # Build observation space:
        # - Keep qgym defaults except connection_graph
        # - Add V14-lite scheduling features
        # - Add V18 per-edge features
        old_space = env.observation_space
        kept = {}
        if isinstance(old_space, DictSpace):
            for k, v in old_space.spaces.items():
                if k == "connection_graph":
                    continue
                kept[k] = v

        kept["qubit_finish_times"] = Box(
            low=0.0, high=norm_high, shape=(self._n_qubits,), dtype=np.float32,
        )
        kept["critical_path_so_far"] = Box(
            low=0.0, high=norm_high, shape=(1,), dtype=np.float32,
        )
        kept["front_layer_dist"] = Box(
            low=0.0, high=10.0, shape=(self._extended_set_size,), dtype=np.float32,
        )
        kept["edge_dist_now"] = Box(
            low=0.0, high=10.0, shape=(n_e,), dtype=np.float32,
        )
        kept["edge_dist_after"] = Box(
            low=-10.0, high=10.0, shape=(n_e,), dtype=np.float32,
        )
        kept["edge_max_finish"] = Box(
            low=0.0, high=norm_high, shape=(n_e,), dtype=np.float32,
        )
        self.observation_space = DictSpace(kept)

    def observation(self, obs):
        state = self.env._state
        n_q = self._n_qubits
        n_e = self._n_edges
        edges = self._edges
        dist = self._dist
        mapping = np.asarray(state.mapping, dtype=int)
        position = int(state.position)
        circuit = state.interaction_circuit

        # Drop connection_graph from upstream obs
        new_obs = {k: v for k, v in obs.items() if k != "connection_graph"}

        # V14-lite features
        prefix = routing_to_circuit_prefix(state)
        finish = _finish_times(prefix, n_q)
        finish_clipped = np.clip(finish, 0, self._max_makespan).astype(np.float32)
        new_obs["qubit_finish_times"] = (finish_clipped / SCALE_CYCLES).astype(np.float32)
        cp = float(min(int(finish.max()), self._max_makespan)) / SCALE_CYCLES
        new_obs["critical_path_so_far"] = np.array([cp], dtype=np.float32)

        # V18 features
        front_dist = np.zeros(self._extended_set_size, dtype=np.float32)
        end = min(position + self._extended_set_size, len(circuit))
        for k in range(position, end):
            lq1, lq2 = int(circuit[k][0]), int(circuit[k][1])
            front_dist[k - position] = dist[int(mapping[lq1]), int(mapping[lq2])]

        ext_end = min(position + 1 + self._extended_set_size, len(circuit))
        front_sum = float(front_dist[0]) if end > position else 0.0
        ext_sum = 0.0
        ext_count = 0
        for k in range(position + 1, ext_end):
            lq1, lq2 = int(circuit[k][0]), int(circuit[k][1])
            ext_sum += float(dist[int(mapping[lq1]), int(mapping[lq2])])
            ext_count += 1
        if ext_count > 0:
            ext_sum /= ext_count
        h_now = front_sum + self._extended_weight * ext_sum

        edge_dist_now = np.zeros(n_e, dtype=np.float32)
        edge_dist_after = np.zeros(n_e, dtype=np.float32)
        edge_max_finish = np.zeros(n_e, dtype=np.float32)

        for ei, (u, v) in enumerate(edges):
            edge_dist_now[ei] = dist[int(mapping[u]), int(mapping[v])]
            new_mapping = mapping.copy()
            new_mapping[u], new_mapping[v] = int(new_mapping[v]), int(new_mapping[u])
            front_after = float(dist[int(new_mapping[int(circuit[position][0])]),
                                     int(new_mapping[int(circuit[position][1])])]) if end > position else 0.0
            ext_after_sum = 0.0
            ext_after_count = 0
            for k in range(position + 1, ext_end):
                lq1, lq2 = int(circuit[k][0]), int(circuit[k][1])
                ext_after_sum += float(dist[int(new_mapping[lq1]), int(new_mapping[lq2])])
                ext_after_count += 1
            if ext_after_count > 0:
                ext_after_sum /= ext_after_count
            h_after = front_after + self._extended_weight * ext_after_sum
            edge_dist_after[ei] = h_after - h_now

            phys_u = int(mapping[u])
            phys_v = int(mapping[v])
            edge_max_finish[ei] = max(int(finish[phys_u]), int(finish[phys_v])) / SCALE_CYCLES

        norm_high = float(self._max_makespan / SCALE_CYCLES)
        new_obs["front_layer_dist"] = np.clip(front_dist, 0, 10.0).astype(np.float32)
        new_obs["edge_dist_now"] = np.clip(edge_dist_now, 0, 10.0).astype(np.float32)
        new_obs["edge_dist_after"] = np.clip(edge_dist_after, -10.0, 10.0).astype(np.float32)
        new_obs["edge_max_finish"] = np.clip(edge_max_finish, 0.0, norm_high).astype(np.float32)
        return new_obs

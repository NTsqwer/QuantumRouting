"""V14-lite observation wrapper: minimal scheduling-aware features.

Drops from V14:
  - qubit_busyness (redundant with interaction_gates_ahead + mapping)
  - connection_graph (constant per topology, agent memorizes it)

Keeps:
  - qgym defaults except connection_graph: mapping, is_legal_surpass, interaction_gates_ahead
  - qubit_finish_times (normalized by /50)
  - critical_path_so_far (normalized by /50)
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict as DictSpace

from pipeline import routing_to_circuit_prefix


SCALE_CYCLES = 50.0
CNOT_DUR = 2
SWAP_DUR = 6


def qubit_finish_times(routed_gates, n_qubits):
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


class SchedulingAwareObservationWrapperLite(gym.ObservationWrapper):
    """Slimmer obs wrapper: drops connection_graph + qubit_busyness."""

    def __init__(self, env, max_makespan=500):
        super().__init__(env)
        self._n_qubits = env._state.connection_graph.number_of_nodes()
        self._max_makespan = int(max_makespan)
        norm_high = float(self._max_makespan / SCALE_CYCLES)
        old_space = env.observation_space

        # Keep qgym defaults EXCEPT connection_graph
        kept = {}
        if isinstance(old_space, DictSpace):
            for k, v in old_space.spaces.items():
                if k == "connection_graph":
                    continue
                kept[k] = v
        # Add scheduling features
        kept["qubit_finish_times"] = Box(
            low=0.0, high=norm_high, shape=(self._n_qubits,), dtype=np.float32,
        )
        kept["critical_path_so_far"] = Box(
            low=0.0, high=norm_high, shape=(1,), dtype=np.float32,
        )
        self.observation_space = DictSpace(kept)

    def observation(self, obs):
        new_obs = {k: v for k, v in obs.items() if k != "connection_graph"}
        state = self.env._state
        prefix = routing_to_circuit_prefix(state)
        finish = qubit_finish_times(prefix, self._n_qubits)
        finish_clipped = np.clip(finish, 0, self._max_makespan).astype(np.float32)
        new_obs["qubit_finish_times"] = (finish_clipped / SCALE_CYCLES).astype(np.float32)
        cp = float(min(int(finish.max()), self._max_makespan)) / SCALE_CYCLES
        new_obs["critical_path_so_far"] = np.array([cp], dtype=np.float32)
        return new_obs

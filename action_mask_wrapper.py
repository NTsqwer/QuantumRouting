"""Action mask wrapper for the qgym Routing env.

MaskablePPO from sb3-contrib expects either an env method `action_masks()` or
an env attribute that exposes the mask. We expose a method `action_masks()`
that returns a boolean array of length n_actions, where True = action is
allowed and False = action is masked out.

Semantics:
  - actions 0..n_connections-1 = SWAPs at each edge. Always legal.
  - action n_connections        = surpass next gate. Legal only if next gate's
                                  qubits are physically adjacent under the
                                  current mapping (= is_legal_surpass[0]).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class ActionMaskWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Find the inner Routing state (may be nested under other wrappers)
        unwrapped = env
        while hasattr(unwrapped, "env"):
            if hasattr(unwrapped, "_state"):
                break
            unwrapped = unwrapped.env
        self._inner_with_state = unwrapped

    def _state(self):
        # Search for the underlying RoutingState across wrapper layers
        e = self.env
        while True:
            if hasattr(e, "_state"):
                return e._state
            if hasattr(e, "env"):
                e = e.env
            else:
                raise RuntimeError("could not find RoutingState in wrapper stack")

    def action_masks(self) -> np.ndarray:
        s = self._state()
        n_actions = int(s.n_connections) + 1
        mask = np.ones(n_actions, dtype=bool)
        # surpass action = n_connections; legal iff next gate's qubits are adjacent
        if int(s.position) < len(s.interaction_circuit):
            q1, q2 = s.interaction_circuit[int(s.position)]
            surpass_legal = bool(s.is_legal_surpass(int(q1), int(q2)))
        else:
            # episode would be done, but mask harmlessly
            surpass_legal = True
        mask[int(s.n_connections)] = surpass_legal
        return mask

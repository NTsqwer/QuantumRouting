"""Mixed-family interaction circuit generator.

Trains the routing agent on a mix of circuit structures (random / QFT /
parallel / trotter) so it doesn't overfit to qgym's default uniform-random
2q pairs and can generalize to structured families it's evaluated on.

Yields ``np.ndarray`` of shape (length, 2) — same format as qgym's
:class:`BasicInteractionGenerator`.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np

from qgym.generators.interaction import InteractionGenerator
from qgym.utils.input_parsing import parse_seed
from qgym.utils.input_validation import check_graph_is_valid_topology


class MixedInteractionGenerator(InteractionGenerator):
    """Yield random / QFT / parallel-layer / trotter circuits at a mix.

    Default mix: 40% random, 25% qft (all-to-all subset), 20% parallel
    (random matchings), 15% trotter (brick-wall). Each circuit's length is
    capped at ``max_length``.
    """

    def __init__(
        self,
        max_length: int = 10,
        seed: Any = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.max_length = int(max_length)
        self.rng = parse_seed(seed)
        self.finite = False
        self.n_qubits: int

        if weights is None:
            weights = {"random": 0.40, "qft": 0.25, "parallel": 0.20, "trotter": 0.15}
        total = sum(weights.values())
        self._families = list(weights.keys())
        self._probs = np.array([weights[k] / total for k in self._families])

    def set_state_attributes(
        self, *, connection_graph: nx.Graph | None = None, **kwargs: Any
    ) -> None:
        connection_graph = check_graph_is_valid_topology(
            connection_graph, "connection_graph"
        )
        self.n_qubits = connection_graph.number_of_nodes()

    def __next__(self) -> np.ndarray:
        family = self._families[int(self.rng.choice(len(self._families), p=self._probs))]
        n = self.n_qubits

        if family == "random":
            # Random length 1..max_length, random distinct pairs.
            length = int(self.rng.integers(1, self.max_length + 1))
            circuit = np.zeros((length, 2), dtype=int)
            for i in range(length):
                circuit[i] = self.rng.choice(n, size=2, replace=False)
            return circuit

        if family == "qft":
            # Subset of all-to-all interactions, capped at max_length, with
            # random qubit relabeling so each instance differs.
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            self.rng.shuffle(pairs)
            length = min(self.max_length, len(pairs))
            perm = self.rng.permutation(n)
            base = np.array(pairs[:length], dtype=int)
            return perm[base].astype(int)

        if family == "parallel":
            # Random matchings: floor(n/2) disjoint pairs per layer, several layers.
            pairs_per_layer = n // 2
            n_layers = max(1, self.max_length // max(pairs_per_layer, 1))
            circuit = []
            for _ in range(n_layers):
                perm = self.rng.permutation(n)
                for k in range(pairs_per_layer):
                    circuit.append((int(perm[2 * k]), int(perm[2 * k + 1])))
                if len(circuit) >= self.max_length:
                    break
            return np.array(circuit[: self.max_length], dtype=int)

        if family == "trotter":
            # Brick-wall pattern with random qubit permutation.
            even = [(i, i + 1) for i in range(0, n - 1, 2)]
            odd = [(i, i + 1) for i in range(1, n - 1, 2)]
            n_steps = max(1, self.max_length // (len(even) + len(odd)))
            circuit: list[tuple[int, int]] = []
            for _ in range(n_steps):
                circuit.extend(even)
                circuit.extend(odd)
                if len(circuit) >= self.max_length:
                    break
            perm = self.rng.permutation(n)
            base = np.array(circuit[: self.max_length], dtype=int)
            return perm[base].astype(int)

        raise ValueError(f"unknown family {family!r}")

    def __repr__(self) -> str:
        return (
            f"MixedInteractionGenerator(max_length={self.max_length}, "
            f"weights={dict(zip(self._families, self._probs))})"
        )

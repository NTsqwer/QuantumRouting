"""Wrap an InteractionGenerator so each yielded circuit is pre-permuted by
SabreLayout. Used for Regime B training where the agent learns to route
circuits that have already been laid-out by SABRE's layout pass.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import SabreLayout

from qgym.generators.interaction import InteractionGenerator
from qgym.utils.input_parsing import parse_seed
from qgym.utils.input_validation import check_graph_is_valid_topology


class SabreLayoutGeneratorWrapper(InteractionGenerator):
    """Wraps any InteractionGenerator, applying SabreLayout to each emitted
    circuit before yielding.
    """

    def __init__(self, inner: InteractionGenerator, seed: Any = None) -> None:
        self.inner = inner
        self.rng = parse_seed(seed)
        self.finite = getattr(inner, "finite", False)
        self.max_length = getattr(inner, "max_length", 10)
        self.n_qubits: int
        self.coupling: CouplingMap | None = None

    def set_state_attributes(
        self, *, connection_graph: nx.Graph | None = None, **kwargs: Any
    ) -> None:
        connection_graph = check_graph_is_valid_topology(
            connection_graph, "connection_graph"
        )
        self.n_qubits = connection_graph.number_of_nodes()
        self.inner.set_state_attributes(connection_graph=connection_graph, **kwargs)
        edges = list(connection_graph.edges())
        self.coupling = CouplingMap(
            [list(e) for e in edges] + [list(reversed(e)) for e in edges]
        )

    def _apply_sabre_layout(self, circuit: np.ndarray, seed: int) -> np.ndarray:
        if circuit.size == 0:
            return circuit
        qc = QuantumCircuit(self.n_qubits)
        for q1, q2 in circuit:
            qc.cx(int(q1), int(q2))
        pm = PassManager([SabreLayout(coupling_map=self.coupling, seed=seed, max_iterations=2)])
        pm.run(qc)
        layout = pm.property_set.get("layout")
        if layout is None:
            return circuit
        perm = np.zeros(self.n_qubits, dtype=int)
        for i, q in enumerate(qc.qubits):
            perm[i] = layout[q]
        return perm[circuit].astype(int)

    def __next__(self) -> np.ndarray:
        circuit = next(self.inner)
        seed = int(self.rng.integers(0, 2**31 - 1))
        try:
            return self._apply_sabre_layout(circuit, seed)
        except Exception:
            return circuit

    def __repr__(self) -> str:
        return f"SabreLayoutGeneratorWrapper(inner={self.inner!r})"

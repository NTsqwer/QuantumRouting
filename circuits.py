"""Structured interaction circuit families for the characterization study.

These are the circuit families where prior work (HPCA 2022 "Not All SWAPs
Have the Same Cost") showed SWAP count diverging from final makespan. Random
small circuits have a near-perfect SWAP-count -> makespan correlation;
structured circuits with parallelism opportunities are where the failure
mode is expected to appear.

All generators return ``np.ndarray`` of shape (n_gates, 2) -- the same format
used by qgym's :class:`InteractionGenerator`.
"""

from __future__ import annotations

import numpy as np


def random_interactions(n_qubits: int, length: int, rng: np.random.Generator) -> np.ndarray:
    """Random distinct-pair interactions (matches qgym BasicInteractionGenerator).

    Provided for apples-to-apples comparison with our existing random results.
    """
    circuit = np.zeros((length, 2), dtype=int)
    for i in range(length):
        circuit[i] = rng.choice(n_qubits, size=2, replace=False)
    return circuit


def qft_all_to_all(n_qubits: int) -> np.ndarray:
    """All-to-all pairwise interactions in QFT-style cascading order.

    For n qubits emits gates (0,1), (0,2), ..., (0,n-1), (1,2), (1,3), ...,
    (n-2, n-1). High dependency depth on most qubits, lots of forced SWAPs
    on sparse topologies.
    """
    pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    return np.array(pairs, dtype=int)


def parallel_layers(n_qubits: int, n_layers: int, rng: np.random.Generator) -> np.ndarray:
    """``n_layers`` layers, each is a random matching: floor(n/2) disjoint pairs.

    Within a layer all gates are independent and *could* be scheduled in
    parallel if routing preserves their qubit assignments. This maximises
    the gap between SWAP-count-optimal and makespan-optimal routings.
    """
    pairs_per_layer = n_qubits // 2
    circuit = []
    for _ in range(n_layers):
        perm = rng.permutation(n_qubits)
        for i in range(pairs_per_layer):
            circuit.append((int(perm[2 * i]), int(perm[2 * i + 1])))
    return np.array(circuit, dtype=int)


def trotter_brick(n_qubits: int, n_steps: int) -> np.ndarray:
    """Brick-wall Trotter pattern: alternating even/odd nearest-neighbour layers.

    Each Trotter step is two layers: first all (2k, 2k+1) edges, then all
    (2k+1, 2k+2) edges. Standard quantum-simulation circuit shape.
    """
    even_layer = [(i, i + 1) for i in range(0, n_qubits - 1, 2)]
    odd_layer = [(i, i + 1) for i in range(1, n_qubits - 1, 2)]
    circuit: list[tuple[int, int]] = []
    for _ in range(n_steps):
        circuit.extend(even_layer)
        circuit.extend(odd_layer)
    return np.array(circuit, dtype=int)


def staircase(n_qubits: int, n_repeats: int) -> np.ndarray:
    """CNOT staircase: (0,1), (1,2), (2,3), ..., (n-2, n-1), repeated.
    Linear chain: trivially nearest-neighbour. Non-linear topologies need many SWAPs.
    """
    pairs = [(i, i + 1) for i in range(n_qubits - 1)]
    circuit: list[tuple[int, int]] = []
    for _ in range(n_repeats):
        circuit.extend(pairs)
    return np.array(circuit, dtype=int)


def hwea_layer(n_qubits: int, n_layers: int) -> np.ndarray:
    """Hardware-efficient ansatz style: alternating layers of
    CNOT(i, i+1) for even i, then CNOT(i, i+1) for odd i.
    Like Trotter but emitted in a different order; tests scheduling of
    full-layer-then-full-layer structure.
    """
    even_pairs = [(i, i + 1) for i in range(0, n_qubits - 1, 2)]
    odd_pairs = [(i, i + 1) for i in range(1, n_qubits - 1, 2)]
    circuit: list[tuple[int, int]] = []
    for layer_idx in range(n_layers):
        if layer_idx % 2 == 0:
            circuit.extend(even_pairs)
        else:
            circuit.extend(odd_pairs)
    return np.array(circuit, dtype=int)


def all_to_all_layered(n_qubits: int, n_layers: int, rng: np.random.Generator) -> np.ndarray:
    """Each layer is a random complete pairing of qubits.
    More dense than 'parallel' (uses every qubit each layer when possible).
    """
    pairs_per_layer = n_qubits // 2
    circuit = []
    for _ in range(n_layers):
        perm = rng.permutation(n_qubits)
        for k in range(pairs_per_layer):
            circuit.append((int(perm[2 * k]), int(perm[2 * k + 1])))
    return np.array(circuit, dtype=int)


FAMILIES = {"random", "qft", "parallel", "trotter", "deferred", "asymmetric",
             "staircase", "hwea", "all2all"}


def deferred(n_qubits: int, n_phase1: int, rng: np.random.Generator) -> np.ndarray:
    """Phase 1: many CNOTs on qubits 0,1 (keeping them busy).
    Phase 2: a single CNOT on qubits 0 and (n_qubits-1) — the most distant pair.

    During phase 1, qubits 2..n-1 are idle. A smart router can park the
    routing SWAPs for phase 2 *during* phase 1, in parallel with the
    busy work on qubits 0,1.
    """
    out = []
    for _ in range(n_phase1):
        out.append((0, 1))
    out.append((0, n_qubits - 1))
    return np.array(out, dtype=int)


def asymmetric(
    n_qubits: int, n_left_phase: int, n_right_phase: int, rng: np.random.Generator
) -> np.ndarray:
    """Two parallel-friendly phases, each on opposite ends, then a cross-cut.

    Phase A: ``n_left_phase`` CNOTs on qubits 0,1.
    Phase B: ``n_right_phase`` CNOTs on qubits (n-2, n-1).
    Cross: a CNOT spanning logical 1 and logical (n-2).

    A and B can run in parallel from the start (disjoint qubits). The cross
    needs SWAPs in the middle of the topology, which can be parked during
    A or B.
    """
    out = []
    for _ in range(n_left_phase):
        out.append((0, 1))
    for _ in range(n_right_phase):
        out.append((n_qubits - 2, n_qubits - 1))
    out.append((1, n_qubits - 2))
    return np.array(out, dtype=int)


def make_circuits(
    family: str,
    n_qubits: int,
    n_circuits: int,
    *,
    seed: int,
    length: int = 10,
    n_layers: int = 5,
    n_steps: int = 4,
    n_phase1: int = 5,
    n_left_phase: int = 4,
    n_right_phase: int = 4,
) -> list[np.ndarray]:
    """Build ``n_circuits`` test circuits of the given family."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_circuits):
        if family == "random":
            out.append(random_interactions(n_qubits, length, rng))
        elif family == "qft":
            # QFT is deterministic; perturb by random qubit relabelling so
            # each "circuit" is a different instance of the family.
            perm = rng.permutation(n_qubits)
            base = qft_all_to_all(n_qubits)
            out.append(perm[base].astype(int))
        elif family == "parallel":
            out.append(parallel_layers(n_qubits, n_layers, rng))
        elif family == "trotter":
            # Trotter brick pattern is deterministic on a chain. Permute qubit
            # labels so different instances arise.
            perm = rng.permutation(n_qubits)
            base = trotter_brick(n_qubits, n_steps)
            out.append(perm[base].astype(int))
        elif family == "deferred":
            out.append(deferred(n_qubits, n_phase1, rng))
        elif family == "asymmetric":
            out.append(asymmetric(n_qubits, n_left_phase, n_right_phase, rng))
        elif family == "staircase":
            perm = rng.permutation(n_qubits)
            base = staircase(n_qubits, n_repeats=max(1, length // (n_qubits - 1)))
            out.append(perm[base].astype(int))
        elif family == "hwea":
            perm = rng.permutation(n_qubits)
            base = hwea_layer(n_qubits, n_layers=n_layers)
            out.append(perm[base].astype(int))
        elif family == "all2all":
            out.append(all_to_all_layered(n_qubits, n_layers, rng))
        else:
            raise ValueError(f"unknown family {family!r}; choices: {sorted(FAMILIES)}")
    return out

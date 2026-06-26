"""Realistic hardware model for the routing-scheduling experiments.

Adds three constraints to qgym's idealized model, all motivated by superconducting
hardware (IBM Falcon/Eagle scale):

1. **Per-edge CNOT durations** vary by ~20% from the mean. Some couplers are
   slower than others due to qubit frequency detuning and calibration variance.
   Reported on IBM Falcon: 199-309 ns CNOT durations across edges.

2. **CNOT-direction asymmetry**: CNOT(a, b) and CNOT(b, a) have different
   durations on the same edge (cross-resonance is asymmetric). ~10-15% typical
   on IBM hardware.

3. **Crosstalk between adjacent-but-disjoint qubit pairs**: two CNOTs whose
   qubit sets are *not* sharing a qubit but are spatially adjacent in the
   connection graph cannot fire in the same cycle without crosstalk error.
   Standard mitigation in the literature is to enforce a temporal gap.

References:
- IBM Falcon CNOT timing: arxiv 2410.00916
- "Quantum Crosstalk Analysis for Simultaneous Gate Operations" PRX Quantum
- "Not All SWAPs Have the Same Cost" HPCA 2022
- "Timing and Resource-Aware Mapping..." Lao et al., IEEE TCAD 2021

This module provides:
- ``RealisticHardware`` dataclass capturing the constraints.
- ``make_linear5_realistic`` and similar factory functions.
- Helper ``gate_duration`` and ``crosstalk_blocks`` predicates used by
  the realistic scheduler in ``realistic_scheduler.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import networkx as nx

from qgym.custom_types import Gate


@dataclass
class RealisticHardware:
    """Hardware model with realistic per-edge timings + crosstalk."""

    n_qubits: int
    connection_graph: nx.Graph
    # Per-(physical-qubit-pair) CNOT durations, asymmetric. Key is (control, target).
    cnot_durations: dict[tuple[int, int], int] = field(default_factory=dict)
    # SWAP duration per edge = 3 * (avg CNOT duration on that edge); precomputed.
    swap_durations: dict[frozenset[int], int] = field(default_factory=dict)
    # Crosstalk distance: gates whose qubits are within this graph distance
    # (and don't share a qubit) cannot fire in the same cycle.
    crosstalk_max_distance: int = 1

    def gate_duration(self, gate: Gate) -> int:
        """Cycles required by this gate on its specific qubits."""
        # 1-qubit gates: convention is q1 == q2. Duration is 1 cycle on
        # superconducting hardware (~10-50 ns vs CNOT ~200 ns).
        if gate.q1 == gate.q2:
            return 1
        if gate.name in ("cnot", "cx"):
            d = self.cnot_durations.get((gate.q1, gate.q2))
            if d is None:
                raise KeyError(f"no CNOT duration set for ({gate.q1}, {gate.q2})")
            return d
        if gate.name == "swap":
            key = frozenset((gate.q1, gate.q2))
            d = self.swap_durations.get(key)
            if d is None:
                raise KeyError(f"no SWAP duration set for {set(key)}")
            return d
        raise ValueError(f"unknown 2q gate name {gate.name!r}")

    def crosstalk_blocks(self, g1: Gate, g2: Gate) -> bool:
        """True iff scheduling g1 and g2 in overlapping cycles is forbidden by crosstalk.

        Two gates crosstalk iff they share NO qubit (so qubit availability
        doesn't already serialize them) AND their qubit sets contain
        physical qubits within ``crosstalk_max_distance`` hops in the
        connection graph.

        1-qubit gates (q1==q2) do NOT cause crosstalk in this model -- they
        are fast (1 cycle) and well-shielded; the dominant crosstalk effect
        on superconducting hardware is between simultaneous 2-qubit gates.
        """
        if g1.q1 == g1.q2 or g2.q1 == g2.q2:
            return False
        s1 = {g1.q1, g1.q2}
        s2 = {g2.q1, g2.q2}
        if s1 & s2:
            # Share a qubit => qubit availability already serializes.
            return False
        # Min graph distance between any qubit of g1 and any qubit of g2.
        for q1 in s1:
            for q2 in s2:
                try:
                    d = nx.shortest_path_length(self.connection_graph, q1, q2)
                except nx.NetworkXNoPath:
                    continue
                if d <= self.crosstalk_max_distance:
                    return True
        return False


def _build(
    graph: nx.Graph,
    cnot_table: dict[tuple[int, int], int],
    crosstalk_max_distance: int = 1,
) -> RealisticHardware:
    """Construct a RealisticHardware from a (control, target) -> duration table.

    Forward and reverse CNOT durations are independent (asymmetry). SWAP
    duration on an edge is the average of forward + reverse + forward = 3
    CNOT decomposition cost on that pair, computed as
    ``2 * forward + reverse`` cycles (the standard SWAP recipe).
    """
    swap = {}
    for u, v in graph.edges():
        # SWAP(u, v) decomposes as CNOT(u,v), CNOT(v,u), CNOT(u,v) = 2*fwd + rev
        fwd = cnot_table.get((u, v))
        rev = cnot_table.get((v, u))
        if fwd is None or rev is None:
            raise KeyError(f"missing direction for edge ({u}, {v})")
        swap[frozenset((u, v))] = 2 * fwd + rev
    return RealisticHardware(
        n_qubits=graph.number_of_nodes(),
        connection_graph=graph,
        cnot_durations=cnot_table,
        swap_durations=swap,
        crosstalk_max_distance=crosstalk_max_distance,
    )


def make_linear5_realistic(crosstalk_max_distance: int = 1) -> RealisticHardware:
    """5-qubit linear chain with realistic per-edge CNOT durations.

    Edges:  0─1─2─3─4
    Each edge gets a forward and reverse CNOT duration. Asymmetry per edge
    is ~10-15%. Across edges, durations vary ~20% (some edges are
    intrinsically slower due to frequency-collision avoidance).

    Crosstalk: two CNOTs on edges that are within 1 hop of each other (e.g.
    (0,1) and (2,3) on a linear chain — qubits 1 and 2 are adjacent) can't
    co-fire. With crosstalk_max_distance=0, the constraint is removed.
    """
    g = nx.path_graph(5)
    # Edge timings (cycles). Slow edge = (2, 3) coupler; fast edges around it.
    cnot_table = {
        (0, 1): 2, (1, 0): 2,    # fast pair, symmetric
        (1, 2): 2, (2, 1): 3,    # mild asymmetry
        (2, 3): 4, (3, 2): 3,    # slow edge with asymmetry (slower forward)
        (3, 4): 2, (4, 3): 2,    # fast pair
    }
    return _build(g, cnot_table, crosstalk_max_distance=crosstalk_max_distance)


def make_tshape5_realistic(crosstalk_max_distance: int = 1) -> RealisticHardware:
    """5-qubit T-shape: 0-1-2-3 with 4 attached to 2."""
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (2, 3), (2, 4)])
    cnot_table = {
        (0, 1): 2, (1, 0): 2,
        (1, 2): 3, (2, 1): 2,    # asymmetric
        (2, 3): 2, (3, 2): 2,
        (2, 4): 3, (4, 2): 3,    # slow branch
    }
    return _build(g, cnot_table, crosstalk_max_distance=crosstalk_max_distance)


def make_grid3x3_realistic(crosstalk_max_distance: int = 1) -> RealisticHardware:
    """3x3 grid (9 qubits). Layout::

         0 - 1 - 2
         |   |   |
         3 - 4 - 5
         |   |   |
         6 - 7 - 8

    Per-edge CNOT durations vary ~25% (typical of IBM Falcon-class
    hardware). Center qubit (4) has 4 connections; corner qubits have 2.
    """
    g = nx.Graph()
    edges = [
        (0, 1), (1, 2),               # top row
        (3, 4), (4, 5),               # middle row
        (6, 7), (7, 8),               # bottom row
        (0, 3), (3, 6),               # left column
        (1, 4), (4, 7),               # middle column
        (2, 5), (5, 8),               # right column
    ]
    g.add_edges_from(edges)
    # Edge durations: vary by ~25%. Slow center edges (around q4) to make
    # the routing decision interesting -- agent should prefer routing
    # around the slow center if possible.
    cnot_table = {}
    base_durations = {
        # Fast edges (corners)
        frozenset({0, 1}): 2, frozenset({1, 2}): 2,
        frozenset({6, 7}): 2, frozenset({7, 8}): 2,
        frozenset({0, 3}): 2, frozenset({2, 5}): 2,
        # Medium edges
        frozenset({3, 6}): 3, frozenset({5, 8}): 3,
        # Slow center edges (more crosstalk-prone in real hardware)
        frozenset({3, 4}): 3, frozenset({4, 5}): 3,
        frozenset({1, 4}): 4, frozenset({4, 7}): 3,
    }
    for u, v in edges:
        d = base_durations[frozenset({u, v})]
        cnot_table[(u, v)] = d
        # Asymmetric reverse: ~10% lower
        cnot_table[(v, u)] = max(2, d - 1) if d > 2 else d
    return _build(g, cnot_table, crosstalk_max_distance=crosstalk_max_distance)


def gate_durations_table(hw: RealisticHardware) -> dict:
    """Format compatible with the legacy asap_makespan API: maps gate name to
    a single duration value. Used only as a fallback when callers can't pass
    the full hardware model. Returns the *mean* per-name duration.
    """
    if hw.cnot_durations:
        cnot_mean = sum(hw.cnot_durations.values()) / len(hw.cnot_durations)
    else:
        cnot_mean = 2
    if hw.swap_durations:
        swap_mean = sum(hw.swap_durations.values()) / len(hw.swap_durations)
    else:
        swap_mean = 6
    return {"cnot": int(round(cnot_mean)), "swap": int(round(swap_mean))}


__all__ = [
    "RealisticHardware",
    "make_linear5_realistic",
    "make_tshape5_realistic",
    "gate_durations_table",
]

"""Deterministic list-scheduler that respects realistic hardware constraints.

Given a routed circuit and a ``RealisticHardware`` model, produces a schedule
respecting:
- Per-edge gate durations (different durations per qubit pair).
- Direction asymmetry (CNOT(a,b) vs CNOT(b,a)).
- Qubit availability (a qubit can do one gate at a time).
- Crosstalk: two gates with adjacent-but-disjoint qubit sets cannot run in
  the same cycle window.

Output: total makespan in cycles.

This is the realistic-hardware analogue of ``pipeline.asap_makespan``.
"""

from __future__ import annotations

import numpy as np

from qgym.custom_types import Gate
from realistic_machine import RealisticHardware


def realistic_makespan(circuit: list[Gate], hw: RealisticHardware) -> int:
    """Schedule the circuit greedily under realistic constraints.

    Strategy: process gates in input order (preserves the routing's
    intended dependency order). For each gate find the earliest time at
    which (a) both its qubits are free, (b) no concurrently-scheduled
    gate causes crosstalk. Place it there.

    With ``crosstalk_max_distance=0`` and uniform durations, this reduces
    to plain ASAP (and qubit-availability handles disjoint-qubit
    parallelism for free).
    """
    if not circuit:
        return 0

    n_qubits = hw.n_qubits
    qubit_available = np.zeros(n_qubits, dtype=np.int_)
    active: list[tuple[int, int, Gate]] = []

    for gate in circuit:
        duration = hw.gate_duration(gate)
        if gate.q1 == gate.q2:
            # 1-qubit gate: no crosstalk, only one qubit busy.
            start = int(qubit_available[gate.q1])
            finish = start + duration
            active.append((start, finish, gate))
            qubit_available[gate.q1] = finish
            continue

        candidate = int(max(qubit_available[gate.q1], qubit_available[gate.q2]))
        while True:
            blocking_finish = None
            for (s, f, g_other) in active:
                if f <= candidate:
                    continue
                if not hw.crosstalk_blocks(gate, g_other):
                    continue
                if blocking_finish is None or f > blocking_finish:
                    blocking_finish = f
            if blocking_finish is None:
                break
            candidate = blocking_finish

        start = candidate
        finish = start + duration
        active.append((start, finish, gate))
        qubit_available[gate.q1] = finish
        qubit_available[gate.q2] = finish

    return int(qubit_available.max())


__all__ = ["realistic_makespan"]

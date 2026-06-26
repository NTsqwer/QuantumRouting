"""Qiskit-backed gate-cancellation optimizer for routed qgym circuits.

This is the optimization pass missing from qgym. Without it, routings that
look equivalent by SWAP count produce nearly identical makespans (the
structural argument for why scheduling-aware routing rewards don't help in
plain qgym). With it, routings that *create cancellation opportunities*
become measurably cheaper -- which is the HPCA 2022 effect, and is what
gives the routing -> scheduling interaction enough room to be exploitable.

Pipeline:
  1. ``BasisTranslator`` decomposes each SWAP into 3 CNOTs.
  2. ``Collect2qBlocks`` + ``ConsolidateBlocks`` group adjacent 2q gates.
  3. ``UnitarySynthesis(basis=['cx', 'u3'])`` re-synthesizes each block at
     the *minimum* CNOT count (2-qubit Cliffords need 0..3 CNOTs).
  4. ``InverseCancellation`` + ``CommutativeCancellation`` clean up.

Output: the same {cnot, swap} alphabet plus optional 1q gates which we
discard for makespan accounting (1q gates are zero-cost in our model).

Equivalence is verified with ``Operator.equiv`` -- this is unitary
equivalence (allows global phase, no qubit relabelling).
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.circuit.library import CXGate
from qiskit.quantum_info import Operator
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    BasisTranslator,
    Collect2qBlocks,
    CommutativeCancellation,
    ConsolidateBlocks,
    InverseCancellation,
    UnitarySynthesis,
)

from qgym.custom_types import Gate


_PASS_MANAGER = PassManager([
    # Accept any 1q + 2q gates in input. Translate to (cx, u3) basis so the
    # downstream Collect/Consolidate/UnitarySynthesis passes can work uniformly.
    BasisTranslator(SessionEquivalenceLibrary, ["cx", "u3"]),
    Collect2qBlocks(),
    ConsolidateBlocks(basis_gates=["cx"]),
    UnitarySynthesis(basis_gates=["cx", "u3"]),
    InverseCancellation([CXGate()]),
    CommutativeCancellation(),
    InverseCancellation([CXGate()]),
])


def qgym_to_qiskit(circuit: list[Gate], n_qubits: int) -> QuantumCircuit:
    """Convert a qgym ``list[Gate]`` (cnot/swap only) to a qiskit QuantumCircuit."""
    qc = QuantumCircuit(n_qubits)
    for g in circuit:
        if g.name == "cnot":
            qc.cx(g.q1, g.q2)
        elif g.name == "swap":
            qc.swap(g.q1, g.q2)
        else:
            raise ValueError(f"unsupported gate {g.name!r} (expected cnot/swap)")
    return qc


def qiskit_to_qgym_2q_only(qc: QuantumCircuit) -> list[Gate]:
    """Convert qiskit QuantumCircuit to qgym ``list[Gate]``, keeping ONLY 2q gates.

    1-qubit gates are zero-cost in our cycle model and are discarded for
    makespan accounting. The returned list contains only ``cnot`` and
    ``swap`` Gate namedtuples.
    """
    out: list[Gate] = []
    for instr in qc.data:
        op = instr.operation
        if len(instr.qubits) == 1:
            continue
        qubits = [qc.find_bit(q).index for q in instr.qubits]
        if op.name == "cx":
            out.append(Gate("cnot", int(qubits[0]), int(qubits[1])))
        elif op.name == "swap":
            out.append(Gate("swap", int(qubits[0]), int(qubits[1])))
        else:
            raise ValueError(
                f"transpiler emitted unexpected 2q gate {op.name!r}"
            )
    return out


def qiskit_to_qgym_with_1q(qc: QuantumCircuit) -> list[Gate]:
    """Convert qiskit QuantumCircuit to qgym ``list[Gate]``, keeping 1q AND 2q gates.

    1-qubit gates are encoded as ``Gate(name, q, q)`` (q1==q2). They take
    1 cycle in the makespan model.

    Used when the optimizer is run on a full mixed-gate circuit and we
    want the makespan to account for 1q gate timing too.
    """
    out: list[Gate] = []
    for instr in qc.data:
        op = instr.operation
        qubits = [qc.find_bit(q).index for q in instr.qubits]
        if len(qubits) == 1:
            out.append(Gate(op.name, int(qubits[0]), int(qubits[0])))
        elif len(qubits) == 2:
            if op.name == "cx":
                out.append(Gate("cnot", int(qubits[0]), int(qubits[1])))
            elif op.name == "swap":
                out.append(Gate("swap", int(qubits[0]), int(qubits[1])))
            else:
                raise ValueError(
                    f"transpiler emitted unexpected 2q gate {op.name!r}"
                )
        else:
            raise ValueError(
                f"transpiler emitted unsupported {len(qubits)}-qubit gate"
            )
    return out


def optimize_circuit_with_1q(circuit: list[Gate], n_qubits: int) -> list[Gate]:
    """Apply qiskit cancellation, KEEPING 1q gates (encoded as q1==q2).

    Used when the routed circuit has 1q gates we want to preserve for
    realistic makespan accounting (each 1q gate = 1 cycle).
    """
    if not circuit:
        return []
    qc = qgym_to_qiskit_with_1q(circuit, n_qubits)
    optimized = _PASS_MANAGER.run(qc)
    return qiskit_to_qgym_with_1q(optimized)


def qgym_to_qiskit_with_1q(circuit: list[Gate], n_qubits: int) -> QuantumCircuit:
    """Convert a qgym ``list[Gate]`` (1q AND 2q) into a qiskit QuantumCircuit.

    1q gates encoded as ``Gate(name, q, q)`` are dispatched to the matching
    Qiskit gate method (``qc.h(q)``, ``qc.x(q)``, etc.).
    """
    qc = QuantumCircuit(n_qubits)
    for g in circuit:
        if g.q1 == g.q2:
            method = getattr(qc, g.name, None)
            if method is None:
                raise ValueError(f"unknown 1q gate name {g.name!r}")
            method(g.q1)
        elif g.name in ("cnot", "cx"):
            qc.cx(g.q1, g.q2)
        elif g.name == "swap":
            qc.swap(g.q1, g.q2)
        else:
            method = getattr(qc, g.name, None)
            if method is None:
                raise ValueError(f"unknown 2q gate name {g.name!r}")
            method(g.q1, g.q2)
    return qc


def optimize_circuit(circuit: list[Gate], n_qubits: int) -> list[Gate]:
    """Apply qiskit-backed cancellation optimization to a routed qgym circuit.

    Returns a new ``list[Gate]`` containing only ``cnot`` (and possibly
    ``swap``) gates -- 1q gates emitted by qiskit are discarded for the
    makespan model. The returned circuit is unitary-equivalent to the input
    on the 2q subspace; verify with ``assert_equivalent_2q``.
    """
    if not circuit:
        return []
    qc = qgym_to_qiskit(circuit, n_qubits)
    optimized = _PASS_MANAGER.run(qc)
    return qiskit_to_qgym_2q_only(optimized)


def assert_equivalent(c1: list[Gate], c2: list[Gate], n_qubits: int) -> None:
    """Verify two qgym circuits implement the same unitary (up to global phase).

    NOTE: this asserts equivalence including 1q rotations. To check that the
    *2q-only projection* of an optimized circuit is equivalent, the caller
    must rebuild the optimized qiskit circuit *before* discarding 1q gates.
    """
    qc1 = qgym_to_qiskit(c1, n_qubits)
    qc2 = qgym_to_qiskit(c2, n_qubits)
    if not Operator(qc1).equiv(Operator(qc2)):
        raise AssertionError(
            f"circuits not equivalent:\n  before ({len(c1)}): {c1}\n"
            f"  after  ({len(c2)}): {c2}"
        )


def optimize_and_verify(
    circuit: list[Gate], n_qubits: int, *, verify: bool = True
) -> list[Gate]:
    """Optimize a circuit and (optionally) verify unitary equivalence including 1q.

    The verification is done on the FULL optimized circuit (cx + 1q gates),
    which is unitary-equivalent to the input. The returned ``list[Gate]``
    has 1q gates discarded for makespan accounting.
    """
    if not circuit:
        return []
    qc = qgym_to_qiskit(circuit, n_qubits)
    optimized = _PASS_MANAGER.run(qc)
    if verify:
        if not Operator(qc).equiv(Operator(optimized)):
            raise AssertionError(
                "qiskit transpiler produced non-equivalent circuit; "
                "this should not happen with our pass manager"
            )
    return qiskit_to_qgym_2q_only(optimized)

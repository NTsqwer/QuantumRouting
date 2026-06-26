"""Routing->scheduling pipeline bridge for RQ5 pass-interaction experiments.

Provides:
  - routing_to_circuit: convert a completed routing episode to list[Gate] for scheduling
  - routing_to_circuit_prefix: build the routed prefix at any partial RoutingState
  - asap_makespan: deterministic ASAP scheduler (the fixed evaluator)
  - make_machine_properties: helper to create standard hardware spec
  - SchedulingAwareRewarder: scheduling-aware routing rewarder (Condition B)

Condition A baseline is qgym's :class:`SwapQualityRewarder` (use directly,
no wrapper needed). It is the strongest pure-SWAP-objective rewarder qgym
ships -- BasicRewarder is too weak for a fair comparison.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from qgym.custom_types import Gate
from qgym.envs.routing.routing_rewarders import SwapQualityRewarder
from qgym.envs.scheduling.machine_properties import MachineProperties
from qgym.envs.scheduling.rulebook import CommutationRulebook

if TYPE_CHECKING:
    from qgym.envs.routing.routing_state import RoutingState


def _build_routed_circuit(
    interaction_circuit: np.ndarray,
    swap_gates_inserted,
    n_qubits: int,
    up_to_position: int,
    include_pending_swaps: bool,
) -> list[Gate]:
    """Internal: replay mapping evolution and emit a list of Gate namedtuples.

    Args:
        interaction_circuit: Original logical interaction list (n, 2).
        swap_gates_inserted: Iterable of (position, q1, q2) tuples.
        n_qubits: Total qubits.
        up_to_position: Process interactions [0, up_to_position).
        include_pending_swaps: Also include SWAPs whose position == up_to_position.
    """
    swaps_at: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
    for position, pq1, pq2 in swap_gates_inserted:
        swaps_at[int(position)].append((int(pq1), int(pq2)))

    # phys_of_logical[lq] = current physical position of logical qubit lq.
    # Initially identity; each SWAP at physicals (pq1, pq2) exchanges the
    # two logicals currently sitting at those positions.
    phys_of_logical = np.arange(n_qubits, dtype=np.int_)
    # logical_at_phys[pq] = logical qubit currently at physical pq.
    logical_at_phys = np.arange(n_qubits, dtype=np.int_)
    result: list[Gate] = []

    def apply_swap(pq1: int, pq2: int) -> None:
        l1, l2 = int(logical_at_phys[pq1]), int(logical_at_phys[pq2])
        logical_at_phys[pq1], logical_at_phys[pq2] = l2, l1
        phys_of_logical[l1], phys_of_logical[l2] = pq2, pq1

    for i in range(up_to_position):
        for pq1, pq2 in swaps_at.get(i, []):
            result.append(Gate("swap", pq1, pq2))
            apply_swap(pq1, pq2)

        lq1, lq2 = interaction_circuit[i]
        result.append(Gate(
            "cnot",
            int(phys_of_logical[int(lq1)]),
            int(phys_of_logical[int(lq2)]),
        ))

    if include_pending_swaps:
        for pq1, pq2 in swaps_at.get(up_to_position, []):
            result.append(Gate("swap", pq1, pq2))
            apply_swap(pq1, pq2)

    return result


def routing_to_circuit(routing_state: RoutingState) -> list[Gate]:
    """Convert a completed routing episode into a routed gate list.

    Replays the mapping evolution implied by `swap_gates_inserted`, then maps
    each logical interaction gate to its physical qubits, with SWAPs interleaved.

    Args:
        routing_state: A RoutingState after episode completion (is_done() == True).

    Returns:
        Ordered list of Gate namedtuples with names "swap" and "cnot" on physical qubits.
    """
    return _build_routed_circuit(
        routing_state.interaction_circuit,
        routing_state.swap_gates_inserted,
        routing_state.n_qubits,
        up_to_position=len(routing_state.interaction_circuit),
        include_pending_swaps=False,
    )


def routing_to_circuit_prefix(routing_state: RoutingState) -> list[Gate]:
    """Build the routed circuit so far for a partial RoutingState.

    Includes all SWAPs already inserted (including any at position == current,
    which have been physically applied even though their target interaction
    hasn't been surpassed yet) and all interactions [0, position).
    """
    return _build_routed_circuit(
        routing_state.interaction_circuit,
        routing_state.swap_gates_inserted,
        routing_state.n_qubits,
        up_to_position=int(routing_state.position),
        include_pending_swaps=True,
    )


def make_machine_properties(
    n_qubits: int,
    *,
    cnot_duration: int = 2,
    swap_duration: int = 6,
) -> MachineProperties:
    """Return MachineProperties with CNOT and SWAP gates for n_qubits qubits."""
    mp = MachineProperties(n_qubits)
    mp.add_gates({"cnot": cnot_duration, "swap": swap_duration})
    return mp


def asap_makespan(
    circuit: list[Gate], gate_durations: dict[str, int]
) -> int:
    """Compute the ASAP makespan of a circuit given per-gate-type durations.

    Gates issue as early as qubit availability allows; circuit order is
    preserved per qubit; independent gates on different qubits run in
    parallel.

    Supports both 2-qubit gates (``q1 != q2``) and 1-qubit gates
    (``q1 == q2`` -- standard convention in our pipeline). 1-qubit gates
    use the same gate_durations dict; missing keys default to 1 cycle.
    """
    if not circuit:
        return 0

    n_qubits = max(max(g.q1, g.q2) for g in circuit) + 1
    qubit_available = np.zeros(n_qubits, dtype=np.int_)

    for gate in circuit:
        duration = gate_durations.get(gate.name, 1)
        if gate.q1 == gate.q2:
            # 1-qubit gate: only one qubit busy
            start = int(qubit_available[gate.q1])
            qubit_available[gate.q1] = start + duration
        else:
            start = int(max(qubit_available[gate.q1], qubit_available[gate.q2]))
            qubit_available[gate.q1] = start + duration
            qubit_available[gate.q2] = start + duration

    return int(qubit_available.max())


def cnot_same_control(gate1: Gate, gate2: Gate) -> bool:
    """Commutation rule: two CNOTs with the same control qubit commute.

    CNOT_c->t1 and CNOT_c->t2 act independently on disjoint targets and only
    read the control. The standard textbook commutation rule.
    """
    if gate1.name == "cnot" and gate2.name == "cnot":
        # Gate(name, q1, q2) -> q1 = control, q2 = target in qgym convention.
        if gate1.q1 == gate2.q1 and gate1.q2 != gate2.q2:
            return True
    return False


def make_default_rulebook(*, allow_cnot_same_control: bool = True) -> CommutationRulebook:
    """qgym CommutationRulebook with the standard rules + CNOT-same-control.

    Default qgym rules are: gates with disjoint qubits commute, and identical
    gates commute. The disjoint-qubits part is already captured implicitly by
    qubit-availability scheduling, so for the schedule to actually differ from
    plain ASAP we need a richer rule. CNOT-same-control is the textbook
    commutation that real compilers exploit.
    """
    rulebook = CommutationRulebook(default_rules=True)
    if allow_cnot_same_control:
        rulebook.add_rule(cnot_same_control)
    return rulebook


def commutation_aware_makespan(
    circuit: list[Gate],
    gate_durations: dict[str, int],
    rulebook: CommutationRulebook | None = None,
) -> int:
    """Greedy list-schedule the circuit respecting qubit availability + rulebook.

    Builds a blocking matrix from the rulebook (gate i blocks gate j iff i
    precedes j in the input order AND they don't commute). At each iteration,
    finds all ready gates (those whose blocking predecessors are already
    scheduled) and picks the one that can start earliest. This is what makes
    commutation rules matter -- with a richer rulebook, more gates are
    "ready" simultaneously and can be reordered to start earlier.

    With qgym's default rulebook + cnot_same_control, this can return a
    smaller makespan than plain ASAP when commuting reorderings unblock a
    qubit earlier.
    """
    if not circuit:
        return 0
    if rulebook is None:
        rulebook = make_default_rulebook()

    n_qubits = max(max(g.q1, g.q2) for g in circuit) + 1
    qubit_available = np.zeros(n_qubits, dtype=np.int_)
    SENTINEL = -1
    gate_finish = np.full(len(circuit), SENTINEL, dtype=np.int_)
    blocking = rulebook.make_blocking_matrix(circuit)

    while np.any(gate_finish == SENTINEL):
        best_j = -1
        best_start = -1
        for j in range(len(circuit)):
            if gate_finish[j] != SENTINEL:
                continue
            # Ready iff all blocking predecessors already scheduled.
            blockers = np.nonzero(blocking[:j, j])[0]
            if blockers.size and np.any(gate_finish[blockers] == SENTINEL):
                continue
            gate = circuit[j]
            start = int(max(qubit_available[gate.q1], qubit_available[gate.q2]))
            if blockers.size:
                start = max(start, int(gate_finish[blockers].max()))
            if best_j == -1 or start < best_start:
                best_j = j
                best_start = start

        gate = circuit[best_j]
        duration = gate_durations[gate.name]
        gate_finish[best_j] = best_start + duration
        qubit_available[gate.q1] = best_start + duration
        qubit_available[gate.q2] = best_start + duration

    return int(qubit_available.max())


class SchedulingAwareRewarder(SwapQualityRewarder):
    """Routing rewarder with scheduling-aware shaping (Condition B family).

    Inherits the full :class:`SwapQualityRewarder` reward shape (illegal-action
    penalty, surpass reward, swap penalty, observation-enhancement bonus) so the
    only difference vs Condition A is the added scheduling signal.

    Three knobs control the scheduling signal:

    * ``delta_critical_path_weight`` -- per-SWAP penalty on the change in
      makespan of the routed-so-far prefix.
    * ``final_makespan_weight`` -- episode-end penalty on total makespan.
    * ``progress_gate`` -- if True, the per-step delta-CP penalty is applied
      *only* to useless SWAPs (those with non-positive obs-enhancement). This
      avoids the failure mode where useful SWAPs are made more expensive than
      useless ones.

    Variants used in experiments:

    * V1 (failed): ``delta_critical_path_weight=1.0, progress_gate=False``.
      Penalizes useful SWAPs that extend the critical path *more* than useless
      cheap SWAPs that run in parallel -- agent learns to dump useless free
      SWAPs.
    * V2 (episode-end only): ``delta_critical_path_weight=0.0,
      final_makespan_weight=0.3``. No per-step gaming surface; one terminal
      signal. Higher variance but unbiased.
    * V3 (progress-gated): ``delta_critical_path_weight=1.0,
      progress_gate=True, final_makespan_weight=0.3``. Per-step penalty only
      hits useless SWAPs, making them strictly more expensive than useful ones.
    """

    def __init__(
        self,
        gate_durations: dict[str, int],
        illegal_action_penalty: float = -50.0,
        penalty_per_swap: float = -10.0,
        reward_per_surpass: float = 10.0,
        good_swap_reward: float = 5.0,
        delta_critical_path_weight: float = 1.0,
        final_makespan_weight: float = 0.0,
        progress_gate: bool = False,
        optimize_terminal: bool = False,
        optimize_per_step: bool = False,
        realistic_hardware=None,
    ) -> None:
        """Initialize the scheduling-aware rewarder.

        Args:
            gate_durations: Per-gate-type durations used for ASAP makespan.
            illegal_action_penalty: Forwarded to SwapQualityRewarder.
            penalty_per_swap: Forwarded to SwapQualityRewarder.
            reward_per_surpass: Forwarded to SwapQualityRewarder.
            good_swap_reward: Forwarded to SwapQualityRewarder (observation bonus).
            delta_critical_path_weight: Per-SWAP penalty multiplier on
                makespan(prefix_after) - makespan(prefix_before). 0 disables.
            final_makespan_weight: Episode-end penalty multiplier on total
                makespan. 0 disables.
            progress_gate: If True, apply per-step delta-CP penalty only when
                the SWAP did NOT increase the number of executable gates ahead
                (i.e., useless SWAPs). Required to avoid making useful SWAPs
                more expensive than useless ones.
            optimize_terminal: If True, the episode-end makespan term uses
                ``asap_makespan(optimize_circuit(routed))`` instead of the
                un-optimized makespan. This is the signal that decouples
                from SWAP count -- without it the terminal reward is
                redundant with the SWAP penalty. Slow (qiskit transpile per
                episode end) but only fires at termination.
            optimize_per_step: If True, the per-step delta-CP penalty is
                computed on the *optimized* prefix at each SWAP, so the
                signal reflects the post-cancellation cost of the SWAP.
                ~0.5 ms per call -- effectively free.
            realistic_hardware: If not None, a ``RealisticHardware``
                instance from realistic_machine.py. The terminal makespan
                term (and per-step delta-CP if used) will use
                ``realistic_makespan`` instead of ``asap_makespan`` --
                so the agent is trained against the actual hardware-
                constrained scheduling model rather than idealized ASAP.
        """
        super().__init__(
            illegal_action_penalty=illegal_action_penalty,
            penalty_per_swap=penalty_per_swap,
            reward_per_surpass=reward_per_surpass,
            good_swap_reward=good_swap_reward,
        )
        self._gate_durations = dict(gate_durations)
        self._delta_cp_weight = float(delta_critical_path_weight)
        self._final_makespan_weight = float(final_makespan_weight)
        self._progress_gate = bool(progress_gate)
        self._optimize_terminal = bool(optimize_terminal)
        self._optimize_per_step = bool(optimize_per_step)
        self._realistic_hardware = realistic_hardware

    def compute_reward(
        self,
        *,
        old_state: RoutingState,
        action: int,
        new_state: RoutingState,
    ) -> float:
        """SwapQualityRewarder reward + scheduling-aware shaping."""
        reward = super().compute_reward(
            old_state=old_state, action=action, new_state=new_state
        )

        if self._is_illegal(action, old_state):
            return reward

        # Per-step delta on a SWAP action only.
        if action != old_state.n_connections and self._delta_cp_weight != 0.0:
            apply_penalty = True
            if self._progress_gate:
                # Only penalize useless SWAPs (no progress in legal-surpass count).
                enhancement = self._observation_enhancement_factor(old_state, new_state)
                apply_penalty = enhancement <= 0

            if apply_penalty:
                old_prefix = routing_to_circuit_prefix(old_state)
                new_prefix = routing_to_circuit_prefix(new_state)
                if self._optimize_per_step:
                    from optimize import optimize_circuit
                    old_prefix = optimize_circuit(old_prefix, new_state.n_qubits)
                    new_prefix = optimize_circuit(new_prefix, new_state.n_qubits)
                delta = asap_makespan(
                    new_prefix, self._gate_durations
                ) - asap_makespan(old_prefix, self._gate_durations)
                reward -= self._delta_cp_weight * delta

        if self._final_makespan_weight != 0.0 and new_state.is_done():
            full = routing_to_circuit(new_state)
            if self._optimize_terminal:
                # Lazy import: avoids qiskit dependency for non-optimizing variants.
                from optimize import optimize_circuit
                full = optimize_circuit(full, new_state.n_qubits)
            if self._realistic_hardware is not None:
                from realistic_scheduler import realistic_makespan
                makespan = realistic_makespan(full, self._realistic_hardware)
            else:
                makespan = asap_makespan(full, self._gate_durations)
            reward -= self._final_makespan_weight * makespan

        return reward

"""Faithful SABRE reimplementation + makespan-aware extension.

Reimplements qiskit's SabreSwap with the 'lookahead' heuristic and an
attempt-limit, with optional multi-trial selection.

The score function follows qiskit's docstring:

    H_basic(layout, S)    = sum_{(a,b) in S} D[layout(a)][layout(b)]
    H_lookahead(layout)   = (1/|F|) H_basic(layout, F)
                           + 0.5 * (1/|E|) H_basic(layout, E)
    H = H_lookahead

where F = front layer (gates whose preds are all done, themselves not done),
      E = extended set: next gates reachable through F (capped at 20).

The makespan-aware variant adds a term:

    H' = H + lambda * depth_increase(swap)

where depth_increase = how many extra cycles this SWAP would add to the
critical path of the partial routed circuit. Computed by tracking
finish_times[q] = earliest free cycle for physical qubit q in the routed-so-far
prefix.

The "swap" candidate set is exactly the SWAPs adjacent to at least one
qubit in the front layer (qiskit's choice -- restricts search space).

Author: reimplemented for thesis to allow ablation/extension.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import networkx as nx
import numpy as np

from qgym.custom_types import Gate


CNOT_DUR = 2
SWAP_DUR = 6
EXTENDED_SET_WEIGHT = 0.5
EXTENDED_SET_SIZE = 20


def _routed_makespan(routed) -> int:
    """ASAP makespan estimate of a routed gate list (CNOT=2, SWAP=6) on
    physical qubits. Used as the inner-trials selection criterion."""
    if routed is None or len(routed) == 0:
        return 0
    n = 1 + max(max(g.q1, g.q2) for g in routed)
    finish = [0] * n
    for g in routed:
        dur = 2 if g.name == "cnot" else 6
        start = max(finish[g.q1], finish[g.q2])
        finish[g.q1] = start + dur
        finish[g.q2] = start + dur
    return max(finish)


class RoutedList(list):
    """A list of routed Gates that also carries a parallel list mapping
    each entry to its original input-index (or -1 for inserted SWAPs).
    Exists so callers that iterate the list see no change, but consumers
    that need to align routed CNOTs back to the input 2q-gate order can
    read `routed.cnot_input_idx`.
    """
    cnot_input_idx: list  # parallel to self; -1 means SWAP

    def __init__(self, gates=None, cnot_input_idx=None):
        super().__init__(gates or [])
        self.cnot_input_idx = list(cnot_input_idx or [])


@dataclass
class _RouterState:
    """Mutable state during routing."""
    layout: np.ndarray              # layout[logical] = physical (initial: identity)
    inv_layout: np.ndarray          # inv_layout[physical] = logical
    finish_times: np.ndarray        # finish_times[physical] = earliest free cycle
    output: list                    # routed gate list (CNOTs + SWAPs)
    done: np.ndarray                # done[i] = True if circuit[i] applied
    preds: list                     # preds[i] = list of circuit indices that must precede i
    cnot_input_idx: list = None     # parallel to output: input-index for each CNOT, -1 for SWAPs

    def __post_init__(self):
        if self.cnot_input_idx is None:
            self.cnot_input_idx = []

    def apply_swap(self, p1: int, p2: int) -> None:
        """Apply a physical swap on positions p1 and p2."""
        l1 = int(self.inv_layout[p1])
        l2 = int(self.inv_layout[p2])
        self.layout[l1] = p2
        self.layout[l2] = p1
        self.inv_layout[p1] = l2
        self.inv_layout[p2] = l1
        # Schedule the swap
        start = max(int(self.finish_times[p1]), int(self.finish_times[p2]))
        self.finish_times[p1] = start + SWAP_DUR
        self.finish_times[p2] = start + SWAP_DUR
        self.output.append(Gate("swap", p1, p2))
        self.cnot_input_idx.append(-1)

    def execute_cnot(self, p1: int, p2: int, input_idx: int = -1) -> None:
        """Execute a CNOT on adjacent physicals p1, p2."""
        start = max(int(self.finish_times[p1]), int(self.finish_times[p2]))
        self.finish_times[p1] = start + CNOT_DUR
        self.finish_times[p2] = start + CNOT_DUR
        self.output.append(Gate("cnot", p1, p2))
        self.cnot_input_idx.append(input_idx)


def _build_preds(circuit: np.ndarray) -> tuple[list, list]:
    """For each gate i, find which earlier gates share a qubit with it.

    Returns (preds, succs). preds[i] is the set of indices that must complete
    before gate i can be in the front layer.
    """
    n = len(circuit)
    preds: list[set] = [set() for _ in range(n)]
    succs: list[set] = [set() for _ in range(n)]
    last_use: dict[int, int] = {}  # logical_qubit -> last gate index
    for i in range(n):
        lq1, lq2 = int(circuit[i, 0]), int(circuit[i, 1])
        if lq1 in last_use:
            preds[i].add(last_use[lq1])
            succs[last_use[lq1]].add(i)
        if lq2 in last_use and last_use[lq2] != last_use.get(lq1, None):
            preds[i].add(last_use[lq2])
            succs[last_use[lq2]].add(i)
        last_use[lq1] = i
        last_use[lq2] = i
    return preds, succs


def _front_layer(state: _RouterState, succs: list) -> list[int]:
    """Indices of gates whose preds are all done and themselves not done."""
    front = []
    for i, p in enumerate(state.preds):
        if not state.done[i] and all(state.done[k] for k in p):
            front.append(i)
    return front


def _extended_set(front: list[int], succs: list, done: np.ndarray, cap: int = EXTENDED_SET_SIZE) -> list[int]:
    """BFS one step beyond front; collect up to `cap` gates not in front and not done."""
    seen = set(front)
    out: list[int] = []
    for f in front:
        for s in succs[f]:
            if s in seen or done[s]:
                continue
            seen.add(s)
            out.append(s)
            if len(out) >= cap:
                return out
    return out


def _h_lookahead(state, front_gates_qubits, ext_gates_qubits, dist):
    """H_lookahead: (1/|F|)*sum_F + 0.5*(1/|E|)*sum_E."""
    layout = state.layout
    if not front_gates_qubits:
        return 0.0
    f_sum = 0.0
    for a, b in front_gates_qubits:
        f_sum += float(dist[int(layout[a]), int(layout[b])])
    h = f_sum / len(front_gates_qubits)
    if ext_gates_qubits:
        e_sum = 0.0
        for a, b in ext_gates_qubits:
            e_sum += float(dist[int(layout[a]), int(layout[b])])
        h += EXTENDED_SET_WEIGHT * e_sum / len(ext_gates_qubits)
    return h


def _candidate_swaps(front_qubits, edges, layout):
    """Physical-edge SWAPs adjacent to at least one front-layer qubit."""
    front_physicals = set()
    for a, b in front_qubits:
        front_physicals.add(int(layout[a]))
        front_physicals.add(int(layout[b]))
    out = []
    for u, v in edges:
        if int(u) in front_physicals or int(v) in front_physicals:
            out.append((int(u), int(v)))
    return out


def _depth_increase(state, p1, p2):
    """How many cycles this SWAP would add to the longest qubit timeline."""
    start = max(int(state.finish_times[p1]), int(state.finish_times[p2]))
    new_finish = start + SWAP_DUR
    cur_max = int(state.finish_times.max())
    return max(0, new_finish - cur_max)


def _start_cycle(state, p1, p2):
    """When can this SWAP actually start? Lower = more parallelizable."""
    return max(int(state.finish_times[p1]), int(state.finish_times[p2]))


def _swap_load(swap_counts, p1, p2):
    """How loaded are these two qubits with SWAPs already?"""
    return int(swap_counts[p1]) + int(swap_counts[p2])


def _enables_front_cnot(state, interaction_circuit, front, p1, p2, coupling):
    """Will applying SWAP(p1,p2) cause any front-layer gate to become executable
    AND share a qubit with the SWAP? If so, the optimizer can absorb part of the
    SWAP cost. Returns the count of such gates (0, 1, or 2)."""
    # Simulate the swap on inv_layout
    saved_l1 = int(state.inv_layout[p1])
    saved_l2 = int(state.inv_layout[p2])
    # After swap: physical p1 holds logical saved_l2, p2 holds saved_l1
    count = 0
    for i in front:
        lq1, lq2 = int(interaction_circuit[i, 0]), int(interaction_circuit[i, 1])
        # Where do those logicals sit AFTER the swap?
        np1 = p2 if lq1 == saved_l1 else (p1 if lq1 == saved_l2 else int(state.layout[lq1]))
        np2 = p2 if lq2 == saved_l1 else (p1 if lq2 == saved_l2 else int(state.layout[lq2]))
        if (np1, np2) in coupling or (np2, np1) in coupling:
            # And does the gate touch p1 or p2?
            if np1 == p1 or np1 == p2 or np2 == p1 or np2 == p2:
                count += 1
    return count


def _delta_cp_v2(state, p1, p2, fusion_credit):
    """Augmented delta-critical-path:
    cost = max(0, swap_finish - cur_critical_path) - fusion_credit * absorbed
    where absorbed is bounded by SWAP_DUR. fusion_credit in [0, 1]."""
    start = max(int(state.finish_times[p1]), int(state.finish_times[p2]))
    new_finish = start + SWAP_DUR
    cur_max = int(state.finish_times.max())
    delta = max(0, new_finish - cur_max)
    return float(delta) - fusion_credit * float(SWAP_DUR)


def _lookahead_finish(state, interaction_circuit, front, p1, p2, dist):
    """Earliest cycle at which the next front-layer gate involving qubits p1 or p2
    can complete after this SWAP is applied. If no front-layer gate involves
    p1/p2, returns the SWAP finish cycle.

    Captures: a SWAP is bad not just because it idles, but because it has to
    sit on a busy qubit and *then* a CNOT has to run on top of it.
    """
    start = max(int(state.finish_times[p1]), int(state.finish_times[p2]))
    swap_finish = start + SWAP_DUR
    saved_l1 = int(state.inv_layout[p1])
    saved_l2 = int(state.inv_layout[p2])
    # After swap: physical p1 holds logical saved_l2, p2 holds saved_l1
    best_completion = swap_finish
    for i in front:
        lq1 = int(interaction_circuit[i, 0])
        lq2 = int(interaction_circuit[i, 1])
        np1 = p2 if lq1 == saved_l1 else (p1 if lq1 == saved_l2 else int(state.layout[lq1]))
        np2 = p2 if lq2 == saved_l1 else (p1 if lq2 == saved_l2 else int(state.layout[lq2]))
        # Only count if this gate touches the SWAPped qubits (otherwise unrelated)
        if np1 in (p1, p2) or np2 in (p1, p2):
            # Earliest it can start: after both endpoints free; for SWAPped qubits that's swap_finish.
            # For the other endpoint of a 2q gate that's just one finish-time entry.
            end_np1 = swap_finish if np1 in (p1, p2) else int(state.finish_times[np1])
            end_np2 = swap_finish if np2 in (p1, p2) else int(state.finish_times[np2])
            cnot_start = max(end_np1, end_np2)
            cnot_finish = cnot_start + CNOT_DUR
            if cnot_finish > best_completion:
                best_completion = cnot_finish
    return float(best_completion)


def _two_term_score(state, p1, p2, alpha, beta):
    """alpha * start_cycle + beta * delta_critical_path."""
    start = max(int(state.finish_times[p1]), int(state.finish_times[p2]))
    new_finish = start + SWAP_DUR
    cur_max = int(state.finish_times.max())
    delta = max(0, new_finish - cur_max)
    return alpha * float(start) + beta * float(delta)


def _fusion_aware_score(state, interaction_circuit, front, p1, p2, coupling):
    """Score = start_cycle * (1 - absorb_factor)
    where absorb_factor is 1.0 if a front-layer CNOT exactly on (p1,p2) becomes
    executable (full SWAP absorption), 0.5 if a CNOT on one of p1/p2 with a
    different partner does (partial), and 0.0 otherwise.

    Encodes: a SWAP whose entire 3-CNOT cost gets eaten by a fusion is free.
    """
    start = max(int(state.finish_times[p1]), int(state.finish_times[p2]))
    saved_l1 = int(state.inv_layout[p1])
    saved_l2 = int(state.inv_layout[p2])
    absorb = 0.0
    for i in front:
        lq1 = int(interaction_circuit[i, 0])
        lq2 = int(interaction_circuit[i, 1])
        np1 = p2 if lq1 == saved_l1 else (p1 if lq1 == saved_l2 else int(state.layout[lq1]))
        np2 = p2 if lq2 == saved_l1 else (p1 if lq2 == saved_l2 else int(state.layout[lq2]))
        if not ((np1, np2) in coupling or (np2, np1) in coupling):
            continue
        # Adjacency-becoming-executable.
        # Full absorption: gate is exactly on (p1, p2)
        if {np1, np2} == {p1, p2}:
            absorb = max(absorb, 1.0)
        elif np1 in (p1, p2) or np2 in (p1, p2):
            absorb = max(absorb, 0.5)
    return float(start) * (1.0 - absorb)


def sabre_route(
    interaction_circuit: np.ndarray,
    connection_graph: nx.Graph,
    initial_mapping: np.ndarray | None = None,
    lookahead: bool = True,
    makespan_lambda: float = 0.0,
    makespan_mode: str = "depth_increase",  # "depth_increase" or "start_cycle" or "combined"
    swap_load_weight: float = 0.0,
    attempt_limit: int | None = None,
    seed: int = 0,
    trials: int = 1,
    # Two-term mode parameters
    two_term_alpha: float = 1.0,
    two_term_beta: float = 1.0,
) -> list[Gate] | None:
    """SABRE routing with optional makespan-awareness.

    If trials > 1, run `trials` independent attempts with derived seeds
    and return the one with smallest final makespan
    (max of finish_times). Matches Qiskit SabreSwap(trials=N) semantics
    but using makespan instead of (depth, gate_count) as the tiebreaker.

    Returns the routed gate list, or None if routing failed (hit attempt_limit
    without progress, meaning the algorithm got stuck).
    """
    # Multi-trial loop: when trials > 1, run `trials` independent attempts
    # with derived seeds and return the best by final makespan. Single
    # call below recurses with trials=1.
    if trials > 1:
        rng_outer = np.random.default_rng(seed)
        best_routed = None
        best_mks = float("inf")
        for _ in range(trials):
            trial_seed = int(rng_outer.integers(0, 2**31 - 1))
            r = sabre_route(
                interaction_circuit, connection_graph,
                initial_mapping=initial_mapping,
                lookahead=lookahead, makespan_lambda=makespan_lambda,
                makespan_mode=makespan_mode, swap_load_weight=swap_load_weight,
                attempt_limit=attempt_limit, seed=trial_seed, trials=1,
                two_term_alpha=two_term_alpha, two_term_beta=two_term_beta,
            )
            if r is None:
                continue
            mks = _routed_makespan(r)
            if mks < best_mks:
                best_mks = mks
                best_routed = r
        return best_routed

    n_qubits = connection_graph.number_of_nodes()
    edges = list(connection_graph.edges())

    # Distance matrix (shortest paths on undirected graph)
    d = dict(nx.all_pairs_shortest_path_length(connection_graph))
    dist = np.zeros((n_qubits, n_qubits), dtype=np.float32)
    for u in range(n_qubits):
        for v in range(n_qubits):
            dist[u, v] = float(d[u].get(v, n_qubits + 1))

    # Coupling set for legality check
    coupling = set()
    for u, v in edges:
        coupling.add((int(u), int(v)))
        coupling.add((int(v), int(u)))

    if initial_mapping is None:
        initial_mapping = np.arange(n_qubits, dtype=int)
    layout = np.array(initial_mapping, dtype=int).copy()
    inv_layout = np.zeros(n_qubits, dtype=int)
    for lq, pq in enumerate(layout):
        inv_layout[int(pq)] = lq

    preds, succs = _build_preds(interaction_circuit)
    state = _RouterState(
        layout=layout,
        inv_layout=inv_layout,
        finish_times=np.zeros(n_qubits, dtype=int),
        output=[],
        done=np.zeros(len(interaction_circuit), dtype=bool),
        preds=preds,
    )

    if attempt_limit is None:
        attempt_limit = 10 * n_qubits

    rng = np.random.default_rng(seed)
    no_progress_count = 0
    swap_counts = np.zeros(n_qubits, dtype=int)

    while True:
        # 1. Drain executable gates from the front layer
        progressed = True
        while progressed:
            progressed = False
            front = _front_layer(state, succs)
            for i in front:
                lq1, lq2 = int(interaction_circuit[i, 0]), int(interaction_circuit[i, 1])
                p1, p2 = int(state.layout[lq1]), int(state.layout[lq2])
                if (p1, p2) in coupling or (p2, p1) in coupling:
                    state.execute_cnot(p1, p2, input_idx=int(i))
                    state.done[i] = True
                    progressed = True
        # Check done
        front = _front_layer(state, succs)
        if not front:
            break  # all done

        # 2. SWAP selection
        front_qubits = [(int(interaction_circuit[i, 0]), int(interaction_circuit[i, 1]))
                         for i in front]
        ext = _extended_set(front, succs, state.done)
        ext_qubits = [(int(interaction_circuit[i, 0]), int(interaction_circuit[i, 1])) for i in ext]
        candidates = _candidate_swaps(front_qubits, edges, state.layout)
        if not candidates:
            return None  # no SWAPs available; bug

        # Score each candidate
        scored = []
        for u, v in candidates:
            # Simulate applying this swap to the layout (not finish_times)
            new_layout = state.layout.copy()
            lu = int(state.inv_layout[u])
            lv = int(state.inv_layout[v])
            new_layout[lu] = v
            new_layout[lv] = u
            # H_lookahead with new_layout
            saved_layout = state.layout
            state.layout = new_layout
            h = _h_lookahead(state, front_qubits, ext_qubits if lookahead else [], dist)
            state.layout = saved_layout
            # Makespan-aware extension
            if makespan_lambda != 0.0:
                if makespan_mode == "depth_increase":
                    m = float(_depth_increase(state, u, v))
                elif makespan_mode == "start_cycle":
                    m = float(_start_cycle(state, u, v))
                elif makespan_mode == "combined":
                    m = float(_start_cycle(state, u, v)) + 2.0 * float(_depth_increase(state, u, v))
                elif makespan_mode == "delta_cp":
                    m = float(_depth_increase(state, u, v))
                elif makespan_mode == "delta_cp_fusion":
                    absorbed = _enables_front_cnot(state, interaction_circuit, front, u, v, coupling)
                    m = _delta_cp_v2(state, u, v, fusion_credit=0.5 * min(absorbed, 1))
                elif makespan_mode == "lookahead_finish":
                    # Project the completion cycle of the next front-layer gate
                    # that this SWAP enables on the SWAPped qubits. Subtract
                    # current critical path so the term is "extension" not absolute.
                    cur_max = int(state.finish_times.max())
                    m = _lookahead_finish(state, interaction_circuit, front, u, v, dist) - float(cur_max)
                    if m < 0:
                        m = 0.0
                elif makespan_mode == "two_term":
                    # alpha * start_cycle + beta * delta_critical_path
                    m = _two_term_score(state, u, v, two_term_alpha, two_term_beta)
                elif makespan_mode == "fusion_aware":
                    # start_cycle discounted by absorb_factor (1.0 if SWAP enables
                    # CNOT on same pair → fully absorbed).
                    m = _fusion_aware_score(state, interaction_circuit, front, u, v, coupling)
                elif makespan_mode == "esp_aware":
                    # Old esp_aware (fusion bonus on top of start_cycle) — kept for
                    # backward compatibility but the new ESP-aware mode is below.
                    start = float(_start_cycle(state, u, v))
                    saved_l1 = int(state.inv_layout[u])
                    saved_l2 = int(state.inv_layout[v])
                    fusion_credit = 0.0
                    for i in front:
                        lq1 = int(interaction_circuit[i, 0])
                        lq2 = int(interaction_circuit[i, 1])
                        np1 = v if lq1 == saved_l1 else (u if lq1 == saved_l2 else int(state.layout[lq1]))
                        np2 = v if lq2 == saved_l1 else (u if lq2 == saved_l2 else int(state.layout[lq2]))
                        if not ((np1, np2) in coupling or (np2, np1) in coupling):
                            continue
                        if {np1, np2} == {u, v}:
                            fusion_credit = max(fusion_credit, 1.0)
                        elif np1 in (u, v) or np2 in (u, v):
                            fusion_credit = max(fusion_credit, 0.5)
                    m = two_term_alpha * start - two_term_beta * fusion_credit
                elif makespan_mode == "esp_direct":
                    # SABRE-ESP: combine CNOT-count cost (via expected un-absorbed
                    # CNOTs from the SWAP) AND decoherence cost (via max(finish)).
                    #   m = alpha * unabsorbed_cnots + beta * max(finish)
                    # where alpha encodes the CNOT-error penalty and beta the
                    # decoherence-cost-per-cycle.
                    # Both multiplied by makespan_lambda outside.
                    start = float(_start_cycle(state, u, v))
                    saved_l1 = int(state.inv_layout[u])
                    saved_l2 = int(state.inv_layout[v])
                    # Expected un-absorbed CNOTs from this SWAP:
                    #   - If SWAP enables a same-pair front CNOT: full absorption,
                    #     2 of 3 SWAP-CNOTs cancel. Cost = 1 CNOT.
                    #   - If SWAP enables a partner-edge CNOT touching p1/p2:
                    #     partial absorption. Cost = 2 CNOTs.
                    #   - Otherwise: full SWAP cost = 3 CNOTs.
                    unabsorbed = 3.0  # default: no absorption
                    for i in front:
                        lq1 = int(interaction_circuit[i, 0])
                        lq2 = int(interaction_circuit[i, 1])
                        np1 = v if lq1 == saved_l1 else (u if lq1 == saved_l2 else int(state.layout[lq1]))
                        np2 = v if lq2 == saved_l1 else (u if lq2 == saved_l2 else int(state.layout[lq2]))
                        if not ((np1, np2) in coupling or (np2, np1) in coupling):
                            continue
                        if {np1, np2} == {u, v}:
                            unabsorbed = min(unabsorbed, 1.0)
                        elif np1 in (u, v) or np2 in (u, v):
                            unabsorbed = min(unabsorbed, 2.0)
                    # The ESP-aware score combines gate-error term + decoherence term.
                    # two_term_alpha = CNOT-count weight (gate error proxy)
                    # two_term_beta  = max(finish) weight (decoherence proxy)
                    m = two_term_alpha * unabsorbed + two_term_beta * start
                else:
                    m = 0.0
                h += makespan_lambda * m
            if swap_load_weight != 0.0:
                h += swap_load_weight * float(_swap_load(swap_counts, u, v))
            scored.append((h, u, v))
        # Break ties randomly using the rng
        min_h = min(s[0] for s in scored)
        candidates_best = [(u, v) for h, u, v in scored if h <= min_h + 1e-9]
        u, v = candidates_best[int(rng.integers(0, len(candidates_best)))]
        prev_done = int(state.done.sum())
        state.apply_swap(u, v)
        swap_counts[u] += 1
        swap_counts[v] += 1
        new_done_after_drain = prev_done  # will be recomputed next iter

        # Progress tracking
        if int(state.done.sum()) == prev_done:
            no_progress_count += 1
            if no_progress_count > attempt_limit:
                return None
        else:
            no_progress_count = 0

    return RoutedList(state.output, state.cnot_input_idx)


def sabre_route_rollout(
    interaction_circuit: np.ndarray,
    connection_graph: nx.Graph,
    initial_mapping: np.ndarray | None = None,
    rollout_horizon: int = 5,  # how many gates to lookahead
    seed: int = 0,
) -> list[Gate] | None:
    """SABRE variant where each SWAP candidate is scored by running a short
    rollout of vanilla SABRE on the next few gates and measuring the resulting
    makespan."""
    n_qubits = connection_graph.number_of_nodes()
    edges = list(connection_graph.edges())
    d = dict(nx.all_pairs_shortest_path_length(connection_graph))
    dist = np.zeros((n_qubits, n_qubits), dtype=np.float32)
    for u in range(n_qubits):
        for v in range(n_qubits):
            dist[u, v] = float(d[u].get(v, n_qubits + 1))
    coupling = set()
    for u, v in edges:
        coupling.add((int(u), int(v))); coupling.add((int(v), int(u)))

    if initial_mapping is None:
        initial_mapping = np.arange(n_qubits, dtype=int)
    layout = np.array(initial_mapping, dtype=int).copy()
    inv_layout = np.zeros(n_qubits, dtype=int)
    for lq, pq in enumerate(layout):
        inv_layout[int(pq)] = lq

    preds, succs = _build_preds(interaction_circuit)
    state = _RouterState(
        layout=layout, inv_layout=inv_layout,
        finish_times=np.zeros(n_qubits, dtype=int),
        output=[], done=np.zeros(len(interaction_circuit), dtype=bool),
        preds=preds,
    )

    rng = np.random.default_rng(seed)
    attempt_limit = 10 * n_qubits
    no_progress = 0

    while True:
        # Drain executable
        progressed = True
        while progressed:
            progressed = False
            front = _front_layer(state, succs)
            for i in front:
                lq1, lq2 = int(interaction_circuit[i, 0]), int(interaction_circuit[i, 1])
                p1, p2 = int(state.layout[lq1]), int(state.layout[lq2])
                if (p1, p2) in coupling or (p2, p1) in coupling:
                    state.execute_cnot(p1, p2, input_idx=int(i)); state.done[i] = True; progressed = True
        front = _front_layer(state, succs)
        if not front:
            break

        front_qubits = [(int(interaction_circuit[i, 0]), int(interaction_circuit[i, 1])) for i in front]
        candidates = _candidate_swaps(front_qubits, edges, state.layout)
        if not candidates:
            return None

        # Score each candidate by rollout
        scored = []
        for u, v in candidates:
            # Snapshot state, simulate the swap, run a short rollout, get makespan
            saved_layout = state.layout.copy()
            saved_inv = state.inv_layout.copy()
            saved_finish = state.finish_times.copy()
            saved_output_len = len(state.output)
            saved_done = state.done.copy()

            state.apply_swap(u, v)

            # Short rollout: greedy SABRE basic, up to rollout_horizon gates executed
            rollout_done = 0
            r_attempts = 0
            r_attempt_limit = 5 * n_qubits
            while rollout_done < rollout_horizon and r_attempts < r_attempt_limit:
                progressed = True
                while progressed:
                    progressed = False
                    rfront = _front_layer(state, succs)
                    for i in rfront:
                        lq1, lq2 = int(interaction_circuit[i, 0]), int(interaction_circuit[i, 1])
                        p1, p2 = int(state.layout[lq1]), int(state.layout[lq2])
                        if (p1, p2) in coupling or (p2, p1) in coupling:
                            state.execute_cnot(p1, p2, input_idx=int(i))
                            state.done[i] = True
                            progressed = True
                            rollout_done += 1
                            if rollout_done >= rollout_horizon:
                                break
                    if rollout_done >= rollout_horizon:
                        break
                if rollout_done >= rollout_horizon:
                    break
                rfront = _front_layer(state, succs)
                if not rfront:
                    break
                r_front_qubits = [(int(interaction_circuit[i, 0]), int(interaction_circuit[i, 1])) for i in rfront]
                r_cands = _candidate_swaps(r_front_qubits, edges, state.layout)
                if not r_cands:
                    break
                # Score basic SABRE
                best_h = float("inf"); best_sw = r_cands[0]
                for ru, rv in r_cands:
                    new_lay = state.layout.copy()
                    lu = int(state.inv_layout[ru]); lv = int(state.inv_layout[rv])
                    new_lay[lu] = rv; new_lay[lv] = ru
                    s_ = state.layout
                    state.layout = new_lay
                    h = _h_lookahead(state, r_front_qubits, [], dist)
                    state.layout = s_
                    if h < best_h:
                        best_h = h; best_sw = (ru, rv)
                state.apply_swap(*best_sw)
                r_attempts += 1
            mks_rollout = int(state.finish_times.max())

            # Restore
            state.layout = saved_layout
            state.inv_layout = saved_inv
            state.finish_times = saved_finish
            state.output = state.output[:saved_output_len]
            state.done = saved_done

            scored.append((mks_rollout, u, v))

        min_h = min(s[0] for s in scored)
        best = [(u, v) for h, u, v in scored if h == min_h]
        u, v = best[int(rng.integers(0, len(best)))]
        prev_done = int(state.done.sum())
        state.apply_swap(u, v)
        if int(state.done.sum()) == prev_done:
            no_progress += 1
            if no_progress > attempt_limit:
                return None
        else:
            no_progress = 0
    return RoutedList(state.output, state.cnot_input_idx)


def sabre_route_best_of(
    interaction_circuit: np.ndarray,
    connection_graph: nx.Graph,
    n_trials: int = 5,
    initial_mapping: np.ndarray | None = None,
    lookahead: bool = True,
    makespan_lambda: float = 0.0,
    makespan_mode: str = "depth_increase",
    swap_load_weight: float = 0.0,
    scoring=lambda gates: sum(1 for g in gates if g.name == "swap"),
) -> list[Gate]:
    """Run N trials of SABRE with different seeds, return best by scoring fn.

    `scoring` is applied to the gate list to pick best. Default: minimize SWAP count.
    """
    best = None
    best_score = float("inf")
    for trial in range(n_trials):
        out = sabre_route(
            interaction_circuit, connection_graph,
            initial_mapping=initial_mapping,
            lookahead=lookahead,
            makespan_lambda=makespan_lambda,
            makespan_mode=makespan_mode,
            swap_load_weight=swap_load_weight,
            seed=trial,
        )
        if out is None:
            continue
        s = scoring(out)
        if s < best_score:
            best = out
            best_score = s
    return best if best is not None else []

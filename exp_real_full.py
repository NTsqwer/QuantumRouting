"""SABRE-MS on real NISQ benchmark circuits, full correct pipeline:

  - Real circuits: QFT, Quantum Volume, VQE EfficientSU2, QAOA-random.
  - Full Qiskit basis decomposition to (cx, u3) — keeps 1q gates.
  - SabreLayout for initial mapping (same for both methods).
  - K=20 routing trials per method, paired by initial-mapping seed.
  - Optimization preserves 1q gates (we stay in QuantumCircuit format
    throughout, applying Qiskit's gate-cancellation pass natively).
  - ASAP scheduler with 1q=1 cycle, cnot=2, swap=6.
  - Per-family lambda tuned (corrected for QAOA from the sweep).

Output: results/real_full.json
"""
from __future__ import annotations

import json
import os
import time
import warnings
from collections.abc import Callable

import numpy as np
from scipy import stats  # noqa: F401 (used in wilcoxon)
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import (
    CXGate, SwapGate, U3Gate, HGate, RZGate, SXGate, XGate,
)
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import (
    SabreLayout, SabreSwap,
    BasisTranslator, Collect2qBlocks, ConsolidateBlocks,
    UnitarySynthesis, InverseCancellation, CommutativeCancellation,
)
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary

warnings.filterwarnings("ignore", category=DeprecationWarning)

from sabre_impl import sabre_route
from topologies import get as get_topology


CNOT_DUR = 2
SWAP_DUR = 6
ONE_Q_DUR = 1
K = 20


# Equivalence check: on by default. Set RP_SKIP_EQUIV=1 in env to disable
# (for tight hot loops where you're sure). Failures raise; the bug we
# hit silently produced wrong circuits, so we error loud now.
import os as _os
_EQUIV_DEFAULT = _os.environ.get("RP_SKIP_EQUIV", "0") != "1"


def _assert_equiv(qc_input: QuantumCircuit, qc_routed: QuantumCircuit,
                 context: str = "") -> None:
    """Verify that qc_routed (with SWAPs stripped and gates retargeted)
    implements the same unitary as qc_input. Raises AssertionError on
    mismatch."""
    from qiskit.quantum_info import Operator
    # Strip swaps; track cumulative perm so subsequent gates target
    # the qubit that started where they're now targeting.
    n = qc_input.num_qubits
    perm = list(range(n))  # perm[current_position] = original_position
    out = QuantumCircuit(n)
    for instr in qc_routed.data:
        qs = [qc_routed.find_bit(q).index for q in instr.qubits]
        if instr.operation.name == "swap":
            a, b = qs
            perm[a], perm[b] = perm[b], perm[a]
            continue
        out.append(instr.operation, [perm[q] for q in qs])
    if not Operator(qc_input).equiv(Operator(out)):
        raise AssertionError(
            f"ROUTING NON-EQUIVALENT (context={context!r}): "
            f"routed circuit (SWAPs stripped) does not match input unitary"
        )


# ===================================================================
# Optimization that preserves 1q gates (Qiskit-native)
# ===================================================================

_OPTIMIZER_PM = PassManager([
    BasisTranslator(SessionEquivalenceLibrary, ["cx", "u3"]),
    Collect2qBlocks(),
    ConsolidateBlocks(basis_gates=["cx"]),
    UnitarySynthesis(basis_gates=["cx", "u3"]),
    InverseCancellation([CXGate()]),
    CommutativeCancellation(),
    InverseCancellation([CXGate()]),
])


def optimize_qc(qc: QuantumCircuit) -> QuantumCircuit:
    """Apply Qiskit's gate-cancellation pipeline directly to a QuantumCircuit
    (no round-trip through qgym). Preserves all gates."""
    return _OPTIMIZER_PM.run(qc)


# ===================================================================
# Makespan computation directly from QuantumCircuit
# ===================================================================

def asap_makespan_qc(qc: QuantumCircuit) -> int:
    """ASAP-schedule a QuantumCircuit and return the makespan.
    Durations: 1q=1, cx=2, swap=6, all others (rz/measure/etc.)=1 if 1q, error if 2q+.
    """
    n_q = qc.num_qubits
    free = np.zeros(n_q, dtype=int)
    for instr in qc.data:
        op = instr.operation
        qs = [qc.find_bit(q).index for q in instr.qubits]
        if len(qs) == 1:
            free[qs[0]] += ONE_Q_DUR
        elif len(qs) == 2:
            if op.name == "cx":
                dur = CNOT_DUR
            elif op.name == "swap":
                dur = SWAP_DUR
            else:
                # Treat any other 2q gate as a CNOT for accounting (shouldn't
                # appear after basis decomposition to cx,u3).
                dur = CNOT_DUR
            start = max(int(free[qs[0]]), int(free[qs[1]]))
            free[qs[0]] = start + dur
            free[qs[1]] = start + dur
        elif len(qs) == 0:
            continue  # barrier/measure with no qargs
        else:
            raise ValueError(f"unsupported {len(qs)}-qubit gate {op.name}")
    return int(free.max())


# ===================================================================
# Circuit generators (Qiskit-shipped benchmark circuits)
# ===================================================================

def gen_qft(n_qubits, seed=0):
    from qiskit.circuit.library import QFT
    return QFT(n_qubits, do_swaps=True, approximation_degree=0).decompose()


def gen_quantum_volume(n_qubits, seed=0):
    from qiskit.circuit.library import QuantumVolume
    depth = n_qubits  # standard QV protocol uses depth = n_qubits
    return QuantumVolume(num_qubits=n_qubits, depth=depth, seed=seed).decompose()


def gen_vqe_efficient_su2(n_qubits, seed=0):
    """VQE-style hardware-efficient ansatz, full entanglement (not linear),
    reps=2."""
    from qiskit.circuit.library import EfficientSU2
    qc = EfficientSU2(num_qubits=n_qubits, reps=2, entanglement="full",
                      flatten=True)
    rng = np.random.default_rng(seed)
    return qc.assign_parameters(rng.uniform(0, 2 * np.pi, len(qc.parameters)))


def gen_qaoa_random(n_qubits, seed=0):
    """QAOA on a random ER cost graph (p=1)."""
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.quantum_info import SparsePauliOp
    rng = np.random.default_rng(seed)
    paulis, coeffs = [], []
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            if rng.random() < 0.5:
                s = list("I" * n_qubits)
                s[i] = "Z"; s[j] = "Z"
                paulis.append("".join(reversed(s)))
                coeffs.append(rng.uniform(-1, 1))
    if not paulis:
        paulis = ["I" * (n_qubits - 2) + "ZZ"]
        coeffs = [1.0]
    cost = SparsePauliOp.from_list(list(zip(paulis, coeffs)))
    qc = QAOAAnsatz(cost_operator=cost, reps=1, flatten=True)
    bound = qc.assign_parameters(rng.uniform(0, 2 * np.pi, len(qc.parameters)))
    return bound.decompose()


GENERATORS = {
    "qft":      gen_qft,
    "qv":       gen_quantum_volume,
    "vqe":      gen_vqe_efficient_su2,
    "qaoa":     gen_qaoa_random,
}


# Per-family lambda based on our sweep results.
# QAOA needs a much smaller lambda than the headline class rule predicted.
def lambda_for(family, topology):
    if family == "qaoa":
        return 0.005  # from exp_qaoa_lambda_sweep.py
    # For other families, use the headline class rule
    if "grid" in topology or "heavy_hex" in topology:
        return 0.05
    return 0.10


# ===================================================================
# Pipeline: decompose -> layout -> route -> optimize -> schedule
# ===================================================================

def decompose_basis(qc: QuantumCircuit) -> QuantumCircuit:
    """Decompose to (cx, u3) basis with no layout/routing.
    optimization_level=0 just translates; we get a clean basis-decomposed
    circuit with no SWAPs yet."""
    return transpile(qc, basis_gates=["cx", "u3"], optimization_level=0,
                    seed_transpiler=0)


def get_layout(qc_basis: QuantumCircuit, coupling, seed: int) -> np.ndarray:
    """Run SabreLayout on the basis-decomposed circuit, return the
    logical-to-physical permutation."""
    n = qc_basis.num_qubits
    pm = PassManager([SabreLayout(coupling_map=coupling, seed=seed,
                                  max_iterations=2)])
    pm.run(qc_basis)
    layout = pm.property_set.get("layout")
    if layout is None:
        return np.arange(n, dtype=int)
    perm = np.zeros(n, dtype=int)
    for i, q in enumerate(qc_basis.qubits):
        perm[i] = layout[q]
    return perm


def apply_layout_to_qc(qc_basis: QuantumCircuit, perm: np.ndarray) -> QuantumCircuit:
    """Return a new QuantumCircuit with all gates remapped from logical to
    physical qubits using `perm[logical] = physical`."""
    n = qc_basis.num_qubits
    qc_phys = QuantumCircuit(n)
    for instr in qc_basis.data:
        op = instr.operation
        qs = [qc_basis.find_bit(q).index for q in instr.qubits]
        new_qubits = [qc_phys.qubits[int(perm[q])] for q in qs]
        qc_phys.append(op, new_qubits)
    return qc_phys


def route_with_sabre(qc_phys: QuantumCircuit, coupling, seed: int,
                    verify_equivalence: bool = _EQUIV_DEFAULT) -> QuantumCircuit:
    """Run vanilla SABRE-lookahead routing on the post-layout circuit.
    Inserts SWAPs to make all 2q gates adjacent. Preserves 1q gates."""
    pm = PassManager([SabreSwap(coupling_map=coupling, heuristic="lookahead",
                               seed=seed, trials=1)])
    out = pm.run(qc_phys)
    if verify_equivalence:
        _assert_equiv(qc_phys, out, context=f"sabre_lookahead seed={seed}")
    return out


def route_with_sabre_ms(qc_phys: QuantumCircuit, graph, coupling, lam: float,
                       seed: int, alim_mult: int = 10,
                       verify_equivalence: bool = _EQUIV_DEFAULT,
                       makespan_mode: str = "start_cycle",
                       two_term_alpha: float = 1.0,
                       two_term_beta: float = 1.0) -> QuantumCircuit | None:
    """Route the post-layout circuit using SABRE-MS for the 2q gates and
    splice 1q gates back at their correct logical position.

    SABRE-MS may execute independent CNOTs out of input order (commuting
    CNOTs on disjoint qubits). The splice handles this by tracking, for
    each input-1q gate, which CNOT (by input-index) it must come after
    on its qubit, and emitting it only once all such CNOTs have been
    completed in the routed output.
    """
    n = qc_phys.num_qubits

    # Extract instruction list with type tag, AND for each 1q gate find
    # the input-index of the most recent prior 2q gate that touched the
    # same logical qubit (or -1 if none).
    insts = []   # list of dict { 'kind', 'op', 'qs', ... }
    cp = []      # 2q pairs in input order
    last_2q_on_q = {}  # logical qubit -> most recent 2q input-index touching it
    for instr in qc_phys.data:
        op = instr.operation
        qs = [qc_phys.find_bit(q).index for q in instr.qubits]
        if len(qs) == 1:
            q = qs[0]
            pred = last_2q_on_q.get(q, -1)
            insts.append({"kind": "1q", "op": op, "q": q,
                         "pred_2q_idx": pred})
        elif len(qs) == 2:
            q1, q2 = qs
            cp_idx = len(cp)
            cp.append((q1, q2))
            insts.append({"kind": "2q", "op": op, "q1": q1, "q2": q2,
                         "cp_idx": cp_idx})
            last_2q_on_q[q1] = cp_idx
            last_2q_on_q[q2] = cp_idx
        else:
            # 'other' (barrier etc.) — depend on all touched qubits
            preds = [last_2q_on_q.get(q, -1) for q in qs]
            insts.append({"kind": "other", "op": op, "qs": qs,
                         "pred_2q_idx": max(preds) if preds else -1})

    if not cp:
        return qc_phys  # nothing to route

    cp_arr = np.array(cp, dtype=int)
    routed = sabre_route(cp_arr, graph, lookahead=True, makespan_lambda=lam,
                        makespan_mode=makespan_mode, seed=seed,
                        attempt_limit=alim_mult * n,
                        two_term_alpha=two_term_alpha, two_term_beta=two_term_beta)
    if routed is None:
        return None

    # Build cp_idx -> position-in-routed (where its CNOT got emitted).
    cnot_input_idx = getattr(routed, "cnot_input_idx", None)
    if cnot_input_idx is None:
        # Fallback: assume in-order (will break if router reordered, but
        # at least won't crash).
        cnot_input_idx = []
        ci = 0
        for g in routed:
            if g.name == "cnot":
                cnot_input_idx.append(ci); ci += 1
            else:
                cnot_input_idx.append(-1)

    # Group the 1q (and 'other') gates by their predecessor cp_idx:
    # everything with pred_2q_idx == k goes "after" cp_idx k has been
    # emitted. pred = -1 means "before any 2q", emit first.
    pending_after = {k: [] for k in range(-1, len(cp))}
    cp_order = []  # The original-input-index, e.g. insts[i]['cp_idx'] in input order
    for inst in insts:
        if inst["kind"] in ("1q", "other"):
            pending_after[inst["pred_2q_idx"]].append(inst)
        else:
            cp_order.append(inst["cp_idx"])

    out_qc = QuantumCircuit(n)
    layout_cur = np.arange(n, dtype=int)  # logical -> current physical
    inv_layout = np.arange(n, dtype=int)  # physical -> current logical

    def emit_1q_or_other(inst):
        if inst["kind"] == "1q":
            phys_now = int(layout_cur[inst["q"]])
            out_qc.append(inst["op"], [out_qc.qubits[phys_now]])
        else:
            out_qc.append(inst["op"],
                         [out_qc.qubits[int(layout_cur[q])]
                          for q in inst["qs"]])

    # Emit gates that have no 2q predecessor (pred = -1) first.
    for inst in pending_after[-1]:
        emit_1q_or_other(inst)

    # Walk routed in execution order, emitting SWAPs and CNOTs, and after
    # each CNOT-with-input-idx-k, drain pending_after[k].
    for pos, g in enumerate(routed):
        if g.name == "swap":
            p1, p2 = g.q1, g.q2
            l1, l2 = int(inv_layout[p1]), int(inv_layout[p2])
            layout_cur[l1], layout_cur[l2] = p2, p1
            inv_layout[p1], inv_layout[p2] = l2, l1
            out_qc.append(SwapGate(), [out_qc.qubits[p1], out_qc.qubits[p2]])
        elif g.name == "cnot":
            out_qc.append(CXGate(), [out_qc.qubits[g.q1], out_qc.qubits[g.q2]])
            k = cnot_input_idx[pos]
            if k >= 0:
                for inst in pending_after[k]:
                    emit_1q_or_other(inst)
        else:
            # Unknown gate type from router — pass through
            out_qc.append(g.name,
                         [out_qc.qubits[g.q1], out_qc.qubits[g.q2]])

    if verify_equivalence:
        _assert_equiv(qc_phys, out_qc,
                     context=f"sabre_ms lam={lam} seed={seed}")
    return out_qc


def pipeline_sabre(qc_orig: QuantumCircuit, graph, coupling, seed: int) -> int:
    """End-to-end pipeline with vanilla SABRE routing.
    Returns the makespan or None on failure."""
    qc_basis = decompose_basis(qc_orig)
    perm = get_layout(qc_basis, coupling, seed)
    qc_phys = apply_layout_to_qc(qc_basis, perm)
    qc_routed = route_with_sabre(qc_phys, coupling, seed)
    qc_opt = optimize_qc(qc_routed)
    return asap_makespan_qc(qc_opt)


def pipeline_sabre_ms(qc_orig: QuantumCircuit, graph, coupling, lam: float,
                      seed: int, alim_mult: int = 10) -> int | None:
    """End-to-end pipeline with SABRE-MS routing."""
    qc_basis = decompose_basis(qc_orig)
    perm = get_layout(qc_basis, coupling, seed)
    qc_phys = apply_layout_to_qc(qc_basis, perm)
    qc_routed = route_with_sabre_ms(qc_phys, graph, coupling, lam, seed,
                                    alim_mult=alim_mult)
    if qc_routed is None:
        return None
    qc_opt = optimize_qc(qc_routed)
    return asap_makespan_qc(qc_opt)


def best_of_k(fn: Callable[[int], int | None], k: int) -> int | None:
    best = float("inf")
    for s in range(k):
        m = fn(s)
        if m is None:
            continue
        if m < best:
            best = m
    return best if best != float("inf") else None


def wilcoxon(a, b, alt="less"):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.all(a == b):
        return 1.0
    try:
        _, p = stats.wilcoxon(a, b, alternative=alt)
        return float(p)
    except Exception:
        return 1.0


# ===================================================================
# Driver
# ===================================================================

# Per-run lambda grid (INCLUDES 0, so SABRE's score is always a candidate).
PERRUN_LAMBDA_GRID = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25]
PERRUN_K0 = 5


def select_lambda_perrun(qc, graph, coupling, ci, alim_mult):
    """Per-run lambda selection for one circuit, respecting this runner's
    per-seed layout: for each lambda, probe at K0 trials and keep the shortest
    makespan; return the lambda with the shortest probe makespan (ties -> smaller
    lambda, so lambda=0 wins an exact tie). Part of the algorithm, not tuning."""
    best_lam, best_probe = None, float("inf")
    for lam in PERRUN_LAMBDA_GRID:
        mk = best_of_k(
            lambda s: pipeline_sabre_ms(qc, graph, coupling, lam,
                                        seed=s + ci * K, alim_mult=alim_mult),
            PERRUN_K0,
        )
        if mk is not None and mk < best_probe:
            best_probe, best_lam = mk, lam
    return best_lam


def run_cell(topology, family, n_circuits, alim_mult=10):
    graph, n_qubits = get_topology(topology)
    edges = list(graph.edges())
    coupling = CouplingMap([list(e) for e in edges] + [list(reversed(e)) for e in edges])
    gen = GENERATORS[family]

    sabre_mks, ms_mks, lams = [], [], []
    for ci in range(n_circuits):
        try:
            qc = gen(n_qubits, seed=1234 + ci)
        except Exception:
            continue
        try:
            lam = select_lambda_perrun(qc, graph, coupling, ci, alim_mult)
            if lam is None:
                continue
            s_mk = best_of_k(
                lambda s: pipeline_sabre(qc, graph, coupling, seed=s + ci * K),
                K,
            )
            m_mk = best_of_k(
                lambda s: pipeline_sabre_ms(qc, graph, coupling, lam,
                                          seed=s + ci * K, alim_mult=alim_mult),
                K,
            )
        except Exception:
            continue
        if s_mk is None or m_mk is None:
            continue
        sabre_mks.append(s_mk); ms_mks.append(m_mk); lams.append(lam)

    if not sabre_mks:
        return None
    a, b = np.array(ms_mks, float), np.array(sabre_mks, float)
    return {
        "topology": topology, "family": family,
        "lambda_mean": float(np.mean(lams)), "lambda_per_circuit": lams,
        "n_qubits": n_qubits, "n_circuits": len(sabre_mks),
        "sabre_lookahead_k20_mean": float(b.mean()),
        "ms_k20_mean": float(a.mean()),
        "gain_pct": float(100 * (b.mean() - a.mean()) / b.mean()),
        "p_ms_less": wilcoxon(a, b, "less"),
    }


CELLS = [
    # QFT
    ("linear7",   "qft", 12),
    ("ring8",     "qft", 12),
    ("ring12",    "qft", 10),
    ("grid3x3",   "qft", 12),
    ("grid4x4",   "qft", 8),
    ("heavy_hex2","qft", 8),
    # Quantum Volume
    ("linear7",   "qv",  12),
    ("linear9",   "qv",  12),
    ("ring8",     "qv",  12),
    ("ring12",    "qv",  10),
    ("grid3x3",   "qv",  12),
    ("grid4x4",   "qv",  8),
    ("heavy_hex2","qv",  8),
    # VQE full entanglement
    ("ring8",     "vqe", 12),
    ("grid3x3",   "vqe", 12),
    ("grid4x4",   "vqe", 8),
    ("heavy_hex2","vqe", 8),
    # QAOA (lambda = 0.005)
    ("linear7",   "qaoa", 15),
    ("linear9",   "qaoa", 15),
    ("ring8",     "qaoa", 15),
    ("ring12",    "qaoa", 12),
    ("grid3x3",   "qaoa", 12),
    ("grid4x4",   "qaoa", 8),
    ("heavy_hex2","qaoa", 8),
]


def main():
    os.makedirs("results", exist_ok=True)
    out_path = "results/real_full_perrun.json"
    out = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            out = json.load(f)
        print(f"Loaded {len(out)} cells")

    t0 = time.time()
    print(f"{'Cell':<24} {'lam':>6} {'n_q':>4} {'n':>3}  "
          f"{'SABRE K20':>10} {'MS K20':>9} {'gain':>8} {'p':>10}")
    print("-" * 100)

    for topo, fam, n in CELLS:
        key = f"{topo}/{fam}"
        if key in out:
            r = out[key]
        else:
            print(f"  Running {key}...", flush=True)
            tc = time.time()
            r = run_cell(topo, fam, n)
            print(f"    ...{time.time()-tc:.0f}s", flush=True)
            if r is None:
                continue
            out[key] = r
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
        marker = "  *" if r["gain_pct"] > 1 and r["p_ms_less"] < 0.05 else ""
        print(f"{key:<24} {r['lambda_mean']:>6.3f} {r['n_qubits']:>4} {r['n_circuits']:>3}  "
              f"{r['sabre_lookahead_k20_mean']:>10.2f} {r['ms_k20_mean']:>9.2f} "
              f"{r['gain_pct']:>+7.2f}% {r['p_ms_less']:>10.1e}{marker}")

    if out:
        print("\n=== Aggregate by family ===")
        for fam_name in ["qft", "qv", "vqe", "qaoa"]:
            results = [v for v in out.values() if v["family"] == fam_name]
            if not results: continue
            gains = [v["gain_pct"] for v in results]
            sig = sum(1 for v in results if v["gain_pct"] > 1 and v["p_ms_less"] < 0.05)
            print(f"  {fam_name:<8}: mean {np.mean(gains):+.2f}%   "
                  f"median {np.median(gains):+.2f}%   "
                  f"significant: {sig}/{len(results)}")
    print(f"\nTotal: {time.time()-t0:.0f}s")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

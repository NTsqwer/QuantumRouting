"""SABRE-MS vs SABRE family on UNOPTIMIZED makespan (no qiskit optimizer pass).

The goal: does SABRE-MS still beat SABRE if we skip the gate-cancellation step?
If yes -> the makespan term picks better-scheduling SWAPs even without optimizer
If no  -> SABRE-MS's win specifically exploits gate-cancellation opportunities

Pipeline (mirrors normal SABRE-MS evaluation EXCEPT no optimizer):
  1. SabreLayout per-circuit
  2. Route with method under test
  3. ASAP makespan on the raw routed circuit (cnot=2, swap=6 cycles)
                                ^^^^^^^^^^^^^^^^^^^ key change

Topologies: small (5q linear/ring/tshape), medium (linear7/9, ring8/12, grid3x3/4x4,
heavy_hex2), and large (grid5x5, linear15). Mix of families. K=1 and K=20.

Output: results/sabrems_unoptimized.json
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
from scipy import stats
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import SabreLayout, SabreSwap

from circuits import make_circuits
from pipeline import asap_makespan
from sabre_impl import sabre_route
from topologies import get as get_topology


GD = {"cnot": 2, "swap": 6}
K = 20


def sabre_layout_perm(circuit, coupling, n_qubits, seed=0):
    if circuit.size == 0:
        return np.arange(n_qubits, dtype=int)
    qc = QuantumCircuit(n_qubits)
    for q1, q2 in circuit:
        qc.cx(int(q1), int(q2))
    pm = PassManager([SabreLayout(coupling_map=coupling, seed=seed, max_iterations=2)])
    pm.run(qc)
    layout = pm.property_set.get("layout")
    if layout is None:
        return np.arange(n_qubits, dtype=int)
    perm = np.zeros(n_qubits, dtype=int)
    for i, q in enumerate(qc.qubits):
        perm[i] = layout[q]
    return perm


def qiskit_route_gates(circuit_perm, coupling, n_qubits, heuristic, seed=0):
    """One single-seed Qiskit SABRE routing. Returns qgym Gate list, NO optimization."""
    from qgym.custom_types import Gate
    qc = QuantumCircuit(n_qubits)
    for q1, q2 in circuit_perm:
        qc.cx(int(q1), int(q2))
    pm = PassManager([SabreSwap(coupling_map=coupling, heuristic=heuristic, seed=seed, trials=1)])
    routed = pm.run(qc)
    out = []
    for instr in routed.data:
        op = instr.operation
        qs = [routed.find_bit(q).index for q in instr.qubits]
        if len(qs) == 2:
            name = "cnot" if op.name == "cx" else op.name
            out.append(Gate(name, int(qs[0]), int(qs[1])))
    return out


def qiskit_mk_one(circuit_perm, coupling, n_qubits, heuristic, seed):
    """RAW (unoptimized) makespan."""
    gates = qiskit_route_gates(circuit_perm, coupling, n_qubits, heuristic, seed)
    return asap_makespan(gates, GD)  # NO optimize_circuit call


def qiskit_mk_k(circuit_perm, coupling, n_qubits, heuristic, k=K):
    best = float("inf")
    for s in range(k):
        m = qiskit_mk_one(circuit_perm, coupling, n_qubits, heuristic, s)
        if m < best:
            best = m
    return best


def ms_mk_one(circuit_perm, graph, n_qubits, lam, seed, attempt_limit_mult=10):
    gates = sabre_route(circuit_perm, graph, lookahead=True, makespan_lambda=lam,
                        makespan_mode="start_cycle", seed=seed,
                        attempt_limit=attempt_limit_mult * n_qubits)
    if gates is None:
        return None
    return asap_makespan(gates, GD)  # NO optimize_circuit call


def ms_mk_k(circuit_perm, graph, n_qubits, lam, k=K, attempt_limit_mult=10):
    best = float("inf")
    n_success = 0
    for s in range(k):
        m = ms_mk_one(circuit_perm, graph, n_qubits, lam, s, attempt_limit_mult)
        if m is None:
            continue
        n_success += 1
        if m < best:
            best = m
    return best if n_success > 0 else None


def wilcoxon_paired(ms_arr, baseline_arr):
    a = np.asarray(ms_arr, dtype=float)
    b = np.asarray(baseline_arr, dtype=float)
    if np.all(a == b):
        return 1.0
    try:
        _, p = stats.wilcoxon(a, b, alternative="less")
        return float(p)
    except Exception:
        return 1.0


def run_cell(topology, family, lam, n_circuits=60, length=30, seed=1234,
             alim_mult=10):
    graph, n_qubits = get_topology(topology)
    edges = list(graph.edges())
    coupling = CouplingMap([list(e) for e in edges] + [list(reversed(e)) for e in edges])
    cs = make_circuits(family, n_qubits, n_circuits, seed=seed, length=length)

    mks = {"basic_k1": [], "lookahead_k1": [], "decay_k1": [], "ms_k1": [],
           "basic_k20": [], "lookahead_k20": [], "decay_k20": [], "ms_k20": []}

    for ci, c in enumerate(cs):
        perm = sabre_layout_perm(c, coupling, n_qubits, seed=ci)
        cp = perm[c].astype(int)

        # K=1 — single seed per method
        for h in ("basic", "lookahead", "decay"):
            mks[f"{h}_k1"].append(qiskit_mk_one(cp, coupling, n_qubits, h, seed=ci))
        ms_k1_val = ms_mk_one(cp, graph, n_qubits, lam, seed=ci, attempt_limit_mult=alim_mult)
        if ms_k1_val is None:
            # Conservative fallback: best of K=1 SABRE values
            ms_k1_val = min(mks["basic_k1"][-1], mks["lookahead_k1"][-1], mks["decay_k1"][-1])
        mks["ms_k1"].append(ms_k1_val)

        # K=20 — best of 20 seeds
        for h in ("basic", "lookahead", "decay"):
            mks[f"{h}_k20"].append(qiskit_mk_k(cp, coupling, n_qubits, h, k=K))
        ms_k20_val = ms_mk_k(cp, graph, n_qubits, lam, k=K, attempt_limit_mult=alim_mult)
        if ms_k20_val is None:
            ms_k20_val = mks["lookahead_k20"][-1]
        mks["ms_k20"].append(ms_k20_val)

    def gain_pct(better, worse):
        return 100.0 * (float(np.mean(worse)) - float(np.mean(better))) / float(np.mean(worse))

    return {
        "topology": topology, "family": family, "lambda": lam,
        "n_qubits": n_qubits, "n_circuits": n_circuits,
        "attempt_limit_mult": alim_mult,
        "raw_means": {k: float(np.mean(v)) for k, v in mks.items()},
        # K=1 SABRE-MS vs each baseline
        "k1_vs_basic":     {"gain_pct": gain_pct(mks["ms_k1"], mks["basic_k1"]),
                            "p": wilcoxon_paired(mks["ms_k1"], mks["basic_k1"])},
        "k1_vs_lookahead": {"gain_pct": gain_pct(mks["ms_k1"], mks["lookahead_k1"]),
                            "p": wilcoxon_paired(mks["ms_k1"], mks["lookahead_k1"])},
        "k1_vs_decay":     {"gain_pct": gain_pct(mks["ms_k1"], mks["decay_k1"]),
                            "p": wilcoxon_paired(mks["ms_k1"], mks["decay_k1"])},
        # K=20 SABRE-MS vs each baseline
        "k20_vs_basic":     {"gain_pct": gain_pct(mks["ms_k20"], mks["basic_k20"]),
                             "p": wilcoxon_paired(mks["ms_k20"], mks["basic_k20"])},
        "k20_vs_lookahead": {"gain_pct": gain_pct(mks["ms_k20"], mks["lookahead_k20"]),
                             "p": wilcoxon_paired(mks["ms_k20"], mks["lookahead_k20"])},
        "k20_vs_decay":     {"gain_pct": gain_pct(mks["ms_k20"], mks["decay_k20"]),
                             "p": wilcoxon_paired(mks["ms_k20"], mks["decay_k20"])},
    }


# Cells to test (topology, family, oracle_lambda, n_circuits, length, attempt_limit_mult).
# Oracle lambdas from oracle_lambda.json and grid5x5_corrected.json.
CELLS = [
    # Small 5q
    ("linear5",  "random", 0.10, 60, 30, 10),
    ("linear5",  "qft",    0.25, 60, 30, 10),
    ("linear5",  "parallel", 0.02, 60, 30, 10),
    ("ring5",    "qft",    0.25, 60, 30, 10),
    ("tshape5",  "qft",    0.25, 60, 30, 10),

    # Medium (7-9 qubits)
    ("linear7",  "qft",    0.25, 60, 30, 10),
    ("linear9",  "qft",    0.25, 60, 30, 10),
    ("linear9",  "random", 0.10, 60, 30, 10),
    ("ring8",    "qft",    0.25, 60, 30, 10),
    ("ring12",   "qft",    0.25, 60, 30, 10),
    ("ring12",   "random", 0.10, 60, 30, 10),
    ("ring12",   "parallel", 0.02, 60, 30, 10),

    # Grids
    ("grid3x3",  "qft",    0.05, 60, 30, 10),
    ("grid4x4",  "qft",    0.05, 60, 30, 10),
    ("grid4x4",  "random", 0.05, 60, 30, 10),
    ("grid4x4",  "parallel", 0.05, 60, 30, 10),

    # Heavy-hex IBM
    ("heavy_hex2", "qft",    0.25, 60, 30, 10),
    ("heavy_hex2", "parallel", 0.05, 60, 30, 10),

    # Large
    ("linear15", "qft",    0.25, 40, 30, 10),
    ("grid5x5",  "qft",    0.10, 30, 30, 50),  # needs alim 50 like before
    ("grid5x5",  "random", 0.10, 30, 30, 10),
    ("grid5x5",  "parallel", 0.05, 30, 30, 10),
]


def main():
    os.makedirs("results", exist_ok=True)
    out_path = "results/sabrems_unoptimized.json"
    out = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            out = json.load(f)
        print(f"Loaded {len(out)} existing entries")

    t0 = time.time()
    print(f"{'Cell':<22} {'lam':>5} {'n_q':>4}  {'K=1 vs basic':>13} {'K=1 vs look':>13} {'K=1 vs decay':>13}   {'K=20 vs basic':>14} {'K=20 vs look':>13} {'K=20 vs decay':>14}")
    print("-" * 145)

    for topo, fam, lam, n, length, alim in CELLS:
        key = f"{topo}/{fam}"
        if key in out:
            r = out[key]
        else:
            print(f"  Running {key}...", flush=True)
            r = run_cell(topo, fam, lam, n_circuits=n, length=length, alim_mult=alim)
            out[key] = r
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)

        print(f"{key:<22} {r['lambda']:>5.2f} {r['n_qubits']:>4}  "
              f"{r['k1_vs_basic']['gain_pct']:>+7.1f}% p={r['k1_vs_basic']['p']:>5.0e}  "
              f"{r['k1_vs_lookahead']['gain_pct']:>+7.1f}% p={r['k1_vs_lookahead']['p']:>5.0e}  "
              f"{r['k1_vs_decay']['gain_pct']:>+7.1f}% p={r['k1_vs_decay']['p']:>5.0e}    "
              f"{r['k20_vs_basic']['gain_pct']:>+7.1f}% p={r['k20_vs_basic']['p']:>5.0e}  "
              f"{r['k20_vs_lookahead']['gain_pct']:>+7.1f}% p={r['k20_vs_lookahead']['p']:>5.0e}  "
              f"{r['k20_vs_decay']['gain_pct']:>+7.1f}% p={r['k20_vs_decay']['p']:>5.0e}")

    print(f"\nTotal: {time.time()-t0:.0f}s")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

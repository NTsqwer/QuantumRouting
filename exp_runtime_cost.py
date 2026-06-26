"""Runtime cost analysis: how much slower is SABRE-MS (K=20) vs SABRE (K=1)?

For thesis: if SABRE-MS takes 10x longer, it's only useful in offline settings.
If it takes 2x longer, it's practical.

This measures wall-clock routing time (excluding layout and optimization, which
are identical for both methods) for representative cells.

Output: results/runtime_cost.json
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import SabreLayout, SabreSwap

from circuits import make_circuits
from sabre_impl import sabre_route
from topologies import get as get_topology

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


def qiskit_route_gates(circuit_np, coupling, n_qubits, seed=0):
    from qgym.custom_types import Gate
    qc = QuantumCircuit(n_qubits)
    for q1, q2 in circuit_np:
        qc.cx(int(q1), int(q2))
    pm = PassManager([SabreSwap(coupling_map=coupling, heuristic="lookahead",
                                seed=seed, trials=1)])
    routed = pm.run(qc)
    out = []
    for instr in routed.data:
        op = instr.operation
        qs = [routed.find_bit(q).index for q in instr.qubits]
        if len(qs) == 2:
            name = "cnot" if op.name == "cx" else op.name
            out.append(Gate(name, int(qs[0]), int(qs[1])))
    return out


def measure_routing_time(topology, family, lam, n_circuits=20, length=30, seed=1234):
    graph, n_qubits = get_topology(topology)
    edges = list(graph.edges())
    coupling = CouplingMap([list(e) for e in edges] + [list(reversed(e)) for e in edges])
    cs = make_circuits(family, n_qubits, n_circuits, seed=seed, length=length)

    # Precompute layouts
    layouts = []
    for ci, c in enumerate(cs):
        perm = sabre_layout_perm(c, coupling, n_qubits, seed=ci)
        layouts.append(perm[c].astype(int))

    # Time SABRE K=1
    t0 = time.perf_counter()
    for ci, cp in enumerate(layouts):
        qiskit_route_gates(cp, coupling, n_qubits, seed=0)
    sabre_k1_time = time.perf_counter() - t0

    # Time SABRE K=20
    t0 = time.perf_counter()
    for ci, cp in enumerate(layouts):
        for k in range(K):
            qiskit_route_gates(cp, coupling, n_qubits, seed=k)
    sabre_k20_time = time.perf_counter() - t0

    # Time SABRE-MS K=1
    t0 = time.perf_counter()
    for ci, cp in enumerate(layouts):
        sabre_route(cp, graph, lookahead=True, makespan_lambda=lam,
                    makespan_mode="start_cycle", seed=0)
    ms_k1_time = time.perf_counter() - t0

    # Time SABRE-MS K=20
    t0 = time.perf_counter()
    for ci, cp in enumerate(layouts):
        for k in range(K):
            sabre_route(cp, graph, lookahead=True, makespan_lambda=lam,
                        makespan_mode="start_cycle", seed=k)
    ms_k20_time = time.perf_counter() - t0

    return {
        "topology": topology, "family": family, "lambda": lam,
        "n_circuits": n_circuits, "n_qubits": n_qubits,
        "circuit_length": int(len(cs[0])),
        "sabre_k1_ms": sabre_k1_time * 1000 / n_circuits,   # ms per circuit
        "sabre_k20_ms": sabre_k20_time * 1000 / n_circuits,
        "ms_k1_ms": ms_k1_time * 1000 / n_circuits,
        "ms_k20_ms": ms_k20_time * 1000 / n_circuits,
        "k1_overhead_vs_sabre_k1": ms_k1_time / sabre_k1_time,
        "k20_overhead_vs_sabre_k20": ms_k20_time / sabre_k20_time,
        "k20_overhead_vs_sabre_k1": ms_k20_time / sabre_k1_time,
    }


CELLS = [
    ("linear5",    "qft",      0.25, 20, 30),
    ("ring12",     "qft",      0.25, 20, 30),
    ("ring12",     "parallel", 0.02, 20, 30),
    ("ring12",     "random",   0.10, 20, 30),
    ("grid4x4",    "qft",      0.05, 20, 30),
    ("grid4x4",    "parallel", 0.05, 20, 30),
    ("grid5x5",    "qft",      0.02, 20, 30),
    ("grid5x5",    "random",   0.10, 20, 30),
    ("heavy_hex2", "qft",      0.25, 20, 30),
    ("heavy_hex2", "random",   0.10, 20, 30),
]


def main():
    os.makedirs("results", exist_ok=True)
    out_path = "results/runtime_cost.json"
    out = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            out = json.load(f)
        print("Loaded {} existing entries".format(len(out)))

    t0 = time.time()
    print("{:<25} {:>4}  {:>8} {:>8} {:>8} {:>8}  {:>8} {:>8}".format(
        "Cell", "nq", "sab_k1", "sab_k20", "ms_k1", "ms_k20",
        "ms/sab k1", "ms/sab k20"))
    print("-" * 90)

    for topo, fam, lam, n, length in CELLS:
        key = "{}/{}".format(topo, fam)
        if key in out:
            r = out[key]
            print("{:<25} {:>4}  {:>8.1f} {:>8.1f} {:>8.1f} {:>8.1f}  {:>8.2f}x {:>8.2f}x".format(
                key, r["n_qubits"],
                r["sabre_k1_ms"], r["sabre_k20_ms"], r["ms_k1_ms"], r["ms_k20_ms"],
                r["k1_overhead_vs_sabre_k1"], r["k20_overhead_vs_sabre_k20"]))
            continue

        print("  Running {}...".format(key), flush=True)
        r = measure_routing_time(topo, fam, lam, n_circuits=n, length=length)
        out[key] = r
        print("{:<25} {:>4}  {:>8.1f} {:>8.1f} {:>8.1f} {:>8.1f}  {:>8.2f}x {:>8.2f}x".format(
            key, r["n_qubits"],
            r["sabre_k1_ms"], r["sabre_k20_ms"], r["ms_k1_ms"], r["ms_k20_ms"],
            r["k1_overhead_vs_sabre_k1"], r["k20_overhead_vs_sabre_k20"]))

        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

    print("\nTotal: {:.0f}s".format(time.time() - t0))
    print("Saved {}".format(out_path))


if __name__ == "__main__":
    main()

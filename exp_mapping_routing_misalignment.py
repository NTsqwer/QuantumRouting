"""Mapping/routing pass-boundary misalignment.

Tests whether SabreLayout's mapping-quality proxy (minimise inserted SWAPs
during its internal forward-backward iteration) systematically disagrees
with which mapping actually produces the shortest final makespan after
the downstream routing + optimisation + scheduling.

This is the mapping-routing analog of the routing-scheduling misalignment
already characterised in exp_real_full.py / paper section 3.

Per cell (topology, family), per circuit:

  1. Generate a pool of K candidate mappings by running SabreLayout K
     times with distinct seeds (each with swap_trials=1 internal). Capture
     the layout AND the swap count SabreLayout would have used to score it.
  2. For each candidate mapping, route the (basis-decomposed, layout-
     applied) circuit with SABRE-lookahead at K_route fixed seeds, take
     best-by-makespan. Compute the resulting makespan.
  3. Two selection rules:
       - Rule A (the proxy): mapping with the fewest SWAPs in step (1)
       - Rule B (the truth): mapping with the lowest makespan in step (2)
  4. Report: disagreement frequency, mean makespan loss from picking A
     when B is better.

Output: results/mapping_routing_misalignment.json
"""
from __future__ import annotations
import json
import os
import time
import warnings
from collections import defaultdict

import numpy as np
warnings.filterwarnings("ignore", category=DeprecationWarning)

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import SabreLayout, SabreSwap

from exp_real_full import (
    GENERATORS, decompose_basis, apply_layout_to_qc,
    optimize_qc, asap_makespan_qc,
)
from topologies import get as get_topology


CELLS = [
    # QFT (already done — first round of evidence for framing C)
    ("linear7", "qft",  8),
    ("ring8",   "qft",  8),
    ("ring12",  "qft",  6),
    ("grid3x3", "qft",  8),
    # QV
    ("linear7", "qv",   8),
    ("ring8",   "qv",   8),
    ("ring12",  "qv",   6),
    ("grid3x3", "qv",   8),
    # VQE (EfficientSU2 full entanglement)
    ("linear7", "vqe",  8),
    ("ring8",   "vqe",  8),
    ("grid3x3", "vqe",  8),
    # QAOA (random Erdős-Rényi)
    ("linear7", "qaoa", 8),
    ("ring8",   "qaoa", 8),
    ("ring12",  "qaoa", 6),
    ("grid3x3", "qaoa", 8),
]

K_LAYOUT = 20   # candidate mappings per circuit
K_ROUTE  = 5    # downstream routing trials per (mapping, circuit)


def get_layout_with_seed(qc_basis, coupling, seed):
    """Run SabreLayout with given seed. Return (perm, swap_count_under_default).

    The swap_count is what SabreLayout uses internally for its
    layout_trials selection rule ("trial that results in the output with
    the fewest swap gates will be selected"). We reproduce that here so
    we can compare against argmin-by-makespan.
    """
    n = qc_basis.num_qubits
    pm = PassManager([SabreLayout(coupling_map=coupling, seed=seed,
                                  max_iterations=2,
                                  swap_trials=1, layout_trials=1)])
    pm.run(qc_basis)
    layout = pm.property_set.get("layout")
    if layout is None:
        return None, None
    perm = np.zeros(n, dtype=int)
    for i, q in enumerate(qc_basis.qubits):
        perm[i] = layout[q]

    # Score the mapping by SABRE-lookahead K=1 swap count (the proxy
    # SabreLayout effectively optimises against during its iteration).
    qc_phys = apply_layout_to_qc(qc_basis, perm)
    pm2 = PassManager([SabreSwap(coupling_map=coupling,
                                 heuristic="lookahead", seed=seed,
                                 trials=1)])
    qc_routed = pm2.run(qc_phys)
    n_swap = sum(1 for instr in qc_routed.data
                 if instr.operation.name == "swap")
    return perm, n_swap


def route_and_makespan(qc_basis, perm, coupling, k_route):
    """Route with SABRE-lookahead at k_route trials, return best
    post-optimisation makespan."""
    qc_phys = apply_layout_to_qc(qc_basis, perm)
    best = float("inf")
    for s in range(k_route):
        pm = PassManager([SabreSwap(coupling_map=coupling,
                                    heuristic="lookahead", seed=s,
                                    trials=1)])
        qc_routed = pm.run(qc_phys)
        m = asap_makespan_qc(optimize_qc(qc_routed))
        if m is not None and m < best:
            best = m
    return best if best != float("inf") else None


def run_cell(topology, family, n_circuits):
    graph, n_q = get_topology(topology)
    edges = list(graph.edges())
    coupling = CouplingMap([list(e) for e in edges]
                          + [list(reversed(e)) for e in edges])
    gen = GENERATORS[family]

    per_circuit = []
    n_disagree = 0
    swap_gain_sum = 0.0
    swap_gain_count = 0
    mks_gain_sum = 0.0
    distinct_mappings_sum = 0

    for ci in range(n_circuits):
        try:
            qc = gen(n_q, seed=1234 + ci)
        except Exception:
            continue
        qc_basis = decompose_basis(qc)

        # Build the candidate-mapping pool
        mappings = []  # each: (perm_tuple, swap_count, makespan)
        for k in range(K_LAYOUT):
            perm, n_swap = get_layout_with_seed(qc_basis, coupling, seed=ci*K_LAYOUT + k)
            if perm is None or n_swap is None:
                continue
            mks = route_and_makespan(qc_basis, perm, coupling, K_ROUTE)
            if mks is None:
                continue
            mappings.append((tuple(int(x) for x in perm), int(n_swap), float(mks)))

        if len(mappings) < 2:
            continue

        # Distinct mappings (the proxy-quality argmin is meaningful only
        # if the pool has spread)
        distinct = len({m[0] for m in mappings})
        distinct_mappings_sum += distinct

        # Rule A: argmin by swap count (the proxy)
        i_a = int(np.argmin([m[1] for m in mappings]))
        mks_a = mappings[i_a][2]

        # Rule B: argmin by makespan (the truth)
        i_b = int(np.argmin([m[2] for m in mappings]))
        mks_b = mappings[i_b][2]

        # Did they pick the same mapping?
        disagree = mappings[i_a][0] != mappings[i_b][0]
        if disagree:
            n_disagree += 1
            swap_gain_sum  += mappings[i_b][1] - mappings[i_a][1]  # extra swaps under rule B
            mks_gain_sum   += mks_a - mks_b                        # makespan saved
            swap_gain_count += 1

        per_circuit.append({
            "ci": ci,
            "n_distinct_mappings": distinct,
            "rule_a_swaps": mappings[i_a][1],
            "rule_a_makespan": mks_a,
            "rule_b_swaps": mappings[i_b][1],
            "rule_b_makespan": mks_b,
            "disagree": disagree,
        })

    n_used = len(per_circuit)
    return {
        "topology": topology, "family": family,
        "n_qubits": n_q, "n_circuits": n_used,
        "K_layout": K_LAYOUT, "K_route": K_ROUTE,
        "frac_disagree": n_disagree / n_used if n_used > 0 else 0.0,
        "mean_distinct_mappings": distinct_mappings_sum / n_used if n_used > 0 else 0.0,
        "mean_makespan_loss_pct": (
            100 * mks_gain_sum / sum(p["rule_a_makespan"] for p in per_circuit)
            if per_circuit and sum(p["rule_a_makespan"] for p in per_circuit) > 0
            else 0.0
        ),
        "mean_extra_swaps_for_makespan_best": (
            swap_gain_sum / swap_gain_count if swap_gain_count > 0 else 0.0
        ),
        "per_circuit": per_circuit,
    }


def main():
    os.makedirs("results", exist_ok=True)
    out_path = "results/mapping_routing_misalignment.json"
    out = {}

    print(f"Mapping/routing pass-boundary misalignment")
    print(f"  K_layout={K_LAYOUT}, K_route={K_ROUTE}")
    print(f"  Selection rules:")
    print(f"    A = argmin by SABRE-lookahead swap count (SabreLayout's proxy)")
    print(f"    B = argmin by post-optimisation makespan (the truth)")
    print()
    print(f"{'Cell':<22} {'n':>3} {'distinct':>9} {'disagree':>9} "
          f"{'mks_loss':>9} {'extra_swaps':>11}")
    print("-" * 70)

    t0 = time.time()
    for topology, family, n in CELLS:
        tc = time.time()
        r = run_cell(topology, family, n)
        out[f"{topology}/{family}"] = r
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"{topology+'/'+family:<22} {r['n_circuits']:>3} "
              f"{r['mean_distinct_mappings']:>9.1f} "
              f"{r['frac_disagree']*100:>8.1f}% "
              f"{r['mean_makespan_loss_pct']:>+8.2f}% "
              f"{r['mean_extra_swaps_for_makespan_best']:>+11.2f}  "
              f"({time.time()-tc:.0f}s)",
              flush=True)

    print(f"\nTotal: {time.time()-t0:.0f}s")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

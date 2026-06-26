"""IBM real-hardware evaluation: SABRE vs SABRE-MS on Quantum Volume.

Submits paired routed circuits to a real IBM backend (default
ibm_brisbane) and measures Hellinger fidelity to the all-zeros bitstring
using the inverse-test methodology. This is the closest possible
substitute for "did our claimed makespan reduction translate to a real
hardware improvement?"

Protocol per (backend, n_qubits, family) cell:
  1. Generate N=4 random QV/QFT circuits.
  2. For each, compose with its own inverse (ideal output is |0...0>).
  3. Route the *original* QV with SABRE and with SABRE-MS (best of K=5).
  4. Compose each routed circuit with the inverse of the original
     unrouted circuit (NOT its own inverse — we need the *circuit's*
     inverse, not the routing's).
  5. We use the self-inverse test on the routed circuit
     (qc_routed.compose(qc_routed.inverse())), because the routing inserts
     SWAPs that do not cancel in the original-inverse test.
  6. Both circuits go to the backend in one job.
  7. Compute Hellinger fidelity to |0...0> from the returned counts.

Authentication: requires QiskitRuntimeService account saved locally
(use QiskitRuntimeService.save_account once with your API token).

Usage:
  # First time only (replace TOKEN):
  python -c "from qiskit_ibm_runtime import QiskitRuntimeService; \\
             QiskitRuntimeService.save_account(channel='ibm_quantum_platform', \\
             token='YOUR_TOKEN', overwrite=True)"

  # Then:
  python exp_ibm_hardware.py --backend ibm_brisbane --n-qubits 8 \\
    --family qv --n-circuits 4 --shots 4096

Output: results/ibm_hardware_<backend>_<family>_<n>q.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings

import numpy as np
import networkx as nx

warnings.filterwarnings("ignore", category=DeprecationWarning)

from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

from exp_real_full import (
    GENERATORS, lambda_for, decompose_basis, get_layout, apply_layout_to_qc,
    route_with_sabre, route_with_sabre_ms, optimize_qc,
)


# ============================================================
# Hellinger fidelity
# ============================================================

def hellinger_fidelity_against_zeros(counts: dict, n_qubits: int) -> float:
    """Hellinger fidelity between counts distribution and a delta at |0...0>."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    target = "0" * n_qubits
    # Hellinger fidelity = (sum sqrt(p*q))^2; q is delta at target so only
    # the target's probability contributes.
    p_target = counts.get(target, 0) / total
    return float(p_target)  # (sqrt(p)*sqrt(1))^2 = p, since q(target)=1, q(other)=0


# ============================================================
# Routing pipelines (return QuantumCircuit, not just makespan)
# ============================================================

def get_backend_subgraph(backend, n_qubits: int):
    """Get an `n_qubits`-vertex induced subgraph of the backend's coupling
    map. Picks the densest connected subgraph by a greedy walk from the
    most-connected qubit. Returns (subgraph_as_networkx, physical_qubits_list).
    """
    coupling = backend.coupling_map
    # Build networkx graph from the directed coupling
    G = nx.Graph()
    for a, b in coupling.get_edges():
        G.add_edge(a, b)
    # Pick a connected subgraph greedy
    degrees = dict(G.degree())
    start = max(degrees, key=degrees.get)
    selected = [start]
    frontier = set(G.neighbors(start))
    while len(selected) < n_qubits and frontier:
        # Pick frontier node with most connections to current selected
        best = max(frontier,
                   key=lambda v: sum(1 for u in selected if G.has_edge(v, u)))
        selected.append(best)
        frontier.update(G.neighbors(best))
        frontier -= set(selected)
    if len(selected) < n_qubits:
        raise ValueError(f"Backend cannot provide {n_qubits}-qubit subgraph")
    subgraph = G.subgraph(selected).copy()
    # Relabel physical qubits to logical [0..n-1]
    mapping = {p: i for i, p in enumerate(selected)}
    sub_relabel = nx.relabel_nodes(subgraph, mapping)
    return sub_relabel, selected


def route_with_method(qc_orig, method, graph, coupling, lam, seed_base, k=5,
                     alim_mult=10):
    """Route the original circuit with SABRE or SABRE-MS, best of K trials by
    ASAP makespan under the qgym cost model (cnot=2, swap=6, 1q=1). This
    matches the audit's selection criterion exactly: SABRE-MS picks the
    same routing it would in the audit. Equivalence is verified on every
    routed circuit before scoring (any non-equivalent circuit would
    raise AssertionError)."""
    from exp_real_full import asap_makespan_qc
    best_qc, best_mks = None, float("inf")
    best_2q = None
    for k_i in range(k):
        seed = seed_base + k_i
        qc_basis = decompose_basis(qc_orig)
        perm = get_layout(qc_basis, coupling, seed)
        qc_phys = apply_layout_to_qc(qc_basis, perm)
        if method == "sabre":
            qc_routed = route_with_sabre(qc_phys, coupling, seed,
                                         verify_equivalence=True)
        elif method == "sabre_ms":
            qc_routed = route_with_sabre_ms(qc_phys, graph, coupling, lam,
                                           seed, alim_mult,
                                           verify_equivalence=True)
            if qc_routed is None:
                continue
        else:
            raise ValueError(method)
        qc_opt = optimize_qc(qc_routed)
        mks = asap_makespan_qc(qc_opt)
        if mks < best_mks:
            best_mks = mks
            best_qc = qc_opt
            best_2q = sum(1 for instr in qc_opt.data
                         if len(instr.qubits) == 2)
    return best_qc, best_2q


def make_inverse_test_circuit(qc_routed: QuantumCircuit,
                              physical_qubits: list[int],
                              backend) -> QuantumCircuit:
    """Compose qc_routed with its own inverse, lift to backend's full qubit
    space, add measurements, and transpile to backend's basis gates (without
    re-routing — we want to preserve our routing choices).

    The Sampler requires circuits already at the backend's ISA (basis gates +
    coupling-map adherence). Our qc_routed is on n_qubits logical qubits with
    SWAPs/CNOTs/u3 — we need to map those logical qubits to the chosen physical
    qubits on the backend, then translate to the backend's native gate set.
    """
    n = qc_routed.num_qubits
    inverse = qc_routed.inverse()
    closed = QuantumCircuit(n)
    closed.compose(qc_routed, inplace=True)
    closed.compose(inverse, inplace=True)

    # Lift to backend qubit-count and place on chosen physical qubits
    full = QuantumCircuit(backend.num_qubits)
    full.compose(closed, qubits=physical_qubits, inplace=True)
    # Measure only the n logical qubits (the inverse test should drive
    # them to |0>)
    creg = ClassicalRegister(n, "c")
    full.add_register(creg)
    for li, pi in enumerate(physical_qubits):
        full.measure(pi, li)

    # Translate to backend native basis WITHOUT routing (we preserved
    # adjacency by construction; layout is fixed by placing on
    # physical_qubits which form an adjacent subgraph).
    # Use preset pass manager at optimisation level 0 with
    # initial_layout=trivial; on a real backend it will only translate
    # basis gates.
    pm = generate_preset_pass_manager(
        backend=backend, optimization_level=0,
        initial_layout=list(range(backend.num_qubits)),
    )
    return pm.run(full)


# ============================================================
# Driver
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default=None,
                  help="IBM backend name; default = least busy")
    p.add_argument("--n-qubits", type=int, required=True)
    p.add_argument("--family", choices=list(GENERATORS), required=True)
    p.add_argument("--n-circuits", type=int, default=4)
    p.add_argument("--k", type=int, default=5,
                  help="Routing trials per method, best by 2q gate count")
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--lam", type=float, default=None,
                  help="Override SABRE-MS lambda (default from lambda_for)")
    p.add_argument("--out", default=None)
    p.add_argument("--dry-run", action="store_true",
                  help="Build circuits, print sizes, do not submit")
    args = p.parse_args()

    # ---- IBM account / backend selection
    print("Loading IBM Runtime service...")
    service = QiskitRuntimeService()
    if args.backend:
        backend = service.backend(args.backend)
    else:
        backend = service.least_busy(operational=True, simulator=False,
                                    min_num_qubits=args.n_qubits)
    print(f"  Using backend: {backend.name}  ({backend.num_qubits} qubits)")
    print(f"  Pending jobs: {backend.status().pending_jobs}")

    # ---- Pick physical-qubit subgraph
    graph, physical_qubits = get_backend_subgraph(backend, args.n_qubits)
    print(f"  Selected physical qubits: {physical_qubits}")
    print(f"  Edges in subgraph: {list(graph.edges())}")
    # Build a CouplingMap for the SABRE pipeline (it operates in logical
    # space [0..n-1] for our routing helpers)
    edges_logical = list(graph.edges())
    coupling = CouplingMap([list(e) for e in edges_logical] +
                          [list(reversed(e)) for e in edges_logical])

    # Lambda: prefer heldout value if available (methodologically consistent
    # with audit + ESP), else fall back to class-rule.
    lam = args.lam
    if lam is None:
        try:
            with open("results/heldout_lambda_OLD_BROKEN.json") as f:
                heldout = json.load(f)
            # Try ibm_marrakesh family-specific lookups (ring8 since we use 8-qubit subgraph)
            for key in [f"ring8/{args.family}", f"linear7/{args.family}"]:
                if key in heldout:
                    lam = heldout[key]["best_lambda_from_train"]
                    print(f"  Lambda: {lam} (from {key} held-out value)")
                    break
        except FileNotFoundError:
            pass
        if lam is None:
            lam = lambda_for(args.family, "ibm")
            print(f"  Lambda: {lam} (class-rule fallback)")
    else:
        print(f"  Lambda: {lam} (override)")

    # ---- Generate + route circuits
    gen = GENERATORS[args.family]
    isa_circuits = []      # what we submit
    meta = []              # (ci, method, 2q_count)
    print("\nRouting circuits...")
    for ci in range(args.n_circuits):
        qc_orig = gen(args.n_qubits, seed=2026 + ci)
        for method in ("sabre", "sabre_ms"):
            seed_base = (2026 + ci) * 100 + (0 if method == "sabre" else 50)
            qc_routed, cnt = route_with_method(
                qc_orig, method, graph, coupling, lam,
                seed_base, k=args.k,
            )
            if qc_routed is None:
                print(f"  ci={ci} {method}: FAILED to route")
                continue
            isa = make_inverse_test_circuit(qc_routed, physical_qubits, backend)
            isa_circuits.append(isa)
            meta.append({"ci": ci, "method": method, "post_opt_2q_count": cnt,
                        "isa_depth": isa.depth()})
            print(f"  ci={ci} {method:9s}: post-opt 2q={cnt:3d}  isa_depth={isa.depth():4d}")

    if args.dry_run:
        print("\nDry run — not submitting.")
        return

    # ---- Submit as a single Sampler job
    print(f"\nSubmitting {len(isa_circuits)} circuits to {backend.name}...")
    sampler = Sampler(backend)
    sampler.options.default_shots = args.shots
    job = sampler.run(isa_circuits)
    print(f"  Job ID: {job.job_id()}")
    print(f"  Waiting for results (may queue for minutes to hours)...")
    t0 = time.time()
    result = job.result()
    elapsed = time.time() - t0
    print(f"  Result returned after {elapsed:.0f}s")

    # ---- Compute fidelity per circuit, aggregate
    fids = []
    for i, m in enumerate(meta):
        pubresult = result[i]
        counts = pubresult.data.c.get_counts()
        f = hellinger_fidelity_against_zeros(counts, args.n_qubits)
        fids.append(f)
        m["fidelity_to_zeros"] = f
        m["shots"] = sum(counts.values())
        # Save top 3 bitstrings for sanity
        sorted_bs = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
        m["top3_counts"] = sorted_bs

    # Paired per ci
    by_ci = {}
    for m, f in zip(meta, fids):
        by_ci.setdefault(m["ci"], {})[m["method"]] = f
    paired_sabre, paired_ms = [], []
    for ci, dct in by_ci.items():
        if "sabre" in dct and "sabre_ms" in dct:
            paired_sabre.append(dct["sabre"])
            paired_ms.append(dct["sabre_ms"])
    print("\nResults:")
    print(f"  {'ci':>3} {'sabre F':>10} {'MS F':>10} {'ratio':>9}")
    for ci, dct in sorted(by_ci.items()):
        s = dct.get("sabre", 0.0); m = dct.get("sabre_ms", 0.0)
        r = m / s if s > 0 else float("inf")
        print(f"  {ci:>3} {s:>10.4f} {m:>10.4f} {r:>8.3f}x")
    if paired_sabre:
        s_arr = np.array(paired_sabre, float)
        m_arr = np.array(paired_ms, float)
        ratio = (m_arr / s_arr)
        ratio = ratio[np.isfinite(ratio)]
        print(f"\n  Mean SABRE   F: {s_arr.mean():.4f}")
        print(f"  Mean SABRE-MS F: {m_arr.mean():.4f}")
        print(f"  Mean ratio: {ratio.mean():.3f}x   median {np.median(ratio):.3f}x")
        if len(s_arr) >= 3:
            from scipy import stats
            try:
                _, p = stats.wilcoxon(m_arr, s_arr, alternative="greater")
                print(f"  Wilcoxon p (MS > SABRE): {p:.3e}")
            except Exception:
                pass

    # ---- Save
    os.makedirs("results", exist_ok=True)
    out_path = args.out or f"results/ibm_hardware_{backend.name}_{args.family}_{args.n_qubits}q.json"
    summary = {
        "backend": backend.name,
        "family": args.family,
        "n_qubits": args.n_qubits,
        "physical_qubits": physical_qubits,
        "edges": [[int(a), int(b)] for a, b in graph.edges()],
        "shots": args.shots,
        "k_routing_trials": args.k,
        "lambda": lam,
        "job_id": job.job_id(),
        "circuits": meta,
        "paired": [{"ci": ci, "sabre": dct.get("sabre"),
                   "sabre_ms": dct.get("sabre_ms")}
                  for ci, dct in sorted(by_ci.items())],
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()

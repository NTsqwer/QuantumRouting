"""On-hardware lambda sweep: route each cell at every lambda, run on the real
device, report the lambda with the best MEASURED fidelity.

Unlike exp_lambda_for_hardware.py (which picks lambda by an ESP MODEL), this runs
the whole lambda grid on ibm_marrakesh and lets the device decide. For each
(family, size): one SABRE baseline (= lambda 0) plus SABRE-MS at each lambda > 0,
all truncated to --cap2q two-qubit gates so the inverse test stays resolvable.
All circuits go in ONE Sampler job. We report, per cell, the best-fidelity lambda
and its ratio over the SABRE baseline.

Budget-first: --dry-run builds everything, prints depths + estimated runtime,
submits nothing.

Run with RP_SKIP_EQUIV=1. Output: results/ibm_lambda_sweep_<backend>.json
"""
from __future__ import annotations
import argparse
import json
import os
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=DeprecationWarning)

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from mqt.bench import get_benchmark, BenchmarkLevel

from exp_real_full import (
    decompose_basis, get_layout, apply_layout_to_qc,
    route_with_sabre, route_with_sabre_ms, optimize_qc, asap_makespan_qc,
)
from exp_mqt_bench import strip_measurements
from exp_ibm_hardware import (
    hellinger_fidelity_against_zeros, get_backend_subgraph,
    make_inverse_test_circuit,
)
import networkx as nx

LAMBDA_GRID = [0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.5]


def subgraph_from_start(backend, n_qubits, start):
    """Greedy connected n-qubit subgraph grown from a chosen start qubit,
    so we can target a specific physical region (e.g. best-readout qubits)."""
    G = nx.Graph()
    for a, b in backend.coupling_map.get_edges():
        G.add_edge(a, b)
    selected = [start]
    frontier = set(G.neighbors(start))
    while len(selected) < n_qubits and frontier:
        best = max(frontier, key=lambda v: sum(1 for u in selected if G.has_edge(v, u)))
        selected.append(best)
        frontier.update(G.neighbors(best))
        frontier -= set(selected)
    if len(selected) < n_qubits:
        raise ValueError(f"cannot grow {n_qubits}-qubit subgraph from {start}")
    mapping = {p: i for i, p in enumerate(selected)}
    return nx.relabel_nodes(G.subgraph(selected).copy(), mapping), selected


def best_readout_start(backend):
    """Physical qubit with the lowest readout error (best region anchor)."""
    props = backend.properties()
    cand = []
    for q in range(backend.num_qubits):
        try:
            cand.append((q, props.readout_error(q)))
        except Exception:
            pass
    return min(cand, key=lambda x: x[1])[0]


def cap_2q(qc, cap):
    if not cap:
        return qc
    out = QuantumCircuit(qc.num_qubits)
    seen = 0
    for inst in qc.data:
        if len(inst.qubits) == 2:
            if seen >= cap:
                break
            seen += 1
        out.append(inst.operation, inst.qubits)
    return out


def best_route(qc_orig, method, graph, coupling, lam, k):
    """Best-of-K by makespan (same protocol as the main hardware run)."""
    seed_base = hash((method, lam)) % 10000
    best_qc, best_mks, best_2q = None, float("inf"), None
    for ki in range(k):
        seed = seed_base + ki
        qb = decompose_basis(qc_orig)
        perm = get_layout(qb, coupling, seed)
        qp = apply_layout_to_qc(qb, perm)
        if method == "sabre":
            r = route_with_sabre(qp, coupling, seed, verify_equivalence=False)
        else:
            r = route_with_sabre_ms(qp, graph, coupling, lam, seed,
                                    alim_mult=50, verify_equivalence=False)
            if r is None:
                continue
        o = optimize_qc(r)
        mk = asap_makespan_qc(o)
        if mk < best_mks:
            best_mks, best_qc = mk, o
            best_2q = sum(1 for i in o.data if len(i.qubits) == 2)
    return best_qc, best_2q, best_mks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="ibm_marrakesh")
    p.add_argument("--families", nargs="+", default=["qftentangled", "qft"])
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--cap2q", type=int, default=15)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--region", default="default",
                  help="'default' (densest), 'best' (lowest-readout anchor), or an int start qubit")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    print("Loading IBM Runtime service...")
    service = QiskitRuntimeService()
    backend = service.backend(args.backend)
    print(f"  Backend: {backend.name}, queue={backend.status().pending_jobs}")

    nq = args.size
    if args.region == "default":
        graph, physical_qubits = get_backend_subgraph(backend, nq)
    elif args.region == "best":
        start = best_readout_start(backend)
        graph, physical_qubits = subgraph_from_start(backend, nq, start)
    else:
        graph, physical_qubits = subgraph_from_start(backend, nq, int(args.region))
    print(f"  Region '{args.region}' -> physical qubits {physical_qubits}")
    edges = list(graph.edges())
    coupling = CouplingMap([list(e) for e in edges] + [list(reversed(e)) for e in edges])

    isa_circuits, meta = [], []
    print(f"\nBuilding lambda sweep: {args.families} @{nq}q, cap2q={args.cap2q}, "
          f"grid={LAMBDA_GRID}")
    for fam in args.families:
        qc0 = cap_2q(strip_measurements(
            get_benchmark(benchmark=fam, circuit_size=nq, level=BenchmarkLevel.INDEP)),
            args.cap2q)
        # SABRE baseline (lambda 0)
        sq, s2, smk = best_route(qc0, "sabre", graph, coupling, 0.0, args.k)
        isa = make_inverse_test_circuit(sq, physical_qubits, backend)
        isa_circuits.append(isa)
        meta.append({"family": fam, "lam": 0.0, "method": "sabre",
                     "n2q": s2, "makespan": smk, "depth": isa.depth()})
        print(f"  {fam}@{nq}q  sabre        2q={s2:>3} mk={smk:>3} d={isa.depth()}")
        # SABRE-MS at each lambda
        for lam in LAMBDA_GRID:
            mq, m2, mmk = best_route(qc0, "sabre_ms", graph, coupling, lam, args.k)
            if mq is None:
                continue
            isa = make_inverse_test_circuit(mq, physical_qubits, backend)
            isa_circuits.append(isa)
            meta.append({"family": fam, "lam": lam, "method": "sabre_ms",
                         "n2q": m2, "makespan": mmk, "depth": isa.depth()})
            print(f"  {fam}@{nq}q  ms lam={lam:<5} 2q={m2:>3} mk={mmk:>3} d={isa.depth()}")

    sum_depth = sum(m["depth"] for m in meta)
    est = sum_depth / 280.0 * (args.shots / 4096) * 1.0  # rough, like the main script
    print(f"\n  Circuits: {len(isa_circuits)} | sum ISA depth {sum_depth} | "
          f"~{est:.0f}s est @ {args.shots} shots")
    try:
        rem = service.usage().get("usage_remaining_seconds")
        print(f"  Remaining free quota: {rem} s")
    except Exception:
        pass

    if args.dry_run:
        print("\nDry run -- nothing submitted.")
        return

    print(f"\nSubmitting {len(isa_circuits)} circuits...")
    sampler = Sampler(backend)
    job = sampler.run(isa_circuits, shots=args.shots)
    print(f"  Job ID: {job.job_id()}  (waiting...)")
    result = job.result()
    for i, m in enumerate(meta):
        counts = result[i].data.c.get_counts()
        m["fidelity"] = hellinger_fidelity_against_zeros(counts, nq)
        m["shots"] = sum(counts.values())

    # Report best lambda per family.
    out = {"backend": backend.name, "size": nq, "cap2q": args.cap2q,
           "shots": args.shots, "job_id": job.job_id(), "meta": meta, "by_family": {}}
    print("\n=== ON-HARDWARE LAMBDA SWEEP ===")
    for fam in args.families:
        rows = [m for m in meta if m["family"] == fam]
        sabre = next(m for m in rows if m["method"] == "sabre")
        ms_rows = [m for m in rows if m["method"] == "sabre_ms"]
        best = max(ms_rows, key=lambda m: m["fidelity"])
        print(f"\n  {fam}@{nq}q:  SABRE fid={sabre['fidelity']:.4f}")
        for m in ms_rows:
            star = "  <-- BEST" if m is best else ""
            ratio = m["fidelity"] / sabre["fidelity"] if sabre["fidelity"] else float("nan")
            print(f"    lam={m['lam']:<5} fid={m['fidelity']:.4f}  "
                  f"ratio={ratio:.3f}x  (2q={m['n2q']}, mk={m['makespan']}){star}")
        out["by_family"][fam] = {
            "sabre_fid": sabre["fidelity"],
            "best_lam": best["lam"], "best_fid": best["fidelity"],
            "best_ratio": best["fidelity"] / sabre["fidelity"] if sabre["fidelity"] else None,
        }

    outpath = args.out or f"results/ibm_lambda_sweep_{backend.name}.json"
    json.dump(out, open(outpath, "w"), indent=2)
    print(f"\nSaved {outpath}")


if __name__ == "__main__":
    main()

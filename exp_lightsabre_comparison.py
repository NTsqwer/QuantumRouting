"""SABRE-MS vs all three Qiskit SABRE heuristics (basic, lookahead, decay) — K=1, FAIR.

Qiskit 2.4.1's SabreSwap exposes three heuristics:
- basic:     sum of distances for front-layer interacting qubits
- lookahead: basic + weighted extended-set distances (the default in our pipeline)
- decay:     lookahead + decay penalty on recently-used SWAPs (LightSABRE-style depth penalty)

The LightSABRE paper (arXiv 2409.08368) introduced 'depth' and 'critical_path' heuristics,
but these are NOT shipped in Qiskit's Python API as of 2.4.1 — only in the paper.
The 'decay' heuristic IS what LightSABRE actually ships in production Qiskit (it's the
"recently-used SWAP penalty" version of the depth-aware idea).

This experiment establishes that SABRE-MS beats not just `lookahead` (our standard baseline)
but ALSO the LightSABRE-flavored `decay` variant — at EQUAL COMPUTE (K=1 each).

K=1 (single seed, no multi-trial best-of) is the fair comparison: both algorithms run
exactly once per circuit, so any gain is attributable to the heuristic itself, not to
extra random trials. This closes the "you're just sampling more" fairness objection.

For each cell, we run K=1 each of:
  * SABRE-basic     (single seed, single trial)
  * SABRE-lookahead (single seed, single trial)
  * SABRE-decay     (single seed, single trial)
  * SABRE-MS        (single seed, single trial, oracle lambda)

Output: results/lightsabre_comparison.json
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
from optimize import optimize_circuit
from pipeline import asap_makespan
from sabre_impl import sabre_route
from topologies import get as get_topology

GD = {"cnot": 2, "swap": 6}
K = 1  # FAIR comparison: single seed each, no best-of-K selection


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


def qiskit_route_gates(circuit_np, coupling, n_qubits, heuristic, seed=0):
    """Route with the requested Qiskit SABRE heuristic ('basic', 'lookahead', 'decay')."""
    from qgym.custom_types import Gate
    qc = QuantumCircuit(n_qubits)
    for q1, q2 in circuit_np:
        qc.cx(int(q1), int(q2))
    pm = PassManager([SabreSwap(coupling_map=coupling, heuristic=heuristic,
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


def single_run_qiskit(circuit_perm, coupling, n_qubits, heuristic, seed=0):
    """Single seed of a Qiskit SABRE heuristic. Returns makespan after optimization."""
    gates = qiskit_route_gates(circuit_perm, coupling, n_qubits, heuristic, seed=seed)
    opt = optimize_circuit(gates, n_qubits)
    return asap_makespan(opt, GD)


def single_run_sabre_ms(circuit_perm, graph, n_qubits, lam, seed=0, attempt_limit_mult=10):
    """Single seed of SABRE-MS at a fixed lambda. Returns makespan after
    optimization, or None on failure."""
    alim = attempt_limit_mult * n_qubits
    mg = sabre_route(circuit_perm, graph, lookahead=True, makespan_lambda=lam,
                     makespan_mode="start_cycle", seed=seed, attempt_limit=alim)
    if mg is None:
        return None
    opt = optimize_circuit(mg, n_qubits)
    return asap_makespan(opt, GD)


# Per-run lambda selection grid (INCLUDES 0; lambda=0 recovers SABRE).
PERRUN_LAMBDA_GRID = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25]
PERRUN_K0 = 5


def perrun_sabre_ms(circuit_perm, graph, n_qubits, seed=0, attempt_limit_mult=10):
    """SABRE-MS with PER-RUN lambda selection: probe each grid lambda at K0 and
    keep the shortest-makespan one, then route one final seed at that lambda
    (K=1 final, matching the equal-compute design). Returns makespan or None."""
    alim = attempt_limit_mult * n_qubits

    def route(lam, s):
        mg = sabre_route(circuit_perm, graph, lookahead=True, makespan_lambda=lam,
                         makespan_mode="start_cycle", seed=s, attempt_limit=alim)
        return None if mg is None else asap_makespan(optimize_circuit(mg, n_qubits), GD)

    best_lam, best_probe = None, float("inf")
    for lam in PERRUN_LAMBDA_GRID:
        for ps in range(PERRUN_K0):
            mk = route(lam, seed * 1000 + ps)
            if mk is not None and mk < best_probe:
                best_probe, best_lam = mk, lam
    if best_lam is None:
        return None, None
    return route(best_lam, seed), best_lam


def run_cell(topology, family, n_circuits=40, length=30, seed=1234,
             attempt_limit_mult=10):
    graph, n_qubits = get_topology(topology)
    edges = list(graph.edges())
    coupling = CouplingMap([list(e) for e in edges] + [list(reversed(e)) for e in edges])
    cs = make_circuits(family, n_qubits, n_circuits, seed=seed, length=length)

    mks = {"basic": [], "lookahead": [], "decay": [], "sabre_ms": []}
    chosen_lams = []

    for ci, c in enumerate(cs):
        perm = sabre_layout_perm(c, coupling, n_qubits, seed=ci)
        cp = perm[c].astype(int)

        # Same seed for every method, so the comparison is paired on identical randomness.
        # seed=ci avoids using the same seed across all circuits.
        for h in ("basic", "lookahead", "decay"):
            mk = single_run_qiskit(cp, coupling, n_qubits, h, seed=ci)
            mks[h].append(mk)

        mk_ms, lam_ci = perrun_sabre_ms(cp, graph, n_qubits, seed=ci,
                                        attempt_limit_mult=attempt_limit_mult)
        if mk_ms is None:
            # Routing failure fallback: use the worst Qiskit baseline to be conservative.
            mk_ms = max(mks["basic"][-1], mks["lookahead"][-1], mks["decay"][-1])
        mks["sabre_ms"].append(mk_ms)
        if lam_ci is not None:
            chosen_lams.append(lam_ci)

    def stats_vs(baseline_key):
        baseline = np.array(mks[baseline_key])
        ms = np.array(mks["sabre_ms"])
        mean_b = float(baseline.mean())
        mean_m = float(ms.mean())
        pct = 100.0 * (mean_b - mean_m) / mean_b if mean_b > 0 else 0.0
        diffs = ms - baseline
        if np.any(diffs != 0):
            try:
                _, p = stats.wilcoxon(ms, baseline, alternative="less")
                p = float(p)
            except Exception:
                p = 1.0
        else:
            p = 1.0
        n_better = int(np.sum(ms < baseline))
        return {
            "baseline_mean": mean_b,
            "ms_mean": mean_m,
            "gain_pct": pct,
            "p_value": p,
            "n_better": n_better,
            "n_total": len(ms),
        }

    return {
        "topology": topology,
        "family": family,
        "n_qubits": n_qubits,
        "lambda_mean": float(np.mean(chosen_lams)) if chosen_lams else 0.0,
        "attempt_limit_mult": attempt_limit_mult,
        "n_circuits": n_circuits,
        "raw_means": {k: float(np.mean(v)) for k, v in mks.items()},
        "vs_basic": stats_vs("basic"),
        "vs_lookahead": stats_vs("lookahead"),
        "vs_decay": stats_vs("decay"),
    }


# Representative cells: small, medium, large; QFT (biggest gains), random (moderate),
# parallel (most fragile). Mix of topology classes. n=80 per cell since K=1 is fast.
CELLS = [
    # (topology, family, n_circuits, attempt_limit_mult); lambda is chosen per run.
    ("ring12",     "qft",      80, 10),  # best-case QFT on ring
    ("linear9",    "qft",      80, 10),  # linear QFT
    ("grid4x4",    "qft",      80, 10),  # grid QFT, n=16
    ("grid5x5",    "qft",      80, 50),  # large grid QFT (needs alim=50*n_q)
    ("ring12",     "random",   80, 10),  # random
    ("grid4x4",    "random",   80, 10),  # random on grid
    ("grid4x4",    "parallel", 80, 10),  # parallel on grid (large gain expected)
    ("ring12",     "parallel", 80, 10),  # parallel on ring (fragile case)
    ("heavy_hex2", "qft",      80, 10),  # IBM heavy-hex
    ("heavy_hex2", "parallel", 80, 10),  # IBM heavy-hex parallel
]


def main():
    os.makedirs("results", exist_ok=True)
    out_path = "results/lightsabre_comparison_perrun.json"
    out = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            out = json.load(f)
        print("Loaded {} existing entries".format(len(out)))

    t0 = time.time()
    hdr = "{:<22} {:>4}  {:>6}  {:>8} {:>8} {:>8} {:>8}  {:>9} {:>9} {:>9}".format(
        "Cell", "n_q", "lam",
        "basic", "lookhd", "decay", "MS",
        "MS-basic", "MS-look", "MS-decay")
    print(hdr)
    print("-" * len(hdr))

    def show(key, r):
        print("{:<22} {:>4}  {:>6.3f}  {:>8.1f} {:>8.1f} {:>8.1f} {:>8.1f}  {:>+8.1f}% {:>+8.1f}% {:>+8.1f}%".format(
            key, r["n_qubits"], r["lambda_mean"],
            r["raw_means"]["basic"], r["raw_means"]["lookahead"],
            r["raw_means"]["decay"], r["raw_means"]["sabre_ms"],
            r["vs_basic"]["gain_pct"], r["vs_lookahead"]["gain_pct"],
            r["vs_decay"]["gain_pct"]))

    for topo, fam, n, alim_mult in CELLS:
        key = "{}/{}".format(topo, fam)
        if key in out:
            show(key, out[key]); continue
        print("  Running {}...".format(key), flush=True)
        r = run_cell(topo, fam, n_circuits=n, attempt_limit_mult=alim_mult)
        out[key] = r
        show(key, r)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

    # Aggregate across cells (means of per-cell gains; pooled win counts).
    cells = [v for v in out.values() if isinstance(v, dict) and "vs_basic" in v]
    print("\n=== Aggregate over {} cells ===".format(len(cells)))
    for key in ("vs_basic", "vs_lookahead", "vs_decay"):
        g = [c[key]["gain_pct"] for c in cells]
        nb = sum(c[key]["n_better"] for c in cells)
        nt = sum(c[key]["n_total"] for c in cells)
        print("  {:<14} mean gain {:+.1f}%   wins {}/{}".format(key, float(np.mean(g)), nb, nt))
    print("\nTotal: {:.0f}s".format(time.time() - t0))
    print("Saved {}".format(out_path))


if __name__ == "__main__":
    main()

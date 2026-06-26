"""SWAP absorption rate: do SABRE-MS routings have more cancellation opportunities?

For each routing (SABRE-lookahead vs SABRE-MS), measure:
  - n_swaps:        SWAP count inserted by the router
  - pre_cnot:       CNOT count after SWAP -> 3CNOT decomposition (no optimization)
  - post_cnot:      CNOT count after the full Qiskit gate-cancellation pass
  - absorbed_pct:   (pre - post) / pre   -- what fraction of CNOTs the optimizer kills

Hypothesis: SABRE-MS produces routings where the optimizer kills MORE CNOTs,
because finish-time-aware SWAPs land adjacent to upcoming CNOTs on the same pair
(those are the SWAPs that get partially absorbed). If true, SABRE-MS's makespan
win is explained by a measurable mechanism, not just "the optimizer happens to
like it."

Tested on 11 cells × K=20 SABRE trials. Compares mean absorption rates per cell.

Output: results/swap_absorption.json
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
from scipy import stats
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import SabreSwap

from circuits import make_circuits
from exp_sabrems_unoptimized import sabre_layout_perm, wilcoxon_paired, GD
from optimize import optimize_circuit
from pipeline import asap_makespan
from sabre_impl import sabre_route
from topologies import get as get_topology

# Per-run lambda selection grid (INCLUDES 0; lambda=0 recovers SABRE).
PERRUN_LAMBDA_GRID = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25]
PERRUN_K0 = 5


def _decompose_swaps_to_cnots(gates):
    """Each SWAP = 3 CNOTs. Return the expanded CNOT-only gate list count."""
    n = 0
    for g in gates:
        if g.name == "cnot":
            n += 1
        elif g.name == "swap":
            n += 3
    return n


def _count_post_optimizer_cnots(gates, n_qubits):
    """Run the optimizer; count CNOTs that survive."""
    optimized = optimize_circuit(gates, n_qubits)
    return sum(1 for g in optimized if g.name == "cnot")


def sabre_route_qiskit(cp, coupling, n_qubits, heuristic, seed):
    """One Qiskit SABRE routing -> qgym Gate list (cnots and swaps)."""
    from qgym.custom_types import Gate
    qc = QuantumCircuit(n_qubits)
    for q1, q2 in cp:
        qc.cx(int(q1), int(q2))
    pm = PassManager([SabreSwap(coupling_map=coupling, heuristic=heuristic,
                                seed=seed, trials=1)])
    routed = pm.run(qc)
    out = []
    n_swaps = 0
    for instr in routed.data:
        op = instr.operation
        qs = [routed.find_bit(q).index for q in instr.qubits]
        if len(qs) == 2:
            name = "cnot" if op.name == "cx" else op.name
            out.append(Gate(name, int(qs[0]), int(qs[1])))
            if name == "swap":
                n_swaps += 1
    return out, n_swaps


def measure_routing(gates, n_qubits):
    """Return dict with pre/post-optimizer CNOT counts, absorption rate, makespan."""
    n_swaps = sum(1 for g in gates if g.name == "swap")
    pre_cnot = _decompose_swaps_to_cnots(gates)
    post_cnot = _count_post_optimizer_cnots(gates, n_qubits)
    return {
        "n_swaps": n_swaps,
        "pre_cnot": pre_cnot,
        "post_cnot": post_cnot,
        "absorbed_pct": 100.0 * (pre_cnot - post_cnot) / pre_cnot if pre_cnot > 0 else 0.0,
        "mk_raw": asap_makespan(gates, GD),
        "mk_opt": asap_makespan(optimize_circuit(gates, n_qubits), GD),
    }


def run_cell(topology, family, n_circuits, length, alim_mult, K=20):
    graph, n_qubits = get_topology(topology)
    edges = list(graph.edges())
    coupling = CouplingMap([list(e) for e in edges] + [list(reversed(e)) for e in edges])
    cs = make_circuits(family, n_qubits, n_circuits, seed=1234, length=length)

    # Per-circuit, K=1 (single seed) and K=20-by-makespan-opt (best routing under
    # the optimized pipeline) — measure absorption on the chosen routing.
    sabre_k1 = []   # SABRE-lookahead K=1
    sabre_k20 = []  # SABRE-lookahead K=20 best-by-optimized-makespan
    ms_k1 = []
    ms_k20 = []
    chosen_lams = []  # per-run lambda picked per circuit

    for ci, c in enumerate(cs):
        perm = sabre_layout_perm(c, coupling, n_qubits, seed=ci)
        cp = perm[c].astype(int)

        # SABRE-lookahead K=20: collect all routings, pick best by post-opt makespan
        sabre_all = []
        for s in range(K):
            gates, _ = sabre_route_qiskit(cp, coupling, n_qubits, "lookahead", s)
            sabre_all.append(measure_routing(gates, n_qubits))
        sabre_k1.append(sabre_all[ci % K])  # use one fixed-seed trial as K=1
        best_idx = int(np.argmin([m["mk_opt"] for m in sabre_all]))
        sabre_k20.append(sabre_all[best_idx])

        # SABRE-MS K=20, with PER-RUN lambda selection: probe each grid lambda at
        # K0 and keep the one with the shortest optimized makespan, then route K=20.
        def _ms_route(lam_try, seed):
            return sabre_route(cp, graph, lookahead=True, makespan_lambda=lam_try,
                               makespan_mode="start_cycle", seed=seed,
                               attempt_limit=alim_mult * n_qubits)
        best_lam, best_probe = None, float("inf")
        for lam_try in PERRUN_LAMBDA_GRID:
            pm_best = float("inf")
            for s in range(PERRUN_K0):
                g = _ms_route(lam_try, s)
                if g is not None:
                    mk = asap_makespan(optimize_circuit(g, n_qubits), GD)
                    if mk < pm_best:
                        pm_best = mk
            if pm_best < best_probe:
                best_probe, best_lam = pm_best, lam_try
        chosen_lams.append(best_lam if best_lam is not None else 0.0)
        ms_all = []
        if best_lam is not None:
            for s in range(K):
                g = _ms_route(best_lam, s)
                ms_all.append(measure_routing(g, n_qubits) if g is not None else None)
        valid = [m for m in ms_all if m is not None]
        if not valid:
            ms_k1.append(sabre_all[ci % K])  # fallback
            ms_k20.append(sabre_all[best_idx])
            continue
        ms_k1.append(ms_all[ci % K] if ms_all[ci % K] is not None else valid[0])
        best_ms_idx = int(np.argmin([m["mk_opt"] for m in valid]))
        ms_k20.append(valid[best_ms_idx])

    def agg(rs):
        return {
            "mean_n_swaps":       float(np.mean([r["n_swaps"] for r in rs])),
            "mean_pre_cnot":      float(np.mean([r["pre_cnot"] for r in rs])),
            "mean_post_cnot":     float(np.mean([r["post_cnot"] for r in rs])),
            "mean_absorbed_pct":  float(np.mean([r["absorbed_pct"] for r in rs])),
            "mean_mk_raw":        float(np.mean([r["mk_raw"] for r in rs])),
            "mean_mk_opt":        float(np.mean([r["mk_opt"] for r in rs])),
        }

    return {
        "topology": topology, "family": family,
        "lambda_mean": float(np.mean(chosen_lams)), "lambda_per_circuit": chosen_lams,
        "n_qubits": n_qubits, "n_circuits": n_circuits, "K": K,
        "sabre_k1": agg(sabre_k1),
        "sabre_k20": agg(sabre_k20),
        "ms_k1": agg(ms_k1),
        "ms_k20": agg(ms_k20),
        # Head-to-head paired tests on absorption rate
        "p_ms_higher_absorbed_k1": wilcoxon_paired(
            -np.array([r["absorbed_pct"] for r in ms_k1]),
            -np.array([r["absorbed_pct"] for r in sabre_k1]),
        ),
        "p_ms_higher_absorbed_k20": wilcoxon_paired(
            -np.array([r["absorbed_pct"] for r in ms_k20]),
            -np.array([r["absorbed_pct"] for r in sabre_k20]),
        ),
    }


# The 8 on-core-topology cells. lambda is chosen PER RUN by the makespan it
# reaches (the rule the paper describes), so this experiment is consistent with
# the main comparison's methodology.
CELLS = [
    ("linear7",  "qft",      30, 30, 10),
    ("ring8",    "qft",      30, 30, 10),
    ("ring12",   "qft",      25, 30, 10),
    ("ring12",   "random",   25, 30, 10),
    ("grid3x3",  "qft",      30, 30, 10),
    ("grid4x4",  "qft",      20, 30, 10),
    ("grid4x4",  "parallel", 20, 30, 10),
    ("heavy_hex2", "qft",    20, 30, 10),
]


def main():
    os.makedirs("results", exist_ok=True)
    out_path = "results/swap_absorption_perrun.json"
    out = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            out = json.load(f)
        print(f"Loaded {len(out)} cells")

    t0 = time.time()
    print(f"{'Cell':<24} {'n_q':>4}  "
          f"{'SABRE swaps':>11} {'SABRE preC':>11} {'SABRE postC':>12} {'SABRE abs%':>11}   "
          f"{'MS swaps':>9} {'MS preC':>8} {'MS postC':>9} {'MS abs%':>9}  "
          f"{'absDelta':>9} {'p':>9}")
    print("-" * 160)

    for topo, fam, n, length, alim in CELLS:
        key = f"{topo}/{fam}"
        if key in out:
            r = out[key]
        else:
            print(f"  Running {key}...", flush=True)
            tc = time.time()
            r = run_cell(topo, fam, n, length, alim)
            print(f"    ...{time.time()-tc:.0f}s", flush=True)
            out[key] = r
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
        s, m = r["sabre_k20"], r["ms_k20"]
        delta = m["mean_absorbed_pct"] - s["mean_absorbed_pct"]
        print(f"{key:<24} {r['n_qubits']:>4}  "
              f"{s['mean_n_swaps']:>11.2f} {s['mean_pre_cnot']:>11.2f} {s['mean_post_cnot']:>12.2f} {s['mean_absorbed_pct']:>10.2f}%   "
              f"{m['mean_n_swaps']:>9.2f} {m['mean_pre_cnot']:>8.2f} {m['mean_post_cnot']:>9.2f} {m['mean_absorbed_pct']:>8.2f}%  "
              f"{delta:>+8.2f}% {r['p_ms_higher_absorbed_k20']:>9.1e}")

    # Aggregate
    deltas_k20 = [v["ms_k20"]["mean_absorbed_pct"] - v["sabre_k20"]["mean_absorbed_pct"]
                  for v in out.values()]
    swap_deltas = [v["ms_k20"]["mean_n_swaps"] - v["sabre_k20"]["mean_n_swaps"]
                   for v in out.values()]
    mk_deltas = [v["sabre_k20"]["mean_mk_opt"] - v["ms_k20"]["mean_mk_opt"]
                 for v in out.values()]
    # Pearson r between per-cell absorption uplift and makespan reduction --
    # the actual claim of the figure.
    upl = np.array(deltas_k20)
    mkr = np.array([100.0 * (v["sabre_k20"]["mean_mk_opt"] - v["ms_k20"]["mean_mk_opt"])
                    / v["sabre_k20"]["mean_mk_opt"] if v["sabre_k20"]["mean_mk_opt"] else 0.0
                    for v in out.values()])
    r_pearson = float(np.corrcoef(upl, mkr)[0, 1]) if len(upl) > 1 else float("nan")
    print(f"\n=== Aggregate (K=20, MS vs SABRE) ===")
    print(f"  Mean absorption rate delta (MS - SABRE): {np.mean(deltas_k20):+.2f}%  median {np.median(deltas_k20):+.2f}%")
    print(f"  Mean SWAP count delta (MS - SABRE):      {np.mean(swap_deltas):+.2f}   median {np.median(swap_deltas):+.2f}")
    print(f"  Mean makespan reduction (SABRE - MS):    {np.mean(mk_deltas):+.2f}  median {np.median(mk_deltas):+.2f}")
    print(f"  Pearson r(absorption uplift, makespan reduction): {r_pearson:.3f}")
    print(f"\nTotal: {time.time()-t0:.0f}s")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

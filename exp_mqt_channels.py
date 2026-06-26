"""Mechanism decomposition on MQT Bench: parallelization vs absorption.

SABRE-MS helps for two distinct reasons:
  (1) PARALLELIZATION / scheduling: the makespan-aware SWAP overlaps better on
      the ASAP critical path. This shows up even on the RAW routed circuit,
      with no gate cancellation.
  (2) ABSORPTION: SABRE-MS picks SWAPs the optimizer can cancel into adjacent
      gates. This shows up only AFTER the optimizer.

We measure both by computing each method's makespan twice -- raw (no optimizer)
and optimized -- on the full MQT circuit (1q gates included, since they affect
both the schedule and what the optimizer can cancel).

For each (benchmark, topology) cell, at K=20:
  prod_raw / prod_opt : production SABRE (best of K by SWAP count) makespan
  ms_raw   / ms_opt   : SABRE-MS (best of K by makespan) makespan
Channel split:
  gain_raw   = (prod_raw - ms_raw) / prod_raw     -> parallelization (lower bound)
  gain_opt   = (prod_opt - ms_opt) / prod_opt     -> parallelization + absorption
  absorption_channel = gain_opt - gain_raw        -> extra from the optimizer
NOTE: the two channels are NOT additive (the optimizer reshapes the schedule),
so gain_raw and (gain_opt - gain_raw) are existence lower-bounds, not an exact
partition.

Output: results/mqt_channels.json
"""
from __future__ import annotations
import json
import os
import time
import warnings
import numpy as np
from scipy import stats
warnings.filterwarnings("ignore", category=DeprecationWarning)

from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import SabreSwap

from mqt.bench import get_benchmark, BenchmarkLevel

from exp_real_full import (
    decompose_basis, get_layout, apply_layout_to_qc,
    route_with_sabre_ms, optimize_qc, asap_makespan_qc,
)
from exp_mqt_bench import strip_measurements
from lambda_select import select_lambda_ms
from topologies import get as get_topology


K = 20
BENCHMARKS = ["ae", "ghz", "graphstate", "qaoa", "qft", "qftentangled",
              "qpeexact", "vqe_su2", "wstate"]
# Cover small -> medium qubit counts so the channel split is reported across
# scale. (Mechanism does not need the 25q tier; 7-16q spans the range and keeps
# the SABRE-MS attempt-limited search from thrashing on large grids.)
TOPOLOGIES = [
    ("linear7", 7), ("ring8", 8), ("grid3x3", 9),
    ("ring12", 12), ("grid4x4", 16),
]


def sabre_swaps(routed):
    return sum(1 for i in routed.data if i.operation.name == "swap")


def prod_sabre(qc_phys, coupling, K):
    """Production SABRE: K lookahead trials, pick by fewest SWAPs (tiebreak
    raw makespan). Return (raw_makespan, opt_makespan) of the chosen trial."""
    best = None  # (n_swaps, raw_mk, routed)
    for s in range(K):
        pm = PassManager([SabreSwap(coupling_map=coupling, heuristic="lookahead",
                                    seed=s, trials=1)])
        routed = pm.run(qc_phys)
        ns = sabre_swaps(routed)
        raw = asap_makespan_qc(routed)
        if best is None or (ns, raw) < (best[0], best[1]):
            best = (ns, raw, routed)
    return best[1], asap_makespan_qc(optimize_qc(best[2]))


def sabre_ms_best(qc_phys, graph, coupling, lam, K):
    """SABRE-MS: K trials, pick by OPTIMIZED makespan (the deployment rule).
    Return (raw_makespan, opt_makespan) of the chosen trial."""
    best = None  # (opt_mk, raw_mk)
    for s in range(K):
        r = route_with_sabre_ms(qc_phys, graph, coupling, lam, seed=s, alim_mult=10)
        if r is None:
            continue
        opt = asap_makespan_qc(optimize_qc(r))
        raw = asap_makespan_qc(r)
        if best is None or opt < best[0]:
            best = (opt, raw)
    if best is None:
        return None
    return best[1], best[0]


def run_cell(benchmark, topology, n_qubits, K):
    try:
        qc = get_benchmark(benchmark=benchmark, circuit_size=n_qubits,
                           level=BenchmarkLevel.INDEP)
    except Exception as e:
        return {"error": f"get_benchmark: {e}"}
    qc = strip_measurements(qc)
    graph, n_q = get_topology(topology)
    if n_q != n_qubits:
        return {"error": f"{topology} has {n_q}q != {n_qubits}"}
    edges = list(graph.edges())
    coupling = CouplingMap([list(e) for e in edges] + [list(reversed(e)) for e in edges])

    qc_basis = decompose_basis(qc)
    perm = get_layout(qc_basis, coupling, seed=0)
    qc_phys = apply_layout_to_qc(qc_basis, perm)

    prod_raw, prod_opt = prod_sabre(qc_phys, coupling, K)
    sel = select_lambda_ms(qc_phys, graph, coupling, alim_mult=10)
    if sel is None:
        return {"error": "sabre_ms returned None for all seeds"}
    lam = sel["lam"]
    ms_raw, ms_opt = sel["ms_raw"], sel["ms_makespan"]

    gain_raw = 100.0 * (prod_raw - ms_raw) / prod_raw if prod_raw else 0.0
    gain_opt = 100.0 * (prod_opt - ms_opt) / prod_opt if prod_opt else 0.0
    return {
        "benchmark": benchmark, "topology": topology, "n_qubits": n_qubits,
        "lam": lam, "K": K,
        "prod_raw": prod_raw, "prod_opt": prod_opt,
        "ms_raw": ms_raw, "ms_opt": ms_opt,
        "gain_raw_pct": gain_raw,            # parallelization channel (lower bound)
        "gain_opt_pct": gain_opt,            # parallelization + absorption
        "absorption_channel_pct": gain_opt - gain_raw,
    }


def main():
    os.makedirs("results", exist_ok=True)
    out_path = "results/mqt_channels_perrun.json"
    out = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            out = json.load(f)
        print(f"Loaded {len(out)} cells")

    print(f"MQT mechanism channels | K={K} | raw vs optimized makespan")
    print(f"{'cell':<22}{'nq':>3} | {'gain_raw':>9}{'gain_opt':>9}{'absorb':>8}")
    print("-" * 60)
    t0 = time.time()
    for bm in BENCHMARKS:
        for topo, nq in TOPOLOGIES:
            key = f"{bm}/{topo}"
            if key in out and "error" not in out[key]:
                r = out[key]; tag = "cached"
            else:
                tc = time.time()
                r = run_cell(bm, topo, nq, K)
                out[key] = r
                with open(out_path, "w") as f:
                    json.dump(out, f, indent=2)
                if "error" in r:
                    print(f"  {key}: {r['error']}")
                    continue
                tag = f"{time.time()-tc:.0f}s"
            print(f"{key:<22}{r['n_qubits']:>3} | {r['gain_raw_pct']:>+8.2f}%"
                  f"{r['gain_opt_pct']:>+8.2f}%{r['absorption_channel_pct']:>+7.2f}%"
                  f"  ({tag})", flush=True)

    cells = [r for r in out.values() if "error" not in r and "gain_opt_pct" in r]
    if cells:
        gr = np.array([r["gain_raw_pct"] for r in cells])
        go = np.array([r["gain_opt_pct"] for r in cells])
        ab = go - gr
        # restrict the share to cells with a real optimized gain
        pos = go > 1.0
        print(f"\n=== Aggregate ({len(cells)} cells) ===")
        print(f"  Parallelization channel (raw, no optimizer): mean {gr.mean():+.2f}%  median {np.median(gr):+.2f}%")
        print(f"  Total gain (optimized):                      mean {go.mean():+.2f}%  median {np.median(go):+.2f}%")
        print(f"  Absorption channel (optimizer adds):         mean {ab.mean():+.2f} pp  median {np.median(ab):+.2f} pp")
        if pos.sum():
            share = float(np.mean(gr[pos] / go[pos]) * 100)
            print(f"  Parallelization share of total gain (cells with gain>1%): {share:.0f}%")
        # Is SABRE-MS better even on RAW makespan? (scheduling channel exists)
        try:
            p_raw = stats.wilcoxon([r["ms_raw"] for r in cells],
                                   [r["prod_raw"] for r in cells],
                                   alternative="less").pvalue
            print(f"  SABRE-MS < production on RAW makespan: Wilcoxon p={p_raw:.2e}")
        except Exception:
            pass
    print(f"\nTotal: {time.time()-t0:.0f}s | Saved {out_path}")


if __name__ == "__main__":
    main()

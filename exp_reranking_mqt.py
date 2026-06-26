"""Proxy-gap diagnostic (EXP 1) on MQT Bench circuits — external confirmation.

This is the SAME measurement as exp_reranking_phenomenon.py, but the circuits
come from the TU Munich MQT Bench suite instead of our synthetic families.
The point: show the SWAP-count vs makespan selection gap is not an artifact of
our hand-written circuits — it appears on a community-standard benchmark we did
not author.

For each (benchmark, topology) cell:
  1. get_benchmark(level=INDEP), strip measurements/barriers.
  2. decompose to the cx/u3 basis.
  3. Produce K independent compiler attempts (seed 0..K-1): each runs
     SabreLayout(seed) then SabreSwap-lookahead(seed). This is the same
     trial-pool diversity Qiskit uses internally — SabreSwap is seed-
     deterministic on a fixed input, so per-trial layout seeding is what
     generates the candidate spread, exactly as in the synthetic experiment
     (which draws a fresh SabreLayout seed per circuit instance).
  4. For each attempt, record (#SWAPs, optimized ASAP makespan).
  5. Rank the K attempts by fewest-SWAPs vs shortest-makespan; report
     - whether the two picks disagree,
     - makespan discarded by selecting on SWAP count.

Output: results/reranking_mqt.json
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
    optimize_qc, asap_makespan_qc,
)
from exp_mqt_bench import strip_measurements
from topologies import get as get_topology


K = 60

# Same MQT families and topologies as the MQT-Bench headline experiment, so
# the diagnostic is reported on exactly the distribution SABRE-MS is validated
# on. linear7/ring8/grid3x3/ring12/grid4x4 at their native qubit counts.
BENCHMARKS = ["ae", "ghz", "graphstate", "qaoa", "qft", "qftentangled",
              "qpeexact", "vqe_su2", "wstate"]
TOPOLOGIES = [
    ("linear7", 7),
    ("ring8",   8),
    ("grid3x3", 9),
    ("ring12",  12),
    ("grid4x4", 16),
]


def count_swaps(qc) -> int:
    return sum(1 for instr in qc.data if instr.operation.name == "swap")


def attempt_once(qc_basis, coupling, seed):
    """One independent compiler attempt: SabreLayout(seed) then
    SabreSwap-lookahead(seed). Returns (#swaps, optimized ASAP makespan)."""
    perm = get_layout(qc_basis, coupling, seed=seed)
    qc_phys = apply_layout_to_qc(qc_basis, perm)
    pm = PassManager([SabreSwap(coupling_map=coupling, heuristic="lookahead",
                                seed=seed, trials=1)])
    routed = pm.run(qc_phys)
    n_swaps = count_swaps(routed)
    mk_opt = asap_makespan_qc(optimize_qc(routed))
    return n_swaps, mk_opt


def wilcoxon_paired(a, b):
    """One-sided: is a < b? Returns p-value (1.0 if all ties)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if np.all(a == b):
        return 1.0
    try:
        return float(stats.wilcoxon(a, b, alternative="less").pvalue)
    except Exception:
        return 1.0


def run_cell(benchmark, topology, n_qubits, K):
    try:
        qc = get_benchmark(benchmark=benchmark, circuit_size=n_qubits,
                           level=BenchmarkLevel.INDEP)
    except Exception as e:
        return {"error": f"get_benchmark failed: {e}"}
    qc = strip_measurements(qc)

    graph, n_q = get_topology(topology)
    if n_q != n_qubits:
        return {"error": f"topology {topology} has {n_q} qubits, expected {n_qubits}"}
    edges = list(graph.edges())
    coupling = CouplingMap([list(e) for e in edges]
                           + [list(reversed(e)) for e in edges])

    qc_basis = decompose_basis(qc)

    swaps = np.zeros(K, dtype=int)
    mks = np.zeros(K, dtype=int)
    for s in range(K):
        ns, mk = attempt_once(qc_basis, coupling, seed=s)
        swaps[s] = ns
        mks[s] = mk

    idx_by_swap = int(np.argmin(swaps))
    idx_by_mks = int(np.argmin(mks))
    mks_bySwap = int(mks[idx_by_swap])
    mks_byMks = int(mks[idx_by_mks])
    return {
        "benchmark": benchmark, "topology": topology, "n_qubits": n_qubits,
        "K": K,
        "mks_by_swap": mks_bySwap,
        "mks_by_mks": mks_byMks,
        "swap_by_swap": int(swaps[idx_by_swap]),
        "swap_by_mks": int(swaps[idx_by_mks]),
        "misaligned": bool(idx_by_swap != idx_by_mks),
        "mks_loss_pct": 100.0 * (mks_bySwap - mks_byMks) / mks_bySwap
                        if mks_bySwap > 0 else 0.0,
        "extra_swaps_for_better_mks": int(swaps[idx_by_mks] - swaps[idx_by_swap]),
        "n_distinct_swap_counts": int(len(set(swaps.tolist()))),
        "n_distinct_mks": int(len(set(mks.tolist()))),
    }


def main():
    os.makedirs("results", exist_ok=True)
    out_path = "results/reranking_mqt.json"
    out = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            out = json.load(f)
        print(f"Loaded {len(out)} cells from cache")

    print(f"MQT proxy-gap diagnostic | K={K} routings/circuit, SABRE-lookahead")
    print(f"{'cell':<28} {'n':>3} | {'mks_bySwap':>10} {'mks_byMks':>10} "
          f"{'loss%':>7} {'mis':>4} {'dSwp':>5}")
    print("-" * 80)

    t0 = time.time()
    for benchmark in BENCHMARKS:
        for topology, n_qubits in TOPOLOGIES:
            key = f"{benchmark}/{topology}"
            if key in out and "error" not in out[key]:
                r = out[key]
                tag = "cached"
            else:
                tc = time.time()
                r = run_cell(benchmark, topology, n_qubits, K)
                if "error" in r:
                    print(f"  {key}: {r['error']}")
                    out[key] = r
                    with open(out_path, "w") as f:
                        json.dump(out, f, indent=2)
                    continue
                out[key] = r
                with open(out_path, "w") as f:
                    json.dump(out, f, indent=2)
                tag = f"{time.time()-tc:.0f}s"
            print(f"{key:<28} {r['n_qubits']:>3} | "
                  f"{r['mks_by_swap']:>10} {r['mks_by_mks']:>10} "
                  f"{r['mks_loss_pct']:>+6.2f}% {str(r['misaligned']):>4} "
                  f"{r['extra_swaps_for_better_mks']:>+5} ({tag})", flush=True)

    # Aggregate over cells with no error.
    cells = [r for r in out.values() if "error" not in r and "mks_loss_pct" in r]
    if cells:
        mis = 100.0 * sum(1 for r in cells if r["misaligned"]) / len(cells)
        loss = np.array([r["mks_loss_pct"] for r in cells])
        extra = np.array([r["extra_swaps_for_better_mks"] for r in cells])
        a = np.array([r["mks_by_swap"] for r in cells], float)
        b = np.array([r["mks_by_mks"] for r in cells], float)
        print(f"\n=== Aggregate ({len(cells)} cells) ===")
        print(f"  Circuits misaligned (SWAP pick != makespan pick): {mis:.1f}%")
        print(f"  Mean makespan discarded by SWAP selection:        {loss.mean():+.2f}%  "
              f"(median {np.median(loss):+.2f}%)")
        print(f"  Mean extra SWAPs accepted for the better makespan: {extra.mean():+.2f}")
        print(f"  Paired Wilcoxon (byMks < bySwap):                  p={wilcoxon_paired(b, a):.2e}")

    print(f"\nTotal: {time.time()-t0:.0f}s | Saved {out_path}")


if __name__ == "__main__":
    main()

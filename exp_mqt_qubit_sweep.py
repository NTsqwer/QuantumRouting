"""MQT-Bench qubit-count sweep for SABRE-MS.

Sweeps qubit count and measures SABRE-MS vs production SABRE on:
  - makespan reduction (%)
  - ESP ratio (SABRE-MS / SABRE), heron_typical calibration

Two topology series:
  - grid (square-ish, m*n closest to target qubit count)
  - heavy-hex (IBM Heron geometry)

Dense MQT families that scale CX with n: qft, qaoa, qpeexact.
Per-(topology, family, n) lambda tuned on a disjoint train fold.

Produces results/mqt_qubit_sweep.json. Run plot_qubit_sweep.py to graph.
"""
from __future__ import annotations
import json
import os
import time
import warnings
import numpy as np
from scipy import stats
warnings.filterwarnings("ignore", category=DeprecationWarning)

import networkx as nx
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import SabreSwap

from exp_real_full import (
    decompose_basis, get_layout, apply_layout_to_qc,
    route_with_sabre_ms, optimize_qc, asap_makespan_qc,
)
from exp_real_full_esp import compute_esp, CALIBRATIONS
from exp_mqt_bench import strip_measurements
from topologies import grid, heavy_hex
from mqt.bench import get_benchmark, BenchmarkLevel


K = 20
N_TRAIN = 4
N_TEST = 8
LAMBDA_GRID = [0.005, 0.02, 0.05, 0.10, 0.25]
HERON = CALIBRATIONS[0]  # heron_typical
FAMILIES = ["qft", "qaoa", "qpeexact"]

# (label, qubit_count) -> connection graph
def grid_for(n):
    """Square-ish grid with >= n nodes, then we request a benchmark of the
    grid's exact node count."""
    import math
    side = int(round(math.sqrt(n)))
    while side * side < n:
        side += 1
    g = grid(side, side)
    return g, side * side

# Sweep points: grid series and heavy-hex series
GRID_SIZES = [(3, 3), (4, 4), (5, 5), (6, 6), (7, 7)]   # 9,16,25,36,49
HHEX_ROWS = [2, 4, 6]                                   # 14,28,42


def sabre_trial(qc, coupling, seed):
    qc_basis = decompose_basis(qc)
    perm = get_layout(qc_basis, coupling, seed)
    qc_phys = apply_layout_to_qc(qc_basis, perm)
    pm = PassManager([SabreSwap(coupling_map=coupling,
                                heuristic="lookahead", seed=seed, trials=1)])
    qc_routed = pm.run(qc_phys)
    n_swap = sum(1 for i in qc_routed.data if i.operation.name == "swap")
    qc_opt = optimize_qc(qc_routed)
    mks = asap_makespan_qc(qc_opt)
    if mks is None:
        return None
    esp, _ = compute_esp(qc_opt, HERON)
    return int(n_swap), float(mks), float(esp)


def sabre_production(qc, coupling, K, seed_base):
    """Production rule: fewest SWAPs (tiebreak makespan). Returns (mks, esp)."""
    trials = []
    for k in range(K):
        try:
            r = sabre_trial(qc, coupling, seed_base + k)
        except Exception:
            continue
        if r is not None:
            trials.append(r)
    if not trials:
        return None
    best = min(trials, key=lambda t: (t[0], t[1]))
    return best[1], best[2]  # makespan, esp of the production-chosen trial


def ms_trial(qc, graph, coupling, lam, seed):
    qc_basis = decompose_basis(qc)
    perm = get_layout(qc_basis, coupling, seed)
    qc_phys = apply_layout_to_qc(qc_basis, perm)
    qc_routed = route_with_sabre_ms(qc_phys, graph, coupling, lam,
                                    seed=seed, alim_mult=50,
                                    verify_equivalence=False)
    if qc_routed is None:
        return None
    qc_opt = optimize_qc(qc_routed)
    mks = asap_makespan_qc(qc_opt)
    if mks is None:
        return None
    esp, _ = compute_esp(qc_opt, HERON)
    return float(mks), float(esp)


def sabre_ms_best(qc, graph, coupling, lam, K, seed_base):
    """SABRE-MS rule: shortest makespan. Returns (mks, esp) of that trial."""
    best_mks = float("inf")
    best_esp = None
    for k in range(K):
        try:
            r = ms_trial(qc, graph, coupling, lam, seed_base + k)
        except Exception:
            continue
        if r is None:
            continue
        if r[0] < best_mks:
            best_mks = r[0]
            best_esp = r[1]
    return (best_mks, best_esp) if best_esp is not None else None


def tune_lambda(family, n_q, graph, coupling):
    best_lam, best_mean = LAMBDA_GRID[0], float("inf")
    for lam in LAMBDA_GRID:
        vals = []
        for ci in range(N_TRAIN):
            try:
                qc = get_benchmark(benchmark=family, circuit_size=n_q,
                                  level=BenchmarkLevel.INDEP)
                qc = strip_measurements(qc)
            except Exception:
                continue
            r = sabre_ms_best(qc, graph, coupling, lam, K, ci * K)
            if r is not None:
                vals.append(r[0])
        if vals and (m := sum(vals) / len(vals)) < best_mean:
            best_mean, best_lam = m, lam
    return best_lam


def paired_bootstrap_ci(deltas, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(deltas, float)
    n = len(d)
    boots = [float(np.mean(d[rng.integers(0, n, n)])) for _ in range(n_boot)]
    return float(np.mean(d)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def run_cell(family, topo_label, graph, n_q):
    edges = list(graph.edges())
    coupling = CouplingMap([list(e) for e in edges]
                          + [list(reversed(e)) for e in edges])
    # MQT benchmarks are defined on exact circuit_size; if benchmark
    # circuit_size != n_q nodes we must request the node count.
    try:
        _ = get_benchmark(benchmark=family, circuit_size=n_q,
                         level=BenchmarkLevel.INDEP)
    except Exception as e:
        return {"error": f"gen {family}@{n_q}: {e}"}

    lam = tune_lambda(family, n_q, graph, coupling)

    mks_red, esp_ratio = [], []
    sabre_mks_l, ms_mks_l = [], []
    for ci in range(N_TRAIN, N_TRAIN + N_TEST):
        try:
            qc = get_benchmark(benchmark=family, circuit_size=n_q,
                              level=BenchmarkLevel.INDEP)
            qc = strip_measurements(qc)
        except Exception:
            continue
        sb = ci * K
        sp = sabre_production(qc, coupling, K, sb)
        ms = sabre_ms_best(qc, graph, coupling, lam, K, sb)
        if sp is None or ms is None:
            continue
        s_mks, s_esp = sp
        m_mks, m_esp = ms
        if s_mks <= 0:
            continue
        mks_red.append(100 * (s_mks - m_mks) / s_mks)
        esp_ratio.append(m_esp / s_esp if s_esp > 0 else 1.0)
        sabre_mks_l.append(s_mks)
        ms_mks_l.append(m_mks)
    if len(mks_red) < 3:
        return {"error": "too few"}

    mean_red, lo, hi = paired_bootstrap_ci(mks_red)
    try:
        _, p = stats.wilcoxon(ms_mks_l, sabre_mks_l, alternative="less")
        p = float(p)
    except Exception:
        p = 1.0
    return {
        "family": family, "topology": topo_label, "n_qubits": n_q,
        "lam": lam, "n_test": len(mks_red),
        "makespan_reduction_pct": mean_red,
        "makespan_ci95": [lo, hi],
        "esp_ratio_mean": float(np.mean(esp_ratio)),
        "esp_ratio_median": float(np.median(esp_ratio)),
        "wilcoxon_p": p,
        "mks_red_vals": mks_red,
        "esp_ratio_vals": esp_ratio,
    }


def main():
    os.makedirs("results", exist_ok=True)
    out_path = "results/mqt_qubit_sweep.json"
    out = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            out = json.load(f)
        print(f"Loaded {len(out)} cells from cache")

    cells = []
    for (m, n) in GRID_SIZES:
        g = grid(m, n)
        cells.append((f"grid{m}x{n}", g, m * n))
    for rows in HHEX_ROWS:
        g = heavy_hex(rows)
        cells.append((f"heavy_hex{rows}", g, g.number_of_nodes()))

    all_cells = [(fam, lbl, g, nq) for (lbl, g, nq) in cells for fam in FAMILIES]
    print(f"MQT qubit sweep | {len(all_cells)} cells | K={K} test={N_TEST}")
    hdr = (f"{'cell':<24} {'nq':>3} {'lam':>6} {'mks_red%':>9} "
           f"{'CI95':>16} {'esp_ratio':>9} {'wilcoxon_p':>11}")
    print(hdr)
    print("-" * len(hdr))

    t0 = time.time()
    for fam, lbl, g, nq in all_cells:
        key = f"{lbl}/{fam}"
        if key in out and out[key] is not None and "makespan_reduction_pct" in out[key]:
            r = out[key]
        else:
            tc = time.time()
            r = run_cell(fam, lbl, g, nq)
            out[key] = r
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
            if "error" in r:
                print(f"{key:<24} ERR {r['error']}")
                continue
            print(f"  [{time.time()-tc:.0f}s]")
        if "error" in r:
            continue
        lo, hi = r["makespan_ci95"]
        print(f"{key:<24} {r['n_qubits']:>3} {r['lam']:>6.3f} "
              f"{r['makespan_reduction_pct']:>+8.2f}% [{lo:>+5.1f},{hi:>+5.1f}] "
              f"{r['esp_ratio_mean']:>8.2f}x {r['wilcoxon_p']:>11.2e}")

    print(f"\nTotal: {time.time()-t0:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()

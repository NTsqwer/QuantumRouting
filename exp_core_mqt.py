"""Unified MQT CORE experiment -- the PRIMARY benchmark backbone.

MQT Bench is the community standard; this is the broad, externally-authored set
that backs the headline claims, analyzed by family and by topology. Same four
headline metrics as exp_core.py (audit / decomposition / reachability /
channels), all computed from ONE shared trial pool per config so they are
directly comparable.

MQT CORE = 11 families x 6 topologies (matched qubit counts):
  families: ae, ghz, graphstate, qnn, wstate, vqe_su2, qaoa, qft,
            qpeexact, qpeinexact, qftentangled
            (ae/ghz/qnn/graphstate are near-trivial -> honest negative controls)
  topologies: linear7(7), ring8(8), grid3x3(9), ring12(12), heavy_hex2(14),
              grid4x4(16)

lambda: per-RUN selection (lambda_select.select_lambda_ms) -- for each lambda in
the grid {0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25}, probe at K0=5 trials, keep the
shortest-makespan lambda, then route at full K=20. lambda=0 is in the grid, so the
fallback to SABRE is automatic. This is the rule the paper now describes.
K=20, K_big=200 (reachability), alim=50. Run with RP_SKIP_EQUIV=1.
Output: results/core_mqt_perrun.json
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
K_BIG = 200
ALIM = 50

FAMILIES = ["ae", "ghz", "graphstate", "qnn", "wstate", "vqe_su2",
            "qaoa", "qft", "qpeexact", "qpeinexact", "qftentangled"]
TOPOLOGIES = [("linear7", 7), ("ring8", 8), ("grid3x3", 9),
              ("ring12", 12), ("heavy_hex2", 14), ("grid4x4", 16)]


def sabre_trial(qc_phys, coupling, seed):
    pm = PassManager([SabreSwap(coupling_map=coupling, heuristic="lookahead",
                                seed=seed, trials=1)])
    routed = pm.run(qc_phys)
    ns = sum(1 for i in routed.data if i.operation.name == "swap")
    return ns, asap_makespan_qc(routed), asap_makespan_qc(optimize_qc(routed))


def run_cell(family, topology, n_qubits):
    try:
        qc = strip_measurements(get_benchmark(benchmark=family, circuit_size=n_qubits,
                                              level=BenchmarkLevel.INDEP))
    except Exception as e:
        return {"error": str(e)}
    graph, n_q = get_topology(topology)
    if n_q != n_qubits:
        return {"error": f"{topology}={n_q}q != {n_qubits}"}
    edges = list(graph.edges())
    coupling = CouplingMap([list(e) for e in edges] + [list(reversed(e)) for e in edges])

    qc_basis = decompose_basis(qc)
    # MQT circuit is a single instance; vary the SabreLayout seed to get a pool,
    # exactly as the diagnostic does. One mapped circuit per layout seed.
    A_opt = A_raw = B_opt = None
    pool_opt = []
    # Use layout seed 0 as the canonical mapping for prod/ms (matches mqt_bench),
    # and the K-seed SABRE pool for the selection/reachability comparison.
    perm0 = get_layout(qc_basis, coupling, seed=0)
    qc_phys = apply_layout_to_qc(qc_basis, perm0)

    pool = [sabre_trial(qc_phys, coupling, s) for s in range(K)]
    A_idx = min(range(K), key=lambda i: (pool[i][0], pool[i][2]))
    A_opt = pool[A_idx][2]; A_raw = pool[A_idx][1]
    B_opt = min(t[2] for t in pool)

    sel = select_lambda_ms(qc_phys, graph, coupling, alim_mult=ALIM)
    if sel is None:
        return {"error": "ms None all seeds"}
    lam = sel["lam"]
    C_opt = sel["ms_makespan"]; C_raw = sel["ms_raw"]

    reach_opts = [t[2] for t in pool] + [sabre_trial(qc_phys, coupling, s)[2]
                                         for s in range(K, K_BIG)]
    reach = min(reach_opts)

    return {
        "family": family, "topology": topology, "n_qubits": n_qubits, "lam": lam,
        "n_2q": sum(1 for i in qc_basis.data if i.operation.name == "cx"),
        "A_prod": A_opt, "B_sabre_mks": B_opt, "C_ms": C_opt,
        "A_prod_raw": A_raw, "C_ms_raw": C_raw, "reach_best": reach,
        "audit_gain_pct": 100 * (A_opt - C_opt) / A_opt if A_opt else 0.0,
        "rescore_pct": 100 * (A_opt - B_opt) / A_opt if A_opt else 0.0,
        "algorithm_pct": 100 * (B_opt - C_opt) / A_opt if A_opt else 0.0,
        "ms_raw_gain_pct": 100 * (A_raw - C_raw) / A_raw if A_raw else 0.0,
        "reach_residual_pct": 100 * (C_opt - reach) / reach if reach else 0.0,
        "routing": A_opt > 0 and pool[A_idx][0] > 0,
    }


def main():
    os.makedirs("results", exist_ok=True)
    out_path = "results/core_mqt_perrun.json"
    out = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            out = json.load(f)

    print(f"MQT CORE | K={K} K_big={K_BIG} alim={ALIM} | {len(FAMILIES)}x{len(TOPOLOGIES)}")
    print(f"{'cell':<26}{'2q':>4}{'audit':>7}{'rescore':>8}{'algo':>7}{'raw':>7}{'reachR':>8}")
    print("-" * 70)
    t0 = time.time()
    for fam in FAMILIES:
        for topo, nq in TOPOLOGIES:
            key = f"{fam}/{topo}"
            if key in out and "error" not in (out[key] or {"error": 1}):
                r = out[key]
            else:
                r = run_cell(fam, topo, nq)
                out[key] = r
                with open(out_path, "w") as f:
                    json.dump(out, f, indent=2)
                if "error" in r:
                    print(f"{key:<26} ERR {r['error']}"); continue
            print(f"{key:<26}{r['n_2q']:>4}{r['audit_gain_pct']:>+6.1f}%{r['rescore_pct']:>+7.1f}%"
                  f"{r['algorithm_pct']:>+6.1f}%{r['ms_raw_gain_pct']:>+6.1f}%"
                  f"{r['reach_residual_pct']:>+7.1f}%", flush=True)

    cells = [r for r in out.values() if r and "error" not in r]
    routing = [r for r in cells if r.get("routing")]
    if cells:
        def m(rs, k): return float(np.mean([r[k] for r in rs]))
        print(f"\n=== MQT CORE aggregate ===")
        print(f"  All {len(cells)} cells: audit {m(cells,'audit_gain_pct'):+.1f}%")
        print(f"  Routing-meaningful {len(routing)} cells:")
        print(f"    audit gain:    {m(routing,'audit_gain_pct'):+.1f}%")
        print(f"    rescore:       {m(routing,'rescore_pct'):+.1f}%   algorithm: {m(routing,'algorithm_pct'):+.1f}%")
        print(f"    parallel(raw): {m(routing,'ms_raw_gain_pct'):+.1f}%")
        nbeat = sum(1 for r in routing if r["reach_residual_pct"] < -0.5)
        print(f"    reach residual: {m(routing,'reach_residual_pct'):+.1f}%  (MS beats 200-pool on {nbeat}/{len(routing)})")
        print("  by family (audit gain, routing cells):")
        from collections import defaultdict
        fam = defaultdict(list)
        for r in routing: fam[r["family"]].append(r["audit_gain_pct"])
        for f in FAMILIES:
            if fam[f]: print(f"    {f:<14}{np.mean(fam[f]):+.1f}% (n={len(fam[f])})")
    print(f"\nTotal: {time.time()-t0:.0f}s | Saved {out_path}")


if __name__ == "__main__":
    main()

"""ESP on EXACTLY the core-set circuits, same selection as exp_core_mqt.py.

The point: makespan and ESP must describe the SAME circuits, chosen the SAME
way, so the two numbers are directly comparable. This reuses the core protocol
verbatim:
  - same families x topologies (the MQT primary core)
  - same fixed SabreLayout seed 0, same K=20
  - production = the SABRE trial chosen by (fewest SWAPs, then makespan)
  - SABRE-MS  = the SABRE-MS trial chosen by shortest makespan
It then computes, on those two chosen circuits, both the makespan reduction AND
the ESP ratio (Nishio-style, real ibm_marrakesh params, 1q gates counted in the
schedule). No new circuit set, no new selection rule.

Output: results/core_esp.json
"""
from __future__ import annotations
import json
import os
import time
import warnings
import numpy as np
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
from esp import esp_time_aware, ESPParams, DEVICE_PRESETS
from qgym.custom_types import Gate

# Same core as exp_core_mqt.py
K = 20
ALIM = 50
FAMILIES = ["ae", "ghz", "graphstate", "qnn", "wstate", "vqe_su2",
            "qaoa", "qft", "qpeexact", "qpeinexact", "qftentangled"]
TOPOLOGIES = [("linear7", 7), ("ring8", 8), ("grid3x3", 9),
              ("ring12", 12), ("heavy_hex2", 14), ("grid4x4", 16)]
ESP_FLOOR = 1e-3


def to_gates(qc):
    out = []
    for i in qc.data:
        qs = [qc.find_bit(q).index for q in i.qubits]
        nm = i.operation.name
        if len(qs) == 2:
            out.append(Gate("cnot" if nm == "cx" else nm, qs[0], qs[1]))
        elif len(qs) == 1:
            out.append(Gate(nm, qs[0], qs[0]))
    return out


def esp_presets(qc, nq):
    g = to_gates(qc)
    return {k: esp_time_aware(g, nq, ESPParams(**p))["esp"] for k, p in DEVICE_PRESETS.items()}


def n2q(qc):
    return sum(1 for i in qc.data if i.operation.num_qubits == 2)


def run_cell(family, topology, nq):
    try:
        qc = strip_measurements(get_benchmark(benchmark=family, circuit_size=nq,
                                              level=BenchmarkLevel.INDEP))
    except Exception as e:
        return {"error": str(e)}
    graph, n_q = get_topology(topology)
    edges = list(graph.edges())
    coupling = CouplingMap([list(e) for e in edges] + [list(reversed(e)) for e in edges])
    qc_basis = decompose_basis(qc)
    perm = get_layout(qc_basis, coupling, seed=0)     # same fixed layout as core
    qc_phys = apply_layout_to_qc(qc_basis, perm)

    # Production pool: keep the circuit chosen by (fewest SWAPs, then makespan).
    best = None
    for s in range(K):
        pm = PassManager([SabreSwap(coupling_map=coupling, heuristic="lookahead",
                                    seed=s, trials=1)])
        routed = pm.run(qc_phys)
        sw = sum(1 for i in routed.data if i.operation.name == "swap")
        o = optimize_qc(routed); mk = asap_makespan_qc(o)
        if best is None or (sw, mk) < (best[0], best[1]):
            best = (sw, mk, o)
    prod_qc, prod_mk = best[2], best[1]

    # SABRE-MS: per-run lambda selection, then the shortest-makespan trial at the
    # chosen lambda. SAME selector as the makespan run, so ESP and makespan describe
    # the same circuit chosen the same way.
    sel = select_lambda_ms(qc_phys, graph, coupling, alim_mult=ALIM)
    if sel is None:
        return {"error": "ms None"}
    lam = sel["lam"]
    ms_qc, bm = sel["ms_qc"], sel["ms_makespan"]

    pe = esp_presets(prod_qc, nq)
    me = esp_presets(ms_qc, nq)
    return {
        "family": family, "topology": topology, "n_qubits": nq, "lam": lam,
        "prod_makespan": prod_mk, "ms_makespan": bm,
        "makespan_red_pct": 100 * (prod_mk - bm) / prod_mk if prod_mk else 0.0,
        "prod_2q": n2q(prod_qc), "ms_2q": n2q(ms_qc), "n2q_delta": n2q(ms_qc) - n2q(prod_qc),
        "prod_esp": pe, "ms_esp": me,
        "esp_ratio": {k: (me[k] / pe[k] if pe[k] > 0 else None) for k in DEVICE_PRESETS},
        "esp_valid": {k: (max(pe[k], me[k]) >= ESP_FLOOR) for k in DEVICE_PRESETS},
        "routing": best[0] > 0,
    }


def main():
    os.makedirs("results", exist_ok=True)
    out_path = "results/core_esp_perrun.json"
    out = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            out = json.load(f)

    print(f"Core-set ESP | SAME cells/selection as makespan audit | K={K}")
    print(f"{'cell':<22}{'mks_red':>8}{'2q_d':>6}{'mar_esp':>9}{'valid':>7}")
    print("-" * 56)
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
                    continue
            mr = r["esp_ratio"]["marrakesh"]
            cell = f"{mr:.2f}x" if (mr and r["esp_valid"]["marrakesh"]) else "floor"
            print(f"{key:<22}{r['makespan_red_pct']:>+7.1f}%{r['n2q_delta']:>+6}"
                  f"{cell:>9}{str(r['esp_valid']['marrakesh']):>7}", flush=True)

    cells = [r for r in out.values() if r and "error" not in r]
    routing = [r for r in cells if r.get("routing")]
    print(f"\n=== Core-set ESP, routing-meaningful cells ({len(routing)}) ===")
    print(f"  Makespan reduction: mean {np.mean([r['makespan_red_pct'] for r in routing]):+.1f}%  "
          f"median {np.median([r['makespan_red_pct'] for r in routing]):+.1f}%")
    nadd = sum(1 for r in routing if r['n2q_delta'] > 0)
    print(f"  2q-gate delta: mean {np.mean([r['n2q_delta'] for r in routing]):+.1f} "
          f"(SABRE-MS adds gates on {nadd}/{len(routing)})")
    for p in DEVICE_PRESETS:
        valid = [r for r in routing if r["esp_valid"][p] and r["esp_ratio"][p]]
        if valid:
            ratios = [r["esp_ratio"][p] for r in valid]
            print(f"  ESP[{p:<10}] valid={len(valid):>2}/{len(routing)}  "
                  f"mean {np.mean(ratios):.2f}x  median {np.median(ratios):.2f}x  "
                  f"wins {sum(1 for x in ratios if x>1)}/{len(valid)}")
    # by qubit count (marrakesh)
    from collections import defaultdict
    print("  marrakesh ESP by qubit count:")
    byn = defaultdict(list)
    for r in routing:
        if r["esp_valid"]["marrakesh"] and r["esp_ratio"]["marrakesh"]:
            byn[r["n_qubits"]].append(r["esp_ratio"]["marrakesh"])
    for n in sorted(byn):
        print(f"    {n:>3}q: mean {np.mean(byn[n]):.2f}x (n={len(byn[n])})")
    print(f"\nTotal: {time.time()-t0:.0f}s | Saved {out_path}")


if __name__ == "__main__":
    main()

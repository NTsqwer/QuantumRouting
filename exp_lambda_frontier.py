"""Lambda frontier: the tunable makespan <-> SWAP-count trade-off, and how the
ESP-optimal lambda shifts with device coherence.

The story this backs (paper reframe): SABRE-MS does NOT replace the SWAP-count
objective. Its score is H_SABRE + lambda * (makespan term), so lambda is a knob:
  - lambda = 0  -> exactly production SABRE (the never-lose floor).
  - lambda > 0  -> trade more SWAPs for shorter makespan.
The right lambda depends on the device: on a decoherence-limited device (short
T2) the makespan term dominates the success probability, so the ESP-optimal
lambda is large; on a gate-error-limited device (good qubits, long T2) it is
small or zero. This sweep measures, per cell:
  - makespan and post-cancellation 2q-gate count at each lambda (the frontier),
  - ESP at each lambda under each device preset (marrakesh / heron_good / garnet),
  - the ESP-argmax lambda per preset,
all on the SAME core protocol as exp_core_esp.py (fixed layout seed 0, K=20,
SABRE-MS trial chosen by shortest makespan). lambda=0 reuses the production
SABRE pool so the floor is exact.

Output: results/lambda_frontier.json
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
from topologies import get as get_topology
from esp import esp_time_aware, ESPParams, DEVICE_PRESETS
from exp_core_esp import to_gates, n2q

K = 20
ALIM = 50
# lambda grid: 0 (=SABRE) up to a value that saturates the makespan term.
LAMBDA_GRID = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.5]

# Representative routing-heavy cells across the connectivity classes. We keep
# the dense families (the ones routing actually works on) on the four most
# routing-stressed core topologies, so the sweep stays cheap but covers the
# regimes where the trade-off is visible.
CELLS = [
    ("qft", "ring8", 8), ("qft", "ring12", 12), ("qft", "grid4x4", 16),
    ("qftentangled", "ring12", 12), ("qpeexact", "ring12", 12),
    ("qpeexact", "heavy_hex2", 14), ("ae", "ring12", 12),
    ("qaoa", "grid4x4", 16), ("qft", "linear7", 7), ("qft", "heavy_hex2", 14),
]


def esp_presets(qc, nq):
    g = to_gates(qc)
    return {k: esp_time_aware(g, nq, ESPParams(**p))["esp"] for k, p in DEVICE_PRESETS.items()}


def prod_pool(qc_phys, coupling):
    """Production SABRE: keep the (fewest SWAPs, then makespan) trial. This is
    the lambda=0 anchor, computed the same way as exp_core_esp.py."""
    best = None
    for s in range(K):
        pm = PassManager([SabreSwap(coupling_map=coupling, heuristic="lookahead",
                                    seed=s, trials=1)])
        routed = pm.run(qc_phys)
        sw = sum(1 for i in routed.data if i.operation.name == "swap")
        o = optimize_qc(routed)
        mk = asap_makespan_qc(o)
        if best is None or (sw, mk) < (best[0], best[1]):
            best = (sw, mk, o)
    return best[2], best[1]


def ms_pool(qc_phys, graph, coupling, lam, nq):
    """SABRE-MS at this lambda: keep the shortest-makespan trial over K seeds."""
    bm, best = float("inf"), None
    for s in range(K):
        r = route_with_sabre_ms(qc_phys, graph, coupling, lam, seed=s, alim_mult=ALIM)
        if r is None:
            continue
        o = optimize_qc(r)
        mk = asap_makespan_qc(o)
        if mk < bm:
            bm, best = mk, o
    return best, bm


def run_cell(family, topology, nq):
    qc = strip_measurements(get_benchmark(benchmark=family, circuit_size=nq,
                                          level=BenchmarkLevel.INDEP))
    graph, _ = get_topology(topology)
    edges = list(graph.edges())
    coupling = CouplingMap([list(e) for e in edges] + [list(reversed(e)) for e in edges])
    qc_basis = decompose_basis(qc)
    perm = get_layout(qc_basis, coupling, seed=0)
    qc_phys = apply_layout_to_qc(qc_basis, perm)

    # lambda = 0 anchor = production SABRE.
    prod_qc, prod_mk = prod_pool(qc_phys, coupling)
    prod_2q = n2q(prod_qc)
    prod_esp = esp_presets(prod_qc, nq)

    sweep = []
    for lam in LAMBDA_GRID:
        if lam == 0.0:
            qc_l, mk_l, q2_l, esp_l = prod_qc, prod_mk, prod_2q, prod_esp
        else:
            qc_l, mk_l = ms_pool(qc_phys, graph, coupling, lam, nq)
            if qc_l is None:
                continue
            q2_l = n2q(qc_l)
            esp_l = esp_presets(qc_l, nq)
        sweep.append({
            "lam": lam,
            "makespan": mk_l,
            "makespan_red_pct": 100 * (prod_mk - mk_l) / prod_mk if prod_mk else 0.0,
            "n2q": q2_l,
            "n2q_delta": q2_l - prod_2q,
            "esp": esp_l,
            "esp_ratio": {k: (esp_l[k] / prod_esp[k] if prod_esp[k] > 0 else None)
                          for k in DEVICE_PRESETS},
        })

    # ESP-argmax lambda per device preset.
    best_lam = {}
    for p in DEVICE_PRESETS:
        cand = [(row["esp"][p], row["lam"]) for row in sweep if row["esp"][p] > 0]
        best_lam[p] = max(cand)[1] if cand else None

    return {"family": family, "topology": topology, "n_qubits": nq,
            "prod_makespan": prod_mk, "prod_2q": prod_2q, "prod_esp": prod_esp,
            "sweep": sweep, "esp_argmax_lambda": best_lam}


def main():
    os.makedirs("results", exist_ok=True)
    out_path = "results/lambda_frontier.json"
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}

    print(f"Lambda frontier | K={K} | grid {LAMBDA_GRID}")
    print(f"{'cell':<24}{'argmax lam (mar/good/gar)':>28}")
    print("-" * 54)
    t0 = time.time()
    for fam, topo, nq in CELLS:
        key = f"{fam}/{topo}"
        if key in out and out[key] and "sweep" in out[key]:
            r = out[key]
        else:
            r = run_cell(fam, topo, nq)
            out[key] = r
            json.dump(out, open(out_path, "w"), indent=2)
        bl = r["esp_argmax_lambda"]
        print(f"{key:<24}{str(bl['marrakesh']):>8}{str(bl['heron_good']):>10}"
              f"{str(bl['garnet']):>10}", flush=True)

    cells = [r for r in out.values() if r and "sweep" in r]
    print(f"\n=== {len(cells)} cells ===")

    # 1. lambda=0 is exactly SABRE (ratio 1.00 by construction) -> never-lose floor.
    print("\n  Floor check (lambda=0 ESP ratio, should be 1.00x):")
    for p in DEVICE_PRESETS:
        r0 = [row["esp_ratio"][p] for r in cells for row in r["sweep"]
              if row["lam"] == 0.0 and row["esp_ratio"][p]]
        print(f"    {p:<11} mean {np.mean(r0):.3f}x")

    # 2. ESP-argmax lambda shifts with device coherence.
    print("\n  Median ESP-argmax lambda by device (decoherence pressure rises ->):")
    for p in ["heron_good", "marrakesh", "garnet"]:
        lams = [r["esp_argmax_lambda"][p] for r in cells
                if r["esp_argmax_lambda"][p] is not None]
        print(f"    {p:<11} median lambda* = {np.median(lams):.3f}  "
              f"(values {sorted(lams)})")

    # 3. Best-lambda ESP ratio per device (the deployable win if you tune lambda).
    print("\n  Best-achievable ESP ratio at the ESP-argmax lambda:")
    for p in ["heron_good", "marrakesh", "garnet"]:
        best = []
        for r in cells:
            lam = r["esp_argmax_lambda"][p]
            row = next((x for x in r["sweep"] if x["lam"] == lam), None)
            if row and row["esp_ratio"][p]:
                best.append(row["esp_ratio"][p])
        print(f"    {p:<11} median {np.median(best):.2f}x  mean {np.mean(best):.2f}x  "
              f"wins {sum(x >= 1 for x in best)}/{len(best)}")

    # 4. The frontier itself: makespan reduction vs extra 2q gates as lambda rises.
    print("\n  Frontier (mean over cells): lambda -> makespan_red%, extra 2q gates")
    for lam in LAMBDA_GRID:
        mk = [row["makespan_red_pct"] for r in cells for row in r["sweep"] if row["lam"] == lam]
        dq = [row["n2q_delta"] for r in cells for row in r["sweep"] if row["lam"] == lam]
        if mk:
            print(f"    lambda={lam:<6} makespan {np.mean(mk):+6.1f}%   extra 2q {np.mean(dq):+6.1f}")

    print(f"\nTotal {time.time()-t0:.0f}s | Saved {out_path}")


if __name__ == "__main__":
    main()

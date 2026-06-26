"""Re-evaluate baseline RL and schedule-aware (V19) RL at K=20 on
circuits with <= 35 two-qubit gates after basis decomposition.

For each (topology, family) combination on the topologies we have RL
models for (linear5, ring5, ring8), per circuit:
  - generate qc and decompose to (cx, u3)
  - if n_cx > 35: skip the circuit
  - otherwise: route K=20 times with the baseline RL agent and the V19
    RL agent, pick best-by-makespan
  - also run SABRE-lookahead K=20 with the production rule
    (pick by fewest SWAPs, tiebreak by lower makespan)

Output: results/rl_capped35.json
"""
from __future__ import annotations
import json
import os
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import SabreSwap
from sb3_contrib import MaskablePPO

from eval_4way_real import (
    pipeline_rl, make_env_for_condition,
)
from exp_real_full import (
    GENERATORS, decompose_basis, get_layout, apply_layout_to_qc,
    optimize_qc, asap_makespan_qc,
)
from topologies import get as get_topology


K = 20
N_CIRCUITS = 6
MAX_CX = 35


TOPOLOGIES = ["linear5", "ring5", "ring8", "linear7"]
FAMILIES   = ["qft", "qv", "vqe", "qaoa"]


def count_cx(qc):
    qc_b = decompose_basis(qc)
    return sum(1 for instr in qc_b.data if instr.operation.name == "cx")


def sabre_trial(qc, coupling, seed):
    qc_basis = decompose_basis(qc)
    perm = get_layout(qc_basis, coupling, seed)
    qc_phys = apply_layout_to_qc(qc_basis, perm)
    pm = PassManager([SabreSwap(coupling_map=coupling,
                                heuristic="lookahead",
                                seed=seed, trials=1)])
    qc_routed = pm.run(qc_phys)
    n_swap = sum(1 for instr in qc_routed.data
                 if instr.operation.name == "swap")
    mks = asap_makespan_qc(optimize_qc(qc_routed))
    if mks is None:
        return None
    return int(n_swap), float(mks)


def sabre_production_k20(qc, coupling, K, seed_base):
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
    return min(trials, key=lambda t: (t[0], t[1]))[1]


def rl_best_of_k(qc, graph, coupling, model, env, n_qubits, K, seed_base):
    best = float("inf")
    for k in range(K):
        m = pipeline_rl(qc, graph, coupling, model, env, n_qubits,
                        seed_base + k, deterministic=False)
        if m is not None and m < best:
            best = float(m)
    return best if best != float("inf") else None


def run_topology(topology):
    graph, n_q = get_topology(topology)
    edges = list(graph.edges())
    coupling = CouplingMap([list(e) for e in edges]
                          + [list(reversed(e)) for e in edges])

    base_model_path = f"models/baseline_{topology}_TWOPHASE.zip"
    v19_model_path  = f"models/v19_{topology}_TWOPHASE.zip"
    if not (os.path.exists(base_model_path) and os.path.exists(v19_model_path)):
        print(f"  [{topology}] missing model file(s); skipping")
        return None

    m_baseline = MaskablePPO.load(base_model_path)
    m_v19      = MaskablePPO.load(v19_model_path)

    env_baseline = make_env_for_condition("baseline", graph,
                                          max_length=80, obs_reach=80)
    env_v19      = make_env_for_condition("v19", graph,
                                          max_length=80, obs_reach=80)

    out = {"topology": topology, "n_qubits": n_q, "K": K,
           "max_cx_cap": MAX_CX,
           "by_family": {}}

    print(f"\n=== {topology} ===")
    print(f"{'family':<8} {'n_kept':>7} {'mean_cx':>7} | "
          f"{'SABRE':>7} {'base_RL':>8} {'V19_RL':>8} | "
          f"{'V19 gain vs base':>16}")

    for fam in FAMILIES:
        gen = GENERATORS[fam]
        per_circuit = []
        for ci in range(N_CIRCUITS):
            try:
                qc = gen(n_q, seed=1234 + ci)
            except Exception:
                continue
            n_cx = count_cx(qc)
            if n_cx > MAX_CX:
                continue
            seed_base = ci * K
            s = sabre_production_k20(qc, coupling, K, seed_base)
            b = rl_best_of_k(qc, graph, coupling, m_baseline, env_baseline,
                             n_q, K, seed_base)
            v = rl_best_of_k(qc, graph, coupling, m_v19, env_v19,
                             n_q, K, seed_base)
            if any(x is None for x in (s, b, v)):
                continue
            per_circuit.append({"ci": ci, "n_cx": n_cx,
                                "sabre": s, "baseline_rl": b, "v19_rl": v})
        if not per_circuit:
            print(f"{fam:<8} {'no_data':>7}")
            continue
        n = len(per_circuit)
        mean_cx = sum(p["n_cx"] for p in per_circuit) / n
        mean_s = sum(p["sabre"]       for p in per_circuit) / n
        mean_b = sum(p["baseline_rl"] for p in per_circuit) / n
        mean_v = sum(p["v19_rl"]      for p in per_circuit) / n
        v19_gain = 100 * (mean_b - mean_v) / mean_b if mean_b > 0 else 0
        out["by_family"][fam] = {
            "n_kept": n, "mean_n_cx": mean_cx,
            "sabre_mean": mean_s, "baseline_rl_mean": mean_b,
            "v19_rl_mean": mean_v,
            "v19_vs_baseline_pct": v19_gain,
            "per_circuit": per_circuit,
        }
        print(f"{fam:<8} {n:>7} {mean_cx:>7.1f} | "
              f"{mean_s:>7.1f} {mean_b:>8.1f} {mean_v:>8.1f} | "
              f"{v19_gain:>+15.2f}%")
    return out


def main():
    os.makedirs("results", exist_ok=True)
    t0 = time.time()
    all_out = {}
    for topology in TOPOLOGIES:
        tc = time.time()
        r = run_topology(topology)
        if r is not None:
            all_out[topology] = r
        with open("results/rl_capped35.json", "w") as f:
            json.dump(all_out, f, indent=2)
        print(f"  -> saved  ({time.time()-tc:.0f}s)")
    print(f"\nTotal: {time.time()-t0:.0f}s")
    print(f"Saved results/rl_capped35.json")


if __name__ == "__main__":
    main()

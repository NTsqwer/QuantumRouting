"""ESP on real Qiskit benchmark circuits, full pipeline (1q gates preserved).

Same setup as exp_real_full.py (same circuits, same SabreLayout, same K=20,
same per-family lambda, same Qiskit optimizer downstream), but the metric
is Expected Success Probability under three IBM noise calibrations instead
of just makespan.

ESP formula:
    ESP = (1 - eps_cnot)^N_CNOT * (1 - eps_1q)^N_1q * prod_q exp(-t_idle(q)/T2)

with t_idle(q) = makespan - busy_q (cycles q is in the schedule but not
executing). The gate-error term applies to the post-optimization gate count,
so SWAPs absorbed by the optimizer don't contribute their full 3-CNOT cost.

Three calibrations (representative of IBM superconducting 2024-2025):
  heron_typical: cnot=0.7%,  1q=0.03%, T2 = 100us (1000 cycles)
  eagle_typical: cnot=1.5%,  1q=0.05%, T2 =  50us ( 500 cycles)  (noisier)
  heron_best:    cnot=0.3%,  1q=0.02%, T2 = 200us (2000 cycles)  (best case)

Output: results/real_full_esp.json
"""
from __future__ import annotations

import json
import math
import os
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore", category=DeprecationWarning)

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from exp_real_full import (
    GENERATORS, lambda_for, decompose_basis, get_layout,
    apply_layout_to_qc, route_with_sabre, route_with_sabre_ms,
    optimize_qc, K,
)
from topologies import get as get_topology


CNOT_DUR = 2
SWAP_DUR = 6
ONE_Q_DUR = 1


@dataclass
class ESPParams:
    cnot_error: float
    one_q_error: float
    T2_cycles: float
    name: str


CALIBRATIONS = [
    ESPParams(name="heron_typical",
              cnot_error=0.007, one_q_error=0.0003, T2_cycles=1000.0),
    ESPParams(name="eagle_typical",
              cnot_error=0.015, one_q_error=0.0005, T2_cycles=500.0),
    ESPParams(name="heron_best",
              cnot_error=0.003, one_q_error=0.0002, T2_cycles=2000.0),
]


def asap_schedule_qc(qc: QuantumCircuit):
    """Schedule a QuantumCircuit and return (makespan, busy_per_qubit) arrays.
    Treats 1q gates as 1 cycle, cx as 2, swap as 6.
    `busy[q]` is the total cycles qubit q is executing any gate.
    """
    n_q = qc.num_qubits
    free = np.zeros(n_q, dtype=int)
    busy = np.zeros(n_q, dtype=int)
    for instr in qc.data:
        op = instr.operation
        qs = [qc.find_bit(q).index for q in instr.qubits]
        if len(qs) == 1:
            free[qs[0]] += ONE_Q_DUR
            busy[qs[0]] += ONE_Q_DUR
        elif len(qs) == 2:
            if op.name == "cx":
                dur = CNOT_DUR
            elif op.name == "swap":
                dur = SWAP_DUR
            else:
                dur = CNOT_DUR
            start = max(int(free[qs[0]]), int(free[qs[1]]))
            free[qs[0]] = start + dur
            free[qs[1]] = start + dur
            busy[qs[0]] += dur
            busy[qs[1]] += dur
    makespan = int(free.max()) if len(free) else 0
    return makespan, busy


def count_gates(qc: QuantumCircuit):
    """Count CNOTs, SWAPs, and 1q gates in a QuantumCircuit."""
    n_cnot = n_swap = n_1q = 0
    for instr in qc.data:
        op = instr.operation
        qs = [qc.find_bit(q).index for q in instr.qubits]
        if len(qs) == 1:
            n_1q += 1
        elif len(qs) == 2:
            if op.name == "cx":
                n_cnot += 1
            elif op.name == "swap":
                n_swap += 1
            else:
                n_cnot += 1  # treat other 2q as cnot for accounting
    return n_cnot, n_swap, n_1q


def compute_esp(qc: QuantumCircuit, params: ESPParams) -> tuple[float, dict]:
    """Compute ESP for a QuantumCircuit under given calibration.
    Returns (esp, breakdown_dict)."""
    makespan, busy = asap_schedule_qc(qc)
    n_cnot, n_swap, n_1q = count_gates(qc)

    # Gate-error term:
    # CNOT error per gate, SWAP = 3 CNOTs of error (if any SWAPs survived
    # the optimizer; usually 0 after our pipeline)
    log_gate = (n_cnot * math.log(1.0 - params.cnot_error)
                + n_swap * 3.0 * math.log(1.0 - params.cnot_error)
                + n_1q * math.log(1.0 - params.one_q_error))
    # Decoherence on idle time
    log_decoh = 0.0
    for q in range(qc.num_qubits):
        idle = max(0, makespan - int(busy[q]))
        log_decoh += -idle / params.T2_cycles
    log_total = log_gate + log_decoh
    esp = math.exp(log_total)
    return esp, {
        "makespan": makespan,
        "n_cnot": n_cnot, "n_swap": n_swap, "n_1q": n_1q,
        "log_esp": log_total,
        "log_gate_part": log_gate,
        "log_decoh_part": log_decoh,
    }


def pipeline_sabre(qc_orig, graph, coupling, seed) -> QuantumCircuit | None:
    """Vanilla SABRE routing through full pipeline. Returns final QC."""
    qc_basis = decompose_basis(qc_orig)
    perm = get_layout(qc_basis, coupling, seed)
    qc_phys = apply_layout_to_qc(qc_basis, perm)
    qc_routed = route_with_sabre(qc_phys, coupling, seed)
    return optimize_qc(qc_routed)


def pipeline_sabre_ms(qc_orig, graph, coupling, lam, seed,
                     alim_mult=10) -> QuantumCircuit | None:
    """SABRE-MS routing through full pipeline."""
    qc_basis = decompose_basis(qc_orig)
    perm = get_layout(qc_basis, coupling, seed)
    qc_phys = apply_layout_to_qc(qc_basis, perm)
    qc_routed = route_with_sabre_ms(qc_phys, graph, coupling, lam, seed, alim_mult)
    if qc_routed is None:
        return None
    return optimize_qc(qc_routed)


def best_of_k_esp(fn: Callable[[int], QuantumCircuit | None], k: int,
                  params: ESPParams):
    """Run fn(seed) K times, pick the highest-ESP routing under `params`.
    Returns (esp, breakdown) of the best one, or (None, None)."""
    best_esp = -float("inf")
    best_break = None
    for s in range(k):
        try:
            qc = fn(s)
        except Exception:
            continue
        if qc is None:
            continue
        try:
            esp, br = compute_esp(qc, params)
        except Exception:
            continue
        if esp > best_esp:
            best_esp = esp
            best_break = br
    return (best_esp, best_break) if best_break is not None else (None, None)


def wilcoxon(a, b, alt="greater"):
    """Test: is `a` significantly greater than `b`? (For ESP we want MS > SABRE.)"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.all(a == b):
        return 1.0
    try:
        _, p = stats.wilcoxon(a, b, alternative=alt)
        return float(p)
    except Exception:
        return 1.0


def run_cell(topology, family, n_circuits, alim_mult=10):
    graph, n_qubits = get_topology(topology)
    edges = list(graph.edges())
    coupling = CouplingMap([list(e) for e in edges] + [list(reversed(e)) for e in edges])
    gen = GENERATORS[family]
    lam = lambda_for(family, topology)

    # Per calibration, collect best-ESP routings under SABRE and SABRE-MS
    results = {cal.name: {"sabre_esps": [], "ms_esps": [],
                          "sabre_brs": [], "ms_brs": []}
              for cal in CALIBRATIONS}

    for ci in range(n_circuits):
        try:
            qc = gen(n_qubits, seed=1234 + ci)
        except Exception:
            continue
        for cal in CALIBRATIONS:
            s_esp, s_br = best_of_k_esp(
                lambda s: pipeline_sabre(qc, graph, coupling, seed=s + ci * K),
                K, cal,
            )
            m_esp, m_br = best_of_k_esp(
                lambda s: pipeline_sabre_ms(qc, graph, coupling, lam,
                                            seed=s + ci * K, alim_mult=alim_mult),
                K, cal,
            )
            if s_esp is None or m_esp is None:
                continue
            results[cal.name]["sabre_esps"].append(s_esp)
            results[cal.name]["ms_esps"].append(m_esp)
            results[cal.name]["sabre_brs"].append(s_br)
            results[cal.name]["ms_brs"].append(m_br)

    if not results[CALIBRATIONS[0].name]["sabre_esps"]:
        return None

    out = {
        "topology": topology, "family": family, "lambda": lam,
        "n_qubits": n_qubits,
        "n_circuits": len(results[CALIBRATIONS[0].name]["sabre_esps"]),
        "by_calibration": {},
    }
    for cal in CALIBRATIONS:
        r = results[cal.name]
        a = np.array(r["ms_esps"], float)
        b = np.array(r["sabre_esps"], float)
        # Multiplicative gain (ratio)
        ratio = a.mean() / b.mean() if b.mean() > 0 else float("inf")
        # Percentage uplift
        pct = 100.0 * (a.mean() - b.mean()) / b.mean() if b.mean() > 0 else 0.0
        out["by_calibration"][cal.name] = {
            "sabre_mean_esp": float(b.mean()),
            "ms_mean_esp": float(a.mean()),
            "esp_ratio": float(ratio),
            "esp_uplift_pct": float(pct),
            "p_ms_greater": wilcoxon(a, b, "greater"),
            "sabre_mean_cnot": float(np.mean([br["n_cnot"] for br in r["sabre_brs"]])),
            "ms_mean_cnot": float(np.mean([br["n_cnot"] for br in r["ms_brs"]])),
            "sabre_mean_makespan": float(np.mean([br["makespan"] for br in r["sabre_brs"]])),
            "ms_mean_makespan": float(np.mean([br["makespan"] for br in r["ms_brs"]])),
        }
    return out


# Same cells as exp_real_full.py for direct comparison
CELLS = [
    ("linear7",   "qft", 12, 10),
    ("ring8",     "qft", 12, 10),
    ("ring12",    "qft", 10, 10),
    ("grid3x3",   "qft", 12, 10),
    ("grid4x4",   "qft", 8, 10),
    ("heavy_hex2","qft", 8, 10),
    ("linear7",   "qv",  12, 10),
    ("linear9",   "qv",  12, 10),
    ("ring8",     "qv",  12, 10),
    ("ring12",    "qv",  10, 10),
    ("grid3x3",   "qv",  12, 10),
    ("grid4x4",   "qv",  8, 10),
    ("heavy_hex2","qv",  8, 10),
    ("ring8",     "vqe", 12, 10),
    ("grid3x3",   "vqe", 12, 10),
    ("grid4x4",   "vqe", 8, 10),
    ("heavy_hex2","vqe", 8, 10),
    ("linear7",   "qaoa", 15, 10),
    ("linear9",   "qaoa", 15, 10),
    ("ring8",     "qaoa", 15, 10),
    ("ring12",    "qaoa", 12, 10),
    ("grid3x3",   "qaoa", 12, 10),
    ("grid4x4",   "qaoa", 8, 10),
    ("heavy_hex2","qaoa", 8, 10),
]


def main():
    os.makedirs("results", exist_ok=True)
    out_path = "results/real_full_esp.json"
    out = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            out = json.load(f)
        print(f"Loaded {len(out)} cells")

    t0 = time.time()
    # Primary calibration printed
    cal = "heron_typical"
    print(f"Primary calibration: {cal} (CNOT=0.7%, 1q=0.03%, T2=100us)\n")
    print(f"{'Cell':<24} {'lam':>6} {'n_q':>4} {'n':>3}  "
          f"{'SABRE ESP':>10} {'MS ESP':>9} {'ratio':>8} {'uplift':>8} {'p':>9}  "
          f"{'SABRE mk':>9} {'MS mk':>8} {'SABRE cx':>9} {'MS cx':>8}")
    print("-" * 145)

    for topo, fam, n, alim in CELLS:
        key = f"{topo}/{fam}"
        if key in out:
            r = out[key]
        else:
            print(f"  Running {key}...", flush=True)
            tc = time.time()
            r = run_cell(topo, fam, n, alim_mult=alim)
            print(f"    ...{time.time()-tc:.0f}s, got {0 if r is None else r['n_circuits']} circuits",
                  flush=True)
            if r is None:
                continue
            out[key] = r
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
        bc = r["by_calibration"][cal]
        marker = "  *" if bc["esp_uplift_pct"] > 1 and bc["p_ms_greater"] < 0.05 else ""
        print(f"{key:<24} {r['lambda']:>6.3f} {r['n_qubits']:>4} {r['n_circuits']:>3}  "
              f"{bc['sabre_mean_esp']:>10.4f} {bc['ms_mean_esp']:>9.4f} "
              f"{bc['esp_ratio']:>7.3f}x {bc['esp_uplift_pct']:>+7.2f}% {bc['p_ms_greater']:>9.1e}{marker}  "
              f"{bc['sabre_mean_makespan']:>9.1f} {bc['ms_mean_makespan']:>8.1f} "
              f"{bc['sabre_mean_cnot']:>9.1f} {bc['ms_mean_cnot']:>8.1f}")

    if out:
        print("\n=== Aggregate by calibration ===")
        for cal_obj in CALIBRATIONS:
            cal = cal_obj.name
            uplifts = [v["by_calibration"][cal]["esp_uplift_pct"]
                      for v in out.values()
                      if cal in v["by_calibration"]]
            sig = sum(1 for v in out.values()
                     if cal in v["by_calibration"]
                     and v["by_calibration"][cal]["esp_uplift_pct"] > 1
                     and v["by_calibration"][cal]["p_ms_greater"] < 0.05)
            if uplifts:
                print(f"  {cal:<16}: mean ESP uplift {np.mean(uplifts):+7.2f}%   "
                      f"median {np.median(uplifts):+7.2f}%   "
                      f"sig: {sig}/{len(uplifts)}")

        print("\n=== Aggregate by family (heron_typical) ===")
        for fam_name in ["qft", "qv", "vqe", "qaoa"]:
            results = [v for v in out.values() if v["family"] == fam_name]
            if not results: continue
            uplifts = [v["by_calibration"]["heron_typical"]["esp_uplift_pct"]
                      for v in results]
            sig = sum(1 for v in results
                     if v["by_calibration"]["heron_typical"]["esp_uplift_pct"] > 1
                     and v["by_calibration"]["heron_typical"]["p_ms_greater"] < 0.05)
            print(f"  {fam_name:<8}: mean {np.mean(uplifts):+7.2f}%   "
                  f"sig {sig}/{len(results)}")
    print(f"\nTotal: {time.time()-t0:.0f}s")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

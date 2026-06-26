"""Evaluate V19 trained on real benchmark patterns vs SABRE-MS and SABRE-lookahead.

For a given topology + agent file:
  - Generate test circuits from each family (QFT, QV, VQE, QAOA) using the
    Qiskit library generators (same as exp_real_full.py).
  - Decompose to (cx, u3) basis, apply SabreLayout for initial mapping.
  - Extract the 2q-pair sequence, feed it to:
       * SABRE-lookahead K=20 (baseline)
       * SABRE-MS K=20 (paper headline)
       * V19 K=20 stochastic best-of-makespan (the new candidate)
  - Then for each method, splice 1q gates back in and run the Qiskit optimizer.
  - Compare scheduled makespan (cnot=2, swap=6, 1q=1).

Output: results/v19_real_eval_<topology>.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore", category=DeprecationWarning)

from qiskit.transpiler import CouplingMap
from sb3_contrib import MaskablePPO

from qgym.envs.routing import Routing
from qgym.generators.interaction import BasicInteractionGenerator

from action_mask_wrapper import ActionMaskWrapper
from exp_real_full import (
    GENERATORS, lambda_for, decompose_basis, get_layout, apply_layout_to_qc,
    route_with_sabre, route_with_sabre_ms, optimize_qc, asap_makespan_qc, K,
)
from obs_wrapper_v19 import V19ObservationWrapper
from pipeline import routing_to_circuit_prefix
from topologies import get as get_topology


GD = {"cnot": 2, "swap": 6}


def _unwrap_state(env):
    e = env
    while True:
        if hasattr(e, "_state") and not callable(getattr(type(e), "_state", None)):
            return e._state
        if hasattr(e, "env"):
            e = e.env
        else:
            raise RuntimeError("could not find _state")


def make_v19_env(graph, max_length=200, obs_reach=80):
    # Env buffer can be large (max_length=200) for long real circuits.
    # The policy's lookahead window obs_reach MUST match training (80 here).
    env = Routing(connection_graph=graph,
                  interaction_generator=BasicInteractionGenerator(max_length=max_length, seed=0),
                  max_observation_reach=obs_reach,
                  observe_legal_surpasses=True,
                  observe_connection_graph=True)
    env = V19ObservationWrapper(env)
    env = ActionMaskWrapper(env)
    return env


def v19_route(model, env, cp, deterministic, max_steps=2000):
    """Run V19 on the 2q-pair sequence cp, return the routed gate list
    (qgym Gate objects) or None on failure."""
    from pipeline import routing_to_circuit
    obs, _ = env.reset(options={"interaction_circuit": cp})
    state = _unwrap_state(env)
    for _ in range(max_steps):
        mask = env.action_masks()
        a, _ = model.predict(obs, deterministic=deterministic, action_masks=mask)
        obs, _r, term, trunc, _ = env.step(int(a))
        if state.is_done() or term or trunc:
            break
    return routing_to_circuit(state) if state.is_done() else None


def splice_1q_into_route(qc_phys, routed_2q_gates, n_qubits,
                         verify_equivalence: bool = True):
    """Walk through qc_phys (post-layout, on physical labels). When a 2q gate
    appears, consume from routed_2q_gates: emit any preceding SWAPs, then the
    CNOT. 1q gates pass through unchanged but get remapped through the running
    SWAPs.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import CXGate, SwapGate
    n = n_qubits

    layout_cur = np.arange(n, dtype=int)   # layout_cur[logical] = current physical
    inv_layout = np.arange(n, dtype=int)   # inv_layout[physical] = current logical

    out_qc = QuantumCircuit(n)
    ridx = 0
    for instr in qc_phys.data:
        op = instr.operation
        qs = [qc_phys.find_bit(q).index for q in instr.qubits]
        if len(qs) == 1:
            q = qs[0]
            phys_now = int(layout_cur[q])
            out_qc.append(op, [out_qc.qubits[phys_now]])
        elif len(qs) == 2:
            while ridx < len(routed_2q_gates) and routed_2q_gates[ridx].name == "swap":
                sw = routed_2q_gates[ridx]
                p1, p2 = sw.q1, sw.q2
                l1, l2 = int(inv_layout[p1]), int(inv_layout[p2])
                layout_cur[l1], layout_cur[l2] = p2, p1
                inv_layout[p1], inv_layout[p2] = l2, l1
                out_qc.append(SwapGate(), [out_qc.qubits[p1], out_qc.qubits[p2]])
                ridx += 1
            if ridx < len(routed_2q_gates):
                cn = routed_2q_gates[ridx]
                out_qc.append(CXGate(), [out_qc.qubits[cn.q1], out_qc.qubits[cn.q2]])
                ridx += 1
        else:
            out_qc.append(op, [out_qc.qubits[int(layout_cur[q])] for q in qs])
    while ridx < len(routed_2q_gates):
        sw = routed_2q_gates[ridx]
        if sw.name == "swap":
            out_qc.append(SwapGate(), [out_qc.qubits[sw.q1], out_qc.qubits[sw.q2]])
        elif sw.name == "cnot":
            out_qc.append(CXGate(), [out_qc.qubits[sw.q1], out_qc.qubits[sw.q2]])
        ridx += 1
    if verify_equivalence:
        from exp_real_full import _assert_equiv
        _assert_equiv(qc_phys, out_qc, context="rl_splice")
    return out_qc


def pipeline_v19(qc_orig, graph, coupling, model, env, n_qubits, seed,
                deterministic=False):
    """End-to-end pipeline routing 2q gates via V19, 1q gates passed through.
    Returns makespan after optimization."""
    qc_basis = decompose_basis(qc_orig)
    perm = get_layout(qc_basis, coupling, seed)
    qc_phys = apply_layout_to_qc(qc_basis, perm)

    # Extract 2q pairs from qc_phys
    pairs = []
    for instr in qc_phys.data:
        op = instr.operation
        qs = [qc_phys.find_bit(q).index for q in instr.qubits]
        if len(qs) == 2:
            pairs.append((int(qs[0]), int(qs[1])))
    if not pairs:
        # No 2q gates - just optimize and schedule
        return asap_makespan_qc(optimize_qc(qc_phys))
    cp = np.array(pairs, dtype=int)

    # Route via V19
    routed_2q = v19_route(model, env, cp, deterministic=deterministic)
    if routed_2q is None:
        return None
    # Splice 1q gates back in
    qc_routed = splice_1q_into_route(qc_phys, routed_2q, n_qubits)
    qc_final = optimize_qc(qc_routed)
    return asap_makespan_qc(qc_final)


def best_of_k(fn, k):
    best = float("inf")
    for s in range(k):
        m = fn(s)
        if m is None:
            continue
        if m < best:
            best = m
    return best if best != float("inf") else None


def wilcoxon(a, b, alt="less"):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.all(a == b):
        return 1.0
    try:
        _, p = stats.wilcoxon(a, b, alternative=alt)
        return float(p)
    except Exception:
        return 1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--topology", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--n-circuits", type=int, default=12)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--obs-reach", type=int, default=80,
                  help="V19 max_observation_reach (MUST match training value)")
    args = p.parse_args()

    graph, n_qubits = get_topology(args.topology)
    edges = list(graph.edges())
    coupling = CouplingMap([list(e) for e in edges] + [list(reversed(e)) for e in edges])
    env_v19 = make_v19_env(graph, max_length=200, obs_reach=args.obs_reach)
    print(f"Loading {args.model}")
    model = MaskablePPO.load(args.model, device="cpu")

    out_path = f"results/v19_real_eval_{args.topology}.json"
    os.makedirs("results", exist_ok=True)
    out = {}

    t0 = time.time()
    print(f"{'Family':<12} {'lam':>6} {'n_q':>4} {'n':>3}  "
          f"{'SABRE':>9} {'MS':>9} {'V19 det':>9} {'V19 K20':>9}  "
          f"{'V19 vs MS':>15}")
    print("-" * 110)

    for fam in ["qft", "qv", "vqe", "qaoa"]:
        gen = GENERATORS[fam]
        lam = lambda_for(fam, args.topology)
        sabre_mks, ms_mks, v19_det_mks, v19_k20_mks = [], [], [], []

        for ci in range(args.n_circuits):
            try:
                qc = gen(n_qubits, seed=1234 + ci)
            except Exception:
                continue
            try:
                # SABRE K=20
                s_mk = best_of_k(
                    lambda s: _sabre_pipeline(qc, graph, coupling, seed=s + ci * args.K),
                    args.K,
                )
                # SABRE-MS K=20
                m_mk = best_of_k(
                    lambda s: _ms_pipeline(qc, graph, coupling, lam,
                                          seed=s + ci * args.K),
                    args.K,
                )
                # V19 deterministic
                v_det = pipeline_v19(qc, graph, coupling, model, env_v19, n_qubits,
                                    seed=ci, deterministic=True)
                # V19 K=20 stochastic best-of-makespan
                v_k20 = best_of_k(
                    lambda s: pipeline_v19(qc, graph, coupling, model, env_v19, n_qubits,
                                          seed=s + ci * args.K, deterministic=False),
                    args.K,
                )
            except Exception:
                continue
            if any(v is None for v in (s_mk, m_mk, v_det, v_k20)):
                continue
            sabre_mks.append(s_mk); ms_mks.append(m_mk)
            v19_det_mks.append(v_det); v19_k20_mks.append(v_k20)

        if not sabre_mks:
            continue
        s = np.array(sabre_mks, float)
        m = np.array(ms_mks, float)
        vd = np.array(v19_det_mks, float)
        v20 = np.array(v19_k20_mks, float)
        gain_v_ms = 100 * (m.mean() - v20.mean()) / m.mean()
        p_v_lt_ms = wilcoxon(v20, m, "less")
        out[fam] = {
            "topology": args.topology, "family": fam, "lambda": lam,
            "n_qubits": n_qubits, "n_circuits": len(sabre_mks), "K": args.K,
            "sabre_mean": float(s.mean()),
            "ms_mean": float(m.mean()),
            "v19_det_mean": float(vd.mean()),
            "v19_k20_mean": float(v20.mean()),
            "v19_vs_ms_pct": float(gain_v_ms),
            "p_v19_less_than_ms": p_v_lt_ms,
        }
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        marker = "  *" if gain_v_ms > 1 and p_v_lt_ms < 0.05 else ""
        print(f"{fam:<12} {lam:>6.3f} {n_qubits:>4} {len(sabre_mks):>3}  "
              f"{s.mean():>9.1f} {m.mean():>9.1f} {vd.mean():>9.1f} {v20.mean():>9.1f}  "
              f"{gain_v_ms:>+8.2f}% p={p_v_lt_ms:.0e}{marker}")

    print(f"\nTotal: {time.time()-t0:.0f}s")
    print(f"Saved {out_path}")


def _sabre_pipeline(qc, graph, coupling, seed):
    from exp_real_full import (
        decompose_basis, get_layout, apply_layout_to_qc,
        route_with_sabre, optimize_qc, asap_makespan_qc,
    )
    qc_basis = decompose_basis(qc)
    perm = get_layout(qc_basis, coupling, seed)
    qc_phys = apply_layout_to_qc(qc_basis, perm)
    qc_routed = route_with_sabre(qc_phys, coupling, seed)
    return asap_makespan_qc(optimize_qc(qc_routed))


def _ms_pipeline(qc, graph, coupling, lam, seed, alim_mult=10):
    from exp_real_full import (
        decompose_basis, get_layout, apply_layout_to_qc,
        route_with_sabre_ms, optimize_qc, asap_makespan_qc,
    )
    qc_basis = decompose_basis(qc)
    perm = get_layout(qc_basis, coupling, seed)
    qc_phys = apply_layout_to_qc(qc_basis, perm)
    qc_routed = route_with_sabre_ms(qc_phys, graph, coupling, lam, seed, alim_mult)
    if qc_routed is None:
        return None
    return asap_makespan_qc(optimize_qc(qc_routed))


if __name__ == "__main__":
    main()

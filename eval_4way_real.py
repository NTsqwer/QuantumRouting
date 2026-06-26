"""4-way comparison on real benchmark circuits:
   SABRE-lookahead vs SABRE-MS vs baseline qgym RL vs V19 RL

All methods at K=20 best-of-K stochastic, paired per circuit.
Pipeline equivalence-verified throughout.

Usage:
  python eval_4way_real.py --topology linear5 \
      --baseline-model models/abl_baseline_real_linear5_FIXED.zip \
      --v19-model models/v19_real_linear5_FIXED.zip
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
from obs_wrapper_v19 import V19ObservationWrapper
from exp_real_full import (
    GENERATORS, lambda_for, decompose_basis, get_layout, apply_layout_to_qc,
    route_with_sabre, route_with_sabre_ms, optimize_qc, asap_makespan_qc,
)
from topologies import get as get_topology

from eval_v19_real_benchmarks import (
    _unwrap_state, v19_route, splice_1q_into_route,
)
from pipeline import routing_to_circuit


def make_env_for_condition(condition, graph, max_length=200, obs_reach=80):
    """Build the eval env matching the training condition."""
    env = Routing(connection_graph=graph,
                  interaction_generator=BasicInteractionGenerator(max_length=max_length, seed=0),
                  max_observation_reach=obs_reach,
                  observe_legal_surpasses=True,
                  observe_connection_graph=True)
    if condition == "v19":
        env = V19ObservationWrapper(env)
    elif condition == "baseline":
        pass  # qgym defaults, no obs wrapper
    else:
        raise ValueError(condition)
    env = ActionMaskWrapper(env)
    return env


def rl_route(model, env, cp, deterministic, max_steps=2000):
    """Run a policy on the 2q-pair sequence cp, return routed Gate list or None."""
    obs, _ = env.reset(options={"interaction_circuit": cp})
    state = _unwrap_state(env)
    for _ in range(max_steps):
        mask = env.action_masks()
        a, _ = model.predict(obs, deterministic=deterministic, action_masks=mask)
        obs, _r, term, trunc, _ = env.step(int(a))
        if state.is_done() or term or trunc:
            break
    return routing_to_circuit(state) if state.is_done() else None


def pipeline_rl(qc_orig, graph, coupling, model, env, n_qubits, seed,
               deterministic=False):
    qc_basis = decompose_basis(qc_orig)
    perm = get_layout(qc_basis, coupling, seed)
    qc_phys = apply_layout_to_qc(qc_basis, perm)
    pairs = []
    for instr in qc_phys.data:
        qs = [qc_phys.find_bit(q).index for q in instr.qubits]
        if len(qs) == 2:
            pairs.append((int(qs[0]), int(qs[1])))
    if not pairs:
        return asap_makespan_qc(optimize_qc(qc_phys))
    cp = np.array(pairs, dtype=int)
    routed_2q = rl_route(model, env, cp, deterministic=deterministic)
    if routed_2q is None:
        return None
    qc_routed = splice_1q_into_route(qc_phys, routed_2q, n_qubits)
    return asap_makespan_qc(optimize_qc(qc_routed))


def pipeline_sabre(qc, graph, coupling, seed):
    qc_basis = decompose_basis(qc)
    perm = get_layout(qc_basis, coupling, seed)
    qc_phys = apply_layout_to_qc(qc_basis, perm)
    qc_routed = route_with_sabre(qc_phys, coupling, seed)
    return asap_makespan_qc(optimize_qc(qc_routed))


def pipeline_ms(qc, graph, coupling, lam, seed):
    qc_basis = decompose_basis(qc)
    perm = get_layout(qc_basis, coupling, seed)
    qc_phys = apply_layout_to_qc(qc_basis, perm)
    qc_routed = route_with_sabre_ms(qc_phys, graph, coupling, lam,
                                    seed=seed, alim_mult=10)
    if qc_routed is None:
        return None
    return asap_makespan_qc(optimize_qc(qc_routed))


def best_of_k(fn, K):
    best = float("inf")
    for s in range(K):
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
    p.add_argument("--baseline-model", required=True)
    p.add_argument("--v19-model", required=True)
    p.add_argument("--n-circuits", type=int, default=12)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--max-length", type=int, default=80,
                  help="Eval-time env buffer; default 80")
    p.add_argument("--obs-reach", type=int, default=40,
                  help="Eval-time obs reach; should match training")
    args = p.parse_args()

    graph, n_qubits = get_topology(args.topology)
    edges = list(graph.edges())
    coupling = CouplingMap([list(e) for e in edges]
                          + [list(reversed(e)) for e in edges])

    env_baseline = make_env_for_condition("baseline", graph,
                                          max_length=args.max_length,
                                          obs_reach=args.obs_reach)
    env_v19 = make_env_for_condition("v19", graph,
                                     max_length=args.max_length,
                                     obs_reach=args.obs_reach)

    print(f"Loading models...")
    m_baseline = MaskablePPO.load(args.baseline_model, device="cpu")
    m_v19 = MaskablePPO.load(args.v19_model, device="cpu")
    print(f"  baseline: {args.baseline_model}")
    print(f"  v19:      {args.v19_model}")
    print()

    print(f"4-way eval on {args.topology}, K={args.K}, n_circuits={args.n_circuits}\n")
    print(f"{'Family':<10} {'lam':>6} {'n_q':>4} {'n':>3}  "
          f"{'SABRE':>9} {'MS':>9} {'base_k20':>9} {'v19_k20':>9}  "
          f"{'best_RL_vs_MS':>13} {'V19_vs_base':>12}")
    print("-" * 130)

    out = {}
    for fam in ["qft", "qv", "vqe", "qaoa"]:
        gen = GENERATORS[fam]
        lam = lambda_for(fam, args.topology)
        sabre_l, ms_l, base_l, v19_l = [], [], [], []

        for ci in range(args.n_circuits):
            try:
                qc = gen(n_qubits, seed=1234 + ci)
            except Exception:
                continue
            try:
                s = best_of_k(
                    lambda seed: pipeline_sabre(qc, graph, coupling, seed + ci * args.K),
                    args.K)
                m = best_of_k(
                    lambda seed: pipeline_ms(qc, graph, coupling, lam, seed + ci * args.K),
                    args.K)
                b = best_of_k(
                    lambda seed: pipeline_rl(qc, graph, coupling, m_baseline, env_baseline,
                                            n_qubits, seed + ci * args.K, deterministic=False),
                    args.K)
                v = best_of_k(
                    lambda seed: pipeline_rl(qc, graph, coupling, m_v19, env_v19,
                                            n_qubits, seed + ci * args.K, deterministic=False),
                    args.K)
            except Exception as e:
                print(f"  ci={ci} {fam}: ERR {type(e).__name__}: {e}")
                continue
            if any(x is None for x in (s, m, b, v)):
                continue
            sabre_l.append(s); ms_l.append(m); base_l.append(b); v19_l.append(v)

        if not sabre_l:
            print(f"  {fam}: no successful routings")
            continue

        sabre_a = np.array(sabre_l, float)
        ms_a    = np.array(ms_l, float)
        base_a  = np.array(base_l, float)
        v19_a   = np.array(v19_l, float)

        # Best RL of the two
        best_rl = np.minimum(base_a, v19_a)
        best_rl_vs_ms = 100 * (ms_a.mean() - best_rl.mean()) / ms_a.mean()
        v19_vs_base = 100 * (base_a.mean() - v19_a.mean()) / base_a.mean()
        p_v19_vs_base = wilcoxon(v19_a, base_a, "less")

        out[fam] = {
            "topology": args.topology, "family": fam, "lambda": lam,
            "n_qubits": n_qubits, "n_circuits": len(sabre_l), "K": args.K,
            "sabre_mean": float(sabre_a.mean()),
            "ms_mean": float(ms_a.mean()),
            "baseline_mean": float(base_a.mean()),
            "v19_mean": float(v19_a.mean()),
            "v19_vs_baseline_pct": float(v19_vs_base),
            "p_v19_less_than_baseline": p_v19_vs_base,
            "v19_vs_sabre_pct": float(100 * (sabre_a.mean() - v19_a.mean()) / sabre_a.mean()),
            "v19_vs_ms_pct": float(100 * (ms_a.mean() - v19_a.mean()) / ms_a.mean()),
            "baseline_vs_sabre_pct": float(100 * (sabre_a.mean() - base_a.mean()) / sabre_a.mean()),
        }
        sig = "*" if p_v19_vs_base < 0.05 and v19_vs_base > 1 else " "
        print(f"{fam:<10} {lam:>6.3f} {n_qubits:>4} {len(sabre_l):>3}  "
              f"{sabre_a.mean():>9.1f} {ms_a.mean():>9.1f} {base_a.mean():>9.1f} {v19_a.mean():>9.1f}  "
              f"{best_rl_vs_ms:>+12.2f}% {v19_vs_base:>+11.2f}%{sig}",
              flush=True)

    out_path = f"results/4way_real_{args.topology}.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()

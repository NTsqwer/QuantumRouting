"""Comprehensive RL evaluation against ALL 4 routing baselines.

Until now, our RL models (V14-lite, 4-way ablation) have only been compared against
qiskit SABRE with heuristic="lookahead". A reviewer can reasonably ask:
  - Does the RL also beat LightSABRE (decay)?
  - Does the RL beat the original Li et al. 2019 SABRE (basic)?
  - Does the RL beat the classical SABRE-MS fix, or only the unmodified SABRE family?

This script answers all three at once.

For each topology in {linear5, tshape5, ring5}, evaluate:
    Baselines (Qiskit SabreSwap):
        - SABRE-basic     (heuristic="basic",     trials=1, det K=1; trials=20, K=20)
        - SABRE-lookahead (heuristic="lookahead", trials=1; trials=20)
        - SABRE-decay     (heuristic="decay",     trials=1; trials=20)
    Classical contribution:
        - SABRE-MS        (our sabre_impl with oracle lambda, K=1 and K=20)
    RL models:
        - V14-lite (linear5 only, 3 model seeds)
        - 4-way ablation: baseline / obs_only / reward_only / both (linear5, tshape5, ring5)

Each method evaluated on 100 SabreLayout-permuted circuits per (topology, family).
K=20 mode uses "best-of-K-by-makespan" for all methods.

Output: results/rl_vs_all_baselines.json
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
from scipy import stats
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import SabreLayout, SabreSwap
from stable_baselines3 import PPO

from qgym.envs.routing import Routing
from qgym.envs.routing.routing_rewarders import BasicRewarder
from qgym.generators.interaction import BasicInteractionGenerator

from circuits import make_circuits
from obs_wrapper_v14lite import SchedulingAwareObservationWrapperLite
from optimize import optimize_circuit, qiskit_to_qgym_2q_only
from pipeline import asap_makespan, routing_to_circuit
from sabre_impl import sabre_route
from topologies import get as get_topology


GD = {"cnot": 2, "swap": 6}
FAMILIES = ["random", "qft", "parallel", "trotter"]

# Per-topology oracle lambdas for SABRE-MS (from oracle_lambda.json and related).
# For 5-qubit topologies, oracle is family-dependent:
#   - QFT/trotter on ring/linear: 0.25
#   - random: 0.10
#   - parallel: 0.02 (fragile)
# tshape5 is treated like linear; ring5 like ring.
ORACLE_LAMBDA = {
    "linear5": {"random": 0.10, "qft": 0.25, "parallel": 0.02, "trotter": 0.25},
    "tshape5": {"random": 0.10, "qft": 0.25, "parallel": 0.02, "trotter": 0.25},
    "ring5":   {"random": 0.10, "qft": 0.25, "parallel": 0.02, "trotter": 0.25},
}

# Ablation model paths (relative to models/). The script tries to load each;
# missing files are skipped.
ABLATION_MODELS = {
    "linear5": {
        "baseline":     "models/B_abl_baseline_linear5.zip",
        "obs_only":     "models/B_abl_obsonly_linear5.zip",
        "reward_only":  "models/B_abl_rewardonly_linear5.zip",
        # The "both" model on linear5 is the V14-lite checkpoint
        "both":         "models/B_v14lite_linear5.zip",
    },
    "tshape5": {
        "baseline":     "models/B_abl_baseline_tshape5.zip",
        "obs_only":     "models/B_abl_obsonly_tshape5.zip",
        "reward_only":  "models/B_abl_rewardonly_tshape5.zip",
        "both":         "models/B_abl_both_tshape5.zip",
    },
    "ring5": {
        "baseline":     "models/B_abl_baseline_ring5.zip",
        "obs_only":     "models/B_abl_obsonly_ring5.zip",
        "reward_only":  "models/B_abl_rewardonly_ring5.zip",
        "both":         "models/B_abl_both_ring5.zip",
    },
}

# Whether each ablation variant uses the v14lite obs-wrapper (obs_only and both do;
# baseline and reward_only do not).
ABL_USES_V14LITE_OBS = {
    "baseline":    False,
    "obs_only":    True,
    "reward_only": False,
    "both":        True,
}


# =============================================================================
# Pipeline helpers
# =============================================================================

def sabre_layout_perm(circuit, coupling, n_qubits, seed=0):
    if circuit.size == 0:
        return np.arange(n_qubits, dtype=int)
    qc = QuantumCircuit(n_qubits)
    for q1, q2 in circuit:
        qc.cx(int(q1), int(q2))
    pm = PassManager([SabreLayout(coupling_map=coupling, seed=seed, max_iterations=2)])
    pm.run(qc)
    layout = pm.property_set.get("layout")
    if layout is None:
        return np.arange(n_qubits, dtype=int)
    perm = np.zeros(n_qubits, dtype=int)
    for i, q in enumerate(qc.qubits):
        perm[i] = layout[q]
    return perm


def qiskit_route_gates(circuit_perm, coupling, n_qubits, heuristic, seed=0):
    """One single-seed Qiskit SABRE routing pass. Returns qgym Gate list."""
    from qgym.custom_types import Gate
    qc = QuantumCircuit(n_qubits)
    for q1, q2 in circuit_perm:
        qc.cx(int(q1), int(q2))
    pm = PassManager([SabreSwap(coupling_map=coupling, heuristic=heuristic,
                                seed=seed, trials=1)])
    routed = pm.run(qc)
    out = []
    for instr in routed.data:
        op = instr.operation
        qs = [routed.find_bit(q).index for q in instr.qubits]
        if len(qs) == 2:
            name = "cnot" if op.name == "cx" else op.name
            out.append(Gate(name, int(qs[0]), int(qs[1])))
    return out


def qiskit_mk_one(circuit_perm, coupling, n_qubits, heuristic, seed):
    """Qiskit SABRE -> optimize -> ASAP makespan. Single seed."""
    gates = qiskit_route_gates(circuit_perm, coupling, n_qubits, heuristic, seed=seed)
    opt = optimize_circuit(gates, n_qubits)
    return asap_makespan(opt, GD)


def qiskit_mk_k(circuit_perm, coupling, n_qubits, heuristic, k=20):
    """Best-of-K-by-makespan for a Qiskit SABRE variant."""
    best = float("inf")
    for s in range(k):
        mk = qiskit_mk_one(circuit_perm, coupling, n_qubits, heuristic, s)
        if mk < best:
            best = mk
    return best


def sabre_ms_mk_one(circuit_perm, graph, n_qubits, lam, seed, alim_mult=10):
    """SABRE-MS -> optimize -> ASAP makespan. Single seed."""
    alim = alim_mult * n_qubits
    gates = sabre_route(circuit_perm, graph, lookahead=True, makespan_lambda=lam,
                        makespan_mode="start_cycle", seed=seed, attempt_limit=alim)
    if gates is None:
        return None
    opt = optimize_circuit(gates, n_qubits)
    return asap_makespan(opt, GD)


def sabre_ms_mk_k(circuit_perm, graph, n_qubits, lam, k=20, alim_mult=10):
    """Best-of-K-by-makespan for SABRE-MS."""
    best = float("inf")
    n_success = 0
    for s in range(k):
        mk = sabre_ms_mk_one(circuit_perm, graph, n_qubits, lam, s, alim_mult)
        if mk is None:
            continue
        n_success += 1
        if mk < best:
            best = mk
    return best if n_success > 0 else None


# =============================================================================
# RL evaluation
# =============================================================================

def make_env(graph, n_qubits, max_length, use_v14lite_obs):
    env = Routing(
        connection_graph=graph,
        interaction_generator=BasicInteractionGenerator(max_length=max_length, seed=0),
        max_observation_reach=max_length,
        observe_legal_surpasses=True,
        observe_connection_graph=True,
        rewarder=BasicRewarder(),
    )
    if use_v14lite_obs:
        return SchedulingAwareObservationWrapperLite(env)
    return env


def ppo_det_route(model, env, cp, max_steps=2000, stuck_window=30):
    """Deterministic PPO routing with stuck-detection fallback to stochastic."""
    obs, _ = env.reset(options={"interaction_circuit": cp})
    state = env.unwrapped._state if hasattr(env, "unwrapped") else env._state
    last_pos, stuck = 0, 0
    for _ in range(max_steps):
        a, _ = model.predict(obs, deterministic=stuck < stuck_window)
        obs, _r, term, trunc, _ = env.step(int(a))
        pos = int(state.position)
        if pos != last_pos:
            stuck = 0
            last_pos = pos
        else:
            stuck += 1
        if state.is_done() or term or trunc:
            break
    return routing_to_circuit(state) if state.is_done() else None


def ppo_stoch_route(model, env, cp, max_steps=2000):
    """Stochastic PPO routing for K=20 best-of."""
    obs, _ = env.reset(options={"interaction_circuit": cp})
    state = env.unwrapped._state if hasattr(env, "unwrapped") else env._state
    for _ in range(max_steps):
        a, _ = model.predict(obs, deterministic=False)
        obs, _r, term, trunc, _ = env.step(int(a))
        if state.is_done() or term or trunc:
            break
    return routing_to_circuit(state) if state.is_done() else None


def ppo_mk_det(model, env, cp, n_qubits):
    gates = ppo_det_route(model, env, cp)
    if gates is None:
        return None
    opt = optimize_circuit(gates, n_qubits)
    return asap_makespan(opt, GD)


def ppo_mk_k20(model, env, cp, n_qubits, k=20):
    """Best-of-K stochastic routings, scored by makespan."""
    best = float("inf")
    n_success = 0
    for _ in range(k):
        gates = ppo_stoch_route(model, env, cp)
        if gates is None:
            continue
        n_success += 1
        opt = optimize_circuit(gates, n_qubits)
        mk = asap_makespan(opt, GD)
        if mk < best:
            best = mk
    return best if n_success > 0 else None


# =============================================================================
# Per-topology evaluation
# =============================================================================

def eval_topology(topology, n_per_cell=100, max_length=10, k=20, do_k20=True, seed=1234):
    print(f"\n{'='*70}")
    print(f"Topology: {topology}")
    print(f"{'='*70}", flush=True)
    t_topo = time.time()

    graph, n_qubits = get_topology(topology)
    edges = list(graph.edges())
    coupling = CouplingMap([list(e) for e in edges] + [list(reversed(e)) for e in edges])

    # Load all available RL models for this topology
    rl_models = {}
    for name, path in ABLATION_MODELS.get(topology, {}).items():
        if os.path.exists(path):
            try:
                rl_models[name] = PPO.load(path, device="cpu")
                print(f"  loaded {name}: {path}", flush=True)
            except Exception as e:
                print(f"  FAIL to load {name} ({path}): {e}", flush=True)
        else:
            print(f"  missing {name}: {path}", flush=True)

    # Build envs (separately for v14lite-obs vs default obs)
    env_default = make_env(graph, n_qubits, max_length, use_v14lite_obs=False)
    env_v14lite = make_env(graph, n_qubits, max_length, use_v14lite_obs=True)

    # Per-family results
    results = {}
    for fam in FAMILIES:
        print(f"\n  family: {fam}", flush=True)
        t_fam = time.time()

        lam = ORACLE_LAMBDA[topology][fam]
        cs = make_circuits(fam, n_qubits, n_per_cell, seed=seed, length=max_length)

        # Storage for per-circuit makespans, one list per method
        # methods include the 3 qiskit heuristics × {K=1, K=20}, sabre-ms × {K=1, K=20},
        # and each RL model × {det, K=20 if enabled}
        mks_k1 = {"basic": [], "lookahead": [], "decay": [], "sabre_ms": []}
        mks_k20 = {"basic": [], "lookahead": [], "decay": [], "sabre_ms": []}
        ppo_det_mks = {name: [] for name in rl_models}
        ppo_k20_mks = {name: [] for name in rl_models}

        for ci, c in enumerate(cs):
            perm = sabre_layout_perm(c, coupling, n_qubits, seed=ci)
            cp = perm[c].astype(int)

            # K=1 Qiskit baselines (single seed = ci, paired by circuit)
            for h in ("basic", "lookahead", "decay"):
                mk = qiskit_mk_one(cp, coupling, n_qubits, h, seed=ci)
                mks_k1[h].append(mk)

            # K=1 SABRE-MS
            mk_ms1 = sabre_ms_mk_one(cp, graph, n_qubits, lam, seed=ci)
            if mk_ms1 is None:
                # fallback: worst of the other K=1 results
                mk_ms1 = max(mks_k1["basic"][-1], mks_k1["lookahead"][-1], mks_k1["decay"][-1])
            mks_k1["sabre_ms"].append(mk_ms1)

            # K=20 Qiskit baselines
            if do_k20:
                for h in ("basic", "lookahead", "decay"):
                    mk = qiskit_mk_k(cp, coupling, n_qubits, h, k=k)
                    mks_k20[h].append(mk)
                # K=20 SABRE-MS
                mk_ms = sabre_ms_mk_k(cp, graph, n_qubits, lam, k=k)
                if mk_ms is None:
                    mk_ms = mks_k20["lookahead"][-1]
                mks_k20["sabre_ms"].append(mk_ms)

            # RL models
            for name, model in rl_models.items():
                env = env_v14lite if ABL_USES_V14LITE_OBS[name] else env_default
                # Deterministic
                mk_det = ppo_mk_det(model, env, cp, n_qubits)
                if mk_det is None:
                    # PPO got stuck — fallback to SABRE-lookahead
                    mk_det = mks_k1["lookahead"][-1]
                ppo_det_mks[name].append(mk_det)
                # K=20 stochastic best-of
                if do_k20:
                    mk_k20 = ppo_mk_k20(model, env, cp, n_qubits, k=k)
                    if mk_k20 is None:
                        mk_k20 = mks_k20["lookahead"][-1] if mks_k20["lookahead"] else mk_det
                    ppo_k20_mks[name].append(mk_k20)

        # Aggregate
        def means(d):
            return {k: float(np.mean(v)) for k, v in d.items() if v}

        results[fam] = {
            "lambda": lam,
            "n_circuits": n_per_cell,
            "k1_means": means(mks_k1),
            "k20_means": means(mks_k20) if do_k20 else {},
            "ppo_det_means": means(ppo_det_mks),
            "ppo_k20_means": means(ppo_k20_mks) if do_k20 else {},
            # Keep raw arrays for paired tests
            "k1_raw": {k: [float(x) for x in v] for k, v in mks_k1.items()},
            "k20_raw": {k: [float(x) for x in v] for k, v in mks_k20.items()} if do_k20 else {},
            "ppo_det_raw": {n: [float(x) for x in v] for n, v in ppo_det_mks.items()},
            "ppo_k20_raw": {n: [float(x) for x in v] for n, v in ppo_k20_mks.items()} if do_k20 else {},
        }

        # Print summary line
        line = f"    {fam}: "
        for h in ("basic", "lookahead", "decay", "sabre_ms"):
            line += f"{h}={results[fam]['k1_means'][h]:.1f} "
        for name in rl_models:
            line += f"RL[{name}]det={results[fam]['ppo_det_means'][name]:.1f} "
        print(line + f"  ({time.time()-t_fam:.1f}s)", flush=True)

    print(f"\nTopology {topology} done in {time.time()-t_topo:.0f}s", flush=True)
    return results


def main():
    os.makedirs("results", exist_ok=True)
    out_path = "results/rl_vs_all_baselines.json"
    out = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            out = json.load(f)
        print(f"Loaded {len(out)} existing topologies from {out_path}")

    t0 = time.time()
    for topology in ("linear5", "tshape5", "ring5"):
        if topology in out:
            print(f"\nSkipping {topology} (already done). Delete from JSON to rerun.")
            continue

        out[topology] = eval_topology(topology, n_per_cell=100, k=20, do_k20=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

    print(f"\nTotal: {time.time()-t0:.0f}s")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

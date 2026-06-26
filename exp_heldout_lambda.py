"""Held-out lambda validation.

Reviewer attack: "your lambda values are per-class tuned, so the gain
is overfit to your test circuits." Defense: split each cell into a
train half and a test half, pick lambda on the train half by sweeping,
and report the gain on the test half. If the test-half gain is close
to the headline gain, the lambda values generalize.

Protocol per cell:
  1. Generate 2N circuits (deterministic seeds), split into train (first
     N) and test (last N).
  2. Train fold: for each lambda in {0.005, 0.01, 0.02, 0.05, 0.10,
     0.25}, run SABRE-MS K=20 on the train circuits and record mean
     post-optimisation makespan. Pick the lambda with the smallest
     mean.
  3. Test fold: run SABRE (K=20) and SABRE-MS (K=20, best-lambda-from-
     train) on the test circuits. Report mean gain and paired Wilcoxon.

We use the same realistic full Qiskit pipeline as exp_real_full.py so
the numbers are directly comparable to Table 8 in the paper.

Output: results/heldout_lambda.json
"""
from __future__ import annotations

import json
import os
import time
import warnings

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore", category=DeprecationWarning)

from qiskit.transpiler import CouplingMap

from exp_real_full import (
    GENERATORS, K, pipeline_sabre, pipeline_sabre_ms, best_of_k,
)
from topologies import get as get_topology


LAMBDA_GRID = [0.005, 0.01, 0.02, 0.05, 0.10, 0.25]


def wilcoxon_less(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.all(a == b):
        return 1.0
    try:
        _, p = stats.wilcoxon(a, b, alternative="less")
        return float(p)
    except Exception:
        return 1.0


def run_cell(topology, family, n_per_fold, alim_mult=10):
    graph, n_q = get_topology(topology)
    edges = list(graph.edges())
    coupling = CouplingMap([list(e) for e in edges]
                          + [list(reversed(e)) for e in edges])
    gen = GENERATORS[family]

    # Generate 2*n_per_fold circuits deterministically. Train first, test
    # last. Different seed prefixes for train vs test so they're disjoint
    # samples from the same family distribution.
    train_circuits, test_circuits = [], []
    for ci in range(n_per_fold):
        try:
            train_circuits.append(gen(n_q, seed=10_000 + ci))
        except Exception:
            continue
    for ci in range(n_per_fold):
        try:
            test_circuits.append(gen(n_q, seed=20_000 + ci))
        except Exception:
            continue
    if not train_circuits or not test_circuits:
        return None

    # ----- TRAIN: sweep lambda, pick best by mean makespan on train half
    train_means = {}
    for lam in LAMBDA_GRID:
        mks_list = []
        for ci, qc in enumerate(train_circuits):
            m = best_of_k(
                lambda s, qc=qc, lam=lam, ci=ci:
                    pipeline_sabre_ms(qc, graph, coupling, lam,
                                     seed=s + ci * K, alim_mult=alim_mult),
                K,
            )
            if m is not None:
                mks_list.append(m)
        if mks_list:
            train_means[lam] = float(np.mean(mks_list))
    if not train_means:
        return None
    best_lam = min(train_means, key=train_means.get)

    # ----- TEST: evaluate SABRE and SABRE-MS@best_lam on held-out half
    sabre_mks, ms_mks = [], []
    for ci, qc in enumerate(test_circuits):
        s_m = best_of_k(
            lambda s, qc=qc, ci=ci:
                pipeline_sabre(qc, graph, coupling, seed=s + ci * K),
            K,
        )
        m_m = best_of_k(
            lambda s, qc=qc, ci=ci:
                pipeline_sabre_ms(qc, graph, coupling, best_lam,
                                 seed=s + ci * K, alim_mult=alim_mult),
            K,
        )
        if s_m is None or m_m is None:
            continue
        sabre_mks.append(s_m); ms_mks.append(m_m)
    if not sabre_mks:
        return None

    a = np.array(ms_mks, float)
    b = np.array(sabre_mks, float)
    # Per-circuit makespan reduction (paired), for std / CI reporting.
    per_circ_gain = 100.0 * (b - a) / b
    mean_gain = float(np.mean(per_circ_gain))
    std_gain = float(np.std(per_circ_gain, ddof=1)) if len(per_circ_gain) > 1 else 0.0
    sem = std_gain / np.sqrt(len(per_circ_gain)) if len(per_circ_gain) else 0.0
    return {
        "topology": topology, "family": family,
        "n_qubits": n_q,
        "n_train": len(train_circuits),
        "n_test": len(sabre_mks),
        "lambda_grid": LAMBDA_GRID,
        "train_means_per_lambda": train_means,
        "best_lambda_from_train": float(best_lam),
        "test_sabre_mean": float(b.mean()),
        "test_ms_mean": float(a.mean()),
        "test_gain_pct": float(100 * (b.mean() - a.mean()) / b.mean()),
        "test_gain_mean_percircuit": mean_gain,
        "test_gain_std": std_gain,
        "test_gain_sem": float(sem),
        "per_circuit_gain": [float(x) for x in per_circ_gain],
        "test_p_ms_less": wilcoxon_less(a, b),
    }


# Same cell list as exp_real_full.py main table, but doubled n so we can
# split. We use a fraction of that to keep wall-time reasonable.
# Format: (topology, family, n_per_fold, alim_mult)
CELLS = [
    # QFT — representative of the strongest gains
    ("linear7",   "qft",  6, 10),
    ("ring8",     "qft",  6, 10),
    ("ring12",    "qft",  6, 10),
    ("grid3x3",   "qft",  6, 10),
    ("grid4x4",   "qft",  4, 10),
    ("heavy_hex2","qft",  4, 10),
    # Quantum Volume — IBM's hardware benchmark
    ("linear7",   "qv",   6, 10),
    ("ring8",     "qv",   6, 10),
    ("ring12",    "qv",   5, 10),
    ("grid3x3",   "qv",   6, 10),
    ("grid4x4",   "qv",   4, 10),
    # VQE
    ("ring8",     "vqe",  6, 10),
    ("grid3x3",   "vqe",  6, 10),
    # QAOA — the family with the small-lambda exception
    ("linear7",   "qaoa", 8, 10),
    ("ring8",     "qaoa", 8, 10),
    ("ring12",    "qaoa", 6, 10),
    ("grid3x3",   "qaoa", 6, 10),
]


def main():
    os.makedirs("results", exist_ok=True)
    out_path = "results/heldout_lambda.json"
    out = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            out = json.load(f)
        print(f"Loaded {len(out)} cells")

    t0 = time.time()
    print("Held-out lambda validation")
    print(f"  lambda grid: {LAMBDA_GRID}")
    print(f"  K=20 per method, train fold picks lambda, test fold reports gain\n")
    print(f"{'Cell':<24} {'n_q':>4} {'n_tr':>5} {'n_te':>5} "
          f"{'best_lam':>9} {'SABRE':>8} {'MS':>8} {'gain':>9} {'p':>9}")
    print("-" * 100)

    for topo, fam, n, alim in CELLS:
        key = f"{topo}/{fam}"
        if key in out and out[key].get("n_test"):
            r = out[key]
        else:
            print(f"  Running {key}...", flush=True)
            tc = time.time()
            r = run_cell(topo, fam, n, alim_mult=alim)
            elapsed = time.time() - tc
            if r is None:
                print(f"    ...{elapsed:.0f}s, FAILED")
                continue
            print(f"    ...{elapsed:.0f}s, train_means_per_lambda="
                  f"{ {k: round(v,2) for k,v in r['train_means_per_lambda'].items()} }",
                  flush=True)
            out[key] = r
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
        marker = "  *" if r["test_gain_pct"] > 1 and r["test_p_ms_less"] < 0.05 else ""
        print(f"{key:<24} {r['n_qubits']:>4} {r['n_train']:>5} {r['n_test']:>5} "
              f"{r['best_lambda_from_train']:>9.3f} "
              f"{r['test_sabre_mean']:>8.2f} {r['test_ms_mean']:>8.2f} "
              f"{r['test_gain_pct']:>+8.2f}% "
              f"{r['test_p_ms_less']:>9.1e}{marker}")

    if out:
        gains = [v["test_gain_pct"] for v in out.values()]
        sig = sum(1 for v in out.values()
                 if v["test_gain_pct"] > 1 and v["test_p_ms_less"] < 0.05)
        print(f"\nHeld-out summary across {len(out)} cells:")
        print(f"  mean gain : {np.mean(gains):+.2f}%")
        print(f"  median    : {np.median(gains):+.2f}%")
        print(f"  significant: {sig}/{len(out)}")
        # Per-family
        print("  per family:")
        for fam in ["qft", "qv", "vqe", "qaoa"]:
            fg = [v["test_gain_pct"] for v in out.values() if v["family"] == fam]
            if fg:
                print(f"    {fam:<6}: n={len(fg):<3} mean {np.mean(fg):+.2f}%  "
                      f"median {np.median(fg):+.2f}%")

    print(f"\nTotal: {time.time()-t0:.0f}s")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

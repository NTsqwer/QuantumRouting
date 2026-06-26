"""DEFENSE ARTIFACT: run the paper's exact MQT pipeline on EVERY MQT Bench family
that generates at our qubit counts, not just the 11 in the paper.

This reuses exp_core_mqt.run_cell VERBATIM, so the methodology (SabreLayout seed 0,
K=20 pool, per-run lambda selection, ASAP makespan, optimize pass) is byte-identical
to the paper. The ONLY change is the family list: all families discoverable in the
installed mqt.bench, tested at each of the 6 core topology qubit counts.

Output:
  results/mqt_allfamilies_perrun.json   (full per-cell records)
  results/mqt_allfamilies_table.csv     (flat per-circuit table for the appendix)

Purpose: if examiners ask "what about Grover / DJ / the families you omitted?",
this is the evidence the effect holds (or honestly where it does not).
"""
from __future__ import annotations
import json
import os
import csv
import time
import warnings
import pkgutil
import numpy as np
warnings.filterwarnings("ignore")

import mqt.bench.benchmarks as _bench_pkg
from exp_core_mqt import run_cell, TOPOLOGIES, FAMILIES as PAPER_FAMILIES


def all_family_names():
    """Every benchmark module mqt.bench exposes (minus private/registry)."""
    names = [n for _, n, _ in pkgutil.iter_modules(_bench_pkg.__path__)
             if not n.startswith("_")]
    return sorted(names)


def main():
    os.makedirs("results", exist_ok=True)
    out_path = "results/mqt_allfamilies_perrun.json"
    csv_path = "results/mqt_allfamilies_table.csv"
    out = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            out = json.load(f)

    families = all_family_names()
    paper_set = set(PAPER_FAMILIES)
    print(f"MQT ALL-FAMILIES | {len(families)} families x {len(TOPOLOGIES)} topologies")
    print(f"(paper used {len(paper_set)}: {sorted(paper_set)})\n")
    print(f"{'cell':<30}{'inPaper':>8}{'2q':>5}{'audit%':>8}{'route':>7}")
    print("-" * 62)

    t0 = time.time()
    for fam in families:
        for topo, nq in TOPOLOGIES:
            key = f"{fam}/{topo}"
            if key in out and "error" not in (out[key] or {"error": 1}):
                r = out[key]
            else:
                r = run_cell(fam, topo, nq)
                out[key] = r
                with open(out_path, "w") as f:
                    json.dump(out, f, indent=2)
            in_paper = "yes" if fam in paper_set else "NEW"
            if "error" in r:
                # Most "errors" here are just "family doesn't generate at nq" -- expected.
                continue
            print(f"{key:<30}{in_paper:>8}{r['n_2q']:>5}"
                  f"{r['audit_gain_pct']:>+7.1f}%{str(r['routing']):>7}", flush=True)

    # ---- flat per-circuit CSV (one row per generated cell) ----
    rows = [r for r in out.values() if r and "error" not in r]
    rows.sort(key=lambda r: (r["family"] not in paper_set, r["family"], r["topology"]))
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "in_paper", "topology", "n_qubits", "n_2q", "lambda",
                    "sabre_makespan", "ms_makespan", "audit_gain_pct",
                    "rescore_pct", "algorithm_pct", "raw_parallel_pct",
                    "reach_residual_pct", "routing_meaningful"])
        for r in rows:
            w.writerow([r["family"], "yes" if r["family"] in paper_set else "no",
                        r["topology"], r["n_qubits"], r["n_2q"], r["lam"],
                        r["A_prod"], r["C_ms"],
                        round(r["audit_gain_pct"], 2), round(r["rescore_pct"], 2),
                        round(r["algorithm_pct"], 2), round(r["ms_raw_gain_pct"], 2),
                        round(r["reach_residual_pct"], 2), r["routing"]])

    # ---- summary ----
    routing = [r for r in rows if r.get("routing")]
    new_routing = [r for r in routing if r["family"] not in paper_set]
    paper_routing = [r for r in routing if r["family"] in paper_set]
    print(f"\n=== SUMMARY ===")
    print(f"  generated cells: {len(rows)}  (routing-meaningful: {len(routing)})")
    if paper_routing:
        print(f"  PAPER families   routing cells: {len(paper_routing)}  "
              f"mean audit {np.mean([r['audit_gain_pct'] for r in paper_routing]):+.1f}%")
    if new_routing:
        g = [r["audit_gain_pct"] for r in new_routing]
        print(f"  OMITTED families routing cells: {len(new_routing)}  "
              f"mean audit {np.mean(g):+.1f}%  (min {min(g):+.1f}%, max {max(g):+.1f}%)")
        print(f"  omitted families that route: "
              f"{sorted(set(r['family'] for r in new_routing))}")
        nwin = sum(1 for x in g if x > 0)
        print(f"  omitted cells where SABRE-MS wins (audit>0): {nwin}/{len(new_routing)}")
    print(f"\nTotal: {time.time()-t0:.0f}s")
    print(f"Saved: {out_path}\n       {csv_path}")


if __name__ == "__main__":
    main()

"""Plot SABRE-MS improvement vs qubit count from results/mqt_qubit_sweep.json.

Two panels:
  (left)  makespan reduction (%) vs qubit count
  (right) ESP ratio (SABRE-MS / SABRE) vs qubit count
Series: averaged over families, split by topology type (grid vs heavy-hex).
Error bars: 95% bootstrap CI on makespan; min-max across families for ESP.
Saves figures/mqt_qubit_sweep.pdf
"""
from __future__ import annotations
import json
import os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    with open("results/mqt_qubit_sweep.json") as f:
        data = json.load(f)

    # Group by (topology_type, n_qubits) -> list of cell results
    grid_pts = defaultdict(list)   # n -> [results]
    hhex_pts = defaultdict(list)
    for key, r in data.items():
        if r is None or "makespan_reduction_pct" not in r:
            continue
        n = r["n_qubits"]
        if r["topology"].startswith("grid"):
            grid_pts[n].append(r)
        elif r["topology"].startswith("heavy_hex"):
            hhex_pts[n].append(r)

    def series(pts, field):
        """field='mks': mean % makespan reduction with bootstrap CI.
        field='esp': median log10(ESP ratio) with 25-75 percentile band.
        log10 is numerically stable; raw ESP ratios underflow/overflow at
        large n because baseline ESP -> 0 on deep circuits."""
        ns = sorted(pts)
        xs, ys, los, his = [], [], [], []
        for n in ns:
            rs = pts[n]
            if field == "mks":
                vals = np.array([v for r in rs for v in r["mks_red_vals"]], float)
                xs.append(n)
                ys.append(float(np.mean(vals)))
                rng = np.random.default_rng(0)
                boots = [float(np.mean(vals[rng.integers(0, len(vals), len(vals))]))
                         for _ in range(2000)]
                los.append(float(np.percentile(boots, 2.5)))
                his.append(float(np.percentile(boots, 97.5)))
            else:
                ratios = np.array([v for r in rs for v in r["esp_ratio_vals"]
                                   if v > 0], float)
                logr = np.log10(ratios)
                xs.append(n)
                ys.append(float(np.median(logr)))
                los.append(float(np.percentile(logr, 25)))
                his.append(float(np.percentile(logr, 75)))
        return np.array(xs), np.array(ys), np.array(los), np.array(his)

    # Single makespan panel. The ESP ratio is NOT plotted here: at >16 qubits
    # ESP underflows to ~0 for both methods, so the ratio is a floor artifact
    # (see the ESP subsection, which reports ESP only in the meaningful regime).
    fig, ax1 = plt.subplots(1, 1, figsize=(6.0, 4.2))

    for pts, label, color, marker in [
        (grid_pts, "Grid (2D)", "tab:blue", "o"),
        (hhex_pts, "Heavy-hex (IBM)", "tab:red", "s"),
    ]:
        if not pts:
            continue
        x, y, lo, hi = series(pts, "mks")
        if len(x):
            ax1.plot(x, y, marker=marker, color=color, label=label, lw=2)
            ax1.fill_between(x, lo, hi, color=color, alpha=0.15)

    ax1.axhline(0, color="grey", lw=0.8, ls="--")
    ax1.set_xlabel("Number of qubits")
    ax1.set_ylabel("Makespan reduction (\\%)")
    ax1.set_title("SABRE-MS makespan reduction vs scale")
    ax1.legend()
    ax1.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/mqt_qubit_sweep.pdf", bbox_inches="tight")
    fig.savefig("figures/mqt_qubit_sweep.png", dpi=150, bbox_inches="tight")
    print("saved figures/mqt_qubit_sweep.{pdf,png}")

    # Per-n summary: makespan mean %, ESP median log10 ratio
    def summary(pts, label):
        print(f"\n{label}:")
        for n in sorted(pts):
            rs = pts[n]
            mr = np.mean([v for r in rs for v in r["mks_red_vals"]])
            ratios = np.array([v for r in rs for v in r["esp_ratio_vals"] if v > 0], float)
            med_log = float(np.median(np.log10(ratios)))
            print(f"  {n:>3}q: makespan {mr:+.2f}%  median log10(ESP ratio) {med_log:+.2f} "
                  f"(={10**med_log:.1f}x)  ({len(rs)} families)")
    summary(grid_pts, "Grid series")
    summary(hhex_pts, "Heavy-hex series")


if __name__ == "__main__":
    main()

"""The absorption channel, per configuration. Reads the per-run absorption
experiment (results/swap_absorption_perrun.json) and plots, for each
configuration: the percentage-point increase in the fraction of SWAP-CNOTs the
cancellation pass eliminates (SABRE-MS minus SABRE) on the x-axis, against the
percentage makespan reduction on the y-axis. Reports the Pearson r.

Saves figures/mechanism_scatter.pdf
"""
from __future__ import annotations
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    with open("results/swap_absorption_perrun.json") as f:
        data = json.load(f)
    cells = [v for v in data.values() if v and "ms_k20" in v]

    # x: absorption-rate uplift (MS - SABRE), percentage points
    # y: makespan reduction (SABRE - MS) / SABRE, percent
    x, y, labels = [], [], []
    for c in cells:
        s, m = c["sabre_k20"], c["ms_k20"]
        x.append(m["mean_absorbed_pct"] - s["mean_absorbed_pct"])
        sm = s["mean_mk_opt"]
        y.append(100.0 * (sm - m["mean_mk_opt"]) / sm if sm else 0.0)
        labels.append(f"{c['topology']}/{c['family']}")

    x, y = np.array(x), np.array(y)
    r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else float("nan")

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.scatter(x, y, s=42, color="tab:blue", edgecolor="k", linewidth=0.5, zorder=3)
    # least-squares trend line
    if len(x) > 1:
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, a + b * xs, color="0.4", ls="--", lw=1.2, zorder=2,
                label=f"Pearson $r={r:.2f}$")
        ax.legend(fontsize=9, loc="lower right")
    ax.set_xlabel("Absorption-rate uplift: extra SWAP-CNOTs cancelled,\n"
                  "SABRE-MS $-$ SABRE (percentage points)")
    ax.set_ylabel("Makespan reduction (\\%)")
    ax.set_title(f"More absorbed SWAPs, larger makespan gain ({len(cells)} configs)")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/mechanism_scatter.pdf", bbox_inches="tight")
    fig.savefig("figures/mechanism_scatter.png", dpi=150, bbox_inches="tight")
    print(f"saved figures/mechanism_scatter.{{pdf,png}}  (Pearson r={r:.3f}, n={len(cells)})")
    for lab, xi, yi in sorted(zip(labels, x, y), key=lambda t: -t[2]):
        print(f"  {lab:<20} absorb_uplift {xi:+6.2f}pp   makespan {yi:+6.2f}%")


if __name__ == "__main__":
    main()

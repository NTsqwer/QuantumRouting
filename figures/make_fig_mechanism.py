"""Figure: mechanism scatter plot. Per-cell, x = absorption uplift
(SABRE-MS minus SABRE absorption rate, percentage points), y = makespan
gain (post-optimisation, percent). r = 0.88 line + cell labels.

Reads results/swap_absorption_heldout.json (mechanism, run at the held-out
per-topology lambda -- the same rule the audit uses). Absorption uplift and
makespan gain both come from that single file, so the r is self-consistent.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    with open("results/swap_absorption_heldout.json") as f:
        mech = json.load(f)

    # Restrict to cells on the paper's two benchmark sets: the six core
    # topologies (the off-dataset linear5/linear9 diagnostic chains are dropped).
    CORE_TOPOS = {"linear7", "ring8", "grid3x3", "ring12", "heavy_hex2", "grid4x4"}

    # Build paired list using makespan gain from the SAME mechanism file
    # (these are the numbers cited in the text and the r=0.88 claim).
    xs, ys, labels = [], [], []
    for key, d in mech.items():
        if key.split("/")[0] not in CORE_TOPOS:
            continue
        sabre_abs = d["sabre_k20"]["mean_absorbed_pct"]
        ms_abs = d["ms_k20"]["mean_absorbed_pct"]
        uplift = ms_abs - sabre_abs
        s_mk = d["sabre_k20"]["mean_mk_opt"]
        m_mk = d["ms_k20"]["mean_mk_opt"]
        if s_mk <= 0:
            continue
        gain = 100 * (s_mk - m_mk) / s_mk
        xs.append(float(uplift))
        ys.append(float(gain))
        labels.append(key)

    xs = np.array(xs); ys = np.array(ys)
    r = float(np.corrcoef(xs, ys)[0, 1])
    print(f"n = {len(xs)} cells, r = {r:.3f}")

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.scatter(xs, ys, s=44, facecolor="#fafafa", edgecolor="#111111",
              linewidths=1.0, zorder=3)
    # Linear regression line
    coef = np.polyfit(xs, ys, 1)
    xs_line = np.linspace(xs.min() - 1, xs.max() + 1, 10)
    ax.plot(xs_line, np.polyval(coef, xs_line),
           color="#888888", linewidth=1.0, linestyle="--", zorder=2)
    # Label each point. Per-label offsets de-conflict the few near-coincident
    # cells (grid3x3/qft vs grid4x4/parallel sit almost on top of each other).
    OFFSETS = {
        "grid3x3/qft": (4, -10),
        "grid4x4/parallel": (4, 4),
        "grid4x4/qft": (4, 4),
        "ring8/qft": (-46, -12),
    }
    for x, y, lab in zip(xs, ys, labels):
        # Shorten heavy_hex2 -> hhex2 for compactness
        lab_short = lab.replace("heavy_hex2", "hhex2")
        dx, dy = OFFSETS.get(lab, (4, 3))
        ax.annotate(lab_short, (x, y), xytext=(dx, dy),
                   textcoords="offset points", fontsize=7, color="#222222")
    ax.set_xlabel("Absorption uplift  (SABRE-MS $-$ SABRE, pp)", fontsize=9)
    ax.set_ylabel("Makespan gain  (\\%)", fontsize=9)
    ax.text(0.04, 0.94, f"Pearson $r = {r:.2f}$", transform=ax.transAxes,
           fontsize=9, va="top",
           bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                     edgecolor="#888888"))
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout(pad=0.4)
    out = "figures/mechanism_scatter.pdf"
    plt.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight",
               pad_inches=0.05, dpi=200)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

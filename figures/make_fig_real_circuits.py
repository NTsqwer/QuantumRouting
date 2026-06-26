"""Figure: per-cell gain bar chart for the real-circuit benchmark.

Groups bars by circuit family (QFT, QV, VQE, QAOA) so the family-wise
pattern is visible at a glance. Bar shading marks significance
(p < 0.05). The figure complements Table 8 rather than replacing it:
the table provides exact numbers, the figure shows the per-family
shape.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FAMILY_ORDER = ["qft", "qv", "vqe", "qaoa"]
FAMILY_LABEL = {"qft": "QFT", "qv": "QV", "vqe": "VQE", "qaoa": "QAOA"}


def main():
    with open("results/real_full.json") as f:
        d = json.load(f)

    # Build list of (family, topology, gain, sig)
    rows = []
    for key, v in d.items():
        if v.get("n_circuits", 0) < 1:
            continue
        sig = (v["gain_pct"] > 1) and (v["p_ms_less"] < 0.05)
        rows.append((v["family"], v["topology"], v["gain_pct"], sig))

    # Group by family in display order
    by_fam = {fam: [] for fam in FAMILY_ORDER}
    for r in rows:
        if r[0] in by_fam:
            by_fam[r[0]].append(r)

    # Sort each family by topology name length then alpha (visually
    # consistent across families)
    topology_order = ["linear7", "linear9", "ring8", "ring12",
                     "grid3x3", "grid4x4", "heavy_hex2"]
    for fam in by_fam:
        by_fam[fam].sort(key=lambda r: topology_order.index(r[1])
                        if r[1] in topology_order else 999)

    # Flatten with gaps between families
    labels, gains, sigs, family_at_pos = [], [], [], []
    boundaries = []  # x-positions where families end
    for fam in FAMILY_ORDER:
        for _, topo, gain, sig in by_fam[fam]:
            labels.append(topo.replace("heavy_hex2", "hhex2"))
            gains.append(gain)
            sigs.append(sig)
            family_at_pos.append(fam)
        boundaries.append(len(labels) - 0.5)

    xs = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 3.2))

    # Color: significant = solid dark fill, non-sig = hatched light
    bar_color = "#3a3a3a"
    light = "#cccccc"
    for i, (g, s) in enumerate(zip(gains, sigs)):
        ax.bar(i, g, color=bar_color if s else light,
              edgecolor="#222222", linewidth=0.6,
              hatch="" if s else "//")

    ax.axhline(0, color="#000000", linewidth=0.7)

    # Family dividers
    prev = -0.5
    for end in boundaries[:-1]:
        ax.axvline(end, color="#888888", linestyle=":",
                  linewidth=0.7, zorder=0)
        prev = end

    # X-tick labels
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)

    # Family group titles
    starts = [0]
    for b in boundaries[:-1]:
        starts.append(int(np.ceil(b)))
    for i, fam in enumerate(FAMILY_ORDER):
        start = starts[i]
        end = boundaries[i]
        center = (start + end) / 2
        ax.text(center, max(gains) * 1.05, FAMILY_LABEL[fam],
               ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Makespan reduction vs SABRE-lookahead K=20 (%)",
                 fontsize=9)
    ax.set_ylim(0, max(gains) * 1.32)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    # Legend for significance (bottom-right to avoid the QAOA group title)
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=bar_color, edgecolor="#222222",
              label="$p < 0.05$"),
        Patch(facecolor=light, edgecolor="#222222", hatch="//",
              label="not significant"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right",
             frameon=False, bbox_to_anchor=(1.0, 1.02), ncol=2)

    plt.tight_layout(pad=0.4)
    out = "figures/real_circuits_gains.pdf"
    plt.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight",
               pad_inches=0.05, dpi=200)
    print(f"Saved {out}  ({len(labels)} cells, mean gain "
          f"{np.mean(gains):.1f}%, "
          f"sig {sum(sigs)}/{len(sigs)})")


if __name__ == "__main__":
    main()

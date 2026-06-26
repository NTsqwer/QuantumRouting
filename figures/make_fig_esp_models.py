"""Slide figure: ESP model 1 (ours) vs ESP model 2 (Pietrzak) by topology.

Grouped bar chart of the median SABRE-MS/SABRE ESP ratio per topology, under
each of the two success-probability models. Reads results/core_esp_pietrzak.json
(each cell carries both our_ratio and pz_ratio on the same circuits).
Output: figures/esp_models.{pdf,png}.
"""
import json
import os
import statistics as st
import collections

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#1a1a1a"
OURS = "#2f6db0"      # blue: our model
PZ = "#b03030"        # red: Pietrzak model
GREY = "#9a9a9a"

ORDER = ["linear7", "ring8", "grid3x3", "ring12", "heavy_hex2", "grid4x4"]
LABEL = {"linear7": "linear7", "ring8": "ring8", "grid3x3": "grid3x3",
         "ring12": "ring12", "heavy_hex2": "heavy-hex", "grid4x4": "grid4x4"}


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = json.load(open(os.path.join(here, "results", "core_esp_pietrzak.json")))
    by = collections.defaultdict(list)
    for v in d.values():
        by[v["topology"]].append(v)

    topos = [t for t in ORDER if t in by]
    our_med = [st.median([x["our_ratio"] for x in by[t]]) for t in topos]
    pz_med = [st.median([x["pz_ratio"] for x in by[t]]) for t in topos]

    x = np.arange(len(topos))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.bar(x - w / 2, our_med, w, label="Model 1: single $T_2$ decay (ours)",
           color=OURS, edgecolor=INK, linewidth=0.6)
    ax.bar(x + w / 2, pz_med, w, label="Model 2: $T_1/T_2$ Haar idle (Pietrzak)",
           color=PZ, edgecolor=INK, linewidth=0.6)
    ax.axhline(1.0, color=GREY, lw=1.0, ls=(0, (3, 3)), zorder=0)
    ax.text(len(topos) - 0.5, 1.02, "parity", color=GREY, fontsize=8, va="bottom",
            ha="right")

    for xi, (o, p) in enumerate(zip(our_med, pz_med)):
        ax.text(xi - w / 2, o + 0.03, f"{o:.2f}", ha="center", va="bottom",
                fontsize=7.5, color=OURS)
        ax.text(xi + w / 2, p + 0.03, f"{p:.2f}", ha="center", va="bottom",
                fontsize=7.5, color=PZ)

    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[t] for t in topos])
    ax.set_ylabel("median ESP ratio  (SABRE-MS / SABRE)")
    ax.set_title("Predicted success-probability gain under two ESP models",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(our_med) * 1.18)

    plt.tight_layout()
    out = os.path.join(here, "figures", "esp_models.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    print("Saved", out)


if __name__ == "__main__":
    main()

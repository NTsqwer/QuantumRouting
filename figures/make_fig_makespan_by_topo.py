"""Slide figure: SABRE-MS makespan reduction by topology (MQT audit cells).

Bar chart of mean makespan reduction per topology over the routing-meaningful
MQT cells (core_mqt.json, audit_gain_pct). Output: figures/makespan_by_topo.{pdf,png}.
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
BLUE = "#2f6db0"
GREY = "#9a9a9a"

ORDER = ["linear7", "ring8", "grid3x3", "ring12", "heavy_hex2", "grid4x4"]
LABEL = {"linear7": "linear7", "ring8": "ring8", "grid3x3": "grid3x3",
         "ring12": "ring12", "heavy_hex2": "heavy-hex", "grid4x4": "grid4x4"}


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = json.load(open(os.path.join(here, "results", "core_mqt.json")))
    rows = [v for v in d.values()
            if isinstance(v, dict) and v.get("routing") and "audit_gain_pct" in v]
    by = collections.defaultdict(list)
    for v in rows:
        by[v["topology"]].append(v["audit_gain_pct"])

    topos = [t for t in ORDER if t in by]
    means = [st.mean(by[t]) for t in topos]
    overall = st.mean([v["audit_gain_pct"] for v in rows])

    x = np.arange(len(topos))
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    ax.bar(x, means, 0.62, color=BLUE, edgecolor=INK, linewidth=0.6)
    ax.axhline(overall, color=GREY, lw=1.0, ls=(0, (3, 3)), zorder=0)
    ax.text(len(topos) - 0.5, overall + 0.4, f"overall {overall:.1f}%",
            color=GREY, fontsize=8.5, ha="right", va="bottom")
    for xi, m in enumerate(means):
        ax.text(xi, m + 0.4, f"{m:.0f}%", ha="center", va="bottom",
                fontsize=8.5, color=BLUE)

    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[t] for t in topos])
    ax.set_ylabel("mean makespan reduction")
    ax.set_title("SABRE-MS makespan reduction by topology (MQT)", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(means) * 1.18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    plt.tight_layout()
    out = os.path.join(here, "figures", "makespan_by_topo.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    print("Saved", out)


if __name__ == "__main__":
    main()

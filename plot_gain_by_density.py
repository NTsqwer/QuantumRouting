"""Where does the proxy gap come from? The MQT diagnostic
(results/reranking_mqt.json) shows the gap is STRUCTURAL: it concentrates on
circuits with dense/irregular entanglement and vanishes on shallow, regular
state-prep circuits. No single scalar (qubit count, makespan, SWAP count)
predicts it -- the circuit family does. So we report the gap by family.

Saves figures/gain_by_density.pdf
"""
from __future__ import annotations
import json
import os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# A coarse, defensible label for each MQT family's routing character.
CHARACTER = {
    "ghz": "shallow", "wstate": "shallow", "vqe_su2": "shallow",
    "qaoa": "moderate", "qftentangled": "moderate", "qft": "moderate",
    "ae": "moderate", "qpeexact": "deep/irregular",
}
CHAR_COLOR = {"shallow": "tab:green", "moderate": "tab:orange",
              "deep/irregular": "tab:red"}

# graphstate is a sparse *control* family (it needs ~0 SWAPs at these sizes; see
# Setup). Its large within-pool "gap" is an artefact of SABRE breaking ties at
# random when there is almost nothing to route -- not the structural proxy gap
# this figure is about -- and it is not seed-stable. We exclude it here so the
# figure shows the genuine, deterministic gap on routing-meaningful families.
EXCLUDE = {"graphstate"}


def main():
    with open("results/reranking_mqt.json") as f:
        data = json.load(f)
    cells = [v for v in data.values()
             if "mks_loss_pct" in v and v["benchmark"] not in EXCLUDE]

    fam_gap = defaultdict(list)
    fam_mis = defaultdict(list)
    for c in cells:
        fam_gap[c["benchmark"]].append(c["mks_loss_pct"])
        fam_mis[c["benchmark"]].append(c["misaligned"])
    fams = sorted(fam_gap, key=lambda f: np.mean(fam_gap[f]))
    means = [np.mean(fam_gap[f]) for f in fams]
    colors = [CHAR_COLOR[CHARACTER[f]] for f in fams]

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    bars = ax.barh(fams, means, color=colors, edgecolor="k", linewidth=0.5)
    for f, m, b in zip(fams, means, bars):
        mis = 100 * np.mean(fam_mis[f])
        ax.text(m + 0.4, b.get_y() + b.get_height() / 2,
                f"{mis:.0f}% mis", va="center", fontsize=7, color="0.3")
    ax.set_xlabel("Makespan recovered by ranking on makespan\n"
                  "instead of SWAP count (\\%)")
    ax.set_title("Proxy gap is structural: large on deep/irregular circuits,\n"
                 f"absent on shallow state-prep ({len(fams)} MQT-Bench families)")
    ax.grid(alpha=0.3, axis="x")
    # Legend by routing character.
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=CHAR_COLOR[k], edgecolor="k", label=k)
               for k in ["shallow", "moderate", "deep/irregular"]]
    ax.legend(handles=handles, fontsize=8, loc="lower right",
              title="routing character")

    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/gain_by_density.pdf", bbox_inches="tight")
    fig.savefig("figures/gain_by_density.png", dpi=150, bbox_inches="tight")
    print("saved figures/gain_by_density.{pdf,png}")

    print("\nFamily mean gaps (shallow->deep):")
    for f in fams:
        print(f"  {f:<14} {np.mean(fam_gap[f]):+6.2f}%  "
              f"({100*np.mean(fam_mis[f]):.0f}% misaligned, {CHARACTER[f]})")


if __name__ == "__main__":
    main()

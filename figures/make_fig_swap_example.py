"""Figure: the routing problem on a small star, as a worked example.

Three panels on a 5-qubit star (one hub, four leaves):
  (a) the device coupling graph: only hub-leaf pairs are wired;
  (b) the program needs a gate between two leaves -- not connected;
  (c) a SWAP across one wired (hub-leaf) edge brings the two together,
      after which the gate runs.

Black-and-white, publication-ready, matches figures/make_fig_topologies.py
house style. Output: figures/swap_example.{pdf,png}.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Star geometry: hub at centre, four leaves N/E/S/W.
HUB = (0.0, 0.0)
LEAF = {
    "N": (0.0, 1.0),
    "E": (1.0, 0.0),
    "S": (0.0, -1.0),
    "W": (-1.0, 0.0),
}
EDGES = [("HUB", k) for k in LEAF]  # only hub-leaf pairs are coupled
POS = {"HUB": HUB, **LEAF}

NODE_R = 0.22
DARK = "#111111"


def draw_node(ax, xy, label, fill="white", text=DARK, lw=1.4):
    c = plt.Circle(xy, NODE_R, facecolor=fill, edgecolor=DARK, lw=lw, zorder=3)
    ax.add_patch(c)
    ax.text(xy[0], xy[1], label, ha="center", va="center",
            fontsize=9, color=text, zorder=4)


def draw_edges(ax, highlight=None):
    for a, b in EDGES:
        xa, ya = POS[a]
        xb, yb = POS[b]
        hot = highlight is not None and (a, b) in highlight or (b, a) in (highlight or [])
        ax.plot([xa, xb], [ya, yb], color=DARK,
                lw=2.4 if hot else 1.3, zorder=1,
                solid_capstyle="round")


def need_gate(ax, p, q, ok):
    """Draw a curved dashed connector between leaf positions p and q
    annotating the wanted two-qubit gate."""
    style = "-" if ok else (0, (3, 3))
    arr = FancyArrowPatch(
        POS[p], POS[q], connectionstyle="arc3,rad=0.35",
        arrowstyle="-", linestyle=style, lw=1.6,
        color=DARK if ok else "#888888", zorder=2,
    )
    ax.add_patch(arr)


def frame(ax, title):
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.7, 1.6)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.5))

    # --- (a) the device: a star coupling graph ---
    ax = axes[0]
    draw_edges(ax)
    draw_node(ax, POS["HUB"], "$Q_0$")
    draw_node(ax, POS["N"], "$Q_1$")
    draw_node(ax, POS["E"], "$Q_2$")
    draw_node(ax, POS["S"], "$Q_3$")
    draw_node(ax, POS["W"], "$Q_4$")
    ax.text(0, -1.55, "device: only $Q_0$ couples to each $Q_i$",
            ha="center", va="center", fontsize=7.5, style="italic", color=DARK)
    frame(ax, "(a) coupling graph")

    # --- (b) gate wanted between two leaves: not connected ---
    ax = axes[1]
    draw_edges(ax)
    # qubits q1..q4 placed on leaves, q0 on hub
    draw_node(ax, POS["HUB"], "$q_0$")
    draw_node(ax, POS["N"], "$q_1$")
    draw_node(ax, POS["E"], "$q_2$")
    draw_node(ax, POS["S"], "$q_3$")
    draw_node(ax, POS["W"], "$q_4$")
    need_gate(ax, "N", "E", ok=False)  # gate on q1,q2 -- both leaves
    ax.text(0.78, 0.78, "gate on\n$q_1,q_2$", ha="center", va="center",
            fontsize=7.5, color="#666666")
    ax.text(0, -1.55, "$q_1,q_2$ not coupled: cannot run",
            ha="center", va="center", fontsize=7.5, style="italic", color=DARK)
    frame(ax, "(b) gate not executable")

    # --- (c) after SWAP(q0,q1): q1 sits on the hub, now next to q2 ---
    ax = axes[2]
    draw_edges(ax, highlight=[("HUB", "N")])
    # SWAP exchanged states of hub and north leaf
    draw_node(ax, POS["HUB"], "$q_1$", fill="#dddddd")
    draw_node(ax, POS["N"], "$q_0$", fill="#dddddd")
    draw_node(ax, POS["E"], "$q_2$")
    draw_node(ax, POS["S"], "$q_3$")
    draw_node(ax, POS["W"], "$q_4$")
    need_gate(ax, "HUB", "E", ok=True)  # now q1 (hub) and q2 (east) adjacent
    ax.text(1.05, 0.62, "gate on\n$q_1,q_2$", ha="center", va="center",
            fontsize=7.5, color="#444444")
    ax.text(0, -1.55, "after SWAP($q_0,q_1$): now coupled, gate runs",
            ha="center", va="center", fontsize=7.5, style="italic", color=DARK)
    frame(ax, "(c) SWAP, then run")

    plt.tight_layout(pad=0.4)
    out = "figures/swap_example.pdf"
    plt.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight",
                pad_inches=0.05, dpi=200)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

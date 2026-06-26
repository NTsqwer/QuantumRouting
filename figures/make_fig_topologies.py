"""Figure: representative topologies used in the paper.

Three small connectivity-graph panels: ring12, grid4x4, heavy_hex2.
Black-and-white, publication-ready, tight footprint.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from topologies import get as get_topology


def heavy_hex2_layout(n_rows=2, row_len=7):
    """Layout matching the construction in topologies.heavy_hex.
    Row r at y = r, qubit i in that row at x = i."""
    pos = {}
    for r in range(n_rows):
        for i in range(row_len):
            pos[r * row_len + i] = (float(i), -float(r))
    return pos


def draw(ax, graph, title, layout="auto"):
    n = graph.number_of_nodes()
    if layout == "circular":
        pos = nx.circular_layout(graph)
    elif layout == "grid":
        side = int(round(n ** 0.5))
        pos = {i: (i % side, -(i // side)) for i in range(n)}
    elif layout == "heavy":
        pos = heavy_hex2_layout()
    else:
        pos = nx.spring_layout(graph, seed=0)
    nx.draw_networkx_edges(graph, pos, ax=ax, width=1.2, edge_color="#333333")
    nx.draw_networkx_nodes(graph, pos, ax=ax,
                          node_color="white", edgecolors="#111111",
                          node_size=170, linewidths=1.2)
    ax.set_axis_off()
    ax.set_title(f"{title}  (n={n})", fontsize=9)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.0))
    g, _ = get_topology("ring12")
    draw(axes[0], g, "ring12", "circular")
    g, _ = get_topology("grid4x4")
    draw(axes[1], g, "grid4x4", "grid")
    g, _ = get_topology("heavy_hex2")
    draw(axes[2], g, "heavy_hex2", "heavy")
    plt.tight_layout(pad=0.4)
    out = "figures/topologies.pdf"
    plt.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight",
               pad_inches=0.05, dpi=200)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

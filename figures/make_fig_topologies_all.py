"""Figure: the six core coupling graphs used in the paper.

Two rows of three panels: linear7, ring8, grid3x3 / ring12, heavy_hex2, grid4x4
-- the connectivity classes (linear chain, ring, square grid, heavy-hex). Drawn
straight from topologies.get(), so the graphs are exactly the ones the
experiments route on. Black-and-white, tight footprint.

Output: figures/topologies_all.{pdf,png}
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from topologies import get as get_topology


def layout_for(name, graph):
    n = graph.number_of_nodes()
    if name.startswith("ring"):
        return nx.circular_layout(graph)
    if name.startswith("linear"):
        return {i: (float(i), 0.0) for i in range(n)}
    if name == "grid3x3":
        return {i: (i % 3, -(i // 3)) for i in range(n)}
    if name == "grid4x4":
        return {i: (i % 4, -(i // 4)) for i in range(n)}
    if name == "heavy_hex2":
        # two rows of 7, matching topologies.heavy_hex construction; the
        # vertical rungs (0-7, 2-9, 4-11, 6-13) form the heavy-hex cells.
        return {r * 7 + i: (float(i), -1.6 * r) for r in range(2) for i in range(7)}
    return nx.spring_layout(graph, seed=0)


def draw(ax, name):
    g, _ = get_topology(name)
    pos = layout_for(name, g)
    nx.draw_networkx_edges(g, pos, ax=ax, width=1.0, edge_color="#444444")
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color="white",
                           edgecolors="#111111", node_size=90, linewidths=0.9)
    ax.set_axis_off()
    ax.margins(0.12)
    ax.set_title(f"{name} ($n={g.number_of_nodes()}$)", fontsize=9)


def main():
    names = ["linear7", "ring8", "grid3x3", "ring12", "heavy_hex2", "grid4x4"]
    fig, axes = plt.subplots(2, 3, figsize=(6.4, 3.4))
    for ax, name in zip(axes.flat, names):
        draw(ax, name)
    plt.tight_layout(pad=0.5)
    out = "figures/topologies_all.pdf"
    plt.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.savefig(out.replace(".pdf", ".png"), bbox_inches="tight",
                pad_inches=0.04, dpi=200)
    print("Saved", out)


if __name__ == "__main__":
    main()

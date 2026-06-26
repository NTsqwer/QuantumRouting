"""Connection-graph topologies used for the routing experiments."""

from __future__ import annotations

import networkx as nx


def linear_chain(n: int) -> nx.Graph:
    """Path graph: 0 - 1 - 2 - ... - (n-1)."""
    return nx.path_graph(n)


def t_shape_5() -> nx.Graph:
    """5-qubit T-shape: 0-1-2-3 with 4 attached to 2.

    Two parallel branches enable independent gates -- exposes scheduling
    advantage of routes that preserve parallelism.
    """
    g = nx.Graph()
    g.add_edges_from([(0, 1), (1, 2), (2, 3), (2, 4)])
    return g


def grid(m: int, n: int) -> nx.Graph:
    """m x n 2D grid graph with nodes relabeled to integers 0..m*n-1."""
    g = nx.grid_2d_graph(m, n)
    return nx.convert_node_labels_to_integers(g)


def ring(n: int) -> nx.Graph:
    """Cycle graph: 0-1-...-n-1-0."""
    return nx.cycle_graph(n)


def heavy_hex(n_rows: int) -> nx.Graph:
    """Crude approximation of IBM's heavy-hex topology.
    Builds n_rows of length-7 chains, connects alternating qubits with cross-links.
    Returns ~n_rows*7 qubits."""
    g = nx.Graph()
    row_len = 7
    for r in range(n_rows):
        offset = r * row_len
        for i in range(row_len - 1):
            g.add_edge(offset + i, offset + i + 1)
        if r > 0:
            # Cross-links every other qubit
            for i in range(0, row_len, 2):
                g.add_edge((r - 1) * row_len + i, r * row_len + i)
    return g


def star_5() -> nx.Graph:
    """5-qubit star: node 0 is the center, connected to 1, 2, 3, 4.
    All non-center nodes connect only through 0 (high-degree central qubit)."""
    g = nx.Graph()
    g.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4)])
    return g


def star(n: int) -> nx.Graph:
    """n-qubit star: node 0 is the hub, connected to 1..n-1. Every
    interaction not involving the hub must route through it, making the
    hub a hard scheduling bottleneck -- the purest setting for the
    SWAP-count/makespan proxy gap."""
    g = nx.Graph()
    g.add_edges_from([(0, i) for i in range(1, n)])
    return g


TOPOLOGIES = {
    "linear5": (linear_chain(5), 5),
    "tshape5": (t_shape_5(), 5),
    "ring5": (ring(5), 5),
    "star5": (star_5(), 5),
    "star8": (star(8), 8),
    "star10": (star(10), 10),
    "linear7": (linear_chain(7), 7),
    "linear9": (linear_chain(9), 9),
    "linear15": (linear_chain(15), 15),
    "linear20": (linear_chain(20), 20),
    "ring8": (ring(8), 8),
    "ring12": (ring(12), 12),
    "ring20": (ring(20), 20),
    "grid3x3": (grid(3, 3), 9),
    "grid4x4": (grid(4, 4), 16),
    "grid5x5": (grid(5, 5), 25),
    "grid6x6": (grid(6, 6), 36),
    "grid7x7": (grid(7, 7), 49),
    "heavy_hex2": (heavy_hex(2), 14),
    "heavy_hex4": (heavy_hex(4), 28),
    "heavy_hex6": (heavy_hex(6), 42),
}


def get(name: str) -> tuple[nx.Graph, int]:
    """Return (connection_graph, n_qubits) for a named topology."""
    if name not in TOPOLOGIES:
        raise ValueError(
            f"Unknown topology {name!r}. Choices: {list(TOPOLOGIES)}"
        )
    return TOPOLOGIES[name]

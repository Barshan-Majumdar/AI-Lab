"""Experiment 3: Depth First Search (DFS) with graph visualization."""

import networkx as nx
import matplotlib.pyplot as plt


# Create graph
G = nx.Graph()
edges = [
    ("A", "B"),
    ("A", "C"),
    ("B", "D"),
    ("B", "E"),
    ("C", "F"),
    ("C", "G"),
    ("E", "H"),
]
G.add_edges_from(edges)


def dfs(graph: nx.Graph, start: str):
    """Iterative DFS traversal using a stack."""
    visited = []
    seen = set()
    stack = [start]

    while stack:
        node = stack.pop()

        if node in seen:
            continue

        seen.add(node)
        visited.append(node)

        # Reverse-sorted push keeps traversal deterministic (A, B, D, ...).
        neighbors = sorted(set(graph.neighbors(node)) - seen, reverse=True)
        stack.extend(neighbors)

    return visited


start_node = "A"
traversal = dfs(G, start_node)
print("DFS Traversal:", traversal)


# Draw graph
pos = nx.spring_layout(G, seed=42)
node_colors = ["red" if node in traversal else "lightblue" for node in G.nodes()]

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=1000,
    node_color=node_colors,
    font_size=12,
)

plt.title("Depth First Search")
plt.tight_layout()
plt.show()
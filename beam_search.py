import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def beam_search(graph, heuristic, start, goal, beam_width):
    parent = {start: None}
    beam = [start]
    selected_nodes = {start}
    discarded_nodes = set()
    expanded_nodes = set()

    print("\n==============================")
    print(" BEAM SEARCH")
    print("==============================")

    level = 0

    while beam:
        print(f"\nLevel {level}")
        print("Current Beam:", beam)

        # Check if goal is in beam
        if goal in beam:
            print("\nGoal Found!")

            path = []
            current = goal
            while current is not None:
                path.append(current)
                current = parent[current]
            path.reverse()

            print("Final Path:", " -> ".join(path))
            return path, selected_nodes, discarded_nodes, expanded_nodes

        candidates = []

        # Expand nodes in current beam
        for node in beam:
            expanded_nodes.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in selected_nodes:
                    candidates.append(neighbor)
                    if neighbor not in parent:
                        parent[neighbor] = node

        # Remove duplicates while preserving order
        candidates = list(dict.fromkeys(candidates))

        print("Candidates:", candidates)

        # Sort by heuristic (ascending — lower is better)
        candidates.sort(key=lambda x: heuristic[x])

        print("Sorted:", [(node, heuristic[node]) for node in candidates])

        # Select best W nodes
        new_beam = candidates[:beam_width]
        discarded = candidates[beam_width:]

        discarded_nodes.update(discarded)
        selected_nodes.update(new_beam)

        print("Selected:", [(node, heuristic[node]) for node in new_beam])
        print("Discarded:", [(node, heuristic[node]) for node in discarded])

        beam = new_beam
        level += 1

    print("\nGoal could not be reached.")
    return None, selected_nodes, discarded_nodes, expanded_nodes


# ── Graph definition ──────────────────────────────────────────────────────────

graph = {
    'S': ['A', 'B', 'C'],
    'A': ['D', 'E'],
    'B': ['F', 'G'],
    'C': ['H', 'I'],
    'D': [],
    'E': [],
    'F': [],
    'G': [],
    'H': [],
    'I': [],
}

# ── Heuristic values ──────────────────────────────────────────────────────────

heuristic = {
    'S': 10,
    'A': 6,
    'B': 3,
    'C': 5,
    'D': 4,
    'E': 7,
    'F': 2,
    'G': 6,
    'H': 3,
    'I': 8,
}

# ── Parameters ────────────────────────────────────────────────────────────────

start = 'S'
goal = 'F'
beam_width = 2

# ── Run beam search ───────────────────────────────────────────────────────────

path, selected_nodes, discarded_nodes, expanded_nodes = beam_search(
    graph, heuristic, start, goal, beam_width
)

# ── Build NetworkX graph ──────────────────────────────────────────────────────

G = nx.DiGraph()
for node in graph:
    G.add_node(node)
    for neighbor in graph[node]:
        G.add_edge(node, neighbor)

# ── Node positions ────────────────────────────────────────────────────────────

pos = {
    'S': (0, 3),
    'A': (-3, 2),
    'B': (0, 2),
    'C': (3, 2),
    'D': (-4, 1),
    'E': (-2, 1),
    'F': (-0.8, 1),
    'G': (0.8, 1),
    'H': (2, 1),
    'I': (4, 1),
}

# ── Node colors ───────────────────────────────────────────────────────────────

node_colors = []
for node in G.nodes():
    if path and node in path:
        node_colors.append("red")
    elif node in discarded_nodes:
        node_colors.append("lightgray")
    elif node in selected_nodes:
        node_colors.append("orange")
    else:
        node_colors.append("skyblue")

# ── Edge colors ───────────────────────────────────────────────────────────────

edge_colors = []
for u, v in G.edges():
    if path and u in path and v in path:
        is_path_edge = any(
            path[i] == u and path[i + 1] == v
            for i in range(len(path) - 1)
        )
        edge_colors.append("red" if is_path_edge else "gray")
    else:
        edge_colors.append("gray")

# ── Node labels ───────────────────────────────────────────────────────────────

labels = {node: f"{node}\n(h={heuristic[node]})" for node in G.nodes()}

# ── Draw ──────────────────────────────────────────────────────────────────────

plt.figure(figsize=(12, 7))

nx.draw_networkx_edges(
    G, pos,
    edge_color=edge_colors,
    arrows=True,
    arrowsize=20,
    width=2,
)

nx.draw_networkx_nodes(
    G, pos,
    node_color=node_colors,
    node_size=2200,
    edgecolors="black",
    linewidths=2,
)

nx.draw_networkx_labels(
    G, pos,
    labels=labels,
    font_size=11,
    font_weight="bold",
)

plt.title(
    f"Beam Search\nBeam Width = {beam_width} | Start = {start} | Goal = {goal}",
    fontsize=16,
    fontweight="bold",
)

legend_elements = [
    Patch(facecolor="red",       edgecolor="black", label="Final Path"),
    Patch(facecolor="orange",    edgecolor="black", label="Selected by Beam"),
    Patch(facecolor="skyblue",   edgecolor="black", label="Unselected Node"),
    Patch(facecolor="lightgray", edgecolor="black", label="Discarded"),
]

plt.legend(handles=legend_elements, loc="upper right")
plt.axis("off")
plt.tight_layout()
plt.show()
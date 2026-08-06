"""Experiment 5: A* Search algorithm with step-by-step visualization."""

import heapq
import networkx as nx
import matplotlib.pyplot as plt


# Weighted undirected graph matching the experiment reference
graph = {
    "A": [("B", 3), ("C", 1), ("D", 4)],
    "B": [("A", 3), ("E", 5), ("F", 2)],
    "C": [("A", 1), ("E", 2), ("F", 3)],
    "D": [("A", 4), ("F", 4)],
    "E": [("B", 5), ("C", 2), ("G", 2)],
    "F": [("B", 2), ("C", 3), ("D", 4), ("G", 1)],
    "G": [("E", 2), ("F", 1)],
}

heuristic = {"A": 7, "B": 6, "C": 4, "D": 5, "E": 3, "F": 1, "G": 0}

# Fixed positions to resemble the PDF figure
pos = {
    "A": (2, 4),
    "B": (0, 2),
    "C": (2, 2),
    "D": (4, 2),
    "E": (1, 0),
    "F": (3, 0),
    "G": (2, -2),
}

# Create drawable graph
G = nx.Graph()
for node in graph:
    for neighbor, cost in graph[node]:
        G.add_edge(node, neighbor, weight=cost)


def draw_graph(current=None, path=None, step=0):
    plt.figure(figsize=(8, 6))

    node_colors = []
    for n in G.nodes():
        if path and n in path:
            node_colors.append("limegreen")
        elif n == current:
            node_colors.append("orange")
        else:
            node_colors.append("skyblue")

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")
    nx.draw_networkx_edges(G, pos, width=2)

    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    if path is not None and len(path) > 1:
        edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color="green", width=4)

    plt.title(f"A* Search: Step {step}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"step_{step}.png", dpi=150)
    plt.show()


def astar(start, goal):
    pq = []
    heapq.heappush(pq, (heuristic[start], 0, start, [start]))

    best_g = {}
    step = 1

    while pq:
        f, g, node, path = heapq.heappop(pq)

        print("-----------------------------------")
        print("Expand", node)
        print("g =", g)
        print("h =", heuristic[node])
        print("f =", f)

        draw_graph(current=node, step=step)
        step += 1

        if node == goal:
            return path, g

        if node in best_g and best_g[node] <= g:
            continue

        best_g[node] = g

        for neighbor, cost in graph[node]:
            new_g = g + cost
            new_f = new_g + heuristic[neighbor]
            heapq.heappush(pq, (new_f, new_g, neighbor, path + [neighbor]))
            print(f"{neighbor}: g={new_g}, h={heuristic[neighbor]}, f={new_f}")

    return None, None


if __name__ == "__main__":
    draw_graph(step=0)
    path, cost = astar("A", "G")
    print("\nOptimal Path:", " -> ".join(path) if path else None)
    print("Total Cost:", cost)
    if path:
        draw_graph(path=path, step=99)
